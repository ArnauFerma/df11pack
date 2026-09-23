//! df11pack -- compress BF16 model weights into the DFloat11 format.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use df11_codec::arch::ArchDef;
use df11_codec::huffman::LutMode;
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};

#[derive(Parser)]
#[command(
    name = "df11pack",
    about = "Compress BF16 model weights into DFloat11 format"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a model.
    Compress {
        /// A .safetensors file, or a directory of them.
        source: PathBuf,
        /// Architecture definition: a name from `df11pack architectures`, or a path to a .toml.
        #[arg(long)]
        arch: String,
        /// Output directory.
        #[arg(long, short)]
        out: PathBuf,
        /// LUT semantics. `compat` is byte-identical to the official compressor.
        #[arg(long, value_enum, default_value_t = LutModeArg::Compat)]
        luts: LutModeArg,
        /// Memory budget, e.g. 512M or 4G. Workers are sized to fit inside it.
        #[arg(long)]
        ram: Option<String>,
        /// Force a worker count, overriding --ram.
        #[arg(long)]
        workers: Option<usize>,
        /// Read scheduling: auto detects the device; sequential suits spinning disks.
        #[arg(long, value_enum, default_value_t = IoArg::Auto)]
        io: IoArg,
        /// Safe mode: decode every unit and check it against the source before
        /// writing it, so a wrong unit never reaches disk.
        #[arg(long)]
        safe: bool,
        /// Store a SHA-256 of every DF11 tensor in the file metadata, so
        /// `verify` can catch a wrong value without the source. Tensor bytes are
        /// unchanged, but the header is no longer byte-identical to the official
        /// tool's.
        #[arg(long)]
        hashes: bool,
        /// Index scheme. `df11` is the format every DF11 loader reads. `idx8`
        /// (docs/INDEX_SCHEMES.md) is smaller but NOT DF11: the official kernel
        /// cannot read it.
        #[arg(long, value_enum, default_value_t = IndexArg::Df11)]
        index: IndexArg,
        /// Symbols per idx8 block.
        #[arg(long, default_value_t = 64)]
        idx8_block: usize,
    },
    /// Check a written output, optionally against the model it was made from.
    ///
    /// Without --source only the structure is checked: every index the GPU
    /// kernel trusts stays in bounds. With it, weights are decoded and compared.
    Verify {
        /// The output directory (or single ComfyUI file).
        output: PathBuf,
        /// The source model the output was compressed from.
        #[arg(long, requires = "arch")]
        source: Option<PathBuf>,
        /// The architecture definition it was compressed with.
        #[arg(long, requires = "source")]
        arch: Option<String>,
        /// integrity needs no source; sample and full decode against it.
        /// Default: sample with a source, integrity without.
        #[arg(long, value_enum)]
        level: Option<LevelArg>,
        /// Chunks to decode at the sample level. The first and last chunk of every
        /// unit and every tensor-boundary chunk are always included, even past
        /// this; the rest of the budget is spread at random.
        #[arg(long, default_value_t = 1000)]
        samples: usize,
        /// Seed for the sample. Printed on every run, so any run can be repeated.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// List the available architecture definitions.
    Architectures,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum IndexArg {
    Df11,
    Idx8,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum LevelArg {
    Integrity,
    Sample,
    Full,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum IoArg {
    Auto,
    Sequential,
    Concurrent,
}

impl From<IoArg> for df11_codec::io_sched::IoMode {
    fn from(a: IoArg) -> Self {
        use df11_codec::io_sched::IoMode;
        match a {
            IoArg::Auto => IoMode::Auto,
            IoArg::Sequential => IoMode::Sequential,
            IoArg::Concurrent => IoMode::Concurrent,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum LutModeArg {
    Compat,
    Correct,
}

impl From<LutModeArg> for LutMode {
    fn from(a: LutModeArg) -> Self {
        match a {
            LutModeArg::Compat => LutMode::Compat,
            LutModeArg::Correct => LutMode::Correct,
        }
    }
}

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_defs.rs"));
}

/// Every available definition as `(name, toml, origin)`: the embedded set, with
/// any in `DF11PACK_ARCH_DIR` added or overriding by name.
fn all_defs() -> Result<Vec<(String, String, String)>, String> {
    let mut defs: std::collections::BTreeMap<String, (String, String)> = embedded::EMBEDDED
        .iter()
        .map(|(n, t)| (n.to_string(), (t.to_string(), "built-in".to_string())))
        .collect();
    if let Ok(dir) = std::env::var("DF11PACK_ARCH_DIR") {
        let rd = std::fs::read_dir(&dir).map_err(|e| format!("DF11PACK_ARCH_DIR={dir}: {e}"))?;
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "toml") {
                let name = p
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                let text =
                    std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                defs.insert(name, (text, p.display().to_string()));
            }
        }
    }
    Ok(defs.into_iter().map(|(n, (t, o))| (n, t, o)).collect())
}

/// A definition by name, or from a `.toml` path.
fn load_arch(spec: &str) -> Result<ArchDef, String> {
    if spec.ends_with(".toml") {
        let text = std::fs::read_to_string(spec)
            .map_err(|e| format!("could not read architecture file {spec}: {e}"))?;
        return ArchDef::from_toml(&text).map_err(|e| format!("{spec}: {e}"));
    }
    let defs = all_defs()?;
    let (_, text, origin) = defs.iter().find(|(n, _, _)| n == spec).ok_or_else(|| {
        format!(
            "no architecture named {spec:?}. Run `df11pack architectures` to see the \
             available names, or pass a path to a .toml definition."
        )
    })?;
    ArchDef::from_toml(text).map_err(|e| format!("{spec} ({origin}): {e}"))
}

fn list_architectures() -> Result<(), String> {
    let entries = all_defs()?;
    if entries.is_empty() {
        return Err("no definitions available".to_string());
    }
    // Written through a locked handle so a closed pipe -- `df11pack
    // architectures | head` -- ends the command quietly instead of panicking,
    // which is what Rust's println! does on EPIPE.
    let stdout = io::stdout();
    let mut w = stdout.lock();
    let quiet = |r: io::Result<()>| -> Result<bool, String> {
        match r {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    };

    if !quiet(writeln!(
        w,
        "{:<30} {:<16} {:>5}  SOURCE",
        "NAME", "LAYOUT", "UNITS"
    ))? {
        return Ok(());
    }
    for (name, text, origin) in entries {
        let line = match ArchDef::from_toml(&text) {
            Ok(d) => format!(
                "{:<30} {:<16} {:>5}  {}",
                d.name,
                d.layout.as_str(),
                d.units.len(),
                d.source
            ),
            Err(e) => format!("{name:<30} INVALID ({origin}): {e}"),
        };
        if !quiet(writeln!(w, "{line}"))? {
            return Ok(());
        }
    }
    Ok(())
}

/// Parse a size like `512M`, `4G`, `1500000`.
fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1u64 << 10),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1u64 << 20),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1u64 << 30),
        _ => (s, 1),
    };
    num.trim()
        .parse::<u64>()
        .map(|v| v * mult)
        .map_err(|_| format!("could not parse size {s:?}; try 512M or 4G"))
}

/// Everything `compress` needs, so the signature stays readable as flags accrue.
struct CompressArgs {
    source: PathBuf,
    arch: String,
    out: PathBuf,
    luts: LutModeArg,
    ram: Option<String>,
    workers: Option<usize>,
    io: IoArg,
    safe: bool,
    hashes: bool,
    index: IndexArg,
    idx8_block: usize,
}

fn compress(a: CompressArgs) -> Result<(), String> {
    let CompressArgs {
        source,
        arch,
        out,
        luts,
        ram,
        workers,
        io,
        safe,
        hashes,
        index,
        idx8_block,
    } = a;
    let def = load_arch(&arch)?;
    let model = ModelSource::open(&source).map_err(|e| format!("{}: {e}", source.display()))?;

    if luts == LutModeArg::Correct {
        eprintln!(
            "warning: --luts=correct produces output that is NOT byte-identical to the\n\
             official compressor. It zero-fills LUT positions the official encoder leaves\n\
             holding leaked state from the previous prefix table. The output is stamped\n\
             df11pack_luts=\"correct\" so it cannot be mistaken for compat output.\n\
             See docs/COMPATIBILITY.md before using this."
        );
    }

    if index == IndexArg::Idx8 {
        eprintln!(
            "warning: --index=idx8 output is NOT DFloat11. No DF11 loader or the official\n\
             kernel can read it; it needs an idx8 decoder. It is stamped\n\
             df11pack_index=\"idx8\" and carries no dfloat11_config.\n\
             See docs/INDEX_SCHEMES.md."
        );
    }

    let ram_budget = match &ram {
        Some(s) => Some(parse_size(s)?),
        None => None,
    };
    let opts = WriteOptions {
        lut_mode: luts.into(),
        ram_budget,
        workers,
        io: io.into(),
        verify: safe,
        hashes,
        idx8_block: (index == IndexArg::Idx8).then_some(idx8_block),
    };
    let report = write_directory(&model, &def, &out, &opts).map_err(|e| e.to_string())?;

    let ratio = if report.source_bytes > 0 {
        report.output_bytes as f64 / report.source_bytes as f64
    } else {
        0.0
    };
    println!(
        "compressed {} unit(s) from {} shard(s)\n  {:.1} MiB -> {:.1} MiB  ({:.3})",
        report.units,
        model.shards(),
        report.source_bytes as f64 / (1 << 20) as f64,
        report.output_bytes as f64 / (1 << 20) as f64,
        ratio
    );
    if !report.tied_dropped.is_empty() {
        println!(
            "  dropped {} tied tensor(s): {}",
            report.tied_dropped.len(),
            report.tied_dropped.join(", ")
        );
    }
    if !report.limited_units.is_empty() {
        println!(
            "  the 32-bit limiter ran on {} unit(s): {}",
            report.limited_units.len(),
            report.limited_units.join(", ")
        );
    }
    if !report.verified.is_empty() {
        println!(
            "  verified {} unit(s) against the source before writing",
            report.verified.len()
        );
    }
    if hashes {
        println!("  stored a SHA-256 of every DF11 tensor; the headers now differ from the official tool's");
    }
    println!("  reads: {}", report.io.reason);
    println!("  wrote {} -> {}", report.shards.len() + 1, out.display());
    Ok(())
}

struct VerifyArgs {
    output: PathBuf,
    source: Option<PathBuf>,
    arch: Option<String>,
    level: Option<LevelArg>,
    samples: usize,
    seed: Option<u64>,
}

/// Returns whether the output passed.
fn verify(a: VerifyArgs) -> Result<bool, String> {
    use df11_codec::check::{check_output, Level};
    let out = ModelSource::open(&a.output).map_err(|e| format!("{}: {e}", a.output.display()))?;
    let reference = match (&a.source, &a.arch) {
        (Some(s), Some(arch)) => Some((
            ModelSource::open(s).map_err(|e| format!("{}: {e}", s.display()))?,
            load_arch(arch)?,
        )),
        _ => None,
    };
    let seed = a.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let level = match (a.level, reference.is_some()) {
        (Some(LevelArg::Integrity), _) | (None, false) => Level::Integrity,
        (Some(LevelArg::Sample), _) | (None, true) => Level::Sample {
            budget: a.samples,
            seed,
        },
        (Some(LevelArg::Full), _) => Level::Full,
    };
    let r = check_output(&out, reference.as_ref().map(|(s, d)| (s, d)), level)
        .map_err(|e| e.to_string())?;

    let what = match level {
        Level::Integrity => "structure".to_string(),
        Level::Sample { seed, .. } => format!(
            "structure, and {} chunk(s) decoded against the source (seed {seed})",
            r.checked.len()
        ),
        Level::Full => format!("structure, and all {} chunk(s) decoded", r.checked.len()),
    };
    println!("checked {} unit(s): {what}", r.units);
    if r.hashed > 0 {
        println!("  {} stored SHA-256 hash(es) compared", r.hashed);
    } else if level == Level::Integrity && reference.is_none() {
        println!(
            "  note: this output has no stored hashes and no --source was given, so a\n  \
             wrong weight value in a well-formed unit is not detectable"
        );
    }
    for f in &r.failures {
        match f.chunk {
            Some(c) => println!("  FAIL {} chunk {c}: {}", f.unit, f.error),
            None => println!("  FAIL {}: {}", f.unit, f.error),
        }
    }
    if r.ok() {
        println!("  ok");
    } else {
        println!("  {} failure(s)", r.failures.len());
    }
    Ok(r.ok())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Compress {
            source,
            arch,
            out,
            luts,
            ram,
            workers,
            io,
            safe,
            hashes,
            index,
            idx8_block,
        } => compress(CompressArgs {
            source,
            arch,
            out,
            luts,
            ram,
            workers,
            io,
            safe,
            hashes,
            index,
            idx8_block,
        }),
        Command::Verify {
            output,
            source,
            arch,
            level,
            samples,
            seed,
        } => {
            // A failed check is exit 1, distinct from 2 for "could not check".
            return match verify(VerifyArgs {
                output,
                source,
                arch,
                level,
                samples,
                seed,
            }) {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::from(1),
                Err(e) => {
                    eprintln!("df11pack: {e}");
                    ExitCode::from(2)
                }
            };
        }
        Command::Architectures => list_architectures(),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("df11pack: {e}");
            ExitCode::FAILURE
        }
    }
}
