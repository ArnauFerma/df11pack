//! df11pack -- compress BF16 model weights into the DFloat11 format.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
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
    },
    /// List the available architecture definitions.
    Architectures,
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

/// Where the shipped definitions live.
fn arch_dir() -> PathBuf {
    if let Ok(d) = std::env::var("DF11PACK_ARCH_DIR") {
        return PathBuf::from(d);
    }
    // Next to the binary, then the working directory, so both an installed
    // layout and a cargo-run from the repo work.
    if let Ok(exe) = std::env::current_exe() {
        for up in [1usize, 2, 3, 4] {
            let mut p = exe.clone();
            for _ in 0..up {
                p.pop();
            }
            let c = p.join("data/architectures");
            if c.is_dir() {
                return c;
            }
        }
    }
    PathBuf::from("data/architectures")
}

fn load_arch(spec: &str) -> Result<ArchDef, String> {
    let path = if spec.ends_with(".toml") {
        PathBuf::from(spec)
    } else {
        arch_dir().join(format!("{spec}.toml"))
    };
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "could not read architecture {spec:?} at {}: {e}\n\
             Run `df11pack architectures` to see the available names.",
            path.display()
        )
    })?;
    ArchDef::from_toml(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn list_architectures() -> Result<(), String> {
    let dir = arch_dir();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("could not list {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Err(format!("no definitions found in {}", dir.display()));
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
    for p in entries {
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let line = match ArchDef::from_toml(&text) {
            Ok(d) => format!(
                "{:<30} {:<16} {:>5}  {}",
                d.name,
                format!("{:?}", d.layout).to_lowercase(),
                d.units.len(),
                d.source
            ),
            Err(e) => format!("{:<30} INVALID: {e}", p.display()),
        };
        if !quiet(writeln!(w, "{line}"))? {
            return Ok(());
        }
    }
    Ok(())
}

fn compress(source: &Path, arch: &str, out: &Path, luts: LutModeArg) -> Result<(), String> {
    let def = load_arch(arch)?;
    let model = ModelSource::open(source).map_err(|e| format!("{}: {e}", source.display()))?;

    if luts == LutModeArg::Correct {
        eprintln!(
            "warning: --luts=correct produces output that is NOT byte-identical to the\n\
             official compressor. It zero-fills LUT positions the official encoder leaves\n\
             holding leaked state from the previous prefix table. The output is stamped\n\
             df11pack_luts=\"correct\" so it cannot be mistaken for compat output.\n\
             See docs/COMPATIBILITY.md before using this."
        );
    }

    let opts = WriteOptions {
        lut_mode: luts.into(),
    };
    let report = write_directory(&model, &def, out, &opts).map_err(|e| e.to_string())?;

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
    println!("  wrote {} -> {}", report.shards.len() + 1, out.display());
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Compress {
            source,
            arch,
            out,
            luts,
        } => compress(&source, &arch, &out, luts),
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
