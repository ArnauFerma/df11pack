//! Writing a DF11 model directory.
#![allow(unused_imports)]

use crate::arch::{ArchDef, Layout};
use crate::config::{build_config, ConfigMode};
use crate::discover::{discover, DiscoverError};
use crate::huffman::LutMode;
use crate::io_sched::{plan, IoMode, IoPlan};
use crate::safetensors::{
    write_bytes_atomic, write_file, Dtype, OutTensor, Payload, StError, StreamingWriter, TensorDecl,
};
use crate::source::{ModelSource, View, COMFYUI_PREFIX};
use crate::unit::encode_unit_streaming;
use crate::EncodeError;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// How to compress.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Default `Compat`. `Correct` is not byte-identical, by design, and stamps
    /// the output; see `docs/COMPATIBILITY.md`.
    pub lut_mode: LutMode,
    /// Upper bound on resident memory, in bytes. Workers are limited so that
    /// roughly `1.35 x N_max` bytes per worker fits inside it (DESIGN 5.2).
    /// `None` means "use every core", which is only safe when memory is ample.
    pub ram_budget: Option<u64>,
    /// Force a worker count, overriding the budget calculation.
    pub workers: Option<usize>,
    /// Whether source reads may overlap. `Auto` detects the device.
    pub io: IoMode,
    /// Safe mode: decode every unit and check it against the source before its
    /// shard is written, so a wrong unit never reaches disk.
    ///
    /// Costs a second pass over the unit's source (2 N bytes held during the
    /// check) and the decode itself. That is the trade safe mode exists to make.
    pub verify: bool,
    /// Store a SHA-256 of each unit's six DF11 tensors in the shard's metadata,
    /// so `verify` can catch a wrong value without the source. Off by default:
    /// the header then differs from the official tool's (COMPATIBILITY.md).
    pub hashes: bool,
}

/// How many units to encode at once, given the budget and the largest unit.
///
/// A worker holds the exponent stream (N bytes), `sign_mantissa` (N) and the
/// encoded output (~0.34 N): about **2.34 N**. DESIGN §5.2's 1.35 N assumes the
/// exponents are re-derived on a second pass rather than kept, which is not done
/// yet — the constant here is what is actually held, measured, not the target.
/// Always at least one: a budget too small for a single unit still has to make
/// progress rather than refuse.
/// Bytes a worker holds per weight of the largest unit, in fast mode.
///
/// **Measured, and machine-dependent.** Qwen3-0.6B's largest unit is 15,728,640
/// weights. On the 4-thread i3 a worker cost 3.7–4.25 bytes per weight; on a
/// 128-thread EPYC, after intra-unit parallelism landed, one worker peaked at
/// 85.7 MiB, about 5.7. The difference is per-thread allocator state and chunk
/// buffers, which grow with core count. 6.0 covers both observations.
pub const BYTES_PER_WEIGHT_HELD: f64 = 6.0;

/// Extra bytes per weight that safe mode holds: the unit's source (2 N) plus
/// the decoder's window and gap tables. Measured at ~3.6 on the full model
/// (267 -> 492 MiB at four workers); 4.0 is used.
pub const SAFE_MODE_EXTRA_PER_WEIGHT: f64 = 4.0;

/// Share of available memory the default budget may use when `--ram` is absent.
pub const DEFAULT_MEMORY_FRACTION: f64 = 0.8;

/// Bytes per weight a worker holds under these options.
pub fn bytes_per_weight(opts: &WriteOptions) -> f64 {
    if opts.verify {
        BYTES_PER_WEIGHT_HELD + SAFE_MODE_EXTRA_PER_WEIGHT
    } else {
        BYTES_PER_WEIGHT_HELD
    }
}

/// `MemAvailable` from `/proc/meminfo`, in bytes, if it can be read.
pub fn available_memory() -> Option<u64> {
    parse_mem_available(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// `MemAvailable` from the text of `/proc/meminfo`, in bytes.
///
/// Separate from the file read so it can be tested on fixed input: against the
/// live file, any field with a plausible size -- `SwapFree`, say -- would pass.
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// How many units to encode at once. Reads available memory for the default.
pub fn worker_count(opts: &WriteOptions, largest_unit_weights: u64, cores: usize) -> usize {
    worker_count_with(opts, largest_unit_weights, cores, available_memory())
}

/// The decision itself, with available memory passed in so it can be tested.
///
/// Without `--ram`, the budget is a fraction of available memory rather than
/// "every core": on the 3.6 GB target machine, one worker per core for a
/// Flux-sized unit would need roughly 5.8 GB and be killed. Only when available
/// memory cannot be read does it fall back to the core count. Always at least
/// one worker, so a budget smaller than one unit still makes progress.
pub fn worker_count_with(
    opts: &WriteOptions,
    largest_unit_weights: u64,
    cores: usize,
    available: Option<u64>,
) -> usize {
    if let Some(w) = opts.workers {
        return w.max(1);
    }
    let budget = opts
        .ram_budget
        .or_else(|| available.map(|a| (a as f64 * DEFAULT_MEMORY_FRACTION) as u64));
    let by_budget = match budget {
        Some(b) => {
            let per_worker = ((largest_unit_weights as f64) * bytes_per_weight(opts)).max(1.0);
            ((b as f64) / per_worker).floor() as usize
        }
        None => cores,
    };
    by_budget.clamp(1, cores.max(1))
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct WriteReport {
    pub units: usize,
    pub shards: Vec<String>,
    pub remainder: String,
    pub source_bytes: u64,
    pub output_bytes: u64,
    /// Tensors dropped because they share storage with another, as the official
    /// compressor's `save_pretrained` does for tied embeddings.
    pub tied_dropped: Vec<String>,
    /// Units where the 32-bit limiter had to run.
    pub limited_units: Vec<String>,
    /// How reads were scheduled, and why.
    pub io: IoPlan,
    /// The key prefix stripped from the source, if any (`model.diffusion_model.`
    /// for a ComfyUI checkpoint).
    pub prefix_stripped: Option<String>,
    /// Units checked against their source before being written. Empty unless
    /// `verify` was set.
    pub verified: Vec<String>,
    /// The most source reads that were ever in flight at once.
    ///
    /// Reported so the scheduling decision is observable rather than merely
    /// declared: under `Sequential` this must be 1, and a missing gate shows up
    /// here instead of only as a timing difference.
    pub max_concurrent_reads: usize,
    /// The config file written, if the layout has one. ComfyUI-native output is
    /// a single file with no config, so this is `None` there.
    pub config: Option<String>,
}

#[derive(Debug)]
pub enum WriteError {
    Discover(DiscoverError),
    Encode {
        unit: String,
        source: EncodeError,
    },
    /// Safe mode found a unit that does not decode back to its source.
    Verify {
        unit: String,
        source: crate::verify::VerifyError,
    },
    St(StError),
    Io(std::io::Error),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discover(e) => write!(f, "{e}"),
            Self::Encode { unit, source } => write!(f, "unit {unit:?}: {source}"),
            Self::Verify { unit, source } => write!(
                f,
                "unit {unit:?} failed verification and was NOT written: {source}"
            ),
            Self::St(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<DiscoverError> for WriteError {
    fn from(e: DiscoverError) -> Self {
        Self::Discover(e)
    }
}
impl From<StError> for WriteError {
    fn from(e: StError) -> Self {
        Self::St(e)
    }
}
impl From<std::io::Error> for WriteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// One unit's tensors, ready to write: the six DF11 tensors followed by the
/// siblings that travel with it.
struct Built {
    tensors: Vec<OutTensor>,
    limited: Option<String>,
    verified: bool,
}

type Reader<'r> = dyn Fn(&str) -> Result<Vec<u8>, StError> + Sync + 'r;

/// Encode one unit and, in safe mode, check it against its source.
///
/// Produces the tensors but writes nothing, so the directory and single-file
/// layouts can share it. The chunked encoder inside uses rayon's global pool, so
/// a single unit can still occupy every core.
fn build_unit(
    src: &View,
    u: &crate::discover::DiscoveredUnit,
    threads: usize,
    bpt: usize,
    read: &Reader,
    verify: bool,
) -> Result<Built, WriteError> {
    // Sizes come from the header; no tensor data is read yet.
    let counts: Vec<u64> = u
        .tensors
        .iter()
        .map(|n| src.info(n).map(|i| i.nbytes() / 2).unwrap_or(0))
        .collect();
    let enc = encode_unit_streaming(&u.name, &counts, |i| read(&u.tensors[i]), threads, bpt)
        .map_err(|e| WriteError::Encode {
            unit: u.name.clone(),
            source: e,
        })?;
    let limited = (enc.limiter_iterations > 0).then(|| u.name.clone());

    // Safe mode checks the unit BEFORE anything is written, so a wrong one never
    // reaches disk. `verified` records that the check RAN, not that it was
    // requested: the report is evidence, not an echo of the flag.
    let mut verified = false;
    if verify {
        let mut source_bytes: Vec<u8> = Vec::with_capacity((enc.weights() * 2) as usize);
        for name in &u.tensors {
            source_bytes.extend_from_slice(&read(name)?);
        }
        let luts: Vec<u8> = enc.luts.iter().flat_map(|r| r.iter().copied()).collect();
        let view = crate::verify::UnitView {
            luts: &luts,
            encoded_exponent: &enc.encoded_exponent,
            sign_mantissa: &enc.sign_mantissa,
            output_positions: &enc.output_positions,
            gaps: &enc.gaps,
            bytes_per_thread: bpt,
            threads_per_block: threads,
        };
        crate::verify::verify_unit(&view, &source_bytes).map_err(|e| WriteError::Verify {
            unit: u.name.clone(),
            source: e,
        })?;
        verified = true;
    }

    let mut tensors: Vec<OutTensor> = Vec::with_capacity(6 + u.siblings.len());
    let rows = enc.luts.len() as u64;
    for (name, data) in enc.tensors() {
        let (dtype, shape) = if name.ends_with(".luts") {
            (Dtype::new(Dtype::U8), vec![rows, 256])
        } else if name.ends_with(".split_positions") {
            (Dtype::new(Dtype::I64), vec![(data.len() / 8) as u64])
        } else {
            (Dtype::new(Dtype::U8), vec![data.len() as u64])
        };
        tensors.push(OutTensor::owned(name, dtype, shape, data));
    }
    // Siblings travel with their unit, as upstream does (FINDINGS 0.7).
    for s in &u.siblings {
        tensors.push(borrowed(src, s));
    }
    Ok(Built {
        tensors,
        limited,
        verified,
    })
}

/// Metadata key prefix for a tensor's SHA-256, and the stamp that says an
/// output carries them.
pub const HASH_KEY_PREFIX: &str = "df11pack_sha256:";
pub const HASH_STAMP: &str = "df11pack_hashes";

/// Lowercase hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Add the hash of every in-memory tensor -- the six DF11 tensors; siblings are
/// borrowed copies of the source -- to `meta`.
fn add_hashes(meta: &mut BTreeMap<String, String>, tensors: &[OutTensor]) {
    meta.insert(HASH_STAMP.to_string(), "sha256".to_string());
    for t in tensors {
        if let Payload::Owned(data) = &t.data {
            meta.insert(format!("{HASH_KEY_PREFIX}{}", t.name), sha256_hex(data));
        }
    }
}

/// A source tensor copied by byte range rather than read into memory.
fn borrowed(src: &View, name: &str) -> OutTensor {
    let info = src.info(name).expect("discovered from this source").clone();
    let (path, offset, len) = src.locate(name).expect("discovered from this source");
    OutTensor {
        name: name.to_string(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data: Payload::Borrowed { path, offset, len },
    }
}

fn decls_of(ts: &[OutTensor]) -> Vec<TensorDecl> {
    ts.iter()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            len: t.data.len(),
        })
        .collect()
}

/// Run `f` over every unit, at most `workers` at a time, and return the results
/// in definition order.
///
/// `workers` bounds how many units are in flight, because each one holds
/// memory. It is deliberately not the CPU limit: the encoder inside a unit uses
/// rayon's global pool. Sizing one pool for both jobs is what capped throughput
/// at 16 workers in the scaling measurement.
fn run_units<T: Send>(
    units: &[crate::discover::DiscoveredUnit],
    workers: usize,
    f: &(dyn Fn(&crate::discover::DiscoveredUnit) -> Result<T, WriteError> + Sync),
) -> Result<Vec<T>, WriteError> {
    let permits = std::sync::Mutex::new(workers);
    let cv = std::sync::Condvar::new();
    let results: std::sync::Mutex<Vec<(usize, Result<T, WriteError>)>> =
        std::sync::Mutex::new(Vec::with_capacity(units.len()));

    std::thread::scope(|scope| {
        for (i, u) in units.iter().enumerate() {
            {
                let mut n = permits.lock().expect("permits");
                while *n == 0 {
                    n = cv.wait(n).expect("permits");
                }
                *n -= 1;
            }
            let (permits, cv, results) = (&permits, &cv, &results);
            scope.spawn(move || {
                let r = f(u);
                results.lock().expect("results").push((i, r));
                *permits.lock().expect("permits") += 1;
                cv.notify_one();
            });
        }
    });

    let mut done = results.into_inner().expect("results");
    // Restore definition order, which completion order does not preserve.
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

/// The official shard name for a unit: dots become underscores.
pub fn shard_name(unit: &str) -> String {
    format!("{}.safetensors", unit.replace('.', "_"))
}

/// The file holding everything outside any unit -- or, for ComfyUI-native,
/// everything.
pub fn remainder_name(layout: Layout) -> &'static str {
    match layout {
        Layout::Diffusers | Layout::DiffusersSingle => "diffusion_pytorch_model.safetensors",
        _ => "model.safetensors",
    }
}

/// Compress a model into a DF11 output: a directory of shards for the
/// transformers and diffusers layouts, a single file for ComfyUI-native and
/// diffusers-single (the latter with a `config.json` beside it).
pub fn write_directory(
    source: &ModelSource,
    def: &ArchDef,
    out_dir: &Path,
    opts: &WriteOptions,
) -> Result<WriteReport, WriteError> {
    std::fs::create_dir_all(out_dir)?;

    // ComfyUI checkpoints carry `model.diffusion_model.` on every diffusion key;
    // the definitions do not. Strip it when present (DESIGN 5.5).
    let prefix = (def.layout == Layout::ComfyuiNative
        && source.names().iter().any(|n| n.starts_with(COMFYUI_PREFIX)))
    .then_some(COMFYUI_PREFIX);
    let src = View::new(source, prefix);

    let names: Vec<String> = src.names();
    let found = discover(def, &names)?;
    let source_bytes = src.total_bytes();

    let threads = *def.threads_per_block.first().unwrap_or(&512) as usize;
    let bpt = def.bytes_per_thread as usize;

    let largest = found
        .units
        .iter()
        .map(|u| {
            u.tensors
                .iter()
                .map(|n| src.info(n).map(|i| i.nbytes() / 2).unwrap_or(0))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = worker_count(opts, largest, cores);

    // Header metadata, as upstream writes it: unit shards and the single file come
    // from a bare `save_file(state_dict)` and carry none; only the remainder, which
    // upstream writes with the library's `save_pretrained`, has `format: pt`.
    let mut meta = BTreeMap::new();
    if opts.lut_mode == LutMode::Correct {
        // Never let a deliberately non-byte-identical file be mistaken for one
        // that is. See docs/COMPATIBILITY.md.
        meta.insert("df11pack_luts".to_string(), "correct".to_string());
    }
    let mut remainder_meta = meta.clone();
    remainder_meta.insert("format".to_string(), "pt".to_string());

    // On a spinning disk, overlapping readers make the head seek; one reader at
    // a time is much faster. Encoding still overlaps -- only the reads queue.
    let io = plan(src.dir(), opts.io);
    let read_gate: Option<std::sync::Mutex<()>> = io.sequential.then(|| std::sync::Mutex::new(()));
    let in_flight = std::sync::atomic::AtomicUsize::new(0);
    let peak_reads = std::sync::atomic::AtomicUsize::new(0);
    let read = |name: &str| -> Result<Vec<u8>, StError> {
        use std::sync::atomic::Ordering;
        let _held = read_gate.as_ref().map(|m| m.lock().expect("read gate"));
        let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        peak_reads.fetch_max(n, Ordering::SeqCst);
        let r = src.read(name);
        in_flight.fetch_sub(1, Ordering::SeqCst);
        r
    };

    // The source config drives the tie check and the output config.
    let source_config: Option<serde_json::Value> = Some(src.dir().join("config.json"))
        .filter(|p| p.is_file())
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok());

    // Tied tensors: upstream's save_pretrained refuses to write two tensors that
    // share storage, and drops the tied view. The tie is a property of the model
    // config, so that is where we read it from -- not guessed from the bytes.
    let mut tied_dropped = Vec::new();
    let tied = source_config
        .as_ref()
        .and_then(|v| v.get("tie_word_embeddings").and_then(|t| t.as_bool()))
        .unwrap_or(false);
    if tied {
        const TIED_VIEW: &str = "lm_head.weight";
        const TIED_SOURCE: &str = "model.embed_tokens.weight";
        if found.passthrough.iter().any(|n| n == TIED_VIEW)
            && found.passthrough.iter().any(|n| n == TIED_SOURCE)
            && src.tensors_equal(TIED_VIEW, TIED_SOURCE)?
        {
            tied_dropped.push(TIED_VIEW.to_string());
        }
    }
    let passthrough: Vec<&String> = found
        .passthrough
        .iter()
        .filter(|n| !tied_dropped.contains(n))
        .collect();

    let remainder = remainder_name(def.layout).to_string();
    let mut shards = Vec::new();
    let mut limited_units = Vec::new();
    let mut verified = Vec::new();
    let mut output_bytes: u64 = 0;

    if def.layout.single_file() {
        // One file. Its header must list every tensor's byte range before any
        // data is written, but encoded sizes are only known after encoding.
        // Holding every unit in memory would break the budget, and staging to
        // temporary shards would double the disk needed. So: encode each unit
        // once to learn its sizes, write the header, then encode again in order
        // and stream straight to disk. The encoder is deterministic, and the
        // writer checks every tensor against its declared size, so a mismatch
        // is an error rather than a corrupt file.
        //
        // The library's layout order (dtype descending, then name) interleaves
        // units: every I64 `split_positions` first, then the BF16 siblings and
        // passthrough, then each unit's U8 tensors. So pass one keeps each unit's
        // small tensors, and pass two encodes a unit once, when its first large
        // tensor comes up; a unit's U8 tensors are contiguous in that order.
        const KEEP: u64 = 1 << 20;
        let pass1 = run_units(&found.units, workers, &|u| {
            let b = build_unit(&src, u, threads, bpt, &read, false)?;
            // Hashed here, before the header: pass two must then write exactly
            // these bytes, which `verify` will confirm.
            let mut h = BTreeMap::new();
            if opts.hashes {
                add_hashes(&mut h, &b.tensors);
            }
            let small: Vec<(String, Vec<u8>)> = b
                .tensors
                .iter()
                .filter_map(|t| match &t.data {
                    Payload::Owned(v) if (v.len() as u64) <= KEEP => {
                        Some((t.name.clone(), v.clone()))
                    }
                    _ => None,
                })
                .collect();
            Ok((decls_of(&b.tensors), b.limited, h, small))
        })?;
        let mut decls = Vec::new();
        let mut meta = meta.clone();
        // Which unit made each tensor we must produce, and the small ones' bytes.
        let mut owner: BTreeMap<String, usize> = BTreeMap::new();
        let mut kept: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (ui, (d, limited, h, small)) in pass1.into_iter().enumerate() {
            meta.extend(h);
            for decl in &d {
                owner.insert(decl.name.clone(), ui);
            }
            decls.extend(d);
            kept.extend(small);
            if let Some(l) = limited {
                limited_units.push(l);
            }
        }
        let rem: Vec<OutTensor> = passthrough.iter().map(|n| borrowed(&src, n)).collect();
        decls.extend(decls_of(&rem));
        // Everything copied from the source -- passthrough and every unit's
        // siblings -- is written straight from it, never via its unit.
        let siblings: Vec<OutTensor> = found
            .units
            .iter()
            .flat_map(|u| u.siblings.iter().map(|n| borrowed(&src, n)))
            .collect();
        let rem_by_name: BTreeMap<&str, &OutTensor> = rem
            .iter()
            .chain(siblings.iter())
            .map(|t| (t.name.as_str(), t))
            .collect();

        let path = out_dir.join(&remainder);
        let mut w = StreamingWriter::begin(&path, decls, &meta)?;
        let order: Vec<String> = w.order().iter().map(|d| d.name.clone()).collect();
        let mut current: Option<(usize, BTreeMap<String, OutTensor>)> = None;
        let mut encoded = vec![false; found.units.len()];
        for name in &order {
            if let Some(t) = rem_by_name.get(name.as_str()) {
                w.put(t)?;
                continue;
            }
            let ui = owner[name];
            if let Some(v) = kept.get(name) {
                w.write(name, v)?;
                continue;
            }
            if current.as_ref().map(|(i, _)| *i) != Some(ui) {
                // Only a large encoded tensor brings a unit in.
                if encoded[ui] {
                    return Err(WriteError::Io(std::io::Error::other(format!(
                        "unit {} needed twice while streaming; its tensors are not contiguous",
                        found.units[ui].name
                    ))));
                }
                let b = build_unit(&src, &found.units[ui], threads, bpt, &read, opts.verify)?;
                if b.verified {
                    verified.push(found.units[ui].name.clone());
                }
                encoded[ui] = true;
                // The small tensors already written came from pass one; the
                // encoder is deterministic, and this makes sure of it.
                for t in &b.tensors {
                    if let (Payload::Owned(v), Some(k)) = (&t.data, kept.get(&t.name)) {
                        if v != k {
                            return Err(WriteError::Io(std::io::Error::other(format!(
                                "{}: second encoding differs from the first",
                                t.name
                            ))));
                        }
                    }
                }
                current = Some((
                    ui,
                    b.tensors.into_iter().map(|t| (t.name.clone(), t)).collect(),
                ));
            }
            let t = &current.as_ref().expect("just set").1[name];
            w.put(t)?;
        }
        // A unit whose every tensor was small was never re-encoded; safe mode
        // must still check it.
        if opts.verify {
            for (ui, u) in found.units.iter().enumerate() {
                if !encoded[ui] {
                    let b = build_unit(&src, u, threads, bpt, &read, true)?;
                    for t in &b.tensors {
                        if let (Payload::Owned(v), Some(k)) = (&t.data, kept.get(&t.name)) {
                            if v != k {
                                return Err(WriteError::Io(std::io::Error::other(format!(
                                    "{}: second encoding differs from the first",
                                    t.name
                                ))));
                            }
                        }
                    }
                    verified.push(u.name.clone());
                }
            }
        }
        w.finish()?;
        output_bytes += std::fs::metadata(&path)?.len();
    } else {
        let written = run_units(&found.units, workers, &|u| {
            let b = build_unit(&src, u, threads, bpt, &read, opts.verify)?;
            let fname = shard_name(&u.name);
            let path = out_dir.join(&fname);
            let mut meta = meta.clone();
            if opts.hashes {
                add_hashes(&mut meta, &b.tensors);
            }
            write_file(&path, &b.tensors, &meta)?;
            let sz = std::fs::metadata(&path)?.len();
            Ok((fname, sz, b.limited, b.verified.then(|| u.name.clone())))
        })?;
        for (fname, sz, limited, was_verified) in written {
            output_bytes += sz;
            limited_units.extend(limited);
            verified.extend(was_verified);
            shards.push(fname);
        }
        let rem: Vec<OutTensor> = passthrough.iter().map(|n| borrowed(&src, n)).collect();
        let rpath = out_dir.join(&remainder);
        write_file(&rpath, &rem, &remainder_meta)?;
        output_bytes += std::fs::metadata(&rpath)?.len();
    }
    limited_units.sort();

    let config = match def.layout {
        // ComfyUI-native output is a single file and carries no config; the
        // Extended node discards it (DESIGN 5.5).
        Layout::ComfyuiNative => None,
        layout => {
            let mode = if matches!(layout, Layout::Diffusers | Layout::DiffusersSingle) {
                // What every published diffusers release actually ships.
                ConfigMode::Minimal
            } else {
                ConfigMode::PreserveSource
            };
            let cfg = build_config(source_config.as_ref(), def, mode);
            let path = out_dir.join("config.json");
            write_bytes_atomic(&path, &serde_json::to_vec_pretty(&cfg).unwrap_or_default())?;
            output_bytes += std::fs::metadata(&path)?.len();
            Some("config.json".to_string())
        }
    };

    // A generation config the model author shipped is carried over byte for
    // byte. One is never invented: upstream's is synthesised by transformers from
    // config.json and stamped with the installed version, and transformers
    // regenerates it from the config when it is absent.
    if def.layout == Layout::Transformers {
        let g = src.dir().join("generation_config.json");
        if g.is_file() {
            let path = out_dir.join("generation_config.json");
            write_bytes_atomic(&path, &std::fs::read(&g)?)?;
            output_bytes += std::fs::metadata(&path)?.len();
        }
    }

    Ok(WriteReport {
        units: found.units.len(),
        shards,
        remainder,
        source_bytes,
        output_bytes,
        tied_dropped,
        limited_units,
        io,
        prefix_stripped: src.prefix().map(str::to_string),
        verified,
        max_concurrent_reads: peak_reads.load(std::sync::atomic::Ordering::SeqCst),
        config,
    })
}
