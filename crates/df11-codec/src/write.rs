//! Writing a DF11 model directory.
#![allow(unused_imports)]

use crate::arch::{ArchDef, Layout};
use crate::config::{build_config, ConfigMode};
use crate::discover::{discover, DiscoverError};
use crate::huffman::LutMode;
use crate::io_sched::{plan, IoMode, IoPlan};
use crate::safetensors::{write_file, Dtype, OutTensor, Payload, StError};
use crate::source::ModelSource;
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

/// Encode one unit and write its shard. Runs on its own thread; the chunked
/// encoder inside uses rayon's global pool, so a single unit can still occupy
/// every core.
#[allow(clippy::too_many_arguments)]
fn encode_one_unit(
    source: &ModelSource,
    u: &crate::discover::DiscoveredUnit,
    index: usize,
    out_dir: &std::path::Path,
    threads: usize,
    bpt: usize,
    meta: &BTreeMap<String, String>,
    read_gate: &Option<std::sync::Mutex<()>>,
    counters: (
        &std::sync::atomic::AtomicUsize,
        &std::sync::atomic::AtomicUsize,
    ),
    verify: bool,
) -> Result<(usize, String, u64, Option<String>, bool), WriteError> {
    use std::sync::atomic::Ordering;
    let (in_flight, peak) = counters;
    let read = |name: &str| -> Result<Vec<u8>, crate::safetensors::StError> {
        let _held = read_gate.as_ref().map(|m| m.lock().expect("read gate"));
        let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(n, Ordering::SeqCst);
        let r = source.read(name);
        in_flight.fetch_sub(1, Ordering::SeqCst);
        r
    };
    // Sizes come from the header; no tensor data is read yet.
    let counts: Vec<u64> = u
        .tensors
        .iter()
        .map(|n| source.info(n).map(|i| i.nbytes() / 2).unwrap_or(0))
        .collect();
    let enc = encode_unit_streaming(&u.name, &counts, |i| read(&u.tensors[i]), threads, bpt)
        .map_err(|e| WriteError::Encode {
            unit: u.name.clone(),
            source: e,
        })?;
    let limited = (enc.limiter_iterations > 0).then(|| u.name.clone());

    let mut out: Vec<OutTensor> = Vec::with_capacity(6 + u.siblings.len());
    let rows = enc.luts.len() as u64;
    for (name, data) in enc.tensors() {
        let (dtype, shape) = if name.ends_with(".luts") {
            (Dtype::new(Dtype::U8), vec![rows, 256])
        } else if name.ends_with(".split_positions") {
            (Dtype::new(Dtype::I64), vec![(data.len() / 8) as u64])
        } else {
            (Dtype::new(Dtype::U8), vec![data.len() as u64])
        };
        out.push(OutTensor::owned(name, dtype, shape, data));
    }
    // Siblings travel with their unit, as upstream does (FINDINGS 0.7).
    for s in &u.siblings {
        let info = source.info(s).expect("discovered from this file").clone();
        let (path, offset, len) = source.locate(s).expect("discovered from this file");
        out.push(OutTensor {
            name: s.clone(),
            dtype: info.dtype.clone(),
            shape: info.shape.clone(),
            data: Payload::Borrowed { path, offset, len },
        });
    }
    // Safe mode checks the unit BEFORE it is written, so a wrong one never
    // reaches disk at all -- there is nothing to clean up afterwards.
    //
    // `did_verify` records that the check RAN, not that it was requested. The
    // report is then evidence rather than an echo of the flag: if this block is
    // skipped, nothing downstream claims the unit was checked.
    let mut did_verify = false;
    if verify {
        let mut source_bytes: Vec<u8> = Vec::with_capacity((enc.weights() * 2) as usize);
        for name in &u.tensors {
            source_bytes.extend_from_slice(&read(name)?);
        }
        let pos = enc.output_positions.clone();
        let luts: Vec<u8> = enc.luts.iter().flat_map(|r| r.iter().copied()).collect();
        let view = crate::verify::UnitView {
            luts: &luts,
            encoded_exponent: &enc.encoded_exponent,
            sign_mantissa: &enc.sign_mantissa,
            output_positions: &pos,
            gaps: &enc.gaps,
            bytes_per_thread: bpt,
            threads_per_block: threads,
        };
        crate::verify::verify_unit(&view, &source_bytes).map_err(|e| WriteError::Verify {
            unit: u.name.clone(),
            source: e,
        })?;
        did_verify = true;
    }

    let fname = shard_name(&u.name);
    let path = out_dir.join(&fname);
    write_file(&path, &out, meta)?;
    let sz = std::fs::metadata(&path)?.len();
    Ok((index, fname, sz, limited, did_verify))
}

/// The official shard name for a unit: dots become underscores.
pub fn shard_name(unit: &str) -> String {
    format!("{}.safetensors", unit.replace('.', "_"))
}

/// The file holding everything outside any unit.
pub fn remainder_name(layout: Layout) -> &'static str {
    match layout {
        Layout::Diffusers => "diffusion_pytorch_model.safetensors",
        _ => "model.safetensors",
    }
}

/// Compress a model into a DF11 directory.
pub fn write_directory(
    source: &ModelSource,
    def: &ArchDef,
    out_dir: &Path,
    opts: &WriteOptions,
) -> Result<WriteReport, WriteError> {
    std::fs::create_dir_all(out_dir)?;
    let names: Vec<String> = source.names();
    let found = discover(def, &names)?;

    let source_bytes = source.total_bytes();

    let threads = *def.threads_per_block.first().unwrap_or(&512) as usize;
    let bpt = def.bytes_per_thread as usize;

    let largest = found
        .units
        .iter()
        .map(|u| {
            u.tensors
                .iter()
                .map(|n| source.info(n).map(|i| i.nbytes() / 2).unwrap_or(0))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = worker_count(opts, largest, cores);

    let mut shards = Vec::new();
    let mut limited_units = Vec::new();
    let mut output_bytes: u64 = 0;

    let mut meta = BTreeMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    if opts.lut_mode == LutMode::Correct {
        // Never let a deliberately non-byte-identical file be mistaken for one
        // that is. See docs/COMPATIBILITY.md.
        meta.insert("df11pack_luts".to_string(), "correct".to_string());
    }

    // Two different limits, deliberately not the same number.
    //
    // `workers` bounds how many units are IN FLIGHT, because each one holds
    // memory. CPU parallelism is a separate axis: the chunked encoder inside a
    // unit uses rayon's global pool, i.e. every core. Sizing one pool to do both
    // jobs is what capped throughput at 16 in the scaling measurement -- beyond
    // that, more units in flight bought memory pressure rather than speed, while
    // a model with few large units could not use the cores at all.
    type UnitResult = Result<(usize, String, u64, Option<String>, bool), WriteError>;

    // On a spinning disk, overlapping readers make the head seek; one reader at
    // a time is much faster. Encoding still overlaps -- only the reads queue.
    let io = plan(source.dir(), opts.io);
    let read_gate: Option<std::sync::Mutex<()>> = io.sequential.then(|| std::sync::Mutex::new(()));
    let read_gate = &read_gate;
    let in_flight = std::sync::atomic::AtomicUsize::new(0);
    let peak_reads = std::sync::atomic::AtomicUsize::new(0);
    let counters = (&in_flight, &peak_reads);

    let permits = std::sync::Arc::new(std::sync::Mutex::new(workers));
    let cv = std::sync::Arc::new(std::sync::Condvar::new());
    let results: std::sync::Mutex<Vec<UnitResult>> = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for (ui, u) in found.units.iter().enumerate() {
            // Block until a memory permit is free, then start this unit.
            {
                let mut n = permits.lock().expect("permits");
                while *n == 0 {
                    n = cv.wait(n).expect("permits");
                }
                *n -= 1;
            }
            let permits = std::sync::Arc::clone(&permits);
            let cv = std::sync::Arc::clone(&cv);
            let results = &results;
            let meta = &meta;
            scope.spawn(move || {
                let r = encode_one_unit(
                    source,
                    u,
                    ui,
                    out_dir,
                    threads,
                    bpt,
                    meta,
                    read_gate,
                    counters,
                    opts.verify,
                );
                results.lock().expect("results").push(r);
                *permits.lock().expect("permits") += 1;
                cv.notify_one();
            });
        }
    });

    let mut done = results.into_inner().expect("results");
    let mut collected = Vec::with_capacity(done.len());
    for r in done.drain(..) {
        collected.push(r?);
    }
    // Restore definition order, which the completion order does not preserve.
    collected.sort_by_key(|(i, _, _, _, _)| *i);
    let mut verified = Vec::new();
    for (i, fname, sz, limited, was_verified) in collected {
        output_bytes += sz;
        if let Some(l) = limited {
            limited_units.push(l);
        }
        if was_verified {
            verified.push(found.units[i].name.clone());
        }
        shards.push(fname);
    }
    limited_units.sort();

    // Tied tensors: upstream's save_pretrained refuses to write two tensors that
    // share storage, and drops the tied view. The tie is a property of the model
    // config, so that is where we read it from -- not guessed from the bytes.
    let mut tied_dropped = Vec::new();
    let tied = Some(source.dir().join("config.json"))
        .filter(|p| p.is_file())
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.get("tie_word_embeddings").and_then(|t| t.as_bool()))
        .unwrap_or(false);
    if tied {
        const TIED_VIEW: &str = "lm_head.weight";
        const TIED_SOURCE: &str = "model.embed_tokens.weight";
        if found.passthrough.iter().any(|n| n == TIED_VIEW)
            && found.passthrough.iter().any(|n| n == TIED_SOURCE)
            && source.tensors_equal(TIED_VIEW, TIED_SOURCE)?
        {
            tied_dropped.push(TIED_VIEW.to_string());
        }
    }

    let mut rem: Vec<OutTensor> = Vec::new();
    for n in &found.passthrough {
        if tied_dropped.contains(n) {
            continue;
        }
        let info = source.info(n).expect("from this file").clone();
        let (path, offset, len) = source.locate(n).expect("from this file");
        rem.push(OutTensor {
            name: n.clone(),
            dtype: info.dtype.clone(),
            shape: info.shape.clone(),
            data: Payload::Borrowed { path, offset, len },
        });
    }
    // The source config drives both the tie check above and the output config.
    let source_config: Option<serde_json::Value> = Some(source.dir().join("config.json"))
        .filter(|p| p.is_file())
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok());

    let config = match def.layout {
        // ComfyUI-native output is a single file and carries no config; the
        // Extended node discards it (DESIGN 5.5).
        Layout::ComfyuiNative => None,
        layout => {
            let mode = if layout == Layout::Diffusers {
                // What every published diffusers release actually ships.
                ConfigMode::Minimal
            } else {
                ConfigMode::PreserveSource
            };
            let cfg = build_config(source_config.as_ref(), def, mode);
            let path = out_dir.join("config.json");
            std::fs::write(&path, serde_json::to_vec_pretty(&cfg).unwrap_or_default())?;
            output_bytes += std::fs::metadata(&path)?.len();
            Some("config.json".to_string())
        }
    };

    let remainder = remainder_name(def.layout).to_string();
    let rpath = out_dir.join(&remainder);
    write_file(&rpath, &rem, &meta)?;
    output_bytes += std::fs::metadata(&rpath)?.len();

    Ok(WriteReport {
        units: found.units.len(),
        shards,
        remainder,
        source_bytes,
        output_bytes,
        tied_dropped,
        limited_units,
        io,
        verified,
        max_concurrent_reads: peak_reads.load(std::sync::atomic::Ordering::SeqCst),
        config,
    })
}
