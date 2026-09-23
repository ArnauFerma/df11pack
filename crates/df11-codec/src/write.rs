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
}

/// How many units to encode at once, given the budget and the largest unit.
///
/// A worker holds the exponent stream (N bytes), `sign_mantissa` (N) and the
/// encoded output (~0.34 N): about **2.34 N**. DESIGN §5.2's 1.35 N assumes the
/// exponents are re-derived on a second pass rather than kept, which is not done
/// yet — the constant here is what is actually held, measured, not the target.
/// Always at least one: a budget too small for a single unit still has to make
/// progress rather than refuse.
/// Bytes a worker holds per weight of the largest unit.
///
/// **Measured, not derived.** Compressing Qwen3-0.6B (largest unit 15,728,640
/// weights) peaks at 63.8 MiB with one worker and 230.2 MiB with four, so the
/// marginal cost of a worker is ~55 MiB and the first costs ~64 MiB — 3.7 to 4.25
/// bytes per weight. The higher figure is used so the budget errs toward fewer
/// workers.
///
/// The accounting: the exponent stream (N bytes), `sign_mantissa` (N), the
/// encoded output (~0.34 N), the chunked encoder's per-chunk buffers and window
/// tables, and one source tensor in flight. DESIGN §5.2's 1.35 N assumes the
/// exponents are re-derived on a second pass instead of kept; that is not done,
/// and this constant reflects what is actually held.
pub const BYTES_PER_WEIGHT_HELD: f64 = 4.25;

pub fn worker_count(opts: &WriteOptions, largest_unit_weights: u64, cores: usize) -> usize {
    if let Some(w) = opts.workers {
        return w.max(1);
    }
    let by_budget = match opts.ram_budget {
        Some(b) => {
            let per_worker = ((largest_unit_weights as f64) * BYTES_PER_WEIGHT_HELD).max(1.0);
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
    Encode { unit: String, source: EncodeError },
    St(StError),
    Io(std::io::Error),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discover(e) => write!(f, "{e}"),
            Self::Encode { unit, source } => write!(f, "unit {unit:?}: {source}"),
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
) -> Result<(usize, String, u64, Option<String>), WriteError> {
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
    let fname = shard_name(&u.name);
    let path = out_dir.join(&fname);
    write_file(&path, &out, meta)?;
    let sz = std::fs::metadata(&path)?.len();
    Ok((index, fname, sz, limited))
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
    type UnitResult = Result<(usize, String, u64, Option<String>), WriteError>;

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
                    source, u, ui, out_dir, threads, bpt, meta, read_gate, counters,
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
    collected.sort_by_key(|(i, _, _, _)| *i);
    for (_, fname, sz, limited) in collected {
        output_bytes += sz;
        if let Some(l) = limited {
            limited_units.push(l);
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
        max_concurrent_reads: peak_reads.load(std::sync::atomic::Ordering::SeqCst),
        config,
    })
}
