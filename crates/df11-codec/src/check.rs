//! Post-hoc verification of a written output. DESIGN §7.3, adapted.
//!
//! Three levels:
//!
//! - **Integrity** needs no source model. It finds the units in the output
//!   itself and checks their structure: shapes, index monotonicity, the per-chunk
//!   delta bounds, LUT jump targets, split positions. DESIGN defined this level
//!   by journal hashes; the journal was dropped with resume in Phase 4, so this
//!   level now catches *structural* damage but not a value flipped inside an
//!   otherwise well-formed tensor. That needs the source.
//! - **Sample** decodes a stratified, seeded set of kernel chunks and compares
//!   them with the source.
//! - **Full** decodes everything.

use crate::arch::ArchDef;
use crate::discover::discover;
use crate::sample::{chunk_of, plan, UnitShape};
use crate::source::{ModelSource, View};
use crate::verify::{chunk_range, verify_chunk, verify_unit, UnitView};
use crate::MAX_PREFIX_TABLES;
use std::collections::BTreeMap;

/// How thoroughly to check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Integrity,
    Sample { budget: usize, seed: u64 },
    Full,
}

/// One thing that is wrong, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub unit: String,
    /// The kernel chunk, when the check was chunk-level.
    pub chunk: Option<usize>,
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct CheckReport {
    pub units: usize,
    /// `(unit, chunk)` for every chunk decoded against the source.
    pub checked: Vec<(String, usize)>,
    /// Stored per-tensor hashes that were compared (outputs written with
    /// `--hashes`). Zero means a value error was not looked for without a source.
    pub hashed: usize,
    /// The seed a sampled run used, so it can be reproduced exactly.
    pub seed: Option<u64>,
    pub failures: Vec<Failure>,
}

impl CheckReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug)]
pub enum CheckError {
    /// Sample and full levels compare against the source; one must be given.
    NeedsSource,
    Other(String),
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NeedsSource => write!(
                f,
                "this level decodes against the source model; pass --source and --arch"
            ),
            Self::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CheckError {}

/// Kernel geometry: bytes per thread and threads per block.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    bpt: usize,
    tpb: usize,
}

impl Geometry {
    fn chunk_bytes(self) -> usize {
        self.bpt * self.tpb
    }
}

/// A unit's tensors, read from the output.
struct Loaded {
    luts: Vec<u8>,
    enc: Vec<u8>,
    sm: Vec<u8>,
    pos: Vec<u32>,
    gaps: Vec<u8>,
    split: Vec<i64>,
    /// How many stored hashes were compared, and the first that did not match.
    hashed: usize,
    hash_error: Option<String>,
}

impl Loaded {
    fn read(out: &ModelSource, unit: &str) -> Result<Self, String> {
        use crate::write::{sha256_hex, HASH_KEY_PREFIX};
        let hashed = std::cell::Cell::new(0usize);
        let hash_error = std::cell::RefCell::new(None);
        let g = |f: &str| {
            let name = format!("{unit}.{f}");
            if out.info(&name).is_none() {
                return Err(format!("{name} is missing from the output"));
            }
            let bytes = out.read(&name).map_err(|e| format!("{name}: {e}"))?;
            let stored = out
                .metadata_of(&name)
                .and_then(|m| m.get(&format!("{HASH_KEY_PREFIX}{name}")));
            if let Some(want) = stored {
                hashed.set(hashed.get() + 1);
                let got = sha256_hex(&bytes);
                if &got != want && hash_error.borrow().is_none() {
                    *hash_error.borrow_mut() = Some(format!(
                        "{name}: SHA-256 {got} does not match the stored {want}"
                    ));
                }
            }
            Ok(bytes)
        };
        let pos = g("output_positions")?;
        let split = g("split_positions")?;
        if pos.len() % 4 != 0 || split.len() % 8 != 0 {
            return Err("output_positions or split_positions has a partial element".into());
        }
        Ok(Loaded {
            luts: g("luts")?,
            enc: g("encoded_exponent")?,
            sm: g("sign_mantissa")?,
            pos: pos
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            gaps: g("gaps")?,
            split: split
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            hashed: hashed.get(),
            hash_error: hash_error.into_inner(),
        })
    }

    fn view(&self, geo: Geometry) -> UnitView<'_> {
        UnitView {
            luts: &self.luts,
            encoded_exponent: &self.enc,
            sign_mantissa: &self.sm,
            output_positions: &self.pos,
            gaps: &self.gaps,
            bytes_per_thread: geo.bpt,
            threads_per_block: geo.tpb,
        }
    }
}

/// Everything that can be checked about a unit without its source.
///
/// These are the invariants the kernel depends on to index safely: a violation
/// here is an out-of-bounds read or a garbage weight on the GPU, not a crash
/// anywhere a user would see it.
fn structure(l: &Loaded, geo: Geometry) -> Result<(), String> {
    // LUTs: whole rows, at least one prefix table plus the lengths row, and every
    // jump pointing at a table that exists.
    if l.luts.len() % 256 != 0 || l.luts.len() < 512 {
        return Err(format!(
            "luts is {} bytes, not whole 256-byte rows",
            l.luts.len()
        ));
    }
    let tables = l.luts.len() / 256 - 1;
    if tables > MAX_PREFIX_TABLES {
        return Err(format!(
            "{tables} prefix tables, over the limit of {MAX_PREFIX_TABLES}"
        ));
    }
    for (t, row) in l.luts[..tables * 256].chunks_exact(256).enumerate() {
        for (b, &v) in row.iter().enumerate() {
            if v >= 240 && 256 - v as usize >= tables {
                return Err(format!(
                    "luts[{t}][{b}] = {v} jumps to table {}, but there are {tables}",
                    256 - v as usize
                ));
            }
        }
    }

    // Sizes: the chunk count follows from the bitstream, and output_positions
    // and gaps are sized from it.
    let n = l.sm.len();
    let chunks = l.enc.len().div_ceil(geo.chunk_bytes());
    if l.pos.len() != chunks + 1 {
        return Err(format!(
            "output_positions has {} entries; a {}-byte bitstream has {chunks} chunks, so {}",
            l.pos.len(),
            l.enc.len(),
            chunks + 1
        ));
    }
    let gap_bytes = (chunks * geo.tpb * 5).div_ceil(8);
    if l.gaps.len() != gap_bytes {
        return Err(format!(
            "gaps is {} bytes; {chunks} chunks need {gap_bytes}",
            l.gaps.len()
        ));
    }

    // output_positions: from 0 to n, never decreasing, and no chunk claiming more
    // elements than it has bits (every code is at least one bit long).
    if l.pos.first() != Some(&0) {
        return Err(format!("output_positions[0] = {:?}, not 0", l.pos.first()));
    }
    if *l.pos.last().unwrap() as usize != n {
        return Err(format!(
            "output_positions ends at {}, but there are {n} weights",
            l.pos.last().unwrap()
        ));
    }
    let chunk_bits = 8 * geo.chunk_bytes();
    for (c, w) in l.pos.windows(2).enumerate() {
        if w[1] < w[0] || (w[1] - w[0]) as usize > chunk_bits {
            return Err(format!(
                "output_positions[{c}..={}] = {}, {}: not a valid chunk",
                c + 1,
                w[0],
                w[1]
            ));
        }
    }

    // split_positions: strictly increasing internal boundaries.
    let mut prev = 0i64;
    for (i, &s) in l.split.iter().enumerate() {
        if s <= prev || s as usize >= n {
            return Err(format!(
                "split_positions[{i}] = {s} is not an internal boundary"
            ));
        }
        prev = s;
    }
    Ok(())
}

/// A stored hash that did not match. Checked before structure: it is the most
/// exact statement of what is wrong.
fn verified(l: &Loaded) -> Result<(), String> {
    l.hash_error.clone().map_or(Ok(()), Err)
}

/// Geometry from the output's own config.json, as the loader reads it.
fn geometry_from_config(out: &ModelSource) -> Geometry {
    let fallback = Geometry { bpt: 8, tpb: 512 };
    let Ok(text) = std::fs::read_to_string(out.dir().join("config.json")) else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return fallback;
    };
    let c = &v["dfloat11_config"];
    Geometry {
        bpt: c["bytes_per_thread"]
            .as_u64()
            .map_or(fallback.bpt, |x| x as usize),
        tpb: c["threads_per_block"][0]
            .as_u64()
            .map_or(fallback.tpb, |x| x as usize),
    }
}

/// Read source weights `[a, b)` of a unit whose tensors are concatenated in
/// `tensors` order, touching only the byte ranges involved.
fn read_range(
    src: &View,
    tensors: &[String],
    counts: &[u64],
    a: u64,
    b: u64,
) -> Result<Vec<u8>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut out = Vec::with_capacity(((b - a) * 2) as usize);
    let mut base = 0u64;
    for (name, &n) in tensors.iter().zip(counts) {
        let (lo, hi) = (a.max(base), b.min(base + n));
        if lo < hi {
            let (path, offset, _) = src.locate(name).ok_or_else(|| format!("{name} vanished"))?;
            let mut f =
                std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            f.seek(SeekFrom::Start(offset + (lo - base) * 2))
                .map_err(|e| e.to_string())?;
            let start = out.len();
            out.resize(start + ((hi - lo) * 2) as usize, 0);
            f.read_exact(&mut out[start..])
                .map_err(|e| format!("{name}: {e}"))?;
        }
        base += n;
        if base >= b {
            break;
        }
    }
    Ok(out)
}

/// Check a written output. `reference` is the source model and the definition it
/// was compressed with; required for `Sample` and `Full`.
pub fn check_output(
    output: &ModelSource,
    reference: Option<(&ModelSource, &ArchDef)>,
    level: Level,
) -> Result<CheckReport, CheckError> {
    let fail = |unit: &str, chunk: Option<usize>, error: String| Failure {
        unit: unit.to_string(),
        chunk,
        error,
    };
    let mut report = CheckReport {
        units: 0,
        hashed: 0,
        checked: Vec::new(),
        seed: match level {
            Level::Sample { seed, .. } => Some(seed),
            _ => None,
        },
        failures: Vec::new(),
    };

    let Some((source, def)) = reference else {
        if level != Level::Integrity {
            return Err(CheckError::NeedsSource);
        }
        // No source: the units are whatever the output declares.
        let geo = geometry_from_config(output);
        let units: Vec<String> = output
            .names()
            .iter()
            .filter_map(|n| n.strip_suffix(".encoded_exponent").map(str::to_string))
            .collect();
        report.units = units.len();
        for u in units {
            match Loaded::read(output, &u) {
                Ok(l) => {
                    report.hashed += l.hashed;
                    if let Err(e) = verified(&l).and_then(|()| structure(&l, geo)) {
                        report.failures.push(fail(&u, None, e));
                    }
                }
                Err(e) => report.failures.push(fail(&u, None, e)),
            }
        }
        return Ok(report);
    };

    // With a source, the units are what the definition finds in it -- so a unit
    // the output lacks is a failure rather than something nobody looked for.
    let src = View::for_def(source, def).map_err(|e| CheckError::Other(e.to_string()))?;
    let found = discover(def, &src.names()).map_err(|e| CheckError::Other(e.to_string()))?;
    let geo = Geometry {
        bpt: def.bytes_per_thread as usize,
        tpb: *def.threads_per_block.first().unwrap_or(&512) as usize,
    };
    report.units = found.units.len();

    // Structure first. A unit that fails it is not decoded: its indices cannot
    // be trusted to stay in bounds.
    let mut good: Vec<(usize, Loaded, Vec<u64>)> = Vec::new();
    for (i, u) in found.units.iter().enumerate() {
        let counts: Vec<u64> = u
            .tensors
            .iter()
            .map(|n| src.info(n).map_or(0, |t| t.nbytes() / 2))
            .collect();
        let l = match Loaded::read(output, &u.name).and_then(|l| {
            report.hashed += l.hashed;
            verified(&l).and_then(|()| structure(&l, geo)).map(|()| l)
        }) {
            Ok(l) => l,
            Err(e) => {
                report.failures.push(fail(&u.name, None, e));
                continue;
            }
        };
        let total: u64 = counts.iter().sum();
        if l.sm.len() as u64 != total {
            report.failures.push(fail(
                &u.name,
                None,
                format!("the unit holds {} weights, the source {total}", l.sm.len()),
            ));
            continue;
        }
        if l.split != ArchDef::split_positions(&counts) {
            report.failures.push(fail(
                &u.name,
                None,
                format!(
                    "split_positions {:?} do not match the source tensors {:?}",
                    l.split,
                    ArchDef::split_positions(&counts)
                ),
            ));
            continue;
        }
        good.push((i, l, counts));
    }

    match level {
        Level::Integrity => {}
        Level::Full => {
            for (i, l, counts) in &good {
                let u = &found.units[*i];
                let total = counts.iter().sum();
                let bytes = match read_range(&src, &u.tensors, counts, 0, total) {
                    Ok(b) => b,
                    Err(e) => return Err(CheckError::Other(e)),
                };
                if let Err(e) = verify_unit(&l.view(geo), &bytes) {
                    report.failures.push(fail(&u.name, None, e.to_string()));
                }
                let n = l.pos.len() - 1;
                report.checked.extend((0..n).map(|c| (u.name.clone(), c)));
            }
        }
        Level::Sample { budget, seed } => {
            // The chunks holding a boundary between concatenated tensors are
            // always sampled: that is where an ordering fault would show first.
            let shapes: Vec<UnitShape> = good
                .iter()
                .map(|(_, l, _)| {
                    let mut b: Vec<usize> = l
                        .split
                        .iter()
                        .filter_map(|&s| chunk_of(&l.pos, s as u64))
                        .collect();
                    b.dedup();
                    UnitShape {
                        chunks: l.pos.len() - 1,
                        boundary_chunks: b,
                    }
                })
                .collect();
            let mut by_unit: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (k, c) in plan(&shapes, budget, seed) {
                by_unit.entry(k).or_default().push(c);
            }
            for (k, chunks) in by_unit {
                let (i, l, counts) = &good[k];
                let u = &found.units[*i];
                let view = l.view(geo);
                for c in chunks {
                    report.checked.push((u.name.clone(), c));
                    let Some((a, b)) = chunk_range(&view, c) else {
                        report
                            .failures
                            .push(fail(&u.name, Some(c), "no valid range".into()));
                        continue;
                    };
                    let want = read_range(&src, &u.tensors, counts, a as u64, b as u64)
                        .map_err(CheckError::Other)?;
                    if let Err(e) = verify_chunk(&view, c, &want) {
                        report.failures.push(fail(&u.name, Some(c), e.to_string()));
                    }
                }
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal well-formed unit: one prefix table, `chunks` chunks.
    fn unit(pos: Vec<u32>) -> Loaded {
        let chunks = pos.len() - 1;
        Loaded {
            luts: vec![0; 512],
            enc: vec![0; chunks * 64],
            sm: vec![0; *pos.last().unwrap() as usize],
            pos,
            gaps: vec![0; chunks * 5],
            split: vec![],
            hashed: 0,
            hash_error: None,
        }
    }

    /// Geometry of 8 bytes x 8 threads: 512 bits, so at most 512 weights a chunk.
    const GEO: Geometry = Geometry { bpt: 8, tpb: 8 };

    #[test]
    fn a_chunk_may_hold_one_weight_per_bit_and_no_more() {
        assert!(structure(&unit(vec![0, 512, 600]), GEO).is_ok());
        // Monotonic, starts at 0, ends at n -- but chunk 0 claims 513 weights in
        // 512 bits, which no code of length >= 1 allows.
        let e = structure(&unit(vec![0, 513, 600]), GEO).unwrap_err();
        assert!(e.contains("not a valid chunk"), "{e}");
    }
}
