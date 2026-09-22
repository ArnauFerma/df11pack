//! Access to the frozen Phase 0 fixtures: official DFloat11 output, hashed
//! per tensor, that Phase 1 grades byte-identity against.
//!
//! The fixtures themselves are large and untracked. `phase0/fixtures/MANIFEST.json`
//! is tracked and indexes them. When the outputs are absent, [`Fixtures::load`]
//! returns [`None`] so tests can skip cleanly rather than fail — see
//! [`skip_if_missing`].
//!
//! Regenerate with `phase0/env/bin/python phase0/freeze_fixtures.py`.

use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// One tensor in an official output, with the hash it must reproduce.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorFixture {
    #[serde(skip)]
    pub name: String,
    #[serde(skip)]
    pub file: PathBuf,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
struct RawSet {
    name: String,
    provenance: String,
    note: String,
    output_dir: String,
    source_dir: String,
    shards: u64,
    unit_tensors: u64,
    non_unit_tensors: u64,
    files: std::collections::BTreeMap<String, std::collections::BTreeMap<String, TensorFixture>>,
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    sets: Vec<RawSet>,
}

/// One frozen official output.
#[derive(Debug)]
pub struct FixtureSet {
    pub name: String,
    /// Directory of the BF16 model this output was compressed from.
    pub source_dir: PathBuf,
    /// `"locally-compressed"` or `"published-release"`. A byte-identity failure
    /// against a published release must be triaged against its provenance
    /// caveat before being treated as an encoder bug.
    pub provenance: String,
    pub note: String,
    pub shards: u64,
    pub unit_tensors: u64,
    pub non_unit_tensors: u64,
    tensors: Vec<TensorFixture>,
}

impl FixtureSet {
    pub fn tensors(&self) -> &[TensorFixture] {
        &self.tensors
    }

    /// Every tensor belonging to DF11 compression units, i.e. the encoder's output.
    pub fn unit_tensors(&self) -> impl Iterator<Item = &TensorFixture> {
        self.tensors.iter().filter(|t| t.is_unit_tensor())
    }

    /// Tensors of one named unit, e.g. `"model.layers.0"`.
    pub fn unit(&self, unit: &str) -> Vec<&TensorFixture> {
        let prefix = format!("{unit}.");
        let mut v: Vec<_> = self
            .tensors
            .iter()
            .filter(|t| t.is_unit_tensor() && t.name.starts_with(&prefix))
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Names of every unit in this set, sorted.
    pub fn unit_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .tensors
            .iter()
            .filter(|t| t.is_unit_tensor())
            .filter_map(|t| t.unit_name().map(str::to_owned))
            .collect();
        v.sort();
        v.dedup();
        v
    }
}

const UNIT_SUFFIXES: [&str; 6] = [
    "luts",
    "encoded_exponent",
    "sign_mantissa",
    "output_positions",
    "gaps",
    "split_positions",
];

impl TensorFixture {
    pub fn is_unit_tensor(&self) -> bool {
        self.suffix().is_some_and(|s| UNIT_SUFFIXES.contains(&s))
    }

    fn suffix(&self) -> Option<&str> {
        self.name.rsplit_once('.').map(|(_, s)| s)
    }

    /// `"model.layers.0.gaps"` -> `"model.layers.0"`.
    pub fn unit_name(&self) -> Option<&str> {
        if !self.is_unit_tensor() {
            return None;
        }
        self.name.rsplit_once('.').map(|(head, _)| head)
    }

    /// The expected bytes, read from the official output on disk.
    pub fn read(&self) -> std::io::Result<Vec<u8>> {
        let (offsets, base) = locate(&self.file, &self.name)?;
        let mut f = File::open(&self.file)?;
        f.seek(SeekFrom::Start(base + offsets.0))?;
        let mut buf = vec![0u8; (offsets.1 - offsets.0) as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Hash the on-disk bytes without holding them, and check the manifest.
    pub fn verify_on_disk(&self) -> std::io::Result<bool> {
        let (offsets, base) = locate(&self.file, &self.name)?;
        let mut f = File::open(&self.file)?;
        f.seek(SeekFrom::Start(base + offsets.0))?;
        let mut left = offsets.1 - offsets.0;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        while left > 0 {
            let n = (left as usize).min(buf.len());
            f.read_exact(&mut buf[..n])?;
            h.update(&buf[..n]);
            left -= n as u64;
        }
        Ok(hex::encode(h.finalize()) == self.sha256)
    }
}

/// Parse a safetensors header and return one tensor's `(start, end)` plus the
/// offset the data section begins at.
fn locate(path: &Path, tensor: &str) -> std::io::Result<((u64, u64), u64)> {
    let mut f = File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)?;
    let off = v
        .get(tensor)
        .and_then(|t| t.get("data_offsets"))
        .and_then(|o| o.as_array())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {tensor:?} not in {}", path.display()),
            )
        })?;
    let s = off[0].as_u64().unwrap_or(0);
    let e = off[1].as_u64().unwrap_or(0);
    Ok(((s, e), 8 + n))
}

/// The loaded manifest.
#[derive(Debug)]
pub struct Fixtures {
    pub sets: Vec<FixtureSet>,
}

impl Fixtures {
    /// Load the manifest, or `None` when the fixtures are not present.
    pub fn load() -> Option<Self> {
        let root = workspace_root()?;
        let man = root.join("phase0/fixtures/MANIFEST.json");
        let raw: RawManifest = serde_json::from_slice(&std::fs::read(&man).ok()?).ok()?;
        let mut sets = Vec::new();
        for s in raw.sets {
            let dir = resolve(&root, &s.output_dir);
            if !dir.exists() {
                continue;
            }
            let mut tensors = Vec::new();
            for (fname, ts) in s.files {
                let fpath = dir.join(&fname);
                for (tname, mut t) in ts {
                    t.name = tname;
                    t.file = fpath.clone();
                    tensors.push(t);
                }
            }
            sets.push(FixtureSet {
                name: s.name,
                source_dir: resolve(&root, &s.source_dir),
                provenance: s.provenance,
                note: s.note,
                shards: s.shards,
                unit_tensors: s.unit_tensors,
                non_unit_tensors: s.non_unit_tensors,
                tensors,
            });
        }
        if sets.is_empty() {
            return None;
        }
        Some(Fixtures { sets })
    }

    pub fn set(&self, name: &str) -> Option<&FixtureSet> {
        self.sets.iter().find(|s| s.name == name)
    }
}

fn resolve(root: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

fn workspace_root() -> Option<PathBuf> {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if d.join("phase0").is_dir() {
            return Some(d);
        }
        if !d.pop() {
            return None;
        }
    }
}

/// Load the fixtures, or print why the test is being skipped and return `None`.
///
/// Prefer this in tests over silently passing: a test that skips must say so.
#[must_use]
pub fn skip_if_missing(test: &str) -> Option<Fixtures> {
    match Fixtures::load() {
        Some(f) => Some(f),
        None => {
            eprintln!(
                "SKIP {test}: Phase 0 fixtures absent. \
                 Regenerate with `phase0/env/bin/python phase0/freeze_fixtures.py`."
            );
            None
        }
    }
}

/// Where and how two byte strings first differ.
#[derive(Debug, PartialEq, Eq)]
pub struct ByteDiff {
    pub byte_offset: usize,
    pub bit_offset: usize,
    pub expected: u8,
    pub actual: u8,
    pub expected_len: usize,
    pub actual_len: usize,
    pub context_expected: Vec<u8>,
    pub context_actual: Vec<u8>,
}

impl fmt::Display for ByteDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.expected_len != self.actual_len {
            writeln!(
                f,
                "length differs: expected {} bytes, got {}",
                self.expected_len, self.actual_len
            )?;
        }
        writeln!(
            f,
            "first difference at byte {} (bit {}): expected 0x{:02x} ({:08b}), got 0x{:02x} ({:08b}), xor 0x{:02x}",
            self.byte_offset,
            self.bit_offset,
            self.expected,
            self.expected,
            self.actual,
            self.actual,
            self.expected ^ self.actual
        )?;
        write!(f, "  expected: ")?;
        for b in &self.context_expected {
            write!(f, "{b:02x} ")?;
        }
        writeln!(f)?;
        write!(f, "  actual:   ")?;
        for b in &self.context_actual {
            write!(f, "{b:02x} ")?;
        }
        Ok(())
    }
}

/// First difference between two byte strings, or `None` if identical.
pub fn diff_bytes(expected: &[u8], actual: &[u8]) -> Option<ByteDiff> {
    let n = expected.len().min(actual.len());
    let mut at = None;
    for i in 0..n {
        if expected[i] != actual[i] {
            at = Some(i);
            break;
        }
    }
    let at = match at {
        Some(i) => i,
        None if expected.len() == actual.len() => return None,
        // Equal on the overlap but different lengths: report at the truncation.
        None => n,
    };
    let lo = at.saturating_sub(8);
    let hi_e = (at + 8).min(expected.len());
    let hi_a = (at + 8).min(actual.len());
    Some(ByteDiff {
        byte_offset: at,
        bit_offset: at * 8,
        expected: expected.get(at).copied().unwrap_or(0),
        actual: actual.get(at).copied().unwrap_or(0),
        expected_len: expected.len(),
        actual_len: actual.len(),
        context_expected: expected[lo..hi_e].to_vec(),
        context_actual: actual[lo..hi_a].to_vec(),
    })
}

/// Assert `actual` reproduces the fixture exactly, panicking with the location
/// and surrounding bytes of the first difference.
pub fn assert_matches(fixture: &TensorFixture, actual: &[u8]) {
    let expected = fixture
        .read()
        .unwrap_or_else(|e| panic!("reading fixture {}: {e}", fixture.name));
    if let Some(d) = diff_bytes(&expected, actual) {
        panic!(
            "tensor {} does not match the official output\n  file: {}\n{}",
            fixture.name,
            fixture.file.display(),
            d
        );
    }
}

/// Reader for the BF16 source model a fixture set was compressed from.
///
/// Phase 1 needs the encoder's *input*, not just its output: to grade
/// `sign_mantissa` byte-for-byte you must feed in exactly the bytes the official
/// compressor fed in, concatenated in `attr_names` order.
pub struct SourceModel {
    path: PathBuf,
}

impl SourceModel {
    pub fn open(set: &FixtureSet) -> Option<Self> {
        let p = set.source_dir.join("model.safetensors");
        p.is_file().then_some(SourceModel { path: p })
    }

    /// Raw little-endian bytes of one tensor.
    pub fn tensor(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let (off, base) = locate(&self.path, name)?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(base + off.0))?;
        let mut buf = vec![0u8; (off.1 - off.0) as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// The concatenated BF16 buffer for one unit, in `attr_names` order.
    ///
    /// The order is load-bearing: it fixes `split_positions` and therefore every
    /// compressed byte. See FINDINGS 0.9.
    pub fn unit_input(&self, unit: &str, attr_names: &[&str]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        for a in attr_names {
            out.extend_from_slice(&self.tensor(&format!("{unit}.{a}.weight"))?);
        }
        Ok(out)
    }
}

/// One generated case for the 32-bit code-length limiter.
///
/// No real fixture reaches the limiter, so these come from the H4 Python spec
/// (itself validated against `dahuffman`). See `phase0/gen_limiter_cases.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct LimiterCase {
    pub name: String,
    pub freqs: Vec<(u8, u64)>,
    pub max_bits_uncapped: u32,
    pub triggered: bool,
    pub max_bits_final: u32,
    pub iterations: usize,
    /// True when the limiter had to choose among equal frequencies above 1.
    pub ambiguous: bool,
    pub first_ambiguous: Option<AmbiguousDetail>,
    /// Symbol (as a decimal string) -> code length.
    pub lengths: std::collections::BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AmbiguousDetail {
    pub min_k: usize,
    pub boundary_value: u64,
    pub tied: usize,
    pub slots: usize,
}

#[derive(Debug, Deserialize)]
struct LimiterFile {
    cases: Vec<LimiterCase>,
}

/// Load the generated limiter cases, or `None` if they are absent.
pub fn limiter_cases() -> Option<Vec<LimiterCase>> {
    let root = workspace_root()?;
    let p = root.join("phase0/fixtures/limiter_cases.json");
    let f: LimiterFile = serde_json::from_slice(&std::fs::read(p).ok()?).ok()?;
    Some(f.cases)
}
