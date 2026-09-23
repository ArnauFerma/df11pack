//! Assembling one compression unit: source tensors in, the six DF11 tensors out.

use crate::arch::ArchDef;
use crate::bitstream::Encoded;
use crate::chunked::{encode_chunked, DEFAULT_CHUNK};
use crate::huffman::{build_limited, build_luts};
use crate::{check_unit_limits, split_fields_into, EncodeError, Histogram};

/// The six tensors DF11 stores for one unit, with the official name suffixes.
#[derive(Debug, Clone)]
pub struct UnitOutput {
    pub name: String,
    pub luts: Vec<[u8; 256]>,
    pub encoded_exponent: Vec<u8>,
    pub sign_mantissa: Vec<u8>,
    pub output_positions: Vec<u32>,
    pub gaps: Vec<u8>,
    /// `n - 1` internal boundaries; empty for a single-tensor unit.
    pub split_positions: Vec<i64>,
    /// How many demotion rounds the 32-bit limiter needed; 0 when it never ran.
    pub limiter_iterations: usize,
    /// The idx8 index, when asked for. Its tensors then replace `gaps` and
    /// `output_positions` (docs/INDEX_SCHEMES.md). Not DF11-compatible.
    pub idx8: Option<crate::idx8::Idx8>,
}

impl UnitOutput {
    /// `(tensor name, raw little-endian bytes)` in the format's own dtypes,
    /// ready to write into a safetensors shard.
    pub fn tensors(&self) -> Vec<(String, Vec<u8>)> {
        let n = &self.name;
        if let Some(ix) = &self.idx8 {
            return vec![
                (
                    format!("{n}.luts"),
                    self.luts.iter().flat_map(|r| r.iter().copied()).collect(),
                ),
                (
                    format!("{n}.encoded_exponent"),
                    self.encoded_exponent.clone(),
                ),
                (format!("{n}.sign_mantissa"), self.sign_mantissa.clone()),
                (format!("{n}.idx8_lengths"), ix.lengths.clone()),
                (format!("{n}.idx8_superblocks"), ix.superblock_bytes()),
                (format!("{n}.idx8_meta"), ix.meta_bytes()),
                (
                    format!("{n}.split_positions"),
                    self.split_positions
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                ),
            ];
        }
        vec![
            (
                format!("{n}.luts"),
                self.luts.iter().flat_map(|r| r.iter().copied()).collect(),
            ),
            (
                format!("{n}.encoded_exponent"),
                self.encoded_exponent.clone(),
            ),
            (format!("{n}.sign_mantissa"), self.sign_mantissa.clone()),
            (
                format!("{n}.output_positions"),
                self.output_positions
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            (format!("{n}.gaps"), self.gaps.clone()),
            (
                format!("{n}.split_positions"),
                self.split_positions
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
        ]
    }

    /// Total weights in the unit.
    pub fn weights(&self) -> u64 {
        self.sign_mantissa.len() as u64
    }
}

/// Encode one unit from its source tensors.
///
/// `tensors` must be the BF16 little-endian bytes of each attribute **in the
/// definition's order**. That order fixes `split_positions` and therefore every
/// compressed byte, so passing them in a different order produces a structurally
/// valid but wrong file.
pub fn encode_unit(
    name: &str,
    tensors: &[&[u8]],
    threads_per_block: usize,
    bytes_per_thread: usize,
) -> Result<UnitOutput, EncodeError> {
    let counts: Vec<u64> = tensors.iter().map(|t| (t.len() / 2) as u64).collect();
    let mut it = tensors.iter();
    encode_unit_streaming(
        name,
        &counts,
        |_| Ok::<_, EncodeError>(it.next().expect("one per count").to_vec()),
        threads_per_block,
        bytes_per_thread,
    )
}

/// Encode a unit without ever holding all its source tensors at once.
///
/// `counts` is each tensor's weight count, known from the file header without
/// reading any data. `fetch(i)` yields the i-th tensor's bytes; it is called once
/// per tensor, in order, and each is dropped before the next is fetched. That
/// keeps 2 N bytes of source data out of the worker's footprint, which on a real
/// unit is the largest single allocation it would otherwise make.
pub fn encode_unit_streaming<E>(
    name: &str,
    counts: &[u64],
    fetch: impl FnMut(usize) -> Result<Vec<u8>, E>,
    threads_per_block: usize,
    bytes_per_thread: usize,
) -> Result<UnitOutput, EncodeError>
where
    EncodeError: From<E>,
{
    encode_unit_streaming_with(
        name,
        counts,
        fetch,
        threads_per_block,
        bytes_per_thread,
        None,
    )
}

/// [`encode_unit_streaming`], optionally also building the idx8 index with
/// blocks of `idx8_block` symbols.
pub fn encode_unit_streaming_with<E>(
    name: &str,
    counts: &[u64],
    mut fetch: impl FnMut(usize) -> Result<Vec<u8>, E>,
    threads_per_block: usize,
    bytes_per_thread: usize,
    idx8_block: Option<usize>,
) -> Result<UnitOutput, EncodeError>
where
    EncodeError: From<E>,
{
    let total_weights: u64 = counts.iter().sum();
    check_unit_limits(total_weights, 0)?;

    // Split each tensor straight into the two streams. The concatenated copy of
    // the whole unit, 2 N bytes, is never built.
    let n = total_weights as usize;
    let mut exponents: Vec<u8> = Vec::with_capacity(n);
    let mut sign_mantissa: Vec<u8> = Vec::with_capacity(n);
    for i in 0..counts.len() {
        let t = fetch(i)?;
        split_fields_into(&t, &mut exponents, &mut sign_mantissa);
        // Dropped here, before the next is fetched.
    }

    // Gates first: a model that cannot be represented must fail before any
    // output exists, not after.
    let hist = Histogram::build(&exponents)?;
    let built = build_limited(&hist.frequencies())?;
    let luts = build_luts(&built.codebook)?;

    let Encoded {
        bytes,
        gaps,
        output_positions,
    } = encode_chunked(
        &exponents,
        &built.codebook,
        bytes_per_thread,
        threads_per_block,
        DEFAULT_CHUNK,
    );

    check_unit_limits(total_weights, bytes.len() as u64)?;

    // idx8 needs only each symbol's code length, in stream order.
    let idx8 = match idx8_block {
        None => None,
        Some(block) => {
            let (bits, _) = crate::chunked::code_table(&built.codebook);
            Some(
                crate::idx8::build(exponents.iter().map(|&e| bits[e as usize]), block)
                    .map_err(EncodeError::Idx8)?,
            )
        }
    };

    Ok(UnitOutput {
        name: name.to_string(),
        luts,
        encoded_exponent: bytes,
        sign_mantissa,
        output_positions,
        gaps,
        split_positions: ArchDef::split_positions(counts),
        limiter_iterations: built.iterations,
        idx8,
    })
}
