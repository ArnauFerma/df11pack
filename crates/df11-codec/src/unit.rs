//! Assembling one compression unit: source tensors in, the six DF11 tensors out.

use crate::arch::ArchDef;
use crate::bitstream::{encode, Encoded};
use crate::huffman::{build_limited, build_luts};
use crate::{check_unit_limits, split_fields, EncodeError, Histogram};

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
}

impl UnitOutput {
    /// `(tensor name, raw little-endian bytes)` in the format's own dtypes,
    /// ready to write into a safetensors shard.
    pub fn tensors(&self) -> Vec<(String, Vec<u8>)> {
        let n = &self.name;
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
    let total_weights: u64 = counts.iter().sum();
    check_unit_limits(total_weights, 0)?;

    let mut concatenated = Vec::with_capacity(tensors.iter().map(|t| t.len()).sum());
    for t in tensors {
        concatenated.extend_from_slice(t);
    }

    let (exponents, sign_mantissa) = split_fields(&concatenated);
    drop(concatenated);

    // Gates first: a model that cannot be represented must fail before any
    // output exists, not after.
    let hist = Histogram::build(&exponents)?;
    let built = build_limited(&hist.frequencies())?;
    let luts = build_luts(&built.codebook)?;

    let Encoded {
        bytes,
        gaps,
        output_positions,
    } = encode(
        &exponents,
        &built.codebook,
        bytes_per_thread,
        threads_per_block,
    );

    check_unit_limits(total_weights, bytes.len() as u64)?;

    Ok(UnitOutput {
        name: name.to_string(),
        luts,
        encoded_exponent: bytes,
        sign_mantissa,
        output_positions,
        gaps,
        split_positions: ArchDef::split_positions(&counts),
        limiter_iterations: built.iterations,
    })
}
