//! df11-codec -- the DFloat11 encoder.
//!
//! Nothing is implemented yet. Phase 1 fills this in, graded against the
//! frozen Phase 0 fixtures (see `df11-fixtures` and `docs/PLAN.md`).
//!
//! Read `docs/COMPATIBILITY.md` before touching the LUT path: the default
//! mode reproduces a bug in the official encoder on purpose.

/// Split a little-endian BF16 buffer into the two streams DF11 stores.
///
/// A BF16 weight is `s eeeeeeee mmmmmmm`. The official encoder derives, from the
/// int16 pattern `W`:
///
/// ```text
/// exponent      = (W >> 7) & 0xFF
/// sign_mantissa = ((W >> 8) & 0x80) | (W & 0x7F)
/// ```
///
/// so `sign_mantissa` carries the sign in bit 7 and the mantissa in bits 0..=6.
/// The official code shifts a *signed* int16, but both results are masked, so an
/// unsigned shift is equivalent.
///
/// Returns `(exponents, sign_mantissa)`, one byte each per weight.
///
/// # Panics
/// If `bf16_le` has an odd length; a BF16 buffer is always an even number of bytes.
pub fn split_fields(bf16_le: &[u8]) -> (Vec<u8>, Vec<u8>) {
    assert!(
        bf16_le.len() % 2 == 0,
        "BF16 buffer must have an even length, got {}",
        bf16_le.len()
    );
    let n = bf16_le.len() / 2;
    let mut exponents = Vec::with_capacity(n);
    let mut sign_mantissa = Vec::with_capacity(n);
    for c in bf16_le.chunks_exact(2) {
        let w = u16::from_le_bytes([c[0], c[1]]);
        exponents.push(((w >> 7) & 0xFF) as u8);
        sign_mantissa.push((((w >> 8) & 0x80) | (w & 0x7F)) as u8);
    }
    (exponents, sign_mantissa)
}

pub mod arch;
pub mod bitstream;
pub mod huffman;
pub mod safetensors;
pub mod unit;

use std::fmt;

/// Exponents 240..=255 collide with the LUT's "jump to table 256-v" convention,
/// so the kernel would misread them. DESIGN 1.3.
pub const RESERVED_EXPONENT_MIN: u8 = 240;
/// The kernel holds `n_elements` in an `int`.
pub const MAX_UNIT_WEIGHTS: u64 = (1 << 31) - 1;
/// The kernel holds `n_bytes` in an `int`, and `output_positions` is `uint32`.
pub const MAX_UNIT_BYTES: u64 = (1 << 31) - 1;
/// A LUT value >= 240 encodes a jump to table `256 - v`, bounding the count.
pub const MAX_PREFIX_TABLES: usize = 16;
/// `gaps` stores a 5-bit offset per 64-bit window, which requires this.
pub const MAX_CODE_BITS: u32 = 32;

/// A condition that must abort encoding rather than produce a file.
///
/// Every variant carries the measured value, so the message is actionable
/// rather than merely a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// An exponent in the range the LUT jump convention reserves.
    ReservedExponent {
        value: u8,
        index: usize,
    },
    TooManyWeights {
        weights: u64,
    },
    TooManyBytes {
        bytes: u64,
    },
    TooManyPrefixTables {
        tables: usize,
    },
    /// A code the 32-bit limiter failed to bring within range.
    CodeTooLong {
        bits: u32,
    },
    /// The 32-bit limiter had to choose between symbols of equal frequency, and
    /// the choice changes the codebook, so byte-identity cannot be guaranteed.
    ///
    /// The official encoder resolves this with `np.argpartition`, whose ordering
    /// among equal elements NumPy does not specify. Rather than emit a file that
    /// might silently differ, df11pack refuses. See `docs/COMPATIBILITY.md`.
    AmbiguousLimiterTie {
        min_k: usize,
        boundary_frequency: u64,
        tied: usize,
        slots: usize,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedExponent { value, index } => write!(
                f,
                "exponent {value} at weight {index} is in the reserved range \
                 {RESERVED_EXPONENT_MIN}..=255 (infinities or NaNs); the LUT jump \
                 convention cannot represent it, so no file was written"
            ),
            Self::TooManyWeights { weights } => write!(
                f,
                "unit has {weights} weights, over the kernel's limit of {MAX_UNIT_WEIGHTS}; \
                 split the unit"
            ),
            Self::TooManyBytes { bytes } => write!(
                f,
                "unit encodes to {bytes} bytes, over the kernel's limit of {MAX_UNIT_BYTES}; \
                 split the unit"
            ),
            Self::TooManyPrefixTables { tables } => write!(
                f,
                "codebook needs {tables} prefix tables, over the limit of {MAX_PREFIX_TABLES}"
            ),
            Self::AmbiguousLimiterTie {
                min_k,
                boundary_frequency,
                tied,
                slots,
            } => write!(
                f,
                "the 32-bit code-length limiter reached an ambiguous tie at k={min_k}: \
                 {tied} symbols share frequency {boundary_frequency} but only {slots} \
                 can be demoted. The official encoder picks among them with \
                 np.argpartition, whose order NumPy leaves unspecified, and the choice \
                 changes the codebook. df11pack will not emit a file that might \
                 silently differ; compress this unit with the official compressor, or \
                 see docs/COMPATIBILITY.md"
            ),
            Self::CodeTooLong { bits } => write!(
                f,
                "longest code is {bits} bits, over the limit of {MAX_CODE_BITS}; \
                 the 32-bit limiter failed"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Counts of each exponent value in one compression unit.
#[derive(Debug, Clone)]
pub struct Histogram {
    counts: [u64; 256],
    total: u64,
}

impl Histogram {
    /// Count exponents, rejecting any in the reserved range.
    ///
    /// The check runs here rather than later so that a model which cannot be
    /// represented fails before anything is written.
    pub fn build(exponents: &[u8]) -> Result<Self, EncodeError> {
        let mut counts = [0u64; 256];
        for (index, &e) in exponents.iter().enumerate() {
            if e >= RESERVED_EXPONENT_MIN {
                return Err(EncodeError::ReservedExponent { value: e, index });
            }
            counts[e as usize] += 1;
        }
        Ok(Histogram {
            counts,
            total: exponents.len() as u64,
        })
    }

    pub fn counts(&self) -> &[u64; 256] {
        &self.counts
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    /// How many distinct exponent values occur.
    pub fn distinct(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// `(symbol, frequency)` pairs, ascending by symbol -- the order
    /// `torch.unique` yields, which the official encoder feeds to dahuffman.
    pub fn frequencies(&self) -> Vec<(u8, u64)> {
        (0..=255u8)
            .filter(|&s| self.counts[s as usize] > 0)
            .map(|s| (s, self.counts[s as usize]))
            .collect()
    }

    /// The exponent values that occur, ascending.
    pub fn present_symbols(&self) -> Vec<u8> {
        (0..=255u8)
            .filter(|&s| self.counts[s as usize] > 0)
            .collect()
    }
}

/// Reject a unit the kernel's 32-bit fields cannot address.
pub fn check_unit_limits(weights: u64, encoded_bytes: u64) -> Result<(), EncodeError> {
    if weights > MAX_UNIT_WEIGHTS {
        return Err(EncodeError::TooManyWeights { weights });
    }
    if encoded_bytes > MAX_UNIT_BYTES {
        return Err(EncodeError::TooManyBytes {
            bytes: encoded_bytes,
        });
    }
    Ok(())
}

/// Reject a codebook needing more prefix tables than the jump convention allows.
pub fn check_prefix_tables(tables: usize) -> Result<(), EncodeError> {
    if tables > MAX_PREFIX_TABLES {
        return Err(EncodeError::TooManyPrefixTables { tables });
    }
    Ok(())
}

/// Reject a code longer than `gaps` can describe.
pub fn check_code_len(bits: u32) -> Result<(), EncodeError> {
    if bits > MAX_CODE_BITS {
        return Err(EncodeError::CodeTooLong { bits });
    }
    Ok(())
}
