//! Safe mode: an independent CPU decoder that checks a unit before it ships.
//!
//! **This is deliberately not written from the encoder.** It is derived from what
//! `decode.cu` does, because a verifier that shares the encoder's understanding
//! of the format cannot catch the encoder misunderstanding it — the two would
//! agree on the same wrong answer. The only things it takes from our side are the
//! six emitted tensors and the source weights; everything about how to read them
//! comes from the kernel's semantics.
//!
//! What the kernel does, and therefore what this replicates:
//!
//! 1. A code is found by walking the hierarchical byte LUTs: look up the next
//!    byte in the current table; a value ≥ 240 means "jump to table 256 − v" and
//!    consume that byte; otherwise the value is the symbol and its length comes
//!    from the final row.
//! 2. Each 64-bit window is entered at the bit offset its `gaps` entry gives,
//!    which is where the first code that *starts* in that window begins.
//! 3. `output_positions` gives the element index at the start of each
//!    4096-byte chunk.
//!
//! Checking 2 and 3 matters as much as checking 1: an encoder bug that produced
//! a valid bitstream with wrong indices would decode correctly here and wrongly
//! on the GPU, which is the failure this exists to prevent.

use std::fmt;

/// Why a unit failed verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The decoded weight at `index` differs from the source.
    WeightMismatch {
        index: usize,
        expected: u16,
        got: u16,
    },
    /// Decoding ran out of symbols before the unit's weight count.
    Truncated {
        decoded: usize,
        expected: usize,
    },
    /// A `gaps` entry does not point at a code boundary.
    GapNotACodeBoundary {
        window: usize,
        gap: u32,
        bit: usize,
    },
    /// An `output_positions` entry disagrees with where decoding actually is.
    OutputPositionMismatch {
        chunk: usize,
        stored: u32,
        actual: u32,
    },
    /// A byte selected a LUT table that does not exist.
    BadLutJump {
        table: usize,
        byte: u8,
        target: usize,
    },
    /// A symbol was decoded whose code length is zero, i.e. it has no code.
    SymbolWithoutCode {
        symbol: u8,
    },
    Malformed(String),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightMismatch {
                index,
                expected,
                got,
            } => write!(
                f,
                "weight {index} decoded as 0x{got:04x}, source has 0x{expected:04x}"
            ),
            Self::Truncated { decoded, expected } => {
                write!(f, "decoded {decoded} weights but the unit holds {expected}")
            }
            Self::GapNotACodeBoundary { window, gap, bit } => write!(
                f,
                "gaps[{window}] = {gap} does not land on a code boundary (bit {bit}); \
                 the GPU would start this window mid-code"
            ),
            Self::OutputPositionMismatch {
                chunk,
                stored,
                actual,
            } => write!(
                f,
                "output_positions[{chunk}] = {stored} but decoding reaches element {actual}; \
                 the GPU would write this chunk to the wrong place"
            ),
            Self::BadLutJump {
                table,
                byte,
                target,
            } => write!(
                f,
                "LUT {table} byte {byte} jumps to table {target}, which does not exist"
            ),
            Self::SymbolWithoutCode { symbol } => {
                write!(f, "decoded symbol {symbol}, which has no code length")
            }
            Self::Malformed(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// One unit's emitted tensors, as the loader would see them.
pub struct UnitView<'a> {
    /// `(n_prefixes + 1) x 256`, the final row holding code lengths.
    pub luts: &'a [u8],
    pub encoded_exponent: &'a [u8],
    pub sign_mantissa: &'a [u8],
    /// The uint32 values, already decoded from their uint8 view.
    pub output_positions: &'a [u32],
    /// Packed 5 bits per 64-bit window.
    pub gaps: &'a [u8],
    pub bytes_per_thread: usize,
    pub threads_per_block: usize,
}

/// Decode a unit and check it against the source weights.
pub fn verify_unit(view: &UnitView, source_bf16_le: &[u8]) -> Result<(), VerifyError> {
    let n_weights = view.sign_mantissa.len();
    if source_bf16_le.len() != n_weights * 2 {
        return Err(VerifyError::Malformed(format!(
            "source has {} weights, the unit has {n_weights}",
            source_bf16_le.len() / 2
        )));
    }
    if view.luts.len() % 256 != 0 || view.luts.len() < 512 {
        return Err(VerifyError::Malformed(format!(
            "luts is {} bytes, not a whole number of 256-byte rows plus lengths",
            view.luts.len()
        )));
    }
    let rows = view.luts.len() / 256;
    let tables = rows - 1;
    let lens = &view.luts[tables * 256..];

    let window_bits = 8 * view.bytes_per_thread;
    let chunk_bits = window_bits * view.threads_per_block;
    let total_bits = view.encoded_exponent.len() * 8;

    // Walk the stream once, as the kernel's decode pass does, recording where
    // each window and chunk is entered so the indices can be checked against it.
    let mut bit = 0usize;
    let mut index = 0usize;
    let mut window_entry: Vec<Option<usize>> = vec![None; total_bits.div_ceil(window_bits) + 1];
    let mut chunk_entry: Vec<Option<u32>> = vec![None; total_bits.div_ceil(chunk_bits) + 1];

    while index < n_weights {
        let w = bit / window_bits;
        if w < window_entry.len() && window_entry[w].is_none() {
            window_entry[w] = Some(bit % window_bits);
        }
        let c = bit / chunk_bits;
        if c < chunk_entry.len() && chunk_entry[c].is_none() {
            chunk_entry[c] = Some(index as u32);
        }

        let (symbol, len) = decode_at(view.luts, tables, lens, view.encoded_exponent, bit)?;
        // Recombine exactly as the kernel writes it: the decoded exponent in
        // bits 14..7, the stored sign in bit 15 and mantissa in bits 6..0.
        let sm = view.sign_mantissa[index];
        let got = ((u16::from(sm & 0x80)) << 8) | (u16::from(symbol) << 7) | u16::from(sm & 0x7F);
        let expected =
            u16::from_le_bytes([source_bf16_le[index * 2], source_bf16_le[index * 2 + 1]]);
        if got != expected {
            return Err(VerifyError::WeightMismatch {
                index,
                expected,
                got,
            });
        }
        bit += len as usize;
        index += 1;
        if bit > total_bits {
            return Err(VerifyError::Truncated {
                decoded: index,
                expected: n_weights,
            });
        }
    }
    if index != n_weights {
        return Err(VerifyError::Truncated {
            decoded: index,
            expected: n_weights,
        });
    }

    // Every gap must name the bit at which its window is entered. A window the
    // stream never enters is unconstrained.
    let unpacked = unpack_gaps(view.gaps);
    for (w, entry) in window_entry.iter().enumerate() {
        let Some(want) = entry else { continue };
        let Some(&got) = unpacked.get(w) else {
            continue;
        };
        if got as usize != *want {
            return Err(VerifyError::GapNotACodeBoundary {
                window: w,
                gap: got,
                bit: *want,
            });
        }
    }

    // And every output position must name the element the chunk starts at.
    for (c, entry) in chunk_entry.iter().enumerate() {
        let Some(want) = entry else { continue };
        let Some(&stored) = view.output_positions.get(c) else {
            continue;
        };
        if stored != *want {
            return Err(VerifyError::OutputPositionMismatch {
                chunk: c,
                stored,
                actual: *want,
            });
        }
    }
    // The trailing entry is the element count.
    if let Some(&last) = view.output_positions.last() {
        if last as usize != n_weights {
            return Err(VerifyError::OutputPositionMismatch {
                chunk: view.output_positions.len() - 1,
                stored: last,
                actual: n_weights as u32,
            });
        }
    }
    Ok(())
}

/// Eight bits starting at an arbitrary bit offset, MSB first.
pub fn peek8(bytes: &[u8], bit: usize) -> u8 {
    let i = bit / 8;
    let off = bit % 8;
    let hi = u16::from(bytes.get(i).copied().unwrap_or(0));
    let lo = u16::from(bytes.get(i + 1).copied().unwrap_or(0));
    (((hi << 8) | lo) >> (8 - off)) as u8
}

/// Decode one code, walking the hierarchical tables the way the kernel does.
///
/// Public so the jump boundary can be tested directly: real units here use four
/// tables, whose jump values are 253..=255, so no fixture exercises the value
/// **240** — which is what a unit with the maximum sixteen tables would use.
pub fn decode_at(
    luts: &[u8],
    tables: usize,
    lens: &[u8],
    bytes: &[u8],
    bit: usize,
) -> Result<(u8, u8), VerifyError> {
    let mut table = 0usize;
    let mut level = 0usize;
    loop {
        let b = peek8(bytes, bit + level * 8);
        let v = luts[table * 256 + b as usize];
        if v >= 240 {
            let target = 256 - v as usize;
            if target >= tables {
                return Err(VerifyError::BadLutJump {
                    table,
                    byte: b,
                    target,
                });
            }
            table = target;
            level += 1;
            if level > tables {
                return Err(VerifyError::Malformed("LUT jumps do not terminate".into()));
            }
        } else {
            let len = lens[v as usize];
            if len == 0 {
                return Err(VerifyError::SymbolWithoutCode { symbol: v });
            }
            return Ok((v, len));
        }
    }
}

/// Unpack the 5-bit-per-window gap array.
fn unpack_gaps(packed: &[u8]) -> Vec<u32> {
    let n = packed.len() * 8 / 5;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * 5;
        let mut v = 0u32;
        for k in 0..5 {
            let bit = start + k;
            let byte = packed[bit / 8];
            let b = (byte >> (7 - (bit % 8))) & 1;
            v = (v << 1) | u32::from(b);
        }
        out.push(v);
    }
    out
}
