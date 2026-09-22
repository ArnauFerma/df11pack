//! The bitstream, `gaps` and `output_positions`.
//!
//! Ported from `dfloat11_utils.encode`. Three details are easy to lose:
//!
//! 1. A gap or output position is recorded **before** the symbol is emitted, so
//!    it is the offset of the first code that *starts* in that window or chunk.
//! 2. The trailing EOF writes **exactly one byte** and drops any bits beyond it.
//!    When the stream happens to end byte-aligned, no EOF byte is written at all.
//! 3. `gaps` is zero-padded to `512 * blocks_per_grid` entries, then packed
//!    5 bits each, MSB first.

use crate::huffman::{Codebook, Sym};

/// One unit's encoded exponent stream and its two index tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// Packed 5-bit offsets, one per 64-bit window.
    pub gaps: Vec<u8>,
    /// Element index at the start of each 4096-byte chunk, plus the total.
    pub output_positions: Vec<u32>,
}

impl Encoded {
    /// `output_positions` as the uint8 view the format stores.
    pub fn output_positions_bytes(&self) -> Vec<u8> {
        self.output_positions
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}

/// Encode one unit's exponent stream.
pub fn encode(
    exponents: &[u8],
    cb: &Codebook,
    bytes_per_thread: usize,
    threads_per_block: usize,
) -> Encoded {
    let window_bits = 8 * bytes_per_thread;
    let chunk_bits = window_bits * threads_per_block;

    let mut table: [Option<crate::huffman::Code>; 256] = [None; 256];
    for (s, c) in cb.entries() {
        if let Sym::Val(v) = s {
            table[*v as usize] = Some(*c);
        }
    }
    let eof = code_for(cb, Sym::Eof);

    let mut bytes: Vec<u8> = Vec::with_capacity(exponents.len() / 2);
    let mut gaps: Vec<u32> = Vec::new();
    let mut output_positions: Vec<u32> = Vec::new();

    let mut buffer: u64 = 0;
    let mut size: u32 = 0;
    let mut total_size: usize = 0;
    let mut element_count: u32 = 0;

    for &s in exponents {
        // Recorded before the symbol is written, so it is the offset of the
        // first code that STARTS in this window / chunk.
        if total_size / window_bits + 1 > gaps.len() {
            gaps.push((total_size % window_bits) as u32);
        }
        if total_size / chunk_bits + 1 > output_positions.len() {
            output_positions.push(element_count);
        }

        let c = table[s as usize].expect("exponent present in the codebook");
        buffer = (buffer << c.bits) + c.value;
        size += c.bits;
        total_size += c.bits as usize;
        element_count += 1;

        while size >= 8 {
            let byte = (buffer >> (size - 8)) as u8;
            bytes.push(byte);
            buffer -= u64::from(byte) << (size - 8);
            size -= 8;
        }
    }

    // A byte-aligned stream gets no EOF byte at all.
    if size > 0 {
        if total_size / window_bits + 1 > gaps.len() {
            gaps.push((total_size % window_bits) as u32);
        }
        if total_size / chunk_bits + 1 > output_positions.len() {
            output_positions.push(element_count);
        }
        buffer = (buffer << eof.bits) + eof.value;
        size += eof.bits;
        // Exactly one byte; any bits past it are dropped, as upstream does.
        let byte = if size >= 8 {
            (buffer >> (size - 8)) as u8
        } else {
            (buffer << (8 - size)) as u8
        };
        bytes.push(byte);
    }

    output_positions.push(exponents.len() as u32);

    let blocks = bytes.len().div_ceil(threads_per_block * bytes_per_thread);
    let target = threads_per_block * blocks;
    // Extend only. Upstream multiplies by a possibly-negative count, which
    // yields an empty list rather than truncating.
    if target > gaps.len() {
        gaps.resize(target, 0);
    }

    let mut packed = Vec::with_capacity(gaps.len() * 5 / 8 + 1);
    let mut acc: u16 = 0;
    let mut nbits: u32 = 0;
    for g in &gaps {
        acc = (acc << 5) | (*g as u16 & 0x1F);
        nbits += 5;
        while nbits >= 8 {
            packed.push((acc >> (nbits - 8)) as u8);
            nbits -= 8;
            acc &= (1 << nbits) - 1;
        }
    }
    if nbits > 0 {
        packed.push((acc << (8 - nbits)) as u8);
    }

    Encoded {
        bytes,
        gaps: packed,
        output_positions,
    }
}

/// Look up a symbol's code, or EOF's.
pub(crate) fn code_for(cb: &Codebook, s: Sym) -> crate::huffman::Code {
    cb.entries()
        .iter()
        .find(|(sym, _)| *sym == s)
        .map(|(_, c)| *c)
        .expect("symbol present in codebook")
}
