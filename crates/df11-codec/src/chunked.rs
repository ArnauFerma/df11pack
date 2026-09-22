//! The chunked encoder: DESIGN §5.3.
//!
//! The serial encoder in [`crate::bitstream`] is the reference. This one must
//! produce **byte-identical** output, which is the whole test.
//!
//! The idea: a chunk's bit length is `Σ hist[s] × len[s]`, computable from its
//! own histogram and the codebook without encoding anything. Prefix-summing
//! those lengths gives every chunk the exact bit offset it starts at, so all
//! chunks can encode into their own regions at once. Only the bytes straddling a
//! chunk boundary are shared, and those combine with OR because each side writes
//! only its own bits.
//!
//! `gaps` and `output_positions` fall out of the same offsets: a gap is recorded
//! for the first code that *starts* in each 64-bit window, and a chunk knows the
//! global bit position of every symbol it holds, so it can emit the entries for
//! the windows it opens without consulting its neighbours.

use crate::bitstream::Encoded;
use crate::huffman::{Codebook, Sym};

/// Symbols per chunk. Only affects scheduling, never the output.
pub const DEFAULT_CHUNK: usize = 1 << 20;

/// Encode in parallel. Output is byte-identical to [`crate::bitstream::encode`].
pub fn encode_chunked(
    exponents: &[u8],
    cb: &Codebook,
    bytes_per_thread: usize,
    threads_per_block: usize,
    chunk_symbols: usize,
) -> Encoded {
    use rayon::prelude::*;

    let chunk_symbols = chunk_symbols.max(1);
    let (bits, vals) = code_table(cb);
    let eof = crate::bitstream::code_for(cb, Sym::Eof);
    let window_bits = 8 * bytes_per_thread;
    let block_bits = window_bits * threads_per_block;

    if exponents.is_empty() {
        // The serial encoder writes no bytes, no gaps, and only the trailing
        // element count.
        return Encoded {
            bytes: Vec::new(),
            gaps: Vec::new(),
            output_positions: vec![0],
        };
    }

    // Pass 1: each chunk's bit length, from its own histogram. No encoding yet.
    let chunks: Vec<&[u8]> = exponents.chunks(chunk_symbols).collect();
    let lengths: Vec<u64> = chunks
        .par_iter()
        .map(|c| {
            let mut hist = [0u64; 256];
            for &s in c.iter() {
                hist[s as usize] += 1;
            }
            (0..256).map(|s| hist[s] * u64::from(bits[s])).sum::<u64>()
        })
        .collect();

    // Exclusive prefix sum: where each chunk starts, in bits.
    let mut starts = Vec::with_capacity(chunks.len());
    let mut acc = 0u64;
    for l in &lengths {
        starts.push(acc);
        acc += *l;
    }
    let total_bits = acc;

    let body_bytes = (total_bits / 8) as usize;
    let tail_bits = (total_bits % 8) as u32;
    let n_bytes = body_bytes + usize::from(tail_bits > 0);

    // Pass 2: each chunk packs its own bits, independently.
    let packed: Vec<(usize, Vec<u8>)> = chunks
        .par_iter()
        .zip(starts.par_iter())
        .map(|(c, &start)| {
            let first_byte = (start / 8) as usize;
            let mut buf: Vec<u8> = Vec::with_capacity((c.len() * 4) / 8 + 2);
            let mut acc: u64 = 0;
            // Bits already occupied in the first byte by the previous chunk.
            let mut size: u32 = (start % 8) as u32;
            if size > 0 {
                acc = 0; // those bits belong to the neighbour; we write zeros
            }
            for &s in c.iter() {
                let b = bits[s as usize];
                acc = (acc << b) | vals[s as usize];
                size += b;
                while size >= 8 {
                    buf.push((acc >> (size - 8)) as u8);
                    size -= 8;
                    acc &= (1u64 << size) - 1;
                }
            }
            if size > 0 {
                // Leftover bits sit in the high positions of one more byte; the
                // next chunk ORs its own bits into the low positions.
                buf.push((acc << (8 - size)) as u8);
            }
            (first_byte, buf)
        })
        .collect();

    let mut bytes = vec![0u8; n_bytes.max(1)];
    for (first, buf) in &packed {
        for (i, b) in buf.iter().enumerate() {
            let idx = first + i;
            if idx < bytes.len() {
                // Each side writes only its own bits, so OR is exact.
                bytes[idx] |= *b;
            }
        }
    }
    bytes.truncate(n_bytes);

    // The EOF tail, exactly as the serial encoder writes it: one byte, and only
    // when the stream did not end byte-aligned.
    if tail_bits > 0 {
        let pending = u64::from(bytes[body_bytes] >> (8 - tail_bits));
        let combined = (pending << eof.bits) | eof.value;
        let size = tail_bits + eof.bits;
        bytes[body_bytes] = if size >= 8 {
            (combined >> (size - 8)) as u8
        } else {
            (combined << (8 - size)) as u8
        };
    }

    // gaps and output_positions: a window or block is opened by the first code
    // that STARTS in it, so each chunk can emit its own and the earliest wins.
    let n_windows = (total_bits as usize).div_ceil(window_bits).max(1);
    let mut gaps: Vec<u32> = vec![0; n_windows];
    let mut gap_set = vec![false; n_windows];
    let n_blocks_pos = (total_bits as usize).div_ceil(block_bits).max(1);
    let mut pos: Vec<u32> = vec![0; n_blocks_pos];
    let mut pos_set = vec![false; n_blocks_pos];

    let per_chunk: Vec<(Vec<(usize, u32)>, Vec<(usize, u32)>)> = chunks
        .par_iter()
        .zip(starts.par_iter())
        .enumerate()
        .map(|(ci, (c, &start))| {
            let mut g = Vec::new();
            let mut o = Vec::new();
            let mut p = start;
            let mut elem = (ci * chunk_symbols) as u32;
            let mut last_w: Option<usize> = None;
            let mut last_b: Option<usize> = None;
            for &s in c.iter() {
                let w = (p as usize) / window_bits;
                if last_w != Some(w) {
                    g.push((w, (p as usize % window_bits) as u32));
                    last_w = Some(w);
                }
                let b = (p as usize) / block_bits;
                if last_b != Some(b) {
                    o.push((b, elem));
                    last_b = Some(b);
                }
                p += u64::from(bits[s as usize]);
                elem += 1;
            }
            (g, o)
        })
        .collect();

    for (g, o) in &per_chunk {
        for &(w, v) in g {
            if w < n_windows && !gap_set[w] {
                gaps[w] = v;
                gap_set[w] = true;
            }
        }
        for &(b, v) in o {
            if b < n_blocks_pos && !pos_set[b] {
                pos[b] = v;
                pos_set[b] = true;
            }
        }
    }

    // The serial encoder repeats both checks once more before writing EOF.
    if tail_bits > 0 {
        let w = (total_bits as usize) / window_bits;
        if w >= n_windows {
            gaps.push((total_bits as usize % window_bits) as u32);
        }
        let b = (total_bits as usize) / block_bits;
        if b >= n_blocks_pos {
            pos.push(exponents.len() as u32);
        }
    }

    let mut output_positions: Vec<u32> = pos
        .into_iter()
        .zip(pos_set)
        .filter_map(|(v, set)| set.then_some(v))
        .collect();
    output_positions.push(exponents.len() as u32);

    let mut gaps: Vec<u32> = gaps
        .into_iter()
        .zip(gap_set)
        .filter_map(|(v, set)| set.then_some(v))
        .collect();

    let blocks = bytes.len().div_ceil(threads_per_block * bytes_per_thread);
    let target = threads_per_block * blocks;
    if target > gaps.len() {
        gaps.resize(target, 0);
    }

    let mut packed_gaps = Vec::with_capacity(gaps.len() * 5 / 8 + 1);
    let mut a: u16 = 0;
    let mut nb: u32 = 0;
    for g in &gaps {
        a = (a << 5) | (*g as u16 & 0x1F);
        nb += 5;
        while nb >= 8 {
            packed_gaps.push((a >> (nb - 8)) as u8);
            nb -= 8;
            a &= (1 << nb) - 1;
        }
    }
    if nb > 0 {
        packed_gaps.push((a << (8 - nb)) as u8);
    }

    Encoded {
        bytes,
        gaps: packed_gaps,
        output_positions,
    }
}

/// Per-symbol code lengths and values, indexed by exponent.
pub(crate) fn code_table(cb: &Codebook) -> ([u32; 256], [u64; 256]) {
    let (mut bits, mut vals) = ([0u32; 256], [0u64; 256]);
    for (s, c) in cb.entries() {
        if let Sym::Val(v) = s {
            bits[*v as usize] = c.bits;
            vals[*v as usize] = c.value;
        }
    }
    (bits, vals)
}
