//! The `idx8` index scheme (docs/INDEX_SCHEMES.md), from the sibling project
//! `bf16-exponent-compression` (`kernel_idx8.py`, `build_index8`).
//!
//! **Not DF11-compatible.** The official kernel cannot read it. It shares
//! everything with DF11 except the index: the same codebook, LUTs, exponent
//! bitstream and sign/mantissa bytes. In place of `gaps` and `output_positions`:
//!
//! - blocks of a fixed `block` symbols, so each block's first output element is
//!   simply `b * block` -- no output-side index at all;
//! - `idx8_lengths`: one `u8` per block, its length in bits minus the unit's
//!   minimum block length;
//! - `idx8_superblocks`: one `u32` per 32 blocks (a warp), the absolute bit offset
//!   of its first block. A block's start is its superblock's base plus the
//!   lengths of the blocks before it in that superblock.
//! - **Escapes** (df11pack's extension, chosen by the user after measuring real
//!   layers): the u8 value 255 means "this block's length is in the side table".
//!   `idx8_escape_blocks` lists those blocks (sorted u32) and
//!   `idx8_escape_lengths` their exact lengths. Real Qwen3 layers need 0 to ~130
//!   per 245,760 blocks -- rare exponents with 20-25-bit codes clustering -- so
//!   the kernel's prefix sum is unchanged and a lane with code 255 does one
//!   lookup. Without escapes most real units cannot be indexed at all.
//!
//! The kernel pads both arrays to its grid; the file stores them unpadded.

use std::fmt;

/// Blocks per superblock: one warp.
pub const SUPERBLOCK: usize = 32;

/// The length code that means "see the escape table".
pub const ESCAPE: u8 = 255;

/// The index for one unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idx8 {
    pub block: u32,
    pub minlen: u32,
    pub lengths: Vec<u8>,
    pub superblocks: Vec<u32>,
    /// `(block, exact length in bits)` for every block coded [`ESCAPE`], sorted.
    pub escapes: Vec<(u32, u32)>,
    /// Total bits of the symbol stream, excluding the EOF code.
    pub total_bits: u64,
}

/// Why a unit cannot be indexed this way. Never a silent fallback to DF11.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Idx8Error {
    /// The stream is longer than a `u32` bit offset can address.
    TooLong {
        bits: u64,
    },
    BadBlock(usize),
}

impl fmt::Display for Idx8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong { bits } => write!(
                f,
                "idx8: the unit's stream is {bits} bits, past the u32 offsets idx8 stores"
            ),
            Self::BadBlock(b) => write!(f, "idx8: block size {b} must be at least 1"),
        }
    }
}

impl std::error::Error for Idx8Error {}

/// Build the index from each symbol's code length, in stream order.
pub fn build(code_lengths: impl Iterator<Item = u32>, block: usize) -> Result<Idx8, Idx8Error> {
    if block == 0 {
        return Err(Idx8Error::BadBlock(block));
    }
    let mut lens: Vec<u64> = Vec::new();
    let (mut cur, mut in_block, mut total) = (0u64, 0usize, 0u64);
    for l in code_lengths {
        cur += u64::from(l);
        total += u64::from(l);
        in_block += 1;
        if in_block == block {
            lens.push(cur);
            cur = 0;
            in_block = 0;
        }
    }
    if in_block > 0 {
        lens.push(cur);
    }
    if total > u64::from(u32::MAX) {
        return Err(Idx8Error::TooLong { bits: total });
    }
    // The minimum comes from full blocks: a final partial block can be far
    // shorter, and would push every other block toward an escape.
    let full = if in_block > 0 && lens.len() > 1 {
        &lens[..lens.len() - 1]
    } else {
        &lens[..]
    };
    let min = full.iter().copied().min().unwrap_or(0);
    let mut superblocks = Vec::with_capacity(lens.len().div_ceil(SUPERBLOCK));
    let mut off = 0u64;
    for (b, l) in lens.iter().enumerate() {
        if b % SUPERBLOCK == 0 {
            superblocks.push(off as u32);
        }
        off += l;
    }
    let mut escapes = Vec::new();
    let mut lengths = Vec::with_capacity(lens.len());
    for (b, &l) in lens.iter().enumerate() {
        match l.checked_sub(min).filter(|d| *d < u64::from(ESCAPE)) {
            Some(d) => lengths.push(d as u8),
            None => {
                // Too long for a u8 -- or, for a short final block, below the
                // minimum. Either way the exact length goes in the side table.
                lengths.push(ESCAPE);
                escapes.push((b as u32, l as u32));
            }
        }
    }
    Ok(Idx8 {
        block: block as u32,
        minlen: min as u32,
        lengths,
        superblocks,
        escapes,
        total_bits: total,
    })
}

impl Idx8 {
    /// Block `b`'s start and end bit, recovered exactly as the kernel does.
    pub fn block_range(&self, b: usize) -> Option<(u64, u64)> {
        if b >= self.lengths.len() {
            return None;
        }
        let base = u64::from(*self.superblocks.get(b / SUPERBLOCK)?);
        let mut start = base;
        for i in b - b % SUPERBLOCK..b {
            start += self.length(i)?;
        }
        Some((start, start + self.length(b)?))
    }

    /// Block `i`'s length: its code plus the minimum, or its escape entry.
    pub fn length(&self, i: usize) -> Option<u64> {
        let code = *self.lengths.get(i)?;
        if code == ESCAPE {
            let k = self
                .escapes
                .binary_search_by_key(&(i as u32), |e| e.0)
                .ok()?;
            Some(u64::from(self.escapes[k].1))
        } else {
            Some(u64::from(code) + u64::from(self.minlen))
        }
    }

    /// The `idx8_meta` tensor: `[block, minlen, total_bits]` as little-endian
    /// i64. `total_bits` is where the last block ends, which no length encodes.
    pub fn meta_bytes(&self) -> Vec<u8> {
        [
            i64::from(self.block),
            i64::from(self.minlen),
            self.total_bits as i64,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
    }

    /// The escape table as its two tensors: block indices, then lengths.
    pub fn escape_bytes(&self) -> (Vec<u8>, Vec<u8>) {
        (
            self.escapes
                .iter()
                .flat_map(|e| e.0.to_le_bytes())
                .collect(),
            self.escapes
                .iter()
                .flat_map(|e| e.1.to_le_bytes())
                .collect(),
        )
    }

    /// Rebuild the index from its five stored tensors.
    pub fn from_tensors(
        lengths: &[u8],
        superblocks: &[u8],
        meta: &[u8],
        escape_blocks: &[u8],
        escape_lengths: &[u8],
    ) -> Option<Self> {
        if meta.len() != 24
            || superblocks.len() % 4 != 0
            || escape_blocks.len() % 4 != 0
            || escape_blocks.len() != escape_lengths.len()
        {
            return None;
        }
        let u32s = |b: &[u8]| -> Vec<u32> {
            b.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let m: Vec<i64> = meta
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some(Idx8 {
            block: u32::try_from(m[0]).ok()?,
            minlen: u32::try_from(m[1]).ok()?,
            total_bits: u64::try_from(m[2]).ok()?,
            lengths: lengths.to_vec(),
            superblocks: u32s(superblocks),
            escapes: u32s(escape_blocks)
                .into_iter()
                .zip(u32s(escape_lengths))
                .collect(),
        })
    }

    pub fn superblock_bytes(&self) -> Vec<u8> {
        self.superblocks
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}
