//! The `dahuffman`-compatible codebook and the hierarchical LUTs.
//!
//! Ported from `phase0/h4_codec_spec.py`, itself derived line by line from
//! `dahuffman` 0.4.2 and `dfloat11_utils`. Two things here are load-bearing and
//! easy to lose in a rewrite:
//!
//! 1. **Tie-breaking.** Heap nodes are ordered by `(frequency, representative)`,
//!    where a node's representative is the first symbol it was built from,
//!    inherited from whichever child sorted *smaller at merge time* — not the
//!    smallest symbol in its subtree. EOF compares below every real symbol.
//! 2. **Order.** The codebook is an ordered list, not a map. Its iteration order
//!    determines the order prefixes are discovered, hence `prefixes.index(..)`,
//!    hence the jump values written into the LUTs.

use crate::{check_code_len, check_prefix_tables, EncodeError};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A codebook symbol: a real exponent, or the end-of-stream marker.
///
/// `Eof` orders below every real symbol, unconditionally.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sym {
    Eof,
    Val(u8),
}

impl Ord for Sym {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Sym::Eof, Sym::Eof) => Ordering::Equal,
            (Sym::Eof, Sym::Val(_)) => Ordering::Less,
            (Sym::Val(_), Sym::Eof) => Ordering::Greater,
            (Sym::Val(a), Sym::Val(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for Sym {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// One symbol's code: `bits` long, `value` read MSB-first.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Code {
    pub bits: u32,
    pub value: u64,
}

struct Node {
    freq: u64,
    rep: Sym,
    leaves: Vec<(Sym, Code)>,
}

impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.freq == o.freq && self.rep == o.rep
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap and we need the smallest first.
        o.freq.cmp(&self.freq).then_with(|| o.rep.cmp(&self.rep))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// A `dahuffman`-compatible codebook, in the order the merge produced it.
#[derive(Clone, Debug)]
pub struct Codebook {
    entries: Vec<(Sym, Code)>,
}

impl Codebook {
    /// Build from symbol frequencies, ascending by symbol as `torch.unique` gives them.
    ///
    /// EOF is injected at frequency 1 when not already present, exactly as
    /// `dahuffman` does.
    pub fn build(freqs: &[(u8, u64)]) -> Self {
        assert!(!freqs.is_empty(), "need at least one symbol");
        let mut heap: BinaryHeap<Node> = freqs
            .iter()
            .map(|&(s, f)| Node {
                freq: f,
                rep: Sym::Val(s),
                leaves: vec![(Sym::Val(s), Code { bits: 0, value: 0 })],
            })
            .collect();
        heap.push(Node {
            freq: 1,
            rep: Sym::Eof,
            leaves: vec![(Sym::Eof, Code { bits: 0, value: 0 })],
        });

        while heap.len() > 1 {
            let a = heap.pop().expect("two nodes");
            let b = heap.pop().expect("two nodes");
            let mut leaves = Vec::with_capacity(a.leaves.len() + b.leaves.len());
            // `a` takes bit 0: length grows, value unchanged.
            for (s, c) in &a.leaves {
                leaves.push((
                    *s,
                    Code {
                        bits: c.bits + 1,
                        value: c.value,
                    },
                ));
            }
            // `b` takes bit 1, prepended at the leaf's current width.
            for (s, c) in &b.leaves {
                leaves.push((
                    *s,
                    Code {
                        bits: c.bits + 1,
                        value: (1u64 << c.bits) + c.value,
                    },
                ));
            }
            heap.push(Node {
                freq: a.freq + b.freq,
                rep: a.rep,
                leaves,
            });
        }
        Codebook {
            entries: heap.pop().expect("one node").leaves,
        }
    }

    /// Entries in merge order. The order matters; see the module docs.
    pub fn entries(&self) -> &[(Sym, Code)] {
        &self.entries
    }

    pub fn max_bits(&self) -> u32 {
        self.entries.iter().map(|(_, c)| c.bits).max().unwrap_or(0)
    }

    pub fn code_of(&self, s: u8) -> Option<Code> {
        self.entries
            .iter()
            .find(|(sym, _)| *sym == Sym::Val(s))
            .map(|(_, c)| *c)
    }

    /// Per-symbol code lengths, indexed by exponent value; 0 where absent.
    pub fn lengths(&self) -> [u8; 256] {
        let mut l = [0u8; 256];
        for (s, c) in &self.entries {
            if let Sym::Val(v) = s {
                l[*v as usize] = c.bits as u8;
            }
        }
        l
    }
}

fn bits_string(code: &Code) -> String {
    format!("{:0width$b}", code.value, width = code.bits as usize)
}

/// Build the hierarchical byte LUTs: `(n_prefixes + 1, 256)`, the final row
/// holding per-symbol code lengths.
///
/// Reproduces the official forward-fill, **including its carry-over across
/// rows**. Within a row the fill is deliberate: a code shorter than the byte
/// boundary sets one entry and the fill replicates it across every byte value
/// sharing that prefix. Across rows it is accidental — the accumulator is never
/// reset — and a row whose first key is not 0 therefore begins with the previous
/// row's trailing value. Byte-identity requires both. See `docs/COMPATIBILITY.md`.
pub fn build_luts(cb: &Codebook) -> Result<Vec<[u8; 256]>, EncodeError> {
    let mut prefixes: Vec<String> = vec![String::new()];
    for (s, c) in cb.entries() {
        if matches!(s, Sym::Val(_)) {
            let b = bits_string(c);
            let keep = ((c.bits as usize).saturating_sub(1)) / 8 * 8;
            let p = b[..keep.min(b.len())].to_string();
            if !prefixes.contains(&p) {
                prefixes.push(p);
            }
        }
    }
    // Stable, so equal-length prefixes keep discovery order.
    prefixes.sort_by_key(|p| p.len());
    check_prefix_tables(prefixes.len())?;
    check_code_len(cb.max_bits())?;

    let mut rows: Vec<[u8; 256]> = Vec::with_capacity(prefixes.len() + 1);
    let mut curr_val: u8 = 0; // never reset between rows, on purpose

    for p in &prefixes {
        let pl = p.len() / 8;
        let mut bytes_map: Vec<Option<u8>> = vec![None; 256];
        for (s, c) in cb.entries() {
            let Sym::Val(key) = s else { continue };
            let bin_val = bits_string(c);
            if !bin_val.starts_with(p.as_str()) {
                continue;
            }
            let (dict_key, dict_value) = if ((c.bits as usize).saturating_sub(1)) / 8 == pl {
                let mut tail = bin_val[(pl * 8).min(bin_val.len())..].to_string();
                while tail.len() < 8 {
                    tail.push('0');
                }
                (u32::from_str_radix(&tail, 2).unwrap_or(0) as usize, *key)
            } else {
                let lo = pl * 8;
                let hi = (pl * 8 + 8).min(bin_val.len());
                let seg = &bin_val[lo..hi];
                let full = &bin_val[..hi];
                let idx = prefixes
                    .iter()
                    .position(|x| x == full)
                    .expect("prefix discovered during the first pass");
                (
                    u32::from_str_radix(seg, 2).unwrap_or(0) as usize,
                    (256 - idx) as u8,
                )
            };
            match bytes_map[dict_key] {
                Some(v) if v != dict_value => {
                    unreachable!("LUT key {dict_key} already holds {v}, would be {dict_value}")
                }
                _ => bytes_map[dict_key] = Some(dict_value),
            }
        }
        let mut row = [0u8; 256];
        for (i, cell) in row.iter_mut().enumerate() {
            if let Some(v) = bytes_map[i] {
                curr_val = v;
            }
            *cell = curr_val;
        }
        rows.push(row);
    }
    rows.push(cb.lengths());
    Ok(rows)
}
