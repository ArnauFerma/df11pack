//! Which chunks to check when not checking all of them. DESIGN §7.3.
//!
//! Uniform sampling alone detects a *systematic* fault (one touching a fraction f
//! of chunks) with probability 1 − (1 − f)^k, but an *isolated* one -- a single bad
//! chunk among a million -- only about k in a million times. So the plan is
//! stratified: the places faults concentrate are always included, and uniform
//! samples fill the rest of the budget.

/// What the planner needs to know about one unit.
#[derive(Debug, Clone)]
pub struct UnitShape {
    /// Number of kernel chunks.
    pub chunks: usize,
    /// Chunks containing a boundary between concatenated tensors.
    pub boundary_chunks: Vec<usize>,
}

/// One chunk to check: `(unit index, chunk index)`.
pub type Pick = (usize, usize);

/// A deterministic plan: the same seed always yields the same picks, so a run
/// can be reproduced exactly from its report.
///
/// Always included, even past `budget`: the first and last chunk of every unit
/// (so every unit is covered at least once) and every boundary chunk. The rest of
/// the budget is spent uniformly at random over the remaining chunks.
pub fn plan(units: &[UnitShape], budget: usize, seed: u64) -> Vec<Pick> {
    use std::collections::BTreeSet;

    let mut picks: Vec<Pick> = Vec::new();
    let mut taken: BTreeSet<Pick> = BTreeSet::new();
    let add = |p: Pick, picks: &mut Vec<Pick>, taken: &mut BTreeSet<Pick>| {
        if taken.insert(p) {
            picks.push(p);
        }
    };

    for (u, s) in units.iter().enumerate() {
        if s.chunks == 0 {
            continue;
        }
        add((u, 0), &mut picks, &mut taken);
        add((u, s.chunks - 1), &mut picks, &mut taken);
        for &c in &s.boundary_chunks {
            if c < s.chunks {
                add((u, c), &mut picks, &mut taken);
            }
        }
    }

    let total: usize = units.iter().map(|s| s.chunks).sum();
    let want = budget.saturating_sub(picks.len());
    let free = total - taken.len();
    if want == 0 || free == 0 {
        return picks;
    }

    // Map a flat index over all chunks to (unit, chunk).
    let starts: Vec<usize> = units
        .iter()
        .scan(0usize, |acc, s| {
            let here = *acc;
            *acc += s.chunks;
            Some(here)
        })
        .collect();
    let at = |i: usize| -> Pick {
        let u = starts.partition_point(|&s| s <= i) - 1;
        (u, i - starts[u])
    };

    let mut state = seed;
    if want >= free {
        for i in 0..total {
            add(at(i), &mut picks, &mut taken);
        }
    } else if want * 2 > free {
        // Dense: shuffle the free chunks and take a prefix.
        let mut pool: Vec<Pick> = (0..total).map(at).filter(|p| !taken.contains(p)).collect();
        for i in 0..want {
            let j = i + (splitmix64(&mut state) % (pool.len() - i) as u64) as usize;
            pool.swap(i, j);
            let p = pool[i];
            add(p, &mut picks, &mut taken);
        }
    } else {
        // Sparse: draw and reject repeats, without materialising every chunk.
        let target = picks.len() + want;
        while picks.len() < target {
            let i = (splitmix64(&mut state) % total as u64) as usize;
            add(at(i), &mut picks, &mut taken);
        }
    }
    picks
}

/// splitmix64: small, fast, and fully specified, so a seed means the same thing on
/// every platform and in every future version. No dependency.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The chunk that contains element `e`, given a unit's `output_positions`.
///
/// With empty chunks -- a chunk no code starts in, whose start equals the next
/// one's -- the element belongs to the last of the chunks sharing its start,
/// because that is the one that actually decodes it.
pub fn chunk_of(output_positions: &[u32], e: u64) -> Option<usize> {
    let (&total, starts) = output_positions.split_last()?;
    if e >= u64::from(total) || starts.is_empty() {
        return None;
    }
    let c = starts.partition_point(|&p| u64::from(p) <= e);
    c.checked_sub(1)
}
