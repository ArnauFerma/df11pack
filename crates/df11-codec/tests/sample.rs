//! Phase 6 -- the stratified sampling plan.

use df11_codec::sample::{chunk_of, plan, UnitShape};
use std::collections::BTreeSet;

fn shapes() -> Vec<UnitShape> {
    vec![
        UnitShape {
            chunks: 1278,
            boundary_chunks: vec![170, 256, 341, 512, 768, 1023],
        },
        UnitShape {
            chunks: 1,
            boundary_chunks: vec![],
        },
        UnitShape {
            chunks: 50,
            boundary_chunks: vec![0, 49],
        },
        UnitShape {
            chunks: 900,
            boundary_chunks: vec![],
        },
    ]
}

#[test]
fn every_unit_is_covered_at_its_first_and_last_chunk() {
    let p: BTreeSet<_> = plan(&shapes(), 1000, 7).into_iter().collect();
    for (u, s) in shapes().iter().enumerate() {
        assert!(p.contains(&(u, 0)), "unit {u}: first chunk missing");
        assert!(
            p.contains(&(u, s.chunks - 1)),
            "unit {u}: last chunk missing"
        );
    }
}

#[test]
fn every_boundary_chunk_is_included() {
    let p: BTreeSet<_> = plan(&shapes(), 1000, 7).into_iter().collect();
    for (u, s) in shapes().iter().enumerate() {
        for &c in &s.boundary_chunks {
            assert!(p.contains(&(u, c)), "unit {u}: boundary chunk {c} missing");
        }
    }
}

#[test]
fn the_budget_is_spent_and_not_exceeded_beyond_the_mandatory_picks() {
    let p = plan(&shapes(), 1000, 7);
    assert_eq!(p.len(), 1000, "the budget must be used in full");
    let unique: BTreeSet<_> = p.iter().collect();
    assert_eq!(unique.len(), p.len(), "no chunk picked twice");
    for &(u, c) in &p {
        assert!(c < shapes()[u].chunks, "pick ({u}, {c}) is out of range");
    }
}

/// A budget smaller than the mandatory set still includes all of it: coverage of
/// every unit and every boundary is not traded away to hit a number.
#[test]
fn mandatory_picks_survive_a_tiny_budget() {
    let p: BTreeSet<_> = plan(&shapes(), 3, 7).into_iter().collect();
    // 4 units -> first/last (unit 1 has one chunk), plus 8 boundaries, deduplicated.
    assert!(
        p.len() >= 3 + 4,
        "mandatory picks must not be cut to fit the budget"
    );
    for (u, s) in shapes().iter().enumerate() {
        assert!(p.contains(&(u, 0)) && p.contains(&(u, s.chunks - 1)));
    }
}

#[test]
fn a_budget_covering_everything_picks_every_chunk() {
    let total: usize = shapes().iter().map(|s| s.chunks).sum();
    let p = plan(&shapes(), total + 100, 1);
    assert_eq!(p.len(), total, "every chunk, each once");
}

#[test]
fn the_same_seed_gives_the_same_plan_and_a_different_one_does_not() {
    assert_eq!(plan(&shapes(), 500, 42), plan(&shapes(), 500, 42));
    assert_ne!(
        plan(&shapes(), 500, 42),
        plan(&shapes(), 500, 43),
        "the seed must actually drive the uniform part"
    );
}

/// The uniform part must reach the whole unit, not cluster at one end.
#[test]
fn the_uniform_part_spreads_across_the_unit() {
    let s = vec![UnitShape {
        chunks: 10_000,
        boundary_chunks: vec![],
    }];
    let p = plan(&s, 400, 99);
    let (lo, hi) = p.iter().fold(
        (0, 0),
        |(lo, hi), &(_, c)| {
            if c < 5000 {
                (lo + 1, hi)
            } else {
                (lo, hi + 1)
            }
        },
    );
    assert!(lo > 150 && hi > 150, "picks cluster: {lo} low, {hi} high");
}

#[test]
fn chunk_of_finds_the_chunk_containing_an_element() {
    // Chunks start at elements 0, 10, 10 (an empty chunk), 25; the trailing entry
    // is the total, 40.
    let pos = [0u32, 10, 10, 25, 40];
    assert_eq!(chunk_of(&pos, 0), Some(0));
    assert_eq!(chunk_of(&pos, 9), Some(0));
    assert_eq!(chunk_of(&pos, 10), Some(2), "an empty chunk holds nothing");
    assert_eq!(chunk_of(&pos, 24), Some(2));
    assert_eq!(chunk_of(&pos, 25), Some(3));
    assert_eq!(chunk_of(&pos, 39), Some(3));
    assert_eq!(chunk_of(&pos, 40), None, "past the end");
}
