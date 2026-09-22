//! Step 1.5 -- the 32-bit code-length limiter.
//!
//! Real models do not reach it: tier-1's longest code is 26 bits against a limit
//! of 32. So the oracle here is the H4 Python spec, which was itself validated
//! against `dahuffman` on 682 histograms.
//!
//! The contract under test is deliberately narrower than "reproduce NumPy".
//! Where the limiter's choice is forced, df11pack must match exactly. Where
//! `np.argpartition` picks among equal frequencies above 1 -- a choice NumPy does
//! not specify and which changes the codebook -- df11pack must refuse rather than
//! emit a file that might silently differ.

use df11_codec::huffman::{build_limited, Codebook};
use df11_codec::EncodeError;
use df11_fixtures::limiter_cases;

#[test]
fn unambiguous_cases_match_the_spec_exactly() {
    let Some(cases) = limiter_cases() else {
        eprintln!("SKIP: run phase0/gen_limiter_cases.py");
        return;
    };
    let mut checked = 0;
    for c in cases.iter().filter(|c| !c.ambiguous) {
        let built = build_limited(&c.freqs)
            .unwrap_or_else(|e| panic!("{}: expected success, got {e}", c.name));
        let cb = &built.codebook;

        // Pins WHICH demotion produced this table. Without it, a limiter that
        // demotes the wrong number of symbols converges on the same codebook an
        // iteration later and goes unnoticed -- verified: mutating the demotion
        // to take one symbol too few left every other assertion green.
        assert_eq!(
            built.iterations, c.iterations,
            "{}: limiter took {} iterations, spec took {}",
            c.name, built.iterations, c.iterations
        );

        assert_eq!(
            cb.max_bits(),
            c.max_bits_final,
            "{}: final max code length disagrees with the spec",
            c.name
        );
        assert!(cb.max_bits() <= 32, "{}: still over 32 bits", c.name);

        for (sym, expected) in &c.lengths {
            let s: u8 = sym.parse().expect("symbol");
            let got = cb
                .code_of(s)
                .unwrap_or_else(|| panic!("{}: no code for {s}", c.name));
            assert_eq!(
                got.bits, *expected,
                "{}: symbol {s} got {} bits, spec says {expected}",
                c.name, got.bits
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 5,
        "expected several unambiguous cases, got {checked}"
    );
}

#[test]
fn ambiguous_cases_are_refused_with_the_measured_numbers() {
    let Some(cases) = limiter_cases() else {
        return;
    };
    let mut checked = 0;
    for c in cases.iter().filter(|c| c.ambiguous) {
        let d = c.first_ambiguous.as_ref().expect("detail present");
        let err = build_limited(&c.freqs)
            .expect_err(&format!("{}: an ambiguous tie must be refused", c.name));
        assert_eq!(
            err,
            EncodeError::AmbiguousLimiterTie {
                min_k: d.min_k,
                boundary_frequency: d.boundary_value,
                tied: d.tied,
                slots: d.slots,
            },
            "{}: refusal must report the same numbers the spec measured",
            c.name
        );
        let msg = err.to_string();
        assert!(
            msg.contains("argpartition"),
            "message must name the cause: {msg}"
        );
        assert!(
            msg.contains("silently"),
            "message must say why we refuse rather than guess: {msg}"
        );
        checked += 1;
    }
    assert!(checked >= 2, "expected ambiguous cases, got {checked}");
}

#[test]
fn a_codebook_already_within_32_bits_is_returned_untouched() {
    let freqs: Vec<(u8, u64)> = vec![(1, 100), (2, 80), (3, 60), (4, 10)];
    let plain = Codebook::build(&freqs);
    let limited = build_limited(&freqs).expect("well under the limit");
    assert_eq!(
        limited.iterations, 0,
        "the limiter must not have run at all"
    );
    assert_eq!(plain.max_bits(), limited.codebook.max_bits());
    assert_eq!(
        plain.entries(),
        limited.codebook.entries(),
        "must not be rebuilt"
    );
}
