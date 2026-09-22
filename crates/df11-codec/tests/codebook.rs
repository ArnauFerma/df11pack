//! Step 1.5 -- the dahuffman-compatible codebook and the LUTs.
//!
//! Graded against real official output. Comparing the LUTs byte-for-byte
//! exercises the tie-breaking, the merge order, the prefix discovery order and
//! the forward-fill all at once: any of them wrong moves bytes.

use df11_codec::huffman::{build_luts, Codebook, Sym};
use df11_codec::{split_fields, Histogram};
use df11_fixtures::{skip_if_missing, SourceModel};

const QWEN3_LAYER: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[test]
fn eof_orders_below_every_real_symbol() {
    assert!(Sym::Eof < Sym::Val(0));
    assert!(Sym::Eof < Sym::Val(255));
    assert!(Sym::Val(0) > Sym::Eof);
    assert!(Sym::Val(3) < Sym::Val(4));
}

#[test]
fn codes_are_prefix_free_and_kraft_exact() {
    let freqs: Vec<(u8, u64)> = vec![(1, 45), (2, 13), (3, 12), (4, 16), (5, 9), (6, 5)];
    let cb = Codebook::build(&freqs);
    // Kraft equality holds for a complete Huffman code.
    let sum: f64 = cb
        .entries()
        .iter()
        .map(|(_, c)| 2f64.powi(-(c.bits as i32)))
        .sum();
    assert!((sum - 1.0).abs() < 1e-12, "Kraft sum was {sum}, expected 1");

    // No code is a prefix of another.
    for (sa, ca) in cb.entries() {
        for (sb, cb2) in cb.entries() {
            if sa == sb || ca.bits > cb2.bits {
                continue;
            }
            let shifted = cb2.value >> (cb2.bits - ca.bits);
            assert_ne!(
                shifted, ca.value,
                "{sa:?} is a prefix of {sb:?}: {ca:?} vs {cb2:?}"
            );
        }
    }
}

#[test]
fn eof_is_injected_even_though_it_is_never_an_exponent() {
    let cb = Codebook::build(&[(7, 100), (9, 3)]);
    assert!(
        cb.entries().iter().any(|(s, _)| *s == Sym::Eof),
        "dahuffman always adds an EOF symbol"
    );
    assert_eq!(cb.entries().len(), 3);
}

/// The load-bearing one.
#[test]
fn luts_match_official_output_byte_for_byte() {
    let Some(fx) = skip_if_missing("luts_match_official_output_byte_for_byte") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };

    let mut checked = 0;
    for unit in set.unit_names() {
        let input = src.unit_input(&unit, &QWEN3_LAYER).expect("source weights");
        let (exp, _sm) = split_fields(&input);
        let h = Histogram::build(&exp).expect("no reserved exponents");
        let cb = Codebook::build(&h.frequencies());
        let rows = build_luts(&cb).expect("luts within limits");

        let fixture = set
            .unit(&unit)
            .into_iter()
            .find(|t| t.name.ends_with("luts"))
            .expect("luts fixture");
        let official = fixture.read().expect("read luts");

        assert_eq!(
            rows.len() as u64,
            fixture.shape[0],
            "{unit}: produced {} LUT rows, official has {}",
            rows.len(),
            fixture.shape[0]
        );
        let flat: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(flat.len(), official.len());
        if flat != official {
            let i = flat
                .iter()
                .zip(&official)
                .position(|(a, b)| a != b)
                .expect("lengths equal");
            panic!(
                "{unit}: LUTs differ at row {} col {}: ours {} official {}",
                i / 256,
                i % 256,
                flat[i],
                official[i]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 4);
}

/// The leak must be reproduced, not merely tolerated: assert it is present in
/// our own output, so an implementation that "fixed" it fails here.
#[test]
fn the_cross_row_carry_over_is_reproduced() {
    let Some(fx) = skip_if_missing("the_cross_row_carry_over_is_reproduced") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let input = src
        .unit_input("model.layers.0", &QWEN3_LAYER)
        .expect("source");
    let (exp, _) = split_fields(&input);
    let h = Histogram::build(&exp).unwrap();
    let rows = build_luts(&Codebook::build(&h.frequencies())).unwrap();

    // Row 3 of this unit has no entry for byte 0, so its leading cells carry
    // row 2's trailing value (FINDINGS 0.5). A per-row reset would make them 0.
    let leaked = rows[3][0];
    assert_eq!(
        leaked, rows[2][255],
        "row 3 must begin with row 2's trailing value"
    );
    assert_ne!(leaked, 0, "a per-row reset would leave 0 here");
    assert!(
        rows[3][..128].iter().all(|&v| v == leaked),
        "the whole leaked run must carry the same value"
    );
}

/// `--luts=correct` must differ from compat in exactly the leaked positions and
/// nowhere else. A mode that changed anything further would not be "the same
/// file with the leak removed", it would be a different codec.
#[test]
fn correct_mode_differs_only_where_the_leak_is() {
    use df11_codec::huffman::{build_luts_with, LutMode};

    let Some(fx) = skip_if_missing("correct_mode_differs_only_where_the_leak_is") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };

    let mut units_with_leak = 0;
    for unit in set.unit_names() {
        let input = src.unit_input(&unit, &QWEN3_LAYER).expect("source");
        let (exp, _) = split_fields(&input);
        let h = Histogram::build(&exp).unwrap();
        let cb = Codebook::build(&h.frequencies());

        let compat = build_luts_with(&cb, LutMode::Compat).unwrap();
        let correct = build_luts_with(&cb, LutMode::Correct).unwrap();
        assert_eq!(compat.len(), correct.len(), "{unit}: same number of rows");

        // Predict the leaked span independently: for each row after the first
        // whose own entries do not start at byte 0, the leading run carries the
        // previous row's trailing value in compat and must be 0 in correct.
        let mut leaked_cells = 0;
        for r in 0..compat.len() {
            for c in 0..256 {
                if compat[r][c] == correct[r][c] {
                    continue;
                }
                leaked_cells += 1;
                assert!(r > 0, "{unit}: row 0 cannot leak, nothing precedes it");
                assert_eq!(correct[r][c], 0, "{unit}: correct mode must zero-fill");
                assert_eq!(
                    compat[r][c],
                    compat[r - 1][255],
                    "{unit}: a differing cell must hold the previous row's trailing value"
                );
                assert!(
                    (0..c).all(|k| compat[r][k] == compat[r][c]),
                    "{unit}: the leak is a prefix run, so every earlier cell matches"
                );
            }
        }
        if leaked_cells > 0 {
            units_with_leak += 1;
        }
        // The lens row is data, never filled, so it must be untouched.
        assert_eq!(
            compat.last().unwrap(),
            correct.last().unwrap(),
            "{unit}: the code-length row must be identical in both modes"
        );
    }
    assert!(
        units_with_leak > 0,
        "no unit exhibited the leak, so this test proved nothing"
    );
}
