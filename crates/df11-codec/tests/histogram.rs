//! Step 1.4 -- the exponent histogram and the gates that must never be skipped.

use df11_codec::{
    check_code_len, check_prefix_tables, check_unit_limits, split_fields, EncodeError, Histogram,
    MAX_UNIT_BYTES, MAX_UNIT_WEIGHTS,
};
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
fn counts_every_symbol() {
    let h = Histogram::build(&[5, 5, 7, 0, 5, 0]).expect("no reserved exponents here");
    assert_eq!(h.total(), 6);
    assert_eq!(h.counts()[5], 3);
    assert_eq!(h.counts()[7], 1);
    assert_eq!(h.counts()[0], 2);
    assert_eq!(h.counts()[9], 0);
    assert_eq!(h.distinct(), 3);
    assert_eq!(h.present_symbols(), vec![0, 5, 7]);
}

#[test]
fn empty_input_is_an_empty_histogram() {
    let h = Histogram::build(&[]).expect("empty is valid");
    assert_eq!(h.total(), 0);
    assert_eq!(h.distinct(), 0);
}

#[test]
fn reserved_exponents_abort_with_the_value_and_position() {
    for v in [240u8, 250, 255] {
        let mut input = vec![7u8; 100];
        input[63] = v;
        let e = Histogram::build(&input).expect_err("reserved exponent must abort");
        assert_eq!(
            e,
            EncodeError::ReservedExponent {
                value: v,
                index: 63
            }
        );
        let msg = e.to_string();
        assert!(msg.contains("reserved"), "message must explain why: {msg}");
        assert!(
            msg.contains("no file was written"),
            "message must say nothing was emitted: {msg}"
        );
    }
}

#[test]
fn the_boundary_is_240_not_239() {
    assert!(
        Histogram::build(&[239u8]).is_ok(),
        "239 is a legal exponent"
    );
    assert!(Histogram::build(&[240u8]).is_err(), "240 is reserved");
}

#[test]
fn unit_limits_reject_only_past_the_kernels_range() {
    assert!(check_unit_limits(MAX_UNIT_WEIGHTS, MAX_UNIT_BYTES).is_ok());
    assert_eq!(
        check_unit_limits(MAX_UNIT_WEIGHTS + 1, 10).unwrap_err(),
        EncodeError::TooManyWeights {
            weights: MAX_UNIT_WEIGHTS + 1
        }
    );
    assert_eq!(
        check_unit_limits(10, MAX_UNIT_BYTES + 1).unwrap_err(),
        EncodeError::TooManyBytes {
            bytes: MAX_UNIT_BYTES + 1
        }
    );
}

#[test]
fn prefix_table_and_code_length_gates() {
    // 17 is the real bound: jump values 240..=255 reach tables 1..=16, plus
    // table 0. An 18th table's jump value would fall below 240 and be read as a
    // symbol instead.
    assert!(check_prefix_tables(17).is_ok());
    assert_eq!(
        check_prefix_tables(18).unwrap_err(),
        EncodeError::TooManyPrefixTables { tables: 18 }
    );
    assert!(check_code_len(32).is_ok());
    assert_eq!(
        check_code_len(33).unwrap_err(),
        EncodeError::CodeTooLong { bits: 33 }
    );
}

/// Fixture-backed: the symbols our histogram finds must be exactly the symbols
/// the official codebook assigned a code length to. The LUT's final row holds
/// one length per symbol, zero where the symbol does not occur.
#[test]
fn present_symbols_match_the_official_codebook() {
    let Some(fx) = skip_if_missing("present_symbols_match_the_official_codebook") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0 set");
    let Some(src) = SourceModel::open(set) else {
        return;
    };

    let mut checked = 0;
    for unit in set.unit_names() {
        let input = src.unit_input(&unit, &QWEN3_LAYER).expect("source weights");
        let (exp, _sm) = split_fields(&input);
        let h = Histogram::build(&exp).expect("real weights have no reserved exponents");
        assert_eq!(h.total(), (input.len() / 2) as u64);

        let luts = set
            .unit(&unit)
            .into_iter()
            .find(|t| t.name.ends_with("luts"))
            .expect("luts fixture");
        let bytes = luts.read().expect("read luts");
        let rows = luts.shape[0] as usize;
        assert_eq!(luts.shape[1], 256);
        // The last row is the per-symbol code lengths.
        let lens = &bytes[(rows - 1) * 256..rows * 256];

        let official: Vec<u8> = (0..=255u8).filter(|&s| lens[s as usize] > 0).collect();
        assert_eq!(
            h.present_symbols(),
            official,
            "{unit}: histogram symbols disagree with the official codebook"
        );
        assert!(h.distinct() > 1, "{unit}: expected a real distribution");
        checked += 1;
    }
    assert_eq!(checked, 4, "tier0 has 4 units");
}
