//! Step 1.3 -- the field split, graded against real official output.
//!
//! `sign_mantissa` is stored verbatim by the official encoder, so it can be
//! compared byte-for-byte against the fixtures without any of the codec being
//! written yet. That makes it the earliest possible real check on the pipeline.

use df11_codec::split_fields;
use df11_fixtures::{assert_matches, skip_if_missing, SourceModel};

/// The Qwen3 layer unit, in the order the official `pattern_dict` lists it.
/// Order is load-bearing: it fixes the concatenation (FINDINGS 0.9).
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
fn split_is_reversible() {
    // Every 16-bit pattern, so no exponent or mantissa value is untested.
    let mut buf = Vec::with_capacity(65536 * 2);
    for w in 0u32..=0xFFFF {
        buf.extend_from_slice(&(w as u16).to_le_bytes());
    }
    let (exp, sm) = split_fields(&buf);
    assert_eq!(exp.len(), 65536);
    assert_eq!(sm.len(), 65536);
    for w in 0u32..=0xFFFF {
        let i = w as usize;
        let rebuilt =
            ((u16::from(sm[i] & 0x80)) << 8) | ((u16::from(exp[i])) << 7) | u16::from(sm[i] & 0x7F);
        assert_eq!(rebuilt, w as u16, "round trip failed for 0x{w:04x}");
    }
}

#[test]
fn known_bit_patterns() {
    // 0x3F80 = 1.0f32 in BF16: sign 0, exponent 0x7F, mantissa 0.
    // 0xBF80 = -1.0: sign 1, same exponent and mantissa.
    let buf = [0x80u8, 0x3F, 0x80, 0xBF];
    let (exp, sm) = split_fields(&buf);
    assert_eq!(exp, vec![0x7F, 0x7F], "exponent of +/-1.0 is 0x7F");
    assert_eq!(sm, vec![0x00, 0x80], "sign bit lands in bit 7");
}

/// The real check: reproduce `sign_mantissa` byte-for-byte for every unit of a
/// real official output, from the real source weights.
#[test]
fn sign_mantissa_matches_official_output_for_every_unit() {
    let Some(fx) = skip_if_missing("sign_mantissa_matches_official_output_for_every_unit") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0 set");
    let Some(src) = SourceModel::open(set) else {
        eprintln!("SKIP: source model absent at {}", set.source_dir.display());
        return;
    };

    let units = set.unit_names();
    assert_eq!(units.len(), 4, "tier0 has 4 layer units");

    for unit in &units {
        let input = src
            .unit_input(unit, &QWEN3_LAYER)
            .unwrap_or_else(|e| panic!("reading source for {unit}: {e}"));
        let (_exp, sm) = split_fields(&input);

        let fixture = set
            .unit(unit)
            .into_iter()
            .find(|t| t.name.ends_with("sign_mantissa"))
            .unwrap_or_else(|| panic!("no sign_mantissa fixture for {unit}"));

        assert_eq!(
            sm.len() as u64,
            fixture.bytes,
            "{unit}: produced {} bytes, official has {}",
            sm.len(),
            fixture.bytes
        );
        assert_matches(fixture, &sm);
    }
}

/// The exponent stream is not stored raw, but its length and range are checkable,
/// and DESIGN 1.3 forbids 240..=255 (the LUT jump convention would misread them).
#[test]
fn exponents_are_in_range_for_real_weights() {
    let Some(fx) = skip_if_missing("exponents_are_in_range_for_real_weights") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0 set");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let unit = "model.layers.0";
    let input = src.unit_input(unit, &QWEN3_LAYER).expect("source");
    let (exp, _sm) = split_fields(&input);
    assert_eq!(exp.len(), input.len() / 2);
    assert!(
        exp.iter().all(|&e| e < 240),
        "real BF16 weights must not produce exponents 240..=255"
    );
}
