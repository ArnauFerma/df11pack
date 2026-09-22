//! Phase 1 exit gate: all six DF11 tensors, byte-identical, for every unit.

use df11_codec::unit::encode_unit;
use df11_fixtures::{assert_matches, skip_if_missing, SourceModel};

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
fn all_six_tensors_match_official_output_for_every_unit() {
    let Some(fx) = skip_if_missing("all_six_tensors_match_official_output_for_every_unit") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };

    let mut units_checked = 0;
    let mut tensors_checked = 0;
    for unit in set.unit_names() {
        let owned: Vec<Vec<u8>> = QWEN3_LAYER
            .iter()
            .map(|a| {
                src.tensor(&format!("{unit}.{a}.weight"))
                    .unwrap_or_else(|e| panic!("{unit}.{a}: {e}"))
            })
            .collect();
        let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

        let out = encode_unit(&unit, &refs, 512, 8)
            .unwrap_or_else(|e| panic!("{unit}: encode failed: {e}"));
        assert_eq!(out.name, unit);

        let produced = out.tensors();
        assert_eq!(produced.len(), 6, "{unit}: six tensors per unit");

        let official = set.unit(&unit);
        assert_eq!(official.len(), 6, "{unit}: fixture has six tensors");

        for (tname, bytes) in &produced {
            let f = official
                .iter()
                .find(|t| &t.name == tname)
                .unwrap_or_else(|| panic!("{unit}: official has no {tname}"));
            assert_eq!(
                bytes.len() as u64,
                f.bytes,
                "{tname}: produced {} bytes, official has {}",
                bytes.len(),
                f.bytes
            );
            assert_matches(f, bytes);
            tensors_checked += 1;
        }
        units_checked += 1;
    }
    assert_eq!(units_checked, 4, "tier0 has 4 units");
    assert_eq!(tensors_checked, 24, "4 units x 6 tensors");
}

/// A single-tensor unit -- the mainstream case for lm_head and embed_tokens --
/// must produce an empty split_positions rather than a one-element one.
#[test]
fn a_single_tensor_unit_has_empty_split_positions() {
    let Some(fx) = skip_if_missing("a_single_tensor_unit_has_empty_split_positions") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let one = src
        .tensor("model.layers.0.mlp.down_proj.weight")
        .expect("source tensor");
    let out = encode_unit("solo", &[&one], 512, 8).expect("encodes");
    assert!(
        out.split_positions.is_empty(),
        "one tensor means no internal boundaries"
    );
    assert_eq!(out.weights(), (one.len() / 2) as u64);
    let sp = out
        .tensors()
        .into_iter()
        .find(|(n, _)| n.ends_with("split_positions"))
        .expect("still emitted");
    assert!(sp.1.is_empty(), "emitted as a zero-length tensor");
}

#[test]
fn a_reserved_exponent_aborts_before_anything_is_produced() {
    // 0xFF80 is -infinity in BF16: exponent 0xFF, which the LUT jump convention
    // reserves. Two weights so the buffer is well-formed.
    let bad = [0x80u8, 0xFF, 0x00, 0x3F];
    let err = encode_unit("u", &[&bad], 512, 8).expect_err("must abort");
    assert!(
        matches!(
            err,
            df11_codec::EncodeError::ReservedExponent { value: 255, .. }
        ),
        "got {err:?}"
    );
}
