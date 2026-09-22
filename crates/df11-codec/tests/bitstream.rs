//! Step 1.7 -- the bitstream and the two index tensors, against real output.
//!
//! This is the first byte-identity check on `encoded_exponent`. Until now the
//! exponent stream had only synthetic coverage, because DF11 never stores it raw.

use df11_codec::bitstream::encode;
use df11_codec::huffman::Codebook;
use df11_codec::{split_fields, Histogram};
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
const BYTES_PER_THREAD: usize = 8;
const THREADS_PER_BLOCK: usize = 512;

/// All three tensors, byte for byte, for every unit.
#[test]
fn encoded_gaps_and_positions_match_official_output() {
    let Some(fx) = skip_if_missing("encoded_gaps_and_positions_match_official_output") else {
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
        let enc = encode(&exp, &cb, BYTES_PER_THREAD, THREADS_PER_BLOCK);

        let get = |suffix: &str| {
            set.unit(&unit)
                .into_iter()
                .find(|t| t.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("{unit}: no {suffix} fixture"))
        };

        let f_enc = get("encoded_exponent");
        assert_eq!(
            enc.bytes.len() as u64,
            f_enc.bytes,
            "{unit}: encoded length {} vs official {}",
            enc.bytes.len(),
            f_enc.bytes
        );
        assert_matches(f_enc, &enc.bytes);

        let f_gaps = get("gaps");
        assert_eq!(
            enc.gaps.len() as u64,
            f_gaps.bytes,
            "{unit}: gaps length {} vs official {}",
            enc.gaps.len(),
            f_gaps.bytes
        );
        assert_matches(f_gaps, &enc.gaps);

        let f_pos = get("output_positions");
        let pos = enc.output_positions_bytes();
        assert_eq!(
            pos.len() as u64,
            f_pos.bytes,
            "{unit}: output_positions length {} vs official {}",
            pos.len(),
            f_pos.bytes
        );
        assert_matches(f_pos, &pos);

        checked += 1;
    }
    assert_eq!(checked, 4);
}

/// Structural relations DESIGN 1.3 states, checked on our own output so a change
/// that breaks them fails here rather than at the invariant checker later.
#[test]
fn index_tensors_have_the_shapes_the_format_requires() {
    let Some(fx) = skip_if_missing("index_tensors_have_the_shapes_the_format_requires") else {
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
    let enc = encode(&exp, &Codebook::build(&h.frequencies()), 8, 512);

    let chunk = BYTES_PER_THREAD * THREADS_PER_BLOCK; // 4096
    let blocks = enc.bytes.len().div_ceil(chunk);
    assert_eq!(
        enc.output_positions.len(),
        blocks + 1,
        "one entry per chunk plus the trailing total"
    );
    assert_eq!(
        *enc.output_positions.last().unwrap() as usize,
        exp.len(),
        "the trailing value is the element count, not a byte length"
    );
    assert!(
        enc.output_positions.windows(2).all(|w| w[0] <= w[1]),
        "output_positions must be non-decreasing"
    );
    // 512 gaps per block, 5 bits each.
    assert_eq!(enc.gaps.len(), 320 * blocks, "gaps are padded per block");
}
