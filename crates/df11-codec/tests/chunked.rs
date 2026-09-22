//! Phase 3 -- the chunked encoder must be byte-identical to the serial one.

use df11_codec::bitstream::encode;
use df11_codec::chunked::{encode_chunked, DEFAULT_CHUNK};
use df11_codec::huffman::Codebook;
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

fn check(exp: &[u8], chunk: usize) {
    let h = Histogram::build(exp).expect("no reserved exponents");
    let cb = Codebook::build(&h.frequencies());
    let want = encode(exp, &cb, 8, 512);
    let got = encode_chunked(exp, &cb, 8, 512, chunk);
    assert_eq!(
        got.bytes.len(),
        want.bytes.len(),
        "chunk={chunk}: encoded length differs"
    );
    assert!(got.bytes == want.bytes, "chunk={chunk}: bitstream differs");
    assert!(got.gaps == want.gaps, "chunk={chunk}: gaps differ");
    assert!(
        got.output_positions == want.output_positions,
        "chunk={chunk}: output_positions differ"
    );
}

#[test]
fn matches_the_serial_encoder_on_synthetic_streams() {
    // Deterministic pseudo-random exponents in a realistic range.
    let mut x: u64 = 0x2545F491_4F6CDD1D;
    let mut exp = Vec::with_capacity(300_000);
    for _ in 0..300_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        exp.push(96 + (x % 31) as u8);
    }
    for chunk in [1, 2, 3, 7, 64, 1000, 4096, 99_991, 300_000, 1 << 20] {
        check(&exp, chunk);
    }
}

#[test]
fn matches_on_degenerate_streams() {
    // One symbol: every code is the same length, so chunk boundaries land
    // everywhere.
    check(&vec![7u8; 100_000], 997);
    // Two symbols, wildly skewed: long codes that straddle windows.
    let mut v = vec![3u8; 100_000];
    v[50_000] = 200;
    check(&v, 1024);
    // Very short streams, where the EOF tail dominates.
    for n in [1usize, 2, 3, 7, 8, 9, 63, 64, 65] {
        check(&vec![11u8; n], 4);
    }
}

#[test]
fn chunk_size_never_changes_the_output() {
    let mut x: u64 = 12345;
    let mut exp = Vec::with_capacity(50_000);
    for _ in 0..50_000 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        exp.push(100 + ((x >> 33) % 17) as u8);
    }
    let h = Histogram::build(&exp).unwrap();
    let cb = Codebook::build(&h.frequencies());
    let base = encode_chunked(&exp, &cb, 8, 512, 1);
    for chunk in [2usize, 5, 13, 512, 4096, 50_000, 1 << 20] {
        let other = encode_chunked(&exp, &cb, 8, 512, chunk);
        assert!(
            other.bytes == base.bytes
                && other.gaps == base.gaps
                && other.output_positions == base.output_positions,
            "chunk={chunk} produced different output from chunk=1"
        );
    }
}

/// The real check: every tier-0 unit, against the serial encoder that is already
/// proved byte-identical to the official compressor.
#[test]
fn matches_the_serial_encoder_on_every_real_unit() {
    let Some(fx) = skip_if_missing("matches_the_serial_encoder_on_every_real_unit") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let mut n = 0;
    for unit in set.unit_names() {
        let input = src.unit_input(&unit, &QWEN3_LAYER).expect("source");
        let (exp, _) = split_fields(&input);
        check(&exp, DEFAULT_CHUNK);
        check(&exp, 65_536);
        n += 1;
    }
    assert_eq!(n, 4);
}
