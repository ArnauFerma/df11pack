//! Phase 5 -- the independent CPU verifier.

use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::verify::{verify_unit, UnitView, VerifyError};
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

struct Loaded {
    luts: Vec<u8>,
    enc: Vec<u8>,
    sm: Vec<u8>,
    pos: Vec<u32>,
    gaps: Vec<u8>,
}

fn load(shard: &std::path::Path, unit: &str) -> Loaded {
    let f = SafeTensorsFile::open(shard).expect("open shard");
    let g = |s: &str| f.read(&format!("{unit}.{s}")).expect(s);
    let pos_bytes = g("output_positions");
    Loaded {
        luts: g("luts"),
        enc: g("encoded_exponent"),
        sm: g("sign_mantissa"),
        pos: pos_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        gaps: g("gaps"),
    }
}

fn view(l: &Loaded) -> UnitView<'_> {
    UnitView {
        luts: &l.luts,
        encoded_exponent: &l.enc,
        sign_mantissa: &l.sm,
        output_positions: &l.pos,
        gaps: &l.gaps,
        bytes_per_thread: 8,
        threads_per_block: 512,
    }
}

/// The real check: the official compressor's own output must verify against the
/// source weights it was made from.
#[test]
fn official_output_verifies_against_the_source() {
    let Some(fx) = skip_if_missing("official_output_verifies_against_the_source") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let mut n = 0;
    for unit in set.unit_names() {
        let shard = &set.unit(&unit)[0].file;
        let l = load(shard, &unit);
        let source = src.unit_input(&unit, &QWEN3_LAYER).expect("source weights");
        verify_unit(&view(&l), &source)
            .unwrap_or_else(|e| panic!("{unit}: official output failed verification: {e}"));
        n += 1;
    }
    assert_eq!(n, 4);
}

/// A single flipped bit in the payload must be caught.
#[test]
fn a_corrupted_bitstream_is_caught() {
    let Some(fx) = skip_if_missing("a_corrupted_bitstream_is_caught") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let unit = "model.layers.0";
    let mut l = load(&set.unit(unit)[0].file, unit);
    let source = src.unit_input(unit, &QWEN3_LAYER).expect("source");

    let target = l.enc.len() / 3;
    l.enc[target] ^= 0b0000_1000;
    let e = verify_unit(&view(&l), &source).expect_err("a flipped bit must be caught");
    assert!(
        matches!(
            e,
            VerifyError::WeightMismatch { .. } | VerifyError::Truncated { .. }
        ),
        "expected a decode failure, got {e}"
    );
}

/// The index tensors are exactly what a sequential decoder would not check, and
/// exactly what a wrong encoder would get wrong. Each must be caught.
#[test]
fn a_corrupted_gap_is_caught() {
    let Some(fx) = skip_if_missing("a_corrupted_gap_is_caught") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let unit = "model.layers.0";
    let mut l = load(&set.unit(unit)[0].file, unit);
    let source = src.unit_input(unit, &QWEN3_LAYER).expect("source");

    // Change one 5-bit gap entry without touching the bitstream. A decoder that
    // only walks the stream start to finish would see nothing wrong.
    l.gaps[100] ^= 0b0000_0111;
    let e = verify_unit(&view(&l), &source).expect_err("a bad gap must be caught");
    assert!(
        matches!(e, VerifyError::GapNotACodeBoundary { .. }),
        "expected GapNotACodeBoundary, got {e}"
    );
}

#[test]
fn a_corrupted_output_position_is_caught() {
    let Some(fx) = skip_if_missing("a_corrupted_output_position_is_caught") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let unit = "model.layers.0";
    let mut l = load(&set.unit(unit)[0].file, unit);
    let source = src.unit_input(unit, &QWEN3_LAYER).expect("source");

    // Exactly the corruption that defeated the structural checker in Phase 0:
    // swap two adjacent, individually plausible deltas.
    let (a, b) = (l.pos[500], l.pos[501]);
    l.pos[501] = a + (l.pos[502] - b);
    let e = verify_unit(&view(&l), &source).expect_err("a bad output_position must be caught");
    assert!(
        matches!(e, VerifyError::OutputPositionMismatch { .. }),
        "expected OutputPositionMismatch, got {e}"
    );
}

#[test]
fn a_corrupted_sign_mantissa_is_caught() {
    let Some(fx) = skip_if_missing("a_corrupted_sign_mantissa_is_caught") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };
    let unit = "model.layers.0";
    let mut l = load(&set.unit(unit)[0].file, unit);
    let source = src.unit_input(unit, &QWEN3_LAYER).expect("source");
    l.sm[12345] ^= 0x01;
    let e = verify_unit(&view(&l), &source).expect_err("a bad sign_mantissa must be caught");
    assert!(
        matches!(e, VerifyError::WeightMismatch { index: 12345, .. }),
        "expected the mismatch at the corrupted weight, got {e}"
    );
}

/// The LUT jump boundary, pinned directly.
///
/// A value of exactly 240 means "jump to table 16", which only a unit using the
/// maximum sixteen prefix tables produces. No real fixture here has more than
/// four, so the boundary is unreachable from end-to-end tests and has to be
/// tested on its own — verified by mutation: changing `>= 240` to `>= 241` passed
/// every other test in this file.
#[test]
fn a_lut_value_of_exactly_240_is_a_jump_not_a_symbol() {
    use df11_codec::verify::decode_at;

    // 17 tables plus the lengths row: a jump value of 240 targets table 16, so
    // table 16 must exist for that value to be legal at all.
    let tables = 17usize;
    let mut luts = vec![0u8; (tables + 1) * 256];
    // Table 0, byte 0x00 -> jump to table 256 - 240 = 16.
    luts[0] = 240;
    // Table 16, byte 0x00 -> symbol 5.
    luts[16 * 256] = 5;
    // Lengths row: symbol 5 is 16 bits long.
    luts[tables * 256 + 5] = 16;
    let lens_start = tables * 256;
    let lens = luts[lens_start..].to_vec();

    let stream = [0u8, 0, 0, 0];
    let (sym, len) = decode_at(&luts, tables, &lens, &stream, 0).expect("must follow the jump");
    assert_eq!(sym, 5, "240 must be followed as a jump to table 16");
    assert_eq!(len, 16);
}

/// And the value just below the threshold is a symbol, not a jump.
#[test]
fn a_lut_value_of_239_is_a_symbol() {
    use df11_codec::verify::decode_at;
    let tables = 2usize;
    let mut luts = vec![0u8; (tables + 1) * 256];
    luts[0] = 239;
    luts[tables * 256 + 239] = 3;
    let lens = luts[tables * 256..].to_vec();
    let (sym, len) = decode_at(&luts, tables, &lens, &[0u8, 0], 0).expect("239 is a symbol");
    assert_eq!(sym, 239);
    assert_eq!(len, 3);
}

/// The EOF-window entry. When a unit's last code straddles into a new window, no
/// code starts there, yet the format records a gap for it. The chunked encoder
/// once dropped that entry, and this verifier did not notice, because it only
/// checked windows where a code starts. The synthetic FLUX unit below has that
/// shape; zeroing its EOF gap must be caught.
#[test]
fn a_wrong_eof_window_gap_is_caught() {
    let Some(fx) = skip_if_missing("a_wrong_eof_window_gap_is_caught") else {
        return;
    };
    let Some(set) = fx.set("synthetic-flux-comfyui") else {
        eprintln!("SKIP: run phase0/make_synthetic.py");
        return;
    };
    let unit = "double_blocks.0";
    let shard = &set.unit(unit)[0].file;
    let mut l = load(shard, unit);

    // The source, concatenated in the definition's order, from the synthetic model.
    let src = SafeTensorsFile::open(set.source_dir.join("model.safetensors")).unwrap();
    let attrs = [
        "img_mod.lin",
        "img_attn.qkv",
        "img_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mod.lin",
        "txt_attn.qkv",
        "txt_attn.proj",
        "txt_mlp.0",
        "txt_mlp.2",
    ];
    let mut source = Vec::new();
    for a in attrs {
        source.extend(src.read(&format!("{unit}.{a}.weight")).unwrap());
    }
    verify_unit(&view(&l), &source).expect("the official unit must verify");

    // Window 95531 is the one the synthetic run exposed: the last code ends 3
    // bits into it. Zero that 5-bit entry.
    let w = 95531usize;
    for k in 0..5 {
        let bit = w * 5 + k;
        l.gaps[bit / 8] &= !(1 << (7 - bit % 8));
    }
    let e = verify_unit(&view(&l), &source).expect_err("a wrong EOF-window gap must be caught");
    assert!(
        matches!(e, VerifyError::GapNotACodeBoundary { window: 95531, .. }),
        "expected the EOF window to be named, got {e}"
    );
}
