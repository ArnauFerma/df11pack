//! Phase 6 -- decoding a single kernel chunk, as the GPU does.

use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::verify::{chunk_count, chunk_range, verify_chunk, UnitView, VerifyError};
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
    let f = SafeTensorsFile::open(shard).unwrap();
    let g = |s: &str| f.read(&format!("{unit}.{s}")).unwrap();
    Loaded {
        luts: g("luts"),
        enc: g("encoded_exponent"),
        sm: g("sign_mantissa"),
        pos: g("output_positions")
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

fn setup() -> Option<(Loaded, Vec<u8>)> {
    let fx = skip_if_missing("chunk_verify")?;
    let set = fx.set("tier0-qwen3-trunc-layers-only")?;
    let src = SourceModel::open(set)?;
    let unit = "model.layers.0";
    let l = load(&set.unit(unit)[0].file, unit);
    let source = src.unit_input(unit, &QWEN3_LAYER).ok()?;
    Some((l, source))
}

#[test]
fn every_chunk_of_a_real_unit_verifies_on_its_own() {
    let Some((l, source)) = setup() else { return };
    let v = view(&l);
    let n = chunk_count(&v);
    assert_eq!(n, 1278, "tier-0 layer 0 has 1278 kernel chunks");
    let mut covered = 0usize;
    for c in 0..n {
        let (a, b) = chunk_range(&v, c).expect("every chunk has a range");
        assert!(a <= b);
        covered += b - a;
        verify_chunk(&v, c, &source[a * 2..b * 2]).unwrap_or_else(|e| panic!("chunk {c}: {e}"));
    }
    assert_eq!(covered, l.sm.len(), "the chunks partition the unit exactly");
}

/// A fault is attributed to the chunk that contains it, and only that chunk.
#[test]
fn a_corrupted_chunk_fails_alone() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    // A byte in the middle of chunk 600's bitstream region.
    let target = 600 * 4096 + 2000;
    l.enc[target] ^= 0x10;
    let v = view(&l);
    let mut failed = Vec::new();
    for c in 595..606 {
        let (a, b) = chunk_range(&v, c).unwrap();
        if verify_chunk(&v, c, &source[a * 2..b * 2]).is_err() {
            failed.push(c);
        }
    }
    assert_eq!(
        failed,
        vec![600],
        "only the chunk holding the corrupted byte may fail"
    );
}

/// The first gap of a chunk is what lets it start mid-stream; a wrong one must
/// fail that chunk even though sequential decoding would never read it.
#[test]
fn a_wrong_first_gap_fails_its_chunk() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    let c = 300usize;
    let w = c * 512; // first window of chunk c
                     // Flip the lowest bit of that window's 5-bit gap.
    let bit = w * 5 + 4;
    l.gaps[bit / 8] ^= 1 << (7 - bit % 8);
    let v = view(&l);
    let (a, b) = chunk_range(&v, c).unwrap();
    let e =
        verify_chunk(&v, c, &source[a * 2..b * 2]).expect_err("a wrong gap must fail the chunk");
    let _: VerifyError = e;
    // Its neighbours are unaffected.
    for d in [c - 1, c + 1] {
        let (a, b) = chunk_range(&v, d).unwrap();
        verify_chunk(&v, d, &source[a * 2..b * 2]).expect("neighbours still verify");
    }
}

#[test]
fn a_wrong_output_position_fails_the_chunks_it_bounds() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    let c = 700usize;
    l.pos[c] += 1; // chunk c now claims to start one element late
    let v = view(&l);
    let (a, b) = chunk_range(&v, c).unwrap();
    assert!(
        verify_chunk(&v, c, &source[a * 2..b * 2]).is_err(),
        "a shifted start must fail the chunk"
    );
}

/// The two cases the weight comparison cannot see. Moving the END of a chunk by
/// one element leaves every decoded weight correct -- the chunk simply decodes one
/// too few or one too many -- so only the boundary checks can catch it.
#[test]
fn a_chunk_that_ends_one_element_early_is_caught() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    let c = 400usize;
    l.pos[c + 1] -= 1;
    let v = view(&l);
    let (a, b) = chunk_range(&v, c).unwrap();
    assert!(
        verify_chunk(&v, c, &source[a * 2..b * 2]).is_err(),
        "a chunk whose next element also starts inside it must fail"
    );
}

#[test]
fn a_chunk_that_ends_one_element_late_is_caught() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    let c = 400usize;
    l.pos[c + 1] += 1;
    let v = view(&l);
    let (a, b) = chunk_range(&v, c).unwrap();
    assert!(
        verify_chunk(&v, c, &source[a * 2..b * 2]).is_err(),
        "a chunk claiming an element that starts in the next chunk must fail"
    );
}

/// A pure value error: sign_mantissa is stored beside the bitstream, so
/// corrupting it leaves every code boundary intact. Only the weight comparison
/// can see it -- verified: with that comparison disabled, every other test in
/// this file still passed, because each of their corruptions also broke a
/// boundary and was caught by a boundary check first.
#[test]
fn a_corrupted_sign_mantissa_fails_its_chunk_by_value() {
    let Some((mut l, source)) = setup() else {
        return;
    };
    let c = 900usize;
    let (a, _) = chunk_range(&view(&l), c).unwrap();
    l.sm[a + 17] ^= 0x01;
    let v = view(&l);
    let (a, b) = chunk_range(&v, c).unwrap();
    match verify_chunk(&v, c, &source[a * 2..b * 2]) {
        Err(VerifyError::WeightMismatch { index, .. }) => assert_eq!(index, a + 17),
        other => panic!("expected a WeightMismatch at the corrupted weight, got {other:?}"),
    }
}
