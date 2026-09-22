//! Tests for the fixture harness itself.
//!
//! The harness is what every later byte-identity claim rests on, so it is
//! tested against the real frozen output, not against synthetic buffers alone.

use df11_fixtures::{assert_matches, diff_bytes, skip_if_missing};

#[test]
fn diff_is_none_for_identical() {
    let a = vec![1u8, 2, 3, 4];
    assert!(diff_bytes(&a, &a).is_none());
}

#[test]
fn diff_reports_exact_position_and_bit() {
    let a = vec![0u8; 100];
    let mut b = a.clone();
    b[37] = 0x08; // one bit set
    let d = diff_bytes(&a, &b).expect("a single changed byte must be reported");
    assert_eq!(d.byte_offset, 37);
    assert_eq!(d.bit_offset, 37 * 8);
    assert_eq!(d.expected, 0x00);
    assert_eq!(d.actual, 0x08);
}

#[test]
fn diff_reports_length_mismatch_even_when_prefix_matches() {
    let a = vec![7u8; 64];
    let b = vec![7u8; 63];
    let d = diff_bytes(&a, &b).expect("a truncation must be reported");
    assert_eq!(d.expected_len, 64);
    assert_eq!(d.actual_len, 63);
    assert_eq!(d.byte_offset, 63);
}

/// The one that matters: a single flipped bit in a multi-megabyte real tensor.
#[test]
fn finds_one_flipped_bit_in_a_real_multi_mb_tensor() {
    let Some(fx) = skip_if_missing("finds_one_flipped_bit_in_a_real_multi_mb_tensor") else {
        return;
    };
    let set = fx
        .set("tier0-qwen3-trunc-layers-only")
        .expect("tier0 set present");

    let t = set
        .unit_tensors()
        .find(|t| t.name.ends_with("sign_mantissa") && t.bytes > 4 * 1024 * 1024)
        .expect("a multi-MB sign_mantissa fixture");

    let good = t.read().expect("read fixture");
    assert!(good.len() > 4 * 1024 * 1024, "expected a multi-MB tensor");

    // Unmodified bytes must match.
    assert!(diff_bytes(&good, &good).is_none());

    // Flip exactly one bit, deep inside, and require the harness to pin it.
    let target = good.len() / 3;
    let mut bad = good.clone();
    bad[target] ^= 0b0001_0000;

    let d = diff_bytes(&good, &bad).expect("a single flipped bit must be caught");
    assert_eq!(d.byte_offset, target, "must report the exact byte");
    assert_eq!(d.expected ^ d.actual, 0b0001_0000, "must report which bit");
}

/// `assert_matches` must actually panic on a mismatch. A harness whose assertion
/// silently passes would make every later byte-identity claim worthless.
#[test]
fn assert_matches_panics_on_mismatch_and_names_the_tensor() {
    let Some(fx) = skip_if_missing("assert_matches_panics_on_mismatch_and_names_the_tensor") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").unwrap();
    let t = set.unit_tensors().next().expect("at least one unit tensor");

    let mut bad = t.read().expect("read fixture");
    bad[0] ^= 0xff;

    let msg = std::panic::catch_unwind(|| assert_matches(t, &bad))
        .expect_err("assert_matches must panic when the bytes differ");
    let msg = msg
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or("<non-string panic>");
    assert!(msg.contains(&t.name), "panic must name the tensor: {msg}");
    assert!(
        msg.contains("first difference at byte 0"),
        "panic must locate the difference: {msg}"
    );
}

#[test]
fn assert_matches_accepts_the_real_bytes() {
    let Some(fx) = skip_if_missing("assert_matches_accepts_the_real_bytes") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").unwrap();
    for t in set.unit_tensors().take(6) {
        let good = t.read().expect("read fixture");
        assert_matches(t, &good);
    }
}

/// The manifest must describe the files on disk. If this fails, the fixtures
/// drifted and every byte-identity result is against the wrong reference.
#[test]
fn manifest_hashes_match_the_files_on_disk() {
    let Some(fx) = skip_if_missing("manifest_hashes_match_the_files_on_disk") else {
        return;
    };
    let mut checked = 0;
    for set in &fx.sets {
        // Smallest first: this is a correctness check, not a throughput test,
        // and hashing the 297 MiB embedding buys nothing here.
        let mut ts: Vec<_> = set.tensors().iter().collect();
        ts.sort_by_key(|t| t.bytes);
        for t in ts.iter().take(12) {
            assert!(
                t.verify_on_disk().expect("hashing fixture"),
                "manifest sha256 does not match disk for {} in {}",
                t.name,
                set.name
            );
            checked += 1;
        }

        // The negative case. Without it this test cannot tell a real hash check
        // from a function that always returns true -- verified: mutating
        // verify_on_disk to `|| true` left the positive-only version green.
        let mut wrong = (*ts[0]).clone();
        wrong.sha256 = "0".repeat(64);
        assert!(
            !wrong
                .verify_on_disk()
                .expect("hashing fixture with a deliberately wrong expected hash"),
            "verify_on_disk must reject a tensor whose expected hash is wrong \
             (tensor {} in {})",
            wrong.name,
            set.name
        );
    }
    assert!(
        checked >= 12,
        "expected to check some tensors, got {checked}"
    );
}

/// Structural facts Phase 0 established, asserted so a regenerated manifest
/// that lost them fails loudly.
#[test]
fn fixture_sets_have_the_expected_shape() {
    let Some(fx) = skip_if_missing("fixture_sets_have_the_expected_shape") else {
        return;
    };
    let t1 = fx
        .set("tier1-qwen3-0.6b-layers-only")
        .expect("tier1 present");
    assert_eq!(t1.unit_names().len(), 28, "Qwen3-0.6B has 28 layer units");

    for unit in t1.unit_names() {
        let ts = t1.unit(&unit);
        assert_eq!(ts.len(), 6, "unit {unit} must have all six DF11 tensors");
    }

    // split_positions holds n-1 internal boundaries for a 7-tensor unit (FINDINGS 0.2).
    let sp = t1
        .unit("model.layers.0")
        .into_iter()
        .find(|t| t.name.ends_with("split_positions"))
        .expect("split_positions present");
    assert_eq!(sp.shape, vec![6], "7 concatenated tensors => 6 boundaries");
}
