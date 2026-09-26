//! Flag combinations refused at parse time, before any file is read.

use std::process::Command;

/// Exit code and stderr. The source and arch do not exist: a refusal must come
/// from argument parsing, before they are looked at.
fn compress(extra: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_df11pack"))
        .args(["compress", "/nonexistent/model", "--arch", "qwen3-4b"])
        .args(["-o", "/nonexistent/out"])
        .args(extra)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn idx8_block_needs_index_idx8() {
    for args in [
        &["--idx8-block", "32"][..],
        &["--index", "df11", "--idx8-block", "32"][..],
    ] {
        let (code, err) = compress(args);
        assert_eq!(code, 2, "{args:?} is a usage error: {err}");
        assert!(err.contains("--index"), "{args:?}: {err}");
    }
}

#[test]
fn idx8_block_zero_is_refused_at_parse_time() {
    let (code, err) = compress(&["--index", "idx8", "--idx8-block", "0"]);
    assert_eq!(code, 2, "a usage error: {err}");
    assert!(err.contains("--idx8-block"), "{err}");
}

/// The control: the same flags with a valid block get past parsing and fail
/// only on the missing source (exit 1, not 2).
#[test]
fn a_valid_idx8_block_parses() {
    let (code, err) = compress(&["--index", "idx8", "--idx8-block", "32"]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("/nonexistent/model"), "{err}");
}
