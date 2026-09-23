//! `df11pack verify` end to end: exit codes a script can rely on.

use df11_codec::source::ModelSource;
use df11_fixtures::skip_if_missing;
use std::path::PathBuf;
use std::process::Command;

fn run(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_df11pack"))
        .args(args)
        .env(
            "DF11PACK_ARCH_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/architectures"),
        )
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
    )
}

#[test]
fn verify_exits_0_on_a_good_output_and_1_on_a_corrupted_one() {
    let Some(fx) = skip_if_missing("verify_cli") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").unwrap();
    let official = set.tensors()[0].file.parent().unwrap().to_path_buf();
    let out: PathBuf = std::env::temp_dir().join(format!("df11pack_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    for e in std::fs::read_dir(&official).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, out.join(p.file_name().unwrap())).unwrap();
    }
    let source = set.source_dir.join("model.safetensors");
    let (o, s) = (out.to_str().unwrap(), source.to_str().unwrap());
    let args = [
        "verify",
        o,
        "--source",
        s,
        "--arch",
        "qwen3-4b",
        "--samples",
        "50",
        "--seed",
        "7",
    ];

    let (code, stdout) = run(&args);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("seed 7"), "the seed is printed: {stdout}");

    // Flip one weight's sign bit in the first chunk of a unit, on disk.
    let (path, offset, _) = ModelSource::open(&out)
        .unwrap()
        .locate("model.layers.0.sign_mantissa")
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[offset as usize + 3] ^= 0x80;
    std::fs::write(&path, bytes).unwrap();

    let (code, stdout) = run(&args);
    assert_eq!(code, 1, "a failed check is exit 1: {stdout}");
    assert!(stdout.contains("FAIL model.layers.0 chunk 0"), "{stdout}");

    // Without the source, that corruption is invisible -- and the output says so.
    let (code, stdout) = run(&["verify", o]);
    assert_eq!(code, 0);
    assert!(stdout.contains("not detectable"), "{stdout}");

    // A level that needs the source, without it, is 2: could not check.
    assert_eq!(run(&["verify", o, "--level", "full"]).0, 2);
    let _ = std::fs::remove_dir_all(&out);
}

/// With --hashes, a flipped value is caught with no source at all.
#[test]
fn a_hashed_output_catches_a_flipped_value_without_the_source() {
    let Some(fx) = skip_if_missing("verify_cli_hashes") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").unwrap();
    let source = set.source_dir.join("model.safetensors");
    let out: PathBuf = std::env::temp_dir().join(format!("df11pack_cli_h_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let (o, s) = (out.to_str().unwrap(), source.to_str().unwrap());
    let (code, stdout) = run(&["compress", s, "--arch", "qwen3-4b", "-o", o, "--hashes"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("differ from the official"), "{stdout}");

    let (code, stdout) = run(&["verify", o]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("24 stored SHA-256"), "{stdout}");
    assert!(!stdout.contains("not detectable"), "{stdout}");

    let (path, offset, _) = ModelSource::open(&out)
        .unwrap()
        .locate("model.layers.3.encoded_exponent")
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[offset as usize + 777] ^= 0x04;
    std::fs::write(&path, bytes).unwrap();

    let (code, stdout) = run(&["verify", o]);
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("FAIL model.layers.3"), "{stdout}");
    let _ = std::fs::remove_dir_all(&out);
}
