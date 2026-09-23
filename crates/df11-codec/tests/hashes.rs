//! Opt-in per-tensor hashes: what lets `verify` catch a wrong value without the
//! source, at the cost of a header that differs from the official tool's.

use df11_codec::arch::ArchDef;
use df11_codec::check::{check_output, Level};
use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing};
use std::path::{Path, PathBuf};

const FIELDS: [&str; 6] = [
    "luts",
    "encoded_exponent",
    "sign_mantissa",
    "output_positions",
    "gaps",
    "split_positions",
];

fn outdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("df11pack_h_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn def(name: &str) -> ArchDef {
    let defs = architecture_defs().unwrap();
    let (_, toml) = defs.iter().find(|(n, _)| n == name).unwrap();
    ArchDef::from_toml(toml).unwrap()
}

/// Compress `set` with hashes on; returns (output dir, source file, official dir).
fn compress(tag: &str, set: &str, arch: &str, hashes: bool) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let fx = skip_if_missing(tag)?;
    let set = fx.set(set)?;
    let official = set.tensors()[0].file.parent()?.to_path_buf();
    let source = set.source_dir.join("model.safetensors");
    let out = outdir(tag);
    let src = ModelSource::open(&source).unwrap();
    let opts = WriteOptions {
        hashes,
        ..Default::default()
    };
    write_directory(&src, &def(arch), &out, &opts).unwrap();
    Some((out, source, official))
}

fn flip(file: &Path, tensor: &str, at: u64) {
    let (path, offset, _) = ModelSource::open(file).unwrap().locate(tensor).unwrap();
    let mut b = std::fs::read(&path).unwrap();
    b[(offset + at) as usize] ^= 0x01;
    std::fs::write(&path, b).unwrap();
}

#[test]
fn hashes_are_off_by_default_and_on_when_asked() {
    let Some((plain, _, _)) = compress("off", "tier0-qwen3-trunc-layers-only", "qwen3-4b", false)
    else {
        return;
    };
    let f = SafeTensorsFile::open(plain.join("model_layers_0.safetensors")).unwrap();
    assert!(
        !f.metadata().keys().any(|k| k.starts_with("df11pack")),
        "default output must carry no df11pack metadata: {:?}",
        f.metadata()
    );

    let (hashed, _, _) = compress("on", "tier0-qwen3-trunc-layers-only", "qwen3-4b", true).unwrap();
    let f = SafeTensorsFile::open(hashed.join("model_layers_0.safetensors")).unwrap();
    let m = f.metadata();
    assert_eq!(m.get("df11pack_hashes").map(String::as_str), Some("sha256"));
    for field in FIELDS {
        let k = format!("df11pack_sha256:model.layers.0.{field}");
        let v = m.get(&k).unwrap_or_else(|| panic!("{k} missing"));
        assert_eq!(v.len(), 64, "{k} is hex SHA-256");
    }
}

/// The hashes must be of the bytes actually written -- checked here against an
/// independent SHA-256, not the one the writer used.
#[test]
fn a_stored_hash_is_the_sha256_of_the_tensor_bytes() {
    let Some((out, _, _)) = compress("sha", "tier0-qwen3-trunc-layers-only", "qwen3-4b", true)
    else {
        return;
    };
    let path = out.join("model_layers_2.safetensors");
    let f = SafeTensorsFile::open(&path).unwrap();
    let bytes = f.read("model.layers.2.gaps").unwrap();
    let dump = std::env::temp_dir().join(format!("df11pack_h_{}_gaps.bin", std::process::id()));
    std::fs::write(&dump, &bytes).unwrap();
    let Ok(o) = std::process::Command::new("sha256sum").arg(&dump).output() else {
        eprintln!("SKIP: no sha256sum");
        return;
    };
    let want = String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    assert_eq!(f.metadata()["df11pack_sha256:model.layers.2.gaps"], want);
}

/// Only the header changes: every tensor is still byte-identical to the
/// official output.
#[test]
fn hashing_changes_no_tensor_bytes() {
    let Some((out, _, official)) =
        compress("same", "tier0-qwen3-trunc-layers-only", "qwen3-4b", true)
    else {
        return;
    };
    let ours = ModelSource::open(&out).unwrap();
    let theirs = ModelSource::open(&official).unwrap();
    assert_eq!(ours.names(), theirs.names());
    for n in theirs.names() {
        assert_eq!(ours.read(&n).unwrap(), theirs.read(&n).unwrap(), "{n}");
    }
}

/// The point of the feature: a value flipped on disk, invisible to the structural
/// check alone, is caught with no source -- and located.
#[test]
fn integrity_catches_a_flipped_value_when_hashes_are_present() {
    let Some((out, _, _)) = compress("flip", "tier0-qwen3-trunc-layers-only", "qwen3-4b", true)
    else {
        return;
    };
    let clean = check_output(&ModelSource::open(&out).unwrap(), None, Level::Integrity).unwrap();
    assert!(clean.ok(), "{:?}", clean.failures);
    assert_eq!(clean.hashed, 24, "4 units x 6 tensors");

    flip(&out, "model.layers.1.sign_mantissa", 12_345);
    let r = check_output(&ModelSource::open(&out).unwrap(), None, Level::Integrity).unwrap();
    assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
    assert_eq!(r.failures[0].unit, "model.layers.1");
    assert!(
        r.failures[0].error.contains("sign_mantissa"),
        "{:?}",
        r.failures
    );
}

/// And for the single-file layout, where the header is written before the
/// second encoding pass: the hashes come from pass one and must still match what
/// pass two wrote.
#[test]
fn native_single_file_hashes_verify() {
    let Some((out, _, _)) = compress("native", "synthetic-flux-comfyui", "flux-comfyui", true)
    else {
        return;
    };
    let file = out.join("model.safetensors");
    let r = check_output(&ModelSource::open(&file).unwrap(), None, Level::Integrity).unwrap();
    assert!(r.ok(), "{:?}", r.failures);
    assert_eq!(r.hashed, 4 * 6);

    flip(&file, "double_blocks.1.encoded_exponent", 999);
    let r = check_output(&ModelSource::open(&file).unwrap(), None, Level::Integrity).unwrap();
    assert!(
        r.failures.iter().any(|f| f.unit == "double_blocks.1"),
        "{:?}",
        r.failures
    );
}

/// Without hashes nothing is claimed: `hashed` is zero, so the CLI can say a
/// value error is undetectable rather than implying it was checked.
#[test]
fn an_unhashed_output_reports_zero_hashes() {
    let Some((out, _, _)) = compress("zero", "tier0-qwen3-trunc-layers-only", "qwen3-4b", false)
    else {
        return;
    };
    let r = check_output(&ModelSource::open(&out).unwrap(), None, Level::Integrity).unwrap();
    assert_eq!(r.hashed, 0);
}
