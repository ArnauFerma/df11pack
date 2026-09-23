//! Whole-file byte identity, header included.
//!
//! Every other byte-identity test compares tensors. That let the writer add a
//! `{"format": "pt"}` metadata block the official tool never writes to unit shards
//! or to the single file -- tensors identical, files not. These compare the files.

use df11_codec::arch::ArchDef;
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing};
use std::path::{Path, PathBuf};

fn run(tag: &str, set: &str, arch: &str) -> Option<(PathBuf, PathBuf)> {
    let fx = skip_if_missing(tag)?;
    let set = fx.set(set)?;
    let official = set.tensors()[0].file.parent()?.to_path_buf();
    let out = df11_fixtures::scratch(&format!("wf_{tag}"));
    let defs = architecture_defs()?;
    let (_, toml) = defs.iter().find(|(n, _)| n == arch)?;
    let def = ArchDef::from_toml(toml).ok()?;
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).ok()?;
    write_directory(&src, &def, &out, &WriteOptions::default()).unwrap();
    Some((out, official))
}

/// Files that must be identical: every safetensors file whose name we share,
/// except the ones `skip` names (written by a stand-in, not the official code).
fn assert_identical(out: &Path, official: &Path, skip: &[&str]) -> usize {
    let mut n = 0;
    for e in std::fs::read_dir(official).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".safetensors") || skip.contains(&name.as_str()) {
            continue;
        }
        let ours = std::fs::read(out.join(&name)).unwrap();
        let theirs = std::fs::read(&p).unwrap();
        assert!(
            ours == theirs,
            "{name}: not byte-identical as a file ({} vs {} bytes)",
            ours.len(),
            theirs.len()
        );
        n += 1;
    }
    n
}

#[test]
fn transformers_directory_is_identical_file_for_file() {
    let Some((out, official)) = run("tf", "tier0-qwen3-trunc-layers-only", "qwen3-4b") else {
        return;
    };
    assert_eq!(assert_identical(&out, &official, &[]), 5);
}

#[test]
fn comfyui_single_file_is_identical() {
    for (set, arch) in [
        ("synthetic-flux-comfyui", "flux-comfyui"),
        ("synthetic-sdxl-comfyui", "sdxl-comfyui"),
    ] {
        let Some((out, official)) = run(arch, set, arch) else {
            return;
        };
        assert_eq!(assert_identical(&out, &official, &[]), 1, "{set}");
    }
}

/// Diffusers unit shards are written by the official code itself; the remainder
/// in the synthetic set is from a stand-in `save_pretrained`, so it is skipped.
#[test]
fn diffusers_unit_shards_are_identical() {
    let Some((out, official)) = run("df", "synthetic-flux-dev-diffusers", "flux-dev-diffusers")
    else {
        return;
    };
    let n = assert_identical(&out, &official, &["diffusion_pytorch_model.safetensors"]);
    assert_eq!(n, 4);
}
