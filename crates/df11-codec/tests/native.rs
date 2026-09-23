//! The ComfyUI-native layout: one file, no config, source prefix stripped.
//!
//! DESIGN 5.5. The review found the writer produced a directory of shards for
//! this layout -- the same shape as diffusers -- and the only native test checked
//! that no config.json was written, which a wrong shape also satisfies.

use df11_codec::arch::ArchDef;
use df11_codec::safetensors::{write_file, OutTensor, Payload, SafeTensorsFile};
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};
use df11_fixtures::skip_if_missing;
use std::collections::BTreeMap;

const NATIVE_QWEN: &str = r#"
name = "native-probe"
layout = "comfyui-native"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = 'model\.layers\.\d+'
attrs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
"#;

fn outdir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("df11pack_n_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn safetensors_in(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    v.sort();
    v
}

#[test]
fn native_output_is_one_file_and_no_config() {
    let Some(fx) = skip_if_missing("native_output_is_one_file_and_no_config") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let def = ArchDef::from_toml(NATIVE_QWEN).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");
    let out = outdir("one");
    write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    assert_eq!(
        safetensors_in(&out),
        vec!["model.safetensors".to_string()],
        "native output must be exactly one safetensors file"
    );
    assert!(!out.join("config.json").exists());
    let _ = std::fs::remove_dir_all(&out);
}

/// Every tensor the official compressor produced for this model -- across all
/// its shards -- must be in our single file, byte-identical, with the same dtype
/// and shape, and nothing else may be.
#[test]
fn the_single_file_holds_every_official_tensor_exactly() {
    let Some(fx) = skip_if_missing("the_single_file_holds_every_official_tensor_exactly") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let def = ArchDef::from_toml(NATIVE_QWEN).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");
    let out = outdir("exact");
    write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    let ours = SafeTensorsFile::open(out.join("model.safetensors")).expect("one file");

    let mut n = 0;
    for t in set.tensors() {
        let info = ours
            .info(&t.name)
            .unwrap_or_else(|| panic!("{} missing from the single file", t.name));
        assert_eq!(info.dtype.0, t.dtype, "{}", t.name);
        assert_eq!(info.shape, t.shape, "{}", t.name);
        assert!(
            ours.read(&t.name).unwrap() == t.read().unwrap(),
            "{}: bytes differ",
            t.name
        );
        n += 1;
    }
    assert_eq!(n, 42);
    assert_eq!(ours.len(), 42, "no tensors beyond the official set");
    let _ = std::fs::remove_dir_all(&out);
}

/// ComfyUI checkpoints usually carry `model.diffusion_model.` on every key. The
/// definitions are written without it, so it must be stripped, and the output
/// must not depend on whether the source had it.
#[test]
fn a_prefixed_source_gives_the_same_output() {
    let Some(fx) = skip_if_missing("a_prefixed_source_gives_the_same_output") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let def = ArchDef::from_toml(NATIVE_QWEN).expect("parses");
    let plain = SafeTensorsFile::open(set.source_dir.join("model.safetensors")).expect("source");

    // Build a prefixed copy of the source, config included.
    let pdir = outdir("prefixed_src");
    let tensors: Vec<OutTensor> = plain
        .names()
        .map(|n| {
            let i = plain.info(n).unwrap().clone();
            OutTensor {
                name: format!("model.diffusion_model.{n}"),
                dtype: i.dtype.clone(),
                shape: i.shape.clone(),
                data: Payload::Owned(plain.read(n).unwrap()),
            }
        })
        .collect();
    write_file(pdir.join("model.safetensors"), &tensors, &BTreeMap::new()).unwrap();
    std::fs::copy(set.source_dir.join("config.json"), pdir.join("config.json")).unwrap();

    let a = outdir("from_plain");
    let b = outdir("from_prefixed");
    write_directory(
        &ModelSource::open(set.source_dir.join("model.safetensors")).unwrap(),
        &def,
        &a,
        &WriteOptions::default(),
    )
    .expect("plain");
    write_directory(
        &ModelSource::open(pdir.join("model.safetensors")).unwrap(),
        &def,
        &b,
        &WriteOptions::default(),
    )
    .expect("a prefixed source must be recognised, not rejected as matching nothing");

    let fa = SafeTensorsFile::open(a.join("model.safetensors")).unwrap();
    let fb = SafeTensorsFile::open(b.join("model.safetensors")).unwrap();
    let na: Vec<&str> = fa.names().collect();
    let nb: Vec<&str> = fb.names().collect();
    assert_eq!(na, nb, "the prefix must be stripped from the output names");
    for n in na {
        assert!(fa.read(n).unwrap() == fb.read(n).unwrap(), "{n} differs");
    }
    for d in [pdir, a, b] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Native output must also respect the memory budget: the fix is not allowed to
/// hold every unit's output in memory at once.
#[test]
fn native_output_is_identical_across_worker_counts() {
    let Some(fx) = skip_if_missing("native_output_is_identical_across_worker_counts") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let def = ArchDef::from_toml(NATIVE_QWEN).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");
    let a = outdir("w1");
    let b = outdir("w4");
    let one = WriteOptions {
        workers: Some(1),
        ..Default::default()
    };
    let four = WriteOptions {
        workers: Some(4),
        ..Default::default()
    };
    write_directory(&src, &def, &a, &one).unwrap();
    write_directory(&src, &def, &b, &four).unwrap();
    assert_eq!(
        std::fs::read(a.join("model.safetensors")).unwrap(),
        std::fs::read(b.join("model.safetensors")).unwrap(),
        "the single file must not depend on the worker count"
    );
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}
