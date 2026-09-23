//! Corpus case 2: FLUX and Chroma, both layouts, against the official compressor.
//!
//! Until this file existed, every byte-identity result was Qwen3 and no Flux or
//! Chroma unit had ever been encoded. The fixtures come from
//! `phase0/make_synthetic.py`: small models with the real module names, compressed
//! by the official tool.

use df11_codec::arch::ArchDef;
use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing};
use std::collections::BTreeSet;

fn check(set_name: &str, def_name: &str) {
    let Some(fx) = skip_if_missing(set_name) else {
        return;
    };
    let Some(set) = fx.set(set_name) else {
        eprintln!("SKIP {set_name}: not in the manifest; run phase0/make_synthetic.py");
        return;
    };
    let defs = architecture_defs().expect("definitions");
    let (_, toml) = defs.iter().find(|(n, _)| n == def_name).expect(def_name);
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    let out = df11_fixtures::scratch(&format!("syn_{set_name}"));
    let _ = std::fs::remove_dir_all(&out);
    let report = write_directory(&src, &def, &out, &WriteOptions::default())
        .unwrap_or_else(|e| panic!("{set_name}: {e}"));
    assert!(report.units > 0, "{set_name}: no units found");

    // Every official tensor, in the same file, byte-identical, same header entry.
    let mut n = 0;
    for t in set.tensors() {
        let fname = t.file.file_name().unwrap().to_string_lossy().to_string();
        let ours = out.join(&fname);
        assert!(ours.exists(), "{set_name}: we did not produce {fname}");
        let f = SafeTensorsFile::open(&ours).unwrap();
        let info = f
            .info(&t.name)
            .unwrap_or_else(|| panic!("{set_name}: {} missing from our {fname}", t.name));
        assert_eq!(info.dtype.0, t.dtype, "{set_name}: {} dtype", t.name);
        assert_eq!(info.shape, t.shape, "{set_name}: {} shape", t.name);
        assert!(
            f.read(&t.name).unwrap() == t.read().unwrap(),
            "{set_name}: {} bytes differ from the official output",
            t.name
        );
        n += 1;
    }
    assert!(n > 0);

    // And nothing extra: the same safetensors files, holding the same tensors.
    let files = |dir: &std::path::Path| -> BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".safetensors"))
            .collect()
    };
    let official_dir = set.tensors()[0].file.parent().unwrap().to_path_buf();
    assert_eq!(
        files(&out),
        files(&official_dir),
        "{set_name}: file sets differ"
    );
    let ours_total: usize = files(&out)
        .iter()
        .map(|f| SafeTensorsFile::open(out.join(f)).unwrap().len())
        .sum();
    assert_eq!(
        ours_total, n,
        "{set_name}: we emitted tensors the official output does not have"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn flux_comfyui_native_matches_the_official_output() {
    check("synthetic-flux-comfyui", "flux-comfyui");
}

#[test]
fn chroma_comfyui_native_matches_the_official_output() {
    check("synthetic-chroma-comfyui", "chroma-comfyui");
}

#[test]
fn flux_diffusers_matches_the_official_output() {
    check("synthetic-flux-dev-diffusers", "flux-dev-diffusers");
}

#[test]
fn chroma_diffusers_matches_the_official_output() {
    check("synthetic-chroma-diffusers", "chroma-diffusers");
}

/// Phase 7: every other definition, each against its own synthetic model
/// compressed by the official tool.
#[test]
fn every_definition_matches_the_official_output() {
    let Some(fx) = skip_if_missing("every_definition_matches_the_official_output") else {
        return;
    };
    // Covered elsewhere, or not yet at all -- named, so a new definition without a
    // fixture fails here instead of passing unnoticed.
    const ELSEWHERE: [&str; 5] = [
        "qwen3-4b",     // tier0 / tier1, real weights
        "flux-comfyui", // the four original synthetic tests above
        "chroma-comfyui",
        "flux-dev-diffusers",
        "chroma-diffusers",
    ];
    const UNCOVERED: [&str; 0] = [];
    let mut checked = 0;
    for (name, _) in architecture_defs().expect("definitions") {
        if ELSEWHERE.contains(&name.as_str()) || UNCOVERED.contains(&name.as_str()) {
            continue;
        }
        let set = format!("synthetic-{name}");
        assert!(
            fx.set(&set).is_some(),
            "{name}: no synthetic fixture; run phase0/make_synthetic.py {name}"
        );
        check(&set, &name);
        checked += 1;
    }
    assert!(checked >= 29, "checked {checked}");
}
