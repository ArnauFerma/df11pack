//! Phase 2 -- writing a DF11 directory, graded against the official output.

use df11_codec::arch::ArchDef;
use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::write::{remainder_name, shard_name, write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing, SourceModel};

fn outdir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("df11pack_w_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn shard_names_follow_the_official_convention() {
    assert_eq!(shard_name("model.layers.0"), "model_layers_0.safetensors");
    assert_eq!(
        shard_name("distilled_guidance_layer"),
        "distilled_guidance_layer.safetensors"
    );
    assert_eq!(shard_name("lm_head"), "lm_head.safetensors");
}

#[test]
fn the_remainder_file_is_named_per_layout() {
    use df11_codec::arch::Layout;
    assert_eq!(remainder_name(Layout::Transformers), "model.safetensors");
    assert_eq!(
        remainder_name(Layout::Diffusers),
        "diffusion_pytorch_model.safetensors"
    );
    assert_eq!(remainder_name(Layout::ComfyuiNative), "model.safetensors");
}

/// The Phase 2 gate for the writer: our whole output directory must match the
/// official one, file by file and tensor by tensor.
#[test]
fn the_written_directory_matches_the_official_one() {
    let Some(fx) = skip_if_missing("the_written_directory_matches_the_official_one") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(_s) = SourceModel::open(set) else {
        return;
    };
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");

    let src = SafeTensorsFile::open(set.source_dir.join("model.safetensors")).expect("source");
    let out = outdir("dir");
    let report = write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");

    assert_eq!(report.units, 4);
    assert_eq!(report.shards.len(), 4);
    assert_eq!(report.remainder, "model.safetensors");

    // Every DF11 tensor must be byte-identical, and must live in the same file
    // the official compressor put it in.
    let mut checked = 0;
    for t in set.tensors() {
        let official_file = t.file.file_name().unwrap().to_string_lossy().to_string();
        let ours = out.join(&official_file);
        assert!(
            ours.exists(),
            "we did not produce {official_file}, which the official output has"
        );
        let f = SafeTensorsFile::open(&ours).expect("open ours");
        let got = match f.read(&t.name) {
            Ok(b) => b,
            Err(e) => panic!("{}: not in our {official_file}: {e}", t.name),
        };

        // The header is part of the file. Matching bytes with a wrong dtype or
        // shape is not a match -- the loader reshapes `luts` to
        // (n_prefixes + 1, 256) and reads `split_positions` as I64, so either
        // being wrong breaks decoding while leaving the data identical.
        // Verified by mutation: without this, flattening luts' shape and typing
        // split_positions as U8 both passed.
        let info = f.info(&t.name).expect("present");
        assert_eq!(
            info.dtype.0, t.dtype,
            "{}: dtype {} vs official {}",
            t.name, info.dtype.0, t.dtype
        );
        assert_eq!(
            info.shape, t.shape,
            "{}: shape {:?} vs official {:?}",
            t.name, info.shape, t.shape
        );
        assert!(
            info.is_consistent(),
            "{}: shape must explain the bytes",
            t.name
        );

        let want = t.read().expect("fixture");
        assert_eq!(
            got.len(),
            want.len(),
            "{}: {} bytes vs official {}",
            t.name,
            got.len(),
            want.len()
        );
        assert!(
            got == want,
            "{}: bytes differ from the official output",
            t.name
        );
        checked += 1;
    }
    assert_eq!(checked, 42, "tier0 official output has 42 tensors in total");

    // And we must not have invented files or tensors.
    let ours: std::collections::BTreeSet<String> = std::fs::read_dir(&out)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    let official: std::collections::BTreeSet<String> = set
        .tensors()
        .iter()
        .map(|t| t.file.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    assert_eq!(ours, official, "the set of safetensors files must match");

    let _ = std::fs::remove_dir_all(&out);
}

/// `tie_word_embeddings` makes the official output drop `lm_head.weight`
/// entirely (FINDINGS 0.2). We must do the same, and say we did.
#[test]
fn a_tied_tensor_is_dropped_and_reported() {
    let Some(fx) = skip_if_missing("a_tied_tensor_is_dropped_and_reported") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = SafeTensorsFile::open(set.source_dir.join("model.safetensors")).expect("source");

    let out = outdir("tied");
    let report = write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    assert_eq!(
        report.tied_dropped,
        vec!["lm_head.weight".to_string()],
        "lm_head is byte-identical to embed_tokens and must be dropped, as upstream does"
    );

    let rem = SafeTensorsFile::open(out.join("model.safetensors")).expect("remainder");
    let names: Vec<&str> = rem.names().collect();
    assert_eq!(
        names,
        vec!["model.embed_tokens.weight", "model.norm.weight"],
        "the remainder holds exactly what the official one holds"
    );
    let _ = std::fs::remove_dir_all(&out);
}
