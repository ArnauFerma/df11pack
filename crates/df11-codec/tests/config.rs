//! Phase 2 -- config.json generation.

use df11_codec::arch::ArchDef;
use df11_codec::config::{build_config, dfloat11_config, ConfigMode};
use df11_fixtures::{architecture_defs, official_pattern_dicts};
use serde_json::json;

fn def(name: &str) -> ArchDef {
    let defs = architecture_defs().expect("definitions present");
    let (_, toml) = defs.iter().find(|(n, _)| n == name).expect(name);
    ArchDef::from_toml(toml).expect("parses")
}

#[test]
fn dfloat11_config_has_the_four_keys_the_format_defines() {
    let d = def("qwen3-4b");
    let c = dfloat11_config(&d);
    let o = c.as_object().expect("an object");
    let mut keys: Vec<&str> = o.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "bytes_per_thread",
            "pattern_dict",
            "threads_per_block",
            "version"
        ]
    );
    assert_eq!(o["version"], json!("0.2.0"));
    assert_eq!(o["bytes_per_thread"], json!(8));
    assert_eq!(o["threads_per_block"], json!([512]));
}

/// The strong check: our generated block must equal a published official
/// release's, key for key and **in the same order**, since the order is the
/// concatenation order.
#[test]
fn dfloat11_config_matches_published_releases_exactly() {
    let official = official_pattern_dicts().expect("fixture");
    let mut checked = 0;
    for (name, o) in &official {
        let d = def(name);
        let ours = dfloat11_config(&d);
        let pd = ours
            .get("pattern_dict")
            .and_then(|v| v.as_object())
            .unwrap_or_else(|| panic!("{name}: no pattern_dict object"));

        assert_eq!(pd.len(), o.pattern_dict.len(), "{name}: unit count");
        for ((ok, ov), (mk, mv)) in o.pattern_dict.iter().zip(pd.iter()) {
            assert_eq!(mk, ok, "{name}: pattern order differs");
            assert_eq!(mv, ov, "{name}: attribute list differs for {ok}");
        }
        assert_eq!(
            ours["bytes_per_thread"],
            json!(o.bytes_per_thread),
            "{name}"
        );
        assert_eq!(
            ours["threads_per_block"],
            json!(o.threads_per_block),
            "{name}"
        );
        if let Some(v) = &o.version {
            assert_eq!(ours["version"], json!(v), "{name}: format version");
        }
        checked += 1;
    }
    assert!(
        checked >= 8,
        "expected all shipped definitions, got {checked}"
    );
}

#[test]
fn minimal_mode_matches_what_published_diffusers_releases_ship() {
    let d = def("chroma-diffusers");
    let src =
        json!({"_class_name": "ChromaTransformer2DModel", "model_type": "llama", "num_layers": 19});
    let c = build_config(Some(&src), &d, ConfigMode::Minimal);
    let o = c.as_object().expect("object");
    let mut keys: Vec<&str> = o.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec!["dfloat11_config", "model_type"],
        "published releases carry exactly these two"
    );
    assert_eq!(o["model_type"], json!("llama"));
}

#[test]
fn minimal_mode_omits_model_type_when_the_source_has_none() {
    let d = def("chroma-diffusers");
    let c = build_config(Some(&json!({"num_layers": 19})), &d, ConfigMode::Minimal);
    let keys: Vec<&str> = c.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["dfloat11_config"]);
}

#[test]
fn preserve_mode_keeps_every_source_key_and_adds_ours() {
    let d = def("qwen3-4b");
    let src = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "tie_word_embeddings": true
    });
    let c = build_config(Some(&src), &d, ConfigMode::PreserveSource);
    let o = c.as_object().expect("object");
    assert_eq!(o["architectures"], json!(["Qwen3ForCausalLM"]));
    assert_eq!(o["hidden_size"], json!(1024));
    assert_eq!(o["tie_word_embeddings"], json!(true));
    assert!(o.contains_key("dfloat11_config"));
    assert_eq!(o.len(), 4, "source keys plus exactly one added");
}

#[test]
fn a_source_config_is_never_required() {
    let d = def("qwen3-4b");
    for mode in [ConfigMode::Minimal, ConfigMode::PreserveSource] {
        let c = build_config(None, &d, mode);
        assert!(
            c.get("dfloat11_config").is_some(),
            "{mode:?}: must still emit the block the loader requires"
        );
    }
}

#[test]
fn an_existing_dfloat11_config_in_the_source_is_replaced_not_merged() {
    let d = def("qwen3-4b");
    let src = json!({"dfloat11_config": {"version": "0.0.1", "stale": true}, "x": 1});
    let c = build_config(Some(&src), &d, ConfigMode::PreserveSource);
    let block = &c["dfloat11_config"];
    assert_eq!(block["version"], json!("0.2.0"), "ours must win");
    assert!(
        block.get("stale").is_none(),
        "a stale key from the source config must not survive"
    );
}
