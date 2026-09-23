//! Phase 2 -- unit discovery.

use df11_codec::arch::ArchDef;
use df11_codec::discover::{discover, DiscoverError};

fn qwen3_def() -> ArchDef {
    ArchDef::from_toml(
        r#"
name = "t"
layout = "transformers"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = 'model\.layers\.\d+'
attrs = ["self_attn.q_proj", "mlp.down_proj"]
"#,
    )
    .expect("valid")
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn finds_units_and_keeps_attribute_order() {
    let d = qwen3_def();
    // Deliberately listed with the attrs in the wrong order.
    let n = names(&[
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.embed_tokens.weight",
    ]);
    let got = discover(&d, &n).expect("discovers");
    assert_eq!(got.units.len(), 2);
    assert_eq!(got.units[0].name, "model.layers.0");
    assert_eq!(
        got.units[0].tensors,
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight"
        ],
        "tensors must follow the definition's order, not the file's"
    );
    assert_eq!(got.units[1].name, "model.layers.1");
    assert_eq!(got.passthrough, vec!["model.embed_tokens.weight"]);
}

#[test]
fn units_are_ordered_numerically_not_lexicographically() {
    let d = qwen3_def();
    let mut n = Vec::new();
    for i in [0usize, 2, 10, 1] {
        n.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        n.push(format!("model.layers.{i}.mlp.down_proj.weight"));
    }
    let got = discover(&d, &n).expect("discovers");
    let got_names: Vec<&str> = got.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(
        got_names,
        vec![
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.10"
        ],
        "layer 10 must not sort between 1 and 2"
    );
}

/// The placement rule from FINDINGS 0.7: tensors under a unit's module that are
/// not compressed still travel with it.
#[test]
fn sibling_tensors_are_attributed_to_their_unit_not_to_passthrough() {
    let d = qwen3_def();
    let n = names(&[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
    ]);
    let got = discover(&d, &n).expect("discovers");
    assert_eq!(got.units.len(), 1);
    assert_eq!(
        got.units[0].siblings,
        vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight"
        ],
        "norms under the unit's module travel with it"
    );
    assert_eq!(
        got.passthrough,
        vec!["model.norm.weight"],
        "only tensors outside every unit's module are passthrough"
    );
}

#[test]
fn a_standalone_unit_is_its_own_tensor() {
    let d = ArchDef::from_toml(
        r#"
name = "t"
layout = "transformers"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = 'lm_head'
attrs = []
"#,
    )
    .unwrap();
    let n = names(&["lm_head.weight", "model.norm.weight"]);
    let got = discover(&d, &n).expect("discovers");
    assert_eq!(got.units.len(), 1);
    assert_eq!(got.units[0].name, "lm_head");
    assert_eq!(got.units[0].tensors, vec!["lm_head.weight"]);
    assert!(got.units[0].siblings.is_empty());
    assert_eq!(got.passthrough, vec!["model.norm.weight"]);
}

#[test]
fn a_partial_unit_is_refused_rather_than_silently_compressed() {
    let d = qwen3_def();
    let n = names(&["model.layers.0.self_attn.q_proj.weight"]);
    match discover(&d, &n) {
        Err(DiscoverError::IncompleteUnit { unit, missing }) => {
            assert_eq!(unit, "model.layers.0");
            assert_eq!(missing, vec!["mlp.down_proj"]);
        }
        other => panic!("expected IncompleteUnit, got {other:?}"),
    }
}

#[test]
fn a_definition_that_matches_nothing_is_an_error() {
    let d = qwen3_def();
    let n = names(&["transformer.h.0.attn.weight", "wte.weight"]);
    assert_eq!(discover(&d, &n).unwrap_err(), DiscoverError::NoUnitsFound);
}

#[test]
fn the_pattern_must_match_the_whole_module_name() {
    let d = qwen3_def();
    // `extra.model.layers.0` must NOT match `model\.layers\.\d+`.
    let n = names(&[
        "extra.model.layers.0.self_attn.q_proj.weight",
        "extra.model.layers.0.mlp.down_proj.weight",
    ]);
    assert_eq!(
        discover(&d, &n).unwrap_err(),
        DiscoverError::NoUnitsFound,
        "an anchored pattern must not match a suffix of a longer name"
    );
}

#[test]
fn an_invalid_regex_is_reported_not_panicked_on() {
    let d = ArchDef::from_toml(
        r#"
name = "t"
layout = "diffusers"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = '['
attrs = ["a"]
"#,
    )
    .unwrap();
    match discover(&d, &names(&["x.a.weight"])) {
        Err(DiscoverError::BadPattern { pattern, .. }) => assert_eq!(pattern, "["),
        other => panic!("expected BadPattern, got {other:?}"),
    }
}

/// End-to-end against the real tier-0 source model and the shipped Qwen3
/// definition. This is what validates the placement rule rather than asserting
/// it: the siblings we attribute to each unit must be exactly the tensors the
/// official compressor put in that unit's shard.
#[test]
fn discovery_on_the_real_model_matches_the_official_shards() {
    use df11_fixtures::{architecture_defs, skip_if_missing, SourceModel};

    let Some(fx) = skip_if_missing("discovery_on_the_real_model_matches_the_official_shards")
    else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(_src) = SourceModel::open(set) else {
        return;
    };
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs
        .iter()
        .find(|(n, _)| n == "qwen3-4b")
        .expect("qwen3-4b");
    let def = ArchDef::from_toml(toml).expect("parses");

    let model =
        df11_codec::safetensors::SafeTensorsFile::open(set.source_dir.join("model.safetensors"))
            .expect("open source model");
    let names: Vec<String> = model.names().map(String::from).collect();
    assert_eq!(names.len(), 47, "the truncated model has 47 tensors");

    let got = discover(&def, &names).expect("discovers");
    assert_eq!(got.units.len(), 4, "4 layers");

    for u in &got.units {
        assert_eq!(u.tensors.len(), 7, "{}: seven compressed linears", u.name);
        assert_eq!(
            u.siblings.len(),
            4,
            "{}: four norms travel with the unit",
            u.name
        );
        // Compare against what the official compressor actually put in the shard.
        let official = set.unit(&u.name);
        assert_eq!(official.len(), 6, "{}: six DF11 tensors", u.name);

        let shard = &official[0].file;
        let f = df11_codec::safetensors::SafeTensorsFile::open(shard).expect("open shard");
        let non_unit: Vec<String> = f
            .names()
            .filter(|n| {
                !matches!(
                    n.rsplit_once('.').map(|(_, s)| s),
                    Some("luts")
                        | Some("encoded_exponent")
                        | Some("sign_mantissa")
                        | Some("output_positions")
                        | Some("gaps")
                        | Some("split_positions")
                )
            })
            .map(String::from)
            .collect();
        let mut ours = u.siblings.clone();
        ours.sort();
        assert_eq!(
            ours, non_unit,
            "{}: the tensors we attribute to this unit must be exactly the \
             non-DF11 tensors the official shard contains",
            u.name
        );
    }

    let mut pt = got.passthrough.clone();
    pt.sort();
    assert_eq!(
        pt,
        vec![
            "lm_head.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string()
        ],
        "everything else stays outside the units"
    );
}

// ---- Phase 7: pattern shapes the Extended definitions introduce ----

fn def_of(units: &str) -> ArchDef {
    ArchDef::from_toml(&format!(
        "name = \"t\"\nlayout = \"comfyui-native\"\nformat_version = \"0.5.0\"\n\
         threads_per_block = [512]\nbytes_per_thread = 8\nsource = \"test\"\n{units}"
    ))
    .expect("valid")
}

/// ACEStep15 upstream writes `layers\.\d++`. Python's `re` (3.11+) reads that as a
/// possessive quantifier; under fullmatch, a possessive quantifier at the very
/// end matches exactly what the greedy one does, since nothing follows it to
/// backtrack for. The Rust engine has no possessive syntax, so it must be read
/// as greedy there -- not rejected, and not silently misparsed.
#[test]
fn a_trailing_possessive_digit_run_matches_like_python() {
    let d = def_of("[[unit]]\npattern = 'enc\\.layers\\.\\d++'\nattrs = [\"q\"]\n");
    let got = discover(
        &d,
        &names(&[
            "enc.layers.0.q.weight",
            "enc.layers.12.q.weight",
            "enc.layersX3.q.weight",
        ]),
    )
    .expect("a trailing \\d++ must be accepted");
    let units: Vec<&str> = got.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(units, ["enc.layers.0", "enc.layers.12"]);
}

/// A possessive quantifier anywhere else can change what matches; the engine
/// cannot reproduce it, so it must be refused rather than approximated.
#[test]
fn a_possessive_quantifier_mid_pattern_is_refused() {
    let d = def_of("[[unit]]\npattern = 'a\\d++\\.b'\nattrs = [\"q\"]\n");
    let e = discover(&d, &names(&["a1.b.q.weight"])).expect_err("must refuse");
    assert!(matches!(e, DiscoverError::BadPattern { .. }), "{e}");
}

/// Units must not nest. The official compressor detaches a unit's weights as it
/// goes, so a unit inside another fails upstream; here it would also claim the
/// inner unit's tensors as the outer one's siblings, writing them twice.
#[test]
fn nested_units_are_refused() {
    let d = def_of(
        "[[unit]]\npattern = 'blk\\.\\d+'\nattrs = [\"a\"]\n\
         [[unit]]\npattern = 'blk\\.\\d+\\.inner'\nattrs = [\"b\"]\n",
    );
    let e = discover(&d, &names(&["blk.0.a.weight", "blk.0.inner.b.weight"]))
        .expect_err("nesting must be refused");
    assert!(matches!(e, DiscoverError::NestedUnits { .. }), "{e}");
}

/// Upstream: when the matched module is itself an nn.Linear or nn.Embedding, its
/// own weight is compressed and the attrs are ignored. From tensor names alone a
/// module with its own `.weight` could be either case, so a definition that
/// gives attrs for such a module is ambiguous and must be refused.
#[test]
fn attrs_on_a_module_with_its_own_weight_are_refused() {
    let d = def_of("[[unit]]\npattern = 'proj'\nattrs = [\"1\"]\n");
    let e =
        discover(&d, &names(&["proj.weight", "proj.1.weight"])).expect_err("ambiguous module kind");
    assert!(matches!(e, DiscoverError::AmbiguousModule { .. }), "{e}");
}

/// Character classes, as SDXL uses them, select exactly the listed indices.
#[test]
fn a_character_class_pattern_selects_only_its_indices() {
    let d = def_of("[[unit]]\npattern = 'out\\.[01]\\.0'\nattrs = [\"e\"]\n");
    let got = discover(
        &d,
        &names(&["out.0.0.e.weight", "out.1.0.e.weight", "out.2.0.e.weight"]),
    )
    .unwrap();
    let units: Vec<&str> = got.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(units, ["out.0.0", "out.1.0"]);
    assert!(got.passthrough.contains(&"out.2.0.e.weight".to_string()));
}

/// `\\d++` is a literal backslash then a possessive `d++` -- not the digit class.
/// And a pattern with a possessive earlier on is refused even if it also ends in
/// `\d++`.
#[test]
fn only_a_genuine_trailing_digit_class_is_translated() {
    for p in ["a\\\\\\\\d++", "a\\\\d++\\\\.b\\\\d++"] {
        let d = def_of(&format!("[[unit]]\npattern = \"{p}\"\nattrs = [\"q\"]\n"));
        let e = discover(&d, &names(&["a1.b2.q.weight", "a\\ddd.q.weight"]))
            .expect_err(&format!("{p} must be refused"));
        assert!(matches!(e, DiscoverError::BadPattern { .. }), "{p}: {e}");
    }
}
