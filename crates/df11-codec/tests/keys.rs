//! Per-definition key rules, and the tied-`lm_head` alias.
//!
//! Found against real checkpoints (FINDINGS, "Definitions against real
//! checkpoints"): the Extended releases are ComfyUI's in-memory state_dict, after
//! its load-time key conversion -- RMSNorm `.scale` saved as `.weight` for the Flux
//! family (but NOT Krea-2, whose release keeps `.scale`), and the Cosmos `net.`
//! prefix stripped with its training state dropped. And the official compressor
//! compresses a tied `lm_head` as its own unit. Names change; values never do.

use df11_codec::arch::ArchDef;
use df11_codec::keys::{map_names, KeyError};

fn def(extra: &str, layout: &str, units: &str) -> ArchDef {
    ArchDef::from_toml(&format!(
        "name = \"t\"\nlayout = \"{layout}\"\nformat_version = \"0.5.0\"\n\
         threads_per_block = [512]\nbytes_per_thread = 8\nsource = \"t\"\n{extra}\n{units}"
    ))
    .expect("valid")
}

const BLOCKS: &str = "[[unit]]\npattern = 'blocks\\.\\d+'\nattrs = [\"a\"]\n";

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_definition_without_rules_maps_names_to_themselves() {
    let d = def("", "diffusers", BLOCKS);
    let m = map_names(&d, &names(&["blocks.0.a.weight", "x.scale"]), false).unwrap();
    assert_eq!(m.physical("x.scale"), Some("x.scale"));
    assert_eq!(m.visible(), ["blocks.0.a.weight", "x.scale"]);
}

#[test]
fn rename_changes_only_matching_names() {
    let d = def(
        "[[keys.rename]]\npattern = '\\.scale$'\nreplacement = '.weight'\n",
        "comfyui-native",
        BLOCKS,
    );
    let m = map_names(
        &d,
        &names(&[
            "blocks.0.a.weight",
            "blocks.0.norm.key_norm.scale",
            "scale_shift",
        ]),
        false,
    )
    .unwrap();
    assert_eq!(
        m.physical("blocks.0.norm.key_norm.weight"),
        Some("blocks.0.norm.key_norm.scale")
    );
    assert_eq!(
        m.physical("scale_shift"),
        Some("scale_shift"),
        "only a trailing .scale"
    );
    assert_eq!(
        m.physical("blocks.0.norm.key_norm.scale"),
        None,
        "the old name is gone"
    );
}

#[test]
fn strip_and_drop_as_cosmos_needs() {
    let d = def(
        "[keys]\nstrip_prefix = [\"net.\"]\ndrop = ['^accum_', '\\._extra_state$']\n",
        "comfyui-native",
        BLOCKS,
    );
    let m = map_names(
        &d,
        &names(&[
            "net.blocks.0.a.weight",
            "net.accum_iteration",
            "net.blocks.0.attn.k_norm._extra_state",
            "net.x_embedder.proj.1.weight",
        ]),
        false,
    )
    .unwrap();
    assert_eq!(
        m.visible(),
        ["blocks.0.a.weight", "x_embedder.proj.1.weight"]
    );
    assert_eq!(
        m.physical("blocks.0.a.weight"),
        Some("net.blocks.0.a.weight")
    );
}

/// The ComfyUI checkpoint prefix is stripped first, as before, and hides the rest
/// of a full checkpoint; the rules then apply to what remains.
#[test]
fn rules_apply_after_the_comfyui_prefix() {
    let d = def(
        "[[keys.rename]]\npattern = '\\.scale$'\nreplacement = '.weight'\n",
        "comfyui-native",
        BLOCKS,
    );
    let m = map_names(
        &d,
        &names(&[
            "model.diffusion_model.blocks.0.a.weight",
            "model.diffusion_model.blocks.0.n.scale",
            "first_stage_model.decoder.scale",
        ]),
        false,
    )
    .unwrap();
    assert_eq!(m.visible(), ["blocks.0.a.weight", "blocks.0.n.weight"]);
    assert_eq!(
        m.physical("blocks.0.n.weight"),
        Some("model.diffusion_model.blocks.0.n.scale")
    );
    assert_eq!(m.prefix(), Some("model.diffusion_model."));
}

/// Two source tensors landing on one name would silently lose one of them.
#[test]
fn a_rename_collision_is_refused() {
    let d = def(
        "[[keys.rename]]\npattern = '\\.scale$'\nreplacement = '.weight'\n",
        "comfyui-native",
        BLOCKS,
    );
    let e = map_names(
        &d,
        &names(&["n.scale", "n.weight", "blocks.0.a.weight"]),
        false,
    )
    .expect_err("collision");
    assert!(matches!(e, KeyError::Collision { .. }), "{e}");
}

#[test]
fn a_bad_rule_regex_is_refused_when_the_definition_loads() {
    let r = ArchDef::from_toml(
        "name = \"t\"\nlayout = \"diffusers\"\nformat_version = \"0.5.0\"\n\
         threads_per_block = [512]\nbytes_per_thread = 8\nsource = \"t\"\n\
         [keys]\ndrop = ['(']\n[[unit]]\npattern = 'a'\nattrs = []\n",
    );
    assert!(r.is_err());
}

const LLM: &str = "[[unit]]\npattern = 'lm_head'\nattrs = []\n\
                   [[unit]]\npattern = 'model\\.embed_tokens'\nattrs = []\n";

/// Tied embeddings: the source has no `lm_head.weight`, the official compressor
/// still compresses `lm_head` -- the shared weight -- as its own unit.
#[test]
fn a_tied_lm_head_is_an_alias_of_the_embedding() {
    let d = def("", "transformers", LLM);
    let n = names(&["model.embed_tokens.weight", "model.norm.weight"]);
    let m = map_names(&d, &n, true).unwrap();
    assert_eq!(
        m.physical("lm_head.weight"),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(m.visible().len(), 3);

    // Only when the config says tied...
    assert_eq!(
        map_names(&d, &n, false).unwrap().physical("lm_head.weight"),
        None
    );
    // ...only when the definition compresses lm_head...
    let no_unit = def("", "transformers", BLOCKS);
    assert_eq!(
        map_names(&no_unit, &n, true)
            .unwrap()
            .physical("lm_head.weight"),
        None
    );
    // ...and never over a real lm_head.
    let with = names(&["model.embed_tokens.weight", "lm_head.weight"]);
    assert_eq!(
        map_names(&d, &with, true)
            .unwrap()
            .physical("lm_head.weight"),
        Some("lm_head.weight")
    );
}

// ---- through the writer ----

use df11_codec::safetensors::{write_file, Dtype, OutTensor, SafeTensorsFile};
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};

fn bf16(n: usize, seed: u8) -> Vec<u8> {
    // Plausible small weights: exponent around 120, varied mantissa.
    (0..n)
        .flat_map(|i| {
            let v: u16 = (0x3C00 | ((i as u16).wrapping_mul(37).wrapping_add(seed as u16) & 0x3FF))
                ^ ((i as u16 & 1) << 15);
            v.to_le_bytes()
        })
        .collect()
}

#[test]
fn a_renamed_tensor_is_written_under_its_new_name_with_its_bytes() {
    let d = def(
        "[[keys.rename]]\npattern = '\\.scale$'\nreplacement = '.weight'\n",
        "comfyui-native",
        BLOCKS,
    );
    let dir = df11_fixtures::scratch("keys_rename_src");
    let norm = bf16(64, 9);
    write_file(
        dir.join("model.safetensors"),
        &[
            OutTensor::owned(
                "model.diffusion_model.blocks.0.a.weight",
                Dtype::new("BF16"),
                vec![64, 64],
                bf16(4096, 1),
            ),
            OutTensor::owned(
                "model.diffusion_model.blocks.0.n.scale",
                Dtype::new("BF16"),
                vec![64],
                norm.clone(),
            ),
        ],
        &Default::default(),
    )
    .unwrap();
    let out = df11_fixtures::scratch("keys_rename_out");
    let src = ModelSource::open(dir.join("model.safetensors")).unwrap();
    let opts = WriteOptions {
        verify: true,
        ..Default::default()
    };
    write_directory(&src, &d, &out, &opts).unwrap();
    let f = SafeTensorsFile::open(out.join("model.safetensors")).unwrap();
    assert!(f.info("blocks.0.n.scale").is_none());
    assert_eq!(
        f.read("blocks.0.n.weight").unwrap(),
        norm,
        "renamed, bytes unchanged"
    );
}

#[test]
fn a_tied_lm_head_is_compressed_as_its_own_unit() {
    let d = def("", "transformers", LLM);
    let dir = df11_fixtures::scratch("keys_tied_src");
    write_file(
        dir.join("model.safetensors"),
        &[
            OutTensor::owned(
                "model.embed_tokens.weight",
                Dtype::new("BF16"),
                vec![128, 64],
                bf16(8192, 3),
            ),
            OutTensor::owned(
                "model.norm.weight",
                Dtype::new("BF16"),
                vec![64],
                bf16(64, 5),
            ),
        ],
        &Default::default(),
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), br#"{"tie_word_embeddings": true}"#).unwrap();
    let out = df11_fixtures::scratch("keys_tied_out");
    let src = ModelSource::open(dir.join("model.safetensors")).unwrap();
    let opts = WriteOptions {
        verify: true,
        ..Default::default()
    };
    let r = write_directory(&src, &d, &out, &opts).unwrap();
    assert_eq!(r.units, 2);
    assert_eq!(r.verified.len(), 2, "both decode back to the embedding");
    let head = SafeTensorsFile::open(out.join("lm_head.safetensors")).unwrap();
    let embed = SafeTensorsFile::open(out.join("model_embed_tokens.safetensors")).unwrap();
    assert_eq!(
        head.read("lm_head.sign_mantissa").unwrap(),
        embed.read("model.embed_tokens.sign_mantissa").unwrap(),
        "the same weight, compressed twice, as upstream does"
    );
}
