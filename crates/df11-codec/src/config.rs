//! Generating the output `config.json`.
//!
//! Two shapes exist in the wild, and the difference is not cosmetic.
//!
//! Every published **diffusers** DF11 release carries exactly two keys —
//! `dfloat11_config` and a vestigial `model_type` — not the diffusers schema
//! DESIGN §1.6 assumed. The loader, given `bfloat16_model=`, reads the file as a
//! plain dict and requires only `dfloat11_config` (FINDINGS 0.8).
//!
//! The **transformers** path is different: upstream's `save_pretrained` does not
//! add to the source config, it re-normalises the whole schema against whatever
//! transformers version is installed, and `transformers_version` is that
//! library's own version string — uncomputable from a Rust binary. Reproducing
//! that byte for byte therefore needs a *declared* version target plus a table
//! of per-version transformations; `phase0/rebuild_config.py --mode full` does
//! it and is byte-exact on both fixtures. This module deliberately does not,
//! and [`ConfigMode::PreserveSource`] says what it does instead.

use crate::arch::ArchDef;
use serde_json::{Map, Value};

/// Which output shape to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigMode {
    /// `dfloat11_config` plus `model_type` if the source had one. What every
    /// published diffusers release actually ships.
    Minimal,
    /// The source config verbatim, with `dfloat11_config` added.
    ///
    /// **Not byte-identical to `save_pretrained`'s output**, which rewrites the
    /// whole schema. It is a faithful superset of the source and carries
    /// everything the loader needs.
    PreserveSource,
}

/// The `dfloat11_config` block, with `pattern_dict` in definition order.
pub fn dfloat11_config(def: &ArchDef) -> Value {
    let mut pattern = Map::new();
    for u in &def.units {
        pattern.insert(
            u.pattern.clone(),
            Value::Array(u.attrs.iter().map(|a| Value::String(a.clone())).collect()),
        );
    }
    let mut o = Map::new();
    o.insert("version".into(), Value::String(def.format_version.clone()));
    o.insert(
        "threads_per_block".into(),
        Value::Array(def.threads_per_block.iter().map(|&t| t.into()).collect()),
    );
    o.insert("bytes_per_thread".into(), def.bytes_per_thread.into());
    o.insert("pattern_dict".into(), Value::Object(pattern));
    Value::Object(o)
}

/// Build the output `config.json`.
pub fn build_config(source: Option<&Value>, def: &ArchDef, mode: ConfigMode) -> Value {
    let mut out = Map::new();
    match mode {
        ConfigMode::Minimal => {
            if let Some(mt) = source
                .and_then(|s| s.get("model_type"))
                .filter(|v| !v.is_null())
            {
                out.insert("model_type".into(), mt.clone());
            }
        }
        ConfigMode::PreserveSource => {
            if let Some(o) = source.and_then(|s| s.as_object()) {
                for (k, v) in o {
                    // Any dfloat11_config already present describes a different
                    // compression of this model; ours replaces it outright
                    // rather than merging, so no stale key survives.
                    if k != "dfloat11_config" {
                        out.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }
    out.insert("dfloat11_config".into(), dfloat11_config(def));
    Value::Object(out)
}
