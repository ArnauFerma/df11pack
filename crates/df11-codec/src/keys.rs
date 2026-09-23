//! Output names from source names.
//!
//! Most definitions write every tensor under its source name. Two things change
//! that, both found checking definitions against real checkpoints (FINDINGS):
//!
//! - **Key rules** a definition carries ([`crate::arch::KeyRules`]). The Extended
//!   releases are ComfyUI's in-memory `state_dict`, saved after its load-time key
//!   conversion: RMSNorm `.scale` becomes `.weight` for the Flux family, and the
//!   Cosmos family loses its `net.` prefix and training state. Per definition, not
//!   global -- Krea-2's release keeps `.scale`.
//! - **A tied `lm_head`.** With `tie_word_embeddings`, a source has no
//!   `lm_head.weight`, yet the official compressor walks the tied Linear and
//!   compresses the shared weight as its own unit. When a definition compresses
//!   `lm_head`, the name is made an alias of the embedding.
//!
//! Only names change. A mapped tensor's bytes are read from its physical name.

use crate::arch::{ArchDef, Layout};
use crate::source::COMFYUI_PREFIX;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyError {
    /// Two source tensors would be written under one name.
    Collision {
        name: String,
        first: String,
        second: String,
    },
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collision {
                name,
                first,
                second,
            } => write!(
                f,
                "source tensors {first:?} and {second:?} would both be written as {name:?}"
            ),
        }
    }
}

impl std::error::Error for KeyError {}

/// Visible (output) name -> physical (source) name.
#[derive(Debug, Clone, Default)]
pub struct NameMap {
    map: BTreeMap<String, String>,
    prefix: Option<String>,
}

impl NameMap {
    /// Where a visible name's bytes live.
    pub fn physical(&self, visible: &str) -> Option<&str> {
        self.map.get(visible).map(String::as_str)
    }

    /// Every visible name, sorted.
    pub fn visible(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    /// The ComfyUI checkpoint prefix that was stripped, if any.
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Names only through a checkpoint prefix, no rules: what `View::new` does.
    pub fn with_prefix(physical: &[String], prefix: Option<&str>) -> Self {
        let p = prefix.unwrap_or("");
        NameMap {
            map: physical
                .iter()
                .filter_map(|n| n.strip_prefix(p).map(|v| (v.to_string(), n.clone())))
                .collect(),
            prefix: prefix.map(str::to_string),
        }
    }
}

/// Map a source's physical names to the names the output carries.
///
/// `tied` is the source config's `tie_word_embeddings`.
pub fn map_names(def: &ArchDef, physical: &[String], tied: bool) -> Result<NameMap, KeyError> {
    // First the checkpoint prefix, as always: it also hides the VAE and text
    // encoders of a full ComfyUI checkpoint.
    let prefix = (def.layout == Layout::ComfyuiNative
        && physical.iter().any(|n| n.starts_with(COMFYUI_PREFIX)))
    .then_some(COMFYUI_PREFIX);
    let base = NameMap::with_prefix(physical, prefix);

    // Validated when the definition loaded.
    let drop: Vec<regex::Regex> = def
        .keys
        .drop
        .iter()
        .map(|r| regex::Regex::new(r).expect("validated"))
        .collect();
    let rename: Vec<(regex::Regex, &str)> = def
        .keys
        .rename
        .iter()
        .map(|r| {
            (
                regex::Regex::new(&r.pattern).expect("validated"),
                r.replacement.as_str(),
            )
        })
        .collect();

    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (visible, phys) in base.map {
        let mut name = visible;
        if let Some(p) = def
            .keys
            .strip_prefix
            .iter()
            .find(|p| name.starts_with(p.as_str()))
        {
            name = name[p.len()..].to_string();
        }
        if drop.iter().any(|r| r.is_match(&name)) {
            continue;
        }
        for (r, to) in &rename {
            name = r.replacen(&name, 1, *to).into_owned();
        }
        if let Some(first) = map.get(&name) {
            return Err(KeyError::Collision {
                name,
                first: first.clone(),
                second: phys,
            });
        }
        map.insert(name, phys);
    }

    // The tied lm_head: only when tied, only when the definition compresses a
    // module named `lm_head`, and never over a real one.
    const HEAD: &str = "lm_head.weight";
    const EMBED: &str = "model.embed_tokens.weight";
    let compresses_head = def.units.iter().any(|u| {
        u.is_standalone()
            && regex::Regex::new(&format!("^(?:{})$", u.pattern))
                .is_ok_and(|r| r.is_match("lm_head"))
    });
    if tied && compresses_head && !map.contains_key(HEAD) {
        if let Some(embed) = map.get(EMBED).cloned() {
            map.insert(HEAD.to_string(), embed);
        }
    }

    Ok(NameMap {
        map,
        prefix: prefix.map(str::to_string),
    })
}
