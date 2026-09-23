//! Architecture definitions: which modules form a compression unit, and in what
//! order their tensors are concatenated.
//!
//! These live in versioned TOML under `data/architectures/`, not in code, so
//! adding a model does not mean recompiling.
//!
//! **The attribute order is load-bearing.** It fixes the concatenation, hence
//! `split_positions`, hence every compressed byte. Phase 0 derived two of these
//! orders from class definitions and got the right set with the wrong order in
//! both cases (FINDINGS 0.9), so every definition shipped here is transcribed
//! from a published official release's own `dfloat11_config`, or from Extended's
//! `pattern_dict.py`, and records which in `source`.

use serde::Deserialize;
use std::fmt;

/// Which ecosystem's key naming and output shape this definition targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    /// A transformers LLM: directory of shards plus `config.json`.
    Transformers,
    /// A diffusers model: one shard per unit plus `config.json`.
    Diffusers,
    /// ComfyUI-native: a single safetensors, no `config.json`.
    ComfyuiNative,
    /// Diffusers, single file: one `diffusion_pytorch_model.safetensors` holding
    /// every tensor, plus `config.json`. What the Qwen-Image releases ship.
    DiffusersSingle,
}

impl Layout {
    /// The name a definition file uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Transformers => "transformers",
            Layout::Diffusers => "diffusers",
            Layout::ComfyuiNative => "comfyui-native",
            Layout::DiffusersSingle => "diffusers-single",
        }
    }

    /// Whether the output is one file rather than a shard per unit.
    pub fn single_file(self) -> bool {
        matches!(self, Layout::ComfyuiNative | Layout::DiffusersSingle)
    }
}

/// One compression unit pattern.
#[derive(Debug, Clone, Deserialize)]
pub struct UnitPattern {
    /// Regex matched against the full module name.
    pub pattern: String,
    /// Attribute paths, **in concatenation order**. Empty means the matched
    /// module is itself a single tensor to compress — the case every published
    /// DF11 LLM at 8B and above uses for `lm_head` and `model.embed_tokens`.
    pub attrs: Vec<String>,
}

impl UnitPattern {
    /// A unit holding exactly one tensor, whose `split_positions` is empty.
    pub fn is_standalone(&self) -> bool {
        self.attrs.is_empty()
    }
}

/// A complete architecture definition.
#[derive(Debug, Clone, Deserialize)]
pub struct ArchDef {
    pub name: String,
    pub layout: Layout,
    /// The `dfloat11_config.version` to stamp. Published releases carry both
    /// `0.2.0` and `0.5.0`; it is per-target, not global.
    pub format_version: String,
    pub threads_per_block: Vec<u32>,
    pub bytes_per_thread: u32,
    /// Where this definition was transcribed from. Required: a definition
    /// without provenance is a guess, and Phase 0 showed guesses are wrong.
    pub source: String,
    #[serde(default, rename = "unit")]
    pub units: Vec<UnitPattern>,
    /// How source tensor names become output names, beyond the ComfyUI
    /// checkpoint prefix. Empty for most definitions. See [`crate::keys`].
    #[serde(default)]
    pub keys: KeyRules,
    /// The single output file's name, for the single-file layouts, when it is
    /// not the layout's usual one.
    #[serde(default)]
    pub file: Option<String>,
}

/// Name rules a definition can carry: what ComfyUI's key conversion does to a
/// checkpoint before the Extended releases were saved. Names only; values are
/// never changed.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRules {
    /// Prefixes removed where present, after the ComfyUI checkpoint prefix.
    #[serde(default)]
    pub strip_prefix: Vec<String>,
    /// Regexes (searched) for tensors left out entirely, such as training state.
    #[serde(default)]
    pub drop: Vec<String>,
    /// Applied in order to every remaining name.
    #[serde(default)]
    pub rename: Vec<Rename>,
}

/// Replace the first match of `pattern` with `replacement`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rename {
    pub pattern: String,
    pub replacement: String,
}

/// Why a definition was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchError {
    Empty(&'static str),
    NoUnits,
    DuplicatePattern(String),
    DuplicateAttr { pattern: String, attr: String },
    BadNumber(&'static str),
    Parse(String),
    BadKeyRule { rule: String, message: String },
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(field) => write!(f, "`{field}` must not be empty"),
            Self::BadKeyRule { rule, message } => {
                write!(f, "key rule {rule:?} is not a valid regex: {message}")
            }
            Self::NoUnits => write!(f, "a definition needs at least one [[unit]]"),
            Self::DuplicatePattern(p) => write!(f, "pattern {p:?} appears more than once"),
            Self::DuplicateAttr { pattern, attr } => write!(
                f,
                "pattern {pattern:?} lists attribute {attr:?} twice; the order is the \
                 concatenation order, so a repeat is always a mistake"
            ),
            Self::BadNumber(field) => write!(f, "`{field}` must be greater than zero"),
            Self::Parse(e) => write!(f, "could not parse the definition: {e}"),
        }
    }
}

impl std::error::Error for ArchError {}

impl ArchDef {
    /// Parse and validate a definition.
    pub fn from_toml(s: &str) -> Result<Self, ArchError> {
        let def: ArchDef = toml::from_str(s).map_err(|e| ArchError::Parse(e.to_string()))?;
        def.validate()?;
        Ok(def)
    }

    fn validate(&self) -> Result<(), ArchError> {
        if self.name.trim().is_empty() {
            return Err(ArchError::Empty("name"));
        }
        if self.source.trim().is_empty() {
            return Err(ArchError::Empty("source"));
        }
        if self.format_version.trim().is_empty() {
            return Err(ArchError::Empty("format_version"));
        }
        if self.threads_per_block.is_empty() || self.threads_per_block.contains(&0) {
            return Err(ArchError::BadNumber("threads_per_block"));
        }
        if self.bytes_per_thread == 0 {
            return Err(ArchError::BadNumber("bytes_per_thread"));
        }
        if self.units.is_empty() {
            return Err(ArchError::NoUnits);
        }
        let rules = self
            .keys
            .drop
            .iter()
            .chain(self.keys.rename.iter().map(|r| &r.pattern));
        for rule in rules {
            regex::Regex::new(rule).map_err(|e| ArchError::BadKeyRule {
                rule: rule.clone(),
                message: e.to_string(),
            })?;
        }
        let mut seen = Vec::new();
        for u in &self.units {
            if u.pattern.trim().is_empty() {
                return Err(ArchError::Empty("unit.pattern"));
            }
            if seen.contains(&u.pattern) {
                return Err(ArchError::DuplicatePattern(u.pattern.clone()));
            }
            seen.push(u.pattern.clone());
            let mut attrs = Vec::new();
            for a in &u.attrs {
                if a.trim().is_empty() {
                    return Err(ArchError::Empty("unit.attrs entry"));
                }
                if attrs.contains(a) {
                    return Err(ArchError::DuplicateAttr {
                        pattern: u.pattern.clone(),
                        attr: a.clone(),
                    });
                }
                attrs.push(a.clone());
            }
        }
        Ok(())
    }

    /// The definition as the `pattern_dict` the official config carries, in order.
    pub fn pattern_dict(&self) -> Vec<(String, Vec<String>)> {
        self.units
            .iter()
            .map(|u| (u.pattern.clone(), u.attrs.clone()))
            .collect()
    }

    /// Byte offsets where each concatenated tensor ends, excluding the total —
    /// the `split_positions` layout confirmed in FINDINGS 0.2: `n - 1` entries
    /// for `n` tensors, and empty for a standalone unit.
    pub fn split_positions(counts: &[u64]) -> Vec<i64> {
        if counts.len() < 2 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(counts.len() - 1);
        let mut acc = 0i64;
        for c in &counts[..counts.len() - 1] {
            acc += *c as i64;
            out.push(acc);
        }
        out
    }
}
