//! Applying an architecture definition to a model's tensor names.

use crate::arch::ArchDef;
use std::fmt;

/// One compression unit found in a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredUnit {
    /// The module name, e.g. `model.layers.0`.
    pub name: String,
    /// Source tensors to compress, **in the definition's attribute order**.
    pub tensors: Vec<String>,
    /// Other tensors under the same module that are not compressed.
    ///
    /// These matter: the official compressor detaches the whole submodule and
    /// saves its entire `state_dict`, so they travel into the unit's shard
    /// rather than staying in the remainder file (FINDINGS 0.7). A writer that
    /// ignores this emits the right set of tensors in the wrong files.
    pub siblings: Vec<String>,
}

/// What an architecture definition found in a model.
#[derive(Debug, Clone)]
pub struct Discovery {
    pub units: Vec<DiscoveredUnit>,
    /// Tensors belonging to no unit and no unit's module.
    pub passthrough: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoverError {
    BadPattern {
        pattern: String,
        message: String,
    },
    /// A module matched but is missing attributes the definition names.
    IncompleteUnit {
        unit: String,
        missing: Vec<String>,
    },
    /// Two patterns claim the same tensor.
    Contested {
        tensor: String,
        units: Vec<String>,
    },
    NoUnitsFound,
    /// One unit lies inside another. Upstream detaches weights as it goes and
    /// fails; here the inner unit's tensors would also be the outer's siblings.
    NestedUnits {
        outer: String,
        inner: String,
    },
    /// The definition gives attrs for a module that has its own `.weight`.
    /// Upstream ignores the attrs when that module is an nn.Linear or
    /// nn.Embedding and uses them otherwise; tensor names cannot tell which.
    AmbiguousModule {
        unit: String,
    },
}

impl fmt::Display for DiscoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadPattern { pattern, message } => {
                write!(f, "pattern {pattern:?} is not a valid regex: {message}")
            }
            Self::IncompleteUnit { unit, missing } => write!(
                f,
                "unit {unit:?} is missing {} attribute(s): {}. Compressing a partial \
                 unit would silently change the concatenation and therefore every byte",
                missing.len(),
                missing.join(", ")
            ),
            Self::Contested { tensor, units } => write!(
                f,
                "tensor {tensor:?} is claimed by more than one unit ({}); the definition \
                 is ambiguous",
                units.join(", ")
            ),
            Self::NestedUnits { outer, inner } => write!(
                f,
                "unit {inner:?} lies inside unit {outer:?}; units must not nest"
            ),
            Self::AmbiguousModule { unit } => write!(
                f,
                "unit {unit:?} has its own weight and the definition also lists attrs for \
                 it. The official compressor ignores the attrs if the module is an \
                 nn.Linear or nn.Embedding, and names alone cannot tell; if it is one, \
                 the definition should give `attrs = []`"
            ),
            Self::NoUnitsFound => write!(
                f,
                "the definition matched no modules in this model -- wrong definition \
                 for this checkpoint, or a key prefix that needs stripping"
            ),
        }
    }
}

impl std::error::Error for DiscoverError {}

/// Find every compression unit the definition describes in `tensor_names`.
pub fn discover(def: &ArchDef, tensor_names: &[String]) -> Result<Discovery, DiscoverError> {
    use regex::Regex;
    use std::collections::BTreeMap;

    // Anchored: a pattern must match the whole module name, so
    // `model\.layers\.\d+` does not match `extra.model.layers.0`.
    let mut res = Vec::with_capacity(def.units.len());
    for u in &def.units {
        let bad = |message: String| DiscoverError::BadPattern {
            pattern: u.pattern.clone(),
            message,
        };
        let translated = python_to_rust(&u.pattern).map_err(bad)?;
        let r = Regex::new(&format!("^(?:{translated})$")).map_err(|e| bad(e.to_string()))?;
        res.push(r);
    }

    // (pattern index, module) -> attr index -> tensor name
    let mut found: BTreeMap<(usize, String), BTreeMap<usize, String>> = BTreeMap::new();
    let mut claims: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (pi, u) in def.units.iter().enumerate() {
        let suffixes: Vec<(usize, String)> = if u.is_standalone() {
            vec![(0, ".weight".to_string())]
        } else {
            u.attrs
                .iter()
                .enumerate()
                .map(|(ai, a)| (ai, format!(".{a}.weight")))
                .collect()
        };
        for name in tensor_names {
            for (ai, suffix) in &suffixes {
                let Some(module) = name.strip_suffix(suffix.as_str()) else {
                    continue;
                };
                if module.is_empty() || !res[pi].is_match(module) {
                    continue;
                }
                found
                    .entry((pi, module.to_string()))
                    .or_default()
                    .insert(*ai, name.clone());
                claims
                    .entry(name.clone())
                    .or_default()
                    .push(module.to_string());
            }
        }
    }

    if let Some((tensor, units)) = claims.iter().find(|(_, u)| u.len() > 1) {
        let mut units = units.clone();
        units.sort();
        units.dedup();
        if units.len() > 1 {
            return Err(DiscoverError::Contested {
                tensor: tensor.clone(),
                units,
            });
        }
    }

    if found.is_empty() {
        return Err(DiscoverError::NoUnitsFound);
    }

    // A module with its own weight, given attrs: see AmbiguousModule.
    for (pi, module) in found.keys() {
        if !def.units[*pi].is_standalone()
            && tensor_names
                .iter()
                .any(|n| *n == format!("{module}.weight"))
        {
            return Err(DiscoverError::AmbiguousModule {
                unit: module.clone(),
            });
        }
    }

    // Units must not nest. Sorted, an enclosing module comes right before the
    // first module inside it... but not necessarily adjacent, so check pairs.
    let modules: Vec<&String> = found.keys().map(|(_, m)| m).collect();
    for outer in &modules {
        let prefix = format!("{outer}.");
        if let Some(inner) = modules.iter().find(|m| m.starts_with(&prefix)) {
            return Err(DiscoverError::NestedUnits {
                outer: (*outer).clone(),
                inner: (*inner).clone(),
            });
        }
    }

    let mut units: Vec<DiscoveredUnit> = Vec::with_capacity(found.len());
    for ((pi, module), by_attr) in &found {
        let want = if def.units[*pi].is_standalone() {
            1
        } else {
            def.units[*pi].attrs.len()
        };
        if by_attr.len() != want {
            let missing: Vec<String> = (0..want)
                .filter(|i| !by_attr.contains_key(i))
                .map(|i| def.units[*pi].attrs[i].clone())
                .collect();
            return Err(DiscoverError::IncompleteUnit {
                unit: module.clone(),
                missing,
            });
        }
        let tensors: Vec<String> = (0..want).map(|i| by_attr[&i].clone()).collect();
        let prefix = format!("{module}.");
        let siblings: Vec<String> = tensor_names
            .iter()
            .filter(|n| n.starts_with(&prefix) && !tensors.contains(n))
            .cloned()
            .collect();
        units.push(DiscoveredUnit {
            name: module.clone(),
            tensors,
            siblings,
        });
    }

    // Pattern order first, then natural order within a pattern, so layer 10
    // follows layer 9 rather than layer 1.
    let order: BTreeMap<&String, usize> = found.keys().map(|(pi, m)| (m, *pi)).collect();
    units.sort_by(|a, b| {
        order[&a.name]
            .cmp(&order[&b.name])
            .then_with(|| natural_cmp(&a.name, &b.name))
    });

    let claimed: Vec<String> = units
        .iter()
        .flat_map(|u| u.tensors.iter().chain(u.siblings.iter()))
        .cloned()
        .collect();
    let passthrough: Vec<String> = tensor_names
        .iter()
        .filter(|n| !claimed.contains(n))
        .cloned()
        .collect();

    Ok(Discovery { units, passthrough })
}

/// Python `re` syntax the Rust engine lacks, translated where it is provably
/// equivalent and refused where it is not.
///
/// The one case upstream uses: a possessive `\d++` at the very end of a pattern.
/// Under fullmatch nothing follows it, so there is nothing to backtrack for and
/// it matches exactly the strings the greedy `\d+` does. A possessive quantifier
/// anywhere else can change the match and is refused.
///
/// The Rust engine happens to accept `\d++` itself, as the nested repetition
/// `(\d+)+` -- the same strings at the end of a pattern, different ones
/// mid-pattern (Python's `a\d++1` never matches). So the translation is explicit
/// rather than left to that coincidence, and the refusal is what matters.
fn python_to_rust(pattern: &str) -> Result<String, String> {
    let possessive =
        |p: &str| p.contains("++") || p.contains("*+") || p.contains("?+") || p.contains("}+");
    if !possessive(pattern) {
        return Ok(pattern.to_string());
    }
    if let Some(head) = pattern.strip_suffix(r"\d++") {
        // An odd run of backslashes before `d` means `\d` is the class, not a
        // literal backslash followed by `d`.
        let slashes = head.chars().rev().take_while(|&c| c == '\\').count();
        if slashes % 2 == 0 && !possessive(head) {
            return Ok(format!(r"{head}\d+"));
        }
    }
    Err(
        "possessive quantifiers are supported only as a trailing `\\d++`, where they \
         match exactly as the greedy form does"
            .to_string(),
    )
}

/// Compare names treating digit runs as numbers, so `layers.10` sorts after
/// `layers.9` rather than between `layers.1` and `layers.2`.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (mut x, mut y) = (a.as_bytes(), b.as_bytes());
    loop {
        match (x.first(), y.first()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(&c), Some(&d)) => {
                if c.is_ascii_digit() && d.is_ascii_digit() {
                    let xi = x
                        .iter()
                        .position(|c| !c.is_ascii_digit())
                        .unwrap_or(x.len());
                    let yi = y
                        .iter()
                        .position(|c| !c.is_ascii_digit())
                        .unwrap_or(y.len());
                    let xv: u128 = std::str::from_utf8(&x[..xi]).unwrap().parse().unwrap_or(0);
                    let yv: u128 = std::str::from_utf8(&y[..yi]).unwrap().parse().unwrap_or(0);
                    match xv.cmp(&yv) {
                        Ordering::Equal => {
                            x = &x[xi..];
                            y = &y[yi..];
                        }
                        o => return o,
                    }
                } else {
                    match c.cmp(&d) {
                        Ordering::Equal => {
                            x = &x[1..];
                            y = &y[1..];
                        }
                        o => return o,
                    }
                }
            }
        }
    }
}
