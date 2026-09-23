//! Definitions against real checkpoints -- names and shapes only.
//!
//! Every other test proves a definition is *reproduced*; stand-in models carry the
//! definition's own module names, so they cannot show the definition fits a real
//! model. These fixtures are the safetensors headers of real source checkpoints
//! and of the real DF11 releases made from them (`phase0/fetch_real_headers.py`;
//! no weights). Discovery runs on the real source names, and what it would write
//! is compared with what the release contains:
//!
//! - the same units;
//! - each unit's weight count equal to the release's `sign_mantissa` length, and
//!   its tensor count to `split_positions` + 1 -- the concatenation membership;
//! - every compressed tensor BF16;
//! - every other tensor in the same place: a directory release's unit shards hold
//!   exactly our siblings, its remainder exactly our passthrough; a single-file
//!   release holds exactly what we would write.

use df11_codec::arch::ArchDef;
use df11_codec::discover::discover;
use df11_codec::keys::map_names;
use df11_codec::write::{remainder_name, shard_name};
use df11_fixtures::architecture_defs;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const DF11: [&str; 6] = [
    "luts",
    "encoded_exponent",
    "sign_mantissa",
    "output_positions",
    "gaps",
    "split_positions",
];

/// name -> (dtype, shape), over every file of one side.
fn tensors(side: &Value) -> BTreeMap<String, (String, Vec<u64>)> {
    let mut out = BTreeMap::new();
    for (_, h) in side["files"].as_object().unwrap() {
        for (n, v) in h["tensors"].as_object().unwrap() {
            let shape = v[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            out.insert(n.clone(), (v[0].as_str().unwrap().to_string(), shape));
        }
    }
    out
}

fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// Everything wrong with one definition against its real checkpoint.
fn check(name: &str, fx: &Value, def: &ArchDef) -> Vec<String> {
    let mut bad = Vec::new();
    let src = tensors(&fx["source"]);
    let rel = tensors(&fx["release"]);

    // What we would see: the writer's own name map -- the ComfyUI prefix, the
    // definition's key rules, and a tied lm_head.
    let tied = fx["source"]["tie_word_embeddings"]
        .as_bool()
        .unwrap_or(false);
    let physical: Vec<String> = src.keys().cloned().collect();
    let map = match map_names(def, &physical, tied) {
        Ok(m) => m,
        Err(e) => return vec![format!("name mapping failed: {e}")],
    };
    let names = map.visible();
    let lookup = |n: &str| src[map.physical(n).expect("visible")].clone();

    let found = match discover(def, &names) {
        Ok(f) => f,
        Err(e) => return vec![format!("discovery failed: {e}")],
    };

    let ours: BTreeSet<&str> = found.units.iter().map(|u| u.name.as_str()).collect();
    let theirs: BTreeSet<&str> = rel
        .keys()
        .filter_map(|n| n.strip_suffix(".encoded_exponent"))
        .collect();
    if ours != theirs {
        let only_ours: Vec<_> = ours.difference(&theirs).take(5).collect();
        let only_theirs: Vec<_> = theirs.difference(&ours).take(5).collect();
        bad.push(format!(
            "units differ: {} ours, {} release; only ours {only_ours:?}, only release {only_theirs:?}",
            ours.len(),
            theirs.len()
        ));
    }

    let mut not_bf16: BTreeMap<String, usize> = BTreeMap::new();
    for u in &found.units {
        if !theirs.contains(u.name.as_str()) {
            continue;
        }
        let n: u64 = u.tensors.iter().map(|t| numel(&lookup(t).1)).sum();
        let want = rel[&format!("{}.sign_mantissa", u.name)].1[0];
        if n != want {
            bad.push(format!("{}: {n} weights, release {want}", u.name));
        }
        let splits = rel[&format!("{}.split_positions", u.name)]
            .1
            .first()
            .copied()
            .unwrap_or(0);
        if u.tensors.len() as u64 != splits + 1 {
            bad.push(format!(
                "{}: {} tensors, release split_positions implies {}",
                u.name,
                u.tensors.len(),
                splits + 1
            ));
        }
        for t in &u.tensors {
            let d = lookup(t).0;
            if d != "BF16" {
                *not_bf16.entry(d).or_insert(0) += 1;
            }
        }
    }
    if !not_bf16.is_empty() {
        bad.push(format!("source not BF16: {not_bf16:?} tensors to compress"));
    }

    // Placement of everything that is not compressed.
    let df11_of = |u: &str| -> Vec<String> { DF11.iter().map(|f| format!("{u}.{f}")).collect() };
    if def.layout.single_file() {
        let mut want: BTreeSet<String> = found.passthrough.iter().cloned().collect();
        for u in &found.units {
            want.extend(u.siblings.iter().cloned());
            want.extend(df11_of(&u.name));
        }
        let got: BTreeSet<String> = rel.keys().cloned().collect();
        diff(&mut bad, "single file", &want, &got);
    } else {
        let files = fx["release"]["files"].as_object().unwrap();
        let in_file = |f: &str| -> BTreeSet<String> {
            files
                .get(f)
                .map(|h| h["tensors"].as_object().unwrap().keys().cloned().collect())
                .unwrap_or_default()
        };
        for u in &found.units {
            let mut want: BTreeSet<String> = u.siblings.iter().cloned().collect();
            want.extend(df11_of(&u.name));
            diff(
                &mut bad,
                &shard_name(&u.name),
                &want,
                &in_file(&shard_name(&u.name)),
            );
        }
        let want: BTreeSet<String> = found.passthrough.iter().cloned().collect();
        let rem = remainder_name(def.layout);
        diff(&mut bad, rem, &want, &in_file(rem));
    }
    let _ = name;
    bad
}

fn diff(bad: &mut Vec<String>, what: &str, want: &BTreeSet<String>, got: &BTreeSet<String>) {
    if want != got {
        let missing: Vec<_> = want.difference(got).take(4).collect();
        let extra: Vec<_> = got.difference(want).take(4).collect();
        bad.push(format!(
            "{what}: {} tensors expected, release has {}; we have and it lacks {missing:?}; it has and we lack {extra:?}",
            want.len(),
            got.len()
        ));
    }
}

/// Divergences that are understood, each with the text every one of its failure
/// lines must contain. The test fails on anything else -- and on a listed
/// divergence that no longer occurs, so this list cannot go stale. See FINDINGS,
/// "Definitions against real checkpoints".
const KNOWN: [(&str, &[&str], &str); 5] = [
    // The release converted the source to BF16 first; df11pack refuses non-BF16.
    ("hidream-i1-diffusers", &["source not BF16"], "F16 source"),
    ("krea2-comfyui", &["source not BF16"], "5 F32 tensors"),
    (
        "omnigen2-transformer-diffusers",
        &["source not BF16"],
        "F32 source",
    ),
    ("wan-diffusers", &["source not BF16"], "F32 source"),
    // Its tied lm_head is now reproduced; only the dtype remains.
    ("omnigen2-mllm", &["source not BF16"], "F32 source"),
];

#[test]
fn definitions_fit_real_checkpoints() {
    let Some(root) = df11_fixtures::workspace_root() else {
        return;
    };
    let dir = root.join("phase0/fixtures/real_headers");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("SKIP: run phase0/fetch_real_headers.py");
        return;
    };
    let defs: BTreeMap<String, ArchDef> = architecture_defs()
        .unwrap()
        .into_iter()
        .map(|(n, t)| (n, ArchDef::from_toml(&t).unwrap()))
        .collect();
    let mut report = Vec::new();
    let mut failed = 0;
    let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        let fx: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        let name = fx["definition"].as_str().unwrap().to_string();
        let bad = check(&name, &fx, &defs[&name]);
        let known = KNOWN.iter().find(|(n, _, _)| *n == name);
        match (bad.is_empty(), known) {
            (true, None) => report.push(format!("  ok     {name}")),
            (true, Some(_)) => {
                failed += 1;
                report.push(format!(
                    "  FIXED? {name}: listed in KNOWN but now fits; remove it"
                ));
            }
            (false, Some((_, marks, why)))
                if bad.iter().all(|b| marks.iter().any(|m| b.contains(m)))
                    && marks.iter().all(|m| bad.iter().any(|b| b.contains(m))) =>
            {
                report.push(format!("  known  {name}: {why}"));
            }
            (false, _) => {
                failed += 1;
                report.push(format!("  FAIL   {name}"));
                report.extend(bad.iter().take(6).map(|b| format!("          {b}")));
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert_eq!(failed, 0, "\n{}", report.join("\n"));
}
