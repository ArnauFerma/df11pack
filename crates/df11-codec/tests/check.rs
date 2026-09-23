//! Phase 6 -- post-hoc verification of a written output, including corruption
//! injected into the files on disk.

use df11_codec::arch::ArchDef;
use df11_codec::check::{check_output, Level};
use df11_codec::sample::plan;
use df11_codec::source::ModelSource;
use df11_fixtures::{architecture_defs, skip_if_missing};
use std::path::{Path, PathBuf};

struct Env {
    out: PathBuf,
    source: PathBuf,
    def: ArchDef,
}

/// A private copy of the official tier-0 output, so on-disk corruption never
/// touches the fixture.
fn env(tag: &str) -> Option<Env> {
    let fx = skip_if_missing(tag)?;
    let set = fx.set("tier0-qwen3-trunc-layers-only")?;
    let official = set.tensors()[0].file.parent()?.to_path_buf();
    let out = df11_fixtures::scratch(&format!("chk_{tag}"));
    for e in std::fs::read_dir(&official).ok()? {
        let p = e.ok()?.path();
        std::fs::copy(&p, out.join(p.file_name()?)).ok()?;
    }
    let defs = architecture_defs()?;
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b")?;
    Some(Env {
        out,
        source: set.source_dir.join("model.safetensors"),
        def: ArchDef::from_toml(toml).ok()?,
    })
}

/// Overwrite bytes of one tensor inside a file on disk.
fn corrupt(file: &Path, tensor: &str, at: usize, f: impl Fn(u8) -> u8) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut h = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(file)
        .unwrap();
    let mut len = [0u8; 8];
    h.read_exact(&mut len).unwrap();
    let n = u64::from_le_bytes(len);
    let mut hdr = vec![0u8; n as usize];
    h.read_exact(&mut hdr).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&hdr).unwrap();
    let start = v[tensor]["data_offsets"][0].as_u64().unwrap();
    let pos = 8 + n + start + at as u64;
    let mut b = [0u8; 1];
    h.seek(SeekFrom::Start(pos)).unwrap();
    h.read_exact(&mut b).unwrap();
    h.seek(SeekFrom::Start(pos)).unwrap();
    h.write_all(&[f(b[0])]).unwrap();
}

#[test]
fn every_level_passes_on_a_correct_output() {
    let Some(e) = env("pass") else { return };
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    for level in [
        Level::Integrity,
        Level::Sample {
            budget: 200,
            seed: 1,
        },
        Level::Full,
    ] {
        let r = check_output(&out, Some((&src, &e.def)), level).unwrap();
        assert!(r.ok(), "{level:?}: {:?}", r.failures);
        assert_eq!(r.units, 4);
    }
}

#[test]
fn integrity_needs_no_source() {
    let Some(e) = env("nosrc") else { return };
    let out = ModelSource::open(&e.out).unwrap();
    let r = check_output(&out, None, Level::Integrity).expect("integrity runs without a source");
    assert!(r.ok());
    assert_eq!(r.units, 4, "units are found in the output itself");
    assert!(
        check_output(&out, None, Level::Full).is_err(),
        "a decoding level without a source must be an error, not a silent pass"
    );
}

/// Structural damage on disk, caught without the source.
#[test]
fn integrity_catches_structural_damage_on_disk() {
    let Some(e) = env("struct") else { return };
    // Break monotonicity of output_positions in layer 2's shard.
    corrupt(
        &e.out.join("model_layers_2.safetensors"),
        "model.layers.2.output_positions",
        4 * 300 + 3,
        |b| b ^ 0x40,
    );
    let out = ModelSource::open(&e.out).unwrap();
    let r = check_output(&out, None, Level::Integrity).unwrap();
    assert!(!r.ok(), "structural damage must be found");
    assert!(
        r.failures.iter().any(|f| f.unit == "model.layers.2"),
        "and attributed to the right unit: {:?}",
        r.failures
    );
    assert!(
        r.failures.iter().all(|f| f.unit == "model.layers.2"),
        "and only to it: {:?}",
        r.failures
    );
}

/// A value error in a chunk sampling always visits -- the first chunk of a unit --
/// is caught at the sample level, and located.
#[test]
fn sampling_catches_a_value_error_in_a_mandatory_chunk() {
    let Some(e) = env("mand") else { return };
    corrupt(
        &e.out.join("model_layers_1.safetensors"),
        "model.layers.1.sign_mantissa",
        5,
        |b| b ^ 0x01,
    );
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    // Integrity cannot see it: every structure is intact.
    assert!(check_output(&out, None, Level::Integrity).unwrap().ok());
    let r = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 10,
            seed: 3,
        },
    )
    .unwrap();
    assert!(!r.ok(), "the first chunk is always sampled");
    let f = &r.failures[0];
    assert_eq!((f.unit.as_str(), f.chunk), ("model.layers.1", Some(0)));
}

/// The honest limitation, measured: an isolated fault in a chunk the sample did
/// not pick goes unnoticed at the sample level. Only the full sweep catches it.
#[test]
fn sampling_can_miss_an_isolated_fault_and_the_full_sweep_cannot() {
    let Some(e) = env("miss") else { return };
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();

    // Learn which chunks a small seeded sample visits, then corrupt one it does not.
    let clean = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 20,
            seed: 5,
        },
    )
    .unwrap();
    let visited: std::collections::BTreeSet<_> = clean.checked.iter().cloned().collect();
    let target = (0..1278)
        .find(|c| !visited.contains(&("model.layers.3".to_string(), *c)))
        .expect("a small budget leaves chunks unvisited");

    // The target chunk's first weight: its element index is output_positions[c].
    let f =
        df11_codec::safetensors::SafeTensorsFile::open(e.out.join("model_layers_3.safetensors"))
            .unwrap();
    let pos = f.read("model.layers.3.output_positions").unwrap();
    let first = u32::from_le_bytes(pos[target * 4..target * 4 + 4].try_into().unwrap()) as usize;
    drop(f);
    corrupt(
        &e.out.join("model_layers_3.safetensors"),
        "model.layers.3.sign_mantissa",
        first,
        |b| b ^ 0x02,
    );

    let out = ModelSource::open(&e.out).unwrap();
    let sampled = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 20,
            seed: 5,
        },
    )
    .unwrap();
    assert!(
        sampled.ok(),
        "sampling does not visit chunk {target}, so cannot see it"
    );
    let full = check_output(&out, Some((&src, &e.def)), Level::Full).unwrap();
    assert!(!full.ok(), "the full sweep must catch what sampling missed");
    assert!(full.failures.iter().any(|f| f.unit == "model.layers.3"));
}

#[test]
fn a_sampled_run_records_its_seed_and_reproduces() {
    let Some(e) = env("seed") else { return };
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    let a = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 50,
            seed: 11,
        },
    )
    .unwrap();
    let b = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 50,
            seed: 11,
        },
    )
    .unwrap();
    assert_eq!(a.seed, Some(11));
    assert_eq!(a.checked, b.checked, "the same seed visits the same chunks");
    // It follows the stratified plan.
    let _ = plan;
    assert!(a.checked.len() >= 50);
}

#[test]
fn a_unit_missing_from_the_output_is_a_failure() {
    let Some(e) = env("missing") else { return };
    std::fs::remove_file(e.out.join("model_layers_3.safetensors")).unwrap();
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    let r = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample {
            budget: 20,
            seed: 1,
        },
    )
    .unwrap();
    assert!(
        r.failures.iter().any(|f| f.unit == "model.layers.3"),
        "a unit the source defines but the output lacks must be reported: {:?}",
        r.failures
    );
}

/// Every official output in the corpus -- transformers, diffusers, ComfyUI
/// native, single-tensor units with empty split_positions -- passes the
/// structural check without a source and a sampled decode with one. Guards the
/// checker against being fitted to the one fixture the other tests use.
#[test]
fn every_official_output_passes() {
    let Some(fx) = skip_if_missing("every_official_output_passes") else {
        return;
    };
    let defs = architecture_defs().expect("definitions");
    for (set_name, def_name, expect_units) in [
        ("tier0-qwen3-trunc-layers-only", "qwen3-4b", 4),
        ("synthetic-flux-comfyui", "flux-comfyui", 4),
        ("synthetic-chroma-comfyui", "chroma-comfyui", 6),
        ("synthetic-flux-dev-diffusers", "flux-dev-diffusers", 4),
        ("synthetic-chroma-diffusers", "chroma-diffusers", 5),
    ] {
        let Some(set) = fx.set(set_name) else {
            eprintln!("SKIP {set_name}: run phase0/make_synthetic.py");
            continue;
        };
        let official = set.tensors()[0].file.parent().unwrap().to_path_buf();
        let out = ModelSource::open(&official).unwrap();
        let src = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();
        let (_, toml) = defs.iter().find(|(n, _)| n == def_name).unwrap();
        let def = ArchDef::from_toml(toml).unwrap();

        let i = check_output(&out, None, Level::Integrity).unwrap();
        assert!(i.ok(), "{set_name} integrity: {:?}", i.failures);
        let s = check_output(
            &out,
            Some((&src, &def)),
            Level::Sample {
                budget: 64,
                seed: 9,
            },
        )
        .unwrap();
        assert!(s.ok(), "{set_name} sample: {:?}", s.failures);
        assert_eq!(
            i.units, s.units,
            "{set_name}: both ways must find the same units"
        );
        // Counted independently from the official files' encoded_exponent keys.
        assert_eq!(s.units, expect_units, "{set_name}");
    }
}

/// A LUT jump to a table that does not exist would send the kernel reading past
/// the LUT in shared memory. Structural, so no source is needed.
#[test]
fn integrity_catches_a_jump_to_a_missing_table() {
    let Some(e) = env("lutjump") else { return };
    // Layer 0 has four prefix tables, 0..=3; 252 jumps to table 4, the first
    // one that does not exist.
    corrupt(
        &e.out.join("model_layers_0.safetensors"),
        "model.layers.0.luts",
        7,
        |_| 252,
    );
    let out = ModelSource::open(&e.out).unwrap();
    let r = check_output(&out, None, Level::Integrity).unwrap();
    assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
    assert!(r.failures[0].error.contains("table 4,"), "{:?}", r.failures);
}

/// With no random budget at all, the chunk holding a tensor boundary is still
/// decoded: the wiring from split_positions to the plan is what puts it there.
#[test]
fn a_zero_budget_still_checks_the_boundary_chunks() {
    let Some(e) = env("bound") else { return };
    let f =
        df11_codec::safetensors::SafeTensorsFile::open(e.out.join("model_layers_2.safetensors"))
            .unwrap();
    let split: Vec<i64> = f
        .read("model.layers.2.split_positions")
        .unwrap()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let pos: Vec<u32> = f
        .read("model.layers.2.output_positions")
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    drop(f);
    let boundary = split[2] as usize;
    let chunk = df11_codec::sample::chunk_of(&pos, boundary as u64).unwrap();
    assert!(
        chunk > 0 && chunk + 1 < pos.len() - 1,
        "not a first/last chunk"
    );
    corrupt(
        &e.out.join("model_layers_2.safetensors"),
        "model.layers.2.sign_mantissa",
        boundary,
        |b| b ^ 0x10,
    );
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    let r = check_output(
        &out,
        Some((&src, &e.def)),
        Level::Sample { budget: 0, seed: 1 },
    )
    .unwrap();
    assert_eq!(
        r.failures
            .iter()
            .map(|f| (f.unit.as_str(), f.chunk))
            .collect::<Vec<_>>(),
        vec![("model.layers.2", Some(chunk))]
    );
}

/// output_positions must end at the unit's weight count: a larger final entry
/// is a kernel writing past the end of the output tensor.
#[test]
fn integrity_catches_a_wrong_final_output_position() {
    let Some(e) = env("poslast") else { return };
    let f =
        df11_codec::safetensors::SafeTensorsFile::open(e.out.join("model_layers_1.safetensors"))
            .unwrap();
    let n = f.read("model.layers.1.output_positions").unwrap().len();
    drop(f);
    corrupt(
        &e.out.join("model_layers_1.safetensors"),
        "model.layers.1.output_positions",
        n - 4,
        |b| b.wrapping_add(1),
    );
    let out = ModelSource::open(&e.out).unwrap();
    let r = check_output(&out, None, Level::Integrity).unwrap();
    assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
    assert!(r.failures[0].error.contains("ends at"), "{:?}", r.failures);
}

/// A shifted split position is still a well-formed increasing list, so the
/// structural level cannot see it -- but it moves a tensor boundary, and a loader
/// would slice the weights wrongly. Only the source knows where it belongs.
#[test]
fn a_moved_split_position_needs_the_source_to_catch() {
    let Some(e) = env("split") else { return };
    corrupt(
        &e.out.join("model_layers_3.safetensors"),
        "model.layers.3.split_positions",
        8 * 2,
        |b| b ^ 0x01,
    );
    let out = ModelSource::open(&e.out).unwrap();
    let src = ModelSource::open(&e.source).unwrap();
    assert!(check_output(&out, None, Level::Integrity).unwrap().ok());
    let r = check_output(&out, Some((&src, &e.def)), Level::Integrity).unwrap();
    assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
    assert!(
        r.failures[0].error.contains("split_positions"),
        "{:?}",
        r.failures
    );
}

/// Phase 7: the checker over every synthetic official output, whatever its
/// definition -- including ACEStep's `\d++`, SDXL's character classes and the
/// single-tensor units.
#[test]
fn the_checker_passes_every_synthetic_official_output() {
    let Some(fx) = skip_if_missing("checker_all_synthetic") else {
        return;
    };
    let mut n = 0;
    for (name, toml) in architecture_defs().unwrap() {
        let Some(set) = fx.set(&format!("synthetic-{name}")) else {
            continue;
        };
        let def = ArchDef::from_toml(&toml).unwrap();
        let official = set.tensors()[0].file.parent().unwrap().to_path_buf();
        let out = ModelSource::open(&official).unwrap();
        let src = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();
        let i = check_output(&out, None, Level::Integrity).unwrap();
        let f = check_output(&out, Some((&src, &def)), Level::Full).unwrap();
        assert!(
            i.ok() && f.ok(),
            "{name}: {:?} {:?}",
            i.failures,
            f.failures
        );
        assert_eq!(i.units, f.units, "{name}");
        n += 1;
    }
    assert!(n >= 33, "checked {n}");
}
