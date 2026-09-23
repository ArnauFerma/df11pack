//! Phase 2 -- writing a DF11 directory, graded against the official output.

use df11_codec::arch::ArchDef;
use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::source::ModelSource;
use df11_codec::write::{remainder_name, shard_name, write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing, SourceModel};

fn outdir(tag: &str) -> std::path::PathBuf {
    df11_fixtures::scratch(&format!("w_{tag}"))
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

    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");
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
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

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

#[test]
fn a_config_is_written_for_layouts_that_have_one() {
    use df11_codec::arch::Layout;
    let Some(fx) = skip_if_missing("a_config_is_written_for_layouts_that_have_one") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    assert_eq!(def.layout, Layout::Transformers);
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    let out = outdir("cfg");
    let report = write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    assert_eq!(report.config.as_deref(), Some("config.json"));

    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("config.json")).expect("read"))
            .expect("json");
    // Preserve mode: the source's own keys survive, plus ours.
    assert_eq!(written["hidden_size"], serde_json::json!(1024));
    assert_eq!(written["tie_word_embeddings"], serde_json::json!(true));
    assert_eq!(
        written["dfloat11_config"]["bytes_per_thread"],
        serde_json::json!(8)
    );
    // The official Qwen3 pattern leaves its dots unescaped -- `model.layers.\d+`
    // -- while Chroma's escapes them. Publishers are inconsistent, and the
    // definition must round-trip whichever it was given, byte for byte.
    let pd = written["dfloat11_config"]["pattern_dict"]
        .as_object()
        .expect("pattern_dict object");
    assert_eq!(
        pd.keys().collect::<Vec<_>>(),
        vec!["model.layers.\\d+"],
        "the pattern must round-trip exactly as the official release wrote it"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn comfyui_native_output_has_no_config() {
    let Some(fx) = skip_if_missing("comfyui_native_output_has_no_config") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    // Reuse the Qwen pattern but declare the native layout, which is what
    // decides whether a config is emitted.
    let def = ArchDef::from_toml(
        r#"
name = "native-probe"
layout = "comfyui-native"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = 'model\.layers\.\d+'
attrs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
"#,
    )
    .expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");
    let out = outdir("native");
    let report = write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    assert_eq!(report.config, None, "native output carries no config");
    assert!(!out.join("config.json").exists());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_worker_count_follows_the_ram_budget() {
    use df11_codec::write::{bytes_per_weight, worker_count_with};
    let unit = 15_728_640u64; // a real Qwen3 layer unit
    let per_worker = |o: &WriteOptions| (unit as f64 * bytes_per_weight(o)).ceil() as u64;

    // No budget and memory unknown: fall back to the core count.
    let none = WriteOptions::default();
    assert_eq!(worker_count_with(&none, unit, 8, None), 8);

    // A budget for exactly four workers.
    let mut four = WriteOptions::default();
    four.ram_budget = Some(per_worker(&four) * 4);
    assert_eq!(worker_count_with(&four, unit, 8, None), 4);

    // Never more than the cores available.
    assert_eq!(worker_count_with(&four, unit, 2, None), 2);

    // A budget too small for even one unit still makes progress.
    let tiny = WriteOptions {
        ram_budget: Some(1024),
        ..Default::default()
    };
    assert_eq!(
        worker_count_with(&tiny, unit, 8, None),
        1,
        "a budget below one unit must still run, single-threaded"
    );

    // An explicit count overrides the budget.
    let forced = WriteOptions {
        ram_budget: Some(per_worker(&four) * 4),
        workers: Some(7),
        ..Default::default()
    };
    assert_eq!(worker_count_with(&forced, unit, 8, None), 7);
}

/// The regression the review found: with no `--ram`, the default was one worker
/// per core regardless of memory. On the 3.6 GB target machine a Flux-sized unit
/// (~340M weights) at four workers needs several gigabytes and is killed.
#[test]
fn without_a_budget_the_default_respects_available_memory() {
    use df11_codec::write::{bytes_per_weight, worker_count_with, DEFAULT_MEMORY_FRACTION};
    let flux_unit = 340_000_000u64;
    let opts = WriteOptions::default();
    let available = 2_700u64 << 20; // what the target machine actually reports free

    let w = worker_count_with(&opts, flux_unit, 4, Some(available));
    let per = flux_unit as f64 * bytes_per_weight(&opts);
    assert!(
        (w as f64) * per <= available as f64 * DEFAULT_MEMORY_FRACTION || w == 1,
        "{w} workers x {per:.0} bytes exceeds {DEFAULT_MEMORY_FRACTION} of {available} available"
    );
    assert_eq!(
        w, 1,
        "a Flux-sized unit on this machine fits one worker, not four"
    );

    // Plenty of memory: the core count is the limit again.
    assert_eq!(worker_count_with(&opts, 15_728_640, 4, Some(64 << 30)), 4);
}

/// Safe mode holds the unit's source and the decoder's tables on top of the
/// encoder's state, so the same budget must admit fewer workers.
#[test]
fn safe_mode_is_counted_against_the_budget() {
    use df11_codec::write::{bytes_per_weight, worker_count_with};
    let unit = 15_728_640u64;
    let fast = WriteOptions::default();
    let safe = WriteOptions {
        verify: true,
        ..Default::default()
    };
    assert!(
        bytes_per_weight(&safe) > bytes_per_weight(&fast),
        "safe mode must cost more per weight than fast mode"
    );
    let budget = (unit as f64 * bytes_per_weight(&fast) * 8.0) as u64;
    let with = |o: &WriteOptions| {
        let mut o = o.clone();
        o.ram_budget = Some(budget);
        worker_count_with(&o, unit, 64, None)
    };
    assert_eq!(with(&fast), 8);
    assert!(
        with(&safe) < 8,
        "the budget that fits 8 fast workers must fit fewer safe ones, got {}",
        with(&safe)
    );
}

/// The field that is read, pinned on fixed input. Every other field here has a
/// plausible size, so only exact values distinguish them -- verified: reading
/// `SwapFree` instead passed a plausibility-only test.
#[test]
fn mem_available_is_the_field_that_is_read() {
    use df11_codec::write::parse_mem_available;
    let meminfo = "MemTotal:        3698176 kB\n\
                   MemFree:          339968 kB\n\
                   MemAvailable:    2780160 kB\n\
                   Buffers:          102400 kB\n\
                   SwapTotal:       4194300 kB\n\
                   SwapFree:        4000000 kB\n";
    assert_eq!(parse_mem_available(meminfo), Some(2_780_160 * 1024));
    assert_eq!(
        parse_mem_available("MemTotal: 1 kB\n"),
        None,
        "absent means unknown"
    );
    assert_eq!(parse_mem_available("MemAvailable: lots kB\n"), None);
}

#[test]
fn available_memory_is_read_on_linux() {
    use df11_codec::write::available_memory;
    if std::path::Path::new("/proc/meminfo").exists() {
        let m = available_memory().expect("MemAvailable must be readable on Linux");
        assert!(m > 64 << 20, "implausibly little memory reported: {m}");
        assert!(m < 1 << 50, "implausibly much memory reported: {m}");
    }
}

/// Parallelism must not change a byte. Compress the same model with one worker
/// and with many, and require identical files.
#[test]
fn worker_count_never_changes_the_output() {
    let Some(fx) = skip_if_missing("worker_count_never_changes_the_output") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    let a = outdir("w1");
    let b = outdir("w4");
    write_directory(
        &src,
        &def,
        &a,
        &WriteOptions {
            workers: Some(1),
            ..Default::default()
        },
    )
    .expect("one worker");
    write_directory(
        &src,
        &def,
        &b,
        &WriteOptions {
            workers: Some(4),
            ..Default::default()
        },
    )
    .expect("four workers");

    let mut n = 0;
    for e in std::fs::read_dir(&a).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let other = b.join(&name);
        assert!(other.exists(), "{name} missing from the 4-worker run");
        assert_eq!(
            std::fs::read(&p).unwrap(),
            std::fs::read(&other).unwrap(),
            "{name}: differs between 1 and 4 workers"
        );
        n += 1;
    }
    assert_eq!(n, 6, "4 shards + remainder + config");
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}

#[test]
fn the_io_plan_is_reported_and_obeyed() {
    use df11_codec::io_sched::IoMode;
    let Some(fx) = skip_if_missing("the_io_plan_is_reported_and_obeyed") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    for (mode, want) in [(IoMode::Sequential, true), (IoMode::Concurrent, false)] {
        let out = outdir(&format!("io{want}"));
        let r = write_directory(
            &src,
            &def,
            &out,
            &WriteOptions {
                io: mode,
                workers: Some(3),
                ..Default::default()
            },
        )
        .expect("writes");
        assert_eq!(r.io.sequential, want, "{mode:?} must be obeyed");
        if want {
            // Not merely declared: with three workers reading seven tensors each,
            // a missing gate shows up here.
            assert_eq!(
                r.max_concurrent_reads, 1,
                "sequential mode must actually serialise reads, not just say so"
            );
        }
        assert!(!r.io.reason.is_empty(), "the plan must say why");
        assert_eq!(r.units, 4, "{mode:?}: output must be unaffected");
        let _ = std::fs::remove_dir_all(&out);
    }
}

/// The Phase 3 exit gate's memory case: a 512 MiB budget must still compress a
/// model whose units do not all fit, by running fewer workers.
#[test]
fn a_512_mib_budget_still_compresses_and_bounds_workers() {
    use df11_codec::write::{bytes_per_weight, worker_count_with};
    let Some(fx) = skip_if_missing("a_512_mib_budget_still_compresses_and_bounds_workers") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    let budget = 512u64 << 20;
    let out = outdir("ram512");
    let r = write_directory(
        &src,
        &def,
        &out,
        &WriteOptions {
            ram_budget: Some(budget),
            ..Default::default()
        },
    )
    .expect("a 512 MiB budget must not prevent compression");
    assert_eq!(r.units, 4);

    // The unit is 15,728,640 weights; the budget admits a bounded number.
    let unit = 15_728_640u64;
    let opts512 = WriteOptions {
        ram_budget: Some(budget),
        ..Default::default()
    };
    let allowed = (budget as f64 / (unit as f64 * bytes_per_weight(&opts512))).floor() as usize;
    assert!(allowed >= 1, "512 MiB must admit at least one worker");
    assert_eq!(worker_count_with(&opts512, unit, 64, None), allowed.min(64));

    // And a budget far below one unit must still make progress, single-threaded.
    let tiny = WriteOptions {
        ram_budget: Some(1 << 20),
        ..Default::default()
    };
    assert_eq!(worker_count_with(&tiny, unit, 64, None), 1);
    let out2 = outdir("ram1m");
    let r2 = write_directory(&src, &def, &out2, &tiny).expect("a 1 MiB budget must still run");
    assert_eq!(r2.units, 4);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&out2);
}

#[test]
fn safe_mode_verifies_every_unit_before_writing_it() {
    let Some(fx) = skip_if_missing("safe_mode_verifies_every_unit_before_writing_it") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).expect("source");

    let out = outdir("safe");
    let r = write_directory(
        &src,
        &def,
        &out,
        &WriteOptions {
            verify: true,
            ..Default::default()
        },
    )
    .expect("safe mode must accept correct output");
    assert_eq!(r.verified.len(), 4, "every unit must be checked");
    assert_eq!(r.units, 4);

    // And without it, nothing is claimed to be verified.
    let out2 = outdir("fast");
    let r2 = write_directory(&src, &def, &out2, &WriteOptions::default()).expect("fast mode");
    assert!(
        r2.verified.is_empty(),
        "fast mode must not claim verification it did not do"
    );

    // Both modes must produce identical files: verification observes, it does
    // not change what is written.
    for e in std::fs::read_dir(&out).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap();
        assert_eq!(
            std::fs::read(&p).unwrap(),
            std::fs::read(out2.join(n)).unwrap(),
            "{n:?}: safe mode changed the output"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&out2);
}

/// generation_config.json: carried over when the source ships one, never invented.
///
/// The official output's copy is synthesised by transformers from config.json
/// (`"_from_model_config": true`) and stamped with the installed version -- the
/// same version trap as config.json, and regenerated by transformers from the
/// config when absent. So df11pack copies a real one and does not fabricate one.
#[test]
fn generation_config_is_copied_when_present_and_never_invented() {
    let Some(fx) = skip_if_missing("generation_config_is_copied_when_present_and_never_invented")
    else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(defs) = architecture_defs() else {
        return;
    };
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").expect("def");
    let def = ArchDef::from_toml(toml).expect("parses");

    // A source directory WITH a generation config.
    let srcdir = outdir("gen_src");
    std::fs::copy(
        set.source_dir.join("model.safetensors"),
        srcdir.join("model.safetensors"),
    )
    .unwrap();
    std::fs::copy(
        set.source_dir.join("config.json"),
        srcdir.join("config.json"),
    )
    .unwrap();
    let gen = br#"{"do_sample": true, "temperature": 0.6, "top_p": 0.95}"#;
    std::fs::write(srcdir.join("generation_config.json"), gen).unwrap();

    let out = outdir("gen_out");
    let src = ModelSource::open(srcdir.join("model.safetensors")).unwrap();
    write_directory(&src, &def, &out, &WriteOptions::default()).expect("writes");
    assert_eq!(
        std::fs::read(out.join("generation_config.json")).expect("must be copied"),
        gen,
        "copied byte for byte, not re-serialised"
    );

    // WITHOUT one: none is invented.
    let out2 = outdir("gen_none");
    let src2 = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();
    write_directory(&src2, &def, &out2, &WriteOptions::default()).expect("writes");
    assert!(
        !out2.join("generation_config.json").exists(),
        "a generation config must not be fabricated when the source has none"
    );
    for d in [srcdir, out, out2] {
        let _ = std::fs::remove_dir_all(d);
    }
}
