//! Step 1.8 -- architecture definitions.

use df11_codec::arch::{ArchDef, ArchError, Layout};
use df11_codec::{split_fields, Histogram};
use df11_fixtures::{architecture_defs, official_pattern_dicts, skip_if_missing, SourceModel};

const QWEN3_LAYER: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[test]
fn every_shipped_definition_parses_and_validates() {
    let Some(defs) = architecture_defs() else {
        eprintln!("SKIP: data/architectures missing");
        return;
    };
    assert!(
        defs.len() >= 8,
        "expected the shipped set, got {}",
        defs.len()
    );
    for (name, src) in &defs {
        let d = ArchDef::from_toml(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(&d.name, name, "name must match the file stem");
        assert!(!d.source.trim().is_empty(), "{name}: needs provenance");
        assert!(!d.units.is_empty());
    }
}

/// Drift guard. These files are generated from verified official releases, so
/// this does not validate the original transcription -- it catches a later
/// hand-edit that silently changes an order.
#[test]
fn definitions_still_match_the_official_pattern_dicts() {
    let (Some(defs), Some(official)) = (architecture_defs(), official_pattern_dicts()) else {
        eprintln!("SKIP: fixtures missing");
        return;
    };
    let mut checked = 0;
    for (name, src) in &defs {
        let Some(o) = official.get(name) else {
            panic!("{name}: no recorded official pattern_dict")
        };
        let d = ArchDef::from_toml(src).expect("parses");
        let ours = d.pattern_dict();

        assert_eq!(
            ours.len(),
            o.pattern_dict.len(),
            "{name}: unit count differs from the official release"
        );
        for (i, (pat, attrs)) in o.pattern_dict.iter().enumerate() {
            assert_eq!(&ours[i].0, pat, "{name}: unit {i} pattern differs");
            let want: Vec<String> = attrs
                .as_array()
                .expect("attrs array")
                .iter()
                .map(|v| v.as_str().expect("attr string").to_owned())
                .collect();
            assert_eq!(
                ours[i].1, want,
                "{name}: attribute ORDER differs for {pat} -- this fixes split_positions \
                 and therefore every compressed byte"
            );
        }
        assert_eq!(d.bytes_per_thread, o.bytes_per_thread, "{name}");
        assert_eq!(d.threads_per_block, o.threads_per_block, "{name}");
        checked += 1;
    }
    assert!(checked >= 8);
}

#[test]
fn standalone_units_are_recognised() {
    let (Some(defs), _) = (architecture_defs(), ()) else {
        return;
    };
    let (_, src) = defs
        .iter()
        .find(|(n, _)| n == "qwen3-8b")
        .expect("qwen3-8b");
    let d = ArchDef::from_toml(src).expect("parses");
    let standalone: Vec<_> = d.units.iter().filter(|u| u.is_standalone()).collect();
    assert_eq!(
        standalone.len(),
        2,
        "lm_head and model.embed_tokens are single-tensor units"
    );
    assert_eq!(d.layout, Layout::Transformers);
}

#[test]
fn validation_rejects_definitions_that_would_corrupt_output() {
    let base = r#"
name = "t"
layout = "diffusers"
format_version = "0.5.0"
threads_per_block = [512]
bytes_per_thread = 8
source = "test"
[[unit]]
pattern = "a"
attrs = ["x", "y"]
"#;
    assert!(ArchDef::from_toml(base).is_ok(), "the base must be valid");

    let dup_attr = base.replace(r#"attrs = ["x", "y"]"#, r#"attrs = ["x", "x"]"#);
    assert_eq!(
        ArchDef::from_toml(&dup_attr).unwrap_err(),
        ArchError::DuplicateAttr {
            pattern: "a".into(),
            attr: "x".into()
        }
    );

    let dup_pat = format!("{base}\n[[unit]]\npattern = \"a\"\nattrs = [\"z\"]\n");
    assert_eq!(
        ArchDef::from_toml(&dup_pat).unwrap_err(),
        ArchError::DuplicatePattern("a".into())
    );

    let no_source = base.replace(r#"source = "test""#, r#"source = """#);
    assert_eq!(
        ArchDef::from_toml(&no_source).unwrap_err(),
        ArchError::Empty("source")
    );

    let no_units = base.split("[[unit]]").next().unwrap().to_string();
    assert_eq!(
        ArchDef::from_toml(&no_units).unwrap_err(),
        ArchError::NoUnits
    );

    let zero_bpt = base.replace("bytes_per_thread = 8", "bytes_per_thread = 0");
    assert_eq!(
        ArchDef::from_toml(&zero_bpt).unwrap_err(),
        ArchError::BadNumber("bytes_per_thread")
    );
}

#[test]
fn split_positions_layout_matches_the_format() {
    assert_eq!(ArchDef::split_positions(&[]), Vec::<i64>::new());
    assert_eq!(
        ArchDef::split_positions(&[100]),
        Vec::<i64>::new(),
        "a single-tensor unit has empty split_positions"
    );
    assert_eq!(ArchDef::split_positions(&[10, 20, 30]), vec![10, 30]);
}

/// End-to-end: the definition, applied to the real source weights, must
/// reproduce the official `split_positions` exactly. This is what proves the
/// attribute order is right, independently of the drift guard above.
#[test]
fn split_positions_reproduce_official_output() {
    let Some(fx) = skip_if_missing("split_positions_reproduce_official_output") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let Some(src) = SourceModel::open(set) else {
        return;
    };

    let mut checked = 0;
    for unit in set.unit_names() {
        let counts: Vec<u64> = QWEN3_LAYER
            .iter()
            .map(|a| {
                let b = src
                    .tensor(&format!("{unit}.{a}.weight"))
                    .unwrap_or_else(|e| panic!("{unit}.{a}: {e}"));
                (b.len() / 2) as u64
            })
            .collect();
        let ours = ArchDef::split_positions(&counts);

        let f = set
            .unit(&unit)
            .into_iter()
            .find(|t| t.name.ends_with("split_positions"))
            .expect("split_positions fixture");
        let bytes = f.read().expect("read");
        let official: Vec<i64> = bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(
            ours, official,
            "{unit}: split_positions disagree -- the attribute order is wrong"
        );
        // And the total, which is NOT stored there, must be the weight count.
        let total: u64 = counts.iter().sum();
        let input = src.unit_input(&unit, &QWEN3_LAYER).expect("input");
        let (exp, _) = split_fields(&input);
        assert_eq!(Histogram::build(&exp).unwrap().total(), total);
        checked += 1;
    }
    assert_eq!(checked, 4);
}
