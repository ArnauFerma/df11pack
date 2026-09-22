//! Phase 2 -- reading single-file and sharded models.

use df11_codec::safetensors::{write_file, Dtype, OutTensor, SafeTensorsFile};
use df11_codec::source::ModelSource;
use df11_fixtures::skip_if_missing;
use std::collections::BTreeMap;

fn dir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("df11pack_src_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn t(name: &str, byte: u8, n: usize) -> OutTensor {
    OutTensor {
        name: name.into(),
        dtype: Dtype::new(Dtype::U8),
        shape: vec![n as u64],
        data: vec![byte; n],
    }
}

#[test]
fn opens_a_single_file() {
    let d = dir("single");
    let p = d.join("model.safetensors");
    write_file(&p, &[t("a", 1, 4), t("b", 2, 8)], &BTreeMap::new()).unwrap();
    let m = ModelSource::open(&p).expect("opens a file");
    assert_eq!(m.shards(), 1);
    assert_eq!(m.names(), vec!["a", "b"]);
    assert_eq!(m.read("b").unwrap(), vec![2u8; 8]);
    assert_eq!(m.total_bytes(), 12);
    assert_eq!(m.dir(), d.as_path());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn opens_a_directory_of_shards_and_unifies_them() {
    let d = dir("sharded");
    write_file(
        d.join("part-1.safetensors"),
        &[t("a", 1, 4)],
        &BTreeMap::new(),
    )
    .unwrap();
    write_file(
        d.join("part-2.safetensors"),
        &[t("b", 2, 8)],
        &BTreeMap::new(),
    )
    .unwrap();
    let m = ModelSource::open(&d).expect("opens a directory");
    assert_eq!(m.shards(), 2);
    assert_eq!(m.names(), vec!["a", "b"], "names unify across shards");
    assert_eq!(m.read("a").unwrap(), vec![1u8; 4]);
    assert_eq!(m.read("b").unwrap(), vec![2u8; 8]);
    assert_eq!(m.total_bytes(), 12);
    let _ = std::fs::remove_dir_all(&d);
}

/// Sharding is a property of the input only: the same tensors must come back
/// identically however they were split.
#[test]
fn a_sharded_source_reads_identically_to_a_single_file() {
    let Some(fx) = skip_if_missing("a_sharded_source_reads_identically_to_a_single_file") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let whole = set.source_dir.join("model.safetensors");
    let single = SafeTensorsFile::open(&whole).expect("open source");
    let names: Vec<String> = single.names().map(String::from).collect();

    // Split the real model across three shards, round-robin.
    let d = dir("split");
    for part in 0..3usize {
        let tensors: Vec<OutTensor> = names
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == part)
            .map(|(_, n)| {
                let info = single.info(n).unwrap();
                OutTensor {
                    name: n.clone(),
                    dtype: info.dtype.clone(),
                    shape: info.shape.clone(),
                    data: single.read(n).unwrap(),
                }
            })
            .collect();
        write_file(
            d.join(format!("model-{}-of-3.safetensors", part + 1)),
            &tensors,
            &BTreeMap::new(),
        )
        .unwrap();
    }

    let m = ModelSource::open(&d).expect("opens the split model");
    assert_eq!(m.shards(), 3);
    assert_eq!(m.len(), names.len(), "no tensor lost in the split");
    assert_eq!(m.names(), names);
    for n in &names {
        assert_eq!(
            m.read(n).unwrap(),
            single.read(n).unwrap(),
            "{n}: bytes must not depend on which shard it landed in"
        );
        let a = m.info(n).unwrap();
        let b = single.info(n).unwrap();
        assert_eq!(
            (a.dtype.clone(), a.shape.clone()),
            (b.dtype.clone(), b.shape.clone())
        );
    }
    assert_eq!(
        m.total_bytes(),
        single
            .names()
            .filter_map(|n| single.info(n))
            .map(|i| i.nbytes())
            .sum::<u64>()
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_tensor_declared_by_two_shards_is_an_error_not_a_coin_flip() {
    let d = dir("dup");
    write_file(
        d.join("x.safetensors"),
        &[t("same", 1, 4)],
        &BTreeMap::new(),
    )
    .unwrap();
    write_file(
        d.join("y.safetensors"),
        &[t("same", 9, 4)],
        &BTreeMap::new(),
    )
    .unwrap();
    assert!(
        ModelSource::open(&d).is_err(),
        "two shards claiming one tensor must be refused, not silently resolved"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_directory_with_no_safetensors_is_an_error() {
    let d = dir("empty");
    std::fs::write(d.join("readme.txt"), b"nothing here").unwrap();
    assert!(ModelSource::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}
