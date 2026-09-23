//! Phase 2 -- the safetensors reader and writer.

use df11_codec::safetensors::{write_file, Dtype, OutTensor, Payload, SafeTensorsFile, StError};
use df11_fixtures::skip_if_missing;
use std::collections::BTreeMap;

fn tmp(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("df11pack_test_{}_{}", std::process::id(), name));
    p
}

#[test]
fn round_trips_tensors_and_metadata() {
    let path = tmp("roundtrip.safetensors");
    let tensors = vec![
        OutTensor {
            name: "b.second".into(),
            dtype: Dtype::new(Dtype::U8),
            shape: vec![4],
            data: Payload::Owned(vec![9, 8, 7, 6]),
        },
        OutTensor::owned(
            "a.first",
            Dtype::new(Dtype::I64),
            vec![2],
            (1i64..=2).flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ];
    let mut meta = BTreeMap::new();
    meta.insert("df11pack_luts".to_string(), "correct".to_string());
    write_file(&path, &tensors, &meta).expect("write");

    let f = SafeTensorsFile::open(&path).expect("open");
    assert_eq!(f.len(), 2);
    assert_eq!(
        f.metadata().get("df11pack_luts").map(String::as_str),
        Some("correct")
    );
    assert_eq!(f.read("b.second").unwrap(), vec![9, 8, 7, 6]);
    assert_eq!(
        f.read("a.first").unwrap(),
        (1i64..=2)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    );
    // Written order is preserved in the header, independent of sorted names.
    assert_eq!(
        f.physical_order(),
        &["b.second".to_string(), "a.first".to_string()]
    );
    assert_eq!(f.names().collect::<Vec<_>>(), vec!["a.first", "b.second"]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_data_section_starts_eight_byte_aligned() {
    let path = tmp("align.safetensors");
    // A name chosen so the JSON length is very unlikely to be a multiple of 8.
    let tensors = vec![OutTensor {
        name: "x".into(),
        dtype: Dtype::new(Dtype::U8),
        shape: vec![3],
        data: Payload::Owned(vec![1, 2, 3]),
    }];
    write_file(&path, &tensors, &BTreeMap::new()).expect("write");
    let raw = std::fs::read(&path).expect("read back");
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap());
    assert_eq!(
        (8 + n) % 8,
        0,
        "data must start 8-byte aligned, got {}",
        8 + n
    );
    assert_eq!(&raw[(8 + n) as usize..], &[1u8, 2, 3]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tensors_are_contiguous_with_no_holes() {
    let path = tmp("contig.safetensors");
    let tensors: Vec<OutTensor> = (0..5u8)
        .map(|i| {
            OutTensor::owned(
                format!("t{i}"),
                Dtype::new(Dtype::U8),
                vec![(i as u64) + 1],
                vec![i; (i as usize) + 1],
            )
        })
        .collect();
    write_file(&path, &tensors, &BTreeMap::new()).expect("write");
    let f = SafeTensorsFile::open(&path).expect("open");
    let mut expect = 0u64;
    for i in 0..5u8 {
        let info = f.info(&format!("t{i}")).expect("info");
        assert_eq!(
            info.offsets.0,
            expect,
            "t{i} must start where t{} ended",
            i - 1
        );
        expect = info.offsets.1;
        assert!(
            info.is_consistent(),
            "t{i}: shape and dtype must explain the byte count"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_missing_tensor_is_an_error_not_a_panic() {
    let path = tmp("missing.safetensors");
    write_file(
        &path,
        &[OutTensor::owned(
            "present",
            Dtype::new(Dtype::U8),
            vec![1],
            vec![0],
        )],
        &BTreeMap::new(),
    )
    .unwrap();
    let f = SafeTensorsFile::open(&path).unwrap();
    match f.read("absent") {
        Err(StError::NotFound(n)) => assert_eq!(n, "absent"),
        other => panic!("expected NotFound, got {other:?}"),
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_truncated_header_is_rejected() {
    let path = tmp("trunc.safetensors");
    std::fs::write(&path, [255u8, 255, 255, 255, 255, 255, 255, 255]).unwrap();
    assert!(
        SafeTensorsFile::open(&path).is_err(),
        "an absurd header length must not be trusted"
    );
    let _ = std::fs::remove_file(&path);
}

/// The negative case. Without it, `is_consistent` returning a constant `true`
/// passes every other test here -- verified by mutation.
#[test]
fn is_consistent_rejects_a_shape_that_does_not_explain_the_bytes() {
    use df11_codec::safetensors::TensorInfo;

    let good = TensorInfo {
        dtype: Dtype::new(Dtype::BF16),
        shape: vec![4, 8],
        offsets: (0, 64), // 32 elements x 2 bytes
    };
    assert!(good.is_consistent());

    let short = TensorInfo {
        dtype: Dtype::new(Dtype::BF16),
        shape: vec![4, 8],
        offsets: (0, 63),
    };
    assert!(
        !short.is_consistent(),
        "63 bytes cannot hold 32 BF16 values"
    );

    let wrong_dtype = TensorInfo {
        dtype: Dtype::new(Dtype::U8),
        shape: vec![4, 8],
        offsets: (0, 64),
    };
    assert!(
        !wrong_dtype.is_consistent(),
        "32 U8 values are 32 bytes, not 64"
    );

    // A dtype we do not know cannot be checked, so it must not be rejected.
    let unknown = TensorInfo {
        dtype: Dtype::new("SOMETHING_NEW"),
        shape: vec![4],
        offsets: (0, 999),
    };
    assert!(
        unknown.is_consistent(),
        "an unknown width is unverifiable, not wrong"
    );
}

#[test]
fn dtype_widths_cover_what_df11_emits() {
    assert_eq!(Dtype::new("BF16").width(), Some(2));
    assert_eq!(Dtype::new("U8").width(), Some(1));
    assert_eq!(Dtype::new("I64").width(), Some(8));
    assert_eq!(Dtype::new("U32").width(), Some(4));
    assert_eq!(Dtype::new("WEIRD").width(), None);
}

/// Reading a real official DF11 shard must agree with the frozen manifest.
#[test]
fn reads_a_real_official_shard() {
    let Some(fx) = skip_if_missing("reads_a_real_official_shard") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").expect("tier0");
    let t = set
        .unit_tensors()
        .find(|t| t.name.ends_with("split_positions"))
        .expect("a split_positions fixture");

    let f = SafeTensorsFile::open(&t.file).expect("open real shard");
    let info = f.info(&t.name).expect("tensor present");
    assert_eq!(info.dtype.0, t.dtype, "{}: dtype", t.name);
    assert_eq!(info.shape, t.shape, "{}: shape", t.name);
    assert_eq!(info.nbytes(), t.bytes, "{}: byte count", t.name);
    assert!(info.is_consistent());
    assert_eq!(f.read(&t.name).expect("read"), t.read().expect("fixture"));

    // The official writer emits the six unit tensors plus the layer's norms.
    assert_eq!(f.len(), 10, "official layer shard holds 10 tensors");
}
