//! Phase 2 -- the safetensors reader and writer.

use df11_codec::safetensors::{write_file, Dtype, OutTensor, Payload, SafeTensorsFile, StError};
use df11_fixtures::skip_if_missing;
use std::collections::BTreeMap;

/// A path inside a directory unique to this test, so a leftover from another
/// test -- or from an earlier run of this one -- can never be mistaken for
/// output of the run under test.
fn tmp(name: &str) -> std::path::PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("df11pack_t{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("test dir");
    d.join(name)
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

/// A file appears complete or not at all.
///
/// Resume was dropped from the plan once compression got fast enough that losing
/// a run costs less than the machinery to resume it. Atomicity was kept, because
/// it protects against something speed does not fix: a half-written file that
/// looks finished.
#[test]
fn a_successful_write_leaves_no_temporary_behind() {
    let path = tmp("atomic_ok.safetensors");
    let dir = path.parent().unwrap().to_path_buf();
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "the test starts from an empty directory"
    );

    write_file(
        &path,
        &[OutTensor::owned(
            "t",
            Dtype::new(Dtype::U8),
            vec![4],
            vec![1, 2, 3, 4],
        )],
        &BTreeMap::new(),
    )
    .expect("write");

    let after: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        after.len(),
        1,
        "exactly the final file should remain, found {after:?}"
    );
    assert!(!after[0].ends_with(".tmp"), "a temporary was left behind");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_failed_write_leaves_no_file_at_the_destination() {
    use df11_codec::safetensors::Payload;

    // A borrowed payload that claims more bytes than its source has, so the copy
    // fails partway through.
    let path = tmp("atomic_fail.safetensors");
    // Deliberately outside the destination directory, so it does not count as a
    // leftover there.
    let src = std::env::temp_dir().join(format!("df11pack_short_{}.bin", std::process::id()));
    std::fs::write(&src, [0u8; 16]).unwrap();

    let r = write_file(
        &path,
        &[OutTensor {
            name: "t".into(),
            dtype: Dtype::new(Dtype::U8),
            shape: vec![1 << 20],
            data: Payload::Borrowed {
                path: src.clone(),
                offset: 0,
                len: 1 << 20, // far more than the 16 bytes available
            },
        }],
        &BTreeMap::new(),
    );
    assert!(r.is_err(), "the write must fail");
    assert!(
        !path.exists(),
        "a failed write must not leave a file at the destination; \
         a truncated safetensors is indistinguishable from a real one to a loader"
    );
    let leftovers: Vec<String> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        leftovers.is_empty(),
        "the destination directory must be empty after a failed write, found {leftovers:?}"
    );
    let _ = std::fs::remove_file(&src);
}

/// A failing rename must be an error, not a silent no-op.
///
/// Without this, swallowing the rename's result reports success while leaving
/// nothing at the destination -- the worst outcome of the three, because the
/// caller believes the file exists.
#[test]
fn a_write_that_cannot_be_renamed_into_place_is_an_error() {
    let path = tmp("rename_blocked.safetensors");
    // Occupy the destination with a directory, which a file cannot be renamed onto.
    let _ = std::fs::remove_file(&path);
    std::fs::create_dir_all(&path).expect("occupy the destination");

    let r = write_file(
        &path,
        &[OutTensor::owned(
            "t",
            Dtype::new(Dtype::U8),
            vec![2],
            vec![7, 7],
        )],
        &BTreeMap::new(),
    );
    assert!(
        r.is_err(),
        "a rename that cannot succeed must surface as an error, not be swallowed"
    );
    assert!(path.is_dir(), "the destination must be left as it was");
    let leftovers: Vec<String> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporaries left behind: {leftovers:?}"
    );
    let _ = std::fs::remove_dir_all(&path);
}

mod streaming {
    use super::tmp;
    use df11_codec::safetensors::{Dtype, SafeTensorsFile, StreamingWriter, TensorDecl};
    use std::collections::BTreeMap;

    fn decl(name: &str, len: u64) -> TensorDecl {
        TensorDecl {
            name: name.into(),
            dtype: Dtype::new(Dtype::U8),
            shape: vec![len],
            len,
        }
    }

    #[test]
    fn writes_declared_tensors_in_order() {
        let p = tmp("stream_ok.safetensors");
        let mut w =
            StreamingWriter::begin(&p, vec![decl("a", 2), decl("b", 3)], &BTreeMap::new()).unwrap();
        w.write("a", &[1, 2]).unwrap();
        w.write("b", &[3, 4, 5]).unwrap();
        w.finish().unwrap();
        let f = SafeTensorsFile::open(&p).unwrap();
        assert_eq!(f.read("a").unwrap(), vec![1, 2]);
        assert_eq!(f.read("b").unwrap(), vec![3, 4, 5]);
    }

    /// The property the single-file layout depends on: a tensor that comes out a
    /// different size than declared must be refused, because the header is
    /// already written and would otherwise describe the wrong bytes.
    #[test]
    fn a_tensor_of_the_wrong_size_is_refused_and_nothing_is_published() {
        let p = tmp("stream_len.safetensors");
        let mut w = StreamingWriter::begin(&p, vec![decl("a", 4)], &BTreeMap::new()).unwrap();
        assert!(
            w.write("a", &[1, 2, 3]).is_err(),
            "3 bytes where 4 were declared"
        );
        drop(w);
        assert!(!p.exists(), "a refused stream must not leave a file");
        let left: Vec<_> = std::fs::read_dir(p.parent().unwrap()).unwrap().collect();
        assert!(left.is_empty(), "nor a temporary");
    }

    #[test]
    fn a_tensor_out_of_order_is_refused() {
        let p = tmp("stream_order.safetensors");
        let mut w =
            StreamingWriter::begin(&p, vec![decl("a", 1), decl("b", 1)], &BTreeMap::new()).unwrap();
        assert!(
            w.write("b", &[9]).is_err(),
            "b written where a was declared"
        );
    }

    #[test]
    fn finishing_with_tensors_unwritten_is_refused() {
        let p = tmp("stream_short.safetensors");
        let mut w =
            StreamingWriter::begin(&p, vec![decl("a", 1), decl("b", 1)], &BTreeMap::new()).unwrap();
        w.write("a", &[1]).unwrap();
        assert!(w.finish().is_err(), "b was declared and never written");
        assert!(!p.exists(), "an incomplete stream must not be published");
    }

    #[test]
    fn an_abandoned_writer_leaves_nothing_behind() {
        let p = tmp("stream_drop.safetensors");
        {
            let mut w = StreamingWriter::begin(&p, vec![decl("a", 1)], &BTreeMap::new()).unwrap();
            w.write("a", &[1]).unwrap();
            // dropped without finish, as on a panic or early return
        }
        let left: Vec<_> = std::fs::read_dir(p.parent().unwrap()).unwrap().collect();
        assert!(
            left.is_empty(),
            "dropping an unfinished writer must remove its temporary"
        );
    }
}
