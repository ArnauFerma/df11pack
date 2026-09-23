//! The idx8 index scheme: construction, the representability check, and the
//! offsets it recovers.

use df11_codec::idx8::{build, Idx8Error, ESCAPE, SUPERBLOCK};

#[test]
fn every_block_range_is_recovered_exactly() {
    // Irregular code lengths, 1000 symbols, block 7: many superblocks, a short
    // final block.
    let lens: Vec<u32> = (0..1000u32).map(|i| 2 + (i * 7 + i / 13) % 9).collect();
    let block = 7;
    let ix = build(lens.iter().copied(), block).unwrap();
    assert_eq!(ix.lengths.len(), 1000usize.div_ceil(block));
    assert_eq!(ix.superblocks.len(), ix.lengths.len().div_ceil(SUPERBLOCK));
    let mut start = 0u64;
    for (b, chunk) in lens.chunks(block).enumerate() {
        let len: u64 = chunk.iter().map(|&l| u64::from(l)).sum();
        assert_eq!(ix.block_range(b), Some((start, start + len)), "block {b}");
        start += len;
    }
    assert_eq!(ix.total_bits, start);
    assert_eq!(ix.block_range(ix.lengths.len()), None);
}

/// A block whose length a u8 cannot hold is escaped, not refused: code 255 and
/// its exact length in the side table. Every offset still comes out exact.
#[test]
fn an_overlong_block_is_escaped_and_still_placed_exactly() {
    // Block 4: most blocks 4 bits; block 2 is 280 bits (delta 276), block 5 is
    // exactly min + 254 (the largest plain code), block 7 min + 255 (escaped).
    let mut lens = [1u32; 40];
    lens[8..12].copy_from_slice(&[70, 70, 70, 70]);
    lens[20..24].copy_from_slice(&[64, 64, 64, 66]);
    lens[28..32].copy_from_slice(&[65, 65, 65, 64]);
    let ix = build(lens.iter().copied(), 4).unwrap();
    assert_eq!(ix.minlen, 4);
    assert_eq!(ix.lengths[2], ESCAPE);
    assert_eq!(ix.lengths[5], 254, "254 is a plain code");
    assert_eq!(
        ix.lengths[7], ESCAPE,
        "255 itself must escape: it is the marker"
    );
    assert_eq!(ix.escapes, vec![(2, 280), (7, 259)]);
    let mut start = 0u64;
    for (b, chunk) in lens.chunks(4).enumerate() {
        let len: u64 = chunk.iter().map(|&l| u64::from(l)).sum();
        assert_eq!(ix.block_range(b), Some((start, start + len)), "block {b}");
        start += len;
    }
}

#[test]
fn a_zero_block_is_refused() {
    assert_eq!(
        build([3u32].into_iter(), 0).unwrap_err(),
        Idx8Error::BadBlock(0)
    );
}

/// A unit's last block is usually partial -- here one 2-bit symbol against
/// 7-symbol blocks of 280 bits. It must not set the minimum (every full block
/// would then escape); being below it, it is escaped itself, and placed exactly.
#[test]
fn a_short_final_block_does_not_set_the_minimum() {
    let mut lens = vec![40u32; 7 * 300];
    lens.push(2);
    let ix = build(lens.iter().copied(), 7).unwrap();
    assert_eq!(ix.minlen, 280);
    let last = ix.lengths.len() - 1;
    assert_eq!(
        ix.escapes,
        vec![(last as u32, 2)],
        "only the final block escapes"
    );
    let start = 280 * last as u64;
    assert_eq!(ix.block_range(last), Some((start, start + 2)));
    assert_eq!(ix.total_bits, start + 2);
}

// ---- through the writer ----

use df11_codec::arch::ArchDef;
use df11_codec::safetensors::SafeTensorsFile;
use df11_codec::source::ModelSource;
use df11_codec::write::{write_directory, WriteOptions};
use df11_fixtures::{architecture_defs, skip_if_missing};

/// idx8 output: the same codebook, bitstream and sign/mantissa bytes as DF11,
/// the idx8 tensors in place of gaps/output_positions, every block verified by
/// safe mode, and nothing that claims to be DF11.
#[test]
fn idx8_output_shares_the_payload_and_never_claims_df11() {
    let Some(fx) = skip_if_missing("idx8_writer") else {
        return;
    };
    let Some(set) = fx.set("synthetic-flux-dev-diffusers") else {
        return;
    };
    let defs = architecture_defs().unwrap();
    let (_, toml) = defs
        .iter()
        .find(|(n, _)| n == "flux-dev-diffusers")
        .unwrap();
    let def = ArchDef::from_toml(toml).unwrap();
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();

    let df11 = df11_fixtures::scratch("idx8_df11");
    write_directory(&src, &def, &df11, &WriteOptions::default()).unwrap();
    let i8 = df11_fixtures::scratch("idx8_out");
    let opts = WriteOptions {
        idx8_block: Some(64),
        verify: true,
        ..Default::default()
    };
    let r = write_directory(&src, &def, &i8, &opts).unwrap();
    assert_eq!(
        r.verified.len(),
        r.units,
        "every unit's idx8 blocks verified"
    );

    let unit = "transformer_blocks.0";
    let a = SafeTensorsFile::open(df11.join("transformer_blocks_0.safetensors")).unwrap();
    let b = SafeTensorsFile::open(i8.join("transformer_blocks_0.safetensors")).unwrap();
    for t in [
        "luts",
        "encoded_exponent",
        "sign_mantissa",
        "split_positions",
    ] {
        let n = format!("{unit}.{t}");
        assert_eq!(
            a.read(&n).unwrap(),
            b.read(&n).unwrap(),
            "{t} is shared with DF11"
        );
    }
    for t in ["gaps", "output_positions"] {
        assert!(b.info(&format!("{unit}.{t}")).is_none(), "{t} replaced");
    }
    assert_eq!(
        b.info(&format!("{unit}.idx8_superblocks")).unwrap().dtype.0,
        "U32"
    );
    assert_eq!(b.info(&format!("{unit}.idx8_meta")).unwrap().shape, vec![3]);
    assert_eq!(
        b.metadata().get("df11pack_index").map(String::as_str),
        Some("idx8")
    );

    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(i8.join("config.json")).unwrap()).unwrap();
    assert!(cfg.get("dfloat11_config").is_none(), "must not claim DF11");
    assert_eq!(cfg["df11pack_idx8_config"]["idx8_block"], 64);

    // And the index is smaller than the one it replaces.
    let idx = |f: &SafeTensorsFile, ts: &[&str]| -> u64 {
        ts.iter()
            .map(|t| f.info(&format!("{unit}.{t}")).unwrap().nbytes())
            .sum()
    };
    assert!(
        idx(&b, &["idx8_lengths", "idx8_superblocks", "idx8_meta"])
            < idx(&a, &["gaps", "output_positions"])
    );
}

/// A wrong stored length is caught: the block's symbols no longer end where
/// the index says. On a real idx8 unit read back from disk.
#[test]
fn a_wrong_idx8_length_on_disk_is_caught() {
    use df11_codec::idx8::Idx8;
    use df11_codec::verify::{verify_idx8_unit, VerifyError};
    let Some(fx) = skip_if_missing("idx8_corrupt") else {
        return;
    };
    let Some(set) = fx.set("synthetic-flux-dev-diffusers") else {
        return;
    };
    let defs = architecture_defs().unwrap();
    let (_, toml) = defs
        .iter()
        .find(|(n, _)| n == "flux-dev-diffusers")
        .unwrap();
    let def = ArchDef::from_toml(toml).unwrap();
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();
    let out = df11_fixtures::scratch("idx8_corrupt");
    let opts = WriteOptions {
        idx8_block: Some(64),
        ..Default::default()
    };
    write_directory(&src, &def, &out, &opts).unwrap();

    let unit = "single_transformer_blocks.1";
    let f = SafeTensorsFile::open(out.join("single_transformer_blocks_1.safetensors")).unwrap();
    let g = |t: &str| f.read(&format!("{unit}.{t}")).unwrap();
    let mut ix = Idx8::from_tensors(
        &g("idx8_lengths"),
        &g("idx8_superblocks"),
        &g("idx8_meta"),
        &g("idx8_escape_blocks"),
        &g("idx8_escape_lengths"),
    )
    .unwrap();
    let source: Vec<u8> = [
        "norm.linear",
        "proj_mlp",
        "proj_out",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
    ]
    .iter()
    .flat_map(|a| src.read(&format!("{unit}.{a}.weight")).unwrap())
    .collect();
    verify_idx8_unit(
        &g("luts"),
        &g("encoded_exponent"),
        &g("sign_mantissa"),
        &ix,
        &source,
    )
    .expect("the unit as written verifies");

    ix.lengths[3] = ix.lengths[3].wrapping_add(1);
    let e = verify_idx8_unit(
        &g("luts"),
        &g("encoded_exponent"),
        &g("sign_mantissa"),
        &ix,
        &source,
    )
    .expect_err("a wrong length must be caught");
    assert!(
        matches!(e, VerifyError::Idx8BlockBoundary { block: 3, .. }),
        "{e}"
    );
}

/// The real layer that plain idx8 had to refuse (block lengths 133..=515 at
/// block 64): with escapes it indexes, every block verifies against the source,
/// and the whole index is still smaller than gaps + output_positions.
#[test]
fn a_real_layer_with_outliers_indexes_and_verifies() {
    let Some(fx) = skip_if_missing("idx8_real") else {
        return;
    };
    let set = fx.set("tier0-qwen3-trunc-layers-only").unwrap();
    let defs = architecture_defs().unwrap();
    let (_, toml) = defs.iter().find(|(n, _)| n == "qwen3-4b").unwrap();
    let def = ArchDef::from_toml(toml).unwrap();
    let src = ModelSource::open(set.source_dir.join("model.safetensors")).unwrap();
    let out = df11_fixtures::scratch("idx8_real");
    let opts = WriteOptions {
        idx8_block: Some(64),
        verify: true,
        ..Default::default()
    };
    let r = write_directory(&src, &def, &out, &opts).expect("escapes make it representable");
    assert_eq!(r.verified.len(), 4);

    let official = set.tensors()[0].file.parent().unwrap().to_path_buf();
    let f = SafeTensorsFile::open(out.join("model_layers_0.safetensors")).unwrap();
    let o = SafeTensorsFile::open(official.join("model_layers_0.safetensors")).unwrap();
    let u = "model.layers.0";
    let n = |f: &SafeTensorsFile, t: &str| f.info(&format!("{u}.{t}")).unwrap().nbytes();
    let escapes = n(&f, "idx8_escape_blocks") / 4;
    assert!(escapes > 0, "this layer has outlier blocks");
    let idx8: u64 = [
        "idx8_lengths",
        "idx8_superblocks",
        "idx8_meta",
        "idx8_escape_blocks",
        "idx8_escape_lengths",
    ]
    .iter()
    .map(|t| n(&f, t))
    .sum();
    let df11 = n(&o, "gaps") + n(&o, "output_positions");
    eprintln!("layer 0: {escapes} escapes; idx8 index {idx8} B vs DF11 {df11} B");
    assert!(idx8 < df11);
}
