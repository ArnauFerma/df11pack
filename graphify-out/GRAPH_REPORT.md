# Graph Report - df11pack  (2026-09-26)

## Corpus Check
- 151 files · ~286,846 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 61 file(s) not represented in the graph (top: .toml 36, .log 8, .rc 7)

## Summary
- 1361 nodes · 2801 edges · 93 communities (86 shown, 7 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 294 edges (avg confidence: 0.85)
- Token cost: 249,235 input · 0 output

## Community Hubs (Navigation)
- Output Format Checker
- Architecture Definitions
- Post-hoc Verification
- Test Fixtures Library
- Key Mapping Rules
- CLI Entry Point
- Model Source Reader
- Unit Discovery
- Phase 2 Exit Gate
- GPU Session Helpers
- Bitstream & Codebook Tests
- Safetensors Output Tensors
- Invariant Checker (Python)
- Safetensors Writer Tests
- Directory Writer Tests
- H4 Huffman Test Cases
- Stratified Chunk Sampling
- Error Plumbing
- Codec Limits & Errors
- Release & Community Docs
- Batched GPU Session Findings
- H6 Reorder Inference Test
- Synthetic Model Generator
- CuPy Shim
- Bitstream & Chunked Encoder
- Byte-Identity Plan & Corpus
- H11 Shard Repacking
- Compatibility Contract & LUT Modes
- Real Header Fetcher
- Directory Writer & Config
- Architecture Tests
- Qwen3-8B Comparison Report
- Huffman Codebook & LUTs
- Huffman Heap Nodes
- Fixture Harness Tests
- Config Rebuilder
- RAM Budget & Workers
- Limiter Case Generator
- CI & Phase 0 Environment
- idx8 Index Builder
- I/O Scheduling
- Safetensors Dtypes
- Error Trait Glue
- Native Output Tests
- idx8 & Unit Tests
- I/O Scheduler Tests
- Encoder & Decoder Design
- Pattern Dict Variants
- GPU Runbook & H10/H11
- H11 Repack Inference Test
- Definition Format & Layouts
- Source Reader Tests
- idx8 Index Tensors
- Streaming Unit Encoder
- Whole-File Identity Tests
- Index Schemes
- Chroma/Flux Units & Pattern Dict
- 32-bit Codec Reference
- 32-bit Code Limiter
- Histogram Tests
- H7 decode.ptx Test
- Tie-Break & LUT Findings
- GPU Session Runner
- EOF Symbol
- Exponent Histogram
- Atomic File Writes
- Verify CLI Tests
- Key Rules, RAM & Refusals
- Config & output_positions Findings
- Refuse-Don't-Guess Limiter
- Synthetic Identity Tests
- DF11 Format & LUT Leak
- Parallel Encoder & idx8 Findings
- Definition Generator
- RSS Sampler
- DF11 Ecosystem
- Unit Build Views
- Output Layouts
- Streaming & I/O Design
- get_luts Reimplementation
- Repack Verifier
- Phase 0 Hypotheses
- Qwen3-8B Benchmark Setup
- Safetensors Order Test
- Verify Command Design
- Phase Run Script
- Fixture Manifest
- Name Iterator
- Env Paths
- Entropy Headroom
- Crates
- encoded_exponent
- df11-fixtures

## God Nodes (most connected - your core abstractions)
1. `skip_if_missing()` - 56 edges
2. `write_directory()` - 42 edges
3. `ArchDef` - 36 edges
4. `check_output()` - 35 edges
5. `StError` - 30 edges
6. `architecture_defs()` - 29 edges
7. `write_file()` - 27 edges
8. `ModelSource` - 26 edges
9. `SafeTensorsFile` - 25 edges
10. `discover()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Qwen3-8B session requirements` --semantically_similar_to--> `Phase 0 pinned requirements`  [INFERRED] [semantically similar]
  gpu_session/qwen3_8b/requirements.txt → phase0/requirements.txt
- `Corruption tests (11 cases)` --semantically_similar_to--> `Tests must be able to fail`  [INFERRED] [semantically similar]
  phase0/INVARIANTS_RESULTS.md → CONTRIBUTING.md
- `phase0/H4_RESULTS.md` --references--> `h4_codec_spec.py executable spec`  [EXTRACTED]
  docs/FINDINGS.md → phase0/H4_RESULTS.md
- `df11pack v0.1.0 release (2026-09-23)` --references--> `32-bit limiter: refuse on ambiguous tie (AmbiguousLimiterTie)`  [INFERRED]
  CHANGELOG.md → docs/COMPATIBILITY.md
- `phase0/official.py wrapper` --references--> `Official DFloat11 compressor (pip dfloat11 0.5.0)`  [INFERRED]
  phase0/README.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **DF11 unit tensor layout** — docs_findings_sign_mantissa, docs_findings_encoded_exponent, docs_findings_gaps, docs_findings_output_positions, docs_findings_luts, docs_findings_split_positions [EXTRACTED 1.00]
- **Byte-identical dahuffman reproduction** — docs_findings_h4, docs_findings_dahuffman_tiebreak, docs_findings_argpartition_tie_order, docs_findings_get_luts_leak_bug [EXTRACTED 1.00]
- **Batched GPU session confirmations** — docs_findings_gpu_session, docs_findings_h7, docs_findings_h6, docs_findings_h11, docs_findings_luts_correct_mode [EXTRACTED 1.00]
- **DF11 parallel-decode metadata used by kernel, CPU decoder and chunked encoder** — docs_design_gaps, docs_design_output_positions, docs_design_decode_kernel, docs_design_cpu_reference_decoder, docs_design_chunked_encoder [EXTRACTED 1.00]
- **Byte-identity verification chain** — docs_plan_official_df11_releases, docs_plan_golden_fixtures, docs_plan_fixture_harness, docs_design_h4_byte_identity [INFERRED 0.85]
- **Concatenation order fixes split_positions and bytes** — docs_design_pattern_dict, docs_design_split_positions, docs_plan_architecture_definitions_toml, docs_design_diffusers_layout [EXTRACTED 1.00]
- **GPU session closing four Phase 0 items** — gpu_session_expected_h7_decode_ptx_driver_api, gpu_session_expected_h6_inference_confirmation, gpu_session_expected_h11_load_confirmation, gpu_session_expected_luts_correct_kernel_gate, gpu_session_run_all [EXTRACTED 1.00]
- **Byte-identical or refuse output policy** — contributing_byte_identity_contract, contributing_refuse_rather_than_guess, docs_compatibility_luts_compat_mode, docs_compatibility_32bit_limiter_refusal, docs_compatibility_metadata_stamp [INFERRED 0.85]
- **Opt-in non-default extras in v0.1.0** — docs_compatibility_luts_correct_mode, readme_hashes_option, docs_index_schemes_idx8_scheme [EXTRACTED 1.00]

## Communities (93 total, 7 thin omitted)

### Community 0 - "Output Format Checker"
Cohesion: 0.06
Nodes (57): check, a_chunk_may_hold_one_weight_per_bit_and_no_more(), check_output(), CheckError, CheckReport, Failure, GEO, Geometry (+49 more)

### Community 1 - "Architecture Definitions"
Cohesion: 0.07
Nodes (42): architecture_defs, ArchDef, ArchError, KeyRules, Layout, Rename, Display, Error (+34 more)

### Community 2 - "Post-hoc Verification"
Cohesion: 0.10
Nodes (47): chunk_count(), chunk_range(), decode_at(), gap_at(), peek8(), Display, Error, Formatter (+39 more)

### Community 3 - "Test Fixtures Library"
Cohesion: 0.09
Nodes (35): AmbiguousDetail, ByteDiff, Fixtures, FixtureSet, limiter_cases(), LimiterCase, LimiterFile, locate() (+27 more)

### Community 4 - "Key Mapping Rules"
Cohesion: 0.10
Nodes (29): arch, comfyui_prefix, KeyError, map_names(), NameMap, BTreeMap, Display, Error (+21 more)

### Community 5 - "CLI Entry Point"
Cohesion: 0.12
Nodes (33): clap, all_defs(), Cli, Command, compress(), CompressArgs, df11_codec::io_sched::IoMode, IndexArg (+25 more)

### Community 6 - "Model Source Reader"
Cohesion: 0.12
Nodes (12): COMFYUI_PREFIX, ModelSource, AsRef, BTreeMap, Option, Path, PathBuf, Self (+4 more)

### Community 7 - "Unit Discovery"
Cohesion: 0.14
Nodes (32): discover(), DiscoveredUnit, DiscoverError, Discovery, natural_cmp(), python_to_rust(), Display, Error (+24 more)

### Community 8 - "Phase 2 Exit Gate"
Cohesion: 0.09
Nodes (24): argparse, ast, collections, dfloat11, Phase 2 exit gate, with a determinism baseline. The skeleton comes from…, json, pathlib, Transcribe ComfyUI-DFloat11-Extended's pattern_dict.py into JSON, pinned. The… (+16 more)

### Community 9 - "GPU Session Helpers"
Cohesion: 0.12
Nodes (28): bytes_equal(), decode_with_cupy(), default_out_path(), find_ptx_path(), gpu_info(), group_units(), launch_geometry(), load_unit_raw() (+20 more)

### Community 10 - "Bitstream & Codebook Tests"
Cohesion: 0.11
Nodes (17): split_fields(), BYTES_PER_THREAD, index_tensors_have_the_shapes_the_format_requires(), QWEN3_LAYER, THREADS_PER_BLOCK, correct_mode_differs_only_where_the_leak_is(), luts_match_official_output_byte_for_byte(), QWEN3_LAYER (+9 more)

### Community 11 - "Safetensors Output Tensors"
Cohesion: 0.18
Nodes (12): header_bytes(), OutTensor, AsRef, BTreeMap, Self, String, Vec, SafeTensorsFile (+4 more)

### Community 12 - "Invariant Checker (Python)"
Cohesion: 0.13
Nodes (18): Structural self-consistency vs content correctness, Exception, math, numpy, check_file(), check_unit(), fail(), Failure (+10 more)

### Community 13 - "Safetensors Writer Tests"
Cohesion: 0.19
Nodes (18): write_file(), a_failed_write_leaves_no_file_at_the_destination(), a_missing_tensor_is_an_error_not_a_panic(), a_successful_write_leaves_no_temporary_behind(), a_tensor_of_the_wrong_size_is_refused_and_nothing_is_published(), a_tensor_out_of_order_is_refused(), a_truncated_header_is_rejected(), a_write_that_cannot_be_renamed_into_place_is_an_error() (+10 more)

### Community 14 - "Directory Writer Tests"
Cohesion: 0.23
Nodes (16): write_directory(), a_wrong_idx8_length_on_disk_is_caught(), idx8_output_shares_the_payload_and_never_claims_df11(), a_512_mib_budget_still_compresses_and_bounds_workers(), a_config_is_written_for_layouts_that_have_one(), a_non_bf16_unit_tensor_is_refused_before_writing(), a_tied_tensor_is_dropped_and_reported(), comfyui_native_output_has_no_config() (+8 more)

### Community 15 - "H4 Huffman Test Cases"
Cohesion: 0.16
Nodes (20): dahuffman, _fib_chain(), find_fib_chain_for_max_len(), h_fib_chain(), h_fib_chain_multiplicity(), h_heavily_skewed(), h_many_ties(), h_near_degenerate() (+12 more)

### Community 16 - "Stratified Chunk Sampling"
Cohesion: 0.14
Nodes (15): BTreeSet, chunk_of(), plan(), Option, Vec, splitmix64(), UnitShape, a_budget_covering_everything_picks_every_chunk() (+7 more)

### Community 17 - "Error Plumbing"
Cohesion: 0.16
Nodes (9): BufWriter, Display, Error, Formatter, From, StError, StreamingWriter, Drop (+1 more)

### Community 18 - "Codec Limits & Errors"
Cohesion: 0.13
Nodes (16): check_code_len(), check_prefix_tables(), check_unit_limits(), EncodeError, MAX_CODE_BITS, MAX_PREFIX_TABLES, MAX_UNIT_BYTES, MAX_UNIT_WEIGHTS (+8 more)

### Community 19 - "Release & Community Docs"
Cohesion: 0.19
Nodes (19): Issue template config, Release workflow, Release notes extracted from CHANGELOG section, Static musl/multi-platform binaries with SHA256SUMS, CHANGELOG, df11pack v0.1.0 release (2026-09-23), Code of Conduct (Contributor Covenant 2.1), CONTRIBUTING (+11 more)

### Community 20 - "Batched GPU Session Findings"
Cohesion: 0.15
Nodes (19): cupy import shim that raises on use, Batched rented GPU session (RTX A4000), H1: official peak RAM ~2x model, H11: loader ignores shard grouping, H12: CPU decoder scales with cores, H2: >90% time in Python encode loop, H5: native encoder makes disk the bottleneck, H6: loaders ignore physical tensor order (+11 more)

### Community 21 - "H6 Reorder Inference Test"
Cohesion: 0.16
Nodes (17): build_reordered_dir(), build_skeleton_model(), default_reordered_shard(), default_src_dir(), main(), test_h6_reorder_inference.py -- H6 inference confirmation. FINDINGS.md 0.8/H6…, repo_root(), run_forward() (+9 more)

### Community 22 - "Synthetic Model Generator"
Cohesion: 0.15
Nodes (17): itertools, build(), generic_shape(), instances(), lin(), main(), put(), Corpus case 2: small synthetic FLUX and Chroma models, compressed by the… (+9 more)

### Community 23 - "CuPy Shim"
Cohesion: 0.13
Nodes (11): cuda, _Device, _Function, __getattr__(), _missing(), _GpuReached, A deliberately non-functional stand-in for cupy. The official `dfloat11`…, Records the PTX path; never parses or loads it. (+3 more)

### Community 24 - "Bitstream & Chunked Encoder"
Cohesion: 0.19
Nodes (14): code_for(), encode(), Encoded, Vec, code_table(), DEFAULT_CHUNK, encode_chunked(), check() (+6 more)

### Community 25 - "Byte-Identity Plan & Corpus"
Cohesion: 0.13
Nodes (18): np.argpartition tie-ordering risk, bf16-exponent-compression (sibling repo), ChromaRadiance, dahuffman library, df11pack, Golden tests corpus, H4: byte-identical output by replicating dahuffman, LeanModels/DFloat11 (+10 more)

### Community 26 - "H11 Shard Repacking"
Cohesion: 0.18
Nodes (16): collect_manifest(), main(), plan_interleaved(), plan_single(), Variant (b): N files, round-robin over ALL keys (so layer N's seven attribute…, repack_shards.py -- H11 evidence builder. Repacks the 28-shard…, key -> dict(dtype, shape, src_path, src_start, src_end), Write one safetensors file containing exactly `keys`, in the given order,… (+8 more)

### Community 27 - "Compatibility Contract & LUT Modes"
Cohesion: 0.18
Nodes (17): Pull request template, Byte-identity is the contract, Tests must be able to fail, COMPATIBILITY.md, get_luts carry-forward LUT leak bug, --luts=compat mode (default), DESIGN.md, FINDINGS.md (df11pack measurement log) (+9 more)

### Community 28 - "Real Header Fetcher"
Cohesion: 0.20
Nodes (15): concurrent_futures, fetch(), get(), header(), info(), main(), Real tensor names, for checking definitions against real checkpoints. For each…, The source config's tie_word_embeddings, from the config.json beside its… (+7 more)

### Community 29 - "Directory Writer & Config"
Cohesion: 0.16
Nodes (14): config, Self, add_hashes(), BYTES_PER_WEIGHT_HELD, DEFAULT_MEMORY_FRACTION, HASH_KEY_PREFIX, HASH_STAMP, BTreeMap (+6 more)

### Community 30 - "Architecture Tests"
Cohesion: 0.13
Nodes (11): encode_unit(), definitions_still_match_the_official_pattern_dicts(), every_shipped_definition_parses_and_validates(), QWEN3_LAYER, split_positions_reproduce_official_output(), standalone_units_are_recognised(), a_reserved_exponent_aborts_before_anything_is_produced(), a_single_tensor_unit_has_empty_split_positions() (+3 more)

### Community 31 - "Qwen3-8B Comparison Report"
Cohesion: 0.17
Nodes (11): datetime, Compare the three Qwen3-8B outputs and write /workspace/out/report.json. -…, sha(), tree(), hashlib, header(), main(), 0.12 -- freeze the Phase 0 reference outputs as graded fixtures. Records per-… (+3 more)

### Community 32 - "Huffman Codebook & LUTs"
Cohesion: 0.30
Nodes (9): binaryheap, bits_string(), build_luts(), build_luts_with(), Code, Codebook, LutMode, String (+1 more)

### Community 33 - "Huffman Heap Nodes"
Cohesion: 0.29
Nodes (9): Node, Option, Ordering, Self, Sym, Eq, Ord, PartialEq (+1 more)

### Community 34 - "Fixture Harness Tests"
Cohesion: 0.25
Nodes (12): encoded_gaps_and_positions_match_official_output(), all_six_tensors_match_official_output_for_every_unit(), assert_matches(), diff_bytes(), skip_if_missing(), assert_matches_accepts_the_real_bytes(), assert_matches_panics_on_mismatch_and_names_the_tensor(), diff_reports_exact_position_and_bit() (+4 more)

### Community 35 - "Config Rebuilder"
Cohesion: 0.23
Nodes (12): copy, build_dfloat11_config(), derive_layer_types(), derive_rope_parameters(), dump(), main(), Qwen3-specific. Confirmed only for sliding_window=None (both fixtures). Real…, Reproduce the FULL transformers-normalized schema (5 transformations). Byte-… (+4 more)

### Community 36 - "RAM Budget & Workers"
Cohesion: 0.26
Nodes (13): available_memory(), bytes_per_weight(), cgroup_headroom(), parse_mem_available(), Option, worker_count(), worker_count_with(), WriteOptions (+5 more)

### Community 37 - "Limiter Case Generator"
Cohesion: 0.28
Nodes (11): case(), fib(), main(), Generate expected results for the 32-bit code-length limiter. No real fixture…, freqs: dict symbol -> frequency, ascending by symbol., build_huffman_table(), get_32bit_codec(), _max_code_len() (+3 more)

### Community 38 - "CI & Phase 0 Environment"
Cohesion: 0.18
Nodes (12): CI workflow, Cross-platform test matrix (Linux/Windows/macOS), MSRV Rust 1.85 check, Phase 0 fixtures, Environment pins (cu124 torch, setuptools<81, transformers 4.51.0), dahuffman library, h4_codec_spec.py executable spec, Phase 0 reference environment README (+4 more)

### Community 39 - "idx8 Index Builder"
Cohesion: 0.18
Nodes (10): build(), ESCAPE, Idx8Error, Display, Error, Formatter, Item, Iterator (+2 more)

### Community 40 - "I/O Scheduling"
Cohesion: 0.36
Nodes (11): device_name_for(), IoMode, IoPlan, parse_rotational(), plan(), read_rotational(), rotational_for(), Option (+3 more)

### Community 41 - "Safetensors Dtypes"
Cohesion: 0.21
Nodes (7): Dtype, .BF16, .I64, dtype_rank(), .U8, Option, TensorInfo

### Community 42 - "Error Trait Glue"
Cohesion: 0.18
Nodes (10): Display, Error, Fn, Formatter, From, Self, T, run_units() (+2 more)

### Community 43 - "Native Output Tests"
Cohesion: 0.23
Nodes (11): a_prefixed_source_gives_the_same_output(), native_output_is_identical_across_worker_counts(), native_output_is_one_file_and_no_config(), NATIVE_QWEN, outdir(), Path, PathBuf, String (+3 more)

### Community 44 - "idx8 & Unit Tests"
Cohesion: 0.18
Nodes (5): archdef, chunked, crate, a_real_layer_with_outliers_indexes_and_verifies(), idx8

### Community 45 - "I/O Scheduler Tests"
Cohesion: 0.18
Nodes (4): rotational_detection_agrees_with_sysfs_on_this_machine(), io_sched, path, write_as

### Community 46 - "Encoder & Decoder Design"
Cohesion: 0.27
Nodes (11): Chunked parallel encoder (histogram prefix-sum), Compression/decompression asymmetry, Parallel CPU reference decoder, decode.cu / decode.ptx kernel, gaps tensor, H12: CPU decoder scales with cores, MIT licence decision, output_positions tensor (+3 more)

### Community 47 - "Pattern Dict Variants"
Cohesion: 0.18
Nodes (11): Chroma distilled_guidance_layer unit (12 attrs), Flux/Chroma diffusers concatenation orders, dfloat11_config version drift (0.2.0 vs 0.5.0), H13: Chroma approximator stays uncompressed (refuted), Layers-only pattern (Qwen3-4B style), Mainstream pattern (lm_head + embed_tokens standalone), dfloat11_config.pattern_dict, Qwen3-0.6B (tier 1) (+3 more)

### Community 48 - "GPU Runbook & H10/H11"
Cohesion: 0.22
Nodes (11): H11 load confirmation (shard grouping), H6 inference confirmation (intra-shard tensor order), GPU session runbook, RunPod RTX A4000 rental, Tier-0 4-layer Qwen3-0.6B official output directory, H10 and H11 results, H10: config.json rebuildable without transformers/diffusers, H11: loader dispatches by tensor name, not shard layout (+3 more)

### Community 49 - "H11 Repack Inference Test"
Cohesion: 0.36
Nodes (10): build_skeleton_model(), copy_config_files(), default_src_dir(), main(), test_h11_repack_inference.py -- H11 load confirmation. FINDINGS.md 0.8/H11…, repack_script(), repo_root(), run_cmd() (+2 more)

### Community 50 - "Definition Format & Layouts"
Cohesion: 0.22
Nodes (10): Model request issue template, Tied lm_head compressed as own unit, [keys] rename/strip/drop rules, Output layouts (transformers, diffusers, diffusers-single, comfyui-native), pattern_dict, TOML model definition format, H8 / H9 / H6 results, H8: non-unit tensors passed through untouched (LLM) (+2 more)

### Community 51 - "Source Reader Tests"
Cohesion: 0.27
Nodes (8): btreemap, a_sharded_source_reads_identically_to_a_single_file(), a_tensor_declared_by_two_shards_is_an_error_not_a_coin_flip(), dir(), opens_a_directory_of_shards_and_unifies_them(), opens_a_single_file(), PathBuf, t()

### Community 52 - "idx8 Index Tensors"
Cohesion: 0.33
Nodes (4): Idx8, Option, Self, Vec

### Community 53 - "Streaming Unit Encoder"
Cohesion: 0.38
Nodes (8): encode_unit_streaming(), encode_unit_streaming_with(), Option, String, Vec, UnitOutput, E, FnMut

### Community 54 - "Whole-File Identity Tests"
Cohesion: 0.31
Nodes (9): assert_identical(), comfyui_single_file_is_identical(), diffusers_unit_shards_are_identical(), Option, Path, PathBuf, run(), transformers_directory_is_identical_file_for_file() (+1 more)

### Community 55 - "Index Schemes"
Cohesion: 0.27
Nodes (10): Format divergence requires own decoder and name, INDEX_SCHEMES.md, bf16-exponent-compression sibling project (kernel_idx8.py), df11 index scheme (gaps + output_positions), gaps tensor, idx8 index scheme with escape table, IndexBuilder seam, output_positions tensor (+2 more)

### Community 56 - "Chroma/Flux Units & Pattern Dict"
Cohesion: 0.20
Nodes (10): Checkpoints, journal and resume (dropped), Chroma architecture, ChromaApproximator (distilled_guidance_layer), Compression unit (UC), Flux architecture, H13: Chroma approximator uncompressed (refuted), pattern_dict, split_positions tensor (+2 more)

### Community 57 - "32-bit Codec Reference"
Cohesion: 0.22
Nodes (10): compare_32bit(), compare_tables(), _probe_tie_sensitivity(), Real dahuffman-built code table for `frequencies`., For an argpartition call that had a real tie at the boundary (more candidates…, Transcription of dfloat11_utils.get_32bit_codec using real dahuffman., ref_eof_key(), ref_get_32bit_codec() (+2 more)

### Community 58 - "32-bit Code Limiter"
Cohesion: 0.28
Nodes (7): build_limited(), LimitedBuild, a_codebook_already_within_32_bits_is_returned_untouched(), ambiguous_cases_are_refused_with_the_measured_numbers(), unambiguous_cases_match_the_spec_exactly(), encodeerror, limiter_cases

### Community 60 - "H7 decode.ptx Test"
Cohesion: 0.33
Nodes (8): ctypes, cu_check(), decode_with_driver_api(), default_unit_file(), load_libcuda(), test_h7_decode_ptx.py -- H7: can decode.ptx be loaded and run WITHOUT CuPy,…, repo_root(), setup_prototypes()

### Community 61 - "Tie-Break & LUT Findings"
Cohesion: 0.25
Nodes (9): np.argpartition tie order (load-bearing), Byte-identity criterion, dahuffman representative-based tie-break rule, diffusers-single layout, get_luts curr_val carry-over bug, H4: dahuffman reproducible byte-for-byte, luts (prefix lookup tables), Two-pass single-file writer (StreamingWriter) (+1 more)

### Community 62 - "GPU Session Runner"
Cohesion: 0.28
Nodes (8): fail(), ITEM_ELAPSED, ITEM_NAMES, ITEM_RC, ITEM_SCRIPTS, log(), run_all.sh script, results.json / SUMMARY.txt

### Community 64 - "Exponent Histogram"
Cohesion: 0.29
Nodes (3): Histogram, Vec, split_fields_into()

### Community 65 - "Atomic File Writes"
Cohesion: 0.46
Nodes (6): Payload, Path, PathBuf, sync_dir(), tmp_path(), write_bytes_atomic()

### Community 66 - "Verify CLI Tests"
Cohesion: 0.32
Nodes (7): a_hashed_output_catches_a_flipped_value_without_the_source(), String, run(), verify_exits_0_on_a_good_output_and_1_on_a_corrupted_one(), modelsource, pathbuf, skip_if_missing

### Community 67 - "Key Rules, RAM & Refusals"
Cohesion: 0.25
Nodes (8): Non-BF16 source refusal, BYTES_PER_WEIGHT_HELD constant, H8: uncompressed tensors passed through (LLM), Per-definition key rules (strip_prefix, drop, rename), --ram budget, Definitions against real checkpoints (23 of 28 fit), Streaming memory reductions (63.8 MiB peak), Tied word embeddings (lm_head / embed_tokens)

### Community 68 - "Config & output_positions Findings"
Cohesion: 0.25
Nodes (8): save_pretrained whole-schema config rewrite, ConfigMode::PreserveSource, Determinism baseline for logits comparison, Diffusers two-key config.json, H10: config.json rebuildable without heavy libs, output_positions, Transformers-version RoPE config trap, Two-sided output_positions chunk bound

### Community 69 - "Refuse-Don't-Guess Limiter"
Cohesion: 0.33
Nodes (7): Refuse rather than guess, 32-bit limiter: refuse on ambiguous tie (AmbiguousLimiterTie), np.argpartition tie ambiguity, phase0/H4_RESULTS.md, get_luts curr_val read-before-assignment crash, H4: reproduce dahuffman codebook from histogram, (frequency, representative) tie-breaking rule

### Community 70 - "Synthetic Identity Tests"
Cohesion: 0.52
Nodes (6): check(), chroma_comfyui_native_matches_the_official_output(), chroma_diffusers_matches_the_official_output(), every_definition_matches_the_official_output(), flux_comfyui_native_matches_the_official_output(), flux_diffusers_matches_the_official_output()

### Community 71 - "DF11 Format & LUT Leak"
Cohesion: 0.29
Nodes (7): LUT carry-forward leakage, DFloat11 (DF11) format, Kernel 32-bit limits, Hierarchical LUTs (>=240 jump convention), sign_mantissa tensor, check_invariants.py format checker, --luts=compat / --luts=correct modes

### Community 72 - "Parallel Encoder & idx8 Findings"
Cohesion: 0.33
Nodes (7): Chunked parallel encoder, EOF window gaps bug in chunked encoder, gaps index tensor, idx8 input-side index, idx8 escape table, Inter-unit parallelism, Intra-unit parallelism

### Community 73 - "Definition Generator"
Cohesion: 0.48
Nodes (6): esc(), extended(), kebab(), keys_toml(), main(), Generate data/architectures/*.toml from verified official pattern_dicts.…

### Community 74 - "RSS Sampler"
Cohesion: 0.38
Nodes (5): main(), mark(), peak_mb(), rss_mb(), Sampler

### Community 75 - "DF11 Ecosystem"
Cohesion: 0.33
Nodes (6): Bug report issue template, ComfyUI-DFloat11-Extended, df11pack compressor, DFloat11 format, DFloat11Model loader, Official DFloat11 compressor (pip dfloat11 0.5.0)

### Community 76 - "Unit Build Views"
Cohesion: 0.40
Nodes (6): View, borrowed(), build_unit(), Built, Vec, Reader

### Community 77 - "Output Layouts"
Cohesion: 0.40
Nodes (6): ComfyUI-DFloat11-Extended, ComfyUI-native layout, dfloat11_config / config.json, Diffusers layout, diffusers-single layout, Phase 7: remaining architectures (35 definitions)

### Community 78 - "Streaming & I/O Design"
Cohesion: 0.33
Nodes (6): H5: native encoder is disk-bound, I/O scheduling (HDD vs SSD), RAM budget per worker (~1.35 N bytes), Streaming pipeline (reader, bounded queue, workers, committer), 3 GB RAM development machine constraint, Phase 3: streaming and parallelism

### Community 79 - "get_luts Reimplementation"
Cohesion: 0.33
Nodes (6): ndarray, get_luts(), Reimplementation of dfloat11_utils.get_luts (minus the final torch.from_numpy…, compare_luts(), Transcription of dfloat11_utils.get_luts, dropping the final torch.from_numpy…, ref_get_luts()

### Community 80 - "Repack Verifier"
Cohesion: 0.53
Nodes (5): bytes_equal(), collect_manifest(), main(), verify_repack.py -- confirm a repacked variant is byte-identical per tensor…, read_header()

### Community 81 - "Phase 0 Hypotheses"
Cohesion: 0.40
Nodes (5): H7: load decode.ptx without CuPy, H8/H9: uncompressed tensors unchanged, Phase 0 hypotheses H1-H13, Phase 0: measure official compressor, Rented GPU session (RTX A4000, ~$0.09)

### Community 82 - "Qwen3-8B Benchmark Setup"
Cohesion: 0.40
Nodes (5): Qwen3-8B session requirements, Qwen3-8B run pip freeze, Phase 0 pinned requirements, Qwen3-8B benchmark (40 s / 6.6 GB vs 37 min / 27.3 GB), DFloat11/Qwen3-8B-DF11 published release

### Community 83 - "Safetensors Order Test"
Cohesion: 0.50
Nodes (4): canonical_order(), Fn, T, layout_follows_the_safetensors_library_order()

### Community 84 - "Verify Command Design"
Cohesion: 0.67
Nodes (4): Kernel chunk (4096 bytes), Post-hoc verification (integrity/sample/full), Stratified chunk sampling, df11pack verify command (Phase 6)

### Community 85 - "Phase Run Script"
Cohesion: 1.00
Nodes (3): log(), run.sh script, step()

### Community 86 - "Fixture Manifest"
Cohesion: 0.50
Nodes (3): generated, note, sets

### Community 88 - "Env Paths"
Cohesion: 0.67
Nodes (3): Env, Option, PathBuf

## Knowledge Gaps
- **85 isolated node(s):** `df11-codec`, `DEFAULT_CHUNK`, `SUPERBLOCK`, `ESCAPE`, `RESERVED_EXPONENT_MIN` (+80 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 352 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ArchDef` connect `Architecture Definitions` to `Output Format Checker`, `Key Mapping Rules`, `CLI Entry Point`, `Model Source Reader`, `Unit Discovery`, `Directory Writer Tests`, `Env Paths`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **Why does `FINDINGS.md (df11pack measurement log)` connect `Compatibility Contract & LUT Modes` to `Invariant Checker (Python)`, `Batched GPU Session Findings`, `Synthetic Model Generator`, `Index Schemes`, `Real Header Fetcher`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Why does `write_directory()` connect `Directory Writer Tests` to `Output Format Checker`, `Atomic File Writes`, `Architecture Definitions`, `RAM Budget & Workers`, `Key Mapping Rules`, `Model Source Reader`, `Unit Discovery`, `I/O Scheduling`, `Synthetic Identity Tests`, `Error Trait Glue`, `Safetensors Output Tensors`, `Unit Build Views`, `Safetensors Writer Tests`, `idx8 & Unit Tests`, `Native Output Tests`, `Whole-File Identity Tests`, `CLI Entry Point`, `Directory Writer & Config`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 52 inferred relationships involving `skip_if_missing()` (e.g. with `split_positions_reproduce_official_output()` and `encoded_gaps_and_positions_match_official_output()`) actually correct?**
  _`skip_if_missing()` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `write_directory()` (e.g. with `build_config()` and `discover()`) actually correct?**
  _`write_directory()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `check_output()` (e.g. with `discover()` and `chunk_of()`) actually correct?**
  _`check_output()` has 22 INFERRED edges - model-reasoned connections that need verification._
- **What connects `df11-codec`, `DEFAULT_CHUNK`, `SUPERBLOCK` to the rest of the system?**
  _85 weakly-connected nodes found - possible documentation gaps or missing edges._