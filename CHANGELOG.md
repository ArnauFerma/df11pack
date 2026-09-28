# Changelog

All notable changes. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) — before 1.0, minor
versions may change the command line.

## [Unreleased]

### Changed

- `phase0/check_comfyui_release.sh` (research script, not the tool): runs without GNU
  time, stops on a failed compress, ends with `CHECK: PASS`/`FAILED` and a matching exit
  status, and strips the binary's directory from its logs.

## [0.1.1] - 2026-09-26

Fixes found by an independent review of the repository against its own data.

### Fixed

- **`--luts=correct` did not change the LUTs in 0.1.0.** It wrote the
  `df11pack_luts="correct"` stamp but encoded in compat mode, so 0.1.0 output
  stamped "correct" is compat output (it decodes correctly; the stamp is wrong).
  The mode now reaches the encoder, with a test that checks the written LUTs.
- Two definition patterns claiming the same module are refused; before, the
  module became two units and its tensors were written twice.
- `--ram` with a huge value gives a parse error instead of overflowing.
- A unit whose tensors are all empty is an error, not a panic.
- `verify` fails a unit whose `gaps`/`output_positions` lack an entry for a window
  or chunk the stream enters (before, it skipped it).
- Opening a safetensors file with a malformed header (non-integer shapes or
  offsets, offsets past the data, sizes that don't match the dtype) fails at once,
  naming the tensor.
- `--io auto` finds the disk type for NVMe and other partitions, instead of
  "device type unknown" (the choice made was already right).

### Changed

- `verify` on `--index idx8` output exits 2 ("idx8 output: not supported by
  verify") instead of reporting every unit as FAIL with exit 1.
- `compress` says what it wrote ("wrote 5 safetensors files + config.json").
- `--idx8-block` requires `--index idx8` and refuses 0 at parse time (exit 2).
- `--ram` help notes there is always at least one worker.
- Definition headers say where each definition comes from (19 from
  ComfyUI-DFloat11-Extended's `pattern_dict.py`, 16 from a published release's
  `dfloat11_config`); the ComfyUI `flux`, `chroma` and `chroma-radiance` sources
  record Extended commit 414506d instead of `master` (content unchanged).
- Tests fail instead of passing silently when a tracked fixture is missing or
  unreadable; only absent generated fixtures are skipped, with a SKIP line.
- CI builds the MSRV job with Rust 1.85.1 (the repo's toolchain file had
  overridden it, so 1.85 was never really tested before).
- Documentation brought up to date with the measurements (FINDINGS, DESIGN,
  PLAN, COMPATIBILITY, INDEX_SCHEMES); NORMS.md added.

### Added

- Contributor documentation: CONTRIBUTING, Code of Conduct (Contributor Covenant
  2.1), SECURITY, issue and pull request templates, CITATION.cff, and
  [docs/DEFINITIONS.md](docs/DEFINITIONS.md) for the model definition format.
- `phase0/make_tier0.py --src`, and fixture regeneration steps in `phase0/README.md`.
- CI uses current GitHub actions (Node 24) and pins Ubuntu 24.04.

### Known limits (in addition to 0.1.0's)

- H12 is unmeasured: the CPU decoder behind `verify` is sequential, and there is
  no GPU verification path.
- `config.json` keeps the source model's schema (for example `rope_theta`,
  `torch_dtype`, the source's `transformers_version`), so it differs from the
  official compressor's `save_pretrained` output.
- `verify --level full` peaked at 12.0 GiB of RAM on Qwen3-8B (6.6 GiB to compress).

## [0.1.0] - 2026-09-23

First release.

### Compression

- DFloat11 output byte-identical to the official compressor (pip `dfloat11`
  0.5.0), as whole files: on real Qwen3-8B, all 39 `.safetensors` files match the official output
  and the published `DFloat11/Qwen3-8B-DF11` release, in 40 s and 6.6 GiB of RAM
  against 37 min and 27.3 GiB.
- Four layouts: transformers, diffusers, diffusers single-file, ComfyUI single-file.
- 35 built-in model definitions: every official DFloat11 release with a
  `dfloat11_config`, and every model in ComfyUI-DFloat11-Extended. Custom
  definitions by TOML path or `DF11PACK_ARCH_DIR`.
- Parallel chunked encoder; memory-aware worker count (respects container limits);
  `--ram`, `--workers`; sequential reads for spinning disks.
- Atomic output: a failed or interrupted run never leaves a half-written file.
- Refuses what it cannot do exactly, rather than guessing: non-BF16 sources, and the
  rare ambiguous tie in the official 32-bit code-length limiter.

### Verification

- `--safe`: every unit decoded and checked against the source before it is written.
- `df11pack verify`: structure without the source; sampled or full decode with it;
  exit codes 0 / 1 / 2 for pass / fail / could not check.

### Opt-in extras (clearly marked, never the default)

- `--luts=correct`: zero-fills leftover LUT entries; unreachable by construction,
  and measured so with the official CUDA kernel on the one real leak. *(Corrected
  2026-09-26: in 0.1.0 the flag only wrote the stamp; fixed in 0.1.1.)*
- `--hashes`: per-tensor SHA-256 in the header; `verify` then catches corruption
  without the source.
- `--index idx8`: experimental 8-bit block index with an escape table, 0.66%
  smaller on Qwen3-8B. **Not DFloat11**; no current loader reads it.

### Known limits

- Image and video models have not yet been loaded from df11pack output in ComfyUI or
  diffusers on a GPU (files are byte-identical to what those loaders read).
- 28 of the 35 definitions were checked against real checkpoints (Llama through
  Mistral-Nemo, which shares it). Not checked: Gemma-3, SD3.5 and FLUX.1-dev/Kontext
  diffusers (gated repositories), and the ComfyUI `flux`, `flux2-alt` and
  `zimage-pixel-space` definitions. *(Corrected 2026-09-26; 0.1.0 listed Llama as
  unchecked and omitted the others.)*
- `verify --level full` decodes on one core: 82 s for Qwen3-8B on an 8-vCPU machine
  (measured, `gpu_session/qwen3_8b/results/df11pack_verify.time`), and roughly 10
  minutes for a 12B model on an old dual-core laptop (estimated from ~19M
  weights/s, FINDINGS "Round trip, and cost").
- The legacy DFloat11 0.1.0 format (pickled releases) is not supported.

