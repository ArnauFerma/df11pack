# Changelog

All notable changes. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) — before 1.0, minor
versions may change the command line.

## [Unreleased]

### Added

- Contributor documentation: CONTRIBUTING, Code of Conduct (Contributor Covenant
  2.1), SECURITY, issue and pull request templates, CITATION.cff, and
  [docs/DEFINITIONS.md](docs/DEFINITIONS.md) for the model definition format.
- `phase0/make_tier0.py --src`, and fixture regeneration steps in `phase0/README.md`.

### Changed

- CI uses current GitHub actions (Node 24) and pins Ubuntu 24.04.

## [0.1.0] - 2026-09-23

First release.

### Compression

- DFloat11 output byte-identical to the official compressor (pip `dfloat11`
  0.5.0), as whole files: on real Qwen3-8B, all 39 files match the official output
  and the published `DFloat11/Qwen3-8B-DF11` release, in 40 s and 6.6 GB of RAM
  against 37 min and 27.3 GB.
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

- `--luts=correct`: zero-fills leftover LUT entries; proven unreachable by the
  official CUDA kernel.
- `--hashes`: per-tensor SHA-256 in the header; `verify` then catches corruption
  without the source.
- `--index idx8`: experimental 8-bit block index with an escape table, 0.66%
  smaller on Qwen3-8B. **Not DFloat11**; no current loader reads it.

### Known limits

- Image and video models have not yet been loaded from df11pack output in ComfyUI or
  diffusers on a GPU (files are byte-identical to what those loaders read).
- Gated repositories (Gemma-3, SD3.5, FLUX.1-dev diffusers, Llama) were not checked
  against real checkpoints.
- `verify --level full` decodes on one core (~10 minutes for a 12B model).
- The legacy DFloat11 0.1.0 format (pickled releases) is not supported.

