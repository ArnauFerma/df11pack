# df11pack

An independent compressor that produces **DFloat11 (DF11)** files: lossless
compression of BF16 model weights, where each weight's 8-bit exponent is
Huffman-coded, the sign and mantissa are stored raw, and decompression runs on
the GPU through a CUDA kernel.

**Status: planning.** There is no implementation yet — only the design and the
ordered plan under [`docs/`](docs/).

## Why

Compatibility with DF11 is at the **output** level, not the code level: the files
produced must be byte-identical to the official compressor's, so existing loaders
and the CUDA kernel work unchanged. The compressor itself is rewritten because
the official one is slow and memory-hungry for implementation reasons:

- it encodes exponents in a Python loop, one symbol at a time (`encode()` calls `.tolist()`) — this, not threading, is the main cause of slowness;
- it loads the entire model into RAM (~48 GB peak reported for 12B Flux);
- compression units are fully independent, and the format already carries everything needed for parallel encoding (`gaps`, `output_positions`), because the decoder is massively parallel. The compressor simply does not use it.

**Guiding principle:** anyone must be able to compress models — including on old
machines, with little RAM, no NVIDIA GPU, and spinning hard drives.

## Scope

- Rust, single static binary, no Python at runtime.
- Both output layouts from the start: ComfyUI-native (single file) and diffusers (directory of shards + `config.json`).
- v1 architectures: Flux and Chroma, both layouts.
- Two modes: *fast* (compress only) and *safe* (verify every unit after encoding), plus a standalone post-hoc verifier.
- The GPU is used only for verification, never for compression.

Non-goals for v1: new codecs or ratio improvements, inference, dtype conversion,
and conversion between layouts.

## Usage

```sh
df11pack architectures                                   # list definitions
df11pack compress model.safetensors --arch flux-comfyui -o out/
df11pack compress model/ --arch qwen3-4b -o out/ --safe   # verify each unit before writing
df11pack verify out/                                     # structure only, no source needed
df11pack verify out/ --source model/ --arch qwen3-4b     # + 1000 sampled chunks vs the source
df11pack verify out/ --source model/ --arch qwen3-4b --level full
```

`verify` exits 0 on pass, 1 when a check fails, 2 when it could not check. Without
`--source` it cannot see a wrong weight value, only broken structure; a sampled run
prints its seed so it can be repeated with `--seed`.

## Documents

- [`docs/DESIGN.md`](docs/DESIGN.md) — the design: format invariants, architecture, checkpoints, verification, test plan.
- [`docs/PLAN.md`](docs/PLAN.md) — ordered implementation plan; Phase 0 and Phase 1 in detail.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — what the output promises, and the one place it deliberately does not.
- [`docs/INDEX_SCHEMES.md`](docs/INDEX_SCHEMES.md) — the swappable index seam, and the `idx8` alternative it is designed to accept.
- [`docs/FINDINGS.md`](docs/FINDINGS.md) — Phase 0 measurements and what they changed.

## Upstream

- Official implementation: <https://github.com/LeanModels/DFloat11> (Apache-2.0)
- ComfyUI integration: <https://github.com/mingyi456/ComfyUI-DFloat11-Extended>

## Licence

Not yet decided — see [`LICENSE`](LICENSE).
