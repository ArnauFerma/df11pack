# df11pack

A fast, independent compressor for **[DFloat11](https://github.com/LeanModels/DFloat11)**:
lossless compression of BF16 model weights to about 70% of their size, decoded on
the GPU by the official CUDA kernel.

df11pack writes **the same files the official compressor writes** — byte for byte —
so every existing DFloat11 loader works unchanged: `DFloat11Model` for LLMs and
diffusers, and [ComfyUI-DFloat11-Extended](https://github.com/mingyi456/ComfyUI-DFloat11-Extended)
for ComfyUI. It is just much faster and lighter, and needs no GPU and no Python.

## Measured

Qwen3-8B, compressed on the same 8-vCPU machine:

| | official compressor | **df11pack** |
|---|---|---|
| time | 37 min | **40 s** |
| peak RAM | 27.3 GB | **6.6 GB** |
| output | 39 files, 11.16 GB | **the same 39 files, byte-identical** |

All 39 files are also byte-identical to the published
[`DFloat11/Qwen3-8B-DF11`](https://huggingface.co/DFloat11/Qwen3-8B-DF11) release
(SHA-256 compared file by file). On an old dual-core laptop with no GPU, Qwen3-0.6B
takes ~13 s against the official 13 min. Details and every other measurement:
[`docs/FINDINGS.md`](docs/FINDINGS.md), raw run data in
[`gpu_session/qwen3_8b/results/`](gpu_session/qwen3_8b/results/).

## Install

Download the archive for your platform from [**Releases**](https://github.com/ArnauFerma/df11pack/releases), unpack, and run
`df11pack`. One static binary; nothing else to install. Linux (x86_64, aarch64),
Windows (x86_64) and macOS (Apple Silicon).

Or build from source (Rust 1.85+): `cargo build --release -p df11pack`.

## Use

```sh
df11pack architectures                      # what it can compress
```

**ComfyUI** — a single `.safetensors` for ComfyUI-DFloat11-Extended:

```sh
df11pack compress flux1-schnell.safetensors --arch flux-schnell-comfyui -o out/
# -> out/model.safetensors
```

**diffusers** — a directory `DFloat11Model.from_pretrained` loads:

```sh
df11pack compress FLUX.1-schnell/transformer/ --arch flux-dev-diffusers -o out/
```

**LLMs (transformers)**:

```sh
df11pack compress Qwen3-8B/ --arch qwen3-8b -o Qwen3-8B-DF11/
```

Useful options:

- `--safe` decodes every unit and checks it against the source **before** writing it;
- `--ram 4G` caps memory (by default it sizes itself to what is free);
- `--io sequential` for spinning hard drives (auto-detected on Linux).

**Check any output**, including ones you did not make:

```sh
df11pack verify out/                                        # structure, no source needed
df11pack verify out/ --source model/ --arch qwen3-8b        # + 1000 sampled chunks decoded
df11pack verify out/ --source model/ --arch qwen3-8b --level full
```

`verify` exits 0 on pass, 1 when a check fails, 2 when it could not check.

## Models

35 definitions ship built in (`df11pack architectures`):

- **every official DFloat11 release** with a `dfloat11_config` — 40 of the 45 in the
  DFloat11 Hugging Face organisation: Qwen3, Qwen2.5, Llama, Mistral, Phi-4, Gemma-3,
  DeepSeek-R1 distills, FLUX.1 (dev/schnell/Kontext/Krea/Fill/Depth/Canny), Chroma,
  HiDream, Qwen-Image, Wan 2.1/2.2, SD3.5, OmniGen2, BAGEL;
- **every model in ComfyUI-DFloat11-Extended** (ComfyUI files): Flux, Flux.2,
  Chroma, Chroma-Radiance, Qwen-Image 2.1, Z-Image, Lumina 2, Cosmos-Predict2,
  Anima, ERNIE-Image, LongCat, Ovis, Krea-2, Lens, SDXL, ACE-Step 1.5.

A new model is a small TOML file; pass its path to `--arch`, or put it in a folder
named by `DF11PACK_ARCH_DIR`. Format and how to add one: [docs/DEFINITIONS.md](docs/DEFINITIONS.md).

## What is checked, and what is not

- Byte-identity with the official compressor is tested per architecture, on small
  models run through the official tool, and on real Qwen3 models (0.6B, 8B).
- Every definition was checked against the **tensor names of real checkpoints and
  real published DF11 files**; 23 of 28 match exactly, and the other 5 have
  sources in F32/F16, which df11pack refuses (convert to BF16 first). Gated
  repositories (Gemma-3, SD3.5, FLUX.1-dev diffusers, Llama) were not checked.
- Not yet done: loading df11pack's output in ComfyUI on a GPU for the image models.
  Since the files are byte-identical to what those loaders already read, it is
  expected to work; reports welcome.

Beyond the official format, all opt-in and clearly marked:

- `--luts=correct` — zero-fills LUT entries the official encoder leaves as leftovers;
  verified unreachable by the kernel ([`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)).
- `--hashes` — per-tensor SHA-256 in the header, so `verify` catches corruption
  without the source; tensor bytes unchanged.
- `--index idx8` — an experimental 8-bit block index, 0.66% smaller. **Not
  DFloat11**: no current loader reads it ([`docs/INDEX_SCHEMES.md`](docs/INDEX_SCHEMES.md)).

## Documents

- [docs/DEFINITIONS.md](docs/DEFINITIONS.md): the model definition format.
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md): what the output promises.
- [docs/FINDINGS.md](docs/FINDINGS.md): every measurement, and what each changed.
- [docs/DESIGN.md](docs/DESIGN.md), [docs/PLAN.md](docs/PLAN.md): the original design and build plan.
- [CHANGELOG.md](CHANGELOG.md).

## Contributing

Testing a model on a GPU and reporting how it went helps most right now. See
[CONTRIBUTING.md](CONTRIBUTING.md); questions go to
[Discussions](https://github.com/ArnauFerma/df11pack/discussions), vulnerabilities to
[SECURITY.md](SECURITY.md). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

MIT — see [`LICENSE`](LICENSE). Free to use, modify and redistribute; keep the
copyright notice.

df11pack reimplements the DFloat11 file format and its codebook construction from
the official implementation (Apache-2.0, <https://github.com/LeanModels/DFloat11>,
by the DFloat11 authors) and reads model definitions derived from
ComfyUI-DFloat11-Extended (by mingyi456). It contains no code from either and does
not redistribute `decode.ptx`. All credit for the format goes to them.
