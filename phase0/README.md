# Phase 0: Reference Environment

This environment runs the OFFICIAL DFloat11 compressor, the reference implementation the project is graded against.

## Host Requirements

- Linux
- Python 3.12.3
- 3 GB RAM
- ~39 GB free disk space
- No NVIDIA GPU

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv phase0/env
source phase0/env/bin/activate
```

Upgrade pip:

```bash
phase0/env/bin/pip install --upgrade pip
```

Install PyTorch (CPU-only index is essential—the default pulls several GB of unused CUDA wheels):

```bash
phase0/env/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Install everything else at the exact versions used (the frozen environment):

```bash
phase0/env/bin/pip install -r phase0/requirements.txt
```

## Pinned Versions

Record these exactly; different dahuffman or dfloat11 versions silently change what "byte-identical" means:

- Python 3.12.3
- torch 2.14.0+cpu
- numpy 2.5.3
- safetensors 0.8.0
- dahuffman 0.4.2
- transformers 5.17.0
- dfloat11 0.5.0

Note: cupy is deliberately NOT installed.

## The cupy Stub

The official `dfloat11` package imports cupy at module scope and builds a `RawModule` from decode.ptx at import time. This makes the package unimportable without cupy, even for compression (pure CPU work). Installing `cupy-cuda12x` would waste ~2.5 GB of CUDA wheels on a machine with no GPU.

Instead, `phase0/shim/cupy.py` is a stub that satisfies the import and raises on any real use. This is stronger than merely cost-saving: if a compression run completes with the stub on `sys.path`, that run provably never touched the GPU. A real use would fail loudly instead.

Import `phase0/official.py` instead of `dfloat11` directly. It places the stub on `sys.path` first and asserts that the stub (not a real cupy) was imported:

```python
import sys; sys.path.insert(0, "phase0")
import official
official.compress_model(...)   # keep check_correctness=False on this machine
```

Anything requiring actual cupy—inference, `check_correctness=True`, later GPU verification—must run on a machine with the real package installed.

## Regenerating the fixtures

The byte-identity tests read official outputs listed in `fixtures/MANIFEST.json`
(paths relative to the repo root, sha256 per tensor). From the repo root, with the
environment above:

```bash
PY=phase0/env/bin/python
hf download Qwen/Qwen3-0.6B --local-dir phase0/models/qwen3-0.6b  # ~1.5 GB
$PY phase0/make_tier0.py --src phase0/models/qwen3-0.6b          # 4-layer tier 0
$PY phase0/run_official.py --model phase0/corpus/tier0/qwen3-trunc \
    --out phase0/out/official/qwen3-trunc-layers-only-dir         # ~2 min
$PY phase0/make_synthetic.py                                     # stand-ins, ~7 min
$PY phase0/freeze_fixtures.py                                    # rewrite MANIFEST.json
```

Optional: tier 1 is the full Qwen3-0.6B (`run_official.py --model phase0/models/qwen3-0.6b
--out phase0/out/official/qwen3-0.6b-layers-only`, ~13 min); `freeze_fixtures.py`
expects its source under `../bf16-exponent-compression/real_model`. Sets that are
absent are skipped.

Caveat: the four original stand-ins (`flux-comfyui`, `chroma-comfyui`,
`flux-dev-diffusers`, `chroma-diffusers`) were generated with a per-process random
seed, so regenerating them gives different (equally valid) weights, and the one test
that pins a position inside `synthetic-flux-comfyui` will need updating. They are
only regenerated when named explicitly.

`fetch_real_headers.py`, `import_extended.py` and `import_official_releases.py`
refresh the small committed fixtures from the network (headers and configs only).
