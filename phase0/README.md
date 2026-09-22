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

Install remaining packages:

```bash
phase0/env/bin/pip install numpy safetensors 'dahuffman==0.4.2' transformers
```

Install the official compressor:

```bash
phase0/env/bin/pip install dfloat11
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
