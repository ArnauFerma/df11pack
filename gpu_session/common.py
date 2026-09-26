"""common.py -- shared helpers for the gpu_session/ scripts.

No torch, no cupy at import time (kept importable even before those are
installed, so --help / argument errors surface without needing the venv).
Only numpy is imported eagerly; every gpu_session script installs numpy
before running any test.

This module is deliberately dependency-light and copies (rather than
imports) the tiny safetensors-header parser already proven in
phase0/check_invariants.py, so each gpu_session script stays runnable with
only this file next to it plus the standard library and numpy -- no
phase0/ import path juggling required on the rented box.
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

UNIT_SUFFIXES = (
    "luts", "encoded_exponent", "sign_mantissa",
    "output_positions", "gaps", "split_positions",
)

THREADS_PER_BLOCK = (512,)
BYTES_PER_THREAD = 8


# ---------------------------------------------------------------------------
# safetensors header parsing -- no torch, no safetensors library
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    data_start = 8 + hlen
    return header, data_start


def read_tensor_bytes(path, data_start, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + s)
        buf = f.read(e - s)
    if len(buf) != e - s:
        raise IOError(f"short read for tensor in {path}: wanted {e - s}, got {len(buf)}")
    return buf


def group_units(header):
    """Group tensor keys into DF11 units by their common prefix.
    Returns (units: {prefix: {suffix: info}}, others: [key, ...])."""
    units = {}
    others = []
    for key, info in header.items():
        if "." in key:
            prefix, suffix = key.rsplit(".", 1)
        else:
            prefix, suffix = "", key
        if suffix in UNIT_SUFFIXES:
            units.setdefault(prefix, {})[suffix] = info
        else:
            others.append(key)
    return units, others


def load_unit_raw(path, prefix, header=None, data_start=None):
    """Return {suffix: raw_bytes} for the five kernel-input tensors of one
    unit (luts, encoded_exponent, sign_mantissa, output_positions, gaps),
    read straight off disk as raw bytes -- exactly what load_file() would
    hand the kernel via .data_ptr(), with no dtype reinterpretation."""
    if header is None:
        header, data_start = read_header(path)
    units, _ = group_units(header)
    if prefix not in units:
        raise KeyError(f"no unit with prefix {prefix!r} in {path}; found {sorted(units)}")
    tensors = units[prefix]
    needed = ("luts", "encoded_exponent", "sign_mantissa", "output_positions", "gaps")
    missing = [s for s in needed if s not in tensors]
    if missing:
        raise KeyError(f"unit {prefix!r} in {path} is missing {missing}")
    return {suf: read_tensor_bytes(path, data_start, tensors[suf]) for suf in needed}, tensors


def only_unit_prefix(header):
    """Convenience for single-unit fixture files: return the one unit
    prefix present, erroring loudly if there isn't exactly one."""
    units, _ = group_units(header)
    if len(units) != 1:
        raise ValueError(f"expected exactly one DF11 unit, found {len(units)}: {sorted(units)}")
    return next(iter(units))


# ---------------------------------------------------------------------------
# ground truth: the unit's source BF16 weights, concatenated in pattern order
# ---------------------------------------------------------------------------
def source_weights_bytes(unit_file, prefix, unit_bytes, split_positions_bytes, source):
    """Return the raw BF16 bytes the unit must decode to: the source tensors,
    flattened and concatenated in the order the unit's pattern_dict lists them
    (the order dfloat11's compress_model uses), read from `source` (a
    .safetensors file or a directory of them). The pattern_dict is read from
    the config.json next to `unit_file`.

    Returns None when `source` does not exist, so a caller can still run
    without it. Raises when the source exists but does not match the unit
    (sizes, split_positions or sign/mantissa bytes differ): that means the
    wrong source, not a decode result."""
    import re

    if not os.path.exists(source):
        return None
    with open(os.path.join(os.path.dirname(os.path.abspath(unit_file)), "config.json")) as f:
        pattern_dict = json.load(f)["dfloat11_config"]["pattern_dict"]
    matches = [p for p in pattern_dict if re.fullmatch(p, prefix)]
    if len(matches) != 1:
        raise ValueError(f"unit {prefix!r} matches {len(matches)} patterns in config.json: {matches}")
    attrs = pattern_dict[matches[0]]
    keys = [f"{prefix}.{a}.weight" for a in attrs] if attrs else [f"{prefix}.weight"]

    files = ([source] if os.path.isfile(source) else
             sorted(os.path.join(source, n) for n in os.listdir(source) if n.endswith(".safetensors")))
    where = {}
    for path in files:
        header, data_start = read_header(path)
        for k in keys:
            if k in header:
                where[k] = (path, data_start, header[k])
    missing = [k for k in keys if k not in where]
    if missing:
        raise KeyError(f"source {source} lacks {missing}")

    parts = []
    for k in keys:
        path, data_start, info = where[k]
        if info["dtype"] != "BF16":
            raise ValueError(f"source tensor {k} is {info['dtype']}, not BF16")
        parts.append(read_tensor_bytes(path, data_start, info))
    truth = b"".join(parts)

    n_elements = len(unit_bytes["sign_mantissa"])
    if len(truth) != 2 * n_elements:
        raise ValueError(f"source tensors hold {len(truth) // 2} weights, the unit {n_elements}")
    sizes = np.array([len(b) // 2 for b in parts], dtype=np.int64)
    split = np.frombuffer(split_positions_bytes, dtype="<i8")
    if not np.array_equal(np.cumsum(sizes)[:-1], split):
        raise ValueError(f"source tensor sizes {sizes.tolist()} do not give the unit's "
                         f"split_positions {split.tolist()} (wrong order or wrong tensors)")
    w = np.frombuffer(truth, dtype="<u2")
    sign_mantissa = (((w >> 8) & 0x80) | (w & 0x7F)).astype(np.uint8)
    if sign_mantissa.tobytes() != unit_bytes["sign_mantissa"]:
        raise ValueError("source weights' sign/mantissa bytes differ from the unit's sign_mantissa")
    return truth


# ---------------------------------------------------------------------------
# kernel launch geometry (shared by cupy and driver-API paths)
# ---------------------------------------------------------------------------
def launch_geometry(luts_bytes, encoded_bytes, output_positions_bytes):
    """Returns (n_luts, n_bytes, grid, block, shared_mem_size)
    using the exact formulas dfloat11.py uses at inference and at
    check_correctness time."""
    n_luts = len(luts_bytes) // 256
    n_bytes = len(encoded_bytes)
    op_u32 = np.frombuffer(output_positions_bytes, dtype="<u4")
    deltas = op_u32[1:].astype(np.int64) - op_u32[:-1].astype(np.int64)
    shared_mem_size = THREADS_PER_BLOCK[0] * 4 + 4 + int(deltas.max()) * 2
    block = THREADS_PER_BLOCK
    grid = (math.ceil(n_bytes / (block[0] * BYTES_PER_THREAD)),)
    # n_elements = number of decoded weights = len(sign_mantissa) bytes,
    # caller supplies it directly (it's just len(sign_mantissa_bytes)).
    return n_luts, n_bytes, grid, block, shared_mem_size


def find_ptx_path():
    """Locate decode.ptx inside the installed dfloat11 package, the same
    way dfloat11.py itself does (pkg_resources.resource_filename)."""
    import pkg_resources
    return pkg_resources.resource_filename("dfloat11", "decode.ptx")


def decode_with_cupy(unit_bytes, n_elements):
    """Decode one unit's five kernel-input tensors with real CuPy --
    RawModule + RawKernel over decode.ptx, exactly the call dfloat11.py's
    own get_hook()/compress_model() make. Returns the raw output bytes
    (n_elements * 2, bf16 bit patterns, uninterpreted)."""
    import cupy as cp

    ptx_path = find_ptx_path()
    module = cp.RawModule(path=ptx_path)
    decode_fn = module.get_function("decode")

    n_luts, n_bytes, grid, block, shared_mem = launch_geometry(
        unit_bytes["luts"], unit_bytes["encoded_exponent"], unit_bytes["output_positions"],
    )

    d_luts = cp.asarray(np.frombuffer(unit_bytes["luts"], dtype=np.uint8))
    d_encoded = cp.asarray(np.frombuffer(unit_bytes["encoded_exponent"], dtype=np.uint8))
    d_sign_mantissa = cp.asarray(np.frombuffer(unit_bytes["sign_mantissa"], dtype=np.uint8))
    d_output_positions = cp.asarray(np.frombuffer(unit_bytes["output_positions"], dtype=np.uint8))
    d_gaps = cp.asarray(np.frombuffer(unit_bytes["gaps"], dtype=np.uint8))
    d_out = cp.empty(n_elements * 2, dtype=cp.uint8)

    decode_fn(grid=grid, block=block, shared_mem=shared_mem, args=[
        d_luts.data.ptr, d_encoded.data.ptr, d_sign_mantissa.data.ptr,
        d_output_positions.data.ptr, d_gaps.data.ptr, d_out.data.ptr,
        n_luts, n_bytes, n_elements,
    ])
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(d_out).tobytes()


# ---------------------------------------------------------------------------
# result reporting
# ---------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def write_result(out_path, item, status, summary, detail, elapsed_seconds):
    """status must be one of CONFIRMED / REFUTED / ERROR."""
    assert status in ("CONFIRMED", "REFUTED", "ERROR"), status
    result = {
        "item": item,
        "status": status,
        "summary": summary,
        "detail": detail,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": now_iso(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[{item}] {status}: {summary}")
    print(f"[{item}] elapsed: {elapsed_seconds:.1f}s   result: {out_path}")
    return result


def default_out_path(item_name):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "out", f"{item_name}.json")


def bytes_equal(a, b):
    return a == b


def sha256_hex(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def gpu_info():
    """Best-effort nvidia-smi summary; never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception as e:
        return f"<nvidia-smi unavailable: {e}>"


def resolved_versions():
    """Best-effort import + __version__ probe for the packages this
    session cares about. Never raises; missing packages are reported as
    such rather than crashing the caller."""
    out = {}
    for mod in ("torch", "cupy", "dfloat11", "transformers", "safetensors", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            out[mod] = f"<not importable: {e.__class__.__name__}: {e}>"
    return out
