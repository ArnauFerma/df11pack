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
# kernel launch geometry (shared by cupy and driver-API paths)
# ---------------------------------------------------------------------------
def launch_geometry(luts_bytes, encoded_bytes, output_positions_bytes):
    """Returns (n_luts, n_bytes, n_elements, grid, block, shared_mem_size)
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
