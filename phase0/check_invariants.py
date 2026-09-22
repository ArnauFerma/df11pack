#!/usr/bin/env python3
"""check_invariants.py -- executable checker for the DF11 safetensors format.

Validates every invariant listed in docs/DESIGN.md section 1.3 ("Format
invariants we must respect") against one or more DF11-compressed
safetensors files, WITHOUT loading whole tensors into memory and WITHOUT
importing torch. Parses the safetensors header directly (8-byte
little-endian header length + JSON) and reads tensor payloads by seek/read,
one small tensor at a time.

Ground truth for the exact formulas below was cross-checked against:
  - docs/DESIGN.md section 1.3
  - phase0/env/lib/python3.12/site-packages/dfloat11/dfloat11_utils.py
    (encode, encode_weights, get_luts -- the official encoder)
  - a real downloaded shard (DFloat11/Qwen3-4B-DF11, model_layers_0.safetensors)

Two invariants in the task's own prompt paraphrase turned out to disagree
with the actual official encoder (see phase0/INVARIANTS_RESULTS.md for the
full writeup):
  - `split_positions`' last value is NOT the total weight count. The
    encoder computes `cumsum(sizes)[:-1]`, i.e. only the *interior* split
    points between concatenated tensors -- the final cumulative total
    (which would equal the weight count) is deliberately dropped. Empty
    for a single-tensor unit.
  - `output_positions`' trailing value is NOT the encoded bitstream byte
    length. It is `len(data)`, the total number of exponent *symbols*
    (= weight count), which equals `sign_mantissa`'s length. It is an
    element index, not a byte offset.

This checker implements the corrected (source-accurate) invariants, and
flags the file as failing if the WRONG version happens to hold instead
(which would itself indicate the file is not a well-formed DF11 unit, or
that the encoder that produced it silently violated its own convention).

Usage:
    check_invariants.py FILE_OR_DIR [FILE_OR_DIR ...]

Exit code is 0 iff every unit in every file passes every invariant.
"""
import json
import math
import os
import sys
import struct
from collections import defaultdict, namedtuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants from the official encoder (dfloat11_utils.py / dfloat11.py, v0.5.0)
# ---------------------------------------------------------------------------
BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512
WINDOW_BITS = 8 * BYTES_PER_THREAD          # 64 bits per gaps window
CHUNK_BYTES = BYTES_PER_THREAD * THREADS_PER_BLOCK  # 4096 bytes per output_positions chunk
GAP_BITS = 5                                # bits packed per gaps window
MAX_CODE_LEN = 32                           # kernel/gaps hard limit
MAX_PREFIX_TABLES = 16                      # luts jump byte range (240..255) -> <=16 targets
JUMP_THRESHOLD = 240
INT32_MAX = 2 ** 31 - 1

UNIT_SUFFIXES = (
    'luts', 'encoded_exponent', 'sign_mantissa',
    'output_positions', 'gaps', 'split_positions',
)

EXPECTED_DTYPE = {
    'luts': 'U8',
    'encoded_exponent': 'U8',
    'sign_mantissa': 'U8',
    'output_positions': 'U8',
    'gaps': 'U8',
    'split_positions': 'I64',
}


class Failure(Exception):
    """A single invariant violation. Carries (invariant, unit, detail)."""

    def __init__(self, invariant, unit, detail):
        self.invariant = invariant
        self.unit = unit
        self.detail = detail
        super().__init__(f"[{invariant}] unit={unit!r}: {detail}")


Result = namedtuple("Result", ["file", "unit", "ok", "failures"])


# ---------------------------------------------------------------------------
# safetensors header parsing (no torch, no safetensors library -- direct)
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: file too short to contain a safetensors header length")
        (hlen,) = struct.unpack("<Q", raw_len)
        raw = f.read(hlen)
        if len(raw) != hlen:
            raise ValueError(f"{path}: truncated header (declared {hlen} bytes, got {len(raw)})")
        header = json.loads(raw)
    header.pop("__metadata__", None)
    data_start = 8 + hlen
    return header, data_start


def read_tensor_bytes(path, data_start, info):
    s, e = info["data_offsets"]
    n = e - s
    with open(path, "rb") as f:
        f.seek(data_start + s)
        buf = f.read(n)
    if len(buf) != n:
        raise ValueError(f"short read for tensor: wanted {n} bytes, got {len(buf)}")
    return buf


def group_units(header):
    """Group tensor keys into DF11 units by their common prefix."""
    units = defaultdict(dict)
    others = []
    for key, info in header.items():
        if "." in key:
            prefix, suffix = key.rsplit(".", 1)
        else:
            prefix, suffix = "", key
        if suffix in UNIT_SUFFIXES:
            units[prefix][suffix] = info
        else:
            others.append(key)
    return units, others


# ---------------------------------------------------------------------------
# Per-unit validation
# ---------------------------------------------------------------------------
def check_unit(path, data_start, unit_name, tensors, failures):
    """tensors: dict suffix -> header info (dtype/shape/data_offsets)."""

    def fail(inv, detail):
        failures.append(Failure(inv, unit_name, detail))

    # --- INV-NAMES: all six present, with the right dtype/rank -------------
    missing = [s for s in UNIT_SUFFIXES if s not in tensors]
    if missing:
        fail("INV-NAMES-COMPLETE", f"missing tensor(s) {missing} (present: {sorted(tensors)})")
        # Can't safely continue: bail out for this unit.
        return

    for suf in UNIT_SUFFIXES:
        info = tensors[suf]
        want_dtype = EXPECTED_DTYPE[suf]
        if info["dtype"] != want_dtype:
            fail("INV-NAMES-DTYPE", f"{unit_name}.{suf} has dtype {info['dtype']!r}, expected {want_dtype!r}")

    luts_info = tensors["luts"]
    ee_info = tensors["encoded_exponent"]
    sm_info = tensors["sign_mantissa"]
    op_info = tensors["output_positions"]
    gaps_info = tensors["gaps"]
    sp_info = tensors["split_positions"]

    if len(luts_info["shape"]) != 2 or luts_info["shape"][1] != 256:
        fail("INV-NAMES-SHAPE", f"luts shape {luts_info['shape']} is not (n_prefixes+1, 256)")
        return
    if len(ee_info["shape"]) != 1:
        fail("INV-NAMES-SHAPE", f"encoded_exponent shape {ee_info['shape']} is not rank-1")
        return
    if len(sm_info["shape"]) != 1:
        fail("INV-NAMES-SHAPE", f"sign_mantissa shape {sm_info['shape']} is not rank-1")
        return
    if len(op_info["shape"]) != 1:
        fail("INV-NAMES-SHAPE", f"output_positions shape {op_info['shape']} is not rank-1")
        return
    if len(gaps_info["shape"]) != 1:
        fail("INV-NAMES-SHAPE", f"gaps shape {gaps_info['shape']} is not rank-1")
        return
    if len(sp_info["shape"]) != 1:
        fail("INV-NAMES-SHAPE", f"split_positions shape {sp_info['shape']} is not rank-1")
        return

    n_bytes = ee_info["shape"][0]           # bitstream length, bytes
    n_elements = sm_info["shape"][0]        # weight count (canonical source #1)
    op_bytes = op_info["shape"][0]
    gaps_bytes = gaps_info["shape"][0]
    n_prefixes = luts_info["shape"][0] - 1  # luts rows minus the trailing "lens" row

    # --- INV-LIMIT: 2**31-1 caps on weights/bytes per unit ------------------
    if n_elements > INT32_MAX:
        fail("INV-LIMIT-WEIGHTS", f"{n_elements} weights exceeds 2**31-1 ({INT32_MAX})")
    if n_bytes > INT32_MAX:
        fail("INV-LIMIT-BYTES", f"{n_bytes} bitstream bytes exceeds 2**31-1 ({INT32_MAX})")

    # --- INV-LUTS-SHAPE: n_prefixes bounds -----------------------------------
    if n_prefixes < 1:
        fail("INV-LUTS-SHAPE", f"luts has {n_prefixes} prefix table rows (need >= 1)")
    if n_prefixes > MAX_PREFIX_TABLES:
        fail("INV-LUTS-SHAPE", f"luts has {n_prefixes} prefix tables, exceeds max {MAX_PREFIX_TABLES}")

    # --- Read the small tensors (luts, output_positions, gaps, split_positions) ---
    luts_raw = read_tensor_bytes(path, data_start, luts_info)
    luts = np.frombuffer(luts_raw, dtype=np.uint8).reshape(luts_info["shape"])

    op_raw = read_tensor_bytes(path, data_start, op_info)
    if op_bytes % 4 != 0:
        fail("INV-OUTPOS-DTYPE", f"output_positions byte length {op_bytes} is not a multiple of 4 (uint32 view)")
        op_u32 = None
    else:
        op_u32 = np.frombuffer(op_raw, dtype='<u4')

    gaps_raw = read_tensor_bytes(path, data_start, gaps_info)
    gaps_packed = np.frombuffer(gaps_raw, dtype=np.uint8)

    sp_raw = read_tensor_bytes(path, data_start, sp_info)
    split_positions = np.frombuffer(sp_raw, dtype='<i8')

    # --- INV-LUTS-JUMP: >=240 jump convention well-formed --------------------
    if n_prefixes >= 1:
        table_rows = luts[:n_prefixes]  # exclude the trailing "lens" row
        jump_mask = table_rows >= JUMP_THRESHOLD
        if jump_mask.any():
            rows, cols = np.nonzero(jump_mask)
            for r, c in zip(rows.tolist(), cols.tolist()):
                v = int(table_rows[r, c])
                target = 256 - v
                if not (0 <= target < n_prefixes):
                    fail(
                        "INV-LUTS-JUMP",
                        f"luts[{r},{c}] = {v} (jump-to-LUT convention) targets table {target}, "
                        f"but only tables 0..{n_prefixes - 1} exist",
                    )

        # --- INV-GAPS-MAXLEN: max Huffman code length <= 32 bits, via the
        # trailing "lens" row of luts (see module docstring for why this is
        # checked here rather than by decoding a literal gap value: a gaps
        # window is packed as a fixed 5-bit field, so it can only ever
        # decode to 0..31 -- it is structurally incapable of representing a
        # violation. The *cause* the format cares about -- max code length
        # > 32 bits -- is checked directly here instead.) ---------------------
        lens_row = luts[-1]
        max_len = int(lens_row.max())
        if max_len > MAX_CODE_LEN:
            bad_symbols = np.nonzero(lens_row > MAX_CODE_LEN)[0].tolist()
            fail(
                "INV-GAPS-MAXLEN",
                f"luts lens row has max code length {max_len} > {MAX_CODE_LEN} "
                f"(symbols {bad_symbols[:10]}{'...' if len(bad_symbols) > 10 else ''})",
            )

    # --- INV-OUTPOS-COUNT / MONO / TOTAL -------------------------------------
    if op_u32 is not None:
        expected_chunks = math.ceil(n_bytes / CHUNK_BYTES) if n_bytes > 0 else 0
        expected_count = expected_chunks + 1
        actual_count = len(op_u32)
        if actual_count != expected_count:
            fail(
                "INV-OUTPOS-COUNT",
                f"output_positions has {actual_count} entries, expected "
                f"ceil(n_bytes={n_bytes} / {CHUNK_BYTES}) + 1 = {expected_count}",
            )

        if actual_count >= 2:
            diffs = np.diff(op_u32.astype(np.int64))
            if (diffs < 0).any():
                bad_idx = int(np.nonzero(diffs < 0)[0][0])
                fail(
                    "INV-OUTPOS-MONO",
                    f"output_positions is not non-decreasing at index {bad_idx}: "
                    f"{int(op_u32[bad_idx])} -> {int(op_u32[bad_idx + 1])}",
                )

        if actual_count >= 1:
            trailing = int(op_u32[-1])
            if trailing != n_elements:
                fail(
                    "INV-OUTPOS-TOTAL",
                    f"output_positions trailing value {trailing} != sign_mantissa length "
                    f"(total weight count) {n_elements} "
                    f"(note: trailing value is an ELEMENT count, not the encoded byte length "
                    f"{n_bytes} -- see module docstring)",
                )

    # --- INV-GAPS-SHAPE: 5 bits/window, padded to multiple of 512 windows ---
    blocks = math.ceil(n_bytes / CHUNK_BYTES) if n_bytes > 0 else 0
    expected_windows = THREADS_PER_BLOCK * blocks
    expected_gaps_bytes = math.ceil(expected_windows * GAP_BITS / 8)
    if gaps_bytes != expected_gaps_bytes:
        fail(
            "INV-GAPS-SHAPE",
            f"gaps is {gaps_bytes} bytes, expected ceil({expected_windows} windows * "
            f"{GAP_BITS} bits / 8) = {expected_gaps_bytes} "
            f"(windows padded to a multiple of {THREADS_PER_BLOCK})",
        )
    else:
        # Every window value is structurally < 32 given 5-bit packing; still
        # decode and assert as a sanity check on our own unpacking logic.
        bits = np.unpackbits(gaps_packed)
        usable_bits = expected_windows * GAP_BITS
        if len(bits) >= usable_bits and expected_windows > 0:
            grouped = bits[:usable_bits].reshape(-1, GAP_BITS)
            weights = (1 << np.arange(GAP_BITS - 1, -1, -1)).astype(np.uint8)
            vals = grouped.astype(np.uint16) @ weights
            if vals.max() >= 32:
                bad = int(np.nonzero(vals >= 32)[0][0])
                fail("INV-GAPS-VALUE", f"gaps window {bad} decodes to {int(vals[bad])} (must be < 32)")

    # --- INV-SPLIT: int64, strictly increasing, last < total, consistent ----
    if len(split_positions) > 0:
        diffs = np.diff(split_positions)
        if (diffs <= 0).any():
            bad_idx = int(np.nonzero(diffs <= 0)[0][0])
            fail(
                "INV-SPLIT-MONO",
                f"split_positions not strictly increasing at index {bad_idx}: "
                f"{int(split_positions[bad_idx])} -> {int(split_positions[bad_idx + 1])}",
            )
        if int(split_positions[0]) <= 0:
            fail("INV-SPLIT-MONO", f"split_positions[0] = {int(split_positions[0])} must be > 0")
        last = int(split_positions[-1])
        if not (last < n_elements):
            fail(
                "INV-SPLIT-BOUND",
                f"split_positions last value {last} must be strictly less than the total "
                f"weight count {n_elements} (split_positions holds only the INTERIOR split "
                f"points -- cumsum(sizes)[:-1] -- it deliberately excludes the final total; "
                f"see module docstring)",
            )

    # --- INV-SM: sign_mantissa is one byte per weight (cross-checked against
    # output_positions' independently-encoded weight count, not tautological) ---
    if op_u32 is not None and len(op_u32) >= 1:
        op_total = int(op_u32[-1])
        if n_elements != op_total:
            fail(
                "INV-SM-LENGTH",
                f"sign_mantissa has {n_elements} bytes (1 per weight) but output_positions' "
                f"trailing total says {op_total} weights",
            )


def check_file(path, results):
    header, data_start = read_header(path)
    units, others = group_units(header)
    if not units:
        print(f"  (no DF11 units found in {path}; {len(others)} uncompressed tensor(s))")
        return
    for unit_name in sorted(units):
        failures = []
        check_unit(path, data_start, unit_name, units[unit_name], failures)
        results.append(Result(path, unit_name, len(failures) == 0, failures))


def iter_input_files(args):
    for a in args:
        if os.path.isdir(a):
            for name in sorted(os.listdir(a)):
                if name.endswith(".safetensors"):
                    yield os.path.join(a, name)
        else:
            yield a


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    results = []
    for path in iter_input_files(argv):
        print(f"Checking {path} ...")
        try:
            check_file(path, results)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(Result(path, "<file>", False, [Failure("INV-FILE", "<file>", str(e))]))

    n_units = len(results)
    n_ok = sum(1 for r in results if r.ok)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.file} :: {r.unit}")
        for f in r.failures:
            print(f"    {f}")

    print(f"\n{n_ok}/{n_units} units passed.")
    return 0 if n_ok == n_units and n_units > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
