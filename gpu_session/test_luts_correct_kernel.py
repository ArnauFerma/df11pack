#!/usr/bin/env python3
"""test_luts_correct_kernel.py -- the --luts=correct gate test.

docs/COMPATIBILITY.md and FINDINGS.md 0.5 describe a real bug in the
official get_luts(): a prefix table after the first that lacks key 0 is
filled, at its low end, with the PREVIOUS table's trailing value rather
than 0 -- "leaked" bytes that describe a different table. df11pack's
default --luts=compat mode reproduces this bug byte-for-bit (required for
byte-identity with every real official file). An experimental
--luts=correct mode instead zero-fills those positions. The argument that
this is safe -- those positions are claimed unreachable during decode --
is INFERRED from reading the kernel's indexing logic, not measured. This
script is the measurement.

Known instance used here (named explicitly in the task brief and
independently re-derived below, not just trusted):
    phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_0.safetensors
    prefix table 3 (0-indexed) is {128: 96}; positions 0..127 of that row
    carry 105, leaked from table 2's trailing value at column 255.

Method:
  1. Load the real unit's five kernel-input tensors as raw bytes.
  2. Independently re-derive the leaked run: for each luts row i>0, find
     the run of identical bytes starting at column 0 that equals the
     PREVIOUS row's value at column 255. This is the generic form of the
     documented bug, not a hardcoded row/column pair -- and it is asserted
     to match the documented row 3 / columns 0..127 / value 105 case as a
     precondition, so a silent mismatch (wrong file, wrong version) fails
     loudly rather than testing the wrong bytes.
  3. Build a byte-identical copy of the file except that exact run is
     zeroed in the luts tensor (everything else -- encoded_exponent,
     sign_mantissa, output_positions, gaps, every other luts byte -- is
     untouched).
  4. Decode BOTH (leaked original, zeroed copy) with the real, unmodified
     CUDA kernel via CuPy (the production decode path).
  5. Compare the two full decoded weight streams bit-for-bit.
     CONFIRMED (safe to ship the mode) iff identical.
     REFUTED (remove the mode, per COMPATIBILITY.md's own stated
     consequence) iff they differ anywhere.

Usage:
    test_luts_correct_kernel.py [--unit-file PATH] [--out PATH]
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_unit_file():
    return os.path.join(
        repo_root(), "phase0", "out", "official",
        "qwen3-trunc-layers-only-dir", "model_layers_0.safetensors",
    )


def find_leaked_run(luts_bytes, n_luts):
    """Generic re-derivation of the get_luts carry-forward bug: for each
    row i (1..n_luts-1), a run starting at column 0 whose value equals row
    (i-1)'s value at column 255 is a candidate leaked run. Returns a list
    of (row, start_col, end_col_exclusive, value) for every such run found
    (there may be more than one across the file's prefix tables; the
    caller picks the documented one)."""
    import numpy as np
    arr = np.frombuffer(luts_bytes, dtype=np.uint8).reshape(n_luts, 256)
    found = []
    for i in range(1, n_luts):
        prev_trailing = int(arr[i - 1, 255])
        if int(arr[i, 0]) != prev_trailing:
            continue
        end = 1
        while end < 256 and int(arr[i, end]) == prev_trailing:
            end += 1
        found.append((i, 0, end, prev_trailing))
    return found, arr


def build_zeroed_copy(src_path, dst_path, luts_info, row, start_col, end_col, n_cols=256):
    """Copy src_path to dst_path with luts[row, start_col:end_col] zeroed;
    every other byte in the file, including the rest of luts, is
    untouched."""
    with open(src_path, "rb") as f:
        data = bytearray(f.read())
    s, _e = luts_info["data_offsets"]
    # luts_info's data_offsets are relative to data_start; we need the
    # absolute file offset, computed by the caller and passed via
    # luts_info["_abs_start"].
    abs_start = luts_info["_abs_start"]
    row_start = abs_start + row * n_cols
    for col in range(start_col, end_col):
        data[row_start + col] = 0
    with open(dst_path, "wb") as f:
        f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-file", default=default_unit_file())
    ap.add_argument("--out", default=common.default_out_path("luts_correct_kernel"))
    ap.add_argument(
        "--expect-row", type=int, default=3,
        help="documented leaked row (0-indexed); mismatch vs. re-derivation is an ERROR, not silently ignored",
    )
    ap.add_argument("--expect-start-col", type=int, default=0)
    ap.add_argument("--expect-end-col", type=int, default=128)
    args = ap.parse_args()

    detail = {"unit_file": args.unit_file, "gpu": common.gpu_info()}

    with common.Timer() as t:
        work_copy = None
        try:
            if not os.path.exists(args.unit_file):
                raise FileNotFoundError(f"unit file not found: {args.unit_file}")

            header, data_start = common.read_header(args.unit_file)
            prefix = common.only_unit_prefix(header)
            unit_bytes, tensor_infos = common.load_unit_raw(
                args.unit_file, prefix, header=header, data_start=data_start,
            )
            n_elements = len(unit_bytes["sign_mantissa"])
            n_luts = len(unit_bytes["luts"]) // 256
            detail["unit_prefix"] = prefix
            detail["n_luts"] = n_luts
            detail["n_elements"] = n_elements

            runs, luts_arr = find_leaked_run(unit_bytes["luts"], n_luts)
            detail["leaked_runs_found"] = [
                {"row": r, "start_col": s, "end_col": e, "value": v} for r, s, e, v in runs
            ]
            if not runs:
                raise RuntimeError(
                    "No carry-forward-leak pattern found in this file's luts tensor. "
                    "This script is calibrated to a known-leaky fixture (see docstring); "
                    "it cannot test --luts=correct against a file that doesn't exhibit the bug."
                )

            match = [r for r in runs if r[0] == args.expect_row
                     and r[1] == args.expect_start_col and r[2] == args.expect_end_col]
            if not match:
                raise RuntimeError(
                    f"Expected the documented leak at row={args.expect_row}, "
                    f"cols=[{args.expect_start_col},{args.expect_end_col}), but re-derivation "
                    f"found: {detail['leaked_runs_found']}. Refusing to guess which run to "
                    "test -- pass --expect-row/--expect-start-col/--expect-end-col explicitly "
                    "if this is intentionally a different fixture."
                )
            row, start_col, end_col, leaked_value = match[0]
            detail["tested_run"] = {"row": row, "start_col": start_col, "end_col": end_col, "leaked_value": leaked_value}

            luts_info = dict(tensor_infos["luts"])
            luts_info["_abs_start"] = data_start + luts_info["data_offsets"][0]

            work_dir = os.path.dirname(os.path.abspath(args.out)) or "."
            os.makedirs(work_dir, exist_ok=True)
            work_copy = os.path.join(work_dir, "_luts_zeroed_copy.safetensors")
            build_zeroed_copy(args.unit_file, work_copy, luts_info, row, start_col, end_col)

            zeroed_header, zeroed_data_start = common.read_header(work_copy)
            zeroed_unit_bytes, _ = common.load_unit_raw(work_copy, prefix, header=zeroed_header, data_start=zeroed_data_start)

            # sanity: every tensor except luts must be byte-identical, and
            # luts must differ from the original in EXACTLY the intended
            # run and nowhere else.
            for suf in ("encoded_exponent", "sign_mantissa", "output_positions", "gaps"):
                if zeroed_unit_bytes[suf] != unit_bytes[suf]:
                    raise RuntimeError(f"patch corrupted {suf}, which must be untouched")
            import numpy as np
            orig_luts = np.frombuffer(unit_bytes["luts"], dtype=np.uint8).reshape(n_luts, 256).copy()
            new_luts = np.frombuffer(zeroed_unit_bytes["luts"], dtype=np.uint8).reshape(n_luts, 256).copy()
            diff_mask = orig_luts != new_luts
            expected_mask = np.zeros_like(diff_mask)
            expected_mask[row, start_col:end_col] = True
            if not np.array_equal(diff_mask, expected_mask):
                raise RuntimeError(
                    "patch touched bytes outside the intended run: "
                    f"diff positions {np.argwhere(diff_mask).tolist()}"
                )
            if not (new_luts[row, start_col:end_col] == 0).all():
                raise RuntimeError("patched positions are not all zero")
            detail["patch_verified_surgical"] = True

            print(f"[luts=correct] decoding ORIGINAL (leaked) unit via CuPy...")
            out_leaked = common.decode_with_cupy(unit_bytes, n_elements)
            print(f"[luts=correct] decoding ZEROED-leak copy via CuPy...")
            out_zeroed = common.decode_with_cupy(zeroed_unit_bytes, n_elements)

            detail["decoded_leaked_sha256"] = common.sha256_hex(out_leaked)
            detail["decoded_zeroed_sha256"] = common.sha256_hex(out_zeroed)
            identical = common.bytes_equal(out_leaked, out_zeroed)
            detail["bit_for_bit_identical"] = identical

            if identical:
                status = "CONFIRMED"
                summary = (
                    f"Zeroing the leaked LUT run (row {row}, cols [{start_col},{end_col}), "
                    f"originally {leaked_value}) and decoding with the real, unmodified CUDA "
                    f"kernel produced output BIT-FOR-BIT IDENTICAL to decoding the original "
                    f"leaked file, across all {n_elements} weights. The leaked positions are "
                    f"confirmed unreachable during decode. --luts=correct is safe to ship as "
                    f"specified in COMPATIBILITY.md."
                )
            else:
                status = "REFUTED"
                summary = (
                    f"Zeroing the leaked LUT run (row {row}, cols [{start_col},{end_col})) "
                    f"changed the decoded output. The leaked positions ARE reachable during "
                    f"decode. Per COMPATIBILITY.md's own stated consequence, --luts=correct "
                    f"must be REMOVED from PLAN step 1.6 and from COMPATIBILITY.md, not "
                    f"documented around."
                )
        except Exception as e:
            status = "ERROR"
            summary = f"{e.__class__.__name__}: {e}"
            detail["traceback"] = traceback.format_exc()
        finally:
            if work_copy and os.path.exists(work_copy):
                try:
                    os.remove(work_copy)
                except OSError:
                    pass

    common.write_result(args.out, "luts_correct_kernel_gate", status, summary, detail, t.elapsed)
    sys.exit(0 if status in ("CONFIRMED", "REFUTED") else 1)


if __name__ == "__main__":
    main()
