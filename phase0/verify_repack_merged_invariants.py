#!/usr/bin/env python3
"""verify_repack_merged_invariants.py -- H11 support tool.

phase0/check_invariants.py groups tensors into DF11 units PER FILE (see its
group_units(), called once per path inside check_file()). That is the right
behavior for every fixture seen so far, because the official writer always
keeps one unit's tensors together in one shard. It is the WRONG behavior for
an H11 arbitrary-regrouping variant, where a unit's six DF11 tensors are
deliberately scattered across different files -- check_invariants.py would
then report every such unit as INV-NAMES-COMPLETE-missing, even though the
tensors all exist, are byte-identical to the original, and are perfectly
loadable (the official loader merges by NAME across the whole directory
before ever grouping into units -- see dfloat11.py's load_and_replace_tensors,
quoted in H10_H11_RESULTS.md).

This script does not modify or reimplement check_invariants.py's invariant
logic (reuse, not rewrite, per instructions). Instead it does exactly what
the real loader does: builds a directory-wide manifest of tensor name ->
(source file, byte range) regardless of which file each tensor physically
lives in, reassembles each logical unit into its own small temporary
single-file safetensors (~21 MiB for a Qwen3 layer unit), and then invokes
the UNMODIFIED phase0/check_invariants.py as a subprocess on that temp file,
one unit at a time, deleting it immediately after.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_shards import collect_manifest, write_group  # noqa: E402

UNIT_SUFFIXES = (
    "luts", "encoded_exponent", "sign_mantissa",
    "output_positions", "gaps", "split_positions",
)


def group_units_merged(manifest):
    units = {}
    others = []
    for key in manifest:
        if "." in key:
            prefix, suffix = key.rsplit(".", 1)
        else:
            prefix, suffix = "", key
        if suffix in UNIT_SUFFIXES:
            units.setdefault(prefix, {})[suffix] = key
        else:
            others.append(key)
    return units, others


def main(argv):
    repacked_dir = argv[0]
    checker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_invariants.py")
    python = sys.executable

    manifest = collect_manifest(repacked_dir)
    units, others = group_units_merged(manifest)
    print(f"{len(units)} logical units reassembled from {len(manifest)} tensors "
          f"across the directory ({len(others)} uncompressed tensors)", file=sys.stderr)

    n_ok = 0
    n_fail = 0
    with tempfile.TemporaryDirectory() as tmp:
        for unit_name in sorted(units):
            keys = [f"{unit_name}.{suf}" for suf in UNIT_SUFFIXES if suf in units[unit_name]]
            if len(keys) != len(UNIT_SUFFIXES):
                print(f"[FAIL] {unit_name}: only {len(keys)}/6 suffix tensors present in manifest "
                      f"at all (not a file-grouping issue -- genuinely missing)")
                n_fail += 1
                continue
            tmp_path = os.path.join(tmp, "unit.safetensors")
            write_group(tmp_path, keys, manifest)
            result = subprocess.run([python, checker, tmp_path], capture_output=True, text=True)
            passed = result.returncode == 0 and "1/1 units passed" in result.stdout
            if passed:
                n_ok += 1
            else:
                n_fail += 1
                print(f"[FAIL] {unit_name}:\n{result.stdout}\n{result.stderr}")
            os.remove(tmp_path)

    print(f"\n{n_ok}/{len(units)} logical units passed check_invariants.py "
          f"after cross-file reassembly by name.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
