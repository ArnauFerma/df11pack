#!/usr/bin/env python3
"""verify_repack.py -- confirm a repacked variant is byte-identical per
tensor against the original 28-shard directory. No torch. Streams by byte
range; never materializes a whole tensor unless comparing it in chunks."""
import json
import os
import struct
import sys

COPY_BUF = 4 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def collect_manifest(dir_or_files):
    manifest = {}
    files = []
    if isinstance(dir_or_files, str) and os.path.isdir(dir_or_files):
        for name in sorted(os.listdir(dir_or_files)):
            if name.endswith(".safetensors"):
                files.append(os.path.join(dir_or_files, name))
    else:
        files = dir_or_files
    for path in files:
        header, data_start = read_header(path)
        for key, info in header.items():
            s, e = info["data_offsets"]
            manifest[key] = {
                "dtype": info["dtype"], "shape": info["shape"],
                "path": path, "start": data_start + s, "end": data_start + e,
            }
    return manifest


def bytes_equal(m1, m2):
    n1, n2 = m1["end"] - m1["start"], m2["end"] - m2["start"]
    if n1 != n2:
        return False, f"length {n1} != {n2}"
    with open(m1["path"], "rb") as f1, open(m2["path"], "rb") as f2:
        f1.seek(m1["start"]); f2.seek(m2["start"])
        remaining = n1
        while remaining > 0:
            b = min(COPY_BUF, remaining)
            c1, c2 = f1.read(b), f2.read(b)
            if c1 != c2:
                return False, "content mismatch"
            remaining -= b
    return True, None


def main(argv):
    orig_dir, repacked_dir = argv[0], argv[1]
    orig = collect_manifest(orig_dir)
    repacked = collect_manifest(repacked_dir)

    orig_keys, repacked_keys = set(orig), set(repacked)
    missing = orig_keys - repacked_keys
    extra = repacked_keys - orig_keys
    ok = True
    if missing:
        print(f"MISSING in repacked: {len(missing)} keys, e.g. {sorted(missing)[:5]}")
        ok = False
    if extra:
        print(f"EXTRA in repacked: {len(extra)} keys, e.g. {sorted(extra)[:5]}")
        ok = False

    n_checked = 0
    n_mismatch = 0
    for key in sorted(orig_keys & repacked_keys):
        o, r = orig[key], repacked[key]
        if o["dtype"] != r["dtype"] or o["shape"] != r["shape"]:
            print(f"MISMATCH dtype/shape {key}: {o['dtype']}/{o['shape']} vs {r['dtype']}/{r['shape']}")
            n_mismatch += 1
            continue
        same, reason = bytes_equal(o, r)
        if not same:
            print(f"MISMATCH bytes {key}: {reason}")
            n_mismatch += 1
        n_checked += 1

    print(f"{n_checked} tensors compared, {n_mismatch} mismatches, "
          f"{len(missing)} missing, {len(extra)} extra")
    ok = ok and (n_mismatch == 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
