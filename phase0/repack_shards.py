#!/usr/bin/env python3
"""repack_shards.py -- H11 evidence builder.

Repacks the 28-shard qwen3-0.6b-layers-only official output into an
arbitrary tensor-to-file grouping (a single merged file, or N files with
layers interleaved and names that match no unit/tensor name), preserving
every tensor's name, dtype, shape and raw bytes exactly.

No torch. No safetensors library (avoids any dependency on how *it* chooses
to serialize -- we hand-roll the format directly, so we control every byte).
Streams tensor payloads through a bounded-size buffer; never holds a whole
tensor's twin copies (source + dest) in RAM at once for tensors above the
buffer size, and never holds more than one shard's header in memory at a time
beyond the small combined manifest (key -> dtype/shape/source location).

Safetensors format (as documented by the reference implementation, and as
phase0/check_invariants.py already parses it):
    8 bytes little-endian uint64: header length N
    N bytes: UTF-8 JSON header, {tensor_name: {"dtype", "shape",
             "data_offsets": [start, end]}, ...} plus optional "__metadata__"
    remaining bytes: raw tensor data, concatenated, referenced by
             data_offsets relative to the end of the header.
There is no required alignment/padding between header and data, and none
between tensors, beyond what data_offsets says. The official writer happens
to produce contiguous, gap-free layouts; we reproduce that property (not
required by the format, but keeps this script simple and matches DESIGN.md
5.5's "tensors end up contiguous, with no holes").
"""
import argparse
import json
import os
import struct
import sys

COPY_BUF = 4 * 1024 * 1024  # 4 MiB streaming buffer


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def collect_manifest(src_dir):
    """key -> dict(dtype, shape, src_path, src_start, src_end)"""
    manifest = {}
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src_dir, name)
        header, data_start = read_header(path)
        for key, info in header.items():
            s, e = info["data_offsets"]
            if key in manifest:
                raise ValueError(f"duplicate tensor key {key!r} in {name} and elsewhere")
            manifest[key] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "src_path": path,
                "src_start": data_start + s,
                "src_end": data_start + e,
            }
    return manifest


def write_group(out_path, keys, manifest):
    """Write one safetensors file containing exactly `keys`, in the given
    order, streaming each tensor's bytes from its source file."""
    header = {}
    offset = 0
    for key in keys:
        m = manifest[key]
        n = m["src_end"] - m["src_start"]
        header[key] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [offset, offset + n]}
        offset += n

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for key in keys:
            m = manifest[key]
            remaining = m["src_end"] - m["src_start"]
            with open(m["src_path"], "rb") as src:
                src.seek(m["src_start"])
                while remaining > 0:
                    chunk = src.read(min(COPY_BUF, remaining))
                    if not chunk:
                        raise IOError(f"short read copying {key} from {m['src_path']}")
                    out.write(chunk)
                    remaining -= len(chunk)


def plan_single(manifest):
    """Variant (a): every tensor in one file, sorted by key (arbitrary but
    deterministic; deliberately NOT the official writer's own internal
    order, which is irrelevant to the invariant we're testing -- H11 is
    about FILE grouping, not intra-file tensor order)."""
    return {"model_merged.safetensors": sorted(manifest.keys())}


def plan_interleaved(manifest, n_files=4):
    """Variant (b): N files, round-robin over ALL keys (so layer N's seven
    attribute tensors + its six DF11 suffix tensors end up split across
    different files from each other, and from other tensors of the SAME
    unit), with file names that match no unit name and no tensor name."""
    keys = sorted(manifest.keys())
    groups = {f"blob_{i:02d}_of_{n_files}.bin.safetensors": [] for i in range(n_files)}
    names = list(groups.keys())
    for i, key in enumerate(keys):
        groups[names[i % n_files]].append(key)
    return groups


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=("single", "interleaved"), required=True)
    ap.add_argument("--n-files", type=int, default=4)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = collect_manifest(args.src_dir)
    print(f"collected {len(manifest)} tensor keys from {args.src_dir}", file=sys.stderr)

    if args.mode == "single":
        plan = plan_single(manifest)
    else:
        plan = plan_interleaved(manifest, args.n_files)

    for fname, keys in plan.items():
        out_path = os.path.join(args.out_dir, fname)
        write_group(out_path, keys, manifest)
        size = os.path.getsize(out_path)
        print(f"wrote {out_path}: {len(keys)} tensors, {size} bytes", file=sys.stderr)

    print(json.dumps({fname: len(keys) for fname, keys in plan.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
