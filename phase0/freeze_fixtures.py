"""0.12 -- freeze the Phase 0 reference outputs as graded fixtures.

Records per-tensor sha256 for every official output produced or validated in
Phase 0, with provenance, so Phase 1 can grade against them without Python,
without the official compressor, and without a large machine.
"""
import hashlib, json, struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT = {'luts', 'encoded_exponent', 'sign_mantissa', 'output_positions', 'gaps', 'split_positions'}

SETS = [
    dict(name="tier0-qwen3-trunc-layers-only", provenance="locally-compressed",
         out=ROOT/"out/official/qwen3-trunc-layers-only-dir",
         src=ROOT/"corpus/tier0/qwen3-trunc",
         note="4-layer truncated Qwen3-0.6B, vocab 4096. Layer unit is full size (15,728,640 weights)."),
    dict(name="tier1-qwen3-0.6b-layers-only", provenance="locally-compressed",
         out=ROOT/"out/official/qwen3-0.6b-layers-only",
         src=Path("../bf16-exponent-compression/real_model"),
         note="Full Qwen3-0.6B, 28 units. Official peak RSS 2287.6 MiB, 788.9 s."),
]


def header(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h, 8 + n


def sha(p, base, off):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        f.seek(base + off[0]); left = off[1] - off[0]
        while left:
            b = f.read(min(1 << 20, left)); m.update(b); left -= len(b)
    return m.hexdigest()


def main():
    man = {"generated": datetime.now(timezone.utc).isoformat(),
           "note": "Per-tensor sha256 of official DFloat11 output. Phase 1 grades byte-identity against these.",
           "sets": []}
    for s in SETS:
        if not s["out"].exists():
            print(f"  SKIP {s['name']} (missing)"); continue
        files, n_unit, n_plain, total = {}, 0, 0, 0
        for f in sorted(s["out"].glob("*.safetensors")):
            h, base = header(f)
            ent = {}
            for name, meta in sorted(h.items()):
                dig = sha(f, base, meta["data_offsets"])
                nb = meta["data_offsets"][1] - meta["data_offsets"][0]
                ent[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                             "bytes": nb, "sha256": dig}
                total += nb
                if name.rsplit('.', 1)[-1] in UNIT: n_unit += 1
                else: n_plain += 1
            files[f.name] = ent
        cfg = {}
        for j in ("config.json", "generation_config.json"):
            p = s["out"]/j
            if p.exists():
                cfg[j] = hashlib.sha256(p.read_bytes()).hexdigest()
        man["sets"].append({
            "name": s["name"], "provenance": s["provenance"], "note": s["note"],
            "output_dir": str(s["out"]), "source_dir": str(s["src"]),
            "shards": len(files), "unit_tensors": n_unit, "non_unit_tensors": n_plain,
            "total_bytes": total, "json_sha256": cfg, "files": files,
        })
        print(f"  {s['name']}: {len(files)} shards, {n_unit} unit + {n_plain} non-unit tensors, {total/2**20:.1f} MiB")
    out = ROOT/"fixtures"; out.mkdir(exist_ok=True)
    (out/"MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(f"\nwrote {out/'MANIFEST.json'} ({(out/'MANIFEST.json').stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
