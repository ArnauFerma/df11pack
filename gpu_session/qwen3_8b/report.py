"""Compare the three Qwen3-8B outputs and write /workspace/out/report.json.

- sha256 of every safetensors file in each output;
- df11pack vs official: file for file;
- both vs the published DFloat11/Qwen3-8B-DF11 release (its LFS sha256, recorded
  in PREDICTION.json before the run -- nothing downloaded);
- sizes, and idx8's measured size and escapes against the prediction;
- wall time and peak RSS of every step.
"""
import hashlib, json, re, struct
from pathlib import Path

W = Path("/workspace/out")
HERE = Path(__file__).resolve().parent
pred = json.loads((HERE / "PREDICTION.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tree(d):
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(Path(d).glob("*.safetensors"))}


def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def timing(name):
    t = W / f"{name}.time"
    if not t.exists():
        return None
    txt = t.read_text()
    wall = re.search(r"Elapsed \(wall clock\) time.*: (.+)", txt)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    rc = (W / f"{name}.rc").read_text().strip() if (W / f"{name}.rc").exists() else None
    return {"wall": wall.group(1) if wall else None,
            "peak_rss_mib": int(rss.group(1)) / 1024 if rss else None, "exit": rc}


out = {"steps": {s: timing(s) for s in
                 ["build", "download", "df11pack", "df11pack_verify", "idx8", "official", "report"]}}
trees = {k: tree(W / k) for k in ["df11pack", "official", "idx8"] if (W / k).exists()}
out["files"] = trees
out["totals_bytes"] = {k: sum(f["bytes"] for f in v.values()) for k, v in trees.items()}

if "df11pack" in trees and "official" in trees:
    a, b = trees["df11pack"], trees["official"]
    out["df11pack_vs_official"] = {
        "same_file_set": sorted(a) == sorted(b),
        "identical": sorted(n for n in a if n in b and a[n]["sha256"] == b[n]["sha256"]),
        "different": sorted(n for n in a if n in b and a[n]["sha256"] != b[n]["sha256"]),
        "only_df11pack": sorted(set(a) - set(b)),
        "only_official": sorted(set(b) - set(a)),
    }
rel = pred["release_lfs_sha256"]
for k in ["df11pack", "official"]:
    if k in trees:
        t = trees[k]
        out[f"{k}_vs_release"] = {
            "identical": sorted(n for n in t if rel.get(n) == t[n]["sha256"]),
            "different": sorted(n for n in t if n in rel and rel[n] != t[n]["sha256"]),
            "not_in_release": sorted(n for n in t if n not in rel),
        }

if "idx8" in trees:
    esc, blocks = 0, 0
    per_unit = {}
    for p in sorted((W / "idx8").glob("*.safetensors")):
        h = header(p)
        for k, v in h.items():
            if k.endswith(".idx8_escape_blocks"):
                u = k[: -len(".idx8_escape_blocks")]
                n = v["shape"][0]
                per_unit[u] = n
                esc += n
            if k.endswith(".idx8_lengths"):
                blocks += v["shape"][0]
    measured = out["totals_bytes"]["idx8"]
    df11 = out["totals_bytes"].get("df11pack") or pred["release_safetensors_bytes"]
    out["idx8"] = {
        "escapes": esc, "blocks": blocks, "escapes_per_unit": per_unit,
        "total_bytes_measured": measured,
        "total_bytes_predicted": pred["idx8_total_bytes_predicted"],
        "saving_vs_df11_pct_measured": (df11 - measured) / df11 * 100,
        "saving_vs_df11_pct_predicted": pred["idx8_saving_pct_predicted"],
    }

(W / "report.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=1)[:4000])
