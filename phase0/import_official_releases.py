"""Record the pattern_dict of every official DFloat11 release, and map each release
to the shipped definition that reproduces it.

Reads each release's config.json at a pinned commit (small JSON only; no weights).
Releases whose pattern_dict is already a shipped definition are mapped to it;
the rest become new entries in official_pattern_dicts.json, one per distinct
pattern_dict, named after one canonical release.

    python phase0/import_official_releases.py

Outputs:
  phase0/fixtures/official_pattern_dicts.json  (new entries added)
  phase0/fixtures/official_releases.json       (release -> definition, sha, version)
"""
import json, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
API = "https://huggingface.co/api/models"

# One definition per distinct pattern_dict: (canonical release, layout).
NEW = {
    "bagel-7b-mot": ("DFloat11/BAGEL-7B-MoT-DF11", "transformers"),
    "flux-kontext-dev-diffusers": ("DFloat11/FLUX.1-Kontext-dev-DF11", "diffusers"),
    "hidream-i1-diffusers": ("DFloat11/HiDream-I1-Full-DF11", "diffusers"),
    "llama-3.3-70b": ("DFloat11/Llama-3.3-70B-Instruct-DF11", "transformers"),
    "omnigen2-mllm": ("DFloat11/OmniGen2-mllm-DF11", "transformers"),
    "omnigen2-transformer-diffusers": ("DFloat11/OmniGen2-transformer-DF11", "diffusers"),
    "phi-4": ("DFloat11/Phi-4-reasoning-plus-DF11", "transformers"),
    "qwen-image-diffusers-single": ("DFloat11/Qwen-Image-DF11", "diffusers-single"),
    "wan-diffusers": ("DFloat11/Wan2.1-T2V-14B-Diffusers-DF11", "diffusers"),
    "gemma-3": ("DFloat11/gemma-3-4b-it-DF11", "transformers"),
    "sd35-large-diffusers": ("DFloat11/stable-diffusion-3.5-large-DF11", "diffusers"),
}


def get(url):
    return json.load(urllib.request.urlopen(url))


def key(cfg):
    return json.dumps([list(cfg["pattern_dict"].items()), cfg["threads_per_block"],
                       cfg["bytes_per_thread"]])


def main():
    fx_path = ROOT / "fixtures/official_pattern_dicts.json"
    fx = json.loads(fx_path.read_text())
    releases = {}
    for m in sorted(get(f"{API}?author=DFloat11&limit=500"), key=lambda m: m["id"]):
        rid = m["id"]
        info = get(f"{API}/{rid}")
        sha = info["sha"]
        cfg = None
        if any(s["rfilename"] == "config.json" for s in info.get("siblings", [])):
            c = get(f"https://huggingface.co/{rid}/resolve/{sha}/config.json")
            cfg = c.get("dfloat11_config") if isinstance(c, dict) else None
        releases[rid] = {"sha": sha, "config": cfg}

    # New definitions from their canonical release.
    for name, (rid, layout) in NEW.items():
        cfg = releases[rid]["config"]
        fx[name] = {"repo": rid, "revision": releases[rid]["sha"], "layout": layout,
                    "version": cfg["version"], "threads_per_block": cfg["threads_per_block"],
                    "bytes_per_thread": cfg["bytes_per_thread"],
                    "pattern_dict": cfg["pattern_dict"]}

    by_key = {}
    for name, d in fx.items():
        by_key.setdefault(key(d), []).append(name)

    out = {}
    for rid, r in releases.items():
        cfg = r["config"]
        if cfg is None:
            out[rid] = {"sha": r["sha"], "definition": None,
                        "why": "no dfloat11_config: the legacy 0.1.0 format, or a ComfyUI file"}
            continue
        names = by_key.get(key(cfg))
        assert names, f"{rid}: no definition reproduces its pattern_dict"
        out[rid] = {"sha": r["sha"], "version": cfg["version"], "definition": sorted(names)[0]}
    for name in NEW:
        users = sorted(r.split("/")[1] for r, v in out.items() if v["definition"] == name)
        fx[name]["used_by"] = users

    fx_path.write_text(json.dumps(fx, indent=2) + "\n")
    (ROOT / "fixtures/official_releases.json").write_text(json.dumps(out, indent=1) + "\n")
    covered = sum(1 for v in out.values() if v["definition"])
    print(f"{len(out)} releases, {covered} mapped to a definition, {len(NEW)} new definitions")
    for rid, v in out.items():
        print(f"  {rid:50s} -> {v['definition']}")


if __name__ == "__main__":
    main()
