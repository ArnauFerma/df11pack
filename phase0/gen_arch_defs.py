"""Generate data/architectures/*.toml from verified official pattern_dicts.

Generated rather than hand-written so there is no transcription step to get
wrong -- Phase 0 showed that deriving these orders by reading class definitions
produces the right set of attributes in the wrong order (FINDINGS 0.9).

Input:  phase0/fixtures/official_pattern_dicts.json, transcribed from published
        releases' own dfloat11_config and from Extended's pattern_dict.py;
        phase0/fixtures/extended_pattern_dicts.json (import_extended.py), every
        other Extended model, all ComfyUI-native.
Output: data/architectures/<name>.toml
"""
import json, pathlib

LAYOUT = {"qwen3-4b": "transformers", "qwen3-8b": "transformers",
          "flux-dev-diffusers": "diffusers", "chroma-diffusers": "diffusers",
          "chroma-base-diffusers-mingyi": "diffusers",
          "flux-comfyui": "comfyui-native", "chroma-comfyui": "comfyui-native",
          "chroma-radiance-comfyui": "comfyui-native"}
DESC = {"qwen3-4b": "Qwen3-class LLM, transformer layers only; embeddings left uncompressed.",
        "qwen3-8b": "Qwen3-class LLM, layers plus lm_head and embed_tokens as standalone single-tensor units.",
        "flux-dev-diffusers": "FLUX.1 in the diffusers layout.",
        "chroma-diffusers": "Chroma in the diffusers layout; distilled_guidance_layer is ONE unit of 12.",
        "chroma-base-diffusers-mingyi": "Chroma, diffusers, community variant: guidance layer split per layer.",
        "flux-comfyui": "FLUX.1 in the ComfyUI-native layout.",
        "chroma-comfyui": "Chroma in the ComfyUI-native layout.",
        "chroma-radiance-comfyui": "ChromaRadiance, native; adds nerf_blocks with a single-tensor unit."}


def esc(s):
    return "'" + s + "'" if "\\" in s or '"' in s else '"' + s + '"'


# Extended models already defined from official_pattern_dicts.json. The generator
# checks they have not drifted from the pinned import rather than emitting twice.
EXTENDED_EXISTING = {"Flux": "flux-comfyui", "Chroma": "chroma-comfyui",
                     "ChromaRadiance": "chroma-radiance-comfyui"}


# Where splitting on case gives a poor name.
NAMES = {"CosmosT2IPredict2": "cosmos-t2i-predict2", "LongCatImage": "longcat-image"}


def kebab(model):
    if model in NAMES:
        return NAMES[model]
    out = ""
    for i, c in enumerate(model):
        if c.isupper() and i and (model[i - 1].islower() or model[i - 1].isdigit()):
            out += "-"
        out += c.lower()
    return out


def extended(root, fx):
    ext = json.loads((root / "phase0/fixtures/extended_pattern_dicts.json").read_text())
    src = f"github.com/mingyi456/ComfyUI-DFloat11-Extended pattern_dict.py @ {ext['commit'][:7]}"
    names = {}
    for model, pairs in ext["models"].items():
        pd = {p: a for p, a in pairs}
        if model in EXTENDED_EXISTING:
            key = EXTENDED_EXISTING[model]
            assert fx[key]["pattern_dict"] == pd and list(fx[key]["pattern_dict"]) == list(pd), \
                f"{key} has drifted from Extended {model} @ {ext['commit'][:7]}"
            continue
        key = f"{kebab(model)}-comfyui"
        names[key] = model
        LAYOUT[key] = "comfyui-native"
        DESC[key] = f"{model} in the ComfyUI-native layout, from Extended's pattern_dict."
        fx[key] = {"pattern_dict": pd, "repo": src, "threads_per_block": [512],
                   "bytes_per_thread": 8, "version": "0.5.0"}
    # Definition name -> upstream model, so the drift test can find each one's
    # pinned transcription.
    (root / "phase0/fixtures/extended_def_names.json").write_text(json.dumps(names, indent=1) + "\n")


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    fx = json.loads((root / "phase0/fixtures/official_pattern_dicts.json").read_text())
    extended(root, fx)
    outdir = root / "data/architectures"
    outdir.mkdir(parents=True, exist_ok=True)
    for key, d in fx.items():
        # Entries from import_official_releases.py carry their own layout and the
        # releases they reproduce.
        if "layout" in d:
            LAYOUT[key] = d["layout"]
        if d.get("used_by"):
            DESC[key] = "Official DF11 layout of: " + ", ".join(d["used_by"]) + "."
        ver = d.get("version") or "0.5.0"
        lines = ["# Generated from verified official output by phase0/gen_arch_defs.py -- do not hand-edit.",
                 f"# {DESC[key]}", "",
                 f'name = "{key}"', f'layout = "{LAYOUT[key]}"', f'format_version = "{ver}"',
                 f'threads_per_block = {d["threads_per_block"]}',
                 f'bytes_per_thread = {d["bytes_per_thread"]}',
                 f'source = "{d["repo"]}"', ""]
        for pat, attrs in d["pattern_dict"].items():
            lines += ["[[unit]]", f"pattern = {esc(pat)}"]
            if attrs:
                lines.append("attrs = [")
                lines += [f'    "{a}",' for a in attrs]
                lines.append("]")
            else:
                lines += ["# empty: the matched module is itself one tensor", "attrs = []"]
            lines.append("")
        (outdir / f"{key}.toml").write_text("\n".join(lines))
        print(f"  wrote {key}.toml ({len(d['pattern_dict'])} units)")


if __name__ == "__main__":
    main()
