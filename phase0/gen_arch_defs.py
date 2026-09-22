"""Generate data/architectures/*.toml from verified official pattern_dicts.

Generated rather than hand-written so there is no transcription step to get
wrong -- Phase 0 showed that deriving these orders by reading class definitions
produces the right set of attributes in the wrong order (FINDINGS 0.9).

Input:  phase0/fixtures/official_pattern_dicts.json, transcribed from published
        releases' own dfloat11_config and from Extended's pattern_dict.py.
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


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    fx = json.loads((root / "phase0/fixtures/official_pattern_dicts.json").read_text())
    outdir = root / "data/architectures"
    outdir.mkdir(parents=True, exist_ok=True)
    for key, d in fx.items():
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
