"""Real tensor names, for checking definitions against real checkpoints.

For each definition: the safetensors **headers** (names, dtypes, shapes -- no
weights; a few KB each, by HTTP range request) of a real source checkpoint, and of
the real DF11 release made from it. `tests/real_names.rs` then runs discovery on
the source names and compares with the release, offline.

Ungated repositories only. Pinned to the commit read.

    python phase0/fetch_real_headers.py [definition ...]

Output: phase0/fixtures/real_headers/<definition>.json
"""
import json, pathlib, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "fixtures/real_headers"

# definition: (source repo, source files or a subfolder holding an index),
#             (release repo, release files; None = every root .safetensors)
CASES = {
    # transformers / diffusers, against the official DFloat11 releases
    "qwen3-4b": (("Qwen/Qwen3-4B", "index:model.safetensors.index.json"),
                 ("DFloat11/Qwen3-4B-DF11", None)),
    "qwen3-8b": (("Qwen/Qwen3-8B", "index:model.safetensors.index.json"),
                 ("DFloat11/Qwen3-8B-DF11", None)),
    "llama-3.3-70b": (("mistralai/Mistral-Nemo-Instruct-2407", "index:model.safetensors.index.json"),
                      ("DFloat11/Mistral-Nemo-Instruct-2407-DF11", None)),
    "phi-4": (("microsoft/Phi-4-reasoning-plus", "index:model.safetensors.index.json"),
              ("DFloat11/Phi-4-reasoning-plus-DF11", None)),
    "omnigen2-mllm": (("OmniGen2/OmniGen2", "index:mllm/model.safetensors.index.json"),
                      ("DFloat11/OmniGen2-mllm-DF11", None)),
    "omnigen2-transformer-diffusers": (("OmniGen2/OmniGen2", "dir:transformer"),
                                       ("DFloat11/OmniGen2-transformer-DF11", None)),
    "bagel-7b-mot": (("ByteDance-Seed/BAGEL-7B-MoT", ["ema.safetensors"]),
                     ("DFloat11/BAGEL-7B-MoT-DF11", None)),
    "chroma-diffusers": (("lodestones/Chroma1-HD", "dir:transformer"),
                         ("DFloat11/Chroma-DF11", None)),
    "chroma-base-diffusers-mingyi": (("lodestones/Chroma1-Base", "dir:transformer"),
                                     ("mingyi456/Chroma1-Base-DF11", None)),
    "hidream-i1-diffusers": (("HiDream-ai/HiDream-I1-Full", "dir:transformer"),
                             ("DFloat11/HiDream-I1-Full-DF11", None)),
    "qwen-image-diffusers-single": (("Qwen/Qwen-Image", "dir:transformer"),
                                    ("DFloat11/Qwen-Image-DF11", None)),
    "wan-diffusers": (("Wan-AI/Wan2.1-T2V-14B-Diffusers", "dir:transformer"),
                      ("DFloat11/Wan2.1-T2V-14B-Diffusers-DF11", None)),
    # ComfyUI-native, against the Extended author's releases
    "flux-schnell-comfyui": (("Comfy-Org/flux1-schnell", ["flux1-schnell.safetensors"]),
                             ("mingyi456/FLUX.1-schnell-DF11-ComfyUI", ["flux1-schnell-DF11.safetensors"])),
    "chroma-comfyui": (("Comfy-Org/Chroma1-HD_repackaged", ["split_files/diffusion_models/Chroma1-HD.safetensors"]),
                       ("mingyi456/Chroma1-HD-DF11-ComfyUI", ["Chroma1-HD-DF11.safetensors"])),
    "chroma-radiance-comfyui": (("lodestones/Chroma1-Radiance", ["latest_x0.safetensors"]),
                                ("mingyi456/Chroma1-Radiance-DF11-ComfyUI",
                                 ["Chroma1-Radiance-latest_x0-DF11-54840d542dd6242a8263fc684a57c64a195d6d9c870bc0375a7478a8b7098744.safetensors"])),
    "flux2-comfyui": (("Comfy-Org/vae-text-encorder-for-flux-klein-4b", ["split_files/diffusion_models/flux-2-klein-4b.safetensors"]),
                      ("mingyi456/FLUX.2-klein-4B-DF11-ComfyUI", ["flux-2-klein-4b-DF11.safetensors"])),
    "lumina2-comfyui": (("Comfy-Org/Lumina_Image_2.0_Repackaged", ["split_files/diffusion_models/lumina_2_model_bf16.safetensors"]),
                        ("mingyi456/Lumina-Image-2.0-DF11-ComfyUI", ["lumina_2_model_bf16-DF11.safetensors"])),
    "zimage-comfyui": (("Comfy-Org/z_image", ["split_files/diffusion_models/z_image_bf16.safetensors"]),
                       ("mingyi456/Z-Image-DF11-ComfyUI", ["z_image_bf16-DF11.safetensors"])),
    "cosmos-t2i-predict2-comfyui": (("Comfy-Org/Cosmos_Predict2_repackaged", ["cosmos_predict2_2B_t2i.safetensors"]),
                                    ("mingyi456/Cosmos-Predict2-2B-Text2Image-DF11-ComfyUI", ["cosmos_predict2_2B_t2i-DF11.safetensors"])),
    "anima-comfyui": (("circlestone-labs/Anima", ["split_files/diffusion_models/anima-base-v1.0.safetensors"]),
                      ("mingyi456/Anima-DF11-ComfyUI", ["anima-base-v1.0-DF11.safetensors"])),
    "ernie-image-comfyui": (("Comfy-Org/ERNIE-Image", ["diffusion_models/ernie-image.safetensors"]),
                            ("mingyi456/ERNIE-Image-DF11-ComfyUI", ["ernie-image-DF11.safetensors"])),
    "longcat-image-comfyui": (("Comfy-Org/LongCat-Image", ["split_files/diffusion_models/longcat_image_bf16.safetensors"]),
                              ("mingyi456/LongCat-Image-DF11-ComfyUI", ["longcat_image_bf16-DF11.safetensors"])),
    "ovis-image-comfyui": (("Comfy-Org/Ovis-Image", ["split_files/diffusion_models/ovis_image_bf16.safetensors"]),
                           ("mingyi456/Ovis-Image-7B-DF11-ComfyUI", ["ovis_image_bf16-DF11.safetensors"])),
    "krea2-comfyui": (("Comfy-Org/Krea-2", ["diffusion_models/krea2_raw_bf16.safetensors"]),
                      ("mingyi456/Krea-2-Raw-DF11-ComfyUI", ["krea2_raw_bf16-DF11.safetensors"])),
    "lens-comfyui": (("Comfy-Org/Lens", ["diffusion_models/lens_bf16.safetensors"]),
                     ("mingyi456/Lens-DF11-ComfyUI", ["lens_bf16-DF11.safetensors"])),
    "qwen-image21-comfyui": (("Comfy-Org/Qwen-Image-2.1", ["diffusion_models/qwen_image_2.1_bf16.safetensors"]),
                             ("mingyi456/Qwen-Image-2.1-DF11-ComfyUI", ["qwen_image_2.1_bf16-DF11.safetensors"])),
    "acestep15-comfyui": (("Comfy-Org/ace_step_1.5_ComfyUI_files", ["split_files/diffusion_models/acestep_v1.5_base.safetensors"]),
                          ("mingyi456/Ace-Step1.5-DF11-ComfyUI", ["acestep_v1.5_base-DF11.safetensors"])),
    "sdxl-comfyui": (("ChenkinNoob/ChenkinNoob-XL-V0.2", ["ChenkinNoob-XL-V0.2.safetensors"]),
                     ("mingyi456/ChenkinNoob-XL-V0.2-DF11-ComfyUI", ["ChenkinNoob-XL-V0.2-DF11.safetensors"])),
}


def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60).read()


def info(repo):
    return json.loads(get(f"https://huggingface.co/api/models/{repo}"))


def header(repo, sha, f):
    url = f"https://huggingface.co/{repo}/resolve/{sha}/{f}"
    n = struct.unpack("<Q", get(url, {"Range": "bytes=0-7"}))[0]
    h = json.loads(get(url, {"Range": f"bytes=8-{7 + n}"}))
    meta = h.pop("__metadata__", None)
    # Kept compact: name -> [dtype, shape], plus the physical order.
    order = sorted(h, key=lambda k: h[k]["data_offsets"][0])
    return {"metadata": meta, "order": order,
            "tensors": {k: [v["dtype"], v["shape"]] for k, v in h.items()}}


def resolve(repo, spec):
    i = info(repo)
    sha = i["sha"]
    names = [s["rfilename"] for s in i["siblings"]]
    if spec is None:
        files = sorted(n for n in names if n.endswith(".safetensors") and "/" not in n)
    elif isinstance(spec, list):
        files = spec
    elif spec.startswith("index:"):
        idx = json.loads(get(f"https://huggingface.co/{repo}/resolve/{sha}/{spec[6:]}"))
        sub = spec[6:].rsplit("/", 1)[0] + "/" if "/" in spec[6:] else ""
        files = sorted({sub + f for f in idx["weight_map"].values()})
    elif spec.startswith("dir:"):
        d = spec[4:] + "/"
        files = sorted(n for n in names if n.startswith(d) and n.endswith(".safetensors")
                       and "/" not in n[len(d):])
    else:
        raise ValueError(spec)
    return sha, files


def tied(repo, sha, files):
    """The source config's tie_word_embeddings, from the config.json beside its
    weights; None if there is none."""
    d = files[0].rsplit("/", 1)[0] + "/" if "/" in files[0] else ""
    try:
        return json.loads(get(f"https://huggingface.co/{repo}/resolve/{sha}/{d}config.json")).get(
            "tie_word_embeddings")
    except Exception:  # noqa: BLE001 -- no config is a normal answer
        return None


def fetch(repo, spec):
    sha, files = resolve(repo, spec)
    with ThreadPoolExecutor(8) as ex:
        heads = list(ex.map(lambda f: header(repo, sha, f), files))
    return {"repo": repo, "sha": sha, "tie_word_embeddings": tied(repo, sha, files),
            "files": dict(zip(files, heads))}


def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in which or CASES:
        (srepo, sspec), (rrepo, rspec) = CASES[name]
        try:
            rec = {"definition": name, "source": fetch(srepo, sspec), "release": fetch(rrepo, rspec)}
        except Exception as e:  # noqa: BLE001 -- report and go on; the test lists what is missing
            print(f"  {name:32s} FAILED: {e}")
            continue
        (OUT / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")) + "\n")
        ns = sum(len(h["tensors"]) for h in rec["source"]["files"].values())
        nr = sum(len(h["tensors"]) for h in rec["release"]["files"].values())
        print(f"  {name:32s} source {ns:5d} tensors / {len(rec['source']['files'])} files;"
              f" release {nr:5d} / {len(rec['release']['files'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
