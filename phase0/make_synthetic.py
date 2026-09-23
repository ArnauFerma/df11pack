"""Corpus case 2: small synthetic FLUX and Chroma models, compressed by the
OFFICIAL compressor, so the diffusers and ComfyUI-native paths have real
fixtures to be graded against.

The review found that no Flux or Chroma unit had ever been encoded: every
byte-identity result was Qwen3, and the eight architecture definitions were
checked only as pattern_dict transcriptions. This closes that.

Models are built as plain nn.Module trees whose module names are exactly those
the architecture definitions name. The official `compress_model` walks
`named_modules()` and the attribute paths, so it cannot tell these from the real
classes -- the tensors it emits are its own output, not an imitation. The only
stand-in is `save_pretrained` on the diffusers models, which the official code
calls to write the remainder file; ours writes the remaining state_dict to
`diffusion_pytorch_model.safetensors`, which is what diffusers itself does. It
touches passthrough tensors only, never a compressed unit.
"""
import json, re, shutil, sys, zlib
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import official  # noqa: E402  (cupy stub, then dfloat11)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

H = 256  # hidden size: small, but units span ~100 kernel chunks
PATTERNS = json.loads((ROOT / "fixtures/official_pattern_dicts.json").read_text())
# Every definition, including those generated from Extended's pattern_dict, read
# back from the TOML the generator wrote -- so the fixture tests the definition
# df11pack actually ships.
for _p in (ROOT.parent / "data/architectures").glob("*.toml"):
    import tomllib
    _d = tomllib.loads(_p.read_text())
    PATTERNS.setdefault(_d["name"], {"pattern_dict": {u["pattern"]: u["attrs"] for u in _d["unit"]}})


def put(root, path, module):
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        nxt = cur._modules.get(p) if p in cur._modules else None
        if nxt is None:
            nxt = nn.Module()
            cur.add_module(p, nxt)
        cur = nxt
    cur.add_module(parts[-1], module)


def lin(i, o):
    return nn.Linear(i, o, bias=True)


# Shapes chosen so every attribute in a unit has a distinct size, which makes
# split_positions sensitive to the concatenation order.
SHAPE = {
    "qkv": (H, 3 * H), "proj": (H, H), "mlp.0": (H, 4 * H), "mlp.2": (4 * H, H),
    "mod.lin": (H, 6 * H), "linear1": (H, 7 * H), "linear2": (5 * H, H),
    "modulation.lin": (H, 3 * H),
    "to_q": (H, H), "to_k": (H, H // 2), "to_v": (H, H // 4),
    "add_q_proj": (H, H + 8), "add_k_proj": (H, H + 16), "add_v_proj": (H, H + 24),
    "to_out.0": (H, H - 8), "to_add_out": (H, H - 16),
    "ff.net.0.proj": (H, 4 * H), "ff.net.2": (4 * H, H),
    "ff_context.net.0.proj": (H, 3 * H), "ff_context.net.2": (3 * H, H),
    "proj_mlp": (H, 4 * H), "proj_out": (5 * H, H),
    "norm1.linear": (H, 6 * H), "norm1_context.linear": (H, 5 * H), "norm.linear": (H, 3 * H),
    "in_proj": (64, H), "out_proj": (H, 2 * H), "linear_1": (H, H + 32), "linear_2": (H + 32, H),
    "in_layer": (H, H + 40), "out_layer": (H + 40, H),
}


def shape_for(attr):
    for key, s in SHAPE.items():
        if attr == key or attr.endswith("." + key) or attr.endswith(key):
            return s
    raise KeyError(attr)


# The four original sets, generated before the seed was made stable. Their frozen
# outputs are kept; see the note on `seed_for`.
ORIGINAL = ["flux-comfyui", "chroma-comfyui", "flux-dev-diffusers", "chroma-diffusers"]


def seed_for(name):
    """A seed that is the same in every process.

    The first version used `hash(name)`, which Python randomises per process
    (PYTHONHASHSEED), so the four original fixtures cannot be regenerated
    bit-exactly -- they are frozen by SHA-256 in MANIFEST.json instead, and
    tests pin positions inside them (the EOF window 95531). Every set generated
    since uses this.
    """
    return zlib.crc32(name.encode())


def instances(pattern):
    """Concrete module names a pattern fullmatches: digit runs as 0 and 1, and
    every member of a character class. Only the syntax the definitions use."""
    toks = re.findall(r"\\d\+\+?|\[[^\]]+\]|\\.|.", pattern)
    choices = []
    for t in toks:
        if t.startswith("\\d"):
            choices.append(["0", "1"])
        elif t.startswith("["):
            choices.append(list(t[1:-1]))
        elif t.startswith("\\"):
            choices.append([t[1]])
        else:
            choices.append([t])
    names = ["".join(c) for c in product(*choices)]
    for n in names:
        assert re.fullmatch(pattern, n), (pattern, n)
    return names


def generic_shape(i):
    """Attribute i of a unit: every attribute in a unit has a distinct size."""
    return (H, H * (1 + i % 4) + 8 * (i + 1))


def build(name):
    """A model with exactly the module names the definition names."""
    pd = PATTERNS[name]["pattern_dict"]
    m = nn.Module()
    for pattern, attrs in pd.items():
        if name in ORIGINAL:
            # Instantiate each pattern twice (or once when it names a single module).
            base = pattern.replace("\\.", ".").replace("\\d+", "{}")
            units = [base.format(i) for i in (0, 1)] if "{}" in base else [base]
        else:
            units = instances(pattern)
        for unit in units:
            if not attrs:
                # Empty attrs: the matched module is itself a Linear; its bias is
                # the sibling.
                put(m, unit, lin(H, 3 * H + 8))
                continue
            for k, a in enumerate(attrs):
                put(m, f"{unit}.{a}", lin(*(shape_for(a) if name in ORIGINAL else generic_shape(k))))
            # A sibling that is not compressed, so the placement rule is exercised.
            put(m, f"{unit}.extra_norm", nn.LayerNorm(H))
    # Passthrough tensors outside every unit.
    put(m, "img_in", lin(64, H))
    put(m, "final_layer.linear", lin(H, 64))

    def save_pretrained(path):
        save_file({k: v.contiguous() for k, v in m.state_dict().items()},
                  str(Path(path) / "diffusion_pytorch_model.safetensors"))

    if "diffusers" in name:
        m.save_pretrained = save_pretrained

    torch.manual_seed(abs(hash(name)) % (2**31) if name in ORIGINAL else seed_for(name))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.02)
    return m.to(torch.bfloat16)


def main():
    # Default: every definition not yet covered. The originals are regenerated
    # only when named explicitly, since they cannot be reproduced bit-exactly.
    names = sys.argv[1:] or sorted(
        n for n in PATTERNS
        if (n.endswith("-comfyui") or n == "chroma-base-diffusers-mingyi") and n not in ORIGINAL)
    manifest = ROOT / "out/official/synthetic_manifest.json"
    out = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name in names:
        native = "comfyui" in name
        src = ROOT / "corpus/synthetic" / name
        dst = ROOT / "out/official" / f"synthetic-{name}"
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)
        src.mkdir(parents=True)
        dst.mkdir(parents=True)

        m = build(name)
        sd = {k: v.contiguous() for k, v in m.state_dict().items()}
        save_file(sd, str(src / "model.safetensors"))
        n_weights = sum(v.numel() for v in sd.values())

        official.compress_model(m, pattern_dict=PATTERNS[name]["pattern_dict"],
                                save_path=str(dst), save_single_file=native,
                                check_correctness=False)
        files = sorted(p.name for p in dst.iterdir())
        out[name] = {"weights": n_weights, "native": native, "files": files}
        print(f"  {name:22s} {n_weights/1e6:5.2f}M weights -> {len(files)} files: {files[:4]}{' ...' if len(files) > 4 else ''}")
        manifest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
