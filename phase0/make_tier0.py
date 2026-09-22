"""Build the tier-0 development model: a truncated Qwen3-0.6B.

Small enough that the OFFICIAL compressor runs on a 3 GB machine, while keeping
the real architecture, the real tensor names and real BF16 weight values -- so
codebooks and bitstreams exercise a genuine exponent distribution, not a
synthetic one.

Shrinks two dimensions only:
  * number of transformer layers  (28 -> N)
  * vocabulary                    (151936 -> V, embedding rows sliced)
Everything else keeps Qwen3-0.6B's real shapes.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("../bf16-exponent-compression/real_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("phase0/corpus/tier0/qwen3-trunc"))
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    keep_layers = set(range(a.layers))

    out = {}
    with safe_open(SRC / "model.safetensors", framework="pt") as f:
        names = list(f.keys())
        for n in names:
            if n.startswith("model.layers."):
                idx = int(n.split(".")[2])
                if idx in keep_layers:
                    out[n] = f.get_tensor(n)
            elif n in ("model.embed_tokens.weight", "lm_head.weight"):
                # slice rows instead of materialising 296 MiB
                out[n] = f.get_slice(n)[: a.vocab, :].clone()
            else:
                out[n] = f.get_tensor(n)

    cfg["num_hidden_layers"] = a.layers
    cfg["vocab_size"] = a.vocab
    for k in ("bos_token_id", "eos_token_id"):
        if k in cfg and isinstance(cfg[k], int) and cfg[k] >= a.vocab:
            cfg[k] = a.vocab - 1
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(out, str(a.out / "model.safetensors"), metadata={"format": "pt"})

    n_w = sum(t.numel() for t in out.values())
    size = (a.out / "model.safetensors").stat().st_size
    print(f"wrote {a.out}")
    print(f"  tensors : {len(out)}")
    print(f"  weights : {n_w/1e6:.2f}M")
    print(f"  size    : {size/2**20:.1f} MiB")
    for n, t in sorted(out.items(), key=lambda kv: -kv[1].numel())[:3]:
        print(f"  largest : {t.numel()/1e6:7.2f}M  {n}")


if __name__ == "__main__":
    main()
