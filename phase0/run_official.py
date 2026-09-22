"""Run the OFFICIAL DFloat11 compressor and measure time and peak RSS.

Produces the reference output df11pack is graded against, plus the H1/H2 data.
GPU is never touched: the cupy stub in phase0/shim raises if it is.
"""
import argparse, json, os, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official  # noqa: E402  (installs the cupy stub, then imports dfloat11)

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

LAYER_ATTRS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def peak_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return 0.0


class Sampler(threading.Thread):
    daemon = True

    def __init__(self, interval=0.05):
        super().__init__()
        self.interval, self.samples, self.stop = interval, [], False

    def run(self):
        t0 = time.time()
        while not self.stop:
            self.samples.append((time.time() - t0, rss_mb()))
            time.sleep(self.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", choices=["layers-only", "mainstream"], default="layers-only")
    ap.add_argument("--single-file", action="store_true", default=False)
    a = ap.parse_args()

    pattern_dict = {r"model\.layers\.\d+": LAYER_ATTRS}
    if a.pattern == "mainstream":
        pattern_dict = {"lm_head": [], "model.embed_tokens": [], **pattern_dict}

    s = Sampler(); s.start()
    marks = {}

    def mark(name):
        marks[name] = (time.time(), rss_mb(), peak_mb())
        print(f"  [{name:22s}] t={time.time()-t0:7.1f}s  rss={rss_mb():7.1f} MiB  peak={peak_mb():7.1f} MiB", flush=True)

    t0 = time.time()
    mark("start")
    cfg = AutoConfig.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, config=cfg)
    model.eval()
    mark("model loaded")

    a.out.mkdir(parents=True, exist_ok=True)
    official.compress_model(model, pattern_dict=pattern_dict, save_path=str(a.out),
                            save_single_file=a.single_file, check_correctness=False)
    mark("compressed")

    s.stop = True; s.join(timeout=1)
    total = time.time() - t0
    peak = peak_mb()

    src = sum(f.stat().st_size for f in a.model.glob("*.safetensors"))
    dst = sum(f.stat().st_size for f in a.out.rglob("*.safetensors"))
    report = {
        "model": str(a.model), "pattern": a.pattern,
        "pattern_dict": {k: v for k, v in pattern_dict.items()},
        "wall_seconds": round(total, 2),
        "peak_rss_mib": round(peak, 1),
        "rss_at": {k: round(v[1], 1) for k, v in marks.items()},
        "source_bytes": src, "output_bytes": dst,
        "ratio": round(dst / src, 4) if src else None,
        "samples": [(round(t, 2), round(r, 1)) for t, r in s.samples],
    }
    (a.out.parent / f"{a.out.name}.run.json").write_text(json.dumps(report, indent=2))
    print(f"\n  wall {total:.1f}s   PEAK RSS {peak:.1f} MiB   {src/2**20:.1f} -> {dst/2**20:.1f} MiB  ({dst/src:.3f})")


if __name__ == "__main__":
    main()
