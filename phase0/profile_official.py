"""H2: where does the official compressor's time actually go?

Runs the official compressor under py-spy and, separately, with coarse stage
timers, so the flame graph and the stage table can be cross-checked against each
other and against wall clock.

Usage:
    phase0/env/bin/py-spy record -o phase0/out/h2/flame.svg --rate 100 --subprocesses -- \
        phase0/env/bin/python phase0/profile_official.py --model <dir> --out <dir>
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official  # noqa: E402
from dfloat11 import dfloat11_utils as U  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

LAYER_ATTRS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

STAGE_FNS = ["get_codec", "get_32bit_codec", "get_luts", "encode_weights", "encode"]
timings = defaultdict(float)
counts = defaultdict(int)


def instrument():
    """Wrap the official stage functions with timers, without changing behaviour."""
    for name in STAGE_FNS:
        fn = getattr(U, name, None)
        if fn is None:
            print(f"  (no {name} in dfloat11_utils -- skipping)")
            continue

        def make(fn, name):
            def wrapped(*a, **k):
                t = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    timings[name] += time.perf_counter() - t
                    counts[name] += 1
            return wrapped

        setattr(U, name, make(fn, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    instrument()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.eval()
    t_load = time.perf_counter() - t0

    a.out.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    official.compress_model(model, pattern_dict={r"model\.layers\.\d+": LAYER_ATTRS},
                            save_path=str(a.out), save_single_file=False,
                            check_correctness=False)
    t_compress = time.perf_counter() - t1
    total = time.perf_counter() - t0

    # encode is called by encode_weights, so it is nested -- report both and flag it.
    report = {
        "wall_total_s": round(total, 3),
        "model_load_s": round(t_load, 3),
        "compress_call_s": round(t_compress, 3),
        "stages_s": {k: round(v, 3) for k, v in sorted(timings.items(), key=lambda kv: -kv[1])},
        "stage_calls": dict(counts),
        "note": "encode is nested inside encode_weights; do not sum them.",
    }
    top = max((v, k) for k, v in timings.items() if k != "encode_weights") if timings else (0, "-")
    report["dominant_leaf_stage"] = top[1]
    report["dominant_leaf_share_of_compress"] = round(top[0] / t_compress, 4) if t_compress else None

    (a.out.parent / f"{a.out.name}.h2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
