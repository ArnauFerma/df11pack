"""Phase 2 exit gate, with a determinism baseline.

The skeleton comes from AutoModelForCausalLM.from_config, which initialises
randomly. Two skeletons built independently differ, so any comparison must first
establish that loading the SAME directory twice gives identical logits. Without
that baseline a difference proves nothing -- which is exactly what a first run
without it appeared to show.
"""
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
from dfloat11 import DFloat11Model

def logits_of(dirpath, seed=0):
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(dirpath)
    skeleton = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).to("cuda")
    m = DFloat11Model.from_pretrained(dirpath, bfloat16_model=skeleton, device="cuda")
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with torch.no_grad():
        out = m(ids).logits.clone()
    del m, skeleton
    torch.cuda.empty_cache()
    return out

official, ours = sys.argv[1], sys.argv[2]
a1 = logits_of(official); print("[gate] baseline A", flush=True)
a2 = logits_of(official); print("[gate] baseline B", flush=True)
baseline_ok = torch.equal(a1, a2)
print(f"[gate] determinism baseline: {baseline_ok}", flush=True)
if not baseline_ok:
    print(json.dumps({"baseline_identical": False}))
    print("[gate] INCONCLUSIVE: loading the same directory twice already differs.")
    sys.exit(2)

b = logits_of(ours); print("[gate] loaded df11pack output", flush=True)
same = torch.equal(a1, b)
print(json.dumps({"baseline_identical": True, "identical": bool(same),
                  "shape": list(a1.shape),
                  "max_abs_diff": float((a1.float()-b.float()).abs().max())}, indent=2))
print("[gate] " + ("CONFIRMED: the official loader and the real CUDA kernel produced "
      "EXACTLY identical logits from df11pack output and from official output."
      if same else "REFUTED: logits differ."))
sys.exit(0 if same else 1)
