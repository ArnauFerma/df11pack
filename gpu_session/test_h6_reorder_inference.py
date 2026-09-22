#!/usr/bin/env python3
"""test_h6_reorder_inference.py -- H6 inference confirmation.

FINDINGS.md 0.8/H6 confirmed STRUCTURALLY (three independent parsers agree
a physically-reordered shard round-trips byte-identical) and confirmed the
loader dispatches purely by tensor name (source reading). What was never
done: actually load the reordered file through the official
`DFloat11Model` and run inference through the real CUDA kernel.

This script does exactly that:
  1. Build a bf16 model skeleton from the ORIGINAL directory's config.json
     (architecture only; all weights come from the safetensors files).
  2. Load it twice with `DFloat11Model.from_pretrained`:
       (a) the untouched qwen3-trunc-layers-only-dir
       (b) a copy of that directory with model_layers_0.safetensors
           swapped for the pre-built reordered fixture
           (phase0/out/h6/model_layers_0.reordered.safetensors) --
           same tensors, same bytes, different physical order/header order.
  3. Run identical input_ids through both, EXACT (torch.equal) logit
     comparison.
  4. As a determinism baseline, also runs (a) twice and requires those two
     runs to already be exactly equal -- otherwise a difference between
     (a) and (b) would be ambiguous (kernel nondeterminism vs. order
     sensitivity), and that ambiguity must be reported, not swallowed.

No tokenizer is needed or available in these directories (they are
compressor output, not full model repos): input_ids are a fixed-seed
pseudo-random sequence in [0, vocab_size).

Usage:
    test_h6_reorder_inference.py [--src-dir PATH] [--reordered-shard PATH] [--out PATH]
"""
import argparse
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_src_dir():
    return os.path.join(repo_root(), "phase0", "out", "official", "qwen3-trunc-layers-only-dir")


def default_reordered_shard():
    return os.path.join(repo_root(), "phase0", "out", "h6", "model_layers_0.reordered.safetensors")


def build_reordered_dir(src_dir, reordered_shard, work_dir):
    dst = os.path.join(work_dir, "reordered_dir")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src_dir):
        if name == "model_layers_0.safetensors":
            continue
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dst, name))
    shutil.copy2(reordered_shard, os.path.join(dst, "model_layers_0.safetensors"))
    return dst


def build_skeleton_model(config_dir, torch, transformers):
    from transformers.modeling_utils import no_init_weights
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with no_init_weights():
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.tie_weights()
        model.eval()
    return model, config


def run_forward(model_dir, config_dir, torch, transformers, dfloat11_cls, device, input_ids):
    model, config = build_skeleton_model(config_dir, torch, transformers)
    loaded = dfloat11_cls.from_pretrained(model_dir, bfloat16_model=model, device=device)
    loaded.eval()
    with torch.no_grad():
        out = loaded(input_ids.to(device))
    logits = out.logits.detach().to("cpu")
    del loaded
    torch.cuda.empty_cache()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=default_src_dir())
    ap.add_argument("--reordered-shard", default=default_reordered_shard())
    ap.add_argument("--out", default=common.default_out_path("h6_reorder_inference"))
    args = ap.parse_args()

    detail = {"src_dir": args.src_dir, "reordered_shard": args.reordered_shard, "gpu": common.gpu_info()}

    with common.Timer() as t:
        tmp = None
        try:
            if not os.path.isdir(args.src_dir):
                raise FileNotFoundError(f"missing {args.src_dir}")
            if not os.path.exists(args.reordered_shard):
                raise FileNotFoundError(f"missing {args.reordered_shard}")

            import torch
            import transformers
            from dfloat11 import DFloat11Model

            device = "cuda:0"
            vocab_size = transformers.AutoConfig.from_pretrained(args.src_dir).vocab_size
            torch.manual_seed(0)
            input_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
            detail["vocab_size"] = vocab_size
            detail["input_ids"] = input_ids.tolist()

            tmp = tempfile.mkdtemp(prefix="h6_")
            reordered_dir = build_reordered_dir(args.src_dir, args.reordered_shard, tmp)

            print("[H6] pass 1/3: original directory (determinism baseline run A)...")
            logits_orig_a = run_forward(args.src_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)

            print("[H6] pass 2/3: original directory again (determinism baseline run B)...")
            logits_orig_b = run_forward(args.src_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)

            print("[H6] pass 3/3: reordered shard directory...")
            logits_reordered = run_forward(reordered_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)

            baseline_equal = torch.equal(logits_orig_a, logits_orig_b)
            baseline_max_abs_diff = (logits_orig_a - logits_orig_b).abs().max().item()
            detail["determinism_baseline_exactly_equal"] = baseline_equal
            detail["determinism_baseline_max_abs_diff"] = baseline_max_abs_diff

            reorder_equal = torch.equal(logits_orig_a, logits_reordered)
            reorder_max_abs_diff = (logits_orig_a - logits_reordered).abs().max().item()
            detail["reorder_vs_original_exactly_equal"] = reorder_equal
            detail["reorder_vs_original_max_abs_diff"] = reorder_max_abs_diff
            detail["logits_shape"] = list(logits_orig_a.shape)

            if not baseline_equal:
                status = "ERROR"
                summary = (
                    "Cannot draw a conclusion: two runs of the SAME unmodified directory "
                    f"produced different logits (max abs diff {baseline_max_abs_diff}). "
                    "The kernel/pipeline is not deterministic on this box, so a reorder-vs-"
                    "original difference would be uninterpretable. Investigate before retrying."
                )
            elif reorder_equal:
                status = "CONFIRMED"
                summary = (
                    "Physically reordering model_layers_0.safetensors' tensors (same names, "
                    "dtypes, shapes, bytes; different data_offsets/header order) produced "
                    "EXACTLY identical logits after loading through DFloat11Model and running "
                    "the real CUDA kernel. H6 is fully confirmed, not just structurally."
                )
            else:
                status = "REFUTED"
                summary = (
                    "Loading the reordered shard produced DIFFERENT logits from the original "
                    f"(max abs diff {reorder_max_abs_diff}) even though the determinism "
                    "baseline was exact. Physical tensor order DOES matter to the official "
                    "loader/kernel path."
                )
        except Exception as e:
            status = "ERROR"
            summary = f"{e.__class__.__name__}: {e}"
            detail["traceback"] = traceback.format_exc()
        finally:
            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)

    common.write_result(args.out, "H6_reorder_inference", status, summary, detail, t.elapsed)
    sys.exit(0 if status in ("CONFIRMED", "REFUTED") else 1)


if __name__ == "__main__":
    main()
