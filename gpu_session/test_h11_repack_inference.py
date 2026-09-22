#!/usr/bin/env python3
"""test_h11_repack_inference.py -- H11 load confirmation.

FINDINGS.md 0.8/H11 confirmed STRUCTURALLY: a single merged file and a
4-way interleaved repacking of the 28-shard official output (built by
phase0/repack_shards.py, verified byte-identical per tensor by
phase0/verify_repack.py) both pass check_invariants.py once reassembled
by tensor name. Never done: actually loading either variant through
DFloat11Model and comparing inference.

Cost-scoped deliberately: the original H11 evidence used the FULL 28-layer
/ 870 MiB official output. Loading and shard-grouping dispatch is a
name-only lookup (load_and_replace_tensors, see FINDINGS 0.8) with no
layer-count dependence, so this script reuses the already-uploaded 90 MiB
4-layer qwen3-trunc-layers-only-dir instead of re-deriving the 870 MiB
directory on the box. That is the same source directory test_h6 uses. If
you specifically need the 28-layer variant re-tested, pass --src-dir at a
copy of qwen3-0.6b-layers-only re-generated on this box (not scripted here
-- see RUNBOOK.md "optional, if you want the full-size H11 run too").

Method:
  1. Run the real (unmodified) phase0/repack_shards.py against --src-dir,
     producing (a) a single merged file and (b) 4 interleaved files whose
     names match no unit and no tensor.
  2. Run the real (unmodified) phase0/verify_repack.py against both --
     structural byte-identity re-check, cheap, and a required gate before
     trusting the inference comparison below.
  3. Copy config.json (+ generation_config.json if present) into each
     repacked directory -- DFloat11Model.from_pretrained's bfloat16_model
     branch reads config.json itself for dfloat11_config, independent of
     the model skeleton.
  4. Build one bf16 skeleton model from --src-dir's config, load it three
     ways (original directory, merged-file directory, interleaved
     directory), run identical input_ids, and require EXACT (torch.equal)
     agreement across all three -- plus a same-directory determinism
     baseline, as in test_h6.

Usage:
    test_h11_repack_inference.py [--src-dir PATH] [--out PATH]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_src_dir():
    return os.path.join(repo_root(), "phase0", "out", "official", "qwen3-trunc-layers-only-dir")


def repack_script():
    return os.path.join(repo_root(), "phase0", "repack_shards.py")


def verify_script():
    return os.path.join(repo_root(), "phase0", "verify_repack.py")


def run_cmd(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def copy_config_files(src_dir, dst_dir):
    for name in ("config.json", "generation_config.json"):
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def build_skeleton_model(config_dir, torch, transformers):
    from transformers.modeling_utils import no_init_weights
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with no_init_weights():
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.tie_weights()
        model.eval()
    return model


def run_forward(model_dir, config_dir, torch, transformers, dfloat11_cls, device, input_ids):
    model = build_skeleton_model(config_dir, torch, transformers)
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
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--out", default=common.default_out_path("h11_repack_inference"))
    args = ap.parse_args()

    detail = {"src_dir": args.src_dir, "gpu": common.gpu_info()}

    with common.Timer() as t:
        tmp = None
        try:
            if not os.path.isdir(args.src_dir):
                raise FileNotFoundError(f"missing {args.src_dir}")
            if not os.path.exists(repack_script()) or not os.path.exists(verify_script()):
                raise FileNotFoundError(
                    "phase0/repack_shards.py and phase0/verify_repack.py must be uploaded "
                    "alongside gpu_session/ (see RUNBOOK.md)."
                )

            tmp = tempfile.mkdtemp(prefix="h11_")
            merged_dir = os.path.join(tmp, "merged")
            interleaved_dir = os.path.join(tmp, "interleaved")

            print("[H11] repacking: single merged file...")
            rc, out, err = run_cmd([sys.executable, repack_script(), args.src_dir, merged_dir, "--mode", "single"])
            detail["repack_single_rc"] = rc
            detail["repack_single_stderr"] = err[-4000:]
            if rc != 0:
                raise RuntimeError(f"repack_shards.py --mode single failed:\n{err}")

            print("[H11] repacking: interleaved files...")
            rc, out, err = run_cmd([
                sys.executable, repack_script(), args.src_dir, interleaved_dir,
                "--mode", "interleaved", "--n-files", str(args.n_files),
            ])
            detail["repack_interleaved_rc"] = rc
            detail["repack_interleaved_stderr"] = err[-4000:]
            if rc != 0:
                raise RuntimeError(f"repack_shards.py --mode interleaved failed:\n{err}")

            print("[H11] verifying byte-identity: merged...")
            rc, out, err = run_cmd([sys.executable, verify_script(), args.src_dir, merged_dir])
            detail["verify_single_stdout"] = out[-4000:]
            verify_single_ok = rc == 0
            detail["verify_single_ok"] = verify_single_ok

            print("[H11] verifying byte-identity: interleaved...")
            rc, out, err = run_cmd([sys.executable, verify_script(), args.src_dir, interleaved_dir])
            detail["verify_interleaved_stdout"] = out[-4000:]
            verify_interleaved_ok = rc == 0
            detail["verify_interleaved_ok"] = verify_interleaved_ok

            if not (verify_single_ok and verify_interleaved_ok):
                raise RuntimeError(
                    "verify_repack.py reported a structural mismatch before inference was even "
                    "attempted -- see verify_single_stdout / verify_interleaved_stdout in detail."
                )

            copy_config_files(args.src_dir, merged_dir)
            copy_config_files(args.src_dir, interleaved_dir)

            import torch
            import transformers
            from dfloat11 import DFloat11Model

            device = "cuda:0"
            vocab_size = transformers.AutoConfig.from_pretrained(args.src_dir).vocab_size
            torch.manual_seed(0)
            input_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
            detail["vocab_size"] = vocab_size
            detail["input_ids"] = input_ids.tolist()

            print("[H11] pass 1/4: original directory (determinism baseline run A)...")
            logits_orig_a = run_forward(args.src_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)
            print("[H11] pass 2/4: original directory again (determinism baseline run B)...")
            logits_orig_b = run_forward(args.src_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)
            print("[H11] pass 3/4: single merged file...")
            logits_merged = run_forward(merged_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)
            print("[H11] pass 4/4: interleaved files...")
            logits_interleaved = run_forward(interleaved_dir, args.src_dir, torch, transformers, DFloat11Model, device, input_ids)

            baseline_equal = torch.equal(logits_orig_a, logits_orig_b)
            detail["determinism_baseline_exactly_equal"] = baseline_equal
            detail["determinism_baseline_max_abs_diff"] = (logits_orig_a - logits_orig_b).abs().max().item()

            merged_equal = torch.equal(logits_orig_a, logits_merged)
            interleaved_equal = torch.equal(logits_orig_a, logits_interleaved)
            detail["merged_vs_original_exactly_equal"] = merged_equal
            detail["merged_vs_original_max_abs_diff"] = (logits_orig_a - logits_merged).abs().max().item()
            detail["interleaved_vs_original_exactly_equal"] = interleaved_equal
            detail["interleaved_vs_original_max_abs_diff"] = (logits_orig_a - logits_interleaved).abs().max().item()
            detail["logits_shape"] = list(logits_orig_a.shape)

            if not baseline_equal:
                status = "ERROR"
                summary = (
                    "Cannot draw a conclusion: two runs of the SAME unmodified directory "
                    "produced different logits, so a repack-vs-original difference would be "
                    "uninterpretable. Investigate determinism before retrying."
                )
            elif merged_equal and interleaved_equal:
                status = "CONFIRMED"
                summary = (
                    "Both repacked variants (single merged file; 4 interleaved files whose "
                    "names match no unit) loaded through DFloat11Model and produced EXACTLY "
                    "identical logits to the original per-layer-shard directory. H11 is fully "
                    "confirmed: shard grouping/file naming does not matter to the official "
                    "loader or kernel path."
                )
            else:
                bad = []
                if not merged_equal:
                    bad.append("merged")
                if not interleaved_equal:
                    bad.append("interleaved")
                status = "REFUTED"
                summary = (
                    f"Repacked variant(s) {bad} loaded and ran, but produced DIFFERENT logits "
                    "from the original directory. Shard grouping/file naming DOES matter to "
                    "the official loader or kernel path, contrary to the structural-only "
                    "confirmation in FINDINGS.md 0.8/H11."
                )
        except Exception as e:
            status = "ERROR"
            summary = f"{e.__class__.__name__}: {e}"
            detail["traceback"] = traceback.format_exc()
        finally:
            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)

    common.write_result(args.out, "H11_repack_load_and_inference", status, summary, detail, t.elapsed)
    sys.exit(0 if status in ("CONFIRMED", "REFUTED") else 1)


if __name__ == "__main__":
    main()
