#!/usr/bin/env bash
# run_all.sh -- single entry point for the df11pack GPU session.
#
# Runs all four Phase-0-closing GPU tests (H7, H6 inference, H11 inference,
# --luts=correct kernel gate) in priority order, independently, so that one
# failing does not stop the others. Writes one machine-readable result file
# and one human summary. Idempotent and re-runnable: safe to run again if
# it was interrupted, a package failed to resolve, or you just want to
# re-confirm before tearing the instance down.
#
# See gpu_session/RUNBOOK.md for what to upload and what instance to rent.
# See gpu_session/EXPECTED.md for what each result means, decided BEFORE
# this ran.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUT_DIR="$HERE/out"
VENV_DIR="$HERE/venv"
mkdir -p "$OUT_DIR"

SESSION_T0=$(date +%s)

log()  { printf '[run_all] %s\n' "$*"; }
fail() { printf '\n[run_all] FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites -- fail fast, name exactly what's missing
# ---------------------------------------------------------------------------
log "checking prerequisites..."

command -v nvidia-smi >/dev/null 2>&1 || fail \
  "nvidia-smi not found. This must run on a GPU instance with the NVIDIA driver installed. Did you rent a GPU pod? See RUNBOOK.md."

SMI_OUT="$(nvidia-smi 2>&1)" || fail \
  "nvidia-smi is present but failed to run (driver not loaded / no GPU visible). Output:
$SMI_OUT"

CUDA_SMI_VERSION="$(printf '%s\n' "$SMI_OUT" | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)"
[ -n "$CUDA_SMI_VERSION" ] || fail "could not parse a CUDA version out of nvidia-smi's output; cannot pick a matching cupy/torch build. Raw output:
$SMI_OUT"
CUDA_MAJOR="${CUDA_SMI_VERSION%%.*}"
CUDA_MINOR="${CUDA_SMI_VERSION#*.}"
log "GPU driver reports CUDA $CUDA_SMI_VERSION (major $CUDA_MAJOR)"

# torch must come from the PyTorch index for a CUDA version the driver
# supports: unpinned, pip pulls a cu128 wheel that refuses a CUDA 12.4 driver
# (RUNBOOK.md, "Environment pins", pin 1). Pick the newest index not newer
# than the driver. Set TORCH_INDEX_URL to override.
DRIVER_CU=$(( CUDA_MAJOR * 100 + CUDA_MINOR ))
TORCH_CU=""
for cu in 130 129 128 126 124 121 118; do
  if [ $(( (cu / 10) * 100 + cu % 10 )) -le "$DRIVER_CU" ]; then TORCH_CU="cu$cu"; break; fi
done
if [ -z "${TORCH_INDEX_URL:-}" ]; then
  [ -n "$TORCH_CU" ] || fail "driver reports CUDA $CUDA_SMI_VERSION, older than any PyTorch CUDA index this script knows (cu118 and up). Set TORCH_INDEX_URL by hand and re-run."
  TORCH_INDEX_URL="https://download.pytorch.org/whl/$TORCH_CU"
fi
log "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"

command -v python3 >/dev/null 2>&1 || fail "python3 not found on this instance."
PY_VER="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
log "python3 version: $PY_VER"
python3 -c 'import sys; assert sys.version_info[:2] >= (3, 9), "need >=3.9"' \
  || fail "python3 is older than 3.9 (found $PY_VER). dfloat11 requires >=3.9."

command -v pip3 >/dev/null 2>&1 || python3 -m pip --version >/dev/null 2>&1 || fail \
  "no usable pip found for python3 ($PY_VER)."

FREE_KB="$(df -Pk "$HERE" | tail -1 | awk '{print $4}')"
FREE_GIB=$(( FREE_KB / 1024 / 1024 ))
[ "$FREE_GIB" -ge 8 ] || fail \
  "only ${FREE_GIB} GiB free on the filesystem holding $HERE; need at least ~8 GiB for the torch/cupy CUDA wheels. Attach more disk or pick a bigger instance."
log "free disk: ${FREE_GIB} GiB (ok)"

# Required uploaded inputs -- list every one by name so a missing upload
# fails immediately with an exact, actionable message rather than a
# confusing failure three steps later inside a test script.
REQUIRED_PATHS=(
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/config.json"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/generation_config.json"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/model.safetensors"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_0.safetensors"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_1.safetensors"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_2.safetensors"
  "$REPO_ROOT/phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_3.safetensors"
  "$REPO_ROOT/phase0/out/h6/model_layers_0.reordered.safetensors"
  "$REPO_ROOT/phase0/corpus/tier0/qwen3-trunc/model.safetensors"
  "$REPO_ROOT/phase0/repack_shards.py"
  "$REPO_ROOT/phase0/verify_repack.py"
  "$HERE/common.py"
  "$HERE/test_h7_decode_ptx.py"
  "$HERE/test_h6_reorder_inference.py"
  "$HERE/test_h11_repack_inference.py"
  "$HERE/test_luts_correct_kernel.py"
)
MISSING=()
for p in "${REQUIRED_PATHS[@]}"; do
  [ -e "$p" ] || MISSING+=("$p")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  printf '[run_all] FATAL: missing %d required file(s):\n' "${#MISSING[@]}" >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  fail "upload is incomplete. See RUNBOOK.md 'What to upload'."
fi
log "all required files present (${#REQUIRED_PATHS[@]} checked)"

# Optional: the source weights of the tier-0 unit (136 MiB). With it, H7 and the
# --luts=correct gate also compare their decodes with the source weights;
# without it they compare the two decodes only with each other.
SOURCE_WEIGHTS="$REPO_ROOT/phase0/corpus/tier0/qwen3-trunc/model.safetensors"
if [ -e "$SOURCE_WEIGHTS" ]; then
  log "source weights present: H7 and the LUT gate will check decodes against them"
else
  log "WARNING: $SOURCE_WEIGHTS not uploaded; H7 and the LUT gate will not check decodes against the source weights"
fi

# ---------------------------------------------------------------------------
# 2. Install exactly what's needed -- torch from the index matching the
#    driver's CUDA version, nothing else pinned; record what actually resolved.
# ---------------------------------------------------------------------------
if [ ! -x "$VENV_DIR/bin/python3" ]; then
  log "creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR" || fail "python3 -m venv failed"
else
  log "reusing existing venv at $VENV_DIR"
fi
PY="$VENV_DIR/bin/python3"
"$PY" -m pip install --upgrade pip -q || fail "pip upgrade failed"

log "installing torch (CUDA-enabled wheel from $TORCH_INDEX_URL)..."
"$PY" -m pip install -q torch --index-url "$TORCH_INDEX_URL" || fail \
  "torch install failed. If this instance has an unusual CUDA/driver combo, install torch manually per https://pytorch.org/get-started/locally/ and re-run this script (it will skip already-satisfied installs)."

"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || fail \
  "torch installed but torch.cuda.is_available() is False. GPU not visible to torch -- check the instance's CUDA/driver setup before continuing."

if [ "$CUDA_MAJOR" -ge 12 ]; then
  CUPY_EXTRA="cuda12"
elif [ "$CUDA_MAJOR" -eq 11 ]; then
  CUPY_EXTRA="cuda11"
else
  fail "driver reports CUDA major version $CUDA_MAJOR, which dfloat11's published extras (cuda11, cuda12) don't cover. Install a matching cupy-cudaXXx by hand and re-run."
fi
log "installing dfloat11[$CUPY_EXTRA] (pulls the matching cupy build, plus transformers/safetensors/accelerate/huggingface_hub/tqdm/dahuffman)..."
"$PY" -m pip install -q "dfloat11[$CUPY_EXTRA]" || fail "pip install dfloat11[$CUPY_EXTRA] failed"

# Explicit installs too (belt and suspenders -- dfloat11 already pulls
# these transitively, but the task calls them out by name and an explicit
# install makes the resolved versions easy to find in pip's own output).
"$PY" -m pip install -q transformers safetensors numpy || fail "pip install transformers/safetensors/numpy failed"

log "verifying the installed stack imports cleanly and reaches decode.ptx..."
"$PY" - <<'PYEOF' || fail "post-install import check failed -- see traceback above"
import sys
import torch
import cupy as cp
import transformers
import safetensors
import numpy
from dfloat11 import DFloat11Model, compress_model  # noqa: F401
import pkg_resources
ptx = pkg_resources.resource_filename("dfloat11", "decode.ptx")
assert __import__("os").path.exists(ptx), f"decode.ptx not found at {ptx}"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"OK: torch={torch.__version__} cupy={cp.__version__} transformers={transformers.__version__} "
      f"safetensors={safetensors.__version__} numpy={numpy.__version__} decode.ptx={ptx}")
PYEOF

RESOLVED_VERSIONS="$("$PY" - <<'PYEOF'
import json
out = {}
for mod in ("torch", "cupy", "dfloat11", "transformers", "safetensors", "numpy"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "unknown")
    except Exception as e:
        out[mod] = f"<not importable: {e}>"
print(json.dumps(out))
PYEOF
)"
log "resolved versions: $RESOLVED_VERSIONS"

# ---------------------------------------------------------------------------
# 3. Run the four tests, in priority order, independently.
# ---------------------------------------------------------------------------
declare -a ITEM_NAMES=(h7_decode_ptx h6_reorder_inference h11_repack_inference luts_correct_kernel)
declare -a ITEM_SCRIPTS=(test_h7_decode_ptx.py test_h6_reorder_inference.py test_h11_repack_inference.py test_luts_correct_kernel.py)
declare -a ITEM_ELAPSED=()

for i in "${!ITEM_NAMES[@]}"; do
  name="${ITEM_NAMES[$i]}"
  script="${ITEM_SCRIPTS[$i]}"
  out_json="$OUT_DIR/${name}.json"
  log ""
  log "=== [$((i+1))/4] $name ($script) ==="
  t0=$(date +%s)
  # A result left by an earlier run must not be reported as this run's.
  rm -f "$out_json"
  "$PY" "$HERE/$script" --out "$out_json"
  rc=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))
  ITEM_ELAPSED+=("$elapsed")
  if [ "$rc" -eq 0 ]; then
    log "$name finished in ${elapsed}s (see $out_json)"
  else
    log "$name EXITED NON-ZERO ($rc) after ${elapsed}s -- treated as ERROR, continuing to the next item. See $out_json"
  fi
done

# ---------------------------------------------------------------------------
# 4. Aggregate: one machine-readable result file, one human summary.
# ---------------------------------------------------------------------------
SESSION_T1=$(date +%s)
SESSION_ELAPSED=$((SESSION_T1 - SESSION_T0))

ITEM_NAMES_CSV="$(IFS=,; echo "${ITEM_NAMES[*]}")"
ITEM_ELAPSED_CSV="$(IFS=,; echo "${ITEM_ELAPSED[*]}")"

"$PY" - "$OUT_DIR" "$SESSION_ELAPSED" "$ITEM_NAMES_CSV" "$ITEM_ELAPSED_CSV" "$RESOLVED_VERSIONS" "$CUDA_SMI_VERSION" <<'PYEOF'
import json, os, sys, datetime

out_dir, session_elapsed, names_csv, elapsed_csv, resolved_json, cuda_smi_version = sys.argv[1:7]
names = names_csv.split(",")
elapsed_list = [int(x) for x in elapsed_csv.split(",")]
resolved = json.loads(resolved_json)

items = {}
for name, elapsed in zip(names, elapsed_list):
    p = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(p):
        with open(p) as f:
            items[name] = json.load(f)
    else:
        items[name] = {
            "item": name, "status": "ERROR",
            "summary": f"{p} was not written -- the test script crashed before it could report.",
            "detail": {}, "elapsed_seconds": elapsed,
        }
    items[name]["shell_elapsed_seconds"] = elapsed

results = {
    "session": {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_elapsed_seconds": int(session_elapsed),
        "cuda_smi_version": cuda_smi_version,
        "resolved_versions": resolved,
    },
    "items": items,
}
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

order = ["h7_decode_ptx", "h6_reorder_inference", "h11_repack_inference", "luts_correct_kernel"]
title = {
    "h7_decode_ptx": "1. H7 -- decode.ptx without CuPy (driver API)",
    "h6_reorder_inference": "2. H6 -- reordered shard, inference confirmation",
    "h11_repack_inference": "3. H11 -- repacked variants, load confirmation",
    "luts_correct_kernel": "4. --luts=correct kernel gate",
}
lines = []
lines.append("df11pack GPU session -- summary")
lines.append(f"generated: {results['session']['timestamp']}")
lines.append(f"total session wall time: {int(session_elapsed)}s ({session_elapsed/60:.1f} min)")
lines.append(f"GPU driver CUDA version: {cuda_smi_version}")
lines.append(f"resolved versions: {json.dumps(resolved)}")
lines.append("")
for name in order:
    it = items[name]
    lines.append(f"{title[name]}")
    lines.append(f"  status:  {it.get('status')}")
    lines.append(f"  elapsed: {it.get('elapsed_seconds', it.get('shell_elapsed_seconds', '?'))}s")
    lines.append(f"  summary: {it.get('summary')}")
    lines.append("")
lines.append("Full detail (including tracebacks on ERROR) is in results.json next to this file.")
lines.append("Read RUNBOOK.md 'How to read the result' and EXPECTED.md before acting on any REFUTED/ERROR item.")

with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
PYEOF

log ""
log "=== SESSION COMPLETE in ${SESSION_ELAPSED}s ($(awk "BEGIN{printf \"%.1f\", $SESSION_ELAPSED/60}") min) ==="
log "results:  $OUT_DIR/results.json"
log "summary:  $OUT_DIR/SUMMARY.txt"
log ""
log ">>> Bring both files back, then STOP THE INSTANCE. See RUNBOOK.md. <<<"

# Exit 0 even if individual items ERRORed -- run_all.sh's job is to
# produce a complete report, not to succeed only when every hypothesis is
# confirmed. A REFUTED or ERROR item is a valid, useful outcome, visible
# in results.json / SUMMARY.txt.
exit 0
