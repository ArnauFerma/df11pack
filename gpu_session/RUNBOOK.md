# GPU session runbook

One rented GPU session, scripted end to end, to close the four Phase 0
items that need a real GPU: H7, H6's inference confirmation, H11's load
confirmation, and the `--luts=correct` kernel gate. Read
[`EXPECTED.md`](EXPECTED.md) once before you run anything — it says, for
each of the four, what result means what, decided in advance so the
outcome can't be rationalised after the fact.

This is a correctness session, not a benchmark. Every test here runs one
CUDA kernel launch over one small tensor (the largest is 15.7M weights).
Nothing is timed for speed and nothing needs multiple GPUs.

---

## What instance to rent

**Smallest/cheapest single GPU you can find with ≥8 GB VRAM and a driver
reporting CUDA 12.x.** That's it. These are correctness tests, not
benchmarks — there is no reason to pay for an A100, H100, or anything with
more than one GPU.

Concrete recommendation: **RunPod Community Cloud, RTX A4000 (16 GB)**,
historically around **$0.17–0.25/hr**, billed per second. If that specific
card isn't available when you look, sort Community Cloud by price
ascending and take the cheapest card that lists ≥8 GB VRAM — an RTX 3060,
3070, A4000, or A5000 are all comfortably enough. Avoid Secure Cloud (same
hardware, higher price) unless Community Cloud has no stock. Prices
fluctuate; check RunPod's own pricing page at rental time rather than
trusting a number written here.

Requirements on the template:
- Linux, NVIDIA driver present (`nvidia-smi` works) reporting CUDA ≥ 11 in
  its own "CUDA Version" line (the scripts detect and handle 11 or 12).
  `decode.ptx` targets PTX ISA 8.2 / `sm_52`, which any driver from the
  last several years JITs fine.
- Python ≥ 3.9 (any standard RunPod PyTorch/CUDA template has this; you do
  **not** need a template with PyTorch preinstalled — `run_all.sh`
  installs its own venv and its own torch build).
- ~8 GB free disk for the torch + cupy CUDA wheels. Any default template
  disk (20 GB+) is plenty.
- **No internet access to Hugging Face is required.** Every test uses only
  the small fixtures you upload; nothing is downloaded from the Hub. This
  is deliberate (see "What to upload" below) and makes the session
  immune to HF rate limits or auth issues.

---

## What to upload — minimal, itemised

Total: **~110 MiB**, not the 959 MiB fixture set and not a full model.
This works because H6's inference test, H11's repack test, the H7 kernel
test, and the `--luts=correct` gate test all only need **one small
already-compressed shard directory** that Phase 0 already produced —
nothing here needs to be re-derived from a raw model, and the one thing
that genuinely must be built fresh on the box (the H11 repacked variants,
~90 MiB each) is generated there, not uploaded, exactly as instructed.

Preserve the relative paths shown — the scripts assume this layout rooted
at the repo root:

| Path | Size | Why |
|---|---|---|
| `phase0/out/official/qwen3-trunc-layers-only-dir/` (all 5 files: `config.json`, `generation_config.json`, `model.safetensors`, `model_layers_0..3.safetensors`) | 89.8 MiB | The only model needed. Real official compressor output, 4-layer truncated Qwen3-0.6B, `tie_word_embeddings: true`. Its own `config.json` fully describes the architecture — no download needed to rebuild the skeleton model. Used by **all four** tests: H7 and the LUT gate read `model_layers_0.safetensors` directly; H6 and H11 load the whole directory through `DFloat11Model`. |
| `phase0/out/h6/model_layers_0.reordered.safetensors` | 20.4 MiB | The pre-built H6 fixture: same tensor names/dtypes/shapes/bytes as `model_layers_0.safetensors` above, physically reordered. Used only by the H6 test. |
| `phase0/repack_shards.py` | <15 KiB | Regenerates the H11 repacked variants **on the box** from the uploaded directory above — this is the "regenerate rather than upload the ~870 MB repacks" step the brief asked for, just against the 90 MiB tier-0 directory instead of the full 870 MiB tier-1 one (see the docstring of `test_h11_repack_inference.py` for why that's a faithful substitute: the loader dispatches purely by tensor name, with no layer-count dependence). |
| `phase0/verify_repack.py` | <5 KiB | Structural byte-identity gate the H11 script runs before trusting the inference comparison. |
| `gpu_session/` (this whole directory: `run_all.sh`, `common.py`, the four `test_*.py` scripts, this file, `EXPECTED.md`) | <100 KiB | The session itself. |

Nothing else. No `phase0/env/`, no fixtures `MANIFEST.json`, no full
Qwen3-0.6B, no HF token.

A one-line way to stage exactly this on your local machine before
uploading (run from the repo root):

```bash
mkdir -p /tmp/df11pack_gpu_upload
rsync -a --relative \
  phase0/out/official/qwen3-trunc-layers-only-dir \
  phase0/out/h6/model_layers_0.reordered.safetensors \
  phase0/repack_shards.py \
  phase0/verify_repack.py \
  gpu_session \
  /tmp/df11pack_gpu_upload/
```

Then `scp -r` or `rsync` `/tmp/df11pack_gpu_upload/*` to the pod, landing
at the same relative paths under wherever you put the repo root on the
box (e.g. `~/df11pack/`).

---

## The command to run

On the pod, from the repo root (the directory containing `phase0/` and
`gpu_session/`):

```bash
bash gpu_session/run_all.sh
```

That's the entire session. It:

1. Checks `nvidia-smi`, CUDA version, python, free disk, and every
   uploaded file above — and refuses to proceed with a specific message
   naming whatever is missing, rather than failing confusingly partway
   through.
2. Creates `gpu_session/venv/` (skipped if it already exists — safe to
   re-run) and installs torch (unpinned, CUDA-enabled), `dfloat11[cuda11]`
   or `dfloat11[cuda12]` (auto-selected from the driver's reported CUDA
   version, which also pulls cupy, transformers, safetensors, accelerate,
   huggingface_hub, tqdm, dahuffman), plus explicit transformers/
   safetensors/numpy. It prints and records exactly what resolved.
3. Runs the four tests in priority order (H7, H6, H11, `--luts=correct`),
   each independently — one failing does not stop the others — printing
   elapsed time per item as it goes.
4. Writes `gpu_session/out/results.json` (machine-readable) and
   `gpu_session/out/SUMMARY.txt` (human-readable), and prints the total
   session wall time.

Re-running it is safe and cheap: the venv is reused, and each test
regenerates its own small working files from the uploaded fixtures every
time, so nothing accumulates incorrectly across runs.

Expect it to take on the order of **15–20 minutes** end to end on a fresh
pod (see "Honest wall-time estimate" below) — mostly the torch/cupy CUDA
wheel downloads, not the tests themselves.

---

## What to bring back

Two files, from `gpu_session/out/` on the pod:

- `results.json` — the full machine-readable record (also embeds each
  item's own detail: hashes, exact logit diffs, tracebacks on error).
- `SUMMARY.txt` — the same thing, human-readable, one item per block.

`scp` or `rsync` both back to your local machine, e.g.:

```bash
scp pod:~/df11pack/gpu_session/out/results.json  gpu_session/out/
scp pod:~/df11pack/gpu_session/out/SUMMARY.txt    gpu_session/out/
```

(Everything else on the pod — the venv, the installed CUDA wheels, the
regenerated repack variants — is disposable. Don't bother bringing any of
it back.)

---

## How to tell, from `results.json` alone, whether each item is confirmed or refuted

Each of the four entries under `items` has a top-level `"status"` field,
one of exactly three values:

| `status` | Meaning |
|---|---|
| `CONFIRMED` | The pre-registered confirming outcome in EXPECTED.md happened. Read `"summary"` for the one-line reason. |
| `REFUTED` | The pre-registered refuting outcome happened. The test ran correctly and gave a definitive, negative answer — this is not a script failure, and EXPECTED.md names the specific consequence to act on. |
| `ERROR` | The test could not reach a conclusion (crash, missing dependency, non-deterministic baseline, etc). `"detail"."traceback"` has the cause. This is the only status that means "something went wrong," and it means the item is still open — rerun after fixing the cause, don't treat it as an answer either way. |

`items.<name>.summary` is written to be read standalone (it names the
specific measurement, not just pass/fail). `items.<name>.detail` has the
raw evidence behind it — sha256 hashes of decoded output for H7 and the
LUT gate, exact max-abs-logit-diff numbers for H6 and H11, and the
repack/verify subprocess output for H11.

The four keys are `h7_decode_ptx`, `h6_reorder_inference`,
`h11_repack_inference`, `luts_correct_kernel` — matching priority order.

`SUMMARY.txt` is the same information formatted for reading directly,
no `jq` required.

---

## Stop the instance

**Before you close the terminal:** confirm you've copied `results.json`
and `SUMMARY.txt` off the pod, then **terminate/stop the pod from the
RunPod dashboard (or `runpodctl stop pod <id>` / `remove pod <id>`).**
Community Cloud bills by the second while the pod exists, running or not
if it's not actually terminated — there is nothing left to do on it once
those two files are copied out, so there is no reason to leave it up.
This is the last step. Do it now, not "in a minute."

---

## Honest wall-time estimate

| Phase | Estimate | Why |
|---|---|---|
| Pod boot + SSH ready | 1–3 min | Instance-dependent, not scripted. |
| Upload ~110 MiB | 1–4 min | Depends on your local upload bandwidth, not the pod's. |
| `run_all.sh` prereq checks | <10 s | Local checks only. |
| venv + pip installs (torch, cupy, transformers, deps) | 3–7 min | The dominant scripted cost — CUDA wheels are large (torch ~2 GB, cupy ~100–200 MB) even though the tests themselves are tiny. Datacenter-to-datacenter bandwidth on RunPod is usually fast; budget the high end if the mirror is slow. |
| H7 (decode.ptx via driver API vs CuPy) | 15–40 s | Dominated by CUDA context init and first-kernel JIT/compile overhead, not the decode itself (15.7M elements is small). |
| H6 (reorder inference, 3 forward passes) | 30–90 s | `transformers`/`accelerate` import overhead plus three `DFloat11Model.from_pretrained` calls on a 4-layer model. |
| H11 (repack + verify + 4 forward passes) | 60–120 s | Repack/verify are pure CPU I/O on 90 MiB (seconds); most of this is four separate model loads/forward passes, each reimporting torch/transformers fresh. |
| `--luts=correct` gate | 10–30 s | One unit, two decodes. |
| Aggregation + printing | <5 s | |
| **Total scripted time (`run_all.sh` alone)** | **~6–15 min** | |
| **Total session, boot to teardown** | **~15–25 min** | Budget **30 minutes** to be safe on a first run; a re-run (venv cached) is close to the scripted-time figure alone. |

At $0.17–0.25/hr this session costs on the order of **$0.05–0.10** total.
The number that matters is not the GPU-second cost, it's not leaving the
pod running after you're done — see "Stop the instance" above.
