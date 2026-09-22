# Pre-registration: what would confirm or refute each item

Written **before** the GPU session runs, against the scripts as they
stand in this directory. Not edited afterwards. The point of writing this
first is that once `results.json` exists, it is too late to decide what a
given number would have meant — that decision is made here, now, while
the answer is still unknown.

Each item names: what CONFIRMS it, what REFUTES it, and the specific,
concrete consequence for the project in each case. `ERROR` (the test
could not reach a conclusion at all — crash, missing dependency,
non-deterministic baseline) is not a third outcome to interpret; it means
the item is still open and must be rerun, and is covered once at the end
rather than per item.

---

## 1. H7 — `decode.ptx` without CuPy, via the raw CUDA driver API

**Script:** `test_h7_decode_ptx.py`

**Setup:** one real DF11 unit's five kernel-input tensors, decoded two
ways — CuPy's `RawModule`/`RawKernel` (the control, exactly what
`dfloat11.py` itself does) and the CUDA driver API reached directly via
`ctypes` against `libcuda.so` (`cuInit` → `cuCtxCreate` →
`cuModuleLoadData` → `cuModuleGetFunction` → `cuMemAlloc`/`cuMemcpyHtoD` →
`cuLaunchKernel` → `cuMemcpyDtoH`), using the exact kernel parameter types
read off `decode.ptx`'s own `.visible .entry decode(...)` declaration (6×
`u64` pointers, 3× `u32` scalars), no CuPy or torch involved in that path.

**CONFIRMS H7:** the driver-API path runs to completion and its output is
**bit-for-bit identical** to CuPy's output for the same input.

**REFUTES H7:** the driver-API path either (a) fails to load/launch the
kernel at all while CuPy succeeds on the same PTX, or (b) runs but
produces output that differs from CuPy's, even in one byte.

**Consequence if CONFIRMED:** df11pack's GPU verification path (whatever
compares its own decoder's output against the real kernel) can be
implemented as direct FFI from the Rust binary to `libcuda.so` — no
Python process, no CuPy dependency, no subprocess shim needed at
verification time. This removes a whole class of packaging complexity
from the Rust binary's GPU-verification story.

**Consequence if REFUTED:** the Rust binary cannot talk to `decode.ptx`
on its own. GPU verification has to shell out to (or embed) a Python
process with CuPy installed, adding a real runtime dependency and a
process-boundary to whatever design assumed a self-contained binary. This
would need to be reflected wherever the GPU-verification step is
specified in PLAN.md.

---

## 2. H6 inference confirmation — physical tensor order inside a shard

**Script:** `test_h6_reorder_inference.py`

**Setup:** the same 4-layer directory loaded through the real
`DFloat11Model.from_pretrained`, twice — once untouched, once with
`model_layers_0.safetensors` swapped for the pre-built reordered fixture
(same tensor names/dtypes/shapes/bytes, different physical byte order and
header key order) — run on identical `input_ids`, logits compared with
`torch.equal` (exact, not `allclose`). A same-directory, run-twice
determinism baseline is required to already be exact before the
reorder-vs-original comparison is treated as meaningful.

**CONFIRMS H6 (fully, upgrading it from "confirmed structurally"):** the
determinism baseline is exact, **and** the reordered-directory logits are
**exactly equal** to the original's.

**REFUTES H6:** the determinism baseline is exact, but the
reordered-directory logits differ from the original's by any amount.

**Consequence if CONFIRMED:** no constraint is added — df11pack's writer
already only needs to match the official *placement* rule (FINDINGS 0.7:
which shard a tensor lands in) and tensor names, not intra-shard physical
byte order. FINDINGS.md 0.8/H6 moves from "confirmed structurally,
inference-level open" to fully confirmed.

**Consequence if REFUTED:** physical tensor order inside a shard is
load-bearing for the official runtime despite the loader's own source
dispatching purely by name (as read in Phase 0). df11pack's writer would
need to reproduce the official compressor's *intra-shard tensor order*
exactly, as a new invariant, in addition to file grouping and tensor
placement — a real scope addition to Phase 2's writer and to
DESIGN.md/PLAN.md wherever shard-writing is specified.

---

## 3. H11 load confirmation — shard grouping and file naming

**Script:** `test_h11_repack_inference.py`

**Setup:** `phase0/repack_shards.py` run fresh on the box against the
uploaded 4-layer directory, producing a single-merged-file variant and a
4-file interleaved variant (names matching no unit, one unit's tensors
split across files), each verified byte-identical to the original by
`phase0/verify_repack.py` before proceeding. All three layouts (original,
merged, interleaved) loaded through `DFloat11Model.from_pretrained` and
run on identical `input_ids`; logits compared with `torch.equal`. Same
determinism-baseline precondition as H6.

**Scope note, stated once here rather than re-litigated per result:**
this deliberately reuses the 90 MiB tier-0 (4-layer) directory rather than
regenerating the full 870 MiB tier-1 (28-layer) output on the box. The
mechanism under test — `load_and_replace_tensors` dispatching purely by
tensor name, independent of file count or layer count (FINDINGS 0.8/H11,
confirmed from source) — has no layer-count dependence, so this is a
faithful, much cheaper substitute for the same hypothesis. If this
result and the full-size structural check in FINDINGS.md ever disagree in
some future run, that disagreement would itself be a new finding worth
its own investigation — not expected, not observed here.

**CONFIRMS H11 (fully):** determinism baseline exact, **and** both the
merged-file and interleaved-file variants produce logits **exactly equal**
to the original directory's.

**REFUTES H11:** determinism baseline exact, but either variant's logits
differ from the original's.

**Consequence if CONFIRMED:** df11pack's writer is free to choose any
shard-splitting scheme (size-balanced files, one file per pattern-match
group, a single file, whatever suits the Rust implementation) without
needing to replicate the official compressor's one-shard-per-layer
convention. FINDINGS.md 0.8/H11 moves from "confirmed structurally" to
fully confirmed.

**Consequence if REFUTED:** df11pack must replicate the official
per-layer shard grouping exactly (one `model_layers_N.safetensors` per
matched submodule, remainder tensors in `model.safetensors`), as a hard
compatibility requirement rather than an implementation choice — a
correction to DESIGN.md/PLAN.md wherever the writer's shard layout is
currently treated as free.

---

## 4. `--luts=correct` kernel gate

**Script:** `test_luts_correct_kernel.py`

**Setup:** the real, already-on-disk leaked unit
(`phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_0.safetensors`,
prefix table 3, positions 0..127, carrying 105 leaked from table 2 — the
exact instance named in the task brief, independently re-derived by the
script itself as a precondition check rather than only trusted). A
byte-identical copy is made with only those 128 leaked bytes zeroed
(verified surgical: every other tensor and every other LUT byte
unchanged). Both files' unit is decoded with the real, unmodified CUDA
kernel via CuPy; the two full decoded weight streams (15,728,640 elements
each) are compared bit-for-bit.

**CONFIRMS the mode is safe to ship:** the zeroed-copy decode is
**bit-for-bit identical** to the original leaked-file decode. This is the
direct test COMPATIBILITY.md says is missing — "nobody has run the
official kernel over a `--luts=correct` file and compared the decoded
tensor bit-for-bit against the source."

**REFUTES the mode:** the two decodes differ in any byte.

**Consequence if CONFIRMED:** `--luts=correct` stays exactly as specified
in COMPATIBILITY.md and PLAN.md step 1.6 — off by default, printing its
warning, stamped in `__metadata__`, and now able to have its gating
condition ("not permitted in any released artefact until the kernel test
in Phase 5 passes") marked satisfied by this result. COMPATIBILITY.md's
open question is closed.

**Consequence if REFUTED:** per COMPATIBILITY.md's own stated rule —
*"If the kernel test ever fails ... `--luts=correct` is not 'more
correct', it is simply wrong, and it must be removed rather than
documented around"* — **`--luts=correct` is removed from PLAN.md step 1.6
and from COMPATIBILITY.md.** Not disabled, not re-gated further: removed.
The only supported LUT mode becomes `--luts=compat` (the carry-forward
bug, reproduced exactly), unconditionally.

---

## On `ERROR` results

An `ERROR` status (crash, missing upload, `torch.cuda.is_available()`
false, a determinism baseline that isn't exact, etc.) is not evidence for
or against the hypothesis — it means the experiment didn't run. Re-read
`detail.traceback` in `results.json`, fix the cause, and re-run
`gpu_session/run_all.sh` (safe and idempotent — see RUNBOOK.md). Do not
record an `ERROR` item as either confirmed or refuted in FINDINGS.md; it
stays open, exactly as it is today, until it produces a real
CONFIRMED/REFUTED result.
