# df11pack — Implementation plan

Ordered plan derived from [`DESIGN.md`](DESIGN.md). Phase 0 and Phase 1 are
specified step by step; Phases 2–7 are outlined, to be expanded when their
predecessor's exit gate is met.

**Rule for the whole plan:** no step in Phase 1 or later starts until the Phase 0
table of hypotheses is closed. Where a step can be reordered, that is stated
explicitly; otherwise the order is a dependency order.

Effort estimates are in **days of focused work by one person**, and they are
rough. They exclude model download time and waiting on rented hardware.

---

## Resource inventory

### The development machine is the binding constraint

Measured, not assumed: **3 GB of total RAM** (~2 GB available) and **39 GB free
disk**. Every sizing decision below follows from those two numbers, and they are
tighter than anything in DESIGN.md contemplates.

This is not a handicap to work around — it is the target environment. The guiding
principle says anyone must be able to compress models on a modest machine. If
df11pack is developed on one, that property is tested continuously instead of
being asserted and discovered false at the end.

What it does constrain is the **reference** side: the official compressor is the
thing that needs RAM, and we need to run it to produce fixtures. Hence the
tiered corpus below, whose first tier is sized so the official tool fits
comfortably in ~1 GB.

### Hardware

| Resource | Needed by | Note |
|---|---|---|
| **This machine (3 GB RAM, 39 GB disk)** | **All of Phase 1, and Phase 0 tiers 0 and 2** | Sufficient for the whole development spine. df11pack's own budget is ~1.35 × N per worker; on tier 0 that is single-digit MB. |
| **~64 GB RAM — optional, one step** | Step 0.4 only, and only to confirm H1 *at full scale* | This is the RAM the **official** compressor needs (~48 GB peak on 12B Flux), not anything df11pack needs. H1 is the hypothesis that this peak is unnecessary — it motivates the project but constrains no design decision. Without such a machine, measure the scaling curve on the models that do fit and cite the official README's figure for the endpoint. |
| **NVIDIA GPU** | Steps 0.10 (H7), 0.11 (`check_correctness` path), and Phase 5 | Verification only. There is no local GPU, so these steps need rented hardware; the sibling repo `../bf16-exponent-compression/RENT_A_GPU.md` already documents a working rental and setup procedure. |
| A machine with **4–8 GB** | Tier 1 only: running the *official* compressor on full Qwen3-0.6B | Needed only if step 0.2c finds that the LLM `pattern_dict` puts the 155.6M-weight embedding in a single UC — see 0.2c. Rentable by the hour; not on the critical path. |
| An old machine (4 cores / 8 GB / HDD) | Phase 3 exit gate, success metrics | The development machine already is the low-RAM reference. What it may not provide is an HDD and a multi-core comparison point; emulate with cgroup limits and `--workers 1` where it cannot. |

### Corpus, in three tiers

**Tier 0 — the development spine.** Models small enough that the *official*
compressor runs locally in well under 1 GB, so every fixture is produced and
compared on this machine, with no downloads and no rentals. This is where Phase 1
is graded. Built in step 0.2.

**Tier 1 — one real LLM.** Full Qwen3-0.6B, already on disk. Proves the encoder
on a real weight distribution rather than a synthetic one.

**Tier 2 — real diffusion models, read-only.** The published DF11 releases,
used for structural questions only (which tensors exist, `split_positions`, LUT
shapes). Never compressed locally. Disk-bound: 39 GB free means one
source/release pair at a time, deleted before fetching the next.

| Model | Tier | Size | Needed by | Source |
|---|---|---|---|---|
| Truncated Qwen3-0.6B + synthetic cases | 0 | <100 MB each | 0.2, and all of Phase 1 | Generated locally from the model below |
| **Qwen3-0.6B (BF16) — already on disk** | 1 | 1.5 GB | 0.2, 0.2c, 0.12 | `../bf16-exponent-compression/real_model/model.safetensors`, already used by the sibling project. **No download needed.** |
| **Official pre-compressed DF11 releases** | ~11–17 GB each | **0.2b, 0.7, 0.8, 0.9, 0.12** | `DFloat11/Chroma-DF11`, `DFloat11/FLUX.1-dev-DF11`, `DFloat11/FLUX.1-schnell-DF11` (diffusers layout) and `mingyi456/Chroma1-Base-DF11` (ComfyUI-native). These **are** the official compressor's output — downloading them replaces running it. See 0.2b. |
| `lodestones/Chroma1-HD` | ~18 GB | 0.7, 0.9 (as the *source* side of the comparison) | single-file, ComfyUI-native layout |
| `imnotednamode/Chroma-v36-dc-diffusers` (`subfolder="transformer"`) | ~18 GB | 0.8 (H10), 0.9, 1.8 | diffusers layout; also settles the §1.7 "still to verify" item |
| Flux BF16 (both layouts) | 2 | ~24 GB each | 0.7, 0.12 | Source side for the Flux comparisons. **Does not fit alongside its DF11 release** in 39 GB — fetch, compare, delete, one at a time, or defer to Phase 2. |

Tier 0 is the only corpus most steps need, and it is generated, not downloaded.

### Software to pin

The official toolchain is the ground truth, so its versions are part of the
experiment. Pin and record: `dfloat11` (official), `dahuffman` **0.4.2**,
`diffusers` **0.40.0**, `torch`, `safetensors`, `numpy`, plus the
ComfyUI-DFloat11-Extended checkout. Record exact versions and commit SHAs in the
Phase 0 findings; a different `dahuffman` or `diffusers` can silently change what
"byte-identical" means.

---

## Phase 0 — Measure the official compressor, settle H1–H13

**Phase goal:** turn every [HYPOTHESIS] in DESIGN.md into a measured
[CONFIRMED] or [REFUTED], and produce the golden fixtures Phase 1 will be graded
against. No Rust is written in this phase.

**Phase exit gate:** DESIGN §2's table is closed — every row has a result, a
method, and the raw data committed. `docs/FINDINGS.md` exists and each refuted
hypothesis has a stated consequence for the design.

**Phase effort:** ~12–18 days, of which a large share is waiting on downloads and
on the official compressor's own runtime.

### 0.1 — Pinned reference environment

- **Goal:** a reproducible environment running the official compressor, and a readable local checkout of the sources DESIGN.md cites.
- **Inputs:** none.
- **Actions:** create `phase0/` with a pinned virtualenv (requirements file with exact versions); install the official `dfloat11`, `dahuffman==0.4.2`, `diffusers==0.40.0`, torch, safetensors; clone `LeanModels/DFloat11` and `mingyi456/ComfyUI-DFloat11-Extended` at recorded SHAs for reading; write `phase0/README.md` describing how to rebuild the environment from scratch.
- **Produces:** `phase0/requirements.txt`, `phase0/README.md`, recorded SHAs.
- **Verification:** compress Qwen3-0.6B end to end with the official tool and load the result back with the official loader. If that does not work, nothing downstream is trustworthy.
- **Depends on:** —
- **Effort:** 1 day.

### 0.2 — Corpus generation and acquisition

- **Goal:** have every test corpus from DESIGN §8 on disk, in a documented layout.
- **Inputs:** 0.1.
- **Actions:** build **tier 0 first**: (a) a truncated Qwen3-0.6B — take the on-disk model, keep 2–4 transformer layers, drop or shrink the embedding, and write a fresh safetensors of well under 100 MB, so the official compressor runs on it in a few hundred MB of RAM; (b) write generator scripts for the reduced synthetic Flux in **both** layouts (instantiate `FluxTransformer2DModel` with a reduced config for diffusers, the ComfyUI equivalent class for native), for the sharded diffusers variant with `index.json`, and for the adversarial distributions (2/3/4 LUT levels, the 32-bit limit path, single-exponent UC, awkward sizes, exponents 240–255, single-tensor UC, very small UC). Start the Chroma downloads in the background — they are large and gate steps 0.8–0.9.
- **Produces:** `phase0/corpus/` with a manifest (name, tier, layout, size, generator seed, sha256 per file), plus the measured official-compressor peak RSS for each tier-0 case, so the corpus is provably runnable here.
- **Verification:** each synthetic model loads with the loader of its own ecosystem; the adversarial cases provably hit the paths they claim to (assert on the codebook the official `get_32bit_codec` builds).
- **Depends on:** 0.1.
- **Effort:** 3 days. The adversarial generators are most of it — hitting the 32-bit limit path on purpose is fiddly.

### 0.2b — Official DF11 releases as reference output

- **Goal:** obtain the official compressor's output for real models **without running the official compressor**, which is what the large-RAM requirement was really about.
- **Inputs:** 0.1.
- **Actions:** download the published DF11 files — `DFloat11/Chroma-DF11`, `DFloat11/FLUX.1-dev-DF11`, `DFloat11/FLUX.1-schnell-DF11` for the diffusers layout, `mingyi456/Chroma1-Base-DF11` for ComfyUI-native — together with the matching BF16 sources. Record for each: which upstream revision it was compressed from, the `dfloat11_config` it carries, and its file hashes. Where the compressed release and the BF16 source are not provably the same revision, mark that pair as usable for *structural* questions (which tensors exist, `split_positions`, LUT shapes) but not for byte-identity claims.
- **Produces:** `phase0/corpus/official/` plus a provenance note per model.
- **Verification:** each downloaded file passes the 0.6 invariant checker and loads with its own ecosystem's loader.
- **Depends on:** 0.1.
- **Effort:** 1 day, mostly download time.
- **Why this step exists:** the pass criterion is "byte-identical to the official compressor's output". That needs the output, not the compressor. Reading a published DF11 file is a streaming read bounded by its largest tensor — a few hundred MB — so every real-model question below drops from ~64 GB to a laptop.

### 0.2c — Does the LLM pattern put the embedding in one UC? — **DONE**

- **Status:** resolved. Full write-up in [`FINDINGS.md`](FINDINGS.md). Short version: the pattern is a per-model choice; `Qwen3-4B-DF11` compresses only `model.layers.\d+`, while 8B and above add `lm_head` and `model.embed_tokens` as **standalone single-tensor units**. No 0.6B release exists, so the choice is ours. Under layers-only the largest unit here is 15.73M weights and the official compressor is predicted to fit in ~1.6 GiB; under the mainstream pattern it is 155.58M weights and ~3.3 GiB, which does not fit. **Tier 1 runs locally under layers-only**, pending empirical confirmation once 0.1 exists. Two consequences promoted into Phase 1: single-tensor units are mainstream rather than an edge case, and step 1.8's schema must express them.
- **Goal:** find out whether tier 1 is runnable on this machine, before planning around it.
- **Inputs:** the official example `pattern_dict` for Qwen-class LLMs; the on-disk Qwen3-0.6B.
- **Actions:** the file holds 751.6M weights in 311 tensors, of which **two are 155.6M each**: `model.embed_tokens.weight` and `lm_head.weight`, which are **byte-identical** (`tie_word_embeddings: true`, yet both are stored). Determine from the official `pattern_dict` whether either falls inside a compression unit. If neither does, the largest UC is a per-layer group of a few million weights and the official compressor runs here in a few hundred MB — tier 1 needs no special machine. If one does, that single UC's `.tolist()` alone is ~1.24 GB and the official tool will not fit in 3 GB; tier 1 then needs a rented 4–8 GB box for one run, and the fixture is kept forever after.
- **Produces:** a yes/no that decides whether the 4–8 GB row in the hardware table is ever exercised.
- **Verification:** confirm by running the official compressor on tier 1 under a memory cap and observing either success or a clean OOM at the predicted unit.
- **Depends on:** 0.1.
- **Effort:** half a day.
- **Second finding, recorded here:** those two identical 296.8 MiB tensors mean a tied-embedding LLM presents df11pack with two units whose input bytes are identical, so their entire output would be identical too. Deduplicating them would halve the work for this whole model class — and would **break the byte-identity criterion**, because the official compressor emits both. So: detect and report it, never act on it by default. Revisit as an opt-in flag after v1. The sibling project hit the same duplication from the other direction, where it mattered for weight counts rather than for work.

### 0.3 — H2: where does the time go

- **Goal:** confirm or refute that >90% of official compression time is the Python `encode` loop.
- **Inputs:** 0.1, 0.2 (Qwen3-0.6B and synthetic Flux).
- **Actions:** run the official compressor under `py-spy record`; produce a flame graph and a per-stage breakdown (cat / histogram / codec / LUT / encode / correctness check / serialise). Repeat on two model sizes to see how the split scales.
- **Produces:** flame graphs and a stage-time table in `phase0/out/h2/`.
- **Verification:** the stage times sum to wall-clock within a few percent; the result is stable across runs.
- **Depends on:** 0.2.
- **Effort:** 1 day.
- **Consequence if refuted:** if another stage dominates, the §5.3 encoder design is no longer the main lever, and Phase 1's ordering must be reconsidered before any Rust is written.

### 0.4 — H1: where does the memory go

- **Goal:** decompose the official peak RAM into loading, per-UC temporaries and serialisation, and check that the peak scales with model size.
- **Inputs:** 0.2.
- **Actions:** `/usr/bin/time -v` for peak RSS, plus an RSS sampler tagging each stage, plus `tracemalloc` for the Python-side allocations named in DESIGN §1.2. Run on Qwen3-0.6B and the synthetic Flux; run on real Flux **only if a ~64 GB machine is available**.
- **Produces:** peak-RSS-by-stage table and the per-UC temporary breakdown.
- **Verification:** the measured breakdown accounts for the observed peak; the ~48 GB figure for 12B Flux is reproduced or explained.
- **Depends on:** 0.2.
- **Effort:** 1–2 days.
- **Machine note:** this is the *only* step that would benefit from ~64 GB, and only for the 12B endpoint. Measure the curve across Qwen3-0.6B, the synthetic models and the largest model that fits; if it scales as predicted, the endpoint is extrapolation plus the official README's own figure. Do not block the phase on it.

### 0.5 — H4: replicate `dahuffman` exactly

- **Goal:** prove, before writing Rust, that `dahuffman`'s codebook is deterministically reproducible from the histogram alone.
- **Inputs:** 0.2 corpus histograms.
- **Actions:** extract exponent histograms from every corpus UC; write an independent reimplementation of `dahuffman`'s tree construction and tie-breaking (heap of `(frequency, [(symbol, code)])`, EOF comparing smallest) in Python **without importing `dahuffman`**; compare code lengths and codes symbol by symbol on every histogram, including the adversarial ones. Do the same for `get_32bit_codec`'s limiting loop and for `get_luts`.
- **Produces:** the reimplementation (as an executable spec for the Rust port), plus golden fixtures: histogram → codebook → LUT triples in `phase0/out/golden/codebooks/`.
- **Verification:** exact match on 100% of the corpus histograms. Any mismatch is a refutation and must be characterised, not patched around.
- **Depends on:** 0.2.
- **Effort:** 2–3 days.
- **Known risk:** `get_32bit_codec` uses `np.argpartition`, whose tie ordering is implementation-defined. Isolate how often ties actually occur on real histograms — this determines whether the documented exception in the brief is a corner case or a routine one.

### 0.6 — H3 write-up and format-invariant assertions

- **Goal:** turn DESIGN §1.3's invariants into executable checks, so later phases can assert rather than assume.
- **Inputs:** 0.2, official outputs.
- **Actions:** write a checker that, given an official DF11 file, validates every invariant in §1.3 (bit order, EOF and padding, `gaps` width and padding to 512 windows, `output_positions` layout and trailing `len(data)`, LUT ≥240 jump convention and ≤16 tables, `split_positions` as cumulative sums in `attr_names` order, the 2³¹ limits). H3 itself needs no measurement — it is confirmed in source — but it is recorded here with its evidence.
- **Produces:** `phase0/check_invariants.py`, run over every official output produced in this phase.
- **Verification:** it passes on every official file and fails on deliberately corrupted copies.
- **Depends on:** 0.2.
- **Effort:** 2 days.

### 0.7 — H8 / H9: are the uncompressed tensors untouched

- **Goal:** settle the critical question — we work on the *file*, the official tool works on the *instantiated model*.
- **Inputs:** 0.2 corpus, official outputs in both layouts.
- **Actions:** for ComfyUI-native (H8) and diffusers (H9), compare tensor by tensor between the source and the official output: names, dtypes, shapes and bytes of everything that belongs to no UC. Catalogue every difference found (added buffers, dtype casts, renames, `swap_scale_shift`-style transforms) per architecture and per layout.
- **Produces:** a difference catalogue; if empty, that is the result.
- **Verification:** byte comparison, not shape comparison.
- **Depends on:** 0.2, 0.2b.
- **Effort:** 2 days. Runs on any machine: both sides are read tensor by tensor from disk, never instantiated. Use the 0.2b downloads as the official side, subject to their provenance caveat.
- **Consequence if refuted:** every catalogued transform becomes a required, architecture-specific feature of the writer in Phase 2, and the "byte-identical" criterion must be restated per tensor class.

### 0.8 — H10 / H11 / H6: what the loaders actually depend on

- **Goal:** find out how much freedom the writer has.
- **Inputs:** 0.2 (diffusers corpus, sharded variant), official outputs.
- **Actions:**
  - H10: rebuild `config.json` from the source config + the fields `save_pretrained` adds + `dfloat11_config`, **without diffusers installed**, and diff against the official one. Note every field that turns out to be diffusers-version-dependent.
  - H11: repack the official output into one shard, and into arbitrary groupings, and load each with `DFloat11Model.from_pretrained`.
  - H6: reorder the tensors physically inside a single-file output and load it with both the official loader and the Extended node.
- **Produces:** a "loader degrees of freedom" note; a `config.json` reconstruction script.
- **Verification:** identical inference output (fixed seed, fixed prompt) across all variants, not merely "it loads".
- **Depends on:** 0.2, Chroma diffusers download.
- **Effort:** 2–3 days.

### 0.9 — H13 and the Chroma diffusers naming

- **Goal:** confirm that Chroma's approximator `in_proj`, `out_proj` and RMSNorms stay uncompressed, and nail down the diffusers UC composition and order that DESIGN §1.7 leaves marked "still to verify".
- **Inputs:** both Chroma checkpoints.
- **Actions:** read the tensor list straight out of `DFloat11/Chroma-DF11` and `mingyi456/Chroma1-Base-DF11` — that list *is* the answer to H13, since an uncompressed tensor appears under its own name and a compressed one does not. Compare against DESIGN §1.7. Dump the real key names from `imnotednamode/Chroma-v36-dc-diffusers`. Derive the concatenation order empirically from the `split_positions` carried in the published file, divided by the per-tensor shapes from the BF16 source.
- **Produces:** the confirmed per-layout Chroma definitions, ready to be transcribed into the Phase 1 architecture data files.
- **Verification:** `split_positions` computed from our derived order matches the official file exactly.
- **Depends on:** 0.2b, 0.7.
- **Effort:** 2 days. Header reads and one `split_positions` tensor per UC — negligible RAM.
- **Note:** this is the single most design-relevant unknown left in DESIGN.md. The diffusers concatenation order cannot be guessed; it has to come out of an official file.

### 0.10 — H7: loading `decode.ptx` without CuPy

- **Goal:** decide whether GPU verification can live inside the Rust binary.
- **Inputs:** official `decode.ptx`.
- **Actions:** minimal native program that loads the PTX through the CUDA driver API and runs the kernel on a small known UC; compare against the CuPy path.
- **Produces:** a yes/no with a working example either way.
- **Verification:** bit-identical decoded output from both paths.
- **Status: DONE — CONFIRMED.** Run in the batched GPU session; `decode.ptx` loads and executes through the raw CUDA driver API via `ctypes`, bit-identical to CuPy. Open decision 1 is closed: no shim is needed.
- **Depends on:** 0.1. **Needed an NVIDIA GPU — rented, ~$0.09.**
- **Effort:** 1–2 days.

### 0.11 — H5 groundwork: disk and encode-speed baselines

- **Goal:** establish the target the native encoder has to beat for "disk-bound" to be a meaningful claim.
- **Inputs:** the reference machines.
- **Actions:** measure sequential read throughput on each reference disk (HDD and SSD/NVMe, cold cache); measure the official encoder's symbols/s as the floor; compute the symbols/s a native encoder must reach to saturate each disk.
- **Produces:** a target table: disk → required encode throughput.
- **Verification:** —
- **Depends on:** 0.1.
- **Effort:** 1 day.
- **Note:** H5 itself **cannot be settled in Phase 0** — it needs the native encoder. What Phase 0 produces is the target; H5 closes at the Phase 3 exit gate. See "Inconsistencies" below.

### 0.12 — Golden fixture dump

- **Goal:** freeze the official outputs that Phase 1 will be graded against, so Phase 1 needs neither Python nor a large machine.
- **Inputs:** everything above.
- **Actions:** for every corpus case, store the official output tensors plus the intermediate artefacts (histogram, codebook, LUTs, `gaps`, `output_positions`, `split_positions`) with a manifest and per-tensor sha256. Tier 0 and (per 0.2c) tier 1 come from running the official compressor locally — these are the fixtures that matter, because they are produced from a source we control end to end. Tier 2 comes from the 0.2b downloads. Keep the large ones out of git (see `.gitignore`) with a documented regeneration script and recorded hashes.
- **Produces:** `fixtures/` plus `fixtures/MANIFEST.json`, each entry tagged `locally-compressed` or `published-release` so a byte-identity failure on a published-release fixture is triaged against its provenance caveat before being treated as an encoder bug.
- **Verification:** the regeneration script reproduces identical hashes on a second machine.
- **Depends on:** 0.2–0.9, 0.2b.
- **Effort:** 2 days.

### 0.13 — `docs/FINDINGS.md`

- **Goal:** close the phase.
- **Actions:** one section per hypothesis: method, raw numbers, verdict, and — for anything refuted — the specific consequence for DESIGN.md. Amend DESIGN.md where a [HYPOTHESIS] marker becomes [CONFIRMED] or [REFUTED].
- **Produces:** `docs/FINDINGS.md`; an updated DESIGN.md.
- **Verification:** no row of the §2 table is left open, including an explicit "deferred to Phase N, here is why" for H5 and H12.
- **Effort:** 1 day.

---

## Phase 1 — Native single-threaded encoder for one UC

**Phase goal:** given one UC's worth of BF16 weights in memory, produce every DF11
tensor for it, byte-identical to the official compressor. Single-threaded, no
streaming, no file writing yet.

**Phase exit gate:** byte-for-byte identity against the tier-0 fixtures —
truncated Qwen3, synthetic Flux in both layouts, and every adversarial
distribution — including a correct abort on the exponents 240–255 case. Full
Qwen3-0.6B (tier 1) is the first thing tried after the gate, not part of it.

**Phase effort:** ~12–16 days. No GPU, no downloads, no rentals: it runs entirely
against the tier-0 fixtures on the development machine. Tier-0 units are small
enough that the whole encode chain fits in single-digit MB, so memory is never
the thing under test here — correctness is.

### 1.1 — Cargo workspace skeleton

- **Goal:** the project structure, chosen once.
- **Actions:** create the workspace with a library crate (the codec) and a thin binary crate (CLI later). Pin the Rust toolchain in `rust-toolchain.toml`. Add `safetensors`, `memmap2`, `serde`/`serde_json`, `toml`, `xxhash-rust`; leave `rayon` out until Phase 3 so nothing accidentally parallelises early. Set up CI running `cargo test`, `clippy -D warnings` and `fmt --check`.
- **Produces:** `Cargo.toml`, crate skeletons, CI config.
- **Verification:** CI green on an empty test suite.
- **Depends on:** Phase 0 closed.
- **Effort:** 1 day.

### 1.2 — Fixture harness

- **Goal:** make "byte-identical to the official output" a one-command test from day one, before any codec code exists.
- **Actions:** a test-support module that loads `fixtures/MANIFEST.json`, exposes each case as a test input, and provides an assertion that compares a produced tensor against the fixture byte for byte with a useful diff on failure (first differing byte, its bit offset, the surrounding symbols).
- **Produces:** the harness plus one deliberately failing test proving the diff output is readable.
- **Verification:** the harness detects a single flipped bit in a multi-MB tensor and reports its position correctly.
- **Depends on:** 1.1, 0.12.
- **Effort:** 2 days.
- **Note:** this step comes *before* the codec on purpose. A test suite that has never been observed failing for the right reason is not evidence.

### 1.3 — Field split

- **Goal:** BF16 buffer → exponent stream + `sign_mantissa` bytes.
- **Actions:** implement the split per DESIGN §1.3 (exponent = bits 7..14 of the int16 pattern; `sign_mantissa` = sign bit | 7 mantissa bits). Straightforward scalar code first; leave vectorisation to the compiler and confirm with a disassembly check rather than hand-written intrinsics.
- **Produces:** the split function.
- **Verification:** `sign_mantissa` byte-identical to the fixture for every corpus UC; round-trip property test (split then recombine = original) over random BF16.
- **Depends on:** 1.2.
- **Effort:** 1 day.

### 1.4 — Exponent histogram and validation gates

- **Goal:** the 256-bin histogram, plus the cheap checks that must never be skipped.
- **Actions:** histogram; then the gates from DESIGN §7.1 — abort with a clear, actionable error on any exponent in 240–255, on a UC exceeding 2³¹−1 weights or bytes, on more than 16 prefix tables, and on a maximum code length above 32 bits that the limiter fails to fix.
- **Produces:** histogram + a typed error enum for every abort condition.
- **Verification:** histograms match the fixtures; the adversarial 240–255 case aborts with the right error and writes nothing.
- **Depends on:** 1.3.
- **Effort:** 1 day.

### 1.5 — `dahuffman`-compatible codebook

- **Goal:** the highest-risk step in the phase — port the Phase 0 reimplementation to Rust.
- **Actions:** port the tree construction and tie-breaking exactly as characterised in 0.5 (heap of `(frequency, [(symbol, code)])`, EOF comparing smallest); then the 32-bit limiting loop of `get_32bit_codec`, including the `np.argpartition` tie behaviour as characterised in 0.5.
- **Produces:** the codebook builder.
- **Verification:** exact codebook match against the 0.5 golden fixtures for every corpus histogram, plus the adversarial ones. This is where H4 is finally proven in the real implementation.
- **Depends on:** 1.4, 0.5.
- **Effort:** 3 days.
- **Risk:** if the `argpartition` tie ordering cannot be replicated, invoke the brief's documented exception — but only here, only with the divergence recorded, and only with the result verified against the kernel.

### 1.6 — Hierarchical LUT builder

- **Goal:** the `(n_prefixes + 1, 256)` uint8 LUTs with the ≥240 jump convention, in **both** modes from the start.
- **Actions:** port `get_luts` including its carry-forward accumulator, which leaks across prefix tables (DESIGN §1.3, FINDINGS §0.5) — this is the default `--luts=compat` path and it must reproduce the leak exactly. Then add `--luts=correct`, which deterministically zero-fills those positions, behind the gate specified in [`COMPATIBILITY.md`](COMPATIBILITY.md): off by default, warns on use, stamps `df11pack_luts="correct"` into the safetensors `__metadata__`, and is barred from any released artefact until the Phase 5 kernel test passes. Enforce the ≤16-table bound in both.
- **Produces:** the LUT builder with a mode switch; the metadata stamp; the warning.
- **Verification:** in compat mode, LUT tensors byte-identical to the fixtures including the adversarial 2/3/4-level cases **and** a purpose-built case whose second or later prefix table lacks key `0`, asserting the leaked bytes match. In correct mode, assert the two outputs differ in exactly the positions predicted and nowhere else, and that the stamp is present. A test that never observes the leak is not evidence the leak is reproduced.
- **Depends on:** 1.5.
- **Effort:** 3 days (was 2; the second mode and the leak fixture are the addition).

### 1.7 — Bit writer, `gaps`, `output_positions`

- **Goal:** the bitstream itself, single-threaded, plus the two index tensors.
- **Actions:** a 64-bit accumulator bit writer, MSB-first within each byte, branchless in the common path; emit EOF and pad to a byte boundary; derive `gaps` (5 bits per 64-bit window, padded with zeros to a multiple of 512 windows, then `packbits`-equivalent) and `output_positions` (uint32 stored as a uint8 view, one entry per 4096-byte chunk, with the trailing `len(data)`) **in the same pass**, exactly as DESIGN §5.3 step 4 requires.
- **Produces:** `encoded_exponent`, `gaps`, `output_positions`.
- **Structural requirement:** `gaps` and `output_positions` are produced by a **swappable index component**, not inline in the bit writer. The interface, its two rules (a builder may refuse; only `df11` may claim compatibility) and the already-designed `idx8` alternative are specified in [`INDEX_SCHEMES.md`](INDEX_SCHEMES.md), whose "Phase 1 obligations" section is normative for this step. Implement `df11` behind the interface and a failing stub to prove the seam is load-bearing; do **not** implement `idx8`. The two are derived in the same pass for speed, so the seam is an interface the pass calls, not a separate pass. This costs nothing now and is what makes any future alternative index scheme additive rather than a rewrite — see [`COMPATIBILITY.md`](COMPATIBILITY.md) "Format divergence". It does **not** imply any such scheme will be built.
- **Verification:** all three byte-identical to the fixtures on every corpus case; the 0.6 invariant checker passes on the produced tensors.
- **Depends on:** 1.6.
- **Effort:** 3 days.
- **Note:** deliberately written as a single-threaded reference here. The chunked version from §5.3 lands in Phase 3 and must reproduce this one's output exactly — that equality is Phase 3's real test.

### 1.8 — Architecture definitions as data

- **Goal:** Flux and Chroma, both layouts, as versioned TOML — not compiled in.
- **Actions:** define the schema (architecture, layout, key prefix and its stripping rule, UC patterns, `attr_names` in concatenation order, format version, `threads_per_block`, `bytes_per_thread`); transcribe Flux from Extended's `pattern_dict.py` and the official examples, and Chroma from the definitions **confirmed empirically in 0.9**; include a loader with schema validation and clear errors.
- **Produces:** `data/architectures/*.toml`, the loader.
- **Verification:** `split_positions` computed from each definition matches the official file's for every corpus case — this is what proves the concatenation order is right.
- **Depends on:** 0.9, 1.7.
- **Effort:** 2–3 days.
- **Note:** ChromaRadiance is *not* included unless open decision 5 says so; if it is, the single-tensor UC case must be in the suite before the definition lands.

### 1.9 — `split_positions` and UC assembly

- **Goal:** tie it together — given a UC definition and the source tensors, produce the full set of six output tensors.
- **Actions:** concatenate in `attr_names` order, compute `split_positions` as int64 cumulative sums (empty for a bare `nn.Linear`), run the whole chain, name the outputs `<uc>.luts`, `.encoded_exponent`, `.sign_mantissa`, `.output_positions`, `.gaps`, `.split_positions`.
- **Produces:** the per-UC encode entry point.
- **Verification:** the phase exit gate — all six tensors byte-identical for every UC of corpus cases 1, 2 and 4.
- **Depends on:** 1.8.
- **Effort:** 2 days.

---

## Phases 2–7 — outline

To be expanded into step detail when the preceding exit gate is met.

### Phase 2 — Readers, writers, config

Single-file and sharded-with-index readers; writers for both layouts (single
safetensors, and the directory of one shard per UC + `diffusion_pytorch_model.safetensors`);
`config.json` reconstruction from the 0.8 script; the header-reservation trick so
the file is written sequentially with no holes.
**Exit gate:** the official loaders load our output and produce identical
inference, in both ecosystems; H9–H11 confirmed against real files.
**Needs:** the real models, and a machine that can run inference to compare.
Rough effort: 10–14 days.

### Phase 3 — Streaming and parallelism

The chunked §5.3 encoder (per-chunk histogram → bit-length prefix sum → encode
from a known offset, OR-combining boundary bytes); inter-UC parallelism with
`rayon`; the bounded UC queue; the worker-count formula from the RAM budget; the
HDD/SSD detection and single-sequential-reader policy.
**Exit gate:** measured bounded RAM (including the 512 MB / one-worker case),
output identical to Phase 1's single-threaded encoder, and **H5 settled** against
the 0.11 target table.
**Needs:** the old machine (or a cgroup-limited emulation of it), and an HDD.
Rough effort: 12–16 days.

### Phase 4 — Journal, committer, resume

Append-only journal with xxh3 hashes and the source fingerprint; the in-order
committer with spill for out-of-order UCs; `.tmp` + fsync + atomic rename for
diffusers shards; truncate-to-last-good-entry on resume.
**Exit gate:** killing the process at 100 random points yields, on resume, output
identical to an uninterrupted run.
Rough effort: 8–12 days.

### Phase 5 — Safe mode

The parallel CPU decoder mirroring the kernel's three phases (independent
windows from their `gap`, Blelloch-style prefix sum, chunk-wise comparison
against the source without materialising the decoded UC), and the GPU path via
the official kernel.
**Exit gate:** deliberately injected errors in the bitstream, in `gaps` and in
`output_positions` are all caught and localised; **H12 measured** across core
counts; and **the `--luts=correct` kernel test passes** — run the official kernel
over a `--luts=correct` file and confirm the decoded tensor is bit-identical to
the source, proving the leaked LUT positions are genuinely unreachable. Until
this passes, `--luts=correct` ships in no released artefact; if it fails, the mode
is removed rather than documented around.
**Needs:** a rented NVIDIA GPU for the kernel path.
Rough effort: 10–14 days.

### Phase 6 — Standalone post-hoc verification

The three levels: journal-hash integrity; stratified sampling (≥1 chunk per UC,
first/last chunk of each UC, every `split_positions` boundary chunk, the rest
uniform to 1000, seed recorded); full sweep.
**Exit gate:** detects corruption injected directly into the file on disk.
Rough effort: 6–8 days.

### Phase 7 — Remaining architectures

The rest of Extended's `pattern_dict.py` plus the official examples (Wan2.1,
LLMs), each as a data file with its own golden tests per layout.
**Exit gate:** golden tests pass per architecture and per layout.
Rough effort: open-ended; roughly 1–2 days per architecture once the pattern is
established.

---

## Open decisions

Carried from DESIGN §11 — listed here, not decided.

1. **If H7 fails:** an optional Python + CuPy shim for GPU verification, or CPU decoder only? Blocks nothing before Phase 5, but it changes what "no Python at runtime" means.
2. **Licence.** The official code is Apache-2.0. Replicating the format and construction algorithm is compatible with any permissive licence; redistributing `decode.ptx` carries its licence and attribution. Should be settled before any public release, and it affects whether Phase 5 can ship the PTX.
3. **ComfyUI integration:** CLI binary only, or also a thin node that invokes it?
4. **Final project name** (`df11pack` is provisional). Cheapest to change now.
5. **Does ChromaRadiance ship in v1?** One extra definition, but it drags in the single-tensor UC edge case (DESIGN §1.7). Affects step 1.8.

---

## Inconsistencies and open questions raised while writing this plan

1. **H5 and H12 cannot be settled in Phase 0.** The brief's Phase 0 gate says "settle H1–H13", but H5 ("with a native encoder the disk becomes the bottleneck") requires the native encoder, and H12 ("a CPU decoder scales near-linearly with cores") requires the CPU decoder. DESIGN §10 agrees with the later placement — it puts H12's measurement in Phase 5. This plan therefore produces the H5 *target* in step 0.11 and settles H5 at the Phase 3 gate, and settles H12 at the Phase 5 gate. Flagged rather than silently resolved.
2. **DESIGN §10's Phase 0 row says "H1–H12"**, omitting H13, which was added later to §2. The brief says H1–H13. This plan uses H1–H13.
3. **Flux's ComfyUI double block: 10 or 8 linears?** The brief §4 says the ComfyUI double block has 10 linears (against 14 for diffusers), but DESIGN §1.7 lists Chroma's ComfyUI double block with 8 named `attr_names` and says the Chroma/Flux delta is only the modulations. Those two statements are hard to reconcile; step 0.9 derives the truth empirically from `split_positions` rather than from either document.
4. **The diffusers concatenation order is unconfirmed**, as both documents note. It cannot be guessed, and step 1.8 depends on it, so step 0.9 is on the critical path to Phase 1 — worth starting its downloads first.
5. **~~No local GPU.~~** *Resolved.* The batched GPU session ran on a rented RTX A4000 for about $0.09 and confirmed H7, H6, H11 and the `--luts=correct` gate. Phase 5's GPU work will need another such session.
6. **The development machine has 3 GB of RAM and 39 GB of disk** — measured during review, and tighter than anything DESIGN.md assumes. It does not constrain df11pack itself (~1.35 × N per worker; tier-0 units need single-digit MB) but it does constrain running the *official* compressor to produce fixtures. Hence the tiered corpus: tier 0 is sized so the reference tool fits in ~1 GB here. The one open question is 0.2c — whether the LLM `pattern_dict` puts Qwen3-0.6B's 155.6M-weight embedding in a single unit, which would put tier 1 out of reach locally. Developing on the constrained machine is treated as a feature: it tests the guiding principle continuously rather than at the end.

7. **~~Large-RAM access is unknown.~~** *Resolved during review.* An earlier draft of this plan required ~64 GB for steps 0.4, 0.7, 0.9 and 0.12, on the assumption that real-model fixtures meant running the official compressor. They don't: the official team and the Extended maintainer both publish the compressed output on Hugging Face, and reading it is a streaming operation. Step 0.2b downloads those instead. The only residue is step 0.4's 12B endpoint for H1, which is optional and does not gate anything. It was a mistake to import the reference implementation's memory requirement into a plan whose entire purpose is to eliminate it.
