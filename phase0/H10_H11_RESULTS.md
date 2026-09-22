# Phase 0 — H10 and H11 results

Method, raw evidence, and verdicts for the two hypotheses. Scripts referenced
here (`phase0/rebuild_config.py`, `phase0/repack_shards.py`,
`phase0/verify_repack.py`, `phase0/verify_repack_merged_invariants.py`) are
all under `phase0/`; none of them import torch, transformers or diffusers.
All large repacked artefacts built while producing this file were deleted
immediately after verification (see the "H11 — evidence" section for the
build/test/delete sequence actually run).

---

## H10 — can the output `config.json` be rebuilt without diffusers/transformers?

### 1. Key-by-key diff, both fixtures

Diffed programmatically: SOURCE `../bf16-exponent-compression/real_model/config.json`
(Qwen3-0.6B, written by transformers 4.51.0) vs OUTPUT
`phase0/out/official/qwen3-0.6b-layers-only/config.json` (written by dfloat11
0.5.0 + transformers 5.17.0), and the same diff repeated on the 4-layer
truncated fixture (`phase0/corpus/tier0/qwen3-trunc/config.json` vs
`phase0/out/official/qwen3-trunc-layers-only-dir/config.json`). **Both diffs
are identical in kind** (same keys added/removed/renamed, only values differ
by model size / vocab size), which is why one table covers both.

| Key | Source | Output | Change | Origin |
|---|---|---|---|---|
| `torch_dtype` | `"bfloat16"` | — | removed | `save_pretrained` (transformers re-normalizes to `dtype`) |
| `dtype` | — | `"bfloat16"` | **added** | `save_pretrained` |
| `rope_theta` | `1000000` | — | removed | `save_pretrained` (folded into `rope_parameters`) |
| `rope_scaling` | `null` | — | removed | `save_pretrained` (folded into `rope_parameters`) |
| `rope_parameters` | — | `{"rope_theta": 1000000, "rope_type": "default"}` | **added** | `save_pretrained` |
| `layer_types` | — | `["full_attention"] * num_hidden_layers` | **added** | `save_pretrained` (Qwen3-specific derived field) |
| `pad_token_id` | — | `null` | **added** | `save_pretrained` (base `PretrainedConfig` always tracks it) |
| `transformers_version` | `"4.51.0"` | `"5.17.0"` | **changed** | `save_pretrained` — the *installed* library's version, unrelated to the source value |
| `dfloat11_config` | — | `{version, threads_per_block, bytes_per_thread, pattern_dict}` | **added** | dfloat11 injection (`dfloat11.py` line 615, `model.config.dfloat11_config = {...}` set *before* `save_pretrained`) |
| all other 21 keys | as source | unchanged | — | passthrough |

Every other key (`architectures`, `attention_bias`, `attention_dropout`,
`bos_token_id`, `eos_token_id`, `head_dim`, `hidden_act`, `hidden_size`,
`initializer_range`, `intermediate_size`, `max_position_embeddings`,
`max_window_layers`, `model_type`, `num_attention_heads`,
`num_hidden_layers`, `num_key_value_heads`, `rms_norm_eps`,
`sliding_window`, `tie_word_embeddings`, `use_cache`,
`use_sliding_window`, `vocab_size`) passes through byte-for-byte.

**Which came from `save_pretrained` vs from the dfloat11 injection.** Read
directly from `dfloat11.py` lines 606–620:

```python
if save_model:
    dfloat11_config = {
        'version': version,
        'threads_per_block': threads_per_block,
        'bytes_per_thread': bytes_per_thread,
        'pattern_dict': pattern_dict,
    }
    if hasattr(model, 'config'):
        try:
            model.config.dfloat11_config = dfloat11_config
        except Exception:
            pass

    if hasattr(model, 'save_pretrained') and not save_single_file:
        model.save_pretrained(save_path)
```

Only `dfloat11_config` itself is injected by dfloat11's own code — as an
ordinary extra attribute set on the live `PretrainedConfig` object. **Every
other change** (`torch_dtype`→`dtype`, `rope_theta`/`rope_scaling`→
`rope_parameters`, `layer_types`, `pad_token_id`, the `transformers_version`
bump) comes entirely from `model.save_pretrained()` — i.e. from
`transformers.PretrainedConfig.to_json_file()` re-serializing the **whole**
config object against the schema of whatever transformers version is
installed, not from anything DF11-specific. This means config.json
reconstruction is not "source + one injected key" as DESIGN.md §1.6's
optimistic framing suggested — it is "replicate one version of transformers'
config (de)serialization schema for this model's config class."

Two more observed facts, not in the diff table because there is no source
counterpart to diff against:
- Key order in the output is **plain alphabetical** (`json.dumps(...,
  indent=2, sort_keys=True)`) — confirmed against both fixtures' exact byte
  layout.
- `generation_config.json` has **no source counterpart at all** — it does
  not exist in either source directory. It is synthesized from scratch by
  transformers' `GenerationConfig` machinery
  (`{"_from_model_config": true, "bos_token_id", "eos_token_id",
  "transformers_version", "use_cache"}`), which is even more clearly
  library-version-dependent than `config.json` and is out of this task's
  scope but worth flagging for Phase 1/2.

### 2. `phase0/rebuild_config.py` — verification

Two modes, both stdlib-only (`json`, `argparse`, `copy` — no torch,
transformers or diffusers):

- **`--mode full`**: replicates all five `save_pretrained` transformations
  above, given an explicit `--transformers-version-target` string (since that
  value cannot be computed, only declared — see §3).
- **`--mode minimal`**: keeps the source schema verbatim and only adds
  `dfloat11_config` (the literal fallback dfloat11.py itself uses when the
  model has no `save_pretrained`, `dfloat11.py` lines 632–636:
  `json.dump({'dfloat11_config': dfloat11_config}, config_file, indent=2)`).

Measured result, both fixtures, `--mode full --transformers-version-target 5.17.0`:

```
$ python phase0/rebuild_config.py ../bf16-exponent-compression/real_model/config.json \
    --pattern-dict-json pattern_dict.json --mode full \
    --transformers-version-target 5.17.0 -o /tmp/rebuilt_full.json
$ diff /tmp/rebuilt_full.json phase0/out/official/qwen3-0.6b-layers-only/config.json
(no output)  ->  BYTE IDENTICAL

$ python phase0/rebuild_config.py phase0/corpus/tier0/qwen3-trunc/config.json \
    --pattern-dict-json pattern_dict.json --mode full \
    --transformers-version-target 5.17.0 -o /tmp/rebuilt_trunc.json
$ diff /tmp/rebuilt_trunc.json phase0/out/official/qwen3-trunc-layers-only-dir/config.json
(no output)  ->  BYTE IDENTICAL
```

**Both fixtures reproduce the official `config.json` byte-for-byte**, given
the correct `transformers_version_target` and given that the one
architecture-specific derivation exercised (`layer_types` for Qwen3) matches
what transformers 5.17.0 does. `--mode minimal` reproduces neither
byte-for-byte (diff shown in the script's own test run: missing `dtype`,
`layer_types`, `pad_token_id`, `rope_parameters`; keeps `rope_scaling`/
`rope_theta` instead) — that gap is exactly the five-row diff table above,
by construction.

**What it cannot reproduce, and why**, in `--mode full`:
- `transformers_version`: cannot be *derived*; it must be supplied as an
  explicit target string (see §3). Given the right string, the rest of the
  reconstruction is correct on both fixtures tested.
- `layer_types` for a Qwen3 checkpoint that actually uses sliding-window
  attention (`sliding_window` not null and `use_sliding_window: true`): the
  script explicitly refuses (`raise NotImplementedError`) rather than guess,
  because no fixture on this machine exercises that path — see "what remains
  unproven".
- `rope_parameters` for a non-null `rope_scaling` (e.g. `rope_type: "yarn"`
  or `"linear"`): implemented as a best-effort merge, untested — same reason.

### 3. Library-version-dependent fields, and how Rust should handle each

| Field | Version-dependent? | Recommendation |
|---|---|---|
| `transformers_version` | **Yes, unconditionally.** It is `str(transformers.__version__)` of whatever is installed at compress time — nothing else. A Rust binary with no transformers install has no source for this value at all. | Do not try to derive it. Make it a declared, versioned constant the operator picks per run (`--transformers-version-target`, or a `df11pack.toml` entry), exactly as `pattern_dict` is already a declared per-model choice (FINDINGS.md 0.2c). Ship a small hand-maintained table mapping `transformers_version -> {schema transformations that version applies}` (initially just the 5 rows above; grows as more architectures/versions are added to the golden-test corpus). |
| `dtype`/`rope_parameters`/`layer_types`/`pad_token_id` presence | **Schema-version-dependent, not value-dependent.** *Whether* these keys/renames exist is a function of the transformers schema version being targeted, but once that target is fixed, the *values* are computable from the source config's own fields (`torch_dtype`, `rope_theta`, `rope_scaling`, `num_hidden_layers`, `sliding_window`, `use_sliding_window`). | Implement each transformation as an explicit, named, testable function keyed off the declared target schema version (this is what `rebuild_full()` already does for the one version pair measured here). Extend the version table in lockstep with new golden fixtures — never infer a new version's schema from guessing. |
| `dfloat11_config.version` | **Not library-dependent.** Confirmed by reading `dfloat11.py` line 44: `version = '0.5.0'` is a hardcoded module-level constant in the dfloat11 *package itself*, not read from any installed library's `__version__`. | Treat as a df11pack-format-version constant df11pack owns and picks explicitly per target release, matching FINDINGS.md 0.2's existing note ("df11pack must emit the right version per target and tolerate both on read"). No Rust-specific risk here at all. |
| `generation_config.json` (`_from_model_config`, `transformers_version`) | **Yes**, same as `config.json`'s `transformers_version`, and additionally it has no source file to diff against at all — it is manufactured, not transformed. | Same declared-target-version approach; lower priority since it carries almost no information (3 real fields) and models can run without it (transformers falls back to config-derived generation defaults if it's absent). |

**Bottom line for H10 (LLM/ComfyUI-adjacent, single-file-config layout):**
`config.json` **is reconstructable without transformers/diffusers installed**,
stdlib-only, **provided** the target transformers-version schema is known and
declared (not derived). Given that declaration, reconstruction was measured
byte-identical on both available fixtures. The one field that is
fundamentally unknowable from the source alone is `transformers_version`
itself — everything else is a deterministic, portable transformation once
that target is pinned.

**Untested, and stated as a hypothesis, not a fact**: transformers'
`AutoConfig.from_pretrained` is widely documented/observed to accept legacy
kwarg names (`rope_theta`, `rope_scaling`, `torch_dtype`) even on versions
that themselves *write* the new schema, for backward compatibility. If true,
a `--mode minimal`-style config.json (source schema + `dfloat11_config`
only, no transformers-version emulation at all) would still **load**
correctly on the end user's transformers even though it is not
byte-identical to what `compress_model` would have written — turning this
from a hard blocker into a "nice to have byte-identity, not load-bearing"
concern. This machine has no transformers install to confirm or refute that
claim; it is recorded as the load-bearing open question for whoever validates
Phase 2 with a real transformers environment.

### 4. Diffusers layout — a surprise that revises DESIGN.md §1.6's assumption

DESIGN.md §1.6 assumed: *"`config.json` must come out the same as
`save_pretrained`'s: the transformer's original configuration plus the fields
diffusers adds (`_class_name`, `_diffusers_version`, etc.) and
`dfloat11_config`."* Fetched two **published** diffusers DF11 `config.json`
files (small files only, per the environment limits — no model weights
downloaded):

**`DFloat11/Chroma-DF11/config.json`** (1053 bytes):
```json
{
  "dfloat11_config": {
    "bytes_per_thread": 8,
    "pattern_dict": { "distilled_guidance_layer": [...], "transformer_blocks\\.\\d+": [...], "single_transformer_blocks\\.\\d+": [...] },
    "threads_per_block": [512],
    "version": "0.2.0"
  },
  "model_type": "llama"
}
```

**`DFloat11/FLUX.1-dev-DF11/config.json`** (839 bytes, irregular 4-then-2-space
indentation — visibly hand-edited, not machine-generated with a consistent
indent level):
```json
{
    "dfloat11_config": {
      "bytes_per_thread": 8,
      "pattern_dict": { "transformer_blocks.\\d+": [...], "single_transformer_blocks.\\d+": [...] },
      "threads_per_block": [512],
      "version": "0.2.0"
    },
    "model_type": "llama"
}
```

**Neither file has `_class_name`, `_diffusers_version`, `in_channels`, or any
other diffusers `ModelMixin` config field.** Both consist of exactly
`dfloat11_config` plus one extra, semantically odd key (`"model_type":
"llama"` on a diffusion transformer). Confirmed via the HF API's file listing
for `Chroat-DF11` that there is no `transformer/config.json` subfolder either
— this two-key file **is** the repo's only config.json.

This matches the *code path* `dfloat11.py` actually has for exactly this
scenario — the fallback at lines 632–636 (quoted in §2 above), triggered
whenever the model has no `save_pretrained` **or** the caller already
supplies `bfloat16_model=` and the loader only requires the on-disk
`config.json` to contain `dfloat11_config` (`dfloat11.py` lines 393–406: when
`bfloat16_model` is given and not `from_single_file`, the loader does
`config = json.load(...)` as a **plain dict** and never runs it through
`AutoConfig`/diffusers at all; it only requires `'dfloat11_config' in
config`, checked at line 429). `model_type: "llama"` is not explained by
anything in the dfloat11 source read here — likely added by hand by the
repo maintainers, or by a different code path/version not exercised in this
environment — recorded as an open question, not resolved.

**Revised verdict for the diffusers case:** the *full* diffusers schema
(`_class_name`, `_diffusers_version`, etc.) is something `compress_model`'s
code **can** produce (via `model.save_pretrained()`, when the model object is
a real diffusers `ModelMixin` and `save_model=True`), but the **published,
real-world DF11 diffusers releases inspected here do not actually ship
it** — they ship the two-key minimal form. For df11pack's own diffusers
output, DESIGN.md §1.6's fuller reconstruction is still the correct target
*if* byte-identity to a `save_pretrained()`-style output is the goal, but the
minimal `{"dfloat11_config": ...}` form is demonstrably what the ecosystem
actually consumes today (the loader's own `bfloat16_model=` path requires
nothing more), and it is trivially reproducible with the stdlib
(`rebuild_config.py --mode minimal` already produces exactly this shape for
the LLM case; the same function works unchanged for a diffusers source
config, since it makes no LLM-specific assumptions).

---

## H11 — does the loader depend on how tensors are split across shards?

### 1–2. Repack, parse, invariants, byte-identity (measured)

Built from `phase0/out/official/qwen3-0.6b-layers-only` (28 layer shards +
`model.safetensors`, 282 tensors total, 870 MB), one variant at a time, each
deleted before the next was built (peak disk usage stayed at the ~870 MB of
whichever single variant was live, well inside the 37 GB free / 10 GB floor):

**Variant (a) — single merged file**, all 282 tensors in one
`model_merged.safetensors` (`phase0/repack_shards.py --mode single`):

| Check | Result |
|---|---|
| Build peak RSS (`/usr/bin/time -v`) | **20.9 MiB** |
| `check_invariants.py` | **28/28 units passed** |
| `verify_repack.py` (byte-identity per tensor vs. original) | **282/282 tensors compared, 0 mismatches, 0 missing, 0 extra — PASS** |

**Variant (b) — 4-file arbitrary interleaving**, file names
`blob_00_of_4.bin.safetensors` … `blob_03_of_4.bin.safetensors` (match no
unit name and no tensor name), tensors assigned round-robin over the
alphabetically-sorted key list so that a single unit's tensors land in
**multiple** files (`phase0/repack_shards.py --mode interleaved --n-files 4`):

| Check | Result |
|---|---|
| Build peak RSS | **20.8 MiB** |
| Confirmed scattering (example, `model.layers.5`) | its 6 DF11 suffix tensors landed in `blob_03` (`encoded_exponent`, `output_positions`, `sign_mantissa`), `blob_00` (`gaps`, `split_positions`), `blob_02` (`luts`) — 3 different files for one unit |
| `check_invariants.py` run **per file**, unmodified | **0/84 "units" — FAIL**, `INV-NAMES-COMPLETE` ("missing luts, gaps, split_positions") on every unit whose tensors span files |
| `check_invariants.py` run **per logical unit**, tensors reassembled by name across the whole directory first (`phase0/verify_repack_merged_invariants.py`) | **28/28 logical units passed** |
| `verify_repack.py` (byte-identity per tensor vs. original) | **282/282 tensors compared, 0 mismatches, 0 missing, 0 extra — PASS** |

**A genuine finding, not a bug in the repack**: `check_invariants.py` groups
tensors into units *per file* (its `group_units()` is called once inside
`check_file()`, on one file's header at a time). That is correct for every
official fixture, where a unit's tensors always live together in one shard,
but it is the wrong granularity for testing H11's premise, because it
silently assumes the very thing H11 is asking whether the *loader* assumes.
`phase0/verify_repack_merged_invariants.py` does not modify or reimplement
`check_invariants.py`'s invariant logic; it reassembles each logical unit
from wherever its tensors physically live (exactly what the real loader
does, see below) into a small temporary single-file safetensors, then runs
the **unmodified** `check_invariants.py` on that — reuse, not rewrite. All 28
units pass this way, matching the byte-identity result.

### 3. Loader code — quoted, with the dispatch-by-name conclusion

Source: `phase0/env/lib/python3.12/site-packages/dfloat11/dfloat11.py`,
`load_and_replace_tensors()`, called from `DFloat11Model.from_pretrained()`.

**File names are read, then never inspected again — only used to open the file:**

```python
# Get all .safetensors files in the directory
safetensors_files = [
    f for f in os.listdir(directory_path) if f.endswith('.safetensors')
] if not from_single_file else [directory_path]
...
for file_name in tqdm(safetensors_files, desc=loading_desc):
    file_path = os.path.join(directory_path, file_name) if not from_single_file else file_name

    # Load the tensors from the file
    loaded_tensors = load_file(file_path)

    # Iterate over each tensor in the file
    for tensor_name, tensor_value in loaded_tensors.items():
```
(`dfloat11.py` lines 204–222)

The only use of `file_name`/`file_path` is `os.listdir` (to discover the set
of files) and `load_file(file_path)` (to open one). There is no
`if file_name == ...`, no regex on the filename, no assumption that a
particular unit's tensors live in a particular file — the loop is a flat
`for file_name in <all files> / for tensor_name in <that file's tensors>`
with **no cross-tensor or cross-file state carried between iterations**
except the model object being mutated.

**Dispatch is entirely by `tensor_name`, walking the model's own module
tree**, independent of which file it came from:

```python
else:
    # Split the tensor name to get module path
    parts = tensor_name.split('.')
    module = model

    # Navigate to the correct module
    for i, part in enumerate(parts[:-1]):
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            print(f"Cannot find module path for {tensor_name}", file=stderr)
            break
    else:
        if parts[-1] == 'split_positions':
            setattr(module, 'split_positions', tensor_value.tolist())
        else:
            ...
            module.register_buffer(parts[-1], tensor_value)

        # Set up decompression for encoded weights
        if parts[-1] == 'encoded_exponent':
            # Register the decode hook to decompress weights during forward pass
            module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))
            ...
```
(`dfloat11.py` lines 240–271, elided at `...` only inside the
`cpu_offload` branch and the pattern_dict device-injection branch, neither of
which touches file identity)

**Conclusion from the code, not inference**: the loader's only two uses of
"which file" are (1) enumerate every `.safetensors` file in the directory
with no name filter beyond the extension, and (2) open each one and iterate
its tensors. Every subsequent decision — which module to attach a tensor to,
whether it's `split_positions` (special-cased as a Python list, not a
buffer), whether it's `encoded_exponent` (triggers registering the
decompression hook on that module) — keys **exclusively** on `tensor_name`,
split on `.` and walked against the live model's attribute tree via
`getattr`. Nothing in this function, or in `from_pretrained` around it
(`dfloat11.py` lines 343–465, which only reads `config.json` for
`dfloat11_config` and constructs/dispatches the model), reads or compares a
file name against a unit name or a `pattern_dict` key. **H11's premise is
confirmed from the code**: file names and file grouping are structurally
irrelevant to the loader; only tensor names matter, and they may arrive from
any file in any order.

### 4. What remains unproven

This machine has no GPU and does not run PyTorch (both prohibited by the
environment), so nothing here constitutes an end-to-end confirmation.
Stated precisely:

**Proven, statically and structurally, on real data:**
- The loader's source code has no file-name-dependent branch (quoted above,
  read line-by-line, not paraphrased).
- Two independently-built repackings (merge-to-one-file, and
  scatter-one-unit-across-four-files-with-non-matching-names) preserve every
  tensor's name, dtype, shape and raw bytes exactly (`verify_repack.py`,
  282/282 both times).
- Both repackings parse as valid safetensors and, once grouped by tensor
  name the way the loader groups them (not by physical file, which
  `check_invariants.py` currently assumes), satisfy every structural DF11
  invariant in `docs/DESIGN.md` §1.3 that `check_invariants.py` checks
  (28/28 logical units both times).

**Not proven, and would require a GPU this machine does not have:**
- That `DFloat11Model.from_pretrained` actually **runs** to completion
  against either repacked variant without raising elsewhere (e.g. in
  `get_no_split_classes`/`infer_auto_device_map`, which were not exercised
  here — only `load_and_replace_tensors`'s logic was read and reasoned
  about, not executed against a live `nn.Module`).
- That the decompression kernel, once loaded from either repacked variant,
  produces **numerically identical inference output** to the original
  28-shard directory (DESIGN.md's own stated criterion for H11:
  "load variants and compare inference"). Byte-identical input tensors make
  this extremely likely given the loader-code evidence above, but "likely"
  is not "measured" — the DF11 kernel's correctness depends on the bytes
  reaching the GPU unchanged, which was confirmed, not on anything about
  in-GPU behavior, which was not exercised at all.
- Real-world diffusers loading via `from_pretrained(..., bfloat16_model=)`
  was not exercised end-to-end either; only `load_and_replace_tensors`
  itself was read, and it is layout-agnostic between the LLM/ComfyUI and
  diffusers call paths (same function, same logic, called either way per
  `dfloat11.py` lines 436–441).

**Verdict:** H11 is **confirmed by source code and by structural/byte-level
testing** to the extent achievable without a GPU. The remaining gap is a
real GPU load-and-infer comparison, explicitly out of reach on this
3 GB/no-GPU machine, and should be the first thing scheduled once a rented
GPU session is available (DESIGN.md §1.5 already recommends batching GPU
work into a single rented session rather than spreading it across steps —
this is a candidate for that batch, alongside H7).

---

## Files produced

- `phase0/rebuild_config.py` — stdlib-only config.json reconstructor, `full` and `minimal` modes.
- `phase0/repack_shards.py` — stdlib-only safetensors repacker (single-file and N-way interleaved).
- `phase0/verify_repack.py` — per-tensor byte-identity verifier across directories/files.
- `phase0/verify_repack_merged_invariants.py` — cross-file logical-unit reassembly + unmodified `check_invariants.py` reuse.
- `phase0/H10_H11_RESULTS.md` — this file.

All repacked `.safetensors` variants built during this work (single-file
~870 MB, interleaved 4-file ~870 MB total) were deleted immediately after
verification; none remain on disk. No files under `docs/` were edited.
