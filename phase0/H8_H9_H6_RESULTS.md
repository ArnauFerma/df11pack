# H8 / H9 / H6 — Results

Machine: 3 GB RAM, no GPU, no torch imported anywhere in this session. All
comparisons parse safetensors headers directly (8-byte LE length + JSON) and
read tensor payloads by `seek`/`read` in 4 MiB chunks, hashed with
`hashlib.sha256` streaming. Peak RSS measured with `/usr/bin/time -v`.

No model was downloaded: both source models already existed on disk
(`phase0/corpus/tier0/qwen3-trunc/model.safetensors`, 136.0 MiB; and
`../bf16-exponent-compression/real_model/model.safetensors`, 1.5 GB). `df -h
/home` showed 37 GB free before starting and after finishing; nothing was
fetched or deleted.

Scripts used (kept in the session scratchpad, not under `phase0/`, since they
are throwaway probes rather than reusable tooling — the reusable artifact is
this file plus the one reordered fixture under `phase0/out/h6/`):
- header/byte comparator: standalone script, logic reproduced inline below
- reorder probe: standalone script, logic reproduced inline below

---

## H8 / H9 — method

For each of the two local official outputs:

1. Parsed the source safetensors header and the headers of **every**
   `.safetensors` file in the output directory (the remainder file
   `model.safetensors` **and** every `model_layers_N.safetensors` shard —
   this matters, see catalogue item 3 below).
2. Classified every output tensor key as a **DF11 unit tensor** (suffix in
   `{luts, encoded_exponent, sign_mantissa, output_positions, gaps,
   split_positions}`) or **non-unit**, using the same grouping rule as
   `phase0/check_invariants.py::group_units`.
3. For every non-unit output tensor, looked up the same name in the source
   header and compared dtype, shape, and a streaming SHA-256 of the raw
   bytes (via `data_offsets` range reads, never loading a whole tensor into
   a numpy/torch object).
4. Recorded every name present in source but absent from the non-unit output
   set, and every name present in the non-unit output set but absent from
   source.
5. Read `phase0/env/lib/python3.12/site-packages/dfloat11/dfloat11.py`,
   `compress_model` (lines 496–636), to explain every difference found.

Peak RSS of the comparison script (both models, sequentially, in one
process): **23,848 KiB (23.3 MiB)** — well under the 300 MB budget.

## H8 / H9 — measured numbers

| Model | Source tensors | Output non-unit tensors (all files) | Missing from output | Extra in output | dtype mismatch | shape mismatch | byte mismatch | byte-identical |
|---|---|---|---|---|---|---|---|---|
| `qwen3-trunc` (4 layers) | 47 | 18 | 29 | 0 | 0 | 0 | 0 | **18 / 18** |
| `qwen3-0.6b` (28 layers, full) | 311 | 114 | 197 | 0 | 0 | 0 | 0 | **114 / 114** |
| **Total** | 358 | 132 | 226 | 0 | 0 | 0 | 0 | **132 / 132** |

Every non-unit output tensor that has a same-named source tensor is
byte-identical to it: dtype `BF16` in both, shape unchanged, SHA-256 equal.
**Zero** exceptions across 132 tensors in two independently-sized models (4
and 28 layers). **Zero** tensors appear in either output that don't exist
in the corresponding source.

## H8 / H9 — the full difference catalogue

| # | Item | Source | Output | Verdict | Why (from `compress_model`, dfloat11.py) |
|---|---|---|---|---|---|
| 1 | `lm_head.weight` | present, BF16, shape `[4096,1024]` (trunc) / `[151936,1024]` (full) | **absent** entirely (not in remainder, not in any shard) | expected, documented | `config.json` has `tie_word_embeddings: true`. Confirmed by hash: `lm_head.weight` and `model.embed_tokens.weight` are byte-identical in *both* source files (SHA-256 match, first 12 hex: `b669daa93f1d…` trunc, `8f29acf51943…` full). Directory mode (`save_single_file=False`) reaches L619–620, `model.save_pretrained(save_path)` — transformers' own saver drops the tied duplicate before calling `safetensors.save_file` (the same sharing that made *single-file* mode's direct `save_file(model.state_dict())` at L622 crash with `RuntimeError: Some tensors share memory`, per `docs/FINDINGS.md` 0.2). |
| 2 | 7 linear weights/layer: `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight` — 28 tensors (trunc, 4 layers × 7), 196 tensors (full, 28 layers × 7) | present, BF16 | **absent as plain tensors**; replaced by the 6 DF11 unit tensors under the same `model.layers.N` prefix | expected — this **is** the compression, not an H8/H9 violation | L541–550: for each `attr_path` in `attr_names`, `weights.append(target.weight.data.detach().cpu().flatten())` then `delattr(target, 'weight')`. L593–598: `sub_module.register_buffer(...)` installs `luts/encoded_exponent/sign_mantissa/output_positions/gaps/split_positions` instead. Excluded from the H8/H9 non-unit scope by definition. |
| 3 | 4 norm tensors/layer: `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.q_norm.weight`, `self_attn.k_norm.weight` — 16 tensors (trunc), 112 tensors (full) | present in the single monolithic source file | present, **byte-identical**, but **physically relocated** into `model_layers_N.safetensors` instead of staying in the remainder `model.safetensors` | content unchanged; **location** changed — worth flagging so a naive H8/H9 check that only reads the remainder file misses 114/132 of the non-unit tensors | L513–519: `if not save_single_file: setattr(parent, child_name, None)` detaches the **whole decoder-layer submodule** from the model *before* `save_pretrained()` runs. L600–604: `state_dict = sub_module.state_dict()` captures **everything** in that submodule — not just the attributes named in `attr_names` — and writes it to `<name>.safetensors`. The norms live in the same submodule but match no compression pattern, so they ride along into the shard untouched rather than remaining in the remainder file. |
| 4 | `model.embed_tokens.weight`, `model.norm.weight` | present, BF16 | present in remainder `model.safetensors`, byte-identical | unchanged, in place | Not matched by the layers-only `pattern_dict` (`model.layers.\d+` → the 7 linears), never touched by `compress_model`'s loop, written as-is by `model.save_pretrained()`. |
| 5 | Every other non-unit tensor (132 total, see table above) | present | present, byte-identical, same file role as source | unchanged | Not reached by the compression loop at all. |

No dtype casts, no shape changes, no renames, and no invented tensors were
found anywhere in either output.

Additional corroboration for item 1 (from `phase0/out/*/config.json`): both
outputs carry `"tie_word_embeddings": true`, matching both sources.

## H8 / H9 — verdict

**H8 (LLM path): CONFIRMED**, empirically, with the caveats above being
*explained* differences, not violations. Every non-unit tensor in both local
LLM outputs (132/132 across two model sizes) is byte-identical to its
same-named tensor in the source safetensors file: same dtype, same shape,
identical SHA-256. The only tensor that disappears without becoming a DF11
unit (`lm_head.weight`) does so because it is a weight-tying duplicate of a
tensor that *is* preserved byte-identically — not because instantiation
altered anything. Non-unit tensors are passed through **untouched**, i.e.
**yes**, on this evidence. The one implementation-relevant catch (item 3) is
that "untouched" does not mean "stays in the remainder file" — a compressed
submodule's *other* parameters travel with it into its own shard. Anyone
implementing df11pack's grouping logic must replicate that placement, not
just the placement of the DF11-unit tensors themselves.

**H9 (diffusers path): NOT independently tested in this session** — no
diffusers-layout local output exists to compare (per the task's provided
assets, only the two Qwen3/transformers outputs were available, and no
multi-GB diffusion model was downloaded to build one, per the environment
budget). What supports H9 by analysis rather than fresh measurement:
`compress_model` (dfloat11.py L496–636) is a **single function** shared
verbatim between the LLM and diffusers layouts — the only layout-dependent
step is which `save_pretrained` ends up being called on the remainder model
(transformers' vs diffusers'), and the detach-then-shard mechanism (L513–519,
600–604) that item 3 above depends on is layout-agnostic. `docs/DESIGN.md`
§1.6 already records this flow as `[CONFIRMED]` from source reading. This
session adds byte-level, cross-model empirical confirmation of the
LLM/transformers half of that shared code path; it does **not** add a fresh
diffusers-specific measurement. Treat H9 as *supported* by H8's evidence plus
the shared-code argument, but explicitly **not settled by measurement here**.

---

## H6 — method

Took one real official shard, `phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_0.safetensors`
(21,383,221 bytes; 10 tensors: the 6 DF11 unit tensors of `model.layers.0`
plus the 4 non-unit norm tensors from catalogue item 3 above), and rewrote it
with:

- the **same** tensor names, dtypes, shapes, and bytes,
- a **different physical order** in the data section (`data_offsets`),
- and a different header key-insertion order to match (safetensors permits
  header key order and physical data order to differ; both were changed here
  to stress the claim harder).

New physical order = exact reverse of the original physical order (ordering
by each tensor's original `data_offsets[0]`):

```
original: split_positions, input_layernorm.weight, post_attention_layernorm.weight,
          self_attn.k_norm.weight, self_attn.q_norm.weight, encoded_exponent,
          gaps, luts, output_positions, sign_mantissa
reordered: sign_mantissa, output_positions, luts, gaps, encoded_exponent,
           self_attn.q_norm.weight, self_attn.k_norm.weight,
           post_attention_layernorm.weight, input_layernorm.weight, split_positions
```

Output: `phase0/out/h6/model_layers_0.reordered.safetensors` (21,383,346
bytes — 125 bytes larger, entirely explained by JSON header text length
changing with the reordered/renumbered `data_offsets` integers; confirmed
below to have zero effect on any tensor payload).

Verification, in three independent ways:

1. **My own header parser** (same one used for H8/H9): re-parsed the
   reordered file's header, confirmed the key set is unchanged (10/10), and
   confirmed the on-disk physical order matches the intended new order
   exactly. Recomputed SHA-256 for each of the 10 tensors by name from the
   reordered file and compared against the pre-reorder hash of the same
   name: **10/10 byte-identical**.
2. **`phase0/check_invariants.py`**, run on both the original and the
   reordered file in the same invocation: **2/2 units passed** (`[PASS]` for
   both `model.layers.0` units — original and reordered are structurally
   indistinguishable to the checker).
3. **The real `safetensors` library's own reader** (`safetensors.safe_open`,
   `framework="numpy"`, no torch — this is the same Rust-backed offset parser
   `safetensors.torch.load_file` uses internally, minus the torch tensor
   conversion). Opened both files, confirmed identical key sets, and
   compared the 6 tensors whose dtype numpy understands directly (`U8`/`I64`
   — the BF16 norm tensors raise `TypeError: data type 'bfloat16' not
   understood` in plain numpy and were already covered by SHA-256 in step 1,
   which is bytewise and doesn't depend on numpy understanding the dtype):
   **6/6 identical** (dtype, shape, and `np.array_equal` content). Peak RSS
   for this step: 89,748 KiB (87.6 MiB).

## H6 — verdict

**Structurally CONFIRMED, inference-CONFIRMED status remains open.**
A real DF11 shard, physically reordered (different `data_offsets` sequence
*and* different header key order, same names/dtypes/shapes/bytes), still:
parses correctly under two independent parsers (a from-scratch one and the
official `safetensors` Rust library); passes every structural invariant in
`check_invariants.py`; and every one of its 10 tensors round-trips
byte-identical, verified three ways. The format itself is provably
order-independent at the file level — nothing in the safetensors container
requires or assumes a particular physical layout, since every tensor's
location is stored explicitly in its own `data_offsets` entry.

Reading `load_and_replace_tensors` (dfloat11.py, ~L195–275) supports the same
conclusion for the *loader's* logic, independent of running it: it calls
`safetensors.torch.load_file(file_path)`, which returns a **name-keyed dict**
(not a list), then does
`for tensor_name, tensor_value in loaded_tensors.items(): ... parts =
tensor_name.split('.'); module = model; for part in parts[:-1]: module =
getattr(module, part) ...; module.register_buffer(parts[-1], tensor_value)`
— every dispatch decision is driven by the **tensor_name string**, matched
against `pattern_dict` with `re.fullmatch`, never by iteration position, list
index, or which file a tensor came from. The outer loop over files
(`for file_name in tqdm(safetensors_files, ...)`, built from
`os.listdir(directory_path)`) has no ordering guarantee either, and nothing
downstream depends on it: whichever shard a tensor is read from, it lands at
the same module path by name.

## What remains unproven for H6

Not attempted in this session, and explicitly **not** claimed:

- **Actually loading the reordered file through `DFloat11Model.from_pretrained`
  / `load_and_replace_tensors`.** This requires `import torch` to build the
  real `nn.Module` these functions mutate — forbidden in this environment
  (torch import alone costs ~328 MiB RSS per `docs/FINDINGS.md` 0.2, and the
  task's hard constraint is "DO NOT import torch"). So the loader's
  name-driven dispatch is confirmed by reading the source, not by executing
  it against the reordered fixture.
- **Running the CUDA `decode.ptx` kernel and comparing inference output**
  (the refutation criterion H6 names explicitly: "load a reordered file and
  compare inference output"). This needs a real GPU and real `cupy`; this
  machine has neither, and `phase0/shim/cupy.py` deliberately raises on any
  real kernel call (`_Function.__call__` → `_GpuReached`) specifically so
  this can't be silently skipped. Even the "load buffers but don't run
  inference" middle ground is theoretically reachable with the cupy stub
  (buffer registration doesn't call `_decode`, only the forward-pre-hook
  does), but it still requires `import torch`, so it wasn't attempted either.
- **A reordered multi-shard *directory*, not just one shard.** This session
  reordered tensors *within* one file. It did not test scrambling which
  tensors live in *which* shard (e.g. moving `model.layers.0.gaps` into
  `model_layers_1.safetensors`) — though the source-reading argument above
  (`os.listdir` order + name-keyed dispatch) implies this should also be
  order-independent across files, that combination was not built and tested
  as a fixture here.

**What would prove it:** on a machine with a GPU and real `cupy`, run the
official `DFloat11Model.from_pretrained(bfloat16_model=<instantiated
Qwen3-trunc>, ...)` twice — once against the untouched
`qwen3-trunc-layers-only-dir`, once against a copy with every shard's tensors
reordered as done here (and, for full rigor, tensors also redistributed
across shard files) — and diff the two runs' logits for the same input. Exact
equality (not just closeness) would settle H6 completely; the source-reading
argument above predicts exact equality but does not establish it.

---

## Files produced under `phase0/`

- `phase0/H8_H9_H6_RESULTS.md` — this file.
- `phase0/out/h6/model_layers_0.reordered.safetensors` — the reordered H6
  fixture (21,383,346 bytes), kept as evidence; derived entirely from the
  already-local `qwen3-trunc-layers-only-dir` output, no download involved.

No files were created outside `phase0/`. No `git add`/`git commit` was run.
`df -h /home` before and after: 37 GB free, unchanged (no downloads).
