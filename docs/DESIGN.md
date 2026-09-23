# df11pack — Design document (v0.2, draft)

An independent DFloat11 compressor: streaming, with crash recovery.
Guiding principle: **anyone must be able to compress models**, including on old
machines, with little RAM, without an NVIDIA GPU, or on a spinning hard drive.

Scope for v1: **both layouts from the start**, ComfyUI-native (single file) and
diffusers (directory with shards + `config.json`).

Changes from v0.1: added §1.5 (compression/decompression asymmetry), §1.6
(diffusers layout), the kernel's 32-bit limits, H9–H12, a parallel CPU decoder
mirroring the kernel, and reordered phases.

Status of this document: no code. Every claim is marked **[CONFIRMED]** (read in
the source), **[HYPOTHESIS]** (to be measured in Phase 0) or **[DECISION]**
(open, to be closed before implementing).

---

## 0. Glossary (so we don't confuse "blocks")

DF11 uses the word *block* for three different things. They are separated here:

| Term | What it is | Typical size |
|---|---|---|
| **UC** (compression unit) | A module matching the `pattern_dict` (e.g. `double_blocks.7`). It has its own Huffman codebook. | 141M–340M weights in Flux |
| **Kernel chunk** | A group of 512 CUDA threads × 8 bytes = 4096 bytes of bitstream. `output_positions` stores one index per chunk. | ~12k weights |
| **Thread window** | 8 bytes (64 bits) of bitstream. `gaps` stores 5 bits per window. | ~24 weights |
| **Checkpoint** | A resume point persisted to disk. | 1 UC (see §6) |

---

## 1. What the official compressor does today

### 1.1 Actual flow [CONFIRMED]

Source: `LeanModels/DFloat11`, `dfloat11/dfloat11.py::compress_model` and
`dfloat11/dfloat11_utils.py`. The node in `mingyi456/ComfyUI-DFloat11-Extended`
(`dfloat11_model_loader.py::DFloat11ModelCompressor`) **has no compressor of its
own**: it instantiates the full model with `comfy.sd.load_diffusion_model` and
calls the official `compress_model` with `save_single_file=True`,
`check_correctness=True`.

For each UC, in order and on a single thread:

1. Takes the `.weight` tensors of the layers listed in the `pattern_dict`, flattens them and does a `torch.cat` (a full copy in RAM).
2. `get_codec`: exponent histogram via `torch.unique`.
3. `get_32bit_codec`: builds the Huffman code with the `dahuffman` library. If any code exceeds 32 bits, it forces the least frequent symbols to frequency 1 and rebuilds until it fits.
4. `get_luts`: generates the hierarchical byte-wise LUT (a 256-entry table per prefix, plus a final row with lengths).
5. `encode_weights` → `encode`: **converts the exponents to a Python list (`.tolist()`) and encodes them with a Python loop, one symbol at a time**, accumulating bytes into another Python list. At the same time it generates `gaps` (as 5-character strings, then a list of bits, then `packbits`) and `output_positions`.
6. If `check_correctness`: uploads everything to the GPU, runs the real `decode` kernel and compares against the original weights (another copy on the CPU).
7. Registers the resulting buffers on the module itself and deletes `.weight`.
8. At the end: `save_file(model.state_dict(), 'model.safetensors')` for the whole model at once.

### 1.2 Consequences

- **UCs are completely independent** [CONFIRMED]: each one has its own codebook, LUTs and indices. There is no global state between UCs other than the configuration (`version`, `threads_per_block=(512,)`, `bytes_per_thread=8`, `pattern_dict`). Parallelising across UCs has no conceptual barrier whatsoever.
- **The slowness does not come from "being single-threaded", but from the interpreted loop.** [HYPOTHESIS, consistent with the figures] Flux compresses ~11.8B weights (19 double blocks × ~340M + 38 single blocks × ~141M). At 300–600 ns per Python iteration that comes to 1–2 hours, exactly what the README describes. Parallelising that loop with multiprocessing would give ×cores; removing it with native code should give ×100 or more per core. This corrects my earlier explanation (the GIL is not the main cause).
- **The RAM comes from having the whole model instantiated plus huge per-UC temporaries.** [HYPOTHESIS] Estimate for a large UC (~340M weights): `cat` 0.7 GB, int16/uint8 temporaries ~1.3 GB, `.tolist()` ~2.7 GB (8 B per pointer), the `encoded` list ~0.9 GB, gaps as strings and bit lists ~1.3 GB. That is some 6–7 GB transient on top of the model's ~24 GB, plus whatever ComfyUI's loading costs (state_dict and model possibly coexisting) and the final serialisation. Reaching ~48 GB is plausible.

### 1.3 Format invariants we must respect [CONFIRMED]

Read from the encoder and from the kernel (`decode.cu`):

- `sign_mantissa[i] = (sign bit) | (7 mantissa bits)`, 1 byte per weight.
- Exponent = bits 7..14 of the int16 pattern. Bitstream is MSB-first within each byte.
- After the last symbol, `dahuffman`'s **EOF** code is written and padded to a byte boundary.
- `gaps`: 5 bits per 64-bit window = the offset of the first code that begins in that window; padded with zeros up to a multiple of 512 windows and packed with `packbits`. Requires a maximum code length ≤ 32 bits.
- `output_positions`: index of the first element beginning in each 4096-byte chunk, plus the total element count at the end; uint32 stored as a uint8 view. Its length is `ceil(len(encoded_exponent) / 4096) + 1`, and its final value is the **weight count**, equal to `len(sign_mantissa)` — not the encoded byte length. [CONFIRMED by measurement, 0.1/0.2; the earlier wording "plus `len(data)`" was ambiguous and was read wrongly once already.]
- `luts`: uint8, shape `(n_prefixes + 1, 256)`. In the kernel, a value ≥ 240 means "jump to LUT 256−v". Therefore:
  - **no real exponent may be 240–255** (enormous values, Inf or NaN), and
  - the number of prefix tables is bounded at **17**, not 16. Jump values 240..=255 reach targets 1..=16, and table 0 makes seventeen. An eighteenth table's jump value would fall below 240 and be read as a symbol instead. [CORRECTED, Phase 5 — derived from `get_luts` and consistent with `decode.ptx`, not measured: no real unit here exceeds four tables. This section previously said 16, which would refuse a legal unit.]
  If a model violates this, we must abort with a clear error, never emit a file.
  **Carry-forward leakage [CONFIRMED by measurement, 0.5].** The official
  `get_luts` fills each table with a carry-forward loop whose accumulator is
  function-scoped, so a table after the first that lacks key `0` begins filled
  with the *previous* table's trailing value. Verified in a real official output:
  128 bytes of one row held the prior row's value. Byte-identity therefore
  requires reproducing this. See [`COMPATIBILITY.md`](COMPATIBILITY.md) for the
  default (`--luts=compat`) and the gated opt-out (`--luts=correct`).
- `split_positions`: int64, the **internal** boundaries of the concatenated tensors — for n tensors it has **n−1** entries, equal to `cumsum(sizes)[:-1]`. The total is *not* stored here; it is `len(sign_mantissa)`. Empty for a single-tensor unit (a bare `nn.Linear` or `nn.Embedding`). **The concatenation order is the order of `attr_names` in the `pattern_dict`.** [CONFIRMED by measurement, 0.1/0.2: a 7-tensor unit produced 6 entries.]
- **Kernel 32-bit limits**: `n_bytes` and `n_elements` are `int` and `output_positions` is `uint32`. A UC cannot exceed 2³¹−1 weights or 2³¹−1 bytes of bitstream. Flux is far from this (max ~340M), but an LLM with a huge UC (e.g. giant embeddings) could hit it. This must be checked before encoding.
- Output names per UC: `<uc>.luts`, `.encoded_exponent`, `.sign_mantissa`, `.output_positions`, `.gaps`, `.split_positions`. The compressed `.weight` tensors disappear; the rest of the state_dict (biases, norms, layers outside the pattern) is preserved.
- `dahuffman`'s Huffman tree tie-breaking: a heap of tuples `(frequency, list of (symbol, code))`; ties are resolved by comparing the lists, and EOF always compares as smallest. It is deterministic and reproducible in any language.

### 1.4 What `bf16-exponent-compression` contributes

- Its format **is not DF11-compatible** (canonical Huffman, 8-bit index). So the format is not reused, but three things are:
  1. `extract_real_weights.py`: streaming safetensors reading without torch, tensor by tensor.
  2. `bitpack.py`: the idea of encoding vectorised and proving byte-for-byte identity against the reference. Note: its implementation materialises an array of 1 byte per bit, which is precisely the RAM pattern we want to avoid.
  3. The methodological discipline: verify bit-exactness before measuring times, don't combine optimisations without measuring them separately, and don't extrapolate across architectures.
- Scope note: that repo records that DF11 is no longer the state of the art in decoding speed. This does not affect this project, whose goal is **compatibility with the existing DF11 ecosystem**, not a new codec.

### 1.5 Asymmetry: compression is serial, decompression is parallel [CONFIRMED]

DF11 **does not use the same method in both directions**. The project is designed
around inference (the paper and the kernel are from April–May 2025; the
compression code was published in August 2025 as a utility).

| | Compression (`dfloat11_utils.py::encode`) | Decompression (`decode.cu`) |
|---|---|---|
| Where | CPU | GPU |
| Parallelism | None: Python loop, one symbol per iteration | 512 threads per CUDA block, thousands of blocks |
| Unit of work | The UC's entire stream | An 8-byte window per thread |
| Cost per symbol | Hundreds of ns (interpreter) | One LUT access plus a shift |

How the kernel decompresses, in three synchronised phases (`__syncthreads`):

1. **Counting.** Each thread takes its 8 bytes (plus 4 of margin), uses its `gaps` value to know at which bit the first code born in its window starts, and counts how many symbols start there. No thread depends on another.
2. **Parallel prefix sum.** With those counts it performs a Blelloch scan (up-sweep and down-sweep) in shared memory. It starts from `output_positions[block]` and obtains, for each thread, the output index it writes to.
3. **Decoding and writing.** Each thread decodes its window again and directly **writes the fused BF16** (decoded exponent + `sign_mantissa` byte) at its position.

**Why this is a good starting point for this project:**

- The metadata that makes decompression parallel (`gaps` and `output_positions`) is exactly what the encoder computes serially. The information needed to parallelise exists; the compressor simply does not exploit it.
- The technique in §5.3 (per-chunk histogram → bit length of each chunk → prefix sum → each chunk encodes from its own offset) is **the kernel's trick in reverse**: the kernel counts symbols per window and prefix-sums symbols; the encoder counts bits per chunk and prefix-sums bits.
- The CPU decoder used in safe mode can copy the kernel's structure (independent windows + prefix sum) and be just as parallel with `rayon`. That way verification without a GPU is not serial either.
- It also confirms that the official team already knew compression is parallelisable across UCs: their `compress_flux_parallel.sh` launches separate processes over block ranges with `taskset`. But **each process loads the whole model** (the README warns that it needs a lot of memory), so parallelising multiplies the RAM instead of dividing it. That is exactly the problem this project solves.

### 1.6 Diffusers layout [CONFIRMED]

With `save_single_file=False`, `compress_model` does the following:

- Before processing each UC, it **detaches it from the parent model** (`setattr(parent, child, None)`).
- Saves each UC in its own shard: `<name with dots replaced by _>.safetensors` (e.g. `transformer_blocks_0.safetensors`) with keys `transformer_blocks.0.luts`, etc.
- At the end it calls `model.save_pretrained(save_path)`. Since the UCs no longer hang off the model, that saves **only the remaining parameters** (input embeddings, final norms, etc.) into `diffusion_pytorch_model.safetensors` (diffusers shards above 10 GB by default) and writes `config.json`.
- It injects `dfloat11_config` (version, `threads_per_block`, `bytes_per_thread`, `pattern_dict`) into `model.config` before saving. If `config.json` does not contain it, it rewrites the file with only that key.

How it is loaded (`DFloat11Model.from_pretrained` + `load_and_replace_tensors`):

- Reads `config.json` and requires `dfloat11_config`.
- Loads **all** the `.safetensors` files in the directory, in any order, and dispatches each tensor by name. The file name is irrelevant to the loader (useful for us).
- In diffusers, the BF16 model is supplied by the pipeline (`bfloat16_model=pipe.transformer`).

Differences from ComfyUI-native that affect the design:

- **Different names and grouping.** In Flux, ComfyUI has `double_blocks.N` with a fused `img_attn.qkv`; diffusers has `transformer_blocks.N` with separate `to_q`, `to_k`, `to_v` (14 linears per double block versus 10). The weights are the same, but the concatenation order changes, so **the bitstream is not interchangeable** between layouts. They are two distinct outputs from two distinct sources.
- **There are more transformations.** The Extended node converts diffusers → ComfyUI on the fly (`convert_fixed_tensors.py`), including a `swap_scale_shift` on some norms. That does not touch the compressed UCs, but it shows that uncompressed tensors can require per-architecture transformations.
- **`config.json` must come out the same as `save_pretrained`'s**: the transformer's original configuration plus the fields diffusers adds (`_class_name`, `_diffusers_version`, etc.) and `dfloat11_config`. We do not instantiate the model, so it has to be rebuilt from the source `config.json` (H10).

Existing ways of consuming a diffusers DF11 model that must be supported: a
directory with one shard per UC (the default output) and a single file with
`from_single_file` plus an explicit `pattern_dict`.

---

### 1.7 Chroma: definitions per layout [CONFIRMED in source]

Verified by reading `pattern_dict.py` from ComfyUI-DFloat11-Extended and
`diffusers/models/transformers/transformer_chroma.py` (diffusers wheel 0.40.0).

**Constructor** (compared to Flux): `patch_size=1`, `in_channels=64`,
`num_layers=19`, `num_single_layers=38`, `attention_head_dim=128`,
`num_attention_heads=24`, `joint_attention_dim=4096`, `axes_dims_rope=(16,56,56)`
all match. It adds `approximator_num_channels=64`,
`approximator_hidden_dim=5120`, `approximator_layers=5`. It removes
`pooled_projection_dim` and `guidance_embeds`. ~8.9B parameters, based on
FLUX.1-schnell.

**Why the modulations disappear.** The classes `ChromaAdaLayerNormZeroPruned` and
`ChromaAdaLayerNormZeroSinglePruned` contain no `nn.Linear` at all: they are a
`LayerNorm` with no affine parameters that receives the modulation already
computed from outside (`emb=temb`). It is produced by `ChromaApproximator`, a
shared network. Hence the parameter-count difference against Flux 12B.

**ComfyUI-native layout** (the `"Chroma"` entry already present in Extended; this
concatenation order must be replicated exactly):

- `distilled_guidance_layer\.layers\.\d+` → `in_layer`, `out_layer`
- `double_blocks\.\d+` → `img_attn.qkv`, `img_attn.proj`, `img_mlp.0`,
  `img_mlp.2`, `txt_attn.qkv`, `txt_attn.proj`, `txt_mlp.0`, `txt_mlp.2` (8)
- `single_blocks\.\d+` → `linear1`, `linear2` (2)

Corrections to the earlier estimate: the native single blocks of Flux and of
Chroma have **the same 2** elements (the modulation was already outside the UC in
Flux), so there is no delta there. And the approximator is **not a single UC**: it
is split into one UC per layer (5 UCs of 2 linears each), a much finer
granularity that is favourable to the RAM budget.

**Real structure of `ChromaApproximator`**: `in_proj` (Linear), `layers`
(5 × `PixArtAlphaTextProjection`, each with `in_layer` and `out_layer`), `norms`
(5 × RMSNorm), `out_proj` (Linear). **In the diffusers layout `in_proj` and
`out_proj` are compressed** along with the `layers` linears, in one unit — see
below and FINDINGS §0.9. Only the RMSNorms stay uncompressed. Whether the
ComfyUI-native pattern in Extended makes the same choice is **not yet verified
against a published native release**; treat the native claim as unconfirmed.

**Diffusers layout** — [CONFIRMED by measurement, 0.9]. Taken from the
`dfloat11_config.pattern_dict` inside the published `DFloat11/Chroma-DF11` and
`DFloat11/FLUX.1-dev-DF11` releases, i.e. the configuration the official
compressor actually ran with. **The orders below are load-bearing**: they fix the
concatenation, hence `split_positions`, hence every compressed byte. An earlier
draft of this section derived them from the class definitions and got the right
*set* with the wrong *order* in both patterns — see FINDINGS §0.9.

- `transformer_blocks\.\d+` → 12, in this order: `attn.to_q`, `attn.to_k`,
  `attn.to_v`, **`attn.add_k_proj`, `attn.add_v_proj`, `attn.add_q_proj`**
  (k, v, q — not q, k, v), `attn.to_out.0`, `attn.to_add_out`, `ff.net.0.proj`,
  `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`.
- `single_transformer_blocks\.\d+` → 5, in this order: **`proj_mlp`,
  `proj_out`**, `attn.to_q`, `attn.to_k`, `attn.to_v` — the projections come
  *first*. There is no `attn.to_out.0` and there are no `add_*`, because the
  single block builds `FluxAttention` with `pre_only=True` and without
  `added_kv_proj_dim`.
- `distilled_guidance_layer` → **one unit of 12**, with no layer index:
  `in_proj`, `layers.0.linear_1`, `layers.0.linear_2`, … `layers.4.linear_1`,
  `layers.4.linear_2`, `out_proj`.

For reference, Flux diffusers is the same shape plus the modulation linears:
`transformer_blocks\.\d+` → 14 (`norm1.linear`, `norm1_context.linear`, then the
12 above) and `single_transformer_blocks\.\d+` → 6 (`norm.linear`, then the 5
above).

**H13 is refuted [0.9].** This section previously claimed the approximator splits
into five units of two and that `in_proj`, `out_proj` and the RMSNorms stay
uncompressed. In fact `in_proj` and `out_proj` **are compressed**, inside a single
unit, and the sub-modules are named `linear_1`/`linear_2` rather than
`in_layer`/`out_layer`. Only the RMSNorms are genuinely left uncompressed. The
structural conclusion — Chroma is Flux minus the modulations, plus the
approximator — survives; the granularity and the naming did not.

**ChromaRadiance.** Extended additionally defines this variant: identical to
Chroma plus `nerf_blocks\.\d+` → `param_generator` (a single element). Adding it
is almost free, but it forces support for **single-tensor UCs**, an edge case the
encoder and the golden tests must cover explicitly.

---

## 2. Hypotheses to verify (Phase 0)

Nothing gets implemented until this table is closed. Every hypothesis has a
refutation criterion.

| ID | Hypothesis | How it is measured | Refuted if… |
|---|---|---|---|
| H1 | Official peak RAM ≈ 2× the model, decomposing into loading + per-UC temporaries + serialisation | `/usr/bin/time -v`, `tracemalloc` and RSS sampling per stage, on a small model and (if a machine is available) on Flux | the peak does not scale with model size |
| H2 | >90% of the time is in the Python `encode` loop | `py-spy` on the official compressor | another stage dominates |
| H3 | UCs are independent | already confirmed in source | — |
| H4 | We can produce tensors **byte-identical** to the official ones by replicating `dahuffman` | golden tests (§8) | any difference in any tensor |
| H5 | With a native encoder the bottleneck becomes the disk, not the CPU | isolated encode throughput vs sequential disk read | the CPU still limits below HDD speeds |
| H6 | The loaders (official DF11, Extended node) do not depend on the physical order of tensors inside the safetensors file | load a reordered file and compare inference output | some loader fails or differs |
| H7 | The `decode.ptx` kernel can be loaded without CuPy (CUDA driver API from a native binary) | minimal load-and-run test | it only works via CuPy |
| H8 | The uncompressed tensors in the official file are identical to those in the source safetensors | compare tensor by tensor | ComfyUI adds buffers, casts dtypes or renames on instantiation |
| H9 | In diffusers, the official `diffusion_pytorch_model.safetensors` contains exactly the source tensors that belong to no UC, unchanged | compare tensor by tensor | tensors are missing, extra or different |
| H10 | The output `config.json` can be rebuilt without diffusers: source config + `save_pretrained` fields + `dfloat11_config` | diff against the official one | fields appear that depend on the installed diffusers version |
| H11 | Splitting into shards (one per UC) is not mandatory for the loader; a single shard or arbitrary groupings also load | load variants and compare inference | the loader depends on file names |
| H12 | A CPU decoder with an independent-window structure scales nearly linearly with cores | measure on 1, 2, 4, 8… cores | it saturates on memory bandwidth very early |
| H13 | Chroma's approximator (`in_proj`, `out_proj`, RMSNorms) stays **uncompressed** and the official loader expects it that way | compress Chroma with the official tool and compare its list of compressed tensors against §1.7 | Extended's pattern differs from what the official loader assumes |

H8 is critical: the official tool works on the **instantiated model**, we work on
the **file**. If ComfyUI or diffusers alter anything on instantiation, we will
have to replicate it explicitly per architecture and per layout. H8 and H9 are
the same question in the two ecosystems.

---

## 3. Goals and non-goals

**Goals**

1. Output compatible with the existing loaders, in both layouts: ComfyUI-native (original node and Extended) and diffusers (`DFloat11Model.from_pretrained` and `from_single_file`). Strong criterion: tensors byte-identical to the official compressor's for the same input, and an equivalent `config.json`.
2. RAM bounded by a budget the user sets, independent of model size.
3. Parallelism that adapts to the machine (cores, RAM, disk type).
4. Resume after interruption without losing more than one UC of work.
5. Two modes: fast and safe. Plus independent post-hoc verification.
6. Self-contained binary, no Python environment on the main path.

**Non-goals (v1)**

- New codecs or ratio improvements. The format is DF11 as it stands.
- Inference. Compression and verification only.
- dtype conversion: the input must be BF16.
- Conversion between layouts (diffusers ↔ ComfyUI). Each output is generated from a source in its own layout. This stays a possible extension, since the Extended node shows the conversion is feasible.

---

## 4. Choice of language per task

Single criterion: efficiency (time, RAM, energy) and absence of blockers.
Familiarity does not count.

| Task | Character | Recommendation | Reason |
|---|---|---|---|
| safetensors reading (JSON header + data) | I/O, `pread`/`mmap` | **Rust** | The `safetensors` crate is the format's reference implementation; `memmap2` and direct `pread`. |
| Field separation (exponent / sign+mantissa) | Pure SIMD | **Rust** | Autovectorises well; C would be equivalent in speed. |
| Exponent histogram | SIMD, trivially parallel | **Rust** (rayon) | 256 counters, no contention. |
| Huffman construction compatible with `dahuffman` + 32-bit limit + LUT | ~257 symbols, negligible cost | **Rust** | The cost is irrelevant; what matters is replicating the exact tie-breaking. |
| **Bitstream encoder + gaps + output_positions** | **The hot path** | **Rust**, CPU | See §5.3. A branchless 64-bit bit writer should be in the low single-digit ns per symbol. |
| safetensors writer + journal | Sequential I/O, `fsync`, `rename` | **Rust** | Fine control over writes and atomicity. |
| Reference CPU decoder (safe mode without GPU) | Exact replica of the kernel's semantics, with its same parallel structure (§1.5) | **Rust** (rayon) | Implementation independent of the encoder, so a bug cannot "self-confirm". Independent 8-byte windows + prefix sum = parallel across cores (H12). |
| diffusers `config.json` | JSON, once per model | **Rust** (`serde_json`) | Trivial; the hard part is knowing which fields to put in (H10), not writing them. |
| Verification with the official GPU kernel | Calling `decode.ptx` | **Rust** if H7 is confirmed; otherwise an optional **Python + CuPy shim** | The real kernel is the ground truth of the format; it is best to use exactly it. |
| Architecture definitions (`pattern_dict`, key prefixes) | Data | Versioned **TOML/JSON** | Derived from Extended's `pattern_dict.py`; this way adding models does not require recompiling. |

**Why not C, C++ or Zig?** On performance, Rust and C are equivalent for this
problem. Rust wins because of the reference safetensors crate, because of `rayon`
for data parallelism without writing a scheduler, because of shipping a single
static binary (key for old machines with no Python environment), and because of
memory safety in bit-manipulation code, where silent errors are the main risk.

**Why not GPU for encoding?** [HYPOTHESIS H5] If the native encoder exceeds disk
read speed, the GPU accelerates nothing, adds PCIe transfers, high idle power
draw, and a dependency that excludes anyone without NVIDIA. The GPU is reserved
for verification. If Phase 0 refutes H5, this reopens.

---

## 5. Architecture

### 5.1 Pipeline

```
source safetensors (HDD/SSD)
   │  reader (1 thread on HDD, N on SSD) — in physical offset order
   ▼
bounded UC queue (bounds RAM)
   ▼
encoding workers (rayon, N per budget)
   │  histogram → Huffman → LUT → encode → gaps/positions
   │  [safe mode] decode and compare before handing off
   ▼
in-order committer (1 thread)
   │  writes UCs in order, fsync, updates journal
   ▼
destination DF11 safetensors + journal
```

### 5.2 RAM budget per worker

For a UC with N weights, a worker needs roughly:

- `sign_mantissa`: N bytes
- `encoded_exponent`: ~0.34 N bytes (≈ 2.7 bits per exponent)
- gaps, positions, LUT: negligible

That is about **1.35 N bytes**. On Flux's largest UC (~340M weights) that is
~460 MB [HYPOTHESIS, calculation]. Source data is read through the page cache,
which the system frees under pressure and which does not count as our own memory.

- Workers = `min(cores, RAM_budget / (1.35 × N_max))`.
- With a 4 GB budget: ~6–8 workers. With 1 GB: 2. With 512 MB: 1 (still works).
- In safe mode with the CPU decoder, add ~2 N bytes for the decoded output, unless the comparison is done chunk by chunk without materialising it (preferable, see §7).

### 5.3 Encoder within one UC

Key idea: **the per-chunk histogram gives the bit offsets for free**.

1. Split the UC into chunks of M symbols (e.g. 1–4M).
2. Pass 1 (parallel): histogram of each chunk. Their sum gives the UC's codebook. With the codebook, each chunk's bit count is `Σ hist[s] × len[s]`. The cumulative sum gives the exact bit offset where each chunk starts.
3. Pass 2 (parallel): each chunk encodes into its own region with a 64-bit accumulator. Boundary bytes between chunks are combined with OR.
4. `gaps` and `output_positions` are derived in each chunk's same pass, because every chunk knows its global offset.
5. Tail: EOF and padding, identical to the official tool.

No per-symbol offset array is ever materialised (that would be 8 B/weight).

### 5.4 I/O scheduling

- **HDD**: a single sequential reader in physical offset order. Several concurrent readers cause head seeks and destroy throughput. Automatic detection (`/sys/block/*/queue/rotational` on Linux, a latency heuristic on other systems) with a manual override.
- **Source and destination on the same HDD**: warn. Use a large write buffer to alternate in bursts and reduce seeks.
- **SSD/NVMe**: concurrent readers.
- **Disk space**: source + destination only (~24 GB + ~16 GB on Flux). No intermediate copy space is needed (see §6).

### 5.5 Output format

Both layouts share the whole pipeline (reader, workers, committer, journal). Only
three pluggable pieces change: the **architecture definition** (key prefixes and
`pattern_dict`), the **writer**, and the **config generation**.

**ComfyUI-native**
- Source: a single safetensors with ComfyUI keys (`double_blocks.N...`), possibly with a prefix (`model.diffusion_model.`) that is detected and stripped per architecture.
- Output: a single safetensors with UCs + uncompressed tensors. No `config.json` (the Extended node discards it).

**Diffusers**
- Source: the `transformer/` directory of a diffusers repo (`config.json` + one or more `diffusion_pytorch_model-*.safetensors` shards with their `index.json`). The reader walks the shards following the index, so a sharded source changes nothing in the pipeline.
- Output, two variants:
  - **Directory** (default, like the official tool): one shard per UC with the same name the official tool generates, `diffusion_pytorch_model.safetensors` with the uncompressed tensors, and `config.json` with `dfloat11_config`.
  - **Single file**, for `from_single_file`.
- Advantage of the directory variant for this project: **one shard per UC is also a natural checkpoint**, because each file is written once and closed.

**Common**
- The file is written sequentially with the header reserved up front (the safetensors format allows padding the header with spaces) and filled in at the end. Tensors end up contiguous, with no holes.
- The physical order of tensors may differ from the official one (official safetensors sorts by dtype alignment and name). The compatibility criterion is **per tensor**, not per file, pending H6.

---

## 6. Checkpoints and resume

### 6.1 Granularity: one UC

The original proposal was "checkpoint every n blocks depending on the machine".
With the architecture above, the natural checkpoint is **every UC**:

- A UC is the idempotent unit: the same input produces the same bytes.
- Each UC is hundreds of MB of output, so the cost of an `fsync` + journal update is negligible against its work.
- The most that is lost in a crash is the workers' in-flight work (≤ N UCs in parallel).

The "n" parameter still exists, but only as an **fsync policy** (useful on very
slow disks or SD cards), not as recovery granularity.

### 6.2 Mechanics

- **Diffusers as a directory**: no ordering needed. Each UC is written to `<name>.safetensors.tmp`, `fsync`, then an atomic `rename`. The journal only records hashes. Resuming = skip the complete and valid shards. The uncompressed tensors and `config.json` are written at the end, also atomically.
- **Single file (both layouts)**: workers finish UCs in any order; the committer **writes only in order**. UCs finished out of order wait in memory (bounded by the budget) or, if they do not fit, in a temporary spill file.
- Journal (separate file, append-only), one entry per committed UC:
  - UC identifier
  - offset and length written in the destination
  - hash of the source tensors (xxh3) and of the output
  - names, dtypes and shapes of the emitted tensors
  - mode (fast/safe) and verification result
- Job fingerprint at the head of the journal: hash of the source header, file size, df11pack version and DF11 format version, `pattern_dict` used.

### 6.3 Resume

1. Read the journal and validate the fingerprint. If the source or the configuration changed, refuse to resume.
2. Validate the last entry (hash of the bytes on disk). A corrupt or incomplete entry is discarded.
3. Truncate the destination to the end of the last valid UC.
4. Continue from the next UC.

---

## 7. Verification

### 7.1 Fast mode

Compresses only. Even so it performs cheap checks that should never be disabled:
out-of-range exponents (§1.3), maximum code length, number of LUTs, symbol
counts, and hashes in the journal.

### 7.2 Safe mode

Before committing each UC, it is decoded and compared bit for bit against the
source.

- **With an NVIDIA GPU**: the official `decode` kernel (what the real loader will use). VRAM needed ≈ the UC's compressed size + 2 N bytes, about 1.2 GB on Flux's largest UC.
- **Without a GPU**: an independent CPU decoder that replicates the kernel's semantics, including gaps and output_positions (not just sequential decoding, because those indices are exactly what an encoder bug can break). It uses the kernel's same structure: each 8-byte window starts from its `gap` and is decoded in parallel. If a window starts wrong, its symbols do not match the prefix sum and the error shows up localised.
- The comparison is done per kernel chunk against the source as read, without materialising the whole decoded UC.

### 7.3 Post-hoc verification (standalone command)

Over the already-generated file and the source, three levels:

| Level | What it detects | Cost |
|---|---|---|
| **Integrity** (journal hashes) | On-disk corruption, incomplete writes | Read the destination, no decoding |
| **Semantic sampling** | Systematic encoder errors | Seconds |
| **Full sweep** | Everything | One complete decoding pass |

**On the 1k random samples.** The minimum useful unit of random access is the
**kernel chunk** (~12k weights): `output_positions` says where it starts in
symbols and `gaps` where it starts in bits, so a single chunk can be decoded and
compared against the equivalent stretch of the source (accessed with `pread`).
1000 chunks ≈ 12M weights.

But we have to be clear about what that detects:

- A **systematic error** (a bug affecting a fraction f of the chunks) is detected with probability 1 − (1 − f)^1000. At f = 0.5% that is already above 99%.
- An **isolated error** (one corrupt chunk out of ~1M) is detected with probability ~0.1%. That is what the integrity level is for, not sampling.

Hence **stratified sampling** is proposed instead of uniform:

1. At least one chunk per UC (Flux: 57).
2. First and last chunk of each UC, and the chunks containing each `split_positions` boundary, which is where boundary errors concentrate.
3. The rest uniform, up to 1000.
4. Seed recorded in the report, so the verification can be reproduced exactly.

---

## 8. Test plan (golden tests)

Minimum corpus:

1. **Qwen3-0.6B** (a small real LLM; an official example `pattern_dict` exists).
2. **Reduced synthetic Flux, in both layouts** (same name structure, few layers, small hidden size, weights ~ normal) to validate both `pattern_dict`s, `split_positions`, `config.json` and the shard split without needing 48 GB. In diffusers it is generated by instantiating `FluxTransformer2DModel` with a reduced config; in ComfyUI, with its equivalent class.
3. **Sharded diffusers source**: the same synthetic model saved across several shards with `index.json`, to validate the multi-file reader.
4. **Adversarial synthetic distributions**:
   - codes that force 2, 3 and 4 LUT levels
   - a histogram that triggers the 32-bit limit path
   - a UC with a single distinct exponent
   - a UC whose weight count is not a multiple of anything convenient
   - exponents 240–255 (must abort)
   - **single-tensor UC** (ChromaRadiance's `param_generator` case, §1.7)
   - **very small UC**: Chroma's 5 approximator UCs are orders of magnitude smaller than a double block; verify that chunk distribution and the prefix sum behave when there is less work than threads
5. **Real Flux and Chroma models in both layouts.** For Chroma both sources exist: `lodestones/Chroma1-HD` (single file) and `imnotednamode/Chroma-v36-dc-diffusers` (`subfolder="transformer"`), so the diffusers layout is no longer a blocker on a machine with enough RAM to run the official tool as a final proof.

Criterion: for every case, all tensors in df11pack's result must be byte-identical
to the official compressor's, and the official loader must produce identical
inference output, both in ComfyUI (Extended node) and in diffusers
(`DFloat11Model`).

**Delicate case:** the 32-bit limit path uses `np.argpartition`, whose ordering
among ties depends on the implementation. If it cannot be replicated exactly, a
different but valid result is accepted **only** in that case, documented and
verified with the kernel.

---

## 9. Success metrics

Always measured with the same protocol (warm-up, median of several runs,
environment recorded):

- Total time and time per stage (reading, encode, verification, writing).
- Peak RSS.
- Energy: RAPL (`/sys/class/powercap`) on Intel/AMD CPUs; `nvidia-smi` for the GPU in safe mode.
- Isolated encode throughput (symbols/s per core), to confirm or refute H5.
- Reference machines: a modern one, an old one (e.g. 4 cores, 8 GB, HDD) and one without a GPU.

---

## 10. Phases

Diffusers is in from Phase 1: the per-layout pieces (architecture definition,
writer, config) are designed from the start, not bolted on afterwards.

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0 | Measurements of the official compressor in both layouts; results for H1–H13 | Table §2 closed |
| 1 | Native single-threaded in-memory encoder over one UC; Flux and Chroma architecture definitions for ComfyUI and diffusers | Byte-for-byte identity on corpus §8 (cases 1, 2 and 4) |
| 2 | Readers (single file, and shards with index), writers for both layouts, `config.json` | The official loaders load our output; H9–H11 confirmed |
| 3 | Streaming, intra- and inter-UC parallelism, RAM budget, I/O scheduler | Bounded RAM measured; throughput limited by disk (H5) |
| 4 | Journal, ordered committer, atomic shards, resume | Kill the process at random points × 100 with no differences in the output |
| 5 | Safe mode: parallel CPU decoder and GPU kernel | Detects deliberately injected errors in bitstream, gaps and positions; H12 measured |
| 6 | Post-hoc verification: integrity, stratified sampling, full sweep | Detects corruption injected on disk |
| 7 | The remaining architectures from Extended's `pattern_dict.py` and the official examples (Wan2.1, LLMs) | Golden tests per architecture and layout |

---

## 11. Open decisions

1. ~~v1 architectures and layouts~~ **Closed:** Flux and Chroma, both layouts from the start.
2. **[DECISION] Python dependency for GPU verification**, if H7 is not confirmed: an optional shim, or CPU decoder only?
3. **[DECISION] Licence.** The official code is Apache-2.0; replicating its format and its construction algorithm is compatible with any permissive licence, but if `decode.ptx` is redistributed its licence and attribution must be respected. **Decided (2026-09-23): MIT.** `decode.ptx` is not redistributed.
4. **[DECISION] ComfyUI integration**: CLI binary only, or also a thin node that invokes it?
5. **[DECISION] Name** (`df11pack` is provisional).
6. ~~Chroma in diffusers~~ **Closed:** `ChromaTransformer2DModel` is stable (diffusers ≥0.34) and both layout definitions are derived in §1.7. It stays in Phase 1.
7. **[DECISION] Does ChromaRadiance ship in v1?** The cost is one more definition, but it drags in the single-tensor UC edge case (§1.7).
