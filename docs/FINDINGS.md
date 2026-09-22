# Phase 0 — Findings

One section per hypothesis or step: method, raw numbers, verdict, and the
consequence for [`DESIGN.md`](DESIGN.md) or [`PLAN.md`](PLAN.md).

Anything not yet measured says so. Predictions are recorded before the
measurement that would confirm them and are not edited afterwards.

---

## 0.2c — Does the LLM `pattern_dict` put the embedding in one unit?

**Question.** Whether tier 1 (full Qwen3-0.6B, already on disk) can be compressed
by the *official* compressor on the 3 GB development machine, or whether it needs
rented hardware.

**Method.** Two sources of evidence, neither requiring the official compressor to
run:

1. The `dfloat11_config.pattern_dict` carried in the `config.json` of published
   DF11 LLM releases, fetched directly from Hugging Face. This is authoritative:
   it is the configuration the official compressor actually used.
2. The safetensors header of the local `Qwen3-0.6B` BF16 file
   (`../bf16-exponent-compression/real_model/model.safetensors`), read without
   torch, to compute exact unit sizes.

### Finding 1 — the pattern is a per-model choice, not an architecture constant

| Release | Compression units | Standalone units (empty `attr_names`) |
|---|---|---|
| `DFloat11/Qwen3-4B-DF11` | `model.layers.\d+` | none |
| `DFloat11/Qwen3-8B-DF11` | `model.layers.\d+`, `lm_head`, `model.embed_tokens` | `lm_head`, `model.embed_tokens` |
| `DFloat11/Qwen3-14B-DF11` | same as 8B | `lm_head`, `model.embed_tokens` |
| `DFloat11/Qwen3-32B-DF11` | same as 8B | `lm_head`, `model.embed_tokens` |
| `DFloat11/Llama-3.1-8B-Instruct-DF11` | same as 8B | `lm_head`, `model.embed_tokens` |

The per-layer unit is identical across all of them — seven tensors, in this
order: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`,
`self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`.

Everything from 8B up also compresses `lm_head` and `model.embed_tokens` as
**standalone units with an empty `attr_names` list**. `Qwen3-8B-DF11`'s shard
listing confirms they are emitted as their own files, `lm_head.safetensors` and
`model_embed_tokens.safetensors`, alongside 37 `model_layers_N.safetensors`.

The 4B release is the odd one out. Nothing in the published artefacts explains
why; it is recorded as an observation, not a rule.

**No `Qwen3-0.6B-DF11` or `Qwen3-1.7B-DF11` release exists.** So for tier 1 there
is no official output to be byte-identical *to*, and the pattern is ours to
choose — with the fixture produced by running the official compressor under our
chosen pattern, which keeps both sides of the comparison under our control.

### Finding 2 — the two patterns differ by an order of magnitude here

Computed from the local file's header. Qwen3-0.6B: 311 tensors, 751.6M weights,
28 layers, all 7 attributes present in every layer.

| | Largest unit | BF16 bytes |
|---|---|---|
| Layers-only (4B style) | **15.73M weights** | 30.0 MiB |
| Mainstream (8B and up) | **155.58M weights** | 296.8 MiB |

All 28 layer units together are 440.4M weights; `model.embed_tokens.weight` and
`lm_head.weight` are 155.58M each and, as recorded in 0.2c's earlier note, are
**byte-identical** to one another despite both being stored.

### Prediction (recorded before measurement)

From DESIGN §1.2's decomposition — `.tolist()` at 8 bytes per weight dominating,
plus the `cat` copy and the int16/uint8 temporaries — on top of a ~1.40 GiB
instantiated model:

| Pattern | `.tolist()` | Other temporaries | Per-unit transient | Predicted peak |
|---|---|---|---|---|
| Layers-only | 0.12 GiB | ~0.07 GiB | ~0.19 GiB | **~1.6 GiB — fits** |
| Mainstream | 1.16 GiB | ~0.72 GiB | ~1.88 GiB | **~3.3 GiB — does not fit** |

Available RAM on this machine is ~2 GiB of 3 GiB total.

**Prediction:** under the layers-only pattern the official compressor completes on
full Qwen3-0.6B here; under the mainstream pattern it does not, failing while
processing `model.embed_tokens` or `lm_head`. The layers-only margin is thin —
about 0.4 GiB — and rests on the model being instantiated once. If the loader
holds a `state_dict` alongside the model, it is gone.

**Not yet measured.** Confirming this requires the pinned environment from step
0.1. Until then it is a calculation from the header, not a result.

### Verdict

Tier 1 is runnable locally **if** we adopt the layers-only pattern. The 4–8 GB
row in PLAN.md's hardware table stays unexercised for tier 1.

### Consequences

1. **Single-tensor units are mainstream, not an edge case.** DESIGN §1.7 and §8
   treat a unit holding one tensor as a ChromaRadiance curiosity. It is in fact
   how every published DF11 LLM at 8B and above stores its embedding and output
   head. It must be a first-class path in Phase 1, tested from the start, not an
   adversarial case bolted on at the end. `split_positions` is empty for these
   units, per DESIGN §1.3.
2. **Phase 1's step 1.8 gains a schema requirement:** an architecture definition
   must express a unit with an empty `attr_names`, meaning "compress this
   module's own `.weight` as one unit", alongside the usual list-of-attributes
   form.
3. **A cheap first proof that the streaming design earns its keep.** The
   622.3M-weight `model.embed_tokens` unit of `Qwen3-8B-DF11` is a published
   official output of exactly the standalone-unit kind. Reproducing that single
   unit needs ~840 MB under df11pack's ~1.35 × N budget — it **fits on this
   machine**, while the official compressor would need roughly 7 GiB of
   transients for the same unit and would not. Both sides can be fetched by byte
   range from the safetensors files rather than downloading either model whole.
   This is a strong, early, low-cost target: the first thing df11pack does that
   the reference implementation cannot do here.
4. The layers-only choice for tier 1 is a **local** decision for fixtures. df11pack
   must still support the mainstream pattern, and consequence 3 is how that gets
   tested without a large machine.

---

## 0.1 — Reference environment, and the GPU that turned out not to be needed

**Result: the official compressor runs on this machine, with no GPU and no cupy.**

Pinned: Python 3.12.3, torch 2.14.0+cpu, numpy 2.5.3, safetensors 0.8.0,
dahuffman 0.4.2, transformers 5.17.0, dfloat11 0.5.0. Rebuild steps in
[`../phase0/README.md`](../phase0/README.md).

The official `dfloat11` package does `import cupy` at module scope and builds a
`RawModule` from `decode.ptx` at import time, so it cannot be imported at all
without cupy — even to compress, which is pure CPU work. Reading the source
shows cupy is otherwise reached in only two places, both inside
`cp.cuda.Device` contexts: inference, and `check_correctness=True`.

Rather than install ~2.5 GB of CUDA wheels on a GPU-less machine,
`phase0/shim/cupy.py` satisfies the import and **raises on any real use**. That
is the stronger option, not merely the cheaper one: a compression run that
completes with the stub on `sys.path` has *provably* never touched the GPU.

**Consequence.** Compression is GPU-free in the reference implementation too.
Nothing in Phase 0 through Phase 4 needs a GPU. The first thing that does is H7
(loading `decode.ptx` natively), so all GPU work should be batched into a single
rented session rather than spread across steps.

## 0.2 — Tier-0 corpus, and the first reference output

**Built:** `phase0/corpus/tier0/qwen3-trunc` — Qwen3-0.6B truncated to 4 layers
with the vocabulary sliced to 4096, keeping real tensor names, real shapes and
real BF16 values. 47 tensors, 71.31M weights, 136.0 MiB. Generator:
`phase0/make_tier0.py`. Crucially, **a layer unit is the same size here as in the
full model** (15,728,640 weights), so per-unit memory behaviour is identical —
only the number of units changes.

**Run:** official compressor, layers-only pattern, `check_correctness=False`.

| | |
|---|---|
| Wall time | **113.0 s** for 4 units (62.9M weights) |
| Peak RSS | **956.4 MiB** |
| RSS at start (torch imported) | 328.1 MiB |
| RSS after model load | 360.7 MiB |
| RSS after compression | 657.0 MiB |
| Size | 142,632,024 → 94,168,127 bytes (**0.660**) |

Two things stand out. The torch import alone costs 328 MiB — a third of peak,
before any work. And loading the model added only 32 MiB, because safetensors
mmaps it; the model is in page cache, not in RSS. So the official peak is driven
by **per-unit transients**, not by the model, at least at this size.

### Finding: single-file mode fails on tied-embedding models

The first attempt used `save_single_file=True` and **failed at the very last
step**, after compressing everything successfully (it had already printed
`Compression factor: 68.14%`):

```
RuntimeError: Some tensors share memory ... [{'model.embed_tokens.weight', 'lm_head.weight'}]
```

`config.json` has `tie_word_embeddings: true`, so transformers ties the two on
load; `compress_model` leaves both uncompressed under the layers-only pattern,
and the final `save_file(model.state_dict())` refuses to write shared tensors.
This is a limitation of the official compressor, not of our setup — it will hit
any tied-embedding LLM in single-file mode.

Directory mode (`save_single_file=False`) succeeds, and it is also how every
published DF11 LLM is actually shipped. **Note what it emits:** the remainder
file contains `model.embed_tokens.weight` and `model.norm.weight` — and *not*
`lm_head.weight`, which is dropped entirely because it is tied. A loader must
reconstruct it from the tie. Anything df11pack writes has to match that.

### Two invariants in DESIGN §1.3 were wrong, and are now corrected

Both found by checking the real output rather than trusting the prose, on a unit
of 7 tensors totalling 15,728,640 weights.

1. **`split_positions` holds n−1 entries, not n.** Measured `[2097152, 3145728,
   4194304, 6291456, 9437184, 12582912]` for 7 tensors — exactly
   `cumsum(sizes)[:-1]`. The total, 15,728,640, is not stored; it is
   `len(sign_mantissa)`. A single-tensor unit therefore has an **empty**
   `split_positions`.
2. **`output_positions`' trailing value is an element count, not a byte length.**
   `encoded_exponent` is 5,233,601 bytes → `ceil(/4096) = 1278` chunks →
   `len(output_positions) = 1279`, and its last value is **15,728,640**, the
   weight count. DESIGN's phrase "plus `len(data)`" meant the element stream.

Confirmed exactly as written: the `gaps` length rule (654,201 windows padded to
654,336, × 5 bits = **408,960 bytes**, predicted and measured identical), and
`luts` shape `(n_prefixes + 1, 256)` — this unit gives `(5, 256)`, so four prefix
levels are exercised by default rather than needing an adversarial case.

### Format version drift

This output carries `dfloat11_config` version **`0.5.0`**; the published
Qwen3-8B/14B/32B releases carry **`0.2.0`**. df11pack must emit the right version
per target and tolerate both on read. Recorded now so it is not discovered late.

### H1 prediction for tier 1, refined with measured numbers

A layer unit is identical in the truncated and full models, so the per-unit
transient (~490 MiB above the torch baseline) does not grow with model size, and
the model itself is mmapped. Revised prediction for full Qwen3-0.6B under the
layers-only pattern: **peak RSS in the 1.0–1.6 GiB range**, against ~2.0 GiB
available. The original 0.2c prediction of ~1.6 GiB stands as an upper bound.
Measurement follows in 0.4.

---

## 0.5 / H4 — Reproducing `dahuffman` exactly

**Verdict: H4 CONFIRMED**, with two caveats that are themselves the valuable part.
Full detail in [`../phase0/H4_RESULTS.md`](../phase0/H4_RESULTS.md); spec in
`phase0/h4_codec_spec.py` (numpy only, never imports dahuffman).

**Core codebook: exact.** 682/682 histograms matched in the agent's suite. I
re-verified independently on 300 fresh histograms with a different seed —
including all-frequencies-equal cases, which is where tie-breaking is most
exposed — and got 300/300. The rule:

> dahuffman's heap items are `(total_frequency, [(symbol, (bits, value)), ...])`.
> Every live symbol belongs to exactly one node, so two nodes' first list
> elements are never equal, and a frequency tie is always resolved by comparing
> each node's **representative** — the first symbol it was built from — never the
> `(bits, value)` tail. A representative is inherited from whichever child sorted
> smaller at merge time (structural inheritance, not "smallest symbol in the
> subtree"). EOF is injected at frequency 1 and compares as an unconditional
> minimum, so it wins every tie it takes part in.

That key is a total order, so it is portable to Rust without reference to
Python's `heapq`.

### Caveat 1 — `np.argpartition`'s tie order is load-bearing on real models

The brief allowed a documented divergence here on the assumption it was a corner
case. It is not, for any model that reaches the 32-bit cap.

Measured by exhaustively probing every single-swap alternative selection at every
ambiguous boundary across the suite (2,513 probes, 77 ambiguous iterations), the
result splits cleanly on the boundary frequency:

- boundary value **== 1**: ties are inert — **0 of 2,429** probes changed anything. Forcing an already-frequency-1 symbol to frequency 1 is a no-op.
- boundary value **> 1**: ties are decisive — **84 of 84** probes changed the resulting codebook, in **7 of 7** ambiguous iterations.

And the second regime is the realistic one: of 135 histograms shaped like real
billion-parameter BF16 exponent distributions, 5 reached the 32-bit limiter, and
in **every one** the final table-determining iteration had boundary value 2 with
13–21 tied candidates competing for a single slot.

**Consequence:** a byte-identical Rust port must reproduce numpy 2.5.3's exact
`argpartition` tie order, not merely *a* valid k-smallest selection. This
promotes the brief's "documented exception" from a footnote to a real porting
task, and it is deterministic, so it is tractable — introselect on identical
input is reproducible. It must be pinned to a numpy version in the fixtures.

### Caveat 2 — `get_luts` has a real bug, and byte-identity requires reproducing it

This is the most consequential finding of the session so far.

The official `get_luts` fills each prefix table with a carry-forward loop:

```python
for i in range(256):
    if i in bytes_dict:
        curr_val = bytes_dict[i]
    luts[pi, i] = curr_val
```

`curr_val` is **function-scoped, not loop-scoped**, so it survives across
iterations of the outer `pi` loop. Two distinct consequences:

1. **A crash.** If the *first* prefix table has no key `0`, `curr_val` is unbound
   on the first write — `UnboundLocalError`. The agent hit this in 21 of 682
   synthetic histograms, and in 0 of 135 realistic ones.
2. **Silent leakage, which actually happens on real data.** For any table after
   the first that lacks key `0`, the positions before its first key are filled
   with the **trailing value of the previous table**.

I confirmed (2) in our own official output rather than in theory. In
`model_layers_0.safetensors`, prefix table 2 ends with value 105 at key 192, and
table 3 is `{128: 96}` — it has no key 0. The real file contains:

```
row 2, positions 192..255  ->  105
row 3, positions   0..127  ->  105     <-- leaked from row 2
row 3, positions 128..255  ->   96
```

So **128 bytes of that LUT row carry state from a different table.** Those
positions are probably unreachable during decode, which is why the bug has gone
unnoticed — but they are real bytes in the file, and our criterion is
byte-identity.

**Consequence for Phase 1, step 1.6:** the LUT builder must reproduce this
carry-over exactly, including across table boundaries. An implementation that
"does the right thing" — zero-filling, or restarting `curr_val` per table —
produces a *more correct* LUT and **fails byte-identity on real models**. This is
now a required test case, not an edge case: build a unit whose second or later
prefix table lacks key 0 and assert the leaked bytes match.

This is exactly the failure mode the whole byte-identity criterion exists to
catch, and it would not have been found by reading either codebase — only by
comparing bytes against a real output.
