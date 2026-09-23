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

---

## 0.6 — Format invariant checker

`phase0/check_invariants.py`, results in
[`../phase0/INVARIANTS_RESULTS.md`](../phase0/INVARIANTS_RESULTS.md). It passes on
every real file tested: a published `DFloat11/Qwen3-4B-DF11` shard (100.9M
weights, 4-level LUT, max code length 27) and all four of our own tier-0 units
(15.7M weights each, max code length 25, min 2).

**No invariant in DESIGN §1.3 was violated by any real file** — but two were
stated wrongly in §1.3 itself and are corrected above under 0.2. The checker was
built independently from the official source and arrived at both corrections
before being told, which is why they are recorded as confirmed rather than
assumed.

### A corruption that got through

The first version of the checker caught 10 of 10 injected corruptions. I then
invented an eleventh outside its author's model of the format: set
`output_positions[600] = output_positions[599]`. Still non-decreasing, trailing
total untouched, `sign_mantissa` length unchanged — every checked property holds,
and the file no longer decodes correctly. **It passed, exit 0.**

The cause is worth stating carefully, because it generalises: the ten corruptions
were derived from the same understanding of the format as the checker, so each
one probed a property the checker already knew to look at. A test suite written
by the same mind as the thing it tests inherits its blind spots. Only an
adversary outside that model found this one.

The fix is a two-sided bound rather than mere monotonicity. A 4096-byte chunk
must contain many code starts, since a code is between `min_code_len` and
`max_code_len` bits, both readable from the LUT lengths row:

```
ceil(bits / max_code_len)  <=  output_positions[i+1] - output_positions[i]  <=  floor(bits / min_code_len)
```

With max 25 and min 2 on our unit, a full chunk's delta must lie in [1311,
16384]; the corruption's delta of 0 is now caught, naming the chunk and both
bounds. Re-verified: 11 of 11 corruptions detected, all four real units still
pass, and an independent second published shard passes.

### A gap that remains open, deliberately

Swapping two *individually valid* deltas between adjacent chunks passes
everything. I confirmed this directly: exchanging chunks 500 and 501's real
deltas of 12,337 and 12,391 leaves both values in band and the total intact, and
the checker exits 0.

Closing it would require counting real code starts from `gaps` + `luts` against
the bitstream — a partial decode, not a structural check. That belongs to the
decoder work, so it stays open and documented rather than half-solved. The same
root cause leaves residual gaps for value swaps within a LUT row and among
individual gap values: this tool verifies **structural self-consistency**, not
**content correctness**. `INV-LIMIT-*` is the one category with no residual gap,
since the check's entire content is the threshold.

That boundary is the useful output here. It says precisely which class of
encoder bug Phase 1 cannot rely on this tool to catch, and must catch by
comparing bytes against fixtures instead.

---

## 0.4 / 0.2c confirmation — full Qwen3-0.6B, and a prediction that missed

**Tier 1 completed on this machine.** Full Qwen3-0.6B, layers-only pattern, 28
units, directory mode, `check_correctness=False`.

| | Tier 0 (4 layers) | **Tier 1 (28 layers)** |
|---|---|---|
| Weights compressed | 62.9M | **440.4M** |
| Wall time | 113.0 s | **788.9 s** |
| RSS at start (torch only) | 328.1 MiB | 328.1 MiB |
| RSS after model load | 360.7 MiB | **951.1 MiB** |
| RSS after compression | 657.0 MiB | **1840.7 MiB** |
| **Peak RSS** | 956.4 MiB | **2287.6 MiB** |
| Size | 142.6 → 94.2 MiB (0.660) | **1433.7 → 869.5 MiB (0.607)** |

Every unit passes `check_invariants.py`.

### The verdict of 0.2c holds; its number did not

0.2c predicted that tier 1 would complete here, and it did — that call was right,
and it is what mattered for planning. The refined prediction recorded under 0.2,
however, was **peak RSS in the 1.0–1.6 GiB range**. Measured: **2.29 GiB**. The
prediction is wrong, by about 50% at the upper bound, and is left standing above
as written.

The error is instructive. I inferred from tier 0 that the model stays mmapped and
barely enters RSS — loading it there cost only 32 MiB. That inference did not
survive scaling: at tier 1, loading cost **591 MiB** of RSS. The tier-0 signal
was not evidence of mmap behaviour, it was evidence that 136 MiB is small. I
generalised from a measurement taken in the regime where the effect I was
measuring could not show up.

The practical consequence is that the margin was far thinner than believed. Peak
touched 2.29 GiB on a machine with 3.6 GiB total and roughly 2.0 GiB nominally
available; it completed only because page cache was evicted under pressure. The
0.2c conclusion should be read as "tier 1 fits, with little room to spare",
not "tier 1 fits comfortably".

### H1 — first real data point

Peak RSS / model size = 2287.6 / 1433.7 = **1.60×** under the layers-only
pattern. H1 proposed "≈ 2× the model". The right order, below the stated figure,
and this is the *cheap* pattern — the mainstream pattern, which adds a
155.6M-weight embedding unit, would add roughly 1.9 GiB of transients on top and
would not fit here at all. That is consistent with 0.2c's original reasoning and
with the official README's ~48 GB for 12B Flux.

Not yet a full H1 answer: this is one model at one size under one pattern, and
the stage decomposition still needs `tracemalloc`.

### H2 — strong evidence before profiling

Throughput is essentially identical across a 7× change in workload:

- tier 0: 62.9M weights / 113.0 s = **556,700 symbols/s**
- tier 1: 440.4M weights / 788.9 s = **558,244 symbols/s**

0.3% apart. Time is linear in symbol count with a negligible fixed component,
which is what a per-symbol interpreted loop predicts and what a
fixed-overhead-dominated or allocation-dominated cost would not. This does not
replace the `py-spy` breakdown, but it is independent of it, and it already makes
the central premise of the project hard to doubt: at ~558k symbols/s, Flux's
11.8B weights would take about **5.9 hours** on this machine.

### Ratio note

Tier 1 compresses to 0.607 versus tier 0's 0.660, because the truncated model's
sliced embedding is a much larger share of a much smaller file and is left
uncompressed under this pattern. Compression ratio is therefore not comparable
across tiers, and only per-unit comparisons are meaningful.

---

## Where the bytes actually go — and what an alternative index would buy

Measured directly from two real official units (tier-0 layer 0 and tier-1 layer
27, both 15,728,640 weights). Not modelled.

| Tensor | Bytes | Share of unit | bits/weight |
|---|---|---|---|
| `sign_mantissa` | 15,728,640 | **73.6%** | 8.0000 |
| `encoded_exponent` | 5,233,601 | 24.5% | 2.6619 |
| `gaps` | 408,960 | **1.91%** | 0.2080 |
| `output_positions` | 5,116 | **0.024%** | 0.0026 |
| `luts` | 1,280 | 0.006% | 0.0007 |
| `split_positions` | 48 | 0.000% | 0.0000 |
| **total** | 21,377,645 | | **10.873** |

The second unit agrees to three decimals, so this is the shape of a DF11 unit in
general, not a quirk of one layer.

**Three consequences.**

1. **`output_positions` is already free** at 0.024% of the unit — 0.0026
   bits/weight. Nothing any index scheme does to it can matter.

   **Correction to an earlier version of this section.** I first compared the
   sibling project's 8-bit index against `output_positions` and concluded it
   would save 0.018%. That was the wrong pairing. idx8 is an **input-side**
   index — it tells a thread where its block begins in the bitstream, which is
   what **`gaps`** does; `output_positions` is an output-side index and is not
   what idx8 replaces. The sibling project's own `HANDOFF.md` makes exactly this
   pairing, and its estimate of DF11's `gaps` cost, 0.21 bits/weight, agrees with
   the 0.2080 measured here. Against `gaps`, idx8 costs 0.14 bits/weight at
   BLOCK=64 and 0.07 at BLOCK=128, so it would save **0.6–1.3% of output size**,
   not 0.018%. Modest but real, and it requires a different kernel; the
   speed comparison between the two designs has never been measured. Full
   treatment in [`INDEX_SCHEMES.md`](INDEX_SCHEMES.md).

2. **The index cost that matters is `gaps`, at 1.91%** — eighty times
   `output_positions`, and the only index tensor worth a design decision. It is
   not inefficiency: 5 bits per 64-bit window is the price of letting a thread
   start mid-stream, and DF11 pays for its compactness with a two-pass decode
   (count, block-wide prefix sum, then decode). idx8 trades that for a
   single-pass decode with a warp prefix-sum on the input side, at 0.6–1.3% less
   output. Which is faster is unmeasured.

3. **The remaining headroom over the entropy bound is ~2.7%.** The sibling
   project measured this exact model family's field entropies as H(exp) = 2.645
   and H(mant) = 6.973 bits, joint 10.578 bits/weight. DF11 here achieves
   **10.873** — within **0.295 bits/weight**, or 2.7%, of that bound. Nearly all
   of the remainder is the untouched 8.000 bits/weight of `sign_mantissa`, which
   the same project showed to be near-incompressible.

So DF11's format is close to the measured floor for this approach, and the
distance left is concentrated in the field that is hardest to move. Any future
format work should be justified against these numbers rather than against
intuition about index overhead.

---

## 0.7 / H8, H9 — are the uncompressed tensors passed through untouched?

**H8 (LLM path): CONFIRMED by measurement.** Every non-unit tensor in both local
official outputs was compared against the same-named tensor in its source
safetensors: dtype, shape, and streaming SHA-256 of the bytes.

| Model | Source tensors | Non-unit output tensors | Byte-identical | Altered |
|---|---|---|---|---|
| tier-0 truncated | 47 | 18 | **18 / 18** | 0 |
| tier-1 full Qwen3-0.6B | 311 | 114 | **114 / 114** | 0 |

**132 of 132 byte-identical across two model sizes.** No tensor was invented,
cast, renamed or altered. I re-verified a sample of five independently, including
norms from two different layer shards and both remainder tensors: 5 of 5
identical.

Every absent tensor is explained: the seven compressed linears per layer (that is
the compression), and `lm_head.weight`, dropped because `tie_word_embeddings` makes
transformers share its storage with `model.embed_tokens.weight` — the same tie
that breaks single-file mode (0.2).

**H9 (diffusers path): NOT settled by measurement here.** No diffusers-layout
output was produced locally, and downloading a multi-GB diffusion model was not
justified for this question. What supports it is analysis only: `compress_model`
is a single function shared verbatim by both layouts, and the detach-then-shard
mechanism below is layout-agnostic. DESIGN §1.6 records it as confirmed from
source reading. **Recorded as open**, to be closed when a diffusers output exists
— most cheaply as a by-product of Phase 2 rather than as its own download.

### The placement rule nobody would guess

Non-unit tensors do not all stay put. The four norm tensors per layer —
`input_layernorm.weight`, `post_attention_layernorm.weight`,
`self_attn.q_norm.weight`, `self_attn.k_norm.weight` — are byte-identical but
**physically relocated into their layer's own shard**, not left in the remainder
file.

Verified by counting rather than asserting. `model_layers_0.safetensors` holds 10
tensors: the 6 DF11 unit tensors plus exactly those 4 norms. The remainder
`model.safetensors` holds only two tensors in the entire model:
`model.embed_tokens.weight` and `model.norm.weight`.

The cause is in `compress_model`: it detaches the whole layer submodule
(`setattr(parent, child, None)`) and then saves `sub_module.state_dict()` — *all*
of that submodule's state, not only the attributes named in the `pattern_dict`.

**Consequence for Phase 2.** df11pack's writer must replicate this placement, not
merely the unit tensors' placement. A writer that emits the unit tensors into the
shard and leaves every non-unit tensor in the remainder file produces a
tensor-for-tensor identical *set* with a different *distribution across files* —
which H11 may or may not forgive, and which would not be caught by any per-tensor
comparison. This is exactly the class of bug the per-file checks would miss.

## 0.8 / H6 — does physical tensor order matter?

**Structurally confirmed; inference-level open.**

A real shard (`model_layers_0.safetensors`, 21,383,221 bytes, 10 tensors) was
rewritten with fully reversed physical byte order and header key order, keeping
names, dtypes, shapes and bytes. Verified three independent ways: an independent
header parser (10/10 tensors SHA-256 identical, on-disk order confirmed
reordered), `check_invariants.py` (both files pass), and the real Rust-backed
`safetensors` reader.

Reading `load_and_replace_tensors` confirms the loader dispatches purely by
tensor-name string — regex pattern match plus module-path navigation over
`loaded_tensors.items()` — never by position or file order.

**What remains unproven**, and it is not a formality: nothing here loaded the
reordered file through `DFloat11Model.from_pretrained` or ran the CUDA kernel
over it. That needs a GPU. The cupy stub raises rather than silently skipping, so
this could not have been accidentally glossed. Also untested: redistributing
tensors *across* shard files, as opposed to reordering within one — which is
precisely what H11 asks and what the placement rule above makes non-trivial.

Both go on the batched GPU session with H7.

---

## 0.9 / H13 — Chroma's real definitions, and why guessing them would have failed

Settled from the `dfloat11_config.pattern_dict` carried inside the published
official releases — `DFloat11/Chroma-DF11` and `DFloat11/FLUX.1-dev-DF11`. These
are the configurations the official compressor actually ran with, so they are
ground truth, and they cost two `config.json` fetches rather than 36 GB of
downloads.

### H13 is REFUTED

DESIGN §1.7 and the handoff brief both state that Chroma's approximator
`in_proj`, `out_proj` and RMSNorms stay **uncompressed**. The published release
says otherwise. `distilled_guidance_layer` is **one unit of 12 attributes**:

```
in_proj, layers.0.linear_1, layers.0.linear_2, layers.1.linear_1, layers.1.linear_2,
layers.2.linear_1, layers.2.linear_2, layers.3.linear_1, layers.3.linear_2,
layers.4.linear_1, layers.4.linear_2, out_proj
```

Three corrections in one:

1. **`in_proj` and `out_proj` ARE compressed.** H13's central claim is wrong.
2. **It is a single unit, not five.** DESIGN §1.7 said the approximator splits into one unit per layer (`distilled_guidance_layer\.layers\.\d+`), which it called "much more favourable to the RAM budget". It does not; the pattern has no layer index at all, and the shard listing confirms exactly one `distilled_guidance_layer.safetensors`.
3. **The sub-modules are `linear_1`/`linear_2`,** not `in_layer`/`out_layer`.

Only the RMSNorms survive the original claim: they are absent from the attribute
list, so they do stay uncompressed.

### The concatenation orders, which could not have been guessed — and were not

Step 0.9 warned that the diffusers order "cannot be guessed; it has to come out
of an official file". That was right. DESIGN §1.7's derived orders have the
correct **set** of attributes in both cases and the wrong **order** in both:

**`transformer_blocks`** — the `add_*` projections are **k, v, q**, not q, k, v:

| pos | DESIGN §1.7 guessed | actual |
|---|---|---|
| 3 | `attn.add_q_proj` | **`attn.add_k_proj`** |
| 4 | `attn.add_k_proj` | **`attn.add_v_proj`** |
| 5 | `attn.add_v_proj` | **`attn.add_q_proj`** |

**`single_transformer_blocks`** — the projections come **first**, not last:

| pos | DESIGN §1.7 guessed | actual |
|---|---|---|
| 0 | `attn.to_q` | **`proj_mlp`** |
| 1 | `attn.to_k` | **`proj_out`** |
| 2 | `attn.to_v` | **`attn.to_q`** |
| 3 | `proj_mlp` | **`attn.to_k`** |
| 4 | `proj_out` | **`attn.to_v`** |

Order determines the concatenation, hence `split_positions`, hence the exponent
stream, hence every compressed byte. A Phase 1 built on the derived order would
have produced structurally valid, invariant-passing, **byte-wrong** output for
every Chroma and Flux unit — and the failure would have looked like an encoder
bug, not a definition bug.

### Ground truth for step 1.8

**Flux (diffusers).** `transformer_blocks\.\d+` → 14: `norm1.linear`,
`norm1_context.linear`, `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.add_k_proj`,
`attn.add_v_proj`, `attn.add_q_proj`, `attn.to_out.0`, `attn.to_add_out`,
`ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`.
`single_transformer_blocks\.\d+` → 6: `norm.linear`, `proj_mlp`, `proj_out`,
`attn.to_q`, `attn.to_k`, `attn.to_v`.

**Chroma (diffusers).** Identical to Flux minus the modulation linears —
`transformer_blocks` drops `norm1.linear` and `norm1_context.linear` (→ 12),
`single_transformer_blocks` drops `norm.linear` (→ 5) — plus the
`distilled_guidance_layer` unit above. So DESIGN §1.7's *structural* conclusion
("Flux minus the modulations, plus the approximator") was right; only its
orderings and the approximator's granularity were wrong.

## 0.8 / H10 — rebuilding `config.json` without the heavy libraries

**Reconstructable, given one declared input.** `phase0/rebuild_config.py` uses
only the standard library and reproduces the official output **byte-for-byte on
both local fixtures** in `--mode full`.

The diff between source and output config is identical in shape for both, and
almost none of it is dfloat11's doing:

- **removed:** `torch_dtype`, `rope_theta`, `rope_scaling`
- **added:** `dtype`, `rope_parameters` (merging the two removed rope keys), `layer_types`, `pad_token_id: null`, `dfloat11_config`
- **changed:** `transformers_version` (4.51.0 → 5.17.0)

Only `dfloat11_config` is injected by dfloat11. Everything else is
`save_pretrained` → `PretrainedConfig.to_json_file()` re-normalising the *entire*
config against whatever transformers version is installed. DESIGN §1.6's framing
— "source config + save_pretrained fields + dfloat11_config" — understated this:
it is not an additive step, it is a whole-schema rewrite.

**One field is genuinely uncomputable:** `transformers_version`, which is the
installed library's own version string. A Rust binary has no library to read it
from, so it must be a **declared per-run target**, like `pattern_dict` already
is, backed by a small table mapping version → which schema transformations apply.
By contrast `dfloat11_config.version` is *not* library-dependent: it is a
hardcoded constant in the dfloat11 package.

### The diffusers config is far simpler than DESIGN §1.6 assumed

DESIGN §1.6 says the output `config.json` "must come out the same as
`save_pretrained`'s ... plus the fields diffusers adds (`_class_name`,
`_diffusers_version`, etc.)". Every published diffusers DF11 release checked
contains **exactly two keys**:

```json
{"dfloat11_config": {...}, "model_type": "llama"}
```

No `_class_name`, no `_diffusers_version`, no diffusers schema at all — in
`Chroma-DF11`, `FLUX.1-dev-DF11` and `FLUX.1-schnell-DF11` alike. And
`model_type` is `"llama"` in all three, for two diffusion transformers, which is
plainly vestigial rather than meaningful.

This matches the loader: with `bfloat16_model=` supplied, it reads `config.json`
as a plain dict and requires only `dfloat11_config`, never routing it through
`AutoConfig` or diffusers. So the full-schema reconstruction DESIGN assumed is
achievable but **is not what the ecosystem ships**, and the two-key form is
trivially reproducible. Phase 2 should emit the minimal form for diffusers
targets and keep `--mode full` for the transformers path.

## 0.8 / H11 — shard grouping and file names

**Confirmed, by code and by construction.** `load_and_replace_tensors` enumerates
`*.safetensors` in the directory and dispatches purely on `tensor_name`, walked
against the live module tree; the file name is used only to open the file and is
never compared to anything.

The 28-shard output was repacked two ways, each built and deleted one at a time:
a **single merged file** (282/282 tensors byte-identical, all 28 units pass
invariants), and **4 interleaved files** named `blob_00_of_4.bin.safetensors`…
matching no unit name, with one unit's six tensors deliberately split across
three different files (282/282 byte-identical).

**A checker limitation surfaced, and was handled correctly.** Running
`check_invariants.py` per-file on the interleaved variant fails 84 checks with
`INV-NAMES-COMPLETE` — because the checker groups tensors into units *per file*,
an assumption that holds for every official fixture and is exactly what H11
tests. Rather than weaken the checker, a separate script reassembles each logical
unit by name across the whole directory, as the real loader does, and runs the
**unmodified** checker on that: 28/28 units pass. The checker's assumption is
correct for files the official compressor produces and wrong as a statement about
the format; both facts are now on the record.

**Unproven, as with H6:** nothing here loaded either variant through
`DFloat11Model` or compared inference. Batched with H7.

---

## 0.3 / H2 — where the time goes: CONFIRMED

Instrumented the official stage functions and ran tier 0 again.

| | seconds | share |
|---|---|---|
| wall total | 108.46 | |
| model load | 0.78 | 0.7% |
| `compress_model` call | 107.68 | 99.3% |
| **of which `encode`** | **101.76** | **94.5% of the call, 93.8% of wall** |

H2 proposed ">90% of the time is in the Python `encode` loop". Measured **94.5%**.
Confirmed.

**Limitation, stated rather than glossed:** only `encode` was successfully
instrumented. `compress_model` imports the other stage helpers by name at module
load, so rebinding them on the module afterwards does not intercept the already-bound
references — `get_codec`, `get_32bit_codec`, `get_luts` and `encode_weights`
recorded zero calls. That does not weaken the result: `encode` alone accounts for
94.5%, so the unmeasured remainder is at most 5.5% however it divides. It does
mean this run cannot break that remainder down, and the earlier throughput
evidence (556.7k vs 558.2k symbols/s across a 7× workload change) remains the
independent cross-check.

---

# Phase 0 closure

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Official peak RAM ≈ 2× model | **Confirmed, partially** | 1.60× measured on full Qwen3-0.6B under the cheap pattern; stage decomposition not done |
| H2 | >90% of time in the Python encode loop | **CONFIRMED** | 94.5%, plus linear-throughput cross-check |
| H3 | Units are independent | **CONFIRMED** | source; no shared state beyond configuration |
| H4 | `dahuffman` reproducible byte-for-byte | **CONFIRMED, 2 caveats** | 682+300 histograms exact; argpartition ties load-bearing; `get_luts` leak must be reproduced |
| H5 | Native encoder makes disk the bottleneck | **Deferred to Phase 3** | needs the encoder; baseline recorded |
| H6 | Loaders ignore physical tensor order | **CONFIRMED** | identical logits through the real kernel (GPU session) |
| H7 | `decode.ptx` loadable without CuPy | **CONFIRMED** | raw CUDA driver API via ctypes, bit-identical to CuPy |
| H8 | Uncompressed tensors identical (LLM) | **CONFIRMED** | 132/132 byte-identical across two models |
| H9 | Same, diffusers | **OPEN** | no diffusers output locally; closes as a by-product of Phase 2 |
| H10 | `config.json` rebuildable without heavy libs | **CONFIRMED** | stdlib rebuild, byte-for-byte on both fixtures |
| H11 | Loader ignores shard grouping | **CONFIRMED** | identical logits from merged and interleaved variants |
| H12 | CPU decoder scales with cores | **Deferred to Phase 5** | needs the decoder |
| H13 | Chroma approximator stays uncompressed | **REFUTED** | `in_proj`/`out_proj` are compressed, in one unit of 12 |

**Phase 0 exit gate: met.** Every hypothesis has a verdict or an explicit
deferral with the reason and the phase that closes it. Nothing is left silently
open.

**What Phase 0 changed in the design**, beyond settling hypotheses:

1. Two invariants in DESIGN §1.3 were stated wrongly and are corrected (0.2).
2. `get_luts` leaks state across prefix tables, so byte-identity requires reproducing a bug — this produced [`COMPATIBILITY.md`](COMPATIBILITY.md) and the gated `--luts=correct` mode (0.5).
3. `np.argpartition` tie order is load-bearing on realistic models, promoting a footnote to a porting task (0.5).
4. The diffusers concatenation orders in DESIGN §1.7 were wrong; the real ones are now recorded, and H13 is refuted (0.9).
5. Non-unit norm tensors relocate into their layer's shard — a placement rule no per-tensor check would catch (0.7).
6. `save_pretrained` rewrites the whole config schema rather than adding to it, and published diffusers releases ship a two-key `config.json`, not the diffusers schema (0.8).

**Fixtures frozen** (0.12): `phase0/fixtures/MANIFEST.json`, per-tensor sha256 for
34 shards, 324 tensors, 959 MiB of official output, tagged by provenance. Phase 1
grades against this and needs neither Python, nor the official compressor, nor a
large machine.

**The batched GPU session has run** — H7, H6, H11 and the `--luts=correct` gate
are all confirmed. See the section below. H5 and H12 remain deferred to Phases 3
and 5, where the code they need exists.


---

# GPU session — all four deferred items confirmed

One rented RTX A4000 (16 GB, driver 550.144.03, CUDA 12.4), **32 minutes,
about $0.09**. 111 MiB uploaded; nothing downloaded from Hugging Face. The
pre-registration in `gpu_session/EXPECTED.md` was written before the session.

## H7 — `decode.ptx` without CuPy: **CONFIRMED**

The kernel was loaded and run through the raw CUDA driver API via `ctypes`
(`cuInit` → `cuCtxCreate` → `cuModuleLoadData` → `cuLaunchKernel`), with CuPy as
the control. Both decoded the same 15,728,640-element unit to **bit-for-bit
identical** output.

**Consequence:** GPU verification can live inside the Rust binary. Open decision
1 — "optional Python + CuPy shim, or CPU decoder only?" — is **closed**: neither
is needed, and df11pack keeps its "no Python at runtime" property on the
verification path too.

## H6 — physical tensor order: **CONFIRMED, not merely structural**

Loading the reordered shard through the real `DFloat11Model` and running the real
kernel produced **exactly identical logits**, against a same-directory
determinism baseline. Phase 0 could only show this structurally; it is now
measured end to end.

## H11 — shard grouping and file names: **CONFIRMED**

Both repacked variants — a single merged file, and four interleaved files whose
names match no unit, with one unit's tensors split across three of them — loaded
and produced **exactly identical logits** to the original per-layer directory.

**Consequence:** the writer has real freedom in how it groups tensors into files.
The constraint that remains is the *placement* rule from 0.7 (norm tensors travel
with their layer), which is about matching the official output, not about what
the loader tolerates.

## `--luts=correct` gate — **CONFIRMED, and the mode is now releasable**

Zeroing the leaked LUT run — row 3, columns [0, 128), originally holding 105
carried over from row 2 — and decoding with the real, unmodified CUDA kernel gave
output **bit-for-bit identical** to decoding the original file, across all
15,728,640 weights.

So the inference that those positions are unreachable during decode was correct,
and it is now measured rather than argued. Per `COMPATIBILITY.md`'s own rule, the
mode may ship. Had it failed, the rule required removing the mode outright.

## Environment findings, for anyone reproducing this

Three pins the session needed, none of which were obvious in advance. They are
now in the runbook:

1. **`pip install torch` unpinned pulls a cu128 wheel** that refuses a CUDA 12.4 driver ("driver is too old, found version 12040"). Install from the index matching the card: `--index-url https://download.pytorch.org/whl/cu124`.
2. **`dfloat11` 0.5.0 imports `pkg_resources`**, which setuptools ≥81 removes. Needs `setuptools<81`.
3. **`dfloat11` 0.5.0 needs transformers 4.x.** transformers 5.x removed `no_init_weights` from `transformers.modeling_utils`, and the import fails. `transformers==4.51.0` works — the version the source model's own config records.

The script's fail-fast design earned its keep on all three: each surfaced as a
clear message naming the cause rather than as a confusing downstream failure, and
the three items that were already working were not re-run.

---

# End of Phase 2 (writer path): the first full-model comparison

`df11pack compress` on the full Qwen3-0.6B, against the official compressor on
the same machine and the same input.

| | official | df11pack | |
|---|---|---|---|
| Wall time | 788.9 s | **26.9 s** | **29.3× faster** |
| Peak RSS | 2287.6 MiB | **598 MiB** | **3.8× less** |
| Output | 869.5 MiB | 869.5 MiB | identical |
| Ratio | 0.6065 | 0.607 | identical |
| Tensors matching | — | **282 / 282** | byte-identical, dtype and shape included |

df11pack is still **single-threaded** here: streaming and parallelism are Phase 3.
Throughput is 16.4M weights/s against the official 0.558M/s.

**What this says about H5.** The disk reads at ~852 MiB/s, which a saturating
encoder would have to match at ~446M weights/s. At 16.4M/s we are still **27×
short**, so on this machine the CPU, not the disk, is still the bottleneck — H5 is
*not* yet satisfied by a single-threaded encoder, which is the honest reading. It
closes or fails at the Phase 3 gate, where the chunked encoder and inter-unit
parallelism land. What has already been removed is the interpreted per-symbol
loop, worth 29×; the remaining 27× has to come from parallelism and I/O overlap,
and may not all be there.

**On memory.** 598 MiB peak for a 1.4 GiB model, against the official's 2.3 GiB,
without any streaming yet — the whole unit is still held in memory. Phase 3's
bounded-RAM work starts from a much better position than the plan assumed.

---

# Phase 2 exit gate: passed, and it found a version trap

A second GPU session (RTX 3070, ~12 minutes, about **$0.03**) loaded df11pack's
output and the official compressor's output through the real `DFloat11Model` and
the real CUDA kernel, and compared logits.

## The result

**Identical logits, exactly, `max_abs_diff = 0.0`** — once the two directories
carry the same `config.json` schema. The tensor path is exact: the decoded model
from df11pack's output is indistinguishable from the official one.

## Two things that nearly went wrong, and what they taught

### A missing baseline produced a false "REFUTED"

The first run compared the two directories and reported a difference of 1.64.
That was the test's fault, not the output's: the skeleton comes from
`AutoModelForCausalLM.from_config`, which initialises randomly, so two
independently built skeletons differ for reasons that have nothing to do with
compression. The GPU session's own H6 script had a determinism baseline for
exactly this reason; the gate script I wrote did not.

With the baseline added — load the same directory twice, require identical logits
before comparing anything else — the baseline passed and the difference persisted,
which is what made it worth chasing rather than dismissing.

**A comparison without a baseline cannot tell "different" from "nondeterministic".**

### The difference was the config, and it is a real trap

With the baseline holding, the difference was real but not in our tensors. The
decisive test: take the **official tensors** and swap in **our `config.json`**.
Logits then matched ours exactly, `max_abs_diff = 0.0`. So the compressed data was
never in question; the config was.

The two configs differ because upstream's `save_pretrained` rewrote the schema
against whatever transformers was installed:

| | official output | df11pack output |
|---|---|---|
| RoPE | `rope_parameters: {rope_theta: 1000000, rope_type: "default"}` | `rope_theta: 1000000`, `rope_scaling: null` |
| also | `dtype`, `layer_types`, `pad_token_id` | `torch_dtype` |
| stamped | `transformers_version: 5.17.0` | `4.51.0` (the source's own) |

Under transformers **4.51.0** — the version this model's config was authored with
— the official directory's 5.x-schema config is misread, RoPE falls back to a
default, and inference changes. **A DF11 directory produced by one transformers
version can silently change inference when loaded under another**, because the
rope keys moved. Nothing in the file warns you; the weights are fine.

This vindicates preserving the source config rather than re-normalising it
(`ConfigMode::PreserveSource`). df11pack's output stays readable by the
transformers version the model was authored for, which is the version its own
config names. Matching `save_pretrained` byte for byte would mean inheriting
whichever schema happened to be installed — portability traded for a byte
comparison nobody benefits from.

**Recorded as a limitation, not a solved problem:** df11pack currently emits the
source's schema. A user who needs output for a *newer* transformers must convert
the config themselves. Making the target schema an explicit flag belongs in
Phase 2's remaining work.

## Status

- **H9 remains open for diffusers.** This closed the transformers path end to end. A diffusers output still has not been produced or loaded.

---

# Phase 3, first half: the chunked encoder and inter-unit parallelism

Developed and verified on the local machine — an **i3-2365M, 2 physical cores at
1.4 GHz, 3.6 GB RAM**. Scaling beyond 2 cores has not been measured and is not
claimed.

## Correctness first

The chunked encoder (DESIGN §5.3) computes each chunk's bit length from its own
histogram and the codebook, prefix-sums those to give every chunk its exact
starting bit offset, and lets all chunks pack in parallel. Only bytes straddling a
boundary are shared, and they combine with OR because each side writes only its
own bits. `gaps` and `output_positions` fall out of the same offsets.

It is **byte-identical to the serial encoder** on synthetic streams, degenerate
streams, every chunk size from 1 to 2²⁰, and all four real units. Chunk size is a
scheduling choice that provably cannot change output.

Five mutations, all caught. One is worth naming: **removing the EOF tail byte was
caught only by the real-unit test.** Every synthetic stream happened to end
byte-aligned, so the tail never ran. Real data covered a case the synthetic corpus
did not.

Whole-model output is identical across 1, 4 and budget-selected worker counts.

## Measured, on 2 physical cores

| | official | df11pack, 1 worker | df11pack, 4 threads |
|---|---|---|---|
| Wall | 788.9 s | 28.4 s | **11.8 s** |
| Peak RSS | 2287.6 MiB | 381 MiB | 653 MiB |

**66.9× faster than the official compressor and 3.5× smaller in memory**, on a
2012 ultrabook. Parallel speedup is 2.41× on 2 physical cores.

## A memory bug the budget work exposed

Peak RSS was 678 MiB at one worker, for a model whose largest unit needs ~70 MiB.
The cause: the tied-embedding check read **both** `lm_head.weight` and
`model.embed_tokens.weight` into memory to compare them — 297 MiB each, the two
largest tensors in the model, for a boolean. Comparing them in 1 MiB chunks
instead dropped peak to **381 MiB**, a 44% reduction for a change that touches one
line of intent.

## Streaming: what it bought (superseded by the section below)

*The figures in this subsection were measured before the streaming work landed
and are kept for the record. The current numbers are in the next section.*

## What the RAM budget did not yet do (resolved below)

`--ram` sizes the worker pool, and the arithmetic is DESIGN §5.2's: ~1.35 N bytes
per worker. **That constant is currently wrong, and the budget is advisory rather
than binding.** Two reasons, both honest consequences of streaming not being
implemented yet:

1. A worker still materialises the concatenated unit (2 N bytes) and the exponent
   stream (N) alongside `sign_mantissa` (N) and the encoded output (~0.34 N) —
   about **4.3 N**, not 1.35 N. The 1.35 N figure is the post-streaming target,
   not today's cost.
2. The remainder file is written by materialising each passthrough tensor, so peak
   is floored by the largest one — 297 MiB here regardless of budget.

So `--ram 512M` currently selects a worker count without guaranteeing 512 MiB.
Making the budget binding is the rest of Phase 3, and the claim should not be made
until it is.


---

# Phase 3, streaming: memory down 6x, output unchanged

Three allocations were removed, in order of size.

| change | peak RSS, 1 worker |
|---|---|
| starting point | 678 MiB |
| tied-embedding check compared in chunks instead of loading both 297 MiB tensors | 381 MiB |
| passthrough and sibling tensors copied through a 1 MiB buffer instead of being materialised | 92.9 MiB |
| source tensors fetched one at a time instead of all seven before splitting | **63.8 MiB** |

**63.8 MiB to compress a 1.4 GiB model**, against the official compressor's 2288
MiB — **36x less**. Output stayed byte-identical to the official compressor at
every step: 282 of 282 tensors, checked after each change.

## Measured scaling, 2 physical cores

| workers | wall | peak RSS |
|---|---|---|
| 1 | 25.9 s | 63.8 MiB |
| 2 | 13.3 s | 119.7 MiB |
| 4 | 10.9 s | 230.2 MiB |

Against the official 788.9 s and 2288 MiB: **72x faster, 10x smaller** at four
threads, or 36x smaller single-threaded.

## The budget constant is now measured rather than assumed

`BYTES_PER_WEIGHT_HELD` was 1.35, taken from DESIGN §5.2. That figure assumes the
exponent stream is re-derived on a second pass rather than kept, which is not what
the code does. The marginal cost of a worker is ~55 MiB and the first costs ~64
MiB on a 15,728,640-weight unit, so the real figure is **3.7 to 4.25 bytes per
weight**. The constant is set to 4.25 so the budget errs toward fewer workers,
and its doc comment carries the measurement rather than the derivation.

Getting to 1.35 N means not keeping the exponents — a second pass that re-reads
and re-splits to feed the encoder. That is a real further saving and is not done.

## Still open

- **H5 is not settled.** At 4 threads on 2 cores this machine does 40M weights/s; saturating its disk would need ~446M/s. Whether parallelism closes that gap needs a machine with more cores and a characterised disk.
- **H12 unmeasured.** Scaling past 2 cores has not been observed.

---

# Scaling measured: 128 cores, and where it stops helping

One rented box, ~10 minutes, **about $0.02**. It advertised 16 vCPU and turned out
to be an **AMD EPYC 7B12, 128 threads, 251 GB RAM**. Full Qwen3-0.6B, 440.4M
weights across 28 units. Raw data in `gpu_session/out/scaling_sweep.json`.

| workers | wall | peak RSS | speedup | Mweights/s |
|---|---|---|---|---|
| 1 | 11.11 s | 63.3 MiB | 1.00× | 39.6 |
| 2 | 5.73 s | 120.4 MiB | 1.94× | 76.8 |
| 4 | 2.95 s | 230.4 MiB | 3.76× | 149.1 |
| 8 | 1.72 s | 501.3 MiB | 6.45× | 255.5 |
| **16** | **1.12 s** | 842.5 MiB | **9.95×** | **394.5** |
| 28 | 1.34 s | 1421.6 MiB | 8.27× | 327.6 |
| 32 | 1.31 s | 1421.6 MiB | 8.47× | 335.8 |
| 64 | 1.64 s | 1421.6 MiB | 6.76× | 267.9 |
| 128 | 2.17 s | 1421.6 MiB | 5.13× | 203.4 |

**Scaling is near-linear to 8 workers (6.45× of 8) and best at 16 (9.95×). Past
16 it gets worse**, losing half its throughput by 128.

Two causes, and only one of them is a limit of the machine:

1. **The model has 28 units.** Parallelism here is *between* units, so more than 28 workers cannot help by construction. That is a property of the design, not the hardware, and it means a model with few large units parallelises badly however many cores are available. The chunked encoder already parallelises *within* a unit; it is currently called with the whole unit as one call, so that axis is unused. Exploiting it is the obvious next step and would lift the ceiling for exactly the models that need it most.
2. **Memory pressure and allocator contention.** Peak flattens at 1421.6 MiB from 28 workers on — every unit resident at once — and throughput falls as workers climb. The RAM budget already exists to prevent this; what it lacked was a reason to believe the default should be lower than the core count. It now has one: **more workers than units is never right, and past ~16 the return is negative on this machine.**

**Best observed: 1.12 s for a full 1.4 GiB model** on a 128-thread EPYC. *(Errata: an
earlier version called this 705× faster than the official 788.9 s. That compared
two different machines — the official run was on the 2-core i3. The same-machine
figure is 788.9 s → 10.7 s, about 74×.)*

## H5: settled, and the answer is "it depends on the disk"

At 394.5 Mweights/s the encoder consumes source at **789 MB/s**. So:

- On a **spinning disk** (~100–200 MB/s) *paired with this EPYC*, the encoder outruns the disk: disk-bound.
- *(Errata)* On the machine the project actually targets — the 2-core i3 — best throughput is ~41 Mweights/s, i.e. **~82 MB/s of source**, which is *below* a typical HDD. **Old CPU + old disk is still CPU-bound.** The earlier conclusion that H5 holds "for the machines the guiding principle targets" paired a fast CPU with a slow disk and was wrong.
- On this box's page cache, measured at 10.3 GB/s, the encoder is still 13× short: **CPU-bound, H5 does not hold.**
- On a typical NVMe (2–7 GB/s), still CPU-bound.

So H5 as originally stated holds only when the CPU is much faster than the disk
(fast CPU + HDD). It fails on NVMe, and — correcting an earlier version of this
paragraph — it also fails on an old CPU with an old disk, which is the target case. That is not a dodge: the project exists to serve old machines with
slow disks, and on those it is now disk-bound. On fast hardware there is more CPU
work to reclaim, and intra-unit parallelism is where it is.

## H12 remains unmeasured

H12 is about the *CPU decoder*, which does not exist yet. The numbers above are
the encoder. It stays deferred to Phase 5.

---

# Separating memory from CPU: same speed, half the memory

Re-measured on the same class of machine (EPYC 7B12, 128 threads) after the
worker count stopped doing two jobs. ~12 minutes, **about $0.03**. Raw data in
`gpu_session/out/scaling_sweep2.json`.

## Qwen3-0.6B, layers-only pattern (28 units, 440.4M weights)

| workers | before | after | peak RSS before → after |
|---|---|---|---|
| 1 | 11.11 s (39.6 Mw/s) | **4.96 s (88.7 Mw/s)** | 63.3 → 85.7 MiB |
| 4 | 2.95 s | 1.57 s | 230.4 → 284.9 MiB |
| **8** | 1.72 s | **1.16 s (381.0 Mw/s)** | 501.3 → 474.8 MiB |
| 16 | **1.12 s (394.5 Mw/s)** | 1.24 s | 842.5 → 858.3 MiB |
| 64 | 1.64 s | 1.42 s | 1421.6 → 1215.2 MiB |

**Single-worker throughput more than doubled**, and peak performance now arrives
at **8 workers instead of 16** — the same speed for **44% less memory**. Best
observed throughput is essentially unchanged (381 vs 394.5 Mw/s); what changed is
that you no longer have to spend memory to get it.

That matters because memory is the binding constraint on the machines this
project exists for. Before, a 512 MiB budget forced few workers and therefore few
cores. Now few workers still use every core.

## Few large units: the case the change was for

The `qwen3-8b` pattern compresses `lm_head` and `model.embed_tokens` as standalone
units of 155.6M weights each — ten times a layer. Single-worker throughput there
is **88.9 Mw/s**, indistinguishable from the 28-small-unit case's 88.7. Before the
change a unit was encoded by one thread, so a model shaped like this would have
been ten times slower per unit with no way to use the cores. Throughput is now
independent of how the model divides into units, which was the point.

## What limits a single unit now, measured

Chunk size was the obvious suspect and is **not** the answer. Sweeping it across a
256× range at one worker:

| chunk (symbols) | 1,048,576 | 262,144 | 65,536 | 16,384 | 4,096 |
|---|---|---|---|---|---|
| Mweights/s | 82.6 | 90.8 | **93.9** | 90.4 | 87.6 |

14% between best and worst. The cap is elsewhere: **the field split, the histogram
and the merge passes are all sequential O(N) work**, three passes over the unit
that no amount of chunking touches. Amdahl's law does the rest.

Both are per-chunk reductions and could be parallelised the same way the encoder
was. That is a real and available optimisation, and it is **not being taken now**:
the encoder is already 683× the official compressor and, per H5, disk-bound on the
machines this project targets. Phases 4 through 7 are correctness features the
project actually promises — journal, resume, verification — and they are worth more
than throughput nobody is waiting on.

Recorded so the next person does not have to rediscover where the ceiling is.

---

# Phase 3 exit gate: met

| gate item | status |
|---|---|
| Output identical to the single-threaded encoder | **yes** — byte-identical at every worker count, chunk size and I/O mode; 282/282 tensors against the official compressor |
| Bounded RAM measured, including the 512 MiB case | **yes** — and a 1 MiB budget still compresses, single-threaded, rather than refusing |
| H5 settled | **yes** — disk-bound on spinning disks, CPU-bound on NVMe; see above |

**I/O scheduling** rounds out the phase. `--io auto` resolves the backing device
through `/sys/dev/block/<major>:<minor>` and `queue/rotational`; on a spinning
disk reads are serialised through a gate while encoding still overlaps, because
concurrent readers make the head seek. An unresolvable device defaults to
**concurrent**, since serialising on an NVMe costs real throughput while failing
to serialise on a slow disk merely forgoes an optimisation.

Measured on this SSD: `--io sequential` is 12.59 s against `auto`'s 11.92 s, which
is the right sign — the gate is doing something, and on this hardware that
something is a small loss.

## Two mutations that survived, and what fixed them

**The rotational flag's meaning was untested.** Inverting `1` and `0` passed every
test, because this machine has only SSDs and no end-to-end test can distinguish a
mapping from its inverse when every device answers the same way. Fixed by testing
the parse directly: `1` means spinning, `0` means solid state, pinned as a fact
rather than inferred from hardware that cannot disagree.

**The read gate's existence was untested.** Deleting it left `io.sequential`
reporting `true` while reads ran concurrently — the plan said one thing and the
mechanism did another, and only a timing difference betrayed it. Fixed by making
the mechanism observable: the report now carries `max_concurrent_reads`, a
watermark of reads in flight. Under `Sequential` it must be 1. Without the gate it
reads 3, and the test fails with *"sequential mode must actually serialise reads,
not just say so"*.

That second one is the more useful lesson. A configuration flag that is only ever
checked against *itself* verifies nothing: the test must observe the behaviour the
flag is supposed to cause.


---

# Errata and open gaps, from a full review (2026-09-23)

Everything below was re-checked against code and data rather than memory.

## Claims that were wrong

1. **"705×" / "683× faster" compared two machines.** The official compressor ran on the
   2-core i3; the 1.12 s run was on a 128-thread EPYC. Same-machine: 788.9 s → 10.7 s,
   **~74×**. Corrected above.
2. **H5's verdict for target machines was wrong.** On the i3, df11pack consumes ~82 MB/s of
   source, below a typical HDD, so an old machine with an old disk is still **CPU-bound**.
   Corrected above.
3. **The Phase 4 rationale quoted a fast-machine number.** "Flux in ~30 s" is the EPYC. On the
   i3, Flux's 11.8B weights at ~41 Mweights/s is **~5 minutes** (the official tool: ~6 h on the
   same machine). The decision to drop resume still holds; the number did not.
4. **Phase 1's exit gate was declared met with one case missing.** PLAN names corpus cases 1, 2
   and 4. Case 2 — a reduced synthetic Flux in both layouts — was never built. **No Flux or
   Chroma unit has ever been encoded or byte-checked**; those eight definitions are verified
   only as `pattern_dict` transcriptions. Every byte-identity result is Qwen3.
5. **`MAX_PREFIX_TABLES = 17` was labelled "corrected by measurement". It was derived**, from
   `get_luts` and the 240 jump convention. `decode.ptx` is consistent with it — LUTs are read
   from global memory and the decode unrolls three jump levels, so table *count* is not bounded
   by the kernel — but no real unit has more than four tables, so 17 is unexercised.

## Gaps in the code, not yet fixed

| gap | consequence |
|---|---|
| ~~ComfyUI-native writes a directory of shards~~ | **Fixed.** One file, written in two passes (see below); `model.diffusion_model.` stripped when present |
| ~~Default worker count is the core count, ignoring available RAM~~ | **Fixed.** Without `--ram` the budget is 80% of `MemAvailable`; core count is only the fallback when memory cannot be read. A Flux-sized unit on the target machine now gets one worker, not four |
| ~~`--ram` ignores `--safe`~~ | **Fixed.** Safe mode adds 4.0 bytes/weight to the per-worker cost (measured ~3.6) |
| ~~`BYTES_PER_WEIGHT_HELD = 4.25` is machine-dependent~~ | **Fixed.** Raised to 6.0, covering both the 4-thread (4.25) and 128-thread (5.7) measurements |
| ~~`generation_config.json` is not copied~~ | **Resolved, and the item was mis-stated.** Upstream's is *synthesised* by transformers from config.json and version-stamped, not copied. We copy one the source ships and never invent one |
| H9 (diffusers path) | **Partly closed** — see below |


---

# Corpus case 2 built: FLUX and Chroma, both layouts — and an encoder bug it found

`phase0/make_synthetic.py` builds small models (hidden 256, two instances of every
pattern, 4–7M weights) whose module names are exactly those the architecture
definitions name, and compresses them with the **official** tool. `compress_model`
walks `named_modules()` and the attribute paths, so it cannot tell these from the
real classes; the tensors it emits are its own.

| set | layout | result |
|---|---|---|
| `synthetic-chroma-diffusers` | diffusers | byte-identical, first attempt |
| `synthetic-flux-dev-diffusers` | diffusers | byte-identical, first attempt |
| `synthetic-chroma-comfyui` | ComfyUI-native | byte-identical, first attempt |
| `synthetic-flux-comfyui` | ComfyUI-native | **one byte-range wrong: a single `gaps` entry** |

## The bug: the EOF window

When a unit's final code straddles into a new 64-bit window, no code *starts*
there — but the reference encoder checks once more at EOF and records a gap for
that window anyway. The chunked encoder (Phase 3) handled that case only when the
window lay *past* its pre-sized table, which never happens, so the entry was
silently dropped and zero-padded. `output_positions` had the identical fault at
chunk boundaries.

**This was in every output since the chunked encoder landed.** Any unit whose last
code straddled a window got a wrong final index entry. None of the four Qwen units
happened to; `double_blocks.0` of the synthetic FLUX did. A test over every stream
length from 1 to 2,500 fails at the **31st** — the case is common, not exotic.

**Safe mode would not have caught it.** The verifier checked gaps only for windows
where a code starts, and this window has none. It now checks the EOF position too,
and a test zeroing exactly that entry in the real FLUX unit fails without the fix.

Why nothing earlier caught it, in order:

1. The chunked-vs-serial tests used fixed-length streams that happened to end
   byte-aligned or without a straddle.
2. The real-unit tests covered four Qwen units, none with the shape.
3. The every-length test, once written, still could not reach a chunk boundary —
   5,000 bits against a 32,768-bit chunk — so the `output_positions` half of the fix
   was untested until the test also ran with one thread per block.
4. The verifier shared the blind spot.

The every-length test counts how often the case actually occurs and fails if it
drops below 25, so it cannot pass by never reaching it. Trimming its range for speed
tripped that guard once — at 47 — which is the guard working.

## The single file, without breaking the budget

A single file's header must list every tensor's byte range before any data, but
encoded sizes are known only after encoding. Holding every unit in memory breaks
the RAM budget; staging to temporary shards doubles disk (Flux on the target
machine: ~56 GB against 39 GB free). So the native writer **encodes twice**: once to
learn sizes, then the header, then again in order, streamed straight to disk.
`StreamingWriter` checks every tensor against its declaration, so a size mismatch
is an error rather than a corrupt file. Cost: double the CPU, which is now cheap.

Also found: **the official tool writes `config.json` in single-file mode too**,
which DESIGN §5.5 says native output does not carry. We still write none; the node
does not read it.

## H9, partly

The synthetic diffusers outputs confirm, byte for byte, the **unit shards and the
sibling tensors inside them** — both written by the official code itself via
`save_file(sub_module.state_dict())`. The **remainder file** was written by a
stand-in `save_pretrained`, since the synthetic models are not diffusers classes.
So what diffusers' real `save_pretrained` does to passthrough tensors is **still
untested**, and H9 stays open for that part only.

## Also fixed in passing

`config.json` was written with a plain `std::fs::write`, contradicting Phase 4's
promise that every file is atomic. It and `generation_config.json` now go through
`write_bytes_atomic`.

---

# Phase 6 — Post-hoc verification

## What each level can and cannot see

Every case below is a test that writes the damage into a copy of the official
output **on disk**, then runs the checker.

| damage | integrity (no source) | sample | full |
|---|---|---|---|
| `output_positions` made non-monotone | caught, right unit only | caught | caught |
| last `output_positions` ≠ weight count | caught | caught | caught |
| LUT jump to a table that does not exist | caught | caught | caught |
| `split_positions` moved by one, still increasing | **missed** | caught (checked against the source's tensor sizes) | caught |
| one weight wrong in a mandatory chunk (first/last/boundary) | **missed** | caught, located to the chunk | caught |
| one weight wrong in a chunk the sample skipped | **missed** | **missed** | caught |
| a unit the source defines but the output lacks | not looked for | caught | caught |
| any DF11 tensor byte changed, output written with `--hashes` | **caught** | caught | caught |

The last-but-one row is the honest limit of sampling, and a test pins it: it finds
a chunk that a seeded 20-chunk sample does not visit, corrupts it, and requires the
sample to pass and the full sweep to fail. Sampling finds *systematic* faults; only
`full` finds an isolated one.

Integrity without a source cannot see a wrong value at all — the journal that would
have held hashes was dropped in Phase 4. **Decided:** `compress --hashes` (opt-in,
off by default) stores a SHA-256 per DF11 tensor in the header metadata; with it,
integrity catches a flipped value with no source and names the tensor. Tensor bytes
are unchanged; only the header differs (COMPATIBILITY.md). In single-file mode the
hashes come from the first encoding pass and describe bytes the second pass writes,
so a clean `verify` also confirms the two passes agreed. Every part — the stamp, the
count, the comparison, the native-mode hashes, hashing the right bytes (checked
against `sha256sum`) — has a mutation that turns a test red.

## Every structural check is proven by a mutation

Mutating the checker one condition at a time, four survived the first round of tests
(the LUT off-by-one, the final-position check, the per-chunk bound, the split
comparison). Each now has a test that fails when it is removed. The per-chunk bound
cannot be reached by corrupting a real file — a single changed entry breaks
monotonicity first — so it is tested on a hand-built unit.

## Round trip, and cost

df11pack's own output for all five fixture sets (transformers, diffusers,
ComfyUI-native with the key prefix) passes `--level full`. On the i3, full decode
runs at about 19M weights/s on one core: ~3.3 s for the 63M-weight Qwen fixture, so
roughly **10 minutes for a 12B model**. The default sample (1000 chunks, ~12M weights at ~12k per chunk)
takes under a second plus the source reads. The decoder is still sequential;
the parallel one is the remaining H12 work.

Erratum found while writing this: COMPATIBILITY.md said `--luts=correct` output is
also stamped `df11pack_version`. The code never wrote it. The document now matches
the code.

---

# Phase 7 — The remaining architectures

## Source, pinned

Extended's `pattern_dict.py` at commit `414506d` holds 19 models. It is parsed with
`ast.literal_eval`, never executed, and only the data is kept
(`phase0/fixtures/extended_pattern_dicts.json`, with the file's SHA-256); the
repository has no licence file, so its source is not copied. The three models we
already shipped matched upstream exactly. 16 new definitions were generated;
`definitions_still_match_the_official_pattern_dicts` now covers all 24.

## What the new patterns needed

The official compressor matches with `re.fullmatch` and walks patterns in dict
order. Three shapes in the new definitions are not like Flux:

| upstream | issue | handling |
|---|---|---|
| ACEStep15 `layers\.\d++` | Python 3.11+ reads it as a *possessive* quantifier. The Rust engine accepts the same text as the nested `(\d+)+`, which matches the same strings **only at the end of a pattern** | a trailing `\d++` is translated to `\d+` (provably equivalent under fullmatch); a possessive anywhere else is refused. Mutation note: dropping the translation is an equivalent mutant for exactly that reason — the refusal is what the tests pin |
| SDXL `output_blocks\.[678]\.0` | character classes | work as-is; tested |
| ErnieImage `adaLN_modulation.1` | unescaped dots match any character | same in both engines; no change |

Two more rules from reading the official walk, both refusals rather than guesses:

- **Nested units** are refused. Upstream detaches weights as it walks, so an inner
  unit fails there; here the inner unit's tensors would also be the outer unit's
  siblings and be written twice. No shipped definition nests.
- **Attrs on a module that has its own `.weight`** are refused. Upstream ignores the
  attrs when the matched module is an `nn.Linear` or `nn.Embedding`, and uses them
  otherwise; tensor names cannot tell which. No shipped definition hits it.

## Byte-identity, all 17 sets

`make_synthetic.py` now instantiates any pattern (digit runs as 0 and 1, every member
of a character class), gives every attribute in a unit a distinct size, and builds
single-tensor units as a bare Linear. The official compressor ran on all 17 (2–37M
weights each, 260 s total, 437 MiB peak). **df11pack matched every one, byte for byte,
on the first run**, and the Phase 6 checker passes on all 21 synthetic official
outputs at `full`.

What this does and does not show: the official tool walks `named_modules()`, so it
cannot tell these models from the real classes, and the definition is reproduced
exactly. It does **not** show that a real checkpoint of that model uses these module
names, or needs no key prefix beyond `model.diffusion_model.`. That needs one real
file per architecture.

## Found in passing

`make_synthetic.py` seeded from `hash(name)`, which Python randomises per process.
The four original fixtures therefore cannot be regenerated bit-exactly. They stay
frozen by SHA-256 in MANIFEST.json (and a test pins a window inside one), and are
regenerated only when named explicitly. Every new set uses `zlib.crc32(name)`.

`df11pack architectures` printed the layout as `comfyuinative`; it now prints the
name the definition files use, `comfyui-native`.

## `qwen3-8b`, the last uncovered definition

A stand-in with standalone `lm_head` (Linear) and `model.embed_tokens` (Embedding)
units, and a `save_pretrained` stand-in writing `model.safetensors`, run through
the official tool: **byte-identical**, and it passes `verify --level full`. Every
shipped definition now has a byte-identity test.

---

# Erratum: "byte-identical" held for tensors, not files

Every byte-identity test compared **tensors**. Comparing whole files (new
`tests/whole_file.rs`) showed every unit shard and the ComfyUI single file differed
from the official ones, in the header only, two ways:

1. **Metadata.** We wrote `{"format": "pt"}` into every file. Upstream writes unit
   shards and the single file with a bare `save_file(state_dict)` — no metadata;
   only the remainder, written by the library's `save_pretrained`, carries it.
2. **Layout order.** The `safetensors` library's `serialize` sorts tensors by dtype
   (descending, in its `Dtype` enum order) and then bytewise by name, and lays the
   data out in that order. We wrote them in encoding order. The remainder matched
   only because it was all BF16 in name order.

Loaders read tensors by name, so no file we wrote loaded wrongly. But the claim was
"byte-identical", and at the file level it was false until now. Fixed: headers match
upstream per file kind, and every file is laid out in the library's order. The
single file now streams in that order, which interleaves units (every I64
`split_positions` first, then BF16, then each unit's U8 tensors): pass one keeps
the small tensors, pass two encodes each unit once and checks its small tensors
against pass one's.

**Now tested as whole files:** the tier-0 transformers directory (all 5 files), the
Flux and SDXL single files, and the Flux diffusers unit shards. Not the synthetic
diffusers remainder, which comes from a stand-in `save_pretrained`.

## Test output filled the disk

Tests wrote their outputs to the temp directory and a failing or interrupted test
never removed them; after the mutation runs this reached 29 GB and the disk filled
mid-suite. All test output now goes through `df11_fixtures::scratch`, under
`df11pack-tests/<pid>/`, and each run removes the directories of processes no
longer running. After a full suite, 8 KB remains.

---

# Phase 7 — The official releases

`phase0/import_official_releases.py` reads the `config.json` of every release in the
DFloat11 Hugging Face organisation at a pinned commit (JSON only, no weights) and
maps each to the definition that reproduces its `pattern_dict`
(`phase0/fixtures/official_releases.json`).

- **45 releases; 40 map to a definition.** The other 5 carry no `dfloat11_config`:
  four use the legacy 0.1.0 pickle format (Dolphin3.0 ×2, Llama-3.1-405B,
  Qwen2.5-32B), out of scope; one is a ComfyUI file, which Extended's Flux covers.
- **15 distinct pattern_dicts.** Four were already shipped (Chroma, FLUX.1-dev,
  Qwen3-4B, Qwen3-8B; the last also covers the DeepSeek distills, Llama-3.1-8B,
  QwQ, Qwen2.5-14B and Qwen3-14B/32B). **11 new definitions**: BAGEL, FLUX.1-Kontext
  (and Krea — same patterns with escaped dots), HiDream-I1, Llama-3.3-70B (and both
  Mistrals), OmniGen2 ×2, Phi-4, Qwen-Image (three releases), Wan (2.1 and all four
  2.2), Gemma-3 (three sizes), SD3.5-large.
- **Each is byte-identical** against a stand-in run through the official
  compressor, and passes `verify --level full`.

## A fourth layout: diffusers, single file

The Qwen-Image releases are one 26 GB `diffusion_pytorch_model.safetensors` and a
`config.json` — neither the shard-per-unit diffusers layout nor ComfyUI's
`model.safetensors` without config. Added as `diffusers-single`: the single-file
writer, that file name, and the diffusers `config.json`.

Its header was read from the real release by HTTP range request (the first few KB,
not the 26 GB): no metadata, 1,453 tensors, contiguous, laid out in exactly the
`safetensors` library's order — **the first check of our layout rule against a real
release**, and it holds. pip `dfloat11` 0.5.0 would name this file
`model.safetensors`; the releases do not, so the stand-in renames it — the one
place the fixture follows the release rather than the tool.

## Also in the stand-in generator

SD3.5 splits `transformer_blocks` with an alternation group,
`([0-9]|[1-2][0-9]|3[0-6])` and a separate `transformer_blocks\.37`; the generator
now instantiates groups by searching the integers they accept. BAGEL's `vit_model`
unit concatenates 157 attributes; wide units get narrower stand-in tensors.

## Version strings

A definition carries one `format_version`, from its canonical release. Releases
sharing a pattern_dict do not always share it (Wan2.1 is 0.2.0, Wan2.2 0.3.1;
Qwen-Image 0.3.1, its Edit variants 0.3.2 and 0.5.0). It only reaches
`config.json`'s `dfloat11_config.version`; the tensors do not depend on it.
