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
