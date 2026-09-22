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
