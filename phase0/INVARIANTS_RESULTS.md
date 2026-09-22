# Phase 0, step 0.6 — DF11 format invariant checker: results

`phase0/check_invariants.py` is a standalone command-line tool that validates
a DF11 safetensors file (or a directory of shards) against the format
invariants in `docs/DESIGN.md` §1.3, without importing torch and without
loading any tensor's data fully into memory beyond the small metadata
tensors (`luts`, `output_positions`, `gaps`, `split_positions` — all at most
a few MB even for a huge compression unit). `encoded_exponent` and
`sign_mantissa`, which can be hundreds of MB, are only ever read from their
safetensors header (dtype/shape), never their data.

## Two corrections to the invariants as originally briefed

Before writing the checker, reading the official encoder
(`phase0/env/.../dfloat11/dfloat11_utils.py`) turned up two invariants in
the step-0.6 brief that do **not** match what the encoder actually does.
This was independently confirmed against two real files: a downloaded
official shard (`DFloat11/Qwen3-4B-DF11`, `model_layers_0.safetensors`) and
a locally-produced reference (`phase0/out/official/qwen3-trunc-layers-only-dir/`,
produced by the coordinator with the real official compressor). Both give
identical structural numbers, so this is not a one-off artifact of a
particular file.

**1. `split_positions` does NOT end with the total weight count.**

Source (`dfloat11_utils.py::encode_weights`):
```python
split_positions = torch.cumsum(torch.LongTensor([x.numel() for x in weights]), dim=0)[:-1]
```
The trailing `[:-1]` drops the final cumulative sum (which would equal the
total). `split_positions` holds only the **interior** boundaries between the
concatenated per-tensor weight blocks.

Measured on `qwen3-trunc-layers-only-dir/model_layers_0.safetensors` (a
`model.layers.0` unit concatenating 7 linears):
```
per-tensor sizes   = [2097152, 1048576, 1048576, 2097152, 3145728, 3145728, 3145728]
full cumsum        = [2097152, 3145728, 4194304, 6291456, 9437184, 12582912, 15728640]
split_positions    = [2097152, 3145728, 4194304, 6291456, 9437184, 12582912]   (6 entries, last dropped)
sign_mantissa len  = 15728640  (== the dropped final cumsum value, NOT the last split_positions value)
```
Corrected invariant, as implemented (`INV-SPLIT-MONO`, `INV-SPLIT-BOUND`):
- `len(split_positions) == (number of concatenated tensors) - 1`
- strictly increasing
- every value `< total weight count` (== `len(sign_mantissa)`), not equal to it
- **empty** for a single-tensor unit (a bare `nn.Linear`/`nn.Embedding`
  compression unit, e.g. `lm_head`/`model.embed_tokens` in an 8B+ LLM — this
  is a mainstream case, not an edge case). Verified structurally with a
  hand-built synthetic single-tensor unit (`split_positions` of length 0):
  the checker accepts it (see "What could not be checked" below for why no
  *real* single-tensor unit was tested).

**2. `output_positions`' trailing value is an element count, not a byte count.**

Source (`dfloat11_utils.py::encode`): the function's `data` parameter is
`exponent_8bits.tolist()`, i.e. the flat list of per-weight exponent
*symbols*. `output_positions.append(len(data))` therefore appends the total
**weight count**, not the encoded bitstream's byte length.

Measured on the same unit:
```
encoded_exponent length (bytes) = 5,233,601
ceil(5,233,601 / 4096)          = 1278 chunks
len(output_positions)           = 1279  (1278 + 1 trailing)
output_positions[-1]            = 15,728,640   == total weight count == len(sign_mantissa)
                                 != 5,233,601   (the byte length)
```
Corrected invariant, as implemented (`INV-OUTPOS-TOTAL`, `INV-OUTPOS-COUNT`):
- `len(output_positions) == ceil(len(encoded_exponent) / 4096) + 1`
- `output_positions[-1] == len(sign_mantissa)` (total weight count)

`docs/DESIGN.md` §1.3 itself is accurate on both points ("index of the first
element beginning in each 4096-byte chunk, plus `len(data)` at the end" and
"cumulative sums of the sizes of the concatenated tensors (empty for a bare
`nn.Linear`)") — it was the task brief's paraphrase of these two bullets
that was wrong. The checker implements the source-accurate version and both
real files pass it; neither file passes the brief's original (uncorrected)
phrasing, which is expected and correct.

## What was validated

All invariants from DESIGN.md §1.3 that are checkable from the file alone
(no GPU, no original weights to decode against):

| Tag | Invariant |
|---|---|
| `INV-NAMES-COMPLETE` / `-DTYPE` / `-SHAPE` | the six per-unit tensor names, dtypes (`luts`/`encoded_exponent`/`sign_mantissa`/`output_positions`/`gaps` = U8, `split_positions` = I64), and shapes are present and consistent |
| `INV-LUTS-SHAPE` | `luts` is `(n_prefixes+1, 256)`, `1 <= n_prefixes <= 16` |
| `INV-LUTS-JUMP` | every `>=240` cell in a real (non-`lens`) `luts` row has a valid jump target `256-v` among the existing prefix tables |
| `INV-GAPS-MAXLEN` | max Huffman code length (the `luts` `lens` row) `<= 32` bits |
| `INV-GAPS-SHAPE` | `gaps` byte length matches `ceil(512*ceil(n_bytes/4096) * 5 / 8)` (5 bits/window, padded to a multiple of 512 windows) |
| `INV-GAPS-VALUE` | every decoded 5-bit gap value `< 32` (see note below — structurally guaranteed, kept as a self-check) |
| `INV-OUTPOS-DTYPE` | `output_positions` byte length is a multiple of 4 (uint32-as-uint8 view) |
| `INV-OUTPOS-COUNT` | one entry per 4096-byte bitstream chunk, plus a trailing entry |
| `INV-OUTPOS-MONO` | non-decreasing |
| `INV-OUTPOS-TOTAL` | trailing entry == total weight count (== `sign_mantissa` length) — corrected, see above |
| `INV-SPLIT-MONO` | `split_positions` strictly increasing, first value `> 0` |
| `INV-SPLIT-BOUND` | last value `< total weight count` — corrected, see above |
| `INV-SM-LENGTH` | `sign_mantissa` length cross-checked against `output_positions`' independently-encoded total (non-circular: two different tensors must agree) |
| `INV-LIMIT-WEIGHTS` / `-BYTES` | weight count and bitstream byte count both `<= 2**31 - 1` |

## Result on real files

- `DFloat11/Qwen3-4B-DF11/model_layers_0.safetensors` (downloaded, 136,721,446
  bytes, one unit `model.layers.0`, 100,925,440 weights, 4-level LUT,
  max code length 27): **PASS**, exit code 0.
- `phase0/out/official/qwen3-trunc-layers-only-dir/model_layers_{0,1,2,3}.safetensors`
  (locally produced by the coordinator with the real official compressor,
  4 units, 15,728,640 weights each, 4-level LUT, max code length 25):
  **PASS** on all 4, exit code 0.
- `phase0/out/official/qwen3-trunc-layers-only-dir/model.safetensors`:
  correctly recognized as containing zero DF11 units (only uncompressed
  tensors) and skipped rather than reported as a failure.

No invariant from DESIGN.md §1.3 was found violated by a real official
output. The two "violations" that showed up during development were, as
detailed above, artifacts of the task brief's paraphrase rather than of the
files — the checker was corrected to match the source, not loosened to
match the files' actual (and correct) behavior.

## Corruption tests: does the checker actually fail, for the right reason?

Ten corrupted copies were made from the local reference file
(`qwen3-trunc-layers-only-dir/model_layers_0.safetensors`, 21,383,221 bytes),
each targeting exactly one invariant via a direct byte-level or
header-JSON-level patch (never a full re-encode — see
`run_corruption_tests.py`, executed from the session scratchpad, not
committed). Each was run through the checker and the failure tags were
compared against the expected tag.

| # | Corruption | Expected tag | Checker result | Tag(s) actually reported | Right reason? |
|---|---|---|---|---|---|
| 0 | none (sanity) | — | PASS | — | — |
| 1 | `output_positions` trailing uint32 changed (+12345) | `INV-OUTPOS-TOTAL` | FAIL | `INV-OUTPOS-TOTAL`, `INV-SM-LENGTH` | Yes |
| 2 | `luts` lens-row byte set to 33 | `INV-GAPS-MAXLEN` | FAIL | `INV-GAPS-MAXLEN` | Yes |
| 3 | `output_positions[100]` set below `[99]` | `INV-OUTPOS-MONO` | FAIL | `INV-OUTPOS-MONO` | Yes |
| 4 | `luts[0,0]` (a literal cell) set to 245 (target table 11, only 0..3 exist) | `INV-LUTS-JUMP` | FAIL | `INV-LUTS-JUMP` | Yes |
| 5 | `sign_mantissa` truncated by 1 byte (file + header shape both shrunk) | `INV-SM-LENGTH` | FAIL | `INV-OUTPOS-TOTAL`, `INV-SM-LENGTH` | Yes |
| 6 | `split_positions[2]` duplicated from `[1]` (monotonicity break) | `INV-SPLIT-MONO` | FAIL | `INV-SPLIT-MONO` | Yes |
| 7 | `split_positions[-1]` set equal to total weight count | `INV-SPLIT-BOUND` | FAIL | `INV-SPLIT-BOUND` | Yes |
| 8 | `sign_mantissa` shape inflated to 2,147,483,700 (> 2³¹−1) | `INV-LIMIT-WEIGHTS` | FAIL | `INV-LIMIT-WEIGHTS`, `INV-OUTPOS-TOTAL`, `INV-SM-LENGTH` | Yes |
| 9 | `gaps` shape shrunk by 8 bytes vs. the padded-windows formula | `INV-GAPS-SHAPE` | FAIL | `INV-GAPS-SHAPE` | Yes |
| 10 | `gaps` tensor key removed from header entirely | `INV-NAMES-COMPLETE` | FAIL | `INV-NAMES-COMPLETE` | Yes |

All 10/10 corruptions were caught, all with the expected invariant tag
present in the failure output (some also trip a second, legitimately
related invariant — e.g. corrupting the weight count trips both the
specific check and the cross-tensor `INV-SM-LENGTH` consistency check,
which is correct: they are two independent detectors of the same
underlying inconsistency, not a single test miscounted). Exit code was
verified non-zero (1) for a representative corrupted file via the CLI
directly, and 0 for the unmodified copy, confirming the tool's exit-code
contract end to end (not just the internal `Failure` bookkeeping).

Full detail (per-test failure messages) is in
`run_corruption_tests.py`'s `results.json` output, kept only in the session
scratchpad per the task's disk/scope rules (not part of this repo).

### Why "set a gap value to 33" isn't literally possible

The task brief's example corruption for the `gaps` invariant was "set a gap
value to 33". This is structurally impossible to construct: each `gaps`
window is packed as a fixed 5-bit field (`format(gap, '05b')` then
`packbits`), so decoding 5 bits can only ever yield 0–31 — there is no bit
pattern that decodes to 33. The *underlying* constraint the brief is
pointing at — "the format requires max code length ≤ 32 bits" — is real and
is checked directly instead, via the `luts` `lens` row (`INV-GAPS-MAXLEN`,
corruption test #2 above: a lens byte set to 33, i.e. a Huffman code length
of 33 bits, correctly rejected). `INV-GAPS-VALUE` (unpacking every window
and asserting `< 32`) is kept in the checker as a self-consistency check on
the unpacking code itself, but it can never be the thing that fails given
how 5-bit packing works — this is noted in the code so a future reader
doesn't mistake it for a meaningful corruption target.

## What could not be checked

- **Real single-tensor compression unit (empty `split_positions`).** No
  downloaded or locally-produced file contained one — `Qwen3-4B/8B-DF11`'s
  shards and the local `layers-only` reference both use multi-linear
  `pattern_dict` entries (7-tensor `model.layers.N` units). A real
  single-tensor unit exists in the official ecosystem (e.g. `lm_head` /
  `model.embed_tokens` on 8B+ LLMs, downloaded separately as
  `lm_head.safetensors` / `model_embed_tokens.safetensors`, ~840 MB each for
  Qwen3-8B) but downloading one would have meant an 840 MB, RAM-risky
  fetch for a single edge case, which the task's spirit ("roughly 100 MB")
  argues against. Instead this was validated with a small hand-built
  synthetic safetensors file (`synthetic_single_tensor_unit.safetensors`,
  ~1.4 KB, in the session scratchpad, not part of this repo) that has a
  correctly-empty `split_positions`; the checker accepts it (`PASS`). This
  confirms the checker's *logic* handles the documented edge case, but it
  is not evidence about a real official encoder output — flagging this
  explicitly as requested.
- **Bit-exact correctness of the Huffman/LUT decode against the original
  BF16 weights.** Out of scope for this step (that's H4/H7 in DESIGN.md
  §2, needing either the CUDA kernel or a from-scratch decoder); this
  checker only validates structural/format invariants, not that decoding
  the bitstream reproduces the original weights.
- **`dahuffman` tie-breaking determinism** (DESIGN.md §1.3's last bullet).
  Not a per-file structural invariant — it's a property of the *encoder's*
  construction algorithm, unobservable from a single output file alone.
- **The "no real exponent may be 240–255" constraint** was checked only
  indirectly, via `INV-LUTS-JUMP`'s target-existence check (which is what
  actually breaks if a literal symbol's LUT byte value collides with the
  jump range) and `INV-GAPS-MAXLEN`'s lens-row check. Directly verifying
  that no *literal* LUT cell holds an exponent value that also happens to
  be `>= 240` (as opposed to a jump marker) would require correlating
  every literal cell against the codebook's known symbol set, which is
  reconstructible from the `lens` row but wasn't implemented as a separate
  tag; in practice `INV-LUTS-JUMP` already flags the dangerous case (an
  unresolvable jump), which is the actual hazard the design doc calls out.
- **Version-string handling**: this checker never reads `dfloat11_config`
  or `config.json` (it validates the safetensors payload only), so the
  `0.5.0` vs `0.2.0` version-string difference the coordinator flagged does
  not affect it either way.

## Cleanup

The downloaded shard (`.../scratchpad/df11_layer0.safetensors`, 136,721,446
bytes) and all corrupted copies / synthetic test files were deleted from the
scratchpad at the end of this task; nothing outside `phase0/` in the repo
was modified, and no file in `phase0/out/` (the coordinator's reference
output) was touched — corruption tests operated on copies only.
