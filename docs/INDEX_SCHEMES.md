# Index schemes

DF11 spends most of its non-payload bytes on *indices* — the metadata that lets a
GPU thread start decoding mid-stream without replaying everything before it. This
document specifies the seam that keeps index generation swappable, and the one
alternative scheme already designed and measured elsewhere.

Nothing here changes the default. `df11` is the only scheme that produces files
the official kernel can read, and it is what df11pack emits unless told otherwise.

---

## Why a seam at all

Measured on two real official units (see [`FINDINGS.md`](FINDINGS.md)):

| Tensor | bits/weight | share of unit |
|---|---|---|
| `gaps` (input-side index) | **0.2080** | 1.91% |
| `output_positions` (output-side index) | 0.0026 | 0.024% |

The §5.3 encoder already computes, for every chunk, its exact bit length and the
prefix sum of those lengths — it has to, in order to let chunks encode in
parallel into their own regions. **That is precisely the input an index builder
needs.** So the seam is not an abstraction invented for flexibility; it is the
natural shape of data the encoder already produces.

Putting it behind an interface costs nothing now. Wiring index emission inline
into the bit writer would make any alternative a rewrite of the hot path later.

---

## The interface

Conceptually, for one compression unit:

```
IndexBuilder:
    inputs  (already computed by the §5.3 encoder):
        chunk_bit_lengths : [u64]      bits emitted by each chunk
        chunk_bit_offsets : [u64]      exclusive prefix sum of the above
        symbol_counts     : [u64]      symbols encoded per chunk
        total_symbols     : u64
        total_bytes       : u64
        code_lengths      : [u8; 256]  from the codebook

    output:
        named tensors to emit, plus their dtypes
        a validity verdict: Ok, or Unrepresentable(reason)

    properties:
        name()            stable identifier recorded in the journal
        is_df11_compatible() -> bool
```

Two rules the interface must enforce, whatever the scheme:

1. **A builder may refuse.** Some schemes are only representable when the data
   cooperates (see idx8 below). Refusal must be a typed error naming the
   violated condition and the measured value, never a silent fallback to another
   scheme — a file whose index scheme is not what the caller asked for is worse
   than no file.
2. **Only `df11` may claim DF11 compatibility.** Anything else must make
   `is_df11_compatible()` false, which the writer uses to refuse to emit a
   `dfloat11_config` or to name its output as a DF11 model. See
   [`COMPATIBILITY.md`](COMPATIBILITY.md).

---

## Scheme `df11` — the default, and the only compatible one

Reproduces the official format exactly:

- `gaps`: 5 bits per 64-bit window, giving the bit offset of the first codeword
  that *starts* in that window; zero-padded to a multiple of 512 windows and
  bit-packed. Requires max code length ≤ 32 bits.
- `output_positions`: uint32 (stored as a uint8 view), the index of the first
  element beginning in each 4096-byte chunk, plus a trailing total element count.

The kernel pays for the compactness of `gaps` with a **two-pass decode**: a
counting pass, a block-wide prefix sum in shared memory to establish each
thread's output position, then the real decode pass.

---

## Scheme `idx8` — designed and measured in `bf16-exponent-compression`

Not implemented here. Specified so that implementing it later is additive.

**Credit and source.** This is the design from the sibling project
`bf16-exponent-compression` (`kernel_idx8.py`), where it was implemented,
benchmarked on three GPUs and found to cost nothing in decode time. This section
restates it; it does not originate it.

Instead of one absolute `uint32` offset per block:

- one `uint32` per **superblock** of 32 blocks (one warp) → 0.125 B/block
- one `uint8` per block holding that block's **length in bits, minus a global
  minimum** → 1.0 B/block

**1.125 B/block instead of 4.** A block's absolute offset is recovered as the
superblock base plus an exclusive prefix sum of the preceding lanes' lengths
within the warp — five `__shfl_up_sync` steps, measured as unmeasurably cheap.

**The representability condition.** The `uint8` holds a length *delta*, so it
works only while the spread of block lengths is under 256. Measured at BLOCK=64
on Qwen3-0.6B: lengths ran 130–273 bits, a range of 143. The builder **must
check this and fail** if it does not hold — this is the canonical case for rule 1
above.

### What it would actually buy in DF11

The honest pairing is against `gaps`, not `output_positions`, because idx8 is an
**input-side** index — it tells a thread where its block starts in the bitstream,
which is what `gaps` does. `output_positions` is an output-side index and is not
what idx8 replaces.

Using the sibling project's own figures alongside ours:

| Scheme | bits/weight |
|---|---|
| uint32 per block (BLOCK=64) | 0.50 |
| **DF11 `gaps`** | **0.208** (measured here; their estimate 0.21) |
| idx8 at BLOCK=64 | 0.14 |
| idx8 at BLOCK=128 | 0.07 |

So replacing `gaps` with idx8 would save **0.07–0.14 bits/weight**, which against
a measured total of 10.873 bits/weight is **0.6% to 1.3% of output size**.

That is real but modest, and it is not free: DF11's `gaps` buys its compactness
with a two-pass decode, while idx8 is single-pass with a warp prefix-sum on the
input side. **Which decodes faster has never been measured** — the sibling
project states this explicitly as unmeasured, and nothing here changes that.

### What adopting it would require

Not a flag on the DF11 path. An `idx8` file cannot be read by the official
kernel, so it is a different format sharing an encoder, and it needs:

- its own decoder (the sibling project has a working CUDA implementation);
- its own verification chain and fixtures, since there is no official output to be byte-identical to;
- a name that does not claim to be DF11, and `is_df11_compatible() == false`;
- a decision on block geometry, since the bits/weight figures above depend on BLOCK.

The encoder machinery — field split, histogram, codebook, LUTs, bit writer,
chunked parallel encode — is shared and needs no change. That is the whole point
of the seam.

---

## Phase 1 obligations

Step 1.7 must:

1. define the `IndexBuilder` interface above;
2. implement `df11` behind it, with the single-threaded reference encoder producing byte-identical `gaps` and `output_positions` through the interface rather than inline;
3. add a test that the interface is actually load-bearing — a stub builder returning `Unrepresentable` must cause a clean typed failure, not a panic or a silent fallback;
4. **not** implement `idx8`. It is specified, not scheduled.
