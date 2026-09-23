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

**Implemented (2026-09-23), strict:** `df11pack compress --index idx8
[--idx8-block 64]`. Per unit it writes `idx8_lengths` (u8 per block),
`idx8_superblocks` (u32 per 32 blocks) and `idx8_meta` (i64
`[block, minlen, total_bits]`) in place of `gaps` and `output_positions`; `luts`,
`encoded_exponent`, `sign_mantissa` and `split_positions` are byte-identical to
DF11's. Files are stamped `df11pack_index = "idx8"`; `config.json` carries
`df11pack_idx8_config`, never `dfloat11_config`. Safe mode decodes every block from
its indexed start and requires it to end exactly where the index says. The final
partial block does not set the range (the kernel never reads its length).

**Measured on real Qwen3-0.6B layers, plain idx8 is usually not representable.**
At block 64, block lengths spread 237–543 bits across layers (range must be
≤ 255): 0 to 130 of 245,760 blocks overflow per layer — at most 0.05%, where rare
exponents with 20–25-bit codes cluster. Only layer 1 of those sampled fits. The
sibling project's 130–273 came from a slice without those outliers.

**Decided: an escape table** (the user's choice). The u8 code 255 means "this
block's exact length is in the side table": `idx8_escape_blocks` (sorted u32 block
indices) and `idx8_escape_lengths` (u32). The warp prefix sum is unchanged; a lane
whose code is 255 does one lookup. Every unit is now representable (up to u32 bit
offsets). **The sibling project's kernel must learn the same branch** before it can
read these files. On the real tier-0 layer 0: 14 escapes, index 276.6 KB against
DF11's 414.1 KB (`gaps` + `output_positions`) — 0.07 bits/weight, 0.64% of the
unit, within the predicted 0.6–1.3%.

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
