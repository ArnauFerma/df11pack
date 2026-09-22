# Compatibility contract

What df11pack promises about its output, where it deliberately does not, and how
to tell which you have.

Read this before changing anything in the LUT or index path.

---

## The default is bug-for-bug, not correct

df11pack's purpose is producing files the **existing** DF11 ecosystem loads
unchanged: the official `DFloat11Model`, the ComfyUI-DFloat11-Extended node, and
the official CUDA kernel. That makes the official compressor's *output* the
specification — including the parts of it that are mistakes.

There is at least one such mistake, found by comparing bytes rather than reading
code (see [`FINDINGS.md`](FINDINGS.md) §0.5). The official `get_luts` fills each
prefix table with a carry-forward loop whose accumulator is function-scoped
rather than loop-scoped:

```python
for i in range(256):
    if i in bytes_dict:
        curr_val = bytes_dict[i]
    luts[pi, i] = curr_val        # curr_val survives across prefix tables
```

So any prefix table after the first that has no key `0` begins filled with the
**previous table's trailing value**. This is not hypothetical. In a real official
output, prefix table 2 ends at value 105 and table 3 (`{128: 96}`) contains:

```
row 3, positions   0..127  ->  105     <-- leaked from row 2
row 3, positions 128..255  ->   96
```

128 bytes of that row describe a different table. A correct implementation, which
zero-filled or reset the accumulator per table, would produce a **better** LUT and
**fail** byte-identity against every real DF11 file in existence.

**Therefore the default mode reproduces this behaviour exactly.** That is not an
oversight to be cleaned up later; it is the contract.

---

## `--luts` modes

| Mode | Flag | Byte-identical to official | Status |
|---|---|---|---|
| **Compat** (default) | `--luts=compat` | **Yes** | Supported |
| **Correct** | `--luts=correct` | **No, by design** | **Experimental — see the gate below** |

### `--luts=compat` — the default

Reproduces the official carry-forward exactly, including leakage across prefix
table boundaries. Output is byte-identical to the official compressor for the
same input and pattern. This is what you want unless you have a specific reason
otherwise, and it is what the golden tests grade against.

### `--luts=correct` — opt-in, and currently unverified

Fills the positions the official code leaves to leaked state with a deterministic
`0x00` instead, so a LUT row describes only its own table.

**The argument that this is safe:** those positions should be unreachable during
decode. The kernel only indexes a prefix table with byte values that are valid
continuations of the prefix that selected it, and positions before a table's
first key are not valid continuations. If that holds, the leaked bytes are never
read and replacing them changes nothing observable.

**Why that argument is not yet good enough to ship unguarded:** it is a claim
about the CUDA kernel's indexing behaviour that we have inferred, not measured.
Nobody has run the official kernel over a `--luts=correct` file and compared the
decoded tensor bit-for-bit against the source. Until that test exists and passes,
this mode can produce a file that decodes *differently* on real hardware, and the
difference would be silent.

So the mode is gated:

- it is **off by default** and cannot be reached implicitly;
- using it prints a warning naming this document;
- files it produces are stamped (see below) so they can never be mistaken for compat output;
- the verification commands know not to expect byte-identity from them;
- **it is not permitted in any released artefact until the kernel test in Phase 5 passes.** That test is a required exit-gate item, not a nice-to-have.

If the kernel test ever *fails* — that is, the leaked bytes turn out to be
reachable — then `--luts=correct` is not "more correct", it is simply wrong, and
it must be removed rather than documented around.

---

## How to tell which mode produced a file

Compat output carries no extra marking: it must stay byte-identical, so nothing
may be added to it.

`--luts=correct` output is stamped in the safetensors `__metadata__` header:

```
df11pack_luts = "correct"
df11pack_version = "<version>"
```

`__metadata__` is a free-form string map that the official loaders ignore, so the
stamp costs no compatibility. The absence of a stamp means compat mode — which is
also true of every file the official compressor ever produced, which is the point.

Never stamp compat output. Never emit correct output unstamped.

---

## The 32-bit limiter: refuse rather than guess

When a unit's longest Huffman code exceeds 32 bits, the official encoder demotes
the least frequent symbols to frequency 1 and rebuilds, repeating until the code
fits. It selects them with `np.argpartition`, **whose ordering among equal values
NumPy does not specify**.

That ambiguity is not academic. Measured across 2,513 probes (FINDINGS 0.5):

- boundary frequency **1**: inert — 0 of 2,429 probes changed anything, because demoting an already-frequency-1 symbol is a no-op;
- boundary frequency **above 1**: decisive — **84 of 84** probes changed the resulting codebook.

How close is this? Real units measured here run 24–27 bits against the limit of
32: tier-1's worst is 26, and a published `Qwen3-4B` shard reaches 27. Code length
grows with unit size, so a 622M-weight embedding unit — the standalone-unit case
that every published DF11 LLM at 8B and above uses — plausibly approaches it. The
limiter is reachable, not theoretical.

**df11pack's rule.** The choice is not always ambiguous, and where it is forced we
are provably identical. So:

| situation | behaviour |
|---|---|
| limiter never fires | normal encode |
| fires, no tie at the boundary | selection is forced; byte-identical |
| fires, tie at boundary frequency 1 | inert; byte-identical |
| **fires, tie at boundary frequency > 1** | **abort** with `AmbiguousLimiterTie` |

The refusal names the measured numbers — `min_k`, the boundary frequency, how
many symbols are tied and how many slots exist — so it is diagnosable rather than
merely obstructive, and it points here.

This is deliberately narrower than the original brief, which allowed a blanket
"different but valid" result wherever `argpartition` was involved. That would have
meant emitting files that silently differ from the official compressor's with no
way to know which. The rule above converts an unbounded silent divergence into a
bounded, detected, loud one: df11pack is either byte-identical or it refuses.

The alternative — reimplementing NumPy's introselect to reproduce its tie order
exactly — was considered and not taken. It would pin the project to one NumPy
version forever, and an upstream change to that selection would break
compatibility silently, which is the failure mode this rule exists to eliminate.

---

## Format divergence more generally

The same reasoning governs any future change to the on-disk format, including
ideas imported from sibling work such as a narrower block index.

A DF11 file is only useful because the official kernel can decode it. Any change
to the bitstream, the index tensors (`gaps`, `output_positions`), or the LUT
layout produces a file that kernel cannot read. That is not a compatibility
*risk*; it is a different format that happens to share an encoder.

So such changes are never a flag on the default path. They require, at minimum:
their own decoder, their own verification chain, their own fixtures, and a name
that does not claim to be DF11. The encoder machinery — field split, histogram,
codebook, bit writer, chunked parallel encode — is shared and reusable; the
ecosystem is not.

To keep that option cheap without paying for it now, the encoder keeps **index
generation behind a seam** (see [`PLAN.md`](PLAN.md) step 1.7): `gaps` and
`output_positions` are produced by a swappable component rather than inline in
the bit writer. Adding an alternative index scheme later is then additive. This
costs nothing today and is expensive to retrofit, which is the only reason it is
specified this early.
