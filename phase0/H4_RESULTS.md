# H4 — Can we reproduce dahuffman's codebook exactly, from the histogram alone, without the library?

Phase 0, step 0.5. Environment: `phase0/env/bin/python` (dahuffman 0.4.2, numpy 2.5.3).
No torch import anywhere in this work (peak RSS measured at **34.9 MB**, well under the
3 GB box's budget).

## Files produced

- `phase0/h4_codec_spec.py` — self-contained reimplementation (numpy only, no
  dahuffman import) of: the Huffman tree build, `get_32bit_codec`, and
  `get_luts`. This is the executable spec a Rust port should be checked
  against; its module docstring states the tie-break rule in full, in
  Rust-portable terms.
- `phase0/test_h4.py` — 682 histograms compared against the real dahuffman
  library and line-by-line (torch-stripped) transcriptions of
  `dfloat11_utils.get_32bit_codec` / `get_luts`.
- This file.

## Method

`test_h4.py` builds 682 frequency histograms across 12 categories: uniform,
heavily-skewed, near-degenerate, many-exact-ties, single-symbol, two-symbol,
Fibonacci chains engineered to land max code length in the 9–16 / 17–24 /
25–32 bit ranges (forcing 2/3/4 LUT levels), Fibonacci chains ≥34 levels
deep (forcing the `get_32bit_codec` compression loop), Fibonacci chains with
each frequency level duplicated across several symbols (forcing genuine
`argpartition`-boundary ties *while* the 32-bit loop is active), 135
"realistic" histograms shaped like real bf16/fp16 model-exponent
distributions (Gaussian bulk of huge counts + long thin outlier tail down to
counts of 1–3, at param counts from 1M to 7B and three peak widths), and 400
randomly generated histograms (fixed seed, 4 distribution shapes, symbol
counts 2–256).

For every histogram: build the codebook (`h4_codec_spec.build_huffman_table`
vs `dahuffman.HuffmanCodec.from_frequencies(...).get_code_table()`), the
32-bit-limited codebook (`h4_codec_spec.get_32bit_codec` vs a torch-free
transcription of `dfloat11_utils.get_32bit_codec` built on real dahuffman),
and the LUTs (`h4_codec_spec.get_luts` vs a torch-free transcription of
`dfloat11_utils.get_luts`). Compared per-symbol `(bits, value)` and full LUT
arrays, element for element.

## The tie-breaking rule (precise statement)

dahuffman's heap items are `(total_frequency, [(symbol, (bits, value)), ...])`.
Because every live symbol belongs to exactly one heap node, two distinct
nodes' first list-elements are never equal, so Python's tuple/list
comparison always resolves a frequency tie by comparing the **first
symbol** each node was ever built from (its "representative"), and that
comparison never needs to fall through into the `(bits, value)` part.

Concretely: every heap node has a key `(frequency, representative)`.
`representative` is the symbol itself for a leaf; for a merged node it is
`representative(a)`, where `a` is whichever of its two children sorted
smaller (by this same key) when they were merged — this is a structural
inheritance, **not** "the numerically smallest symbol anywhere in the
subtree" (a child can pass on a non-minimal representative if it won a
*non-tied*, frequency-only comparison against a sibling that happened to
contain a smaller symbol). EOF is injected at frequency 1 if not already a
key, and compares as an unconditional minimum (`EOF < x` is always true,
`x < EOF` is always false, for any real symbol `x`) — it wins every tie it
participates in. On a pop of the two smallest nodes `a` (smaller) and `b`
(larger): every leaf under `a` gets bit `0` appended, every leaf under `b`
gets bit `1` appended; the merged node's representative is `a`'s. This KEY
order is total (representatives never collide among live nodes) and
reproduces dahuffman's `heapq` ordering exactly. Full detail, including a
worked argument for why the `(bits, value)` tail is never inspected, is in
`h4_codec_spec.py`'s module docstring.

## Results

| Component | Histograms | Exact matches | Notes |
|---|---|---|---|
| Huffman codebook (bits+value per symbol, incl. EOF) | 682 | **682 / 682 (100%)** | zero mismatches over every category, including 77 argpartition-boundary-tie iterations and hundreds of frequency ties in the tree build itself |
| `get_32bit_codec` (compressed table + frequencies) | 682 | **682 / 682 (100%)** | 18 histograms actually entered the compression loop (max_len > 32 uncompressed) |
| `get_luts` (hierarchical LUT arrays) | 682 | **661 / 682 (96.9%)**; **661 / 661 (100%) of the cases where the original algorithm doesn't crash** | the 21 "misses" are all the *same* root cause — see below, not array-content divergences |

Overall naive "everything matched" rate: 661/682 = **96.93%**. The
qualifier matters: every one of the 21 non-matches is a crash in the
literal original algorithm, not a case where our output differs in content
from a value the original successfully produced. We never observed our
implementation and a *successfully-completed* run of the original disagree
on LUT contents.

### The `get_luts` crash (dfloat11 bug, not a divergence we introduced)

`get_luts`'s inner loop does `if i in bytes_dict: curr_val = bytes_dict[i]`
then unconditionally `luts[pi, i] = curr_val`, and **never initializes
`curr_val`** before the very first read. This is intentional forward-fill
(it's how a code shorter than a full byte gets broadcast across every byte
value sharing its prefix) but it silently assumes index 0 of every LUT row
is always explicitly populated before it's first read. It is not: `EOF`'s
own codeword is deliberately excluded from the LUT (`isinstance(key, int)`
skips it), so whenever EOF's *assigned code value is exactly 0* (i.e. EOF
was on the "smaller" side of every single merge on its path to the root),
address 0 of the top-level row has no real-symbol owner, and the literal
original Python raises `UnboundLocalError`.

We verified this precisely: in **all 21** failing histograms, EOF's final
`(bits, value)` had `value == 0` (e.g. `two_symbol(seed=5): EOF=(bits=2,
val=0)`, `random(seed=61, poisson_like, n=128): EOF=(bits=8, val=0)`). This
occurs for small alphabets (single/two-symbol), near-degenerate
distributions, and "staircase"-like tied/skewed random histograms — cases
where EOF's frequency-1 leaf rides as the globally-smallest node all the
way to the root.

**This is a real but narrow edge case for actual model data.** We built
135 histograms shaped like real bf16/fp16 weight tensors (Gaussian bulk of
counts up to 7B, plus a long thin tail with realistic small outlier counts
down to 1–3, three peak widths, 15 seeds): **zero** of these hit the crash
condition — EOF always ended up with a large `value` (bits 32–33, deep in
the tree but never landing on the all-zero code), because a real model's
exponent histogram has enough independent "levels" of mass that EOF's
branch eventually gets promoted to the "larger" side of some merge instead
of staying smallest all the way up. `h4_codec_spec.get_luts` defaults
`curr_val = 0` before the first row so it never crashes; empirically this
default is only ever exercised in exactly the 21 crash-triggering
degenerate cases, and produces a well-defined (LUT slot mapped to symbol 0)
rather than undefined result there.

### Argpartition tie characterization (the most important number here)

`get_32bit_codec` selects "the `min_k` smallest original frequencies" via
`np.argpartition(freq, min_k)[:min_k]`. NumPy's public contract only
guarantees the selected elements are ≤ every unselected element (a valid
partial order) — it makes **no guarantee about which specific elements are
chosen when several share the exact boundary value**; that choice falls
out of the internal partial-sort algorithm's pivoting, which is not part
of NumPy's documented API and is not something a from-scratch Rust
reimplementation can be expected to reproduce without literally porting
NumPy's selection algorithm.

Measured over all 682 test histograms:

- Histograms that ever entered the 32-bit compression loop: **18 / 682**
  (2.6%) overall, but **5 / 135 (3.7%) of the realistic model-exponent-shaped
  histograms** too — specifically the largest ones tested (7 billion
  params, mid-width Gaussian bulk, `sigma=6`); their uncompressed max code
  length was 33 bits, one over the cap. So this is not purely a synthetic
  corner case: a large real model's exponent histogram can plausibly reach
  the 32-bit limiter in practice.
- Total `argpartition` calls (loop iterations) across those 18 histograms:
  **177**.
- Iterations with a **genuine boundary tie** (more candidate symbols at the
  cut-point frequency value than slots available, so the selection is
  actually ambiguous): **77 / 177 (43.5%)**. Ties are common whenever the
  loop runs at all — not a rare corner case within this regime.
- For every one of those 77 ambiguous iterations we exhaustively
  constructed every valid single-swap alternative selection (swap one
  argpartition-selected boundary-value symbol for one not-selected
  boundary-value symbol, holding `min_k` fixed) and rebuilt the compressed
  Huffman table: **2,513 alternative selections probed**.
  - **84 / 2,513 (3.3%) of individual alternative selections produced a
    different final per-symbol code-length assignment** than the one
    argpartition's actual choice produced.
  - At the iteration level, **7 / 77 (9.1%) of ambiguous iterations had at
    least one alternative tie-resolution that changed the final table.**

**The 3.3% / 9.1% headline numbers are misleading on their own — splitting
by the boundary frequency value tells the real story, and it is clean and
decisive:**

| boundary frequency value | probes | changed | rate |
|---|---|---|---|
| `boundary_value == 1` | 2,429 | 0 | **0.00%** |
| `boundary_value > 1`  | 84    | 84 | **100.00%** |

This is not a coincidence, it follows from what the loop actually does:
`compressed_frequencies[k] = 1` for each selected symbol `k`. When the
boundary value is already 1, this is a **no-op** — the frequency doesn't
change regardless of which of the tied freq-1 symbols argpartition
happened to pick, so the rebuilt Huffman tree is byte-identical no matter
the choice. This is true by construction, not just empirically, and it is
why almost all of the 77 ambiguous iterations we found (70 of them) show
zero sensitivity: most low-`min_k` iterations of a long compression loop
are still selecting among symbols that already have frequency 1 (real
model histograms have lots of singleton/rare exponent values). But the
moment the boundary value rises above 1, this protection disappears:
**every one of the 7 ambiguous iterations where boundary_value > 1, and
every one of the 84 individual alternative selections probed within them,
changed the resulting codebook.** This includes the 5 realistic 7B-param
histograms: in each, the loop's *final* iteration (the one that actually
brings max_len down to ≤ 32 and terminates the loop) has boundary_value=2
with 13–21 tied candidate symbols and only 1 slot — and in all 5 cases,
100% of the alternative single-symbol choices we probed produced a
different final codebook than argpartition's actual pick. Concrete
example: histogram `fib_multiplicity(n_levels=40, mult=4)` — swapping
symbol 11 for symbol 8 in a boundary-value=2 tie changed their code
lengths from `{11: 32, 8: 31}` to `{11: 31, 8: 32}`.

**Verdict on this specific question: ties absolutely do NOT wash out
whenever the boundary value is greater than 1, and this is exactly the
situation the loop is in on its *final*, table-determining iteration for
every realistic large-histogram case we constructed that reached the
32-bit cap.** A Rust port that reimplements `get_32bit_codec`'s selection
with "any valid k-smallest subset" (a stable sort, a different
partial-selection algorithm, etc.) is not merely at theoretical risk — it
will reliably diverge from the official DF11 output on real large-model
histograms that trigger the 32-bit-limit loop, unless it reproduces
NumPy's exact `argpartition` tie order for `numpy==2.5.3`. The
`boundary_value == 1` case, by contrast, can be implemented with *any*
selection rule and is provably safe.

## VERDICT

**H4: confirmed, with two precisely-characterized caveats.**

1. The core dahuffman Huffman-tree construction (the actual "codebook")
   is reproduced **exactly** from the histogram alone, with no
   library dependency, across 682 varied histograms including hundreds of
   deliberately engineered exact-frequency ties, single/two-symbol edge
   cases, and uniform/degenerate distributions: **100% match, 0
   mismatches.** The tie-breaking rule is fully and precisely characterized
   above and in `h4_codec_spec.py`'s docstring, in a form directly
   portable to Rust (a `(frequency, representative)` key with EOF as an
   unconditional minimum).
2. `get_32bit_codec`'s fallback path additionally depends on
   `np.argpartition`'s internal (undocumented, NumPy-version- and
   algorithm-dependent) tie resolution. We've shown this empirically and
   decisively, not just as a theoretical risk: whenever the tied boundary
   frequency value is exactly 1, the tie is provably inert (setting an
   already-frequency-1 symbol to frequency 1 is a no-op: 0/2,429 probed
   alternatives changed anything); but whenever the boundary value is
   greater than 1, the choice is essentially always load-bearing (84/84
   probed alternatives changed the resulting codebook, 7/7 ambiguous
   `boundary_value > 1` iterations were sensitive). This is not confined to
   engineered corner cases: 5 of our 135 realistic 7B-parameter
   model-exponent-shaped histograms triggered the 32-bit loop, and in
   every one of them the loop's *final, table-determining* iteration had
   `boundary_value = 2` with 13–21 tied candidates and only 1 slot —
   exactly the regime where the choice matters 100% of the time we tested
   it. A byte-identical Rust port therefore MUST reproduce NumPy's exact
   `argpartition` selection among ties (pinned to numpy==2.5.3) for any
   model whose exponent histogram reaches the 32-bit cap; it cannot rely on
   "any valid k-smallest subset."
3. `get_luts` has a genuine, pre-existing implementation bug in the
   official code (`curr_val` read-before-assignment) that crashes on
   certain small-alphabet/degenerate histograms; it does not affect any
   realistic model-shaped histogram we tested. Our implementation resolves
   it with a documented default (`curr_val = 0` at row start) rather than
   attempting to replicate a Python crash; every case where the original
   *doesn't* crash, our output matches it exactly (661/661).
