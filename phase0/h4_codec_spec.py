"""H4 codec spec: from-scratch reimplementation of dahuffman 0.4.2's Huffman
codebook construction, plus the dfloat11-specific `get_32bit_codec` and
`get_luts` post-processing steps, WITHOUT importing dahuffman or torch.

This module is the executable spec that a Rust port of df11pack's Huffman
stage should be checked against. Every function below is a line-by-line
derivation of the corresponding official function, read from:

  phase0/env/lib/python3.12/site-packages/dahuffman/huffmancodec.py
    (HuffmanCodec.from_frequencies, dahuffman==0.4.2)
  phase0/env/lib/python3.12/site-packages/dfloat11/dfloat11_utils.py
    (get_32bit_codec, get_luts)

======================================================================
THE TIE-BREAKING RULE (read this before porting to Rust)
======================================================================

dahuffman builds the tree with a textbook min-heap merge, EXCEPT that each
heap item is not just a frequency -- it is a tuple::

    (total_frequency, [(symbol, (bits, value)), ...])

i.e. the frequency *and the whole list of (symbol, code-so-far) pairs
carried by that node*. Python's heapq compares these tuples the normal
tuple/list way: first by `total_frequency` (int); if that ties, it falls
through to comparing the *lists* element-by-element, which means comparing
their first elements, which are `(symbol, (bits, value))` tuples, which in
turn compare `symbol` first.

Two facts collapse this into something simple and exact:

1. At every point in time, every live symbol belongs to exactly one heap
   node, so no two *distinct* live nodes ever have the same first-list
   symbol. That means the `(bits, value)` part of the first tuple is NEVER
   inspected during a comparison -- the comparison always short-circuits on
   the symbol itself. (Two nodes' first elements can be `==` only if they
   are the same node.)

2. A node's "first symbol" (call it its REPRESENTATIVE) is propagated
   purely structurally: when two nodes `a` (popped first == smaller) and
   `b` (popped second == larger) are merged, the merged node's list is
   `a.list + b.list`, so the merged representative is simply
   `a.representative`. This is recursive: the representative of any node is
   the representative of whichever of its two children sorted smaller
   at the moment they were merged. IMPORTANT: this is *not* the same as
   "the numerically smallest symbol in the subtree" -- if `a` was chosen
   over `b` purely because `freq(a) < freq(b)` (not a tie), `a`'s
   representative is inherited regardless of whether `b` secretly contains
   a smaller-numbered symbol. Do not try to shortcut this with "track the
   min symbol per node"; you must literally propagate the representative
   through the merge order.

Consequence -- the exact, portable rule:

    Every heap node has a pair KEY = (frequency: u64, representative: Sym)
    where Sym is either a real symbol (u8/u16/...) or the special EOF
    symbol. Nodes are ordered by KEY using:
      - frequency, ascending, as the primary key;
      - on a frequency tie, representative, ascending, as the secondary
        key, where EOF compares as strictly less than every real symbol
        (in every comparison, EOF < x is true and x < EOF is false --
        EOF acts as an unconditional -infinity, NOT as "the numeric value
        -1" or similar; a real symbol never gets to "beat" EOF, and EOF
        never gets to "lose" a comparison).
    This KEY order is a legal total order (no two live nodes ever share a
    KEY, because representatives never repeat among live nodes) and it
    reproduces dahuffman's heapq ordering bit-for-bit.

    Build: push one leaf node (freq=f, representative=s) per symbol s with
    frequency f; if EOF is not already an explicit key in the frequency
    table, push (freq=1, representative=EOF). Repeatedly pop the two
    smallest nodes by KEY, call them `a` (smaller/first) and `b`
    (larger/second; if `a`'s KEY == `b`'s KEY that's impossible per the
    argument above, so ties are always fully resolved by this KEY).
    Merge: new_freq = freq(a) + freq(b); new_representative =
    representative(a); every leaf that was in `a` gets bit '0' appended
    to its code (bits+1, value unchanged); every leaf that was in `b`
    gets bit '1' appended (bits+1, value + (1 << bits)). Push the merged
    node. Stop when one node remains -- its accumulated
    (symbol -> (bits, value)) map, plus EOF's own entry, is the code
    table.

    Practical note: when there are no frequency ties at all in the input
    histogram, this whole KEY machinery never matters and any standard
    heap-based Huffman build agrees. It only matters when two or more
    *nodes* (leaves or internal) reach equal cumulative frequency at the
    same time, which -- because EOF is always injected at frequency 1 --
    is essentially guaranteed to happen at least once for any histogram
    that contains a symbol with frequency 1 (ties with EOF), and routinely
    happens for skewed/near-degenerate histograms where many raw symbol
    counts collide. See H4_RESULTS.md for measured tie frequency.

======================================================================
get_32bit_codec
======================================================================
Mirrors dfloat11_utils.get_32bit_codec exactly:
  - Build the full (uncapped) table once.
  - If max code length <= 32, return it unchanged.
  - Otherwise, starting at min_k = 2 and incrementing by 1 each iteration,
    take `np.argpartition(freq, min_k)[:min_k]` on the ORIGINAL frequency
    array (freq array order == dict insertion order of the frequency
    table passed in -- this matters, see below), force those min_k
    symbols' frequency to 1 in a FRESH copy of the ORIGINAL frequency
    table (not the previously-compressed one -- each iteration restarts
    from the original), rebuild the full Huffman table, and check max
    length again. Repeat until max length <= 32.
  - `np.argpartition`'s selection among frequency values that are exactly
    equal to the boundary ("the min_k-th smallest value") is NOT specified
    by NumPy -- see H4_RESULTS.md item 4 for a direct empirical
    characterisation of how often that ambiguity is real, and whether it
    changes the final table.

======================================================================
get_luts
======================================================================
Mirrors dfloat11_utils.get_luts exactly, MINUS the final
`torch.from_numpy(...)` wrapper (we return the plain numpy array; the
torch conversion is a no-op reinterpretation of the same bytes and is not
part of the codec logic). One non-obvious detail preserved on purpose:
in the original, the loop variable `curr_val` used to forward-fill gaps
in each LUT row is NEVER reset, neither between rows nor even before the
very first row -- it relies on Python's normal scoping (the name simply
has to already be bound by the time it's first read). This forward-fill
is *not* a bug: for symbols whose code is shorter than a full byte
boundary at this LUT level, the code sets exactly ONE dict entry (the
code's bits followed by zero-padding) and relies on the forward-fill to
replicate that entry across every subsequent byte value that shares the
same prefix, up to the next explicit entry. We initialise `curr_val = 0`
before the first row as a defensive default; empirically (see
H4_RESULTS.md) index 0 of every row is always explicitly populated before
it is ever read, for every histogram tested, so this default is never
actually exercised -- it is a completeness guard, not a divergence from
the original semantics. If that invariant is ever violated the original
Python would raise UnboundLocalError; we do not attempt to replicate a
crash, we just document that this is the fallback behaviour.
"""

from copy import copy

import numpy as np


class _EOFSymbol:
    """Sentinel matching dahuffman's `_EndOfFileSymbol` comparison
    semantics EXACTLY (re-derived from dahuffman/huffmancodec.py, not
    imported): `_EOF` in dahuffman is defined to compare as strictly
    less than every other symbol (its `__lt__` returns True
    unconditionally) and never greater than anything (`__gt__` returns
    False unconditionally). It is a singleton; there is exactly one
    instance, `EOF`, exported from this module.
    """

    def __repr__(self) -> str:
        return "_EOF"

    def __lt__(self, other) -> bool:
        return True

    def __gt__(self, other) -> bool:
        return False

    def __le__(self, other) -> bool:
        return True

    def __ge__(self, other) -> bool:
        return isinstance(other, _EOFSymbol)

    def __eq__(self, other) -> bool:
        return isinstance(other, _EOFSymbol)

    def __hash__(self) -> int:
        return hash(_EOFSymbol)


EOF = _EOFSymbol()


def build_huffman_table(frequencies: dict, eof=EOF) -> dict:
    """Reimplementation of dahuffman.HuffmanCodec.from_frequencies's tree
    build. `frequencies` maps symbol (plain Python int, 0..255 for df11's
    exponent byte) -> positive int frequency. Returns dict symbol -> (bits,
    value), including an entry for `eof`.

    See the module docstring for the exact tie-breaking rule this
    replicates.
    """
    assert len(frequencies) >= 1, "need at least one symbol"

    # heap items: (total_frequency, [(symbol, (bits, value)), ...])
    # list-of-one at the leaves; first element of the list is always the
    # node's "representative" per the module docstring's argument.
    heap = [(f, [(s, (0, 0))]) for s, f in frequencies.items()]
    if eof not in frequencies:
        heap.append((1, [(eof, (0, 0))]))

    # Plain O(n^2) selection-based build (n <= ~257 for df11 exponents,
    # trivial cost) so we don't need to reproduce heapq's C internals --
    # we only need the same COMPARISON KEY and the same pop-two-smallest
    # merge order, which is what heapq itself guarantees.
    import heapq as _heapq

    _heapq.heapify(heap)
    while len(heap) > 1:
        a = _heapq.heappop(heap)
        b = _heapq.heappop(heap)
        merged = (
            a[0] + b[0],
            [(s, (n + 1, v)) for (s, (n, v)) in a[1]]
            + [(s, (n + 1, (1 << n) + v)) for (s, (n, v)) in b[1]],
        )
        _heapq.heappush(heap, merged)

    table = dict(_heapq.heappop(heap)[1])
    return table


def _max_code_len(table: dict) -> int:
    return max(bits for bits, _ in table.values())


def get_32bit_codec(frequencies: dict, eof=EOF):
    """Reimplementation of dfloat11_utils.get_32bit_codec.

    Returns (compressed_table, compressed_frequencies, iterations) where
    `iterations` is a list of diagnostic dicts, one per loop iteration
    actually executed (empty if the uncapped table already satisfies
    max_len <= 32), each with:
        min_k            -- the min_k value used this iteration (the
                             "min_k - 1" printed by the original)
        selected_indices -- the np.argpartition(...)[: min_k] result
        selected_keys    -- the symbols corresponding to those indices
        boundary_value   -- freq value of the largest selected element
        boundary_total   -- how many symbols (anywhere in freq, not just
                             selected) share `boundary_value`
        boundary_selected-- how many of the selected indices have
                             freq == boundary_value
        max_len_after    -- max code length after this iteration
    `boundary_total > boundary_selected` means argpartition faced a real
    tie at the cut point (more equal-valued candidates than slots) and had
    to make an implementation-defined choice among them.
    """
    table = build_huffman_table(frequencies, eof=eof)
    max_len = _max_code_len(table)

    compressed_table = table
    compressed_frequencies = frequencies

    iterations = []

    min_k = 2
    freq = np.array(list(frequencies.values()))
    keys = list(frequencies.keys())  # preserve insertion order, plain ints

    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        this_k = min_k
        min_k += 1

        selected_vals = freq[min_indices]
        boundary_value = int(selected_vals.max())
        boundary_total = int((freq == boundary_value).sum())
        boundary_selected = int((selected_vals == boundary_value).sum())

        min_keys = [keys[i] for i in min_indices]

        compressed_frequencies = copy(frequencies)
        for k in min_keys:
            compressed_frequencies[k] = 1
        compressed_table = build_huffman_table(compressed_frequencies, eof=eof)
        max_len = _max_code_len(compressed_table)

        iterations.append(
            dict(
                min_k=this_k,
                selected_indices=min_indices.copy(),
                selected_keys=list(min_keys),
                boundary_value=boundary_value,
                boundary_total=boundary_total,
                boundary_selected=boundary_selected,
                max_len_after=max_len,
            )
        )

    return compressed_table, compressed_frequencies, iterations


def get_luts(table: dict) -> np.ndarray:
    """Reimplementation of dfloat11_utils.get_luts (minus the final
    torch.from_numpy wrap -- see module docstring). `table` maps symbol
    (int; the EOF entry is skipped, exactly as the original does via
    `isinstance(key, int)`) -> (bits, value). Returns a numpy uint8 array
    of shape (num_levels + 1, 256): rows 0..num_levels-1 are the
    hierarchical LUT levels, the final row holds per-symbol code lengths
    at their own symbol index (0 elsewhere).
    """
    prefixes = [""]

    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)
    curr_val = 0  # see module docstring: defensive default, never
    # actually read in practice -- index 0 of every row is always
    # populated before this default would be needed (verified in
    # test_h4.py / H4_RESULTS.md).

    for pi, p in enumerate(prefixes):
        bytes_dict = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, "0")

                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8):].ljust(8, "0"), 2)
                        dict_value = key
                    else:
                        dict_key = int(bin_val[(pl * 8):(pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[: (pl * 8 + 8)])

                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                    else:
                        bytes_dict[dict_key] = dict_value

        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits

    return np.concatenate((luts, lens), axis=0)
