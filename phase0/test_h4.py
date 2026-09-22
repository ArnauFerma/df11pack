"""H4 test: does h4_codec_spec.py reproduce dahuffman 0.4.2's Huffman
codebook, dfloat11's 32-bit-limited codebook, and dfloat11's hierarchical
LUTs, exactly -- across a large, varied set of exponent histograms?

This file IS allowed to import dahuffman (it's the ground truth we test
against). h4_codec_spec.py itself never imports dahuffman.
It must NOT import torch (3 GB RAM box, shared with another job).

Run with:
  phase0/env/bin/python phase0/test_h4.py
"""

import random
from copy import copy as _copy

import numpy as np
import dahuffman

import h4_codec_spec as spec


# ----------------------------------------------------------------------
# Reference implementations: line-by-line transcriptions of
# dfloat11_utils.get_32bit_codec / get_luts, using the REAL dahuffman
# library, with the torch bits stripped out (they are a no-op
# reinterpretation of the same numpy bytes, not part of the codec logic,
# and importing torch is off-limits on this machine).
# ----------------------------------------------------------------------

def ref_table(frequencies):
    """Real dahuffman-built code table for `frequencies`."""
    codec = dahuffman.HuffmanCodec.from_frequencies(dict(frequencies))
    return codec.get_code_table()


def ref_eof_key(table):
    for k in table:
        if not isinstance(k, int):
            return k
    raise AssertionError("no EOF-like key found")


def ref_get_32bit_codec(frequencies):
    """Transcription of dfloat11_utils.get_32bit_codec using real dahuffman."""
    codec = dahuffman.HuffmanCodec.from_frequencies(dict(frequencies))
    table = codec.get_code_table()
    max_len = max(l for _, (l, _) in table.items())

    compressed_codec = codec
    compressed_counter = frequencies

    min_k = 2
    freq = np.array(list(frequencies.values()))
    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = np.array(list(frequencies.keys()))[min_indices]

        compressed_counter = _copy(frequencies)
        for k in min_keys:
            compressed_counter[k] = 1
        compressed_codec = dahuffman.HuffmanCodec.from_frequencies(compressed_counter)
        table = compressed_codec.get_code_table()
        max_len = max(l for _, (l, _) in table.items())

    return compressed_codec, compressed_counter, table


def ref_get_luts(table):
    """Transcription of dfloat11_utils.get_luts, dropping the final
    torch.from_numpy wrap. Deliberately does NOT pre-initialise curr_val,
    to test whether the original's implicit reliance on Python scoping
    ever actually breaks (see H4_RESULTS.md)."""
    prefixes = [""]

    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)

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
                curr_val = bytes_dict[i]  # noqa: F821 (matches original)
            luts[pi, i] = curr_val  # noqa: F821 (matches original)

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits

    return np.concatenate((luts, lens), axis=0)


# ----------------------------------------------------------------------
# Histogram generators
# ----------------------------------------------------------------------

def h_uniform(n=200, val=1000):
    return {i: val for i in range(n)}, "uniform"


def h_heavily_skewed(seed):
    rng = random.Random(seed)
    n = rng.randint(30, 256)
    freqs = {}
    for i in range(n):
        # geometric-ish decay, heavily skewed but bounded (no bignum risk)
        freqs[i] = max(1, int(2_000_000 * (0.55 ** i)) + rng.randint(0, 3))
    return freqs, f"heavily_skewed(seed={seed})"


def h_near_degenerate(seed):
    rng = random.Random(seed)
    n = rng.randint(5, 256)
    freqs = {i: 1 for i in range(n)}
    dominant = rng.randrange(n)
    freqs[dominant] = 10_000_000
    return freqs, f"near_degenerate(seed={seed})"


def h_many_ties(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 256)
    # a handful of distinct frequency values, reused across many symbols
    distinct_vals = rng.sample(range(1, 50), k=rng.randint(2, 8))
    freqs = {i: rng.choice(distinct_vals) for i in range(n)}
    return freqs, f"many_ties(seed={seed})"


def h_single_symbol():
    return {7: 42}, "single_symbol"


def h_two_symbol(seed):
    rng = random.Random(seed)
    a, b = rng.sample(range(256), 2)
    fa, fb = rng.choice([(1, 1), (1, 2), (5, 5), (1, 100), (100, 1)])
    return {a: fa, b: fb}, f"two_symbol(seed={seed})"


def _fib_chain(n):
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]


def h_fib_chain(n, filler=0, label=None):
    """Fibonacci-frequency chain of n symbols (classic Huffman worst case
    for max code depth), optionally with `filler` extra high-frequency
    symbols mixed in for realism."""
    fibs = _fib_chain(n)
    freqs = {i: fibs[i] for i in range(n)}
    rng = random.Random(1000 + n + filler)
    for j in range(filler):
        sym = n + j
        freqs[sym] = rng.randint(10_000, 1_000_000)
    return freqs, (label or f"fib_chain(n={n},filler={filler})")


def find_fib_chain_for_max_len(lo, hi, filler=0):
    """Search n such that the REAL dahuffman max code length for the
    resulting fib-chain histogram falls within [lo, hi] inclusive."""
    for n in range(3, 120):
        freqs, label = h_fib_chain(n, filler=filler)
        table = ref_table(freqs)
        max_len = max(l for _, (l, _) in table.items())
        if lo <= max_len <= hi:
            return freqs, f"{label}[real_max_len={max_len}]"
    raise AssertionError(f"couldn't find chain for range [{lo},{hi}]")


def h_random(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 256)
    mode = rng.choice(["uniform", "poisson_like", "power", "small_ints"])
    freqs = {}
    symbols = rng.sample(range(256), n)
    if mode == "uniform":
        for s in symbols:
            freqs[s] = rng.randint(1, 100000)
    elif mode == "poisson_like":
        lam = rng.randint(1, 500)
        for s in symbols:
            # crude poisson-ish via sum of uniforms, always >=1
            freqs[s] = max(1, sum(rng.randint(0, 2) for _ in range(lam)))
    elif mode == "power":
        base = rng.uniform(1.2, 3.0)
        for i, s in enumerate(symbols):
            freqs[s] = max(1, int(base ** (n - i)) % 5_000_000 + 1)
    else:  # small_ints -> forces many ties
        for s in symbols:
            freqs[s] = rng.randint(1, 5)
    return freqs, f"random(seed={seed},mode={mode},n={n})"


# ----------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------

class Result:
    def __init__(self):
        self.total = 0
        self.table_mismatches = []
        self.codec32_mismatches = []
        self.lut_mismatches = []
        self.tie_iterations = 0
        self.tie_iterations_with_ambiguity = 0
        self.tie_sensitivity_changed = 0
        self.tie_sensitivity_tested = 0
        self.eof_row0_always_set_before_read = True
        self.luts_triggered = 0  # histograms where the 32-bit loop ran
        self.max_lut_levels_seen = 0
        self.categories = {}

    def note_category(self, label, ok):
        cat = label.split("(")[0]
        c = self.categories.setdefault(cat, [0, 0])
        c[0] += 1
        c[1] += 1 if ok else 0


def compare_tables(our, ref, label, res):
    ok = True
    eof_key = ref_eof_key(ref)
    if our.get(spec.EOF) != ref[eof_key]:
        ok = False
        res.table_mismatches.append((label, "EOF", our.get(spec.EOF), ref[eof_key]))
    for sym, (bits, val) in ref.items():
        if not isinstance(sym, int):
            continue
        our_entry = our.get(sym)
        if our_entry != (bits, val):
            ok = False
            res.table_mismatches.append((label, sym, our_entry, (bits, val)))
    if len(our) != len(ref):
        ok = False
        res.table_mismatches.append((label, "SIZE", len(our), len(ref)))
    return ok


def compare_32bit(freqs, label, res):
    our_table, our_freqs, iterations = spec.get_32bit_codec(dict(freqs))
    ref_codec, ref_freqs, ref_tbl = ref_get_32bit_codec(dict(freqs))

    if iterations:
        res.luts_triggered += 1

    ok = compare_tables(our_table, ref_tbl, label + "[32bit]", res)

    # frequency dict equality (order-insensitive is fine here, values must match)
    if dict(our_freqs) != dict(ref_freqs):
        ok = False
        res.codec32_mismatches.append((label, "FREQS", dict(our_freqs), dict(ref_freqs)))

    # tie bookkeeping
    for it in iterations:
        res.tie_iterations += 1
        if it["boundary_total"] > it["boundary_selected"]:
            res.tie_iterations_with_ambiguity += 1
            _probe_tie_sensitivity(freqs, it, res)

    if not ok:
        res.codec32_mismatches.append((label, "TABLE_MISMATCH", None, None))
    return ok, our_table


def _probe_tie_sensitivity(freqs, iteration, res):
    """For an argpartition call that had a real tie at the boundary
    (more candidates at boundary_value than slots), construct an
    ALTERNATIVE valid min_k-selection that swaps out one of the chosen
    boundary-value indices for a different not-chosen boundary-value
    index, rebuild the compressed table with build_huffman_table, and
    check whether the resulting per-symbol code lengths differ from the
    ones argpartition's actual choice produced.
    """
    keys = list(freqs.keys())
    freq_arr = np.array(list(freqs.values()))
    boundary_value = iteration["boundary_value"]
    selected = set(int(i) for i in iteration["selected_indices"])

    all_boundary_idx = set(int(i) for i in np.where(freq_arr == boundary_value)[0])
    not_selected_boundary = all_boundary_idx - selected
    selected_boundary = [i for i in selected if freq_arr[i] == boundary_value]

    if not not_selected_boundary or not selected_boundary:
        return  # no alternative choice actually exists

    res.tie_sensitivity_tested += 1

    swap_out = selected_boundary[0]
    swap_in = next(iter(not_selected_boundary))
    alt_selected = (selected - {swap_out}) | {swap_in}

    orig_keys = [keys[i] for i in selected]
    alt_keys = [keys[i] for i in alt_selected]

    orig_compressed = _copy(freqs)
    for k in orig_keys:
        orig_compressed[k] = 1
    alt_compressed = _copy(freqs)
    for k in alt_keys:
        alt_compressed[k] = 1

    orig_tbl = spec.build_huffman_table(orig_compressed)
    alt_tbl = spec.build_huffman_table(alt_compressed)

    orig_lengths = {s: b for s, (b, v) in orig_tbl.items() if isinstance(s, int)}
    alt_lengths = {s: b for s, (b, v) in alt_tbl.items() if isinstance(s, int)}

    if orig_lengths != alt_lengths:
        res.tie_sensitivity_changed += 1


def compare_luts(our_table, label, res):
    our_luts = spec.get_luts(our_table)
    try:
        ref_luts = ref_get_luts(our_table)
    except UnboundLocalError:
        res.eof_row0_always_set_before_read = False
        res.lut_mismatches.append((label, "REF_UNBOUND_LOCAL_ERROR", None, None))
        return False

    res.max_lut_levels_seen = max(res.max_lut_levels_seen, our_luts.shape[0] - 1)

    if our_luts.shape != ref_luts.shape or not np.array_equal(our_luts, ref_luts):
        res.lut_mismatches.append(
            (label, "ARRAY_DIFF", our_luts.shape, ref_luts.shape)
        )
        return False
    return True


def run_case(freqs, label, res):
    res.total += 1
    ok_all = True

    our_tbl = spec.build_huffman_table(dict(freqs))
    ref_tbl = ref_table(freqs)
    ok = compare_tables(our_tbl, ref_tbl, label, res)
    ok_all &= ok

    ok32, tbl32 = compare_32bit(freqs, label, res)
    ok_all &= ok32

    ok_lut = compare_luts(tbl32, label, res)
    ok_all &= ok_lut

    res.note_category(label, ok_all)
    return ok_all


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    res = Result()
    cases = []

    cases.append((*h_uniform(200, 1000),))
    cases.append((*h_uniform(2, 5),))
    cases.append((*h_uniform(256, 7),))
    cases.append(h_single_symbol())

    for seed in range(30):
        cases.append(h_heavily_skewed(seed))
    for seed in range(30):
        cases.append(h_near_degenerate(seed))
    for seed in range(40):
        cases.append(h_many_ties(seed))
    for seed in range(20):
        cases.append(h_two_symbol(seed))

    # LUT-level-forcing chains: 2, 3, 4 levels (max_len ranges 9-16/17-24/25-32)
    cases.append(find_fib_chain_for_max_len(9, 16))
    cases.append(find_fib_chain_for_max_len(9, 16, filler=50))
    cases.append(find_fib_chain_for_max_len(17, 24))
    cases.append(find_fib_chain_for_max_len(17, 24, filler=50))
    cases.append(find_fib_chain_for_max_len(25, 32))
    cases.append(find_fib_chain_for_max_len(25, 32, filler=20))

    # 32-bit-limit trigger (max_len > 32 before compression kicks in)
    for n in (34, 36, 38, 40, 45, 50):
        cases.append(h_fib_chain(n, filler=0, label=f"trigger32(n={n})"))
        cases.append(h_fib_chain(n, filler=30, label=f"trigger32_filler(n={n})"))

    for seed in range(400):
        cases.append(h_random(seed))

    fail_labels = []
    for freqs, label in cases:
        try:
            ok = run_case(freqs, label, res)
        except Exception as e:
            ok = False
            res.total += 1
            fail_labels.append((label, f"EXCEPTION: {e!r}"))
            continue
        if not ok:
            fail_labels.append((label, "MISMATCH"))

    print("=" * 70)
    print(f"Total histograms tested : {res.total}")
    print(f"Table mismatches        : {len(res.table_mismatches)}")
    print(f"32bit-codec mismatches  : {len(res.codec32_mismatches)}")
    print(f"LUT mismatches          : {len(res.lut_mismatches)}")
    print(f"Overall FAILING cases   : {len(fail_labels)}")
    print(f"Exact match rate        : {(res.total - len(fail_labels)) / res.total * 100:.4f}%")
    print()
    print(f"Histograms triggering the 32-bit-limit loop : {res.luts_triggered}")
    print(f"argpartition calls total (loop iterations)  : {res.tie_iterations}")
    print(f"  ...with a genuine boundary tie            : {res.tie_iterations_with_ambiguity}")
    print(f"  tie-sensitivity probes actually runnable  : {res.tie_sensitivity_tested}")
    print(f"  probes where final lengths CHANGED        : {res.tie_sensitivity_changed}")
    print(f"Max LUT levels observed (excluding len row) : {res.max_lut_levels_seen}")
    print(f"EOF/row0-always-set-before-read invariant held: {res.eof_row0_always_set_before_read}")
    print()
    print("Per-category (category: passed/total):")
    for cat, (tot, passed) in sorted(res.categories.items()):
        print(f"  {cat:20s} {passed}/{tot}")

    if fail_labels:
        print()
        print("FAILURES (first 30):")
        for label, why in fail_labels[:30]:
            print(f"  {label}: {why}")
        if res.table_mismatches:
            print("Sample table mismatches (first 10):")
            for m in res.table_mismatches[:10]:
                print(f"  {m}")
        if res.lut_mismatches:
            print("Sample LUT mismatches (first 10):")
            for m in res.lut_mismatches[:10]:
                print(f"  {m}")

    return res, fail_labels


if __name__ == "__main__":
    main()
