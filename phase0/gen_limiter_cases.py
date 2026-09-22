"""Generate expected results for the 32-bit code-length limiter.

No real fixture reaches the limiter -- tier-1's longest code is 26 bits -- so the
Python spec in h4_codec_spec.py (validated against dahuffman in H4) is the oracle
for this path. This writes cases the Rust port is graded against.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h4_codec_spec import build_huffman_table, get_32bit_codec, _max_code_len, EOF


def fib(n, start=(1, 1)):
    a, b = start
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


def case(name, freqs):
    """freqs: dict symbol -> frequency, ascending by symbol."""
    uncapped = build_huffman_table(freqs)
    before = _max_code_len(uncapped)
    triggered = before > 32
    table, capped_freqs, iters = get_32bit_codec(dict(freqs))
    after = _max_code_len(table)
    # Ambiguity that actually matters: more symbols share the boundary frequency
    # than there are slots, AND that frequency is above 1 (at 1 it is inert).
    amb = [it for it in iters
           if it["boundary_total"] > it["boundary_selected"] and it["boundary_value"] > 1]
    return {
        "name": name,
        "freqs": [[int(s), int(f)] for s, f in freqs.items()],
        "max_bits_uncapped": int(before),
        "triggered": bool(triggered),
        "max_bits_final": int(after),
        "iterations": len(iters),
        "ambiguous": bool(amb),
        "first_ambiguous": ({"min_k": int(amb[0]["min_k"]),
                             "boundary_value": int(amb[0]["boundary_value"]),
                             "tied": int(amb[0]["boundary_total"]),
                             "slots": int(amb[0]["boundary_selected"])} if amb else None),
        "lengths": {str(int(k)): int(v[0]) for k, v in table.items() if k is not EOF},
    }


def main():
    cases = []

    # Fibonacci frequencies force maximal code depth: n symbols -> ~n-1 bits.
    for n in (30, 36, 40, 45):
        f = fib(n)
        cases.append(case(f"fib{n}", {i: f[i] for i in range(n)}))

    # Same, but with a block of equal frequencies at the low end, so the limiter
    # must choose among ties. Frequency 1 ties are inert; >1 ties are not.
    f = fib(40)
    ones = dict({i: f[i] for i in range(40)})
    for i in range(6):
        ones[i] = 1
    cases.append(case("fib40_ties_at_1", ones))

    twos = dict({i: f[i] for i in range(40)})
    for i in range(6):
        twos[i] = 2
    cases.append(case("fib40_ties_at_2", twos))

    threes = dict({i: f[i] for i in range(40)})
    for i in range(10):
        threes[i] = 3
    cases.append(case("fib40_ties_at_3", threes))

    # A histogram shaped like a real exponent distribution but with a rare tail.
    real = {96 + i: max(1, int(2 ** (20 - abs(i - 12) * 1.7))) for i in range(31)}
    cases.append(case("realistic_skew", real))

    out = {"note": "Expected results for the 32-bit limiter, from the H4 Python spec.",
           "cases": cases}
    p = Path(__file__).resolve().parent / "fixtures" / "limiter_cases.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for c in cases:
        flag = "AMBIGUOUS" if c["ambiguous"] else ("capped" if c["triggered"] else "no-op")
        print(f"  {c['name']:18s} uncapped {c['max_bits_uncapped']:3d} -> {c['max_bits_final']:3d} bits, "
              f"{c['iterations']} iters, {flag}")


if __name__ == "__main__":
    main()
