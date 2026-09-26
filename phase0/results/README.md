# Phase 0 raw results

Small run records from the official compressor, copied on 2026-09-26 from the
gitignored `phase0/out/` so the numbers in `docs/FINDINGS.md` have a committed
source. They are not edited, except as noted.

| file | from | written by | cited in FINDINGS |
|---|---|---|---|
| `qwen3-trunc-layers-only-dir.run.json` | `phase0/out/official/` | `run_official.py`, tier 0 | 0.2, 0.4 |
| `qwen3-0.6b-layers-only.run.json` | `phase0/out/official/` | `run_official.py`, tier 1 | 0.4, H1, H2 |
| `qwen3-trunc.h2.json` | `phase0/out/h2/` | `profile_official.py` | 0.3 / H2 |

The two `.run.json` files had a `samples` list removed: the RSS sampler's
`[seconds, MiB]` series (1,594 and 11,132 points, 61 KB and 443 KB). Every other
field is as written. The series stays local only.

Not here, because it is not small or was never saved: the official outputs
themselves (hashed per tensor in `phase0/fixtures/MANIFEST.json`), the runs' stdout
logs, the H6 reordered shard, and the Phase 2 exit gate's stdout.
