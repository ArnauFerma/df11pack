# Norms for the compression projects

These rules apply to the author's BF16 compression repos (bf16-exponent-compression,
df11pack and the others in the same family), and to everything written about them:
docs, papers, posts, drafts, commit messages. Every repo carries an identical copy
of this file; if you reuse these rules elsewhere, they stand on their own.

The aim is simple: **everything written must match what was measured, and anyone
must be able to check it.** Each rule below exists because the 2026-09-26 review
found it broken at least once; the example is given so the reason stays visible.

## 1. Evidence

1.1 **Every number in a doc traces to a committed file.** A raw log, a JSON result,
or a script whose output reproduces it. Say where when it isn't obvious
(`results/a100/gpu_bench.json`, field `mhz`).
*Found: profile numbers printed on a pod and never saved; a 16-bit index gain
(+1.94 points) nobody could reproduce.*

1.2 **Save output the moment it exists.** Every command that produces a result
writes to a file under `results/` (use `| tee`), and the file is fetched before
the machine is deleted. Terminal output that is only on screen is lost.
If a number survives only as a transcription of terminal output, put it in a
clearly labelled file and say it is weaker evidence than a data file.

1.3 **Never round away information in raw logs.** Store full precision; round only
in the doc. *Found: IQRs printed as whole milliseconds, read as "IQR 0" for
sub-millisecond timings.*

1.4 **Record the conditions with the result**, in the same file or next to it:
date, commit hash, exact command line, machine, GPU name, driver, library versions.
Record conditions *during* the measurement, not only before it.
*Found: GPU clocks read once before warm-up (1140 MHz) and quoted as the clock
"throughout", while the clock during timing was 1410 MHz.*

1.5 **Keep the inputs that matter** or say how to regenerate them exactly
(model repo + revision hash, checksums).

## 2. Claims and wording

2.1 **Label every number:** *measured*, *estimated* (say from what), or
*hypothesis*. Predictions are marked as predictions.

2.2 **No absolutes without proof.** "Every", "never", "all", "always",
"byte-identical", "proven", "first", "fastest" need data that covers the whole
claim. Otherwise state the scope: "39 `.safetensors` files", "on one unit",
"on three GPUs". *Found: "byte-identical" that covered only the safetensors files;
"proven unreachable" resting on one measured instance; "never faster" where the
ladder tied.*

2.3 **Compare like with like, and say how.** Same kernel, same configuration,
same data, or say explicitly what differs ("best configuration of each", "same
thread count"). Give the direction of every ratio (time A / time B).
*Found: a "32x slower" that divided an unoptimised time by an optimised one.*

2.4 **Units are exact.** GB (10⁹) vs GiB (2³⁰), ms vs s, bits/weight vs bits/symbol.
State denominators: "52.6 ms for 7.57 B weights (36 layers + LM head)".

2.5 **Quote papers from the primary text**, with section or figure, never from an
abstract summary or memory. *Found: "11x over DFloat11" where the paper says
~6-7x over DFloat11 and 11x over NeuZip.*

2.6 **Don't write "never measured", "nobody has done"**: write "we found no
measurement of".

2.7 **Plain, concise English. No marketing tone.** A result is described by its
number and its scope, not by adjectives.

## 3. Predictions

3.1 Write predictions **before** the run, in their own commit, so git dates them.
*Found: a predictions file committed together with its results, which proves
nothing about when it was written.*

3.2 Never edit a prediction after the result. Add a dated note below it instead.

3.3 After the run, compare every prediction with the result, one by one, and keep
the misses.

## 4. Documentation and logs (the bf16-exponent-compression model)

Each research repo has, and keeps current:

| file | holds |
|---|---|
| `README.md` | what it is, the headline results with their scope, how to reproduce, "How this was made" |
| `METHODOLOGY.md` | the question, data (and which subset each experiment used), hardware and software, correctness protocol, timing protocol, accounting, threats to validity, how to reproduce, where the raw data is |
| `RESULTS.md` | every result, per phase, each table naming the file it comes from |
| `results/<machine>/` | raw logs, JSON, `env_info.json`, GPU info, the run log of each session |
| `CHANGELOG.md` or a dated **Corrections** section | what changed after something was published |

Tools and libraries (like df11pack) also keep `docs/FINDINGS.md`: every measurement
behind the tool, with errata.

4.1 **Log extensively.** Each GPU session leaves: the script it ran, its full log,
the result files, the pod type, image, cost and duration.

4.2 **Scripts must reproduce the published results.** For each result, the exact
script and configuration that produced it is written down (or pinned in the pod
script). *Found: pod scripts whose defaults had moved on, so they no longer
reproduced the results they were cited for.*

4.3 **One place per fact.** A number lives in one file; others link to it rather
than copying it. When a result changes, search every repo, the drafts and any
notes for the old value (`grep -rn`) and fix every copy the same day.

4.4 **Published documents get corrections, not silent rewrites.** After a release,
a paper or a post, fixes go in a dated Corrections/errata section or a CHANGELOG
entry that says what was wrong.

4.5 **Superseded text is marked**, not left to read as current: strike through with
a one-line resolution and date, or a dated status note at the top.

4.6 **Dates are absolute** (2026-09-26), never "yesterday" or "last week".

## 5. Code and tests

5.1 **A test must be able to fail.** When adding a test, show it fails on the old or
a deliberately broken version (a mutant) before trusting it.
*Found: a test script that always exited 0; unit tests that passed with the logic
replaced by a constant.*

5.2 **No silent skips.** A test that can't run prints SKIP and why. A missing
input that should exist is a failure, not a pass.

5.3 **Check against ground truth**, not only against another copy of the same
computation. *Found: two GPU decodes compared only with each other.*

5.4 **A flag must do what its name says**, with a test that checks the effect, not
only that the option parses. *Found: `--luts correct` wrote the "correct" stamp
but never changed the tables.*

5.5 **Comments and docstrings are checked like code.** A comment that disagrees
with the code is a bug.

5.6 Run the full test suite, the linter and the formatter before every commit that
touches code.

## 6. Publishing

6.1 **Before anything goes public** (a release, a paper, a post, an arXiv upload),
run an independent review with fresh eyes: agents that did not write the work,
checking every file, number and claim against the data. Fix, then recheck.

6.2 **Drafts are checked against the repos** before posting, and after posting the
draft file records what was actually posted.

6.3 **Posts that are already live** are corrected on the platform when they turn out
wrong, with the owner's edit (GitHub keeps edit history).

6.4 **Read the destination's AI policy before publishing anywhere.** Every platform,
forum, subreddit, repository or venue (its rules, CONTRIBUTING file, code of conduct,
submission guidelines) may restrict or forbid AI-generated or AI-assisted content, or
require a specific disclosure. Read it the same day, before posting, and follow it; if
it forbids the content, do not post there. Record in the draft which policy was read
and when. *Found (2026-09-28): an account blocked for breaking a platform's AI policy,
an easily avoided mistake.*

## 7. Authorship and privacy

7.1 The code, docs and measurements are made largely with Claude Code (Anthropic)
under the author's direction. Every repo says so in "How this was made"; docs use
"we" for that work; every commit carries the `Co-Authored-By` trailer. Posts say
"I made X with Claude Code, under my direction", never "I wrote X".

7.2 Docs are about the work. Personal matters stay out of them.

7.3 No local paths or user names in committed files, no tokens or keys anywhere.
Unpublished drafts stay out of git.

## 8. Rented machines

8.1 Before running: check the pod's real memory limit (a small allocation test) and
disk; prefer Secure Cloud when Community hosts misbehave.

8.2 Keep the SSH session open for the whole run, or confirm detached jobs survive.

8.3 Fetch every result and log before deleting the pod; record the pod type, cost
and duration in the session log.

## 9. When a rule is broken

Fix it, note it in the relevant errata or CHANGELOG, and if it reveals a new kind
of mistake, add a rule here with the example.
