# Review: G4 warm replay vs the native arm

Scope: `feat/rsi-gate-g4-native`, one commit 6409fb7 on top of
`origin/feat/semantics-rsi` 1f00e5a. Files: `rsi/gates/run_gate.py`,
`rsi/gates/test_oracle_capacity.py`, `docs/rsi/GATES.md`,
`docs/rsi/PROGRESS.md`, `.claude/skills/pegaflow-new-model/SKILL.md`.
The rule comes from the user (2026-10-26, hard). The arm under test's warm
replay TTFT p50 and req/s must be no worse than the in-run `native` arm
(vLLM OffloadingConnector). The overhead checks and the G2 references stay
against baseline. `native` is required in official runs.

What I checked:
- the code diff, against `_cmp` / `_perf_verdicts` / `verdicts` /
  `official_problems` / `main` at 6409fb7;
- a full re-judge of three official records with the old tree (1f00e5a)
  and the new tree (6409fb7) side by side, with every verdict leaf
  flattened and compared. The runs are Qwen3-4B it7 (`2026-10-25-qwen3-4b-p4-it7-off-a59a254`),
  DSV2-Lite it7 (`2026-10-25-dsv2-lite-p4-it7-off-a59a254`) and Qwen3.5-4B
  it2 (`2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`). I wrote my own scripts
  (`dump.py`, `cmp.py`, `timing.py`) in `~/kexi/tmp/review-g4-native/` on
  .44 and did not use the author's `g4-native-rejudge` scripts. The scripts
  read the run dirs and write nothing to them;
- the new tests run against the old `run_gate.py`;
- a constructed degenerate-native case (`degen.py`, same dir);
- CI on a full `git archive 6409fb7` export in
  `~/kexi/tmp/review-g4-native/ci`.

No GPU job was run.

## Verdict: ACCEPT-WITH-FIXES

The new checks have the same statistic, pass set, pairing and strictness
as warm vs baseline. They apply to the arm under test only and are void
when its replay digests fail. Every path where native timing is missing
fails closed. No existing verdict changed on the three records. The
author's table reproduces to the last digit.

There is one must-fix. A native arm that ran but served no loads still
gives an official run, and the arm under test then passes `*_vs_native`
trivially.

## 1. Statistic, strictness, scope, void

- Same statistic, passes and pairing. `_native_perf_verdicts`
  (run_gate.py:2622-2643) calls the same `_cmp` with slack 0.0 on
  `_pass_values(rep, _timed_warm(rep), key)` against
  `_pass_values(native_rep, _timed_warm(native_rep), key)`. The baseline
  check does the same thing: `_perf_verdicts` (run_gate.py:2672-2680) passes
  `("replay_warm", timed(rep), 0.0)` and uses `timed(base_rep)` for theirs.
  Both compare the median of each side's own warm2..N passes; neither
  pairs pass by pass. The helpers were moved out of `_perf_verdicts`
  without changing their bodies.
- The baseline rule really is strict. `_cmp` (run_gate.py:2599) uses
  `ok = a < b if lower else a > b` when slack == 0, so a tie fails both
  checks. The new checks inherit that rule and have zero tolerance. Test
  case `tie` (test_oracle_capacity.py:1909) covers it.
- Scope. The checks run under `if kind not in _REFERENCE`
  (run_gate.py:3040). `plain` and `baseline` are skipped earlier, and
  selftest arms `continue` before this point, so `old`, `native`,
  `lmcache` and `selftest-*` never get the keys. On the three records only
  `PegaKVConnectorV2` gained leaves.
- Void. `v["pass"] = v["pass"] and replay_ok` uses the arm's
  `G2.digest_self.replay` pass, like every replay G4 check.

## 2. Fail-closed

Every "no native timing" path makes the run non-official and also fails
both checks:

| case | official | `*_vs_native` |
|---|---|---|
| `native` not in `--arms` | `arm native missing` (run_gate.py:3466) | fail (`nat` None) |
| native replay stage crashed (record-only, so `error` is recorded) | `g4_native_problems` (run_gate.py:2652) | fail (`nat` -> None, run_gate.py:3045) |
| native replay without a timed warm pass, or a pass without `ttft_p50`/`req_per_s` | `g4_native_problems` (run_gate.py:2656) | a missing pass gives an empty median and fails; a partial set is judged, but the run is non-official |
| baseline has no timed warm pass | `not want` -> problem | baseline checks fail too |
| replay workload not run | `workload ... missing` (official_problems) | n/a |

A failed HTTP request in a replay pass raises (`raise_for_status`,
workloads.py:455), so the native stage crashes rather than recording a
partial pass. Contamination (`contamination`, run_gate.py:3408) covers
native's timed passes too, so a native stall with p99 above 10x baseline
also makes the run non-official.

Native's own G0 / G2 failures do not suppress its timing. On all three
records native fails `G0.spec` (and `G2.digest_vs_baseline.gsm8k` or
`G2.resume_reference`), and its timing is still used. The test that
checks this sets native's `digest_self` to None (test_oracle_capacity.py:1928-1929).

**The gap (MUST-1).** Nothing checks that native's replay actually worked
as an offloading store. Take a native replay that ran but whose loads all
failed (recompute) or never hit: for example, OffloadingConnector
disabled or degraded on a new model or vLLM version, or a CPU cache that
does not cover the replay. Its warm timing is about baseline's. The
problem list is then empty and the run is official. An arm that barely
beats baseline then passes `*_vs_native` trivially. I reproduced this with
the gate's own functions (`degen.py`): native had `load_errors` = 1 and
warm hit 0, with warm passes at baseline speed, and the arm was 2% faster
than baseline. The result was:

```
g4_native_problems: []
G4.replay_warm.ttft_p50_vs_native True
G4.replay_warm.req_per_s_vs_native True
native G3.replay_warm pass: False
native G4.replay_warm.ttft_p50 (vs baseline) pass: False
```

The verdicts already flag that native is broken: its `G3.replay_warm` and
its own `G4.replay_warm.*` vs baseline fail. Nothing reads those flags,
because native's verdicts are `required: false`.

The bad direction is a native that is too slow. A native that is wrongly
fast only makes the check harder to pass, so it fails safe; native's
`G2.digest_self.replay` passes on all three records anyway.

## 3. Re-judge (old 1f00e5a vs new 6409fb7, every leaf)

| record | leaves old -> new | changed leaves | only-new leaves | `g4_native_problems` |
|---|---|---|---|---|
| Qwen3-4B it7 | 909 -> 917 | `PegaKVConnectorV2/pass` True -> False | 8 (the two `*_vs_native` dicts) | [] |
| DSV2 it7 | 949 -> 957 | none, apart from the tree path inside the `G0.spec` "spec missing" message | 8 | [] |
| Qwen3.5 it2 | 1238 -> 1246 | `PegaKVConnectorV2/pass` True -> False | 8 | [] |

No leaf was removed. Every `baseline`, `old` and `native` leaf is
identical. DSV2's v2 already failed (no committed spec under this tree), so
its arm-level `pass` does not change.

Medians of warm2..N, recomputed from `.gate/<arm>/replay/result.json`:

| record | v2 TTFT p50 | native TTFT p50 | `ttft_p50_vs_native` | v2 req/s | native req/s | `req_per_s_vs_native` |
|---|---|---|---|---|---|---|
| Qwen3-4B it7 | 0.5640 | 0.5619 | fail | 9.646 | 9.359 | pass |
| DSV2 it7 | 0.2996 | 0.2919 | fail | 12.346 | 13.085 | fail |
| Qwen3.5 it2 | 0.2974 | 0.1729 | fail | 14.633 | 13.155 | pass |

These match the author's numbers exactly. All three records are official
under the new tree as well (native present, warm2 and warm3 recorded). By
S1 these records keep the verdict of the gate rev they ran under. Their
next official runs fail G4.

## 4. Noise: is a strict median over 2 passes decidable here?

Per-pass TTFT p50 (warm2, warm3) and the spread within one arm:

| record | arm | warm2 | warm3 | range | range / median |
|---|---|---|---|---|---|
| Qwen3-4B | v2 | 0.5514 | 0.5766 | 25.2 ms | 4.5% |
| Qwen3-4B | native | 0.5612 | 0.5626 | 1.5 ms | 0.3% |
| Qwen3-4B | baseline | 3.697 | 3.565 | 132 ms | 3.6% |
| Qwen3-4B | old | 0.6494 | 0.6721 | 22.7 ms | 3.4% |
| DSV2 | v2 | 0.2957 | 0.3035 | 7.8 ms | 2.6% |
| DSV2 | native | 0.2799 | 0.3039 | 24.0 ms | 8.2% |
| DSV2 | old | 0.3930 | 0.3357 | 57.3 ms | 15.7% |
| Qwen3.5 | v2 | 0.2996 | 0.2952 | 4.4 ms | 1.5% |
| Qwen3.5 | native | 0.1745 | 0.1712 | 3.3 ms | 1.9% |

Qwen3-4B TTFT: v2 is 2.1 ms (0.37%) behind native. That margin is 12x
smaller than v2's own warm2-to-warm3 range of 25.2 ms. v2's warm2 alone
(0.5514) beats both native passes by 10 ms; its warm3 loses by 14 ms. With
two passes the median is the mean of the two, so this verdict depends on
which pass happened to be slow. **At this margin the comparison cannot be
decided with these data.**

DSV2 TTFT is in the same situation. v2 is 7.7 ms (2.6%) behind, but v2's
whole range [0.2957, 0.3035] lies inside native's [0.2799, 0.3039].

The other four comparisons can be decided:
- DSV2 req/s fails: v2's best pass, 12.49, is below native's worst, 13.01.
- Qwen3.5 TTFT fails by 72%.
- Qwen3-4B req/s passes: 9.63 against 9.43, so the ranges are disjoint.
- Qwen3.5 req/s passes by 11%.

So DSV2 and Qwen3.5 fail G4 vs native whatever the noise. For Qwen3-4B
the result is a coin flip. The vs-baseline checks have margins of 6x or
more on all three models, which is why this problem did not show up
before.

Each arm also runs in its own vLLM process at a different time, so
drift between arms adds to the spread within an arm. More passes within
one arm narrow the spread within the arm, but they do not remove the
drift between arms.

This is a SHOULD-FIX recommendation for the user (SHOULD-1). The
zero-tolerance rule stays as it is.

## 5. Docs and tests

- GATES.md (G4 intro, official run, precise definitions), SKILL.md
  (step text, report table, lesson) and PROGRESS (log entry) state the same
  rule. Specifically, they agree on these points:
  - the medians of warm2..N;
  - the strict rule, where a tie fails;
  - the arm under test only;
  - native required for an official run;
  - G0 / G2 failures of native ignored;
  - overhead checks and G2 references kept on baseline.

  The old wording elsewhere is stale (NIT-1).
- Tests (test_oracle_capacity.py:1891-1974) are table-driven, with ids.
  Against the old `run_gate.py`, all 11 new cases fail: `KeyError` on the
  new keys and `AttributeError: g4_native_problems`. The two other cases
  that `-k native` selects are pre-existing and pass on both trees.

## 6. CI (full `git archive 6409fb7` export in `~/kexi/tmp/review-g4-native/ci`, native `.so` from `oracle-fallback-ci-e67c03f`)

- gate pytest: `rsi/gates`, 374 passed, 0 skipped. `pegaflow.pegaflow`
  was loaded from the export's `.so`.
- typos: clean.
- ruff 0.14.14 (the prek rev), `--config python/pyproject.toml`: `check`
  and `format --check` are clean on `rsi/`. On `python/` there are 6 `I001`
  and 11 files that would be reformatted. All of them are in files this
  change does not touch, and the counts are identical on 1f00e5a, so they
  are pre-existing.
- `cz check`: the 6409fb7 message validates. So does this review's
  message.

## Findings

### MUST

- **MUST-1** run_gate.py:2646-2658 (`g4_native_problems`); also
  run_gate.py:3040-3047. A native replay that ran but did not work as an
  offloading store is accepted as the reference. Examples are load errors,
  or warm passes without external hits so the timing is about baseline's.
  The run is then official, and `*_vs_native` passes for any arm that
  beats baseline (`degen.py` above). This is a fail-open path in a hard
  user rule. It is realistic for new models (P6 gemma) or vLLM bumps, where
  OffloadingConnector may not engage.

  Fix: make the run non-official when native's replay is not a valid
  store. Base this on native's own replay data, not on its G0 / G2.
  Examples: `native G3.replay_warm` fails (load errors or excess), or the
  hit rate of native's timed warm passes is well below the oracle's
  (`result.hit["warm2".."warmN"]` has `hit_rate` and `oracle_hit_rate`;
  on all three records native is at the oracle: 0.998 / 0.998 / 0.946).
  Another option is that native's own `G4.replay_warm.*` vs baseline fails.
  If a model truly has no working native, list it in GATES.md, like the
  lmcache status table, instead of passing silently. Add a table case,
  `native-degenerate`, to `test_official_needs_native_warm_replay`.

### SHOULD

- **SHOULD-1 (for the user)** The statistic is GATES.md:1184-1209 with
  `OFFICIAL["replay_repeats"] = 3`, which gives 2 timed passes. A strict
  median over 2 passes cannot decide margins below the spread from pass
  to pass. That spread is 0.3-16% of TTFT p50 in these records. Qwen3-4B
  fails by 0.37% with a 4.5% spread in v2's own passes (section 4).

  Two options, both of which keep zero tolerance:
  - (a) Raise the number of timed passes for official runs (for example
    `replay_repeats` 6, which gives 5 timed passes; one Qwen3-4B pass takes
    about 27 s per arm). Then define "no worse" with a statistic that has
    an explicit "inconclusive" outcome, for example a one-sided rank test
    or "every pass of mine beats native's median". An inconclusive result
    counts as a fail or requires a rerun, never a pass.
  - (b) Keep medians, but require an independent repeat run whenever
    |delta| is smaller than either arm's pass-to-pass range.

  Whichever option is chosen, record the decision next to the rule.
- **SHOULD-2** docs/rsi/PROGRESS.md:110-118. The log entry does not say
  what the rule does to the current models. Under it, Qwen3-4B it7, DSV2
  it7 and Qwen3.5 it2 all fail G4 vs native:
  - Qwen3-4B fails TTFT by 2.1 ms;
  - DSV2 fails both checks;
  - Qwen3.5 fails TTFT by 125 ms.

  The P3 row ("G0-G4 accepted") and the P4 / P5 rows should note that
  the next official run fails G4 unless the load path improves. Add the
  table from section 3 there, or in each model's ITERATIONS.md.

### NIT

- **NIT-1** Stale wording:
  - GATES.md:170 and GATES.md:558 still say "phase 2 will gate warm replay
    ... against native"; warm replay is gated now.
  - GATES.md:554 and run_gate.py:16 call native "record only". Its
    verdicts are still `required: false`, but the arm and its replay timing
    are now required.
  - GATES.md:98 says "Compared against baseline vLLM (no PegaFlow) and the
    old connector". It should say native.
- **NIT-2** test_oracle_capacity.py:1960-1974 builds the problem list by
  hand (`official_problems` + `g4_native_problems`) instead of going
  through the `main` wiring (run_gate.py:4028-4029). A regression in the
  `if "native" in arms and "replay" in wls` guard would go unnoticed. This
  is acceptable given how much `main` would need, but worth a comment.
- **NIT-3** Re-judging a pre-2026-10-26 record without a native arm now
  fails `*_vs_native`, with no `na`. That is consistent with S1 (native
  was an optional arm, not an input the gate failed to produce), but
  GATES.md S1 could say so in one line.
