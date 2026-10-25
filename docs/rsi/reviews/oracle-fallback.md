# Review: fallback-aware G5 oracle

Scope: `feat/rsi-gate-oracle-fallback`, one commit 3c19024 on top of
`origin/feat/semantics-rsi` a4523f3. Files: `rsi/gates/oracle.py`,
`rsi/gates/run_gate.py`, `rsi/gates/test_oracle_capacity.py`, and
`docs/rsi/GATES.md` (G5). The finding it fixes is in
`origin/feat/rsi-qwen3.5-4b`:`docs/rsi/reviews/P5-it2.md` ("G5 oracle finding").

What I checked:
- the code diff, against `pegaflow-semantics/src/groups.rs` (`needed`,
  `resolve`);
- a re-score of every cap stage of three official runs, with the old tree
  (a4523f3) and the new tree (3c19024) side by side. The runs are Qwen3-4B
  it7, DSV2-Lite it7 and Qwen3.5-4B it2 c5d3f04; each has 3 arms x 3
  fractions. My script is `~/kexi/tmp/review-oracle-fallback/rescore.py`
  on .44. I wrote it myself and did not use the dev scripts. It reads the
  run dirs and writes nothing to them;
- a test run of the new tests against the old `oracle.py` / `run_gate.py`;
- CI on a full `git archive 3c19024` export in
  `~/kexi/tmp/review-oracle-fallback/ci`.

No GPU job was run.

## Verdict: ACCEPT-WITH-FIXES

The change is correct:
- Every candidate is a real store of `C` bytes run over the whole trace, so
  `H*_C` is reachable.
- On every segment, `H*_C` is at least `H_lru` in `Σ ext*_C`.
- `prefix_closed` comes from the Kind through `semantics.needed`.
- Full-only records re-score identical, and the Qwen3.5 numbers reproduce.
- No `pass` changes.

There is no must-fix. The two should-fix items are about reporting and doc accuracy.

## 1. Correctness

Candidates are feasible stores:
- `summarize_capacity` (oracle.py:671-672) runs each policy in
  `oracle_c_policies` through `simulate`, over the whole trace. Each run uses
  the same `capacity`, the same store events and visibility, the same `ref`
  local hit, and the same cap at `max(unlimited, ref)`.
- `belady_fallback` differs from `belady` in one thing only: the use set it
  feeds the eviction choice (oracle.py:456, 463, 580). Admission and eviction
  still respect `capacity`, and hits are still `resolve` over resident blocks.
- So each candidate's per-request hits come from one store of `C` bytes.

Per segment max:
- The winner is chosen per segment (oracle.py:712-723). `warm_timed` is its
  own segment over the same three whole-trace runs, so its bar is one
  candidate's sum. It is not a sum of per-pass winners, which would mix
  policies inside a segment and could be unreachable.
- Different segments may report different winners (Qwen3.5 0.5:
  `cap_warm`/`cap_warm2` belady, `cap_warm3` and `warm_timed`
  belady_fallback). That is fine: each segment is scored on its own.
- `efficiency_c_own_local` picks its own max (oracle.py:723). That is also a
  single store per segment (own-local runs), and the field is diagnostic.

Never below LRU: `lru` is a candidate, so `ext*_C >= ext_lru` holds by
construction on every segment.

`prefix_closed` (oracle.py:357-380):
- It checks that the hull of the earlier non-empty `needed` ranges stays
  inside the current one, at every lcm step up to the longest prompt. That is
  the generic version of `needed(q) ⊆ needed(p)`.
- Against `groups.rs:79-88`:
  - Full `0..n`: closed.
  - Checkpoint `n-1..n`: open from the second step.
  - Window `(p-w)/bt..n`: closed while `p - w < bt`, open after.
  - Scratch: skipped.
- The tests cover all five shapes (window-short/long included). The function
  names no model. On the runs it returns True for Qwen3-4B and DSV2 and False
  for Qwen3.5.
- When it holds, `_fallback_keys` equals `_external_keys` as a set and in
  order (group then block). The fallback run would duplicate `belady`, so
  skipping it changes nothing.

## 2. Nothing relaxed

- `_capacity_verdict` `pass` (run_gate.py:2516-2522) is unchanged.
  `efficiency_c` is still only `reported` (2530).
- The diff in `run_gate.py` adds `oracle_c_policy` to the verdict, the curve
  and `hit_report` (2550, 3002, 3032), and nothing else.
- Re-scoring all 27 stages, I rebuilt the verdict from the new capacity block
  and from the record. `pass` is identical in every stage. On Qwen3.5 0.25
  and 0.5 only reported fields differ: `efficiency_c`, `*_vs_oracle_c`,
  `oracle_c_hit_rate`, `reported.below`. Elsewhere only the new
  `oracle_c_policy` differs.
- The bar only rises, so `E_C` can only drop. Qwen3.5 0.25 goes from 1.18 to
  0.88, which removes the impossible `E_C > 1`.
- One indirect effect is not a verdict but an officiality input (S2):
  `freeze_vs_oracle_c` feeds `g5_problems`.

## 3. Re-score

Shared fields of the `capacity` block are compared against the stored
`result.json`, with the new keys stripped: `oracle_c_policies`,
`oracle_c_policy`, `oracle_c_candidates`, `diagnostics.oracle_c_fallback`.
Old tree: all 27 stages are identical to the record. New tree: all 18
Qwen3-4B / DSV2 stages are identical, `prefix_closed` is True, and the
candidates are belady + lru. Belady wins every segment.

Qwen3-4B it7 and DSV2 it7, `warm_timed`. Old and new are equal:

| run | arm | C/U | H*_C | H_lru | E_C |
|---|---|---|---|---|---|
| Qwen3-4B | v2 | 0.25 / 0.5 / 0.75 | 0.8629 / 0.9294 / 0.9703 | 0.5836 / 0.8576 / 0.9158 | 0.1114 / 0.7011 / 0.7406 |
| Qwen3-4B | old | 0.25 / 0.5 / 0.75 | 0.8663 / 0.9297 / 0.9710 | 0.5821 / 0.8571 / 0.9153 | 0.0708 / 0.6749 / 0.6932 |
| Qwen3-4B | native | 0.25 / 0.5 / 0.75 | 0.8629 / 0.9296 / 0.9711 | 0.5903 / 0.8576 / 0.9147 | 0.2963 / 0.7154 / 0.7755 |
| DSV2-Lite | v2 | 0.25 / 0.5 / 0.75 | 0.8536 / 0.9245 / 0.9676 | 0.6706 / 0.8466 / 0.9182 | 0.0699 / 0.6203 / 0.6586 |
| DSV2-Lite | old | 0.25 / 0.5 / 0.75 | 0.8523 / 0.9250 / 0.9678 | 0.6511 / 0.8466 / 0.9180 | 0.1805 / 0.6135 / 0.6157 |
| DSV2-Lite | native | 0.25 / 0.5 / 0.75 | 0.8535 / 0.9246 / 0.9678 | 0.6799 / 0.8471 / 0.9170 | 0.2817 / 0.6217 / 0.7085 |

Qwen3.5-4B it2 c5d3f04 (`warm_timed`, old -> new):

| arm | C/U | H*_C | winner | H_lru | E_C | E_C own-local | freeze / H*_C |
|---|---|---|---|---|---|---|---|
| v2 | 0.25 | 0.6900 -> 0.8412 | belady_fallback | 0.7937 | 1.1831 -> 0.8813 | 0.9276 | 0.458 -> 0.341 |
| v2 | 0.5 | 0.8779 -> 0.8894 | belady_fallback | 0.7919 | 0.8277 -> 0.8128 | 0.8649 | 0.667 -> 0.655 |
| v2 | 0.75 | 0.9457 -> 0.9457 | belady | 0.9064 | 0.8800 -> 0.8800 | 0.9532 | 0.913 |
| old | 0.25 / 0.5 / 0.75 | 0.6903 / 0.8779 / 0.9457 -> 0.8414 / 0.8885 / 0.9457 | bf / bf / belady | 0.7938 / 0.7922 / 0.9053 | 1.2348 / 0.8667 / 0.9431 -> 0.9200 / 0.8524 / 0.9431 | | |
| native | 0.25 / 0.5 / 0.75 | 0.6901 / 0.8779 / 0.9457 -> 0.8426 / 0.8908 / 0.9457 | bf / bf / belady | 0.7937 / 0.7919 / 0.8616 | 1.1754 / 0.8150 / 0.9278 -> 0.8738 / 0.7986 / 0.9278 | | |

This reproduces the claimed H*_C 0.841 / 0.889 / 0.946 and E_C v2 0.881 /
0.813 / 0.880. At 0.75, `belady_fallback` reaches 0.9408 (0.0049 below
plain), matching the doc.

Requests where LRU beats the winning store (per request, `lru > winner`),
counted independently of `lru_above_requests`:
- Qwen3.5, `warm_timed`: 0 at every fraction and in every arm. Before the
  change it was 995 at 0.25 and 239 at 0.5.
- Qwen3.5 per pass: 0 everywhere except `cap_warm2` at 0.5, with 15 in each
  arm. That pass's winner is plain Belady (aggregate ext 5.74M vs fallback
  5.61M vs LRU 4.75M).
- Full-only, at 0.25 only:
  - Qwen3-4B: up to 9 per pass (`cap_cold`/`cap_warm`), 0 in `warm_timed`;
  - DSV2: 1-8 in `warm_timed`.
  - These are unchanged from before: the same plain Belady.

So "0" holds on the scored segment for the hybrid model. It is not a
per-request property of the design: the max is per segment by `Σ ext*_C`,
and the greedy Belady can lose single requests. This is acceptable; the
doc claims only the segment bound.

No candidate is ever above `max(unlimited, ref)` (0 requests).
`oracle_c_tokens >= lru_tokens` held on every segment as well.

## 4. Tests

- `test_oracle_c_is_never_below_lru`: 3 table cases. The fallback trace is
  minimized from the P5 shape: one Full + Checkpoint case and one Qwen3.5
  3-ckpt + Full case.
- `test_fallback_use_set_differs_only_where_needed_is_not_prefix_closed`: 5
  Kind cases.
- Both are table-driven with ids, and each protects one contract: the bar is
  never below LRU, and full-only bars do not move.
- On the old `oracle.py` / `run_gate.py`, exactly these 8 cases fail and the
  rest of the file passes (334). The two fallback cases fail for the real
  reason, not only a missing key: the old `ext*_C` is 0 against `ext_lru` 4,
  and the new one is 8.
- The full-only case and the `prefix_closed` cases fail on old code only for
  missing API, which is expected for new contracts.
- No noise.

## 5. CI (full `git archive 3c19024` export, .44)

- Gate pytest (`PYTHONPATH=. pytest -q -p no:cacheprovider rsi/gates`,
  shared venv): 361 passed in 52 s.
- typos: clean.
- ruff 0.14.14:
  - `ruff check` (CI scope): all passed (one pre-existing `noqa` warning in
    `examples/`);
  - `ruff check --config python/pyproject.toml rsi`: all passed;
  - `ruff format --check`: 13 files formatted.
- `cz check` on the commit message: successful.

## Must-fix

None.

## Should-fix

- S1 (run_gate.py:2561-2563, GATES.md:1346-1350): the verdict's
  `oracle_c_groups` / `oracle_c_passes` come from `diagnostics.oracle_c`,
  which stays the plain Belady (oracle.py:681). GATES.md:1350 still calls
  them the "stranded bytes for `H*_C`". On Qwen3.5 at 0.25 and 0.5 the store
  that sets `H*_C` is `belady_fallback`, so a reader of the per-group
  stored / evicted / stranded numbers sees a different store than the bar.
  Fix: either report the `warm_timed` winner's diagnostics as
  `oracle_c_*` (and the plain Belady under its own name), or also put
  `diagnostics.oracle_c_fallback` in the verdict. In both cases, fix the
  GATES.md sentence.
- S2 (GATES.md:1313-1314 "nothing else changes"; run_gate.py:3324):
  `freeze_vs_oracle_c` is an officiality input (`FREEZE_SEPARATION` in
  `g5_problems`). A higher bar lowers it, so for non-prefix-closed models a
  trace that did not separate freeze from `H*_C` before can now be
  official.
  - That direction is justified: the bar is a reachable store, so the
    separation is real.
  - No current run flips: Qwen3.5 0.5 goes 0.667 -> 0.655, and full-only
    runs are unchanged.
  - Still, it is a change beyond recording fields. Say so in GATES.md
    instead of "nothing else changes".

## Nits

- N1 (GATES.md:143, oracle.py:649): "never below `H_lru`" holds on
  `Σ ext*_C` (the selection key), not strictly on `oracle_c_hit_rate`.
  `max(0, h - ref)` and raw tokens could rank differently. They agreed on
  every segment of the three runs. Say "in `Σ ext*_C`".
- N2 (GATES.md:1311-1313): full-only records re-score identical in every
  shared field. The record itself gains `oracle_c_policies` /
  `oracle_c_policy` / `oracle_c_candidates`, so "byte-identical" should say
  "identical apart from the new fields".
- N3 (GATES.md:1329): `efficiency_c_own_local` still says "and the Belady
  run"; it now uses the same candidates (as 1310-1311 says).
- N4 (GATES.md:412): the suite is 361 tests, not 311.
- N5 (oracle.py:368, 373): `prefix_closed` probes up to `prompt_len`.
  `resolve` stops at `prompt_len - 1`, so a Window prompt ending exactly on
  the boundary can run the extra candidate needlessly. That is harmless: an
  extra feasible store can only raise the bar.
