# Cap sizing review (feat/rsi-gate-cap-sizing)

Scope: `f8eac30...origin/feat/rsi-gate-cap-sizing` (df5f1e3, 4f3cb97,
bc9dcee). Line numbers are at bc9dcee. vLLM references are the 0.32.0 venv
on .44 (`/root/kexi/pegaflow/python/.venv`). Host evidence was read only;
review scripts and the exports are under `~/kexi/tmp/review-capsizing/`.

Verdict: **ACCEPT**

Nothing is relaxed. The floor (`MIN_CAPACITY_GPU_KV = 2.0`), the margin
(`CAP_FLOOR_MARGIN = 1.1`), `CAP_MIN_CONVS` and `OFFICIAL` are unchanged.
The capacity floor is still judged on the measured `U` of the baseline
`cap_cold` trace. The estimate only picks how many conversations, from the
start of the file, go into the trace, so a wrong estimate can waste a run but
cannot make one official. The new `prefill_step_ends` matches vLLM 0.32's
`_mamba_block_aligned_split` exactly on 654,352 cases. Qwen3-4B and DSV2
plans are byte-identical before and after the change.

The SHOULD-FIX items are doc claims that are stronger than the evidence,
plus one pre-existing ordering bug in picking the plan's spec dump. The bug
fails closed.

## MUST-FIX

None.

## SHOULD-FIX

### S1. The plan's spec can be the preempt stage's dump (small GPU pool)

`rsi/gates/run_gate.py:2830-2832` (`cap_plan_spec`) returns the first
workload in `wls` order that has a baseline `probe/spec.json`. `main` moves
only `cap` to the end (`:3415-3416`), so `--workloads
preempt,gsm8k,replay,cap` picks `baseline/preempt/probe/spec.json`. That
stage runs with `--num-gpu-blocks-override`. On the s1 run its dump has
`gpu_kv_bytes` 2.77 GB, while gsm8k, replay, cap and the committed dump
have 33.29 GB. I checked this order directly against `cap_plan_spec`.

Failure scenario: an official Qwen3.5-4B run with preempt listed first
sizes `need` from 2.77 GB (12.2 GB of `U`), stops at 64 conversations
(measured 42.7 GB, 0.64x GPU KV), and the measured floor fails the run
closed. The GPU hours are lost and `cap_plan.spec` is the only clue. The old
inline `next(...)` had the same bug, and 4f3cb97 moved it into the helper
unchanged. Fix: skip `PREEMPT` in `cap_plan_spec`, or prefer `gsm8k`. Add
the preempt-first case to `test_cap_only_run_sizes_from_the_committed_spec`.

### S2. "The G0 check holds the two equal" is false for the runs that use the fallback

`rsi/gates/run_gate.py:2827-2829` (and GATES.md:958-959) justify the
committed-dump fallback with G0. `g0` (`:1509-1537`) compares the committed
dump with the baseline **gsm8k** dump. A cap-only run has none, so no check
compares the committed dump with the spec that actually ran. The fallback is
still safe: it only triggers when no earlier baseline dump exists, and that
only happens in a run that `official_problems` already rejects
(`workloads != gsm8k,replay,cap`). In an official run, gsm8k always comes
before cap. Also, `capacity_floor` reads `gpu_kv_bytes` from the cap stage's
own dump, not from the plan's. On the 4f3cb97 run the cap dump differs from
the committed dump only in `compilation.inductor_compile_config`
(`deterministic`, `benchmark_combo_kernel`); KV bytes are equal. Fix: change
the docstring and the GATES sentence to "only non-official runs reach the
fallback; the floor uses the cap stage's own dump". Optionally, record
`spec_diff(committed, baseline/cap dump)` in `cap_plan` after the stage.

### S3. "Concurrent requests ... only add step ends, so the Checkpoint part is a lower bound" is not what the data shows

GATES.md:971-974, `run_gate.py:2768-2771`. I compared each `cap_cold`
request on the 4f3cb97 run (1858 requests) with `prefill_step_ends` from
the recorded `first_computed`:

| | count |
|---|---|
| requests with fewer aligned stores than the lone model | 0 |
| lone-model aligned ends **not** a subset of the measured ones | 782 |
| measured strictly more | 587 |

Example: prompt 4123 resumed at 1584. Alone it stores 3168 and 3696.
Measured, it stores 2640 and 3696, because a shared budget moved the chunk
end. So concurrency moves step ends as well as adding them. The per-request
count was never lower in this trace. Per group, 16 estimated prompt
checkpoint keys are not in the measured set (630 the other way). The
direction (0.97 at s1, 0.81 at scale) is an observation, not a bound. The
proof does not matter for safety: the floor is measured. It does matter
that GATES does not state a bound that a future model could break silently.
Fix: say "observed below measured on the recorded traces; an over-estimate
fails the measured floor closed (fewer conversations, run not official)".

## NITS

- N1. PROGRESS.md:854-856 says the old 203-conversation plan "would have
  measured below the floor". On the 4f3cb97 trace, the first 203
  conversations' `cap_cold` requests measure 133.196 GB = 2.001x GPU KV
  against a floor of 133.152 GB. Before this fix the plan would have been at
  the floor with no margin, not clearly below it. A real 203-conversation
  run interleaves differently at the tail, so either outcome is possible.
- N2. `prefill_step_ends` (`run_gate.py:2724-2750`) models
  `hash_block_size == block_size`, no Eagle block drop and
  `num_prefill_checkpoint_blocks == 0`. Qwen3.5-4B's dump has exactly these
  values (`hash_block_size` 528, `num_prefill_checkpoint_blocks` 0). The
  estimator does not read them from the spec, so a later hybrid model with
  a different value is sized wrong without any notice. That fails closed, but
  a one-line `print` or a `cap_plan["unmodeled"]` entry when the spec
  differs would make the cause obvious.
- N3. `test_cap_only_run_sizes_from_the_committed_spec`
  (`test_oracle_capacity.py:635-645`) tests the helper, not the `main()`
  wiring at `run_gate.py:3474-3479`. Inlining the old `next(...)` again
  would keep it green. The wiring is three lines, so this is acceptable.
- N4. Qwen3.5-4B's official plan now uses the whole file
  (`enough: false`, 256/256) and measures 2.52x. There is no file headroom
  left: a vLLM change that cuts measured `U` by more than about 21% fails
  every official Qwen3.5-4B run closed, with `(whole replay file used)`.
  That is correct behavior, but worth a line in PROGRESS so that
  `enough: false` on an official record is not read as a defect.

## Verified

- Floor on measured U: `capacity_floor` (`:2708-2721`) uses
  `base_cap["unique_bytes"]` = `oracle.unique_bytes(spec, reqs,
  {"cap_cold"})` (`:1382`) and `gpu_kv_bytes` of the cap stage dump
  (`:1383`). `g5_problems` (`:2920-2934`) adds a problem whenever it fails,
  whatever `cap_plan` says. The estimate never reaches the verdict.
- "An under-estimate only adds conversations": `est.bytes` grows
  monotonically as conversations are added, and the loop stops at the first
  `used >= 64 and est >= need`, or at `len(lines)` (`:2854-2858`). A lower
  estimate therefore gives the same or more conversations, capped at the
  file size (`replay_skip` 0 is official). The trace is always a prefix of
  the file, so the count is the only free variable, and the arm under test
  cannot influence it: it is computed from the baseline or committed dump
  before the baseline cap stage. No property of the trace is selected
  beyond "more of the same file".
- `prefill_step_ends` vs vLLM 0.32: a differential run against
  `Scheduler._mamba_block_aligned_split`, called on a stub with the
  Qwen3.5-4B flags (block 528, no partial hits, no Eagle, no internal
  checkpoints) and iterated to the prompt end. It covered prompt lengths
  1..1587 plus 3000 random lengths up to 8200, budgets
  {512, 528, 1056, 2048}, every aligned start, and junctions at 0,
  start+528, start+1056, the last cacheable boundary, plen-1 and start+700.
  Result: 654,352 cases, 0 mismatches. `last = (plen-1)//bt*bt` is
  `get_mamba_prefill_checkpoint_position(plen, bt, False)`. The junction
  floor is the non-`mamba_shared_prefix_checkpoint` branch, which is the
  one this config takes.
- Oracle rule (H*_C store): `_store_steps(exact=True)` stores block end `e`
  only when a `progress` entry equals `e`. The estimator stores the aligned
  prefill ends plus every decode-crossed boundary
  (`end//bt - plen//bt`, with `end = plen + out - 1` = `computed`). A
  prompt that ends exactly on a boundary is counted once, in the prefill
  (`ch[plen//bt - 1]`), and excluded from `generated`. The resume point is
  `semantics.resolve` over all groups of what the trace has stored so far,
  the same call `oracle.replay` makes. The junction is the resolve of the
  non-Checkpoint groups (vLLM: the longest single-group hit). There are no
  model names or 528 constants in the code (only in docstrings). Align
  mode is read from `spec.cache.mamba_cache_mode`, which gate.json pins and
  `G0.vllm_args` checks.
- Fixture: `testdata/qwen3.5-4b-s1-cap-cold.json` (5958 B) equals the
  `cap_cold` requests of
  `2026-10-25-qwen3.5-4b-p5-it2-s1-25fcccb/baseline/cap/probe/requests.jsonl`
  as sorted `(conv, turn, prompt_len, prompt-block keys)` tuples (39
  requests). `oracle.unique_bytes` on that run = 3494117376, matching
  `measured_unique_bytes`. The committed dump's `block_bytes` equal the run
  dump's.
- Numbers, recomputed offline with OFFICIAL args from the bc9dcee and f8eac30
  exports on .44:

  | | old (f8eac30) | new (bc9dcee) | measured |
  |---|---|---|---|
  | Qwen3.5-4B s1 8 x 6 | 4472832000 | 3391094784 | 3494117376 |
  | Qwen3.5-4B 256 x 12 | 186964377600 | 136328773632 | 167956709376 |
  | Qwen3.5-4B auto plan | 203 convs, 147.05 GB | 256 convs, 136.33 GB, enough=false | 2.52x GPU KV, floor passes |
  | Qwen3-4B auto plan | 64, 54433677312 | 64, 54433677312 | |
  | DSV2-Lite auto plan | 76, 13734033408 | 76, 13734033408 | |

  Per group on the 4f3cb97 run: the full-attention group is exact (2515
  prompt + 202 generated = 2717 measured). Each GDN group has 1532 + 202 =
  1734 estimated vs 2348 measured. The run's `cap_plan.spec` points at the
  committed dump, `available_convs` 256, `capacity_over_gpu_kv` 2.5228. DSV2
  used `docs/rsi/models/DeepSeek-V2-Lite-Chat/spec.json` from
  `origin/feat/rsi-deepseek-v2-lite` (not on the base). The replay pool
  pre-estimate (`estimate_unique`) is also unchanged for Qwen3-4B
  (27714650112) and DSV2 (6085435392). For Qwen3.5-4B it drops from 25.12
  to 18.68 GB. That check is advisory: the measured `check_pool` on the
  baseline trace (`:3460-3470`) still runs before every store arm.
- Tests fail on the old code. The 4 new tests plus the updated
  shared-prefix test fail on f8eac30 (9 cases, mostly signature errors).
  With a signature shim, the two behavior tests still fail:
  `8000 == 5000` (every Checkpoint boundary counted) and
  `4472832000 <= 3494117376`.
- CI: `typos` clean; `cz check --rev-range 1749353..HEAD` successful;
  `pytest -q -p no:cacheprovider rsi/gates` from a `git archive` export of
  bc9dcee on .44: 311 passed in 52.3 s (the export's `semantics.py`
  is identical to the venv's).
