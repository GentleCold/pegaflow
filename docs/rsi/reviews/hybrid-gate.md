# Hybrid gate review (feat/rsi-gate-hybrid)

Scope: `origin/feat/semantics-rsi...origin/feat/rsi-gate-hybrid`, d252bfe..ffdad48.
Line numbers are at ffdad48. The vLLM references are the 0.32.0 venv on .44
(`/root/kexi/pegaflow/python/.venv`). Host evidence was read only, under
`~/kexi/`.

Verdict: **ACCEPT-WITH-FIXES** (M1, M2).

Nothing is relaxed for Qwen3-4B or DSV2. The two new baseline checks only
tighten, but neither has run on those models yet (S1). The hybrid oracle
rule, the preempt fail-closed logic, the claim/measurement mismatch and the
gsm8k compile flag are all correct and match the run data.

There is one way to fool the digests on a non-invariant model: they judge
the bytes in a slot *after* the forward has run, not the bytes the forward
read (M1). The self-test also cannot show that state groups are judged
(M2).

## MUST-FIX

### M1. Start states are digested after the forward, not as the forward reads them

`rsi/gates/rsi_probe.py:621-645` (`_start_states`) queues the episode's
start state (block `start / bt - 1`). `:894-908` (`wait_for_save`) digests
it after the step's forward. vLLM 0.32 runs each step in this order:

1. `handle_preemptions` (`gpu_model_runner.py:3971`), called every step.
2. `_update_states` (`:3979`).
3. `preprocess_mamba` (`:4091`). For a new or resumed request,
   `prev_state_idx = (num_computed_tokens - 1) // block_size`
   (`mamba_utils.py:1490-1493`), which is the start-state block. The align
   precopy copies it into the running-state column
   `cdiv(computed + scheduled, bs) - 1` (`:1519-1535`).
4. A sync `start_load_kv` (`kv_connector_model_runner_mixin.py:83-85`, when
   `has_sync_kv_loads`).
5. The forward, which reads the running column.

So a sync Mamba load lands *after* the precopy, and the GDN layers run from
whatever the start block held before the load. In `wait_for_save` the probe
then reads the correct loaded bytes and records `matched`. The self-test
connector's docstring states this hazard (`selftest/connector.py:25-26`),
but the gate does not detect it. This comes from reading the code; I did not
reproduce it on GPU.

With `batch_invariant=false`, completions are not gated. The only signal
left is gsm8k accuracy, and replay, cap and preempt have none at all.
Computed blocks are not required exact either (`run_gate.py:1797`).

Failure scenario: a connector tuned for G4 sets `sync_load_tokens >= 528`.
`pegaflow/connector_v2/core/scheduler.py:195` already has this knob; today it
is disabled for more than one group, but an RSI iteration can change that.
Then:
- gsm8k loads are 1056 tokens, so they stay async and correct;
- 528-token replay and cap loads go sync and run from stale state;
- every digest matches and every gate passes, while the model computes the
  wrong thing on replay and cap.

The same blind spot covers an async load whose `finished_recving` is
reported before its copy has landed: precopy reads early, and the probe reads
later.

Fix:
- Digest start states in `RsiProbe.handle_preemptions`
  (`rsi_probe.py:882`). It runs every step before the precopy, and the
  precopy reads exactly that block. A sync load has not landed by then, so
  it shows up as mismatched. `ProbeMetadata` is passed in there, so the start
  jobs move from `wait_for_save` to that call; offers stay in
  `wait_for_save`.
- Add a self-test mode `sync` (honest bytes, `load_kv_async=False`) that must
  fail both digest checks.

### M2. The self-test cannot show that state groups are judged

Negative arms pass when "not any(digest check passes)" over all groups
(`run_gate.py:2188`), so a single failing full-attention group satisfies
them.

`offbyone`'s state sabotage is a no-op on this workload. Every external
state load is of one key, the shared 1056 prefix, and it is the only
prompt state the connector saves: `_offered` keeps only boundaries up to
the prompt length (`selftest/connector.py:188`), and prompts are at most 1191
tokens. So `_neighbour` (`:158-162`) returns `done[0]`, which is the correct
state.

Run data, offbyone, (external compared, external mismatched) per group:

| stage | GDN groups 0-2 | full attention, group 3 |
|---|---|---|
| gsm8k | 15 / 0 | 30 / 15 (+113 local) |
| preempt | 21 / 0 | 42 / 21 |

Bitflip corrupts both kinds at once. So no arm shows "attention right,
state wrong", and nothing would notice if the state digest path regressed
(for example, digesting a slot that always matches). Today, detection rests
only on bitflip's per-group counts (groups 0-2: 27/27 mismatched on gsm8k,
21/21 on preempt).

GATES.md:977-978 ("offbyone ... state groups: another saved state") does
not hold on this workload.

Fix:
- Require each negative arm to have mismatched or unreferenced loads in
  *every* transferred group, in both digest checks.
- Make offbyone's state sabotage independent of other saved prefixes, e.g.
  load GDN group `g+1`'s state into group `g` (same shapes), or save the 528
  state and load it into the 1056 slot.
- Have the connector count the sabotages it applied, and fail the arm if the
  count is 0.

## SHOULD-FIX

### S1. The new hard checks have not run on the accepted models

`G2.batch_control` (`run_gate.py:2129`) and `G0.vllm_args` (`:2120`) now gate
every model's baseline. No Qwen3-4B or DSV2 run exists on this gate. On
.44, the only runs with `G2.batch_control` are the three Qwen3.5 self-tests.

Re-judging any older record now fails both checks: `batch_control` returns
None because there is no `alone` pass, and there is no `vllm_cmd`. That
breaks the offline re-judge practice (PROGRESS.md:408, :814).

Fix:
- Do one plain+baseline gsm8k run per accepted model on this branch before
  merging.
- Have the re-judge mark these checks `na: record predates`, and say so in
  GATES.md.

### S2. The accuracy gate sits inside batch noise on Qwen3.5

Honest warm accuracy vs baseline cold, n=128, against `ACC_TOL` = 1 pp
(`run_gate.py:88`):

| run | delta |
|---|---|
| 5bc2e25 | -1.56 pp |
| cb57dd5 | -0.78 pp |
| aa42a44 | +0.78 pp |

The honest arm is not held to `G2.gsm8k` (`:2182-2185`). There is no
evidence that a correct connector passes it, and it is now the only
end-to-end output check.

Fix: at official size (1319), measure the spread for a correct pass, e.g.
honest or baseline warm under different load timing. Then either show
|Δ| ≪ 1 pp, or derive the tolerance from the measured spread. Do not loosen
it without numbers. Report `G2.gsm8k` for the honest arm.

### S3. On block 528, gsm8k judges a single prefix

Every gsm8k prompt's last full block is the shared 8-shot prefix (1056). So
every gsm8k load is of one key per group: honest has 1 state key per GDN
group and 2 full-attention keys (all 1275 judged loads). Preempt external
loads are the same key, because the self-test saves prompt blocks only.

Loads of diverse keys are judged only by cap (exhaustive) and replay
(partial). Neither has run on Qwen3.5, not even baseline. `preempt` is
optional in the official profile (`run_gate.py:2523`).

Fix:
- Require `preempt` in the official profile for state-group models.
- Run baseline cap and replay once on Qwen3.5, so the exhaustive state path
  (offers, `not_offered`, preempted states) is shown to hold on many keys.
- State the coverage limit in GATES.md.

### S4. The G0.vllm_args parser misses equivalent spellings

`_flag_values` (`run_gate.py:1351`) does not parse `--flag=value`. It also
does not normalize `_` to `-`, which vLLM's `FlexibleArgumentParser`
accepts. So `--prefix-cache-retention-interval=0` or
`--prefix_cache_retention_interval 0` placed later overrides the gate flag
and still passes. The docstring's claim (`:1364-1367`) is false for these
forms.

This is reachable only via `--vllm-arg`, which already makes a run
non-official, so it is defense in depth. The effective compilation config is
also not in the spec dump, so `G0.spec` does not pin it.

Fix:
- Split on `=` and normalize underscores before comparing.
- Dump `compilation_config` (mode and `inductor_compile_config`) in
  `spec.json` and pin it in `G0.spec`.

### S5. The torch compile cache is shared across arms and runs

The connector arms share one compile hash, so they load one artifact.
selftest-honest and selftest-liar both "Directly load AOT"
`c3d69611f93c`, the one the baseline produced, from
`~/.cache/vllm/torch_compile_cache`.

Under the M7 residual (uid 0, `CAP_DAC_OVERRIDE`), the arm under test can
rewrite that artifact. The next run's baseline and arm then both run the
tampered model, so digest-vs-baseline compares tampered with tampered. The
provenance watch does not cover the cache.

Fix: give each run its own compile cache dir (`VLLM_CACHE_ROOT`) inside the
run dir. Baseline populates it, and it is bound read-only into the isolated
arm. Alternatively, hash the artifact in the watch.

### S6. Failure-mode tests are missing, and CI does not run the gate tests

Missing tests:
- a `start_*` state skip failing `_digest_verdict` (`run_gate.py:1778`);
- `preempt_gpu_blocks` sizing;
- `_plain_equals_baseline` with equal completions but different logprobs;
- the probe's step logic (`_start_states`, `_offered_states`, `_unoffered`,
  `_preempt_job`, `_track`), which is exercised only by the GPU self-test.

PR CI runs `python/tests` only (`ci.yml:162-165`), so none of the 224 gate
tests run in CI.

Fix: add these as table-driven tests. Add `rsi/gates` to a CI job, or state
that it is a manual gate.

## NITS

- N1. GATES.md:762-767 says `-0.388626` vs `-0.390268` came from two fresh
  compiles of the plain config. In the probe, `-0.388626` is the older
  global-cache plain artifact. The two fresh plain compiles (`plain_fresh`,
  `plain_fresh3`) agree at `-0.390268`. This is still nondeterminism across
  compile times, and the swap evidence holds. The wording should say that.
- N2. `G2.split_control` on Qwen3.5 is vacuous: `same_start` is 63/64, and
  the pre-warm check is met by the shared prefix alone. It still reports
  `measured_invariant: true`. The gate.json notes say "both claims are
  measured in every run". Report `na` or `vacuous` when
  `same_start >= 0.9 n`. It cannot pass a wrong load, only fail a right one.
- N3. `not_offered` is logged at every decode boundary (192 per preempt
  stage) because offers arrive in the next step's scheduler output. Those
  states are then digested as `offer`. Compare against the next step's
  offers, or drop the record.
- N4. `summary.provenance.pegaflow_server.path` points at the gate tree
  (missing, `sha256: null`). No server ran in this self-test, so record
  "not used". The server is not in `_PROVENANCE_ROLES`.
- N5. The venv `pegaflow` is at 056eb4b, not the base head d252bfe. The diff
  is `connector_v2` only, and the self-test does not use it.
- N6. GATES.md:268 says `preempt` is "required on hybrid models" for the
  self-test. Nothing enforces it.
- N7. For self-test arms, `G0.spec` and `G0.provenance` are computed but
  are not in `need` (`run_gate.py:2183`). This predates the branch.

## Verified

Branch and docs:
- `git fetch` confirmed origin head ffdad48 and base d252bfe.
- Read the full diff of `run_gate.py`, `digest_compare.py`, `rsi_probe.py`,
  `selftest/connector.py`, `workloads.py`, the tests, GATES.md and both
  gate.json files.
- 5bc2e25..ffdad48 changes GATES.md only.

Accepted models (batch_invariant true):
- `_plain_equals_baseline` returns `cold_differ == 0`, as before.
- `split_control(…, True)` uses the same pair and basis.
- `_outputs_verdict` is unchanged. The non-invariant branch was already
  `pass: True`.
- `digest_compare` keys are now `(kind, key)`. Kind is always "block" for
  non-state groups, so the counts are unchanged.
  `occurrences + preempted >= computers` is preserved, because
  preempt-digested episodes move from `preempted` to `occurrences`.
- Self-test full-attention paths are unchanged: `bool(self._state)` is
  False, so loads stay sync.
- Qwen3-4B gate.json is unchanged. The DSV2 gate.json is on
  `feat/rsi-deepseek-v2-lite`.

Run 2026-10-24-qwen3.5-4b-selftest-5bc2e25:
- Copied `summary.json`, `requests.jsonl`, digests and outputs to
  `/tmp/hgr`, and re-ran `digest_compare` per group locally.
- Gate tree is rev 5bc2e25, verified (75 files). Venv is 056eb4b, verified.
- `provenance_watch`: 12 checks, 0 changes. 11 port leases.
- `summary.gate` equals the committed gate.json.
- Not official, as expected for a self-test (`gsm8k_limit`, no under_test).
  Isolation and scan are N/A: there is no under-test arm.
- Baseline:
  - plain_equals_baseline: `alone_ref`, 64/64 completions and logprobs
    equal; batched cold_differ 0.
  - batch_control: 16 completions and 64/64 logprobs differ, so measured
    false, matching the claim.
  - `alone` and `alone_ref` accuracy are both 0.71875 (plain and baseline).
  - G2.preempt: 16 preempted requests, 62 resumed local loads judged, 0
    mismatched, 40 GPU blocks.
- Honest:
  - gsm8k: external 150/150, local 1125/1125.
  - preempt: after-preemption local 46/46, external 55/55. That is 11
    resumed episodes with an external load at 1056, × 5 groups.
  - warm_differ 16.
- Liar:
  - gsm8k digest_self: local unreferenced 1265, external judged 0.
  - With `min_judged` 0.9, the check requires `judged > 0`, so it fails.
  - digest_vs_baseline: everything mismatched.
  - preempt: unreferenced.
- Bitflip: mismatched in all four groups on gsm8k and preempt.
- Offbyone: full-attention group only (M2).
- warm_differ 16 comes from batch composition:
  - every loaded block and state matched (local and external);
  - honest warm requests all start at 1056, as in baseline cold;
  - the tails (1056 to ~1150, plus decode) are computed, and no block in
    them is complete;
  - 7 of the 16 differing prompts are in prompt-length groups where every
    warm request was a local hit (no external load);
  - plain `alone` vs batched `split` differs 16/64 the same way.
- Recorded `vllm_cmd`: every gsm8k stage of every arm has `--compilation-config
  {"inductor_compile_config":{"deterministic":true}}` once, after
  `vllm_args`. Preempt stages lack it.
- Compile cache: baseline and connector arms share AOT `c3d69611f93c`; plain
  uses `2f978f6a27ea`.

Compile-determinism probe (`~/kexi/tmp/rsi-gate-hybrid-test/numerics`,
first-token logprob):
- `plain_det`, `plain_det_repeat` and `baseline_det` are equal on all
  logprobs (-0.385402).
- `plain_fresh` = `plain_fresh3` = `plain_load_fresh` = global `baseline` =
  -0.390268.
- Global `plain` = `baseline_fresh` = `baseline_load_fresh2` = -0.388626.
- vLLM `KVTransferConfig.compute_hash` has no factors
  (`config/kv_transfer.py:104`), and `vllm.py:541` appends it.

vLLM code read:
- Step order and precopy, for M1: `gpu_model_runner.py:3971/4091/4202`;
  `mamba_utils.py:1446-1550`; `kv_connector_model_runner_mixin.py:83-92`;
  `scheduler.py:1346-1348` (sync loads).
- Boundary offers: `single_type_kv_cache_manager.py:1887-2005`, including
  the drop on free; `scheduler.py:1457`.

Oracle: `test_qwen35_hit_needs_every_group` and
`test_checkpoint_needs_and_saves_only_block_ends` pin:
- `needed(Checkpoint(528), p) = (p/528 - 1, p/528)`;
- a state is saveable only when a step ends exactly on a boundary;
- a hit only where all three GDN groups and the full prefix are present.

This matches the precopy source block `(p - 1) // 528` for `p` on a boundary.

Tests and CI:
- `git archive ffdad48` to `~/kexi/tmp/review/rsi-gate-hybrid-review-ffdad48`.
  `pytest rsi/gates` with `CUDA_VISIBLE_DEVICES=""`: 224 passed in 44 s.
  The first try had 7 failures only because I had not exported
  `python/pyproject.toml`.
- `/tmp/ci-venv/bin/typos`: clean.
- `cz check --rev-range 1749353..HEAD`: 122 commits OK.
- `ruff check .` (0.14.14, the CI version): passes.
- `ruff format` would reformat 9 files; `workloads.py` was already
  unformatted on the base, and CI does not run `ruff format`.
- Per-group counts (M2 table) are from
  `digest_compare.compare(baseline/<wl>/probe, selftest-<arm>/<wl>/probe)`
  on the copied run. Honest gsm8k: groups 0-2 have local 225 and external
  30, group 3 has local 450 and external 60, all matched. Bitflip gsm8k
  external mismatched: 27/27 in groups 0-2, 27/54 in group 3.
- Accuracies (baseline cold = warm = 0.765625 in all three runs): honest
  warm 0.75 (5bc2e25), 0.7578 (cb57dd5), 0.7734 (aa42a44).

## Response (author, 2026-10-25)

Run evidence for every item is the self-test and Qwen3-4B slice listed at
the end (tags filled in when the runs finish).

| item | resolution |
|---|---|
| M1 | Start states go to `ProbeMetadata.start_states` and are digested in `RsiProbe.handle_preemptions`, before `_update_states`, the align precopy and a sync `start_load_kv`; offers stay in `wait_for_save`. New `selftest:sync` (honest bytes, `load_kv_async=False`, state-group models only) must be caught in every state group. Unit: `test_probe_steps.py::test_start_state_is_queued_for_before_the_forward`, `::test_worker_digests_start_states_before_the_forward`. |
| M2 | Negative arms are judged by `G2.caught_per_group`: every corrupted group (all transferred groups; `sync`: the state groups), on its own, needs a load equal to no reference (mismatched or unreferenced) in both gsm8k digest checks and a nonzero sabotage count. The connector writes per-group sabotage counts (`<stage>/selftest/sabotage-*.jsonl`, `result.sabotage`). offbyone's state sabotage loads the next state group's state of the same prefix (same shapes), independent of other saved prefixes; the `_neighbour` save-order logic is gone. GATES.md:977-978 rewritten. Units: `test_negative_arm_caught_in_every_group`, `test_selftest_sabotage_hits_every_group`. |
| S1 | GATES.md "Record vs gate rev": a record is judged by the rev it ran under. `G0.vllm_args` (no `vllm_cmd` anywhere), `G2.batch_control` (plain gsm8k without an `alone` pass), the committed `compilation` block (dump without one) and the sabotage count are `na: record predates this check` with `would_be`. Unit: `test_new_baseline_checks_are_na_for_older_records`. Qwen3-4B gsm8k slice (plain, baseline, connector_v2) on GPU5 below. DSV2 is on its own branch; it gets the same slice when that branch takes this gate. |
| S2 | Not widened. `G2.gsm8k` without batch invariance gates every arm, honest included, on `alone_ref` (64 prompts one at a time after reset + poison, now run by every arm) vs baseline `alone_ref`, ACC_TOL 0.01; batched deltas are reported (`batched_delta_pp`). Plain and baseline `alone_ref` already agree in every logprob (5bc2e25), so a correct load at the same composition gives the same accuracy and 0.01 is outside noise. The official-size spread measurement is not needed for this rule; if a correct arm's `alone_ref` ever differs, the run shows it. |
| S3 | `preempt` is required in the official profile when the committed spec has a Mamba group, and the honest self-test fails without it there (N6). GATES.md states the coverage limit: gsm8k loads one state key per group; cap and replay coverage on Qwen3.5 come with the P5 it2 official run. |
| S4 | `_flag_values` reads `--flag=value`, `_` for `-` (before any `.`), argparse abbreviations of the gate's flags, `-cc`, and dotted keys (appended last, as vLLM does). The probe dumps `compilation` (mode, backend, `inductor_compile_config`, `cudagraph_mode`) and both committed specs carry it (values from `EngineArgs.create_engine_config` on .44). Units: `test_gate_args_read_every_vllm_spelling`, `test_g0_spec_pins_the_compile_config`. |
| S5 | Per-arm `VLLM_CACHE_ROOT` = `<run>/.compile-cache/<label>`, emptied before the arm's first stage; `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` inside it; bound rw only into its own isolated arm. Per arm rather than "baseline populates, ro into the arm": the connector arms share a compile hash, so a shared ro artifact would still be one artifact for both sides; per arm, a tampered artifact only changes its own arm's numbers, which the baseline comparisons judge. plain == baseline with separate caches: self-test below. Unit: `test_each_arm_gets_its_own_compile_cache`. |
| S6 | Added (table-driven): `start_*` skips fail `_digest_verdict`, `preempt_gpu_blocks` sizing (incl. committed Qwen3.5 = 40), `_plain_equals_baseline` with equal completions and one logprob off, probe step logic (`_start_states`, `_offered_states`, `_unoffered`, `_preempt_job`, `_track`) in the new `test_probe_steps.py`. 282 tests. CI: workflows not edited; PROGRESS open question "add rsi/gates tests to CI" for the user's acceptance, and GATES.md says the suite is a manual gate until then. |
| N1 | Fixed: the older global plain artifact gives -0.388626, two fresh plain compiles both -0.390268. |
| N2 | `G2.split_control` reports `vacuous` and `measured_invariant: null` when `same_start >= 0.9 n` with no divergence; verdict unchanged (can only fail a right load). gate.json notes no longer say both claims are measured. |
| N3 | `not_offered` is recorded only when neither the step nor the next one offered the boundary (or the request finished in between). |
| N4 | `provenance.pegaflow_server` is `{"used": false}` and the watch skips it when no arm starts a server. |
| N5 | Not changed: the venv is shared with the P4 runs; the 056eb4b..d252bfe diff is `connector_v2` only and the self-test and gate do not import it. The slice's connector_v2 arm runs from the venv's 056eb4b and is a reference point for S1, not an iteration. |
| N6 | Enforced: honest self-test on a state-group model fails without `G2.preempt`. |
| N7 | Self-test arms now need `G0.spec` and `G0.provenance`: honest in its `need`, negative arms must pass G0 (caught by their loads, not by a layout difference). |
