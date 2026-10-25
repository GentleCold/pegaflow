# Review: P5 Qwen3.5-4B it2

Scope: `feat/rsi-qwen3.5-4b`, diff `origin/feat/semantics-rsi...02fa4ad`.
New in it2: H1 pinned saves (5667cd7), H2 lookup skip (2efc6da, 800630d),
SF3 hit-stat dedup (863cc99), the hit-stat logging commits (25fcccb, bc9c38a),
the P4 merge (554ad9d), the semantics-rsi merge (be0622f = 9a02fb5), and the
it2 docs. Official run:
`~/kexi/rsi-runs/2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04` on .44.

What I checked:
- the code diff in `python/pegaflow/connector_v2/**` and its tests;
- vLLM 0.32.0 in the shared venv: the BlockPool `touch`/`free_blocks`/
  `reset_prefix_cache`, `Scheduler.__init__` binding, `has_pending_push_work`,
  worker-meta build on both model runners (forward and no-forward), and
  `KVConnectorOutput` aggregation;
- `summary.json` of the official run: every verdict of every arm,
  `summary.hit_report`, `G5.curve`, scan and provenance;
- `rsi/gates/oracle.py` at HEAD. It is byte-identical to the c5d3f04 run tree:
  `oracle.py` and `run_gate.py` md5 match;
- the agent's `~/kexi/tmp/p5-g5-attr/` scripts and outputs. I also ran my own
  CPU-only checks in `~/kexi/tmp/review-p5-it2/` (`g5check.py`,
  `ownlocal.py`, `freeze.py`). They import the gate code and edit nothing;
- the generic pegaflow-core read cache (`storage/read_cache.rs`,
  `storage/prefetch.rs`), for H2's server-side effect;
- CI on a full `git archive HEAD` export in `~/kexi/tmp/review-p5-it2-ci`.

I did not read the old connector, except grepping it for the patterns
listed under "Isolation". I did not read the model branches in pegaflow-server
or pegaflow-core. No GPU job was run.

## Verdict: ACCEPT-WITH-FIXES

The connector changes are correct against vLLM 0.32. The official run is
genuine, and v2 passes every verdict. Every hit-rate number in ITERATIONS
reproduces from the record. The G5 oracle finding is correct, and it changes
no verdict. There is no must-fix.

Before this branch folds into `feat/semantics-rsi`:
- merge a4523f3;
- run a Qwen3-4B / DSV2 regression slice (S1). H2 is not quite the
  no-op the docs claim on full-only layouts.

## Verified

### H1: pinned saves (5667cd7)

Trigger:
- `_pin_saves` (connector.py:88-95) is `may_zero_or_copy_blocks or any
  checkpoint group`.
- Qwen3.5 has a checkpoint group and needs zeroing, so it pins.
- Qwen3-4B (FullAttention, block 16 = hash 16) and DSV2 (MLA, 16 = 16) have
  no zeroing and no checkpoint group. They stay on P4's async batched path.
  On the base they also had `sync_saves=False`.

Ordering:
- A save's blocks are pinned in `plan_saves` (scheduler.py:622-624) when the
  plan is built, before the step's output reaches the runner.
- `touch` raises `ref_cnt` and takes a free block out of the free queue. A
  block the request frees meanwhile therefore never re-enters the queue. It
  cannot be reallocated, zeroed (zeroing only applies to newly allocated
  blocks), or be a COW target until `free_blocks` returns it.
- A prefix hit on a pinned hashed block is read-only. This covers the V2
  runner's zero/COW before `handle_preemptions`, and the checkpoint offer
  freed right after the offering step (vLLM's own docstring: "must pin them
  first").
- The pinned path drops the preemption flush correctly (scheduler.py:528-538).
  The blocks are held anyway.

Release:
- `saved_through()` (worker.py:433-441) advances on completion, failed or
  not, because `_run_job` advances `_completed` in `finally`.
- A dead saver releases everything (`completed = _submitted`).
- Non-writer ranks report `_last_step`.
- `aggregate` takes the min over ranks (connector.py:77-85), and
  `on_saved_through` unpins steps up to it.
- Worker meta is built on no-forward steps by both runners:
  - `v1/worker/gpu/kv_connector.py:131-158` (`no_forward`, then `post_forward`);
  - `kv_connector_model_runner_mixin.py:25-37`, then `_get_kv_connector_output`.
- `KVConnectorOutput.is_empty` counts worker meta.
- `has_pending_push_work` keeps the engine stepping until the pins drain, so
  no pin can outlive the work.

Finished requests:
- Pinned layouts never add to `_saving`, so `request_finished` returns False
  and vLLM frees at once. The pin is the only extra reference, so nothing
  leaks and finished_sending has nothing to hold.
- Failed saves free the blocks: SKILL lesson "report finished_sending even
  on failure".

Bind:
- vLLM 0.32 calls `bind_kv_cache_manager` in `Scheduler.__init__`
  (`v1/core/sched/scheduler.py:360`), before the first
  `build_connector_meta`.
- The fail-closed branch (scheduler.py:528-534) is unreachable on 0.32, and
  it is safe if hit.

`reset_prefix_cache` with pins outstanding:
- `BlockPool.reset_prefix_cache` refuses while any block is used, and the
  gate retries 30 x 1 s (run_gate.py:1078-1084).
- Pins drain within a step or two. "Failed to reset prefix cache" appears 0
  times in every v2 `vllm.log` of the run.

### H2: lookup skip (2efc6da, 800630d)

- The bound is `semantics.resolve` over optimistic "everything present"
  answers (scheduler.py:272-279). It is Kind-generic and names no model.
  If `resolve` raises, the lookup queries as usual.
- Pre-query skip (281-283): no query, no lease.
- Post-group-0 skip (305-307):
  - runs only with more than one group;
  - releases group 0's lease before returning;
  - an exception mid-loop still releases (308-310).
  - No lease leaks.
- `_plan_hit` still runs on a skip, with empty `present`. So the lookup is
  counted, `p <= num_computed` gives no hit, and `request_finished` and save
  planning are untouched. Hit and save bookkeeping is unchanged.
- Full-only layouts:
  - only the pre-query skip can fire;
  - its bound is `floor((num_tokens-1)/bt)*bt`, and when that is
    `<= num_computed` the base code also returned 0;
  - the hit result is unchanged. The server-side effect is not (S1).

### SF3 (863cc99)

- `_counted[req_id] = (num_computed, p_full)` (scheduler.py:659-662)
  dedups vLLM's per-pass re-queries.
- It is popped in `request_finished` (717).
- It touches only stats.

### Namespace

- `QWEN3_NAMESPACE_V1` moved to `a25ea150` / `bdda8997` because P4 put
  `storage_tp` in the fingerprint (0a956a1, layout.py:153). That change was
  reviewed in `P4-dsv2-lite.md`.
- P5's own layout change appends `LayerDecl.layout` only when declared
  (layout.py:181), so full-attention and MLA namespaces do not move. The pin
  test passes.

### Isolation and policy

- connector_v2 has no import of `pegaflow.connector`, `rsi`, `RsiProbe` or
  `sys.modules`. The run's scan is clean: flagged [] and allowlisted [].
- Copying:
  - The old connector shares only vLLM API method names with v2.
  - My targeted grep of the old connector found the same vLLM mechanism (`pool.touch` / `pool.free_blocks`,
    `has_pending_push_work`, a `PegaWorkerMetadata` class name).
  - The behaviour differs. The old class carries a per-job completion-count
    dict that is summed. v2 carries one `saved_through` watermark that is
    min-aggregated, keyed by plan step.
  - The pattern is vLLM's own: the mooncake store scheduler
    `touch`/`free_blocks`, `OffloadingWorkerMetadata.aggregate`, and the
    Mamba offer docstring. ITERATIONS (lines 180-200, 445-453) cites those
    as the design source.
  - Not a copy.
- `policy.py`:
  - 119 lines, no model or spec class names;
  - 4 knobs (`loading_wait_ms`, `sync_load_tokens`, `save_batch_steps`,
    `save_batch_max_kv_usage`), all consumed (connector.py:199-204,
    scheduler.py:222/245/549);
  - only `default.toml`, no per-model toml;
  - P5 added no knob. The two batch knobs came with the P4 merge.

### Gate integrity

- The branch does not change the gate: `rsi/` and `GATES.md` are empty in
  merge-base..HEAD. HEAD only lacks a4523f3 (S2).
- Official: `not_official_because` [], `rev == under_test_rev == c5d3f04`,
  contamination [].
- Floor: 2.52x GPU KV (U 167.96 GB, C 83.98 GB, GPU KV 33.29 GB).
- gate.json: `--mamba-cache-mode align --prefix-cache-retention-interval 528`,
  `split_invariant` true (measured).
- v2 verdicts, all pass:
  - G0.spec: 0 diffs.
  - Digests, local + external matched:

    | workload | local | external |
    |---|---|---|
    | gsm8k | 13460 | 45/45 (also vs baseline) |
    | preempt | 311 | 106/106 |
    | replay | 456 | 540/540 |
    | cap | 15240 | 59627/59627 |

  - `G2.gsm8k` alone_ref delta 0.0.
  - G3 E_ext 1.0, no_eviction 0 everywhere.
  - G4:
    - replay warm TTFT p50 0.300/0.295 s vs baseline 1.112/1.119 s;
    - replay cold TTFT p50 0.512 s vs 1.204 s;
    - req/s warm 14.61/14.66 vs 3.73; cold 8.19 vs 3.61;
    - gsm8k cold 95.00 vs 94.68 s (+0.34%), warm 94.55 vs 94.88 s.
  - G5.replay_warm: pass. excess 0, load errors 0, pool_ok, loads_ok,
    unmatched 0.
- old passes. native fails G0.spec, digest vs baseline and G4 gsm8k cold. It
  is a reference arm.
- `gsm8k_outputs` is not gated (split-invariant model). No verdict is
  relaxed.

## Hit rates (verified)

Recomputed from the probe traces with `oracle.summarize_capacity` and the
baseline reference locals (timed passes cap_warm2 + cap_warm3: 3716
requests, 17,483,836 prompt tokens, ref local 0.2484).

Unlimited pool:
- G3 replay warm: hit 0.946 = H*, for v2, old and native.
- G3 gsm8k warm: hit 0.926 = H*.
- E_ext 1.0 for every arm.

Capacity (H*_C, H_lru and H_freeze on v2's trace):

| C/U | H*_C | H_lru | H_freeze | hit v2 / old / native | E_C v2 / old / native | v2 E_C with own-local added back |
|---|---|---|---|---|---|---|
| 0.25 | 0.6900 | 0.7937 | 0.4506 | 0.7939 / 0.7940 / 0.7939 | 1.1831 / 1.2348 / 1.1754 | 1.2353 |
| 0.5 (gated C) | 0.8779 | 0.7919 | 0.6686 | 0.7939 / 0.7940 / 0.7884 | 0.8277 / 0.8667 / 0.8150 | 0.8666 |
| 0.75 | 0.9457 | 0.9064 | 0.8851 | 0.9066 / 0.9061 / 0.9457 | 0.8800 / 0.9431 / 0.9278 | 0.9438 |

These match `summary.hit_report`, the G5 verdicts and the ITERATIONS
tables.

Own-local arithmetic, checked per request:
- ext loses exactly `Σ max(0, own − ref)` for v2 (403,392 / 428,208 /
  777,744 tokens). old's own local hit is 0, so it loses nothing.
- Adding v2's loss back gives 1.2353 / 0.8666 / 0.9438, against old's
  1.2348 / 0.8667 / 0.9431, within 0.001.
- The claim "E_C 0.828 vs old 0.867 is v2's own GPU-local hits above the
  reference" is correct.
- Native's add-back is 1.2351 / 0.8582 / 1.0000.

At 0.5, v2's hit equals old's and equals H_lru. The gap to H*_C is the base
server's plain LRU on a cyclic scan. I accept the "close enough" judgment.

## G5 oracle finding: correct; no verdict affected

Claim: Belady's next-use sets are built only at each request's unlimited
resume point P (oracle.py:380-385). A Checkpoint's `needed(q)` for q < P is
not in `needed(P)`, so fallback checkpoints look unused, and H*_C is
understated.

My independent check (`~/kexi/tmp/review-p5-it2/g5check.py`, v2 trace):
- `needed(q) ⊆ needed(P)` for every block end q < P holds for the full
  group and fails for the three checkpoint groups.
- Every request where H_lru > H*_C resumes in LRU below its unlimited P:
  995/995 at 0.25 and 239/239 at 0.5. There are none at 0.75.
- The blocks LRU loads at those requests:
  - checkpoint, 0.25: 2985 blocks, 0 with a use at that request; median next
    use 1347 requests later (about one pass); 1887 never used again;
  - checkpoint, 0.5: 717 blocks, 0 with a use at that request; median next
    use 1407 requests later;
  - full: 5585 / 5585 have a use at the request (median next use 32
    requests).
- Belady therefore declines or evicts exactly the blocks a fallback needs.
  The agent's no-LRU-knowledge counterfactual (`attr2.py`, uses at every
  q ≤ P) puts the bar above H_lru on every request.

Effect:
- No verdict changes. E_C is reported, not gated
  (`_capacity_verdict`, run_gate.py:2201-2212), the curve is never gated,
  and GATES.md allows E_C > 1.
- The finding also applies at the gated fraction. 239 requests at 0.5 make
  H*_C slightly low, so v2's E_C 0.828 is optimistic against a
  fallback-aware bar (0.813).
- The finding is in the gate, not the connector. I endorse the
  suggestion: report the max of the two Belady variants, or make the use set
  fallback-aware for Kinds where `needed(q) ⊄ needed(P)`. It is a follow-up
  for the gate owner, outside this branch.

## Regressions (Qwen3-4B, DSV2)

- H1: not active on either model (above), so unchanged.
- Namespace: unchanged by P5 code. The pin test passes.
- SF3: stats only.
- H2: the hit result is unchanged, but server recency is not (S1). No
  official run of either model exists at a post-H2 head. A regression
  slice is needed.

## CI (HEAD 02fa4ad, full `git archive` export on .44)

- typos: clean.
- ruff 0.14.14 (CI's pinned version):
  - `ruff check` (CI scope): all passed.
  - `ruff check --config python/pyproject.toml python rsi`: 6 I001 in
    `python/tests`. That includes `test_connector_v2_core.py:11`, which has
    the same I001 on the base file, so it predates this branch.
  - `ruff format --check --config ...`: 11 files would be reformatted, all
    untouched by this branch. The 25 files the branch touches, plus `rsi/`,
    are formatted.
- `cz check --rev-range origin/feat/semantics-rsi..HEAD`: successful,
  merges included.
- Default Python unit gate, shared venv, no native extension: 502 passed,
  88 skipped, 1 failed. With the native extension copied in: 623 passed,
  1 failed.
  - The failure is `test_e2e_version_preflight::test_preflight_uses_vllm_interpreter_instead_of_pytest_interpreter`.
  - It is the known host failure recorded in P3 and P5 it0; the branch does
    not touch it.
- Source-only gate (the CI job): 504 passed, 87 skipped, 0 failed.
- `test_connector_v2_core.py`: 236 passed, stub and native.
- Gate pytest (`rsi/gates`, native extension from the run tree): 312 passed.

## Must-fix

None.

## Should-fix

S1. H2 changes server LRU order on full-only layouts too. A Qwen3-4B / DSV2
regression slice is needed.
- ITERATIONS.md:752-756 says "the only change is one query and one lease
  fewer". But a prefix query refreshes the recency of every block it hits:
  `prefetch.rs:327` calls `ReadCache::get_prefix_blocks`, which calls
  `refresh_recency` (read_cache.rs:128-137).
- Before H2, vLLM's per-pass re-queries of a waiting request refreshed the
  blocks vLLM already held locally. After H2 (scheduler.py:281-283) they do
  not.
- Server eviction order, and so G5 hit at capacity, can move on Qwen3-4B and
  DSV2. It is likely neutral or better, but unmeasured.
- Fix:
  - correct the doc;
  - before folding into `feat/semantics-rsi`, run at least gsm8k (G4) plus
    an official-size cap (G5) slice at a post-H2 head for each model, and
    compare hit and E_C with their accepted runs.

S2. Merge a4523f3 (the lmcache gate fixes) into the branch before folding.
- `git merge-tree` shows one content conflict, `docs/rsi/PROGRESS.md`.
- No verdict of this run should move. The new resume stage applies only to
  split-variant models (`resume_stage_needed`), and gate.json measures
  `split_invariant` true.
- The lmcache arm is still pending (ITERATIONS.md:989). Record it once it
  runs.

S3. `Worker.sync_saves` is now dead in production, and its docstring is
wrong.
- The connector no longer passes it (connector.py:195-217).
- The module and constructor docstrings (worker.py:20-26, 126-131) say it
  "remains for callers without a block pool". But a pinned layout without a
  pool disables saves (scheduler.py:528-534); it never falls back to sync
  saves.
- Fix: delete `sync_saves` (worker.py:126, 156, 370, 424-425, 444) and its
  test (`test_connector_v2_core.py:1440`), or wire it in as that fallback.

## Nit

- ITERATIONS.md:891-898 "v2's own GPU-local hits above the baseline
  reference" is per request. In aggregate, v2's own local rate (0.182) is
  below the reference (0.248). Say "on requests where own > ref" to avoid
  misreading.
- `PegaWorkerMetadata` (connector.py:76) shares its name with an unrelated
  old-connector class. Its semantics differ (watermark vs job counts), and
  it is fine as is. A distinct name such as `SavedThroughMetadata` would
  avoid confusion in mixed logs and tracebacks.
- Pinned blocks raise `ref_cnt`. vLLM's `is_block_writable` (ref_cnt == 1)
  is used only for speculative Mamba blocks, and spec decode is out of
  scope here. Note it in the README before enabling spec decode on a
  checkpoint layout.
