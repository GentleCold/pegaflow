# Review: P4 DeepSeek-V2-Lite-Chat onboarding (`feat/rsi-deepseek-v2-lite`)

Reviewer: independent agent, fresh context, 2026-10-24. Range: the brief says
b358ceb..3095e0c, but the branch was rebased. b358ceb is the pre-rebase copy
of c330abd, so the reviewed DSV2 diff is `8462ed9..3095e0c` (8462ed9 = merge
base with `feat/semantics-rsi`). Qwen3-4B acceptance reference: 8a05b7c, official
run `2026-10-23-qwen3-4b-p3-it2d`.

What I checked:
- the code diff (adapter, `StorageRanks`, worker batching, KV-pressure send,
  policy/toml);
- vLLM v0.32.0 source, fetched into /tmp: `kv_cache_interface.py`,
  `mla_attention.py`, `sched/scheduler.py`, `block_pool.py`,
  `kv_connector_model_runner_mixin.py`, `gpu_model_runner.py`, `multi_connector.py`;
- every `summary.json` the ITERATIONS file cites, on .44, read-only;
- the remote feature tree against HEAD (md5: byte-identical for
  `connector_v2/**` and `rsi/gates/**`), `p4probe.py`, and the held-out file;
- local stub tests: `test_connector_v2_core.py`, 115 passed, 60 skipped
  (the native cases need the remote venv).

Isolation: I did not open `python/pegaflow/connector/**`, the pd/nixl
connectors, `service.rs`, `instance.rs`/`instance/`, `offload.rs`, or model
history. The diff touches none of them (see S4 for one citation).

## Findings that hold

- MLA adapter (`adapters/mla_attention.py`) is derivable from vLLM 0.32:
  - `MLAAttention.get_kv_cache_spec` hard-codes `num_kv_heads=1`,
    `max_tp_shards=1` (mla_attention.py:1356-1357);
  - `MLAAttentionSpec.head_size_v = 0`, so the page is 16 x 576 x 2 = 18432 B,
    one segment.
  - `Shard.replicated()` with writer = first holder is what seal_plan gives.
  - The fail-closed list covers every MLA-only field of the 0.32 spec:
    `alignment`, `model_version`, `cache_role` (a str enum, compared by value),
    `is_index_group_leader`, `non_causal_multi_token_decode`,
    `cache_dtype_str`, `head_size_v`, plus the inherited Full fields.
  - Namespace: the spec class name and `storage_tp` are in the fingerprint.
    `test_mla_and_full_attention_never_share_a_namespace` is the "different
    bytes, different namespace" test (equal 18432 B pages).
- Single writer: only `writer[tp_rank]` builds save entries (worker.py:345).
  Both ranks register storage rank 0 with tp_size 1 and load from it. A
  non-writer has `_submitted == 0`, so it reports `finished_sending`
  immediately, and vLLM's `KVOutputAggregator` frees blocks only after every
  rank reports. So the writer's fence decides when blocks are freed. The it2
  slice shows 497664 B per block (27 x 18432) on the server, equal to the old
  arm. Write amplification is gone.
- Batching lifecycle (worker.py:332-417):
  - Held jobs own their sequence numbers before release. A request finishing
    while its saves are held gets fence = `_submitted` (worker.py:240), so it
    is reported only after the merged job (the last seq, the last compute
    mark) completes. Blocks cannot be freed or reused before the copy lands.
  - Preemption: `handle_preemptions` runs before `_update_states` and the
    forward (gpu_model_runner.py:3971), and `flush` releases held jobs first.
  - Idle engine: `has_finished_requests()` keeps vLLM stepping while
    finished-but-held requests exist (scheduler.py:2813-2821), and
    `kv_connector_no_forward` still calls `wait_for_save`, so held jobs age
    (`test_save_batching[idle-steps-count]`).
  - Save failure: `_run_job`'s `finally` advances `_completed` to the merged
    seq, so every fence is released. The cost is a bigger miss (up to 32
    steps).
  - Shutdown: `flush(30 s)` releases held jobs first.
  - KV-pressure send: `KVCacheManager.usage` = 1 - free/total, with free
    including evictable cached blocks. So it measures blocks pinned by
    requests, held ones included, which is the right signal.
- Sync-save layouts skip batching (`_release_due` returns True when
  `sync_saves`), tested by `test_layouts_that_zero_or_copy_save_synchronously`.
  It sets 32 steps / 1 s and asserts one RPC in the same step for sync and a
  held job for async.
- Gates are not relaxed: `8462ed9..3095e0c` touches no file under `rsi/` and
  not `GATES.md`. `gate.json` adds `split_invariant: false`, backed by the
  plain-vLLM split control (`2026-10-23-dsv2-lite-split-control`: 647/1319
  differ, every logprob differs, 1319 prewarmed, `ref_equals_cold`).
- Evidence: every G4, hit and RPC number I checked in ITERATIONS matches its
  summary.json. That covers it5-rep-01, press-01, press-02, heldout,
  sweep-01, def-01, it4 rep-01, knob-01, qwen it4-nonreg, it5-nonreg and
  qrep-b. Example: it5-rep-01 KD/KDb/KDc cold p50 0.1916/0.1876/0.1867 vs
  0.1904, req/s 5.671/5.666/5.679 vs 5.696; KOff 0.2000 / 5.581, F/F.
  Warm hits are 108528/153852 local, 45024 external, H* 0.998, E 1.0.
- Held-out slice is disjoint:
  - `r2egym-heldout-skip8.jsonl` md5 8fd2884a… equals `tail -n +9
    r2egym-256.jsonl`. `build_turns` slices `lines[skip:skip+n]` before
    filtering, so tuning = lines 1-8 and held-out = lines 9-16.
  - The two slices share no lines and no first user message. They share the
    system prompt, but no run mixes slices.
- Policy rules:
  - `policy.py` names no model or spec class.
  - 6 knobs (5 after rebasing past fc06c9b), all in `default.toml`.
  - 0 lines of per-Kind policy, 121 lines total.
  - No `models/<model>.toml`; the DSV2 and Qwen3 settings are identical.
- Hit tables (local / external / total, H*, E_ext) are present for every run.
  G5 is pending: the gate is not merged. That is not a failure.

## Must-fix

M1. No DSV2 run of the shipped default has been judged as GATES defines
split_invariant=false. G2/G3 gsm8k and G4.gsm8k are unmeasured at HEAD.
(`docs/rsi/models/DeepSeek-V2-Lite-Chat/ITERATIONS.md:96-100`,
`rsi/gates/run_gate.py:636-641`, `docs/rsi/GATES.md:214-221`)
- On this branch, GATES.md and run_gate.py have no split mode. The
  `split_invariant` definition is on unmerged `feat/rsi-g5`:
  `MIN_COMPUTED_COVERAGE = 0.9`, judged == 0 fails, zero unjudged warm blocks
  on non-aligned prompts, warm outputs vs the plain `split` pass.
- Every DSV2 connector run with gsm8k fails `G2.gsm8k_outputs` (647 differ)
  and `digest_vs_baseline.gsm8k` (27526/39042). Those runs are it1/it2 slice
  and `p5-it1-nonreg-b`, all from before batching.
- Every run from it3 on is `--workloads replay`. G0.spec, G2.digest_*.gsm8k,
  G2.gsm8k, G3.gsm8k_warm and G4.gsm8k_{cold,warm}.seconds read "not
  evaluated".
- No run is official (`official: false` in all of them). ITERATIONS states
  this honestly ("The official run waits for the merge").
- Failure scenario: it3b shows that save submissions on a TP2 pair slow the
  NCCL SHM all-reduce. Batching changes the submission cadence on gsm8k too,
  where every request saves its suffix. The last DSV2 G4.gsm8k (it2, 3d1de18:
  +0.2% / -2.3%) predates batching, so a >2% gsm8k regression at HEAD would go
  unseen.
- Fix: after `feat/rsi-g5` merges, rebase and run one official DSV2 run
  (plain, baseline, `pegaflow.connector_v2:PegaKVConnectorV2`; the cap
  workload too, if rsi-g5 makes it official). Accept on its verdicts:
  `G2.split_control` pass, computed coverage >= 0.9 per pass, zero unjudged
  non-aligned warm blocks, warm == plain split, G4 gsm8k within 2%. Paste the
  hit table (local / ext / total / H* / E_ext; H*_C / H_lru / E_C) next to the
  G4 table, plus `pressure_rpcs` and the cold hold. The official 32x12 replay
  is KV-pressured, see M2.

M2. The Qwen3-4B non-regression evidence was measured in a different regime
from the accepted run. (`python/pegaflow/connector_v2/models/default.toml:27-33`;
ITERATIONS.md:803-856, 1030-1049)
- 3e6f7a3 turns batching on for every model, Qwen3-4B included.
- The Qwen3 nonreg runs (`qwen3-4b-p4-it4-nonreg`, `-it5-nonreg`,
  `-it4-qrep-b`) pass every evaluated verdict, but at 8x6 with max KV usage
  31%. All are non-official (8x6, and `gsm8k_limit=1319`).
- The accepted run `2026-10-23-qwen3-4b-p3-it2d` used the official 32x12
  replay and reached 99.6% GPU KV usage (baseline 96.9%; from its vllm.log).
  That is the regime where batching hurt:
  - press-01: +53% cold p50 vs batching off without the pressure send;
  - press-02: -2 to -3% cold req/s vs batching off even with it.
- Failure scenario: in that regime the official Qwen3 run at HEAD lands >5%
  behind it2d's v2 numbers (warm p50 0.563/0.537, cold p50 1.287, cold req/s
  6.70). That violates "no >5% regression vs last accepted iteration", and no
  slice run can show it.
- Fix: run Qwen3-4B with the official constants on the rebased HEAD and
  compare with it2d per verdict.

M3. `save_batch_ms` has no gate evidence. Fold it into a worker constant
(or delete it). (`policy.py:57`, `default.toml:28`, `connector.py:177,190`,
`core/worker.py:101,145`)
- The skill rule is "every knob needs gate evidence that it helps".
- No run changed it alone and moved a verdict: sweep-01 KB32 vs KB32T1000
  are both P/P, 50 vs 51 RPCs.
- It never even bound the hold. Max cold hold is 1030-1121 ms for KB32 with
  no time bound, and 1066-1106 ms with 1000 ms (it5-rep-01, sweep-01, Qwen3
  qrep-b). The bound is checked once per step, so it cannot cap below about
  one step past 1 s.
- The case it targets (slow steps holding finished requests' blocks) is now
  covered by `save_batch_max_kv_usage`, which has evidence (press-01 vs
  press-02).
- What remains is "bounded staleness of saved blocks for other requests".
  That is a liveness bound, not a tuned policy.
- Fix: replace the knob with a `Worker` class constant, like
  `SHUTDOWN_FLUSH_S`, set to 1.0 s and documented as a staleness cap. Or drop
  it. A 1.0 s constant changes no behaviour, so the tests suffice and no gate
  rerun is needed. Also fix the `default.toml:23-24` comment: the hold can
  exceed the bound by about one step.

## Should-fix

S1. A merged save RPC has no size bound. (`core/worker.py:396-417`;
gRPC limit 64 MiB at `python/src/lib.rs:226`, `pegaflow-server/src/lib.rs:724`)
- One merged job repeats every held block id and hash in every layer entry,
  about 37 B per (layer, block) on the wire.
- At the gate's pinned `--max-num-batched-tokens 2048` it is at most
  128 x 32 x 27 x 37 B ≈ 4 MB.
- Failure scenario: `--max-num-batched-tokens 16384` on a 61-layer MLA model
  gives 1024 x 32 x 61 x 37 B ≈ 74 MB per merged RPC. Every merged save then
  fails with a message-size error, logged only as a warning ("save failed",
  `failed += 1`). External hits drop to about zero, and no gate config
  catches it. Unbatched, one step is about 2.3 MB.
- Fix: release when held (layer, block) entries x the per-entry size pass a
  budget well under the limit. Add a unit case.

S2. Run provenance cannot be checked.
(ITERATIONS.md:970, 993-996, 1030, 1051)
- `.rsi-rev` holds free-form labels (`it4-wip`, `it4-default-wip`,
  `it5-kv090`), not shas.
- ITERATIONS says the it5-rep-01 tree lacks the later order fix. Yet the
  remote `worker.py` now equals HEAD byte for byte, and its mtime
  (00:54:08Z, preserved by tar) predates that run's start (00:54:26Z). Which
  tree ran cannot be told.
- The default arm in it5-rep-01, press-01/02 and qrep-b is `p4probe:KD`, a
  no-op subclass in a remote-only module (`~/kexi/tmp/dsv2-p4/p4probe.py:287-297`,
  checked). It is fine for probes, but the official run must use the real
  class.
- Fix: write the commit sha (plus `-dirty` and a diff hash) into `.rsi-rev`,
  or use rsi-g5's `--under-test-path` snapshot.

S3. PROGRESS overstates the it5 result. (`docs/rsi/PROGRESS.md:606-607`)
- "(G4 passes 3/3)" should read "G4 replay verdicts pass 3/3 (replay-only,
  non-official; gsm8k G4 not evaluated)". ITERATIONS.md:988 already words it
  correctly.
- A reader of PROGRESS could take P4 G4 as met.

S4. The registration contract is cited from an isolation-listed module.
(ITERATIONS.md:253-257)
- It cites pegaflow-core's `mla_replica_registration_seals`, an MLA-named
  test in `pegaflow-core/src/instance/tests.rs`, i.e. the instance slot
  logic on the isolation list. I did not open it.
- The connector code does not depend on it. `StorageRanks` comes from
  seal_plan holders/writer and the public `register_context_batch(tp_rank,
  tp_size)`. The it2 gate log confirms it: `storage_rank=0 writer=True/False`,
  27 x 18432 B per block on the server.
- Fix: replace the citation with that derivation, and state whether instance
  code was read.

S5. The branch is behind `feat/semantics-rsi`.
- fc06c9b (drop `min_hit_tokens`; async loads after a failed sync load) is
  not on this branch.
- `git merge-tree` auto-merges all code, with a conflict only in
  `PROGRESS.md`. The merged tree (fc06c9b's scheduler change plus batching
  plus storage ranks) has never been tested or run.
- Fix: rebase first, rerun the stub and native tests, then do M1/M2 on that
  tree. The budget table becomes 5 knobs.

S6. The KV-usage read fails silently. (`connector.py:233-239`)
- `getattr(self, "_kv_cache_manager", None)` falls back to 0.0. If vLLM does
  not bind the manager (another wrapper, or a vLLM upgrade renames the
  private field), the pressure send is silently off. That is the press-01
  regime: +53% cold p50 vs batching off, +8% vs baseline.
- Fix: log once at the first plan when it is None, and treat unknown usage as
  pressured (send every step = batching off). Add a unit case.

## Nit

N1. Byte identity of replicas was validated only on .44, where the 4090s
have no P2P and use NCCL SHM all-reduce (`gate.json` notes; ITERATIONS.md:30).
Rank 1 loads rank 0's latent. On a host with custom or symmetric-memory
all-reduce, rerun a TP2 probe with per-rank digests once.

N2. The held-out run uses `--replay-file`, which this branch's
`official_problems` does not flag. Its `not_official_because` omits it; the
run is non-official for other reasons. On rsi-g5 use `--replay-skip 8`,
which records `held_out: true`.

N3. The policy budget table (ITERATIONS.md:1000-1003) lists `min_hit_tokens`,
which upstream deleted.

N4. The it5-rep-01 and held-out hit numbers are prose, not tables. Skill
step 7 wants the table next to the G4 table at acceptance.

N5. No unit case covers a merged job failing with several finished requests
fenced on different held seqs. The code is correct (`_completed` jumps to
the last seq), but the batching tests only use successful saves.

## Verdict

accept after must-fix: M1 (official DSV2 run under rsi-g5's split mode,
including gsm8k G2/G3/G4), M2 (official Qwen3-4B non-regression vs it2d at
the rebased HEAD), M3 (fold `save_batch_ms` into a constant). G5 is pending
the gate merge.
