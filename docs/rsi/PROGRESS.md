# RSI: vLLM connector + PegaFlow — progress log

This file is the source of truth for the RSI effort. Any session (human or agent)
that resumes this work MUST read this file first, then `docs/rsi/GATES.md`, then
the skill `.claude/skills/pegaflow-new-model/SKILL.md`. Append, do not rewrite history.

Design doc: Feishu "Offloading rsi"
https://ppio-cloud.feishu.cn/wiki/XRmWwEfWjiIUHok6coIcz34fnpf (rev 62 at start).

## Scope (fixed by the user, 2026-10-23)

- Build the vLLM connector + PegaFlow path of the design; onboard models one by
  one in RSI mode.
- Models, in order: Qwen3-4B, DeepSeek-V2-Lite-Chat, Qwen3.5-4B, gemma-4-E4B-it.
- vLLM: latest release (0.32.0). PD disaggregation: out of scope.
- New semantics (Kind/Shard/...) may be added without asking; every addition is
  logged in "Semantics changelog" below.
- An independent review agent checks every iteration.
- Onboarding agents must NOT read or copy the old connector implementation
  (see "Isolation"). Old connector is a black-box baseline only.
- Every step is recorded here so a lost context never derails the RSI loop.

## Layering (from the design doc)

| Layer | Language | Who changes it |
|---|---|---|
| adapter (per KVCacheSpec type) | Python | onboarding agent |
| policy (what to save/load/prefetch) | Python | onboarding agent (main RSI surface) |
| semantics (`pegaflow-semantics`: needed/resolve/saveable/key/seal_plan) | Rust, PyO3 | agent proposes, review agent + human audit |
| server (storage only) | Rust | not model-aware |
| gates (`docs/rsi/GATES.md`, `rsi/gates/`) | — | human only; agents may add checks, never relax |

## Isolation (no reference to existing implementation)

Forbidden for onboarding agents (read or grep):
- `python/pegaflow/connector/**` (old connector)
- model-specific branches in server: `pegaflow-server/src/service.rs` query
  branches, `pegaflow-core/src/instance.rs` slot math, `offload.rs` page_first
- git history/PRs of model adaptations (e.g. Gemma4 #455, K3)

Allowed: `pegaflow-proto`, `pegaflow-semantics`, the new connector package,
vLLM 0.32 source, `docs/rsi/**`, gate outputs, the Feishu design doc.

The main session has already read the old connector, so it only orchestrates;
model onboarding and optimization are done by fresh sub-agents with the
forbidden list in their prompt. The review agent checks for copying (structure,
names, magic constants that only exist in the old code).

## Environment

- Remote: 10.121.196.3 (8x4090, CUDA 13.0, driver 580); everything under `~/kexi/`.
- Shared venv `~/kexi/pegaflow/python/.venv` (py3.12, vLLM 0.32.0, torch 2.13.0,
  flashinfer-python 0.7.0.post1). No per-branch venv.
- Feature dir `~/kexi/pegaflow-semantics-rsi`, sync via
  `.local/bin/pegaflow-remote-sync.sh --ip 10.121.196.3 --feature semantics-rsi`.
- Build: maturin with `cuda-13,rdma`, `PYO3_PYTHON` = shared venv python.
- Models: `/root/kexi/models/{Qwen3-4B,DeepSeek-V2-Lite-Chat,Qwen3.5-4B,gemma-4-E4B-it}`.
- Logs `~/kexi/logs`, run records `~/kexi/rsi-runs`.
- Local worktree `.worktrees/semantics-rsi`, branch `feat/semantics-rsi`
  (base upstream/master 1749353), trellis task `.trellis/tasks/10-23-semantics-rsi`.

## Phases

| Phase | Content | Status |
|---|---|---|
| P0 | gates: spec dump, V3 digest, gsm8k, perf workload, hit-rate metrics, old-connector baseline | in progress |
| P1 | `pegaflow-semantics` crate + PyO3 + V1 vectors | done (review: `reviews/P1-semantics.md`) |
| P2 | new connector skeleton (core glue + adapters + policy + models), coexists with old | done, review fixes applied (`reviews/P2-connector-skeleton.md` Resolution) |
| P3 | onboard Qwen3-4B | it1 code done, GPU validation pending (.3 unreachable) |
| P4 | onboard DeepSeek-V2-Lite-Chat | todo |
| P5 | onboard Qwen3.5-4B | todo |
| P6 | onboard gemma-4-E4B-it | todo |

## Log

### 2026-10-23

- Worktree + remote env set up; vLLM upgraded to 0.32.0 in shared venv.
- Incident: `rm -rf /ppio/kexi` deleted an old `gds` dir and in-flight model
  downloads (outside `~/kexi`). Rule: `ls` before delete; only write under `~/kexi`.
- Models: Qwen3-4B, Qwen3.5-4B done; gemma-4-E4B-it was still downloading.
- Old-connector baseline (Qwen3-4B e2e) failed at vLLM startup:
  `flashinfer-cubin (0.6.12) does not match flashinfer (0.7.0.post1)`.
  No 0.7.x cubin on PyPI and vLLM 0.32 does not require it -> uninstalled the
  stale `flashinfer-cubin`. Rerun log: `~/kexi/rsi-runs/baseline-old-qwen3.log`.
- Second failure: pegaflow-server (cuda-13 build) picked `libcudart.so` 12.5 via
  ldconfig. Fix: `LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib`
  for every gate run on .3.
- All 4 models downloaded (DeepSeek-V2-Lite-Chat 30G, gemma-4-E4B-it 15G).
- Old-connector black-box baseline, Qwen3-4B e2e (`~/kexi/rsi-runs/baseline-old-qwen3-full.log`):
  4 passed / 1 failed. Outputs match baseline; saves 74 blocks (174.6MB), hits
  40 blocks (94.4MB) in the restart pass (vLLM "External prefix cache hit rate"
  71.1%). FAILED `test_same_process_hma_load_uses_pegaflow`: the warm repeat in
  the same vLLM process (prefix caching off) got hit_delta=0, load_delta=0.
  Recorded as a known baseline gap; the new connector must pass it (G2).
- Skill `.claude/skills/pegaflow-new-model/SKILL.md` created.
- Gate metric plan drafted in `docs/rsi/GATES.md`.
- User: "继续任务" without answering what agentx is -> keep the R2EGym fallback
  for G4 until told otherwise.
- P1 code: `pegaflow-semantics` crate (groups/keys/sharding), shared JSON vectors
  (`pegaflow-semantics/vectors/`) replayed by `tests/vectors.rs` and
  `python/tests/test_semantics_bindings.py`; PyO3 submodule
  `pegaflow.semantics` (`python/src/semantics.rs`, `semantics.py`, `.pyi`).
  Remote: 12 Rust tests pass, clippy clean (fixed one `is_multiple_of` lint).
  Independent semantics review agent launched.
- P1 review: APPROVE-WITH-FIXES, 6 should-fix + nits, all resolved or deferred
  with reason in `docs/rsi/reviews/P1-semantics.md`. Notable: `Group.window`
  arg order now `(block_tokens, tokens)` per design doc; `Group::new` fallible;
  scratch excluded from lcm step; seal_plan requires equal segments across all
  parts; `pegaflow-common::group_hash` now delegates to `semantics::key`.
  Evidence: semantics 16 tests, common 48, bindings 35, workspace clippy clean.
- Next: P0 gate tooling (spec dump, V3 digest, gsm8k runner, hit-rate/oracle),
  then P2 connector skeleton.
- P0 tooling written under `rsi/gates/` (gate-owned, agents read-only):
  - `rsi_probe.py`: `RsiProbe(KVConnectorBase_V1, SupportsHMA)`, never loads
    or saves. Scheduler: `spec.json` (kv_cache_config, spec fields + derived
    page sizes, tensors), `requests.jsonl` (prompt_len, local_hit from
    `get_num_new_matched_tokens`, first_computed = local + external at first
    schedule, first/finish scheduler step, prefix-key chain at gcd(block_size)).
    Worker: `spec-pp*-tp*.json` (per-layer view shape/stride/dtype) and V3
    digests of every fully computed block of finished requests (sampled by
    prompt hash), addressed via `KVCacheTensor` geometry on the raw backing
    buffer, so Mamba/raw specs are covered without view assumptions.
    Digest = two 64-bit weighted byte sums + float L1, computed on GPU.
  - `digest_compare.py`: tags blocks `hit` (covered <= first_computed), refs
    are the first computed occurrence; hit blocks must be byte-identical.
  - `oracle.py`: replays `requests.jsonl` through `semantics.resolve` with an
    unlimited store (saved at finish, visible from the next step); Kinds come
    from the spec dump (Full / Window{sw-1} / Checkpoint for Mamba), not from
    the connector under test.
  - `workloads.py`: gsm8k 8-shot (fixed train[:8] prefix), R2EGym multi-turn
    replay (token-id prompts, streaming TTFT), Prometheus scrape + diff.
  - `run_gate.py`: arms baseline / old / `<module>:<Class>`; per arm+workload
    fresh PegaFlow + vLLM (probe alone or MultiConnector[under_test, probe]),
    cold pass, local `reset_prefix_cache`, warm pass; writes
    `~/kexi/rsi-runs/<date>-<tag>/summary.json` with G2/G3/G4 verdicts.
    gsm8k runs with `VLLM_BATCH_INVARIANT=1` + FLASH_ATTN for byte comparison.
- Probe smoke (Qwen3-4B, .3 GPU0): 601-token prompt twice with prefix cache:
  2nd request first_computed=592, 38 blocks x 36 layers digested, hit blocks
  identical. Baseline without prefix cache (no batch invariance): 1333/1368
  layer-blocks byte-equal, max rel L1 1.9e-3, so cross-run byte equality
  needs batch-invariant mode (now default for gsm8k).
- Gate smoke (Qwen3-4B, 64 gsm8k q + 8x6 replay, baseline + old arms, run dirs
  `~/kexi/rsi-runs/2026-10-23-qwen3-4b-smoke{,-replay}`): all verdicts pass for
  the old connector. gsm8k 0.859 everywhere; old cold/warm completions identical
  to baseline (64/64); digest_vs_baseline 306432 hit layer-blocks, 0 mismatched,
  37296/37296 computed exact under batch invariance; old warm E=1.0008
  (external hit rate 0.946); replay old warm E=1.0004, TTFT p50 0.132s vs 0.209s.
- Oracle fix found by the smoke: the first version made blocks visible only at
  finish, so baseline cold scored E=1.97 (vLLM reuses the 8-shot prefix across
  concurrent requests). Now prompt blocks are visible after the first schedule
  step, generated blocks after finish; baseline cold E=1.000.
- Added G2.gsm8k_outputs (completion texts must equal baseline cold under batch
  invariance), GATES.md "Implementation" section with the precise definitions.
- Full Qwen3-4B P0 run started: `--tag qwen3-4b-p0-baseline` (gsm8k 1319,
  replay 32 convs x 12 turns). Baseline gsm8k 0.8582 cold/warm; replay warm
  TTFT p50 2.54s at concurrency 32.
- P0 review agent launched -> `docs/rsi/reviews/P0-gates.md`.
- P2 plan (started while the P0 run finishes):
  - Package `python/pegaflow/connector_v2/` (`PegaKVConnectorV2`), coexists with
    `pegaflow.connector`; layout per design doc: `core/` glue, `adapters/` (one
    file per KVCacheSpec class), `policy.py`, `models/*.toml`.
  - P2 ships core + adapter registry with NO adapters: startup on any model
    fails with a clear `Unsupported` naming the spec class. Default-gate unit
    tests drive the core state machine through a stub adapter. Adapters are
    added by the onboarding agent of each model (P3 adds the first one).
  - Server stays as is for P2-P6: all 4 models run TP1, where the current
    seal logic equals seal_plan. Server-side seal_plan + layout fingerprint
    namespace go to a later server phase (open question below).
  - Written by an isolated sub-agent (this session read the old connector).
- Full P0 run `2026-10-23-qwen3-4b-p0-baseline` finished. Baseline: gsm8k 0.8582,
  replay TTFT p50 cold/warm 2.525/2.541s, replay local hit rate 0.377 (GPU cache
  small at 8192 x 32), oracle hit rate warm 0.998. Old arm: gsm8k 0.8582 with
  outputs identical, but warm external hit rate 0 (gsm8k) / 0.004 (replay);
  G2.digest_self (0 hits compared), G3 (E 0.94 / 0.40) and G4 fail.
  Root cause is the gate, not the connector: the server ran with NUMA-split
  pools (60 GB / 8 nodes = 7.5 GiB on GPU0's node), 10752 evictions in gsm8k
  cold. Fix in `run_gate`: `--pool-size 160gb --disable-numa-affinity`, plus a
  new verdict G3.no_eviction (oracle assumes unlimited capacity, so evictions
  invalidate E). Rerun: `--tag qwen3-4b-p0-baseline2`, log
  `~/kexi/rsi-runs/p0-baseline2.log`.
- P2 skeleton: `python/pegaflow/connector_v2/` (`PegaKVConnectorV2`, lazy
  export). The vLLM-free `core/` does three things:
  - Hits: queries each group (storage group 0 = a full group via prefix query;
    the others via membership), then `resolve` -> `policy.accept`.
  - Leases: loads only the `needed` blocks past the local hit; leases with
    nothing wanted are released, unwanted blocks get `None` targets.
  - Saves: per step via `saveable` + `choose_saves`, synchronous in
    `wait_for_save`; failed loads are reported through `failed_recving`.
  - The namespace is a sha256 of the adapter declarations, the parallel shape
    and the model/dtypes.
  - No concrete adapters yet: unknown specs raise `Unsupported` at init on
    both roles.
  - Policy knobs come from `models/default.toml`.
  - `tests/test_connector_v2_core.py`: 7 passed; ruff clean; import proof OK.
  - Written in isolation: the old connector, `instance.rs` and `offload.rs`
    were not read.
- P2 review fixes (`reviews/P2-connector-skeleton.md`, "Resolution"):
  - M1/M2: the policy only sees immutable copies (frozensets, a tuple of
    SaveItems).
    - `accept` is capped at the legal `p`, then re-checked against the server
      answers.
    - Every wanted block must be covered by a lease of its group, so an empty
      hit is impossible.
    - `choose_saves` may only return offered items; anything else raises.
  - M3: checkpoint groups are `Unsupported` until the core handles
    `boundary_state_offloads` (P5 core task).
  - M4: `semantics` is injectable and imported lazily. The tests use a
    pure-Python stub plus native cross-checks.
  - S2: segment size == `bytes_per_block`, and a local `seal_plan` (V0) over
    all TP ranks.
  - S3: `LayerDecl.transfer_backend` / `page_first`, included in the namespace.
  - S6: hit and alloc planning never leak leases or raise into vLLM.
  - S1: one load RPC per request, a vLLM-trimmed hit is re-planned, and the
    README requires `kv_load_failure_policy=recompute`.
  - Also fixed: S4 (`Unsupported` without a full group), n2, n4-n8, part of n3.
  - Deferred:
    - S4 fetch for sid>0 (G3; before SSD/remote gates, P6 at latest);
    - S5 async saves (G4, after P3);
    - n3 remainder (server phase);
    - `run_gate` `kv_load_failure_policy=recompute` (owner of `rsi/`,
      before P3's gate run).
  - Test results on .3:
    - `test_connector_v2_core.py`: 77 passed (shared venv).
    - Source-only uv with no `.so` in the tree: 51 passed, 26 native-only
      cases skipped.
    - ruff format/check clean.
- P0 review fixes (`reviews/P0-gates.md` "Resolution", all M1-M9, S1-S12 and
  nits addressed; S5 mapping check stays a P5 task). Main changes:
  - Probe: schedule episodes + per-step progress, `load_errors.jsonl`,
    geometry self-check, digest tags via `vllm_xargs.rsi_digest`,
    `RsiWorkerExt.rsi_poison` (KV poisoned after every reset).
  - Digest: local / external / computed tags, any-copy reference with an
    exhaustiveness rule (`uncertain` otherwise).
  - Oracle: connector-only `E_ext`, `excess`, step-accurate store visibility,
    checkpoint-only-at-step-end.
  - run_gate: `plain` and `selftest:<mode>` arms, `kv_load_failure_policy=recompute`,
    `gate.json`, pinned scheduler flags, 3 replay passes, official profile.
  - Committed `docs/rsi/models/Qwen3-4B/spec.json` refreshed from the new probe
    (old dump lacked `real_page_size_bytes`, max_model_len 4096).
- Gate self-test, Qwen3-4B, gsm8k 128 q (`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p0-selftest`):
  - baseline controls pass: plain == baseline outputs (0/128 differ), poisoned
    warm == cold (0 differ), 16510 local blocks matched, 2576/2576 computed exact.
  - honest: 579 external blocks matched, 0 mismatched, outputs identical, E_ext 1.0.
  - liar: 1158 external mismatched vs baseline (unreferenced in self), acc 0.
  - bitflip: 128 external mismatched (exactly one per request), 28/128
    warm outputs differ, acc 0.844 (an accuracy-only gate would have passed it).
  - offbyone: 451 external mismatched, acc 0.18.
  - All four arms behave as expected -> the digest gates are not blind.
  - Found by the liar arm: gsm8k answer parsing crashed on garbage digit runs
    (`float('1e999')`); `_norm` keeps non-finite numbers as strings.
- Commits on feat/semantics-rsi: `feat(rsi): add P0 gate tooling`,
  `feat(connector): add connector v2 skeleton`.
- Full official run started: `--tag qwen3-4b-p0-final --arms plain,baseline,old`
  (log `~/kexi/rsi-runs/p0-final.log`, dir `2026-10-23-qwen3-4b-p0-final`).
  Baseline gsm8k 0.8582 cold/warm, replay cold TTFT p50 2.84s. Then .3 became
  unreachable from the jump host ("No route to host", .2 too). NEXT on resume:
  check whether the run finished (`summary.json`), else rerun the same command;
  read the old-arm verdicts, record them here, mark P0 done, start P3.
- P3 plan (isolated onboarding agent, Qwen3-4B, record `docs/rsi/models/Qwen3-4B/ITERATIONS.md`):
  - Spec dump: one `FullAttentionSpec` group, layout LBNHC, `block_stride == page`
    (65536 B). In LBNHC a block is `[N tokens][H heads][K|V]`, so K and V are
    interleaved and cannot be stored as two segments: declare ONE segment = the
    whole page.
  - New `adapters/full_attention.py`: `Group.full(block_size)`, `Heads` shard
    (replicas beyond `max_tp_shards`), view checked to be dense pages at the
    server's default block stride (the gRPC register path has no block stride),
    else `Unsupported`. No policy profile needed (defaults apply).
  - The core already matches vLLM 0.32 (block_state, transfer results,
    MultiConnector, empty extra, `PEGAFLOW_PORT`). No semantics change needed.
- P3 it1 written (`ITERATIONS.md` it1): `adapters/full_attention.py` + registry
  entry, adapter tests in `test_connector_v2_core.py` (local stub run 66 passed,
  41 native skipped; ruff clean), README adapter table, SKILL lessons. No core,
  semantics or policy change. .3 still "No route to host" at the end. NEXT on
  resume: sync, native unit run, then the it1 slice gate (command in
  ITERATIONS.md), then the official run only if the slice passes.

## Semantics changelog

Each entry: what was added, why, which vectors cover it.

- S1 (P1, initial): `Kind::{Full, Window{tokens}, Checkpoint, Scratch}`,
  `Group{kind, block_tokens}`; `needed`, `resolve` (lcm step, cap prompt_len-1,
  all-scratch -> 0), `saveable` (checkpoint only on block boundary).
  Vectors: `vectors/resolve.json` (11 cases incl. design-doc example P=12),
  randomized brute-force cross-check in `groups.rs`.
- S2 (P1, initial): `key(hash, group)` = raw hash for group 0, else hash ++ BE u32.
  Vectors: `vectors/key.json`.
- S3 (P1, initial): `Shard::{Replicated, Heads{index,count}, Tokens{index,count,interleave}}`,
  `Region`, `RankDecl`, `seal_plan` -> per-group fragment list (slot = position,
  ordered by layer name then part; identical parts merge, lowest rank = writer).
  Errors: duplicate rank/layer, invalid shard, group/scheme/segment mismatch,
  missing part. Unit tests in `sharding.rs`.
- S4 (P1 review): `Group::new` rejects `block_tokens == 0` and `Window{tokens: 0}`;
  resolve step = lcm over non-scratch groups only (scratch loads nothing);
  `Window.tokens` documented as "trailing cached tokens" (vLLM `sliding_window - 1`
  exact). seal_plan: all parts of a layer must share segment sizes; zero-byte
  segments rejected; writer = de-dup hint, store accepts any holder.
  Vectors: `resolve.json` (18), `seal_plan.json` (12), `key.json` (3).

## Open questions

- "agentx" perf workload: not found in repo; asked the user. Fallback: multi-turn
  agent traces from `/ppio/data_sets/R2EGym-SFT-Trajectories`.
- Server-side seal_plan (replace slot counting) and layout-fingerprint
  namespace in the server: not needed for TP1 onboarding; schedule after P6
  or when a TP>1 gate is added.
