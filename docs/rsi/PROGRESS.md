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
| server (storage only) | Rust | not changed by agents (user, 2026-10-24): gates run on the base server; a server limit is attributed and listed in "Server limitations" below for a human to decide |
| gates (`docs/rsi/GATES.md`, `rsi/gates/`) | — | human only; agents may add checks, never relax |

## PR policy (user, 2026-10-23)

- Connector code is NOT per model: one `connector_v2`, adapters per vLLM
  KVCacheSpec class, policy shared, per-model knobs only in
  `connector_v2/models/<model>.toml`. A model iteration may touch adapters /
  policy as long as every earlier accepted model still passes its gates.
- Review unit = one model: each model's onboarding (all its iterations, docs,
  accepted run) is one branch `feat/rsi-<model>` and one PR, stacked on the
  previous one. #473 is the observation PR for the foundation (semantics,
  gates, connector core) + Qwen3-4B.
- Server stays at the base version (user, 2026-10-24: "server还是保持原始版本，
  把server的局限问题汇报给人就行，gate不要被server受限，命中率不用过度追求最高，
  因为淘汰策略不一定是全局最优的"). Every gate run uses the pegaflow-server
  built from `feat/semantics-rsi`. Agents do not write or tune server code;
  they attribute a server-caused gap and list it under "Server limitations".
  A model is never blocked or held "pending server" because of one. The
  earlier server work (H1-H5, PRs GentleCold/pegaflow#3 and #4) is parked as
  a proposal for the human; nothing depends on it.
- Gate changes (`rsi/gates/**`, GATES.md) stay human-owned: agent-written gate
  code lands on its own branch (e.g. `feat/rsi-g5`) and is reviewed before it
  is used to accept anything.
- Review flow (user, 2026-10-23: "这些评审都不要等我"): nothing blocks on the
  user. Every review (iteration, gate branch, server branch) is done by an
  independent review agent; the orchestrator merges gate branches and moves on
  once that review passes. The user does one acceptance pass after all four
  models are supported, covering the gate PRs, model PRs and the "Server
  limitations" list (with the parked server proposals) together.
- Every progress report and ITERATIONS entry includes the hit-rate table
  (local / external / total hit, oracle H*, E_ext, and at capacity H*_C, H_lru,
  E_C), not only the perf numbers.

### Server limitations (report to the human, do not fix)

Measured on Qwen3-4B G5 (single Full group); details and numbers in
`feat/rsi-server-reclaim:docs/rsi/server/reclaim-eviction.md` and
`feat/rsi-server-evict-tail:docs/rsi/server/evict-tail.md`. Current plan
(r3, C(0.5) = 11536 blocks) at C/U 0.5: base server E_C 0.664-0.684,
H1+H3+H4 0.699-0.719, + H5 0.759-0.769 (hit 0.86 -> 0.89). H1+H3+H4 add
+0.10 at 0.25 and +0.06 at 0.75; H5 costs 0.01-0.08 E_C at 0.25.

- Reclaim drops a fixed `RECLAIM_BATCH_SIZE = 512` blocks per round. On a
  small pool one round frees about half of the cache (1019-block slice: 502
  resident after cap_cold). Proposal H1: `clamp(resident / 64, 1, 512)`.
- Pinned-region fragmentation: a layer-first save call takes one region per
  layer, freed only when the call's last block is evicted, so dead bytes
  stay pinned. Large on the older 32-conv plan (C = 5873 blocks: residency
  after cap_cold 5473 v2 / 4750 old, E_C 0.1-0.25 before H3), small on r3
  (see the E_C line above). Proposal H3: one region per saved block.
- H3's many small copies cost warm TTFT with per-range `cuMemcpyAsync`.
  Proposal H4: one `cuMemcpyBatchAsync` per transfer.
- Eviction is plain LRU per (hash, group) key. On r3 most of the gap to
  tail-first is fully evicted ranges (0.28 -> 0.23 of ext*_C), not
  stranding (1.7% v2 / 2.5% old). Heads aging out mostly hits `old`, which
  queries from its local hit and never refreshes them; v2 queries from
  block 0. Proposal H5: opt-in tail-first eviction in the retained class.
- Not group-coherent: a resume point needs every group, but each group key
  is evicted on its own. Seen on Qwen3.5-4B, but only on a sub-floor pool
  (P5 it2 s2cap d2a484d, C = 1.63 GiB = 0.05x GPU KV, so no G5 conclusion):
  `hit stats` `missing=1,2,3` in 4 of 160 lookups, meaning the full group
  backed a resume point and all three GDN groups were evicted; 0 bytes
  loaded. On that pool one reclaim round (`RECLAIM_BATCH_SIZE` 512 blocks
  of ~17 MB) exceeds the whole pool (~100 blocks): 176-265 evictions and
  3.3-5.0 GB reclaimed per pass against ~3.5 GB saved, 21-30 blocks resident
  at pass end. Needs re-measuring at the official cap size.
  Re-measured at the official cap size (P5 it2 official c5d3f04, C(0.5)
  = 2.52x GPU KV): `missing=1,2,3` in 4 of 7441 lookups at 0.25, 4 of
  7443 at 0.5, 2 of 7433 at 0.75 (2640 / 2640 / 1584 lost tokens):
  negligible at the floor.

### Gate changes

- 2026-10-24, user (human gate owner): "命中率那块，不要设置硬性门槛，只要agent自己判断命中率足够就行了，不要差太多".
  Hit-rate levels are reported, not gated: G3 `E_ext >= 0.95` and
  `oracle_hit_rate >= 0.5`, G5 `E_C >= 0.9`, the G2 "> 0.9 of oracle" item
  and the no-`H`/`E_C`-regression clauses. Every number is still computed
  (`reported` per verdict, `summary.hit_report`). Correctness and run
  validity stay hard and unchanged. The onboarding agent writes a hit-rate
  judgment in ITERATIONS.md; the review agent checks it (GATES.md "Hit
  rate"). Branch `fix/rsi-gate-old-plugin`.
- 2026-10-24: the `old` reference arm runs with `VLLM_PLUGINS=pegaflow` (vLLM
  0.32 looks MultiConnector children up by registered name to rebuild stats;
  P4 MF1 got HTTP 500 on every old-arm request). Every other arm keeps `""`.
  Branch `fix/rsi-gate-old-plugin` 3fe119c.

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

- Remote: 10.1.0.44 since 2026-10-23 (8x4090, driver 595, `LD_LIBRARY_PATH=~/kexi/cuda13-lib`);
  before that 10.121.196.3 (CUDA 13.0, driver 580). Everything under `~/kexi/`.
- Shared venv `~/kexi/pegaflow/python/.venv` (py3.12, vLLM 0.32.0, torch 2.13.0,
  flashinfer-python 0.7.0.post1). No per-branch venv.
- Feature dir `~/kexi/pegaflow-semantics-rsi`, sync via
  `.local/bin/pegaflow-remote-sync.sh --ip 10.1.0.44 --feature semantics-rsi`.
- Build: maturin with `cuda-13,rdma`, `PYO3_PYTHON` = shared venv python.
- Models: `/root/kexi/models/{Qwen3-4B,DeepSeek-V2-Lite-Chat,Qwen3.5-4B,gemma-4-E4B-it}`.
- Logs `~/kexi/logs`, run records `~/kexi/rsi-runs`.
- Local worktree `.worktrees/semantics-rsi`, branch `feat/semantics-rsi`
  (base upstream/master 1749353), trellis task `.trellis/tasks/10-23-semantics-rsi`.

## Phases

| Phase | Content | Status |
|---|---|---|
| P0 | gates: spec dump, V3 digest, gsm8k, perf workload, hit-rate metrics, old-connector baseline | done (p0-final 2026-10-23); G5 capacity gate in progress |
| P1 | `pegaflow-semantics` crate + PyO3 + V1 vectors | done (review: `reviews/P1-semantics.md`) |
| P2 | new connector skeleton (core glue + adapters + policy + models), coexists with old | done, review fixes applied (`reviews/P2-connector-skeleton.md` Resolution) |
| P3 | onboard Qwen3-4B | G0-G4 accepted at 8a05b7c (it2d official `2026-10-23-qwen3-4b-p3-it2d`, review ACCEPT-WITH-FIXES, fixes in it2e); G5 E_C reported (not gated since 2026-10-24): 0.684 on the base server (official r3, `models/Qwen3-4B/ITERATIONS.md` "G5 official (r3)"). Hit-rate judgment on the base server: close enough. G5 hit at C/U 0.5 is v2 0.838-0.843 before P4 it7 (r3, P4 it6 cap, P4 MF5) and 0.859 on it7 (P4 M1b official), vs old 0.820-0.840, native 0.861-0.862, H_lru 0.857-0.858; the gap from H_lru to H*_C is the base server's (server proposals H1-H5 parked, see "Server limitations"). P4 regression record: `models/Qwen3-4B/ITERATIONS.md` "P4 regression runs on this model" |
| P4 | onboard DeepSeek-V2-Lite-Chat | accept: it7 a59a254 on `feat/rsi-deepseek-v2-lite` (TP2, MLA). ACCEPTED 2026-10-25: final review ACCEPT-WITH-FIXES, M1, M2 and S1-S5 applied, re-review ACCEPT (`reviews/P4-final.md`, 4164173). Open for the user's final pass: DSV2 G5 at C/U 0.25 is 0.07 below old on the same server, cause unknown (connector side by elimination; below the 2x GPU-KV floor); lmcache numbers go into the P4 tables when `feat/rsi-gate-lmcache` merges; the gate owner should revisit the preempted-uncertain bound (DSV2 v2 at 0.85% of 1%). Official on it7: DSV2 `2026-10-25-dsv2-lite-p4-it7-off-a59a254` and Qwen3-4B `2026-10-25-qwen3-4b-p4-it7-off-a59a254`, where v2 passes every verdict (Qwen `G2.digest_self.cap` 0.45%). Hit judgment: close enough. Tables: "P4 acceptance tables" below. lmcache: pending (arm not comparable yet) |
| P5 | onboard Qwen3.5-4B | it2 official `2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04` (`feat/rsi-qwen3.5-4b` c5d3f04, hybrid gate, base server): official, v2 passes every verdict; awaiting independent review. it1 accepted (core checkpoint saves from offers, ordering check, Qwen3-4B + DSV2 non-regression); it2 = H1 pinned async checkpoint saves + H2 lookup skip. G4: gsm8k cold/warm +0.34%/-0.35%, replay cold TTFT p50 0.512 vs 1.204 s, warm 0.300 vs 1.112 s. Hits: v2 = H* on every unlimited pass (E_ext 1.0); G5 hit v2/old/native 0.794/0.794/0.794 (0.25), 0.794/0.794/0.788 (0.5), 0.907/0.906/0.946 (0.75), v2 = H_lru within 0.002; E_C 0.5 v2/old/native 0.828/0.867/0.815 (gap to old = own-local accounting). E_C > 1 at 0.25 is an oracle gap for Checkpoint groups (finding for the gate owner, 2026-10-25). Judgment: close enough. lmcache: pending. Record: `models/Qwen3.5-4B/ITERATIONS.md` "it2 official" |
| P6 | onboard gemma-4-E4B-it | todo |

### P4 acceptance tables (it7 a59a254, gate ce645d7, base server d1970ead)

Both runs are official: arms plain, baseline, old, native, v2, GPU 0(,1).
Full records are in `models/DeepSeek-V2-Lite-Chat/ITERATIONS.md` ("it7",
M1a and M1b).

G3, unlimited store (local / ext / total; H*; E_ext; excess tokens):

| model / pass | v2 | old | native | lmcache | H* |
|---|---|---|---|---|---|
| DSV2 gsm8k warm | 0.937 / 0.056 / 0.993; E 1.0; 0 | 0.938 / 0.055 / 0.993; E 1.0; 1230 | 0.940 / 0.053 / 0.993; E 1.0; 1230 | pending (arm not comparable yet) | 0.993 |
| DSV2 replay cold | 0.493 / 0.349 / 0.843; E 1.0 | 0.491 / 0.352 / 0.843; E 1.0 | 0.498 / 0.345 / 0.842; E 0.9998 | pending | 0.843 |
| DSV2 replay warm2 | 0.465 / 0.533 / 0.998; E 1.0; 0 | 0.519 / 0.479 / 0.998; E 1.0; 210 | 0.461 / 0.537 / 0.998; E 1.0; 210 | pending | 0.998 |
| Qwen3-4B gsm8k warm | 0.922 / 0.070 / 0.992; E 1.0; 0 | 0.926 / 0.067 / 0.993; E 1.0; 1170 | 0.933 / 0.060 / 0.993; E 1.0; 1170 | pending | 0.992 |
| Qwen3-4B replay cold | 0.385 / 0.479 / 0.864; E 1.0 | 0.381 / 0.483 / 0.864; E 1.0 | 0.384 / 0.480 / 0.864; E 1.0 | pending | 0.864 |
| Qwen3-4B replay warm2 | 0.347 / 0.651 / 0.998; E 1.0; 0 | 0.361 / 0.638 / 0.998; E 1.0; 195 | 0.378 / 0.621 / 0.998; E 1.0; 195 | pending | 0.998 |

G5, cap trace, warm_timed. Each cell is hit (E_C); H_lru and H*_C are
computed on v2's trace:

| model, C/U | v2 | old | native | lmcache | H_lru | H*_C |
|---|---|---|---|---|---|---|
| DSV2 0.25 (1.14x GPU KV) | 0.452 (0.070) | 0.522 (0.180) | 0.579 (0.282) | pending (arm not comparable yet) | 0.671 | 0.854 |
| DSV2 0.5 (2.27x) | 0.847 (0.620) | 0.845 (0.614) | 0.851 (0.622) | pending | 0.847 | 0.925 |
| DSV2 0.75 | 0.899 (0.659) | 0.886 (0.616) | 0.933 (0.708) | pending | 0.918 | 0.968 |
| Qwen3-4B 0.25 (1.22x) | 0.409 (0.111) | 0.396 (0.071) | 0.533 (0.296) | pending | 0.584 | 0.863 |
| Qwen3-4B 0.5 (2.44x) | 0.859 (0.701) | 0.829 (0.675) | 0.862 (0.715) | pending | 0.858 | 0.929 |
| Qwen3-4B 0.75 | 0.908 (0.741) | 0.878 (0.693) | 0.927 (0.775) | pending | 0.916 | 0.970 |

G4, change vs the same-run baseline (warm = warm2 / warm3):

| model / arm | replay warm p50 | warm req/s | cold p50 | cold req/s | gsm8k cold / warm |
|---|---|---|---|---|---|
| DSV2 v2 | -68.2 / -67.3% | +131 / +126% | -31.8% | +59.7% | +0.32 / -1.96% |
| DSV2 old | -57.8 / -63.8% | +131 / +126% | -30.8% | +57.6% | +1.04 / -0.98% |
| DSV2 native | -69.9 / -67.2% | +141 / +144% | -38.8% | +63.8% | +0.63 / -1.57% |
| Qwen3-4B v2 | -85.1 / -83.8% | +288 / +277% | -62.3% | +145.0% | +0.92 / -0.43% |
| Qwen3-4B old | -82.4 / -81.1% | +294 / +278% | -62.3% | +146.3% | +1.82 / +1.52% |
| Qwen3-4B native | -84.8 / -84.2% | +273 / +269% | -62.2% | +130.1% | +6.38 / +6.42% (fail) |

Judgment: close enough. At the gated point, 0.5, v2 is at H_lru on both
models and level with the reference arms. The one gap is DSV2 0.25, where v2
is 0.07 below old. Old's run-to-run noise there is 0.038 (0.512 to 0.550). C
there is 1.14x GPU KV, below the 2x floor. The sim rules out the
local-hit query as a cause, and the cause is unknown (DSV2 file, "M2").

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
- P3 it1b: review `docs/rsi/reviews/P3-qwen3-4b-it1.md` (APPROVE-WITH-FIXES) addressed;
  Resolution table appended there. M1: `kv_cache_layout` in the namespace, plus
  kernel-split HNC pages rejected. S1: `non_causal` fails closed. S2: worker
  `save stats` log line every 500 steps and at shutdown. n1-n6 fixed. Local stub
  run 69 passed, 43 skipped; ruff clean. .3 still "No route to host". NEXT: the
  same as it1 (sync, native units, slice gate on GPU 1 after checking GPU 0 is
  not running the P0 final run), then read `save stats` for the S5 decision.

- Host move (user, 2026-10-23): .3/.2 stayed "No route to host"; the 4090
  host is now `10.1.0.44` (8x4090, driver 595 / CUDA 13.2, 1 TiB RAM,
  8 NUMA nodes, fresh box). Rebuilt from scratch under `~/kexi/`:
  - apt build deps (protobuf, ibverbs/rdmacm, numa, clang); rustup with the
    default dist server (the tuna rustup mirror 404s on 1.97.0), crates via
    the tuna sparse index in `~/.cargo/config.toml`.
  - Shared venv `~/kexi/pegaflow/python/.venv`: py3.12, vLLM 0.32.0,
    torch 2.13.0+cu130 (tuna). `maturin develop --uv` (the uv venv has no pip).
  - pegaflow-server must be built with `PYO3_PYTHON` = venv python (otherwise
    it links the system libpython3.10). The system toolkit is CUDA 12.2 and
    cudarc dlopens `libcudart.so` unversioned, so the server picked 12.2 and
    failed the preflight. Fix: `~/kexi/cuda13-lib` symlinks the venv's
    `nvidia/cu13/lib` with unversioned names; every gate run on .44 uses
    `LD_LIBRARY_PATH=~/kexi/cuda13-lib` (replaces the .3 cuda-13.0 path).
  - Data: gsm8k from the same OSS mirror (7473/1319); R2EGym parquet from
    hf-mirror, `r2egym-256.jsonl` regenerated with `prep-r2egym --n 256`.
    Models via modelscope into `~/kexi/models/`.
  - Remote checks: `test_connector_v2_core.py` + `test_semantics_bindings.py`
    147 passed (native included); server smoke healthy.
  - First launch died in vLLM startup: flashinfer JIT (sampling) used the
    system nvcc 12.2 (`--compress-mode` unknown). Fix: pin venv
    `nvidia-cuda-nvcc`/`nvidia-cuda-crt`/`nvidia-nvvm` to 13.0.x (13.4 nvcc
    rejects the 13.0 headers; 13.4 cicc emits PTX 9.4 that ptxas 13.0
    refuses), and `~/kexi/cuda13` = CUDA_HOME (bin/include/nvvm from the
    venv, lib -> cuda13-lib). Gate env on .44:
    `CUDA_HOME=~/kexi/cuda13 PATH=$VENV/bin:~/kexi/cuda13/bin:$PATH LD_LIBRARY_PATH=~/kexi/cuda13-lib`.
  - The .3 p0-final record is lost with the host, so p0-final is rerun from
    scratch on .44 GPU0 (`p0-final.log`). P3 it1 slice on GPU1
    (`p3-it1-slice.log`), rev 41fb158.

- P3 it1b slice gate (.44 GPU1, rev 41fb158, 64 gsm8k / 8x6 replay, not
  official): every verdict passes. G0.spec 0 diffs. G2 digest: external
  1492/1492 (gsm8k) and 617/617 (replay) matched, 0 mismatched; gsm8k 0.8594
  cold = warm = baseline, outputs identical. G3: gsm8k warm H=0.978 vs oracle
  0.992, E_ext=1.0; replay warm H=0.991 vs oracle 0.998, E_ext=1.0; excess 0,
  no evictions, no load errors. G4: warm TTFT p50 0.101/0.104 s vs baseline
  0.209 s (2.0x); cold 0.266 s vs 0.211 s (save cost). Save stats (gsm8k):
  ~21 ms per save step, 30% of steps save, sync 3.2 s of a 13.8 s window, rpc
  77 ms, 0 failed. Sync dominates, so S5 async saves are the it2 candidate
  (cold TTFT). Official run (full gsm8k, 32x12 replay) launched on GPU1:
  `~/kexi/rsi-runs/p3-it1.log`. p0-final on GPU0 still running (old arm).

- G4 made hard (user: "性能需要达标才行，需要比基准快"). New checks in
  `run_gate._perf_verdicts`, defined in GATES.md: warm replay TTFT p50 and
  req/s strictly better than baseline; cold replay and gsm8k cold/warm wall
  time at most 2% worse (`MAX_OVERHEAD`). Re-judged the it1b slice offline:
  warm wins (TTFT 0.10 vs 0.21 s, 6.6 vs 5.5 req/s) but cold replay TTFT
  0.266 vs 0.211 s (+26%), cold req/s 4.68 vs 5.11 (-8%), gsm8k +4.1% cold /
  +3.2% warm -> it1b NOT accepted. The official it1 run keeps going on GPU1 as
  full-scale data (judged with the old G4; re-judge its result.json offline).
  NEXT: it2 = remove the synchronous save wait (S5), slices on GPU2, by an
  isolated onboarding agent.

- Official it1 (`2026-10-23-qwen3-4b-p3-it1`) finished, all verdicts passed
  under the old G4. Re-judged with the hard G4: replay warm 4.9x TTFT / 2.8x
  req/s and replay cold 2.1x TTFT / 1.9x req/s vs baseline all pass; gsm8k
  cold +8.3% / warm +7.4% wall time FAIL. Not accepted; it2 (async save)
  targets exactly the gsm8k overhead. Details in Qwen3-4B/ITERATIONS.md.
  Save stats: device sync ~50% of each 500-step window (e.g. 10.5 s of 19.0 s
  on gsm8k), save RPC ~1.5%, 0 failed saves. Decision (timer check): stay on
  P3, it2 (async save) continues; P4 starts only after it2 is accepted.

- p0-final (`2026-10-23-qwen3-4b-p0-final`, arms plain/baseline/old) done:
  baseline controls pass; the `old` reference arm passes every G0-G3 check
  (E_ext 1.0 gsm8k and replay). Reference numbers (black box): gsm8k cold
  170.1 s vs baseline 169.5 s (+0.4%), replay warm2 req/s 10.2 vs 2.59,
  TTFT p50 0.62 vs 3.57 s. So a <2% gsm8k overhead is reachable. Note
  `plain` gsm8k is 148 s vs baseline 169.5 s: the probe (digesting every
  gsm8k request) costs ~13%; G4 compares arms that both carry the probe.
  P0 done.

- User: actual hit rate must approach the theoretical hit rate at a FIXED
  capacity; e.g. Kimi K3-style MLA + checkpoint groups whose boundaries differ:
  if the MLA blocks are evicted, later checkpoints are useless, which is a
  PegaFlow server limit (`cache.rs` LRU evicts each (hash, group) key on its
  own). Decisions: (1) new hard gate G5 (GATES.md): pool = 0.5 x unique bytes
  of the trace, bar = clairvoyant oracle at that capacity, `E_C >= 0.9`, plus
  per-group stranded/evicted/write-amp diagnostics; (2) the server is now in
  the RSI surface (Layering table): an onboarding agent may change eviction /
  admission / pools when a gate points there, Kind-aware and model-agnostic,
  with `cargo test`, an `old`-arm non-regression run and a separate review;
  (3) skill step 5 gets a hypothesis checklist (window tail, checkpoint
  boundaries, never-resumable state, group-coherent eviction, scan
  resistance, async save). Qwen3.5-4B (P5, linear attention + full) is the
  first model where cross-group coherence matters on the 4090s. G5 tooling is
  being implemented by a gate agent on branch `feat/rsi-g5`.

- User asked whether baseline should include vLLM's official offload. Yes:
  vLLM 0.32 ships `OffloadingConnector` (`--kv-offloading-size <GiB>`,
  backend `native`; `SimpleCPUOffloadConnector` with
  `VLLM_USE_SIMPLE_KV_OFFLOAD`), both `SupportsHMA`. New arm `native`
  (OffloadingConnector + RsiProbe, CPU bytes = PegaFlow pool size; at G5 the
  same capacity C). Phase 1: recorded next to `baseline` in G3/G4/G5 tables.
  Phase 2, after the first numbers: G4 warm replay and G5 `E_C` must also be
  at least as good as `native` (hard). Added to the G5 agent's scope.
  PR policy section added (per-model PRs, server PRs separate and
  human-reviewed).

- User worried policy would grow too complex. Agreed rules (SKILL.md step 5,
  checked by the review agent in step 6): no model / spec class names in
  `policy.py`; every knob needs gate evidence, toml only overrides defaults,
  unused knobs are deleted; budget ~150 lines per Kind and <= 12 knobs, refactor
  first when exceeded; simplification is a valid iteration; held-out replay
  slice against tuning to the trace; growing policy logic means it belongs in
  semantics (new Kind) or the server (separate PR).

- P3 it2 (isolated onboarding agent, `ITERATIONS.md` it2), two measured
  connector-core changes, no semantics or server change:
  - it2a (0bb44bf) async saves: `wait_for_save` records a CUDA event and a
    saver thread issues the save RPC after it. Finished requests with saves
    return `request_finished=True` and are released via `finished_sending`
    once their jobs land; preemption (and any window-group layout) flushes
    pending saves in `handle_preemptions`. Slice1 (full gsm8k + 8x6 replay,
    GPU2): all pass except gsm8k warm +2.9%; cold +1.1% (was +8.3%), replay
    cold TTFT within 0.5%. Save step-thread cost 10.4 s -> 15 ms per 500 steps.
  - it2b (2ff447e) sync short loads: warm gsm8k paid the async load round
    trip (7.6 steps admission -> first compute; TTFT sum 282 s vs 179 s).
    Hits <= `sync_load_tokens` (new knob, 256) load before the forward;
    failures go through `invalid_block_ids`. Enabled only for one KV group
    without block zeroing.
  - Slice2 `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2-slice2` (rev 2ff447e):
    every verdict passes. G4: replay warm TTFT 0.091 vs 0.210 s, req/s 6.85
    vs 5.51; replay cold TTFT 0.208 vs 0.210 s, req/s 5.07 vs 5.08; gsm8k
    164.6 vs 163.1 s cold (+0.9%), 163.5 vs 163.1 s warm (+0.3%). Hits: gsm8k
    warm local 0.916 / external 0.077 / total 0.992 = H* 0.992, E_ext 1.0;
    replay warm 0.730 / 0.268 / 0.998 = H* 0.998, E_ext 1.0. G2 digests 0
    mismatched. NEXT (orchestrator): review agent + official run.

- G5 branch review (independent agent): APPROVE-WITH-FIXES. Must-fix: pool
  check fails open when the metric is missing; nothing proves external hits
  came from the C-byte store (add load-bytes >= ext blocks x block bytes).
  Should-fix: E_C rewards lowering vLLM's own local hit (score against the
  baseline arm's per-request local instead); crashing reference arms abort the
  run; `old` still counted as required. Also: a "fill once, never evict" policy
  scores E_C ~0.95 because warm passes replay the same trace. User flagged the
  slice capacity as too small (C = 2.24 GiB = 0.22x GPU KV; official 12.9 GiB =
  1.24x). Decision: G5 gets its own larger workload (64 convs), C/U in
  {0.25, 0.5, 0.75} (gate at 0.5), hard floor C >= 2x vLLM GPU KV for an
  official run; slice G5 = smoke test only. Fixes + capacity change assigned to
  the gate agent on `feat/rsi-g5`; merge after a re-review.
- Server iteration started on `feat/rsi-server-reclaim` (separate PR):
  reclaim frees a fixed 512 blocks per round (~0.5 C on the slice) and the LRU
  is prefix-oblivious (chain heads served from vLLM's GPU cache age out first,
  breaking every lookup that starts at block 0). GPU6/7.
- P3 it2 official run `qwen3-4b-p3-it2` started on GPU1; independent review of
  it2 running.

- Official it2 (`2026-10-23-qwen3-4b-p3-it2`, rev 2ff447e): official, exit 0,
  all G0-G4 pass. replay warm TTFT 0.54 vs 2.55 s (4.7x), req/s 10.0 vs 3.4;
  cold TTFT 1.24 vs 2.79 s; gsm8k +0.9% cold / +0.0% warm. Hits: replay warm
  total 0.998 = H*, E_ext 1.0 (local 0.35 + external 0.65); gsm8k warm 0.992 =
  H*. Digests 0 mismatched. Table in Qwen3-4B/ITERATIONS.md.

- G5 official run, PRE-FIX gate code (`2026-10-23-qwen3-4b-g5-official`, gate
  rev 369b253, GPU4, same run for all arms; v2 numbers match it2). Reference
  only: the gate fixes change E_C (per-arm local subtraction) and capacity.

  | arm | replay warm2 TTFT | warm2 req/s | cold TTFT | gsm8k cold/warm s | replay warm hit (E_ext) |
  |---|---|---|---|---|---|
  | baseline | 2.538 | 3.41 | 2.81 | 164.6 / 164.8 | 0.377 (0) |
  | old | 0.630 | 10.24 | 1.356 | 166.7 / 169.5 | 0.998 (1.0) |
  | native (vLLM OffloadingConnector) | 0.500 | 9.83 | 1.308 | 169.6 / 169.4 | 0.998 (1.0) |
  | connector_v2 | 0.522 | 10.02 | 1.245 | 166.0 / 165.1 | 0.998 (1.0) |

  At capacity C = 12.9 GiB (0.5 x U = 25.8 GiB, ~1.24x GPU KV):

  | arm | cap hit | H*_C | H_lru | E_C | cap_warm2 TTFT | req/s |
  |---|---|---|---|---|---|---|
  | old | 0.459 | 0.814 | 0.523 | 0.12 | 2.087 | 3.73 |
  | native | 0.649 | 0.843 | 0.561 | 0.56 | 1.378 | 4.70 |
  | connector_v2 | 0.440 | 0.785 | 0.517 | 0.22 | 2.007 | 3.67 |

  Findings: unlimited, v2 beats native on cold TTFT, throughput and gsm8k but
  NOT on warm TTFT (0.522 vs 0.500 s, +4%) -> would fail the planned phase-2
  "at least as good as native" G4 check. At capacity native is clearly ahead
  (hit 0.65 vs 0.44, TTFT 1.38 vs 2.01 s): PegaFlow server reclaim/eviction is
  the gap (`feat/rsi-server-reclaim`). H*_C differs per arm because of the
  per-arm local subtraction (review S1, being fixed).
- User: add LMCache with the same config as another reference arm. Assigned to
  the gate agent after the G5 fixes: `LMCacheConnectorV1` + RsiProbe, local CPU
  only, capacity = pool size / C, default chunk size recorded. lmcache is not
  installed; installing into the shared venv would downgrade numpy/protobuf/otel,
  so it goes into an overlay dir on PYTHONPATH for that arm only.
- Reporting rule: every comparison table carries baseline, old, native,
  lmcache (when available) and connector_v2.
- P3 it2c (review fixes, 50e186b + 056eb4b): M1 saves held after sync loads
  until vLLM saw the outcome, no saves from un-rewound requests on failed
  blocks; S1 sync saves for zeroing/CoW layouts; S2 no sync-load deadline;
  S4 saver death releases holds (hung RPC still waits, no client deadline).
  Slice `2026-10-23-qwen3-4b-p3-it2c-slice1` (GPU1): every verdict passes;
  gsm8k v2 163.8 s cold / 162.3 s warm (+0.2%), replay warm TTFT 0.094 vs
  0.210 s; hits unchanged (replay warm 0.998 = H*). Held-out replay
  `2026-10-23-qwen3-4b-p3-it2c-heldout` (convs 33..): all replay verdicts
  pass. (Corrected by the re-review: replay warm TTFT moved +3.2% / +6.0% vs
  slice2, +1.4 / +2.6 pp vs baseline, and warm3 req/s -1.5 pp, so G4 was NOT
  within 0.5 pp of slice2; official rerun `qwen3-4b-p3-it2d` instead.)

- it2c done (50e186b, 056eb4b, 8af8fe4): all it2 review findings resolved
  (S4 partly: a hung save RPC still blocks, needs a server-side deadline).
  Slice and held-out replay slice pass. (The "G4 within 0.5 pp of slice2, the
  official it2 run stands" claim was wrong for replay warm, see the it2c
  re-review entry below.) Independent re-review of it2c running; Qwen3-4B is
  accepted for G0-G4 when it passes (G5 pending).
- P4 DeepSeek-V2-Lite-Chat started on branch `feat/rsi-deepseek-v2-lite`
  (model PR, stacked on #473), TP=2 (31 GB bf16 does not fit one 4090),
  GPUs 0-3, isolated onboarding agent. Its runs import connector_v2 from its
  own remote tree via PYTHONPATH (the shared venv's .pth points at
  `~/kexi/pegaflow-semantics-rsi/python`).

- P4 gate finding (DeepSeek-V2-Lite-Chat, TP2, TRITON_MLA, batch invariant):
  vLLM 0.32 MLA prefill attends to the cached context separately
  (`_compute_prefill_context` + `merge_attn_states` at the computed-token
  boundary), so outputs after a hit depend on where the hit ends; plain vLLM,
  no connector: prefix warmed to 256 changes logprobs, 512 diverges at output
  token 3. The honest selftest therefore fails `G2.gsm8k_outputs` (60/128) and
  `computed_exact`, while every loaded block is byte-exact and the negative
  selftests still fail. Gate fix assigned to the gate owner on `feat/rsi-g5`:
  `split_invariant` established by a gate-run split control (not declared by
  the agent); loaded-block byte checks and accuracy stay hard; computed_exact
  only against same-split references; outputs bounded by the control's own
  divergence rate. P4 official run waits for that merge.

- it2c re-review: APPROVE-WITH-FIXES. M1': a later turn of a tainted request
  can prefix-hit blocks computed on the bad bytes (recompute policy evicts
  nothing) and save on top of them; also a rewound loader leaving before it
  recomputes. Fix direction: fail closed (stop saves for the process after a
  taint). Should-fix: the "within 0.5 pp" no-rerun rule was misapplied
  (replay warm TTFT +1.4/+2.6 pp vs baseline), so the it2 official run does
  NOT stand for the final code: official rerun `qwen3-4b-p3-it2d` (GPU4, with
  old + native arms) after the fix. Qwen3-4B acceptance waits for it.

- G5 re-review: APPROVE-WITH-FIXES, merge after: (1) `G2.digest_self.cap`
  could not catch wrong loads (reference built from undigested timed passes is
  never exhaustive; 79 corrupted blocks passed as `uncertain`), (2) a vacuous
  cap digest must fail for the arm under test. Same change: E_C must not credit
  the arm's own local-cache variance; replicated MLA bytes were doubled at TP2;
  `cap_convs` auto-scales until C(0.5) >= 2x GPU KV (fixed 64x12 gives
  DSV2-Lite only 1.87x); GPU KV bytes from distinct tensors, not per group.
  Verified: G3 identical on 125 probe dirs, M1/M2/S1/S2/S3 fixed in code.

- P3 it2d (8a05b7c): M1' fixed by failing closed. The first sync-load failure
  vLLM reports (`invalid_block_ids`) disables all saves for the process: under
  the recompute policy vLLM evicts nothing, so blocks computed on bad bytes
  stay prefix-cached and any later request can extend them. Tests
  `sync-failed-sharer-next-turn`, `sync-failed-loader-preempted`; 166 passed.
  Teardown backport df01403 fast-forwarded (first official attempt
  `...-it2d-aborted1` died in gate teardown; unchanged relaunch is official).
  Official `2026-10-23-qwen3-4b-p3-it2d` (GPU4, `official: true`): every v2
  verdict passes; `old` (reference) fails G4.gsm8k_warm +3.4%, recorded only
  (this branch's gate counts it, so exit 1).

  | G4 check | v2 | old | baseline |
  |---|---|---|---|
  | replay warm TTFT p50 (warm2/3) | 0.563 / 0.538 s | 0.677 / 0.601 s | 2.544 / 2.547 s |
  | replay warm req/s (warm2/3) | 10.12 / 9.94 | 9.99 / 10.03 | 3.42 / 3.42 |
  | replay cold TTFT p50 | 1.287 s | 1.334 s | 2.804 s |
  | replay cold req/s | 6.70 | 6.75 | 3.26 |
  | gsm8k cold s | 165.96 (+0.9%) | 166.73 (+1.4%) | 164.45 |
  | gsm8k warm s | 164.29 (+0.0%) | 169.93 (+3.4%) | 164.28 |

  | hits | local | ext | total | H* | E_ext | excess |
  |---|---|---|---|---|---|---|
  | baseline gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0 | 0 |
  | old gsm8k warm | 0.928 | 0.065 | 0.993 | 0.992 | 1.0 | 1170 |
  | v2 gsm8k warm | 0.920 | 0.072 | 0.992 | 0.992 | 1.0 | 0 |
  | baseline replay warm | 0.378 | 0 | 0.378 | 0.998 | 0 | 0 |
  | old replay cold / warm | 0.386 / ~0.36 | 0.478 / ~0.64 | 0.864 / 0.998 | 0.864 / 0.998 | 1.0 | 0 / 195 |
  | v2 replay cold / warm | 0.381 / 0.348 | 0.484 / 0.650 | 0.864 / 0.998 | 0.864 / 0.998 | 1.0 | 0 |

  Save stats: 0 failed, 0 flushes, max 3 pending, step-thread submit 13-16 ms.
  Qwen3-4B: G0-G4 accepted pending the it2d review; G5 waits for the gate
  rerun.

- it2d review: ACCEPT-WITH-FIXES, no must-fix. Qwen3-4B G0-G4 accepted at
  8a05b7c; G5 pending r3. it2e (review fixes, no success-path change):
  - S1: every load is async once saves are disabled;
  - S2: lifecycle tests pin the two surviving mutants;
  - S3: provenance sentence fixed, and the connector logs its import dir;
  - S5: `min_hit_tokens` deleted, `loading_wait_ms` kept with the evidence
    gap recorded;
  - S4: exit 1 is the reference `old` arm only.
- Native reference (vLLM OffloadingConnector), record only and noisy (four
  gate runs on the host at once), `2026-10-23-qwen3-4b-p3-it2d-native`:
  - gsm8k cold/warm +7.5% / +4.0% vs the same run's baseline (G4 fail);
  - replay warm2/3 TTFT 0.505 / 0.459 s, req/s 9.63 / 9.76;
  - E_ext 1.0, excess 1170 / 195 (same as `old`);
  - G0/G2 digest fail because it runs LBHNC, not LBNHC (accuracy and
    outputs match).

  The first attempt died to the gate's aiohttp keep-alive race (fixed by
  8462ed9). The same-run v2 vs native comparison waits for G5 r3. ITERATIONS
  has an "it3 candidate" on load-path cost: native submits one batched
  `cuMemcpyBatchAsync` per load, v2 one `cuMemcpyAsync` per merged run;
  14.2 vs 21.9 GB/s effective H2D on replay warm2. Both copy 64 KiB
  per-layer pages.

- P4 start (isolated onboarding agent, DeepSeek-V2-Lite-Chat, MLA, TP2 on
  .44 GPU pairs 0,1 / 2,3; worktree `.worktrees/rsi-deepseek-v2-lite`, branch
  `feat/rsi-deepseek-v2-lite` from `feat/semantics-rsi` 8af8fe4; remote dir
  `~/kexi/pegaflow-rsi-deepseek-v2-lite`). Plan: (1) self-test at TP2
  (plain, baseline, 4 selftest arms, 128 gsm8k) to get spec.json + gate.json;
  (2) map MLAAttentionSpec -> (Kind, Shard, unit) from the dump and vLLM
  0.32 source; (3) adapter for `MLAAttentionSpec` (the registry dispatches
  by exact class name, so the full_attention adapter does not apply); (4)
  slices, then one official run with `old`; (5) Qwen3-4B slice if shared code
  changes. Runs import connector_v2 from the P4 remote tree (PYTHONPATH),
  native `.so` copied from `pegaflow-semantics-rsi`, server binary reused.
  Record: `docs/rsi/models/DeepSeek-V2-Lite-Chat/ITERATIONS.md`.

- P4 it4/it5 (2026-10-24, connector-only, no semantics or server change):
  save batching is a model-agnostic policy knob, on by default:
  `save_batch_steps = 32`, plus `save_batch_max_kv_usage = 0.9`, which sends
  held saves at once when vLLM's GPU KV usage is at least 90%. A fixed 1 s
  staleness cap (`Worker.SAVE_HOLD_MAX_S`, formerly the `save_batch_ms` knob,
  folded per review M3) and a 16 MiB merged-RPC cap are constants.
  Sync-save layouts skip batching. Every run below is non-official.
  - Unpressured DSV2 TP2 3x (`it5-rep-01`, replay only, 8x6): the G4
    replay verdicts pass 3/3 (cold p50 -1.9 to +0.6% vs baseline), against
    +5% for batching off. gsm8k G2/G3/G4 have not been evaluated with
    batching on.
  - Pressured DSV2 (`--num-gpu-blocks 1200`): without the KV-pressure send,
    the batch hold cost +53% cold p50. With it, the default is level with
    batching off, with no added preemptions.
  - Qwen3-4B TP1 (8x6, max KV usage 31%): no regression (`it4-nonreg`,
    `it4-qrep-b`, `it5-nonreg`). The official-regime check (32x12, KV up to
    99.6%, vs it2d) is pending (review M2).
  - Review `reviews/P4-dsv2-lite.md`: accept after must-fix. M1 (official
    DSV2 run under rsi-g5's split mode, including gsm8k) and M2 wait for
    the rsi-g5 merge.
  - Gate issue seen: a vLLM `--port` bind race (`EADDRINUSE`) killed one
    run_gate (`qwen3-4b-p4-it4-qrep`).
- P4 official runs and G5 (2026-10-24, connector-only, base server
  d1970ead; record `models/DeepSeek-V2-Lite-Chat/ITERATIONS.md`):
  - MF1 `2026-10-24-dsv2-lite-p4-mf1` (07f5d96) and MF2
    `2026-10-24-qwen3-4b-p4-mf2` (d56fcb2) are official and pass G0-G4.
    MF2 is no regression vs it2d. The DSV2 old supplement
    (`...-dsv2-lite-p4-old-supp`) has old passing everything.
  - G5 regressed with batching (Qwen3-4B 0.5: 0.777 vs 0.842 at 8a05b7c).
    A/B (`...-ab-a-d56fcb2` / `...-ab-b-706006c`): batching off restores
    it.
  - Cause: `_release_held` folded every held step into one save entry
    per layer. The base server pins one region per entry until its last
    block is evicted (see "Server limitations").
  - it6 (43a3759) keeps one entry per held step. Same RPC count, DSV2
    8 x 6 cold p50 -0.5% vs baseline.
  - Official runs on the it6 tree a1e7fd2 (arms plain, baseline, old,
    native, v2; lmcache pending, arm not comparable yet):
    - MF4 DSV2 `2026-10-24-dsv2-lite-p4-it6-off-a1e7fd2` (pgid
      3030292): v2 passes every verdict.
    - MF5 Qwen3-4B `2026-10-24-qwen3-4b-p4-it6-off-a1e7fd2` (pgid
      3092773): v2 passes G0, G3, G4 and every other G2. It fails
      `G2.digest_self.cap`: 1.008% preempted-uncertain vs the 1% limit,
      0 mismatched. Open for the coordinator and the gate owner.

    | model | C/U | v2 | old | native | lmcache | H_lru | H*_C |
    |---|---|---|---|---|---|---|---|
    | DSV2 | 0.25 | 0.436 | 0.512 | 0.578 | pending | 0.673 | 0.851 |
    | DSV2 | 0.5 | 0.844 | 0.845 | 0.850 | pending | 0.847 | 0.925 |
    | DSV2 | 0.75 | 0.863 | 0.889 | 0.933 | pending | 0.919 | 0.968 |
    | Qwen3-4B | 0.25 | 0.392 | 0.398 | 0.542 | pending | 0.581 | 0.862 |
    | Qwen3-4B | 0.5 | 0.838 | 0.820 | 0.861 | pending | 0.857 | 0.929 |
    | Qwen3-4B | 0.75 | 0.883 | 0.873 | 0.927 | pending | 0.915 | 0.970 |

  - DSV2 still trails old by 0.076 at 0.25 and 0.026 at 0.75. The
    per-request entry split is untested and deliberately not done.
    Listed for the reviewer.
- P3 it2e non-regression (fc06c9b):
  - Slice `2026-10-23-qwen3-4b-p3-it2e-slice`: all correctness and replay
    verdicts pass. gsm8k cold failed at +9.9%. A 10 s zero-throughput stall
    matches a P4 160 GiB pool teardown plus an allocation on the host.
  - gsm8k-only rerun `...-it2e-gsm8k-rerun`: cold +0.5%, warm -0.2%, pass.
  - Second native record-only sample (`...-it2d-native-e9f`, gate e9f3f33):
    the same hits, noisy timings.
- P3 it3 (measurement only, `2026-10-23-qwen3-4b-p3-it3-backend`):
  - v2 loads are blocks x 36 layers x 64 KiB copies (p50 7704 per load).
  - Direct backend: 38.2 ms per load at 15.0 GB/s.
  - Kernel backend: 48.3 ms at 12.3 GB/s (+26%); warm2 TTFT 0.709 vs
    0.518 s. Kernel rejected.
  - The batched-submit lever (native: 21.9 GB/s) is server-owned, so the
    numbers went to the server agent. P3 it3 stops.
- P5 it0 (Qwen3.5-4B, `feat/rsi-qwen3.5-4b`):
  - Spec dump: 3 `MambaSpec` (GDN, align) groups + 1 `FullAttentionSpec`,
    block 528, aliased 2162688 B pages. Self-test stopped at the honest
    arm's `NotImplementedError` for MambaSpec; the probe digested group 3
    only. Gate items 1-4 accepted, built on `feat/rsi-gate-hybrid` after
    feat/rsi-g5 (940def4 already fails G2 on undigested groups).
  - Mapping: GDN = `Kind::Checkpoint(528)`, attention = `Full(528)`. No new
    Kind; no Semantics changelog entry.
  - Adapter draft `adapters/mamba.py` + stub-spec tests. `LayerDecl.layout`
    (namespace-only) added for the packed state shapes/dtypes; empty for
    existing adapters, so Qwen3-4B's namespace is unchanged.
- P5 it1 (Qwen3.5-4B, `feat/rsi-qwen3.5-4b`):
  - Core: checkpoint groups save from vLLM's `boundary_state_offloads`
    (only the boundary `saveable` picks, only if offered); layouts with a
    checkpoint group save synchronously; loads unchanged (the state at p is
    always past the local hit). `hit stats` log: per-group miss attribution.
  - `--prefix-cache-retention-interval 528` requested from the gate owner
    for every arm; not set by the connector.
  - Ordering check on GPU4 (temporary tool `~/kexi/rsi-tools/p5`, not in the
    repo): all checkpoint saves from offered blocks, saved bytes equal an
    unprimed recompute up to rounding, every load byte-equal to the saved
    page after poisoning the GPU cache, outputs equal (order2: 6/6; order3
    with 6 preemptions: 5/6; p4 diverges at token 152 before its own
    preemption). Rerun at 863cc99 as order4 (review should-fix 2): 27/27
    checkpoint and 29/29 full loads byte-equal, p4 flips at token 152 by
    one bf16 step and matches the unprimed alone run.
  - it2 leg 1 (`2026-10-24-qwen3.5-4b-p5-it2-ctrl`, plain vLLM, no
    connector, GPU4, pgid 2328944, 02:55:32Z-02:59:27Z): batched p4 flips
    at token 152 to the same token with the same 0.125 margin under the
    tight config (0.5 with 200 blocks), and p4 alone equals order4 C. A
    local-hit resume equals alone. Closed as GDN decode batch
    non-invariance in vLLM 0.32, not the connector. Gate implication
    (`batch_invariant` for this model) goes to the gate owner.
  - Non-regression on c4c1615: the Qwen3-4B slice
    (`2026-10-24-qwen3-4b-p5-it1-nonreg-b`, GPU4) passes every verdict, with
    the same deltas as P3 it2e. The DSV2 TP2 slice
    (`2026-10-23-dsv2-lite-p5-it1-nonreg-b`, GPU 4,5, 64gb, P4 55e4ff1 +
    P5 merged) matches P4's it2 slice: loads are byte-exact, the split-only
    failures are unchanged, and replay cold TTFT fails at +10.2%, the open
    P4 item. See ITERATIONS.md it1.
  - Two earlier non-regression launches were invalid. They lacked
    `PYTHONPATH=<tree>/python`, so vLLM imported the shared tree's
    connector through the venv `.pth`. They are discarded, and the skill
    has a lesson on it.
  - `LayerDecl.layout` is connector-side (python `adapters/__init__.py`),
    not in pegaflow-semantics and not in key derivation: no S-entry. The
    namespace V1 vector for Qwen3-4B is unchanged: stub `cf6b5725`, native
    `cd8acc0c` at 6545352; on P4's base (55e4ff1) `a25ea150` / `bdda8997`
    before and after the P5 changes.

- G5 implemented on `feat/rsi-g5` (gate agent): `oracle.py` capacity
  simulation (`U`, clairvoyant `H*_C`, group-oblivious `H_lru`, per-group
  stored / evicted / stranded diagnostics), `run_gate` workload `cap` (fresh
  PegaFlow `--pool-size C`, `C = 0.5 x U` from the baseline replay cold pass,
  passes `cap_cold`, `cap_warm`, `cap_warm2..N`), verdicts `G2.digest_self.cap`
  and `G5.replay_warm` (`E_C >= 0.9`) required for the connector under test,
  `cap` now part of the official profile, `native` reference arm
  (OffloadingConnector + RsiProbe, record only). G3 numbers re-derived
  identically on all p0-final / p3-it1 probe dirs. Slice
  `2026-10-23-qwen3-4b-g5-slice` (8 x 6, U 4.48 GiB, C 2.24 GiB): E_C
  connector_v2 0.40, native 0.20, old 0.0 (0 server hits); oracle_C hit rate
  0.903, LRU 0.827, local 0.79. Server limit found: reclaim drops 512 blocks
  per round, so a small pool keeps only ~0.5 C resident (GATES.md G5).

- G5 review (APPROVE-WITH-FIXES) applied on `feat/rsi-g5`: pool size must be
  proven for PegaFlow arms (M1); server `pegaflow_load_bytes_total` must
  cover every external-hit block, connector-side caches are a violation
  (M2); `E_C` subtracts the baseline's local hit for the same
  pass/conversation/turn, per-arm local kept as a diagnostic (S1); a
  reference arm's crash is recorded, not fatal (S2); `old` and `native` are
  `required: false` throughout (S3). User capacity concern (slice C 2.24 GiB
  = 0.22x GPU KV): cap is now its own 64 x 12 trace, baseline runs it first
  (U + reference locals), fractions 0.25/0.5/0.75 per arm (0.5 gated, rest
  a curve), official only if C(0.5) >= 2x GPU KV. Freeze policy: warm passes
  start rotated by j/4; a run whose under-test freeze/H*_C >= 0.9 is not
  official. G3 re-derived identically on all 45 recorded probe dirs.

- G5 pre-fix reference `2026-10-23-qwen3-4b-g5-official` (rev 369b253,
  32 x 12, U 25.81 GiB, C 12.91 GiB = 1.24x GPU KV): `warm_timed` E_C
  connector_v2 0.216, old 0.124, native 0.558; H*_C 0.79-0.84, H_lru
  0.52-0.56. Both PegaFlow arms are below H_lru, consistent with the server
  reclaim-batch / prefix-oblivious LRU limit (GATES.md G5). Not comparable to
  post-fix runs (different trace, per-arm local, no rotation).

- `lmcache` reference arm (LMCacheConnectorV1, local CPU = pool / C bytes
  exactly, chunk 256) wired on `feat/rsi-g5`, lmcache 0.5.6 in the overlay
  `~/kexi/overlays/lmcache` (shared venv untouched). lmcache: not comparable
  on vLLM 0.32 (wrong KV): smoke `2026-10-23-qwen3-4b-g5-lmcache-smoke` gsm8k
  warm accuracy 0, external digests 64/64 (gsm8k) and 468/468 (replay)
  mismatched, not poison and not another key's bytes; the in-tree adapter
  fails to import (GATES.md G5). Left out of the official run.
- `G2.digest_self.cap` fail for `old` / `native` on the fix slice was a gate
  artifact: cap sampled every 4th conversation and their few external loads
  (old: none; native: 6 requests in convs 2/5/6/7) fell outside it, so
  `judged > 0` failed with 0 mismatches (connector_v2: 2 sampled). Cap now
  digests every conversation and an empty sample passes flagged `vacuous`.
  Under the old rule the pre-fix official run had all three passing (old
  541/541, native 8567/8585 + 18 uncertain, v2 1375/1375 matched).

- G5 re-review (APPROVE-WITH-FIXES on 20e6b58..6510116) applied:
  `G2.digest_self.cap` accepted wrong loads (undigested timed passes made
  the reference non-exhaustive; the reviewer's 79/158 corruption passed):
  cap now digests every request of every pass and requires every loaded
  block judged (only preempted-episode uncertainty, <= 1%, flagged);
  vacuous passes only for reference arms. `ext` subtracts
  `max(ref, own local)` (slice: connector_v2 / old E_C 0.0775 -> 0 with 0
  server load bytes). Block / GPU KV bytes count `tp / kv_tp_replicas`
  copies (PegaFlow saves one MLA copy at TP2: 5,799,278,592 B = 11,653 x
  497,664) and GPU KV = Σ KV tensor sizes. `--cap-convs auto` (Qwen3-4B 64,
  est. C/GPU-KV 2.44; DeepSeek-V2-Lite TP2 76, 2.22; a fixed 64 x 12 gives
  it ~1.87). Re-scored offline: pre-fix official E_C unchanged (v2 0.216,
  old 0.124, native 0.558) but its `G2.digest_self.cap` now fails for
  native (18 external + 1 local unjudged, all from undigested requests):
  old-sampling records cannot pass the exhaustive rule; a rerun is needed.
  G3 `summarize` identical on all 153 recorded probe dirs.

- Gate teardown (d646ac3, backport df01403 on fix/rsi-teardown-backport):
  pegaflow-server with a 160 GB pinned pool took > 30 s to exit after
  SIGKILL and `proc.wait(timeout=30)` aborted it2d's official run; the gate
  now waits on the whole process group (120 s after SIGTERM, 300 s after
  SIGKILL) and logs the exit time.
- `split_invariant` (P4 finding): vLLM 0.32 MLA prefill attends to the cached
  context separately (`_compute_prefill_context` + `merge_attn_states` at
  `num_computed_tokens`), so on DeepSeek-V2-Lite the computed KV and greedy
  outputs depend on where a hit ends; an honest connector failed
  `G2.gsm8k_outputs` (warm 60/128) and `computed_exact` with every loaded
  block byte-exact. New gate.json field (default true), established by a
  plain-arm split control (`split_ref` vs `split`, each prompt pre-warmed);
  with false, `computed_exact` judges only same-split copies and
  `G2.gsm8k_outputs` is bounded by the control's rate + 0.02. Controls
  measured (full gsm8k 1319, pre-warm took on 1319/1319):
  Qwen3-4B FLASH_ATTN 0 differ / 0 logprob differ (split-invariant);
  DeepSeek-V2-Lite TP2 TRITON_MLA 647 differ (0.4905; first 128: 0.4688),
  1319/1319 logprobs differ (split-variant). Re-judged offline with
  `split_invariant=false` + that control: P4 self-test honest passes every
  G2 check (ext 1084/1084, warm 60/128 = 0.469 <= 0.489), liar / bitflip /
  offbyone still fail both digest checks (2168 / 256 / 828 ext mismatched);
  P4 old slice: old loaded blocks 10390/10390 matched, computed_exact
  27526/27526 on same-split copies (coverage 0.70 vs baseline),
  `G2.gsm8k_outputs` 642/1319 = 0.487 <= 0.511. Qwen3-4B: every verdict
  unchanged on the 24 recorded runs (5580 verdict leaves) apart from the
  new `G2.split_control`, which old records lack. DeepSeek-V2-Lite's
  gate.json must set `"split_invariant": false` (P4's file).
- G5 official r3 (2026-10-23, .44 GPU5): gate e9f3f33, connector_v2 at
  8a05b7c (`--under-test-path`, tree verified blob-for-blob afterwards),
  official, uncontaminated (max timed p99 ratio 3.2x vs the 10x limit).
  Qwen3-4B connector_v2 passes G0-G4. It fails G5: E_C(0.5) is 0.684 vs
  0.9, with hit 0.842 vs H*_C 0.929, about LRU level (H_lru 0.857). At
  C/U 0.25 / 0.75, E_C is 0.084 / 0.669. Tables are in
  `models/Qwen3-4B/ITERATIONS.md` "G5 official (r3)".
- G5 re-review 2 (accept-with-fixes on 4797a05..e9f3f33) applied in
  91b6a8e..940def4; resolution table in `reviews/G5-gate-rereview2.md`.
  Split-variant G2 is now exact: warm outputs == plain `split` (aligned
  prompts reported only), cold == baseline cold, warm computed KV judged
  against the baseline's `split` pass with coverage >= 0.9 per pass (DSV2
  honest self-test warm coverage 0 -> 1.0; liar/bitflip/offbyone fail;
  P4 old and it1 slices pass G2). Provenance: overlay with only `pegaflow`
  for `--under-test-path`, `G0.provenance` on module files, verified tree
  revs (blob-for-blob vs GitHub) and binary sha256 in `summary.provenance`.
  Teardown failures abort the run; SIGTERM/SIGHUP tear stages down;
  `--pool-size` 64gb with an early U check; untimed `_warmup` request per
  stage; contaminated timed passes (p99 > 10x baseline) make a run
  non-official; `G2.digest_groups` fails a transferred group without
  digests (hybrid / Mamba models fail G2 until Mamba state is digested).

### 2026-10-24

- G5 re-review 3 (M4-M7, S7-S9, N5) applied in c7f7520..b3d43b4. The
  resolution table is in `reviews/G5-gate-rereview3.md`.
  - Every new run passes `G0.spec` again (M4).
  - The arm under test imports a verified snapshot of the rev's blobs.
    Symlinks, untracked files, extra `.so` and pyc all reject the tree.
    Trees for `--under-test-path` must be pyc-free (`git archive`).
  - Unverified gate and venv trees make a run non-official.
  - The arm runs in a private mount namespace that hides the other arms,
    with SYS_ADMIN and SYS_PTRACE dropped. This is deterrence, not a
    boundary (uid 0).
  - A static scan of the snapshot flags instrument access.
  - `VLLM_PLUGINS=""` for every vLLM.
  - Gate fixes and run hygiene:
    - The connector module's import path is checked.
    - The aligned-prompt exemption is exact.
    - Contamination is two-sided.
    - `G4.warmup_seconds` is reported.
    - Stage ports are leased (`PortRace`, never retried).
    - CUDA toolchain preflight.
    - `--replay-skip` for held-out slices (never official).
- GPU3 run on 7d477c5 (02:07:41Z-02:14:40Z, gsm8k 64, 16gb pool):
  - G0.spec and G0.provenance pass.
  - connector_v2 at 8a05b7c served inside the isolation: accuracy equal to
    baseline, G3 E 1.0.
  - The namespaced groups were torn down.
  - The server log shows the full 16 GiB pool and 36 CUDA IPC layers
    registered, with no errors.
- The Qwen3-4B re-judge (39cd826 vs b3d43b4, 49 records) left the G3
  numbers unchanged (147/147) and overall passes unchanged. The only flag
  change is M4's `G0.spec` fix.

### 2026-10-25

- Hybrid gate review (`reviews/hybrid-gate.md`, ACCEPT-WITH-FIXES at
  8789dde) fixes on feat/rsi-gate-hybrid; resolution in the review file's
  response section:
  - M1: start states digested in `handle_preemptions`, before the Mamba
    precopy and any sync load; new `selftest:sync` arm.
  - M2: negative arms judged per group (`G2.caught_per_group`), real
    offbyone state sabotage, sabotage counts.
  - S1: records judged by the gate rev they ran under (`na: record
    predates`). S2: `G2.gsm8k` gated on `alone_ref` without batch
    invariance, honest arm included. S3: `preempt` official on
    state-group models. S4: flag parser spellings, compile config in the
    spec dump. S5: per-arm compile cache. S6: tests (282).
  - S5 follow-ups: fresh per-arm caches re-timed vLLM's FLA autotune
    kernels per arm (Qwen3.5 baseline != plain), fixed by seeding each arm
    with the baseline's Triton autotune picks (3d6b8e6); Qwen3-4B gsm8k
    now compiles with deterministic inductor too (2404394).
  - Self-test `qwen3.5-4b-selftest-3d6b8e6` passed (every negative caught
    in every group); Qwen3-4B slice `qwen3-4b-gate-hybrid-slice-2404394`
    baseline passes `G2.batch_control` and `G0.vllm_args`.
  - Re-review (363441c) R1-R3 fixed: `G0.spec` judges spec fields the
    record's probe did not write as `na_fields` (re-judge of
    `qwen3-4b-p4-it5-nonreg` no longer flips); preempted blocks vLLM zeroes
    or COW-copies into this step are dropped (`preempt_dropped`), an
    all-zero digest is never a reference, the spec dump records
    `model_runner` (V2) and `probe.schema`; `G2.caught_per_group` needs bad
    >= sabotaged per group. N2: seeding refuses non-tile autotune entries.
  - FOLLOW-UP (DSV2 branch, R4, not this branch): when DeepSeek-V2-Lite
    adopts this gate after P4 acceptance, in the same commit: gate.json
    `gsm8k_vllm_args` with deterministic inductor; regenerate the committed
    `spec.json` (retention, `compilation`, `model_runner`, `probe.schema`);
    one plain+baseline gsm8k slice. Until then DSV2 records are re-judged
    with ce645d7.
- P5 it2 started (`feat/rsi-qwen3.5-4b`, merged `feat/semantics-rsi`
  f8eac30 = hybrid gate). Plan: (1) adapt connector_v2 to the hybrid gate
  if a slice shows it needs it (preempt pool, start-state digests before
  the forward, sync-load rules), `hit stats` logged unconditionally at
  shutdown; unit gate. (2) GPU5 slices (plain, baseline, old, native, v2;
  gsm8k + replay + cap + preempt), one hypothesis per iteration, SKILL
  step 5 checklist (checkpoint saves only at resume boundaries, no
  unresumable state, group-coherent resume, save cost off the step).
  (3) Tell the orchestrator before the official run (P4 merge first).
  Server stays at the base build (`pegaflow-semantics-rsi` 056eb4b).
- P5 it2 slice s1 (25fcccb, GPU5, not official): correctness all pass
  (digests in all 4 groups, preempt 106/106, alone_ref = baseline, E_ext
  1.0); fails G4 cold replay (TTFT p50 +27.5 %, req/s -6.2 %) and G5 cap
  (0 bytes loaded at every fraction). `hit stats` never reached vllm.log
  (the API server force-kills EngineCore, `shutdown` never runs): bc9c38a
  logs it when idle and every 30 s. Next: cap-only diagnostic slice
  (baseline + v2) to attribute the 0-load per group, then the save-cost
  hypothesis (checkpoint saves off the step, fenced on block reuse).
- P5 it2 s2cap (d2a484d, cap only, sub-floor pool C = 1.63 GiB = 0.05x
  GPU KV, no G5 conclusion): v2 loads 0 bytes; `hit stats` `missing=1,2,3`
  4 of 160 lookups (full group present, every GDN group evicted). Official
  `--cap-convs auto` sizing for Qwen3.5 reported to the orchestrator (est.
  U 147 GB at 203 convs; estimate/measured 0.78 at 8 x 6 suggests the
  measured floor misses at 203). Merged P4 (563a693) as 554ad9d.
- P5 it2 hypothesis H1 (next): G4 cold replay pays the sync checkpoint
  save (`flush_ms` 1518 / 33.5 s). Change: for layouts that must save
  synchronously (block zeroing / COW / checkpoint groups), the scheduler
  pins every block it plans to save in vLLM's block pool (`touch`) and
  unpins it once every worker reports the save done (worker metadata,
  min over ranks; `has_pending_push_work` keeps the engine stepping
  meanwhile). A pinned block is never reallocated, so it is never zeroed,
  COW-written or reused before the copy lands; the worker then saves
  asynchronously (no flush in `wait_for_save`, no preemption flush, no
  finish hold). Layouts that save async today are unchanged. Measure: v2
  replay cold TTFT p50 / req/s vs baseline, `save stats` flush_ms.
- P5 it2 s3 (5667cd7 = H1, not official): correctness all pass (digests
  in 4 groups, preempt 106/106 ext, alone_ref delta 0, no eviction).
  `flush_ms` 0; replay cold req/s -1.7% (pass), TTFT p50 +3.8% (fail;
  s1 +27.5%). gsm8k +5.3% cold/warm (fail) with 1 save step, so not from
  saves: 6046 prefix queries for 2703 lookups, each pass 4 group queries
  + lease releases for a hit that cannot beat vLLM's local 1056 tokens.
- P5 it2 hypothesis H2 (next): skip lookups that cannot beat vLLM's
  local hit. The bound is `semantics.resolve` over optimistic answers
  (unqueried groups count as fully present), checked before any query
  and after group 0. Full-only layouts skip only when no full block lies
  past the local hit. A skipped group creates no lease. No policy knob.
  Measure: gsm8k cold/warm s, replay cold TTFT p50 vs baseline (read
  replay cold p50 on the official-size run; the slice is noisy).
  Correctness unchanged.
- P5 it2 s4b (800630d = H2, not official; no cap): every v2 verdict that
  ran passes. gsm8k cold -0.1%, warm -0.7%; replay cold TTFT p50 +1.6%,
  req/s -1.3%. gsm8k prefix queries 6046 -> 23. Hits unchanged (replay
  warm 0.936 = H*). Merged `feat/semantics-rsi` 9a02fb5 (cap sizing) as
  be0622f. Next: official run on GPU5 (gsm8k, replay, cap, preempt).
- P5 it2 official (`2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`, c5d3f04,
  GPU5, base server, all 256 cap conversations, C(0.5) 2.52x GPU KV):
  official, and v2 passes every verdict. 0 digest mismatches in 4 groups
  (cap 59627/59627 external). G4: gsm8k +0.34% / -0.35%, replay cold TTFT
  p50 0.512 vs 1.204 s, warm 0.300 vs 1.112 s. Hits: v2 = H* unlimited;
  at capacity v2 = H_lru = old (within 0.002). E_C at 0.5 is v2 0.828,
  old 0.867, native 0.815. The v2-old gap is v2's own local hits above
  the baseline reference, which earn nothing by design; adding them back
  gives old's E_C. Judgment: close enough. Next: independent review.
- For the gate owner (P5 finding, gate code not edited): G5 E_C > 1 at
  C/U 0.25 on Qwen3.5-4B (H*_C 0.690 < H_lru 0.794, every arm) is the
  oracle, not the accounting. Belady's use sets are `needed(P)` at each
  request's unlimited P only. For a Checkpoint group, `needed(q < P)` is a
  different checkpoint, so fallback checkpoints look unused, and Belady
  refuses or evicts them (3368 not admitted vs 2562 stored per GDN group).
  LRU resumes there: 995 requests, 2.69M tokens (0.154 of prompt) at
  0.25, 239 / 0.52M at 0.5, 0 at 0.75. A Belady with uses at every block
  end <= P gives 0.841-0.843 / 0.889-0.891 / 0.941-0.946, above H_lru on
  every request, and E_C v2 / old / native 0.881 / 0.920 / 0.874 at 0.25.
  Suggest a fallback-aware use set for Kinds whose `needed` is not
  prefix-closed (or the max of both variants). Full-only models are
  unaffected. Details: `models/Qwen3.5-4B/ITERATIONS.md` "it2 official".
- P4 final-review fixes (`reviews/P4-final.md`, ACCEPT-WITH-FIXES).
  Changes are in connector_v2's scheduler, its unit test and the docs; no
  semantics or server change.
  - M2: per-request save entries became it7 (a59a254, `plan_saves` emits
    one GroupSave per storage group and request). The base server keeps a
    whole pinned region alive until every block in it is evicted, so mixed
    entries strand bytes.
    - DSV2 A/B (`2026-10-25-dsv2-lite-p4-m2-a59a254`): resident/C went
      from 0.78-0.90 to 0.91-0.96, and hit at 0.75 from 0.863 to 0.899.
    - Qwen3-4B cap regression: 0.5 went from 0.838 to 0.859.
    - The DSV2 8 x 6 cold check failed only on baseline noise; the
      same-time control on a1e7fd2 confirmed this, and it was accepted.
  - The DSV2 0.25 gap stays unexplained. The server-faithful sim (both
    query modes) rules out the local-hit query, the resident deficit, lost
    saves and the server. The gap is 0.06-0.12, against old's own noise of
    0.038.
  - M1: official reruns on it7, DSV2
    `2026-10-25-dsv2-lite-p4-it7-off-a59a254` and Qwen3-4B
    `2026-10-25-qwen3-4b-p4-it7-off-a59a254`. v2 passes every verdict on
    both. Qwen `G2.digest_self.cap` is 0.45% (MF5 1.008%), DSV2 0.85%.
  - S1-S5: done. The tables are in "P4 acceptance tables" above. Gate stays
    at ce645d7 until P4 is accepted.
- Cap sizing (feat/rsi-gate-cap-sizing): `UniqueEstimate` counted every
  block boundary of every Checkpoint group, but the oracle stores a
  Checkpoint block only at step ends on its boundary; Qwen3.5-4B's auto
  plan (203 conversations, est. 147.05 GB) was at the floor with no
  margin: on the 4f3cb97 trace its first 203 conversations measure
  133.20 GB = 2.001x GPU KV vs a 133.15 GB floor (a real 203-conversation
  run interleaves differently at the tail, so either outcome). It now applies the oracle's rules per Kind with vLLM 0.32's
  align-mode step ends (`prefill_step_ends`). s1 8 x 6: 3.39 GB est vs
  3.49 GB measured (was 4.47 GB). Official plans: Qwen3-4B 64 and
  DeepSeek-V2-Lite 76 unchanged (same bytes); Qwen3.5-4B all 256
  conversations est. 136.33 GB, above the floor (133.15 GB) but under the
  1.1 margin (146.47 GB). Floor and margin unchanged. Baseline-only cap
  run on GPU6 at that size (`qwen3.5-4b-cap-sizing-baseline-4f3cb97`, not
  official by design): measured U 167.96 GB = 5.05x GPU KV, C(0.5) 2.52x,
  floor passes. Full-attention blocks exact (2717); Checkpoint 2348 per
  GDN group measured vs 1734 estimated: 32 concurrent workers split
  prefills into more step ends than one request alone (est/meas 0.81).
  A cap-only run
  (`--workloads cap`) had no earlier baseline spec dump and fell back to
  64 conversations; auto now sizes from the committed dump there.
  - Qwen3.5-4B's official plan uses all 256 conversations
    (`enough: false`, measured 2.52x GPU KV). That is expected, not a
    defect: the file has no headroom left, so a vLLM change that cuts the
    measured `U` by more than about 21% fails every official Qwen3.5-4B
    run closed (`capacity below 2x GPU KV (whole replay file used)`).
  - Review (`docs/rsi/reviews/cap-sizing.md`, ACCEPT) fixes: the plan's
    spec is the baseline gsm8k (else replay) dump, never preempt's shrunk
    pool (S1); docs say only non-official runs reach the committed
    fallback (S2) and that the estimate ratios are observations, not a
    bound (S3); a Checkpoint spec with `hash_block_size != block_size`,
    Eagle / speculative or prefill checkpoint blocks is recorded as
    `cap_plan.unmodeled` and the run is not official (N2).

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
- Audit note, not an S-entry (P5 it1): `LayerDecl.layout` (connector
  python, `adapters/__init__.py`) adds the Mamba state shapes/dtypes to the
  connector's namespace fingerprint, only when non-empty. pegaflow-semantics,
  `key(hash, group)` and `seal_plan` are untouched. V1 check: Qwen3-4B's
  namespace is unchanged (stub `cf6b5725`, native `cd8acc0c` at 6545352;
  `a25ea150` / `bdda8997` on P4's 55e4ff1 base with and without P5), pinned by
  `test_namespace_vector_unchanged_for_full_attention`.

## Open questions

- "agentx" perf workload: not available. User (2026-10-23): use the R2EGym
  replay for now. R2EGym (`~/kexi/datasets/r2egym-256.jsonl`, 32 x 12 official)
  is the G4/G3 replay workload until agentx is provided; results only claim
  this workload.
- For final acceptance (user's final pass; agent review counts as acceptance
  until then, GATES.md stays DRAFT): the `split_invariant=false` rule replaces
  cross-split `computed_exact` and exact outputs-vs-baseline-cold on
  split-variant models (MLA prefill, e.g. DeepSeek-V2-Lite). Instead: warm
  outputs == plain `split` exactly (aligned prompts reported only), cold ==
  baseline cold, `computed_exact` only against same-split copies with
  per-pass coverage >= 0.9; the flag is checked by `G2.split_control` in
  every run. Needs the user's confirmation.
- For final acceptance, next to the split rule: the untimed warmup request
  (dc70805, `passes._warmup`) is a measurement change in the lenient
  direction. It warms every arm equally, but the connector's one-time
  save-path cost (first save, gRPC setup) now lands outside
  `G4.replay_cold` and `G4.gsm8k_cold`. It is reported per arm vs baseline
  as `G4.warmup_seconds` (not gated). Needs the user's confirmation.
- For final acceptance: add rsi/gates tests to CI. PR CI runs
  `python/tests` only (`ci.yml`), so the 311 gate tests
  (`rsi/gates/test_oracle_capacity.py`, `test_probe_steps.py`) are a manual
  gate today: whoever changes `rsi/gates` runs them (GATES.md
  "Implementation"). They need the built `pegaflow.semantics`, and
  `test_probe_steps.py` needs vllm + torch (CPU), so the CI job needs the
  native build. Workflows are not edited by agents; needs the user's
  decision.
- Capacity-blind gates (raised 2026-10-23, user asked how the loop would
  find "manage SWA and full blocks separately" for DeepSeek-V4.1-Flash):
  every gate runs with an unlimited pool (`G3.no_eviction`), and write
  amplification / saved-but-never-loaded bytes are listed in GATES.md but not
  computed. So a policy that saves only the trailing window of SWA groups, or
  evicts them first, scores the same as saving everything. Proposal: G5
  capacity gate (fixed pool sized to a fraction of the trace's unique KV bytes,
  oracle-at-capacity as the bar), per-group write-amp / unused-bytes metrics in
  `oracle.py`, and an "optimization hypotheses" checklist in the skill.
  DSV4.1 under vLLM 0.32: kv-source layers `MLAAttentionSpec`
  (`tokens_per_state = compress_ratio`), SWA part `SlidingWindowMLASpec`
  (`DeepseekV4SWACache`, block 64), compressor `CircularBufferSpec`
  (one ring block per request, `prefix_cacheable=False`); the oracle has no
  Kind for `CircularBufferSpec` yet (fails as unknown). Not in the four-model
  scope and no weights downloaded.
- Server-side seal_plan (replace slot counting) and layout-fingerprint
  namespace in the server: not needed for TP1 onboarding; schedule after P6
  or when a TP>1 gate is added.
