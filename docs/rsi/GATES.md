# RSI gates (human-owned)

Agents may add checks or tighten thresholds here; they may not remove or relax
anything (only the human gate owner can, as on 2026-10-24 for hit-rate levels). Changes to this file need the review agent's sign-off and a log line in
`PROGRESS.md`. Status: DRAFT until the user confirms the metric set.

A model is "supported" only when every hard gate passes on that model and no
earlier model regresses.

## Hit rate: reported, judged by the agent (user, 2026-10-24)

User, 2026-10-24: "命中率那块，不要设置硬性门槛，只要agent自己判断命中率足够就行了，不要差太多"
("for hit rate, set no hard threshold; the agent judges whether the hit rate
is enough, it must not be far off"). The human gate owner's decision; it
replaces the hit-rate bars below (G2 "> 0.9 of oracle", G3 `E_ext >= 0.95`
and `oracle_hit_rate >= 0.5`, G5 `E_C >= 0.9`, and the "no `H` / `E_C`
regression" clauses).

- Reported, never gated: `H`, `H*`, `E_ext`, `H*_C`, `H_lru`, `H_freeze`,
  `E_C`, the oracle hit rate and per-group attribution. Every number is
  still computed. Each verdict carries a `reported` block (value, former
  bar as `reference`, `below`). `summary.hit_report` (also printed as the
  `[gate] hit rates (reported, not gated)` line) lists them per arm with
  `status: reported`.
- Still hard, unchanged: load errors, mismatched / unreferenced blocks,
  digest coverage (`MIN_JUDGED_*`, `MIN_COMPUTED_COVERAGE`,
  `MIN_PREWARMED`), `excess` within one block per request, `G3.no_eviction`
  on the unlimited runs, the cap pool and load proofs (`pool_ok`,
  `loads_ok`, `unmatched_requests == 0`), the G5 capacity floor and freeze
  separation (run validity, `not_official_because`), and every G0 / G2 / G4
  check.
- Hit-rate judgment: the onboarding agent writes a "hit-rate judgment" in
  the model's `ITERATIONS.md` for each accepted run. It compares the
  measured hit rate with `H*` / `H*_C` and the reference arms (`H_lru`,
  `old`, `native`), explains every gap (Kind limits, retention, eviction,
  per-group misses), and says whether it is close enough. The review agent
  checks that judgment (numbers against `summary.hit_report`, the
  explanation against the attribution) and can reject the run on it.

## G0 Layout (hard)

- Spec dump of the model under vLLM 0.32 (`kv_cache_config`, per-group spec
  class, block_size, page bytes, layer→group map, TP rank shard info) is
  committed to `docs/rsi/models/<model>/spec.json`.
- Adapter's declared layout (Kind, Shard, unit size, needed set) is checked
  against the dump: byte sizes add up, every layer belongs to exactly one group.

## G1 Semantics vectors (hard)

- `pegaflow-semantics` V1 test vectors pass (`cargo test -p pegaflow-semantics`).
- Every new Kind/Shard ships with vectors for resolve/saveable/needed/seal_plan.

## G2 Correctness (hard)

1. Per-block digest (V3): for every block loaded from PegaFlow, its bytes equal
   the bytes vLLM computes for the same prefix without PegaFlow. Zero mismatches.
2. Greedy output identity: the e2e prompt set
   (`python/tests/test_vllm_e2e_correctness.py`) gives the same tokens as
   baseline vLLM, cold and warm.
3. gsm8k accuracy (8-shot CoT, full test split 1319, greedy, temp 0):
   - runs: baseline (no PegaFlow), cold (PegaFlow saves), warm (GPU prefix cache
     reset via `/reset_prefix_cache` before the pass, so reuse must come from
     PegaFlow)
   - gate: `|warm - baseline| <= 1.0pp` and `|cold - baseline| <= 1.0pp`
   - warm external hit rate vs oracle: reported. A warm pass that loads little
     says little about loaded KV, and the hit-rate judgment has to say so.

## G3 Hit rate (reported, primary RSI objective; see "Hit rate" above)

Definitions, per run, prompt tokens only:

- `local_hit` = tokens served by vLLM's own GPU prefix cache
- `external_hit` = tokens loaded through the connector
- `computed` = prompt tokens actually prefilled
- external hit rate `H = external_hit / (prompt_tokens - local_hit)`
- oracle `H*` = same trace replayed offline through `semantics.resolve` with
  unlimited CPU capacity (upper bound given the model's Kinds; e.g. a
  Checkpoint group can only hit at checkpoint boundaries)
- hit efficiency `E = H / H*`

Also reported, per group: hit tokens, resolve fallback count and tokens lost to
fallback (P shrunk because some group lacked data), lease/prefetch failures,
loaded-but-unused bytes, saved-but-never-loaded bytes (write amplification).

Gate: none on the level since 2026-10-24. `E` (reference 0.95) and `H` against
the model's last accepted iteration go into the hit-rate judgment. Old
connector's `H` on the same trace is recorded as a black-box reference number
only.

## G4 Performance (hard gate, RSI objective)

Workload: multi-turn agent traces (user-provided "agentx" — TBD; fallback
`/ppio/data_sets/R2EGym-SFT-Trajectories` replayed as multi-turn chat with
growing context), plus gsm8k few-shot as a short-prefix workload.

Metrics: TTFT p50/p99, end-to-end throughput (req/s, output tok/s), load and
save bandwidth (GB/s), connector scheduler-side CPU time per step, GPU memory
overhead. Compared against baseline vLLM (no PegaFlow) and the old connector.

Gate (hard): warm replay TTFT p50 and throughput must beat baseline vLLM and
the `native` reference arm (vLLM's OffloadingConnector, same run; user,
2026-10-26); cold replay and gsm8k may be at most 2% worse than baseline (see G4 under
precise definitions); no >5% regression vs. last accepted iteration of the
same model.

## G5 Capacity hit rate (RSI objective; added 2026-10-23; `E_C` reported since 2026-10-24, correctness and validity hard; see G5 under precise definitions)

User: actual hit rate has to approach the theoretical one at a fixed capacity.
G3 runs with an unlimited pool, so it cannot see eviction policy, write
amplification or cross-group coherence. Example: a checkpoint group's state at
token P is useless once the full-attention blocks before P are evicted, and
full blocks past the last resident checkpoint are useless too. The server's
LRU (`pegaflow-core/src/cache.rs`) evicts each (hash, group) key on its own,
so such stranded bytes are possible today.

- Workload `cap`: its own multi-turn replay, larger than `replay` so the
  store matters: `--cap-convs auto x --cap-turns 12` conversations from the
  same r2egym file (official; `auto` = the fewest conversations, at least
  64, whose `C(0.5)` clears the capacity floor below, chosen before the
  baseline cap stage and recorded as `summary.cap_plan`; Qwen3-4B 64,
  DeepSeek-V2-Lite TP2 76). The baseline arm runs it first (`cap_cold`,
  `cap_warm`, no store) to fix `U` and the per-request reference local hits.
  Each store arm (under test, `old`, `native`, `lmcache`) then runs `cap_cold`,
  `cap_warm`, `cap_warm2..N` once per capacity fraction, each on a fresh vLLM
  and a fresh store of `C` bytes. Warm pass `j` starts the conversations
  rotated by `j/4` of their count, so the passes differ in arrival order.
- Capacity: `C = f x U`, `f` in `{0.25, 0.5, 0.75}` (official). Only
  `f = 0.5` is gated; the others are a reported curve. `U` = bytes of the
  distinct (group, key) blocks the unlimited oracle stores over the baseline
  `cap_cold` pass. A run is official only if `C(0.5) >= 2 x` vLLM's GPU KV
  bytes, so G5 measures the external store and not vLLM's own cache. Server
  fragmentation and headroom count against the connector/server.
- All external KV must be loaded from the PegaFlow server. A connector-side
  cache (holding KV in vLLM-process memory or anywhere outside the C-byte
  server) is a gate violation; G5 checks it through the server's load bytes.
- Oracle at capacity `H*_C`: the G3 store events, with a store of `C` bytes and
  clairvoyant eviction: an item (group, key, bytes) whose next use is furthest
  in the future goes first; next use = the next request (by first schedule)
  whose unlimited-oracle resolved prefix covers the item; an item never used
  again is not stored. Hits are `semantics.resolve` over present blocks, as in
  G3. For Kinds whose `needed` is not prefix-closed (Checkpoint, Window) a
  second, fallback-aware Belady also counts every earlier resume point as a
  use, and `H*_C` is the best of the candidate stores (Belady, that variant,
  LRU), so it is never below `H_lru` in `Σ ext*_C` (since 2026-10-26;
  full-only models unchanged). This is a strong reference, not a proven optimum; `E_C > 1` is
  allowed.
- References `H_lru` (group-oblivious LRU, everything stored) and `H_freeze`
  (store until full, never evict): reported only, to show how far naive
  policies are from the bar.
- Score over the timed cap warm passes, against the baseline's local hit for
  the same request (`ref`), so an arm cannot raise its own score by keeping
  less in vLLM's local cache, nor gain from keeping more:
  `ext = max(0, min(actual, oracle) - max(ref, own local))` (capped by the
  unlimited oracle), `ext*_C = max(0, oracle_C - ref)`, `E_C = Σext / Σext*_C`.
- Diagnostics, per group (not gated): resident bytes and stranded bytes at the
  end of each pass (resident blocks that lie on no resolvable prefix of any
  request in the trace), evictions, saved / loaded bytes. Implemented for the
  simulated stores; the server side reports eviction and save/load byte
  counters and its resident / used bytes at each pass end. TODO: actual
  stranded bytes need a server-side resident-key dump (server change, own PR).
- Gate at `f = 0.5`: no load errors; the cap passes' digests pass
  `G2.digest_self` (eviction must never produce a wrong load); server load
  bytes cover every external hit; `excess` bounded as in G3 against the
  unlimited oracle. `E_C` (reference 0.9) and its change vs the last
  accepted iteration are reported and go into the hit-rate judgment. A slice-size G5 (below the capacity floor) is a smoke test of
  flow and digests only; its `E_C` supports no conclusion.
- Reference arms: `old`, `native` (vLLM `OffloadingConnector` with exactly
  `C` CPU bytes) and `lmcache` (LMCache local CPU backend of exactly `C`
  bytes, 256-token chunks) record their G5 numbers and are never required;
  phase 2 will gate warm replay and `E_C` against `native`.
- lmcache store size. LMCache sizes its local CPU backend per worker
  process, not per instance. On an MLA model (DeepSeek-V2-Lite, TP2) only
  the first rank stores (`save only first rank: True`, one `Created
  backend: LocalCPUBackend` for two `Creating LMCacheEngine`), so a
  backend of `C` bytes is the whole store: the same one-copy convention as
  `native` and PegaFlow. On a non-MLA model at TP > 1 every rank stores
  its own shard, so the gate gives each storing rank `C / tp`
  (`lmcache_storing_ranks`, from the committed spec dump: all groups
  `MLAAttentionSpec` = 1 rank, else `tp`). The stage fails closed unless
  vLLM's log shows exactly that many backends, and an lmcache arm at
  TP > 1 without a committed spec dump does not start.
  `result.lmcache` records `storing_ranks`, `per_rank_bytes`, `backends`.
- lmcache build per model. gate.json `lmcache_dist` names the overlay
  dist (`<name>-<version>`) a model's lmcache numbers come from:
  Qwen3-4B `lmcache-0.5.7.dev0+pr4467.e05c2ea`
  (`~/kexi/overlays/lmcache-pr4467`), DeepSeek-V2-Lite-Chat
  `lmcache-0.5.6` (`~/kexi/overlays/lmcache`; PR #4467 crashes on MLA,
  below). A run whose `--lmcache-overlay` holds another dist, or a model
  without `lmcache_dist`, is not official, and the arm fails
  `G0.lmcache_build` (each stage's recorded `result.lmcache.dist`). The
  dist is what the overlay holds; `lmcache.__file__` from the worker is
  not logged (review N3).
- lmcache status per model (vLLM 0.32.0, shared venv torch 2.13.0+cu130).
  The arm is comparable on a model only when its loads are digest-verified
  there (`G2.digest_self.*` and `G2.digest_vs_baseline.gsm8k` pass with 0
  mismatches, warm accuracy within `ACC_TOL` of baseline):

  | model | lmcache | chunk | status | evidence |
  |---|---|---|---|---|
  | Qwen3-4B (FLASH_ATTN, `LBNHC`) | 0.5.6 (PyPI) | 256 | not comparable: wrong KV | below |
  | Qwen3-4B | source build of LMCache PR #4467 head `e05c2ea` | 256 | comparable, except the `cap075` curve point (wrong loads) | below |
  | Qwen3-4B | PR #4467 `e05c2ea` | 16 (secondary) | not comparable: claimed hits never loaded | below |
  | DeepSeek-V2-Lite-Chat (TP2, MLA) | PR #4467 `e05c2ea` | 256 | not comparable: crashes | below |
  | DeepSeek-V2-Lite-Chat | 0.5.6 | 256 | comparable with caveat (record only): G3 / G5 numbers usable; `G2.digest_vs_baseline.gsm8k` fails S7 on 14 warm blocks of 1 request (a reference gap, not wrong bytes; rule unchanged) | below |

  The gate pins an unmerged upstream PR (#4467) for the lmcache arm on
  Qwen3-4B: an open item for the user's final acceptance.

  `native` on Qwen3-4B (`2026-10-24-qwen3-4b-lmcache-ref-1afec5a`):
  `G2.digest_self.cap` fails at `C/U = 0.5` on the preempted-uncertain
  bound only. 0 mismatched; all 5140 unjudged loaded blocks are keys whose
  only undigested computations are preempted episodes (every request is
  digested), 1.04% of loaded blocks vs `MAX_PREEMPTED_UNCERTAIN` 1%
  (0.25: 0.24%, 0.75: 0.79%, both pass). That is the bound working as
  written, not wrong KV; its G5 numbers are reported with that flag.

- Qwen3-4B, PR #4467, chunk 256: `2026-10-24-qwen3-4b-lmcache-ref-1afec5a`
  (`--arms baseline,native,lmcache`, official flags, cap plan 64 convs as
  in the official runs). gsm8k (1319) and replay: every G2 digest check
  passes, external 64/64 (gsm8k) and 14971 (replay) matched, accuracy =
  baseline. `G2.digest_self.cap` passes at the gated `C/U = 0.5` and at
  0.25. At 0.75 it fails: 738 external blocks of 2 warm requests
  (`cap_warm` conv 43 turn 5, `cap_warm2` conv 0 turn 6), every one of 36
  layers, are wrong: 722 carry the bytes of computed blocks of other
  conversations at other positions (e.g. `cap_cold` conv 25 turn 2:
  LMCache served another prefix's chunks), 16 equal no computed copy. No load error was reported. That curve point is wrong KV
  and is not used; the gated point and the rest stand.
- Qwen3-4B, PR #4467, chunk 16 (secondary config, to separate chunk
  granularity from policy: `2026-10-24-qwen3-4b-lmcache-chunk16-91fbc41`,
  gsm8k 64 + cap): 4 of 64 warm requests got a hit from LMCache's lookup
  (`need to load` 48-112 tokens) but no `Retrieved` line ever followed. Their
  19 external blocks hold the poison: one digest, all 36 layers, in all 19.
  Warm accuracy 0.8125 vs 0.8594, 4 completions differ: not comparable.
  So the 256 config stays the only one.

- DeepSeek-V2-Lite-Chat, 2026-10-24, gsm8k 64 questions, TP2,
  TRITON_MLA, batch invariant, not split-invariant. vLLM 0.32 hands MLA
  layers to connectors as rank-4 `[NB, BS, 1, 576]`, which LMCache's vLLM
  detector reads as fused K/V (`NL_X_NB_BS_NH_CS`, NH 1, CS 576), not as
  MLA (its rank-4 branch only special-cases the DSA indexer plane).
  - PR #4467 build (`2026-10-24-dsv2-lite-lmcache-pr4467-slice`): the first
    store raises in `multi_layer_kv_transfer` (the PR's new layout check
    wants a split `[2, L, T, NH*HS]` object for a fused format; the MLA
    object is `[1, L, T, 576]`): every request HTTP 500, stage error
    recorded. Upstream bug in the PR for MLA on vLLM >= 0.29, not patched
    here.
  - 0.5.6 (`2026-10-24-dsv2-lite-lmcache-056-slice`): for MLA the packed
    addressing is the object's own layout, so bytes round-trip: external
    128/128 matched vs baseline and vs self, local 0 mismatched vs self,
    cold outputs = baseline. It still fails `G2.digest_vs_baseline.gsm8k`
    (504 local mismatched, warm computed 0/1010 exact, warm computed
    coverage 0.77 < 0.9), `G2.digest_self.gsm8k` (computed 0/2020 exact)
    and `G2.gsm8k` (warm 0.672 vs 0.688, one question). Cause: the first
    warm request after poison loads 4 x 256-token chunks (1024 tokens) and
    computes blocks 64-67 from token 1024, a split the cold pass never had;
    every one of the 504 mismatched local blocks equals that request's own
    warm-computed copy (split `(1024, ())`, layers 1-26 differ). Later
    requests reuse those blocks locally, so their computed KV differs from
    the cold copies with the same split key from layer 2 on. This looks like
    MLA split-variance carried through the context (the split key did not
    capture it), not a wrong load.
  - It was the gate's reference. The baseline `split` pass resumes each
    request at its last full prompt block (where a block-granular
    connector's hit ends: 63 of 64 warm requests of `native` resume at
    1120-1184). LMCache resumes at a 256-token chunk end, so every lmcache
    warm request computes from 1088 (vLLM's own local hit) or 1024. None of
    its warm computed blocks had a same-split copy except by accident: 0.77
    coverage, 258 of them on non-aligned prompts. No load was missing (every
    lookup hit was retrieved, 128/128 external matched). The fix, for every
    arm (gate `b05e507`): the split key also fingerprints the context bytes
    an episode starts from, and a baseline stage `resume-<label>` replays
    each arm's warm pass at that arm's own resume points (Implementation,
    "Split-variant references").
  - Rerun with it (`2026-10-24-dsv2-lite-lmcache-056-resume-b05e507`,
    0.5.6, 64 questions): `G2.resume_reference` 64/64 requests resumed
    where lmcache did, 31 waves. `G2.digest_vs_baseline.gsm8k` passes:
    local 17136/17136 and external 128/128 matched, computed 2708/2708
    exact, cold and warm coverage 1.0. `G2.gsm8k_outputs` passes (warm 0
    differ from the resume replay). `G2.digest_self.gsm8k` passes; its
    computed blocks are all unjudged there, since no two copies in one run
    share a context. `G2.gsm8k` fails: warm 0.672 vs 0.688, one question
    (-1.56 pp, `ACC_TOL` 1 pp). The baseline's own replay at lmcache's
    resume points scores 0.672 too, with the same completions. That is the
    cost of resuming at chunk ends on a split-variant model, not of the
    bytes.
  - Full size (`2026-10-24-dsv2-lite-lmcache-ref-b05e507`, gate `b05e507`,
    0.5.6, official flags, cap plan 76 convs = the official run's,
    U 13730052096 B): `G2.resume_reference` 1319/1319 resumed where lmcache
    did (1104 waves). `G2.gsm8k` passes: cold 0.6725 = baseline, warm
    0.6717 (-0.08 pp), and the replay also scores 0.6717. `G2.gsm8k_outputs`
    passes (0 non-aligned differ from the replay). `G2.digest_self` passes
    for gsm8k, replay and every cap fraction (0 mismatched).
    `G2.digest_vs_baseline.gsm8k` fails, with local 358496/358496 and
    external 152/152 matched and computed 55026/55026 exact: 14 warm
    computed blocks of one non-aligned request (prompt 1077, 1284 tokens)
    have no same-context copy (S7 needs 0). lmcache resumed it at 1280
    from local 1088 plus 12 loaded blocks, whose bytes equal baseline cold
    copies (computed from 1088 in a prefill over the whole 1284-token
    prompt). The replay's prewarm recomputed them from 1088 in a prefill
    that ends at 1280, which gives other bytes on this build, so the
    resumed request's context fingerprint differs. The reference is
    incomplete for that request; nothing judged is wrong (the review
    reproduced it). The fix would be a prewarm that reproduces the prefill
    the loaded blocks came from. That is not done, so the check keeps
    failing and the rule is unchanged; the model's lmcache status is
    "comparable with caveat": its G3 / G5 numbers are usable, since every
    digest check behind them passes with 0 mismatched, and the row says
    which check fails.
  - The official DSV2 records (`2026-10-24-dsv2-lite-p4-it6-off-a1e7fd2`,
    `2026-10-25-dsv2-lite-p4-it7-off-a59a254`, gate `ce645d75`) predate the
    resume stage. Their digest checks and coverage re-score identically
    under the merged gate, but every arm (`old`, `native` and the required
    connector_v2) now fails `G2.resume_reference` ("no baseline replay of
    this arm's resume points"): fail-closed and correct for a record
    without the stage. DSV2's acceptance on those records does not stand
    under the merged gate until an official TP2 rerun with the resume
    stage passes (review M2).

- lmcache 0.5.6 is wrong on vLLM 0.32's fused K/V page, 2026-10-23,
  re-confirmed 2026-10-24 (`2026-10-24-qwen3-4b-lmcache-repro056`, vLLM's
  builtin `LMCacheConnectorV1`; `2026-10-24-qwen3-4b-lmcache-056-dynamic`,
  LMCache's dynamic connector: same result). vLLM's page is
  `[NB, BS, NH, 2*HS]` (K | V per head); LMCache detects it as
  `NL_X_NB_BS_NH_CS`. gsm8k (64 questions, batch invariant) cold accuracy
  0.859 (= baseline), warm 0; external digests 64/64 mismatched vs baseline
  and vs self, every one of 36 layers. Earlier evidence (2026-10-23): the
  wrong blocks are not the poison pattern and equal no computed copy of
  any key; per-layer L1 differs from the right block by 6-46%. Cause
  (LMCache issue #4463, fix PR #4467, open): the in-process
  `VLLMPagedMemGPUConnectorV2` keeps split `KV_2LTD` objects
  (`[2, L, T, NH*HS]`) while the transfer kernel addresses the fused
  buffer as packed (`[1, L, T, NH*CS]`): the slot stride is half the row,
  slots alias and the V half never round-trips. `use_gpu_connector_v3`
  crashes on the first store (shape mismatch, same issue). vLLM's in-tree
  adapter (`use_native: true`) does not start with 0.5.6 (`ImportError:
  CudaIPCWrapper`), and is missing `get_kv_events` (vLLM issue #62216).
- lmcache build used by the arm: `~/kexi/overlays/lmcache-pr4467`, a source
  build of https://github.com/LMCache/LMCache/pull/4467 head
  `e05c2ea9781b4021081f34dfb982189ec0fabc15` (base: dev `18651e8f`, 2026-10-21),
  (an open, unmerged PR: pinned by commit; final acceptance is the user's),
  unmodified, wheel version `0.5.7.dev0+pr4467.e05c2ea`, CUDA 13.0 nvcc,
  `TORCH_CUDA_ARCH_LIST=8.9`, `--no-build-isolation` against the shared
  venv's torch (build tools in `~/kexi/tmp/lmcache-buildtools`, never in the
  venv), plus the pure deps the venv lacks (aiofile, aiofiles, caio, redis,
  sortedcontainers, as in the 0.5.6 overlay). No local patch: PR #4467 is
  upstream's fix for #4463 and is used as published (validated upstream on
  vLLM 0.32.0, A100, by the issue reporter). Rebuild recipe: below
  ("lmcache overlay").
- Known server limit (slice 2026-10-23, `old` and connector_v2 on the same
  build): reclaim evicts `RECLAIM_BATCH_SIZE = 512` blocks per round
  (`pegaflow-core/src/storage/mod.rs`), so on a small pool each eviction
  burst drops about half of `C` (Qwen3-4B: 512 x 2.25 MiB = 1.13 GiB of a
  2.24 GiB pool) and only ~0.5 C stays resident. LFU admission is off
  (all saves inserted). The server LRU is also not prefix-aware: a chain's
  head blocks, served from vLLM's local cache and never re-read from the
  server, age out first and strand the chain's tail for any lookup that
  starts at block 0 (`old` slice: 0 server hits in every cap pass, pool =
  C, no alloc failure or admission reject). Fixing either is a server
  change (own PR).
- Pre-fix reference (`2026-10-23-qwen3-4b-g5-official`, gate rev 369b253,
  server from `feat/semantics-rsi`; before the review fixes: cap = replay
  trace 32 x 12, `C = 0.5 x U` from the baseline replay, per-arm local, no
  rotation, so not comparable to later runs). U 25.81 GiB, C 12.91 GiB
  (1.24x GPU KV, below the current floor). `warm_timed`, prompt 2.55 M tokens:

  | arm | E_C | hit | H*_C | H_lru | evictions / pass | load bytes / warm pass |
  |---|---|---|---|---|---|---|
  | connector_v2 | 0.216 | 0.440 | 0.785 | 0.517 | 42.5k-45.6k | 16.6-19.2 GB |
  | old | 0.124 | 0.459 | 0.814 | 0.523 | 35.3k-40.4k | 8.8-9.6 GB |
  | native | 0.558 | 0.649 | 0.843 | 0.561 | n/a | n/a |

  `H*` (unlimited) 0.998 for all. All three are far below 0.9 and below
  `H_lru`; native (vLLM's CPU offload, same C) beats both PegaFlow arms.
  Warm TTFT p50: native 1.38-1.51 s, connector_v2 2.01-2.08 s, old
  2.09-2.14 s, baseline (replay, no store) 2.54 s.

## Run record

Every gate run writes `~/kexi/rsi-runs/<date>-<model>-<iter>/` with
`summary.json` (all numbers above), raw logs, the commit hash, and the vLLM /
PegaFlow command lines. `PROGRESS.md` links the summary.

## Implementation (`rsi/gates/`, P0)

Tooling is gate-owned like this file: onboarding agents run it and read its
output, they do not edit it.

| File | Role |
|---|---|
| `rsi_probe.py` | `RsiProbe` connector, never loads/saves. Stacked after the connector under test in `MultiConnector`; dumps G0 spec, per-request trace, V3 digests |
| `digest_compare.py` | G2.1 byte comparison of digested blocks |
| `oracle.py` | G3 `H*` replay through `semantics.resolve`; G5 `U`, `H*_C` (clairvoyant) and `H_lru` capacity simulation |
| `test_oracle_capacity.py` | G5 oracle and gate contract on synthetic traces, no GPU (command below) |
| `test_probe_steps.py` | probe and self-test connector step logic on synthetic scheduler outputs (CPU; imports vllm, torch) |
| `workloads.py` | gsm8k, multi-turn replay, Prometheus scrape |
| `run_gate.py` | orchestrator: arms x workloads, cold/warm passes, verdicts |
| `provenance.py` | tree verification, under-test snapshot, static scan + allowlist |
| `isolate.py` | runs the arm under test with the runs root hidden (mount namespace) |
| `selftest/connector.py` | gate self-test connector: `honest` / `liar` / `bitflip` / `offbyone` / `sync` |

Tests. They need `pytest`, `requests` and a built `pegaflow.semantics` (the
native module), so they run in a venv where `python/` is built. Everything
else they import is stdlib. From a checkout:

```bash
cd python && uv sync --extra test                 # pytest, requests
uv pip install 'maturin>=1.0,<2.0'
uv run maturin develop -r                         # CUDA 13: add --no-default-features --features cuda-13
cd .. && PYTHONPATH=. python/.venv/bin/python -m pytest -q -p no:cacheprovider rsi/gates
```

The full suite is the gate's test run, signal and teardown tests (real
subprocesses) included: 363 tests in about 52 s on the 4090 host. Nothing is
deselected. PR CI does not run them (it runs `python/tests` only); they are
a manual gate for every change under `rsi/gates` until CI does (PROGRESS,
user acceptance item). The isolation tests (`isolate.py`) need Linux and root and skip
elsewhere. The shared venv on the GPU hosts already has the built
`pegaflow`, so there only the last line is needed.

Trees on the host carry no bytecode by construction (M5 rejects any `.pyc`
in the tree under test): extract a rev with
`git archive <rev> | tar -x -C <dir>` plus `echo <rev> > <dir>/.rsi-rev`,
or sync a checkout with `tar --exclude=__pycache__ --exclude='*.pyc'`. Then
build the native module in place (`maturin develop` / copy
`pegaflow<EXT_SUFFIX>` into `python/pegaflow/`).

Usage (on the 4090 host 10.1.0.44, from the feature dir, shared venv):

```bash
CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
LD_LIBRARY_PATH=~/kexi/cuda13-lib \
~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate \
  --model ~/kexi/models/<model> --tag <model>-it<N> \
  --arms plain,baseline,old,native,lmcache,<module>:<Class> --workloads gsm8k,replay,cap --gpu 0 \
  --lmcache-overlay <the overlay of gate.json lmcache_dist: ~/kexi/overlays/lmcache-pr4467 on Qwen3-4B>
# G5 smoke on a slice (below the capacity floor: not official, no E_C conclusion):
#   ... --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --cap-convs 8 --cap-turns 6
```

Official runs include `native` and `lmcache` on every model where the arm
is comparable (G5 "lmcache status per model"); where it is not, drop
`lmcache` and cite that table. A reference-arm rerun that is not part of an
official run (e.g. `--arms plain,baseline,native,lmcache`) lines up with
an official run only with the same model, flags and cap plan (`cap_convs`
`auto` resolves to the same count from the same baseline trace; check
`summary.args.cap_plan`) and the same baseline `U`.

lmcache overlay (a source build of one pinned LMCache commit; on .44):

```bash
git -C ~/kexi/tmp/lmcache-src fetch origin pull/4467/head && git -C ~/kexi/tmp/lmcache-src checkout e05c2ea9781b
# build tools (pip, grpcio-tools 1.78, setuptools_scm, wheel) in a --target dir, never the venv;
# the venv's python only provides torch / ninja (pip wheel installs nothing)
cd ~/kexi/tmp/lmcache-src && CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
  LD_LIBRARY_PATH=~/kexi/cuda13-lib PYTHONPATH=~/kexi/tmp/lmcache-buildtools TORCH_CUDA_ARCH_LIST=8.9 \
  LMCACHE_CUDA_MAJOR=13 SETUPTOOLS_SCM_PRETEND_VERSION=0.5.7.dev0+pr4467.e05c2ea \
  ~/kexi/pegaflow/python/.venv/bin/python -m pip wheel . --no-build-isolation --no-deps -w dist
<any pip> install --no-deps --target ~/kexi/overlays/lmcache-pr4467 dist/lmcache-*.whl
# + aiofile aiofiles caio redis sortedcontainers (copied from the 0.5.6 overlay)
```

Self-test at onboarding (gsm8k, 128 questions per self-test arm, no PegaFlow;
add `preempt` to judge loads after preemption; on models with a Mamba state
group the honest arm fails without it, and add `selftest:sync`):

```bash
... run_gate --model ... --tag <model>-selftest \
  --arms plain,baseline,selftest:honest,selftest:liar,selftest:bitflip,selftest:offbyone \
  --workloads gsm8k,preempt --gsm8k-limit 128
```

Exit code 0 only for an **official** run in which every required verdict passes.
A run is official only with the constants of `run_gate.OFFICIAL` (full gsm8k, 8
shots, 32 x 12 replay, 3 replay passes, pinned scheduler flags), the arms
`plain`, `baseline`, `native` and one connector under test, all three workloads
(`gsm8k`, `replay`, `cap`; cap `auto` x 12 at fractions 0.25,0.5,0.75), the G5
capacity floor (`C(0.5) >= 2 x` GPU KV bytes), no
`--num-gpu-blocks`/`--vllm-arg` override, and committed
`docs/rsi/models/<model>/{gate.json,spec.json}`. On a model with a Mamba
state group (committed spec) `preempt` is a required workload too: gsm8k
loads a single prefix there (below), and `preempt` is the only official
stage that judges every resumed episode's state load. Loads of many state
keys are judged by cap (exhaustive) and replay (sampled); neither has run on
Qwen3.5-4B yet, not even baseline: that coverage comes with the P5 it2
official run. A run is also not official
when it is contaminated: any arm's G4-timed replay pass (`cold`, `warm2..`)
has p99 TTFT `> 10x` the in-run baseline's p99 for the same pass, or the
baseline's p99 is `> 10x` the median of the arms' p99 for that pass (a
baseline stall inflates every arm's margin) (`summary.contamination`; the pass is kept and still judged, never
dropped). A run is not official either when the `native` arm's replay
stage is missing, crashed, or lacks one of the baseline's timed warm passes
(`warm2..N`, with `ttft_p50` and `req_per_s`): G4 vs native has no reference
then (`run_gate.g4_native_problems`). Native's own G0 / G2 verdicts never
make a run non-official (it fails `G0.spec` by construction).
`summary.json` lists `not_official_because`. A required verdict that cannot be evaluated is a fail.
Required verdicts: `baseline` and the connector under test, including
`G2.digest_self.cap` and `G5.replay_warm`. Not required (`required: false`,
every verdict recorded): `old` (black-box reference), `native` and `lmcache`. A crash of
a reference arm's stage is recorded under `errors` in its verdicts and the
run continues; a crash of baseline or the connector under test aborts it.
Every stage's vLLM / pegaflow-server runs in its own process group; teardown
SIGTERMs the group, waits up to 120 s, SIGKILLs it, waits up to 300 s, and a
group that survives that aborts the run (`TeardownError`, also for reference
arms: the next stage would start on a live GPU). For the arm under test the
same signals and waits also cover every process in its private mount
namespace (S11), so a process that `setsid()`s out of the group is stopped
too. `isolate.py` blocks right after its `unshare` until the gate has opened
`/proc/<pid>/ns/mnt` and is holding it, so the namespace is recorded before any arm
code runs. Processes are matched by namespace inode only and signaled through
pidfds; the arm cannot leave the namespace (no `CAP_SYS_ADMIN`).
Stage ports (vLLM, pegaflow-server gRPC / HTTP) are leased from the
host-reserved range 20000-29999 (`ip_local_reserved_ports`: never an
ephemeral port) under per-port `flock` files in `~/kexi/rsi-ports`, each
bind-checked at lease time. vLLM also gets a leased block of 8 contiguous
ports as `VLLM_PORT`: its own TCP ports (`get_open_port`) scan up from
there. On CUDA, TP>1 rendezvous goes through a file store and the engine
core's ZMQ sockets are `ipc://` paths, so the block is normally unused.
A stage process that fails before it is healthy, with EADDRINUSE on any bind
in its log (a leased port, or one vLLM / torch / ZMQ / the server picked
itself), fails the stage with `PortRace`. The error names each port parsed
from the failing lines, plus any leased port another process holds once
the stage process is dead. If no port is found, it names "unknown". It also
records whether the gate leased each port, the lease files and the failing
line. A port race is never retried. A stage's leases are released when the
stage ends, on every path: success, exception, stop signal, teardown. Every
failed stage writes `stage-error.json` (arm, workload, exception type,
error, `port_leases`) in its dir. `summary.port_leases` lists every lease of
the run (log, ports, block, lock files, leased/released times).
CUDA toolchain preflight: vLLM / FlashInfer JIT-build kernels with the
`nvcc` of `CUDA_HOME` or `PATH`, so a run is official only when every nvcc
found there has the venv torch's CUDA major. A missing nvcc, or a
`CUDA_HOME` without one, makes the run non-official.
`summary.cuda_env` records torch's CUDA version, `CUDA_HOME`,
`LD_LIBRARY_PATH`, `PATH`, and the path and version of each nvcc.

Stopping a run: `kill -TERM -<run_gate pgid>` (or `-INT`, `-HUP`; launched
with `setsid`, the gate's pgid is its pid). Each aborts the whole run: the
gate turns the signal into `SystemExit` (its own SIGINT handler replaces an
inherited SIG_IGN and asyncio's cancel-only handler), records it so no later
pass or stage starts even if a pass converts the exception, stops the
current stage's process groups as above (a signal during teardown or a
launch waits for the group; one landing in the few bytecodes before
`stop` owns a group leaves it in a live-group registry that an `atexit`
hook stops) and exits 128 + signal, with no `summary.json`. Never SIGKILL the
gate: its children, in their own sessions, would keep the GPU.

Exit 0 does **not** cover G1 (run `cargo test -p pegaflow-semantics`) or the
"no regression vs last accepted iteration" clauses of G3/G4: compare against
the last accepted run record by hand until `--accepted` exists.

### Precise definitions (these override the prose above where narrower)

- Arms: `plain` (no kv-transfer-config, gsm8k only), `baseline` (probe only),
  `old` (black-box reference), `native` (reference arm, record only: vLLM's
  `OffloadingConnector` as `--kv-offloading-size` sets it up,
  `kv_connector_extra_config = {"cpu_bytes_to_use": B}`, no PegaFlow server,
  `B` = `--pool-size` bytes for gsm8k/replay and exactly `C` for `cap`;
  phase 2 will gate warm replay and `E_C` against it; OffloadingConnector
  switches `kv_cache_layout` to `LBHNC`, so its `G0.spec` and
  `G2.digest_vs_baseline.gsm8k` fail by construction: layout, not data;
  `G2.digest_self.*` and `G2.gsm8k_outputs` are its correctness evidence),
  `lmcache` (reference arm, record only: LMCache's in-process vLLM
  connector, loaded by module path as LMCache documents it
  (`lmcache.integration.vllm.lmcache_connector_v1:LMCacheConnectorV1Dynamic`;
  vLLM's builtin `LMCacheConnectorV1` imports a module LMCache dev moved),
  from the overlay `--lmcache-overlay` (on `PYTHONPATH` of that arm's vLLM
  only: its pins would downgrade numpy / protobuf / otel in the shared venv;
  official runs pass the overlay of the model's gate.json `lmcache_dist`
  (G5 "lmcache build per model": Qwen3-4B `~/kexi/overlays/lmcache-pr4467`,
  DeepSeek-V2-Lite the default `~/kexi/overlays/lmcache`, 0.5.6, which is
  wrong KV on Qwen3-4B); `result.lmcache.dist` records the overlay's
  lmcache version and `G0.lmcache_build` checks it),
  no PegaFlow server, configured by env only (inherited `LMCACHE_*` dropped):
  `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE` = `B / r / 2^30`
  GiB per storing rank (`r` = 1 on MLA or TP1, else `tp`; exact: LMCache
  allocates `int(GiB x 2^30)` bytes = `B / r`, so the store totals the
  same `B` as `native`; G5 "lmcache store size"), `LMCACHE_MAX_LOCAL_DISK_SIZE=0`, no disk / remote URL, chunk
  size LMCache's default 256 tokens, telemetry off (`LMCACHE_TRACK_USAGE=false`,
  `DO_NOT_TRACK=1`); `result.lmcache` records the env and LMCache's logged
  configuration. LMCache stores and looks up 256-token chunks (a trailing
  partial chunk is not saved by default), so its hit lengths are multiples of
  256 where the oracle counts 16-token blocks: its `E` / `E_C` include that
  granularity loss and are not a pure eviction-policy comparison),
  `selftest:<mode>` (gsm8k and preempt only), `<module>:<Class>`. `old`, `native`,
  `lmcache` and the connector under test run as `MultiConnector[arm, RsiProbe]`.
  `--under-test-path DIR` (another tree's `python/`) stages a snapshot,
  `<run>/under-test-overlay/pegaflow`, and puts only that dir first on the
  `PYTHONPATH` of the connector-under-test arm (`provenance.stage`). The
  snapshot holds the verified rev's tracked blobs of `python/pegaflow` plus
  the one expected native module `pegaflow<EXT_SUFFIX>`; it is verified again
  after the copy (blob ids, sha256 equal to the source's). The source tree
  must hold nothing else under `python/pegaflow`: a symlink, an untracked or
  ignored file, another `.so`, or any `.pyc` makes it unverified. Editing the
  tree mid-run changes nothing the arm imports, and the tree cannot replace
  the probe, its worker extension, vLLM or `sitecustomize`. Every vLLM and
  pegaflow-server process runs with
  `PYTHONPYCACHEPREFIX=<run>/.pycache-stages/<arm>/<workload>`. That cache is
  made fresh before the stage, is outside every arm dir, and no later stage
  reads it (M10), so a stage cannot plant bytecode for a later one. The
  gate re-execs itself with
  `PYTHONPYCACHEPREFIX=<run>/.pycache`. So no process reads an in-tree
  `__pycache__`, and the gate and venv trees may carry `__pycache__/*.pyc`
  (any other untracked file still unverifies them). Gate code (`rsi/`)
  always comes from the gate tree. `pegaflow` does not: for the gate process
  (the oracle's `pegaflow.semantics`), the reference arms (`old`) and
  pegaflow-server's embedded Python it resolves through the shared venv's
  `pegaflow_llm.pth` (on the 4090 host:
  `/root/kexi/pegaflow-semantics-rsi/python`), not the gate tree's
  `python/`. `summary.provenance` records, before any stage: the gate
  tree's, the venv `pegaflow` tree's and the under-test tree's rev with
  `verified` (a clean git toplevel, or `.rsi-rev` whose `git ls-tree -r
  <rev> python/pegaflow` blob ids equal the files, fetched blob-less from
  `provenance.REMOTE`), each tree's native-extension sha256, and the
  pegaflow-server binary's sha256 with its tree's `.rsi-rev` (a claim: a
  binary is not tied to a rev). A run with `--under-test-path` is official
  only if the under-test snapshot is verified, and any run only if the gate
  tree and the venv `pegaflow` tree are verified too (an unverified judge or
  oracle is not official; `run_gate.provenance_problems`).
  Re-verification (M10): right after that, the gate fingerprints by blob
  id:
  - the gate tree (`rsi/`, `python/pegaflow`, the model's
    `docs/rsi/models/<model>`);
  - the venv `pegaflow` tree;
  - the snapshot the arm under test imports;
  - the pegaflow-server binary.
  It compares them again before every stage and after the last one. Any
  change (a file edited, added, removed or turned into a symlink) makes the
  run non-official, with the role, the path and the stage it was found
  before named in `not_official_because`. Every check is in
  `summary.provenance_watch`. The run goes on. Not covered: other venv or
  system files, and a change made and undone within one stage.
  G0.provenance (every arm, and
  baseline): every spec dump (scheduler and each worker) records the module
  files of `rsi.gates.rsi_probe`, the worker extension, vLLM, torch,
  `pegaflow` (+ its loaded native module, `site/usercustomize` if loaded)
  and the interpreter prefix; probe and extension must resolve under the gate
  tree, vLLM / torch / prefix to the shared venv, and `pegaflow` to the arm's
  expected dir (the snapshot for the arm under test, the venv's
  otherwise). The probe also records `__file__` of the arm's connector
  module (`connector:<module>`, named by the gate in the probe's
  `connector_modules`). The root it must lie under is chosen by arm kind,
  never by module name. Only the declared self-test arms (`selftest:*`)
  resolve under the gate tree. The arm under test, whatever its module
  name, must resolve under its snapshot, and `old` under the venv
  `pegaflow`. The arm under test's module must also be in the `pegaflow`
  package, the only one the snapshot holds. An `rsi.*` module for the arm
  under test makes the run non-official. An official run needs
  `--under-test-path`; without it the arm under test imports the venv's
  tree, which nothing snapshots.
  The connector under test runs as `MultiConnector[under_test, RsiProbe]` with
  `kv_load_failure_policy=recompute`, reads its server port from
  `PEGAFLOW_PORT`, and gets an empty `kv_connector_extra_config`. The probe
  keeps `prefer_cross_layer_blocks=False`, so cross-layer KV layouts are out of
  scope for now.
- Boundary (M7). The connector under test runs in the same processes as
  `RsiProbe` / `RsiWorkerExt`, and `MultiConnector` imports its module in
  vLLM's API server too (`requires_piecewise_for_cudagraph`). So it *can*
  patch the instrument at runtime, and nothing in-process can detect that:
  a probe self-check would run in the patched process. The gate bounds this;
  it does not prevent it:
  - Isolation (a). The arm under test's vLLM and pegaflow-server are exec'd
    through `rsi/gates/isolate.py`, which uses a private mount namespace. The
    runs root is under an empty tmpfs. Only the current stage's dir
    `<run>/<arm>/<workload>` (rw) and the snapshot (ro) are bound back, along
    with the arm's bytecode cache. Its earlier stages and the gate dir stay
    hidden. Block devices and `/dev/{mem,kmem,port}`
    are masked. `CAP_SYS_ADMIN` / `CAP_SYS_PTRACE` / ... leave the bounding
    set, so the arm cannot unmount the cover or reach the host tree through
    `/proc/<gate pid>/root`. The baseline's outputs, digests and requests,
    and every other arm's, are not visible to it. The pid, IPC and network
    namespaces are the host's (CUDA IPC, loopback ports): the exec is in
    place, so `stop` signals and waits for the same process group as before,
    plus everything in the arm's mount namespace (S11).
    The command prefix is in `summary.isolation`. A run whose arm under test
    could not be isolated (not Linux, not root) is not official.
    Residual: this is deterrence, not a boundary. What holds (tested in
    `test_isolated_arm_cannot_read_other_arms`): with those capabilities out
    of the bounding set, opening `/proc/<pid>/ns/*` or `/proc/<pid>/root` of
    a full-capability process fails the kernel's ptrace access check
    (EACCES), so `setns` / `nsenter` into the host mount namespace and reads
    through `/proc/<gate pid>/root` fail, and the tmpfs cannot be unmounted.
    What does not hold: the arm stays uid 0 with `CAP_DAC_OVERRIDE` in the
    host pid / IPC / network namespaces, so it can write any host file outside
    the runs root. That includes code that later stages or other root
    processes run with full capabilities (gate tree, venv, system files).
    The controls for that:
    - re-verification before every stage and after the last one (M10). It
      covers the gate tree, the venv `pegaflow` tree, the snapshot and the
      server binary;
    - a fresh per-stage bytecode cache that no later stage reads;
    - the static scan and the review.
    Not covered: other venv or system files, and a change made and undone
    within one stage (`summary.isolation.residual` says the same).
  - Static scan (b). `provenance.scan` scans the snapshot, every text line
    and the strings of the native module, for `provenance.SCAN_PATTERNS`:
    names of the instrument (`rsi`, `rsi_probe`, `RsiProbe`, `RsiWorkerExt`,
    `rsi_poison`) and of run artifacts (`rsi-runs`, `probe_dir`, `digest-`,
    `-outputs.json`, `-tokens.json`, `requests.jsonl`), plus runtime
    patching (`sys.modules` writes, `gc.get_objects`/referrers, trace /
    audit hooks, `sys.meta_path` / `path_hooks`, `importlib.reload`). `rsi`
    matches the gate package as code (`import rsi`, `from rsi`, a `"rsi.`
    string, `rsi.gates` / `rsi/gates`), not a path containing `rsi`. A hit
    makes the run non-official unless it is on `provenance.SCAN_ALLOWLIST`
    (path glob + pattern + reason; adding an entry is a gate change, with
    review sign-off). Allowlisted hits are recorded with their reason in
    `summary.scan.allowlisted`, the others in `summary.scan.flagged`.
    Residual: the scan matches literal text, so it is deterrence too. It
    does not chase every route by pattern. These known gaps are covered by
    review of the snapshot, not by the scan (S12):
    - frame walking (`inspect.currentframe` / `inspect.stack`,
      `sys._getframe`, `sys._current_frames`, a traceback's frames) up to
      `MultiConnector._connectors` and the probe object;
    - class discovery through `__subclasses__()` (of `KVConnectorBase_V1`,
      `object`, ...), then `setattr` on the probe class;
    - `importlib.import_module` / `__import__` with a computed name, and any
      built string (`"r" "si"`), which the `rsi` pattern cannot see;
    - `ctypes` (raw memory, or calls into the interpreter's C API), and
      the native module, whose strings are scanned but whose code is not.
  - No plugins (d). Every vLLM process runs with `VLLM_PLUGINS=""`: no
    `vllm.general_plugins` entry point runs (the venv's `pegaflow` plugin
    would run a tree's code in every vLLM process, the API server included).
    Arms name their connector by module path. The connector module is still
    imported in the API server, by `MultiConnector`'s class lookup.
    Exception: the `old` reference arm runs with `VLLM_PLUGINS=pegaflow` (that
    entry point only). vLLM 0.32's API server rebuilds each MultiConnector
    child's stats by registered name
    (`MultiConnector.build_kv_connector_stats` ->
    `KVConnectorFactory.get_connector_class_by_name`). `PegaKVConnector`
    reports stats and only that plugin registers its name, so without it
    every old-arm request fails with HTTP 500. The old arm is a reference arm
    whose connector is imported from the venv's `pegaflow`, and the plugin
    module comes from that same package, never from the tree under test. It
    only registers connector names (lazy imports). The other arms' children
    report no stats (probe, selftest, `connector_v2` today), or are vLLM
    builtins registered under their class names (native
    `OffloadingConnector`), or report none (lmcache's dynamic connector). If the connector
    under test starts reporting stats, it hits the same lookup and fails
    loudly (HTTP 500 on every request), never silently. The fix is in that
    connector: its own module registers its class under its class name when
    it is imported (`KVConnectorFactory.register_connector(name, __name__,
    name)`). vLLM imports that module in the API server by module path. The
    arm under test never gets the plugin, which would run the venv's
    `pegaflow` code in the arm's processes. `summary.vllm_plugins` records
    each arm's value.
  - Independent signal. Completions are read by the gate *client-side*, from
    the `/v1/completions` HTTP response (`workloads.run_gsm8k`), and judged
    in the gate process. What the gate judges from and writes itself (pass
    outputs and tokens, `result.json`, `stage-error.json`) is in
    `<run>/.gate/<arm>/<workload>` (M9). No arm process ever gets that dir.
    Only what the arm's processes must write (the probe's records, logs) is
    in its stage dir, and it is untrusted. A symlink or non-regular file left
    there fails the stage. A later stage of the arm cannot see an earlier
    one. A connector that fakes digests while loading
    garbage still changes completions (exact vs baseline cold, or vs `split`
    on non-aligned prompts on split-variant models). A connector patching
    the API server could in principle forge completions too. That is what
    the scan and the review checklist below are for.
  - Review checklist (skill step 6): the connector does not touch the probe,
    the worker extension, `sys.modules`, the poison RPC or any run artifact.
    No instrument or run-artifact access, by code or by string.
- Per-model settings: `docs/rsi/models/<model>/gate.json` (gate-owned):
  `batch_invariant` (established by `G2.batch_control`, see below),
  `split_invariant` (default true, established by
  `G2.split_control`, see below), `invariant_attention_backend`,
  `attention_backend`, `vllm_args` (e.g. `mamba_cache_mode`,
  `prefix_cache_retention_interval`), applied to every arm's vLLM and
  checked by `G0.vllm_args` (baseline verdicts: each stage's recorded
  `vllm_cmd` in summary.json carries every gate.json flag with its value as
  the last occurrence). The spec dump records `cache.mamba_cache_mode` and
  `prefix_cache_retention_interval` (cache config and `KVCacheConfig`), so
  `G0.spec` also pins them per probed arm against the committed dump. Pinned for every arm:
  `--max-num-batched-tokens 2048 --max-num-seqs 256 --enable-prefix-caching
  --seed 42 --generation-config vllm`, `PYTHONHASHSEED=0`.
- Passes: each arm x workload gets fresh vLLM (+ fresh PegaFlow for `old` and
  under-test arms). gsm8k and replay first send one untimed warmup request
  (`workloads.warmup_prompt()`: 512 fixed pseudo-random token ids, 16 output
  tokens, `rsi_pass=_warmup`, never scored) and `POST /reset_prefix_cache`;
  the stage fails unless no workload request shares a prefix block with it
  (`oracle.warmup_problems` on the probe trace). Its seconds are reported
  per arm vs baseline as `G4.warmup_seconds` (not gated; the one-time
  save-path cost it moves out of the cold passes, PROGRESS "Open
  questions"). Cap stages have no warmup
  (untimed; its blocks would take capped store space). `cold`, then per warm pass: sleep `--settle` s,
  `POST /reset_prefix_cache`, then `POST /collective_rpc {"method": "rsi_poison"}`
  (`RsiWorkerExt`, fills every byte of every KV allocation with 0x3F, a finite
  bf16/fp16 value), then the pass. Warm reuse can only come from the
  connector; stale cold-pass bytes cannot rescue a connector that loads nothing.
  gsm8k: `cold`, `warm` (plain arm also `split_ref`, `split`: the split
  control, and `alone`: the batch control, below; with `split_invariant=false` the baseline arm also runs
  `split`, the same-split references for warm computed KV; its pre-warm
  requests are tagged `<pass>_prewarm` and never scored). Replay: `cold`, `warm` (digested), `warm2..warmN`
  (timed, N = `--replay-repeats`). Each request carries
  `vllm_xargs.rsi_pass = <pass>`; the oracle scores by that tag.
- Probe trace per request: `episodes = [step, local_hit, start]` per schedule
  episode (a resume after preemption opens a new one), `progress = [step,
  computed_after]` whenever a step crosses a key-token boundary. `local_hit` is
  the value passed to `get_num_new_matched_tokens` at the last attempt before
  the episode was scheduled; `first_computed` = `start` of the first episode.
  Load failures (`invalid_block_ids`, `failed_recving`) go to
  `load_errors.jsonl`. `preempt_digested` lists the episodes digested at
  their preemption (below).
- Digest (V3): requests tagged `vllm_xargs.rsi_digest = full` (gsm8k: all;
  replay `warm`/`cold`: every 4th conversation, all of its turns) are digested
  at finish: every fully computed block of every transferable group, per layer,
  on the raw allocation with `KVCacheTensor` geometry, unpadded page bytes (two
  64-bit position-weighted sums + float L1). `register_kv_caches` checks that
  geometry against every layer view and raises on mismatch. Tags, from the
  request's last episode: `local` (`covered <= local_hit`), `external`
  (`<= start`), `computed`. Two kinds of record carry `episode` and are
  tagged from that episode instead:
  - Mamba state groups in `mamba_cache_mode=align` (spec.json
    `probe.state_groups`): vLLM frees a block-end state once compute moves
    past it, so none is left at finish. The probe digests a state in the
    step that holds it. Offers (`kv_connector_block_state.boundary_state_offloads`,
    for requests that step computes, boundary above the episode start) are
    digested in `wait_for_save`, after the forward wrote them. Each
    episode's start state (block `start / block_size - 1`: the local hit or
    what a connector loaded) is digested in `handle_preemptions`. vLLM
    0.32's V2 model runner (what runs: `Using V2 Model Runner` in every
    `vllm.log`, spec.json `model_runner`, pinned by `G0.spec`) calls it
    every step from `kv_connector.pre_forward`, after `update_requests`
    (zeroes new blocks, applies COW copies) and `preprocess_state` (the
    align precopy, which reads that block into a later one: src
    `(start-1)//bs`, dst `cdiv(start+sched, bs)-1`), and before a
    synchronous `start_load_kv` and the forward. So the digest is of the
    bytes the forward computes from: a sync load lands after it and is
    `mismatched`. The preempted-episode digest runs there too, so a freed
    block vLLM reallocated and overwrote earlier in the step (in
    `new_block_ids_to_zero`, or a `kv_cache_block_copies` destination) is
    dropped, recorded in `requests.jsonl` `preempt_dropped`, and counts as
    an undigested computation; an all-zero computed digest is never a
    reference (`ref_zero_copies`, `computed_zero`), since a load that
    wrote nothing into a zeroed block would match it. `requests.jsonl` `states` lists `[episode, covered,
    group, what]`; `what` other than `offer` / `start` is a skip
    (`not_offered`: a step ended on a boundary vLLM offered neither in
    that step nor in the next one, e.g. past the prompt; `start_*`: a
    start state not digestable). A state
    exists only where a step ended exactly on the boundary, so the
    reference counts only those computations (`digest_compare`).
    `state_skipped` reports the skips; any `start_*` fails the digest
    verdict (a loaded state nobody judged).
  - Preempted episodes: `handle_preemptions` (before the step that reuses
    the freed blocks) digests the fully computed blocks the preempted
    episode left, so its computations are references and its loads are
    judged.
- Digest comparison (`digest_compare.compare(REF, CAND, invariant)`): a loaded
  block (local or external) is `matched` if all its layers equal some computed
  copy of the same (rank, group, prefix key) in REF; `mismatched` if it equals
  none and the reference is exhaustive (batch-invariant, or every computation
  of that key in REF was digested); `uncertain` otherwise; `unreferenced` if REF
  never computed the key. A computed block is `exact` if it equals another
  computed copy; with `split_invariant=false` only copies with the same
  *split* count as references: split = (start of the computing episode, the
  `progress` chunk ends of that episode below the block's end, a hash of the
  digests of every block of the request at or below the start on that rank),
  i.e. where the cached context ended, how prefill was chunked up to the
  block, and which context bytes it attended to (a context block computed at
  another split has other bytes, and so does everything computed on top of
  it; wrong context bytes are judged where they enter, as loaded blocks).
  REF may be several probe dirs (their computed copies pooled). A computed
  block with copies only at other splits is `computed_unmatched_split`, one
  REF never computed is `computed_unreferenced` (REF != CAND); both are not
  judged. `computed_by_pass` breaks these counts down by the candidate's
  `rsi_pass`. The split key misses a prefill chunk that ends without
  crossing a key-token boundary; that can only give two different MLA splits
  the same key, i.e. make an honest block inexact (fail-closed), never a
  wrong block exact.
- G0.spec: the arm's scheduler spec dump and per-layer view layouts (shape
  without the block dim, stride, dtype) equal the baseline's, and the
  baseline's equals the committed `spec.json`. Volatile fields (`num_blocks`,
  tensor `size`, `layer_stride`, model path, `max_model_len`) are excluded.
  The dump includes the effective `compilation` config (mode, backend,
  `inductor_compile_config`, `cudagraph_mode`; the gsm8k stage's), so a
  compile override is caught here as well as by `G0.vllm_args`, whose
  parser reads every spelling vLLM accepts (`--flag=value`, `_` for `-`,
  abbreviations, `-cc`, dotted keys).
- Record vs gate rev (S1): a run record is judged by the gate rev it ran
  under (`provenance.gate.rev`; re-judge with that tree when it matters).
  When an offline re-judge with a later rev meets a check whose inputs the
  record lacks because the gate did not produce them yet, the check is
  `{"pass": true, "na": "record predates this check"}` (with `would_be`),
  never a retroactive fail: `G0.vllm_args` (no stage has a `vllm_cmd`),
  `G2.batch_control` (plain gsm8k ran without an `alone` pass) and the
  self-test sabotage count. `G0.spec` (R1) skips, as `na_fields`, the
  committed spec fields the record's probe did not write yet
  (`run_gate.SPEC_FIELDS`: retention since e620462, `compilation` since
  9204863, `model_runner` since schema 3). The record's schema is its
  `probe.schema`, else the newest of those fields present (fields were only
  ever added). A field its schema knew that is missing, or any field
  present with another value, still fails. A record that has them is
  judged in full. Adding a dump field: bump `rsi_probe.SPEC_SCHEMA` and add
  it to `SPEC_FIELDS`.
- G2 controls (baseline arm): `plain_equals_baseline` (gsm8k cold completions
  identical: the probe changes nothing), `warm_equals_cold` (under invariance:
  reset + poison change nothing), `digest_self.gsm8k` (no local mismatch).
- G2.digest_self.gsm8k / digest_vs_baseline.gsm8k: local and external
  `mismatched == 0`, `unreferenced == 0`, `external_matched > 0`, judged
  (matched + mismatched) >= 0.9 of all external blocks, and under invariance
  `computed_exact == computed_compared`. With `split_invariant=false`,
  digest_vs_baseline.gsm8k also needs, in each of `cold` and `warm`, judged
  computed blocks > 0 and `compared / (compared + unmatched_split +
  unreferenced) >= 0.9` (`MIN_COMPUTED_COVERAGE`; REF is the baseline's
  gsm8k probe plus its `resume-<label>` replay of this arm, below), and in
  `warm` zero unjudged computed blocks on
  prompts whose length is not a multiple of the hit alignment (a prompt
  without a recorded length counts as non-aligned; only aligned prompts,
  where the plain split stops one block short of a connector's hit, may
  leave warm blocks unjudged). The split control's `aligned` exemption
  likewise never applies to a request without a token record.
  G2.digest_self.replay: same, with no
  judged-fraction floor (untagged conversations share prefixes).
- G2.digest_groups (baseline and every arm, gsm8k probe): every group the
  oracle scores (not skipped, not scratch) has digest records and is not in
  `probe.digest_skipped_groups`. The probe lists a `MambaSpec` group there
  outside `mamba_cache_mode=align` (a state block is the running state) or
  with speculative blocks; align-mode state groups are digested per step.
- G2.preempt (when `--workloads` includes `preempt`; part of the official
  profile on state-group models only): the first 32 gsm8k prompts, each decoding exactly 640
  tokens (`ignore_eos`, no stop), cold then warm, on a GPU pool of
  `preempt_gpu_blocks` blocks (4 requests of 2048 tokens, at least one
  `max_model_len` request; `max_num_seqs` capped at the block count, which
  vLLM requires for Mamba). Requests outgrow the pool, vLLM preempts and
  resumes them, and every request and preempted episode is digested.
  Passes iff the exhaustive digest verdict passes (as `G2.digest_self.cap`,
  judged >= 0.9 for an arm), vLLM preempted some request
  (`preempted_requests > 0`), and some load of a resumed episode was
  compared (`after_preemption`). Required for the honest self-test when run
  and, on state-group models, always (missing = fail); a negative arm is
  judged by the gsm8k digest checks. On Qwen3.5-4B (40
  blocks) the baseline preempts 16 of 64 requests and judges 62 resumed
  loads, 0 mismatched.
- G2.gsm8k: accuracy within 0.01 of the baseline at the same batch
  composition, for every arm, the honest self-test included. Batch
  invariant: `|acc(pass) - acc(baseline cold)| <= 0.01` for cold and warm.
  `batch_invariant=false`: batched accuracy moves with load timing (honest
  warm vs baseline cold on Qwen3.5-4B, n=128: -1.56 / -0.78 / +0.78 pp in
  three runs), so the gated pass is `alone_ref` (the first 64 prompts one
  at a time, after reset + poison) vs baseline `alone_ref`, and the batched
  deltas are reported (`batched_delta_pp`). One request at a time is the
  composition every arm shares, so an honest arm loads at the baseline's
  split and computes the same numbers (baseline and plain `alone_ref`
  agree in every logprob); the tolerance stays 0.01 and is no longer
  inside batch noise. `alone_ref_differ` (completions vs baseline
  `alone_ref`) is reported.
  G2.gsm8k_outputs (when gate.json declares batch invariance; otherwise
  reported only, see `G2.batch_control`): every completion equals
  baseline cold, cold and warm; with `split_invariant=false`: every cold
  completion equals baseline cold (same split), and every warm completion
  equals the baseline `resume-<label>` replay's completion of the same
  prompt (`warm_reference`; without that stage, the plain arm's `split`
  completion), except on prompts with `prompt_len %
  cache_hit_alignment_tokens == 0` that the replay could not resume where
  the arm did (`warm_differ_vs_split_aligned`, reported only: vLLM's own
  hit stops one block short there, a connector's need not); a non-aligned
  prompt the replay missed or resumed elsewhere counts as differing. The
  warm differ count vs baseline cold is reported only.
- Split-variant references (`split_invariant=false`, every arm with a
  gsm8k stage, needs the baseline arm): right after the arm's gsm8k stage
  the gate runs a fresh baseline stage `baseline/resume-<label>`
  (`run_gate.resume_plan`, `workloads.run_resume`). From the arm's probe
  trace it takes each warm request's prompt index (`rsi_conv`, set on every
  gsm8k request), its first episode `[step, local, start]` and groups the
  requests by `step` into waves in the arm's schedule order. After a
  warmup and reset + poison it replays the waves. Each request whose start
  exceeds its local hit is first sent as its first `start` prompt tokens
  with `max_tokens=1` (`resume_prewarm`, never scored), then whole
  (`resume`, streamed). A wave starts once every request of the previous
  wave has its first token, so the replay's local hits come from its own
  earlier requests, as the arm's did. A warm request vLLM preempted in the
  arm (several episodes; the probe digests every one) gets its later
  episodes chained: after the replay's request of that prompt finishes,
  each episode `k` runs as a request of the first `episode_tokens[k]`
  tokens of the replay's own sequence (prompt + its output so far,
  `return_token_ids`), pre-warmed to its start by the same rule
  (`resume_ep`, `rsi_turn` = k), so its context is the replay's previous
  episode, held in vLLM's prefix cache, and its computed blocks get a
  same-split copy. Its completion is the replay's text up to that point
  plus the episode's. An episode the replay cannot form (its output is
  shorter, or a streamed chunk straddles the episode's first token) stops
  the chain; that prompt then has no completion (differs) and the episode
  is `episode_missing`. A trace without `episode_tokens` (probe before
  this field) makes the plan fail closed. Its digests join REF and its
  completions are the warm outputs reference. `G2.resume_reference` (per
  arm, and required for the honest self-test on split-variant models)
  passes iff the stage ran, its `digest_self` passes and the plan has no
  problem (every prompt has exactly one warm request with a valid index,
  and every preempted one its episode lengths); how many requests or
  later episodes the replay resumed elsewhere (`start_differs`,
  `episode_start_differs`), or missed, is reported, and their blocks stay
  unjudged (coverage) and their completions count as differing
  (non-aligned). The resume stage gets gate.json `gsm8k_vllm_args` like
  the gsm8k stage it replays (`G0.vllm_args` checks it). A resume stage
  that crashes fails that arm's `G2.resume_reference`, not the baseline,
  for a record-only arm; for the arm under test the gate re-raises and the
  run aborts (review N1). The replay shows what the
  baseline computes from the arm's resume points; it cannot make wrong
  loaded bytes look right: those are matched against computed copies,
  never against the arm's own.
- Split invariance (`split_invariant`, `G2.split_control`, baseline verdicts).
  Batch invariance makes a token's KV independent of batch composition, not
  of *where the request's computation starts*. vLLM 0.32's MLA prefill
  (`vllm/model_executor/layers/attention/mla_attention.py`: `forward_mha`
  attends to the new tokens, then, when `prefill_metadata.chunked_context`
  is set, i.e. `num_computed_tokens > 0`, runs `_compute_prefill_context` over
  the cached context and combines the two with `merge_attn_states`) splits the
  softmax at `num_computed_tokens`, so a prompt hit that ends at block 68
  computes the remaining KV and logits with different rounding than a hit
  ending at 72. Dense FLASH_ATTN attends over the whole context in one kernel
  and is split-invariant. Evidence on DeepSeek-V2-Lite TP2 TRITON_MLA, plain
  vLLM, `VLLM_BATCH_INVARIANT=1`: one 900-token prompt, prefix cache warmed to
  0 / 256 / 512 tokens: 256 changes logprobs, 512 diverges at output token 3.
  An honest connector's warm hit ends at the last full prompt block, the
  baseline's at the few-shot prefix, so on such models warm outputs differ from
  baseline cold and computed copies differ across runs although every loaded
  block is byte-exact. Onboarding writes the flag in its model's gate.json
  (on its own branch), but the run never trusts it: the plain arm
  (no connector, no probe) runs `split_ref` (the gsm8k prompts, greedy, with
  logprobs, after a reset + poison; hits end at the shared few-shot prefix)
  and `split` (same, but each request is preceded by the same prompt with
  `max_tokens=1`, so its hit ends at the last full prompt block, where a
  connector's warm load ends). `--enable-prompt-tokens-details` (plain arm
  only) reports cached prompt tokens per request. `G2.split_control` passes
  iff measured invariance (every completion and every sampled-token logprob
  equal between `split` and `split_ref`) equals gate.json's
  `split_invariant`, `split_ref` equals plain cold, and the pre-warm took
  (`split` cached `>= (prompt_tokens - 1) // align * align` tokens, i.e. up to
  the last full prompt block) for >= 0.9 of requests. So `false` with an
  invariant control, or `true` with any divergence, fails the run. With
  `split_invariant=false` the loaded-block checks (`mismatched == 0`,
  `unreferenced == 0`, judged >= 0.9, `external_matched > 0`) and G2.gsm8k
  stay as they are; only `computed_exact` (same-split references, with the
  coverage floor above) and `G2.gsm8k_outputs` (warm vs the plain `split`
  completions) change. A correct connector's warm pass is the control's
  `split` pass with the prefix served from the store instead of vLLM's
  prefix cache: same split, same bytes (byte-exact loads), so under batch
  invariance its completions equal `split`'s exactly; only the aligned
  prompts end their hit elsewhere (the connector loads to `prompt_len - 1`).
  Measured on DeepSeek-V2-Lite TP2: honest self-test 0/128 differ from
  `split`, liar/bitflip/offbyone 128/128; the old connector 38/1319, all 38
  aligned (82 aligned prompts).
- Batch invariance (`batch_invariant`, `G2.batch_control`, baseline
  verdicts). It decides whether outputs are compared at all
  (`G2.warm_equals_cold`, `G2.gsm8k_outputs`), so the run measures it too:
  the plain arm's `alone` pass repeats the first 64 prompts of `split` (same
  pre-warm, so the same split) at concurrency 1. `G2.batch_control` passes
  iff measured invariance (every completion and sampled-token logprob of
  `alone` equal to the same prompt's batched `split` one) equals gate.json's
  `batch_invariant` and the `alone` pre-warm took for >= 0.9 of them. vLLM
  0.32 cannot be batch invariant on GDN (Qwen3.5): the backend selector
  rejects `VLLM_BATCH_INVARIANT` for Mamba backends without
  `supports_batch_invariance`, and plain vLLM batched vs alone flips a
  completion. So `false` with an invariant control (alone == batched),
  or `true` with any divergence, fails the run with the reason named.
  Output rule with `batch_invariant=false`: a warm pass's batches depend on
  load timing, so no reference shares its batch composition, and
  `G2.warm_equals_cold` / `G2.gsm8k_outputs` report the differ counts
  (`gated: false`; vs baseline cold, and `warm_differ_vs_split`) without
  gating. The output controls compare one batch composition instead: the
  plain and baseline arms also run `alone_ref` (the same 64 prompts, one at
  a time, no pre-warm). `G2.plain_equals_baseline` requires plain and
  baseline `alone_ref` equal in every completion and sampled-token logprob
  (batched `cold_differ` reported), and `G2.split_control` compares plain
  `alone_ref` with plain `alone` (basis: baseline `alone_ref`).
  `same_start` reports pairs whose two passes started at the same point
  (no split compared there); on Qwen3.5-4B gsm8k every prompt's last full
  block is the shared prefix (1056), so the split claim rests on the P5
  control and is not exercised by this workload. `G2.split_control`
  reports `vacuous` (and `measured_invariant: null`) when `same_start >=
  0.9 n` with no divergence; the verdict is unchanged (it can only fail a
  right load, never pass a wrong one). G2.gsm8k (accuracy within 0.01) gates, and every digest check
  stays byte-exact and hard: gsm8k digests every computation, so the
  reference is exhaustive (`digest_compare`) and a loaded block or state
  that equals no computed copy is `mismatched`.
- Compile cache (S5): every arm has its own `VLLM_CACHE_ROOT`,
  `<run>/.compile-cache/<label>` (torch compile artifacts and the Inductor
  cache vLLM keeps under it; `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`
  and `FLASHINFER_WORKSPACE_BASE` point inside it too), emptied before
  the arm's first stage and shared by its later stages. The isolated arm
  under test gets only its own bound rw. So no arm loads an artifact another
  arm or an earlier run wrote, and a tampered artifact can only change its
  own arm's numbers, which the baseline comparisons judge. Each arm
  compiles once per config (gsm8k and the others differ), a few minutes per
  arm.
- Triton autotune picks (`seed_autotune`). Kernels with
  `@triton.autotune` outside Inductor (vLLM's FLA ops for GDN:
  `chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h`,
  `solve_tril`, `cumsum`, ...) pick a tile config by timing a fixed list
  on first use, and the pick changes the numbers (BK/BV split a
  reduction). Inductor's `deterministic` does not reach them. Fresh
  per-arm caches re-timed them per arm, so in
  `qwen3.5-4b-selftest-fbcc3fd` every arm ran different picks and
  baseline differed from plain and honest in 14/64 alone_ref completions
  (plain == honest: 0). Each fresh arm root therefore gets a copy of the
  baseline's `*.autotune.json` (Triton keys each by kernel, compiler,
  cache-invalidating env and the config list, and loads it instead of
  timing under `TRITON_CACHE_AUTOTUNING=1`, which vLLM sets by default and
  the gate sets explicitly). Only the picks are copied, never a compiled
  kernel or graph, and it is a copy: an arm can change only its own.
  Triton's file format can also name IR to compile instead of the kernel
  (`ir_override`) or a `pre_hook`; seeding refuses any file whose entries
  hold anything but integer tile configs and numeric timings
  (`autotune_file_problem`). Probe
  (GPU5, `~/kexi/tmp/rsi-gate-hybrid-test/swap`, first-token logprobs of
  4 gsm8k prompts): the baseline's full cache gives `-0.389455 -1.307971
  -0.90808 -0.769167`, plain's `-0.385402 -1.285329 -0.907576
  -0.780117`, and fresh compiles of the plain and the connector config
  with only the baseline's picks both give the baseline's numbers.
- Compile determinism (`gsm8k_vllm_args`). vLLM 0.32 compiles the model
  once per config hash (`<cache root>/torch_compile_cache/<hash>`; before
  S5 the global `~/.cache/vllm`, reused by later runs). `VllmConfig.compute_hash` appends
  `kv_transfer_config.compute_hash()` iff a connector is configured, and
  that hash has no factors, so every connector arm (baseline, selftest,
  the arm under test) shares one compile and plain has another (`old` a
  third: its `VLLM_PLUGINS` differ). Inductor picks some kernel configs by
  benchmarking at compile time, so two compiles of one config need not
  match. On Qwen3.5-4B they do not: concurrency 1, greedy, first sampled
  token of gsm8k prompt 0, the plain artifact compiled earlier into the
  global cache gives `-0.388626` and the plain config compiled fresh twice
  gives `-0.390268` both times, and the numbers follow the artifact
  (loading the other compile's cache swaps them; GPU5 probe 2026-10-24,
  `~/kexi/tmp/rsi-gate-hybrid-test/numerics`). So it is nondeterminism
  across compile times, not within one. In selftest run
  `qwen3.5-4b-selftest-aa42a44` plain and baseline therefore differed from
  the first token in all 64 `alone_ref` requests (12 completions), which
  failed `G2.plain_equals_baseline` and `G2.split_control`. The two
  artifacts' recorded autotune choices differ in two pointwise kernels
  only, which cannot change values; the choice that does is not pinned.
  `inductor_compile_config.deterministic=true` (no benchmark-driven
  choices) makes three fresh compiles, plain twice and baseline once,
  give identical logprobs. gate.json `gsm8k_vllm_args` is appended to
  `vllm_args` on every arm's gsm8k stage, where the cross-arm exact
  comparisons run (`G2.plain_equals_baseline`, `G2.split_control`,
  `G2.gsm8k_outputs`, `digest_vs_baseline`). Replay, cap and preempt
  stages keep vLLM's default compile: they compare arms by timing and
  hit rate (G3/G4/G5), and their digests are judged within the stage's own
  process (`digest_self`, `G2.preempt`). `G4.gsm8k_{cold,warm}.seconds`
  times both sides under the gsm8k config (as on Qwen3-4B, where gsm8k
  runs `VLLM_BATCH_INVARIANT=1`); `G4.replay_*` runs the default config.
  `G0.vllm_args` requires each gsm8k flag as the last occurrence on every
  gsm8k stage. Before S5, Qwen3-4B and DeepSeek-V2-Lite set no
  `gsm8k_vllm_args`: their plain and connector compiles in the global
  cache happened to make the same choices. With per-arm caches every arm
  compiles fresh, so that luck is gone: `qwen3-4b-gate-hybrid-slice-3d6b8e6`
  (default compile) had plain, baseline and connector_v2 pick different
  reduction configs and give three different gsm8k output sets
  (`G2.plain_equals_baseline` 35 differ, connector_v2
  `digest_vs_baseline` all mismatched), while `G2.batch_control` and
  `G0.vllm_args` passed. Qwen3-4B's gate.json now sets the same
  `gsm8k_vllm_args` (and its committed `compilation` block the
  deterministic config). Its accepted records ran under the old gate.json
  and stay judged by it (S1). DeepSeek-V2-Lite needs the same line when
  its branch takes this gate.
- G3 (`oracle.py`): replays `requests.jsonl` with unlimited capacity. A block
  ending at token e is stored at the first step whose `progress` reaches e
  (checkpoint groups: only a step ending exactly on e) and is visible to
  requests first scheduled after that step. Kinds come from the spec dump
  (no window -> Full, sliding window sw -> Window{sw-1}, sw == 1 -> Scratch,
  Mamba -> Checkpoint, `UniformTypeKVCacheSpecs` -> its inner Kind, unknown ->
  error). Per request: `ext = max(0, min(actual, oracle) - local)`,
  `ext* = max(0, oracle - local)`; `E_ext = Σext / Σext*`;
  `excess = Σ max(0, actual - max(oracle, local))`. G3.gsm8k_warm and
  G3.replay_warm: `excess <= one block per request`, no load errors; replay
  G3 also requires G2.digest_self.replay. Reported (`reported`, not in
  `pass`): `E_ext` (reference 0.95, `None` is `below`) and
  `oracle_hit_rate` (reference 0.5), with hit, local and external rates.
- G3.no_eviction: `pegaflow_cache_block_evictions_total` delta = 0 in every
  gsm8k and replay pass (the oracle has unlimited capacity; `cap*` passes are
  meant to evict and are excluded; `native` / `lmcache` have no PegaFlow server: n/a, pass).
  The server runs with
  `--pool-size 64gb --disable-numa-affinity` (per-NUMA pools give one GPU only
  `pool_size / numa_nodes`; 64gb since 2026-10-24: a 160 GiB pinned pool took
  2-3 min to allocate, 60-95 s to free, and stalled other processes' JIT;
  measured U: Qwen3-4B gsm8k 30.3 GiB, replay ~26 GiB, DeepSeek-V2-Lite TP2
  <= 7.8 GiB). The run fails before any store stage (`PoolTooSmall`) when the
  uncapped workload's `UniqueEstimate` from the committed spec (replay:
  prompts + output tokens; gsm8k: prompts, a lower bound), or later the U
  measured on the baseline's trace, exceeds `--pool-size`;
  `summary.pool_checks` records each check.
- G4 (hard since 2026-10-23, user: "性能需要达标才行，需要比基准快"; every
  check is against the baseline arm on the same run unless it says native,
  medians over the listed passes, `MAX_OVERHEAD = 0.02`):
  - `G4.replay_warm.ttft_p50` / `.req_per_s`: timed warm passes (warm2..N)
    strictly better than baseline's.
  - `G4.replay_warm.ttft_p50_vs_native` / `.req_per_s_vs_native` (hard
    since 2026-10-26): the same timed warm passes strictly better than the
    in-run `native` arm's (vLLM's OffloadingConnector), by the same rule
    as vs baseline (`_cmp` with slack 0: median of warm2..N vs median of
    native's warm2..N, `<` for TTFT, `>` for req/s; a tie fails). User
    decision, 2026-10-26 (human gate owner): G4's warm-gain checks also
    compare against native, not only the no-offload baseline; the arm
    under test's replay warm TTFT p50 and req/s must be no worse than
    native's in the same run, as a strict hard check with no tolerance.
    "No worse" is implemented with the warm-vs-baseline comparison
    (strictly better), so equal medians fail. Applied to the connector
    under test only, not to `old` / `native` / `lmcache` or the self-test
    arms. Only native's timing is used: its own G0 / G2 failures (it fails
    `G0.spec` by construction) do not disqualify it as a performance
    reference. A missing or crashed native replay stage fails both checks
    and makes the run non-official (Official run, above). Void (fail) when
    the arm's `G2.digest_self.replay` fails, like every replay G4 check.
    Why the overhead checks below stay against baseline: they bound what
    the connector costs when it has nothing to load (cold replay: vLLM's
    own prefix cache serves the earlier turns; gsm8k: the few-shot prefix
    stays local), and the only cost-free reference for that is vLLM
    without any connector. Native is another offloading store with its own
    save cost, so "within 2% of native" would let the two costs add up;
    the warm checks are where an offloading store competes, so native is
    added there. G2 correctness references likewise stay baseline's (native
    is not a byte-identity reference: it fails `G0.spec`).
  - `G4.replay_cold.ttft_p50` / `.req_per_s`: cold pass at most 2% worse than
    baseline's cold (nothing to load yet; vLLM's own prefix cache serves the
    earlier turns, so cold can only tie; this bounds the save cost).
  - `G4.gsm8k_cold.seconds` / `G4.gsm8k_warm.seconds`: wall time at most 2%
    worse than baseline (the shared few-shot prefix stays in vLLM's local
    cache, so gsm8k bounds per-step connector overhead).
  - Replay G4 checks are void (fail) if G2.digest_self.replay fails. p99 values
    are recorded in pass results but not gated.
- G5 (`oracle.simulate` / `summarize_capacity`, `run_gate` workload `cap`):
  - Store events, visibility and scoring are G3's; per (group, key) block
    bytes = `real_page_size_bytes` (unpadded, as digested; `page_size_bytes`
    if absent) x the group's `num_layers` (`UniformTypeKVCacheSpecs` already
    sums its layers: x 1) x distinct TP copies `tp / kv_tp_replicas`
    (`KVCacheConfig.kv_tp_replicas`: ranks holding identical bytes; MLA's one
    latent head at TP2 = 2, so one copy). Stores keep one copy of replicated
    bytes: PegaFlow on DeepSeek-V2-Lite TP2 (2026-10-23) reported
    `pegaflow_save_bytes_total` 5,799,278,592 = 11,653 insertions x 497,664 B
    (18,432 page x 27 layers, one copy); vLLM's offloading CPU spec stores
    one copy under `replicated_layout` and `cpu_bytes_to_use` is the total
    over workers (`v1/kv_offload/cpu/spec.py`), so `native` gets `C` as is.
    Skipped groups (no kv transfer, eagle) hold no bytes. GPU KV bytes = Σ
    `kv_cache_config.tensors[].size` of one rank (not host-resident; groups
    sharing a tensor count once) x the same TP copies (Qwen3-4B:
    11,161,829,376 = `num_blocks x` block bytes; DeepSeek-V2-Lite TP2:
    3,096,465,408).
  - Trace: `--cap-convs` x `--cap-turns` (official `auto` x 12) conversations
    of `--replay-file`, `--replay-out-tokens` outputs, `--concurrency` workers.
    `--replay-skip N` (replay and cap) starts at conversation N of the file.
    The official value is 0. Any N > 0 is a held-out slice
    (`summary.held_out`, `summary.replay_skip`), never official.
    `auto`: before the baseline cap stage, from the baseline's gsm8k spec
    dump of this run (else replay; never preempt, whose GPU pool is shrunk;
    else the committed dump; `cap_plan.spec`). An official run always has
    gsm8k before cap, so only non-official runs such as `--workloads cap`
    reach the committed fallback; the floor reads GPU KV from the cap
    stage's own dump in every case. Then conversations are added from the file start (at least 64) until the
    estimated `U` reaches `1.1 x 2 x GPU KV / 0.5`. `UniqueEstimate` applies
    the oracle's store rules per group Kind, at block bytes: Full / Window
    groups count distinct prompt-prefix blocks plus each request's generated
    complete blocks (equal to the measured `U` on the recorded Qwen3-4B
    32 x 12 and 8 x 6 traces). Checkpoint groups count a block only where a
    step ends on its boundary: each turn resumes at its local hit (resolve
    over what the trace stored so far), its prefill step ends follow vLLM
    0.32's align-mode split (`prefill_step_ends`: non-final chunks end on a
    block boundary, stops at the last cacheable boundary and at the
    shared-prefix junction, `--max-num-batched-tokens` per step), and decode
    stores every boundary it crosses. A request running alone is assumed;
    concurrent requests share the step budget, which moves step ends as
    well as adding them, so this is not a bound. The ratios below are
    observations (estimate below measured on the recorded traces); an
    over-estimate gives fewer conversations and the measured floor fails
    the run closed (not official). The Checkpoint model also assumes
    `hash_block_size == block_size`, no Eagle / speculative blocks and
    `num_prefill_checkpoint_blocks == 0`; a spec with a Checkpoint group
    that shows otherwise is recorded in `cap_plan.unmodeled` and the run is
    not official. Qwen3.5-4B: s1 8 x 6 estimates 3.39 GB vs 3.49 GB
    measured (0.97; the Full-only rule gave 4.47 GB, 1.28x); the official
    256 x 12 (32 workers) estimates 136.33 GB vs 167.96 GB measured (0.81,
    full-attention group exact; run `qwen3.5-4b-cap-sizing-baseline-4f3cb97`).
    If the whole file is
    not enough every conversation is used and the measured floor fails the
    run closed (`capacity below 2x GPU KV (whole replay file used)`).
    Every replay request carries `vllm_xargs.rsi_conv` / `rsi_turn`; the probe
    records them as `conv` / `turn`. Pass `cap_cold` starts conversations in
    order 0..n-1; warm pass `j` (`cap_warm` = 1, `cap_warmK` = K) starts them
    rotated by `j * n // 4`.
  - Baseline cap stage (runs after gsm8k and replay, before any store arm's
    cap): passes `cap_cold`, `cap_warm` (reset + poison before it, digested
    like replay `warm`), no connector. `U` = bytes of the distinct (group,
    key) blocks the unlimited oracle stores over its `cap_cold` pass (every
    block of the trace appears there; a rotated warm pass adds only differing
    generated tails, which `U` does not count). Reference local hit
    `ref(pass, conv, turn)` = the baseline's `local_hit` at that position, from
    `cap_cold` for `cap_cold` and from `cap_warm` for every warm pass.
    `G5.capacity_floor` (in `summary.json`, not a verdict): `C(0.5) >= 2 x`
    the baseline's GPU KV bytes, else `not_official_because: capacity below
    2x GPU KV`.
  - Store-arm cap stages, per fraction `f` in `--cap-fractions` (must include
    0.5; official `0.25,0.5,0.75`), dir `cap` for 0.5 and `capNNN` (percent)
    otherwise: fresh vLLM and (except `native` / `lmcache`, CPU bytes =
    `C`) a fresh PegaFlow server with `--pool-size C`
    `--disable-numa-affinity`, `C = int(f x U)`; passes `cap_cold`,
    `cap_warm`, `cap_warm2..N` (timed), all digested (every request of
    every pass: the timed passes are where `E_C` is scored, so their loads
    must be judged too, and with nothing undigested the reference is
    exhaustive; cap timing is not gated, the digest cost is accepted), the
    replay
    settle / reset / poison protocol. `pool_capacity_bytes` = the server's
    `pegaflow_pool_capacity_bytes` at start; the server allocates the whole
    pool up front and rounds allocations to 512-byte units and
    power-of-two-mantissa size classes (at most 1/8 waste): server headroom,
    counts against the server.
  - Simulated stores over the arm's own cap trace (all its `cap_*` passes, in
    first-schedule order), `C` bytes each, with `ref` as every request's local
    hit:
    - `H*_C` (clairvoyant): the next use of an item is the next request whose
      unlimited-oracle resolved prefix `P` needs it (`semantics.needed` at
      `P`, minus what `needed` at `ref` already gives); an arriving item with
      no next use is not stored; otherwise items whose next use is furthest
      are evicted first, and an arriving item whose own next use is the
      furthest is not stored. Greedy under variable item sizes and blind to
      partial-prefix coherence: a strong reference, not a proven optimum
      (`E_C > 1` is allowed).
    - Fallback-aware candidate (since 2026-10-26). The use sets above are
      complete only when every group's `needed` is prefix-closed:
      `needed(q) ⊆ needed(P)` for every resume point `q < P`, as for Full.
      A Checkpoint's `needed(q)` is the state at `q`, which `needed(P)` does
      not contain, so the plain Belady saw every earlier checkpoint as
      unused, refused or evicted it, and lost the fallback resume point a
      smaller store needs. LRU kept it and resumed there. On Qwen3.5-4B
      (`2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`) that put `H*_C` 0.690
      below `H_lru` 0.794 at `C/U` 0.25 (`E_C` > 1), and below LRU on 239
      requests at 0.5 (P5-it2 review). `oracle.prefix_closed` checks the
      property from `semantics.needed` (so from the Kind, never the model
      name) over the trace's resume points. Where it fails,
      `belady_fallback` runs too: the same Belady, but a request uses the
      union of `needed(q)` over every resume point `q` in `(ref, P]`
      (`resolve`'s step, the lcm of the block sizes). Where it holds, that
      union is `needed(P)`, so the variant would equal the plain Belady and
      is not run.
    - `H*_C` = per segment, the candidate store with the largest `Σ ext*_C`
      among `belady`, `belady_fallback` (if run) and `lru`, ties in that
      order. Each candidate is one store of `C` bytes over the whole trace,
      so the bar is a hit rate some real store reaches, and it is never
      below `H_lru` in `Σ ext*_C` (the selection key; the hit rates could
      in principle rank differently, and agreed on every segment of the
      re-scored runs). Exact optimality (variable item sizes, all-groups
      coherence) is not tractable, and neither variant dominates: spreading
      uses over every fallback point dilutes the greedy choice (P5-it2: the
      fallback variant is up to 0.005 below the plain one at 0.75). Recorded
      per segment: `oracle_c_policy` (the winner) and `oracle_c_candidates`
      (tokens, `ext*_C` tokens and `lru_above_requests` per candidate);
      `oracle_c_policies` per stage; `diagnostics.oracle_c` stays the plain
      Belady, with `diagnostics.oracle_c_fallback` next to it. The
      verdict's `oracle_c_groups` / `oracle_c_passes` are the diagnostics of
      the `warm_timed` winner, and when that is not the plain Belady, the
      plain Belady's are reported too, as `belady_groups` /
      `belady_passes`. `efficiency_c_own_local` uses the same candidates at
      the own local hit. Full-only models (Qwen3-4B, DeepSeek-V2-Lite) run
      the same two stores as before and the plain Belady wins every
      segment, so their records re-score identical in every shared field
      (the record only gains `oracle_c_policies`, `oracle_c_policy` and
      `oracle_c_candidates`). `E_C` stays reported, not gated. One indirect
      effect: `freeze_vs_oracle_c` feeds `g5_problems` (officiality:
      `FREEZE_SEPARATION`), and a higher bar lowers it, so for a
      non-prefix-closed model a trace that did not separate freeze from
      `H*_C` before can now be official. That is the right direction, since
      the bar is a reachable store and the separation is real. No current
      run flips: Qwen3.5-4B 0.5 goes from 0.667 to 0.655 (v2), and full-only
      runs are unchanged.
    - `H_lru`: group-oblivious LRU, every item stored, a re-store of a
      resident item does not refresh it, a hit refreshes the blocks loaded.
    - `H_freeze`: items stored in arrival order until the next one does not
      fit, never evicted.
    - Hits: `resolve` over resident blocks plus each group's `needed` range
      at `ref`, capped at `max(unlimited oracle, ref)`.
  - Scores per segment (`warm_timed` = `cap_warm2..N`, and per pass):
    `ext = max(0, min(actual, oracle) - max(ref, own))` (unlimited oracle;
    `own` = the arm's own local hit: tokens vLLM held locally are never store
    credit, also when the arm's vLLM kept more than the baseline did),
    `ext*_C = max(0, oracle_C - ref)`, `E_C = Σ ext / Σ ext*_C` (`None` when
    `Σ ext*_C = 0`: fails). Reported: `lru_vs_oracle_c` and
    `freeze_vs_oracle_c` (`Σ max(0, H - ref) / Σ ext*_C`),
    `efficiency_c_own_local` (the same `E_C` with the arm's own local hit in
    place of `ref` for `ext`, `ext*_C` and the candidate runs; diagnostic only),
    `efficiency_c_cold` / `freeze_vs_oracle_c_cold` (`cap_cold`: within-pass
    multi-turn reuse after vLLM evicted the earlier turn; diagnostic only).
    `unmatched_requests` = requests without a `ref` (no position tag, or the
    baseline has no request at that position).
  - Load proof: `external_hit_bytes(pass)` = Σ over the pass's requests of the
    bytes of the blocks `needed` at `actual` that `needed` at the request's
    own local hit does not hold; `loaded_bytes(pass)` = the
    `pegaflow_load_bytes_total` delta of the pass.
  - `G5.replay_warm` (gated fraction 0.5): `E_C` on `warm_timed` reported
    (reference 0.9, not in `pass`); gated: `excess <= one block per request`
    (unlimited oracle, own local, as G3),
    no load errors, `unmatched_requests == 0`, `G2.digest_self.cap` passes,
    and for a
    PegaFlow arm `pool_capacity_bytes` present and `<= C` and Σ
    `loaded_bytes >= Σ external_hit_bytes` over `warm_timed` (`native` / `lmcache`: no
    server, pool and load proof not applicable). Reported with it: `U`, `C`,
    hit / oracle / oracle_C / LRU / freeze hit rates, per-pass eviction
    counter deltas, per-pass server I/O (`saved_bytes`, `loaded_bytes`,
    `external_hit_bytes`) and end-of-pass resident / used bytes, per-group
    stored / not-admitted / evicted counts and bytes and per-pass resident /
    stranded bytes for `H_lru` and for the store that set `H*_C` on
    `warm_timed` (`oracle_c_policy`; plus the plain Belady's as `belady_*`
    when another candidate won).
  - `G2.digest_self.cap` (cap probe dir against itself, every request
    digested): 0 mismatched and 0 unreferenced (local and external), and
    exhaustive: every loaded block judged. The only allowance is a block
    whose undigested computations are all preempted (non-last) episodes of
    digested requests (`*_uncertain_preempted`), at most 1% of the loaded
    blocks, reported as `preempted_uncertain`; any other `uncertain` fails.
    The connector under test must have external blocks (`judged > 0`: a
    passing G5 needs loads, and they must be judged); a reference arm
    (`old`, `native`, `lmcache`) with none passes flagged `vacuous`. Before
    2026-10-24 cap digested only every 4th conversation of `cap_warm` and no
    timed pass: undigested computations in `cap_warm2..N` made every key
    non-exhaustive, so wrong external bytes counted as `uncertain` and one
    match passed (re-review: 79 of 158 blocks corrupted, verdict passed).
    Baseline: the same, with no external block required.
  - `G5.curve`: the same evaluation at each other fraction (`E_C`, oracle_C,
    LRU, freeze, evictions, load proof, digests); never gated.
  - Why rotate: with identical warm passes a store that fills once and never
    evicts already holds what the next pass needs. Recorded traces with
    identical order: `freeze / H*_C` on the warm passes 0.96 (slice,
    connector_v2) and 0.92-0.99 (32 x 12 `old` arm), so a freeze policy could
    pass. A step-level vLLM simulation of the 64 x 12 trace (GPU KV 4745
    blocks, `C = 0.5 U`) gives 0.76-0.77 identical and 0.65-0.72 rotated
    (`H_lru / H*_C` 0.87-0.89 either way; `cap_cold` alone 0.71); it
    under-estimates freeze against real traces, so rotation is the gate's
    mechanism and the self-check below is its guard.
  - Separation self-check: if `freeze_vs_oracle_c` of the connector under
    test at the gated fraction is `>= 0.9` (`FREEZE_SEPARATION`), the trace
    does not separate a frozen store from the bar and the run is not official
    (`not_official_because: G5 trace does not separate freeze from H*_C`).
    This is workload validity, not a bar on the arm, so it stays hard.
- Self-test arms (`rsi/gates/selftest/connector.py`, full-attention groups
  and align-mode Mamba state groups): `honest` saves full blocks at finish and
  states when vLLM offers them (in `wait_for_save` of that step), and serves
  the longest prefix whose full blocks and states (every state group) are
  saved (must pass G0 and every G2 check, `G2.gsm8k` and, on state-group
  models, `G2.preempt` included); `liar` claims every complete unit and
  loads nothing; `bitflip` flips one byte of the first loaded block (state
  groups: an exponent bit of the first packed state); `offbyone` loads block
  i+1 into slot i (state groups: the same prefix's state of the next state
  group, same shapes, other layers: it does not depend on another saved
  prefix, which gsm8k on Qwen3.5-4B never has). On state-group models
  every load is async except `sync`'s: vLLM's align-mode precopy runs before
  a synchronous `start_load_kv`, so `sync` (honest bytes, synchronous
  loads) computes from a stale state and corrupts the state groups only.
  `sync` runs gsm8k only: in the preempt pool vLLM 0.32's scheduler dies on
  its synchronous Mamba loads (`allocate_slots`: "Cannot get 1 free blocks
  from the pool", run `qwen3.5-4b-selftest-687c5b2`).
  The connector reports what it corrupted per group
  (`<stage>/selftest/sabotage-*.jsonl`, `result.sabotage`). A negative arm
  passes (is caught) only if G0 passes and `G2.caught_per_group` passes:
  every group it corrupts (all transferred groups; `sync`: the state
  groups), each judged on its own, has in both `G2.digest_self.gsm8k` and
  `G2.digest_vs_baseline.gsm8k` at least as many loads equal to no
  reference (mismatched or unreferenced) as the connector sabotaged there,
  less the external loads that check left uncertain, and a nonzero
  sabotage count (R3: a gate that catches only some sabotaged loads fails
  its self-test).
  Preempt per-group counts are reported. Output deltas are reported only
  (one flipped byte need not change a greedy completion).
