# Gate evidence (moved out of docs/rsi/GATES.md)

Evidence and status narratives that used to live in `docs/rsi/GATES.md` on
`feat/semantics-rsi` (0f4e07a). GATES.md now states the rules only and points
here. Compaction: keep a passage only while a GATES.md rule or table row still
cites it; drop superseded numbers.

## lmcache status per model: evidence

Old GATES.md lines 200-378 (G5 summary), verbatim:

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
    each arm's warm pass at that arm's own resume points (GATES.md, G2
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
  vLLM 0.32.0, A100, by the issue reporter). Rebuild recipe: section
  "lmcache overlay build (PR #4467)" below.
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


## lmcache overlay build (PR #4467)

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

## Unlimited-store pool size (G3)

`run_gate` starts the server for gsm8k / replay with `--pool-size 64gb
--disable-numa-affinity` (per-NUMA pools give one GPU only
`pool_size / numa_nodes`). 64gb, not the earlier 160gb: on .44 a 160 GiB
pinned pool took 2-3 minutes to allocate and 60-95 s to free, and stalled
other processes' JIT builds. Measured `U` fits: Qwen3-4B gsm8k 30.3 GiB,
replay ~26 GiB, DeepSeek-V2-Lite TP2 <= 7.8 GiB; `PoolTooSmall` guards
models that do not.

## Split invariance: DeepSeek-V2-Lite evidence

Plain vLLM, DeepSeek-V2-Lite TP2 TRITON_MLA, `VLLM_BATCH_INVARIANT=1`: one
900-token prompt, prefix cache warmed to 0 / 256 / 512 tokens: 256 changes
logprobs, 512 diverges at output token 3.

## Working native

Native measured `E_ext = 1.0` on every timed pass of the Qwen3-4B / DSV2 it7
and Qwen3.5 it2 records.

## G5 fallback-aware candidate

On Qwen3.5-4B (`2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`) the plain Belady
put `H*_C` 0.690 below `H_lru` 0.794 at `C/U` 0.25 (`E_C` > 1), and below LRU
on 239 requests at 0.5 (P5-it2 review): it saw every earlier checkpoint as
unused and lost the fallback resume point. `belady_fallback` fixes that; it is
up to 0.005 below the plain one at 0.75 (P5-it2), so neither dominates.

## G4 vs native noise

Official runs time 2 warm passes; Qwen3-4B it7: v2 behind native by 2.1 ms
with a 25 ms warm2-warm3 range of its own (review `g4-native.md` SHOULD-1).
See "Decisions" in PROGRESS.md for how the user resolved it.

## gate.json per model, not per TP: Qwen3-4B TP2

The committed spec dump is per TP (`spec-tp<N>.json`); gate.json is not,
because the controls re-measure its claims every run. Qwen3-4B at TP2: the
slice `2026-10-26-qwen3-4b-tp2-slice-4bb7d8d` has `G2.split_control` 0/64,
`G2.batch_control` 0/64, `warm_equals_cold` and `plain_equals_baseline` with
0 differ; the self-test `2026-10-26-qwen3-4b-tp2-selftest-478187e` has split
0/128, batch 0/64, both output controls 0 differ. `spec-tp2.json` is the
slice's baseline gsm8k dump; the self-test's fresh one has 0 `G0.spec` diffs
against it.

## KV pool size follows the compile cache

Since fcd0534 every arm gets a fresh compile cache, and a fresh compile profiles 9.82 GiB of KV (71,472 tokens on Qwen3-4B TP1). An AOT load from the shared `~/.cache/vllm` profiles 10.4 GiB (75,696 tokens). Within a run all arms compile fresh, so v2 vs native stays fair. Cross-run comparisons with records whose logs show `Directly load AOT` (e.g. it7 `2026-10-25-qwen3-4b-p4-it7-off-a59a254`) need this caveat: the -5.6% pool shifted v2's warm p50 vs native from -1% to +17.5% in `2026-10-26-qwen3-4b-p5-it3-slice-c8a8842` (P5 it3).
