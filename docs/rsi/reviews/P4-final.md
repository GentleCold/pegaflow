# Final review: P4 DeepSeek-V2-Lite-Chat + Qwen3-4B regression (`feat/rsi-deepseek-v2-lite`)

Reviewer: independent agent, fresh context, 2026-10-25. Head 4f7cb61 (this
review is the next commit), base `origin/feat/semantics-rsi` d252bfe, diff
`origin/feat/semantics-rsi...origin/feat/rsi-deepseek-v2-lite`.

Official evidence (both on .44, read-only):
- MF4 DSV2: `~/kexi/rsi-runs/2026-10-24-dsv2-lite-p4-it6-off-a1e7fd2`
- MF5 Qwen3-4B: `~/kexi/rsi-runs/2026-10-24-qwen3-4b-p4-it6-off-a1e7fd2`
- References: MF1 `...-dsv2-lite-p4-mf1`, MF2 `...-qwen3-4b-p4-mf2`, A/B
  `...-ab-a-d56fcb2` / `...-ab-b-706006c`, it6 `...-it6-cap-b5cf567` /
  `...-dsv2-lite-p4-it6-b5cf567`, old supplement `...-dsv2-lite-p4-old-supp`,
  Qwen3-4B r3 `2026-10-23-qwen3-4b-g5-official-e9f3f33-r3` and it2d
  `2026-10-23-qwen3-4b-p3-it2d` (8a05b7c).

Every number below was read by me from `summary.json`, the vLLM logs, or the
probe dirs. To check the save path behind it6 I also read the base server's
save path (`pegaflow-core/src/offload.rs` `batch_save_kv_blocks_from_ipc` /
`build_insert_entries`, `storage/write_path.rs`). I read the old connector
only for the copying check. Neither is part of the onboarding work.

## Verdict: ACCEPT-WITH-FIXES

The connector change is correct. it6 (43a3759) fixes the G5 regression
without losing a save, and its tests fail on the pre-it6 code. DSV2 passes
every hard verdict on an official run.

P4 is accepted only after M1 and M2. M1: the Qwen3-4B regression run fails a
hard verdict (`G2.digest_self.cap`, 39 blocks over the limit). I find no
connector fault (Q2), but a reviewer cannot waive a hard gate. M2: the DSV2
0.25 gap is connector-attributable, and the explanation offered does not
hold on its own (Q3).

## MUST-FIX

- **M1. Qwen3-4B official run fails `G2.digest_self.cap`, so record one
  rerun.** The run is
  `2026-10-24-qwen3-4b-p4-it6-off-a1e7fd2` (verdicts `PegaKVConnectorV2`).
  - It has 4879 preempted-uncertain blocks against a limit of 4840 (1.008%;
    1.018% with the corrected denominator, N3). 0 are mismatched.
    `G5.replay_warm` fails with it.
  - Failure scenario: as recorded, the regression model has a failing hard
    correctness verdict on its only official run on the it6 tree. The
    PROGRESS P4 row says so.
  - Fix: one full official Qwen3-4B run on the accepted connector tree. Use
    a1e7fd2's connector content, or the M2 tree if M2 changes code. Record
    it next to MF5; do not replace MF5.
  - If the rerun passes, M1 is done. If it is again over 1% with 0
    mismatched, the next step is the gate owner's, not a connector reject:
    an allowance, or the `feat/rsi-gate-hybrid` remedy that digests
    preempted episodes in `handle_preemptions` (see Q2). A mismatched or
    unreferenced block would be a reject.
- **M2. DSV2 C/U 0.25 gap vs old: test the lever or fix the attribution,
  then record the result.**
  - Source: `docs/rsi/models/DeepSeek-V2-Lite-Chat/ITERATIONS.md:1919-1937`.
  - The same-server gap is connector-attributable: hit 0.436 vs 0.512, ext
    tokens 82,032 vs 327,152 (4x fewer), E_C 0.040 vs 0.161.
  - The judgment's explanation does not hold on its own. v2 keeps less C
    resident than old at 0.5 too (0.78-0.83 vs 0.92-0.95, a per-pass deficit of
    0.11 to 0.16), and there v2 matches old, native and H_lru exactly. At 0.25 the
    deficit is actually smaller (v2 0.78 / 0.90 / 0.83 / 0.86 vs old 0.92 /
    0.95 / 0.90 / 0.91, i.e. 0.05 to 0.14). So the resident deficit alone
    cannot cost 0.076 at 0.25.
  - The signature specific to 0.25 is a thrash. v2's server insertions grow
    pass over pass (74.5k / 83.8k / 88.1k / 94.9k vs old 66.3k / 61.1k /
    62.6k / 78.4k) while its warm ext hit falls (0.069 -> 0.050).
  - Fix, either one:
    - (a) One non-official cap-only A/B of the per-request-entry lever
      (`plan_saves`, one entry per (group, request)) on DSV2. Same gate
      ce645d7, base server, GPU 0,1, cap plan 76. Adopt it only if
      Qwen3-4B cap does not regress, and then fold it into the M1 run.
    - (b) An attribution that explains why the deficit costs nothing at 0.5
      and 0.076 at 0.25.
  - Hit rate is not gated, so acceptance does not depend on the A/B result,
    only on the judgment being explained and recorded.

## SHOULD-FIX

- **S1. Strengthen the MF5 note (ITERATIONS.md:1708-1837) with the evidence
  it omits.**
  - The old connector failed this same verdict on Qwen3-4B r3 (1.096%,
    0 mismatched; `...-g5-official-e9f3f33-r3` verdicts `old`).
  - native exceeded the limit on MF4 DSV2 (1.151%).
  - v2's own range is 0.41% (r3, 8a05b7c) to 1.01%.
  - Add the byte-level check from Q2. Note that v2 DSV2 MF4 passes at 0.929%,
    so the DSV2 margin is thin too.
- **S2. The final judgment should give the context of the 0.25 point.**
  - C(0.25) is 3.43 GB = 1.14x GPU KV for DSV2 and 1.22x for Qwen3-4B, below
    the 2x floor that makes C(0.5) an external-store measurement.
  - Old itself is 0.16 below H_lru at 0.25 on DSV2 (0.512 vs 0.673), so the
    server dominates that point.
  - v2's local hit is 0.015 to 0.02 below old/native in every cap run,
    including r3 and batching-off B. It is constant, not new; state it once
    instead of counting it in the 0.25 gap.
- **S3. The final hit-rate tables have no E_C for old and native.** They are
  at ITERATIONS.md:1888-1913 and MF4/MF5 G5. PROGRESS "PR policy" requires
  E_C. Values are in the tables below.
- **S4. Qwen3-4B's own record is stale.**
  - `docs/rsi/models/Qwen3-4B/ITERATIONS.md` has no MF2, MF5 or it6 entry;
    the regression runs and Qwen3-4B's hit-rate judgment exist only in the
    DSV2 file.
  - PROGRESS P3 row (`PROGRESS.md:143`) still says "hit-rate judgment pending
    on the base server".
  - Add a short Qwen3-4B entry that links the DSV2 sections, and update the
    P3 row.
- **S5. The PROGRESS accept row must carry the skill step 7 tables.** After
  M1 and M2, the P4 row (`PROGRESS.md:144`) needs local / ext / total, H*,
  E_ext, H*_C, H_lru, E_C and the G4 numbers, for v2 and the reference arms.
  Today it has G5 hit only.
- **S6. CI not observed.** No PR or Actions run exists for this branch on
  GentleCold/pegaflow (`gh pr list` / `gh run list` empty). Open the stacked
  PR and confirm it is green. The diff has no Rust files, so clippy and
  `cargo check` are inherited from the base. I did not run them (macOS, no
  CUDA).

## NITS

- **N1. it6 changes the server's insert order within a merged RPC.**
  - Per-step entries give the layers different hash lists, so the insert
    worker takes the hashed path (`offload.rs:81-85`, `:124-141`) instead
    of the ordered fast path. Pre-it6 folded calls and unbatched calls both
    took the ordered path.
  - LRU insertion order inside one RPC is then HashMap order, not block
    order.
  - At about 1.3 steps and about 28 blocks per RPC under pressure (MF4 cap
    0.5: 3828 RPCs, 1.29 save steps per RPC) this is likely negligible.
    Only a server change could restore the order. Worth one line in the it6
    notes.
- **N2. One unexplained 8.4 s finish-to-report hold.** MF4 cap 0.5 has
  `max_sent_wait_ms=8437.5` in the first save-stats window
  (`PegaKVConnectorV2/cap/vllm.log:403`). Every other window of every P4 run
  is at or below 1.9 s, and the stage mean is 186 ms. It coincides with an
  engine throughput dip (log 19:25:06). Not root-caused; record it.
- **N3. Gate denominator (for the gate owner, not this PR).**
  `run_gate.py:1566` adds `*_uncertain` on top of `*_compared`. But
  `digest_compare.py:242` already counts an uncertain block that has a copy
  in `compared`. The loaded count is inflated by the uncertain count:
  484,099 vs 479,221 actual, so the limit is slightly looser than stated.
- **N4.** The MF4/MF5 G3 tables have no explicit E_ext column. It is 1.0
  everywhere: v2 excess 0, old/native excess 1230 / 210 (DSV2) and
  1170 / 195 (Qwen).

## Answers

**Q1. it6 (`Worker._release_held`, worker.py:421-442): correct.**
- **Per-step entries are kept.** The change only concatenates
  `[s for j in jobs for s in j.saves]` in submit order. Each held job keeps
  its own `(layer, ids, hashes)` entries.
- **The sequence logic is unchanged.** The sent job takes the last job's
  `seq` and `ready`. The compute stream is in order, so the last mark covers
  every earlier step's forward. No block is read before its step's KV is
  written, and none is released early:
  - `poll` reports `finished_sending` only once `fence <= _completed`.
  - `_run_job` sets `_completed = job.seq` in `finally`, also on a failed
    compute wait or a failed RPC.
  - A dead saver settles everything (`_settled`, `poll` uses `_submitted`).
- **Preemption flush is unchanged.** `handle_preemptions` -> `flush()`
  releases held jobs first and then waits for `_submitted`.
- **Nothing lost or duplicated.**
  - Each step's (id, hash) pairs go out exactly once.
  - The server treats repeated layer names in one call as separate entries
    (one `LayerContext` and one pinned allocation each).
  - The same hash in two steps of one RPC seals once: the insert worker
    returns `Duplicate` for the second copy, and a resident block is
    dropped (`write_path.rs:144-149`, `:243-256`).
  - Run data: `failed=0` in every save-stats window of MF4/MF5. 0
    mismatched or unreferenced blocks in every digest of both runs.
- **The tests cover the regression.** With the pre-it6 `worker.py`
  (`git show 43a3759^:...`) in a HEAD export,
  `test_save_batching[steps|cap|cap-before-steps|steps-before-cap]`, all four
  `test_kv_pressure_sends_held_saves` cases and
  `test_merged_save_rpc_stays_under_size_budget` fail (9 failed, 3 passed).
  All pass on HEAD.

**Q2. MF5 `G2.digest_self.cap`: (b), one recorded rerun (M1). Not (c), and
not waivable by me as (a).**

What I verified, using the gate's own `digest_compare` (ce645d7) on the
cap(0.5) probe dirs (scripts in `~/kexi/tmp/review-p4-final/`):
- All 4879 uncertain blocks are `uncertain_preempted` (local 1519, ext
  3360). None is plain uncertain, unreferenced or mismatched.
- 4878 of them have a digested copy of the same key. Their closest
  per-layer rel-L1 to that copy matches ordinary non-invariant noise:

  | set | n | p50 | p99 | max | share > 0.0175 | share > 0.01 |
  |---|---|---|---|---|---|---|
  | v2 preempted-uncertain vs same-key copy | 4878 | 0.00095 | 0.0071 | 0.048 | 0.123% | 0.49% |
  | v2 computed vs other computed copy (honest) | 94996 | 0.00081 | 0.0071 | 0.068 | 0.107% | 0.42% |
  | same blocks vs the adjacent key's copy (wrong block) | 4878 | 0.061 | | min 0.0175 | | |

  A wrong-block load would sit at 0.0175 or more. Only 6 of 4878 do, as
  expected from the honest tail (about 5). old (4205, p99 0.0098) and
  native (2353, p99 0.0052) look the same.
- The share moves a lot from run to run for every arm:

  | run | v2 | old | native |
  |---|---|---|---|
  | r3 (8a05b7c) | 0.41% | 1.10% (fail) | 0.87% |
  | MF2 | 0.51% | 0.22% | |
  | A / B / it6-cap | 0.57% / 0.53% / 0.66% | | |
  | MF5 | 1.01% (fail) | 0.88% | 0.48% |
  | DSV2 MF1 / it6 / MF4 | 0.67% / 0.64% / 0.93% | MF4 0.50%, old-supp 0.71% | MF4 1.15% (fail) |

- `connector_v2` differs between d56fcb2 (MF2) and a1e7fd2 only in
  `_release_held`, so it6 cannot change which blocks are saved.

Hence not (c). But the verdict is hard and the run is official, so the
reviewer cannot record a pass. Rerun once (M1). The gate-side remedy is
`feat/rsi-gate-hybrid` digesting preempted episodes in
`handle_preemptions`: most of these blocks would then be judged instead of
uncertain. I recommend it to the gate owner. It is not counted as evidence
here.

**Q3. DSV2 G5 0.25 / 0.75: 0.75 is close enough; 0.25 needs M2.**
- **At 0.5 (gated):** v2 0.844 vs old 0.845, native 0.850, H_lru 0.847. Close
  enough.
- **At 0.75:** 0.026 below old and 0.056 below H_lru (old 0.027 below). This
  is within what I accept as close.
- **At 0.25:** 0.076 below old with 4x fewer external tokens. That is "差太多"
  unless explained, and the current explanation fails the 0.5 cross-check
  (M2).
- **Mitigating context (S2):** 0.25 is C = 1.14x GPU KV, below the floor,
  and the server already costs old 0.16 there.
- I do not require the lever to succeed, only to be tried, or the gap
  attributed, and recorded.

**Q4. Qwen3-4B G5: no regression.**

| C/U | MF5 v2 hit / E_C | r3 v2 (8a05b7c) hit / E_C | it6-cap v2 hit / E_C | MF5 old hit / E_C |
|---|---|---|---|---|
| 0.25 | 0.392 / 0.065 | 0.399 / 0.084 | 0.396 / 0.080 | 0.398 / 0.081 |
| 0.5 | 0.838 / 0.667 | 0.842 / 0.684 | 0.843 / 0.688 | 0.820 / 0.645 |
| 0.75 | 0.883 / 0.698 | 0.876 / 0.669 | 0.878 / 0.682 | 0.873 / 0.674 |

The differences vs 8a05b7c are at most 0.007 in hit and within the it6-cap
spread. v2 is at or above old at 0.5 and 0.75, and 0.006 below at 0.25.
Server insertions at 0.5: v2 93,389 vs old 98,334.

**Q5. G4: every required verdict passes for v2 in both runs.** See the
tables below.
- **DSV2:** warm TTFT p50 -67.6 / -66.6% and warm req/s +131 / +129% vs
  baseline; cold -34.1% p50, +60.7% req/s; gsm8k +0.15% cold, -2.26% warm.
- **Qwen3-4B:** warm -84.4 / -83.6% and +277 / +274%; cold -63.0% and
  +146.6%; gsm8k +0.37% and -0.76%.
- **Qwen3-4B vs it2d (last accepted, 8a05b7c):** no regression relative to
  the same-run baseline:
  - it2d: warm -77.9 / -78.9%, cold p50 -54.1%, cold req/s +105.5%.
  - In absolute terms MF5's cold p50 is 1.377 vs 1.287 s and cold req/s
    6.26 vs 6.70. The baseline moved more over the same span (cold p50
    2.804 -> 3.721, req/s 3.26 -> 2.54: different GPU and pool size).
  - The comparison basis is therefore same-run relative. The doc's MF5 G4
    table already uses it.

**Q6. split_invariant = false (DSV2): handled.**
- **Split control:** gate.json `split_invariant: false`, and
  `G2.split_control` measured 647 / 1319 differing, which matches the claim.
- **Outputs (v2 `G2.gsm8k_outputs`):** cold_differ 0, warm_differ 647 (vs
  baseline, expected), warm_differ_vs_split 0, warm_differ_vs_split_aligned
  0. old has 38 aligned differences, reported only.
- **Digests:**
  - `digest_vs_baseline.gsm8k`: 45008 / 45008 computed exact, coverage 1.0
    in the cold and warm passes, 10390 ext matched.
  - `digest_self`: replay 27232 ext matched.
  - Cap: 512,214 ext and 567,136 local matched, 10,214 preempted-uncertain
    (0.93%), 0 mismatched.
- **Accuracy:** 0.6725 baseline, cold delta 0.0 pp, warm +0.53 pp.

**Q7. Policy complexity: compliant.**
- `policy.py` is 119 lines. A grep for model or spec-class names (Qwen,
  DeepSeek, MLA, Mamba, *Spec, ...) finds none.
- 4 knobs (`loading_wait_ms`, `sync_load_tokens`, `save_batch_steps`,
  `save_batch_max_kv_usage`), all read in `core/scheduler.py` or
  `connector.py`, each with gate evidence (P3, it4/it5).
- No per-model toml exists (`models/default.toml` only; vllm.log: "no model
  profile DeepseekV2ForCausalLM.toml, using defaults").
- it6 added no knob.

**Q8. Isolation: clean.**
- `summary.scan` has `flagged: []` and `allowlisted: []` in both runs.
- My grep of `connector_v2` for RsiProbe, RsiWorkerExt, rsi_poison,
  sys.modules, probe, rsi-runs, outputs and requests.jsonl finds only
  `hashlib...hexdigest()` (layout.py:165, the namespace hash). There are no
  nixl, P2pNccl, OffloadingConnector or old-connector imports.
- No server or gate file is in the diff.
- MLA replica collapse:
  - The old connector also collapses MLA TP ranks, but by a different
    mechanism and different names (`effective_tp_rank` / `collapse_mla_tp`
    keyed on an MLA flag).
  - v2 derives it generically from `seal_plan` (`StorageRanks`,
    `Shard.replicated()`), from vLLM's `max_tp_shards == 1`.
  - That is the same conclusion from the same vLLM fact, not copying.

**Q9. Provenance: verified.**
- Both runs: `official: true`, `not_official_because: []`, contamination
  `[]`, 20 provenance_watch checks with no change.
- Gate rev ce645d7. Server sha256 `d1970ead...c579d7` (claims 056eb4b).
- CUDA 13.0 env: CUDA_HOME `~/kexi/cuda13`, LD_LIBRARY_PATH
  `~/kexi/cuda13-lib`, nvcc 13.0.
- `vllm_plugins`: old `pegaflow`, others empty.
- Under test: `under_test_rev` a1e7fd2 (post-reword; not in the sha map,
  which only lists rewritten shas).
  - I hashed all 64 non-`.so` files of each run's `under-test-overlay` with
    `git hash-object`. Both runs are identical to each other and to the
    local `git ls-tree a1e7fd2 python/pegaflow`.
  - a1e7fd2..4f7cb61 touches docs only.

**Q10. CI (local).**
- Passing:
  - `/tmp/ci-venv/bin/typos`: clean.
  - `cz check --rev-range 1749353..HEAD`: successful, including this
    review's commit.
  - `ruff check python`: all passed.
  - `ruff format --check` on connector_v2 and the test file: clean.
  - Default Python gate (source-only venv: pytest, numpy, requests):
    473 passed, 60 skipped.
- Not run: clippy and `cargo check` (no Rust diff). No remote CI exists (S6).

**Q11. Docs: mostly accurate; fixes in S3 to S5.**
- The judgment tables have v2 / old / native / lmcache (pending) with H_lru
  and H*_C.
- Every number I cross-checked matches `summary.json`: all G5 cells, the
  resident/C ranges, insertions, the G3 and G4 tables, and the MF5 note's
  1519 / 3360 split.
- Missing: E_C for the reference arms (S3), Qwen3-4B's own record (S4), and
  the accept-row tables (S5).
- The PROGRESS P4 row is accurate for "final review pending".

## Hit-rate tables (from summary.json)

G5, `warm_timed` (H*_C and H_lru from the v2 arm's trace; the reference arms
differ by at most 0.006).

DSV2 (MF4, cap 76 x 12, C(0.5) = 6.87 GB = 2.27x GPU KV):

| C/U | v2 hit | old hit | native hit | lmcache | H*_C | H_lru | v2 E_C | old E_C | native E_C | resident/C v2 / old (4 passes) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.436 | 0.512 | 0.578 | pending | 0.851 | 0.673 | 0.040 | 0.161 | 0.271 | 0.78-0.90 / 0.90-0.95 |
| 0.5 | 0.844 | 0.845 | 0.850 | pending | 0.925 | 0.847 | 0.615 | 0.621 | 0.624 | 0.78-0.83 / 0.92-0.95 |
| 0.75 | 0.863 | 0.889 | 0.933 | pending | 0.968 | 0.919 | 0.581 | 0.632 | 0.712 | 0.79-0.82 / 0.94 |

Qwen3-4B (MF5, cap 64 x 12, C(0.5) = 27.2 GB = 2.44x GPU KV):

| C/U | v2 hit | old hit | native hit | lmcache | H*_C | H_lru | v2 E_C | old E_C | native E_C | resident/C v2 / old |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.392 | 0.398 | 0.542 | pending | 0.862 | 0.581 | 0.065 | 0.081 | 0.308 | 0.74-0.85 / 0.81-0.93 |
| 0.5 | 0.838 | 0.820 | 0.861 | pending | 0.929 | 0.857 | 0.667 | 0.645 | 0.718 | 0.75-0.82 / 0.76-0.84 |
| 0.75 | 0.883 | 0.873 | 0.927 | pending | 0.970 | 0.915 | 0.698 | 0.674 | 0.767 | 0.71-0.83 / 0.73-0.79 |

Cap stages: unmatched requests 0 and load errors 0 for every arm. v2 excess
is 0 everywhere. old excess is at most 330, native at most 540.

G3, unlimited store (fractions of prompt tokens; E_ext = `efficiency` from
`summary.results.<arm>.<wl>.hit`):

| model | pass | v2 local / ext / total | old local / ext / total | native local / ext / total | H* | E_ext v2 / old / native | excess v2 / old / native |
|---|---|---|---|---|---|---|---|
| DSV2 | gsm8k warm | 0.938 / 0.054 / 0.993 | 0.938 / 0.055 / 0.993 | 0.940 / 0.053 / 0.993 | 0.993 | 1.0 / 1.0 / 1.0 | 0 / 1230 / 1230 |
| DSV2 | replay cold | 0.497 / 0.345 / 0.843 | 0.500 / 0.343 / 0.843 | 0.511 / 0.332 / 0.842 | 0.843 | 1.0 / 1.0 / 0.9998 | 0 / 0 / 0 |
| DSV2 | replay warm2 | 0.487 / 0.511 / 0.998 | 0.508 / 0.490 / 0.998 | 0.467 / 0.532 / 0.998 | 0.998 | 1.0 / 1.0 / 1.0 | 0 / 210 / 210 |
| DSV2 | replay warm3 | 0.505 / 0.493 / 0.998 | 0.518 / 0.481 / 0.998 | 0.467 / 0.531 / 0.998 | 0.998 | 1.0 / 1.0 / 1.0 | 0 / 210 / 210 |
| Qwen3-4B | gsm8k warm | 0.912 / 0.081 / 0.992 | 0.927 / 0.067 / 0.993 | 0.933 / 0.060 / 0.993 | 0.992 | 1.0 / 1.0 / 1.0 | 0 / 1170 / 1170 |
| Qwen3-4B | replay cold | 0.379 / 0.486 / 0.864 | 0.382 / 0.482 / 0.864 | 0.384 / 0.480 / 0.864 | 0.864 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 |
| Qwen3-4B | replay warm2 | 0.347 / 0.651 / 0.998 | 0.345 / 0.653 / 0.998 | 0.378 / 0.621 / 0.998 | 0.998 | 1.0 / 1.0 / 1.0 | 0 / 195 / 195 |
| Qwen3-4B | replay warm3 | 0.348 / 0.651 / 0.998 | 0.360 / 0.638 / 0.998 | 0.378 / 0.621 / 0.998 | 0.998 | 1.0 / 1.0 / 1.0 | 0 / 195 / 195 |

`G3.no_eviction` passes for v2 and old in both runs (native has no
PegaFlow server).

## G4 tables (same-run baseline; warm = warm2 / warm3)

DSV2 (MF4):

| metric | baseline | v2 | old | native |
|---|---|---|---|---|
| replay warm TTFT p50 (s) | 0.923 / 0.926 | 0.299 / 0.310 (-67.6 / -66.6%) | 0.344 / 0.334 (-62.7 / -63.9%) | 0.279 / 0.290 (-69.8 / -68.7%) |
| replay warm req/s | 5.41 / 5.41 | 12.48 / 12.38 (+131 / +129%) | 12.35 / 12.37 (+128 / +129%) | 13.24 / 13.19 (+145 / +144%) |
| replay cold TTFT p50 (s) | 0.909 | 0.599 (-34.1%) | 0.630 (-30.7%) | 0.557 (-38.7%) |
| replay cold req/s | 5.37 | 8.63 (+60.7%) | 8.49 (+58.1%) | 8.77 (+63.2%) |
| gsm8k cold (s) | 193.9 | 194.2 (+0.15%) | 196.0 (+1.08%) | 195.2 (+0.67%) |
| gsm8k warm (s) | 193.8 | 189.4 (-2.26%) | 192.0 (-0.94%) | 190.9 (-1.50%) |
| G4 verdicts | | all pass | all pass | all pass |

Qwen3-4B (MF5):

| metric | baseline | v2 | old | native |
|---|---|---|---|---|
| replay warm TTFT p50 (s) | 3.563 / 3.504 | 0.555 / 0.574 (-84.4 / -83.6%) | 0.644 / 0.661 (-81.9 / -81.1%) | 0.455 / 0.504 (-87.2 / -85.6%) |
| replay warm req/s | 2.55 / 2.56 | 9.63 / 9.58 (+277 / +274%) | 9.99 / 10.18 (+291 / +297%) | 9.62 / 9.60 (+277 / +274%) |
| replay cold TTFT p50 (s) | 3.721 | 1.377 (-63.0%) | 1.384 (-62.8%) | 1.420 (-61.8%) |
| replay cold req/s | 2.54 | 6.26 (+146.6%) | 5.56 (+119.1%) | 5.93 (+133.7%) |
| gsm8k cold (s) | 170.2 | 170.8 (+0.37%) | 172.1 (+1.12%) | 172.2 (+1.20%) |
| gsm8k warm (s) | 171.5 | 170.2 (-0.76%) | 174.9 (+2.01%, fail) | 180.4 (+5.22%, fail) |
| G4 verdicts | | all pass | gsm8k warm fails | gsm8k warm fails |

Accuracy: DSV2 0.6725 in every cold pass, warm 0.6778 (v2) / 0.6770 (old,
native). Qwen3-4B 0.8582 in every pass of every arm.
