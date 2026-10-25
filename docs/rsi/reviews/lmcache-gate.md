# lmcache gate review (feat/rsi-gate-lmcache)

Scope: `d252bfe...origin/feat/rsi-gate-lmcache` at a30a27d (89f5a19..a30a27d:
lmcache arm by module path, `--lmcache-chunk-size`, b05e507 resume
references, docs). Line numbers are at a30a27d unless marked f8eac30. Host
evidence was read only, under `~/kexi/` on .44; scratch scripts and the two
tree exports are under `~/kexi/tmp/review-lmcache/`. No GPU run.

Verdict: **ACCEPT-WITH-FIXES** (M1, M2).

I found no way for a connector under test to fool the gate through b05e507.
The resume replay is a baseline stage. It never sees the arm's bytes, only
the arm's resume points. Wrong loaded bytes are still judged against honest
computed copies. The context fingerprint makes the computed-block key finer,
so it can only move a block from "judged" to "unjudged", and unjudged warm
blocks still fail closed through coverage and S7. Nothing changes for
split-invariant models: Qwen3-4B verdicts re-scored with the branch are
identical to the base tree. Every run-data claim I checked holds, with one
exception. The claim that the official DSV2 run "still passes" is only true
for the digest checks: under the branch's `verdicts()` every DSV2 arm fails
the new `G2.resume_reference` (M2). The merge with f8eac30 has 8 conflict
hunks. Taking one side naively drops the hybrid's zero-digest rule (M1).

## MUST-FIX

### M1. The merge with f8eac30 must keep the hybrid's reference rules inside the pooled Reference

`digest_compare.py` conflicts in `records()` and `Reference.__init__`. The
lmcache side moves the copy loop into `Reference._add` (one call per probe
dir). The hybrid side adds the R2 zero-digest skip, `(kind, key)` counters,
`state_groups`, per-episode views and the `episode` field inside the old
`__init__` loop. Resolving the hunk to "theirs" gives a `_add` without
`is_zero`. Then an all-zero computed copy becomes a reference again, and a
load that wrote nothing matches it. That is hybrid R2, a real way to fool the
gate. The auto-merged test file keeps the hybrid's zero tests, which should
catch it, but only if they run on the merged tree.

Fix (the merge plan below lists every hunk): keep R2 in `_add`, sum
`zero_copies` over dirs, union `state_groups` over dirs, and compute the
fingerprint per `(req_id, rank, episode)` from the records of that episode
(`covered <= that episode's start`), not from the last episode. Run the
merged `rsi/gates` pytest, then re-score the recorded Qwen3.5 / Qwen3-4B /
DSV2 runs with the merged tree before pushing the merge.

### M2. Existing official DSV2 records fail the new gate; the "still pass" claim covers only the digests

GATES.md:278 and the brief say the official DSV2 run
`2026-10-24-dsv2-lite-p4-it6-off-a1e7fd2` re-scores to pass under
`b05e507`'s key. That is true for `G2.digest_vs_baseline.gsm8k` and the
coverage. I re-scored it with both trees: identical. But
`run_gate.verdicts()` at a30a27d on the same record adds
`G2.resume_reference = {"pass": false, "reason": "no baseline replay of this
arm's resume points"}` to old, native and connector_v2 (run_gate.py:2082).
So every arm fails, including the required v2 arm. This is fail-closed and
correct for a record without the stage. But it means DSV2's current
acceptance (it6 / it7, gate ce645d75) does not stand under the merged gate.

Fix: say so in GATES.md:278 and PROGRESS.md, and rerun DSV2 official under
the merged gate before DSV2 is called accepted on it. The rerun needs a GPU,
TP2. The arm's warm stage takes about 200 s; the replay of lmcache's 1319
prompts took 200 s in 1104 waves.

## SHOULD-FIX

### S1. LMCache capacity is per process: TP > 1 without MLA would get `tp x C`

`lmcache_env` (run_gate.py:730) gives every worker
`LMCACHE_MAX_LOCAL_CPU_SIZE = C`. On DSV2 (MLA, TP2) this is fair. The log
shows `save only first rank: True` and one `Created backend:
LocalCPUBackend` for two `Creating LMCacheEngine` lines, so total = C =
6865026048 B, one copy, the same convention as native and PegaFlow. On a
non-MLA model at TP > 1, each rank creates its own backend of C, so the arm
gets `tp x C`. No current model hits this. Fix: size each rank as
`C / storing ranks`, or reject lmcache on non-MLA `tp > 1` until that is
done, and add a unit case.

### S2. Nothing enforces which lmcache build a model's official run uses

The table needs PR #4467 on Qwen3-4B (0.5.6 is wrong KV there) and 0.5.6 on
DSV2 (#4467 crashes on MLA). The default `--lmcache-overlay` is the 0.5.6
overlay (run_gate.py:719). SKILL.md says to always pass `lmcache-pr4467`,
which crashes on DSV2. `official_problems` checks only the chunk size
(:2464). The arm is record-only, and a wrong build fails its own digests, so
this cannot change a required verdict. It can put a wrong reference number
in a table, or crash it. Fix: put the expected dist per model (gate.json or
the status table), check `result.lmcache.dist` against it in
`official_problems`, and make SKILL.md name the per-model overlay.

### S3. DSV2 lmcache status: the doc and the orchestrator's label disagree

GATES.md:176 / :276 say "not comparable yet". The brief says the
orchestrator labelled it "comparable with caveat". Pick one. My
recommendation is below (status table): "comparable with caveat" for its
G3 / G5 numbers. Every digest check that backs those numbers passes with
0 mismatched. The one failing check (`G2.digest_vs_baseline.gsm8k`, S7,
14 blocks of 1 request) is a reference gap I reproduced (Verified 5), not a
wrong byte. The rule stays unchanged and the row must say that check fails.

### S4. Tests stop at helpers; add verdict-level and adversarial-composition cases

The new tests (test_oracle_capacity.py:1972, :1988, :2017, :2039) cover the
key, pooling, plan and differ helpers, and each would fail on a lying
helper. Missing:

- `verdicts()` on a split-variant record: without a `resume-<label>` stage
  the arm fails `G2.resume_reference`; with one, REF is pooled and
  `resume_differ` is used.
- Split-invariant: no `G2.resume_reference` key, and `needs_resume` is
  false (:2904).
- End to end: a warm computed block with wrong bytes, on a non-aligned
  prompt, over an honest but other-split context. It ends `unmatched_split`,
  and the verdict must fail through S7 (`unjudged_nonaligned`). This is the
  one path where the finer key turns "inexact" into "unjudged".

### S5. Self-test with the resume stage on DSV2 (GPU)

Offline, the 2026-10-23 DSV2 self-test negatives are still caught with the
new key: liar / bitflip / offbyone fail both digest checks with
2168 / 256 / 828 external mismatched, the same as the base tree. That run
predates the resume stage, so the honest arm with pooled REF and the
negatives with a resume stage have not run. That needs a GPU run (TP2,
128 questions per arm). Also add `G2.resume_reference` to the honest arm's
required list when `split_invariant=false`.

### S6. A preempted warm request is replayed only at its last episode

`resume_plan` (run_gate.py:1066-1069) takes `episodes[-1]`. After the merge,
the hybrid digests preempted episodes too. On a split-variant model, an
honest arm's earlier-episode computed blocks then have no same-split
reference: unjudged, and a fail on non-aligned prompts. This fails closed,
but honest arms fail. Document it, or replay each episode's start.

## NITS

- N1. GATES.md:827: "a resume stage that crashes fails that arm's
  `G2.resume_reference`, not the baseline" holds for record-only arms. For
  the arm under test, `run_stage(record_only=False)` re-raises (:2753) and
  the run aborts.
- N2. GATES.md:194-197: of the 738 mismatched external blocks at cap075,
  722 equal another key's computed copy; 16 equal no computed copy.
- N3. `result.lmcache.dist` (:757) lists the overlay's dist-info, not the
  module vLLM imported. Log `lmcache.__file__` / version from the worker.
- N4. The #4467 source checkout has one modified file, a generated
  `common_pb2.pyi` stub with no runtime effect. The overlay's `.py` tree
  equals the checkout at e05c2ea. "Unmodified" is right for runtime code.
- N5. `2026-10-24-qwen3-4b-lmcache-ref-1afec5a` predates the chunk flag, so
  its `result.lmcache.env` has no `LMCACHE_CHUNK_SIZE`. LMCache's own log
  says `chunk_size: 256` in all 5 stages.
- N6. The aligned-prompt exemption still lets up to 10% of warm computed
  blocks go unjudged when the arm resumes where vLLM's prefix cache cannot
  (`start = L - 1` on `L % 16 == 0`). This predates the branch (it was the
  same against `split`) and is bounded: reused bytes are judged as local.
  Noted only.

## Adversarial analysis of b05e507

- Isolation. The replay is a fresh `baseline` stage (`kind == "baseline"`
  enforced at run_gate.py:1132-1133), with no connector. From the arm it
  reads only `requests.jsonl` (prompt index, `[step, local, start]`,
  `prompt_len`), after `untrusted_problems` has rejected symlinks in the
  arm's dir. A forged trace can only make the honest baseline compute at
  other points. Forged digests are the pre-existing M7 residual.
- Shaping resume points. The arm picks `start` through its claimed hits. The
  replay then computes honest copies at that split, so more of the arm's
  computed blocks are judged, not fewer. A wrong loaded block is
  `external_mismatched` (batch invariant: any honest copy of the key
  decides), whatever the split. Its fingerprint then also differs, so the
  blocks computed on top of it are unjudged, but the load has already
  failed the check.
- Can the fingerprint let a wrong block match? It only adds a component to
  the key (digest_compare.py:140-146). A match still needs exact equality
  with an honest copy at that key. Pooled copies are all honest. The finer
  key turns some "compared, inexact" into `unmatched_split`. That is the
  only loosening, and it is bounded: warm needs coverage >= 0.9 and 0
  unjudged on non-aligned prompts (S7), cold needs coverage >= 0.9, and
  every context byte is judged as a local / external load. S4 asks for the
  end-to-end test.
- Unjudged counts fail closed: start_differs / missing are reported, their
  blocks are unjudged (coverage, S7), and their completions count as
  differing unless the prompt is aligned (`resume_differ`, :1589-1615).
  Plan problems fail `G2.resume_reference` and `resume_differ` returns None,
  which fails `G2.gsm8k_outputs`.
- Completions as reference are used only when `resumed`, which requires
  `split_invariant=false` (:2075-2086). Split-invariant models keep
  baseline cold.

## Verified

1. CI in the worktree: `/tmp/ci-venv/bin/typos` clean; `cz check --rev-range
   1749353..origin/feat/rsi-gate-lmcache`: successful.
2. Gate pytest on a full `git archive` of a30a27d
   (`~/kexi/tmp/review-lmcache/tree`): `185 passed in 44.29s`.
3. Split-invariant unchanged: `rescore.py` on
   `2026-10-24-qwen3-4b-p4-it6-off-a1e7fd2` with the d252bfe and a30a27d
   trees gives identical `G2.digest_vs_baseline.gsm8k` for old (22127/22127
   exact, 6058 ext matched), v2 (22129/22129, 7423) and native (fails by
   layout, as recorded).
4. DSV2 official it6 re-score: digests and coverage identical on both trees
   (v2 computed 45008/45008, warm 17482/17482; old / native 43836/43836,
   warm 16310/17492, 0 non-aligned). Full `verdicts()` at a30a27d adds a
   failing `G2.resume_reference` to all three arms (M2). (`G0.spec` also
   failed in my script because I passed no committed spec: a script
   artifact.) The baseline's own split-variant digest_self is identical on
   both trees.
5. DSV2 lmcache full run `2026-10-24-dsv2-lite-lmcache-ref-b05e507`:
   `G2.resume_reference` 1319 planned, 1104 waves, 0 missing / start_differs
   / prompt_len_differs; `G2.digest_vs_baseline.gsm8k` local
   358496/358496, ext 152/152, computed 55026/55026 exact,
   `computed_unmatched_split` 14, warm `unjudged_nonaligned` 14 (cold
   27526/27526). S7 request conv 1077 (prompt 1284 tokens): arm episode
   `[8089, 1088, 1280]`; its 12 loaded blocks (covered 1104-1280) equal the
   baseline cold copies and differ from the replay's prewarm (computed from
   1088, prefill ending at 1280). Its 4 local blocks (1040-1088) equal the
   replay's and differ from baseline cold's. So no reference shares its
   context: fingerprints `a5df8edf` (arm), `5cb03832` (base cold / warm),
   `ad724db7` (base split), `ba92cfdf` (replay). Accuracy cold 0.6725 =
   baseline, warm 0.6717; outputs 0 differ from the replay. Cap / replay
   digest_self pass at every fraction. Capacity: `cpu_bytes` 6865026048 =
   `capacity_bytes`; one `LocalCPUBackend` (save only first rank).
   Of 1319 warm requests 1317 resumed at their local hit with no load and
   2 loaded; for comparison, v2 loaded on 1319/1319 in it6.
6. Qwen3-4B #4467 chunk 256 (`...-ref-1afec5a`): gsm8k ext 64/64, replay
   14971 matched; cap 0.5 passes (0 mismatched, 4121 preempted-uncertain);
   cap075: 738 ext mismatched in 2 requests (`cap_warm` 43/5: 417,
   `cap_warm2` 0/6: 321), every layer (first_bad layers 0-35), 0 load errors.
   Cap plan and U (64 convs, 54433677312 B) equal it6 / it7 official.
7. native Qwen3-4B cap 0.5: 5140 uncertain, all preempted (4138 ext +
   1002 local), 0 mismatched, over 494121 loaded = 1.040% > 1%.
8. chunk 16 (`...-chunk16-91fbc41`): 19 ext mismatched in 4 requests, one
   digest, identical across all 36 layers (poison); warm -4.69 pp.
9. 0.5.6 on Qwen3-4B (`repro056`, `056-dynamic`): ext 0/64 matched, warm
   -85.9 pp. #4467 on DSV2 (`...-pr4467-slice`): `RuntimeError: SPLIT_KV_2LTD
   requires a [2, num_layers, num_tokens, num_heads * HS] buffer`, HTTP 500.
10. Overlay provenance: `~/kexi/tmp/lmcache-src` HEAD
    e05c2ea9781b4021081f34dfb982189ec0fabc15; overlay dist
    `lmcache-0.5.7.dev0+pr4467.e05c2ea`; `diff -rq` of `.py` sources: none.
    PR #4467 is open (last updated 2026-10-22, head e05c2ea): pinning an
    unmerged upstream PR is the user's call at final acceptance, as
    GATES.md says.
11. Merge: `git merge-tree` and a scratch merge in `/tmp/lmc-merge`
    (outside the main checkout).

## lmcache status per model (endorsed)

| model | lmcache | chunk | status | evidence |
|---|---|---|---|---|
| Qwen3-4B | 0.5.6 | 256 | not comparable: wrong KV | ext 0/64 matched, warm -85.9 pp (repro056, 056-dynamic) |
| Qwen3-4B | #4467 e05c2ea | 256 | comparable at gsm8k, replay, cap 0.25 / 0.5; cap 0.75 excluded (silent wrong KV under pressure, report upstream) | ref-1afec5a: E_C(0.5) 0.690 (H 0.842, H*_C 0.929, H_lru 0.857); native 0.723 |
| Qwen3-4B | #4467 | 16 | not comparable: claimed hits never loaded | 19 poison blocks, warm -4.69 pp |
| DeepSeek-V2-Lite (TP2, MLA) | #4467 | 256 | not comparable: crashes | SPLIT_KV_2LTD RuntimeError |
| DeepSeek-V2-Lite | 0.5.6 | 256 | comparable with caveat (record only): G3 / G5 numbers usable; `G2.digest_vs_baseline.gsm8k` fails S7 on 14 blocks of 1 request (reference gap, verified), rule unchanged | ref-b05e507: E_C(0.5) 0.616 (H*_C 0.924, H_lru 0.847); v2 0.615, old 0.621, native 0.624 (it6) |
| Qwen3.5-4B | - | - | not run | - |

## Merge plan with f8eac30

Conflicts (scratch merge, 8 hunks; the test file auto-merges):

- `rsi/gates/digest_compare.py` (2). `records()`: keep the hybrid's
  per-episode views, `episode` field and 7-tuple, and add the fingerprint
  per `(req_id, rank, episode)` from that episode's records
  (`covered <= start`). `Reference`: keep `_add(probe_dir)` per dir, with R2
  `is_zero` skip, `zero_copies +=`, `state_groups |=`, `(kind, key)`
  counters, `preempt_digested` / `preempt_dropped` inside it (M1). Merge
  the module docstrings.
- `rsi/gates/run_gate.py` (4). Arms docstring: lmcache line from lmc, self-test
  line from HEAD. `run_arm_workload`: keep both `pre` and `resume` flags and
  the resume `ValueError`; the names block takes HEAD's plain / baseline /
  `ALONE_REF` lists plus lmc's `elif resume:` branch next to `elif pre:`.
  `verdicts`: keep HEAD's `G2.preempt` / `gsm8k_accuracy_verdict` /
  `alone_ref_differ` and lmc's `warm_reference`.
- `rsi/gates/workloads.py` (1): `body = _gsm8k_body(...)`, then HEAD's
  `free_run` override.
- `docs/rsi/GATES.md` (1): keep both (lmcache overlay section, then the
  hybrid self-test paragraph).

Semantic items that do not conflict but must be handled in the merge:

- The resume stage must get `gsm8k_vllm_args` as the gsm8k stage does:
  f8eac30 passes `gsm8k=gsm` to `vllm_server` (f8eac30 run_gate.py:1243),
  and `G0.vllm_args` requires them only for `wl == "gsm8k"` (:1598). Use
  `gsm or resume` in both. Otherwise a split-variant model with
  `gsm8k_vllm_args` replays under other compile flags and fails closed (no
  DSV2 gate.json has them today).
- Honest self-test on split-variant models: add `G2.resume_reference` to
  `need` (S5). Per-group negatives (`caught_per_group`) use `vs_base`, which
  is then pooled. Loaded-byte mismatches do not depend on the pool, so
  negatives stay caught.
- `alone_ref` (non-batch-invariant models) is not a `warm` pass, so
  `resume_plan` ignores it. `rsi_conv` on gsm8k requests affects nothing
  else (`_pos` needs `turn`).
- Preempted warm requests (S6).
- After resolving: merged pytest, typos, then re-score Qwen3-4B it6/it7,
  the Qwen3.5 selftest / it2 records and DSV2 it6 / lmcache-ref with the
  merged tree. Expect identical verdicts except the new failing
  `G2.resume_reference` on DSV2 records without the stage (M2).
