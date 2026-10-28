# RSI gates (human-owned)

Agents may add checks or tighten thresholds; only the human gate owner may
remove or relax one. A change needs the review agent's sign-off and a line in
`rsi-notes:PROGRESS.md`. Evidence behind the rules (probe numbers, run tags,
the lmcache overlay build) is in `rsi-notes:gates-evidence.md`. Status: draft
until the user's final acceptance. Tooling under `rsi/gates/` is gate-owned
like this file: onboarding agents run it, they do not edit it.

A model is supported only when every hard gate passes on it and no earlier
model regresses.

## Overview

| Gate | What | Gated |
|---|---|---|
| G0 Layout | committed spec dump; adapter layout vs dump; provenance, vLLM args, lmcache build | hard |
| G1 Semantics vectors | `cargo test -p pegaflow-semantics`; each new Kind/Shard ships vectors for resolve / saveable / needed / seal_plan | hard (outside `run_gate`'s exit code) |
| G2 Correctness | byte digests of every loaded block, output identity, gsm8k accuracy, controls | hard |
| G3 Hit rate, unlimited store | `H`, `H*`, `E_ext`, excess, no eviction | excess, load errors, no eviction hard; levels reported |
| G4 Performance | TTFT, req/s, wall time vs the in-run `native` | gsm8k overhead hard; replay cold and warm reported and judged |
| G5 Capacity hit rate | `E_C` at a fixed store size `C` | correctness, load proof, run validity hard; `E_C` reported |

G2 also requires, outside `run_gate`, greedy output identity with baseline vLLM
on the e2e prompt set (`python/tests/test_vllm_e2e_correctness.py`), cold and
warm. gsm8k in `run_gate`: 8-shot CoT, full test split (1319), greedy,
temperature 0, baseline / cold (PegaFlow saves) / warm (GPU prefix cache reset
first); `|warm - baseline|` and `|cold - baseline| <= 1.0pp`.

## Hit rate

No hit-rate level is a hard gate. The agent iterates until it judges that no
connector-side headroom is left, with every remaining gap attributed. G4
replay vs `native` is judged the same way (G4, "Perf vs native judgment").
"Close enough" is not a stopping point.

- Reported, never gated: `H`, `H*`, `E_ext`, `H*_C`, `H_lru`, `H_freeze`,
  `E_C`, oracle hit rate, per-group attribution. Each verdict carries a
  `reported` block (value, `reference`, `below`); `summary.hit_report` (the
  `[gate] hit rates (reported, not gated)` line) lists them per arm.
  Reference bars: G2 warm external hit > 0.9 of oracle, G3 `E_ext` 0.95 and
  oracle hit rate 0.5, G5 `E_C` 0.9; `H` / `E_C` vs the last accepted
  iteration.
- Hard: load errors; mismatched / unreferenced blocks; digest coverage
  (`MIN_JUDGED_*`, `MIN_COMPUTED_COVERAGE`, `MIN_PREWARMED`); excess within
  one block per request; `G3.no_eviction`; cap pool and load proofs
  (`pool_ok`, `loads_ok`, `unmatched_requests == 0`); G5 capacity floor and
  freeze separation (run validity); every G0 / G2 / G4 check (except
  `*_vs_native`, reported).
- Hit-rate judgment, per accepted run, in
  `rsi-notes:models/<model>/ITERATIONS.md`: measured vs `H*` / `H*_C` and the
  reference arms (`H_lru`, `native`; `old`, `lmcache` when run), every gap explained
  (Kind limits, retention, eviction, per-group misses) and split into
  connector-side headroom, server (GATES.md "Known server limits"), Kind /
  vLLM structure and noise; the connector-side hypotheses tried (with
  results) and why each remaining one is not worth trying, with numbers.
  `H*_C` is a ceiling, not a target. A warm gsm8k pass that loads little says little about loaded
  KV; say so. The review agent checks the numbers against
  `summary.hit_report` and the explanation against the attribution, and can
  reject the run on it.

## Verdicts

Constants (`run_gate.py`): `ACC_TOL` 0.01, `MIN_JUDGED_GSM8K` 0.9,
`MIN_COMPUTED_COVERAGE` 0.9, `MIN_PREWARMED` 0.9, `MAX_OVERHEAD` 0.02,
`MAX_PREEMPTED_UNCERTAIN` 0.01, `CAPACITY_FRACTION` 0.5, `MIN_CAPACITY_GPU_KV`
2.0, `FREEZE_SEPARATION` 0.9, `NATIVE_MIN_EFFICIENCY` 0.9, `CONTAMINATION_P99`
10.

Arms column: "every arm" = every arm except `plain` and `baseline` (store
arms and self-test arms); "store arms" = `native`, the arm under test, and
`old` / `lmcache` when run. Self-test arms get no G3 / G4 / G5 verdict;
`plain` and `baseline` get no G3 / G4 / G5 verdict either. "Onboarding" =
the verdict needs the `plain` arm: without it the verdict is `{"pass": true,
"na": "onboarding control: ..."}` and the run is official only with a
current gate.json `controls` record of `--tp` ("Onboarding controls").

| Verdict | Arms | Passes iff | Section |
|---|---|---|---|
| `G0.spec` | every probed arm | spec dump and layer view layouts equal baseline's; baseline's equal the committed dump of `--tp` (G0) | G0 |
| `G0.vllm_args` | baseline verdicts | every stage's `vllm_cmd` carries each gate.json flag as its last occurrence | Per-model settings |
| `G0.provenance` | every arm, baseline | spec dumps' module files resolve under the expected roots | Provenance |
| `G0.lmcache_build` | `lmcache` | every stage's `result.lmcache.dist` = gate.json `lmcache_dist` | lmcache |
| `G2.plain_equals_baseline` | baseline, onboarding | plain and baseline gsm8k cold completions identical | G2 controls |
| `G2.warm_equals_cold` | baseline | warm completions = cold (gated under batch invariance) | G2 controls |
| `G2.split_control` | baseline, onboarding | measured split invariance = gate.json `split_invariant` | Split invariance |
| `G2.batch_control` | baseline, onboarding | measured batch invariance = gate.json `batch_invariant` | Batch invariance |
| `G2.digest_self.gsm8k` | baseline, every arm | digest verdict of the gsm8k probe against itself | Digest verdicts |
| `G2.digest_vs_baseline.gsm8k` | every arm | digest verdict vs baseline's gsm8k probe (+ resume replay) | Digest verdicts |
| `G2.digest_self.replay` | store arms | digest verdict, no judged-fraction floor | Digest verdicts |
| `G2.digest_self.cap` | baseline, store arms | exhaustive digest verdict of the cap probe | G5 verdicts |
| `G2.digest_groups` | baseline, every arm | every scored group has digest records | Digest verdicts |
| `G2.preempt` | arms run with `preempt` | exhaustive digest verdict, preemption happened, a resumed load judged | G2.preempt |
| `G2.gsm8k` | every arm | accuracy within `ACC_TOL` of baseline at the same batch composition | G2.gsm8k |
| `G2.gsm8k_outputs` | every arm | completions equal the reference (gated under batch invariance) | G2.gsm8k_outputs |
| `G2.resume_reference` | gsm8k arms, split-variant models | baseline resume replay ran and is clean | Split-variant references |
| `G2.caught_per_group` | negative self-tests | every sabotaged group caught in both gsm8k digest checks | Self-test arms |
| `G3.gsm8k_warm`, `G3.replay_warm` | store arms | excess within one block per request, no load errors | G3 |
| `G3.no_eviction` | store arms | no server eviction in any gsm8k / replay pass | G3 |
| `G4.replay_{cold,warm}.{ttft_p50,req_per_s}_vs_native` | arm under test only | reported, never fails (no `pass` key); judged | G4 |
| `G4.gsm8k_{cold,warm}.seconds` | store arms but `native` | wall time at most 2% worse than `native`'s same pass | G4 |
| `G4.warmup_seconds` | store arms but `native` | reported only | Passes |
| `G5.replay_warm` | store arms | at `f = 0.5`: excess, load errors, refs, `G2.digest_self.cap`, pool and load proof | G5 verdicts |
| `G5.curve` | store arms | same at the other fractions; never gated | G5 verdicts |

Recorded in `summary.json`, not verdicts: `G5.capacity_floor`,
`contamination`, `cap_plan`, `pool_checks`, `hit_report`, `native_report`, `provenance`,
`provenance_watch`, `scan`, `isolation`, `vllm_plugins`, `port_leases`,
`cuda_env`.

## Official run

Exit 0 only for an official run in which every required verdict passes;
`summary.json` lists `not_official_because`. A required verdict that cannot
be evaluated fails. Required: the verdicts of `baseline` and of the connector
under test (including `G2.digest_self.cap`, `G5.replay_warm`). Recorded but
not required: `native`, and `old` / `lmcache` when run (optional, never
required; evaluated like any store arm when present). A crashed reference-arm stage is
recorded under its `errors` and the run continues; a crash of baseline or the
arm under test aborts.

Official only if all hold:

| Condition | Rule |
|---|---|
| Profile | `run_gate.OFFICIAL`: full gsm8k, 8 shots, replay 32 x 12 with 64 output tokens, 3 replay passes, digest every 4th conversation, `replay_skip` 0, concurrency 32 (cap: the plan's `workers`), `max_model_len` 8192, pinned scheduler flags, cap `auto` x 12 at fraction 0.5 (or the curve 0.25 / 0.5 / 0.75, `--cap-curve`); lmcache chunk 256 |
| Arms | `baseline`, `native`, one connector under test; `plain`, `old`, `lmcache` optional |
| Onboarding controls | with `plain`: its verdicts. Without: gate.json `controls["tp<N>"]` of `--tp` names a run and rev, holds three passing control verdicts, is full-set (`gsm8k_limit` 0), and its `inputs` = this run's `control_inputs` (`run_gate.controls_problem`; a missing, failing, subset or stale record is non-official, naming what is wrong) |
| Workloads | `gsm8k`, `replay`, `cap`; plus `preempt` on a model with a Mamba state group (committed spec): there gsm8k loads a single prefix, and only `preempt` judges every resumed episode's state load |
| No overrides | no `--num-gpu-blocks` / `--vllm-arg` |
| Committed files | `docs/rsi/models/<model>/gate.json` and the committed spec dump of `--tp` (G0), recording `parallel.tp` = `--tp`; a TP without one is non-official (`committed spec dump ... missing (no dump for --tp N)`) |
| Capacity floor | `C(0.5) >= 2 x` vLLM GPU KV bytes |
| Cap plan | not sized from a spec with unmodeled Checkpoint features (`cap_plan.unmodeled`) |
| Freeze separation | arm under test's `freeze_vs_oracle_c` at `f = 0.5` `< 0.9` |
| lmcache build | with `lmcache`: the overlay holds gate.json `lmcache_dist`, and the model declares one |
| Under-test snapshot | `--under-test-path` given and verified; the arm's module is in the `pegaflow` package (not `rsi.*`) |
| Provenance | gate tree and venv `pegaflow` tree verified; re-verification finds no change before any stage or after the last |
| Isolation | arm under test isolated (Linux, root) |
| Static scan | no non-allowlisted hit in the snapshot |
| CUDA toolchain | every nvcc on `CUDA_HOME` / `PATH` has the venv torch's CUDA major; a missing nvcc, or a `CUDA_HOME` without one, is non-official. `summary.cuda_env` records torch CUDA, `CUDA_HOME`, `LD_LIBRARY_PATH`, `PATH`, each nvcc's path and version |
| Not contaminated | per G4-timed replay pass (`cold`, `warm2..`): no other arm's p99 TTFT `> 10x` native's, and native's not `> 10x` the median of the other arms' p99 (baseline not looked at: no G4 role). Recorded in `summary.contamination`; the pass is still judged |
| Working native | native's replay stage present, not crashed, has `cold` and every timed warm pass `warm2..N` of the profile with `ttft_p50` and `req_per_s`, no load error, warm excess within limit, every timed warm pass `E_ext >= 0.9` (`run_gate.g4_native_problems`). Validity of the reference only: native's hit levels stay reported and its own G0 / G2 verdicts never make a run non-official. A model without a working native is listed under "native", never let through |
| Not held out | `--replay-skip 0` |

- Slices (`--gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --cap-convs 8
  --cap-turns 6`) are non-official and exit 1. A slice G5 is below the floor:
  a smoke test of flow and digests; its `E_C` supports no conclusion.
- Exit 0 does not cover G1 (run `cargo test -p pegaflow-semantics`) or "no
  regression vs the last accepted iteration" (G4 5%, reported `H` / `E_C`):
  compare by hand until `--accepted` exists.
- A reference-arm rerun outside an official run lines up with it only with
  the same model, flags, cap plan (`summary.args.cap_plan`) and baseline `U`.

## Arms

| Arm | Official | What |
|---|---|---|
| `baseline` | required | probe only; the correctness reference (gsm8k, preempt, resume, cap; replay `cold` only) |
| `native` | required | vLLM `OffloadingConnector`; every G4 reference; its verdicts are not required, a working native is |
| `<module>:<Class>` | required | connector under test |
| `plain` | optional | no kv-transfer-config; gsm8k only; the onboarding controls (run with the self-test) |
| `old` | optional | black-box reference: old `PegaKVConnector` from the venv's `pegaflow`, record only |
| `lmcache` | optional | LMCache in-process connector, record only |
| `selftest:<mode>` | self-test | gate self-test connector; gsm8k and preempt only |

An optional reference arm is not run by default; when run it gets every
store-arm verdict (recorded, not required) and its G3 / G5 numbers are
reported next to the arm under test's.

Wall time on the 4090 host (`gsm8k,replay,cap` at the official profile):
default arm set `baseline,native,<arm>` about 1 h on Qwen3-4B, about 2 h on
Qwen3.5-4B (with `preempt`). Each optional store arm (`old`, `lmcache`)
adds about 40 min (Qwen3-4B) / 80 min (Qwen3.5-4B); `--cap-curve` adds about
25 / 40 min per store arm; `plain` about 10 min. These are TP 1 numbers.
At TP 2 the cap stages are larger (Qwen3-4B: 65 workers, at least 130
conversations, against 32 x 64 at TP 1), so each cap stage takes longer.

`old`, `native`, `lmcache` and the arm under test run as
`MultiConnector[arm, RsiProbe]`. The arm under test gets
`kv_load_failure_policy=recompute`, `PEGAFLOW_PORT`, and an empty
`kv_connector_extra_config`. The probe keeps
`prefer_cross_layer_blocks=False`; cross-layer KV layouts are out of scope.

## Per-model settings

`docs/rsi/models/<model>/gate.json` (gate-owned):

| Key | Meaning |
|---|---|
| `batch_invariant` | checked by `G2.batch_control` |
| `split_invariant` | default true; checked by `G2.split_control` |
| `invariant_attention_backend`, `attention_backend` | backend with / without batch invariance |
| `vllm_args` | every arm's vLLM (e.g. `mamba_cache_mode`, `prefix_cache_retention_interval`) |
| `gsm8k_vllm_args` | appended on gsm8k and resume stages ("Compile determinism") |
| `lmcache_dist` | the lmcache build the model's lmcache numbers come from |
| `controls` | per TP (`"tp<N>"`): the onboarding control run's `summary.controls_record` (`run`, `rev`, `gsm8k_limit` 0, `verdicts`, `inputs`; `inputs_recomputed` when the inputs postdate the run); "Onboarding controls" |

`G0.vllm_args` parses every spelling vLLM accepts (`--flag=value`, `_` for
`-`, abbreviations, `-cc`, dotted keys). The spec dump records
`cache.mamba_cache_mode` and `prefix_cache_retention_interval`, so `G0.spec`
pins them too. Pinned for every arm: `--max-num-batched-tokens 2048
--max-num-seqs 256 --enable-prefix-caching --seed 42 --generation-config
vllm`, `PYTHONHASHSEED=0`. Cross-run byte equality of computed KV needs batch
invariance (`VLLM_BATCH_INVARIANT=1` + FLASH_ATTN).

## Passes

- Each arm x workload: fresh vLLM (+ fresh PegaFlow server for `old` and the
  arm under test).
- gsm8k and replay start with one untimed warmup request (512 fixed
  pseudo-random tokens, 16 output, `rsi_pass=_warmup`, never scored), then
  `POST /reset_prefix_cache`; the stage fails if a workload request shares a
  prefix block with it (`oracle.warmup_problems`). `G4.warmup_seconds` is
  reported. Cap stages have no warmup.
- `cold`, then per warm pass: sleep `--settle`, reset prefix cache, `POST
  /collective_rpc {"method": "rsi_poison"}` (fills every KV byte with 0x3F, a
  finite bf16/fp16 value), run. Warm reuse can only come from the connector.
- gsm8k: `cold`, `warm`. Plain (onboarding) also runs `split_ref`, `split`, `alone`;
  plain and baseline run `alone_ref` when `batch_invariant=false`; baseline
  runs `split` when `split_invariant=false`. `<pass>_prewarm` requests are
  never scored.
- Replay: `cold`, `warm` (digested), `warm2..warmN` (timed, N =
  `--replay-repeats`). Baseline runs `cold` only (measured `U` for the pool
  check, cap plan fallback): it has no perf role.
- Requests carry `vllm_xargs.rsi_pass` (the oracle scores by it), replay and
  gsm8k also `rsi_conv` / `rsi_turn`.

## Probe trace and digests

- Trace per request: `episodes = [step, local_hit, start]` per schedule
  episode (a resume after preemption opens one); `progress = [step,
  computed_after]` at key-token boundaries. `local_hit` = the value passed to
  `get_num_new_matched_tokens` at the last attempt before scheduling. Load
  failures go to `load_errors.jsonl`. `preempt_digested` lists the episodes
  digested at their preemption (below).
- Digest V3 at finish, for requests tagged `rsi_digest = full` (gsm8k all;
  replay `cold` / `warm` every 4th conversation; cap and preempt all): every
  fully computed block of every transferable group, per layer, on the raw
  allocation with `KVCacheTensor` geometry, unpadded page bytes (two 64-bit
  position-weighted sums + float L1). `register_kv_caches` raises if a layer
  view disagrees with that geometry. Tag from the last episode: `local`
  (`covered <= local_hit`), `external` (`<= start`), `computed`.
- Mamba state groups (`mamba_cache_mode=align`, spec.json
  `probe.state_groups`), tagged by `episode`. vLLM frees a block-end state
  once compute moves past it, so states are digested within their step:
  - offers (`boundary_state_offloads` above the episode start) in
    `wait_for_save`, after the forward wrote them;
  - each episode's start state (block `start / block_size - 1`) in
    `handle_preemptions`, which vLLM 0.32's V2 model runner (pinned via
    spec.json `model_runner` and `G0.spec`) calls every step after
    `update_requests` and the align precopy and before a synchronous
    `start_load_kv` and the forward: the digest is the bytes the forward
    computes from, so a sync load lands after it and is `mismatched`;
  - `requests.jsonl` `states` lists `[episode, covered, group, what]`; `what`
    other than `offer` / `start` is a skip (`not_offered`, `start_*`). Only
    step ends exactly on a boundary are references. The digest verdict
    reports the skips (`state_skipped`); any `start_*` fails it.
- Preempted episodes: `handle_preemptions` digests the blocks a preempted
  episode left (references; its loads judged). A block vLLM reallocated and
  overwrote earlier that step (`new_block_ids_to_zero`, a COW destination) is
  dropped (`preempt_dropped`) and counts as undigested. An all-zero computed
  digest is never a reference (`ref_zero_copies`, `computed_zero`).

### Digest comparison

`digest_compare.compare(REF, CAND, invariant)`; REF may pool several probe
dirs.

| Block | Outcome |
|---|---|
| loaded | `matched`: all layers equal some computed copy of the same (rank, group, prefix key) in REF; `mismatched`: none equal and the reference is exhaustive (batch invariant, or every computation of the key digested); `uncertain`: otherwise; `unreferenced`: REF never computed the key |
| computed | `exact`: equals another computed copy (split-variant: same split only); `computed_unmatched_split` / `computed_unreferenced`: not judged. `computed_by_pass` breaks them down |

Split = (computing episode's start, its `progress` chunk ends below the
block's end, hash of the request's block digests at or below the start on
that rank). Wrong context bytes are judged where they enter, as loaded
blocks. A chunk end that crosses no key-token boundary is missed: that can
only make an honest block inexact, never a wrong one exact.

### Digest verdicts

- `G2.digest_self.gsm8k`, `G2.digest_vs_baseline.gsm8k`: local and external
  `mismatched == 0`, `unreferenced == 0`, no `start_*` skip,
  `external_matched > 0`, judged `>= 0.9` of external blocks, and under
  invariance `computed_exact == computed_compared`. Baseline self: no local
  mismatch.
- Split-variant models, `G2.digest_vs_baseline.gsm8k` also: in `cold` and
  `warm`, judged computed `> 0` and `compared / (compared + unmatched_split +
  unreferenced) >= 0.9` (REF = baseline gsm8k + its `resume-<label>`
  replay); in `warm`, zero unjudged computed blocks on prompts whose length
  is not a multiple of the hit alignment (no recorded length counts as
  non-aligned; the `aligned` exemption needs a token record).
- `G2.digest_self.replay`: same, no judged-fraction floor.
- `G2.digest_groups`: every group the oracle scores has digest records and is
  not in `probe.digest_skipped_groups` (a `MambaSpec` outside align mode, or
  with speculative blocks).

## G0

- Spec dump committed per TP (`run_gate.committed_spec`):
  `docs/rsi/models/<model>/spec.json` for `--tp 1`, `spec-tp<N>.json` for
  `--tp N`. Head counts, page bytes and block strides are per rank, so a dump
  is never judged at another TP. Contents: `kv_cache_config`, per-group spec
  class, block_size, page bytes, layer -> group map, TP shard info, the gsm8k
  stage's effective `compilation` (mode, backend, `inductor_compile_config`,
  `cudagraph_mode`).
- Every consumer reads the selected dump: `G0.spec`, lmcache storing ranks,
  the preempt GPU pool, the `UniqueEstimate` pool check, the cap plan
  fallback and the official rules (Mamba state group -> `preempt` required;
  committed dump present with `parallel.tp` = `--tp`). Without it at `--tp
  N`: `G0.spec` fails, the preempt workload and the lmcache arm at TP > 1
  do not start, the pool estimate is skipped, and the run is non-official.
- gate.json is per model, not per TP: its claims (`batch_invariant`,
  `split_invariant`, backends, vLLM args) hold at every TP the model has a
  dump for, and `G2.batch_control` / `G2.split_control` measure them per TP
  at onboarding (`controls["tp<N>"]`), so a TP where one does not hold fails
  there.
- The adapter's declared layout (Kind, Shard, unit size, needed set) is
  checked against it: sizes add up, every layer in exactly one group.
- `G0.spec`: arm's scheduler dump and layer view layouts (shape without the
  block dim, stride, dtype) = baseline's = committed. Excluded: `num_blocks`,
  tensor `size`, `layer_stride`, model path, `max_model_len`. Catches compile
  overrides too. New field: bump `rsi_probe.SPEC_SCHEMA`, add to
  `run_gate.SPEC_FIELDS`.

## G2 controls and output checks

`G2.plain_equals_baseline` (the probe changes nothing) and
`G2.warm_equals_cold` (reset + poison change nothing) compare completions;
with `batch_invariant=false` they follow the batch-variant rule below.

### Onboarding controls

`G2.plain_equals_baseline`, `G2.split_control` and `G2.batch_control` need
the `plain` arm. They run at onboarding, in a full-set self-test with
`plain` at each TP the model is gated at. A clean run with `plain` whose
three controls pass at `--gsm8k-limit 0` writes `summary.controls_record`.
Clean means verified, unchanged gate and connector trees (provenance and the
provenance watch), a sane CUDA env, this TP's committed dump, and no
`--num-gpu-blocks` / `--vllm-arg` override: a dirty tree never yields a
pasteable record. Any other run writes None there and the reason in
`summary.controls_record_withheld`. The record goes into gate.json
`controls["tp<N>"]` unchanged:

- `run` (run dir name), `tag` (`--tag`) and `rev` (gate rev);
- `files`: sha256 of every file the control verdicts may read, by path in
  the run dir (`control_files`): the plain and baseline
  `.gate/<label>/gsm8k/*-outputs.json` and `*-tokens.json`, and the baseline
  gsm8k spec dump (hit alignment). Both cold outputs must be named;
- `gsm8k_limit`: always 0. A record is full-set because official runs are
  full-set: the split control judges every prompt whose warm pass the
  official run judges, and a subset leaves the rest unmeasured;
- `verdicts`: the three control verdicts, each a measured pass (`na` is not
  one) with its counts, typed (`CONTROL_MEASURED`: counts are ints, claims
  bools);
- `inputs` = `control_inputs` of that run.

A run without `plain` reports the three `na`. It is official only if the
record of `--tp` names a run, tag and rev, hashes its files, holds three
passing verdicts, has
`gsm8k_limit` exactly the integer 0, its verdicts hold when re-judged, and
its `inputs` equal this run's `control_inputs`
(`run_gate.controls_problem`). Re-judged: each verdict carries its counts
with the right types (a malformed value is a problem, not a crash),
plain equals baseline with 0 differing, the split and batch controls
measured gate.json's current `split_invariant` / `batch_invariant` claims
and found them, pre-warmed at least `MIN_PREWARMED` of a non-empty set, the
split reference equals cold, and the batch control compared
`BATCH_CONTROL_N`. A pasted `{"pass": true}` is rejected. Any gap is
non-official and names what is missing or changed.

The gate cannot tell a forged but self-consistent record from a real one;
`tag` and `files` let the reviewer check it on the host where the run dir
lives. From a gate tree at the record's `rev`, run
`run_gate.recheck_controls_record(<run dir>, <record>)`. It must return
`[]`: the run dir's name, summary.json's `tag` and `rev` match the record,
every file hashes as in `files` (none added or missing), and
`control_verdicts` re-run on the run dir (summary.json's `results` and
`gate`) gives the record's `verdicts`. The reviewer also checks
summary.json's `controls_record` against the record and that
`controls_record_withheld` is None. `control_inputs`
holds:

- `identity`: the model (dir name), `--tp`, and sha256 of that TP's
  committed spec dump. A record copied under another TP's key or into
  another model's gate.json is rejected;
- sha256 of `rsi_probe.py` and `workloads.py`;
- sha256 of the AST (docstrings dropped) of every function between the
  control stages' arguments and the verdicts (`CONTROL_CODE`): which arm
  runs which workload with which args (`runs_workload`, `stage_args`,
  `cap_stage_args`, `run_stage`, `arm_label`), the stage
  (`run_arm_workload`), the vLLM launch (`vllm_server`,
  `kv_transfer_config`, `vllm_plugins`, the compile cache and its autotune
  seed), the gsm8k pass plan and per-pass request arguments (`gsm8k_plan`,
  `run_gsm8k_pass`: passes, prompt subset, concurrency, logprobs,
  pre-warm), batch invariance per stage, reset + poison, the output and
  token readers (`_outputs`, `_tokens`, `_differ`, `_hit_alignment`), the
  control functions, the three verdict functions and their assembly
  (`control_verdicts`: which arm's outputs and which claims each verdict
  reads; `_predates`, `_ran_without`). Not hashed: process plumbing that
  reaches neither vLLM's arguments nor its env (port leases, launch and
  teardown, isolation of the arm under test), and the one-line calls into
  these functions from `main` and from `verdicts()` (which places the three
  `control_verdicts` results in `v["baseline"]`);
- the constants `SPLIT_PASSES`, `MIN_PREWARMED`, `BATCH_PASS`,
  `BATCH_CONTROL_N`, `ALONE_REF`, `_ALONE`, `_REFERENCE`, `AUTOTUNE_SOURCE`;
- the run arguments of the gsm8k stages: `shots`, `concurrency`,
  `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`, `gpu_mem`,
  `num_gpu_blocks`, `vllm_arg`, `settle`;
- sha256 of the gsm8k dataset (`train.jsonl`, `test.jsonl` under
  `--gsm8k-dir`);
- gate.json `attention_backend`, `invariant_attention_backend`,
  `batch_invariant`, `split_invariant`, `vllm_args`, `gsm8k_vllm_args`.

The model and TP select the record (the model's gate.json,
`controls["tp<N>"]`) and `identity` ties it to them. vLLM and the model
are pinned by `G0.spec` (committed dump). Any change to these makes the record stale: rerun the
self-test with `plain` (or add `plain` to the run, which then judges the
controls itself) and replace the record. A record whose `inputs` were
computed later than its run, because the run's rev is off this lineage or
the fingerprint grew since, carries `inputs_recomputed`: what differs
between `rev` and the recomputed inputs, and why the controls behave the
same. Without that reason the record is rejected. The reviewer checks the
reason. `test_committed_controls_records_are_current` fails on a stale or
incomplete committed record.

### G2.gsm8k

Every arm, honest self-test included. Batch invariant: `|acc(pass) -
acc(baseline cold)| <= 0.01` for cold and warm. Batch variant: gated on
`alone_ref` (first 64 prompts one at a time, after reset + poison) vs baseline
`alone_ref`; `batched_delta_pp` and `alone_ref_differ` reported.

### G2.gsm8k_outputs

Gated under batch invariance, else reported.

- Split invariant: every completion = baseline cold, cold and warm.
- Split variant: cold = baseline cold; warm = the baseline resume replay's
  completion of the prompt (`warm_reference`; without that stage, plain's
  `split`; with neither the check fails). Reported only (`warm_differ_vs_split_aligned`): prompts with
  `prompt_len % cache_hit_alignment_tokens == 0` the replay could not resume
  where the arm did. A non-aligned prompt missed or resumed elsewhere
  differs.

### G2.preempt

With `preempt` in `--workloads`: the first 32 gsm8k prompts, exactly 640
tokens each (`ignore_eos`), cold then warm, on a pool of `preempt_gpu_blocks`
(4 x 2048 tokens, at least one `max_model_len` request; `max_num_seqs` capped
at the block count). Every request and preempted episode is digested. Passes
iff the exhaustive digest verdict passes (as `G2.digest_self.cap`, judged
`>= 0.9` for an arm), `preempted_requests > 0`, and a resumed episode's load
was compared (`after_preemption`). Required for the honest self-test when run
and, on state-group models, always (missing = fail); a negative arm is
judged by the gsm8k digest checks.

### Split invariance

Batch invariance does not imply independence from where computation starts:
vLLM 0.32's MLA prefill merges new-token and cached-context attention, so
outputs after a hit depend on where it ends; dense FLASH_ATTN does not.
gate.json's `split_invariant` is never trusted:

- Plain (onboarding) runs `split_ref` (gsm8k, greedy, logprobs, after reset + poison) and
  `split` (each request preceded by its prompt with `max_tokens=1`, so the
  hit ends at the last full prompt block, like a connector's warm load).
  `--enable-prompt-tokens-details` (plain only) reports cached tokens.
- `G2.split_control` passes iff measured invariance (completions and sampled
  logprobs equal between `split` and `split_ref`) = gate.json, `split_ref` =
  plain cold, and `split` cached `>= (prompt_tokens - 1) // align * align`
  for `>= 0.9` of requests. `same_start` pairs are reported; `>= 0.9 n` with
  no divergence reports `vacuous` (`measured_invariant: null`), verdict
  unchanged.
- With `split_invariant=false` only `computed_exact` (same split + coverage
  floor) and `G2.gsm8k_outputs` (warm vs resume replay) change; loaded-block
  checks and `G2.gsm8k` do not.

### Batch invariance

`G2.batch_control` passes iff measured invariance (plain `alone`: the first 64
`split` prompts, same pre-warm, concurrency 1, completions and sampled
logprobs equal to batched `split`) = gate.json `batch_invariant`, and the
`alone` pre-warm took for `>= 0.9`. (vLLM 0.32 cannot be batch invariant on
GDN: Mamba backends reject `VLLM_BATCH_INVARIANT`.)

Batch variant: `G2.warm_equals_cold` and `G2.gsm8k_outputs` report differ
counts (`gated: false`). Plain and baseline run `alone_ref` (same 64 prompts,
one at a time, no pre-warm): `G2.plain_equals_baseline` requires their
`alone_ref` equal in completions and sampled logprobs (`cold_differ`
reported); `G2.split_control` compares plain `alone_ref` with plain `alone`.
`G2.gsm8k` and every digest check stay hard (gsm8k digests are exhaustive).

### Split-variant references

`split_invariant=false`, every arm with a gsm8k stage, needs baseline. After
the arm's gsm8k stage the gate runs `baseline/resume-<label>`
(`run_gate.resume_plan`, `workloads.run_resume`):

- Waves from the arm's trace: each warm request's prompt (`rsi_conv`) and
  first episode, grouped by `step` in schedule order. After warmup and reset
  + poison, a request whose start exceeds its local hit is first sent as its
  first `start` tokens with `max_tokens=1` (`resume_prewarm`), then whole
  (`resume`, streamed). A wave starts when the previous wave has its first
  tokens.
- A preempted warm request's later episodes are chained: episode `k` runs as
  the first `episode_tokens[k]` tokens of the replay's own sequence,
  pre-warmed to its start (`resume_ep`, `rsi_turn = k`). An episode the
  replay cannot form stops the chain (`episode_missing`, completion
  differs). No `episode_tokens` fails the plan closed.
- Its digests join REF, its completions are the warm reference; wrong loaded
  bytes are still matched only against computed copies.
- `G2.resume_reference` (required for the honest self-test on split-variant
  models): the stage ran, its `digest_self` passes, the plan has no problem
  (one warm request per prompt, valid indices, episode lengths).
  `start_differs` / `episode_start_differs` / misses are reported; their
  blocks stay unjudged and completions differ.
- The stage gets `gsm8k_vllm_args`. A crash fails that arm's
  `G2.resume_reference` for a record-only arm and aborts for the arm under
  test.

### Compile cache and autotune

- Each arm has its own `VLLM_CACHE_ROOT` `<run>/.compile-cache/<label>`
  (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`
  inside), emptied before its first stage, shared by its later stages; the
  isolated arm gets only its own rw. No arm loads another arm's or run's
  artifacts.
- Triton autotune picks outside Inductor (FLA ops for GDN) change numbers, so
  each arm root gets a copy of baseline's `*.autotune.json`
  (`seed_autotune`), loaded under `TRITON_CACHE_AUTOTUNING=1`. Only picks
  are copied, never kernels; a file with anything but integer tile configs
  and numeric timings is refused (`autotune_file_problem`: no `ir_override`,
  `pre_hook`).

### Compile determinism

vLLM 0.32 compiles once per config hash (plain, connector arms, and `old`
each differ), and Inductor benchmarks some kernel choices, so every gate.json
sets `gsm8k_vllm_args` = `--compilation-config
{"inductor_compile_config":{"deterministic":true}}`, appended on every gsm8k
and resume stage, where the cross-arm exact comparisons run; the committed
`compilation` block shows it and `G0.vllm_args` requires it as the last
occurrence. Replay, cap and preempt keep the default compile: their digests
are judged within the stage. `G4.gsm8k_*` times both sides under the gsm8k
config, `G4.replay_*` under the default.

## G3 Hit rate (unlimited store)

Prompt tokens: `local_hit`, `external_hit`, `computed`. `H = external_hit /
(prompt_tokens - local_hit)`; `H*` = the trace replayed through
`semantics.resolve` with unlimited capacity; `E = H / H*`. Also reported per
group: hit tokens, resolve fallbacks and tokens lost to them, lease /
prefetch failures, loaded-but-unused and saved-but-never-loaded bytes;
`native`'s `H` (and `old`'s when run) as black-box references.

Oracle (`oracle.py`) replays `requests.jsonl`: a block ending at token e is
stored at the first step whose `progress` reaches e (checkpoint groups: a step
ending exactly on e), visible to requests first scheduled after it. Kinds:
no window -> Full; window sw -> Window{sw-1} (sw == 1: Scratch); Mamba ->
Checkpoint; `UniformTypeKVCacheSpecs` -> inner Kind; unknown -> error. Per
request `ext = max(0, min(actual, oracle) - local)`, `ext* = max(0, oracle -
local)`; `E_ext = Σext / Σext*`; `excess = Σ max(0, actual - max(oracle,
local))`.

- `G3.gsm8k_warm`, `G3.replay_warm`: `excess <= one block per request`, no
  load errors; replay also needs `G2.digest_self.replay`. Reported: `E_ext`
  (`None` is `below`), `oracle_hit_rate`, hit / local / external rates.
- `G3.no_eviction`: `pegaflow_cache_block_evictions_total` delta 0 in every
  gsm8k and replay pass (`native` / `lmcache`: n/a). Server: `--pool-size
  64gb --disable-numa-affinity`. `PoolTooSmall` fails the run before any
  store stage when the committed spec's `UniqueEstimate` (replay prompts +
  outputs; gsm8k prompts), or later the baseline-measured `U`, exceeds
  `--pool-size` (`summary.pool_checks`).

## G4 Performance (gsm8k hard, replay judged)

Every G4 reference is the in-run `native` arm. The gsm8k overhead check is
hard (the regression floor); replay cold and warm vs `native` are reported
and judged. Store arms other than `native` get the gsm8k check (required
only for the arm under test; `old` / `lmcache` record it); only the arm
under test gets `*_vs_native`.
Self-test arms, `plain`, `baseline` and `native` get no G4 verdict. Baseline
has no perf role (its replay runs `cold` only).

Workload: multi-turn agent traces (user-provided "agentx" when available;
until then R2EGym-SFT-Trajectories as growing multi-turn chat,
`~/kexi/datasets/r2egym-256.jsonl`) plus gsm8k as a short-prefix workload.
Metrics: TTFT p50/p99, req/s, output tok/s, load / save bandwidth, connector
scheduler CPU per step, GPU memory overhead. Medians over the listed passes;
p99 recorded, not gated.

| Check | Rule |
|---|---|
| `G4.gsm8k_{cold,warm}.seconds` | hard: at most 2% worse than `native`'s same pass (bounds per-step overhead); a missing or crashed native gsm8k stage fails it (no reference) |
| `G4.replay_cold.{ttft_p50,req_per_s}_vs_native` | reported, not gated: `cold` vs `native`'s `cold` (the save cost). Arm under test only |
| `G4.replay_warm.{ttft_p50,req_per_s}_vs_native` | reported, not gated: `warm2..N` vs `native`'s `warm2..N`. Arm under test only |
| last accepted iteration | no > 5% regression (by hand) |

A `*_vs_native` record carries `reason` when native has no replay stage
(missing or crashed: run non-official), or when native's replay or the
arm's has no pass of that stage with the key; it is marked `void` when
`G2.digest_self.replay` fails (the arm fails on that verdict). The gsm8k bar
is relative to native, which carries its own save cost: an arm that costs as
much as vLLM's own offloading passes; one that costs more than 2% beyond it
fails. G2 references stay baseline's (native fails `G0.spec` by
construction).

### Perf vs native judgment

Each `*_vs_native` record holds both arms' pass values, medians and
pass-to-pass ranges, `gap` (median - native median), `gap_pct` (of native's
median; TTFT positive = slower, req/s negative = slower), `noise` (the larger
range) and `within_noise` (`|gap| <= noise`, a diagnostic; `null` for
`replay_cold`, one pass per side, so no range); median, gap and noise are
absent when either side has no pass of that stage with the key.
`summary.native_report` (the `[gate] perf vs native (reported, not gated)`
line) lists median, native median, `gap_pct`, `behind_pct` (positive = the
arm is slower than native, for either metric), `within_noise`, `void`,
`reason` per arm and key.

Stopping rule: a model is not done when it is close to native. The agent
iterates, one hypothesis at a time, until it judges that no connector-side
optimization is left. Per accepted run, in
`rsi-notes:models/<model>/ITERATIONS.md` next to the hit-rate judgment:

- the remaining gap in % for warm replay, cold replay and gsm8k, each split
  into four parts with numbers: connector (load path, save path,
  scheduling), server ("Known server limits": not ours to fix), vLLM /
  native structural differences (e.g. native's in-process CPU copy vs a
  server round trip, layouts), and noise (pass-to-pass ranges);
- the connector-side hypotheses tried, each with its result, and why each
  remaining one is not worth trying, with numbers (measured upper bound of
  its gain, cost, or a guard below that rules it out).

Compare the load paths first: sync vs async load, load round trip in steps,
per-request load size. Guards stay: the server is out of scope, `H*_C` is a
ceiling and not a target, no tuning to the trace (held-out slice), and the
policy complexity budget (skill step 5). The review agent checks the
numbers against `summary.native_report` and that the attribution holds (no
connector-side part labeled server, structural or noise without evidence),
and can reject the run on it.

## G5 Capacity hit rate

G3's unlimited pool cannot see eviction, write amplification or cross-group
coherence; G5 measures the hit rate at a fixed store size against a
clairvoyant bar. All external KV must come from the PegaFlow server: a
connector-side cache is a violation, checked through the server's load bytes.
Server fragmentation and headroom count against the connector/server.

### Workload and capacity

- Trace: `--cap-convs` x `--cap-turns` (official `auto` x 12) conversations
  of `--replay-file`; `--replay-skip N > 0` is a held-out slice
  (`summary.held_out`), never official. `cap_cold` runs conversations in
  order; warm pass `j` rotates them by `j * n // 4` (guarded by the
  separation check).
- Baseline cap stage (before any store arm's): `cap_cold`, `cap_warm` (after
  reset + poison, digested). `U` = bytes of distinct (group, key) blocks the
  unlimited oracle stores in `cap_cold`. `ref(pass, conv, turn)` = baseline's
  `local_hit` there (`cap_cold` for `cap_cold`, `cap_warm` for warm passes).
- `C = int(f x U)`, `--cap-fractions` must include 0.5. Official: 0.5 alone
  (default), or the curve 0.25, 0.5, 0.75 (`--cap-curve`, exclusive with
  `--cap-fractions`; each point is a full cap stage per store arm); only 0.5
  is gated, the curve is reported (`G5.curve`) and adds no hard check.
  `G5.capacity_floor`: `C(0.5) >= 2 x` baseline GPU KV bytes, else
  non-official.
- Block bytes per (group, key) = `real_page_size_bytes` (else
  `page_size_bytes`) x group `num_layers` (`UniformTypeKVCacheSpecs`: x 1) x
  `tp / kv_tp_replicas`. One copy of replicated bytes (PegaFlow, and native's
  `cpu_bytes_to_use` gets `C` as is). Skipped groups hold no bytes. GPU KV
  bytes = Σ `kv_cache_config.tensors[].size` of one rank (shared tensors
  once) x the same copies.
- `auto` (`summary.cap_plan`): spec from this run's baseline gsm8k dump (else
  replay; never preempt; else committed; `cap_plan.spec`); the floor reads
  GPU KV from the cap stage's dump. Conversations from the file start (>= 64
  and >= 2 x `workers`) until estimated `U >= 1.1 x 2 x GPU KV / 0.5`.
  `workers` (`cap_workers`, every cap stage's concurrency, baseline
  included) = `max(--concurrency, ceil(1.5 x GPU KV tokens / mean turn))`,
  turn = prompt + `--replay-out-tokens`, GPU KV tokens = `num_blocks / Σ
  1/block_size` over GPU groups. Fewer in-flight tokens than GPU KV and
  vLLM's prefix cache keeps every live conversation: the store is reused
  only across passes, in pass order, so `H_lru` gets nothing and a store
  frozen after `cap_cold` nearly reaches `H*_C` (Qwen3-4B TP2 at 32
  workers, 0.75 x GPU KV: local hit 0.85, `lru_vs_oracle_c` 0.003,
  `freeze_vs_oracle_c` 0.92 at every fraction; TP1 at 32 is 2 x GPU KV:
  0.39, 0.87, 0.62). Fewer conversations than `2 x workers` (the whole file
  used) or more workers than `--max-num-seqs` are `cap_plan.problems`,
  non-official; too few workers otherwise show as freeze separation. `UniqueEstimate` per Kind:
  Full / Window = distinct prompt-prefix blocks + generated complete blocks;
  Checkpoint = only step ends on a boundary, with vLLM 0.32's align-mode
  prefill split (`prefill_step_ends`) and every decode boundary, assuming a
  request alone (not a bound). Assumes `hash_block_size == block_size`, no
  Eagle / speculative blocks, `num_prefill_checkpoint_blocks == 0`; otherwise
  `cap_plan.unmodeled`, non-official. Over-estimates and an exhausted file
  fail closed at the floor.
- Store-arm cap stages per fraction (dir `cap`, or `capNNN`): fresh vLLM and
  (except `native` / `lmcache`, CPU bytes = `C`) a server with `--pool-size C
  --disable-numa-affinity`; passes `cap_cold`, `cap_warm`, `cap_warm2..N`,
  every request digested, replay settle / reset / poison protocol.
  `pool_capacity_bytes` = `pegaflow_pool_capacity_bytes` at start. Size-class
  rounding (512-byte units, at most 1/8 waste) is server headroom.

### Simulated stores

Over the arm's own cap trace, `C` bytes each, `ref` as local hit, G3's store
events. Hits = `resolve` over resident blocks plus `needed` at `ref`, capped
at `max(unlimited oracle, ref)`.

| Store | Rule |
|---|---|
| `belady` | next use = next request whose unlimited-oracle prefix `P` needs the item (`needed` at `P` minus `needed` at `ref`); no next use: not stored; evict furthest next use; an arrival whose own next use is furthest is not stored |
| `belady_fallback` | only where `oracle.prefix_closed` fails (`needed(q) ⊄ needed(P)` for some resume point `q < P`; decided from the Kind): uses the union of `needed(q)` over resume points `q` in `(ref, P]` |
| `lru` (`H_lru`) | group-oblivious LRU, everything stored; re-store does not refresh, a hit refreshes loaded blocks |
| `freeze` (`H_freeze`) | stored in arrival order until full, never evicted |

`H*_C` = per segment, the candidate (`belady`, `belady_fallback`, `lru`; ties
in that order) with the largest `Σ ext*_C`: a hit rate some real store
reaches, never below `H_lru`; a strong reference, not a proven optimum
(`E_C > 1` allowed). Recorded: `oracle_c_policy`, `oracle_c_candidates`,
`oracle_c_policies`, `diagnostics.oracle_c{,_fallback}`; verdict
`oracle_c_groups` / `oracle_c_passes` of the winner, plus `belady_*` when
another won. `efficiency_c_own_local` uses the same candidates.

### Scores and load proof

Per segment (`warm_timed` = `cap_warm2..N`, and per pass):

- `ext = max(0, min(actual, oracle) - max(ref, own))`, `ext*_C = max(0,
  oracle_C - ref)`, `E_C = Σ ext / Σ ext*_C` (`None` fails). Scoring vs
  baseline's `ref` stops an arm gaining by keeping less or more locally.
- Reported: `lru_vs_oracle_c`, `freeze_vs_oracle_c` (`Σ max(0, H - ref) /
  Σ ext*_C`), `efficiency_c_own_local`, `efficiency_c_cold`,
  `freeze_vs_oracle_c_cold`.
- `unmatched_requests` = requests without a `ref`.
- Load proof: `external_hit_bytes` = Σ bytes `needed` at `actual` not held by
  `needed` at the own local hit; `loaded_bytes` = `pegaflow_load_bytes_total`
  delta.

### G5 verdicts

- `G5.replay_warm` (f = 0.5) gates: excess within one block per request, no
  load errors, `unmatched_requests == 0`, `G2.digest_self.cap`, and for a
  PegaFlow arm `pool_capacity_bytes` present and `<= C` (an arm without the
  pool metric fails) and Σ `loaded_bytes >=` Σ `external_hit_bytes` over
  `warm_timed` (`native` / `lmcache`: n/a).
  Reported: `E_C` and its change vs the last accepted iteration, `U`, `C`,
  hit / oracle / oracle_C / LRU / freeze rates, eviction deltas, server I/O
  and resident / used bytes per pass, and for `H_lru` and the winner
  per-group stored / not-admitted / evicted and resident / stranded bytes.
  Not implemented: actual stranded bytes on the server (needs a server dump).
- `G2.digest_self.cap`: 0 mismatched, 0 unreferenced, every loaded block
  judged, except blocks whose undigested computations are all preempted
  non-last episodes (`*_uncertain_preempted`, at most 1% of loaded blocks,
  reported `preempted_uncertain`). Arm under test needs `judged > 0`; a
  reference arm with none passes flagged `vacuous`; baseline needs no
  external block.
- `G5.curve`: same at the other fractions (only with `--cap-curve` or a
  listed `--cap-fractions`), never gated.
- Separation: arm under test's `freeze_vs_oracle_c >= 0.9` at f = 0.5 makes
  the run non-official (`G5 trace does not separate freeze from H*_C`).

## Self-test arms

`rsi/gates/selftest/connector.py`; full-attention and align-mode Mamba state
groups.

| Mode | Behaviour |
|---|---|
| `honest` | saves full blocks at finish and offered states; serves the longest prefix with all blocks and states saved. Must pass G0 and every G2 check (`G2.gsm8k`; on state-group models `G2.preempt`; split-variant `G2.resume_reference`) |
| `liar` | claims every unit, loads nothing |
| `bitflip` | flips one byte of the first loaded block (states: an exponent bit) |
| `offbyone` | loads block i+1 into slot i (states: the next state group's state) |
| `sync` | honest bytes, synchronous loads: on state-group models corrupts the state groups (precopy runs first). gsm8k only (vLLM dies on sync Mamba loads in the preempt pool); other modes load async there |

A negative arm is caught only if G0 passes and `G2.caught_per_group` passes:
for every group it corrupts (`sync`: state groups), in both gsm8k digest
checks, loads equal to no reference `>=` sabotaged loads (from
`sabotage-*.jsonl`) minus uncertain external loads, and sabotage `> 0`.
Preempt per-group counts and output deltas are reported only.

## Reference arms

### native

`OffloadingConnector` via `--kv-offloading-size`,
`kv_connector_extra_config = {"cpu_bytes_to_use": B}`, no server; `B` =
`--pool-size` (gsm8k / replay) or `C` (cap). Required in every official run:
it is every G4 reference (gsm8k overhead hard, replay cold and warm
judged); `E_C` reported. It switches `kv_cache_layout` to `LBHNC`, so `G0.spec` and
`G2.digest_vs_baseline.gsm8k` fail by construction; `G2.digest_self.*` and
`G2.gsm8k_outputs` are its correctness evidence.

Models without a working native: none. On Qwen3-4B native fails
`G2.digest_self.cap` at 0.5 on the preempted-uncertain bound only (0
mismatched); its G5 numbers carry that flag.

### old

Optional, record only: the old `PegaKVConnector` from the venv's `pegaflow`,
run as a black box when its numbers are wanted. When run it gets every
store-arm verdict (G2 / G3 / G4 vs native / G5), none required.

### lmcache

Optional, record only. Loaded by module path
(`lmcache.integration.vllm.lmcache_connector_v1:LMCacheConnectorV1Dynamic`)
from `--lmcache-overlay`, on that arm's `PYTHONPATH` only; no server; env
only (`LMCACHE_*` inherited dropped): `LMCACHE_LOCAL_CPU=True`,
`LMCACHE_MAX_LOCAL_CPU_SIZE = B / r / 2^30` GiB per storing rank (totals `B`),
`LMCACHE_MAX_LOCAL_DISK_SIZE=0`, no remote, chunk 256, telemetry off
(`LMCACHE_TRACK_USAGE=false`, `DO_NOT_TRACK=1`). `r` = 1 if all groups are
`MLAAttentionSpec`, else `tp` (`lmcache_storing_ranks`, from the committed
dump of `--tp`); the stage fails unless vLLM's log shows exactly `r` backends,
and TP > 1 without that dump does not start. `result.lmcache` records env,
logged config, `dist`, `storing_ranks`, `per_rank_bytes`, `backends`. 256-token
chunks (no trailing partial) are a granularity loss inside its `E` / `E_C`.
`G0.lmcache_build`: each stage's dist (what the overlay holds) = gate.json
`lmcache_dist`; otherwise, or without `lmcache_dist`, non-official.

#### lmcache status per model

Optional: run it when a comparison is wanted, never required. vLLM 0.32.0,
torch 2.13.0+cu130. Comparable only where its loads are
digest-verified (`G2.digest_self.*`, `G2.digest_vs_baseline.gsm8k` with 0
mismatches, warm accuracy within `ACC_TOL`). Never use a wrong-KV number.

| Model | `lmcache_dist` (overlay) | Chunk | Status |
|---|---|---|---|
| Qwen3-4B (FLASH_ATTN, `LBNHC`) | `lmcache-0.5.6` | 256 | not comparable: fused K/V page misread (LMCache #4463) |
| Qwen3-4B | `lmcache-0.5.7.dev0+pr4467.e05c2ea` (`~/kexi/overlays/lmcache-pr4467`) | 256 | comparable, except `cap075` (wrong loads; unused) |
| Qwen3-4B | same | 16 | not comparable: claimed hits never loaded |
| DeepSeek-V2-Lite-Chat (TP2, MLA) | PR #4467 `e05c2ea` | 256 | not comparable: crashes on first store |
| DeepSeek-V2-Lite-Chat | `lmcache-0.5.6` (`~/kexi/overlays/lmcache`) | 256 | comparable with caveat: G3 / G5 usable; `digest_vs_baseline.gsm8k` fails on 14 warm computed blocks without a same-context reference (reference gap, rule unchanged) |

The PR #4467 overlay is an unmodified source build of LMCache
`e05c2ea9781b4021081f34dfb982189ec0fabc15` against the shared venv's torch
(recipe: `rsi-notes:gates-evidence.md`). Qwen3-4B's lmcache numbers thus
come from an unmerged upstream PR: an open item for the user's final
acceptance.

## Known server limits

Gates run on the base pegaflow-server; fixing these is a server change (own
PR), and a gap they cause is explained, not fixed.

- Reclaim evicts `RECLAIM_BATCH_SIZE = 512` blocks per round
  (`pegaflow-core/src/storage/mod.rs`): on a small pool ~0.5 C stays
  resident.
- LFU admission is off (all saves inserted).
- The LRU (`pegaflow-core/src/cache.rs`) evicts each (hash, group) key alone,
  not prefix-aware: chain heads served from vLLM's local cache age out first
  and strand the tails for lookups from block 0.
- Fragmentation can drain the read cache. With `--blockwise-alloc` off
  (the default), a save allocates one pinned region per layer and segment
  for all blocks of the RPC (`offload.rs`). Such a region frees only once
  every block in it is evicted. Reclaim
  (`reclaim_until_allocator_can_allocate`, `storage/mod.rs`) evicts
  512-block batches until a *contiguous* region fits, so a fragmented pool
  can evict every cache-owned block, including a hot shared prefix. The
  same rev then flips per run between a pass with server hits and one
  without: Qwen3-4B TP2 cap075 warm3 ext was 0 or 6352-7856 on identical
  code (P5 `cap075 check`, rsi-notes). Master's #465 / #470 are in the
  tested base and do not change the save path.
  - G5 at f = 0.75 (and any curve fraction near the store's working set)
    can therefore be bimodal per run. A single run's verdict there needs a
    repeat before it is read as a regression or a gain.
  - The per-pass `cache_block_evictions_total` sums every reclaim call in
    the pass. Over-eviction (evictions well past resident + inserts − `C`)
    is the signal; the per-call drain is visible only at debug log level.

## Provenance

- `--under-test-path DIR` (a tree's `python/`): `provenance.stage` snapshots
  the verified rev's tracked blobs of `python/pegaflow` plus the one native
  module `pegaflow<EXT_SUFFIX>` into `<run>/under-test-overlay/pegaflow`,
  re-verified after the copy (blob ids, sha256), first on the arm under
  test's `PYTHONPATH` only. Any symlink, untracked / ignored file, other
  `.so` or `.pyc` in the source unverifies it. An unverified snapshot
  (`.rsi-rev` not a full 40-hex sha, rev not fetchable, tree differs) or an
  empty overlay stops the run before any stage (argument error, exit 2).
- Bytecode: every vLLM / server process gets a fresh
  `PYTHONPYCACHEPREFIX=<run>/.pycache-stages/<arm>/<workload>` no later stage
  reads; the gate re-execs with `<run>/.pycache`. Gate and venv trees may
  carry `__pycache__/*.pyc`.
- Gate code (`rsi/`) comes from the gate tree; `pegaflow` for the gate,
  `old` and the server's embedded Python from the shared venv's
  `pegaflow_llm.pth` (4090 host: `/root/kexi/pegaflow-semantics-rsi/python`).
- `summary.provenance`: gate, venv `pegaflow` and under-test revs with
  `verified` (clean git toplevel, or `.rsi-rev` whose `git ls-tree` blob ids
  equal the files, fetched blob-less from `provenance.REMOTE`), native
  sha256s, server binary sha256 + `.rsi-rev` (a claim). Officiality:
  `run_gate.provenance_problems`.
- Re-verification: blob-id fingerprints of the gate tree (`rsi/`,
  `python/pegaflow`, `docs/rsi/models/<model>`), venv `pegaflow`, the
  snapshot and the server binary, checked before every stage and after the
  last (`summary.provenance_watch`). Any change makes the run non-official
  (role, path, stage named); the run goes on. Not covered: other venv /
  system files, a change undone within one stage.
- `G0.provenance`: every spec dump records the module files of the probe,
  worker extension, vLLM, torch, `pegaflow` (+ native module,
  `site/usercustomize`), interpreter prefix, and `connector:<module>`.
  Probe / extension under the gate tree, vLLM / torch / prefix in the venv,
  `pegaflow` in the arm's expected dir. Connector root by arm kind, never by
  name: `selftest:*` gate tree, arm under test its snapshot (in `pegaflow`),
  `old` the venv.

## Boundary

The arm under test shares processes with `RsiProbe` / `RsiWorkerExt`
(`MultiConnector` imports it in the API server too), so it can patch the
instrument and nothing in-process can detect that. The gate bounds this:

- Isolation: the arm's vLLM and server exec through `rsi/gates/isolate.py`
  in a private mount namespace: runs root under an empty tmpfs with only the
  current stage dir (rw), the snapshot (ro) and its bytecode cache bound
  back (earlier stages and the gate dir stay hidden); block devices and `/dev/{mem,kmem,port}` masked; `CAP_SYS_ADMIN` /
  `CAP_SYS_PTRACE` / ... dropped from the bounding set. Host pid / IPC /
  network namespaces (CUDA IPC, loopback). `isolate.py` waits after
  `unshare` until the gate holds `/proc/<pid>/ns/mnt`. `summary.isolation`.
- Isolation residual (`summary.isolation.residual`), deterrence not a
  boundary: `setns` / reads via `/proc/<gate pid>/root` fail and the tmpfs
  cannot be unmounted (`test_isolated_arm_cannot_read_other_arms`), but the
  arm stays uid 0 with `CAP_DAC_OVERRIDE` and can write host files outside
  the runs root. Controls: re-verification, per-stage bytecode cache, scan,
  review.
- Static scan: `provenance.scan` matches the snapshot's text and native
  strings against `provenance.SCAN_PATTERNS`: instrument names (`rsi` as
  code, `rsi_probe`, `RsiProbe`, `RsiWorkerExt`, `rsi_poison`), run
  artifacts (`rsi-runs`, `probe_dir`, `digest-`, `-outputs.json`,
  `-tokens.json`, `requests.jsonl`), runtime patching (`sys.modules` writes,
  `gc.get_objects` / referrers, trace / audit hooks, `sys.meta_path` /
  `path_hooks`, `importlib.reload`). A hit is non-official unless on
  `provenance.SCAN_ALLOWLIST` (glob + pattern + reason; a gate change).
  `summary.scan.allowlisted` / `flagged`. Gaps, covered by review: frame
  walking, `__subclasses__()` discovery, computed imports and built strings,
  `ctypes` and native code.
- No plugins: every vLLM runs with `VLLM_PLUGINS=""`; arms name connectors
  by module path. Exception: `old` gets `VLLM_PLUGINS=pegaflow` (vLLM 0.32's
  API server rebuilds MultiConnector child stats by registered name; the
  plugin comes from the venv and only registers names); the arm under test
  never gets the plugin. A connector under test that reports stats must register its class under its class name on
  import, else requests fail with HTTP 500. `summary.vllm_plugins`.
- Independent signal: completions are read client-side from
  `/v1/completions` and judged in the gate; what the gate writes is in
  `<run>/.gate/<arm>/<workload>`, which no arm process gets. The stage dir is
  untrusted; a symlink or non-regular file there fails the stage. Faked
  digests with garbage loads still change completions.
- Review checklist (skill step 6): the connector does not touch the probe,
  the worker extension, `sys.modules`, the poison RPC or any run artifact,
  by code or by string.

## Record vs gate rev

A record is judged by the gate rev it ran under (`provenance.gate.rev`). On a
re-judge with a later rev, a check whose inputs the gate did not produce yet
is `{"pass": true, "na": "record predates this check"}` (with `would_be`):
`G0.vllm_args`, `G2.batch_control`, the self-test sabotage count; `G0.spec`
skips (`na_fields`) committed fields the record's probe schema (`probe.schema`,
else the newest present field) did not write. A known field missing, or any
field with another value, still fails.

## Run mechanics

- Teardown: each stage's processes run in their own process group; SIGTERM,
  wait 120 s, SIGKILL, wait 300 s; a survivor aborts the run
  (`TeardownError`, reference arms too). For the arm under test the same
  covers every process in its mount namespace (matched by namespace inode,
  signaled via pidfds).
- Ports: leased from 20000-29999 (`ip_local_reserved_ports`) under per-port
  `flock` files in `~/kexi/rsi-ports`, bind-checked; vLLM also gets 8
  contiguous ports as `VLLM_PORT`. EADDRINUSE before healthy fails the stage
  with `PortRace`, never retried: it names the ports parsed from the failing
  log lines plus any leased port another process holds once the stage
  process is dead (no port found: "unknown"), whether the gate leased each,
  the lease files and the failing line.
  Leases are released on every exit path; failed stages write
  `stage-error.json`; `summary.port_leases` lists all leases.
- Stopping: `kill -TERM -<run_gate pgid>` (or `-INT`, `-HUP`; launched with
  `setsid`). The run aborts: no later pass or stage starts, current groups
  are stopped as above (an `atexit` hook covers unowned groups), exit 128 +
  signal, no `summary.json`. Never SIGKILL the gate.

## Run record

`~/kexi/rsi-runs/<date>-<tag>/`: `summary.json`, raw logs, commit hash, vLLM
/ PegaFlow command lines. The model's iteration notes
(`rsi-notes:models/<model>/ITERATIONS.md`) link it.

## Implementation

| File | Role |
|---|---|
| `rsi_probe.py` | `RsiProbe` (never loads / saves), stacked after the arm in `MultiConnector`; G0 spec, trace, V3 digests |
| `digest_compare.py` | G2 byte comparison |
| `oracle.py` | G3 `H*`; G5 `U`, `H*_C` candidates, `H_lru`, `H_freeze` |
| `test_oracle_capacity.py` | oracle and gate contract on synthetic traces, no GPU |
| `test_probe_steps.py` | probe and self-test step logic (CPU; imports vllm, torch) |
| `test_committed_spec.py` | per-TP committed spec dump: selection, every consumer in `main`, non-official without one (no GPU) |
| `workloads.py` | gsm8k, replay, resume replay, Prometheus scrape |
| `run_gate.py` | orchestrator: arms x workloads, passes, verdicts |
| `provenance.py` | tree verification, snapshot, static scan + allowlist |
| `isolate.py` | mount-namespace isolation of the arm under test |
| `selftest/connector.py` | `honest` / `liar` / `bitflip` / `offbyone` / `sync` |

### Tests

Need `pytest`, `requests` and a built `pegaflow.semantics`:

```bash
cd python && uv sync --extra test                 # pytest, requests
uv pip install 'maturin>=1.0,<2.0'
uv run maturin develop -r                         # CUDA 13: add --no-default-features --features cuda-13
cd .. && PYTHONPATH=. python/.venv/bin/python -m pytest -q -p no:cacheprovider rsi/gates
```

About a minute on the 4090 host, nothing deselected (signal / teardown tests
use real subprocesses). PR CI does not run it, so it is a manual gate for
every change under `rsi/gates`. Isolation tests need Linux and root. The
shared venv on the GPU hosts is already built: only the last line is needed.

### Usage

Host trees carry no bytecode: `git archive <rev> | tar -x -C <dir>` plus
`echo <rev> > <dir>/.rsi-rev`, or `tar --exclude=__pycache__
--exclude='*.pyc'`; then build or copy `pegaflow<EXT_SUFFIX>` into
`python/pegaflow/`. On the 4090 host 10.1.0.44, from the feature dir:

```bash
CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
LD_LIBRARY_PATH=~/kexi/cuda13-lib \
~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate \
  --model ~/kexi/models/<model> --tag <model>-it<N> \
  --arms baseline,native,<module>:<Class> --workloads gsm8k,replay,cap --gpu 0 \
  --under-test-path <tree>/python
# add ,preempt to --workloads on models with a Mamba state group
# optional: --cap-curve (G5 curve 0.25/0.5/0.75, reported); old / lmcache in
#   --arms (lmcache needs --lmcache-overlay <gate.json lmcache_dist overlay:
#   ~/kexi/overlays/lmcache-pr4467 on Qwen3-4B>); plain (re-judges the
#   onboarding controls in this run)
# slice smoke (not official, no E_C conclusion):
#   ... --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --cap-convs 8 --cap-turns 6
```

### Self-test at onboarding

gsm8k, 128 questions per self-test arm (`SELFTEST_LIMIT`), the full set for
`plain` and `baseline`, no PegaFlow, once per TP the model is gated at
(`--tp N --gpu` with N devices), and again whenever the gate.json `controls`
record of a TP goes stale ("Onboarding controls"). The `plain` arm here is
where the onboarding controls run: when they pass, copy
`summary.controls_record` into gate.json `controls["tp<N>"]`. Add `preempt`
(on state-group models the honest arm fails without it) and `selftest:sync`
there:

```bash
... run_gate --model ... --tag <model>-selftest \
  --arms plain,baseline,selftest:honest,selftest:liar,selftest:bitflip,selftest:offbyone \
  --workloads gsm8k,preempt
```

`selftest:sync` applies only to models with a state group; drop it for
pure-attention models, where the connector refuses to start.

`preempt` needs the committed dump of `--tp`: for a TP without one, run
`--workloads gsm8k` first, commit its `baseline/gsm8k/probe/spec.json` as
`spec.json` (TP 1) or `spec-tp<N>.json`, then rerun with `preempt`.
