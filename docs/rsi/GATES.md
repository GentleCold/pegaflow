# RSI gates (human-owned)

Agents may add checks or tighten thresholds here; they may not remove or relax
anything. Changes to this file need the review agent's sign-off and a log line in
`PROGRESS.md`. Status: DRAFT until the user confirms the metric set.

A model is "supported" only when every hard gate passes on that model and no
earlier model regresses.

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
   - warm external hit rate must be > 0.9 of oracle, otherwise the accuracy
     number says nothing about loaded KV.

## G3 Hit rate (hard on regressions, primary RSI objective)

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

Gate: `E >= 0.95` on the replay workload at the configured capacity, and no
model's `H` drops vs. its last accepted iteration. Old connector's `H` on the same
trace is recorded as a black-box reference number only.

## G4 Performance (hard gate, RSI objective)

Workload: multi-turn agent traces (user-provided "agentx" — TBD; fallback
`/ppio/data_sets/R2EGym-SFT-Trajectories` replayed as multi-turn chat with
growing context), plus gsm8k few-shot as a short-prefix workload.

Metrics: TTFT p50/p99, end-to-end throughput (req/s, output tok/s), load and
save bandwidth (GB/s), connector scheduler-side CPU time per step, GPU memory
overhead. Compared against baseline vLLM (no PegaFlow) and the old connector.

Gate (hard): warm replay TTFT p50 and throughput must beat baseline vLLM;
cold replay and gsm8k may be at most 2% worse than baseline (see G4 under
precise definitions); no >5% regression vs. last accepted iteration of the
same model.

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
| `oracle.py` | G3 `H*` replay through `semantics.resolve` |
| `workloads.py` | gsm8k, multi-turn replay, Prometheus scrape |
| `run_gate.py` | orchestrator: arms x workloads, cold/warm passes, verdicts |

| `selftest/connector.py` | gate self-test connector: `honest` / `liar` / `bitflip` / `offbyone` |

Usage (on the 4090 host 10.1.0.44, from the feature dir, shared venv):

```bash
CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
LD_LIBRARY_PATH=~/kexi/cuda13-lib \
~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate \
  --model ~/kexi/models/<model> --tag <model>-it<N> \
  --arms plain,baseline,old,<module>:<Class> --workloads gsm8k,replay --gpu 0
```

Self-test at onboarding (gsm8k only, 128 questions per self-test arm, no PegaFlow):

```bash
... run_gate --model ... --tag <model>-selftest \
  --arms plain,baseline,selftest:honest,selftest:liar,selftest:bitflip,selftest:offbyone \
  --workloads gsm8k --gsm8k-limit 128
```

Exit code 0 only for an **official** run in which every required verdict passes.
A run is official only with the constants of `run_gate.OFFICIAL` (full gsm8k, 8
shots, 32 x 12 replay, 3 replay passes, pinned scheduler flags), the arms
`plain`, `baseline` and one connector under test, both workloads, no
`--num-gpu-blocks`/`--vllm-arg` override, and committed
`docs/rsi/models/<model>/{gate.json,spec.json}`. `summary.json` lists
`not_official_because`. A required verdict that cannot be evaluated is a fail.

Exit 0 does **not** cover G1 (run `cargo test -p pegaflow-semantics`) or the
"no regression vs last accepted iteration" clauses of G3/G4: compare against
the last accepted run record by hand until `--accepted` exists.

### Precise definitions (these override the prose above where narrower)

- Arms: `plain` (no kv-transfer-config, gsm8k only), `baseline` (probe only),
  `old` (black-box reference), `selftest:<mode>` (gsm8k only), `<module>:<Class>`.
  The connector under test runs as `MultiConnector[under_test, RsiProbe]` with
  `kv_load_failure_policy=recompute`, reads its server port from
  `PEGAFLOW_PORT`, and gets an empty `kv_connector_extra_config`. The probe
  keeps `prefer_cross_layer_blocks=False`, so cross-layer KV layouts are out of
  scope for now.
- Per-model settings: `docs/rsi/models/<model>/gate.json` (gate-owned):
  `batch_invariant`, `invariant_attention_backend`, `attention_backend`,
  `vllm_args` (e.g. `mamba_cache_mode`). Pinned for every arm:
  `--max-num-batched-tokens 2048 --max-num-seqs 256 --enable-prefix-caching
  --seed 42 --generation-config vllm`, `PYTHONHASHSEED=0`.
- Passes: each arm x workload gets fresh vLLM (+ fresh PegaFlow for `old` and
  under-test arms). `cold`, then per warm pass: sleep `--settle` s,
  `POST /reset_prefix_cache`, then `POST /collective_rpc {"method": "rsi_poison"}`
  (`RsiWorkerExt`, fills every byte of every KV allocation with 0x3F, a finite
  bf16/fp16 value), then the pass. Warm reuse can only come from the
  connector; stale cold-pass bytes cannot rescue a connector that loads nothing.
  gsm8k: `cold`, `warm`. Replay: `cold`, `warm` (digested), `warm2..warmN`
  (timed, N = `--replay-repeats`). Each request carries
  `vllm_xargs.rsi_pass = <pass>`; the oracle scores by that tag.
- Probe trace per request: `episodes = [step, local_hit, start]` per schedule
  episode (a resume after preemption opens a new one), `progress = [step,
  computed_after]` whenever a step crosses a key-token boundary. `local_hit` is
  the value passed to `get_num_new_matched_tokens` at the last attempt before
  the episode was scheduled; `first_computed` = `start` of the first episode.
  Load failures (`invalid_block_ids`, `failed_recving`) go to
  `load_errors.jsonl`.
- Digest (V3): requests tagged `vllm_xargs.rsi_digest = full` (gsm8k: all;
  replay `warm`/`cold`: every 4th conversation, all of its turns) are digested
  at finish: every fully computed block of every transferable group, per layer,
  on the raw allocation with `KVCacheTensor` geometry, unpadded page bytes (two
  64-bit position-weighted sums + float L1). `register_kv_caches` checks that
  geometry against every layer view and raises on mismatch. Tags, from the
  request's last episode: `local` (`covered <= local_hit`), `external`
  (`<= start`), `computed`.
- Digest comparison (`digest_compare.compare(REF, CAND, invariant)`): a loaded
  block (local or external) is `matched` if all its layers equal some computed
  copy of the same (rank, group, prefix key) in REF; `mismatched` if it equals
  none and the reference is exhaustive (batch-invariant, or every computation
  of that key in REF was digested); `uncertain` otherwise; `unreferenced` if REF
  never computed the key.
- G0.spec: the arm's scheduler spec dump and per-layer view layouts (shape
  without the block dim, stride, dtype) equal the baseline's, and the
  baseline's equals the committed `spec.json`. Volatile fields (`num_blocks`,
  tensor `size`, `layer_stride`, model path, `max_model_len`) are excluded.
- G2 controls (baseline arm): `plain_equals_baseline` (gsm8k cold completions
  identical: the probe changes nothing), `warm_equals_cold` (under invariance:
  reset + poison change nothing), `digest_self.gsm8k` (no local mismatch).
- G2.digest_self.gsm8k / digest_vs_baseline.gsm8k: local and external
  `mismatched == 0`, `unreferenced == 0`, `external_matched > 0`, judged
  (matched + mismatched) >= 0.9 of all external blocks, and under invariance
  `computed_exact == computed_compared`. G2.digest_self.replay: same, with no
  judged-fraction floor (untagged conversations share prefixes).
- G2.gsm8k: `|acc(pass) - acc(baseline cold)| <= 0.01` for cold and warm.
  G2.gsm8k_outputs (when gate.json declares invariance): every completion equals
  baseline cold, cold and warm.
- G3 (`oracle.py`): replays `requests.jsonl` with unlimited capacity. A block
  ending at token e is stored at the first step whose `progress` reaches e
  (checkpoint groups: only a step ending exactly on e) and is visible to
  requests first scheduled after that step. Kinds come from the spec dump
  (no window -> Full, sliding window sw -> Window{sw-1}, sw == 1 -> Scratch,
  Mamba -> Checkpoint, `UniformTypeKVCacheSpecs` -> its inner Kind, unknown ->
  error). Per request: `ext = max(0, min(actual, oracle) - local)`,
  `ext* = max(0, oracle - local)`; `E_ext = Σext / Σext*`;
  `excess = Σ max(0, actual - max(oracle, local))`. G3.gsm8k_warm and
  G3.replay_warm: `E_ext >= 0.95` (None fails), `oracle_hit_rate >= 0.5`,
  `excess <= one block per request`, no load errors; replay G3 also requires
  G2.digest_self.replay.
- G3.no_eviction: `pegaflow_cache_block_evictions_total` delta = 0 in every
  pass (the oracle has unlimited capacity). The server runs with
  `--pool-size 160gb --disable-numa-affinity` (per-NUMA pools give one GPU only
  `pool_size / numa_nodes`).
- G4 (hard since 2026-10-23, user: "性能需要达标才行，需要比基准快"; every
  check is against the baseline arm on the same run, medians over the listed
  passes, `MAX_OVERHEAD = 0.02`):
  - `G4.replay_warm.ttft_p50` / `.req_per_s`: timed warm passes (warm2..N)
    strictly better than baseline's.
  - `G4.replay_cold.ttft_p50` / `.req_per_s`: cold pass at most 2% worse than
    baseline's cold (nothing to load yet; vLLM's own prefix cache serves the
    earlier turns, so cold can only tie; this bounds the save cost).
  - `G4.gsm8k_cold.seconds` / `G4.gsm8k_warm.seconds`: wall time at most 2%
    worse than baseline (the shared few-shot prefix stays in vLLM's local
    cache, so gsm8k bounds per-step connector overhead).
  - Replay G4 checks are void (fail) if G2.digest_self.replay fails. p99 values
    are recorded in pass results but not gated.
- Self-test arms (`rsi/gates/selftest/connector.py`, full-attention groups
  only): `honest` saves at finish and loads saved prefixes (must pass every G2
  check); `liar` claims every complete unit and loads nothing; `bitflip` flips
  one byte of the first loaded block; `offbyone` loads block i+1 into slot i.
  Each negative arm must fail both G2 digest checks; output deltas are reported
  only (one flipped byte need not change a greedy completion).
