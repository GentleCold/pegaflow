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

## G4 Performance (soft gate, RSI objective)

Workload: multi-turn agent traces (user-provided "agentx" — TBD; fallback
`/ppio/data_sets/R2EGym-SFT-Trajectories` replayed as multi-turn chat with
growing context), plus gsm8k few-shot as a short-prefix workload.

Metrics: TTFT p50/p99, end-to-end throughput (req/s, output tok/s), load and
save bandwidth (GB/s), connector scheduler-side CPU time per step, GPU memory
overhead. Compared against baseline vLLM (no PegaFlow) and the old connector.

Gate: warm TTFT p50 must beat baseline vLLM; no >5% regression vs. last accepted
iteration of the same model.

## Run record

Every gate run writes `~/kexi/rsi-runs/<date>-<model>-<iter>/` with
`summary.json` (all numbers above), raw logs, the commit hash, and the vLLM /
PegaFlow command lines. `PROGRESS.md` links the summary.
