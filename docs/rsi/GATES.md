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

## G5 Capacity hit rate (hard, RSI objective; added 2026-10-23, status: implemented, see G5 under precise definitions)

User: actual hit rate has to approach the theoretical one at a fixed capacity.
G3 runs with an unlimited pool, so it cannot see eviction policy, write
amplification or cross-group coherence. Example: a checkpoint group's state at
token P is useless once the full-attention blocks before P are evicted, and
full blocks past the last resident checkpoint are useless too. The server's
LRU (`pegaflow-core/src/cache.rs`) evicts each (hash, group) key on its own,
so such stranded bytes are possible today.

- Workload `cap`: its own multi-turn replay, larger than `replay` so the
  store matters: `--cap-convs 64 x --cap-turns 12` conversations from the same
  r2egym file (official). The baseline arm runs it first (`cap_cold`,
  `cap_warm`, no store) to fix `U` and the per-request reference local hits.
  Each store arm (under test, `old`, `native`) then runs `cap_cold`,
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
  G3. This is a strong reference, not a proven optimum; `E_C > 1` is allowed.
- References `H_lru` (group-oblivious LRU, everything stored) and `H_freeze`
  (store until full, never evict): reported only, to show how far naive
  policies are from the bar.
- Score over the timed cap warm passes, against the baseline's local hit for
  the same request (`ref`), so an arm cannot raise its own score by keeping
  less in vLLM's local cache: `ext = max(0, min(actual, oracle) - ref)`
  (capped by the unlimited oracle), `ext*_C = max(0, oracle_C - ref)`,
  `E_C = Σext / Σext*_C`.
- Diagnostics, per group (not gated): resident bytes and stranded bytes at the
  end of each pass (resident blocks that lie on no resolvable prefix of any
  request in the trace), evictions, saved / loaded bytes. Implemented for the
  simulated stores; the server side reports eviction and save/load byte
  counters and its resident / used bytes at each pass end. TODO: actual
  stranded bytes need a server-side resident-key dump (server change, own PR).
- Gate: `G5.replay_warm.E_C >= 0.9` at `f = 0.5`; no load errors; the cap
  passes' digests pass `G2.digest_self` (eviction must never produce a wrong
  load); server load bytes cover every external hit; `excess` bounded as in G3
  against the unlimited oracle; no `E_C` regression vs the last accepted
  iteration. A slice-size G5 (below the capacity floor) is a smoke test of
  flow and digests only; its `E_C` supports no conclusion.
- Reference arms: `old` and `native` (vLLM `OffloadingConnector` with exactly
  `C` CPU bytes) record their G5 numbers and are never required; phase 2 will
  gate warm replay and `E_C` against `native`.
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
| `test_oracle_capacity.py` | G5 oracle contract on synthetic traces (`python -m pytest rsi/gates/test_oracle_capacity.py`, no GPU) |
| `workloads.py` | gsm8k, multi-turn replay, Prometheus scrape |
| `run_gate.py` | orchestrator: arms x workloads, cold/warm passes, verdicts |

| `selftest/connector.py` | gate self-test connector: `honest` / `liar` / `bitflip` / `offbyone` |

Usage (on the 4090 host 10.1.0.44, from the feature dir, shared venv):

```bash
CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
LD_LIBRARY_PATH=~/kexi/cuda13-lib \
~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate \
  --model ~/kexi/models/<model> --tag <model>-it<N> \
  --arms plain,baseline,old,native,<module>:<Class> --workloads gsm8k,replay,cap --gpu 0
# G5 smoke on a slice (below the capacity floor: not official, no E_C conclusion):
#   ... --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --cap-convs 8 --cap-turns 6
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
`plain`, `baseline` and one connector under test, all three workloads
(`gsm8k`, `replay`, `cap`; cap 64 x 12 at fractions 0.25,0.5,0.75), the G5
capacity floor (`C(0.5) >= 2 x` GPU KV bytes), no
`--num-gpu-blocks`/`--vllm-arg` override, and committed
`docs/rsi/models/<model>/{gate.json,spec.json}`. `summary.json` lists
`not_official_because`. A required verdict that cannot be evaluated is a fail.
Required verdicts: `baseline` and the connector under test, including
`G2.digest_self.cap` and `G5.replay_warm`. Not required (`required: false`,
every verdict recorded): `old` (black-box reference) and `native`. A crash of
a reference arm's stage is recorded under `errors` in its verdicts and the
run continues; a crash of baseline or the connector under test aborts it.

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
  `G2.digest_self.*` and `G2.gsm8k_outputs` are its correctness evidence), `selftest:<mode>`
  (gsm8k only), `<module>:<Class>`. `old`, `native` and the connector under
  test run as `MultiConnector[arm, RsiProbe]`.
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
  gsm8k and replay pass (the oracle has unlimited capacity; `cap*` passes are
  meant to evict and are excluded; `native` has no PegaFlow server: n/a, pass).
  The server runs with
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
- G5 (`oracle.simulate` / `summarize_capacity`, `run_gate` workload `cap`):
  - Store events, visibility and scoring are G3's; per (group, key) block
    bytes = `real_page_size_bytes` (unpadded, as digested; `page_size_bytes`
    if absent) x the group's `num_layers` (`UniformTypeKVCacheSpecs` already
    sums its layers: x 1) x TP ranks. Skipped groups (no kv transfer, eagle)
    hold no bytes. GPU KV bytes = spec `num_blocks` x Σ block bytes over all
    groups.
  - Trace: `--cap-convs` x `--cap-turns` (official 64 x 12) conversations of
    `--replay-file`, `--replay-out-tokens` outputs, `--concurrency` workers.
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
    otherwise: fresh vLLM and (except `native`) a fresh PegaFlow server with
    `--pool-size C` `--disable-numa-affinity`, `C = int(f x U)`; passes
    `cap_cold`, `cap_warm` (digested), `cap_warm2..N` (timed), the replay
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
    - `H_lru`: group-oblivious LRU, every item stored, a re-store of a
      resident item does not refresh it, a hit refreshes the blocks loaded.
    - `H_freeze`: items stored in arrival order until the next one does not
      fit, never evicted.
    - Hits: `resolve` over resident blocks plus each group's `needed` range
      at `ref`, capped at `max(unlimited oracle, ref)`.
  - Scores per segment (`warm_timed` = `cap_warm2..N`, and per pass):
    `ext = max(0, min(actual, oracle) - ref)` (unlimited oracle),
    `ext*_C = max(0, oracle_C - ref)`, `E_C = Σ ext / Σ ext*_C` (`None` when
    `Σ ext*_C = 0`: fails). Reported: `lru_vs_oracle_c` and
    `freeze_vs_oracle_c` (`Σ max(0, H - ref) / Σ ext*_C`),
    `efficiency_c_own_local` (the same `E_C` with the arm's own local hit in
    place of `ref` for `ext`, `ext*_C` and the Belady run; diagnostic only),
    `efficiency_c_cold` / `freeze_vs_oracle_c_cold` (`cap_cold`: within-pass
    multi-turn reuse after vLLM evicted the earlier turn; diagnostic only).
    `unmatched_requests` = requests without a `ref` (no position tag, or the
    baseline has no request at that position).
  - Load proof: `external_hit_bytes(pass)` = Σ over the pass's requests of the
    bytes of the blocks `needed` at `actual` that `needed` at the request's
    own local hit does not hold; `loaded_bytes(pass)` = the
    `pegaflow_load_bytes_total` delta of the pass.
  - `G5.replay_warm` (gated fraction 0.5): `E_C >= 0.9` on `warm_timed`,
    `excess <= one block per request` (unlimited oracle, own local, as G3),
    no load errors, `unmatched_requests == 0`, `G2.digest_self.cap` passes
    (same as `G2.digest_self.replay`, over the cap probe dir), and for a
    PegaFlow arm `pool_capacity_bytes` present and `<= C` and Σ
    `loaded_bytes >= Σ external_hit_bytes` over `warm_timed` (`native`: no
    server, pool and load proof not applicable). Reported with it: `U`, `C`,
    hit / oracle / oracle_C / LRU / freeze hit rates, per-pass eviction
    counter deltas, per-pass server I/O (`saved_bytes`, `loaded_bytes`,
    `external_hit_bytes`) and end-of-pass resident / used bytes, per-group
    stored / not-admitted / evicted counts and bytes and per-pass resident /
    stranded bytes for `H*_C` and `H_lru`.
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
    test at the gated fraction is `>= 0.9`, the trace does not separate a
    frozen store from the bar and the run is not official
    (`not_official_because: G5 trace does not separate freeze from H*_C`).
- Self-test arms (`rsi/gates/selftest/connector.py`, full-attention groups
  only): `honest` saves at finish and loads saved prefixes (must pass every G2
  check); `liar` claims every complete unit and loads nothing; `bitflip` flips
  one byte of the first loaded block; `offbyone` loads block i+1 into slot i.
  Each negative arm must fail both G2 digest checks; output deltas are reported
  only (one flipped byte need not change a greedy completion).
