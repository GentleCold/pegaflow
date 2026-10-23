# Qwen3-4B onboarding iterations (connector_v2)

Spec: `spec.json` (vLLM 0.32.0, TP1, 4090). One transferable group:
`FullAttentionSpec`, 36 layers, block_size 16, bf16, 8 kv heads x 128,
`page_size_bytes == real_page_size_bytes == 65536`, no padding/quant/window,
`kv_cache_layout = LBNHC`, one `KVCacheTensor` with `block_stride = 65536`
(= page) and `layer_stride = num_blocks * page`, hash_block_size 16.

## it1 (2026-10-23): first adapter

Written in isolation. Sources used: the spec dump, the vLLM v0.32.0 source
(fetched from GitHub), `pegaflow.semantics`, the proto/PyO3 client and the
generic `pegaflow-core/src/layout.rs` / `pegaflow-server/src/registry.rs`.

### Design

- **Group**: `Group.full(block_size)`. Qwen3 attends every past token, so
  resuming at `p` needs blocks `[0, p/16)`. Semantics already covers this, so
  no Kind change was needed.
- **Segments**: one segment per block = the whole 65536 B page. The probe's
  layer view is `[B, H=8, N=16, C=256]` with strides `[32768, 256, 2048, 1]`
  (elements), so the page is `[N][H][K128|V128]`. K and V are interleaved per
  (token, head), and two K/V segments would split each block in the wrong
  places. One page is also the unit the attention kernel reads, so a stored
  block round-trips byte for byte.
- **Server addressing**: the gRPC register path takes no block stride, so the
  server addresses block `b` at `data_ptr + b * bytes_per_block` (contiguous,
  one segment). The adapter therefore registers `bytes_per_block =
  kv_stride_bytes = page`. `view()` proves the geometry instead of assuming
  it:
  - the per-block `[H, N, C]` dims tile a dense range;
  - `N` divides `block_size` (kernel-block splitting);
  - the kernel-block cells times the ratio equal the page;
  - dim 0 strides exactly one kernel block.

  Anything else (block-outer layouts such as BLNHC, padded pages, holes) is
  `Unsupported` at init. Silently wrong addressing is not possible.
- **Shard**: `Heads(tp_rank // replicas, shards)`, where shards =
  `min(tp, max_tp_shards or tp)`. TP1 gives `Heads(0, 1)`. Ranks beyond
  `max_tp_shards` hold identical bytes, so `seal_plan` dedupes them. DCP > 1
  is `Unsupported` until a DCP model is onboarded (Tokens shard semantics are
  not pinned down yet).
- **Fail closed** on spec fields this adapter has not been validated on:
  - `sliding_window`, `attention_chunk_size`;
  - `page_size_padded`, `num_head_slots`, `state_content_bytes`;
  - `tokens_per_state != 1`, `kv_quant_mode != 0`;
  - `has_layer_views == False`.

  Each of these needs its own declaration when a model brings it. For
  example, a FullAttentionSpec carrying `sliding_window` (hybrid allocator
  off) is probably still Full for storage, but this is unverified.
- **Policy**: no `models/Qwen3ForCausalLM.toml`. The defaults (min_hit_tokens
  0, loading_wait_ms 1000) are what this model needs, and a profile file
  would only repeat them.
- **Core**: checked against vLLM 0.32 and found compatible, so there are no
  core changes:
  - `kv_connector_block_state.get_block_ids`;
  - `get_transfer_results` / `failed_recving` with recompute;
  - MultiConnector routing: the first child with a hit wins, and the others
    get `update_state_after_alloc(.., 0)`, which releases their leases;
  - `get_loaded_kv_cache_group_ids`;
  - an empty extra config;
  - `PEGAFLOW_PORT` via `endpoint_of`.

### Changes

- `python/pegaflow/connector_v2/adapters/full_attention.py` (new), registered
  as `"FullAttentionSpec": "full_attention"` in `adapters/__init__.py`.
- `python/tests/test_connector_v2_core.py`: adapter section, all cases run on
  both the stub and the native semantics:
  - **view geometry**: Qwen3 LBNHC, LBHNC, kernel-split accepted; BLNHC,
    holes, wrong page rejected;
  - **layout + seal_plan**: TP1, TP2, TP4 with `max_tp_shards=2` replicas;
  - **fail-closed fields**: each unvalidated spec field is rejected;
  - **Qwen3-4B registration**: the 36-layer worker registration as the server
    sees it.
- `connector_v2/README.md`: registered-adapter table.

### Evidence

- Local (Mac, source-only, stub semantics): `test_connector_v2_core.py`
  66 passed, 41 skipped (native cases need the extension). ruff format/check
  clean.
- GPU validation: NOT run. 10.121.196.3 was unreachable from the jump host
  ("No route to host") at both the start and the end of this iteration.
- Verdict table: none yet.

### Next on .3 (when reachable)

1. Sync, write `.rsi-rev`, and rerun `test_connector_v2_core.py` with the
   shared venv. This covers the native cases.
2. Slice gate:
   `run_gate --model ~/kexi/models/Qwen3-4B --tag qwen3-4b-p3-it1-slice --arms plain,baseline,pegaflow.connector_v2:PegaKVConnectorV2 --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --gpu 1`.
3. Only if every slice verdict passes, run the full official run with the
   same arms and no limits.

### Open risks

- **No end-to-end evidence yet.** Registration size checks are proven only
  against the server's layout rules, read from source; the CUDA IPC path has
  not been exercised. The first gate run may expose:
  - a `size_bytes` / bounds issue;
  - the server's handling of `segments=1` with `kv_stride == bytes_per_block`.
- **G4**: saves are synchronous in `wait_for_save` (deferred S5). The replay
  warm TTFT should still beat baseline because loads skip prefill, but cold
  passes pay the copy inline. Async saves are the likely it2 topic if G4
  fails.
- **Spurious warning under MultiConnector**: vLLM builds each child's
  `KVTransferConfig` from the child dict only, so the child sees
  `kv_load_failure_policy="fail"` (the default) even when the parent says
  `recompute`. The scheduler uses the parent's value, so behaviour is
  correct, but the connector logs a spurious "use recompute" warning. If the
  gate wants the child to see it, `run_gate` should also put
  `kv_load_failure_policy` into the child dict (gate-owned, not changed
  here).
- **Hash trimming**: the scheduler queries at most `(num_tokens-1)//16`
  blocks. This is correct for Full (the last token is always recomputed),
  but a prompt of exactly `16k + 1` tokens gets its full `k` blocks, while
  one of `16k` tokens loses its last block by design.

### Gate concerns (not edited; for the gate owner)

- See the MultiConnector child-config note above. G3 is unaffected.

## it1b (2026-10-23): review fixes

Review: `docs/rsi/reviews/P3-qwen3-4b-it1.md` (APPROVE-WITH-FIXES). Its
Resolution table has the per-finding detail.

### Changes

- **M1:**
  - the namespace now includes vLLM's `kv_cache_layout` (`core/layout.py`);
  - `full_attention.view` rejects kernel-split HNC pages, the one byte order
    the layout name does not capture;
  - README namespace text corrected.
- **S1:** `non_causal` specs fail closed.
- **S2:** `Worker.save_stats`, logged as `save stats` every 500 steps and at
  shutdown. This is the it2 input for the async-save decision. `connector.py`
  routes `pegaflow.connector_v2.*` INFO to stderr: vLLM only configures its own
  `vllm` logger, so core INFO lines were silently dropped before.
- **n1:** the child-config recompute warning is now an accurate INFO line.
- **n2:** `Parallel.kv_tp_replicas`, cross-checked by the adapter.
- **n3:** registration assertions merged into `test_full_attention_layout`.
- **n4:** per-rank pages in the TP fixtures.
- **n5:** the test no longer imports `_tp_ranks`.
- **n6:** comment on the `kv_quant_mode` comparison.

### Evidence

- Local stub run: 69 passed, 43 skipped. ruff clean.
- .3 is still unreachable, so there is no GPU verdict yet.

### What to read from the first slice run

- **Save cost:** `grep "save stats" <arm>/<workload>/vllm.log`. A high
  `avg_save_step_ms` relative to the decode step time, or `save_steps`
  approaching `steps` on replay, means async saves (S5) are it2.
- **Load-failure policy:** the line "own kv_load_failure_policy=fail; the
  effective policy is the top-level ..." is expected under the gate's
  MultiConnector and is not an error.

### Open risks (delta)

- The `kv_cache_layout` namespace input relies on vLLM 0.32 setting it on both
  roles (verified in source: `engine/core.py` sets it on every config before
  the scheduler copy). If a future vLLM leaves it None on one role, the
  scheduler and worker namespaces diverge and registration or queries mismatch
  loudly (no hits), not silently.

### it1b slice gate result (.44, 2026-10-23)

Run `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it1-slice` (64 gsm8k, 8 convs x 6
turns; not official). All verdicts pass.

| Check | Value |
|---|---|
| G0.spec | 0 diffs |
| G2 digest external (gsm8k / replay) | 1492/1492, 617/617 matched; 0 mismatched |
| G2 gsm8k | 0.8594 baseline = cold = warm; outputs identical |
| G3 gsm8k warm | H 0.978, oracle 0.992, E_ext 1.0, excess 0 |
| G3 replay warm | H 0.991, oracle 0.998, E_ext 1.0, excess 0 |
| G3 no_eviction / load errors | pass / 0 |
| G4 replay warm TTFT p50 | 0.101-0.117 s vs baseline 0.209 s |
| replay cold TTFT p50 | 0.266 s vs baseline 0.211 s |

Save stats (gsm8k cold window): 151 of 500 steps save, ~21.5 ms avg (max 34 ms)
per save step, sync 3175 ms vs rpc 77 ms, 0 failed. The synchronous D2H wait
is the whole cost; it explains the +26% cold TTFT. Candidate for it2: S5
async saves, judged by the official run's cold numbers.

Next: official run `qwen3-4b-p3-it1` on GPU1 (in flight). If it passes, mark
it1 accepted and start it2 (async save) or move on to P4.

### it1b under the hard G4 (2026-10-23)

G4 became a hard gate (GATES.md). Re-judging the slice above: warm replay
passes both checks; `G4.replay_cold.ttft_p50` (+26%), `G4.replay_cold.req_per_s`
(-8%), `G4.gsm8k_cold.seconds` (+4.1%) and `G4.gsm8k_warm.seconds` (+3.2%) fail
the 2% overhead bound. it1b is not accepted.

### it1 official run under the hard G4 (2026-10-23)

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it1` (official, GPU1, connector code
of 41fb158). Judged at run time with the old G4: every verdict passed (G0, all
G2 digests 0 mismatch, gsm8k accuracy 0.858 = baseline, G3 oracle hit 0.992 /
0.998, no eviction). Re-judged offline with `_perf_verdicts`:

| check | v2 | baseline | verdict |
|---|---|---|---|
| replay warm TTFT p50 (warm2/3) | 0.515 / 0.524 s | 2.534 / 2.543 s | pass (4.9x) |
| replay warm req/s | 9.60 / 9.55 | 3.44 / 3.43 | pass (2.8x) |
| replay cold TTFT p50 | 1.333 s | 2.797 s | pass |
| replay cold req/s | 6.15 | 3.29 | pass |
| gsm8k cold seconds | 175.4 | 162.0 | FAIL (+8.3%) |
| gsm8k warm seconds | 174.0 | 162.0 | FAIL (+7.4%) |

At 32x12 the replay working set exceeds vLLM's GPU cache, so even the cold
pass reloads earlier turns from PegaFlow and beats baseline; the 8x6 slice was
too small to show this. The remaining failure is per-step save overhead on the
short-prefix gsm8k workload, which is it2's target. it1 is not accepted.

Hit numbers (official): gsm8k warm oracle hit 0.992, replay warm oracle hit
0.998, `E_ext` 1.0 on both, excess within one block per request, no eviction,
digests 0 mismatched (gsm8k 6773 external blocks, replay 18286).

Save stats from the official vLLM logs (`save stats` lines, 500-step windows):

| workload | window | save steps | sync ms | rpc ms | avg ms / save step |
|---|---|---|---|---|---|
| gsm8k | 19.0 s | 429/500 | 10474 | 293 | 25.1 |
| gsm8k | 19.3 s | 419/500 | 9929 | 287 | 24.4 |
| gsm8k | 17.7 s | 314/500 | 8719 | 212 | 28.4 |
| replay | 13.2 s | 281/500 | 8534 | 199 | 31.1 |
| replay | 19.6 s | 325/500 | 8818 | 217 | 27.8 |

The device sync is ~50% of wall time in each window and the RPC ~1.5%, so the
synchronous wait in `Worker.save` explains the gsm8k overhead; it2 removes it.
Reference: the black-box `old` arm (p0-final) is within +0.4% of baseline on
gsm8k cold, so the 2% bound is reachable.

## it2 (plan): no synchronous save wait

Hypothesis: the `wait_for_save` path blocks the step on a full device sync
(3.2 s of a 13.8 s gsm8k window, rpc only 77 ms), so cold steps pay the copy
inline. Making the save asynchronous with respect to the model step should
bring cold replay and gsm8k within 2% of baseline without touching warm hits.
Accept only if every G2/G3 check still passes and the full G4 set passes.


## G5 official (r3, 2026-10-23)

Record `~/kexi/rsi-runs/2026-10-23-qwen3-4b-g5-official-e9f3f33-r3` (.44 GPU5,
18:53:55 to 22:00 UTC). `official: true`, `not_official_because: []`.

- Gate rev e9f3f33 (tree `~/kexi/pegaflow-rsi-g5-e9f3f33`, `.rsi-rev`).
- Connector under test: `pegaflow.connector_v2:PegaKVConnectorV2` at
  8a05b7c (`--under-test-path ~/kexi/pegaflow-rsi-g5-7bf8076-ut/python`, a
  `PYTHONPATH` prepend; the run predates the overlay). The tree was checked
  afterwards with `provenance.Trees` (144bfd8): it matches 8a05b7c
  blob-for-blob (63 files).
- Venv `pegaflow` (`old` arm, oracle): 056eb4b at
  `~/kexi/pegaflow-semantics-rsi/python`, also verified.
- pegaflow-server sha256 `d1970ead…c579d7`; its tree claims 056eb4b.
- Run settings: `--pool-size 160gb` (pre-64gb default), cap `auto` (64
  conversations), fractions 0.25/0.5/0.75.
- The run predates 91b6a8e..bf9ef3c, so `summary.provenance`, the warmup
  request and `G2.digest_groups` are absent. None of them changes a Qwen3-4B
  verdict: Qwen3-4B is split-invariant (FLASH_ATTN, `G2.split_control`
  passes), and it has one full-attention group, which is digested.
- Contamination, checked offline with the 325fbd0 rule (timed replay p99 TTFT
  `> 10x` the baseline's p99 in the same pass): none. The largest ratio is
  native `cold` (20.68 s vs 6.38 s, 3.2x); every other arm is below 1x.

Verdict: Qwen3-4B accepted on G0-G4. G5 fails: E_C(0.5) is 0.684 vs 0.9,
which is at LRU level (H_lru/H*_C 0.859).

### Uncapped hit table (tokens)

gsm8k: prompt 1,469,407 tokens per pass; H* is 0.9329 cold, 0.9925 warm.
Accuracy is 0.8582 in every pass of every arm.

| arm | pass | local | ext | total | hit | E_ext |
|---|---|---|---|---|---|---|
| baseline | warm | 1,370,784 | 0 | 1,370,784 | 0.9329 | 0.0 |
| old | warm | 1,361,424 | 98,098 | 1,459,522 | 0.9933 | 1.0 |
| native | warm | 1,370,784 | 88,738 | 1,459,522 | 0.9933 | 1.0 |
| v2 | warm | 1,338,544 | 119,808 | 1,458,352 | 0.9925 | 1.0 |

Cold equals baseline cold (1,370,784 local) for every arm.

Replay: prompt 1,275,208 tokens per pass; H* is 0.8641 cold (native 0.8631),
0.9983 warm.

| arm | pass | local | ext | total | hit | E_ext |
|---|---|---|---|---|---|---|
| baseline | cold / warm / warm2 / warm3 | 482,624 / 482,624 / 481,216 / 481,216 | 0 | same | 0.3785 / 0.3785 / 0.3774 / 0.3774 | 0.0 |
| old | cold | 488,272 | 613,664 | 1,101,936 | 0.8641 | 1.0 |
| old | warm / warm2 / warm3 | 456,672 / 475,248 / 454,224 | 816,515 / 797,939 / 818,963 | 1,273,187 | 0.9984 | 1.0 |
| native | cold | 489,904 | 612,032 | 1,101,936 | 0.8641 | 1.0 |
| native | warm / warm2 / warm3 | 482,512 / 481,856 / 481,856 | 790,675 / 791,331 / 791,331 | 1,273,187 | 0.9984 | 1.0 |
| v2 | cold | 481,648 | 620,288 | 1,101,936 | 0.8641 | 1.0 |
| v2 | warm / warm2 / warm3 | 443,216 / 443,168 / 443,168 | 829,776 / 829,824 / 829,824 | 1,272,992 | 0.9983 | 1.0 |

### G5 (cap trace, 64 conversations x 12 turns)

- U (measured on the baseline's `cap_cold`) is 54.43 GB; the cap_plan
  estimate was 54.43 GB against the 49.27 GB needed.
- GPU KV is 11.16 GB.
- Capacity floor passes: C(0.5)/GPU-KV is 2.44, above the 2.0 minimum.
- The gate is E_C(0.5) >= 0.9; 0.25 and 0.75 are reported only.
- H*_C, H_lru and hit are warm-timed hit rates. H*_C and H_lru are computed
  on each arm's own trace, so they differ slightly between arms.

| C/U | C (GB) | C/GPU-KV | arm | H*_C | H_lru | hit | E_C |
|---|---|---|---|---|---|---|---|
| 0.25 | 13.61 | 1.22 | v2 | 0.8645 | 0.5814 | 0.3987 | 0.084 |
| 0.25 | 13.61 | 1.22 | old | 0.8654 | 0.5815 | 0.3952 | 0.069 |
| 0.25 | 13.61 | 1.22 | native | 0.8653 | 0.5700 | 0.5285 | 0.287 |
| 0.5 | 27.22 | 2.44 | v2 | 0.9290 | 0.8569 | 0.8420 | **0.684** |
| 0.5 | 27.22 | 2.44 | old | 0.9300 | 0.8575 | 0.8324 | 0.681 |
| 0.5 | 27.22 | 2.44 | native | 0.9292 | 0.8579 | 0.8614 | 0.715 |
| 0.75 | 40.83 | 3.66 | v2 | 0.9707 | 0.9155 | 0.8758 | 0.669 |
| 0.75 | 40.83 | 3.66 | old | 0.9708 | 0.9153 | 0.8764 | 0.682 |
| 0.75 | 40.83 | 3.66 | native | 0.9711 | 0.9146 | 0.9269 | 0.770 |

v2 at C/U 0.5:

| metric | value |
|---|---|
| E_C cold | 0.971 |
| E_C own-local | 0.851 |
| H_lru/H*_C | 0.859 |
| freeze/H*_C | 0.623 (the trace separates a frozen store from H*_C) |
| evictions per pass | 14k-25k blocks |
| pool used at pass end | at most 26.4 of 27.2 GB |
| load errors / excess tokens | 0 / 0 |
| loads_ok, pool_ok | true |

`G2.digest_self.cap` passes for v2 at every fraction.

### G4 (vs the in-run baseline; medians)

Slack: replay warm must be strictly better than baseline; replay cold and
gsm8k may be at most 2% worse.

| check | baseline | v2 | old | native |
|---|---|---|---|---|
| replay warm TTFT p50 (s) | 2.780 / 2.785 | 0.538 / 0.542 | 0.606 / 0.633 | 0.499 / 0.491 |
| replay warm req/s | 3.169 / 3.166 | 10.033 / 10.023 | 10.056 / 10.233 | 5.855 / 9.767 |
| replay cold TTFT p50 (s) | 2.973 | 1.254 | 1.340 | 1.537 |
| replay cold req/s | 3.113 | 6.771 | 6.247 | 4.273 |
| gsm8k cold (s) | 164.2 | 165.6 | 166.2 | 168.2 (fail) |
| gsm8k warm (s) | 164.6 | 164.4 | 169.7 (fail) | 174.6 (fail) |

Replay p99 TTFT (s), cold / warm / warm2 / warm3:

| arm | p99 TTFT |
|---|---|
| baseline | 6.38 / 6.52 / 6.25 / 6.27 |
| v2 | 2.99 / 2.14 / 1.81 / 1.82 |
| old | 5.97 / 2.64 / 2.20 / 1.95 |
| native | 20.68 / 2.38 / 19.96 / 2.33 |

Verdicts:

| arm | result | failing checks |
|---|---|---|
| baseline | passes | none |
| v2 (required) | fails | `G5.replay_warm` only |
| old | fails (not required) | `G4.gsm8k_warm.seconds`, `G2.digest_self.cap`, `G5` |
| native | fails (not required) | `G0.spec` (LBNHC vs LBHNC layout), `G2.digest_vs_baseline.gsm8k`, G4 gsm8k, `G5` |
