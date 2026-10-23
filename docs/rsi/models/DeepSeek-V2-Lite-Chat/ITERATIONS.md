# DeepSeek-V2-Lite-Chat onboarding iterations (connector_v2)

Spec: `spec.json` (vLLM 0.32.0, TP2 on two 4090s, `--gpu-mem 0.8`). One
transferable group:
- `MLAAttentionSpec` (MRO `MLAAttentionSpec > FullAttentionSpec > AttentionSpec`),
  27 layers, block_size 16, bf16;
- one latent "head": `num_kv_heads 1`, `head_size 576` (= kv_lora_rank 512 +
  qk_rope_head_dim 64), `head_size_v 0` (no separate V);
- `page_size_bytes == real_page_size_bytes == 18432` (16 tokens x 576 x 2 B),
  no padding (`alignment`, `page_size_padded` None), no quantization
  (`cache_dtype_str auto`, `kv_quant_mode 0`, `state_content_bytes` None);
- `max_tp_shards 1`, `kv_tp_replicas 2`: both TP ranks hold the same latent;
- `cache_role sparse` (vLLM's default; not an indexer cache),
  `is_index_group_leader False`, `model_version None` (not DeepSeek-V4);
- `kv_cache_layout LBNHC`, one `KVCacheTensor` per rank with
  `block_stride 18432` (= page), `layer_stride = num_blocks * page`;
- per-rank layer view `[6222, 1, 16, 576]`, strides `[9216, 576, 576, 1]`
  (elements), identical on tp0 and tp1; hash_block_size 16.

vLLM capacity at TP2 / gpu-mem 0.8: 2.88 GiB KV per GPU, 6222 blocks,
99,552 tokens (12.2x an 8192-token request). Weights + non-torch 14.96 GiB
per GPU. No `--gpu-mem` change needed.

## Mapping to semantics

| | value | why |
|---|---|---|
| Kind | `Full` (`Group.full(16)`) | MLA attends every past token through the latent; resuming at `p` needs blocks `[0, p/16)` |
| unit | one manager block of one layer = one 18432 B page, one segment | the page is `[N=16][H=1][C=576]`, the latent and rope key of one token are adjacent; it is what TRITON_MLA reads |
| Shard | `Shard.replicated()` on every rank | `max_tp_shards 1`: vLLM builds the latent from the full hidden state on each rank, so the bytes are identical (checked: probe digests per rank) |
| seal_plan | one fragment per layer, holders [0, 1], writer 0 | existing semantics (`sharding.rs` test `mla_replicas_collapse_to_one_fragment_with_rank0_writer`); no new Kind or Shard |

No semantics change.

## Gate settings (`gate.json`)

On sm89 vLLM 0.32 has one dense MLA decode backend: `TRITON_MLA`
(`FLASH_ATTN_MLA` needs sm90, `FLASHMLA` sm90/100, `FLASHINFER_MLA` sm100).
It declares `supports_batch_invariance` and uses one KV split under
`VLLM_BATCH_INVARIANT=1`. The MLA prefill backend is `FLASH_ATTN`
(`num_splits=1` under invariance). So `batch_invariant: true`,
`invariant_attention_backend: TRITON_MLA`.

## Self-test (`~/kexi/rsi-runs/2026-10-23-dsv2-lite-selftest`, TP2 GPU 0,1)

Command (rev 8af8fe4, no connector_v2 involved):

```
run_gate --model ~/kexi/models/DeepSeek-V2-Lite-Chat --tag dsv2-lite-selftest \
  --arms plain,baseline,selftest:honest,selftest:liar,selftest:bitflip,selftest:offbyone \
  --workloads gsm8k --gsm8k-limit 128 --tp 2 --gpu 0,1
```

| arm | verdict | key numbers |
|---|---|---|
| baseline controls | pass | plain == baseline 0/128 differ; warm == cold 0 differ; local 34544 matched, computed_exact 5336/5336 |
| honest | FAIL (expected pass) | external 1084/1084 matched vs self and vs baseline, local 34544 matched, 0 mismatched; but warm outputs 60/128 differ, computed_exact 0/2136 (self), 2668/3736 (vs baseline); acc 0.7031 cold / 0.7109 warm |
| liar | caught | external 2168 mismatched vs baseline (unreferenced in self), acc 0.0 |
| bitflip | caught | 256 external mismatched (one block per request per rank, TP2), warm 128/128 outputs differ, acc 0.0 |
| offbyone | caught | 828 external mismatched, acc 0.10 |

(G0.spec failed in this run only because `spec.json` was not committed yet.)

The negative arms are caught, and per-rank digests work at TP2. The honest
arm fails only on checks of tokens computed AFTER a loaded prefix: every
block it loaded is byte-identical to baseline.

### MLA KV is not split-invariant (gate issue, reported to the orchestrator)

vLLM 0.32's MLA prefill (`MLAAttention.forward_mha`) computes attention over
the cached context separately (`_compute_prefill_context`: gather latent,
`kv_b_proj`, FA varlen with LSE) and merges it with the new tokens'
attention (`merge_attn_states`). The split point is `num_computed_tokens`.
So hidden states, and the KV of every later layer, depend on where the
prefix-cache hit ends, even with `VLLM_BATCH_INVARIANT=1`. Full attention on
FLASH_ATTN (Qwen3) runs one paged kernel over context + suffix and is not
affected.

Evidence without any connector (`~/kexi/tmp/dsv2-p4/split_inv.py`, plain
vLLM TP2 GPU 2,3, `VLLM_BATCH_INVARIANT=1`, TRITON_MLA decode, FLASH_ATTN
MLA prefill; same 900-token prompt, prefix cache reset and warmed with the
first `warm` tokens, then 48 greedy tokens; log `split_inv.log`):

| warm prefix | cached tokens | tokens == no-cache run | first differing token | first 4 logprobs |
|---|---|---|---|---|
| 0 | 0 | yes | - | -0.5516 -1.1509 -1.1579 -1.0617 |
| 0 (repeat) | 0 | yes | - | -0.5516 -1.1509 -1.1579 -1.0617 |
| 256 | 256 | yes | - | -0.5777 -1.1524 -1.0198 -1.0996 |
| 512 | 512 | no | 3 | -0.5752 -1.2620 -1.1928 -0.7722 |
| 800 | 528 | no | 3 | -0.5659 -1.1281 -1.0138 -0.8151 |

Consequence: any correct connector fails `G2.gsm8k_outputs` and the
`computed_exact` clause of the gsm8k digest verdicts on this model.
Orchestrator decision (2026-10-23): gate issue; the gate owner adds a
`split_invariant` mode on `feat/rsi-g5` (a plain-vLLM split control decides
it; loaded-block byte checks and gsm8k accuracy stay hard; computed_exact
only against same-split references; gsm8k_outputs bounded by the split
control). Until that merges, iterations are judged on the loaded-block
checks (local/external matched, 0 mismatched/unreferenced), E_ext, gsm8k
accuracy and G4. The official run waits for the merge.

## it1 (2026-10-23): MLAAttentionSpec adapter

Hypothesis: MLA is Full + one replicated page per layer; the existing core
handles it with one new adapter and no core, policy or semantics change.

### Design

- `adapters/mla_attention.py`, registered as `"MLAAttentionSpec"`. The
  registry dispatches by exact class name (`type(spec).__name__`), so
  `full_attention` never sees MLA. The adapter subclasses
  `FullAttentionAdapter` for the page geometry check (`view`: dense
  `[H=1, N, C]` pages a page apart, NHC kernel-split allowed) and the shared
  fail-closed fields (`sliding_window`, padding, `num_head_slots`,
  `state_content_bytes`, `tokens_per_state`, `kv_quant_mode`, `non_causal`,
  `has_layer_views`).
- Shard: `Shard.replicated()`; requires `max_tp_shards == 1` (anything else
  means a head-split latent nobody has validated) and `kv_tp_replicas`
  dividing `tp_size`. DCP fails closed as in `full_attention`.
- MLA-only fail-closed fields: `alignment`, `model_version` (DeepSeek-V4),
  `cache_role != sparse` (indexer caches), `is_index_group_leader`,
  `cache_dtype_str` other than auto (fp8_ds_mla / nvfp4 packed latents),
  `head_size_v != 0`, `non_causal_multi_token_decode`.
- Namespace: the spec class name is already in the layout fingerprint, so an
  MLA page never shares a namespace with a Full page of the same size.
- Saves at TP2: both ranks save their (identical) replica; the server keeps
  the first copy of a key per slot. One writer (`seal_plan` writer 0) would
  halve D2H traffic; not done in it1 (the core has no per-rank save filter
  yet) so the first slice measures whether it matters.
- Loads at TP2: every rank loads into its own GPU (one `load` RPC per rank).
- Sync loads: enabled (one KV group, no zeroing), as for Qwen3-4B.
- `connector.py` logs the package dir it was imported from (`from=`), so a
  run record shows which checkout vLLM used.

### Tests

`test_connector_v2_core.py` (stub + native):
- `test_mla_attention_layout[tp1|tp2|tp4]`: Full group of 16, replicated
  shard on every rank, the DSV2 registration (one 18432 B segment, 6222
  blocks, direct backend);
- `test_mla_seal_plan_one_writer` (native): TP2 replicas seal into one
  fragment, holders [0, 1], writer 0;
- `test_mla_attention_rejects_unvalidated_specs[...]`: 13 fields;
- `test_mla_and_full_attention_never_share_a_namespace`.

Remote (.44, shared venv, native ext copied from `pegaflow-semantics-rsi`):
200 passed (`test_connector_v2_core.py` + `test_semantics_bindings.py`);
ruff check / format clean (run from `python/`; from the repo root ruff
reports 6 pre-existing I001 in other test files on the base tree too).

### Slice (`~/kexi/rsi-runs/2026-10-23-dsv2-lite-p4-it1-slice`, rev b358ceb)

```
run_gate --model ~/kexi/models/DeepSeek-V2-Lite-Chat --tag dsv2-lite-p4-it1-slice \
  --arms plain,baseline,pegaflow.connector_v2:PegaKVConnectorV2 \
  --gsm8k-limit 1319 --replay-convs 8 --replay-turns 6 --tp 2 --gpu 0,1
```

The old arm ran on the same rev and settings (`2026-10-23-dsv2-lite-p4-old-slice`,
arms baseline,old). The vLLM log confirms the import path
(`from=/root/kexi/pegaflow-rsi-deepseek-v2-lite/python/pegaflow/connector_v2`),
one instance, and 27 layers registered on each of the two devices.

Verdicts (v2). Loaded-block checks and other hard checks:

| verdict | result | numbers |
|---|---|---|
| G0.spec | pass | |
| digest gsm8k, loaded blocks (self and vs baseline) | pass | external 10526 matched / 0 mismatched / 0 unreferenced; local 357952 matched / 0 mismatched |
| G2.digest_self.replay | pass | external 1382 / local 7094 matched, 0 mismatched |
| G2.gsm8k (accuracy) | pass | 0.6725 cold / 0.6778 warm vs baseline 0.6725 |
| G3.gsm8k_warm / replay_warm | pass | E_ext 1.0 / 1.0, excess 0, load errors 0 |
| G3.no_eviction | pass | 0 in every pass |
| G4 | FAIL | replay cold TTFT p50 and req/s (below) |

Split-affected (`split_invariant`, re-judge once feat/rsi-g5 lands):

| verdict | v2 | old |
|---|---|---|
| G2.gsm8k_outputs | FAIL: cold 0 / warm 647 differ | FAIL: warm 642 differ |
| computed_exact, digest_self.gsm8k | 0 / 23032 | 0 / 23176 |
| computed_exact, digest_vs_baseline.gsm8k | 27526 / 39042 | 27526 / 39114 |

G4 (v2 / old vs their own baseline; warm = warm2, warm3):

| metric | baseline (v2 run) | v2 | old | old's baseline | v2 verdict |
|---|---|---|---|---|---|
| replay warm TTFT p50 (s) | 0.162 / 0.163 | 0.098 / 0.115 | 0.112 / 0.095 | 0.161 / 0.164 | pass |
| replay warm req/s | 5.98 / 5.98 | 7.41 / 7.34 | 7.46 / 7.58 | 6.02 / 5.98 | pass |
| replay cold TTFT p50 (s) | 0.186 | 0.218 | 0.216 | 0.187 | FAIL (+17%) |
| replay cold req/s | 4.10 | 3.29 | 5.38 | 4.21 | FAIL |
| replay cold TTFT p99 (s) | 1.94 | 4.78 | 0.76 | | |
| gsm8k cold s | 193.9 | 194.6 | 198.2 | 197.0 | pass (+0.4%) |
| gsm8k warm s | 193.7 | 189.6 | 194.2 | 196.6 | pass |

Hits (local / external / total / oracle H* / E_ext):

| pass | baseline | v2 | old |
|---|---|---|---|
| gsm8k cold | 0.940 / 0 / 0.940 / 0.940 / - | 0.940 / 0 / 0.940 / 0.940 / - | 0.940 / 0 / 0.940 / 0.940 / - |
| gsm8k warm | 0.940 / 0 / 0.940 / 0.993 / 0.0 | 0.937 / 0.055 / 0.993 / 0.993 / 1.0 | 0.938 / 0.055 / 0.993 / 0.993 / 1.0 (excess 1230) |
| replay cold | 0.769 / 0 / 0.769 / 0.769 / - | 0.769 / 0 / 0.769 / 0.760 / - | 0.769 / 0 / 0.769 / 0.760 / - |
| replay warm, warm2, warm3 | 0.769 / 0 / 0.769 / 0.998 / 0.0 | 0.705 / 0.293 / 0.998 / 0.998 / 1.0 | 0.705 / 0.293 / 0.998 / 0.998 / 1.0 (excess 30) |

Save bytes. Per rank, from the workers' `save stats` lines summed over the
run (gsm8k cold+warm / replay all passes), and on the server
(`pegaflow_save_bytes_total`, `cache_block_insertions_total`):

| | tp0 | tp1 | server, gsm8k cold | server, replay cold |
|---|---|---|---|---|
| v2 | 11.18 GB / 1.385 GB, 0 failed | 11.18 GB / 1.385 GB, 0 failed | 13.68 GB, 13742 blocks (995328 B = 54 slots x 18432) | 2.334 GB, 2345 blocks |
| old | | | 6.84 GB, 13742 blocks (497664 B = 27 x 18432) | 1.167 GB, 2345 blocks |

So `Shard.replicated()` does not yet mean one writer: both TP ranks copy
the same latent out, and the server registers the instance with
`tp_size 2`, so it keeps two slots per layer and stores every block twice
(2x D2H traffic, 2x host memory). The old arm stores 27 slots per block.

### Cold replay noise (replay-only repeats, rev b358ceb, arms baseline + v2)

`--workloads replay --replay-convs 8 --replay-turns 6 --tp 2`, run at the
same time on GPU 0,1 (`2026-10-23-dsv2-lite-p4-it1-replay-01`) and GPU 2,3
(`-23`):

| sample | baseline cold TTFT p50 / req/s | v2 cold TTFT p50 / req/s | verdict (TTFT, req/s) |
|---|---|---|---|
| slice (GPU 0,1) | 0.186 / 4.10 | 0.218 / 3.29 | fail, fail |
| repeat 0,1 | 0.190 / 5.69 | 0.197 / 5.53 | fail (+3.4%), fail (-2.9%) |
| repeat 2,3 | 0.194 / 4.08 | 0.197 / 5.57 | pass (+1.6%), pass |

Warm passed in all three samples (v2 0.109 to 0.117 s vs 0.159 to 0.169 s;
7.3 to 7.4 vs 5.95 to 6.0 req/s). Cold req/s swings between 4.1 and 5.7
for the SAME arm, so it is noise, set by how 32 workers line up on 8
conversations. Cold TTFT p50 is the steadier signal: v2 is 1.6 to 3.4%
above baseline in the repeats (17% in the slice), and the old arm is 15%
above. Cold replay saves every block it computes, so the save path costs
cold TTFT. it1 verdict: fail on G4 replay cold TTFT; the rest of the
judged checks pass.

## it2 (2026-10-23): one storage rank, one writer per replicated part

Hypothesis: `Shard.replicated()` should cost one copy, not one per TP rank.
Registering both MLA replicas as one server rank halves D2H bytes and host
memory, and fixes the cold replay TTFT regression if that is save
bandwidth.

### Design

- `core/layout.py`: `StorageRanks`, derived from the `seal_plan` that
  `build_layout` already runs. It applies when every layer has the same
  part count and each TP rank holds one part throughout, either as that
  part's writer for all its layers or for none. Ranks that hold the same
  part then share one storage rank. The server sees `tp_size = parts`; for
  MLA at TP2 that is `tp_size 1`, with both devices registering
  `tp_rank 0`. This is the registration contract pegaflow-core's
  `mla_replica_registration_seals` test covers: one slot per layer, sealed
  once both devices have registered. Every other layout gets the identity
  mapping (Qwen3-4B TP1: unchanged).
- `core/worker.py`: register, save and load use the storage rank. Only the
  writer (`Fragment.writer`, the first holder) saves. Every holder loads
  into its own GPU from the same slot.
- `connector.py`: the session watcher announces the storage `tp_size`.
- The namespace includes the storage `tp_size` (`storage_tp`), so blocks
  stored with another slot count never share a namespace. Otherwise a
  load would fail the server's slot-count check.
- No semantics, server or proto change.

Rebased onto `feat/semantics-rsi` df01403 (8a05b7c: saves stop for the
process after a failed sync load). That gate runs in the scheduler before
any save is planned, so the worker's writer filter only applies to plans
that already passed it. A run at the pre-rebase rev 71f2090 was stopped by
its own PIDs and kept as `2026-10-23-dsv2-lite-p4-it2-slice-aborted-71f2090`.

Tests: `test_mla_attention_layout[tp1|tp2|tp4]` now checks that every rank
registers `(tp_rank 0, tp_size 1)`, that only rank 0's save RPC carries the
27 layers, and that every rank issues its load against storage rank 0.
`test_full_attention_layout` checks one storage rank per head shard, with
the first holder writing (tp4-replicas: ranks 1 and 3 do not save).
StubSemantics' `seal_plan` now returns fragments and holders. 200 passed;
ruff clean.

### Slice (`~/kexi/rsi-runs/2026-10-23-dsv2-lite-p4-it2-slice`, rev 3d1de18, GPU 0,1)

Same command as it1 (`--tag dsv2-lite-p4-it2-slice`). vLLM log:
`registered 27 layers ... device=0 storage_rank=0 writer=True` and
`device=1 storage_rank=0 writer=False`; no load failures, no
`saves disabled`.

Verdicts (v2), loaded-block and other hard checks:

| verdict | result | numbers |
|---|---|---|
| G0.spec | pass | |
| digest gsm8k, loaded blocks (self and vs baseline) | pass | external 10526 matched / 0 mismatched / 0 unreferenced; local 357952 / 0 |
| G2.digest_self.replay | pass | external 1382 / local 7094 matched, 0 mismatched |
| G2.gsm8k | pass | 0.6725 cold / 0.6778 warm vs 0.6725 |
| G3.gsm8k_warm / replay_warm | pass | E_ext 1.0 / 1.0, excess 0, load errors 0 |
| G3.no_eviction | pass | 0 |
| G4 | FAIL | replay cold TTFT p50, cold req/s |

Split-affected (re-judge under `split_invariant`): gsm8k_outputs warm 647
differ (cold 0); computed_exact 0 / 23032 (self), 27526 / 39042 (vs
baseline). Same as it1: loaded bytes are unchanged.

G4:

| metric | baseline | v2 it2 | v2 it1 (slice) | verdict |
|---|---|---|---|---|
| replay warm TTFT p50 (s) | 0.164 / 0.165 | 0.102 / 0.110 | 0.098 / 0.115 | pass |
| replay warm req/s | 5.94 / 5.94 | 7.45 / 7.31 | 7.41 / 7.34 | pass |
| replay cold TTFT p50 (s) | 0.188 | 0.200 (+6.4%) | 0.218 | FAIL |
| replay cold req/s | 5.68 | 5.55 (-2.3%) | 3.29 | FAIL |
| gsm8k cold s | 194.1 | 194.4 (+0.2%) | 194.6 | pass |
| gsm8k warm s | 193.9 | 189.4 | 189.6 | pass |

Hits (local / external / total / H* / E_ext): gsm8k cold 0.940 / 0 /
0.940 / 0.940 / -; gsm8k warm 0.937 / 0.055 / 0.993 / 0.993 / 1.0; replay
cold 0.769 / 0 / 0.769 / 0.760 / -; replay warm, warm2, warm3 0.705 /
0.293 / 0.998 / 0.998 / 1.0. Baseline: gsm8k warm 0.940 local, H* 0.993;
replay warm 0.769 local, H* 0.998.

Save bytes:

| | tp0 | tp1 | server gsm8k cold | server replay cold |
|---|---|---|---|---|
| it1 | 11.18 GB / 1.385 GB | 11.18 GB / 1.385 GB | 13.68 GB (995328 B/block) | 2.334 GB |
| it2 | 11.18 GB / 1.385 GB | 0 / 0 | 6.84 GB, 13742 blocks (497664 B/block = 27 x 18432) | 1.167 GB, 2345 blocks |

Write amplification is gone. The server stores the same bytes as the old
arm, and the same block counts as it1.

Cold TTFT did not improve, so save bytes are not the cause. vLLM's own
histograms for cold replay (sums over 37 requests):

| run | arm | wall s | TTFT sum | prefill sum | queue sum | decode sum |
|---|---|---|---|---|---|---|
| it2 slice | baseline | 6.51 | 9.32 | 5.71 | 2.20 | 32.60 |
| it2 slice | v2 | 6.66 | 10.14 | 6.35 | 2.62 | 32.99 |
| it1 repeat 0,1 | baseline | 6.50 | 8.92 | 5.36 | 2.10 | 32.91 |
| it1 repeat 0,1 | v2 | 6.69 | 9.94 | 6.30 | 2.56 | 33.15 |

On the same GPUs, v2 adds 11 to 17% prefill time and about 20% queue
time, while decode is within 1%. Server-side cold RPCs total 303 ms of
save (138 calls) and 9 ms of query_prefetch (37 calls), all off the step
thread. it2 verdict: keep (one writer is what `Shard.replicated` means),
G4 cold still fails. Next hypothesis (it3): what in a cold prefill step
costs more with the connector. Candidates: vLLM's per-step connector
overhead (outputs gathered from every TP rank), the step thread's
`wait_for_save` / `build_connector_meta`, and the server's D2H copies
competing with prefill on the same GPU.
