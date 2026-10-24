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

## Qwen3-4B non-regression for it2 (`2026-10-23-qwen3-4b-p4-it2-nonreg`, rev 3d1de18, GPU 0)

`--model ~/kexi/models/Qwen3-4B --gsm8k-limit 1319 --replay-convs 8
--replay-turns 6 --gpu 0`, arms plain, baseline, v2. All v2 verdicts pass,
judged against the in-run baseline:

| check | it2-nonreg (GPU 0) | reference it2c-slice1 (GPU 1, rev 056eb4b) |
|---|---|---|
| digest gsm8k self / vs baseline | ext 6123 / local 170698 matched, 0 mismatched; computed_exact 16652/16652, 22126/22126 | ext 6578 / local 170243, 0 mismatched; 16652/16652, 22125/22125 |
| gsm8k_outputs, accuracy | 0 / 0 differ; 0.8582 = baseline | 0 / 0; 0.8582 |
| E_ext gsm8k / replay | 1.0 / 1.0, excess 0 | 1.0 / 1.0 |
| replay warm TTFT p50 v2 / baseline | 0.095, 0.092 / 0.218, 0.220 | 0.094, 0.096 / 0.210, 0.210 |
| replay warm req/s | 6.79, 6.79 / 5.37, 5.38 | 6.87, 6.79 / 5.52, 5.52 |
| replay cold TTFT p50 | 0.217 / 0.218 (-0.5%) | 0.208 / 0.211 |
| replay cold req/s | 5.00 / 5.01 | 5.09 / 5.09 |
| gsm8k cold / warm s | 172.1 / 170.8 vs 170.8 / 171.6 (+0.7%, -0.5%) | 163.8 / 162.3 vs 175.6 / 162.0 |
| hits warm gsm8k (local / ext / total) | 0.926 / 0.067 / 0.992 | 0.921 / 0.072 / 0.992 |

Relative to the in-run baseline, every delta is the same as the reference
run's. Absolute times are about 4% higher on GPU 0 for both arms. So the
storage-rank change (identity at TP1) does not regress Qwen3-4B.

## it3a (2026-10-23): what makes cold prefill slower (probe arms, no code change)

Single-writer did not fix cold, so this iteration measures before changing
code. The probe arms live in a remote-only module,
`~/kexi/tmp/dsv2-p4/p4probe.py` (not in the repo, on PYTHONPATH for the
run only), as subclasses of `PegaKVConnectorV2` at rev 3d1de18:

- `NoSaveV2`: `wait_for_save` does nothing. Queries and loads as v2.
- `NullV2`: registers with the server, returns no hit, plans nothing, saves
  nothing. vLLM still runs its connector plumbing.
- `DeferV2`: same saves as v2, but a step's save job is handed to the saver
  thread only at the start of the first later step without prefill tokens
  (or after 8 steps), so the D2H copies overlap decode steps, not
  prefill. The log shows 147 to 297 decode-only releases per window, 1
  forced.

How saves are issued today (pegaflow-core, read-only): the worker's saver
thread waits on a CUDA event recorded on vLLM's compute stream after the
step's forward, then sends one `save` RPC per step. The server's per-GPU
save thread copies with one `cuMemcpyDtoHAsync_v2` per coalesced range
(`TransferMode::Direct`, the DMA copy engines; MLA uses `direct`) on a
cudarc `new_stream()` stream (`CU_STREAM_NON_BLOCKING`, default priority),
then synchronizes an event. No kernel is involved and no stream priority
is set. The 4090s on .44 have no GPU P2P (vLLM: "Custom allreduce is
disabled ... lacks GPU P2P"), so TP all-reduce goes through NCCL over PCIe
host memory. The server's D2H copies cross the same PCIe link.

Cold replay, replay-only, 8x6, same rev:

| run (GPU) | arm | TTFT p50 | vs baseline | req/s | prefill sum s | queue sum s | decode sum s |
|---|---|---|---|---|---|---|---|
| probe-23 (2,3) | baseline | 0.1892 | | 5.689 | 5.65 | 2.19 | 32.61 |
| | v2 | 0.1997 | +5.5% | 5.510 | 6.37 | 2.62 | 33.23 |
| | NoSaveV2 | 0.1909 | +0.9% | 5.687 | 5.78 | 2.17 | 32.53 |
| | NullV2 | 0.1923 | +1.6% | 5.556 | 5.82 | 2.01 | 33.39 |
| defer-01 (0,1) | baseline | 0.1858 | | 5.723 | 5.73 | 2.35 | 32.37 |
| | v2 | 0.1998 | +7.5% | 5.584 | 6.25 | 2.58 | 32.86 |
| | DeferV2 | 0.1951 | +5.0% | 5.626 | 5.98 | 2.29 | 32.96 |
| defer-23 (2,3) | baseline | 0.1872 | | 5.714 | 5.58 | 2.14 | 32.45 |
| | v2 | 0.2004 | +7.1% | 5.568 | 6.29 | 2.60 | 32.96 |
| | DeferV2 | 0.1969 | +5.2% | 5.594 | 5.92 | 2.14 | 33.19 |

Warm replay (warm2 / warm3 TTFT p50): v2 0.098 to 0.114 s, DeferV2 0.100 to
0.118 s, NoSaveV2 and NullV2 0.164 to 0.177 s (no saves, so no external
hits, as expected). DeferV2's loaded-block checks pass (external 1382 /
local 7094 matched, 0 mismatched; E_ext 1.0).

Findings:
- The cold cost is the save path. NoSaveV2 (queries, leases, connector
  plumbing) is within 1% of baseline, and NullV2 within 1.6%. Per-step
  scheduler-side cost (query RPC, plan building) is not it: NullV2 drops
  them and is no faster than NoSaveV2.
- Moving the copies off prefill steps buys back only a third (+7% to +5%).
  So the cost is not only GPU contention between the D2H copy and the
  prefill kernels. The rest moves with the save job, not with its timing.
- What a save job does besides the copy, on tp0 (saver thread): it waits
  for the compute event (`Event.synchronize`, releases the GIL), builds
  the RPC arguments (27 layer names x block ids x hashes, about 2.3k
  blocks per cold pass), and runs the blocking gRPC call (GIL released in
  `py.detach`). Building the arguments holds the GIL while vLLM's
  worker step thread needs it. Next probe: NoRpcV2 (the full worker save
  path without the RPC) splits Python-side cost from the server copy.

Process note: these runs are stopped only by their own gate. Their
run_gate pgids were recorded at launch (probe-23 1647263, defer-01
1689145, defer-23 1689150).

## it3b (2026-10-23): the save RPC vs the server copy (probe arms, no code change)

The question is which part of the save job costs the time. More arms in
`p4probe.py` (remote only), all at rev 1bfb590, which includes the gate
keep-alive fix:

- `NoRpcV2`: the full worker save path (event wait, building the arguments),
  but the save RPC returns True without calling the server.
- `EmptyRpcV2`: the real save RPC with an empty `saves` list. The gRPC round
  trip happens, the server copies nothing.
- `Batch8V2`: the same saves as v2, but the held jobs are merged into one RPC
  every 8 steps (or on flush). The merged job carries the last job's seq and
  compute mark. The stream is in order, so the last mark covers every earlier
  step.

Cold replay, replay-only, 8x6. Prefill, queue and decode are vLLM's
per-request sums in seconds.

| run (GPU) | arm | TTFT p50 | vs baseline | req/s | prefill | queue | decode |
|---|---|---|---|---|---|---|---|
| norpc-01 (0,1) | baseline | 0.1844 | | 5.739 | 5.50 | 2.13 | 32.36 |
| | v2 | 0.1890 | +2.5% | 5.630 | 5.82 | 2.27 | 32.87 |
| | NoRpcV2 | 0.1869 | +1.4% | 5.684 | 5.65 | 2.17 | 32.60 |
| norpc-23 (2,3) | baseline | 0.1720 | | 5.676 | 5.47 | 1.90 | 32.90 |
| | v2 | 0.2032 | +18.1% | 5.467 | 6.53 | 2.87 | 33.29 |
| | NoRpcV2 | 0.1780 | +3.5% | 5.701 | 5.54 | 2.12 | 32.65 |
| rpc-01 (0,1) | baseline | 0.1864 | | 5.728 | 5.73 | 2.35 | 32.36 |
| | v2 | excluded (contaminated, see below) | | | | | |
| | EmptyRpcV2 | 0.1921 | +3.1% | 5.687 | 5.85 | 2.40 | 32.50 |
| | Batch8V2 | 0.1988 | +6.6% | (1.192, contaminated mid-decode) | 6.13 | 2.42 | |
| | NoSaveV2 | 0.1920 | +3.0% | 5.685 | 5.86 | 2.43 | 32.50 |
| rpc-23 (2,3) | baseline | 0.1869 | | 5.709 | 5.80 | 2.36 | 32.44 |
| | v2 | 0.2010 | +7.6% | 5.569 | 6.18 | 2.45 | 32.98 |
| | EmptyRpcV2 | excluded (contaminated) | | | | | |
| | Batch8V2 | 0.1909 | +2.1% | 5.585 | 6.06 | 2.36 | 33.08 |
| | NoSaveV2 | 0.1913 | +2.4% | 5.661 | 5.83 | 2.22 | 32.67 |

Warm (warm2 / warm3 TTFT p50, rpc runs): v2 0.097 to 0.117 s, Batch8V2 0.105
to 0.115 s, baseline 0.159 to 0.167 s. EmptyRpcV2 and NoSaveV2 stay at
baseline (no saves, so no external hits). Batch8V2's loaded-block checks
pass (E_ext 1.0, excess 0, load_errors 0, no eviction). The hit table
matches v2's: warm local 0.705 / ext 0.293 / total 0.998 / H* 0.998.
Batch8V2's rpc-23 warm pass has p99 1.36 s with queue 11.5 s, against 1.8 s
on its other warm passes and on v2. That is one queued burst on the
first warm pass and it does not repeat.

Contaminated passes, excluded. Three passes stalled for 22 to 32 s, each
in its first forward step after vLLM's Triton JIT compile (rpc-01 v2,
rpc-23 EmptyRpcV2), or in one mid-pass step (rpc-01 Batch8V2). They line
up to the second with another agent's 160 GiB `pegaflow-server` finishing
teardown on .44:

- the qwen3-4b-g5 old arm on GPU 5: "gone after SIGTERM in 66.0s" ends
  19:39:47, and rpc-01 v2's first step runs 19:39:22 to 19:39:46;
- the same run's next server: 61.0 s, ends 19:46:31; rpc-23 EmptyRpcV2 runs
  19:46:10 to 19:46:32.

The stalled requests had no load (hit=0), no save backlog (tp0
max_pending 2, max_job_ms 108.6, the same as the clean v2 pass) and no
pending registration (the session opened 19:39:18). Nothing else ran on
our pair. The reading that this is a CUDA driver stall while a 160 GiB
pinned pool is freed is unverified. It sits next to the server's teardown
candidates (SIGTERM not handled, about 0.36 s/GiB release, hugepages),
which are for the server owner. Our own servers take 59 to 95 s to tear
down, so our runs can stall other agents the same way. The gate owner has
two proposals: an untimed warmup request per arm before the cold pass,
and flagging a cold pass whose p99 is more than 10x the in-run baseline.

What the server does per save (v2 rpc-23 cold, tp0, from the server's
metrics): 138 save RPCs, 1.167 GB, `save_duration` sum 0.297 s (129 under
10 ms, max under 25 ms). On the client side, rpc_ms is 481 over 169 jobs,
wait_ms 3360 (compute event), max_pending 2. The copy itself is cheap for
the server. The cost shows up on the GPU's other work.

Findings:
- NoSaveV2, NoRpcV2 and EmptyRpcV2 are all within 1.4 to 3.5% of baseline.
  Their ranges overlap across the pairs. So the Python save path and an
  empty gRPC round trip are not the cost. The server copy is.
- Batch8V2 is +2.1% on 2,3 against v2's +7.6% in the same run. The 0,1 run
  (+6.6%) lost its v2 control to the stall. Batching is the one probe that
  moves cold toward the noise band while keeping warm hits. It is followed
  up in the next section.

### TP2 all-reduce vs a concurrent D2H (microbenchmark, GPU 0,1)

The coordinator's hypothesis: with TP2 and no P2P, NCCL all-reduce goes
through host memory, and the save D2H competes with it. To test it,
`~/kexi/tmp/dsv2-p4/ar_d2h_bench.py` (remote only) runs a separate saver
process (like pegaflow-server) that copies D2H from GPU memory into pinned
host memory on a non-blocking stream. It times one "step": 54 all-reduces
of [T, 2048] bf16 (27 layers x 2), p50 over 30 steps. NCCL 2.29.7 picks
SHM (`Channel 00 : 0[0] -> 1[1] via SHM/direct/direct`).

| saver | T=512 ms | T=2048 | T=4096 |
|---|---|---|---|
| none | 9.0 | 30.6 | 61.5 |
| GPU2 (outside the pair), back-to-back 64 MB, 24.9 GB/s | 9.0 | 30.6 | 61.8 |
| GPU0 back-to-back 64 MB | 54.4 | 215.9 | 430.8 |
| GPU1 back-to-back 64 MB | 43.5 | 171.4 | 341.7 |
| GPU0 back-to-back, saver pinned to NUMA3 (GPU0-local) | 51.0 | 198.9 | 398.3 |
| GPU0 back-to-back, saver pinned to NUMA0 | 54.3 | 215.6 | 430.5 |
| GPU0 + GPU1 back-to-back | 96.0 | 363.9 | 734.7 |
| GPU0 2 MB / 10 ms (0.18 GB/s) | 10.2 | 36.5 | 73.6 |
| GPU0 1 MB / 10 ms (0.09 GB/s) | 9.7 | 34.6 | 69.3 |
| GPU0 + GPU1, 1 MB / 10 ms each | 10.2 | 36.6 | 73.4 |
| GPU0 7 MB / 40 ms, one copy | 9.1 | 35.2 | 69.5 |
| same, as 256 KiB strided pieces | 9.1 | 36.4 | 73.3 |
| same, as 18 KiB pieces (one MLA page) | 9.1 | 36.1 | 72.8 |
| GPU0 2 MB / 10 ms, side stream in a rank-0 thread (same context) | 10.2 | 37.2 | 72.0 |
| GPU0 tiny kernel every 10 ms, other process, no copy | 9.1 | 31.9 | 64.1 |
| GPU0 tiny kernel every 2 ms, other process, no copy | 9.6 | 34.4 | 68.7 |

Findings:
- The interference is per GPU, not per PCIe link. A 25 GB/s D2H on a GPU
  outside the pair costs the all-reduce nothing. Any D2H on a pair GPU slows
  it. Pinning the saver to the GPU-local NUMA node recovers only 8%.
  Candidate (a), checking that rank 0's D2H does not share a root complex
  with the NCCL path, is not the lever. Every 4090 here is on its own root
  port (`pci0000:00/20/40/60`), SYS to every other GPU.
- Chunking does not help. At the same bytes and pacing, one copy, 256 KiB
  pieces and 18 KiB pieces cost the same. Back-to-back small copies are
  worse than large ones. Candidate (b) as "chunk the D2H" does not hold.
  Rate-limiting lowers the cost roughly in proportion, but even 0.09 GB/s
  costs +13% at T=4096. Cold replay saves about 1.2 GB in 6.6 s (about
  0.18 GB/s, the +19% row), so pacing alone cannot reach 2%.
- Splitting the same bytes across both writers is no better.
- About half the cost is there without any copy: a tiny kernel submitted
  from another process every 2 ms costs +12%. An active second context
  takes time from NCCL's SHM proxy kernels. That points at how often the
  server submits work on the GPU more than at bytes.

These measurements went to the server agent, who owns the server copy and
submit path (feat/rsi-server-reclaim). The connector side can change how
often a save reaches the server: batching across steps (candidate (c)),
next.

### Batching across steps (probe arms, `batch-01` / `batch-23`, rev 1bfb590)

`Batch32V2` is Batch8V2 with 32 steps. `CountV2` is v2 plus a counter of
device-contiguous block runs per layer save. That count is a lower bound on
the server's `cuMemcpyDtoHAsync` calls, since the server merges contiguous
ranges.

| run (GPU) | arm | cold TTFT p50 | vs baseline | req/s | prefill | queue | warm2 / warm3 p50 |
|---|---|---|---|---|---|---|---|
| batch-01 (0,1) | baseline | 0.1894 | | 5.691 | 5.83 | 2.38 | 0.172 / 0.161 |
| | v2 | 0.2071 | +9.3% | 5.589 | 6.36 | 2.48 | 0.101 / 0.116 |
| | Batch8V2 | 0.1944 | +2.6% | 5.594 | 6.08 | 2.22 | 0.104 / 0.109 |
| | Batch32V2 | 0.1897 | +0.2% | 5.662 | 5.88 | 2.43 | 0.108 / 0.100 |
| batch-23 (2,3) | baseline | 0.2010 | | 5.657 | 5.39 | 2.06 | 0.162 / 0.151 |
| | v2 | 0.1996 | -0.7% | 5.556 | 6.06 | 2.33 | 0.118 / 0.111 |
| | Batch32V2 | excluded (stalled in decode, see below) | | | | | 0.099 / 0.108 |
| | CountV2 | 0.1966 | -2.2% | 5.590 | 6.16 | 2.54 | 0.100 / 0.107 |

The batch-23 baseline cold p50 (0.2010) is high for this pair: it was
0.1869 to 0.1892 in four earlier runs. So the p50 deltas on 2,3 in this run
are not comparable. Prefill sums still order the same way as on 0,1
(baseline 5.39, v2 6.06, CountV2 6.16). Warm E_ext is 1.0 with excess 0 and
no eviction for every saving arm. Warm hits for Batch8V2 and Batch32V2 equal
v2's (local 0.705 / ext 0.293 / total 0.998 / H* 0.998).

The Batch32V2 cold pass on 2,3 lost 24 s in one decode step. That step
lines up with this run's own other pair: the batch-01 Batch8V2 server tore
down from 20:52:15 to 20:53:18. It is excluded and is rerun one pair at a
time in it4.

Copies per save (CountV2, cold to warm3, tp0): over 500 jobs, 13,500 layer
saves, 74,952 blocks and 16,308 contiguous runs. That is 1.21 runs per
layer save and about 33 per job (27 layers). So the server issues about 33
`cuMemcpyDtoHAsync` calls per save RPC. Batch32 cuts the RPC count about
16x (one RPC per 32 steps, instead of one every ~2 steps that save), and so
the copy submissions per second too.

Findings:
- Fewer save submissions bring cold back toward baseline: Batch8 +2.6%,
  Batch32 +0.2%, against v2 +9.3% in the same run on 0,1. Warm hits and the
  warm speedup are unchanged.
- Each of these is one sample. The claim needs the coordinator's 3x
  in-run repeat on one pair, with no other pegaflow server starting or
  stopping on the host during a timed pass.

## it4 (2026-10-23): save batching as a policy knob (commit 72bffa6)

Hypothesis: the cold cost scales with how often the server submits copies
on vLLM's GPU. Merging consecutive steps' saves into one save RPC brings
cold replay within 2% of baseline, with warm hits unchanged.

### Design

- Two new knobs in `models/default.toml` / `Policy`, model-agnostic.
  `save_batch_steps` and `save_batch_ms`: a step's save job is held, and
  is sent merged with the following steps' jobs once the oldest held job
  has seen `save_batch_steps` engine steps or waited `save_batch_ms`,
  whichever comes first. A value <= 0 disables that bound;
  `save_batch_steps = 1` sends every step. The commit ships both off, so
  the defaults stay at today's behaviour until the gates pick a value.
- Merging: one entry per layer, with the block ids and hashes of every
  held step in order. The merged job takes the last held job's sequence
  number and compute mark. vLLM's compute stream is in order, so the last
  mark covers every earlier held step's forward. A held job already owns
  its sequence number, so the finish fence (`finished_sending`) of a
  request that finishes while its saves are held still waits for them.
- `flush` (preemption, shutdown) sends held jobs first, so a preemption
  never waits on the batch bound. vLLM calls `wait_for_save` on steps
  without a forward too, so held jobs keep aging while the engine only
  polls.
- New `save stats` fields: `rpcs` (save RPCs sent), `sent` /
  `sent_wait_ms` / `max_sent_wait_ms` (how long finished requests were
  held between vLLM's finish and our `finished_sending`).
- Unit tests: `test_save_batching` covers the off, step, idle-step, time
  and either-bound cases. `test_save_batching_flush_and_finish_fence`
  covers the fence plus flush. 207 passed.

### Knob sweep (`2026-10-23-dsv2-lite-p4-it4-knob-01`, GPU 0,1 only, `--pool-size 64gb`)

The arms are v2 subclasses that set the worker's batch fields (KB32:
steps 32; KT100 / KT250: time 100 / 250 ms, no step bound).

| arm | cold TTFT p50 | vs baseline | req/s | prefill | queue | save RPCs (cold window) | finished_sending hold mean / max ms (cold) |
|---|---|---|---|---|---|---|---|
| baseline | 0.1877 | | 5.714 | 5.56 | 2.13 | | |
| v2 | 0.2254 | +20.1% | 5.474 | 6.42 | 2.60 | 169 | 23 / 161 |
| KB32 | 0.1799 | -4.2% | 5.670 | 5.61 | 2.16 | 13 | 480 / 1030 |
| KT100 | 0.1899 | +1.2% | 5.596 | 6.16 | 2.49 | 46 | 78 / 225 |
| KT250 | 0.1902 | +1.3% | 5.621 | 6.05 | 2.50 | 22 | 154 / 282 |

Warm (warm2 / warm3 TTFT p50): baseline 0.162 / 0.161, v2 0.096 / 0.109,
KB32 0.102 / 0.101, KT100 0.111 / 0.104, KT250 0.099 / 0.110. Every saving
arm: E_ext 1.0, excess 0, load_errors 0, no eviction, replay digests 1382
external / 7094 local matched, 0 mismatched. Hits (warm): local 0.705 /
ext 0.293 / total 0.998 / H* 0.998, the same for every arm. Every arm
logs `flushes=0`, so no preemption flush had to wait. Each of the 37
requests per pass has at most 1 preemption: the gate scrapes only the
per-request histogram, whose le=1 bucket holds all 37.

Contamination check:
- v2's +20% has p99 0.83 (not 10x baseline), but it is far above its
  earlier +5 to +9%. Its cold pass (21:14) ran while another agent's 160
  GiB server (qwen3-4b-g5 gsm8k) was up and serving on GPU 5. It is one
  noisy sample. The repeat run carries its own v2 arm.
- KB32's first warm pass lost 18 s (p99 17.8 s, queue 143 s). The 8
  requests waited on async loads (vLLM: "KV fetch: 8 in progress") from
  21:20:38 to 21:20:56, and the engine only polled. The other agent's
  160 GiB server tore down from 21:19:50 to 21:20:56 and ended in the same
  second. This is the teardown stall again (unverified mechanism, see
  it3b), now hitting a load. Pass excluded.

Readings:
- The RPC count is what moves cold: 169 -> 46 -> 22 -> 13 RPCs gives
  +20 -> +1.2 -> +1.3 -> -4.2%.
- The cost of batching is the finished_sending hold. With a step bound
  only (KB32), it grows with step time: a mean of 480 ms and a max of
  1 s in cold, against 23 / 161 ms unbatched. The time bound caps it
  (KT250: max 282 ms) but sends about 2x the RPCs of KB32. Both bounds
  together (steps 32, time 250 ms) are the candidate default, since the
  time bound limits the hold when steps are slow.
- Not yet measured: G5 E_C (the cap workload, where later saves may turn
  a quick next turn into a miss) and Qwen3-4B TP1.

### KB32 3x in-run repeat (`2026-10-23-dsv2-lite-p4-it4-rep-01`, GPU 0,1 only, 64gb)

Arms: baseline, v2 (batching off), and KB32 three times (KB32b / KB32c are
aliases). Host check: no other pegaflow server started or stopped during
the cold passes.

| arm | cold TTFT p50 | vs baseline | cold req/s | vs baseline | prefill | G4 cold p50 / req/s | warm2 / warm3 p50 | save RPCs (cold) | hold mean / max ms (cold) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1861 | | 5.706 | | 5.60 | | 0.175 / 0.164 | | |
| v2 | 0.2011 | +8.1% | 5.562 | -2.5% | 6.24 | fail / fail | 0.117 / 0.106 | 168 | 20 / 104 |
| KB32 | 0.1890 | +1.5% | 5.676 | -0.5% | 5.86 | pass / pass | 0.096 / 0.123 | 15 | 451 / 1117 |
| KB32b | 0.1874 | +0.7% | 5.673 | -0.6% | 5.86 | pass / pass | 0.096 / 0.104 | 15 | 510 / 1121 |
| KB32c | 0.1905 | +2.4% | 5.640 | -1.2% | 5.95 | fail / pass | 0.096 / 0.105 | 15 | 454 / 1119 |

Every saving arm: E_ext 1.0, excess 0, load_errors 0, no eviction, warm
hits local 0.705 / ext 0.293 / total 0.998 / H* 0.998, `flushes=0`.

Findings:
- KB32 brings cold from +8.1% to +0.7 to +2.4% (mean +1.5%). Two of three
  pass the hard G4 bound, and the third misses it by 0.4 pp. Warm is not
  hurt (it is at or below v2).
- The finished_sending hold is the price: a mean of about 0.5 s and a max
  of 1.1 s with a step bound alone. The 250 ms time bound targets that.
### Candidate default 32 steps or 250 ms, 3x (`2026-10-23-dsv2-lite-p4-it4-def-01`, GPU 0,1 only, 64gb)

KD / KDb / KDc use the working-tree `default.toml` (`save_batch_steps =
32`, `save_batch_ms = 250.0`, confirmed in the log). KOff sets batching
off as the in-run control. Host check: no other pegaflow server started
or stopped during the cold passes.

| arm | cold TTFT p50 | vs baseline | cold req/s | vs baseline | prefill | G4 cold p50 / req/s | warm2 / warm3 p50 | save RPCs (cold) | hold mean / max ms (cold) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1811 | | 5.693 | | 5.58 | | 0.161 / 0.158 | | |
| KOff | 0.2024 | +11.8% | 5.550 | -2.5% | 6.07 | fail / fail | 0.097 / 0.105 | 169 | 20 / 101 |
| KD | 0.1907 | +5.3% | 5.610 | -1.5% | 6.12 | fail / pass | 0.094 / 0.107 | 24 | 144 / 282 |
| KDb | 0.1906 | +5.3% | 5.609 | -1.5% | 6.14 | fail / pass | 0.097 / 0.100 | 25 | 124 / 279 |
| KDc | 0.1891 | +4.4% | 5.609 | -1.5% | 6.12 | fail / pass | 0.096 / 0.108 | 24 | 143 / 281 |

All saving arms: E_ext 1.0, excess 0, no eviction, the same warm hits,
`flushes=0`.

Negative result: the 250 ms bound halves the gain. It caps the hold (max
282 ms vs 1.1 s), but it sends 24 RPCs per cold pass against KB32's 15,
and cold is +4.4 to +5.3% against KB32's +0.7 to +2.4%. Under this load a
cold step takes about 40 to 60 ms, so 250 ms is only 4 to 6 steps.
Batching off is +11.8% in this run (v2 ranged +5 to +20% across runs).
The cold cost tracks save RPCs per pass: 169 -> 24 -> 15 gives about
+8 to +12% -> +5% -> +1.5%. The working-tree default stays uncommitted.

Next hypothesis (one change): a larger batch with a looser time bound,
which sends fewer RPCs than KB32 while still capping the hold. Sweep: 64
steps alone, 64 steps or 1000 ms, 32 steps or 1000 ms. The best one then
gets the 3x repeat.

### Sweep (`2026-10-23-dsv2-lite-p4-it4-sweep-01`, GPU 0,1 only, 64gb, run_gate pgid 2017502)

Pressure columns come from `~/kexi/rsi-tools/pressure.py`. It reads vLLM's
10 s "Engine 000" lines (max waiting, max GPU KV usage) and the worker
`save stats`. Holds and RPCs are summed over all four passes. vLLM's
preemption histogram has no le=0 bucket, and the gate does not scrape
`num_preemptions_total`, so the record only shows "requests preempted
more than once": 0 for every arm. "Free GPU blocks" is the inverse of max
KV usage. This workload never goes above 28%, so it applies no block
pressure. The cap workload (G5) does, and it has not been run yet.

| arm | cold p50 | vs baseline | cold req/s | warm2 / warm3 p50 | RPCs (4 passes) | hold mean / max ms | max waiting | max KV usage |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.1897 | | 5.675 | 0.164 / 0.163 | | | 0 | 25.6% |
| KOff | 0.1989 | +4.9% | 5.562 | 0.100 / 0.104 | 498 | 11 / 99 | 6 | 25.6% |
| KB32 | 0.1810 | -4.6% | 5.667 | 0.099 / 0.103 | 50 | 209 / 1030 | 2 | 25.5% |
| KB64 | 0.1882 | -0.8% | 5.673 | 0.103 / 0.110 | 28 | 572 / 1554 | 0 | 28.2% |
| KB64T1000 | 0.1904 | +0.4% | 5.668 | 0.097 / 0.103 | 27 | 359 / 1078 | 0 | 26.4% |
| KB32T1000 | 0.1916 | +1.0% | 5.667 | 0.097 / 0.100 | 51 | 247 / 1068 | 0 | 25.6% |

Every arm: E_ext 1.0, excess 0, no eviction, the same warm hits,
`flushes=0`.

Findings:
- Every batched arm is within 2% of baseline on cold p50 and req/s.
  KOff is +4.9% in this run. Between 27 and 51 RPCs the differences
  (-4.6% to +1.0%) are run noise: the KB32 3x repeat spread was 1.7 pp.
- 64 steps adds hold (mean 0.57 s, max 1.55 s) for no measurable gain
  over 32. The 1000 ms bound trims the 64-step hold to about KB32's level.
  At 32 steps it rarely fires (51 vs 50 RPCs), so it costs nothing, and it
  caps the hold when steps are slow (long prefills, a busy GPU).
- Chosen default: `save_batch_steps = 32`, `save_batch_ms = 1000.0`, on
  by default. The time bound is a safety cap, not the trigger. It has
  no measured cold cost here (KB32T1000 +1.0%, within the KB32 repeat
  spread). The pressure-based flush the coordinator suggested is not
  needed at this load (no waiting, KV under 30%). It is the fallback if
  G5's cap run shows added preemptions or lower E_C.

### Sync saves skip batching

Layouts whose saves are synchronous (vLLM may zero or copy the blocks
after the step, e.g. hybrid models) wait for every step's copy before the
next step, so nothing can be merged. `_release_due` now returns True when
`sync_saves` is set, whatever the knobs say.
`test_layouts_that_zero_or_copy_save_synchronously` sets 32 steps / 1 s on
the worker and checks that a sync layout sends its RPC in the same step
(`rpcs == 1`, nothing held) while an async layout holds it. 172 core
tests pass.

### Qwen3-4B TP1 non-regression (`2026-10-23-qwen3-4b-p4-it4-nonreg`, GPU 0, 64gb, rev it4-default-32-1000-wip)

Arms: plain, baseline, KOff (batching off), v2 with the shipped default
(32 steps / 1000 ms, confirmed in the log). gsm8k is the full 1319.

| arm | pass | gsm8k acc | gsm8k cold / warm s | replay cold p50 | vs baseline | cold req/s | warm / warm2 / warm3 p50 | RPCs (4 passes) | hold mean / max ms |
|---|---|---|---|---|---|---|---|---|---|
| plain | | 0.8582 | 149.4 / 151.3 | | | | | | |
| baseline | pass | 0.8582 | 172.4 / 173.3 | 0.2194 | | 5.016 | 0.2225 / 0.2195 / 0.2192 | | |
| KOff | pass | 0.8582 | 173.3 / 171.2 | 0.2150 | -2.0% | 4.990 | 0.1259 / 0.0962 / 0.0941 | 521 | 23 / 154 |
| v2 default | pass | 0.8582 | 172.7 / 170.9 | 0.2197 | +0.1% | 5.009 | 0.1500 / 0.0944 / 0.0918 | 51 | 249 / 1124 |

Every verdict passes for both saving arms. G2: gsm8k outputs 0 differ,
digests 0 mismatched (v2: 6903 external / 169918 local matched). G3:
E_ext 1.0, excess 0, load_errors 0, no eviction. Hits: gsm8k warm local
0.917 / ext 0.075 / total 0.992 / H* 0.992; replay warm local 0.730 / ext
0.268 / total 0.998 / H* 0.998, the same for KOff and v2. Max waiting 0,
max KV usage 30%, no request preempted more than once.

v2's first warm pass (p50 0.150, p99 0.413) is slower than KOff's (0.126 /
0.204), while warm2 / warm3 match. The gate scores warm2 / warm3, so G4
passes either way, but a warm cost from the hold would matter. The per
request probe shows the same steps-to-first-prefill in both arms (mean
7.4, max 14), the same hits, and no extra episodes, so no request loaded
less or waited on a save. A host check found two co-tenant servers
allocating pinned pools during that pass: the server agent's 25 GiB pool
(GPU 6, 23:27:21 to 23:27:39) and P5's 64 GiB pool (GPU 4,5, 23:27:43 to
23:28:41). v2's warm pass ran 23:27:38 to 23:27:45.

Repeat (`2026-10-23-qwen3-4b-p4-it4-qrep-b`, GPU 0, replay only, the
shipped default 3x plus KOff):

| arm | cold p50 | vs baseline | cold req/s | warm / warm2 / warm3 p50 | warm p99 | RPCs | hold mean / max ms |
|---|---|---|---|---|---|---|---|
| baseline | 0.2178 | | 5.015 | 0.2224 / 0.2264 / 0.2197 | 0.723 | | |
| KOff | 0.2178 | 0.0% | 4.998 | 0.1072 / 0.0954 / 0.0910 | 0.190 | 466 | 21 / 149 |
| KD | 0.2180 | +0.1% | 5.005 | 0.1103 / 0.0967 / 0.0938 | 0.181 | 51 | 251 / 1106 |
| KDb | 0.2191 | +0.6% | 5.002 | 0.1105 / 0.0948 / 0.0931 | 0.183 | 51 | 248 / 1092 |
| KDc | 0.2180 | +0.1% | 5.001 | 0.1077 / 0.0950 / 0.0921 | 0.185 | 51 | 248 / 1092 |

All four saving arms pass G4 replay (the gsm8k G4 verdicts are
unevaluable because the run is replay only). Same hits, E_ext 1.0, excess
0, no eviction, max waiting 3, max KV usage 30%. The first warm pass is
0.108 to 0.111 for the default against 0.107 for KOff, so the 0.150 above
was that pass alone and not the hold. On Qwen3-4B TP1 the RPC count does
not move cold (KOff is at baseline): the cold cost is a TP2 effect (the
NCCL SHM all-reduce, see it3b). Batching neither helps nor hurts here.

The first attempt (`2026-10-23-qwen3-4b-p4-it4-qrep`) died in its third
arm. vLLM's API server failed with `OSError: [Errno 98] Address already
in use` on the port the gate picked, and run_gate exited with it. This is
a race between the gate's free-port pick and vLLM's bind while other
agents start servers. It is a gate issue (reported, not edited). The
baseline and KOff arms it finished agree with the repeat.

### Pressured check (`2026-10-24-dsv2-lite-p4-it4-press-01`, GPU 0,1, 64gb, `--num-gpu-blocks 1200`, run_gate pgid 2131540, 00:11:42 to 00:25:10Z)

`--num-gpu-blocks 1200` (vLLM's override; the profiled size is 6074) is a
non-official flag, recorded here. It puts 19,200 tokens of GPU KV under
8 conversations of up to 7.6k tokens. Arms: baseline, KOff (batching off),
KD (the shipped default, 32 steps / 1000 ms). A read-only sidecar
(`~/kexi/rsi-tools/metrics_poll.py`) scraped each vLLM server's
`/metrics` every 0.5 s: `num_preemptions_total`, `num_requests_waiting`,
`kv_cache_usage_perc`. Host check: no other pegaflow server started or
stopped during the cold passes (P5 ended 00:12:03, before the first
cold pass at 00:13:59).

| arm | cold p50 | vs baseline | cold req/s | warm / warm2 / warm3 p50 | warm p99 | preemptions (4 passes) | max waiting | samples KV >= 90% | RPCs | hold mean / max ms (cold) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.4193 | | 4.317 | 0.410 / 0.397 / 0.395 | 1.505 | 4 | 6 | 18 | | |
| KOff | 0.2968 | -29.2% | 5.017 | 0.152 / 0.134 / 0.134 | 0.940 | 4 | 4 | 20 | 541 | 17 / 71 |
| KD | 0.4537 | +8.2% | 4.658 | 0.163 / 0.140 / 0.130 | 1.093 | 3 | 6 | 21 | 74 | 376 / 926 |

Every saving arm: G2 replay digests 0 mismatched, G3 E_ext 1.0, excess 0,
load_errors 0, no eviction. Cold hits: local 0.56 / ext 0.21 / total 0.77
for both saving arms (baseline 0.56 local only). Warm hits: total 0.998,
H* 0.998; the local / external split moves (KOff ext 0.546, KD ext 0.468 /
0.458 / 0.395) because which turns still sit in GPU KV after the poison
is timing-dependent at this size.

Under pressure the saving arms beat baseline in cold: a turn that vLLM's
small GPU cache already dropped is a pegaflow hit (ext 0.21). Batching
makes that much worse:
- KD cold is 0.454 against KOff's 0.297 (+53%), and req/s 4.66 against
  5.02. It is behind baseline too (+8.2%).
- Preemptions do not explain it (3 vs 4) and nor do the hits (the same
  ext 0.21 in cold). The difference is the hold: in cold a finished
  request waits a mean of 376 ms (max 926 ms) for `finished_sending`
  against 17 ms (max 71 ms) for KOff, and vLLM cannot free its blocks
  in that time. With KV usage at 90 to 100%, those are the blocks the
  waiting next turn needs, so the next turn may queue behind the batch
  bound. This is the likely reading, not a measured one: the sidecar
  shows waiting requests in every arm (baseline in 36 of 115 samples),
  and vLLM's usage counts held blocks as used, so it cannot separate
  "waiting on held blocks" from "waiting on live blocks". The it5 run
  tests it directly.
- Warm is close (KD within 7% of KOff on p50, warm3 better), because warm
  saves almost nothing (the blocks are already stored).

So holding blocks costs real latency under KV pressure. Per the
coordinator's rule, this needs a pressure-based flush before the default
can stay on.

## it5 (2026-10-24): send held saves under KV pressure

Hypothesis: sending held saves at once on steps where vLLM's GPU KV cache
is nearly full removes the batching cost under pressure (KD back to KOff
or better in the pressured check), while leaving the unpressured cold
gain intact (KV usage stays under 30% there, so the flush never fires).

Design (one knob, computed from scheduler state):
- `Policy.save_batch_max_kv_usage` (default.toml `0.9`; > 1 disables).
  The connector's `build_connector_meta` reads vLLM's
  `kv_cache_manager.usage` (vLLM binds the manager via
  `bind_kv_cache_manager`; this is the same value vLLM exports as
  `kv_cache_usage_perc`, measured after the step's allocations) and
  passes it to `Scheduler.plan_saves(kv_usage=...)`.
- At or above the threshold the step's plan carries `send_saves=True`.
  The worker then sends its held jobs together with this step's save,
  without waiting for them (unlike `flush`, the step does not block).
  The finished requests' `finished_sending` follows as soon as that copy
  lands.
- `save stats` gains `pressure_rpcs` (RPCs sent early because of
  pressure).
- Unit test `test_kv_pressure_sends_held_saves` (low / at-threshold /
  disabled): the scheduler sets `send_saves` only at or above the
  threshold, never `flush`; the worker sends both held steps in one RPC
  when it is set and keeps holding when it is not. 210 tests pass
  (connector_v2 core + semantics bindings), ruff clean.

### Pressured check with the KV-pressure send (`2026-10-24-dsv2-lite-p4-it4-press-02`, GPU 0,1, 64gb, `--num-gpu-blocks 1200`, run_gate pgid 2151580, 00:35:21 to 00:53:20Z, rev it4-default-32-1000-kv090)

The same settings as press-01, with the it5 tree (`save_batch_max_kv_usage
= 0.9`, confirmed in the `save stats` lines). KD and KDb are the shipped
default twice. Host check: P5's Qwen3-4B servers (GPU 4) started at
00:37:26 and 00:42:03. Their 64 GiB pool allocations overlap baseline's
and KOff's cold passes, not KD's or KDb's. KD and KDb are judged
against KOff from both runs.

| arm | cold p50 | vs baseline | cold req/s | warm / warm2 / warm3 p50 | warm p99 | preemptions | max waiting | RPCs (pressure) | hold mean / max ms (cold) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.4193 | | 4.304 | 0.417 / 0.394 / 0.397 | 1.563 | 4 | 5 | | |
| KOff | 0.3031 | -27.7% | 4.995 | 0.149 / 0.143 / 0.155 | 0.967 | 5 | 8 | 549 | 17 / 72 |
| KD | 0.3201 | -23.7% | 4.854 | 0.154 / 0.144 / 0.147 | 0.981 | 3 | 6 | 273 (230) | 258 / 920 |
| KDb | 0.2900 | -30.8% | 4.903 | 0.157 / 0.140 / 0.142 | 0.984 | 3 | 5 | 279 (237) | 267 / 920 |
| press-01 KOff | 0.2968 | | 5.017 | 0.152 / 0.134 / 0.134 | 0.940 | 4 | 4 | 541 | 17 / 71 |
| press-01 KD (no pressure send) | 0.4537 | | 4.658 | 0.163 / 0.140 / 0.130 | 1.093 | 3 | 6 | 74 | 376 / 926 |

Every saving arm: replay digests 0 mismatched, E_ext 1.0, excess 0,
load_errors 0, no eviction, cold total hit 0.769, warm total 0.998 /
H* 0.998. Preemptions and max waiting come from the sidecar and are
cumulative over the arm's four passes.

Findings:
- The pressure send closes the press-01 gap. Cold p50 goes from 0.454 to
  0.320 / 0.290, against KOff's 0.297 / 0.303. Warm is level with KOff.
  No added preemptions (3 against KOff's 4 / 5) and no deeper queue.
- Cold req/s stays 2 to 3% below KOff (4.85 / 4.90 against 5.00 / 5.02)
  in both runs, while still 13% above baseline. The cold hold mean is
  still about 260 ms. Most of it is held while KV usage is under 90%,
  where the pressure send does not fire. A lower threshold would cut
  that further at the price of more RPCs; not tried (one hypothesis per
  iteration).
- The pressure send fires on most RPCs in this setting (230 of 273):
  under this pressure the default behaves like a step-sized batch of
  about 2 steps, which is the intent.

### it5 3x in-run repeat, unpressured (`2026-10-24-dsv2-lite-p4-it5-rep-01`, GPU 0,1, 64gb, run_gate pgid 2169351, 00:54:26 to 01:16:00Z, rev it5-kv090)

Arms: baseline, KOff, and the shipped default (32 steps / 1000 ms / KV
0.9) three times (KD / KDb / KDc). Host check: no other pegaflow server
started or stopped from 00:54 to the end of the run.

| arm | cold p50 | vs baseline | cold req/s | vs baseline | G4 cold p50 / req/s | warm2 / warm3 p50 | RPCs (cold) | pressure RPCs | hold mean / max ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1904 | | 5.696 | | | 0.164 / 0.164 | | | |
| KOff | 0.2000 | +5.0% | 5.581 | -2.0% | fail / fail | 0.096 / 0.108 | 168 | | 11 / 97 |
| KD | 0.1916 | +0.6% | 5.671 | -0.4% | pass / pass | 0.107 / 0.109 | 14 | 0 | 309 / 1070 |
| KDb | 0.1876 | -1.5% | 5.666 | -0.5% | pass / pass | 0.097 / 0.106 | 14 | 0 | 244 / 1066 |
| KDc | 0.1867 | -1.9% | 5.679 | -0.3% | pass / pass | 0.095 / 0.109 | 15 | 0 | 308 / 1068 |

Every saving arm: replay digests 1382 external / 7094 local matched, 0
mismatched; E_ext 1.0, excess 0, load_errors 0, no eviction; warm hits
local 0.705 / ext 0.293 / total 0.998 / H* 0.998; `flushes=0`; max KV
usage 28%, so the pressure send never fired (`pressure_rpcs=0`), as
intended. G4 replay passes 3 of 3 for the default (cold p50 -1.9% to
+0.6%, cold req/s -0.3% to -0.5%) and fails for KOff. Every arm's
overall verdict is False only because this run is replay only: the gsm8k
verdicts are unevaluable.

Order fix after this run (unit-tested, not in the run's tree): the
worker checked `send_saves` before the batch bound, so a step where both
fired counted as a pressure RPC. Now the bound is checked first;
`pressure_rpcs` counts only early sends. Behaviour is unchanged.
