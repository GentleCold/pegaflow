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
  `tp_rank 0`. The mapping is derived from seal_plan's holders and writer
  plus the public `register_context_batch(tp_rank, tp_size)` call, and from
  nothing else. The it2 gate log confirms the result:
  `storage_rank=0 writer=True/False`, and the server stores 27 x 18432 B
  per block. No pegaflow-core instance or slot code was read for this.
  Every other layout gets the identity mapping (Qwen3-4B TP1: unchanged).
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

## Policy budget (after review fix MF3, on top of fc06c9b)

The review (`docs/rsi/reviews/P4-dsv2-lite.md` M3) found that
`save_batch_ms` breaks the knob rule. It never moved a verdict, and it
never capped the hold either: the max cold hold is 1030 to 1121 ms with
no time bound and 1066 to 1106 ms with 1000 ms, because the bound is
checked once per step. It is now `Worker.SAVE_HOLD_MAX_S = 1.0`, a
staleness cap with no behaviour change (the shipped value was 1000 ms).
The unit cases cover it by patching the constant. No gate rerun.

- Knobs: 4 in total, all in `models/default.toml`, with no model profile
  (no `models/<arch>.toml`): `loading_wait_ms`, `sync_load_tokens`
  (pre-P4; `min_hit_tokens` was deleted upstream in fc06c9b), and the two
  P4 batching knobs `save_batch_steps` and `save_batch_max_kv_usage`.
- `policy.py`: no line is specific to a Kind. `accept` and
  `choose_saves` are the identity for every Kind, and the knobs are
  Kind-agnostic. `SaveItem.kind` is carried but no policy code reads it.
  P4 added 0 lines of per-Kind policy (MLA maps to `Full`).

Gate evidence per batching knob (DSV2 TP2, G4 replay cold p50 / cold
req/s verdicts, P = pass, F = fail). All are replay-only, non-official
runs. gsm8k G4 has not been evaluated with batching on.

| knob | evidence | changed alone? |
|---|---|---|
| `save_batch_steps` 1 -> 32 | `knob-01`: v2 (1) F/F, KB32 (32) P/P; `rep-01`: v2 F/F, KB32 x3 P/P, P/P, F/P; `sweep-01`: KOff F/P, KB32 P/P; `it5-rep-01`: KOff F/F, default x3 P/P (with the other knobs at their defaults) | yes, in knob-01 / rep-01 / sweep-01: cold moves from +5 to +20% to -4.6 to +2.4% |
| `save_batch_max_kv_usage` > 1 -> 0.9 | `press-01` (none): KD cold p50 F (+8.2% vs baseline, +53% vs KOff); `press-02` (0.9): KD, KDb P/P (cold p50 -23.7 / -30.8%) | yes (the same tree otherwise, the same `--num-gpu-blocks 1200`) |
| `Worker.SAVE_HOLD_MAX_S = 1.0` (constant, not a knob) | `sweep-01`: KB32 P/P, KB32T1000 P/P (50 vs 51 RPCs). The 250 ms variant (`def-01`) moved cold p50 to F 3 of 3. | kept only as a staleness cap: other requests see a step's blocks in the store within about 1 s plus one step when steps are slow |

## Review fixes without a gate rerun (2026-10-24)

- S6, unknown KV usage: if vLLM's KV cache manager is not bound, usage
  used to read as 0.0, which silently turned off the KV-pressure send
  (the press-01 regime). Now `plan_saves(kv_usage=None)` counts as
  pressure, so every step sends (batching off). The scheduler warns once,
  and `save stats` gains `kv_usage_unknown_steps`. Unit case
  `test_kv_pressure_sends_held_saves[unknown]`.
- S1, merged RPC size: a merged save RPC repeats every held block id and
  hash in every layer entry. At the gate's `--max-num-batched-tokens 2048`
  it stays under about 4 MB (128 blocks x 32 steps x 27 layers x about
  37 B). At 16384 tokens on a 61-layer model it would reach about 74 MB,
  over the 64 MiB gRPC limit (`python/src/lib.rs`), and every merged save
  would fail as a warning only. `Worker.SAVE_RPC_MAX_BYTES = 16 MiB` now
  caps it: a step whose job would push the held batch past the budget
  sends the batch first. `_wire_bytes` estimates the size from the
  protobuf layout (name plus framing per entry; a uint32 id and a
  length-prefixed hash per block). A single step larger than the budget
  still goes alone, as it did unbatched. No change at the gate's settings
  (4 MB < 16 MiB), so no gate rerun. Unit case
  `test_merged_save_rpc_stays_under_size_budget`.

## Run provenance rule (from 2026-10-24, review S2)

Up to now, two things make P4 runs hard to tie to code. First, the remote
tree was synced by tar, not checkout. Second, `.rsi-rev` (recorded as
summary.json `rev`) was free-form from it4 on. Per run dir under
`~/kexi/rsi-runs/` (arms read from the run's own files):

| runs | `rev` | p4probe arms | real `PegaKVConnectorV2` arm |
|---|---|---|---|
| it1 slice/replay, old-slice, it2-slice, qwen it2-nonreg | b358ceb / 3d1de18 (commits) | none | yes |
| it3 probe/defer/norpc/rpc/batch | 3d1de18 / 1bfb590 (commits) | NullV2, NoSaveV2, DeferV2, NoRpcV2, EmptyRpcV2, Batch8V2, Batch32V2, CountV2 | no |
| it4-knob-01, it4-rep-01 | `it4-wip` | KB32(b/c), KT100, KT250 | yes |
| it4-def-01, it4-sweep-01 | `it4-default-wip` | KD(b/c), KOff, KB32/64(T1000) | no |
| qwen it4-nonreg | `it4-default-32-1000-wip` | KOff | yes |
| qwen qrep / qrep-b, press-01 | `it4-default-32-1000-syncbypass` (qrep: none) | KD(b/c), KOff | no |
| press-02 | `it4-default-32-1000-kv090` | KD(b), KOff | no |
| it5-rep-01 | `it5-kv090` | KD(b/c), KOff | no |
| it5-heldout, qwen it5-nonreg | `it5-kv090` | KOff | yes |

What this means:
- At commit revs, the tree matched the commit, but the it3 probe arms came
  from an uncommitted module.
- Free-form revs name a working state that no commit pins. The it5
  `worker.py` at it5-rep-01, for example, predates the order fix noted
  above, and no commit holds it.
- KD/KDb/KDc are no-op subclasses, so they ran the tree's default code,
  but they are still not the shipped class path.

All these numbers stand as recorded, as probe evidence. None of them is
acceptance evidence. That comes from the official runs (MF1, MF2).

From now on:
- `.rsi-rev` holds the full commit sha the remote tree was synced from. If
  anything is uncommitted, append `-dirty` and the sha256 of `git diff`
  (rsi-g5's own tree snapshot replaces this once merged).
- Every arm runs committed code: `plain`, `baseline`, `old`, and
  `pegaflow.connector_v2:PegaKVConnectorV2`. A knob comparison runs
  separately from separate commits, or with a probe arm committed to the
  repo for review. No remote-only module.
- p4probe is retired: `p4probe.py` is renamed to `p4probe.py.retired`,
  so `p4probe:*` arms fail to import.

## Open items

- Under KV pressure the default's cold req/s stays 2 to 3% below batching
  off (`press-02`: 4.85 / 4.90 vs KOff 5.00; press-01 KOff 5.02). It is not
  a G4 fail (G4 compares against baseline, 4.30, which the default beats by
  13%), but it is a residual cost of batching. The cold hold is still about
  260 ms: most of it falls while KV usage is under 90%, where the
  pressure send does not fire. Candidates: a lower threshold, or a
  waiting-queue signal. Neither has been tried.
- The gate port race (vLLM `EADDRINUSE` on the gate's chosen port) is
  with the gate owner.
- DSV2 G5 at C/U 0.25: v2 is 0.06 to 0.12 below old across runs (old's
  own spread is 0.04). The cause is unknown after ruling out the resident
  deficit, block-0 query refresh, lost saves and server behaviour (section
  "M2"). Not pursued: below the 2x floor, not gated.
- MF5 `G2.digest_self.cap` (1.008%, 0 mismatched): the recorded rerun is
  M1. If it is over 1% again, the next step is the gate owner's (an
  allowance, or digesting preempted episodes on `feat/rsi-gate-hybrid`).
- lmcache is not compared yet: the arm is being fixed on
  feat/rsi-gate-lmcache.

## MF1: official run (`2026-10-24-dsv2-lite-p4-mf1`, GPU 0,1, TP2, 64gb, run_gate pgid 2454295, 05:40:44 to 08:09:39Z)

The setup:
- `"official": true`, `not_official_because: []`.
- Gate tree `~/kexi/p4-gate-07f5d96`. Its `rsi/` is identical to
  feat/semantics-rsi d16b990 (the rsi-g5 merge). It adds this model's
  gate.json and spec.json.
- Under test: `--under-test-path ~/kexi/p4-ut-07f5d96/python`, rev 07f5d96.
  Both trees came from `git archive` and carry no pyc. The snapshot is
  verified, the static scan has 0 flags and the provenance watch saw no
  change.
- CUDA 13 env, preflight problems [].
- Arms: plain, baseline, old, native, `pegaflow.connector_v2:PegaKVConnectorV2`.
  Workloads gsm8k (full 1319), replay (32 x 12) and cap (auto: 76 convs x
  12).
- Capacity floor: C(0.5) is 6.87 GB, 2.27x GPU KV. No timed pass was
  contaminated.

`old` failed every stage with an HTTP 500. vLLM 0.32's MultiConnector
looks up `PegaKVConnector` in its registry to rebuild kv stats, and the
gate's `VLLM_PLUGINS=""` left the name unregistered. That was a gate
regression, fixed in 3fe119c (feat/semantics-rsi fff1d9b). `old` is
`required: false`, so the run stays official. `old`'s numbers come from
the supplementary run below.

### Verdicts

| check | v2 | native (record only) |
|---|---|---|
| G0.spec / G0.provenance | pass / pass | fail (LBHNC, known) / pass |
| G2.split_control (baseline) | pass: measured split-variant, 647/1319 differ (0.4905), 1319 logprobs differ, 1319 pre-warmed | |
| G2.gsm8k_outputs | pass: cold 0 differ; warm 0 differ vs plain `split` (aligned 0) | pass |
| G2.gsm8k accuracy (baseline 0.6725) | cold +0.0 pp, warm +0.53 pp (= plain split 0.6778) | +0.0 / +0.45 pp |
| G2.digest_self.gsm8k | 10526 ext matched, 0 mismatched | 9982 / 0 |
| G2.digest_vs_baseline.gsm8k | 10526 / 0; computed 45008/45008 exact, coverage cold 1.0, warm 1.0 | 9982 / 0; coverage warm 0.93 |
| G2.digest_self.replay | 27080 ext + 21898 local matched, 0 mismatched | 18012 / 0 |
| G2.digest_self.cap (exhaustive) | 441102 ext matched, 0 mismatched, 6892 preempted-uncertain (allowed) | 504498 / 0 |
| G3.gsm8k_warm / replay_warm | pass (E_ext 1.0 / 1.0, excess 0 / 0) | pass (excess 1230 / 210) |
| G3.no_eviction | pass | n/a |
| G4 (6 verdicts) | pass, all 6 | pass, all 6 |
| G5.replay_warm, hard parts | load_errors 0, excess 0, unmatched 0, pool_ok, loads_ok | same |
| G5.replay_warm in this run's summary | fail: E_C 0.513 < 0.9 | fail: E_C 0.631 |

When this run was judged, the E_C bar was still hard, so the run's own
summary has v2 `pass: false`, on `G5.replay_warm` only. Since
2026-10-24 (86d83fd) hit-rate levels are reported, not gated.
I re-judged the record offline with the fff1d9b `verdicts()` /
`hit_report()`. The output is `~/kexi/tmp/dsv2-p4/mf1-rejudge-fff1d9b.json`,
and the run dir is unchanged. The result:
- v2 `pass: true`.
- The only verdict that changed is `G5.replay_warm`, from false to true.
- `below_reference` lists only `efficiency_c`.

In the G5 curve, v2's `cap025` digest is false: 6206 of 600154 loaded
blocks (1.03%) are preempted-uncertain, against the 1% allowance, and 0
are mismatched. Curve points are not gated.

### G4 (median of timed passes vs baseline)

| metric | baseline | v2 | vs baseline | native | vs baseline |
|---|---|---|---|---|---|
| replay warm p50 (s) | 0.930 | 0.303 | -67.4% | 0.281 | -69.8% |
| replay warm req/s | 5.40 | 12.35 | +128.6% | 13.14 | +143.2% |
| replay cold p50 (s) | 1.323 | 0.665 | -49.8% | 0.623 | -52.9% |
| replay cold req/s | 5.03 | 8.40 | +66.9% | 7.92 | +57.5% |
| gsm8k cold s | 193.9 | 194.4 | +0.28% | 195.2 | +0.67% |
| gsm8k warm s | 194.0 | 189.5 | -2.27% | 190.8 | -1.61% |

At 32 x 12, the cold pass already reuses within the pass: a turn whose
history vLLM evicted is loaded from PegaFlow (cold H* 0.843 vs baseline
local 0.478). That is why cold is so far ahead of baseline here, unlike the
8 x 6 slices.

Save batching ran under KV pressure the whole time:
- Max GPU KV usage was 99.8% for v2 (baseline 95.3%).
- The pressure send fired on 1296 of 1346 replay RPCs and 3641 of 3725 cap
  RPCs.
- The replay finish-to-report hold was mean 132 ms, max 3003 ms. It covers
  the save itself, not just the batch hold.
- No request was preempted more than once.
- gsm8k: 15 finished requests, 3 RPCs.
- `kv_usage_unknown_steps` was 0 everywhere, so vLLM's cache manager was
  bound.

### Hits (unlimited store)

| arm | workload/pass | local | ext | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|---|
| baseline | gsm8k warm | 0.940 | 0 | 0.940 | 0.993 | 0 | 0 |
| v2 | gsm8k warm | 0.937 | 0.055 | 0.993 | 0.993 | 1.0 | 0 |
| native | gsm8k warm | 0.940 | 0.053 | 0.993 | 0.993 | 1.0 | 1230 |
| baseline | replay cold / warm2 | 0.478 / 0.465 | 0 | 0.478 / 0.465 | 0.843 / 0.998 | 0 | 0 |
| v2 | replay cold | 0.492 | 0.351 | 0.843 | 0.843 | 1.0 | 0 |
| v2 | replay warm / warm2 / warm3 | 0.464 / 0.469 / 0.470 | 0.535 / 0.529 / 0.529 | 0.998 | 0.998 | 1.0 | 0 |
| native | replay cold | 0.500 | 0.343 | 0.842 | 0.843 | 0.9998 | 0 |
| native | replay warm2 | 0.467 | 0.531 | 0.998 | 0.998 | 1.0 | 210 |

### G5 (cap trace, warm_timed; `ref_local` = baseline's local)

| arm | C/U | C (GB) | hit | H*_C | H_lru | E_C | E_C own-local | local |
|---|---|---|---|---|---|---|---|---|
| v2 | 0.25 | 3.43 | 0.428 | 0.853 | 0.673 | 0.007 | 0.051 | 0.405 |
| native | 0.25 | 3.43 | 0.570 | 0.855 | 0.657 | 0.266 | 0.357 | 0.417 |
| v2 | 0.5 | 6.87 | 0.765 | 0.925 | 0.847 | 0.513 | 0.695 | 0.401 |
| native | 0.5 | 6.87 | 0.852 | 0.924 | 0.847 | 0.631 | 0.855 | 0.423 |
| v2 | 0.75 | 10.30 | 0.855 | 0.968 | 0.917 | 0.581 | 0.799 | 0.405 |
| native | 0.75 | 10.30 | 0.934 | 0.968 | 0.917 | 0.713 | 0.936 | 0.435 |

H_freeze at 0.5 is 0.714 (freeze/H*_C 0.548), so the trace separates
policies. Per pass at 0.5, v2's hit goes 0.824 (cold, E_C 0.916) / 0.791 /
0.791 / 0.738, and native's 0.840 / 0.851 / 0.851 / 0.853.

The server at 0.5 (v2):
- Pool proven: `pool_capacity_bytes` = C.
- `pegaflow_pool_used_bytes` / C at pass end: 0.97 to 1.00.
- `pegaflow_cache_resident_bytes` / C: 0.64 to 0.77. At 0.25 it is 0.59 to
  0.72, at 0.75 it is 0.66 to 0.76.
- Evictions per pass: 18k to 38k blocks.
- Load bytes are 2x the external-hit bytes (116.8 vs 58.4 GB), because
  both TP ranks load each replicated MLA block (design: every holder
  loads).

### Hit-rate judgment

With unlimited capacity, v2 is at the oracle:
- gsm8k warm and every replay pass reach `H*` exactly (E_ext 1.0), with
  excess 0.
- native reaches the same hit with excess 1230 / 210.
- Close enough: there is no gap.

With bounded capacity, v2 trails the LRU model and native:
- At C/U 0.5 it is 0.082 below H_lru and 0.087 below native, and 0.16
  below H*_C.
- At 0.25 it gets almost no external hit (E_C 0.007, native 0.266).
- At 0.75 it is 0.06 below H_lru.

The evidence puts this gap in the server's retention, not in the
connector:
1. The connector-side contract holds. There is no excess, no load error and
   no unmatched request, every external hit is backed by server load bytes,
   and the unlimited runs are at H*. With one Full group and one MLA copy
   per block, there is no per-group or Kind miss to explain.
2. The server holds only 0.6 to 0.77 C of cache-resident blocks while its
   pool is 0.92 to 1.0 used. The LRU model keeps all of C resident. A store
   that keeps about 0.65 C acts like LRU at a smaller C. That fits v2 at
   0.5 (0.765) falling between H_lru at 0.25 (0.673) and at 0.5 (0.847),
   and v2 at 0.25 getting almost nothing.
3. It is the same mechanism as the Qwen3-4B G5 r3 record (resident 0.74
   to 0.87 C, v2 0.842 vs H_lru 0.857). The gap is larger here, and so is
   the shortfall in resident bytes.
4. One more observation, not explained: v2's saved bytes over the four
   passes (66.3 GB) exceed the LRU model's stored bytes (53.9 GB) by 23%.
   Explaining it needs the server's save accounting, which is
   pegaflow-core (server-owned, not read here).

Judgment, after the coordinator's review of this section:
- G3 (unlimited) is close enough: v2 is at H*.
- G5 on this server is not close enough. v2 is below H_lru and below
  native at every capacity (0.5: 0.765 vs 0.847 / 0.852; 0.25: 0.428 vs
  0.673 / 0.570). That is "差太多", too far off.
- Cache-resident 0.6 to 0.77 C while the pool is 0.92 to 1.0 used means
  pinned bytes stranded in regions with evicted blocks.
- Superseded: the server stays at base (user decision, 2026-10-24), and
  the H4 run was cancelled. Points 2 and 3 above wrongly put the whole gap
  on the server. The old supplement (old reaches H_lru on the same
  server) and the batching A/B show that most of it is connector-side:
  batched saves folded several steps into one entry per layer. The final
  judgment is under "G5 hit-rate judgment (final)".

## MF2: official Qwen3-4B non-regression (`2026-10-24-qwen3-4b-p4-mf2`, GPU 0, TP1, 64gb, run_gate pgid 2605995, 08:12:35 to 10:29:34Z)

The setup:
- `"official": true`, `not_official_because: []`.
- Gate tree `~/kexi/p4-gate-d56fcb2`, whose `rsi/` is identical to
  feat/semantics-rsi fff1d9b (old-arm plugin fix, hit rate reported). Under
  test: d56fcb2 (`~/kexi/p4-ut-d56fcb2`).
- Base server, sha256 d1970ead, the same binary as it2d and G5 r3.
- Arms plain, baseline, old, v2. Workloads gsm8k (1319), replay (32 x 12)
  and cap (64 x 12).
- Capacity floor 2.44x GPU KV, no contamination. `vllm_plugins`: old
  `pegaflow`, every other arm "".

The pool size differs from it2d's 160gb. Measured U is 32.7 GB for
gsm8k and 27.95 GB for replay, both under 64 GB, and `G3.no_eviction`
passes for v2 and old. So the unlimited passes did not evict, and the
pool difference does not invalidate the comparison.

### Verdicts and G4 vs it2d (`2026-10-23-qwen3-4b-p3-it2d`, 8a05b7c)

Every v2 verdict passes (`pass: true`). G2:
- gsm8k outputs: 0 differ cold and warm (split-invariant: control 0/1319).
- Accuracy delta 0.0 pp.
- Digests: gsm8k 7163 ext matched, replay 18592, cap 230419 (1285
  uncertain, all preempted). 0 mismatched.

| verdict | it2d v2 | MF2 v2 | MF2 old |
|---|---|---|---|
| replay warm p50 vs baseline | -78.4% | -85.3% | -78.2% |
| replay warm req/s vs baseline | +193% | +293% | +294% |
| replay cold p50 vs baseline | -54.1% | -65.3% | -63.4% |
| replay cold req/s vs baseline | +105% | +152% | +149% |
| gsm8k cold s vs baseline | +0.92% | +0.80% | +2.51% (fail) |
| gsm8k warm s vs baseline | +0.01% | +0.07% | +1.76% |

The MF2 baseline is slower than it2d's (replay warm p50 3.87 vs 2.55 s).
In absolute terms, v2's warm p50 is 0.567 vs 0.550 s and its warm req/s
9.71 vs 10.03. G4 judges every verdict against the same run's baseline,
and all pass, as they did for it2d.

### Hits

| arm | pass | local | ext | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|---|
| v2 | gsm8k warm | 0.914 | 0.078 | 0.992 | 0.992 | 1.0 | 0 |
| old | gsm8k warm | 0.925 | 0.068 | 0.993 | 0.992 | 1.0 | 1170 |
| v2 | replay cold | 0.381 | 0.483 | 0.864 | 0.864 | 1.0 | 0 |
| v2 | replay warm2 | 0.348 | 0.651 | 0.998 | 0.998 | 1.0 | 0 |
| old | replay warm2 | 0.340 | 0.658 | 0.998 | 0.998 | 1.0 | 195 |
| it2d v2 | replay warm2 | 0.348 | 0.651 | 0.998 | 0.998 | 1.0 | 0 |

| arm | C/U | hit | H*_C | H_lru | E_C | resident/C (4 passes) |
|---|---|---|---|---|---|---|
| v2 MF2 (d56fcb2) | 0.25 | 0.368 | 0.865 | 0.588 | 0.029 | 0.64 to 0.73 |
| old MF2 | 0.25 | 0.392 | 0.862 | 0.583 | 0.068 | 0.82 to 0.91 |
| v2 r3 (8a05b7c) | 0.25 | 0.399 | 0.865 | 0.581 | 0.084 | 0.74 to 0.87 |
| v2 MF2 | 0.5 | 0.777 | 0.929 | 0.857 | 0.592 | 0.59 to 0.69 |
| old MF2 | 0.5 | 0.840 | 0.930 | 0.857 | 0.673 | 0.75 to 0.83 |
| v2 r3 | 0.5 | 0.842 | 0.929 | 0.857 | 0.684 | 0.76 to 0.83 |
| v2 MF2 | 0.75 | 0.865 | 0.971 | 0.916 | 0.673 | 0.65 to 0.74 |
| old MF2 | 0.75 | 0.879 | 0.971 | 0.915 | 0.688 | 0.71 to 0.85 |
| v2 r3 | 0.75 | 0.876 | 0.971 | 0.916 | 0.669 | 0.74 to 0.81 |

### Hit-rate judgment: G5 regressed

- Unlimited (G3): no change. v2 is at H* on every pass, the same as it2d.
- Capacity (G5): **v2 regressed against its own 8a05b7c and against old in
  the same run.**
  - At 0.5, v2's hit is 0.777, against old 0.840 and r3 v2 0.842.
  - At 8a05b7c, v2 was 0.835 to 0.842 in four base-server cap runs (r3,
    srv-reclaim-cap3-base / -b), level with old (0.828 to 0.832).
  - This is "差太多", too far off, and it is new in P4.
- The signature: the server's cache-resident bytes fall from 0.76 to 0.83
  C to 0.59 to 0.69 C at the same pool use. DSV2 MF1 shows the same
  pattern (resident 0.64 to 0.77 C, v2 below native).

The leading suspect is save batching, on by default since 3e6f7a3. It is
the only P4 change on a dense model's save path. MLA and the TP2 single
writer do not apply to Qwen3-4B. Batching merges up to 32 steps of saves
into one RPC (here mostly pressure sends, 4001 of 4089 cap RPCs at 0.5).
That changes how saves reach the server's allocator. A plausible
mechanism is fragmentation under bursty saves, but it is not proven, and
the allocator is pegaflow-core (not read).

The test is a cap-only run with batching off (`save_batch_steps = 1`)
against the default, on the same server, from committed trees. If the
resident drop and the hit gap go away, batching costs G5 and the default
has to change (for example, a byte cap well below 16 MiB, or batching
off when the store is near full).

## DSV2 old supplement (`2026-10-24-dsv2-lite-p4-old-supp`, GPU 0,1, TP2, 64gb, run_gate pgid 2749223, 10:33:40 to 12:01:04Z)

Run setup:
- Arms plain, baseline, old. Workloads gsm8k, replay, cap. The gate tree
  `rsi/` is identical to 4fa78a0 (old-arm plugin fix).
- Base server d1970ead.
- Cap plan 76 x 12, the same as MF1. C(0.5) = 6.87 GB, 2.27x GPU KV.
- Non-official only because there is no under-test arm, which is the
  purpose of this run.

Verdicts:
- old passes every verdict: G0, G2 (gsm8k 10390 ext matched, cap 490342
  matched with 6224 uncertain, 0 mismatched), G3 (E_ext 1.0, excess
  1230 / 210), G4, and G5 loads/pool.
- old's replay warm p50 is 0.388 s vs baseline 0.929. Cold p50 is 0.698
  vs 0.909.

| arm (run) | C/U | hit | H*_C | H_lru | E_C | resident/C |
|---|---|---|---|---|---|---|
| old (old-supp) | 0.25 | 0.531 | 0.853 | 0.667 | 0.192 | 0.85 to 0.94 |
| v2 (MF1, 07f5d96) | 0.25 | 0.428 | 0.853 | 0.673 | 0.007 | 0.59 to 0.72 |
| native (MF1) | 0.25 | 0.570 | 0.855 | 0.657 | 0.266 | |
| old | 0.5 | 0.845 | 0.925 | 0.847 | 0.609 | 0.85 to 0.94 |
| v2 (MF1) | 0.5 | 0.765 | 0.925 | 0.847 | 0.513 | 0.64 to 0.77 |
| native (MF1) | 0.5 | 0.852 | 0.924 | 0.847 | 0.631 | |
| old | 0.75 | 0.889 | 0.968 | 0.917 | 0.638 | 0.85 to 0.94 |
| v2 (MF1) | 0.75 | 0.855 | 0.968 | 0.917 | 0.581 | 0.66 to 0.76 |
| native (MF1) | 0.75 | 0.934 | 0.968 | 0.917 | 0.713 | |

On the same base server and the same trace, old keeps 0.85 to 0.94 C
resident and reaches H_lru at 0.5 (0.845 vs 0.847). v2 keeps 0.64 to
0.77 C. Most of v2's shortfall against H_lru is therefore not a server
ceiling. The server lets a connector reach H_lru at this capacity, and
the difference is in what v2 sends. MF1's v2 ran with batching on
(07f5d96; its connector_v2 is identical to d56fcb2, the Qwen A/B's arm A).

## Batching A/B for the G5 regression (Qwen3-4B, cap only, GPU 0, base server d1970ead, 64gb)

Hypothesis: save batching (default on since 3e6f7a3) causes MF2's G5
regression against 8a05b7c.

Setup:
- The gate tree for both runs is ce645d7. Each run has a same-run baseline,
  cap auto with 64 convs x 12 turns (the r3 plan), and C(0.5) = 27.2 GB.
- Non-official only because the arms and workloads are cap-only.
- A: under test d56fcb2 (the MF2 tree, `save_batch_steps = 32`), tag
  `qwen3-4b-p4-ab-a-d56fcb2`, pgid 2829266, 12:02:53 to 12:48:34Z.
- B: `exp/p4-batch-off` 706006c, exactly one line changed
  (`save_batch_steps = 1`), tag `qwen3-4b-p4-ab-b-706006c`, pgid 2870135,
  12:51:31 to 13:35:30Z.

| tree | C/U | hit | H*_C | H_lru | E_C | server insertions | resident/C |
|---|---|---|---|---|---|---|---|
| r3, 8a05b7c (no batching) | 0.5 | 0.842 | 0.929 | 0.857 | 0.684 | 93357 | 0.76 to 0.83 |
| MF2, d56fcb2 | 0.5 | 0.777 | 0.929 | 0.857 | 0.592 | 124661 | 0.59 to 0.69 |
| A, d56fcb2 | 0.5 | 0.794 | 0.930 | 0.857 | 0.596 | 122955 | 0.68 to 0.72 |
| B, batching off | 0.5 | 0.837 | 0.929 | 0.857 | 0.692 | 93731 | 0.80 to 0.82 |
| A | 0.25 | 0.367 | 0.863 | 0.578 | 0.025 | 352361 | 0.71 to 0.73 |
| B | 0.25 | 0.405 | 0.862 | 0.583 | 0.100 | 321802 | 0.79 to 0.96 |
| A | 0.75 | 0.867 | 0.970 | 0.915 | 0.677 | 83772 | 0.65 to 0.72 |
| B | 0.75 | 0.879 | 0.971 | 0.916 | 0.689 | 79562 | 0.74 to 0.81 |

Batching is the cause. With it off, v2 is back at its 8a05b7c level at
every fraction (0.5: 0.837 vs 0.842), and A reproduces MF2. Both runs
have excess 0, 0 unmatched and 0 load errors, so correctness is the same.

The mechanism, per pass at 0.5 (end of pass):

| | used/C | resident/C | dead = used - resident | ext share | hit |
|---|---|---|---|---|---|
| A cold / warm / warm2 / warm3 | 0.99 / 0.96 / 0.87 / 0.95 | 0.68 / 0.68 / 0.69 / 0.72 | 0.31 / 0.28 / 0.18 / 0.23 | 0.391 / 0.348 / 0.325 / 0.286 | 0.779 / 0.771 / 0.796 / 0.792 |
| B cold / warm / warm2 / warm3 | 0.97 / 1.00 / 0.96 / 0.97 | 0.81 / 0.80 / 0.82 / 0.80 | 0.16 / 0.20 / 0.14 / 0.17 | 0.451 / 0.421 / 0.384 / 0.326 | 0.842 / 0.857 / 0.854 / 0.820 |

- At the same pool use, A keeps about 0.12 C fewer bytes resident. The
  pinned-but-not-resident share is 0.25 C vs 0.17 C. That matches the base
  server's pinned-region limitation (PROGRESS "Server limitations": a save
  call takes one region per layer, freed only when the call's last block
  is evicted). Bigger save calls pin more dead bytes.
- Per cap stage, from the tp0 save stats at 0.5:
  - A sends 4076 RPCs, 3988 of them pressure sends, at 1.26 steps and
    31.7 blocks/layer per RPC.
  - B sends 5042 RPCs at 1.00 steps and 19.0 blocks per RPC.
  - KV usage is about 99%, so the pressure send fires nearly every step,
    and the merge itself is small (1.26 steps).
- The loop amplifies the merge. Fewer resident blocks means more misses
  and more recompute, so A saves 35% more blocks (129k vs 95.6k per
  layer). That means bigger calls, which pin more dead bytes.
- Preemptions are the same in both runs (2 to 8 per pass).
- Local hit is the same (0.386 vs 0.391 cold). The whole gap is external
  hit, i.e. what the store kept.
- The delay explanation (a held save landing after the next turn's query)
  does not fit holds of 1.26 steps.

This is a connector cost on top of a server limitation: the server's
region granularity makes save-call size matter, and batching made the
calls bigger.

More precisely, what matters is the entry, not the call. The save handler
(`pegaflow-server` `save`, read for its request shape only) passes each
`SaveLayer` entry through as one `LayerSave`, and the PROGRESS limitation
says one region is pinned per layer entry. `_release_held` folded every
held step's blocks for a layer into one entry. So a merged RPC pinned one
region spanning several steps' requests, and that region stays pinned
until the last of those requests' blocks is evicted. Per step, a region
holds one step's blocks.

## it6 (2026-10-24): batched saves keep one entry per step

Hypothesis: sending held steps in one RPC but keeping each step's own
layer entries gives the server the same regions as unbatched saves. That
should restore resident/C and the G5 hit (A/B arm B), while keeping the
fewer RPCs that the cold-TTFT gain comes from (it3b/it4: the cost scales
with how often the server submits work on the GPU, not with bytes).

Change (worker only, no policy knob): `_release_held` concatenates the
held jobs' `(layer, ids, hashes)` entries in step order instead of folding
them per layer. The wire size is unchanged (the 16 MiB budget already
counted each step's entries separately). The server's GPU copy is one
batch task per call either way. Unit tests: `test_save_batching`,
`test_kv_pressure_sends_held_saves` and
`test_merged_save_rpc_stays_under_size_budget` now assert one entry per
held step. 213 tests pass (connector_v2 core + semantics bindings, on
.44), ruff 0.14.0 check and format are clean. Commit 43a3759, tree
b5cf567.

### Qwen3-4B cap (`2026-10-24-qwen3-4b-p4-it6-cap-b5cf567`, GPU 0, 64gb, run_gate pgid 2935988, 14:08:44 to 14:55:26Z)

Same setup as the A/B: gate ce645d7, base server d1970ead, arms
baseline + v2, cap 64 x 12 auto. Under test b5cf567, verified. The run is
non-official only because it is cap-only. Excess 0 and load errors 0 at
every fraction. Save stats (tp0, 0.5): 4023 RPCs, 3944 of them pressure
sends, failed 0.

| tree | C/U | hit | H*_C | H_lru | E_C | resident/C (4 passes) | insertions (4 passes) | RPCs |
|---|---|---|---|---|---|---|---|---|
| A, merged per layer | 0.25 | 0.367 | 0.863 | 0.578 | 0.025 | 0.71 to 0.73 | 352361 | |
| B, batching off | 0.25 | 0.405 | 0.862 | 0.583 | 0.100 | 0.79 to 0.96 | 321802 | |
| it6 | 0.25 | 0.396 | 0.865 | 0.580 | 0.080 | 0.74 to 0.93 | 323799 | |
| A | 0.5 | 0.794 | 0.930 | 0.857 | 0.596 | 0.68 to 0.72 | 122955 | 4076 |
| B | 0.5 | 0.837 | 0.929 | 0.857 | 0.692 | 0.80 to 0.82 | 93731 | 5042 |
| it6 | 0.5 | 0.843 | 0.930 | 0.857 | 0.688 | 0.77 to 0.83 | 93161 | 4023 |
| r3, 8a05b7c | 0.5 | 0.842 | 0.929 | 0.857 | 0.684 | 0.76 to 0.83 | 93357 | |
| A | 0.75 | 0.867 | 0.970 | 0.915 | 0.677 | 0.65 to 0.72 | 83772 | |
| B | 0.75 | 0.879 | 0.971 | 0.916 | 0.689 | 0.74 to 0.81 | 79562 | |
| it6 | 0.75 | 0.878 | 0.970 | 0.915 | 0.682 | 0.74 to 0.81 | 79286 | |

Per pass at 0.5 (cold / warm / warm2 / warm3):
- it6 hit 0.843 / 0.851 / 0.855 / 0.831, ext share 0.455 / 0.440 / 0.528
  / 0.471.
- B hit 0.842 / 0.857 / 0.854 / 0.820, ext 0.451 / 0.444 / 0.528 / 0.469.

The hypothesis holds. With the same batching, about 4000 RPCs and 98%
pressure sends as A, it6 is level with unbatched B and with 8a05b7c at
every fraction. Resident/C, insertions and the per-pass external hit come
back with it. The G5 regression came from the per-layer folding, not from
holding or merging RPCs. The RPC count is A's, so the submission
frequency that it4's cold gain rests on is unchanged. The DSV2 run below
checks that directly.

### DSV2 replay + cap (`2026-10-24-dsv2-lite-p4-it6-b5cf567`, GPU 0,1, TP2, 64gb, run_gate pgid 2972151, 14:57:45 to 16:16:58Z)

Run setup:
- Arms baseline, native, v2. Workloads replay (32 x 12) and cap.
- Gate ce645d7, base server d1970ead, under test b5cf567 (verified).
- Cap auto chose 74 x 12 this time (MF1: 76), so C(0.5) = 6.66 GB, 2.20x
  GPU KV.
- Non-official only because of the arms and workloads (no plain, no
  gsm8k).

Correctness: the v2 replay digests match 26200 with 0 mismatched, cap
494906 matched (5458 uncertain), 0 mismatched. G3 replay E_ext is 1.0
with excess 0, and there is no eviction. Every replay G4 verdict passes:

| arm | replay cold p50 | vs baseline | cold req/s | warm2 / warm3 p50 | warm req/s |
|---|---|---|---|---|---|
| baseline | 0.909 | | 5.37 | 0.927 / 0.924 | 5.41 / 5.42 |
| native | 0.541 | -40.5% | 8.75 | 0.281 / 0.295 | 13.14 / 13.03 |
| v2 it6 | 0.621 | -31.6% | 8.59 | 0.302 / 0.306 | 12.28 / 12.55 |

| arm (run, plan) | C/U | hit | H*_C | H_lru | E_C | local | resident/C |
|---|---|---|---|---|---|---|---|
| v2 it6 (74) | 0.25 | 0.439 | 0.854 | 0.652 | 0.038 | 0.389 | 0.79 to 0.88 |
| native (74) | 0.25 | 0.557 | 0.857 | 0.642 | 0.257 | 0.411 | |
| old (old-supp, 76) | 0.25 | 0.531 | 0.853 | 0.667 | 0.192 | 0.419 | 0.85 to 0.94 |
| v2 MF1 (76) | 0.25 | 0.428 | 0.853 | 0.673 | 0.007 | 0.405 | 0.59 to 0.72 |
| v2 it6 | 0.5 | 0.844 | 0.925 | 0.848 | 0.643 | 0.393 | 0.77 to 0.86 |
| native | 0.5 | 0.851 | 0.925 | 0.849 | 0.655 | 0.411 | |
| old | 0.5 | 0.845 | 0.925 | 0.847 | 0.609 | 0.425 | 0.85 to 0.94 |
| v2 MF1 | 0.5 | 0.765 | 0.925 | 0.847 | 0.513 | 0.401 | 0.64 to 0.77 |
| v2 it6 | 0.75 | 0.864 | 0.968 | 0.918 | 0.623 | 0.392 | 0.78 to 0.80 |
| native | 0.75 | 0.934 | 0.968 | 0.918 | 0.734 | 0.433 | |
| old | 0.75 | 0.889 | 0.968 | 0.917 | 0.638 | 0.428 | 0.85 to 0.94 |
| v2 MF1 | 0.75 | 0.855 | 0.968 | 0.917 | 0.581 | 0.405 | 0.66 to 0.76 |

Save stats (tp0, cap 0.5): 3651 RPCs, 3578 of them pressure sends, failed
0. Server insertions per warm pass at 0.5: v2 it6 25.8k to 26.3k, old
26.8k to 27.1k, v2 MF1 33.5k to 36.2k.

At C/U 0.5, v2 goes from 0.765 to 0.844, level with old (0.845), H_lru
(0.848) and native (0.851). Resident/C rises from 0.64 to 0.77 to 0.77
to 0.86, and insertions fall to old's level. At 0.25 and 0.75 a gap
remains (see the final judgment).

### DSV2 cold check, unpressured (`2026-10-24-dsv2-lite-p4-it6-cold-b5cf567`, GPU 0,1, 64gb, run_gate pgid 3009442, 16:19:22 to 16:27:24Z)

This uses the same 8 x 6 replay-only setting as it5-rep-01 (max KV usage
about 28%, no pressure sends), where batching's cold gain was measured.
Arms baseline + v2 b5cf567. Non-official (replay only, non-official
replay size).

| arm | cold p50 | vs baseline | cold req/s | warm2 / warm3 p50 | RPCs per pass (tp0) |
|---|---|---|---|---|---|
| baseline | 0.1880 | | 5.701 | 0.164 / 0.164 | |
| v2 it6 | 0.1871 | -0.5% | 5.692 | 0.103 / 0.109 | 14 / 15 / 15 / 9 |
| it5-rep-01 KD x3 (folded entries) | 0.1867 to 0.1916 | -1.9 to +0.6% | 5.666 to 5.679 | | 14 / 14 / 15 |
| it5-rep-01 KOff (no batching) | 0.2000 | +5.0% | 5.581 | | 168 |

All four replay G4 verdicts pass. Digests: 1382 matched, 0 mismatched.
E_ext is 1.0, excess 0, no eviction. The RPC count and the cold result
are the same as the folded-entry default, so it6 keeps the cold-TTFT
gain.

## MF4: official DSV2 run on the it6 tree (`2026-10-24-dsv2-lite-p4-it6-off-a1e7fd2`, GPU 0,1, TP2, 64gb, run_gate pgid 3030292, 16:44:16 to 19:44:22Z)

Setup:
- `"official": true`, `not_official_because: []`.
- Gate ce645d7 (`rsi/` identical to the gate tree of MF1/MF2).
- Base server d1970ead.
- Under test a1e7fd2, which contains it6 43a3759, verified.
- Arms plain, baseline, old, native, v2. Workloads gsm8k (1319),
  replay (32 x 12), cap (76 x 12, the same plan as MF1 and the old
  supplement).
- C(0.5) = 6.87 GB, 2.27x GPU KV.
- No contamination. `split_invariant: false` (control 647 / 1319 differ).
- lmcache is not run: the arm does not give comparable KV on vLLM 0.32
  yet, and another agent is fixing it.

### Verdicts

v2 passes every verdict (`pass: true`):
- G0: spec and provenance.
- G2:
  - gsm8k 10390 ext matched, 0 mismatched.
  - vs baseline: 45008 computed exact.
  - Accuracy delta 0.0 / +0.53 pp.
  - Replay 27232 matched. Cap 512214 matched, 7164 uncertain, 0
    mismatched.
- G3: gsm8k / replay E_ext 1.0, excess 0, no eviction.
- G4: every verdict passes (table below).
- G5: loads and pool pass.

old passes every verdict. native fails G0.spec, as it does by
construction (LBHNC layout). It also fails `G2.digest_self.cap`: 12774 of
its cap blocks are preempted-uncertain (7872 external, 4902 local), with
0 mismatched. That is over the gate's 1% limit. The gate then fails its
G5 verdict too, although loads and pool pass. native is a reference arm,
so this does not affect P4. It passes every other correctness verdict.

### G4 (vs same-run baseline)

| metric | baseline | v2 it6 | vs baseline | old | vs baseline | native | vs baseline | MF1 v2 vs baseline |
|---|---|---|---|---|---|---|---|---|
| replay warm2 / warm3 p50 (s) | 0.923 / 0.926 | 0.299 / 0.310 | -67.6 / -66.6% | 0.344 / 0.334 | -62.7 / -63.9% | 0.279 / 0.290 | -69.8 / -68.7% | -67.4% |
| replay warm req/s | 5.41 | 12.48 / 12.38 | +131 / +129% | 12.35 / 12.37 | +128% | 13.24 / 13.19 | +145% | +128.6% |
| replay cold p50 (s) | 0.909 | 0.599 | -34.1% | 0.630 | -30.7% | 0.557 | -38.7% | -49.8% |
| replay cold req/s | 5.37 | 8.63 | +60.7% | 8.49 | +58.1% | 8.77 | +63.2% | +66.9% |
| gsm8k cold s | 193.9 | 194.2 | +0.15% | 196.0 | +1.08% | 195.2 | +0.67% | +0.28% |
| gsm8k warm s | 193.8 | 189.4 | -2.26% | 192.0 | -0.94% | 190.9 | -1.50% | -2.27% |

- MF1's replay cold p50 vs baseline (-49.8%) came from a slower baseline
  (1.323 s; this run's is 0.909 s).
- In absolute terms, v2's cold p50 is 0.599 s here vs 0.665 s in MF1, and
  cold req/s 8.63 vs 8.40.
- v2 is ahead of old on every replay metric and within 7 to 8% of native
  on warm.
- gsm8k is at baseline (+0.15% cold).

### Hits (unlimited store)

| pass | v2 local / ext / total | old | native | lmcache | H* |
|---|---|---|---|---|---|
| gsm8k warm | 0.938 / 0.054 / 0.993, excess 0 | 0.938 / 0.055 / 0.993, excess 1230 | 0.940 / 0.053 / 0.993, excess 1230 | pending (arm not comparable yet) | 0.993 |
| replay cold | 0.497 / 0.345 / 0.843 | 0.500 / 0.343 / 0.843 | 0.511 / 0.332 / 0.842 | pending | 0.843 |
| replay warm2 | 0.487 / 0.511 / 0.998, excess 0 | 0.508 / 0.490 / 0.998, excess 210 | 0.467 / 0.532 / 0.998, excess 210 | pending | 0.998 |

### G5 (cap 76 x 12, warm_timed)

| C/U | v2 it6 | old | native | lmcache | H_lru | H*_C | E_C v2 / old / native | resident/C v2 / old |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.436 | 0.512 | 0.578 | pending (arm not comparable yet) | 0.673 | 0.851 | 0.040 / 0.161 / 0.271 | 0.78 to 0.90 / 0.90 to 0.95 |
| 0.5 | 0.844 | 0.845 | 0.850 | pending | 0.847 | 0.925 | 0.615 / 0.621 / 0.624 | 0.78 to 0.83 / 0.92 to 0.95 |
| 0.75 | 0.863 | 0.889 | 0.933 | pending | 0.919 | 0.968 | 0.581 / 0.632 / 0.712 | 0.79 to 0.82 / 0.94 |

Other numbers from this run:
- MF1 v2 for comparison: 0.428 / 0.765 / 0.855.
- Local hit: v2 0.401 / 0.403 / 0.414, old 0.421 / 0.417 / 0.431.
- Server insertions (4 passes): at 0.5, v2 108075 vs old 108123. At 0.25,
  341261 vs 268400; at 0.75, 97701 vs 87080.
- Save stats (tp0): cap 0.5 has 3828 RPCs (3749 pressure sends); replay
  1335 (1286); gsm8k 358 (0). Failed 0.

These match the non-official it6 run (0.439 / 0.844 / 0.864), so the it6
result reproduces on an official run.

## MF5: official Qwen3-4B regression run on the it6 tree (`2026-10-24-qwen3-4b-p4-it6-off-a1e7fd2`, GPU 0, TP1, 64gb, run_gate pgid 3092773, 19:44:40 to 22:48:40Z)

Setup:
- `"official": true`, `not_official_because: []`.
- Gate ce645d7, base server d1970ead, under test a1e7fd2 (the same tree
  as MF4).
- Arms plain, baseline, old, native, v2. Workloads gsm8k (1319),
  replay (32 x 12), cap (64 x 12).
- C(0.5) = 27.2 GB, 2.44x GPU KV. No contamination.
- lmcache is not run (see MF4).

### Verdicts: v2 fails one, `G2.digest_self.cap`

v2 passes everything else:
- G0.
- G2:
  - gsm8k 7423 ext matched, vs baseline 22129 / 22129 exact, accuracy
    delta 0.0 / 0.0 pp.
  - Replay 18590 matched.
- G3: E_ext 1.0, excess 0, no eviction.
- G4: every verdict passes (table below).

The failing verdict:
- `G2.digest_self.cap` fails, and `G5.replay_warm` fails with it (the
  gate ANDs them; loads and pool pass).
- 0 mismatched: 259969 ext and 214373 local matched.
- 4879 blocks are uncertain, all of them from preempted episodes. That is
  1.008% of 484099 loaded blocks, against `MAX_PREEMPTED_UNCERTAIN = 0.01`.
  The limit allows 4840, so the run is 39 blocks over.

The same share in other runs:

| run | arm | preempted-uncertain | share | mismatched |
|---|---|---|---|---|
| MF2 (d56fcb2) | v2 | 2268 | 0.51% | 0 |
| MF2 | old | 1031 | 0.22% | 0 |
| A/B A (d56fcb2) | v2 | 2549 | 0.57% | 0 |
| A/B B (706006c) | v2 | 2557 | 0.53% | 0 |
| it6 cap (b5cf567) | v2 | 3189 | 0.66% | 0 |
| MF5 (a1e7fd2) | v2 | 4879 | 1.01% | 0 |
| MF5 | old | 4205 | 0.88% | 0 |
| MF5 | native | 2353 | 0.48% | 0 |

vLLM preemptions per cap pass are the same across these runs (2 to 9 for
every arm).

Cross-arm history (final review S1; shares as the gate computes them,
`preempted_uncertain / (compared + uncertain)`):

| run | v2 | old | native |
|---|---|---|---|
| Qwen3-4B r3 (8a05b7c) | 0.41% | 1.10% (fail) | 0.87% |
| Qwen3-4B MF2 | 0.51% | 0.22% | |
| Qwen3-4B A / B / it6 cap | 0.57% / 0.53% / 0.66% | | |
| Qwen3-4B MF5 | 1.01% (fail) | 0.88% | 0.48% |
| DSV2 MF1 / it6 / MF4 | 0.67% / 0.64% / 0.93% | MF4 0.50%, old-supp 0.71% | MF4 1.15% (fail) |

- old failed this same verdict on Qwen3-4B r3 (5268 blocks, 1.10%, 0
  mismatched), and native failed it on MF4 DSV2 (12774, 1.15%). Every
  arm that loads from a store has crossed 1% at least once.
- v2's own range is 0.41% (r3) to 1.01% (MF5). On DSV2 MF4 it passes at
  0.93%, so the DSV2 margin is thin too.
- Byte-level check (reviewer, gate's `digest_compare` at ce645d7 on the
  MF5 cap(0.5) probe dirs): 4878 of the 4879 blocks have a digested copy
  of the same key. Their closest per-layer rel-L1 to it is p50 0.00095,
  p99 0.0071, 0.123% above 0.0175. Honest computed-vs-computed copies give
  p50 0.00081, p99 0.0071, 0.107% above 0.0175. The same blocks against
  the adjacent key's copy (what a wrong-block load would look like) give
  p50 0.061 and a minimum of 0.0175. So the stored bytes are the right
  blocks, at ordinary recompute noise. old (4205, p99 0.0098) and native
  (2353, p99 0.0052) look the same.

What this does and does not show:
- In the same run, old's share also went from 0.22% (MF2) to 0.88%, so
  part of the rise is run to run.
- v2's two it6 samples (0.66%, 1.01%) are above its three pre-it6 samples
  (0.51 to 0.57%). One plausible reading is that a store that keeps more
  also loads more blocks first computed in preempted episodes. That is
  not verified, and two samples cannot separate it from noise.
- No block was judged wrong. The verdict fails on the share of blocks the
  digest cannot judge, not on a wrong block.

I did not edit the gate (it is gate-owned), and I did not rerun to get a
pass. old fails G4.gsm8k_warm (+2.01%) and native fails it too (+5.22%)
in this run; v2 passes both gsm8k verdicts.

### MF5 note: where the 4879 preempted-uncertain blocks come from (run data only, no GPU)

Verdict: the rise is variance, not a connector behaviour change. No fix,
no rerun yet; the reviewer decides whether a recorded rerun is warranted.

The gate's definition (`rsi/gates/digest_compare.py`):
- The digest records only a request's *last* schedule episode (module
  docstring, `records`). A computation in an earlier, preempted episode
  of a digested request is counted in `Reference.preempted`
  (lines 133 to 148: `elif k + 1 < len(ranges)`).
- A loaded (local or external) block is `uncertain` when the reference
  has no digested copy that equals it and the reference is not
  exhaustive (lines 236 to 241, 257 to 258).
- It is `uncertain_preempted` when the key has no untagged computation
  and digested copies plus preempted episodes account for every
  computation of the key (`_uncertain`, lines 181 to 185). So the only
  undigested bytes of that key were computed in a preempted episode.
- `run_gate._digest_verdict` (lines 1559 to 1567) allows at most
  `MAX_PREEMPTED_UNCERTAIN = 0.01` of the loaded blocks.

So yes: these are blocks whose stored copy can only have come from a
preempted episode's KV. That KV is valid. It was computed before the
preemption, and the scheduler's preemption flush (`plan_saves`:
`preempted` sets `flush`) completes the save before vLLM reuses the
blocks. The gate just has no digest of those bytes to judge them
against. old produces the same category on the same trace.

The attribution below reuses the gate's own `digest_compare` (`records`,
`Reference`) on each run's cap(0.5) probe dir, so the counts are exactly
the gate's. Script: `~/kexi/tmp/dsv2-p4/p4_uncertain.py` on .44.

Totals by C/U in MF5 (v2):

| C/U | local uncertain_preempted | external uncertain_preempted |
|---|---|---|
| 0.25 | 780 | 163 |
| 0.5 | 1519 | 3360 |
| 0.75 | 951 | 4832 |

Only the 0.5 run is gated (exhaustive digest). Its 4879 blocks are
1.008% of the loaded blocks.

Per pass at 0.5. Each cell is vLLM preemptions / preempted-episode block
computations (exposure) / uncertain_preempted loads:

| run (tree) | cap_cold | cap_warm | cap_warm2 | cap_warm3 | total | share |
|---|---|---|---|---|---|---|
| MF2 v2 (d56fcb2) | 5 / 250 / 62 | 6 / 128 / 879 | 7 / 709 / 799 | 7 / 1094 / 528 | 2268 | 0.51% |
| A/B A v2 (d56fcb2) | 6 / 410 / 459 | 4 / 575 / 1020 | 4 / 281 / 441 | 5 / 476 / 629 | 2549 | 0.57% |
| A/B B v2 (706006c) | 4 / 66 / 266 | 6 / 204 / 1139 | 5 / 153 / 998 | 4 / 262 / 154 | 2557 | 0.53% |
| it6 cap v2 (b5cf567) | 7 / 182 / 166 | 6 / 119 / 924 | 6 / 338 / 1894 | 4 / 66 / 205 | 3189 | 0.66% |
| MF5 v2 (a1e7fd2) | 5 / 46 / 103 | 7 / 399 / 1925 | 4 / 187 / 1623 | 6 / 99 / 1228 | 4879 | 1.01% |
| MF2 old | 5 / 81 / 280 | 3 / 154 / 287 | 2 / 17 / 106 | 3 / 63 / 358 | 1031 | 0.22% |
| MF5 old | 9 / 259 / 788 | 4 / 90 / 321 | 4 / 246 / 1232 | 5 / 906 / 1864 | 4205 | 0.88% |

Findings:
1. Preemptions do not explain the count. Preemptions per pass are 2 to 9
   in every run. Exposure is lower in MF5 (731 block computations) than in
   MF2 (2181), yet MF5 has twice the uncertain loads. Within a run, the
   pass with the most preemptions is not the one with the most uncertain
   loads (MF2 warm2: 7 preemptions, 799; MF5 warm: 7, 1925).
2. Which request is preempted decides the count. An uncertain load is a
   block computed only in a preempted episode, times the number of later
   requests that load it. A preempted early turn of a long conversation
   is reloaded by every later turn of that conversation:
   - MF5: conv 60 turn 3 is preempted in cap_warm, and turns 3 to 6
     each load 253 of its blocks. Conv 13 alone is 2178 of the 4879
     (45%).
   - The top conversation's share in the other v2 runs is 32% (MF2), 38%
     (A), 38% (B) and 34% (it6 cap). In MF5 old it is 40% (conv 27).
3. The same code moves 4x between runs. old has no code change between
   MF2 and MF5 and goes from 0.22% to 0.88%. v2 before it6 ran three
   times at 0.51 to 0.57%. it6 ran twice, at 0.66% and 1.01%. Both are
   inside the spread old shows by itself.
4. it6 does not change what is saved. Between d56fcb2 (MF2) and a1e7fd2
   (MF5), `connector_v2` differs only in `Worker._release_held`
   (`scheduler.py` diff: 0 lines). The scheduler alone picks which
   (block, hash) pairs are saved, including after a preemption. it6 only
   concatenates the held steps' entries instead of folding them, and
   the unit tests check that every id and hash is still sent once, in
   order. So the coordinator's concern, saving blocks across a
   preemption that were not saved before, is ruled out by construction.
   The "own request, earlier episode" category, a request reloading what
   its preempted episode saved, is the same size: 672 (MF2) vs 729
   (MF5).
5. The one category it6 can plausibly enlarge is retention across a pass.
   Blocks computed only in a preempted episode can stay in the store long
   enough to be loaded a pass later. That happened in MF5 (1386 loads, of
   which 726 are conv 13 turn 1, preempted in cap_warm and loaded in
   cap_warm3) and in the it6 cap run (261, one request). It did not
   happen in MF2, A or B, although B (batching off) has it6's resident/C.
   This follows from a store that keeps more. It is not a save-policy
   change, and the blocks are valid KV.

Conclusion: variance in which turns get preempted. v2's save behaviour
is the same as before it6. Not saving across a preemption would only
shrink the gate's unjudged share: it would drop valid hits, and old saves
these blocks too. The final review agreed (no connector fault) but cannot
waive a hard verdict, so M1 is one recorded official rerun, kept next to
MF5 (section "M1" below). The gate-side remedy, digesting preempted
episodes in `handle_preemptions` (`feat/rsi-gate-hybrid`), is for the gate
owner.

### G4 (vs same-run baseline)

| metric | baseline | v2 it6 | vs baseline | old | vs baseline | native | vs baseline | MF2 v2 | it2d v2 |
|---|---|---|---|---|---|---|---|---|---|
| replay warm2 / warm3 p50 (s) | 3.563 / 3.504 | 0.555 / 0.574 | -84.4 / -83.6% | 0.644 / 0.661 | -81.9 / -81.1% | 0.455 / 0.504 | -87.2 / -85.6% | -85.3% | -78.4% |
| replay warm req/s | 2.55 / 2.56 | 9.63 / 9.58 | +277 / +274% | 9.99 / 10.18 | +291 / +297% | 9.63 / 9.60 | +277 / +274% | +293% | +193% |
| replay cold p50 (s) | 3.721 | 1.377 | -63.0% | 1.384 | -62.8% | 1.420 | -61.8% | -65.3% | -54.1% |
| replay cold req/s | 2.54 | 6.26 | +146.6% | 5.56 | +119.1% | 5.93 | +133.7% | +152% | +105% |
| gsm8k cold s | 170.2 | 170.8 | +0.37% | 172.1 | +1.12% | 172.2 | +1.20% | +0.80% | +0.92% |
| gsm8k warm s | 171.5 | 170.2 | -0.76% | 174.9 | +2.01% (fail) | 180.4 | +5.22% (fail) | +0.07% | +0.01% |

All G4 verdicts pass for v2, and there is no regression against MF2 or
it2d. v2 has the best cold req/s of the three arms.

### Hits (unlimited store)

| pass | v2 local / ext / total | old | native | lmcache | H* |
|---|---|---|---|---|---|
| gsm8k warm | 0.912 / 0.081 / 0.992, excess 0 | 0.927 / 0.067 / 0.993, excess 1170 | 0.933 / 0.060 / 0.993, excess 1170 | pending (arm not comparable yet) | 0.992 |
| replay cold | 0.379 / 0.486 / 0.864 | 0.382 / 0.482 / 0.864 | 0.384 / 0.480 / 0.864 | pending | 0.864 |
| replay warm2 | 0.347 / 0.651 / 0.998, excess 0 | 0.345 / 0.653 / 0.998, excess 195 | 0.378 / 0.621 / 0.998, excess 195 | pending | 0.998 |

### G5 (cap 64 x 12, warm_timed)

| C/U | v2 it6 | old | native | lmcache | H_lru | H*_C | E_C v2 / old / native | resident/C v2 / old | insertions v2 / old |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.392 | 0.398 | 0.542 | pending (arm not comparable yet) | 0.581 | 0.862 | 0.065 / 0.081 / 0.308 | 0.74 to 0.85 / 0.81 to 0.93 | 322810 / 308827 |
| 0.5 | 0.838 | 0.820 | 0.861 | pending | 0.857 | 0.929 | 0.667 / 0.645 / 0.718 | 0.75 to 0.82 / 0.76 to 0.84 | 93389 / 98334 |
| 0.75 | 0.883 | 0.873 | 0.927 | pending | 0.915 | 0.970 | 0.698 / 0.674 / 0.767 | 0.71 to 0.83 / 0.73 to 0.79 | 78308 / 81468 |

MF2 v2 for comparison: 0.368 / 0.777 / 0.865. The it6 recovery holds on
an official run: 0.838 here, 0.843 in the it6 cap run, 0.842 at r3.

## G5 hit-rate judgment (final, official runs MF4 / MF5, base server d1970ead)

The references are same-run native, old and H_lru, with H*_C as the
ceiling. lmcache is pending: the arm is not comparable on vLLM 0.32 yet.

The tables before MF4 come from runs that had different arm sets:
- MF1 had no valid old (plugin bug).
- MF2 had no native.
- The A/B and the it6 non-official runs were v2-only or v2 + native by
  design.

Those tables are kept as recorded. MF4, MF5 and this section are the
side-by-side comparison of v2, old, native and lmcache.

Qwen3-4B (MF5, 64 x 12):

| C/U | v2 it6 | old | native | lmcache | H_lru | H*_C | E_C v2 / old / native | resident/C v2 / old |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.392 | 0.398 | 0.542 | pending (arm not comparable yet) | 0.581 | 0.862 | 0.065 / 0.081 / 0.308 | 0.74 to 0.85 / 0.81 to 0.93 |
| 0.5 | 0.838 | 0.820 | 0.861 | pending | 0.857 | 0.929 | 0.667 / 0.645 / 0.718 | 0.75 to 0.82 / 0.76 to 0.84 |
| 0.75 | 0.883 | 0.873 | 0.927 | pending | 0.915 | 0.970 | 0.698 / 0.674 / 0.767 | 0.71 to 0.83 / 0.73 to 0.79 |

Qwen3-4B judgment: close enough.
- v2 is level with or above old at every fraction, 0.019 below H_lru
  and 0.023 below native at 0.5. That is level with its 8a05b7c r3
  (0.842) and the it6 cap run (0.843). The MF2 regression is gone.
- Gap to native at 0.25 / 0.75 (0.150 / 0.044): native's store is not
  the PegaFlow server, and it beats H_lru at 0.75 (0.927 vs 0.915).
  old, on the same server as v2, has the same gap.
- The gap to H_lru is shared by both PegaFlow connectors: they keep
  0.71 to 0.93 C resident against the LRU model's full C. I attribute it
  to the base server (PROGRESS "Server limitations": the 512-block
  reclaim batch, H1, and region pinning, H3, both largest on small
  pools). It is not measured further here.

DSV2 (MF4, 76 x 12; the same plan as MF1 and the old supplement):

| C/U | v2 it6 | old | native | lmcache | H_lru | H*_C | E_C v2 / old / native | resident/C v2 / old |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.436 | 0.512 | 0.578 | pending (arm not comparable yet) | 0.673 | 0.851 | 0.040 / 0.161 / 0.271 | 0.78 to 0.90 / 0.90 to 0.95 |
| 0.5 | 0.844 | 0.845 | 0.850 | pending | 0.847 | 0.925 | 0.615 / 0.621 / 0.624 | 0.78 to 0.83 / 0.92 to 0.95 |
| 0.75 | 0.863 | 0.889 | 0.933 | pending | 0.919 | 0.968 | 0.581 / 0.632 / 0.712 | 0.79 to 0.82 / 0.94 |

DSV2 judgment: close enough at the gated point, 0.5. v2 is level with old,
native and H_lru (0.844 vs 0.845 / 0.850 / 0.847). Every external hit is
verified, and excess is 0.

At 0.25 and 0.75 a gap remains:
- Against old on the same server, v2 trails by 0.076 at 0.25 and 0.026 at
  0.75. This is the connector part. Against native the gap is 0.142 and
  0.070, but native's store is a different store.
- Its signature: v2 keeps 0.05 to 0.15 C less resident than old at the
  same pool use (used/C 0.94 to 0.99). At 0.25, v2 also inserts 27% more
  blocks (341k vs 268k).
  - About 0.02 of the gap is local hit (v2 0.401 to 0.414 vs old 0.417 to
    0.431), which is vLLM-side reuse, not the store.
  - The rest is external: E_C at 0.25 is 0.040 vs 0.161.
- The per-request entry lever was tested after the final review (M2,
  section "M2" below). It restores resident/C to old's level and lifts
  0.75, but leaves the 0.25 gap. The 0.25 attribution is in that section,
  and it replaces this one: the resident deficit does not explain 0.25.

Context for the 0.25 point (final review S2):
- C(0.25) is 3.43 GB, 1.14x GPU KV for DSV2 (Qwen3-4B 13.6 GB, 1.22x).
  That is below the gate's 2x floor, which is what makes C(0.5) a
  measurement of the external store. At 0.25 the store holds little more
  than vLLM's own cache.
- The server dominates that point: old itself is 0.16 below H_lru (0.512
  vs 0.673) and native 0.10 below.
- Local hit: v2's own local hit is 0.015 to 0.02 below old and native in
  every cap run (MF4 0.401 to 0.414 vs old 0.417 to 0.431; also r3 and
  batching-off B on Qwen3-4B). It is constant and predates P4, so it is not
  counted as part of the 0.25 gap.

Server vs connector for MF1's DSV2 gap at 0.5 (v2 0.765 vs native 0.852):
- Connector, from the batching fold: +0.079. MF4's v2 is 0.844, with it6
  as the only connector change since MF1 (07f5d96 -> a1e7fd2: connector_v2
  differs only in `_release_held`). Resident/C goes from 0.64-0.77 to
  0.78-0.83.
- Remaining: 0.006 to native and 0.003 below H_lru, both within run-to-run
  noise. So at 0.5 nothing is left to attribute to the server once v2
  sends per-step entries. Point 2 of the MF1 judgment (all of it is server
  retention) was wrong.

## M2 (final review): per-request save entries, and the 0.25 gap

Hypothesis: a step's save entry mixes every request that saved in that
step. The base server pins one region per entry until its last block is
evicted, so one entry per (storage group, request) should strand fewer
bytes. That should raise resident/C to old's level and close the 0.25 gap.

Change (scheduler only, no knob): `Scheduler.plan_saves` builds one
`GroupSave` per (sid, request) instead of one per sid. Groups stay in sid
order and requests in scheduled order. The worker already sends several
entries per layer (it6). New unit test
`test_saves_keep_one_entry_per_request` (fails on the old code). Commit
a59a254 on `exp/p4-per-request-save`, tree `~/kexi/p4-ut-a59a254`.

### DSV2 cap-only A/B (`2026-10-25-dsv2-lite-p4-m2-a59a254`, GPU 0,1, TP2, 64gb, run_gate pgid 3230141, 00:16:04 to 01:45:54Z)

Setup:
- Gate ce645d7, base server d1970ead, arms baseline, old, native, v2.
- Workload cap only, `--cap-convs 76` (MF4's auto plan). Non-official by
  design; the gate lists `cap_convs=76 (official auto)`, no plain, and
  cap-only. C(0.5) = 6.87 GB = 2.27x GPU KV, the same as MF4.
- Excess 0 and unmatched 0 for v2. `G2.digest_self.cap`: 0 mismatched for
  every arm (v2 9146 preempted-uncertain).

The control is MF4 (same gate, server, plan and GPUs, a1e7fd2 = lever
minus one commit). old and native are rerun in the same run.

| C/U | v2 lever | v2 MF4 | old (this run) | old MF4 | native (this run) | lmcache | H_lru | H*_C | E_C v2 lever / old / native |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.449 | 0.436 | 0.550 | 0.512 | 0.576 | pending (arm not comparable yet) | 0.654 | 0.852 | 0.057 / 0.186 / 0.272 |
| 0.5 | 0.848 | 0.844 | 0.845 | 0.845 | 0.851 | pending | 0.847 | 0.925 | 0.620 / 0.616 / 0.628 |
| 0.75 | 0.899 | 0.863 | 0.888 | 0.889 | 0.932 | pending | 0.917 | 0.968 | 0.662 / 0.633 / 0.695 |

Store churn per pass (cold / warm / warm2 / warm3). Server insertions and
evictions are pass deltas. resident/C is at pass end. ext is
`max(0, first_computed - local)` over prompt tokens:

| C/U | arm | ext | insertions | evictions | resident/C |
|---|---|---|---|---|---|
| 0.25 | v2 MF4 | 0.038 / 0.020 / 0.019 / 0.013 | 74478 / 83805 / 88071 / 94907 | 69120 / 82944 / 88576 / 94720 | 0.78 / 0.90 / 0.83 / 0.86 |
| 0.25 | old MF4 | 0.047 / 0.099 / 0.067 / 0.059 | 66265 / 61096 / 62646 / 78393 | 59904 / 60928 / 62976 / 78336 | 0.92 / 0.95 / 0.90 / 0.91 |
| 0.25 | v2 lever | 0.059 / 0.106 / 0.022 / 0.023 | 70870 / 66985 / 86608 / 92113 | 64512 / 67072 / 86528 / 92160 | 0.92 / 0.91 / 0.92 / 0.91 |
| 0.25 | old (M2 run) | 0.042 / 0.115 / 0.066 / 0.081 | 66974 / 58413 / 61407 / 57623 | 60928 / 58368 / 61440 / 57344 | 0.88 / 0.88 / 0.88 / 0.92 |
| 0.5 | v2 MF4 | 0.337 / 0.370 / 0.289 / 0.286 | 27566 / 26873 / 26691 / 26945 | | 0.81 / 0.79 / 0.83 / 0.78 |
| 0.5 | v2 lever | 0.328 / 0.369 / 0.291 / 0.287 | 27565 / 26192 / 26327 / 26152 | | 0.92 / 0.93 / 0.94 / 0.91 |
| 0.5 | old (M2 run) | 0.328 / 0.364 / 0.293 / 0.284 | 27565 / 27131 / 26589 / 26777 | | 0.92 / 0.92 / 0.92 / 0.89 |
| 0.75 | v2 MF4 | 0.328 / 0.399 / 0.300 / 0.293 | 27564 / 23096 / 23589 / 23452 | | 0.79 / 0.82 / 0.82 / 0.81 |
| 0.75 | v2 lever | 0.330 / 0.422 / 0.343 / 0.333 | 27565 / 18328 / 18225 / 17475 | | 0.94 / 0.96 / 0.95 / 0.95 |
| 0.75 | old (M2 run) | 0.330 / 0.390 / 0.330 / 0.315 | 27564 / 20566 / 19887 / 19122 | | 0.94 / 0.94 / 0.94 / 0.95 |

Miss classes at 0.25, MF4, per warm request against the gate's own LRU
model (`oracle.simulate("lru")` with the baseline's local hit). A
request is "zero" if LRU serves it and the arm serves nothing; "lost" is
the LRU tokens not served, over prompt tokens:

| pass | v2 zero (n / lost) | v2 short | old zero (n / lost) | old short |
|---|---|---|---|---|
| cap_warm | 183 / 0.146 | 13 / 0.005 | 57 / 0.051 | 24 / 0.016 |
| cap_warm2 | 214 / 0.177 | 8 / 0.006 | 151 / 0.124 | 28 / 0.024 |
| cap_warm3 | 214 / 0.222 | 2 / 0.002 | 164 / 0.165 | 35 / 0.038 |

At 0.5 the same classes are level (warm3 zero: v2 100 / 0.096, old 103 /
0.098). The 0.25 loss is whole requests the store serves nothing to,
not short hits.

Result:
- Resident/C: fixed. The lever brings v2 to 0.91 to 0.96 at every
  fraction, at or above old in the same run (0.88 to 0.95). Insertions at
  0.75 fall from 23k to 18k per warm pass, below old's 20k.
- 0.75: +0.036 (0.863 -> 0.899), now 0.011 above old.
- 0.5: unchanged (0.844 -> 0.848; old 0.845, H_lru 0.847).
- 0.25: +0.013 (0.436 -> 0.449), still 0.101 below old in the same run.
  The reviewer's thrash signature is still there: insertions rise from
  67k to 92k per warm pass while ext falls from 0.106 to 0.023. old's
  stay flat (58k to 61k).
- Decision (coordinator, 2026-10-25): keep the lever as it7, provided the
  Qwen3-4B cap regression and the DSV2 8 x 6 cold G4 check pass (below).

### Qwen3-4B cap regression (`2026-10-25-qwen3-4b-p4-m2-a59a254`, GPU 0, TP1, 64gb, run_gate pgid 3280407, 01:55:12 to 03:14:35Z)

Gate ce645d7, base server, cap 64 x 12 auto, arms baseline, old, v2.
Non-official (cap only, no plain). Excess 0 and unmatched 0 for v2.
`G2.digest_self.cap` passes: 2084 preempted-uncertain (0.43%), 0
mismatched (old 0.71%).

| C/U | v2 lever | v2 MF5 | v2 it6 cap | old (this run) | lmcache | H_lru | H*_C | E_C v2 lever / old | resident/C v2 / old |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.404 | 0.392 | 0.396 | 0.400 | pending (arm not comparable yet) | 0.584 | 0.863 | 0.099 / 0.080 | |
| 0.5 | 0.859 | 0.838 | 0.843 | 0.836 | pending | 0.857 | 0.929 | 0.711 / 0.675 | 0.93 to 0.96 / 0.76 to 0.83 |
| 0.75 | 0.908 | 0.883 | 0.878 | 0.876 | pending | 0.915 | 0.971 | 0.724 / 0.692 | |

No regression. Qwen3-4B gains at every fraction (+0.016 to +0.030 over
it6). At 0.5, v2 reaches H_lru (0.859 vs 0.857) with resident/C 0.93 to
0.96.

### DSV2 8 x 6 cold check (`2026-10-25-dsv2-lite-p4-m2-cold-a59a254`, pgid 3315766, 03:17:28 to 03:26:08Z; control `2026-10-25-dsv2-lite-p4-m2-coldctl-a1e7fd2`, pgid 3321088, 03:27:29 to 03:35:42Z)

Same arguments as the it6 cold check (`--workloads replay --replay-convs
8 --replay-turns 6`, GPU 0,1, arms baseline + v2). The control runs the
accepted tree a1e7fd2 directly after, on the same GPUs.

| run | tree | baseline cold p50 | v2 cold p50 | vs baseline | G4 cold p50 | v2 cold req/s | vs baseline | warm2 / warm3 p50 |
|---|---|---|---|---|---|---|---|---|
| it6 cold check | b5cf567 | 0.1880 | 0.1871 | -0.5% | pass | 5.692 | -0.2% | 0.103 / 0.109 |
| lever | a59a254 | 0.1712 | 0.1897 | +10.8% | fail | 5.698 | +0.2% | 0.109 / 0.113 |
| control | a1e7fd2 | 0.1923 | 0.1900 | -1.2% | pass | 5.689 | -0.1% | 0.098 / 0.108 |

v2's cold p50 is the same on the lever and the control (0.1897 vs
0.1900), within the it6 check's 0.1871. The baseline swings 0.171 to
0.192 across three back-to-back runs of identical code. The lever run's
G4 cold p50 fail comes from its fast baseline, against a 2% slack on a
48-request slice. Every other replay verdict passes in both runs (warm
p50 and req/s, cold req/s, digests 0 mismatched, E_ext 1.0).

### 0.25: what the gap is not, and what is left

old against itself at 0.25 (warm_timed hit) is 0.512 (MF4), 0.531 (old
supplement) and 0.550 (this run), a spread of 0.038. v2 is 0.428 (MF1),
0.439 (it6), 0.436 (MF4) and 0.449 (lever). So the v2-old gap is 0.06 to
0.12 across runs, of which up to 0.04 is old's own run noise.

Candidates, each tested:
1. Resident deficit: ruled out. With the lever, v2's resident/C (0.91 to
   0.92) is at or above old's (0.88 to 0.92) in the same run, and 0.25
   moves only +0.013. That matches the reviewer's cross-check: at 0.5 the
   deficit cost nothing.
2. Block-0 query refresh: ruled out on this plan. v2 queries from block
   0, so every query refreshes vLLM-local heads on the server; old queries
   from its local hit. Test: the server-faithful sim from
   feat/rsi-server-reclaim (`~/kexi/rsi-tools/server-reclaim/sim/r3sim.py`,
   sha256 3f50d935...). It is copied to `~/kexi/tmp/dsv2-p4/r3sim_p4.py`
   with two changes: a `query="local"` mode (scan from the request's own
   local hit, refresh only what is scanned) and the newer `.gate/`
   result layout. The driver is `p4sim.py`, which uses the gate's
   block_bytes (C = 6897 DSV2 blocks). It replays each arm's real 0.25
   trace with the base server's 512-block reclaim rounds; only blocks the
   connector computed are saved. The faithful mode reproduces the logged
   per-query hit count for 74 to 92% of timed requests (v2 79 to 92%,
   old 74 to 85%). warm_timed hit:

   | trace | C | measured | faithful (logged window) | query from block 0 | query from local hit |
   |---|---|---|---|---|---|
   | v2 MF4 | 1.0 C | 0.436 | 0.541 | 0.541 | 0.529 |
   | v2 MF4 | 0.9 C | | 0.448 | 0.448 | 0.431 |
   | old MF4 | 1.0 C | 0.512 | 0.558 | 0.557 | 0.570 |
   | old MF4 | 0.9 C | | 0.518 | 0.511 | 0.528 |
   | v2 lever | 1.0 C | 0.449 | 0.535 | 0.535 | 0.517 |
   | v2 lever | 0.9 C | | 0.472 | 0.472 | 0.446 |
   | old M2 run | 1.0 C | 0.550 | 0.599 | 0.619 | 0.608 |
   | old M2 run | 0.9 C | | 0.546 | 0.556 | 0.554 |

   On v2's trace, querying from the local hit costs 0.012 to 0.026; it
   does not gain anything. On old's trace the query mode moves the hit by
   at most +-0.02 either way. The earlier server sim on the old 32-conv
   Qwen3-4B plan found the same sign for v2 (a local-prefix touch helped
   old +0.06 and cost v2 -0.04). So "query from the local hit" is not a
   lever for v2, and it is not proposed.
3. Lost saves: ruled out. v2's server insertions are 0.95 to 0.99 of the
   complete blocks its requests computed past their hit (old: 0.77 to
   0.90).
4. Server behaviour given the trace: the sim matches both arms at about
   0.9 C (v2 0.448 vs 0.436 measured, old 0.518 vs 0.512 in MF4; M2 run
   0.472 vs 0.449 and 0.546 vs 0.550). Given each arm's own trace, the
   server behaves like an LRU at about its measured resident/C for both
   arms.
5. Local hit: v2's own local hit is 0.019 below old's (0.403 vs 0.422 in
   the M2 run). With the baseline's local hit counted present in the sim
   (gate semantics), the trace gap is still 0.04 to 0.05 at 0.9 C (v2
   0.579 / 0.584 vs old 0.618 / 0.632).

What is left: the same server LRU at the same capacity yields 0.05 to 0.07
less on v2's trace than on old's (0.9 C: 0.448 vs 0.518, 0.472 vs 0.546).
So the difference is in the access trace itself: which requests run when,
and what each computes and saves, not in the store's treatment of
it. At 0.25 the loop is closed and steep. In the gate's LRU model, warm hit
falls from 0.673 to 0.579 between 1.0 C and 0.93 C at 0.25, against 0.847
to 0.845 at 0.5. One fewer hit means more recompute, more saves and more
evictions, which is the thrash the reviewer saw. The observed signature:
v2's warm passes get their external hits almost only from the first 1/8
of each pass (32 to 35 of 65 requests, ext 0.27 there) and close to none
after, while old's hits are spread over the middle and end of the pass.
I could not tie that to a connector mechanism from the traces, so the
0.25 cause is recorded as unknown. 0.25 is below the 2x GPU-KV floor, hit
rate is not gated, and the user's guidance is not to chase hit rate, so
it is not pursued further.

Scripts (all read-only over run dirs, on .44 under `~/kexi/tmp/dsv2-p4/`):
`churn.py` (per-pass server metrics), `miss_classes.py` (gate LRU miss
classes), `lru_variants.py` / `head_sim.py` (gate-model variants),
`computed.py` (computed blocks vs insertions), `timeline.py` (ext hits
over the pass), `r3sim_p4.py` + `p4sim.py` (server-faithful sim, sha256
2ae28f28... / 8023ee17...).

## Qwen3-4B TP1 non-regression for it5 (`2026-10-24-qwen3-4b-p4-it5-nonreg`, GPU 0, 64gb, run_gate pgid 2194273, 01:16:19 to 01:56:30Z, rev it5-kv090)

Arms: plain, baseline, KOff, v2 with the shipped default (32 steps /
1000 ms / KV 0.9). gsm8k is the full 1319.

| arm | pass | gsm8k acc | gsm8k cold / warm s | replay cold p50 | vs baseline | cold req/s | warm / warm2 / warm3 p50 | save RPCs gsm8k / replay | pressure RPCs |
|---|---|---|---|---|---|---|---|---|---|
| plain | | 0.8582 | 148.9 / 150.0 | | | | | | |
| baseline | pass | 0.8582 | 169.8 / 171.2 | 0.2196 | | 5.012 | 0.2253 / 0.2190 / 0.2198 | | |
| KOff | pass | 0.8582 | 171.3 / 170.2 | 0.2188 | -0.4% | 5.018 | 0.1187 / 0.0928 / 0.0921 | 7575 / 539 | |
| v2 default | pass | 0.8582 | 171.3 / 171.2 | 0.2181 | -0.7% | 5.005 | 0.1205 / 0.0924 / 0.0916 | 321 / 51 | 0 |

Every verdict passes for both saving arms. G2: gsm8k outputs 0 differ,
accuracy delta 0.0 pp. G3: E_ext 1.0, excess 0, no eviction. Hits:
gsm8k warm local 0.922 / ext 0.070 / total 0.992 / H* 0.992; replay warm
local 0.730 / ext 0.268 / total 0.998 / H* 0.998, the same for both arms.
Max KV usage 31%, so the pressure send never fired. The replay hold
(all passes) is mean 255 / max 1106 ms for the default against 21 / 156 ms
for KOff, with no measurable warm or cold cost on TP1. Batching cuts
gsm8k save RPCs 24x (7575 -> 321) with gsm8k seconds unchanged.

## Held-out check (`2026-10-24-dsv2-lite-p4-it5-heldout`, GPU 0,1, 64gb, run_gate pgid 2260465, 01:56:55 to 02:12:10Z, rev it5-kv090)

The batching knobs were tuned on R2EGym conversations 0..7, the gate's
default slice. This run uses conversations 8..15, which no tuning run
has seen. The gate has no `--replay-skip` flag (`workloads.build_turns`
takes `skip`, but `run_gate` does not pass it), so the run reads
`~/kexi/tmp/dsv2-p4/r2egym-heldout-skip8.jsonl` = lines 9..256 of
`r2egym-256.jsonl` (line 9 checked byte-identical by md5) through
`--replay-file`. That is a non-official flag, recorded here. The other
settings are the same as the tuning runs (8 convs x 6 turns). Host check:
no other pegaflow server started or stopped during the run.

| arm | cold p50 | vs baseline | cold req/s | vs baseline | G4 cold p50 / req/s | warm2 / warm3 p50 | G4 warm p50 / req/s | RPCs | hold mean / max ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2651 | | 6.141 | | | 0.201 / 0.216 | | | |
| KOff | 0.2370 | -10.6% | 5.940 | -3.3% | pass / fail | 0.129 / 0.126 | pass / pass | 568 | 17 / 117 |
| v2 default | 0.2146 | -19.0% | 6.100 | -0.7% | pass / pass | 0.128 / 0.128 | pass / pass | 52 | 167 / 955 |

Both saving arms: replay digests 1768 external / 5520 local matched, 0
mismatched; E_ext 1.0, excess 0, load_errors 0, no eviction; warm hits
local 0.733 / ext 0.265 / total 0.998 / H* 0.998. Max KV usage 30%, and
the pressure send did not fire.

The default passes every G4 replay verdict on the held-out slice, and
batching off fails cold req/s (-3.3%), as on the tuning slice. This
slice's baseline cold p50 (0.265) sits well above its own warm passes
(0.248 / 0.201 / 0.216), so the -10 / -19% cold p50 figures are mostly
baseline noise. Cold req/s is the steadier comparison: -0.7% for the
default, -3.3% for KOff.
