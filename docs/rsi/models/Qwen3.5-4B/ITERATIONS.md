# Qwen3.5-4B (P5) iterations

Branch `feat/rsi-qwen3.5-4b` (off `feat/semantics-rsi` 6545352), remote tree
`~/kexi/pegaflow-rsi-qwen35`, GPU4 on .44, vLLM 0.32.0, TP1.

## it0 (2026-10-24): spec dump, self-test, mapping proposal

### Model and vLLM settings

- `Qwen3_5ForConditionalGeneration`, 32 decoder layers in a 3:1 pattern
  (`linear_attention` x3, then `full_attention`): 24 Gated DeltaNet (GDN)
  layers and 8 full-attention layers. Weights: `~/kexi/models/Qwen3.5-4B`.
- `mamba_cache_mode`: vLLM 0.32 has only `"align"` and `"none"`
  (`config/cache.py:63`). With prefix caching on, the hybrid model config
  forces `"align"` (`model_executor/models/config.py:686-687`); `"none"` means
  no prefix caching. The spec dump confirms `cache.mamba_cache_mode = "align"`
  with no flag set. gate.json should pin `--mamba-cache-mode align` anyway,
  so the setting is explicit.

### Spec dump (`spec.json`, from the baseline arm)

Run `~/kexi/rsi-runs/2026-10-23-qwen3.5-4b-selftest` (gsm8k 128). Page and
hash sizes: block 528 tokens in every group, `hash_block_size` 528,
`cache_hit_alignment_tokens` 528. Other fields: `kv_cache_layout` LBNHC,
481 GPU blocks, `max_model_len` 8192.

| gid | class | layers | page bytes (real) | state |
|---|---|---|---|---|
| 0 | `MambaSpec` (GDN) | 0, 4, ..., 28 | 2162688 (2146304) | conv `[3, 8192]` bf16 + ssm `[32, 128, 128]` fp32 |
| 1 | `MambaSpec` (GDN) | 1, 5, ..., 29 | 2162688 (2146304) | same |
| 2 | `MambaSpec` (GDN) | 2, 6, ..., 30 | 2162688 (2146304) | same |
| 3 | `FullAttentionSpec` | 3, 7, ..., 31 | 2162688 | 4 KV heads x 256, bf16, 528 tokens |

Notes:
- The Mamba page holds conv (49152 B) and ssm (2097152 B), padded to the
  attention page (16384 B of padding).
- All three Mamba groups have `prefix_cacheable: true`,
  `num_speculative_blocks: 0`, `num_prefill_checkpoint_blocks: 0`,
  `prefix_replay_tokens: 0` and `tokens_per_state: -1`.
- The attention group has `block_table_token_alignment: 128` and
  `dcp_sharded: true`.
- `prefix_cacheable_group_ids` and `transfer_group_ids` are both
  `[0, 1, 2, 3]`.
- Four KV tensors, one per group. Each holds 8 layers x 481 blocks,
  `block_stride` 2162688, `layer_stride` 1040252928.

### Self-test result: stopped at step 1

- `plain` and `baseline` ran: gsm8k 128 accuracy 0.7578 / 0.7656. 20 of 128
  cold outputs differ between plain and baseline. There is no gate.json yet,
  so the batch-invariance setting is undeclared.
- `selftest:honest` failed at engine init:
  `NotImplementedError: SelftestConnector: group 0 is MambaSpec; only
  full-attention groups are supported`. The run stopped there, so the
  liar/bitflip/offbyone arms did not run.
- The oracle maps MambaSpec to `checkpoint`. `oracle.group_kinds(spec)` gives
  `[checkpoint(528), checkpoint(528), checkpoint(528), full(528)]`, so the
  oracle side has a mapping.
- The baseline probe digested only group 3: all 256 digest records have
  `group: 3`. No MambaSpec block was digested, yet `digest_skipped_groups`
  is `{}`. So on this model G2's digest checks see the attention bytes only,
  and nothing in the run record says so. Cause: see the mapping section.
- Warm pass: `local_hit` 1056 = 2 x 528 on the first requests, so vLLM's own
  align cache resumes at block multiples.

Sent to the gate owner. The rest waits for the gate fix.

### How vLLM 0.32 allocates, caches and resumes the GDN state

These points come from reading the installed vLLM 0.32.0 package. Paths are relative to `vllm/`.
- **Block size.** `Platform._align_hybrid_block_size` (`platforms/interface.py:1081-1124`)
  makes the attention page at least as large as the Mamba state: 16 x
  cdiv(2146304, 16 x 4096) = 528 tokens. In align mode it then sets
  `mamba_block_size = block_size` and pads the Mamba page by 0.76%.
- **Grouping.** `_get_kv_cache_groups_uniform_page_size` (`v1/core/kv_cache_utils.py`)
  forms one group per 8 layers: the 24 GDN layers are split three ways
  (`layers[i::3]`) and the 8 full-attention layers form the fourth group. All
  four tensors alias at offset 0, so layer slot j of every group shares
  memory, and block ids come from one shared pool.
- **Page bytes.** One page per (block, layer) holds conv state `[0, 49152)`
  (bf16 `(3, 8192)`), then the SSM state `[49152, 2146304)` (fp32
  `(32, 128, 128)`), then 16384 bytes of padding (`layers/mamba/abstract.py:28-42`).
  `kv_cache_layout` does not affect the inside of a Mamba page.
- **Per-request Mamba blocks** (`MambaManager`,
  `v1/core/single_type_kv_cache_manager.py:1415+`).
  - A running request holds one real state block, at index
    `cdiv(computed + scheduled, 528) - 1`. While a block boundary is being
    crossed it briefly also holds the previous block as the copy source.
    Every other index is the null block (id 0).
  - Before each forward the model runner copies the state forward into the
    new column (`precopy_mamba_align_fused_kernel`, `v1/worker/mamba_utils.py:549-631`;
    V2 runner `gpu/model_states/mamba_hybrid.py:182-228`).
  - GDN reads and writes the state in place at block-table column
    `(seq_len - 1) // 528`.
- **Where a checkpoint exists.** The state for "exactly k x 528 tokens" exists
  only when a step ends on k x 528. The scheduler arranges this in three ways:
  - it clips prefill chunks to 528 multiples (`v1/core/sched/scheduler.py:443-556`;
    with a 2048-token budget they end at 1584, 3168, ...);
  - it forces a stop at `floor((P-1)/528) x 528`;
  - decode crosses boundaries one at a time.

  Interior blocks of a chunk are never written.
- **vLLM's own prefix cache.** By default (`prefix_cache_retention_interval=0`)
  vLLM hashes and keeps only the replay checkpoint `floor((P-1)/528) x 528`
  (`kv_cache_coordinator.py:65-86`). Earlier chunk-end checkpoints are freed
  unhashed.
- **Local hits.** `find_longest_cache_hit` for the Mamba groups returns the
  last block cached in all three Mamba groups. The hybrid coordinator caps the
  attention hit to that, so the local hit always equals the Mamba hit.
- **External hits.** An external hit at p (`get_num_new_matched_tokens`
  is a single scalar for all groups) gets exactly one allocated block per
  Mamba group, at index `(p-1)//528` (`single_type_kv_cache_manager.py:326-367`).
  The local Mamba hit is discarded, so the connector must write the state at
  p for all three Mamba groups, plus full-attention blocks
  `[local, p/528)`.
- **Mamba blocks at finish.** At `request_finished_all_groups`, the Mamba
  lists are `[0, ..., 0, running_block]`, and the running block holds the
  state after `num_computed` tokens, which is usually not block-aligned. So
  checkpoints cannot be saved from the finish-time block table.
- **Save hook.** vLLM provides
  `SchedulerOutput.kv_connector_block_state.boundary_state_offloads =
  {req: [(group_id, block_id, boundary_tokens)]}` (`scheduler.py:1454-1471`,
  `sched/output.py:221`). Its source comment says these "cannot be reconstructed
  from a connector's append-only block table". The bytes are valid after that
  step's forward, until the block is freed or reused. vLLM's
  OffloadingConnector stores Mamba from this hook (`offloading/scheduler.py:1333-1403`)
  and transfers the unpadded page (`offloading/worker.py:126-129`).
- **Zeroing.** `needs_kv_cache_zeroing` is True for this model, but only new
  attention blocks are zeroed (`v1/worker/utils.py:105-158`). A fresh Mamba
  state is masked by `has_initial_state`, not zeroed.
- **Spec decode is out of scope.** No MTP here. With MTP enabled,
  `num_speculative_blocks` grows the conv state and the page, and the eagle
  block-drop shifts checkpoints.

Not verified: an external load at a non-aligned p (no assertion found); the
K/V split inside the 1024-byte attention cell. Neither is needed for the
mapping below.

### Mapping proposal (it0, no code)

| group | spec | Kind | unit (bt) | stored bytes per (block, layer) | `needed(p)` | `saveable(before, after)` |
|---|---|---|---|---|---|---|
| 0, 1, 2 | MambaSpec (GDN), align | `Checkpoint` | 528 | 2162688, the padded page (states 2146304 + 16384 padding; see "Adapter draft"), one segment | `[p/528 - 1]` | `[after/528 - 1]` only when `after % 528 == 0` |
| 3 | FullAttentionSpec | `Full` | 528 | 2162688, one segment | `[0, p/528)` | `[before/528, after/528)` |

No new Kind is needed. `Kind::Checkpoint` already states "the state at `p` is
stored under block index `p / block_tokens - 1`", which matches vLLM's
column `(p-1)//528`. `resolve` uses step lcm(528, 528) = 528, so a hit is
usable only at a p where all three Mamba groups hold a checkpoint at p and
the attention group holds every block before it, with p <= prompt_len - 1.

The adapter and core need these changes (it1, after the gate fix):
- **Save source.** `saveable` says when a checkpoint becomes storable, but
  the GPU block id for it must come from `boundary_state_offloads` for that
  (req, group, boundary), not from the block table. Today the core saves
  from the block table at index b, and for Mamba that is wrong: the index is
  right, but the block at that index is null or already replaced once
  compute moves on. Two things must hold:
  - the save is planned in the step whose forward wrote that state;
  - the copy finishes before vLLM can reuse the block.

  The it2 async-save fences already order saves before reuse of finished
  requests' blocks. A copied-forward source block freed mid-request needs
  the same hold, or a sync save for this group.
- **Load target.** `block_ids[g][p/528 - 1]`, the one non-null Mamba entry
  after allocation, for each of the three groups.
- **Bytes.** Register each GDN layer with `bytes_per_block = 2146304` and
  stride 2162688. The adapter for `MambaSpec` must reject `mamba_cache_mode != "align"`,
  `num_speculative_blocks > 0`, and shapes or dtypes it does not declare.
- **Sync loads.** The current core sync-load gate requires one KV group and
  `not needs_kv_cache_zeroing`, so with 4 groups every load is async. That
  is fine for a first cut.
- **Hit-rate expectation.** Only boundary states vLLM writes can be saved:
  chunk ends, prompt-replay stops and decode crossings. A replay turn that
  extends a conversation resumes at the previous turn's last saved boundary,
  so the stored checkpoint density is what bounds H* for this model. G5
  (capacity) then decides whether saving every chunk-end state is worth its
  2 MB per GDN layer per boundary.

### Save offers: what vLLM actually hands the connector

Read after the mapping above (`v1/core/single_type_kv_cache_manager.py:430-443, 1860-2000`):
- `boundary_state_offloads` entries come from `MambaManager.cache_blocks`
  for blocks that got a hash, plus a CoW-preserved partial tail. With the
  default `prefix_cache_retention_interval=0` only the replay boundary and
  shared-prefix junctions are hashed, so only those are offered. Denser
  external checkpoints need `--prefix-cache-retention-interval 528` (or
  `None`, dense), which also changes vLLM's own retention. That is a run
  setting for gate.json and the model profile, not a semantics change.
- vLLM's docstring: offered blocks "are not kept alive by the request
  block table for the whole request, so a caller that reads them
  asynchronously must pin them first". `pop_blocks_for_free` drops
  not-yet-taken offers of a freed request.
- vLLM's OffloadingConnector fences the offered block id and flushes the
  job if the scheduler re-allocates that block.
- For this model `needs_kv_cache_zeroing` is True, so
  `may_zero_or_copy_blocks` already makes the v2 core save synchronously
  in `wait_for_save`, after the forward that wrote the state. If the offer
  for a boundary arrives in the same step whose forward writes it, a sync
  save needs no extra fence. This ordering is not verified yet; it1 checks
  it on GPU4 with a slice.

### Adapter draft (it0, no core change)

`python/pegaflow/connector_v2/adapters/mamba.py`, registered for `MambaSpec`:
- `group`: `Group.checkpoint(block_size)`.
- `layer`: one segment = `page_size_bytes` (2162688, padded). The server
  addresses block b at `data_ptr + b * bytes_per_block` (the gRPC register
  path passes no block stride), so the copy unit must equal the page
  stride. Storing the 16384 B padding costs 0.76% and needs no server
  change. Shard `Heads(tp_rank, tp_size)`, or `Replicated` when
  `tp_replicated`.
- `layout`: state shapes/dtypes, through a new namespace-only
  `LayerDecl.layout` field. It is appended to the namespace fingerprint only
  when non-empty, so existing namespaces do not move.
- `view`: checks vLLM's raw `[B, 1, 1, real]` int8 view with pages `page`
  bytes apart. Verified on .44 (CPU, `create_kv_cache_views`) for LBNHC,
  LBHNC, BLNHC and BLHNC: all give stride `(2162688, ...)` for a group
  tensor.
- Rejects `mamba_cache_mode != "align"`, speculative or prefill-checkpoint
  blocks, `tokens_per_state > 0`, `num_heads != 1`, DCP, no layer views,
  and states larger than the page.

`build_layout` still raises `Unsupported(... boundary_state_offloads)` for
checkpoint groups. The test pins that Qwen3.5-4B now fails there and not at
"no adapter".

Tests (`test_connector_v2_core.py`, stub specs from spec.json, stub +
native semantics): declaration and view, rejected fields, foreign
geometry, layout failure at the core gap, and V0 `seal_plan` of the full
hybrid layout at TP1/TP2. 158 passed on .44 (CPU only). The full default
gate gives 545 passed and 1 failed: `test_e2e_version_preflight`
compares the remote tree's installed version, unrelated.

Kind: none new. Checkpoint covers align mode. The open items are core
work: save from offers, always load the state at p, and fences.

## it1 (2026-10-24): core checkpoint support, ordering check, non-regression

### Core change (Kind-driven, no model or spec names)

- `Scheduler.plan_saves` takes the step's `boundary_state_offloads`. For a
  checkpoint group, the block ids come from the offers (`_offered_ids`:
  boundary `k*bt` -> block `k-1`), not from the block table; a boundary that
  `saveable` picks but vLLM did not offer is skipped. Full and window groups
  are unchanged.
- Loads: no code. The checkpoint's needed block `[p - bt, p)` always lies
  past the local hit (`p - num_computed >= step >= bt`), so the state at p
  is always loaded, and vLLM drops its local state for an external hit.
- `layout.py`: the checkpoint `Unsupported` is gone.
- Worker: a layout with any checkpoint group saves synchronously in
  `wait_for_save` (an offered block is only kept until vLLM frees or reuses
  it). This model already had `may_zero_or_copy_blocks`.
- Miss attribution (`Scheduler.hit_stats`, logged every 500 lookups and at
  shutdown as `hit stats`): for each lookup, the longest resume point the
  full group backs vs the planned `p`; when it is longer, count the set of
  storage groups missing there (`missing=<sids>`) and the lost tokens.
- `--prefix-cache-retention-interval 528` went to the gate owner as a
  gate.json request (every arm, baseline included). The connector does not
  set it. Without it vLLM offers only the replay boundary and junctions.
- Tests: `test_checkpoint_saves_from_offers` (4 cases),
  `test_checkpoint_loads_state_at_p` (cold, after a local hit, older
  checkpoint), `test_miss_attribution_names_the_missing_group`; the
  checkpoint case of `test_layout_rejects` is gone. 174 passed on .44 (stub
  and native semantics).

### Ordering check (GPU4, temporary, not in the repo)

Tool: `~/kexi/rsi-tools/p5/` (`p5_check.py` connector subclass,
`p5_check_run.py` driver, `p5_analyze.py`). One process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), its own pegaflow-server, retention
528, `max_num_batched_tokens 1056`, `max_num_seqs 4`, natural text (the
model's LICENSE). Six prompts cover: decode across a boundary, a forced
stop at exactly 3 x 528, a junction at 1056 shared with another prompt, an
extension of another prompt. Phases:
- R: one request at a time, prefix cache reset before each, loads disabled
  (the unprimed recompute).
- A: all prompts batched with tight GPU blocks; every checkpoint save is
  captured (GPU page bytes after the sync save).
- C: reset vLLM's prefix cache, overwrite every KV byte with random data,
  rerun A's prompts; every loaded page is captured right after the load.

| Run | GPU blocks | Saves from offers | Loads byte-equal to saved | Saved vs recompute (R) | Outputs |
|---|---|---|---|---|---|
| order2 | 26 | 42/42 | 18/18 | rounding only: max conv 0.0625, ssm 0.024 (control between two states: 28.9 / 3.89) | C == A == R for all 6 |
| order3 | 16, 300 output tokens, 2 GB pool, 6 preemptions | 39 saves, all offered (78 offers) | 27/27 | 24 states in both: rounding only, max conv 0.0625, ssm 0.014 (control 28.6 / 4.28) | C == A for 5/6; p4 diverges at token 152 |

- No save used a block that was not offered at that step, and no saved
  state was overwritten before the copy (the bytes match R up to bf16/fp32
  rounding).
- order3 p4 (corrected after review, `docs/rsi/reviews/P5-qwen35-it0-it1.md`
  should-fix 1): the divergence at output token 152 (position 2052) comes
  before p4's preemption in C (C step 3572, after at least 212 output
  tokens), and p4 (A req 4) was never preempted in A. Up to the divergent
  token, the connector supplied the three GDN states at 1584 (byte-equal to
  A's saves) and full blocks 0-2. Blocks 0 and 1 have the same layer-0 sha in
  all their loads. Block `a2bb744a` loads the same in both C loads, but the
  tool did not capture A's save of it (full saves were not captured). So
  open: batch non-invariance in decode (p4 decoded beside reqs 10 and 12 in
  C, beside 3 and 5 in A; GDN has no batch-invariant mode), or a wrong
  full-attention block. A-vs-R is not comparable past token 64 (R ran 64
  output tokens). it2 plan item 3 settles it.
- Unused offers are boundaries already inside a local or external hit, or
  offers on steps whose computed boundary is a different one.
- hit stats (order3): `{"lookups": 1445}`, no group-incoherent miss.

Run dirs: `~/kexi/rsi-runs/2026-10-24-qwen3.5-4b-p5-it1-order2`, `...-order3`.

### Shared-core non-regression: Qwen3-4B slice (GPU4)

`~/kexi/rsi-runs/2026-10-24-qwen3-4b-p5-it1-nonreg-b` (rev c4c1615, full
gsm8k 1319, replay 8 x 6; pgid 2133248, 00:12:47Z-00:42:10Z host UTC). The
v2 vllm.log `from=` line points at `~/kexi/pegaflow-rsi-qwen35/python`. The
run is not official (it is a slice). Every v2 verdict passes. The in-run
deltas match P3's accepted it2e slice (`2026-10-23-qwen3-4b-p3-it2e-slice`):

| Check | it2e (P3) | P5 it1 |
|---|---|---|
| G2 (digests, outputs, acc 0.8582 cold/warm) | pass | pass |
| G3 gsm8k warm E_ext / external hit | 1.0 / 0.899 | 1.0 / 0.899 |
| G3 replay warm E_ext / external hit | 1.0 / 0.991 | 1.0 / 0.991 |
| G3 no_eviction | pass | pass |
| replay warm TTFT p50 (v2 vs baseline, warm2/warm3) | 0.096/0.091 vs 0.213/0.212 | 0.093/0.092 vs 0.210/0.213 |
| replay warm req/s | 6.84/6.84 vs 5.49/5.48 | 6.85/6.86 vs 5.49/5.49 |
| replay cold TTFT p50 / req/s | +1.3% / +0.05% | -1.0% / -0.5% |
| gsm8k cold s (v2 vs baseline) | 181.9 vs 165.5 (fail; the rerun passed) | 166.0 vs 164.8 (+0.7%, pass) |
| gsm8k warm s | 165.3 vs 165.4 | 165.1 vs 164.8 (+0.2%) |

No `hit stats` line was logged. The layout has one full group and no
policy cut, so no miss was counted.

### Shared-core non-regression: DSV2-Lite TP2 slice (GPU 4,5)

`~/kexi/rsi-runs/2026-10-23-dsv2-lite-p5-it1-nonreg-b`. Tree: P4's
`feat/rsi-deepseek-v2-lite` 55e4ff1 (contains 1bfb590, P4's gate.json and
spec) plus the P5 it1 commits, merged in a temporary tree (`.rsi-rev`
7f51197, not pushed). Settings: `--tp 2 --gpu 4,5 --pool-size 64gb`, full
gsm8k, replay 8 x 6, arms plain, baseline, v2. pgid 2094470,
23:36:03Z-00:12:03Z host UTC. The v2 vllm.log `from=` line points at the
merged tree. Compared with P4's it2 slice (`2026-10-23-dsv2-lite-p4-it2-slice`,
3d1de18, GPU 0,1):

| Check | P4 it2 slice | P5 it1 |
|---|---|---|
| loaded blocks, gsm8k ext / local (matched, 0 mismatched) | 10526 / 357952 | 10390 / 358088 |
| loaded blocks, replay ext / local | 1382 / 7094 | 1382 / 7094 |
| accuracy cold / warm (baseline 0.6725) | 0.6725 / 0.6778 | 0.6725 / 0.6778 |
| split-only fails (`split_invariant` false): warm_differ, computed_exact self, vs baseline | 647, 0/23032, 27526/39042 | the same |
| E_ext gsm8k / replay; no_eviction | 1.0 / 1.0; pass | 1.0 / 1.0; pass |
| replay warm TTFT p50 (v2 vs baseline) | 0.102/0.110 vs 0.164/0.165 | 0.096/0.111 vs 0.164/0.178 |
| replay warm req/s | 7.45/7.31 vs 5.94/5.94 | 7.48/7.48 vs 5.95/5.96 |
| replay cold TTFT p50 | +6.4% (fail) | +10.2% (fail) |
| replay cold req/s | -2.3% (fail) | +0.0% (pass) |
| gsm8k cold / warm s | +0.2% / -2.3% | +0.0% / -2.6% |

The cold replay TTFT failure is P4's open item. Save batching is off at
55e4ff1, and P4's in-run batching-off control measured +5 to +20% across
runs. The checkpoint change does not touch this layout: one MLA group, no
offers, `sync_saves` unchanged. No pass is contaminated: the worst p99 is
cold, 0.760 vs 0.732.

An earlier attempt (`...-dsv2-lite-p5-it1-nonreg`, pgid 2065961) is
invalid. The launch lacked `PYTHONPATH=<tree>/python`, so vLLM imported the
shared tree's connector, which has no MLA adapter, and the v2 arm failed at
init. The first Qwen3-4B slice (`2026-10-23-qwen3-4b-p5-it1-nonreg`) had
the same fault and is discarded.

## it2 plan (not run; starts when `feat/rsi-gate-hybrid` lands)

Preconditions:
- The hybrid gate digests the Mamba groups, so `G2.digest_groups` can pass.
- The gate.json has `--prefix-cache-retention-interval 528` on every arm.
- GPUs are assigned by the coordinator.

Every launch sets `PYTHONPATH=<tree>/python`, or `--under-test-path` once
feat/rsi-g5 is merged. I check the `from=` line before reading any verdict.

### 1. Slices

The Qwen3.5-4B hybrid-gate slice uses the same shape as the it1 slices:
- full gsm8k and replay 8 x 6;
- arms plain, baseline and v2;
- then G5 `cap` at the gate's capacity fraction.

The report has these hit-rate tables:
- per pass: local / external / total / `H*` / `E_ext`, v2 vs baseline;
- per group: hit tokens, fallback count and tokens lost (the gate's
  per-group report).

Expectations:
- With retention 528, every boundary is offered, so replay warm should
  reach `E_ext` 1.0 with no eviction.
- If the gate.json is not ready, I run once without it. Then only replay
  boundaries and junctions are saved. That `E_ext` is recorded as the
  cost of not retaining, not judged.

Sync checkpoint saves (review should-fix 4). Layouts with a checkpoint
group save synchronously in `wait_for_save`. With retention 528, every
prefill chunk that ends on a boundary copies 24 GDN pages (about 52 MB)
plus the attention block before the next step. order3's `save stats`:
`flush_ms` 42.9 over 5 save steps and 57.8 over 3 (8.6 to 19 ms per save
step, up to 41 ms of it `wait_ms` on compute). That can fail the ≤ 2% cold
replay G4 checks without any noise. In each slice:
- record per-step save time from the v2 arm's `save stats` lines for the
  cold replay pass (`flush_ms / save_steps`, `wait_ms`, `rpc_ms`), next
  to the step time (pass seconds / steps);
- attribute the cold-replay G4 TTFT and req/s delta against it before
  calling anything noise: Σ `flush_ms` over the pass vs the delta in pass
  seconds;
- if cold G4 fails and the flush time covers the delta, the next
  hypothesis is one change: fence the offered block id (keep vLLM from
  reusing it until the copy is done, as vLLM's OffloadingConnector does,
  see "Save offers") and save asynchronously.

### 2. Per-group miss attribution in the gate slices

`hit stats` is logged today only after a miss is counted. In a gate slice,
silence cannot be told apart from a missing log line. it2 changes this:
- At shutdown, log `hit stats` unconditionally, with lookups and zero
  counts. This is a small connector change, tested in
  `test_miss_attribution_names_the_missing_group`.
- Keep the every-500-lookups line, with its timestamp.

The gate runs one vLLM process per arm and workload, so the counters cover
all passes of that workload. Cold passes count nothing, because the server
starts empty and the full group backs no resume point. For a per-pass
split, I align the timestamped lines with the gate's pass boundaries. If
the run record has no pass timestamps, I ask the gate owner instead of
guessing.

Cross-check per slice:
- Connector `missing=<sids>` counts and `lost_tokens` against the gate's
  per-group "tokens lost to fallback".
- Unlimited pool (gsm8k, replay): both must be 0.
- G5 cap: misses are expected. A `missing=1,2,3` without `missing=0`
  means the server evicted the checkpoint groups but kept the full group,
  and that goes to the server agent as group-coherent eviction evidence.
  Any disagreement between the connector's count and the gate's count is
  a bug in one of them and blocks the slice.

### 3. Settling the order3 p4 divergence

In order3, p4's completion in C (external hit at 1584) diverges from A
(cold) at output token 152, before any preemption of p4 (see the it1
ordering check). The checkpoint loads were byte-equal; the full-attention
blocks 0-2 were loaded identically in C but A's save of block 2 was not
captured. Candidate causes:
- (a) batch non-invariance in decode: p4 decoded beside different
  requests in A and C (no preemption needed);
- (b) split non-invariance: GDN prefill resumed at 1584 is chunked
  differently from a run through, like MLA on DSV2;
- (c) a wrong full-attention block in `[0, 1584)`.

Both legs always run.
- Leg 1, no connector (in-process driver, GPU only, plain vLLM, same
  prompts, retention 528, enough GPU blocks that nothing is preempted):
  - batch control: p4 alone vs p4 decoded beside the other prompts, cold
    both times, compared token by token;
  - split control: p4 alone cold vs p4 alone after vLLM's own local hit at
    1584.
- Leg 2, the connector run (order3 config): capture the saved bytes of
  full blocks too (`P5_SAVE_FULL=1`) and compare every full-attention
  block p4 loads in C, `[0, 1584)`, with A's save of the same hash; the
  checkpoint comparison stays.
- Both legs log the top-2 logprobs per output token. At the first
  divergent token the margin (top-1 minus top-2, A and C) is recorded: a
  near-tie (< ~1e-2) supports rounding, a large margin points at (c).

| Leg 2 (bytes) | Leg 1 | Margin at divergence | Conclusion | Next step |
|---|---|---|---|---|
| any full or checkpoint load differs | any | any | connector load or save bug (c) | Blocks it2 |
| all equal | batched or split p4 diverges | near-tie | vLLM non-invariance (a) or (b) | Gate owner decides `batch_invariant` / `split_invariant` for this model, as P4 did for DSV2 |
| all equal | no divergence | any | open | Back to the coordinator before any gate run |

The hybrid gate's digests catch wrong loaded bytes only for requests that
take an external hit after vLLM dropped their state, and its Mamba
digests only if taken in the offer step. With an unlimited pool and 481
GPU blocks, gsm8k and replay may never preempt or evict locally, so this
evidence needs a preemption-forcing slice or the G5 `cap` pass. The
coordinator passes that to the hybrid-gate brief. Output-level
non-invariance shows only in `G2.*_outputs`, which depends on this
model's invariance flags in gate.json.

### 4. Should `~/kexi/rsi-tools/p5` move into the repo?

No. The tools stay outside the repo, as P4's did.
- The tools subclass `PegaKVConnectorV2` and read its plan internals. The
  gate is connector-agnostic, and `rsi/gates/**` belongs to the gate
  owner, so they cannot go there as written.
- Once the hybrid gate digests Mamba state, its `G2.digest_*` covers the
  load side: external loads matched against computed state. Its
  negative self-tests cover what poisoning showed.
- Two parts are worth giving to the gate owner as technique, not code:
  - When to digest a Mamba state: the state block is valid in the step
    vLLM offers it in `boundary_state_offloads`, after that step's
    forward, and is freed or reused soon after. The probe should digest
    it then, not at finish.
  - The saved-vs-unprimed-recompute comparison with a rounding bound
    (conv ≤ 0.0625, ssm ≤ 0.024 against controls of 28.6 / 4.28). Under
    `batch_invariant` it can become exact equality.

  I send both to the gate owner (a81695e98582ef447) with a pointer to
  `p5_check.py`. Whether they belong in the gate is the gate owner's call.
- The tools stay until Qwen3.5-4B passes G2 on the hybrid gate, and are
  then deleted.
