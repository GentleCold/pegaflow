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
| 0, 1, 2 | MambaSpec (GDN), align | `Checkpoint` | 528 | 2146304 (conv + ssm, unpadded), one segment | `[p/528 - 1]` | `[after/528 - 1]` only when `after % 528 == 0` |
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
