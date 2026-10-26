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
- order3 p4 (corrected after review, it0/it1 review
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

Provenance (review should-fix 2): order2 and order3 ran on the working
copy before c4c1615 was committed, and no revision was recorded. They are
superseded by order4.

#### order4: order3 rerun at the committed revision

`~/kexi/rsi-runs/2026-10-24-qwen3.5-4b-p5-it1-order4`, GPU4 only, pgid
2321219, 02:45:06Z-02:50:41Z host UTC. The run dir's `config` file
records rev 863cc99 (c4c1615 plus the per-request `hit stats` fix), the
start and end times, and the knobs. The `from=` lines point at
`~/kexi/pegaflow-rsi-qwen35/python`. Config: order3's (16 GPU blocks in A,
300 output tokens in every phase including R, 2 GB pool), plus
`P5_SAVE_FULL=1` (full-group saves captured as well) and `P5_LOGPROBS=2`.

| Check | Result |
|---|---|
| checkpoint saves from offers | R 42/42, A 39 saves from 78 offers, all offered |
| saved (A) vs unprimed recompute (R) | 42 states in both: max conv 0.0703, ssm 0.125 (control between two states: 9.67 / 12.4) |
| checkpoint loads (C, after poison) vs A's saves | 27/27 byte-equal |
| full-attention loads (C) vs A's saves, all layers | 29/29 byte-equal, including `a2bb744a` (both loads of p4) |
| preemptions | A: reqs 1, 3, 5; C: 7, 11 (p4), 12 |
| hit stats | `{"lookups": 21}` (it was 1445 per call), no miss |
| outputs C == A | 5/6; p4 diverges at output token 152 again |

p4 at token 152: A picks 16809 (logprob -0.8195) over 6704 (-0.9445), and
C picks 6704 (-0.7964) over 16809 (-0.9214). The top-2 margin is 0.125 in
both, one bf16 step at logit magnitudes 16 to 32. C's completion equals
R's (p4 alone, unprimed, no loads). A (cold but batched) is the odd one
out. p3 also differs between the batched phases (A == C) and R. With
every loaded byte equal to its save and a one-ulp flip, cause (c) is
excluded for this run, and batch composition (a) is the remaining
explanation. The no-connector controls (it2 item 3, leg 1) confirmed it
and closed the divergence (below).

Analyzer fix: a preempted request reloads into reused GPU block ids, so
pairing loads by `(req, sid, block_id)` mixed up two loads of the same
request (a first pass reported 4 false full-block mismatches). Each load
is now paired with its own load plan. order2 and order3 give the same
checkpoint counts (18/18, 27/27) with the fix.

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

Review it0 + it1 (cc12f5e..4ec49a9): no must-fix; should-fix 1 (order3 p4
divergence misattributed to preemption) corrected above, 2 (ordering
check rerun on the committed revision as order4), 3 (`hit stats` call vs
request counts) and 4 (sync checkpoint saves as a G4 risk, H1 in it2)
addressed below; report file deleted when
migrated to rsi-notes (text in `feat/rsi-qwen3.5-4b` 4bb7d8d).

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

Status: closed as batch non-invariance in vLLM 0.32 GDN decode, not the
connector. Leg 2 ran as order4 at 863cc99 (see the it1 ordering check):
every full and checkpoint load is byte-equal, and the margin at the
divergence is one bf16 step. Leg 1 ran without the connector (results
below).

#### Leg 1 result (`~/kexi/rsi-runs/2026-10-24-qwen3.5-4b-p5-it2-ctrl`)

Plain vLLM 0.32.0 (no `kv_transfer_config`, PYTHONPATH unset). Same
in-process driver knobs and prompts as order4 (retention 528,
`max_num_batched_tokens 1056`, `max_num_seqs 4`, 300 output tokens,
top-2 logprobs). Tree rev 863cc99 (the connector is not loaded). GPU4
only, pgid 2328944, 02:55:32Z-02:59:27Z host UTC. Script md5s are in the
run's `config`.

| Run | p4 token 152 (top-2 logprobs) | p4 vs alone |
|---|---|---|
| plain, batched, 16 GPU blocks (order3/order4 tight config) | 16809 (-0.8326) over 6704 (-0.9576), margin 0.125 | diverges at 152 |
| plain, batched, 200 GPU blocks | 16809 (-0.6735) over 6704 (-1.1735), margin 0.5 | diverges at 152 |
| plain, alone, cold (prefix cache reset before each prompt) | 6704 (-0.7964) over 16809 (-0.9214) | reference |
| plain, alone after vLLM's own local hit (split control) | identical to alone | equal, 300/300 |
| order4 A (connector, batched cold) | 16809 (-0.8195) over 6704 (-0.9445), margin 0.125 | diverges at 152 |
| order4 C (connector, external hit at 1584) | identical to alone, same logprobs to all digits | equal, 300/300 |
| order4 R (connector, alone, loads disabled) | identical to alone | equal, 300/300 |

- Plain vLLM with the same tight config gives the same one-bf16-step flip
  at p4 token 152 when batched, so the connector does not cause it. p3
  (first diff at 148) and p2 (at 170) also differ between batched and
  alone runs in plain vLLM. Which ones differ depends on the batch.
- Split: vLLM's own local hit came at 1056, not 1584. The 1056 boundary
  was the longest one in vLLM's prefix cache when p4 reran. Resuming at
  1056 does not change p4's output, and order4 C resumes at 1584 (from
  the connector, byte-checked loads) and also equals the alone run. So
  in this setup GDN has no split non-invariance at 1056 or 1584.
  Batch composition is the only cause seen.
- Preemption counts for these control runs were not collected:
  `LLM.get_metrics()` asserted in-process. The batched 200-block run does
  not need preemption to flip.
- For the hybrid gate, this model is batch-sensitive in decode. A cold
  vs warm `G2.*_outputs` comparison is meaningful only under
  `batch_invariant` (if vLLM supports it for GDN), or with comparisons
  limited to same-batch runs. That decision is the gate owner's.

Tools: `p5_ctrl.py`, `p5_ctrl.sh`, `p5_ctrl_cmp.py` in
`~/kexi/rsi-tools/p5` (out of the repo, like the rest).

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

## it2 (2026-10-25): hybrid gate slices, pinned saves

Branch `feat/rsi-qwen3.5-4b`. Merged `feat/semantics-rsi` (f8eac30,
hybrid gate) as 6f4aff1 and `feat/rsi-deepseek-v2-lite` (563a693, P4
accepted) as 554ad9d. Base server `pegaflow-semantics-rsi` 056eb4b. GPU5
only. Every slice runs with `--under-test-path <git-archive tree>/python`
and is checked against vllm.log `from=`.

Merge notes for the reviewer:
- P4 put `storage_tp` into the layout namespace, so the Qwen3-4B test
  vector `QWEN3_NAMESPACE_V1` moved from cf6b5725 / cd8acc0c to a25ea150 /
  bdda8997 (stub / native). The P4 review covered this. The new values are
  what 563a693 computes on its own.
- `plan_saves(..., kv_usage=, offers=)`: P4's one GroupSave per
  (storage group, request) applies to the checkpoint groups and the full
  group alike.

### Slice s1 (25fcccb): correctness passes, G4 cold and G5 cap fail

Run `2026-10-25-qwen3.5-4b-p5-it2-s1-25fcccb`: arms plain, baseline, old,
native, v2; gsm8k 128, replay 8 x 6, cap 8 x 6 at 0.25/0.5/0.75, preempt.
Not official (slice, capacity floor 0.05x GPU KV).

Correctness of v2, all pass:
- digests: gsm8k 80/80 external, 1515 local, computed 5/5; replay 39/39;
  preempt 106/106 external (18 preempted requests, 56 resumed-episode
  loads); `G2.digest_groups` [0, 1, 2, 3];
- accuracy: `alone_ref` 0.7188 = baseline;
- G3 `E_ext` 1.0 on gsm8k, replay and preempt, no evictions.

G4 (v2 / old / native / baseline):

| metric | v2 | old | native | baseline |
|---|---|---|---|---|
| warm TTFT p50 (warm2/warm3) | 0.138/0.116 | 0.167/0.167 | 0.121/0.123 | 0.210/0.207 |
| warm req/s | 6.87/6.86 | 6.70/6.70 | 6.47/6.49 | 5.40/5.40 |
| cold TTFT p50 | 0.278 (+27.5%) | 0.346 | 0.224 | 0.218 |
| cold req/s | 4.805 (-6.2%) | 4.801 | 5.084 | 5.124 |
| gsm8k cold s (128 q) | 12.166 (+2.7%) | 10.719 | 12.132 | 11.844 |

Hits, unlimited pool (local / ext / total; H* is the unlimited oracle):

| pass | v2 | old | native | H* | E_ext v2 |
|---|---|---|---|---|---|
| replay warm | 0.678 / 0.257 / 0.936 | 0 / 0.936 / 0.936 | 0.678 / 0.257 / 0.936 | 0.936 | 1.0 |
| gsm8k warm | 0.818 / 0.109 / 0.926 | 0 / 0.926 / 0.926 | 0.919 / 0.007 / 0.926 | 0.926 | 1.0 |
| preempt warm | 0.667 / 0.261 / 0.928 | 0 / 0.928 / 0.928 | 0.870 / 0.058 / 0.928 | 0.928 | 1.0 |

lmcache: pending (`feat/rsi-gate-lmcache` is not merged).

G4 cold cost attribution: checkpoint layouts saved synchronously. Over the
replay cold window the `save stats` show 34 save steps, `flush_ms`
1517.7 of 33.5 s, `wait_ms` 1254 (waiting for the forward), `rpc_ms`
261, `max_job_ms` 109. Every chunk that ends on a 528 boundary blocks the
step until its 24 GDN pages and attention block have been copied. That
accounts for the cold TTFT and req/s gap. H1 below targets it.

`hit stats` never reached vllm.log. The API server force-kills EngineCore
at teardown, so `Scheduler.shutdown` never runs. bc9c38a logs the counters
when the last live request finishes and at most every 30 s.

### Slice s2cap (d2a484d): the 0-byte cap is a sub-floor pool

Cap only, baseline + v2, 8 x 6, f = 0.5: C = 1.63 GiB = 0.05x GPU KV
(33.29 GB). The gate's floor is 2x, so this run supports no G5
conclusion. What it shows:
- v2 loaded 0 bytes. `hit stats` `missing=1,2,3` 4 of 160 lookups (lost
  4224 tokens): the full group backed a resume point and every GDN group
  had been evicted.
- per pass the server evicted 176-265 blocks of ~17 MB (3.3-5.0 GB
  reclaimed) against ~3.5 GB saved, leaving 21-30 resident blocks. One
  reclaim round is `RECLAIM_BATCH_SIZE = 512` blocks, more than the whole
  pool (~100 blocks), and eviction is per (hash, group) key. Both are listed
  under PROGRESS "Server limitations" and marked sub-floor.
- H*_C 0.870, H_lru 0.734, H_freeze 0.846, v2 hit 0.720 (= baseline
  local).

Official sizing (gate `plan_cap_convs`, reported to the orchestrator, fix
assigned to the gate owner): `--cap-convs auto` picks 203 of 256
conversations at estimated U 147.1 GB (need 146.5). At 8 x 6 the estimate
is 4.47 GB against 3.49 GB measured (0.78), so the measured floor likely
misses at 203. The official run waits for the sizing branch.

### H1 (5667cd7): pin saved blocks instead of saving synchronously

Change (model-agnostic; it applies to every layout that saved
synchronously before):
- The scheduler pins each planned save's blocks in vLLM's `BlockPool`
  (`touch`, one reference per block) when it builds the plan, and unpins
  them (`free_blocks`) once every worker reports that step's saves done.
  The report is `PegaWorkerMetadata.saved_through`, aggregated as the min
  over ranks.
- A pinned block stays out of the free queue, so vLLM can neither hand it
  out again nor zero or COW-copy into it before `handle_preemptions`
  runs. That covers the V2 runner's zeroing order as well as reuse.
- These layouts drop the preemption flush and the finish hold, and are
  not batched, so pins stay short. `has_pending_push_work` keeps the
  engine stepping until the pins are released. If no block pool was
  bound, saves are disabled (fail closed).
- Unit tests: `test_pinned_saves_hold_blocks_until_every_worker_saved`,
  `test_pinned_saves_fail_closed_without_a_pool`,
  `test_pinned_worker_reports_saved_through`.

Prediction: cold replay TTFT p50 and req/s within 2% of baseline, `save
stats` `flush_ms` ~0. Correctness unchanged: digests, preempt, alone_ref.

### Slice s3 (5667cd7): H1 result

Run `2026-10-25-qwen3.5-4b-p5-it2-s3-5667cd7`: arms plain, baseline, v2;
full gsm8k, replay 8 x 6, preempt, cap 8 x 6 at f = 0.5. Not official
(slice, capacity floor 0.05x GPU KV). vllm.log logs `saves=async-pinned`.

Correctness of v2, all pass: `G2.digest_groups` [0, 1, 2, 3]; gsm8k
digests 13455 local + 50/50 external, also vs baseline; replay 127 local
+ 39/39 external; preempt 311 local + 106/106 external; `G2.gsm8k`
alone_ref delta 0.0 pp (0.7188); G3 E_ext 1.0 (gsm8k, replay);
`G3.no_eviction` 0 in every pass. Accuracy cold / warm 0.7839 / 0.7892
(baseline 0.7824, plain 0.7892).

G4 (v2 vs baseline):

| metric | s1 (sync saves) | s3 (H1) |
|---|---|---|
| replay cold TTFT p50 | +27.5% | 0.2250 vs 0.2167, +3.8% (fail) |
| replay cold req/s | -6.2% | 5.050 vs 5.137, -1.7% (pass) |
| replay warm TTFT p50 (warm2/warm3) | 0.138/0.116 vs 0.210/0.207 | 0.140/0.139 vs 0.207/0.209 |
| replay warm req/s | 6.87/6.86 vs 5.40/5.40 | 6.86/6.87 vs 5.41/5.41 |
| gsm8k cold / warm s (1319 q) | (128 q: +2.7%) | 99.37 / 99.59 vs 94.41 / 94.60, +5.3% (fail) |

Hits, unlimited pool (local / ext / total = H*): replay warm 0.678 /
0.257 / 0.936, preempt warm 0.667 / 0.261 / 0.928, gsm8k warm 0.919 /
0.006 / 0.926. Same as s1. old and native were not run in s3; their s1
values stand. lmcache: pending.

Replay cold `save stats`: 34 save steps, `flush_ms` 0, saver-thread
`wait_ms` 1508 and `rpc_ms` 229. Those are off the step thread. H1 holds:
the save cost is off the step, and req/s is within 2%. TTFT p50 is still
+3.8%.

The gsm8k cost is not from saves. Over the pass there is 1 save step (40
blocks). It has another cause:
- plain takes 64.3 s and baseline (probe connector only) 94.4 s. vLLM's
  connector path costs ~47% on this model before v2 adds anything.
- the scheduler counted 2703 lookups, but the server logged 6046 prefix
  queries (vLLM re-queries a waiting request every scheduling pass). Every
  one returned 2 hit blocks (the 8-shot prefix, 2 x 528).
- each pass runs `query_prefetch` for all 4 groups, and then releases
  every lease: vLLM's local hit (1056 tokens) already equals the largest
  resume point the prompt allows, `(num_tokens - 1) // 528 * 528`. That
  is ~8 gRPC calls per pass on the scheduler thread, ~48k per gsm8k pass,
  for a hit that can never beat vLLM's own.

### H2 (next): skip lookups that cannot beat vLLM's local hit

Change (model-agnostic, `Scheduler._lookup`, no policy knob). The bound is
`semantics.resolve` run over optimistic answers, where every block of a
group not yet queried counts as present. So it follows each Kind's own
resume rule, with no model names and no extra rule:
- before any query: if the bound is not past `num_computed`, nothing is
  queried. For a full-only layout (Qwen3-4B, DSV2) that means "the prompt
  has no loadable full block past the local hit". There the hit result
  was already 0. The query is not side-effect free, though: a server prefix
  query refreshes the LRU recency of every block it hits
  (`pegaflow-core` `ReadCache::get_prefix_blocks` -> `refresh_recency`).
  Before H2, vLLM's per-pass re-queries of a waiting request refreshed the
  blocks vLLM already held locally; after H2 they do not, so server
  eviction order at capacity can move on full-only models too (review
  P5-it2 S1; a Qwen3-4B / DSV2 regression slice measures it). With a
  checkpoint group the bound is the next resume step past the local hit
  (528 on Qwen3.5);
- after storage group 0 (full, prefix query): the same check with group
  0's real answer. If it fails, group 0's lease is released and the other
  groups are never queried, so they create no lease;
- if `resolve` raises inside the shortcut, the lookup queries as before
  and `_plan_hit` reports the error.

Hit and save bookkeeping is unchanged. `_plan_hit` still runs, so the
lookup is counted in `hit stats`. A skipped group cannot be `missing`,
because the counter only attributes resume points past the local hit.
`_hits`, `request_finished` and save planning are untouched.

Unit test `test_lookup_skips_queries_that_cannot_beat_the_local_hit`,
stub and native: full-only with no block past the local hit and with one
block past it; hybrid at and past group 0's bound; and a hybrid layout
whose next resume step lies past the prompt.

Prediction: gsm8k cold / warm within 2% of baseline, replay cold TTFT p50
closer to baseline. Hits and correctness unchanged. Replay cold p50 is
judged on the official-size run: the 8 x 6 slice is noisy (P4 saw
baseline swing 0.171-0.192).

### Slice s4b (800630d): H2 result

Run `2026-10-25-qwen3.5-4b-p5-it2-s4b-800630d`: arms plain, baseline, v2;
full gsm8k, replay 8 x 6, preempt. No cap: the cap was sub-floor at 8 x 6
anyway. Not official. An earlier attempt (`...-s4-800630d`) is void: the
gate looked for the server binary in the gate tree, and the v2 server
failed to start. s4b passes `--pegaflow-server` for the base build.

All v2 verdicts that ran pass (cap verdicts not run):
- G2: digests 110/110 external in gsm8k (also vs baseline), replay 39/39,
  preempt 106/106; 0 local mismatches; `G2.digest_groups`; alone_ref
  delta 0.0 pp;
- G3: E_ext 1.0 (gsm8k, replay); no eviction in any pass.

| G4 metric | s3 (H1) | s4b (H2) v2 vs baseline |
|---|---|---|
| gsm8k cold s | +5.3% (fail) | 95.28 vs 95.33, -0.1% (pass) |
| gsm8k warm s | +5.3% (fail) | 94.67 vs 95.30, -0.7% (pass) |
| replay cold TTFT p50 | +3.8% (fail) | 0.2194 vs 0.2159, +1.6% (pass) |
| replay cold req/s | -1.7% | 5.069 vs 5.133, -1.3% (pass) |
| replay warm TTFT p50 (warm2/warm3) | 0.140/0.139 vs 0.207/0.209 | 0.146/0.135 vs 0.209/0.209 |
| replay warm req/s | 6.86/6.87 vs 5.41/5.41 | 6.85/6.97 vs 5.41/5.41 |

The gsm8k server log has 23 prefix queries, against 6046 in s3. Replay has
141.

Hits, unlimited pool (local / ext / total = H*, E_ext 1.0): replay warm
0.678 / 0.257 / 0.936, preempt warm 0.667 / 0.261 / 0.928, gsm8k warm
0.911 / 0.015 / 0.926. Totals match s3 and s1. gsm8k shifts 0.008 from
local to external.

H2 verdict: prediction met. Replay cold p50 is +1.6% on this slice, and the
official-size run decides it.

## it2 official (c5d3f04): `2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`

Arms: plain, baseline, old, native and v2. Workloads: full gsm8k, replay
32 x 12, cap on all 256 conversations at C/U 0.25 / 0.5 / 0.75, and preempt.
GPU5, base server 056eb4b. `official: true`, with no `not_official_because`.
- Capacity floor: U 167.96 GB, C(0.5) 83.98 GB = 2.52x GPU KV (floor 2x).
  Each cap stage sizes its pool to C. The run's `--pool-size` was the
  default 64gb, which only applies to the uncapped workloads (all of them
  show 0 evictions).
- Gate and under-test rev: c5d3f04.

### Correctness (hard): v2 passes every verdict

- G0: `G0.spec` has 0 diffs; provenance is the overlay.
- G2: `G2.digest_groups` covers [0, 1, 2, 3]. Digests, local + external:
  - gsm8k: 13460 + 45/45 (vs baseline: also 13460 + 45/45);
  - replay: 456 + 540/540, with 21 local uncertain;
  - preempt: 311 + 106/106;
  - cap: 15240 + 59627/59627.
  
  0 mismatches anywhere. `G2.gsm8k` alone_ref delta is 0.0 pp (0.7188).
  Accuracy cold / warm is 0.7908 / 0.7877 (baseline 0.7877).
- G3: E_ext is 1.0 for gsm8k and replay. `G3.no_eviction` is 0 in every
  pass.
- G5 validity: `pool_ok` and `loads_ok` hold at every fraction. Excess is
  0. Unmatched requests: 0.

old passes every verdict. native fails `G0.spec` (its CPU layout is LBNHC),
`G2.digest_vs_baseline.gsm8k` and gsm8k cold +2.6%. It is a reference arm.

### G4 (hard)

v2 vs baseline. old and native are listed for reference.

| metric | v2 | baseline | v2 vs baseline | old | native |
|---|---|---|---|---|---|
| gsm8k cold s | 95.00 | 94.68 | +0.34% (pass, 2% slack) | 91.67 | 97.10 |
| gsm8k warm s | 94.55 | 94.88 | -0.35% (pass) | 92.32 | 95.94 |
| replay cold TTFT p50 s | 0.512 | 1.204 | -57% (pass) | 0.505 | 1.046 |
| replay cold req/s | 8.19 | 3.61 | 2.27x (pass) | 8.06 | 7.79 |
| replay warm TTFT p50 s (warm2/warm3) | 0.300/0.295 | 1.112/1.119 | -73% (pass) | 0.359/0.349 | 0.175/0.171 |
| replay warm req/s (warm2/warm3) | 14.61/14.66 | 3.73/3.73 | 3.9x (pass) | 14.07/14.07 | 13.15/13.16 |

H2's prediction holds at the official size. gsm8k is within 0.4% of
baseline. Replay cold, which the 8 x 6 slice put at +1.6%, is less than half
of baseline: with 32 conversations, vLLM's GPU cache no longer holds the
earlier turns and the store serves them. Cap save stats (0.5) are 28 save
steps out of 499, `flush_ms` 0 and `failed` 0.

### Hit rates, unlimited pool (G3)

Columns are local / ext / total, as shares of prompt tokens, then H* and
E_ext.

| pass | v2 | old | native | H* | E_ext v2 |
|---|---|---|---|---|---|
| gsm8k warm | 0.920 / 0.006 / 0.9255 | 0 / 0.9255 / 0.9255 | 0.925 / 0.001 / 0.9255 | 0.9255 | 1.0 |
| replay cold | 0.232 / 0.571 / 0.803 | 0 / 0.801 / 0.801 | 0.230 / 0.573 / 0.803 | 0.803 | 1.0 (old 0.998) |
| replay warm (x3) | 0.778 / 0.169 / 0.946 | 0 / 0.946 / 0.946 | 0.778 / 0.169 / 0.946 | 0.946 | 1.0 |
| preempt warm | 0.667 / 0.261 / 0.928 | 0 / 0.928 / 0.928 | 0.870 / 0.058 / 0.928 | 0.928 | 1.0 |

v2 reaches H* on every pass. lmcache: pending.

### Hit rates at capacity (G5)

This table is as judged at c5d3f04, with plain Belady as H*_C. The gate
since 1f00e5a has a fallback-aware oracle; the re-scored table is in
"Re-score with the fallback-aware oracle" below.

The numbers are the timed warm passes (warm2 + warm3, 3716 requests,
17.48M prompt tokens). In every simulation the reference local hit is the
baseline's, 0.248 of the prompt. The unlimited H* is 0.946. H_lru and
H_freeze are simulated on v2's trace; on the old and native traces they
are within 0.002, except native's H_lru at 0.75, which is 0.862.

| C/U | H*_C | H_lru | H_freeze | hit v2 / old / native | E_C v2 / old / native |
|---|---|---|---|---|---|
| 0.25 | 0.690 | 0.794 | 0.451 | 0.794 / 0.794 / 0.794 | 1.183 / 1.235 / 1.175 |
| 0.5 (gated C) | 0.878 | 0.792 | 0.669 | 0.794 / 0.794 / 0.788 | 0.828 / 0.867 / 0.815 |
| 0.75 | 0.946 | 0.906 | 0.885 | 0.907 / 0.906 / 0.946 | 0.880 / 0.943 / 0.928 |

At every fraction v2's hit equals H_lru within 0.002 and equals old's hit
within 0.001.
- The E_C gap to old (0.04-0.06) is accounting, not hit. ext counts
  `min(a, o) - max(ref, own)` per request, so on requests where v2's own
  GPU-local hit is above the baseline reference (own > ref) the excess
  earns nothing: 403k / 428k / 778k tokens at 0.25 / 0.5 / 0.75. In
  aggregate v2's own local rate (0.182) is below the reference (0.248).
  old reports a local hit of 0, so it loses nothing. Adding v2's
  over-reference tokens back gives 1.235 / 0.867 / 0.944, old's
  E_C within 0.001. This term is by design (GATES.md: a higher own
  local hit "is vLLM's cache, not the store's"). It does mean that E_C
  across arms is comparable only at equal own-local hits.
- The hit is 0.794 at both 0.25 and 0.5. That is the within-pass reuse:
  turn k+1 reuses turn k, and the cold pass is also 0.794. The warm passes
  replay the 256 conversations in the same order, which is a cyclic scan
  larger than C, so an LRU store keeps none of the earlier pass. The
  server's eviction is plain LRU per key (see PROGRESS "Server
  limitations"). At 0.75 the earlier pass partly survives: 0.906.
- Group-coherence loss at the official cap size is negligible: `hit stats`
  shows `missing=1,2,3` in 4 of 7441 lookups at 0.25 (4 of 7443 at 0.5,
  2 of 7433 at 0.75), 2640 lost tokens.

### G5 at C/U 0.25: E_C > 1 is an oracle modelling gap for Checkpoint groups

Finding for the gate owner. It is offline: `rsi/gates/oracle.py` at
c5d3f04 was imported on the run's probe traces, and the gate code was not
edited. The scripts are `~/kexi/tmp/p5-g5-attr/attr.py` and `attr2.py`,
outputs `out*-*.txt`.

**Not the accounting.**
- H*_C, H_lru and H_freeze come from the same `simulate` with the same
  reference local hit and the same `min(..., max(unlimited, local))` cap.
- The measured hit (0.7939) matches H_lru (0.7937) for every arm.
- C is `f x` the measured U, so the Checkpoint estimate (1734 vs 2348
  blocks) plays no part.

The anomaly is H*_C < H_lru, entirely inside the oracle.

**Cause.** Belady's next-use sets are built only at each request's
unlimited resume point P: `_external_keys(kinds, r, unlimited[i], ...)` =
`semantics.needed(g, P)`.
- For a Full group, `needed(q)` is a subset of `needed(P)` for q < P, so
  every fallback block already has a use at that request.
- For a Checkpoint group, `needed(q)` is the checkpoint at q, which is not
  in `needed(P)`. A checkpoint at an earlier boundary is "used" only by a
  later request whose own P lands there, for example the same conversation
  turn in the next pass. Belady therefore sees its next use far away, and it
  declines to admit it or evicts it first (0.25: 3368 not admitted against
  2562 stored per GDN group).
- When Belady cannot keep a request's P-checkpoint, `resolve` falls back to
  a boundary whose checkpoint it has also dropped. LRU keeps everything
  recent and resumes there.

Full-only models (Qwen3-4B, DSV2) cannot show this, which is why it first
appears on the hybrid model.

**Numbers (v2 trace, timed passes):**

| C/U | requests with H_lru > H*_C | tokens (share of prompt) | of them below the unlimited P | blocks LRU loads there (ckpt / full) |
|---|---|---|---|---|
| 0.25 | 995 | 2.69M (0.154) | 995 | 2985 / 5585 |
| 0.5 | 239 | 0.52M (0.030) | 239 | 717 / 1406 |
| 0.75 | 0 | 0 | 0 | 0 |

(Belady > LRU elsewhere: 871 requests / 0.88M tokens at 0.25, 1995 / 2.03M
at 0.5, 651 / 0.69M at 0.75.)

**Counterfactual bar** (`attr2.py`, no knowledge of LRU). A Belady whose
use set for request i is the union of `needed(q)` over every block end
q <= P, so every fallback resume point counts as a use. It is above H_lru
on every request at every fraction (`lru > bar` on 0 requests):

| C/U | H*_C (gate) | fallback-aware bar | H_lru | E_C v2 / old / native vs that bar |
|---|---|---|---|---|
| 0.25 | 0.690 | 0.841-0.843 | 0.794 | 0.881 / 0.920 / 0.874 |
| 0.5 | 0.878 | 0.889-0.891 | 0.792 | 0.813 / 0.852 / 0.799 |
| 0.75 | 0.946 | 0.941-0.946 | 0.906 | 0.886 / 0.949 / 0.928 |

At 0.75 the fallback-aware variant is 0.005 below the gate's H*_C on two
traces. Spreading uses over every fallback point dilutes the greedy
choice, so neither variant dominates.

Suggestion for the gate owner (not implemented): either make the Belady
use set fallback-aware for Kinds whose `needed(q)` is not a subset of
`needed(P)`, or report H*_C as the max of both variants. Either way E_C
would be at most 1 here: about 0.87-0.92 at 0.25 and 0.80-0.85 at 0.5. The
oracle docstring already says it is "a strong reference, not a proven
optimum". This measures how far it falls short for Checkpoint groups.

### Re-score with the fallback-aware oracle (gate 1f00e5a, merged as 1002b80)

`feat/semantics-rsi` 1f00e5a takes H*_C as the best of `belady`,
`belady_fallback` (uses at every fallback resume point; only for Kinds
whose `needed` is not prefix-closed) and `lru`. This is the suggestion
above. Offline re-score of the official record: every cap stage's
`capacity` is recomputed from its probe with the merged `oracle.py`, and the
verdicts with the merged `run_gate.py`
(`~/kexi/tmp/p5-g5-attr/rescore_cap.py`, output `rescore-1002b80.txt`).
No verdict flips for any arm (v2, old and baseline pass; native fails as
before, on `G0.spec`, `G2.digest_vs_baseline.gsm8k` and
`G4.gsm8k_cold.seconds`), and `g5_problems` stays empty.
Only H*_C and E_C move; hit, H_lru and H_freeze are the same simulations.

| C/U | H*_C old -> new (winner) | H_lru | hit v2 / old / native | E_C v2 / old / native | H_lru / H*_C |
|---|---|---|---|---|---|
| 0.25 | 0.690 -> 0.841 (belady_fallback) | 0.794 | 0.794 / 0.794 / 0.794 | 0.881 / 0.920 / 0.874 | 0.920 |
| 0.5 (gated C) | 0.878 -> 0.889 (belady_fallback) | 0.792 | 0.794 / 0.794 / 0.788 | 0.813 / 0.852 / 0.799 | 0.848 |
| 0.75 | 0.946 -> 0.946 (belady) | 0.906 | 0.907 / 0.906 / 0.946 | 0.880 / 0.943 / 0.928 | 0.944 |

H*_C and H_lru / H*_C are on v2's trace; old's and native's H*_C are within
0.002. E_C is now at most 1 at every fraction, and v2's E_C is within
0.01 of H_lru / H*_C at 0.25 and 0.5. The v2-old E_C gap
(0.039 / 0.040 / 0.063) is the same own-local term as before: the
over-reference tokens are unchanged and only the denominator moved, so
adding them back gives 0.920 / 0.851 / 0.944, old's E_C within 0.002
(old's own H*_C differs by up to 0.001).

### Hit-rate judgment: close enough

- Unlimited: v2 = H* on every pass (E_ext 1.0), the same as old and native.
- Capacity: v2's hit = H_lru (within 0.002) and = old (within 0.001) at
  0.25 / 0.5 / 0.75. The E_C differences to old are the own-local
  accounting term (within 0.002).
- Fallback-aware oracle (gate 1f00e5a): E_C v2 is 0.881 / 0.813 / 0.880.
  The gap to H*_C at 0.5 (0.794 vs 0.889) is LRU on a cyclic trace. That
  is the base server's eviction policy, shared by every PegaFlow arm;
  native, with vLLM's own CPU cache, is 0.788. Server proposals are parked
  (PROGRESS "Server limitations").
- The E_C > 1 at 0.25 under the c5d3f04 gate was the oracle modelling gap
  above, and the merged oracle closes it (0.881).
- Group-coherence loss is 4 lookups out of 7441.
- lmcache: pending.

### Review P5-it2 (ACCEPT-WITH-FIXES, `reviews/P5-it2.md`)

- S1: the H2 text above is corrected: skipping a query also skips its
  server-side LRU refresh, so H2 is not a pure no-op on full-only layouts.
  The regression slices (Qwen3-4B and DSV2, gsm8k + official-size cap at
  this branch's head) are prepared in PROGRESS and wait for free GPUs.
- S2: merged `feat/semantics-rsi` a4523f3 (lmcache gate) as 7247751.
  Offline re-judge of the official record with the merged gate
  (`~/kexi/tmp/p5-g5-attr/rejudge.py`, every stage's `digest_self`
  recomputed with the new digest_compare, verdicts with the new
  run_gate): every verdict of every arm is unchanged, with no changed
  field, and `g5_problems` stays empty. No `G2.resume_reference` is added,
  since Qwen3.5-4B is split-invariant (`resume_stage_needed` is false).
  One artefact of re-judging from another tree: the record's probe files
  live in the run's gate tree (`~/kexi/p5-gate-c5d3f04`), so `G0.provenance`
  flags them "not under the gate tree". With `REPO` aliased to the run's
  tree (`REJUDGE_REPO`) the flag is gone; it is a path check, not a
  record change.
- S3: `Worker.sync_saves` deleted (dead since H1: a pinned layout without a
  block pool disables saves, it never falls back to sync saves). The unit
  test keeps only the `may_zero_or_copy_blocks` classification that
  `_pin_saves` uses.
- Nit, spec decode: pinned blocks raise vLLM's `ref_cnt`. vLLM 0.32's
  `is_block_writable` (`ref_cnt == 1`) is used only for speculative Mamba
  blocks, and spec decode is out of scope here. Before enabling spec
  decode on a checkpoint layout, check that a pinned save cannot make a
  speculative block look unwritable (README "Saves" says the same).
- `PegaWorkerMetadata` keeps its name (a rename touches the vLLM-facing
  metadata class and buys only log clarity).

## it3 (2026-10-26): per-request finish fence

Target: G4 warm replay TTFT p50 and req/s no worse than `native` (judged).

### Attribution (accepted by main)

- On the full-only models (Qwen3-4B it7, DSV2 it7 m2off) the warm gap vs
  native is queue time, not load time. v2 holds more KV: DSV2 warm KV
  usage 94% vs native 65%.
- The cause is the finish fence. A finished request was held until every
  save job submitted before its finish was done (`_finishing[req] =
  _submitted`). With batched saves (`save_batch_steps=32`, hold up to
  1 s) that is a mean hold of 123.6 ms (max 1516) on Qwen3-4B it7 and
  127.4 ms (max 1371) on DSV2 m2off (`save stats` `sent_wait_ms/sent`).
- 86-90% of requests finish 3 or more steps after their last save, so their
  own saves are long done when they finish.
- KV model (probe replay, `p5-it3-kv.py`): mean KV-blocked admission steps
  for DSV2 loaded requests are 7.3 at the measured hold, 4.8 at a 1-step
  hold, and 5.1 for native.
- Qwen3.5-4B (pinned saves, no finish hold) has a separate cohort-admission
  gap. It is a later candidate and is not addressed here.

### Hypothesis H3 (d201ac3): fence a finished request on its own last save

- Mechanism: the scheduler records the plan step of each request's last
  chosen save (`StepPlan.finished: {req: step}`). The worker maps the step
  to the last job sequence number submitted by the end of that step's
  `save`, and releases the request once that job is done.
- Why it is safe:
  - Only the request's own saves read its blocks. A block it shares through
    the prefix cache stays allocated by vLLM for the other holder.
  - A preempted request's freed blocks are covered by the step's flush, so
    it keeps its fence entry across the preemption.
  - The saver is one thread with a FIFO queue, so "job k done" implies every
    earlier job is done. `_completed` advances in `finally`, so failed,
    raised and dead-saver jobs still release.
- Prediction:
  - most holds drop below one step;
  - DSV2 and Qwen3-4B warm TTFT p50 at or better than native, and req/s no
    worse than native;
  - Qwen3.5-4B unchanged (pinned layout, the code path is not used).
- Unit coverage (`tests/test_connector_v2_core.py`, 242 passed natively):
  - (a) sync-load fail-closed hold: `test_sync_load_outcome_gates_saves`
    now also asserts the fences;
  - (b) preemption: `test_preempted_request_fences_on_its_last_save`;
  - (c) no chosen save, no hold, and (d) FIFO / failed / raised jobs:
    `test_finish_fences_on_the_requests_last_save`;
  - (e) zeroing / COW / checkpoint layouts: they are pinned, with
    `finished == {}` (`test_layouts_that_zero_or_copy_blocks`,
    `test_pinned_saves_hold_blocks_until_every_worker_saved`).
- Slices: Qwen3-4B TP1 GPU5 `baseline,native,v2` replay, then DSV2 TP2 if
  confirmed.
- Review: ACCEPT on c8a8842. The blocker was fixed in c8a8842: a
  preempted request's fence is dropped, else an abort during an async
  re-load frees the load's blocks. The should-fix items and nits are fixed.
- DSV2 S1 (`2026-10-26-dsv2-lite-p5-s1-4bb7d8d`, pre-fence) shows no
  regression vs it7 m2off: G5 E_C 0.621 (it7 0.631), gsm8k E 1.000.

### it3 result: not confirmed, inconclusive (confounded)

Slice `2026-10-26-qwen3-4b-p5-it3-slice-c8a8842` (GPU5, 32 convs, 264
warm requests, same replay content as it7):

| run | arm | warm2/3 TTFT p50 | mean | p90 | req/s |
|---|---|---|---|---|---|
| it7 (GPU0, KV 75,696 tok) | native | 0.561/0.563 | 0.907/0.901 | 2.16/2.17 | 9.29/9.43 |
| it7 | v2 (old fence) | 0.551/0.577 | 0.788/0.789 | 1.85/1.83 | 9.66/9.63 |
| slice (GPU5, KV 71,472 tok) | native | 0.564/0.543 | 0.984/0.978 | 2.18/2.18 | 9.17/9.12 |
| slice | v2 (it3 fence) | 0.645/0.655 | 0.819/0.817 | 1.72/1.72 | 9.94/9.94 |

- Hold (`sent_wait_ms/sent`): mean 123.6 -> 68.5 ms, max 1516 -> 1080.
  gsm8k: 565 ms; there, pressure_rpcs is 0 and KV is not short.
- vs native, slice: p50 +17.5%, mean -16.6%, p90 -21%, req/s +8.7%. In it7
  the same numbers were p50 -0.9%/+2.5%, mean -13%, p90 -15%, req/s +3%.
  Only the p50 moved against v2.
- Not comparable with it7:
  - The gate's fresh per-arm compile cache (fcd0534, in the slice's gate
    9e2111f, not in it7's ce645d7) makes every arm compile from scratch.
    A fresh compile profiles 9.82 GiB of KV (71,472 tokens); an AOT load
    from `~/.cache/vllm` profiles 10.4 GiB (75,696). The difference is not
    the GPU: in every run checked, KV size follows `Directly load AOT`.
    So every run under the current gate has -5.6% KV.
  - Host: GPU0 runs (it6, it7, mf2) have a baseline p50 of 3.5-3.9 s, the
    GPU4-7 runs 2.5-2.8 s, and the slice 2.98 s. ms/step for v2 is 24.1
    on GPU0 and 23.0 on GPU5.
  - KV model on it7's own timing (`kvcap.py`, hold 2 steps, R 2):
    shrinking the pool from 4731 to 4467 blocks raises v2's KV-blocked
    loaded requests 95 -> 146 (steps 11.9 -> 16.7), and native's
    75 -> 109 (15.1 -> 18.8). The smaller pool hurts v2 more, independent
    of the fence.
- Queue/load split (probe, steps, turn>0 loaded requests, n=232):
  - Load round trip is identical everywhere (p50 7, mean 7.6-7.8, both
    arms, both runs), so the whole gap is queue.
  - vLLM's prefill time is lower for v2 (56 vs 89 ms).
  - Queue histogram, buckets [0,3) [3,8) [8,16) [16,32) [32,64) [64,inf):
    - native is bimodal: 0/68-89/43-87/2-10/2-3/72-89;
    - v2 in it7: 22/50/25-33/47-53/23-25/55-57;
    - v2 in the slice: 16/47/15-17/35-39/71-72/43-46.
  - The smaller pool moves about 48 v2 requests from [16,32) into
    [32,64), right where the p50 is. That raises the p50, while the mean
    and the >64 tail improve. Native has almost nothing in that range, so
    its p50 is insensitive to it.
- Residual hold: the saver's compute-event wait averages 32 ms/job
  (~1.4 steps), plus the RPC. In warm replay 98% of RPCs are already
  pressure sends (2021/2054), so flushing the held batch earlier on a
  fence in it would gain little there. On gsm8k it would multiply RPCs
  (2635 releases vs 313 RPCs), which risks the 2% bound.
- Decision: keep it3. It is correct (review ACCEPT), the hold is -45%,
  and req/s, mean and p90 are better than native. The p50 change is
  confounded by the KV pool and the host. No revert unless same-run
  evidence says it hurts.
- Next, once the GPU is cleared: a same-run A/B on one GPU under one gate
  rev, back-to-back, replay only, `--arms native,<v2>`. One run with
  under-test 18607c1 (old fence), one with c8a8842. Judge v2-vs-native
  within each run.

### it3 same-run A/B (2026-10-26): keep, neutral to slightly positive

Runs `2026-10-26-qwen3-4b-p5-it3-ab-old-18607c1` and `-new-c8a8842`:
GPU0, TP1, gate c8a8842, back-to-back, replay only, `--arms native,v2`.
KV is 71,472 tokens in all four arms, so the pool is no longer a
confound. Both runs exit 1 only because they are not official (no plain,
baseline or gsm8k).

| run | arm | warm2/3 TTFT p50 | mean | p90 | req/s |
|---|---|---|---|---|---|
| old 18607c1 | native | 0.565/0.469 | 0.975/0.937 | 2.16/2.12 | 9.288/9.220 |
| old 18607c1 | v2 | 0.672/0.736 | 0.865/0.874 | 1.82/1.79 | 9.568/9.485 |
| new c8a8842 | native | 0.505/0.537 | 0.959/0.973 | 2.14/2.15 | 9.133/9.185 |
| new c8a8842 | v2 | 0.663/0.696 | 0.854/0.859 | 1.81/1.80 | 9.568/9.509 |

- G4 medians vs native: p50 +36.1% (old) and +30.5% (new) behind; req/s
  2.9% and 4.2% ahead.
- v2 from old to new: p50 -1.3%/-5.4%, mean -1.3%/-1.7%, req/s equal.
  - Hold (sent_wait/sent) falls from 107.7 to 67.2 ms (max 1521 to 1042).
  - Job wait (33.8 ms) is unchanged, and so is the share of pressure
    RPCs (2032/2066 and 2026/2063).
- Noise: native's p50 moves by about 10% between the two runs (0.469 to
  0.537 on warm3).
- The v2 p50 gap is the same in both runs and is not caused by the fence.
  - Turn 0 (`t0.py`): v2 gives convs 1-31 local=0 and loads 54,144
    external tokens, admitting about one conv per step. Native gives
    local=1264 and loads 14,990, admitting at steps 2-4. This is it4's
    target.
  - Turns > 0 that load: v2's queue p50 is 30-32 steps vs native's
    10-12, while the load round trip is the same (p50 7 steps).
  - ITL: v2 24.4-24.7 ms vs native 23.5-24.3 ms (within about 1 ms).
- Decision: keep it3.

## it4 (2026-10-26): wait for in-flight async loads (e985938, not gated yet)

### Attribution (TP2 official, confirmed offline on TP1)
- Qwen3-4B warm turn 0: convs 1-31 share a 1264-token prefix with conv0.
  - TP2 official: all 32 lookups land within 33 ms (steps are ~18 ms).
    Every lookup is a full hit planned before conv0's load lands.
  - TP1 slice and it7 warm2, from `t0.py`: v2 gives convs 1-31 local=0,
    admitted about one per step (steps 0-32), 54,144 external tokens.
    Native admits them at steps 2-6 with local=1264, 14,990 external
    tokens. Cold is identical in both arms (local=1264, steps 1-7).
- Mechanism: vLLM caches an async load's blocks only once it lands
  (`delay_cache_blocks`, `_update_waiting_for_remote_kv`). vLLM's
  OffloadingConnector returns None while a hit chunk is in
  `_chunks_being_loaded` (`offloading/scheduler.py` ~612-630, 920-943).
  v2 had no such wait, so every sharer loads its own copy.
- Qwen3.5-4B is different: warm2 turn 0 is local=0 in both arms (50,688
  external tokens each). The 1056-token prefix (2 x 528 blocks) is local
  only in cold. Native admits about one conv per 4 steps, v2 all 32
  within steps 0-4. The gain from the wait there is open.

### Hypothesis H4 (e985938)
- When another request's async load writes blocks that, with vLLM's
  local hit, give a resume point past it (`semantics.resolve`),
  `get_num_new_matched_tokens` returns None. It does not query, so no
  lease is held.
- Tracking: (sid, hash) refcounts of the blocks each non-sync, non-failed
  load plan writes. They are cleared on `finished_recving` /
  `failed_recving` in `update_connector_output` (before vLLM caches the
  blocks that pass) and on `request_finished`.
- Bounds: `loading_wait_ms` per request; sync loads are never tracked;
  off without prefix caching. No new knob. Counted as `deferred`.
- Prediction: Qwen3-4B warm turn 0 gets local=1264 for convs 1-31,
  external tokens fall toward native's, warm req/s returns to at least
  native. Turns >= 1 and cold are unchanged. Qwen3.5: neutral or small.
- Gate plan: after the it3 A/B and the Qwen3.5 plain self-test rerun
  (stale controls record). Qwen3-4B TP1 and a TP2 slice, then Qwen3.5,
  e985938 vs c8a8842. It replaces the 512-token async-start budget
  hypothesis unless an admission-burst cost remains after it.
- Order changed by main: the deferral run came before the Qwen3.5 self-test,
  which waits for slim to be merged.

### Review (ACCEPT with nits) and follow-up 02b6767
- No MUST items. The follow-up adds table cases for a local hit with an
  in-flight tail, two loaders of the same blocks, a re-queued request and
  `skip_reading_prefix_cache`. All five mutants (M1, M9, M2,
  skip-reading, clock reset) are killed.
- Edge-case behaviour changes (not on the gate's paths):
  - A deferral resets the prefetch wait clock.
  - Requests that skip the prefix cache are never deferred.
  - The connector's `prefix_caching` fallback is now True.
- Docs: `loading_wait_ms` bounds both waits, and 0 disables both.

### it4 result: confirmed on Qwen3-4B TP1 (`2026-10-26-qwen3-4b-p5-it4-defer-e985938`)

The run used GPU0, TP1 and gate c8a8842, with arms native and v2 at
e985938 on replay and gsm8k. KV is 71,472 tokens. It is not official
because it has no plain, baseline or cap arm. The control is the A/B run
`-new-c8a8842` on the same GPU and gate, run just before.

| run | arm | warm2/3 TTFT p50 | mean | p90 | req/s | cold req/s |
|---|---|---|---|---|---|---|
| defer | native | 0.555/0.467 | 0.989/0.941 | 2.18/2.09 | 9.299/9.263 | 5.766 |
| defer | v2 e985938 | 0.620/0.610 | 0.847/0.849 | 1.84/1.83 | 9.804/9.798 | 6.250 |
| A/B new | native | 0.505/0.537 | 0.959/0.973 | 2.14/2.15 | 9.133/9.185 | 5.808 |
| A/B new | v2 c8a8842 | 0.663/0.696 | 0.854/0.859 | 1.81/1.80 | 9.568/9.509 | 6.297 |

- Turn 0 in warm2 (`t0.py`): v2 now gives convs 1-31 local=1264 and loads
  14,960 external tokens. Native loads 14,990; c8a8842 gave local=0 and
  loaded 54,144. v2 admits them at steps 2-12; native admits them at
  steps 2-4.
- Hit stats: `deferred` is 93 for replay and 10 for gsm8k. G3 replay_warm
  local is 0.354, vs 0.326 before and 0.368 for native. There were no load
  errors.
- G4 medians vs native: req/s is 5.6% ahead (it was 4.2%). TTFT p50 is
  20.3% behind (it was 30.5%).
- v2, e985938 vs c8a8842:
  - req/s +2.5% / +3.0%.
  - p50 -6.5% / -12.4%; mean and p90 are flat.
  - ITL 24.5 -> 23.4 ms.
  - Cold is unchanged (6.250 vs 6.297, within noise).
- gsm8k: accuracy 0.8552 in both arms. v2 takes 172.4 s cold and
  169.9 s warm, vs native's 171.9 and 175.4 (+0.3% and -3.1%, inside the
  2% bound).
- What remains of the p50 gap: for turn>0 requests that load, v2's
  queue p50 is 25-26 steps vs native's 10-11, with the same load round
  trip (p50 7 steps). Turn-0 admission is also spread out (steps 2-12
  vs 2-4).
- The only traceback in the v2 replay log is vLLM's EngineDeadError after
  the gate's SIGTERM at shutdown. It is not a connector error.
- Decision: keep it4. Next are the TP2 slice and Qwen3.5 TP1, comparing
  the follow-up head against c8a8842.

### it4 TP2 pair: Qwen3-4B TP2, c8a8842 vs 02b6767 in-run

The two runs went back to back on GPUs 0-1. Both used gate rev c8a8842,
the replay workload, and native + v2 arms. KV is 203,616 tokens in every
arm. Runs: `~/kexi/rsi-runs/2026-10-26-qwen3-4b-p5-it4-tp2-{c8a8842,02b6767}`.
Neither run is official (no plain or baseline arm, exit 1 as expected).
Neither vllm.log has a traceback.

TTFT in seconds:

| run | arm | warm2 p50 / mean / p90 | warm3 p50 / mean / p90 | req/s w2 / w3 |
|---|---|---|---|---|
| c8a8842 | native | 0.165 / 0.164 / 0.212 | 0.171 / 0.169 / 0.231 | 19.28 / 19.07 |
| c8a8842 | v2 | 0.132 / 0.148 / 0.194 | 0.132 / 0.150 / 0.198 | 17.60 / 17.20 |
| 02b6767 | native | 0.161 / 0.168 / 0.228 | 0.166 / 0.164 / 0.215 | 19.21 / 19.13 |
| 02b6767 | v2 | 0.157 / 0.155 / 0.241 | 0.166 / 0.172 / 0.276 | 18.75 / 18.60 |

- G4 medians vs native in the same run:
  - c8a8842: TTFT p50 21.3% ahead, req/s 9.3% behind.
  - 02b6767: TTFT p50 1.1% behind (noise), req/s 2.6% behind.
- G3: hit rate is 0.99826 in all arms. Local hit was 0.8334 at c8a8842
  and is 0.8641 at 02b6767, equal to native's 0.8641. Turn-0 external
  tokens fell from 54,144 to 14,960; native's are 14,990. 93 requests
  were deferred.
- Loads fell from 31.0 to 25.2 GB, at 9.3-9.7 GB/s. Warm saves are
  0.1 GB. The cold save is 27.7 GB at 5.3 GB/s.
- Step time is v2 18.0/18.2 -> 17.1/17.3 ms, against native's 16.0-16.3.
  The share of warm3 tokens with an ITL of 25-50 ms is 18.3% -> 6.3%;
  native's is 3.4-5.0%. vLLM mean ITL at 02b6767 is 17.4/17.3 ms for v2
  and 16.5/16.7 ms for native.
- Queue steps for turn>0 loaded requests: v2 p50 4 (mean ~3.4), native
  p50 5-6 (mean ~6.2). At TP2, v2 is ahead on admission.
- The save hold is 258 ms (c8a8842) and 274 ms (02b6767), with 0 pressure
  RPCs. KV usage stays below 0.9, so batches wait the full 32 steps or the
  1 s cap.
- The remaining TP2 gap is about 6.5% longer decode steps. That is target
  (a), save/load copy contention. Admission is not the cause.

### it4 attribution: TP1 queue gap and turn-0 spread (offline, deferral run)

Source: `2026-10-26-qwen3-4b-p5-it4-defer-e985938`, cap 4467 blocks.
Scripts are in `~/kexi/tmp/p5-it3-attr/`: qcount, pair, qtime, t0q,
order, pair2, lastsave, inv, kvcap and itl.

**Loaded turn>0 requests: queue p50 of 27 steps for v2 vs 11 for native**

- The inputs are identical. All 232 turn>0 requests load in both arms,
  with about 1772 local and 3480 external tokens.
- This is a phase shift, not a per-request delay.
  - The mean gap is 35.9 vs 33.8 steps.
  - Paired by native's bucket, v2 minus native is +1 for native gaps under
    10. It is +16 at p50 for native gaps of 10-29. For native gaps of 30
    or more it is -3 at p50 and -10 at mean.
  - By turn, the gap p50 (v2/native) is: t1 5/8, t2 5/6, t3 21/8,
    t4 26/10, t5 68/71, t6 73/72, t7 74/73, t8 39/70, t9 9/6.
  - v2's first_step drifts from +5 steps at turn 0 to +34/+44 at turn 9,
    while both runs take the same time.
- In wall time, v2 queues less.
  - vLLM request_queue mean is 756 ms vs 843 ms; the queue_time sum is
    199.7 s vs 222.6 s.
  - Prefill is 60 vs 94 ms, and e2e is 2314 vs 2438 ms.
  - Native's steps are longer (25.3 vs 23.4 ms).
- The mechanism is KV-blocked FIFO admission, in both arms.
  - vLLM breaks out of its waiting loop when `allocate_slots` fails, so a
    request is only looked up once it reaches the head.
  - From finish to the first server query is p50 28.6 steps. The first to
    the last query takes 1.6 steps, and the last query to first_step
    about 0.
  - Requests with a gap of 30 or more average 8.6 queries (max 54) and
    need about 4600 tokens, against about 2700 for the rest.
- v2 has 43 order inversions; native has 0. All 43 are sync loads
  (16 requests, ext ≤ 256) overtaking async loads. This is a probe
  artefact: an async load's first_step includes the load, about 7 steps.
- The one connector-side lever is the hold after finish.
  - v2's `request_finished` returns True whenever the request has a
    planned save. vLLM then keeps the blocks until finished_sending.
  - Native returns False and fences block reuse instead.
  - 86% of requests made their last save at least 3 steps before they
    finished, so their hold is a pure round trip.
  - A KV-capacity model gives 24.4 blocked steps at hold 3, 20.8 at hold
    0, and 24.8 for native.
  - The measured dose-response is weak: hold 108 -> 67 ms moved queue
    p50 from 34/32 to 30/32.
  - Candidate it5: return False at finish when the workers' reported
    saved_through is at or past the request's last save step.

**Turn-0 admission spread: steps 2-12 for v2 vs 2-4 for native**

- v2's 31 server queries all land within about 19 ms, and each is a full
  hit. The spread is in load completion.
- Attribution: server. The load worker runs serially, one task per RPC,
  and syncs after each task. H2D is 14.8 GB/s, against about 22 GB/s for
  native. Recorded under PROGRESS "Server limitations". One load RPC per
  step stays parked (one bad lease fails the whole call).

**Main's decision (2026-10-26)**

- Target (a), save-path ITL, comes first. Attribute it offline before
  writing code: save D2H, load H2D, and saver-thread/CPU contention, and
  whether the burst of saves after the TP2 hold drives the 25-50 ms ITL
  tail.
- it5 (release at finish when the saves are done) is queued next.
- The Qwen3.5 pair (c8a8842 vs 02b6767, GPU5, replay + gsm8k) is running:
  `qwen3.5-4b-p5-it4-pair-{c8a8842,02b6767}`.

### Qwen3.5-4B pair: c8a8842 vs 02b6767 (GPU5, TP1, native in each run)

Runs: `2026-10-26-qwen3.5-4b-p5-it4-pair-{c8a8842,02b6767}`. Both exit 1,
because there are no plain/baseline arms.

| rev | arm | gsm8k cold acc / s | gsm8k warm acc / s | replay warm2/3 TTFT p50 | mean | req/s | ms/step |
|---|---|---|---|---|---|---|---|
| c8a8842 | native | 0.7824 / 97.40 | 0.7885 / 98.74 | 0.210 / 0.204 | 0.350 / 0.351 | 10.77 / 10.80 | 26.2 / 26.0 |
| c8a8842 | v2 | 0.7877 / 94.74 | 0.7885 / 94.29 | 0.285 / 0.293 | 0.295 / 0.293 | 14.73 / 14.68 | 21.0 / 21.1 |
| 02b6767 | native | 0.7824 / 97.59 | 0.7824 / 99.03 | 0.170 / 0.195 | 0.281 / 0.338 | 13.13 / 11.61 | 20.8 / 24.2 |
| 02b6767 | v2 | 0.7824 / 94.62 | 0.7824 / 94.22 | 0.284 / 0.295 | 0.297 / 0.292 | 14.71 / 14.64 | 21.0 / 21.1 |

- **gsm8k.** Within 2% of native: pass on both revs. v2 warm is 4.5% and
  4.9% faster than native.
- **02b6767 vs c8a8842.** Neutral. v2 changes by less than 0.1%. The
  deferral never fires here: hit stats show no `deferred`.
- **Native noise.** Native varies across runs (02b6767 run: warm2 19.19 s
  vs warm3 21.71 s), so the G4 percentages move:
  - c8a8842: TTFT p50 +39.4% (behind), req/s +36.3% (ahead).
  - 02b6767: TTFT p50 +58.6% (behind), req/s +18.6% (ahead).
- **TTFT p50.** The gap is structural. The hybrid layout has 4 KV groups,
  so loads are never sync and every external hit pays the async round
  trip. v2's mean and p90 match or beat native.
- **Hit rates** (G3 warm, both revs, both arms):
  - replay: hit 0.9462, local 0.7776, ext 0.7582, oracle 0.9462, E 1.0.
  - gsm8k: hit 0.9255, oracle 0.9255, E 1.0. Local is 0.9248 and ext
    0.0093, except c8a8842 v2 at 0.9192 local and 0.0782 ext.
- **Turn 0** (32 conversations, 57407 prompt tokens): identical in every
  arm and rev. Cold is local 31680, ext 0; warm2/3 is local 0, ext 50688.
- **02b6767 v2 replay hit stats:** lookups 1011, lost_tokens 1056,
  `missing=1,2,3` 1.

### Target (a) attribution: TP2 warm step time (offline, TP2 pair 02b6767)

- **The gap.**
  - v2 warm runs at 17.13/17.27 ms/step against native's 16.08/16.21. That
    is +1.05 ms/step, about 0.86 s per pass.
  - Cold is 35.4 vs 34.4.
  - At TP1, v2 leads in both cold (37.5 vs 41.2) and warm (23.4 vs 25.3).
  - Warm ITL share in 25-50 ms is 7.4/6.3% for v2 against 3.4/5.0% for
    native; c8a8842 had 16.7/18.3%.
- **Saves are not the cause.**
  - Saver submit costs about 0.05 ms/step.
  - Server RPC totals per warm pass are load 0.10 s / 528, query
    0.07 s / 264, and save 0.07 s / 52 (0.1 GB).
  - The 258/274 ms hold at TP2 covers near-empty jobs.
- **Sync-load stall** (serial FIFO loader model, s per pass, warm2/warm3):
  - c8a8842: 0.15/0.18.
  - 02b6767: 0.66/1.32, of which 0.22/0.35 is behind the step's own async
    loads and 0.34/0.82 behind older ones.
  - TP1: 0.34.
  - At TP2, 84 of 264 warm loads per pass are sync.
- **Loader-busy overlap.** Each ms the async loader is busy costs about
  0.4-0.7 ms of step time: about 0.85 ms/step at TP2 and 2.6 ms/step at
  TP1. This is a server limit (cross-context copies; PROGRESS "Server
  limitations").

H5 model, stall in s per pass (warm2/warm3, 02b6767):

| variant | stall | demoted |
|---|---|---|
| base | 0.71 / 1.37 | 0 |
| part 1: sync loads first within a step | 0.51 / 1.17 | 0 |
| part 1 + demote after an async load in the previous step | 0.30 / 0.19 | 25/84, 31/84 |
| part 1 + defer one step | 0.55 / 1.54 | no help |

On TP1 e985938, part 1 gives 0.29 s and part 1 + demote gives 0.05 s, with
7 of 16 loads demoted.

### H5 (b8501b2): sync loads first, demote after an async load

- **Main's conditions** (2026-10-26):
  - Demotion costs an async round trip, about 7.6 steps on short-prefix
    workloads, so the gate includes TP1 gsm8k warm (≤2% vs native, hard).
  - Report the two parts separately. If demotion hurts gsm8k, ship part 1
    only and record part 2 as rejected.
- **Part 1, worker.** A step's sync load RPCs are sent before its async
  ones.
- **Part 2, scheduler.** A hit of at most `sync_load_tokens` loads async
  when the previous step planned an async load. The current step's async
  loads do not count, because part 1 sends them after the sync loads.
- **Counters.**
  - Save stats line: `sync_loads`, `sync_first_steps`, `sync_wait_ms`,
    `max_sync_wait_ms`.
  - Hit stats: `sync_demoted`.
  - Per demoted load, `demoted sync load landed ... after_ms steps`.
- **Part-1-only arm.** 7bc785a on `exp/rsi-qwen35-h5-part1`.
- **Gate** (queued behind the Qwen3.5 pair):
  - TP2 replay, `qwen3-4b-p5-h5-tp2-{02b6767,7bc785a,b8501b2}`.
  - Then TP1 replay + full gsm8k, `qwen3-4b-p5-h5-tp1-{...}`.
  - Each run includes native.

### H5 review: REJECT as committed, fixed in daa5e11 (2026-10-26)

- **Review of b8501b2.** Part 1 (sync loads first) was accepted. Part 2 was
  rejected:
  - every async load re-armed the step window, demoted ones included, so
    one long load followed by one short hit per step demoted 6 of 6 (M1);
  - one step is too short a window for a copy that can take several (M2).
- **Fix (daa5e11, new commit).**
  - A short hit is demoted only while a long, non-demoted async load from
    an earlier step is in flight, until vLLM reports it done or the request
    finishes. There is no step counting.
  - The demotion sits behind the `async_demote` knob, off by default and
    pending gate evidence (5/12 knobs).
  - The per-request INFO line became hit stats: `sync_planned`,
    `sync_demoted`, `max_demoted_run`, `demoted_done`, `demoted_ms` /
    `demoted_steps` (sum and max), and `deferred_demoted(_ms)` for sharers
    deferred behind a demoted load.
  - New save stats: `load_steps`, `load_issue_ms` and `max_load_issue_ms`,
    covering all of `start_load_kv`.
  - All 15 mutants are killed, the review's 6 included. New cases: the M1
    cascade, a failed demoted load, an abort, a preempted reload, failed
    plans, and a deferred sharer.
- **Gate arms (S1).**
  - `exp/rsi-qwen35-h5-stats` cf67695: stats only, plan order.
  - daa5e11: plus ordering (part 1).
  - `exp/rsi-qwen35-h5-demote` 0f63123: plus `async_demote = true`, held
    until the re-review passes.
  - The b8501b2 gate jobs were dropped.

### H5 TP2 pair: 02b6767 vs 7bc785a (part 1), Qwen3-4B TP2, native in each run

The runs are `2026-10-26-qwen3-4b-p5-h5-tp2-{02b6767,7bc785a}`. They come
from separate sessions, so they are compared only through ratios to their
own native.

| warm pass | 02b6767 v2 / native | 7bc785a v2 / native |
|---|---|---|
| TTFT p50 warm2 | 0.1454 / 0.1776 = 0.82 | 0.1498 / 0.1610 = 0.93 |
| TTFT p50 warm3 | 0.1509 / 0.1635 = 0.92 | 0.1554 / 0.1706 = 0.91 |
| req/s warm2 | 18.70 / 18.89 = 0.990 | 18.66 / 19.35 = 0.964 |
| req/s warm3 | 18.70 / 19.25 = 0.972 | 18.55 / 19.16 = 0.968 |
| gate G4 TTFT p50 (median) | -13.1% | -8.0% |
| gate G4 req/s (median) | -1.9% | -3.4% |
| ms/step warm | 20.11 / 18.87 (+1.24) | 19.87 / 18.90 (+0.97) |
| ms/step warm2 | 17.18 / 16.39 (+0.79) | 17.28 / 15.97 (+1.31) |
| ms/step warm3 | 17.25 / 16.13 (+1.12) | 17.23 / 16.13 (+1.10) |
| ITL 25-50 ms share, warm2/3 | 8.2/7.4% vs 5.1/4.6% | 9.4/8.0% vs 3.7/4.4% |

Hit rates are identical in both runs: v2 0.9983, local 0.8641, ext 0.9872;
native 0.9984.

The sync stall measured with part 1 (7bc785a, tp0 save-stats windows,
apportioned to passes by time):

| pass | sync loads | reordered steps | sync_wait | per step | max one step |
|---|---|---|---|---|---|
| warm | ~67 | ~18 | ~188 ms | 0.29 ms | 83 ms |
| warm2 | ~78 | ~13 | ~115 ms | 0.17 ms | 114 ms |
| warm3 | ~50 | ~8 | ~213 ms | 0.39 ms | 114 ms |

- **Part 1 did not measurably move step time.** The warm2/3 excess over
  native is +0.96 ms with 02b6767 and +1.21 ms with 7bc785a. The
  run-to-run spread of the same arm is ±0.3 ms, so this is noise. The
  req/s ratio stays 3-4% behind.
- **What is left of the 1.05 ms.**
  - The sync wait is now measured at 0.17-0.39 ms/step, at most about a
    third of the gap. The serial-loader model overestimated it (0.43/0.69
    s per pass for part 1, measured ~0.12/0.21 s).
  - The wait is concentrated in a few steps: the max per step is 114 ms,
    from a sync load queued behind long loads of earlier steps. That is
    what demotion targets; part 1 cannot reach it.
  - The remaining ~0.7-0.9 ms/step is not the sync wait. It matches the
    loader-busy overlap (cross-context copies, 0.4-0.7 ms of step per ms
    busy; PROGRESS "Server limitations"). That is a server limit, plus run
    noise.
- **02b6767 has no sync counters,** so a before/after sync_wait is
  impossible here. The TP1 pair cf67695 (stats) vs daa5e11 (part 1) gives
  it in one session.

### Qwen3.5-4B TTFT p50 attribution (offline, both it4 pair runs)

**Independent review verdict: PARTLY (2026-10-27).** The p50 gap is not
closed as "vLLM batching only". The reviewer's scratch scripts are in
`/tmp/q35rev` on the Mac.

What holds:
- The convoys are real: 3.8-4.1 arrivals per step against native's
  1.25-1.33.
- Native's turn-0 async lookups (about 1.1 s each) spread out its starts
  and drive its p99.
- Loads, bytes and tokens are equal in both arms.

What was refuted or not shown (corrections to items 4 and the split table
below):
- "A lone v2 request matches native": v2 has only 22-25 lone requests per
  pass. My own interpolation gives 193-210 ms for them, not 160-167 ms.
- The convoy table is non-monotone at k=7+ because the gaps between log
  anchors reach 125 steps.
- "At equal convoy size v2 matches native": refuted. At k=2-3 v2 is
  45-75 ms worse, and its window prefill is about 3× larger.

Connector-side residuals the reviewer found:
- **L1. Scheduler lookup cost.** Lookups in the same step are spaced
  1.0-1.2 ms apart, while the server RPC takes 0.016 ms. That is about
  1.1 ms of Python per lookup: 4 per-group `query_prefetch` RPCs plus
  `semantics.resolve`. It costs 42-45 ms on the 32-request turn-0 step and
  4-7 ms per convoy step.
- **L2. One extra step to readiness.** Ready loads reach first_step 4 steps
  after arrival in warm2/3, but some cold/warm requests take 3. I checked
  the claim against vLLM 0.32: in
  `vllm/v1/worker/kv_connector_model_runner_mixin.py:83-97`, a step without
  sync loads calls `start_load_kv` in the `finally` block after the forward
  launch, then calls `get_transfer_results` right away. That first poll
  comes microseconds after the issue, so a load that takes about 4 ms to
  copy cannot be ready for it. The a31c073 probe (`async_first_poll`)
  measures this directly.

Plan, one hypothesis at a time, after the H5 queue finishes:
1. Next Qwen3.5 run (a31c073 plus probe additions):
   - per-load poll index at which it was ready;
   - worker issue-to-ready ms;
   - scheduler `_lookup` ms per request.
2. **H6 = L1:** a cheaper lookup within the existing server API, either
   fewer RPCs per lookup or less Python per lookup. A multi-group query RPC
   would be a server change: if it is the only way, record it as a server
   limit.
3. **H7 = L2:** only if the probe shows the first poll is systematically
   too late and there is a connector-side fix. Otherwise attribute it to
   vLLM's structure (the source line above).

The original analysis follows unchanged. Items 4, the split table and
"Next connector-side hypothesis" are superseded by the verdict above.

Main rejected "the p50 gap is structural": native's loads are async too,
yet native reaches p50 0.17-0.21 s against v2's 0.284-0.295 s. Data:
`2026-10-26-qwen3.5-4b-p5-it4-pair-{02b6767,c8a8842}`, warm2/warm3, probe
steps, server query timestamps, vLLM histograms. Scripts are in
`~/kexi/tmp/p5-it3-attr/`: q35rt, q35time, q35hist2, q35wave, q35wave2,
q35seg.

1. **Round trip in steps.** v2 needs as many steps as native or fewer, in
   every class:

   | class (warm2, 02b6767) | n | native steps (gap+pre) | v2 steps | v2 - native (paired) |
   |---|---|---|---|---|
   | t0 (ext 1584) | 32 | 66 p50 | 4 | -58 |
   | t>0, ext=0 | 67 | 3 | 3 | 0 |
   | t>0, ext ≤528 | 81 | 7 | 6 | -1 |
   | t>0, ext ≤1056 | 40 | 7 | 6 | -1 |
   | t>0, ext >1056 | 32 | 7 | 6 | -1 |

   The other pass and the c8a8842 run give the same within ±0.1 steps.
   Async loads cost 3 extra steps in v2 (lookup and issue, the one-step-late
   done report, promotion) and 4 in native. The step-count gap is in v2's
   favour.
2. **Load size and tasks per request.** They are the same. Each pass has
   185 load RPCs, one per external hit (153 t>0 + 32 t0). One RPC covers
   all 4 groups. Native also has 185 loads.

   | per warm pass | v2 | native |
   |---|---|---|
   | bytes loaded | 16.42 GB | 16.35 GB |
   | copy time | 0.744 s, 4.0 ms/load | 0.785 s, 4.2 ms/load |
   | load RPC time | 14 ms total | - |
   | queries | 740 query RPCs (4 groups × 185), 12 ms total | 31 async lookups, 32.5-38.7 s total |

   The 4-group layout does not multiply v2's per-load cost: it has one
   server task per request, and the per-group queries cost 0.016 ms each.
   Native's 31 async lookups are all turn 0, about 1.1 s each, and they are
   native's slow t0 tail.
3. **Where the distributions split: convoys.** v2 starts all 32
   conversations within 3-5 steps of the pass start. Native spreads its
   starts over 48-65 steps (p50; p90 84-112), because its t0 lookups are
   slow. Every turn has the same output length (64 tokens, ignore_eos), so
   the start phase carries through the whole pass:

   | warm2 / warm3 (02b6767) | native | v2 |
   |---|---|---|
   | t>0 arrivals in the same step, mean | 1.2 / 1.3 | 4.1 / 3.8 |
   | t>0 arrivals in the same step, p90 | 2 | 8 / 7 |
   | prefill tokens in a request's first step, p50 | 330 / 349 | 915 / 795 |
   | steps with arrivals | 195 / 189 | 82 / 90 |
   | max finishes in one step | 3 | 9 |

   In the cold pass both arms start staggered (cold t0 is computed), and v2
   arrivals per step are 1.6 vs 1.4. The convoy comes from v2's fast warm
   t0.
4. **Wall cost of the waited steps.** A least-squares fit of v2 step wall
   time on the query-timestamp anchors (R² 1.00) gives about 11.4 ms/step,
   plus 28-38 ms per 1k prefill tokens, plus 0.31 ms per running request.
   Issued and landed loads come out at +6-7 and -2-4 ms per load, which
   cancel over a round trip. An ext>0 request in v2 waits through steps
   that cost about 42-45 ms each, against a pass average of 21 ms. Five
   steps at about 44 ms gives the measured ~210 ms from query to first
   token.

   v2 wall TTFT (query to first token) by convoy size:

   | convoy size | 02b6767 warm2 / warm3 | c8a8842 warm2 / warm3 |
   |---|---|---|
   | 1 | 164 / 160 ms | 167 / 159 ms |
   | 2-3 | 216 / 209 | 219 / 221 |
   | 4-6 | 218 / 232 | 214 / 231 |
   | 7+ | 201 / 216 | 190 / 164 |

   A lone v2 request lands at 160-167 ms. That is native's t>0 ext>0 level:
   146-170 ms (6 steps × native ms/step).

**Split of the p50 gap** (about +0.09-0.12 s client side):

| share | source | why |
|---|---|---|
| ~0 | connector | v2 is 1 step faster per external hit and about 15× faster on warm t0. Load bytes, RPC count and per-load copy time match native. |
| ~0 | server | 4 ms/load and 0.016 ms/query. The 4-group layout adds queries, not load tasks. |
| ~all | vLLM-native batching under workload phase lock | Fast warm t0 phase-locks the 32 conversations, which have equal output lengths. Their turns arrive in convoys of about 4 (up to 8), each step prefills about 3× the tokens, and those steps cost about 2× the pass mean. Native's slow t0 (about 1.1 s per async lookup) staggers the conversations, so its t>0 requests prefill alone in cheap steps. Native pays for that in its tail: t0 TTFT about 1.2-1.4 s, p99 2.3 s vs v2's 0.585 s. |
| ±0.02-0.04 s | noise | Native p50 is 0.170-0.210 across the two runs with the same workload. |

The p50 metric rewards the staggered schedule. Mean TTFT is close
(0.297 vs 0.281 and 0.295 vs 0.350), p99 is 4× better for v2, and req/s
is +12-36%.

**Next connector-side hypothesis: none proposed.**
- At equal convoy size v2 already matches native.
- The 3-step async round trip is vLLM's protocol (async scheduling reports
  the load one step late). The only connector lever that would shorten it
  is sync loads for multi-group layouts, and lifting the one-group limit
  is ruled out.
- Delaying t0 or staggering requests to break convoys would game the
  metric, so I do not propose it.
- One open diagnostic, for when a GPU slot is free: whether the first
  `poll()` after the issuing step already sees the load ready (4 ms copy
  vs a ~40 ms step). If it does, the round trip is at vLLM's minimum and
  this item closes.

### H5 gate result: `async_demote` deleted (2b, 2ab3ae9)

Runs `2026-10-26-qwen3-4b-p5-h5-{tp1-cf67695,tp1-daa5e11,tp1-8df70d5,tp2-daa5e11,tp2-8df70d5}`.
Each has its own native arm. The arms:

- cf67695: send in plan order (stats arm);
- daa5e11: sync loads first, knob off;
- 8df70d5: sync loads first, `async_demote = true`.

Base server. Script `~/kexi/tmp/p5-it3-attr/h5rep.py`. Each run's G4
failures repeat in every arm and in native (same verdict set), so they come
from the session, not the arm.

**TP1** (warm2 / warm3; ms/step from the probe):

| arm | v2 p50 | native p50 | v2/native | v2 rps | native rps | v2 ms/step | native ms/step |
|---|---|---|---|---|---|---|---|
| cf67695 | 0.629 / 0.598 | 0.488 / 0.486 | 1.29 / 1.23 | 9.73 / 9.76 | 9.25 / 9.17 | 23.5 | 25.6-25.8 |
| daa5e11 | 0.570 / 0.657 | 0.495 / 0.505 | 1.15 / 1.30 | 9.84 / 9.74 | 9.22 / 9.16 | 23.4-23.5 | 25.7-25.8 |
| 8df70d5 | 0.555 / 0.596 | 0.505 / 0.492 | 1.10 / 1.21 | 9.88 / 9.83 | 9.26 / 9.24 | 23.4 | 25.5-25.6 |

TP1 gsm8k warm, all 0.8552 = native (0.8552): daa5e11 171.1 s vs 175.6 s
native; 8df70d5 171.3 s vs 183.0 s. The ≤2% hard bar holds in every arm.

Per-pass v2 worker and scheduler counters at TP1 (warm / warm2 / warm3):

| arm | sync_wait_ms (max window) | load_issue ms/step (max) | sync_planned | sync_demoted |
|---|---|---|---|---|
| cf67695 | 127 / 135 / 141 (max 88) | 3.31 / 2.42 / 2.73 (94) | 16 / 19 / 17 | - |
| daa5e11 | 77 / 42 / 172 (max 63) | 2.76 / 1.86 / 2.40 (64) | 17 / 16 / 18 | - |
| 8df70d5 | 4 / 7 / 6 (max 3) | 1.99 / 1.24 / 1.36 (42) | 8 / 6 / 7 | 10 / 10 / 8 |

TP1 8df70d5 demotion detail (per pass):

| metric | warm / warm2 / warm3 |
|---|---|
| demoted share `sync_demoted/(sync_planned+sync_demoted)` | 0.56 / 0.62 / 0.53 |
| `max_demoted_run` | 2 / 2 / 3 |
| `demoted_ms/demoted_done` (max) | 111 / 125 / 123 ms (max 264) |
| `demoted_steps/demoted_done` (max) | 2.4 / 5.2 / 5.3 (max 8) |
| `deferred_demoted` | 0 in every pass |

vLLM reports a load done one step late, so `demoted_steps` includes about
one step of lag.

**TP2** (warm2 / warm3):

| arm | v2 p50 | native p50 | v2/native | v2 mean | native mean | v2 rps | native rps | v2 ms/step | native ms/step |
|---|---|---|---|---|---|---|---|---|---|
| daa5e11 | 0.142 / 0.149 | 0.167 / 0.157 | 0.85 / 0.95 | 0.135 / 0.148 | 0.169 / 0.165 | 18.96 / 18.84 | 19.12 / 19.30 | 17.0 | 16.1 |
| 8df70d5 | 0.152 / 0.146 | 0.170 / 0.159 | 0.90 / 0.92 | 0.155 / 0.152 | 0.168 / 0.164 | 18.57 / 18.24 | 19.32 / 19.23 | 17.0-17.1 | 16.0-16.2 |

TP2 counters (warm / warm2 / warm3):

| arm | sync_wait_ms (max window) | load_issue ms/step (max) | sync_planned | sync_demoted |
|---|---|---|---|---|
| daa5e11 | 161 / 56 / 99 (max 68) | 4.22 / 2.24 / 3.18 (71) | 84 / 84 / 84 | - |
| 8df70d5 | 15 / 23 / 26 (max 3) | 2.23 / 1.80 / 1.59 (14) | 47 / 40 / 46 | 37 / 44 / 38 |

TP2 8df70d5 demotion detail (per pass):

| metric | warm / warm2 / warm3 |
|---|---|
| demoted share | 0.44 / 0.52 / 0.45 |
| `max_demoted_run` | 3 |
| `demoted_ms/demoted_done` | 111 / 72 / 84 ms (max 200-214) |
| `demoted_steps/demoted_done` | 3.1 / 3.5 / 4.5 (max 6-14) |
| `deferred_demoted` | 0 |

Probe gaps (arrival to first scheduled step) for sync-size hits, v2 vs
native: TP2 daa5e11 1 vs 5-6 steps; TP2 8df70d5 3 vs 5 steps. A sync short
hit loads in 1 step at about 17 ms, so it has TTFT about 1 step. A demoted
one waits 72-125 ms on average (4-7 steps).

**Decision: delete `async_demote` (2ab3ae9).**
- Demotion does remove the stall it targets: sync_wait_ms falls from 42-172
  to 4-26 per pass, and the max window from 37-68 ms to 3 ms. It also cuts
  load_issue per step.
- But the stall was small to begin with: at most about 0.25 ms/step, and
  0.05-0.16 ms/step at TP1. Each demoted hit pays a 4-7 step async round
  trip instead.
- At TP2 sync-size hits get 2 steps slower (gap 1 to 3). Mean TTFT gets
  worse (0.135/0.148 to 0.155/0.152), and so does rps relative to native
  (0.99/0.98 to 0.96/0.95).
- At TP1 p50 and rps move within the run-to-run spread of the two
  knob-off arms (v2/native 1.15-1.30). Step time and gsm8k are unchanged.
- No net gain, so per main's rule the knob is deleted, not left off. Gone:
  the policy field, its toml entry and README text, `_long_loads` /
  `_demoted` / `_deferred_demoted`, `_behind_demoted`, the demote
  counters, and the demotion tests. Kept: part 1 (sync loads first) and
  `sync_planned`. CI on a git-archive export: typos, ruff (python, rsi),
  core tests 279 passed, source-only unit gate 531 passed, gate pytest 390
  passed, cz.
- Part 1 has no measurable effect either: cf67695 sends in plan order, and
  only 1-4 steps per pass would have been reordered. It is kept because it
  costs nothing and only matters when a sync load lands behind an async one
  in the same step.

### Qwen3-4B TP1 TTFT p50 gap: convoy breakdown (offline, the three H5 TP1 runs)

Question: does TP1's 15-30% p50 gap (v2 0.56-0.66 vs native 0.49-0.51 at
equal or better rps) come from the same phase lock as Qwen3.5? **No: the
phase lock is reversed.** Scripts: `tp1rt.py`, `q35wave.py`, `tp1hold.py`,
`tp1pair.py`, `tp1q.py`, `tp1t0.py`, `tp1turn.py`, `tp1tc.py`, `tp1occ.py`,
`tp1lk2.py`, `tp1ahead.py` in `~/kexi/tmp/p5-it3-attr/`.

1. **Arrivals per step.** Native has the convoys:

   | per pass, t>0 | native | v2 |
   |---|---|---|
   | arrivals in the same step, mean | 3.8-5.1 | 1.7-2.1 |
   | arrivals in the same step, p90 | 10-18 | 4-5 |
   | steps with arrivals | 114-119 | 178-181 |
   | max finishes in one step | 13-18 | 5-10 |
   | requests arriving alone | 61-82 | 141-155 |

   The source is t0. All TP1 t0 requests share 79 blocks (1264 tokens). The
   first one loads the shared prefix, so native starts its 32 t0 requests
   within 4-5 steps. v2 spreads them over 12 steps: the base server's
   serial load worker plus the in-flight deferral (t0 gap 8 vs 2-4 steps,
   a known server limit). Turn-1 convoys are 10-11 (native) vs 3.2-3.4
   (v2) and decay by turn.
2. **TTFT by convoy size** (TTFT steps, p50):
   - Mixed. Convoy 1: native 17-64 vs v2 30-46. Convoy 2-3: v2 better in
     5 of 6 passes. Convoy 4-7 and 8+: similar.
   - Splitting by turn band shows where the gap is: in turns 3-4. There v2
     is slower at equal convoy size:

     | turns 3-4 | native p50 | v2 p50 |
     |---|---|---|
     | alone (c1) | 8-10 (one pass 67) | 21-25 |
     | c2-3 | 10-12 | 20-66 |
     | c4+ | 6-8 | 17-54 |

   - Turns 1-2 (v2 4-5 vs native 5-8 steps) and turns 5+ (v2 50-60 vs
     native 60-71) favour v2.
3. **TTFT by turn** (p50 steps, native vs v2, every pass of the three
   runs):

   | turns | native | v2 |
   |---|---|---|
   | t1 | 5-8 | 4-5 |
   | t2 | 4-6 | 5-6 |
   | t3 | 7-8 | 18-22 |
   | t4 | 10-11 | 22-52 |
   | t5-t7 | 65-71 | 65-74 |
   | t8 | 64-67 | 38-56 |

   The p50 is set by t3-t4 (110 of 264 requests). In every v2 pass all 55
   t3-4 requests wait more than 10 steps. Native admits 32-35 of them
   within 10 steps.
4. **Load round trip by class** (gap in steps, warm2/3, all three runs):

   | class | native | v2 | note |
   |---|---|---|---|
   | sync-size ext (15-18 req) | 5 | 1 (daa5e11, cf67695); 4 with demotion | |
   | long ext, alloc to first compute | - | about 0-3 | v2 load path itself is fast |
   | long ext, total | p50 12-14, mean 34 | p50 25-31, mean 36-38 | |
   | t0 | 2-4 | 8-9 | server limit |

   The long-ext v2 wait is almost entirely before the first server query:
   about 21-24 steps for t3-4. The "blocked retries" (first to last query)
   take about 1.5 steps, and allocation to first compute about 0. (Query
   times are mapped to steps by interpolating between sync-load anchors,
   about 44 per run, so this split is approximate: some p50s come out
   slightly negative.) vLLM's
   KV-blocked FIFO holds the request: the head of the waiting queue fails
   `allocate_slots`, so the loop breaks (`scheduler.py` line ~1283) and
   requests behind it are not looked up.
   - Native copy throughput: 20.6 ms/load, 22.3 GB/s. v2: 31.0 ms/load,
     14.7 GB/s (pegaflow_load_duration). This is the server's
     cross-context copy contention. It is off the critical path here,
     since v2 loads land within about 1-2 steps of allocation.
5. **Why t3-4 queue longer in v2.** At t3-4 arrival both arms have about
   11-14 requests ahead in the FIFO. Native's slow t3-4 requests see about
   57k need-tokens ahead, v2's about 40-45k. Native's are mostly t3-t4 from
   the same convoy, and the convoy moves together. v2's arrivals are
   staggered 1-2 per step, so each one waits behind the previous ones'
   allocations one at a time. Occupancy computed from the probe (prompt +
   generated tokens of running requests, 71,472-token pool):
   - v2's slow t3-4 requests see about 18-22k free tokens at arrival.
   - Native's fast ones see 20-27k.
   - Native's slow ones see 28-35k (blocked behind a big head-of-line
     request).
   - Need p50 is 3.6-3.9k in both.

   v2 also holds finished requests' blocks until finished_sending (save
   batching), with sent_wait up to 30 s over 146 requests per window, max
   1.06 s. That block-hold is not visible in probe occupancy, which counts
   only running requests. vLLM's logged KV usage is 97-100% in both arms
   at the 10 s samples.

**Answer.** Not the Qwen3.5 phase lock. TP1's phase lock helps native: the
fast shared-prefix t0 makes native's convoys, and those move through vLLM's
FIFO as a block. v2's staggered arrivals queue one by one behind KV-blocked
heads. At equal convoy size in turns 3-4, v2 is still 2-3× slower (21-25 vs
8-12 steps alone), and v2's own load round trip is not the cause.

**Connector headroom, next hypothesis (it5).** v2 keeps a finished
request's blocks until its last batched save is sent and done. Native frees
blocks at finish: its `request_finished` returns False, and its stores are
fenced through `_block_id_to_pending_jobs` (deferred fences on block
reuse). That leaves fewer free blocks for the FIFO head in v2.
- Candidate: release at finish when saved_through ≥ the request's last
  save step. Measure the per-request hold (finish to finished_sending in
  steps).
- Expected signal: t3-4 TTFT drops toward native's 8-12 steps, with
  sent_wait_ms falling.
- The t0 spread (server serial load worker) stays a server limitation.

## PROGRESS log entries (migrated from `feat/rsi-qwen3.5-4b` 4bb7d8d, 2026-10-26)

The P5 bullets of the code branch's `PROGRESS.md` log, moved here
when the journals left the code branches (dates are the log sections they
were under: 2026-10-23 for it0/it1, 2026-10-25 for it2, 2026-10-26 for the
review fixes and merges).

- P5 it0 (Qwen3.5-4B, `feat/rsi-qwen3.5-4b`):
  - Spec dump: 3 `MambaSpec` (GDN, align) groups + 1 `FullAttentionSpec`,
    block 528, aliased 2162688 B pages. Self-test stopped at the honest
    arm's `NotImplementedError` for MambaSpec; the probe digested group 3
    only. Gate items 1-4 accepted, built on `feat/rsi-gate-hybrid` after
    feat/rsi-g5 (940def4 already fails G2 on undigested groups).
  - Mapping: GDN = `Kind::Checkpoint(528)`, attention = `Full(528)`. No new
    Kind; no Semantics changelog entry.
  - Adapter draft `adapters/mamba.py` + stub-spec tests. `LayerDecl.layout`
    (namespace-only) added for the packed state shapes/dtypes; empty for
    existing adapters, so Qwen3-4B's namespace is unchanged.
- P5 it1 (Qwen3.5-4B, `feat/rsi-qwen3.5-4b`):
  - Core: checkpoint groups save from vLLM's `boundary_state_offloads`
    (only the boundary `saveable` picks, only if offered); layouts with a
    checkpoint group save synchronously; loads unchanged (the state at p is
    always past the local hit). `hit stats` log: per-group miss attribution.
  - `--prefix-cache-retention-interval 528` requested from the gate owner
    for every arm; not set by the connector.
  - Ordering check on GPU4 (temporary tool `~/kexi/rsi-tools/p5`, not in the
    repo): all checkpoint saves from offered blocks, saved bytes equal an
    unprimed recompute up to rounding, every load byte-equal to the saved
    page after poisoning the GPU cache, outputs equal (order2: 6/6; order3
    with 6 preemptions: 5/6; p4 diverges at token 152 before its own
    preemption). Rerun at 863cc99 as order4 (review should-fix 2): 27/27
    checkpoint and 29/29 full loads byte-equal, p4 flips at token 152 by
    one bf16 step and matches the unprimed alone run.
  - it2 leg 1 (`2026-10-24-qwen3.5-4b-p5-it2-ctrl`, plain vLLM, no
    connector, GPU4, pgid 2328944, 02:55:32Z-02:59:27Z): batched p4 flips
    at token 152 to the same token with the same 0.125 margin under the
    tight config (0.5 with 200 blocks), and p4 alone equals order4 C. A
    local-hit resume equals alone. Closed as GDN decode batch
    non-invariance in vLLM 0.32, not the connector. Gate implication
    (`batch_invariant` for this model) goes to the gate owner.
  - Non-regression on c4c1615: the Qwen3-4B slice
    (`2026-10-24-qwen3-4b-p5-it1-nonreg-b`, GPU4) passes every verdict, with
    the same deltas as P3 it2e. The DSV2 TP2 slice
    (`2026-10-23-dsv2-lite-p5-it1-nonreg-b`, GPU 4,5, 64gb, P4 55e4ff1 +
    P5 merged) matches P4's it2 slice: loads are byte-exact, the split-only
    failures are unchanged, and replay cold TTFT fails at +10.2%, the open
    P4 item. See ITERATIONS.md it1.
  - Two earlier non-regression launches were invalid. They lacked
    `PYTHONPATH=<tree>/python`, so vLLM imported the shared tree's
    connector through the venv `.pth`. They are discarded, and the skill
    has a lesson on it.
  - `LayerDecl.layout` is connector-side (python `adapters/__init__.py`),
    not in pegaflow-semantics and not in key derivation: no S-entry. The
    namespace V1 vector for Qwen3-4B is unchanged: stub `cf6b5725`, native
    `cd8acc0c` at 6545352; on P4's base (55e4ff1) `a25ea150` / `bdda8997`
    before and after the P5 changes.
- P5 it2 started (`feat/rsi-qwen3.5-4b`, merged `feat/semantics-rsi`
  f8eac30 = hybrid gate). Plan: (1) adapt connector_v2 to the hybrid gate
  if a slice shows it needs it (preempt pool, start-state digests before
  the forward, sync-load rules), `hit stats` logged unconditionally at
  shutdown; unit gate. (2) GPU5 slices (plain, baseline, old, native, v2;
  gsm8k + replay + cap + preempt), one hypothesis per iteration, SKILL
  step 5 checklist (checkpoint saves only at resume boundaries, no
  unresumable state, group-coherent resume, save cost off the step).
  (3) Tell the orchestrator before the official run (P4 merge first).
  Server stays at the base build (`pegaflow-semantics-rsi` 056eb4b).
- P5 it2 slice s1 (25fcccb, GPU5, not official): correctness all pass
  (digests in all 4 groups, preempt 106/106, alone_ref = baseline, E_ext
  1.0); fails G4 cold replay (TTFT p50 +27.5 %, req/s -6.2 %) and G5 cap
  (0 bytes loaded at every fraction). `hit stats` never reached vllm.log
  (the API server force-kills EngineCore, `shutdown` never runs): bc9c38a
  logs it when idle and every 30 s. Next: cap-only diagnostic slice
  (baseline + v2) to attribute the 0-load per group, then the save-cost
  hypothesis (checkpoint saves off the step, fenced on block reuse).
- P5 it2 s2cap (d2a484d, cap only, sub-floor pool C = 1.63 GiB = 0.05x
  GPU KV, no G5 conclusion): v2 loads 0 bytes; `hit stats` `missing=1,2,3`
  4 of 160 lookups (full group present, every GDN group evicted). Official
  `--cap-convs auto` sizing for Qwen3.5 reported to the orchestrator (est.
  U 147 GB at 203 convs; estimate/measured 0.78 at 8 x 6 suggests the
  measured floor misses at 203). Merged P4 (563a693) as 554ad9d.
- P5 it2 hypothesis H1 (next): G4 cold replay pays the sync checkpoint
  save (`flush_ms` 1518 / 33.5 s). Change: for layouts that must save
  synchronously (block zeroing / COW / checkpoint groups), the scheduler
  pins every block it plans to save in vLLM's block pool (`touch`) and
  unpins it once every worker reports the save done (worker metadata,
  min over ranks; `has_pending_push_work` keeps the engine stepping
  meanwhile). A pinned block is never reallocated, so it is never zeroed,
  COW-written or reused before the copy lands; the worker then saves
  asynchronously (no flush in `wait_for_save`, no preemption flush, no
  finish hold). Layouts that save async today are unchanged. Measure: v2
  replay cold TTFT p50 / req/s vs baseline, `save stats` flush_ms.
- P5 it2 s3 (5667cd7 = H1, not official): correctness all pass (digests
  in 4 groups, preempt 106/106 ext, alone_ref delta 0, no eviction).
  `flush_ms` 0; replay cold req/s -1.7% (pass), TTFT p50 +3.8% (fail;
  s1 +27.5%). gsm8k +5.3% cold/warm (fail) with 1 save step, so not from
  saves: 6046 prefix queries for 2703 lookups, each pass 4 group queries
  + lease releases for a hit that cannot beat vLLM's local 1056 tokens.
- P5 it2 hypothesis H2 (next): skip lookups that cannot beat vLLM's
  local hit. The bound is `semantics.resolve` over optimistic answers
  (unqueried groups count as fully present), checked before any query
  and after group 0. Full-only layouts skip only when no full block lies
  past the local hit. A skipped group creates no lease. No policy knob.
  Measure: gsm8k cold/warm s, replay cold TTFT p50 vs baseline (read
  replay cold p50 on the official-size run; the slice is noisy).
  Correctness unchanged.
- P5 it2 s4b (800630d = H2, not official; no cap): every v2 verdict that
  ran passes. gsm8k cold -0.1%, warm -0.7%; replay cold TTFT p50 +1.6%,
  req/s -1.3%. gsm8k prefix queries 6046 -> 23. Hits unchanged (replay
  warm 0.936 = H*). Merged `feat/semantics-rsi` 9a02fb5 (cap sizing) as
  be0622f. Next: official run on GPU5 (gsm8k, replay, cap, preempt).
- P5 it2 official (`2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`, c5d3f04,
  GPU5, base server, all 256 cap conversations, C(0.5) 2.52x GPU KV):
  official, and v2 passes every verdict. 0 digest mismatches in 4 groups
  (cap 59627/59627 external). G4: gsm8k +0.34% / -0.35%, replay cold TTFT
  p50 0.512 vs 1.204 s, warm 0.300 vs 1.112 s. Hits: v2 = H* unlimited;
  at capacity v2 = H_lru = old (within 0.002). E_C at 0.5 is v2 0.828,
  old 0.867, native 0.815. The v2-old gap is v2's own local hits above
  the baseline reference, which earn nothing by design; adding them back
  gives old's E_C. Judgment: close enough. Next: independent review.
- For the gate owner (P5 finding, gate code not edited): G5 E_C > 1 at
  C/U 0.25 on Qwen3.5-4B (H*_C 0.690 < H_lru 0.794, every arm) is the
  oracle, not the accounting. Belady's use sets are `needed(P)` at each
  request's unlimited P only. For a Checkpoint group, `needed(q < P)` is a
  different checkpoint, so fallback checkpoints look unused, and Belady
  refuses or evicts them (3368 not admitted vs 2562 stored per GDN group).
  LRU resumes there: 995 requests, 2.69M tokens (0.154 of prompt) at
  0.25, 239 / 0.52M at 0.5, 0 at 0.75. A Belady with uses at every block
  end <= P gives 0.841-0.843 / 0.889-0.891 / 0.941-0.946, above H_lru on
  every request, and E_C v2 / old / native 0.881 / 0.920 / 0.874 at 0.25.
  Suggest a fallback-aware use set for Kinds whose `needed` is not
  prefix-closed (or the max of both variants). Full-only models are
  unaffected. Details: `models/Qwen3.5-4B/ITERATIONS.md` "it2 official".
- P5 it2 review fixes (`reviews/P5-it2.md`, ACCEPT-WITH-FIXES):
  - S2: merged `feat/semantics-rsi` a4523f3 (lmcache gate) as 7247751.
    Offline re-judge of `2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04` with
    the merged gate: no verdict and no field changes for any arm,
    `g5_problems` empty, no `G2.resume_reference` (split-invariant).
  - S3: dead `Worker.sync_saves` deleted. Nits: own > ref wording and
    the `ref_cnt` / spec-decode caveat (ITERATIONS, README).
  - S1: H2's ITERATIONS text corrected. A skipped prefix query also skips
    the server's LRU refresh (`ReadCache::get_prefix_blocks` ->
    `refresh_recency`), so eviction order at capacity can move on
    full-only models. Regression slices, not launched (waiting for free
    GPUs), one at a time, at this branch's head:
    - export (Mac; gate tree and under-test tree are both the head, the
      native extension is unchanged since c5d3f04 and is copied from the
      base build):

      ```
      WT=/Users/a1234/workspace/pegaflow/.worktrees/rsi-qwen35
      REV=$(git -C $WT rev-parse --short=7 HEAD); FULL=$(git -C $WT rev-parse HEAD)
      for d in gate ut; do git -C $WT archive --format=tar HEAD | ssh -o ConnectTimeout=20 -i ~/.ssh/pegaflow_h200_kexi -p 2222 -l kexi@root@10.1.0.44 jumpcg.ppio.cloud "mkdir ~/kexi/p5-$d-$REV && tar xf - -C ~/kexi/p5-$d-$REV && echo $FULL > ~/kexi/p5-$d-$REV/.rsi-rev && cp ~/kexi/pegaflow-semantics-rsi/python/pegaflow/pegaflow.cpython-312-x86_64-linux-gnu.so ~/kexi/p5-$d-$REV/python/pegaflow/"; done
      ```

    - Qwen3-4B, TP1, GPU5 (remote; set `REV` there to the same short
      sha):

      ```
      cd ~/kexi/p5-gate-$REV && CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH LD_LIBRARY_PATH=~/kexi/cuda13-lib setsid nohup ~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate --model /root/kexi/models/Qwen3-4B --tag qwen3-4b-p5-s1-$REV --arms plain,baseline,old,pegaflow.connector_v2:PegaKVConnectorV2 --workloads gsm8k,cap --tp 1 --gpu 5 --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 256 --gpu-mem 0.8 --pool-size 64gb --concurrency 32 --cap-convs auto --cap-turns 12 --cap-fractions 0.25,0.5,0.75 --pegaflow-server ~/kexi/pegaflow-semantics-rsi/target/release/pegaflow-server --under-test-path ~/kexi/p5-ut-$REV/python --runs-root ~/kexi/rsi-runs > ~/kexi/rsi-runs/qwen3-4b-p5-s1-$REV.log 2>&1 < /dev/null &
      ```

    - DSV2, TP2, GPU4,5 (after the Qwen run; same command with
      `--model /root/kexi/models/DeepSeek-V2-Lite-Chat --tag
      dsv2-lite-p5-s1-$REV --tp 2 --gpu 4,5` and log
      `dsv2-lite-p5-s1-$REV.log`). The merged gate adds the resume stage
      for DSV2 (split-variant), and DSV2's gate.json now has
      `gsm8k_vllm_args` (merged from `feat/rsi-deepseek-v2-lite` fac8e82
      as 46814fc). So the DSV2 reference is its own official rerun on the
      merged gate (connector tree from its branch) once it lands, and it7
      only for G5 hit / E_C.
    - Not official by design (no replay). The other args match the
      accepted it7 runs (`2026-10-25-*-p4-it7-off-a59a254`), so cap auto
      gives the same 64 / 76 conversations. Compare v2's G4 gsm8k, G5
      hit and E_C per fraction, and v2 - old in the same run, with the
      "P4 acceptance tables". Pass = every v2 verdict that runs passes and
      no hit or E_C drop beyond old's run-to-run noise.
- P5: merged `feat/semantics-rsi` 1f00e5a (fallback-aware G5 oracle) as
  1002b80. Offline re-score of `2026-10-25-qwen3.5-4b-p5-it2-off-c5d3f04`
  (every cap stage's `capacity` recomputed from its probe): no verdict
  change for any arm, `g5_problems` empty. H*_C 0.690 / 0.878 / 0.946 ->
  0.841 / 0.889 / 0.946 (belady_fallback wins at 0.25 and 0.5). E_C v2 /
  old / native 0.881 / 0.920 / 0.874, 0.813 / 0.852 / 0.799, 0.880 /
  0.943 / 0.928. Judgment unchanged: close enough
  (`models/Qwen3.5-4B/ITERATIONS.md` "Re-score with the fallback-aware
  oracle").
- Audit note, not an S-entry (P5 it1): `LayerDecl.layout` (connector
  python, `adapters/__init__.py`) adds the Mamba state shapes/dtypes to the
  connector's namespace fingerprint, only when non-empty. pegaflow-semantics,
  `key(hash, group)` and `seal_plan` are untouched. V1 check: Qwen3-4B's
  namespace is unchanged (stub `cf6b5725`, native `cd8acc0c` at 6545352;
  `a25ea150` / `bdda8997` on P4's 55e4ff1 base with and without P5), pinned by
  `test_namespace_vector_unchanged_for_full_attention`.
