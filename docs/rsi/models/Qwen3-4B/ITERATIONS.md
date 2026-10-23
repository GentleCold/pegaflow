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
