# Review: server reclaim H1 + H3 (branch `feat/rsi-server-reclaim`)

Reviewer: independent review agent, 2026-10-24. Scope:
`git diff 75dfa4c..8d00e00 -- pegaflow-core pegaflow-server pegaflow-common`
(75dfa4c is the merge base with local `master`; `origin/master` is older, at
627c8c9). Commits in that range that are not H1/H2/H3 (de58f96 shard rotation,
8360b5c semantics `group_hash`) were read only for context and are not
reviewed here.

Isolation respected: no connector code, no `service.rs` model branches, no
`instance.rs` slot math, and no `offload.rs` page-first body was read.

## Verdict

**accept-with-fixes.** H1 is correct and can merge as is. H3 is memory-safe:
regions are RAII `Arc<PinnedAllocation>`s owned by segments, so there is no
double free and no early free. But H3 regresses the `old` arm's warm replay
TTFT, and PROGRESS.md's layering table requires server changes to keep the
`old` arm unregressed. That has to be resolved or explicitly waived by a human
before merge.

Build status: `cargo test -p pegaflow-core --lib` does not build on this macOS
host (`rdma-mummy-sys` needs `linux/types.h`, `io-uring` fails without the
`rdma` feature, and the lib tests allocate CUDA pinned memory). This is a
static review only. Nothing was run on remote hosts.

## Must-fix

1. **H3 regresses the `old` arm, which the server policy forbids.**
   `docs/rsi/PROGRESS.md:30` requires server changes to keep "the `old` arm
   unregressed". `reclaim-eviction.md` (Evidence) measures old warm2/3 TTFT
   p50 +0.07-0.28 s (0.63 -> 0.70-0.91 s) on H3. This is the known open issue,
   not a new finding, but it blocks merge under the stated policy. Either fix
   it (see the coalescing section: the doc's second follow-up cannot work), or
   record an explicit human waiver in the PR. Failure scenario: master
   deployments on the old connector get 11-44% slower warm TTFT with no pool
   pressure at all. The 160 GB replay pool never reclaims, so H3's capacity
   win is zero there.
2. **No `cargo test` / `scripts/check.sh` evidence is recorded for the
   branch.** The same policy line requires `cargo test` green.
   `reclaim-eviction.md` records only gate runs. The two new tests
   (`offload.rs:683`, `storage/mod.rs:856`) and the removed `blockwise_alloc`
   field have not been shown to compile under clippy `-D warnings`. Run
   `cargo test -p pegaflow-core --lib` and `./scripts/check.sh` on a
   Linux/CUDA host and record the result in the doc or the PR.

## Should-fix

1. **Duplicate hashes inside one save call still pin dead bytes, which
   contradicts H3's stated invariant** (`offload.rs:477-481`,
   `offload.rs:186-197`, `write_path.rs:143-156`). Phase 1 dedupes
   `hashes_to_save` as a set, but `ctx.blocks_to_save.retain(...)` keeps every
   occurrence. If one call carries the same `(group, hash)` k times (two
   requests in one step computing the same new block in different GPU
   blocks), `alloc_block_targets` sizes that block's region k times larger. It
   then copies all k, and the insert worker seals k blocks. The first one goes
   resident, and `batch_insert_refs` drops the rest as `AlreadyExists`. The
   resident block's segments still hold the k-sized region, so (k-1)/k of it
   is dead until that block is evicted, and hot prefix blocks are rarely
   evicted. `memory_footprint` counts only the slot sizes, so
   `cache_resident_bytes` hides this.
   - This is not a regression: before H3 the duplicates sat in the per-layer
     batch region, which was held even longer.
   - It is the remaining way H3's "region frees exactly when the block goes"
     is false.
   - Fix: dedupe `(layer, group, hash)` in `alloc_block_targets`, or in the
     retain at 479, before allocating. This also saves the redundant GPU
     copies.
2. **No test pins the all-or-nothing allocation contract.**
   `alloc_block_targets` allocates every region up front and returns `Err`
   at the first `None` (`offload.rs:202-210`). Dropping `regions` then frees
   the earlier ones, which is correct by RAII. But no test asserts
   `pool_used == 0` after a failed call. A table case in the existing test is
   enough: pool fits 1.5 blocks, save 2 blocks, expect `Err` and 0 used. It
   protects the contract the per-block change introduced: N allocations per
   call instead of L.
3. **The doc's coalescing follow-up "allow merging across allocations that
   are adjacent in one pinned backing" would not help H3**
   (`reclaim-eviction.md:241`). See the coalescing section: under H3, layer L
   of block i and layer L of block i+1 are a full block apart in host memory,
   not one layer stride, so no host-adjacency rule can merge them. Remove it,
   or restate it as applying only to a layer-major per-block allocation order.
4. **Removing `--blockwise-alloc` makes clap reject the flag at startup**
   (`pegaflow-server/src/lib.rs`, the removed `Cli` field). The repo is clean:
   `git grep -i blockwise` outside the reclaim doc finds nothing on HEAD
   (CLI, `docs/server.md`, `rsi/`, `python/` except the excluded connector
   path, scripts). But any deployment script that passes the flag will exit at
   launch after an upgrade. I could not check cluster launch args (no remote
   access in this review). Put this in the PR description and release notes,
   or keep a hidden no-op flag for one release.

## Nits

1. `offload.rs:196`: `region_bytes[i] += ...` is unchecked, while the old
   code used `checked_mul`. It cannot overflow in practice (the allocator
   rejects anything past u32 units first), but `checked_add` keeps the old
   `EngineError` contract instead of a release-mode wrap.
2. Per-block size-class waste is now deterministic per model
   (`allocator.rs:106-134`, `allocator.rs:161`). Each region is rounded up to
   an offset-allocator class boundary, at most 1/8 extra. Qwen3-4B's 2.25 MiB
   block is 4608 units, an exact class, so the G5 run cannot show this. A
   block of, for example, 8784 units rounds to 9216 (+4.9% on every resident
   block). Before H3 the waste varied with the batch size. Worth one line in
   the doc and a check on the next model.
3. Allocation count: `PinnedMemoryPool::MAX_ALLOCS = 4_000_000` per shard
   (`pinned_pool.rs:141`). With per-block (and per-TP-rank) regions, a pool
   holds `pool / region` live allocations. Regions under about 40 KiB on a
   160 GB pool would hit the node limit before the byte limit. Then
   `storage_report` reports 0 free and reclaim evicts to recover nodes. That
   is far from today's sizes, but it is a new failure mode that blockwise
   had and the old batch path did not.
4. `offload.rs:202-210`: one `storage.allocate` per block means one shard
   lock and one `pool_used_bytes` update each. Within a call every block has
   the same size (same layer set), so `allocator.allocate_many` (already
   present, `cfg(rdma)`-gated as dead code otherwise) would take them in one
   lock. Save duration did not move, so this is optional.
5. `offload.rs:177` cites "the G5 capacity trace" in core code. That is a
   gate name, not a model name, so it is not a layering violation, but the
   evidence belongs in the doc. Prefer a neutral comment.
6. `reclaim-eviction.md:235`: "32 KiB for Qwen3-4B" assumes K/V split
   segments. `models/Qwen3-4B/ITERATIONS.md:20-29` says v2 registers one
   65536 B segment per layer. State which arm has which layout.
7. H1 widens a pre-existing race (`storage/mod.rs:562-607`,
   `metaserver_client.rs:712-718`). Removals are now sent at the end of the
   reclaim call, not per round. If an evicted key is re-saved and its
   `Insert` is enqueued before the call ends, the later `Remove` wins the
   consumer's last-op-wins `net` map. MetaServer then forgets a resident
   block. This only causes a false negative for peers, and it needs a reclaim
   call that outlasts a full save. It is unlikely, but worth a comment.
8. `storage/mod.rs:567`: `resident_blocks()` takes the read-cache lock once
   per round, and `remove_lru_batch` takes it again. It is harmless, but the
   size could be computed inside `remove_lru_batch` under one lock.

## Checklist results

### 1. H3 region ownership and lifetime

- **Freed exactly once, after the last holder.** `Segment` owns an
  `Arc<PinnedAllocation>` (`block.rs:62-77`), and `PinnedAllocation::drop`
  frees under the shard lock. Every holder keeps the region alive through
  `Arc<SealedBlock>` -> `RawBlock` -> `Segment`:
  - read cache, pin/consume, load (`HostBlock::Cached`, `gpu_worker.rs:67`);
  - transfer locks (`transfer_lock.rs:17`);
  - SSD write-back (Weak upgraded for the write);
  - prefetch staging.

  Eviction only takes cache-owned-only blocks (`read_cache.rs:653`,
  `cache.rs:84-88`). H3 changes which blocks share an `Arc`, not the ownership
  model, so no new early-free path exists.
- **Partial save.** An allocation failure drops all regions of the call
  (`offload.rs:202-210`). A GPU copy error drops the layers in the worker, so
  nothing reaches the insert worker (`offload.rs:526-530`). A block never
  seals with missing slots (`block.rs:408-437` seals only when `remaining ==
  0`). An unsealed inflight block is dropped by `gc_stale_inflight`. No leak.
  - Pre-existing, out of scope: on a mid-batch `cuMemcpy*Async` error,
    `memcpy.rs:61-77` returns without synchronizing. The worker then drops the
    host buffers (`gpu_worker.rs:333-341`) while earlier DMAs may still be
    queued. That is a potential write into freed and reused pinned memory.
    H3 does not change it.
- **Duplicate saves of the same key.** Across calls, Phase 1 filters
  resident keys. A late duplicate is dropped whole at `write_path.rs:144-149`.
  Its region belongs only to that block in that call, so it frees
  immediately, which is better than before. A slot-level duplicate into an
  inflight block (`write_path.rs:255`) frees only if every slot of that call's
  block was a duplicate. Otherwise the duplicate slot's bytes ride along with
  the inserted slots (partial overlap of two calls; edge case). Within one
  call, see should-fix 1.
- **TP > 1.** Each rank's RPC is its own call, so each block gets one region
  per rank, all owned by the one sealed block. That is correct, with `tp`
  regions per block.
- **Replicated / MLA single writer.** Only the writer saves. The region is
  sized from the writer's layer contexts. Correct.
- **Multi-group.** Regions are keyed `(group, hash)` (`offload.rs:185-191`),
  and each group seals separately (`offload.rs:621-649`). Group 0 and group 1
  never share a region, and the test covers it.
- **SSD / RDMA fetch.** These are unchanged by H3. RDMA fetch already
  allocates per block per NUMA (`rdma_fetch.rs:723-768`, since #465), so H3
  makes saves consistent with fetches. SSD prefetch still uses up-to-256 MiB
  chunks shared across blocks (`ssd_cache.rs:820-847`). A block restored from
  SSD can therefore still pin a chunk after eviction, the same dead-bytes
  pattern H3 removed from saves. This is not measured (no SSD runs) and is
  worth a line under Open risks.

### 2. H3 sizing

Both loops walk `layer_contexts` and `blocks_to_save` in the same order. The
size pass adds `padded_block_bytes()` per (layer, block) (`offload.rs:196`),
and the build pass advances `offset` by the same amount (`offload.rs:228`).
Offsets therefore tile the region exactly, even when layers in one group have
different sizes.

- Split layouts place K at `offset` and V at `offset + padded_segment_bytes`,
  and `padded_block_bytes = padded_segment * segments` with `segments == 2`.
  Correct.
- Interleaved (`segments == 2`, `kv_stride == segment_bytes`) and single
  segment layouts take the `single_segment(padded_block_bytes)` branch.
  Correct.
- Every `Segment::new` is bounds-checked against the allocation
  (`pinned_pool.rs:98-107`).
- With SSD enabled, every layer offset is a multiple of 512, so O_DIRECT
  iovecs stay aligned.

No off-by-one found. Overflow: nit 1.

### 3. Allocator behaviour

No lock is held across a copy: allocation is Phase 2, the copy is Phase 3.
There is no O(n^2): one `HashMap` index plus linear passes. Allocation count
and class waste: nits 2-4.

### 4. H1

- `clamp(resident / 64, 1, 512)` (`storage/mod.rs:758`) is 1 at resident=0
  (no panic: min <= max). It is 1 for any pool under 128 blocks.
- **No spurious OOM or livelock.**
  - `reclaim_until_allocator_can_allocate` loops rounds until
    `largest_free >= required` or a round evicts nothing
    (`storage/mod.rs:564-602`).
  - `allocate` re-loops while `freed_blocks > 0`, and fails only after a
    zero-progress reclaim plus a final retry (`storage/mod.rs:336-351`).
  - Round size bounds only the overshoot, never the total.
  - Referenced blocks are skipped and moved to MRU (`read_cache.rs:657-659`),
    so later rounds do not rescan them.
- **MetaServer batching.** `unregister` collects only keys actually returned
  by `remove_lru_batch` (`storage/mod.rs:574-580`), and nothing between
  eviction and the single `try_unregister` (`storage/mod.rs:605-607`) can
  return early. Removals cannot be lost or sent for blocks that were not
  evicted. One difference: a full queue now drops the whole call's batch in
  one `try_send` (`metaserver_client.rs:525-530`), not per round. The total
  is the same, and the queue is far less likely to fill. Race: nit 7.
- H2 revert is complete: `git diff b0e6d93 e718c7e -- pegaflow-core
  pegaflow-server pegaflow-common` is empty.

### 5. Removed flag

See should-fix 4. `docs/server.md` was updated in 1521d64. The `StorageConfig`
field and the getter are gone, and the Python bindings and `rsi/` have no
references.

### 6. Model-agnostic

No model names and no per-model branches were added to code. The only
evidence reference is a gate name (nit 5). H3 depends only on `KVCacheLayout`
and `group`.

### 7. Tests

- `each_saved_block_owns_one_region_across_layers` (`offload.rs:683-719`)
  fails if blocks share a region again (`a != b`, and the pool-bytes check
  catches a per-layer region). It also fails if K/V or layers of one block
  split into separate regions (`ids(1, 0) == [a, a]`). It covers per-group
  separation and frees after the last slot drops.
- No test covers "free after the last *reader*" (a pinned or transfer-locked
  sealed block surviving eviction). That is generic RAII, H3 does not change
  it, and it is acceptable.
- No partial-save error test: should-fix 2.
- `reclaim_frees_what_the_allocation_needs_not_a_fixed_batch`
  (`storage/mod.rs:856`) would fail under fixed 512-block rounds. It assumes
  the pinned backing is exactly 64 regions. A backing rounded up to a page or
  hugepage size would need no reclaim, and `resident == 63` would fail. I
  could not check that without CUDA.

## Coalescing hypothesis (known open issue)

**Confirmed for contiguous (single-segment) layer layouts. Refuted as a cause
for split K/V layouts.**

Merge condition, `pegaflow-core/src/transfer/memcpy.rs:34-40`: adjacent list
entries merge only if
`device_allocation` and `host_allocation` are equal, `device` is contiguous,
and `host` is contiguous.

- `device_allocation` is the layer index (`gpu_worker.rs:386,394,413`).
- `host_allocation` is the `Arc<PinnedAllocation>` address
  (`gpu_worker.rs:387,395-397,414`; `block.rs:79-81`).
- Descriptors are built layer-major: per layer, per block, K then V
  (`gpu_worker.rs:357-419`).

Contiguous layout:

- Before H3, layer L's blocks from one save call sat in one region at
  `i * stride`. A load of a prefix saved by the same call into consecutive
  GPU blocks merged into one `cuMemcpyHtoDAsync` per layer per run.
- After H3, layer L of block i and block i+1 live in different allocations,
  and even their addresses are a full block apart. The `host_allocation`
  check fails, so nothing merges. Merging layers of one block is also
  impossible, because `device_allocation` differs per layer.
- A warm load of N blocks over L layers is therefore exactly N x L submits
  (64 KiB each for a 65536 B page). Before, it was about L x runs. At a few
  µs of driver cost per call, hundreds of blocks x 36 layers adds tens of ms,
  which matches the +60-90 ms v2 shift.
- `--blockwise-alloc` gave the same shift for the same reason: one allocation
  per (block, layer).

Split K/V layout:

- The list order is K0, V0, K1, V1, so K_i is always followed by V_i, which
  fails `device_contiguous` (`kv_stride > segment_bytes`).
- Split layouts never merged before H3 either, so lost coalescing cannot
  explain a regression on an arm that registers split K/V.
- The isolation rule kept me from checking which layout the `old` arm
  registers. If it is split, the old arm's +0.07-0.28 s needs a different
  cause.

Implication for the follow-ups listed in the doc:

- Relaxing the merge rule to "same backing" cannot help H3 (should-fix 3).
- The viable options are:
  - pick the kernel backend for fragmented batches (one launch regardless of
    fragment count, `transfer/mod.rs:10-13`);
  - a batched-copy API;
  - an allocation order that keeps layer L of consecutive blocks
    VA-adjacent: per-(block, layer) allocations made layer-major, plus the
    relaxed merge rule.

The worktree currently has an uncommitted `memcpy.rs` debug log of
`copies` vs `submits` from the other agent. That is the right instrument to
settle this. One warm load on H1 vs H3 should show submits go from about
`L x runs` to `N x L`.
