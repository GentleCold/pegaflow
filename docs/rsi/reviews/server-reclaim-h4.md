# Review: server reclaim H4 and review fixes (branch `feat/rsi-server-reclaim`)

Reviewer: independent review agent, 2026-10-24. Scope: `git log 5e6f8eb..935b68f`
(a8bf231 batch memcpy, c04b6c7 drain on error, ef53344 dedupe, 5c4cb9f and
935b68f docs). The previous review is `server-reclaim-h1-h3.md`.

Isolation respected: no connector code, no `service.rs` model branches, no
`instance.rs` slot math, no `offload.rs` page-first body, no model-adaptation
history. `git show b96e80c` and upstream PR #278 / issue #195 were read.

What I checked:
- The code statically, plus the cudarc 0.19.8 bindings it calls.
- Every run dir named in the brief on 10.1.0.44 (read only): `summary.json`,
  per-pass `result.json` metrics, and the `pegaflow.log` startup and debug
  lines.
- The recorded test logs in `~/kexi/logs` and the binaries they came from.

I did not build or run anything, locally or remotely.

## Verdict

**accept after must-fix.** The H4 code is correct on every path I traced, and
the old-arm TTFT regression is closed by the measurements. But the recorded
`cargo test` / `check.sh` evidence was produced from master source, not from
this branch. The previous must-fix 2 is therefore still open.

## Previous must-fix and should-fix status

| item | status | evidence |
|---|---|---|
| MF1 old-arm warm TTFT regression | closed | swap pairs re-read from `summary.json`, table below |
| MF2 `cargo test` / `check.sh` recorded on .44 | **open** | must-fix 1 |
| UAF on a failed transfer (`memcpy.rs`) | closed in code | `gpu_worker.rs:432-446`; the test run is not recorded |
| SF1 duplicate hashes in one save call | closed | `offload.rs:178-183`, `offload.rs:492` |
| SF2 pool returns to 0 after a failed allocation | test added, run not recorded | `offload.rs:747-758` |
| SF3 "merge across allocations" follow-up | closed | `reclaim-eviction.md:224-226` |
| SF4 PR note on `--blockwise-alloc` | closed | PR #3 body, "Breaking CLI change" |
| nit 1 `checked_add` | closed | `offload.rs:208-210` |
| nit 5 gate name in core code | fixed in `offload.rs`; a model name is new in `memcpy.rs` | nit 1 |

## Must-fix

1. **The recorded test and check runs tested master, not this branch.**
   `reclaim-eviction.md:411-437` and `PROGRESS.md:466` say the runs used
   "branch head ef53344". They did not:
   - The remote tree `~/kexi/pegaflow-rsi-server-reclaim` has
     `transfer/memcpy.rs`, `gpu_worker.rs`, `storage/mod.rs` and `offload.rs`
     byte-identical to 75dfa4c (master; md5 compared with `git show`).
   - Those files have ctime 21:49 UTC. Their mtimes match the main checkout
     `/Users/a1234/workspace/pegaflow` (on master), not this worktree. The root
     also holds that checkout's untracked `feishu-*.png`. The rsync source was
     the wrong directory.
   - `rsi-server-reclaim-check*.log` (21:51-21:52) and
     `rsi-server-reclaim-test*.log` (21:54-21:56) ran after that sync.
   - The logged lib binary `target/debug/deps/pegaflow_core-4089f011595b5ebd`
     (built 21:53) contains c04b6c7's removed string "Failed to record event".
     It contains no string from a8bf231 or ef53344.
   - The logged test list has none of the branch's five tests:
     `each_saved_block_owns_one_region_across_layers`,
     `reclaim_frees_what_the_allocation_needs_not_a_fixed_batch`,
     `duplicate_and_resident_hashes_are_not_saved`,
     `failed_allocation_returns_every_region`,
     `failed_submit_drains_copies_already_enqueued`.
   - "170 passed, 1 ignored" is master's count. The one ignored test is
     `kernel_matches_direct_both_directions`, so the run never touched the
     batch backend.
   - An earlier binary, `pegaflow_core-59a628b55322c793` (21:22), does contain
     all branch tests. No log of a run of it exists, and the `--ignored` GPU
     run the doc cites has no log at all.

   The server binary under test is fine. `bin/pegaflow-server-h4` (21:27,
   sha bda553a7, used by every H4 gate run) contains strings from all three
   code commits.

   Failure scenario: a clippy `-D warnings` error, or a failing test in a8bf231
   to ef53344, ships with "cargo test green" in the PR. Two cases nobody has
   seen run:
   - the drain test that guards the UAF fix;
   - the batch leg of the kernel test.

   Fix: sync this worktree (not the main checkout) to .44 and run
   `cargo test --no-default-features --features cuda-13,rdma --no-fail-fast`,
   then `cargo test -p pegaflow-core --lib ... -- --ignored`, then clippy and
   `check.sh`. Keep the logs next to the doc's claims and correct the doc and
   the PR body.

## Should-fix

1. **One failed batch turns batching off for the whole process, silently.**
   `memcpy.rs:36` and `memcpy.rs:153-166`: `BATCH_FAILED` latches on any batch
   error, before the per-range retry has shown that the batch API was at fault.
   - Scenario: a caller bug passes one invalid range, such as a stale block
     mapping or the null host pointer from the drain test. Real cases exist:
     vLLM #39491 got `CUDA_ERROR_INVALID_VALUE` from an out-of-bounds CPU
     block.
   - The batch fails, the per-range retry fails too, and the request errors.
     That much is correct. But every device's workers then stay per range
     until restart, which is +20-26% warm TTFT on the old arm by this branch's
     own swap pairs.
   - The only signal is one `warn!`. No metric records it.

   Fix:
   - latch only when the per-range retry of the same ranges succeeds;
   - export the submission mode, or a fallback counter, as a metric.
2. **The fallback path has no test, and the batch test cannot fail on a
   fallback.**
   - `kernel.rs:216,238,250` runs `MemcpyBackend::new()`. If the batch call
     failed, `submit` would retry per range, so the data check still passes.
     The test never asserts `batch.submission() == "batch"` after the run.
     The bench prints it, but the test does not check it.
   - The drain test (`gpu_worker.rs:544-588`) uses `per_range()`, so no test
     reaches batch error -> retry -> data lands.

   A real fallback trigger exists: the doc's legacy default stream returns
   `INVALID_VALUE` for batch and works per range. A test on
   `ctx.default_stream()` would prove the retry copies every range exactly
   once. It needs the latch to be per backend, or resettable under
   `cfg(test)`. Otherwise it flips the global latch for the parallel kernel
   test in the same process.
3. **A missing driver symbol panics instead of falling back.**
   - cudarc's default `fallback-dynamic-loading` resolves
     `cuMemcpyBatchAsync[_v2]` lazily. `load()` panics on a missing symbol
     (`cudarc .../driver/sys/mod.rs:11-13`).
   - The only guard is `cuDriverGetVersion >= MIN_BATCH_DRIVER`
     (`memcpy.rs:28-31,109`). If a driver at or above that version does not
     export the symbol this build calls, the first load panics the GPU worker
     thread, and every later load or save on that device hangs or fails.
   - The case I could not verify: a `cuda-12` build (the default feature of
     `pegaflow-server` and `python`) calling the non-`_v2` symbol on a 13.x
     driver. There is no libcuda under `~/kexi`, and I did not read outside it.

   Fix: probe the symbol once in `MemcpyBackend::new()`, via `cuGetProcAddress`
   or `culib().get`, and fall back to per range if it is missing.
4. **The `cuda-12` batch path has never run** (`memcpy.rs:297-327`).
   - Every recorded run is a `cuda-13` build on driver 595.91.07 / CUDA 13.2,
     RTX 4090 (`driver=13020` in every H4 `pegaflow.log`).
   - The `cuda-12` path has a different signature: `failIdx` is an
     out-parameter.
   - The pre-b96e80c code passed `std::ptr::null_mut()` for `failIdx` on that
     path (`git show b96e80c` hunk for `submit_batch_memcpy`, cuda-12 arm).
     A null required out-pointer is a plausible cause of the unrecorded
     "driver instability". The new code passes a valid slot, which is the
     right fix, but it is unproven.

   Run the kernel test once on a `cuda-12` build with a 12.8+ driver, or
   default the `cuda-12` path to per range until someone has. Also put
   `fail_idx` in the error string (`memcpy.rs:265-269`). It is the one piece
   of diagnosis the cuda-12 API gives, and it is dropped today.
5. **The `resident_bytes` "2x" is an analysis artifact, not a server bug.**
   `reclaim-eviction.md:343-345` should say so. See checklist item 7.
6. **Doc facts that do not match the run dirs:**
   - `reclaim-eviction.md:305`: `full-h4` replay ran on a **64 GB** pool
     (`args.pool_size = 64gb`, `--pool-size 64gb` in its `pegaflow.log`), not
     160 GB.
     - It is still a fair comparison: 0 evictions on every pass, as in the
       160 GB runs.
     - The 160 GB claim at `:162` covers only the H1-H3 runs. The swap pairs
       and `full-h4` are all 64 GB.
   - `reclaim-eviction.md:291`: swA v2 H4 warm2 is 0.468, not 0.469. That is
     trivial.
   - The miss-attribution and policy sims (`reclaim-eviction.md:349-409`) are
     not in any run dir, and I found no script or output for them on .44
     (`~/kexi/rsi-g5-sim` dates from 13:24, before the r3 work). The
     following are unverifiable as recorded:
     - "sim E_C 0.698 / 0.685";
     - the local-hit step to 0.584;
     - the tail-first +0.04;
     - the "r3 official" row.

     Store the sim script and its output with the runs before H5 builds on it.
7. **GPU confound in the r3 reruns.** Both `cap3-base` runs used GPU7 and both
   `cap3-h4` runs used GPU6. E_C is a hit metric, so the effect is small. But
   scheduling interleaving changes cache state, and every other comparison in
   this doc swaps GPUs. Swap one pair, or state it.

## Nits

1. `memcpy.rs:5-7`: the module doc cites "A layer-first load of Qwen3-4B".
   That is a model name in core code. It is not a branch, but the previous
   review asked for the same cleanup in `offload.rs`. Use "a 36-layer, 64 KiB
   page model" or move it to the doc.
2. `memcpy.rs:105`: `is_some_and(|v| v != "0")` also disables batching for
   `PEGAFLOW_DISABLE_BATCH_MEMCPY=` (empty) and `=false`. Document "any value
   but 0", or parse a boolean.
3. `kernel.rs:213`, bench `:388`: moving to `new_stream()` hides the
   legacy-stream failure instead of pinning it. Should-fix 2's test would pin
   it.
4. `offload.rs:178-183`: `seen` is per layer, so two layers could keep
   different GPU block indices for one hash if their `block_ids` orders
   differed. The content is identical by hash, and every layer still adds
   exactly one slot, so this is harmless. A one-line comment would help.

## Checklist results

### 1. Batch memcpy correctness

- **Arguments.** `submit_batch` (`memcpy.rs:216-272`):
  - One attribute: `srcAccessOrder = STREAM`, both loc hints `INVALID`,
    `flags = 0`.
  - `attr_idxs = [0]`, `numAttrs = 1`.
  - `dsts`, `srcs` and `sizes` are `merged.len()` each.
  - Direction mapping is right: H2D `(device, host)`, D2H `(host, device)`.
  - Host pointers pass as `CUdeviceptr` under UVA, which holds for both
    pinned strategies (`cudaHostAlloc` mapped, `cudaHostRegister`).
  - `count == 0` cannot happen (`memcpy.rs:139-141`).
  - The arrays live until the call returns, and the driver consumes them at
    call time.
- **Version gate.** 13000 for the `_v2` (cuda-13) entry and 12080 for the
  cuda-12 entry match the APIs' introduction versions. The weak spot is
  symbol presence (should-fix 3).
- **Streams.** Workers create their stream with `ctx.new_stream()`
  (`gpu_worker.rs:282-284`; cudarc creates it `NonBlocking`), so the
  legacy-default-stream `INVALID_VALUE` cannot happen in the server. That is
  unchanged from the pre-b96e80c code, which also used `new_stream`.
- **Fallback.**
  - Exactly one retry, of every merged range (`memcpy.rs:154-166`).
  - Ranges the failed batch may already have enqueued run twice. Both
    executions are stream-ordered writes of the same source bytes into a
    destination nobody reads before the sync, so the result is identical.
  - A failed retry returns `Err`.
  - Partial enqueues are never reported as success: `Ok` means every range
    was enqueued by one path and the stream drained cleanly.
  - Asynchronous execution errors surface at `synchronize`.
- **Lifetimes.** `submit_and_wait` (`gpu_worker.rs:432-446`) synchronizes on
  every outcome before it returns.
  - Load: `layers` drops after `completion.signal`.
  - Save: `layers` returns through `reply` after the sync, or drops after the
    `Err`.
  - This covers the kernel backend's chunked launches too.
  - The one gap: if `synchronize` itself fails, the buffers drop anyway. That
    only happens on a sticky context error, when no further DMA runs.

### 2. b96e80c history and the default

- b96e80c / PR #278 give no stack, no error, and no repro. Its validation list
  shows only that per-range works.
- Issue #195 is a benchmark data point in favor of batch.
- The removed code had no fallback, so any batch error failed the transfer.
  Its cuda-12 arm passed a null `failIdx` (should-fix 4).
- **Judgment.** Default on is acceptable for `cuda-13` builds, which is what
  `release.yml` ships:
  - Errors are retried, and the kill switch exists.
  - The gate digests show correct bytes on this stack. The `full-h4` replay
    has external blocks compared/mismatched of 18418/0 (old) and 18530/0 (v2),
    max rel L1 of 0.0033-0.0037.
  - The residual risk is "success but wrong" on another driver, which no
    guard catches. Record the kill switch in deploy notes, and run one replay
    on the production driver before rollout.
- For `cuda-12` builds, default off, or run it once first (should-fix 4).

### 3. Dedupe semantics (ef53344)

- **First copy wins.** The first GPU block index per `(group, hash)` per layer
  is kept. That is also the copy that sealed first before the fix, because
  later duplicates hit `AlreadyExists`.
- **Refcounts.** No lease or refcount is tied to a save. Each layer still
  contributes one slot per block, so sealing is unchanged.
- **Metrics.**
  - `pegaflow_save_bytes_total` and the `to_save` trace attribute now count
    unique copies. They used to count duplicates too.
  - Bytes saved now equals bytes copied, which is more accurate, but the
    series dips on duplicate-heavy workloads.
  - `cache_block_insertions` is unchanged, because it only counts
    `InsertedNew`.
- The dedupe runs before the `page_first` branch (`offload.rs:492` vs
  `:514`), so page-first saves dedupe too. That is benign: strictly fewer
  copies.

### 4. Tests

- `failed_submit_drains_copies_already_enqueued` would fail without c04b6c7:
  a 64 MiB D2H is still in flight when the unsynchronized return reads the
  host. It is timing-based, but its failure direction is reliable.
- `failed_allocation_returns_every_region` pins the all-or-nothing contract.
- `duplicate_and_resident_hashes_are_not_saved` is a cheap unit test of a
  local contract.
- None of these three is noise under the Testing Principles.
- Gaps: the fallback path, and the kernel test's blindness to fallback
  (should-fix 2).

### 5. Model-agnostic

No model names in code paths and no per-model branches. One model name in a
module doc comment (nit 1).

### 6. Evidence, re-read from `summary.json`

E_C / hit over the timed warm passes, r3 plan (C/U 0.25 / 0.5 / 0.75):

| run (GPU) | old E_C | v2 E_C | old hit | v2 hit |
|---|---|---|---|---|
| cap3-base (7) | 0.075 / 0.666 / 0.689 | 0.070 / 0.676 / 0.693 | 0.396 / 0.831 / 0.878 | 0.392 / 0.836 / 0.881 |
| cap3-base-b (7) | - / 0.664 / - | - / 0.683 / - | - / 0.828 / - | - / 0.835 / - |
| cap3-h4 (6) | 0.171 / 0.708 / 0.747 | 0.174 / 0.711 / 0.754 | 0.450 / 0.861 / 0.916 | 0.445 / 0.861 / 0.915 |
| cap3-h4-b (6) | - / 0.716 / - | - / 0.710 / - | - / 0.860 / - | - / 0.861 / - |

- H\*_C is 0.861-0.867 / 0.929-0.930 / 0.970-0.971.
- H_lru is 0.568-0.594 / 0.857-0.858 / 0.915-0.916.
- All four runs use gate bf9ef3c, v2 8a05b7c, and 64 convs.
- The doc's table and its +0.04-0.05 / +0.10 / +0.06 deltas all match.
- H4 runs use `pegaflow-server-h4` (sha bda553a7). Base runs use
  `pegaflow-server-base` (sha 99b77ca6).

Dead memory, in blocks: pool_used minus resident blocks x 2359296 at pass
ends. Resident blocks are accumulated from the per-pass deltas, and
pool_used is read from `pegaflow_end`:
- base at 0.5: 1232-2636. That is 10.7-22.9% of 11536 blocks, which the doc
  rounds to 11-23%.
- base at 0.75: 2630-4458.
- H4: 0 on every pass at every C, with `resident_bytes == pool_used_bytes`
  exactly.

All matches the doc.

Swap pairs, warm2 / warm3 TTFT p50 in seconds, 64 GB pool, 0 evictions:

| pair | v2 H1 | v2 H4 | old H1 | old H4 |
|---|---|---|---|---|
| swA (H4 GPU6) | 0.512 / 0.543 | 0.468 / 0.475 | 0.607 / 0.638 | 0.474 / 0.469 |
| swB (H4 GPU7) | 0.521 / 0.554 | 0.487 / 0.495 | 0.607 / 0.606 | 0.470 / 0.486 |

- **H4 vs H1.** v2 is 7-13% lower and old is 20-26% lower. req/s is
  10.28-10.61 (H4) vs 10.00-10.48 (H1).
- **H4 vs base.** H4 old also beats `full-base` old (0.613 / 0.642), though
  that ran at 160 GB.
- **Old arm.** MF1 is closed.
- **`full-h4` capacity.** E_C 0.539 / 0.448 at C = 13857325056 B.
- **`gsm8k-h4`.** 0.8594 on every arm, and every gsm8k G2/G3/G4 verdict
  passes.
- **Startup log.** Every H4 `pegaflow.log` logs `submission: batch
  (driver=13020 ...)`. None logs a fallback, and `load_errors` is empty
  everywhere.
- **Submit count.** `swA-h4` old has 299 debug H2D lines, all `submits=1`
  (for example `copies=10224 ranges=10224`).

Not verified: the sim tables (should-fix 6), and the copy-count rows for
`cnt2-h1log` / `cnt2-h3log`, which I did not re-derive.

### 7. `pegaflow_cache_resident_bytes` "2x"

This is not a bug, and H3 did not introduce it.
- `pegaflow_end.pegaflow_cache_resident_bytes / resident_blocks` is exactly
  2359296 on every pass of every cap3 run, base and H4.
- The doubled number comes from prefix-summing metric keys. The prefix
  `pegaflow_cache_resident_bytes` matches both
  `pegaflow_cache_resident_bytes{...}` and
  `pegaflow_cache_resident_bytes_by_class{class="retained",...}`. On every
  pass of every cap3 run, that sum is exactly 2.00x the unlabeled gauge.
- The `.cap3.py`-style `g(prefix)` helper does exactly that.
- The gate's own `pegaflow_end` filter (`run_gate.py:657-662`) matches
  `"pegaflow_cache_resident_bytes{"`, which excludes `_by_class`, so the gate
  is correct.
- In the server, insert and evict both add and subtract `memory_footprint()`
  (`read_cache.rs:490`, `storage/mod.rs:486,586`), which is the sum of slot
  sizes.

Correct `reclaim-eviction.md:343-345`. Prefer `pegaflow_end` over prefix sums
in the analysis scripts.
