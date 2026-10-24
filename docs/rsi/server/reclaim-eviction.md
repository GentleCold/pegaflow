# Server: reclaim sizing, eviction order, save allocation (branch `feat/rsi-server-reclaim`)

Trigger: G5 (capacity hit rate) on Qwen3-4B. Every Qwen3-4B block is one
Full-attention (group 0) key of 2.25 MiB, so this is pure single-group
eviction. Group coherence across Kinds is not exercised here.

## Result

| Change | Commit | Kept |
|---|---|---|
| H1: reclaim rounds sized to the resident cache | b0e6d93 | yes |
| H2: refresh a prefix hit tail-first | 59083cf, reverted in df1d544/e718c7e | no (negative) |
| H3: one pinned region per saved block | 1521d64 | yes |
| H4: one `cuMemcpyBatchAsync` per transfer (recovers H3's warm TTFT) | a8bf231 | yes |

Official capacity (32x12 replay, C = 12.9 GiB = 5873 blocks), timed warm
passes:

| server (run) | old E_C | v2 E_C | old H*_C / H_lru / H | v2 H*_C / H_lru / H |
|---|---|---|---|---|
| base (`full-base`) | 0.122 | 0.205 | 0.811 / 0.542 / 0.453 | 0.777 / 0.522 / 0.436 |
| H1 (`full-h1`) | 0.286 | 0.242 | 0.823 / 0.551 / 0.511 | 0.771 / 0.515 / 0.454 |
| H1 (`full-h1c`, repeat) | 0.118 | 0.260 | 0.815 / 0.537 / 0.454 | 0.784 / 0.521 / 0.463 |
| H1+H2 (`full-h2`) | 0.131 | 0.204 | 0.806 / 0.517 / 0.447 | 0.793 / 0.516 / 0.441 |
| H1 + `--blockwise-alloc` (`full-blockwise`) | 0.527 | 0.433 | 0.832 / 0.562 / 0.629 | 0.793 / 0.519 / 0.543 |
| H1+H3 (`full-h3`) | 0.544 | 0.438 | 0.837 / 0.551 / 0.634 | 0.795 / 0.518 / 0.547 |

`full-h1`'s old 0.286 did not repeat (`full-h1c`: 0.118 on the same binary),
so treat it as noise. With per-block allocation both arms get more server
hits than the plain LRU reference H_lru over the same accesses (H vs H_lru).
G5 still fails at 0.9.

## Why v2 E_C was ~0.25 against an LRU reference of ~0.39

E_C = Σext / Σext\*_C. Applying the gate's H_lru model to the `full-h1c` v2
trace gives E_C 0.394, against 0.260 measured. The diagnosis used the
`full-h1c` cap traces, the server's per-query hit log, and a trace sim of the
server cache: plain LRU over sealed blocks, the H1 reclaim rounds, and no
allocator.

1. Pool fragmentation, the main cause. A layer-first save took one pinned
   region per layer per save call, and the region was freed only when the
   call's last block was evicted. Pool bytes used minus resident bytes at
   pass ends, in blocks:

   | run | arm | cap_cold | cap_warm / warm2 / warm3 |
   |---|---|---|---|
   | full-h1c | v2 | 341 | 419 / 209 / 214 |
   | full-h1c | old | 885 | 180 / 627 / 266 |
   | slice-h1 | v2 | 193 | 150 / 143 / 120 |
   | slice-h1 | old | 34 | 29-37 each |
   | blockwise | both | 0 | 0 |

   Residency after `cap_cold` on the official run went from 5473 (v2) and
   4750 (old) in full-h1c to 5848 and 5823 in full-h3, out of 5873.

   This is the slice's missing residency, 726 of 1019 after `cap_cold`
   (919 used, 193 dead). It is not reclaim overshoot, because H1 rounds are
   already at most resident/64. It is not unsealed blocks either: the gap
   goes to 0 with per-block allocation.

   At full capacity the sim predicts v2 0.397 and old 0.365. Measured
   E_C (0.260 / 0.118) matches the sim only at a capacity of 4800-5300
   blocks instead of 5873. Removing the fragmentation measured 0.433 / 0.527
   (blockwise).

2. Touch model: not the cause. v2 queries from block 0, so every lookup also
   refreshes the vLLM-local prefix on the server. old queries from its local
   hit, so local heads are never refreshed. The table shows E_C on the
   `full-h1c` traces at C = 5873 as each server behavior is added (v2 / old):

   | sim | v2 | old |
   |---|---|---|
   | gate H_lru model (every block stored, refreshed per request) | 0.394 | 0.321 |
   | + only blocks the connector computed are saved | 0.310 | 0.354 |
   | + the real query window per request | 0.380 | 0.353 |
   | + H1 reclaim rounds (server-faithful) | 0.397 | 0.365 |
   | faithful + touch the local prefix on every query | 0.355 | 0.422 |
   | faithful + re-save refreshes resident blocks | 0.340 | 0.303 |

   For v2 the touch-model differences cancel, so the server-faithful sim
   ends where the gate's H_lru model does. A model-agnostic local-prefix touch
   would help old (+0.06) but cost v2 (-0.04), so it is not a fix for
   either arm.

3. Miss attribution against ext\*_C. For each warm request, the tokens
   between the real hit and the Belady hit are attributed to the block where
   the real scan stopped. Residency comes from the server-faithful sim:

   | class | v2 | old |
   |---|---|---|
   | served | 0.264 | 0.116 |
   | fully evicted: the stopping block and the rest of the needed range are gone | 0.458 | 0.351 |
   | resident in the sim, missing on the server (fragmentation) | 0.153 | 0.308 |
   | stranded: stopping block evicted, a later needed block resident | 0.126 | 0.207 |
   | query starts past the local hit | 0 | 0.018 |
   | not saved yet, or saved <= 8 steps before the query | 0 | 0 |

Conclusion: the fix is in the server allocator. It is not eviction order (H2
measured that) and not the connector. Prefix-aware eviction would target the
stranded class (13-21%). It was not attempted, because fragmentation was the
larger loss and its fix could be measured directly.

## Changes

- H1: `RECLAIM_BATCH_SIZE = 512` is still the upper bound. A round now evicts
  `clamp(resident / 64, 1, 512)` blocks, so pools of 32768 blocks or more
  are unchanged. All MetaServer removals of one reclaim call go out as one
  command, so the removal queue does not fill under small rounds. Before H1,
  one 512-block round on a 1019-block pool freed half the cache: evictions
  came in multiples of 512, and only 502 blocks were resident after
  `cap_cold`. In a sim of the slice, rounds of resident/64, /32 and /16
  give v2 E_C 0.456 / 0.449 / 0.449 and old 0.170 / 0.165 / 0.161, against
  0.402 and 0.093 with fixed 512-block rounds.
- H2 (negative): tail-first refresh of a prefix hit, on the idea that the
  head blocks vLLM serves locally age out and strand the chain. The slice
  moved within noise (0.446 -> 0.461), and at official capacity it cancelled
  H1 for both arms.
- H3: a layer-first save call allocates one region per saved block, shared
  by all layers and K/V segments the call saves for that block. Evicting the
  block frees its whole region.
  - It replaces the opt-in `--blockwise-alloc` flag (one region per block per
    layer, 36x more allocations for Qwen3-4B), which is removed.
  - A block whose layers arrive in several save calls gets one region per
    call.
  - Page-first saves already allocated one page per block and are unchanged.
  - Region size rounds up to an allocator size class (at most 1/8 extra).
    Qwen3-4B's 2.25 MiB block is an exact class. A block of, say, 8784
    units would round to 9216 (+4.9% on every resident block). Check this
    on the next model.
- H4 (a8bf231): the memcpy backend submits all merged ranges of one
  transfer with one `cuMemcpyBatchAsync` (stream-ordered source access),
  instead of one `cuMemcpyAsync` per range. See "H4: H3's warm TTFT cost".
- Review fixes:
  - c04b6c7: a load or save task synchronizes its stream before it returns
    an error. It used to return first and drop host buffers that queued
    copies could still touch.
  - ef53344: one save call saves each hash once per layer. Before, a block
    computed twice in one step had its region sized and copied twice, and
    the dead duplicate stayed pinned while the block was resident.

## Evidence (Qwen3-4B, gate rev 369b253, GPU6/7)

Slice: 64 gsm8k, 8x6 replay, C = 2.24 GiB = 1019 blocks. H\*_C is 0.903, and
H_lru is 0.822-0.831 depending on the arm's trace.

| server (run) | arm | E_C | H | resident after cap_cold |
|---|---|---|---|---|
| base (`slice-base`) | v2 | 0.402 | 0.800 | 502 |
| H1 (`slice-h1`) | v2 | 0.446 | 0.807 | 726 |
| H1+H2 (`slice-h2`) | v2 | 0.461 | 0.810 | 842 |
| H1 + blockwise (`slice-blockwise`) | v2 | 0.529 | 0.822 | 1008 |
| H1+H3 (`slice-h3`) | v2 | 0.466 | 0.811 | 1008 |
| all of the above | old | 0.000 | 0.791 | 502-1009 |

- The old arm gets 0 server hits in the slice cap (1886 misses per pass), so
  the slice cannot show a change for it.
- H3 and blockwise have the same `cap_cold` residency and hit counts
  (4845 / 4236). The warm-pass gap (0.466 vs 0.529, over 50 k ext\*_C
  tokens) is not explained. The official run decides.

Non-regression for H3 (the H1-H3 replays ran on the gate's 160 GB pool,
so reclaim never ran there; the H4 swap pairs and `full-h4` below use
64 GB, also with 0 evictions):
- Correctness: gsm8k accuracy is 0.8594 on every arm, and there are no
  digest mismatches.
- Slice replay (8x6), slice-h3 vs slice-h1: no change.
  - v2 warm2/3 TTFT p50 0.091 / 0.094 s (H1: 0.091 / 0.091), 6.87 / 6.84
    req/s (H1: 6.85 / 6.83).
  - old 0.132 / 0.133 s (H1: 0.132 / 0.133), 6.68 / 6.66 req/s (H1: 6.70 /
    6.70).
- Official replay (32x12): H3 raises warm TTFT p50 and leaves req/s
  unchanged. Two of the pairs below ran H1 and H3 at the same time on
  GPU6/7, and they swapped GPUs between the pairs (`replay-h1`/`replay-h3`,
  then `swap-h1`/`swap-h3`). Cells are warm2 / warm3 TTFT p50 in seconds:

  | runs | v2 H1 | v2 H3 | old H1 | old H3 |
  |---|---|---|---|---|
  | full-h1c / full-h3 | 0.545 / 0.548 | 0.603 / 0.609 | 0.631 / 0.656 | 0.908 / 0.767 |
  | swap-h1 / swap-h3 | 0.537 / 0.546 | 0.627 / 0.602 | 0.630 / 0.631 | 0.695 / 0.706 |
  | replay-h1 / replay-h3 | - | - | 0.635 / 0.610 | 0.771 / 0.723 |

  - That is +0.06-0.09 s (10-17%) for v2 and +0.07-0.28 s for old. Earlier
    H1/H2 runs had v2 0.52-0.59 s. Blockwise allocation showed the same
    shift (v2 0.593 / 0.601 s).
  - req/s stays at 9.8-10.3 for every run.
  - Save duration (old's connector metric) is unchanged at 0.66-0.73 s per
    pass. Old's load-duration sum does not separate the servers: 17-22 s
    on H1 and 17-28 s on H3. The cause is lost copy coalescing. H4 fixes
    it; see "H4: H3's warm TTFT cost".
  - G4 still passes, because baseline replay warm TTFT is 2.5 s.
- full-h3 cap warm passes ran faster: old 61 / 55 / 55 s (H1c: 72 / 70 /
  70 s), v2 68 / 62 / 64 s (H1c: 73 / 70 / 70 s). The likely reason is more
  server hits, not a faster copy path. `cap_cold` was mixed: old 111 s
  (H1c: 74 s), v2 69 s (H1c: 97 s).
- `G4.gsm8k_warm.seconds` failed for v2 in slice-h3: warm 16.48 s vs cold
  14.46 s; slice-h1 warm was 14.55 s. A gsm8k-only repeat on the same binary
  (`gsm8k-h3`, GPU7) passed: v2 warm 14.64 s vs cold 14.57 s, and old
  14.62 s vs 14.62 s. The slice-h3 value is noise.
- `G2.digest_self.cap` fails for old on every server, because old has no
  external blocks to judge in the slice.

Rejected follow-ups (not committed):
- TinyLFU admission (`--enable-lfu-admission`), GPU slice on H1+H2: v2 E_C
  0.429, below H1+H2's 0.461.
  - Adding `--hint-value-size 2.25mb` (on H1) gave 0 hits on both arms even
    though blocks were inserted. The cause was not diagnosed; the hint
    resizes allocator units.
- Insertion-position policies, sim only: a new save enters near the eviction
  end (LIP) and moves up only on a hit. The slice trace gives E_C 0.95 (v2)
  / 0.92 (old), because the gate replays the same loop. In a multi-turn
  workload, though, the next turn re-reads blocks the previous turn just
  saved, and LIP evicts exactly those first. Not tried on a non-looping
  trace. Rejected as fitting the gate rather than the workload.

## H4: H3's warm TTFT cost

Cause: lost copy coalescing.
- The memcpy backend merges copies only when they are in the same device
  allocation (layer) and the same host allocation, and contiguous on both
  sides.
- Before H3, one layer's blocks from a save call sat in one region, so a
  prefix loaded into consecutive GPU blocks merged into about one copy per
  layer.
- Under H3, layer L of block i and of block i+1 are in different regions,
  a full block apart. No host-adjacency rule can merge them, so the
  "merge across allocations" idea is dropped.
- Both arms register one contiguous 65536 B segment per layer (copies =
  blocks x 36, 64 KiB each), so the same cause applies to old.

Copy counts per warm replay run (32x12, `RUST_LOG` debug, 64 GB pool), from
`cnt2-h1log`, `cnt2-h3log`, and `swA-h4`:

| server | arm | load submits / task p50 | copies / submit | load p50 | load GB/s | save GB/s |
|---|---|---|---|---|---|---|
| H1 | v2 | 5256 | 1.4 (87 KiB) | 30.6 ms | 15.9 | 15.4 |
| H3 | v2 | 7722 | 1.0 (64 KiB) | 38.7 ms | 12.85 | 11.7 |
| H4 | v2 | 1 | 7282 | 25.7 ms | 19.7 | 15.4 |
| H1 | old | 12348 | 1.3 (86 KiB) | 73.4 ms | 15.5 | 15.6 |
| H3 | old | 18612 | 1.0 (64 KiB) | 96.2 ms | 12.4 | 11.3 |
| H4 | old | 1 | 22806 | 66.6 ms | 20.0 | 14.2 |

Load bytes are the same on every server (444-458 GB per run).

Options, by measurement:
- Batched submit (`cuMemcpyBatchAsync`), chosen. It keeps H3's
  one-region-per-block ownership.
- Kernel backend: rejected. In the transfer bench it is fastest for H2D
  (below), but P3 measured it 26% slower per load end to end on Qwen3
  (TTFT +37%). P4 found that server submissions on the GPU slow NCCL
  all-reduce.
- Layer-major region per batch with refcounted sub-ranges: not needed.

`pegaflow-core/benches/transfer_h2d.rs --seg-kib 64` on the RTX 4090 (driver
595.91.07, CUDA 13.2), 1 GiB of 64 KiB pages in shuffled order:

| direction | per-range | batch | kernel |
|---|---|---|---|
| H2D | 79.6 ms (12.6 GiB/s) | 52.8 ms (19.0 GiB/s) | 41.0 ms (24.4 GiB/s) |
| D2H | 78.4 ms | 57.4 ms | 60.1 ms |

With 4 KiB pages, batch is 2.9-3.5x faster than per-range. Contiguous
copies run the same on both paths.

The b96e80c guard:
- PR #278 removed `cuMemcpyBatchAsync` for "driver instability on some
  CUDA stacks". It recorded no stack and no failure.
- Issue #195 measured a 2.4-5.7x speedup for small scattered H2D pages.
- vLLM 0.32 uses the same call in `swap_blocks_batch` with stream-ordered
  source access.
- One failure mode was found here: the batch call returns
  `CUDA_ERROR_INVALID_VALUE` on the legacy default stream. It works on a
  created stream, which is what GPU workers use. The bench and the kernel
  test use a created stream. `failed_batch_retries_per_range_and_lands_every_range`
  pins the legacy-stream failure as a real fallback trigger.

Fallback design (`transfer/memcpy.rs`). `MemcpyBackend::new()` picks per
range when any of these holds:
- `PEGAFLOW_DISABLE_BATCH_MEMCPY` is set to anything but empty, `0`,
  `false`, `no` or `off`.
- The build is cuda-12 and `PEGAFLOW_ENABLE_BATCH_MEMCPY` is not `1`,
  `true`, `yes` or `on`. No recorded run has used the cuda-12 entry point
  on a 12.x driver, and the code b96e80c removed passed it a null `failIdx`.
  The cuda-12 call now passes a valid slot and puts `failIdx` in the
  error.
- The driver is older than 13.0 (cuda-13 build, `_v2` entry point) or 12.8
  (cuda-12 build).
- libcuda does not export the entry point. cudarc resolves it lazily and
  would panic, so `new()` looks it up first.

When a batch call fails, the same ranges are retried per range:
- A repeated copy just rewrites the same bytes.
- If the retry works, the batch API is at fault on this stack. That
  worker's backend (one per GPU worker) stays per range and warns once.
- If the retry fails too, the ranges are at fault. The transfer returns
  both errors and batching stays on.
- The counter `pegaflow_memcpy_batch_fallbacks{outcome=retry_ok|retry_failed}`
  counts both outcomes. Prometheus scrapes it as
  `pegaflow_memcpy_batch_fallbacks_total`.

At startup the server logs which path it uses ("memcpy transfer
submission: batch ..."). Every H4 run logged batch, and none fell back.
Those runs predate the per-backend latch, the symbol probe and the cuda-12
opt-in. They are cuda-13 builds, so the path they measured is unchanged.

Tests: the `choose` table covers the env values, the cuda-12 opt-in, old
drivers and a missing symbol. The GPU tests check two things:
- the kernel test asserts the batch backend is still on batch after its
  run, so a silent fallback fails it;
- the retry test covers an injected failure, the real legacy-stream
  failure (both latch, every range lands in its own slot), and bad ranges (both
  errors, no latch).

The GPU tests pass on cuda-13 and cuda-12 builds on driver 13.2, which
exports both entry points.

Swap-pair protocol: H4 and H1 run at the same time on GPU6/7 and then swap
GPUs, 32x12 replay, 64 GB pool. Cells are warm2 / warm3 TTFT p50 in
seconds:

| pair | v2 H1 | v2 H4 | old H1 | old H4 |
|---|---|---|---|---|
| `swA` (H4 GPU6, H1 GPU7) | 0.512 / 0.543 | 0.468 / 0.475 | 0.607 / 0.638 | 0.474 / 0.469 |
| `swB` (H4 GPU7, H1 GPU6) | 0.521 / 0.554 | 0.487 / 0.495 | 0.607 / 0.606 | 0.470 / 0.486 |

- H4 is 7-14% below H1 for v2 and 20-26% below H1 for old, in both GPU
  assignments. The old arm, which H3 regressed, is now faster than on H1.
- req/s is 10.3-10.6 on H4 vs 10.0-10.5 on H1.

Other H4 checks:
- Capacity: `full-h4` used the same cap plan as `full-h3` (32 convs, C =
  12.9 GiB). E_C is old 0.539 / v2 0.448 (H3: 0.544 / 0.438), so it did not
  drop. Cap warm passes take 55-56 s for old (H3: 55 s).
- Correctness: `gsm8k-h4` (64 questions) scores 0.8594 on every arm. Every
  gsm8k G2, G3, and G4 verdict passes for both arms, and digests match the
  baseline.
- `full-h4` replay (64 GB pool, 0 evictions on every pass; `full-base` and
  `full-h3` ran at 160 GB): warm2/3 TTFT p50 is old 0.482 / 0.513 s
  and v2 0.481 / 0.512 s. G4 replay passes on both arms.

The official r3 cap plan (64 convs, C(0.5) = 27.2 GB) reruns for base vs H4
are in "G5 at the official cap plan".

## G5 at the official cap plan (gate bf9ef3c, r3 plan)

The r3 plan: 64 convs, U 54.43 GB, and C(0.5) = 27.2 GB = 11536 blocks
(2.44x GPU KV). The numbers above used the older 32-conv plan, so they are
not comparable with these.

Cap-only reruns. The arms are baseline, old, and v2 (8a05b7c via
`--under-test-path`), all with the same cap plan. The GPUs were not
swapped: both base runs used GPU7 and both H4 runs used GPU6. E_C is a hit
metric, so the confound is small, but scheduling interleaving can change
cache state.

E_C over the timed warm passes (C/U 0.25 / 0.5 / 0.75). The `-b` runs
repeat the gated 0.5 point only:

| server (run) | old E_C | v2 E_C | old hit | v2 hit |
|---|---|---|---|---|
| r3 official (semantics-rsi base) | 0.069 / 0.681 / 0.682 | 0.084 / 0.684 / 0.669 | 0.395 / 0.832 / 0.876 | 0.399 / 0.842 / 0.876 |
| base (`cap3-base`) | 0.075 / 0.666 / 0.689 | 0.070 / 0.676 / 0.693 | 0.396 / 0.831 / 0.878 | 0.392 / 0.836 / 0.881 |
| base (`cap3-base-b`) | - / 0.664 / - | - / 0.683 / - | - / 0.828 / - | - / 0.835 / - |
| H4 (`cap3-h4`) | 0.171 / 0.708 / 0.747 | 0.174 / 0.711 / 0.754 | 0.450 / 0.861 / 0.916 | 0.445 / 0.861 / 0.915 |
| H4 (`cap3-h4-b`) | - / 0.716 / - | - / 0.710 / - | - / 0.860 / - | - / 0.861 / - |

- H\*_C is 0.861-0.867 / 0.929-0.930 / 0.970-0.971 on every run.
- H_lru is 0.568-0.594 / 0.857-0.858 / 0.915-0.916.
- At 0.5, H4 (which includes H1 and H3) adds +0.04-0.05 E_C on both arms:
  base runs range 0.664-0.683, H4 runs 0.708-0.716. Hit rate rises from
  0.828-0.836 to 0.860-0.861, which is LRU level (H_lru 0.857).
- At 0.25 and 0.75 the gain is larger: +0.10 and +0.06 E_C. At 0.75, hit
  reaches 0.915-0.916, against H_lru 0.915-0.916.
- Dead memory (pool bytes used minus resident blocks x 2.25 MiB, at pass
  ends):
  - Base at 0.5 holds 1232-2636 dead blocks, 11-23% of the 11536-block pool.
  - At 0.75 it holds 2630-4458 dead blocks.
  - H4 holds 0 on every pass at every C.
- An earlier version reported `pegaflow_cache_resident_bytes` at twice the
  slot bytes per block. That was an analysis artifact, not a server bug.
  - The script summed every metric whose name starts with that prefix, so
    `pegaflow_cache_resident_bytes_by_class` was added on top.
  - The gauge itself is exactly 2359296 bytes per resident block on every
    pass.
  - `sim/cap3.py` now matches exact metric names.
- G5 still fails at 0.9. Per-block allocation brings the store up to
  plain LRU. Getting past that takes a policy change; see below.

Miss attribution at C/U 0.5 uses the r3 traces (base server). The scripts
are in `~/kexi/rsi-tools/server-reclaim/sim/` on 10.1.0.44:
- `r3sim.py` is the server-faithful sim, miss attribution and policies;
- `gap*.py` walk from the gate model to that sim;
- `cap3.py` produces the cap3 table and the dead-memory numbers.

`sim/run-all.sh` reruns them against
`~/kexi/rsi-runs/2026-10-23-qwen3-4b-g5-official-e9f3f33-r3` with the
gate tree bf9ef3c. The outputs and script sha256s are in
`~/kexi/rsi-tools/server-reclaim/out/`.

A server-faithful sim replays each arm's real query window. It reproduces
the logged per-query hit count for 90% (v2) and 84% (old) of timed
requests. Sim E_C is 0.698 / 0.685 (v2 / old), against 0.684 / 0.681
measured.

| class (share of ext\*_C) | v2 | old |
|---|---|---|
| served | 0.698 | 0.685 |
| evicted: the stopping block and the rest of the needed range are gone | 0.278 | 0.270 |
| stranded: head evicted, a later needed block resident | 0.017 | 0.025 |
| query starts past the local hit | 0 | 0.011 |
| scan reached the prompt end | 0.008 | 0.008 |

- At this C, fragmentation is no longer the main loss. Almost every miss
  is a range LRU evicted whole, and stranding costs at most 2.5%. A
  prefix-aware policy that only avoids stranding can win at most that
  much.
- The gate's H_lru bar is 0.859 for both arms. The steps below go from the
  gate's model to the server-faithful sim, scored the gate's way (v2 /
  old):

  | model | v2 | old |
  |---|---|---|
  | gate H_lru model | 0.859 | 0.859 |
  | + H1 reclaim rounds | 0.859 | 0.858 |
  | + save only blocks the connector computed | 0.828 | 0.831 |
  | + the arm's own local hit (lower than baseline's) | 0.584 | 0.682 |
  | + the arm's real query window | 0.698 | 0.685 |

  Most of the gap to H_lru comes from the arm's own local hit. The arm's
  vLLM keeps fewer blocks locally than baseline does, so the store has to
  serve more, and what it serves first is the blocks LRU evicts first. The
  server cannot change that. With the baseline's local hit, the same
  server sim gives v2 0.853.
- Policies in the sim (v2 / old, sim LRU = 0.698 / 0.685):

  | policy | v2 | old |
  |---|---|---|
  | reverse refresh (H2) | 0.704 | 0.688 |
  | segmented LRU, protected 0.5 | 0.717 | 0.691 |
  | segmented LRU, protected 0.8 / 0.9 | 0.615 / 0.550 | 0.616 / 0.548 |
  | tail-first among the 1024 oldest | 0.734 | 0.720 |
  | tail-first among the 2048 oldest | 0.742 | 0.728 |
  | tail-first among the 4096 oldest | 0.742-0.747 | 0.728 |
  | tail-first among the 8192 oldest | 0.596 | 0.589 |

  "Tail-first" evicts the deepest chain position in an LRU window. It is
  the only candidate that beats LRU on both arms, by +0.04-0.05. Even
  that is far from 0.9, and its window is a tuning knob with a cliff.
  Scan resistance (SLRU) helps only at a small protected share and hurts
  beyond it.
- The sim has no allocator, so its LRU (0.698 / 0.685) is what a
  fragmentation-free server should reach. H4 measures 0.710-0.711 /
  0.708-0.716, so it is at that level, and the remaining gap to H_lru is
  the arm's own local hit, not the server.
- Next hypothesis (H5, not started): tail-first eviction within a window
  of the oldest blocks, model-agnostic (chain position only). The sim
  predicts about +0.04 on both arms at 0.5. That is a policy change for a
  modest gain, which the plain-LRU preference has to weigh. It needs a gate
  run before it can be claimed.

## Tests and checks (10.1.0.44, RTX 4090, driver 595.91.07, CUDA 13.2)

An earlier version of this section reported a run that tested master. The
sync had copied the main checkout instead of this worktree, so `memcpy.rs`,
`gpu_worker.rs`, `storage/mod.rs` and `offload.rs` on the remote matched
75dfa4c. That run is withdrawn.

Runs are now made from a `git archive` of the commit, unpacked into a
dedicated `~/kexi/pegaflow-rsi-server-reclaim-src[-<rev7>]`, with the commit
in `.rsi-rev`.
The script is `~/kexi/rsi-tools/server-reclaim/verify.sh <src> 6`, and it
writes its logs to `~/kexi/logs/rsi-server-reclaim-verify-<rev7>/`. Each
`env.txt` records the sha256 of those four files, and they match
`git show <rev>:<file>`.

Run at 67922b5 (code = ef53344; GPU6):
- `cargo-test.log`: `cargo test --no-default-features --features
  cuda-13,rdma --no-fail-fast`, workspace.
  - pegaflow-core lib: 175 passed, 2 ignored. The branch's tests ran and
    passed: `each_saved_block_owns_one_region_across_layers`,
    `reclaim_frees_what_the_allocation_needs_not_a_fixed_batch`,
    `duplicate_and_resident_hashes_are_not_saved`, and
    `failed_allocation_returns_every_region`.
  - pegaflow-core integration: engine_basic 10, eviction 3, hybrid_groups 3,
    prefetch_lease 2, prefix_semantics 5, save_validation 2.
  - `ssd_cache` fails all 11 with "pinned pool exhausted while allocating
    ... buffer". Master 75dfa4c fails the same 11 on this host, so the
    failure is environmental (as on .117).
  - pegaflow-server: lib 24, mock_vllm_rpc_e2e 8, http_cleanup_hang_repro 1.
    Its binaries need the uv CPython 3.12 lib dir on `LD_LIBRARY_PATH`.
  - pegaflow-common 48, pegaflow-metaserver 52, pegaflow-semantics 13 + 3,
    pegaflow-transfer 17, pegaflow-pd-wire 7.
- `cargo-test-ignored.log`: `cargo test -p pegaflow-core --lib -- --ignored`
  on GPU6. Both pass: `failed_submit_drains_copies_already_enqueued`, and
  `kernel_matches_direct_both_directions` (kernel, per-range, and batch).
- `clippy-cuda13.log`: clippy `-D warnings` with `cuda-13,rdma`. Clean.
- `check.log`: `./scripts/check.sh` with ruff 0.14.14 (the prek pin).
  - cargo fmt passes.
  - typos is not installed, and the script skips it.
  - `ruff format --check python/` fails on 11 files this branch does not
    touch. Master 75dfa4c fails on the same 11.
- `check-rest.log`: the check.sh steps after the failure, run on their own:
  `ruff check python/`, clippy `-D warnings` (default features), and
  `cargo check` for pegaflow-core and pegaflow-server. All pass.

Run at 6a99d43, the final code, including the H4 review fixes 4a51e22,
46e7bb9, 29158e4, 3f42b7a and 6a99d43. GPU6, logs in
`~/kexi/logs/rsi-server-reclaim-verify-6a99d43/`. The `env.txt` sha256s
match `git show 6a99d43:<file>`, for example `memcpy.rs` 41cea3e3.
- `cargo-test.log` (rc 101, `ssd_cache` only, as above):
  - pegaflow-core lib: 177 passed, 4 ignored, including
    `env_switches_err_towards_off` and
    `build_env_and_driver_pick_the_submission_path`;
  - every other crate and integration target has the same counts as
    67922b5.
- `cargo-test-ignored.log` (cuda-13 build): 4 passed. The four tests are
  `failed_submit_drains_copies_already_enqueued`,
  `kernel_matches_direct_both_directions` (it fails on a silent fallback),
  `failed_batch_retries_per_range_and_lands_every_range` and
  `installed_driver_exports_the_batch_symbol`.
- `cargo-test-ignored-cuda12.log`: the three transfer GPU tests pass on a
  default-features (cuda-12) build. They use the build's
  `cuMemcpyBatchAsync` entry point with `failIdx` on driver 13.2. No 12.x
  driver was run.
- `clippy-cuda13.log`: clean.
- `check.log` (rc 1): the same 11 `ruff format` files as on master.
- `check-rest.log` (rc 0): the steps after that failure all pass.

## Server bug candidates (low priority, not on this branch)

- Teardown takes about 0.36 s/GiB of pinned pool. Measured on a standalone
  `pegaflow-server`:
  - On SIGTERM the process exits at once (rc -15) and is a zombie within
    2 s.
  - `waitpid` returns only after 58 s for a 160 GB pool, and 16 s for 40 GB.
    That time is the kernel/driver releasing the `cudaHostAlloc` pages.

  This exceeded the gate's old 60 + 30 s budget in full-h1b, slice-h1b, the
  first full-h3 and it2d. The gate fix is d646ac3 (feat/rsi-g5). This
  branch's runs from the full-h3 rerun on use a gate copy with it
  backported. Hugepage backing would likely shorten the release (fewer
  pages); not measured.
- The server ignores SIGTERM. It handles only ctrl_c (SIGINT) and the
  shutdown RPC (`pegaflow-server/src/lib.rs`, `shutdown_signal`), so under
  SIGTERM no shutdown code runs. Orchestrators such as k8s send SIGTERM, so
  graceful shutdown should cover it.

## Open risks

- H1's divisor of 64 was chosen, not tuned: in the sim, /16, /32 and /64 all
  beat fixed 512-block rounds by similar margins.
- H3 changes the allocation count per layer-first save call. It was 36 per
  call (one per layer for Qwen3-4B) and is now one per saved block, so a
  call saving more than 36 blocks makes more allocations. Save duration did
  not move.
- Batch memcpy has run only on driver 595.91.07 / CUDA 13.2 (RTX 4090).
  The stack b96e80c hit is unknown.
  - A failing batch call falls back to per range, but a call that returns
    success and then misbehaves would not. `PEGAFLOW_DISABLE_BATCH_MEMCPY=1`
    is the kill switch. Alert on any increase of
    `pegaflow_memcpy_batch_fallbacks_total`.
  - Before rollout, run one replay on the production driver.
  - cuda-12 builds stay per range until someone runs them with
    `PEGAFLOW_ENABLE_BATCH_MEMCPY=1` on a 12.8+ driver.
- Each region is one allocator node. `PinnedMemoryPool::MAX_ALLOCS` is
  4,000,000 per shard, so regions under about 40 KiB on a 160 GB pool
  would run out of nodes before bytes. Today's regions are far larger.
- SSD prefetch still restores blocks into shared chunks of up to 256 MiB
  (`ssd_cache.rs`), so a restored block can keep a chunk pinned after
  eviction, the pattern H3 removed from saves.
- Not measured: models with several KV groups, TP > 1, and SSD-enabled
  pools.
