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

Non-regression for H3 (replay always runs on the gate's 160 GB pool, so
reclaim never runs there):
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
    on H1 and 17-28 s on H3. The cause is not identified; see Open risks.
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
- H3 costs warm replay TTFT when the pool is not under pressure (+10-17%
  p50 for v2, see Evidence); throughput is unchanged. The leading candidate
  is lost copy coalescing.
  - The memcpy backend merges copies only when they are adjacent on both
    sides and in the same host allocation.
  - Before H3, one layer's blocks from a save call shared an allocation and
    were host-adjacent. A copy over those blocks into consecutive GPU blocks
    could merge into one copy per layer and segment.
  - Now each block-layer segment is a separate copy of
    block bytes / (layers x segments), which is 32 KiB for Qwen3-4B.

  Not verified: copy counts were not instrumented. Possible follow-ups:
  - Pick the kernel backend for fragmented batches. It issues one launch
    regardless of the fragment count. Today the connector chooses the
    backend per model at registration.
  - Allow merging across allocations that are adjacent in one pinned
    backing.

  Either one needs its own measurement.
- Not measured: models with several KV groups, TP > 1, and SSD-enabled
  pools.
