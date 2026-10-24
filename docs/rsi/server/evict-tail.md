# Server: tail-first eviction (H5, branch `feat/rsi-server-evict-tail`)

Stacked on `feat/rsi-server-reclaim` (H1-H4, see `reclaim-eviction.md`).
Trigger: G5 on Qwen3-4B still fails at 0.9 after H4. H4 brings the store to
plain-LRU level, and the H4 sim showed one policy beating LRU on both arms:
evict the deepest chain block among the oldest blocks.

## What changes

`--tail-window-percent` (default 25, 0 = plain LRU). Pressure eviction, per
replacement class (reclaimable first, then retained):
1. The window is the oldest `max(25% of the class, blocks still wanted)`
   blocks.
2. Candidates are taken deepest first. Ties go to the oldest. A block whose
   depth was never observed counts as deepest.
3. Taking a candidate walks down to its deepest resident descendant and
   evicts that leaf, then the next, up to the candidate. A block is never
   evicted while a block that depends on it is resident.
4. If the next leaf is pinned (query lease, transfer lock, backup copy),
   the subtree is skipped. If pins cover the whole window, the class's
   other leaves are tried in LRU order.

Depth and links come from what the server sees, with no model knowledge:
- A group 0 prefix query (`query_prefetch`, group 0) marks each block of its
  resident hit prefix with its index in the query and links it to the
  previous block. The index is relative to the query offset, so it orders
  blocks within a chain. That is all eviction needs.
- Remote prefix-only transfers and membership fetches learn nothing. Their
  key lists are not this node's chains.
- Links exist only while both ends are resident. Evicting a block drops its
  links.
- `pegaflow_cache_evictions_with_resident_children` counts evictions of a
  block that still had a resident dependent. Tail-first keeps it at 0;
  plain LRU (window 0) does not.

## Recency refresh in the server (checked in code)

| operation | refreshes LRU recency | code |
|---|---|---|
| prefix query (`query_prefetch`, group 0) | yes, each block of the hit prefix | `ReadCache::get_prefix_blocks` via `prefetch.rs` `full_prefix_scan` |
| membership query (group > 0) | yes, each hit | `get_blocks_aligned` via `get_membership` |
| membership fetch (group > 0, with fetch) | yes, each block of the hit prefix | `get_prefix_blocks` |
| serving a remote transfer | yes | `get_blocks` / `get_prefix_blocks` |
| re-saving a resident block | yes | `insert_block`, `AlreadyExists` |
| query lease (pin) | no; it holds an `Arc`, which blocks eviction until consumed | `lease.rs` |
| load (`load` RPC) | no; it consumes the lease and never touches the read cache | `lib.rs` `batch_load_kv_blocks_multi_layer_inner` |
| backup scan, residency checks | no | `peek_blocks`, `contains_keys` |

So the only signal of use is the query. A block under vLLM's local hit is
refreshed only if the connector queries it:
- v2 queries from block 0, so its heads stay fresh.
- old queries from its local hit, so its head blocks are queried once (when
  vLLM first misses them) and then age out while their descendants stay hot.
  Under LRU every later block of that chain is then dead weight.

Tail-first fixes the old case structurally: the head has resident children,
so it is never a victim before them.
`queried_never_loaded_head_outlives_its_resident_descendants` checks this.
The head is queried once and never loaded, then the chain's tail is queried
again. Eviction takes the three descendants one by one, deepest first, and
the head stays resident until the last of them is gone.

## Group coherence

The server knows group ids, not kinds. A resume point at position p needs
group 0's chain up to p and every other group's data that is needed at p.
What that data is depends on the kind: Full needs every block, Window the
last few, Checkpoint only p. The server cannot tell which.

The rule is: a group block never outlives its position.
- A membership query or fetch for group g > 0 ties each resident
  `key(hash, g)` to the resident group 0 block `hash`. The group block
  becomes a leaf under its position and takes the position's depth.
- So when a position is chosen, its group blocks go before it. A position is
  never evicted while its group data stays as weight no resume point can
  use.
- A group block may leave on its own before its position. A window group's
  early blocks are never needed again, so they should be able to go, and a
  rule that kept group data together with its position would pin them.
- What this does not do: it does not evict a position because one of its
  Full group blocks went. That would need the kind.

The first commit (7791c30) took the position along with a group block.
81fdb6d dropped that: for a window group, it would have evicted attention
blocks that deeper resume points still needed. The group path is covered
by `group_blocks_leave_before_their_position_and_pins_hold_the_chain`. It
is not exercised by Qwen3-4B, which is one Full group.

## Choosing the window (sim)

Sim: `~/kexi/rsi-tools/server-reclaim/sim/h5sim.py` on 10.1.0.44. It
replays each arm's real query window from the r3 traces (H1 rounds, no
allocator, which is H4's level) and scores E_C the gate's way. Outputs are in
`~/kexi/rsi-tools/server-reclaim/out/h5*/`. The server policy is `subfg`:
depth from the hit prefix, links from consecutive hits that exist only while
both ends are resident, subtree descent.

E_C, v2 / old:

| policy | C/U 0.25 | 0.5 | 0.75 |
|---|---|---|---|
| LRU | 0.123 / 0.133 | 0.698 / 0.685 | 0.738 / 0.740 |
| reverse refresh (H2) | | 0.704 / 0.688 | |
| deepest in the oldest 4096 blocks, no descent | | 0.762 / 0.726 | |
| deepest in the oldest 8192, no descent | | 0.633 / 0.597 | |
| deepest in the oldest 25%, no descent | | 0.764 / 0.734 | |
| strict: only leaves, no descent, 25% | | 0.580 / 0.554 | |
| head eviction frees the whole chain, 25% | | 0.766 / 0.613 | |
| H5 (`subfg`), 25% | 0.189 / 0.161 | 0.767 / 0.753 | 0.772 / 0.765 |
| H5 (`subfg`), 35% | 0.219 / 0.199 | 0.763 / 0.748 | 0.789 / 0.783 |
| H5 without forgetting links, 50% | 0.265 / 0.239 | 0.697 / 0.698 | 0.806 / 0.803 |
| H5 without forgetting links, 70% | 0.304 / 0.289 | 0.628 / 0.600 | 0.813 / 0.813 |

Miss attribution at 0.5 for H5 25%: v2 served 0.748, evicted 0.236,
stranded 0.003; old served 0.734, evicted 0.233, stranded 0.010. Against
LRU's 0.017 / 0.025 stranded, H5 removes most of the stranding and also
evicts less of what is needed next.

Why a large window falls off at 0.5: the window is the candidate set. With
8192 blocks (0.71 of the 11536-block cache) or 50-70%, the deepest blocks in
it are the tails of chains used recently, not of chains going cold. Victims
move from the LRU end toward the middle (mean victim rank 0.42-0.59 of the
LRU order, against 0.22 at 25%). The evicted tails are needed by the next
turn of the same conversation, and the stranded share rises to 0.06-0.21.
At 0.75 the cache holds most of the working set, so recent tails are rarely
needed and larger windows keep helping. At 0.25 the cache is far too small
and evicting deeper at all helps. 25% is the largest window that does not
lose at 0.5 on either arm, so it is the default.

Rejected:
- Strict leaves without descent picks leaves that are often the only block
  of a short chain, and leaves inner blocks of long chains to age into
  stranding.
- Freeing the whole chain on a head eviction (cascade) puts old 0.07 below
  LRU: old's heads are its oldest blocks, so the cascade discards hot
  chains.

## Result

At the gated C/U 0.5, H5 raises E_C from 0.699-0.719 (H4) to 0.763-0.769
on both arms, and hit from 0.861 to 0.889. That is above LRU (H_lru 0.857)
and 0.04 below the offline bound (H\*_C 0.929). At 0.75 it gains
+0.01-0.03 E_C. At 0.25 it loses 0.01-0.05 E_C, with hit unchanged; this is
recorded as a tradeoff below. No-cap warm TTFT and gsm8k do not regress.

## Gate (G5 cap, official r3 plan)

Gate bf9ef3c, v2 8a05b7c via `--under-test-path`, arms baseline, old, v2,
`--workloads cap`, official cap fractions 0.25 / 0.5 / 0.75. Servers:
- H4: `bin/pegaflow-server-h4-d45a366`, sha256 b2411b45…, built from a
  `git archive` of d45a366;
- H5: `bin/pegaflow-server-h5`, sha256 ace342eb…, built from a `git archive`
  of 7791c30. `-w0` / `-w10` are wrapper scripts that add
  `--tail-window-percent`.

The 81fdb6d change only touches the group > 0 path, which Qwen3-4B never
takes, so the 7791c30 binary measures the same group 0 policy.

Runs, all on 2026-10-24. Logs and meta are in
`~/kexi/rsi-runs/srv-evict-tail-<tag>.{log,meta}`, run dirs in
`~/kexi/rsi-runs/2026-10-24-qwen3-4b-srv-evict-tail-<tag>`. Each pair ran
at the same time, and the second pair swapped GPUs:

| tag | server | GPU | pgid | UTC start - end |
|---|---|---|---|---|
| `cap3-h5` | H5 | 6 | 2385306 | 04:06:14 - 05:12:52 |
| `cap3-h4` | H4 | 7 | 2385307 | 04:06:14 - 05:12:46 |
| `cap3-h5-b` | H5 | 7 | 2432239 | 05:13:00 - 06:19:13 |
| `cap3-h4-b` | H4 | 6 | 2432240 | 05:13:00 - 06:19:33 |
| `replay-h5` | H5, gate `.gate` (369b253 + teardown backport), 64 GB pool | 6 | 2483930 | 06:19:33 - 06:34:48 |
| `gsm8k-h5` | H5, gate `.gate`, 64 GB pool, limit 64 | 7 | 2483492 | 06:19:13 - 06:26:28 |
| `cap-h5-w10` | H5, `--tail-window-percent 10`, C/U 0.25 + 0.5 | 7 | 2540426 | 06:47:56 - 07:37:45 |
| `cap-h5-w0` | H5, `--tail-window-percent 0` (plain LRU), C/U 0.25 + 0.5 | 6 | 2541289 | 06:48:23 - 07:37:56 |

Every run exits rc 1. The gate marks these runs non-official ("arm plain
missing", partial workloads), and G5 fails. The run records are complete.
An earlier `cap025-h5-w0` launch exited at once on an argument error
(`--cap-fractions` must include 0.5) and ran nothing.

Timed warm passes (warm2 + warm3), old / v2. The `-b` rows are on the other
GPU:

| C/U | server (run) | E_C | hit | H\*_C | H_lru |
|---|---|---|---|---|---|
| 0.25 | H4 (`cap3-h4`) | 0.174 / 0.194 | 0.451 / 0.459 | 0.865 / 0.863 | 0.573 / 0.589 |
| 0.25 | H4 (`cap3-h4-b`) | 0.179 / 0.170 | 0.458 / 0.442 | 0.867 / 0.864 | 0.590 / 0.598 |
| 0.25 | H5 (`cap3-h5`) | 0.129 / 0.159 | 0.430 / 0.464 | 0.863 / 0.862 | 0.575 / 0.580 |
| 0.25 | H5 (`cap3-h5-b`) | 0.147 / 0.163 | 0.457 / 0.458 | 0.864 / 0.862 | 0.593 / 0.581 |
| 0.5 | H4 (`cap3-h4`) | 0.699 / 0.719 | 0.861 / 0.861 | 0.930 / 0.929 | 0.858 / 0.857 |
| 0.5 | H4 (`cap3-h4-b`) | 0.702 / 0.713 | 0.860 / 0.861 | 0.930 / 0.929 | 0.857 / 0.857 |
| 0.5 | H5 (`cap3-h5`) | 0.763 / 0.763 | 0.888 / 0.889 | 0.930 / 0.929 | 0.858 / 0.857 |
| 0.5 | H5 (`cap3-h5-b`) | 0.769 / 0.768 | 0.889 / 0.889 | 0.930 / 0.929 | 0.858 / 0.857 |
| 0.75 | H4 (`cap3-h4`) | 0.748 / 0.757 | 0.916 / 0.917 | 0.971 / 0.971 | 0.915 / 0.916 |
| 0.75 | H4 (`cap3-h4-b`) | 0.753 / 0.755 | 0.916 / 0.915 | 0.971 / 0.970 | 0.915 / 0.915 |
| 0.75 | H5 (`cap3-h5`) | 0.766 / 0.792 | 0.931 / 0.935 | 0.971 / 0.971 | 0.917 / 0.916 |
| 0.75 | H5 (`cap3-h5-b`) | 0.760 / 0.788 | 0.930 / 0.935 | 0.971 / 0.971 | 0.916 / 0.916 |

- At 0.5, H5 adds +0.06 E_C on both arms (H4 0.699-0.719, H5 0.763-0.769).
  Hit rises from 0.860-0.861 to 0.888-0.889.
- At 0.75, H5 adds +0.01-0.03 E_C, and hit rises from 0.915-0.917 to
  0.930-0.935.
- At 0.25, E_C drops: old 0.174-0.179 to 0.129-0.147, v2 0.170-0.194 to
  0.159-0.163. Hit does not move beyond run-to-run spread (H4
  0.442-0.459, H5 0.430-0.464). See the next section.
- Both servers have 0 dead blocks at every pass end.
  `pegaflow_cache_evictions_with_resident_children` is 0 on every H5 run,
  so H5 never evicted a block that had a resident dependent.
- Cap warm TTFT p50 does not regress: at 0.5, H5 is 2.07-2.28 s (old) and
  2.10-2.17 s (v2), against H4's 2.22-2.30 and 2.20-2.28 s.

Miss attribution on each run's own traces (`sim/attr.py`) replays the run's
logged queries through the sim policy that matches the server: LRU for H4,
H5 25% for H5. "match" is the share of timed requests whose sim hit equals
the logged hit. Shares are of ext\*_C:

| C/U | run | match | served | evicted | stranded |
|---|---|---|---|---|---|
| 0.5 | H4 old (`cap3-h4`, `-b`) | 0.89 / 0.87 | 0.637 / 0.644 | 0.286 / 0.283 | 0.060 / 0.060 |
| 0.5 | H5 old (`cap3-h5`, `-b`) | 0.65 / 0.81 | 0.644 / 0.712 | 0.232 / 0.230 | 0.100 / 0.038 |
| 0.5 | H4 v2 (`cap3-h4-b`) | 0.96 | 0.692 | 0.273 | 0.027 |
| 0.5 | H5 v2 (`cap3-h5-b`) | 0.78 | 0.688 | 0.232 | 0.068 |
| 0.25 | H4 old (`cap3-h4`, `-b`) | 0.91 / 0.86 | 0.173 / 0.195 | 0.641 / 0.601 | 0.180 / 0.198 |
| 0.25 | H5 old (`cap3-h5`, `-b`) | 0.27 / 0.27 | 0.134 / 0.143 | 0.531 / 0.627 | 0.327 / 0.223 |

- The LRU sim reproduces H4 well (match 0.86-0.96). The H5 sim reproduces
  H5 at 0.5 and 0.75 (0.65-0.82) but not at 0.25 (0.20-0.38). The sim's
  depth model is the hit prefix of the latest query, and the server's
  depths drift from it once a chain is partly evicted and requeried. So
  attribution of the H5 runs is approximate, and at 0.25 not reliable.
- At 0.5, H5 cuts the "evicted" class from 0.28 to 0.23: it evicts less of
  what is needed next. That is where the gain comes from.
- Outputs: `~/kexi/rsi-tools/server-evict-tail/out/attr/`.

### The 0.25 regression

What the runs show:
- It is the policy, not the build. The same binary with
  `--tail-window-percent 0` (plain LRU) gives 0.179 / 0.180 at 0.25 and
  0.698 / 0.706 at 0.5, both at H4's level. It also logs 38-58k evictions
  with resident children per run, which default H5 never does.
- It is not the window size. 10% gives 0.152 / 0.161 at 0.25 and 0.734 /
  0.735 at 0.5. A smaller window does not recover 0.25, and it gives up
  half of the 0.5 gain.
- It shows in E_C, not in hit. Server-served tokens past the baseline's
  local hit (the part E_C credits) drop from 360-376 to 264-307 per request
  (old), and from 366-420 to 337-352 (v2) (`sim/split.py`). Hit rate counts
  every served token, and stays within the run spread.
- In the first pair's old H5 run, 22% of the credited blocks had never been
  hit by an earlier query, against 6-7% in the other three 0.25 runs of
  that pair (`sim/unknown.py`). Those blocks have unknown depth, which H5
  evicts first.

Likely cause (old arm): at 0.25 the cache holds about a quarter of the
working set, so a conversation's blocks rarely survive past its next turn.
Much of what can still serve was written by the previous turn and not yet
requeried, so it has unknown depth, and H5 puts unknown depth first in
line. This fits the old numbers. It does not explain v2's smaller drop
(0.01-0.03), which is inside H4's own v2 spread at 0.25 (0.170-0.194). Ranking unknown depth last instead (sim `subfg-sh`) is worse
at every capacity: 0.640 / 0.702 at 0.5 against 0.753 / 0.767, because
then decode tails and never-requeried blocks pile up. The sim cannot model
0.25 well enough (match 0.20-0.38) to tune a rule for it, and a rule keyed
to eviction pressure would need its own gate runs.

Tradeoff, not fixed: at 0.25 the store serves little under any policy (E_C
0.13-0.19, hit 0.43-0.46 against H_lru 0.57-0.60). H5 loses 0.01-0.05 E_C
there and no measurable hit. At 0.5 it gains +0.06 E_C and +0.03 hit, and
at 0.75 +0.01-0.03 E_C and +0.015 hit. The default stays 25. A deployment
that runs far below its working set can use `--tail-window-percent 0`.

### Hit rate: close enough?

There is no hard G5 threshold any more (user decision, 2026-10-24). G5's
E_C >= 0.9 is reported only. Measured at the gated 0.5, both arms:
- hit 0.888-0.889, against H\*_C 0.929-0.930 (Belady over the same
  accesses) and H_lru 0.857-0.858. H5 is 0.03 above the LRU reference and
  0.04 below the offline bound.
- At 0.75: 0.930-0.935 against H\*_C 0.971 and H_lru 0.915-0.917, so 0.04
  below the bound.
- E_C 0.763-0.769 is lower than the hit ratio suggests because it counts
  only tokens past the baseline's local hit. The arm's own vLLM keeps fewer
  blocks locally than the baseline does (see `reclaim-eviction.md`), so
  the store serves more of the prefix, and E_C credits none of that part.

In E_C terms, most of what H5 still misses is the "evicted" class (0.23 of
ext\*_C): whole needed ranges gone because the cache is half the working
set. Belady keeps them because it knows which conversation comes next. An
online policy can only guess that from recency and chain shape, which H5
already uses. The rest is stranded (0.04-0.10 in the approximate
attribution) and scan end (0.01).

Judgment: at 0.5 and 0.75 the hit rate is close enough. It beats LRU and is
within 0.04 of the offline bound. At 0.25 neither policy is close (hit
0.43-0.46 against 0.86); that point is capacity-bound, and H5 does not
change its hit rate.

## Warm TTFT and correctness (no-cap)

`replay-h5`, 32x12 replay, 64 GB pool, 0 evictions, so this measures the
query-path cost of depth and link tracking, not eviction. Against
`full-h4` (H4, same gate tree, same GPU6):

| arm | server | warm2 / warm3 TTFT p50 (s) | warm2 / warm3 req/s | cold TTFT p50 |
|---|---|---|---|---|
| old | H4 (`full-h4`) | 0.482 / 0.513 | 10.71 / 10.59 | 1.285 |
| old | H5 (`replay-h5`) | 0.496 / 0.495 | 10.36 / 10.62 | 1.274 |
| v2 | H4 (`full-h4`) | 0.481 / 0.512 | 10.27 / 10.22 | 1.222 |
| v2 | H5 (`replay-h5`) | 0.482 / 0.463 | 10.33 / 10.29 | 1.201 |

G4 passes on both arms, and G2 replay digests match. No warm-TTFT
regression. `gsm8k-h5`: accuracy 0.8594 on every arm, cold and warm, as on
H4 (`gsm8k-h4`). G2 gsm8k digests match baseline.

## Tests and checks (10.1.0.44, RTX 4090, driver 595.91.07, CUDA 13.2)

From a `git archive` of 81fdb6d in
`~/kexi/pegaflow-rsi-server-evict-tail-src-81fdb6d` (tar sha256 4c2382e6…,
same on both ends). `~/kexi/rsi-tools/server-reclaim/verify.sh <src> 6`,
logs in `~/kexi/logs/rsi-server-reclaim-verify-81fdb6d/`. The `env.txt`
sha256s match `git show 81fdb6d:<file>`.
- `cargo-test.log` (rc 101, `ssd_cache` only): pegaflow-core lib 180
  passed, 4 ignored. That includes
  `tail_first_evicts_deepest_block_of_oldest_window_first`,
  `queried_never_loaded_head_outlives_its_resident_descendants` and
  `group_blocks_leave_before_their_position_and_pins_hold_the_chain`. Every
  other target passes, 378 tests passed in all. `ssd_cache` fails all 11 with
  "pinned pool exhausted", as on master and every earlier branch run on
  this host.
- `cargo-test-ignored.log`: 4 passed. `cargo-test-ignored-cuda12.log`: 3
  passed.
- `clippy-cuda13.log`: clean.
- `check.log` (rc 1): cargo fmt passes, and typos is not installed. `ruff
  format --check` fails on the same 11 untouched files as master.
- `check-rest.log` (rc 0): `ruff check`, clippy with default features, and
  `cargo check` all pass.

The commit after 81fdb6d only changes docs and the `--tail-window-percent`
help text (it now points at this doc).

## Open risks

- 0.25 regression, above.
- Memory: a linked block costs a parent-map entry (two `BlockKey`s) and a
  slot in its parent's children list, a few hundred bytes with 32-byte
  hashes. That is under 0.02% of a 2.25 MiB block, but it grows with the
  resident count, not bytes.
- A pressure round scans the window under the cache lock: it collects w
  entries (w = 25% of the class, about 2.9k at 11.5k blocks), selects the
  deepest 4 x batch, and sorts only those. The cap runs show no TTFT cost,
  but a 160 GB pool of small blocks scans far more per round.
- Group > 0 linking is unit-tested only. No hybrid model has run on H5.
- Depth is the hit-prefix index of the latest query, so it shifts when a
  chain is requeried from another offset (old: 6-21% of queries start past
  its local hit, 6-10% before it; `sim/offsets.py`). Order within a chain stays right, but depths of two
  chains are not comparable. The gate result is the evidence this is good
  enough.
