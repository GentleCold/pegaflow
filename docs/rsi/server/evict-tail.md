# Server: tail-first eviction (H5, branch `feat/rsi-server-evict-tail`)

Stacked on `feat/rsi-server-reclaim` (H1-H4, see `reclaim-eviction.md`).
Trigger: G5 on Qwen3-4B still fails at 0.9 after H4. H4 brings the store to
plain-LRU level, and the H4 sim showed one policy beating LRU on both arms:
evict the deepest chain block among the oldest blocks.

## What changes

`--tail-window-percent`, default 0 (plain LRU, opt-in). The evaluated
setting is 25, and every gate run passes it explicitly. Default 0 follows
the review (`docs/rsi/reviews/server-evict-tail-h5.md` §7): no hybrid model
has run with the window on, the group > 0 rule depends on a client
contract (§Group coherence), and old loses E_C at C/U 0.25 for a reason
that is not understood. Revisit 25 as the default after one hybrid model
passes G5 cap with it on.

Pressure eviction, reclaimable class first, then retained:
- The reclaimable class is always plain LRU. Each of its blocks has a copy
  on another node (decode backup, a finished P2P transfer, a reclaim hint),
  so dropping a head there strands nothing that RDMA prefetch cannot
  refill. Before review fix M1 the reclaimable pass walked into retained
  descendants. With decode backup it then dropped the cluster's only copy
  of a chain's tail to drop a backed-up head, and freed retained bytes, so
  backup reporting stalled.
- The retained class, with a window set:
  1. The window is the oldest `max(min(25% of the class, 4096), blocks still
     wanted)` blocks. The 4096 cap (review M2/S2) bounds the scan under
     the cache lock. In the sim the oldest 4096 scored 0.762 / 0.726 at
     0.5, against 0.764 / 0.734 for 25% (§Choosing the window). On the
     gated Qwen3-4B caches (5.8k / 11.5k / 17.3k blocks) 25% is under the
     cap at 0.25 and 0.5, and slightly over at 0.75.
  2. Candidates are taken deepest first. Ties go to the oldest. A block
     whose depth was never observed counts as deepest.
  3. Taking a candidate walks down to its deepest retained descendant and
     evicts that leaf, then the next, up to the candidate. The walk never
     follows a reclaimable child, and a block whose children are all
     reclaimable is a leaf. A retained block is never evicted while a
     retained block that depends on it is resident.
  4. If the next leaf is pinned (query lease, transfer lock, backup copy),
     the subtree is skipped. If pins cover every candidate, the oldest
     4096 retained entries are searched for unpinned leaves, and only the
     victims are cloned. If that finds nothing either, the round falls
     back to plain LRU, so reclaim never stops where H4 would progress.
     Those evictions can strand descendants and show in the counter below.

Lock hold per pressure round, from the ignored bench
`tail_first_round_lock_hold_at_2_6m_blocks` (2.6M resident blocks in
64-block chains, batch 512, 20 rounds, release build, 10.1.0.44; the
time is `remove_lru_batch` including its metrics, so it is an upper
bound):

| case | plain LRU | tail-first 25%, capped | tail-first 25%, uncapped (pre-M2) |
|---|---|---|---|
| nothing pinned, mean / max | 0.81 / 0.89 ms | 1.45 / 1.81 ms | 19.6 / 24.8 ms |
| oldest 8192 pinned, mean / max | 0.95 / 3.5 ms | 1.97 / 12.2 ms | 19.8 / 24.8 ms |

The uncapped column is the same bench with the cap set to `usize::MAX`, a
one-off patch in the remote wip tree that was reverted and not committed.
With the oldest 8192 pinned, the whole 4096-block window is pinned. By the
code path (the bench does not time rounds separately), the first round
finds no candidate and no unpinned leaf, and falls back to plain LRU. That
LRU pass moves the pinned blocks to the MRU end, as H4 does, which is why
both policies have their max there. Later rounds see an unpinned window.

Depth and links come from what the server sees, with no model knowledge:
- A group 0 prefix query (`query_prefetch`, group 0) marks each block of its
  resident hit prefix with its index in the query and links it to the
  previous block. The index is relative to the query offset, so it orders
  blocks within a chain. That is all eviction needs.
- Remote prefix-only transfers learn nothing, and membership fetches learn
  no chain links (only the group links of §Group coherence). Their key
  lists are not this node's chains.
- Links exist only while both ends are resident. Evicting a block drops its
  links, and full cleanup resets them. The eviction-order table test
  checks that both link maps are empty once every block has gone, in
  several orders, including parent-first via the reclaimable LRU.
- The invariant covers known dependents only: a block is never evicted
  while a retained block linked under it is resident. Some blocks have no
  parent link yet, and their parent can go first and strand them:
  - a newly saved block that no group 0 prefix query has hit yet (the
    newest turn, decode tails, SSD/RDMA prefetch inserts);
  - the children of a head that was evicted and saved again. They lost
    their link when the head left, and stay unlinked until a query hits
    through them.

  This is the 0.04-0.10 stranded share in the H5 attribution.
- `pegaflow_cache_evictions_with_resident_children` counts pressure
  evictions of a block that still had a known resident dependent. It is
  nonzero under plain LRU (window 0), in the reclaimable class's LRU, and
  in a tail-first round that falls back to LRU because pins cover the
  window. Full cleanup never counts.

## Recency refresh in the server (checked in code)

| operation | refreshes LRU recency | code |
|---|---|---|
| prefix query (`query_prefetch`, group 0) | yes, each block of the hit prefix | `ReadCache::get_prefix_blocks` via `prefetch.rs` `full_prefix_scan` |
| membership query (group > 0) | yes, each hit | `get_group_members` via `get_membership` |
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
  becomes a leaf under its position and takes the position's depth. The
  membership query links its hits under the lookup's own lock, and builds
  position keys for hits only.
- A position no prefix query has covered has unknown depth, so its linked
  members also take unknown depth and rank first. With a hybrid model,
  members of positions that were saved but not yet requeried are therefore
  early victims.
- So when a position is chosen, its group blocks go before it. A position is
  never evicted while its group data stays as weight no resume point can
  use. This holds within the retained class. A demoted (reclaimable) block
  is evicted by plain LRU, whatever is linked to it.
- A group block may leave on its own before its position. A window group's
  early blocks are never needed again, so they should be able to go, and a
  rule that kept group data together with its position would pin them.
- What this does not do: it does not evict a position because one of its
  Full group blocks went. That would need the kind. A Full member at
  position p that leaves early breaks every resume point at or past p.
  The group 0 chain from p on, and the deeper members, then become bytes
  no resume point can use, and nothing evicts them early.

Client contract (the server cannot enforce it without kinds): a client
must refresh every group block a resume point needs at least as often as
that point's group 0 position. In practice, whenever it prefix-queries
group 0 over a range of positions, it also membership-queries each
group g > 0 over the blocks of that range it would load. Members age by
their own queries only, so a client that refreshes group 0 over a wider
range than group g makes the members go first while the descendant rule
keeps their positions. I did not read the connectors (out of scope for
this work), so whether they meet this is unchecked. No hybrid model has
run with the window on. That is why the window is opt-in, and one hybrid
G5 cap run is the bar before 25 becomes the default.

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
+0.01-0.03 E_C. At 0.25 it loses 0.01-0.05 E_C, and old's hit drops by up
to 0.02 in one of two runs. The cause is not known (§The 0.25 regression),
and it is one reason the window is opt-in. No-cap warm TTFT and gsm8k do
not regress. One cap pair rerun on the review-fixed build (10e7afa,
window passed explicitly) is in §Rerun after the review fixes.

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
  0.159-0.163. Old's hit drops up to 0.02 in one of two runs: H5 0.430 /
  0.457 against 0.451-0.459 for every LRU run (H4 0.451 / 0.458, w0
  0.459). v2's hit does not drop (H4 0.459 / 0.442, H5 0.464 / 0.458).
  See §The 0.25 regression.
- Both servers have 0 dead blocks at every pass end.
  `pegaflow_cache_evictions_with_resident_children` is not exported by any
  H5 25% run. An OTel counter that is never added is not exported, and the
  same binary exports it at window 0. So H5 never evicted a block that had
  a known (query-linked) resident dependent. Unlinked dependents are not
  counted (see the metric's doc).
- Cap warm TTFT p50 does not regress: at 0.5, H5 is 2.07-2.28 s (old) and
  2.10-2.17 s (v2), against H4's 2.22-2.30 and 2.20-2.28 s.

### Rerun after the review fixes

One more cap pair on the same gate tree, plan and arms, after the review
fixes M1, M2, the default change and N4. Server builds:
- H5: built from a `git archive` of 10e7afa
  (`~/kexi/pegaflow-rsi-server-evict-tail-src-10e7afa`, tar sha256
  055aa8f9…, same on both ends) as `bin/pegaflow-server-h5-10e7afa`,
  sha256 ebce745b…. The window is passed explicitly: the run uses the
  wrapper `bin/pegaflow-server-h5-10e7afa-w25` (sha256 eb4b824c…, the
  hash in the run's `summary.json`), which execs that binary with
  `--tail-window-percent 25`.
- H4: the same d45a366 binary as before.

The pair ran at the same time, with H5 on GPU7. The first pair had H5 on
GPU6.

| tag | server | GPU | pgid | UTC start - end |
|---|---|---|---|---|
| `cap4-h5fix` | H5 10e7afa, window 25 | 7 | 2656838 | 08:54:14 - 10:00:36 |
| `cap4-h4` | H4 | 6 | 2656839 | 08:54:15 - 10:00:44 |

Both runs exit rc 1, non-official, as above. Timed warm passes, old / v2:

| C/U | run | E_C | hit | H\*_C | H_lru |
|---|---|---|---|---|---|
| 0.25 | H4 `cap4-h4` | 0.208 / 0.195 | 0.474 / 0.455 | 0.865 / 0.863 | 0.584 / 0.588 |
| 0.25 | H5 `cap4-h5fix` | 0.126 / 0.168 | 0.453 / 0.462 | 0.865 / 0.864 | 0.594 / 0.587 |
| 0.5 | H4 `cap4-h4` | 0.712 / 0.712 | 0.860 / 0.860 | 0.929 / 0.929 | 0.858 / 0.857 |
| 0.5 | H5 `cap4-h5fix` | 0.759 / 0.769 | 0.887 / 0.890 | 0.930 / 0.929 | 0.858 / 0.858 |
| 0.75 | H4 `cap4-h4` | 0.749 / 0.750 | 0.916 / 0.917 | 0.971 / 0.971 | 0.915 / 0.916 |
| 0.75 | H5 `cap4-h5fix` | 0.778 / 0.791 | 0.930 / 0.933 | 0.971 / 0.971 | 0.917 / 0.917 |

- The 0.5 gain holds: +0.047 / +0.057 E_C over the H4 run beside it, and
  0.759 / 0.769 against 0.763-0.769 for the 7791c30 build. Hit is
  0.887 / 0.890 against 0.860.
- At 0.75, +0.03 / +0.04 E_C.
- At 0.25, old loses again: 0.126 against 0.208, the widest gap of the
  three pairs, since this H4 run is the highest H4 so far. v2 loses
  0.027.
- Pressure evictions at 0.5: 63.7k / 63.2k against H4's 75.4k / 75.2k,
  the same ratio as the first build. `with_children` is 0 on both
  servers, and dead blocks are 0 at every pass end.
- Warm TTFT p50 at 0.5, warm2 / warm3: H5 2.19 / 2.13 s (old) and
  2.11 / 2.08 s (v2), against H4's 2.29 / 2.33 and 2.24 / 2.31 s.

The fixes barely touch this run's path:
- Qwen3-4B has one group and there is no MetaServer, so nothing is
  demoted to reclaimable (M1).
- 25% of the class is about 1.4k blocks at 0.25 and 2.9k at 0.5, under
  the 4096 cap. At 0.75 it is about 4.3k, so the cap trims the window
  slightly there (M2).
- The pinned-leaf fallback and the LRU fallback only run when pins cover
  the candidates.

So this pair mainly shows that the fixed build keeps the gain. It is not
evidence for the backup path.

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

The cause is not known. What the runs show:
- It is the policy, not the build. The same binary with
  `--tail-window-percent 0` (plain LRU) gives 0.179 / 0.180 at 0.25 and
  0.698 / 0.706 at 0.5, both at H4's level. At window 0 the counter
  `pegaflow_cache_evictions_with_resident_children` is 38.3k (old cap025)
  and 57.7k (v2 cap) per run, summed over the four passes in
  `result.json`. No H5 25% run exports it.
- It is not the window size. 10% gives 0.152 / 0.161 at 0.25 and 0.734 /
  0.735 at 0.5. A smaller window does not recover 0.25, and it gives up
  half of the 0.5 gain.
- It shows mostly in E_C. Server-served tokens past the baseline's local
  hit (the part E_C credits) drop from 360-376 to 264-307 per request
  (old), and from 366-420 to 337-352 (v2) (`sim/split.py`). Old's hit
  drops up to 0.02 in one of the two runs (above). v2's hit does not
  drop.
- The loss is in blocks of known depth. `sim/unknown.py` splits each run's
  credited blocks (timed passes, past the baseline's local hit, capped by
  the unlimited oracle) into those that no earlier query had hit (still at
  unknown depth on H5) and those an earlier query had hit (known depth).
  Rerun for these notes on 10.1.0.44 (`out/unknown/`). The numbers equal
  the reviewer's rerun; the counts are credited x share, rounded:

| arm | run | credited | never hit before | never-hit blocks | previously hit blocks |
|---|---|---|---|---|---|
| old | H4 `cap3-h4` | 21978 | 0.074 | 1.6k | 20.4k |
| old | H5 `cap3-h5` | 16121 | 0.217 | 3.5k | 12.6k |
| old | H4 `cap3-h4-b` | 22958 | 0.112 | 2.6k | 20.4k |
| old | H5 `cap3-h5-b` | 18736 | 0.159 | 3.0k | 15.8k |
| old | H5 w0 | 22769 | 0.082 | 1.9k | 20.9k |
| old | H5 w10 | 19247 | 0.069 | 1.3k | 17.9k |
| v2 | H4 `cap3-h4` | 25622 | 0.070 | 1.8k | 23.8k |
| v2 | H5 `cap3-h5` | 20564 | 0.061 | 1.3k | 19.3k |
| v2 | H4 `cap3-h4-b` | 22328 | 0.055 | 1.2k | 21.1k |
| v2 | H5 `cap3-h5-b` | 21463 | 0.070 | 1.5k | 20.0k |
| v2 | H5 w0 | 24100 | 0.046 | 1.1k | 23.0k |
| v2 | H5 w10 | 21626 | 0.098 | 2.1k | 19.5k |

  - In both pairs, H5 on old serves more never-hit blocks than H4 (3.5k
    against 1.6k, and 3.0k against 2.6k), not fewer. The whole loss is in
    previously hit blocks: -7.8k in the first pair and -4.6k in the
    second.
  - w10 shows the same: fewer never-hit blocks than H4 (1.3k), and still
    2.5k fewer previously hit ones.
  - v2 has no share difference (H5 0.061 / 0.070 against H4 0.070 /
    0.055). Its smaller loss is also in previously hit blocks (-4.5k and
    -1.1k).

An earlier version of this section blamed unknown-depth blocks being
evicted first. That contradicts the table: if it were the cause, H5's
never-hit count would fall, and it rises. That version also cited only the
first pair. The explanation is withdrawn.

One hypothesis fits the data and is not verified: on old, the chain heads
under vLLM's local hit are queried once and never loaded. Tail-first keeps
them while their descendants are resident (by design,
`queried_never_loaded_head_outlives_its_resident_descendants`), and pays
for them with deep, known-depth tails. At 0.5 that trade wins, because the
kept heads serve after vLLM drops its local copy. At 0.25 the space
they take is the difference, and LRU drops them for free while vLLM still
holds them locally. It does not explain v2's loss, since v2 queries from
block 0. Testing it needs the server's resident set per pass, or a sim
that matches H5 at 0.25. Neither exists: attribution match there is
0.20-0.38. Ranking unknown depth last (sim `subfg-sh`) is worse at every
capacity, 0.640 / 0.702 at 0.5 against 0.753 / 0.767, so that is not the
fix either.

Tradeoff, not fixed: at 0.25 the store serves little under any policy (E_C
0.13-0.19, hit 0.43-0.46 against H_lru 0.57-0.60). H5 loses 0.01-0.05 E_C
there, and old's hit drops up to 0.02 in one of two runs. At 0.5 it gains
+0.06 E_C and +0.03 hit, and at 0.75 +0.01-0.03 E_C and +0.015 hit. The
server default is 0 (opt-in, §What changes); 25 is the evaluated setting.

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
0.43-0.46 against 0.86). That point is capacity-bound, and H5 lowers old's
hit there by up to 0.02 in one of two runs.

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

After the review fixes: `verify.sh` on the 10e7afa export (the gate
rerun's source), GPU6, started 10:04:02 UTC, logs in
`~/kexi/logs/rsi-server-reclaim-verify-10e7afa/`.
- `env.txt` now lists rev and a sha256 of every Rust source
  (`src-sha256.txt`, 143 files), not 4 unrelated files (review N2). The 5
  files the review fixes changed (`storage/read_cache.rs`,
  `storage/mod.rs`, `pegaflow-core/src/lib.rs`, `metrics.rs`,
  `pegaflow-server/src/lib.rs`) equal `git show 10e7afa:<file>`.
- `cargo-test.log` (rc 101, `ssd_cache` only): 378 passed. pegaflow-core
  lib has 180 passed and 5 ignored, the 5th being the new bench. These
  pass:
  - the eviction-order table, now 5 cases: narrow, whole, lru,
    reclaimable head keeps its retained tail (M1), and the retained walk
    stopping at a reclaimable child. Each checks that both link maps are
    empty after a full drain (S5);
  - `queried_never_loaded_head_outlives_its_resident_descendants`;
  - `group_blocks_leave_before_their_position_and_pins_hold_the_chain`,
    now through the single-lock membership lookup (N4) and the LRU
    fallback.

  `ssd_cache` fails all 11 with "pinned pool exhausted", as on master
  (below).
- `cargo-test-ignored.log`: 5 passed, including
  `tail_first_round_lock_hold_at_2_6m_blocks`. `cargo-test-ignored-cuda12`:
  3 passed. `clippy-cuda13`: clean. `check-rest` (rc 0): ruff check,
  default-feature clippy and cargo check pass. `check.log` (rc 1): typos is
  not installed, and `ruff format --check` fails on the same 11 untouched
  Python files.

Before the review, from a `git archive` of 81fdb6d in
`~/kexi/pegaflow-rsi-server-evict-tail-src-81fdb6d` (tar sha256 4c2382e6…,
same on both ends). `~/kexi/rsi-tools/server-reclaim/verify.sh <src> 6`,
logs in `~/kexi/logs/rsi-server-reclaim-verify-81fdb6d/`. That run's
`env.txt` hashed 4 files, none of them changed by H5. The reviewer checked
`read_cache.rs`, `lib.rs` and `storage/mod.rs` against git separately.
- `cargo-test.log` (rc 101, `ssd_cache` only): pegaflow-core lib 180
  passed, 4 ignored. That includes
  `tail_first_evicts_deepest_block_of_oldest_window_first`,
  `queried_never_loaded_head_outlives_its_resident_descendants` and
  `group_blocks_leave_before_their_position_and_pins_hold_the_chain`. Every
  other target passes, 378 tests passed in all. `ssd_cache` fails all 11 with
  "pinned pool exhausted", as on every earlier branch run on this host.
  Master fails it the same way. `~/kexi/rsi-tools/server-evict-tail/master-ssd.sh`
  ran `cargo test -p pegaflow-core --test ssd_cache` once from a
  `git archive` of master 75dfa4c (tar sha256 597a473b…, same on both
  ends), on GPU7, pgid 2615128, 2026-10-24 08:29:56 - 08:30:24 UTC. It
  gave 0 passed, 11 failed, all at `harness.rs:379` with `save:
  Storage("pinned pool exhausted while allocating block buffer")`. The log
  is in `~/kexi/logs/rsi-server-evict-tail-master-75dfa4c-ssd_cache/`
  (`cargo-test-ssd_cache.log`, sha256 b97b82b6…).
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

- 0.25 regression, above. The cause is unknown.
- Memory: a linked block costs a parent-map entry (two `BlockKey`s) and a
  slot in its parent's children list, a few hundred bytes with 32-byte
  hashes. That is under 0.02% of a 2.25 MiB block, but it grows with the
  resident count, not bytes.
- A pressure round scans the window under the cache lock: it collects up to
  4096 entries (25% of the class is about 2.9k at 11.5k blocks), selects
  the deepest 4 x batch, and sorts only those. At 2.6M resident blocks that
  is about 1.4 ms per round against 0.8 ms for LRU (§What changes). The
  reclaim loop still has no round cap, as on H4.
- Decode backup with the window on is untested end to end. M1 is a unit
  test only; no gate run has a MetaServer, and reclaimable evictions are 0
  in every gate run.
- Group > 0 linking is unit-tested only. No hybrid model has run on H5, and
  the client contract in §Group coherence is unchecked.
- Depth is the hit-prefix index of the latest query, so it shifts when a
  chain is requeried from another offset (old: 6-21% of queries start past
  its local hit, 6-10% before it; `sim/offsets.py`). Order within a chain stays right, but depths of two
  chains are not comparable. The gate result is the evidence this is good
  enough.
