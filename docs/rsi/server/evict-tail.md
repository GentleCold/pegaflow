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

## Gate (G5 cap, official r3 plan)

Gate bf9ef3c, v2 8a05b7c via `--under-test-path`, arms baseline, old, v2,
`--workloads cap`, official cap fractions 0.25 / 0.5 / 0.75. Servers:
- H4: `bin/pegaflow-server-h4-d45a366`, sha256 b2411b45…, built from a
  `git archive` of d45a366;
- H5: `bin/pegaflow-server-h5`, sha256 ace342eb…, built from a `git archive`
  of 7791c30.

The 81fdb6d change only touches the group > 0 path, which Qwen3-4B never
takes, so the 7791c30 binary measures the same group 0 policy.

Runs (pgid, GPU, UTC start-end), logs and meta in
`~/kexi/rsi-runs/srv-evict-tail-<tag>.{log,meta}`, run dirs
`~/kexi/rsi-runs/2026-10-24-qwen3-4b-srv-evict-tail-<tag>`:

| tag | server | GPU | pgid | UTC start - end |
|---|---|---|---|---|
| `cap3-h5` | H5 | 6 | 2385306 | 04:06:14 - 05:12:52 |
| `cap3-h4` | H4 | 7 | 2385307 | 04:06:14 - 05:12:46 |

All on 2026-10-24. Every run exits rc 1 because the gate marks a cap-only
run non-official ("arm plain missing", workloads) and G5 fails. The run
records are complete.

Timed warm passes (warm2 + warm3), old / v2:

| C/U | server (run) | E_C | hit | H\*_C | H_lru |
|---|---|---|---|---|---|
| 0.25 | H4 (`cap3-h4`) | 0.174 / 0.194 | 0.451 / 0.459 | 0.865 / 0.863 | 0.573 / 0.589 |
| 0.25 | H5 (`cap3-h5`) | 0.129 / 0.159 | 0.430 / 0.464 | 0.863 / 0.862 | 0.575 / 0.580 |
| 0.5 | H4 (`cap3-h4`) | 0.699 / 0.719 | 0.861 / 0.861 | 0.930 / 0.929 | 0.858 / 0.857 |
| 0.5 | H5 (`cap3-h5`) | 0.763 / 0.763 | 0.888 / 0.889 | 0.930 / 0.929 | 0.858 / 0.857 |
| 0.75 | H4 (`cap3-h4`) | 0.748 / 0.757 | 0.916 / 0.917 | 0.971 / 0.971 | 0.915 / 0.916 |
| 0.75 | H5 (`cap3-h5`) | 0.766 / 0.792 | 0.931 / 0.935 | 0.971 / 0.971 | 0.917 / 0.916 |

RESULTS_PENDING
