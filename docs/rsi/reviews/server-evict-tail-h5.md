# Review: server H5 tail-first eviction (d45a366..4a7b4e0)

Reviewer: independent agent, fresh context. Scope: 7791c30, 81fdb6d, d9ea4d1,
4a7b4e0. Line numbers are at 4a7b4e0 unless noted. Remote evidence was read
on 10.1.0.44 under `~/kexi/` only. The one script I ran was `sim/unknown.py`,
CPU only, output to stdout.

Verdict: accept after must-fix (M1, M2). Default recommendation: 0 (opt-in).

## 1. Correctness under the cache lock

Pins hold. Every victim goes through `evict_subtree`, which checks
`is_cache_owned_only` on the exact leaf it is about to evict
(`read_cache.rs:924`). Query leases, transfer locks (`transfer_lock.rs`
holds `Arc<SealedBlock>`) and backup copies all hold a strong ref. A pinned
leaf ends the subtree (`return`), so no ancestor of a pinned block is taken
either. A block that is mid-save is not in the read cache yet. A re-save of
a resident block is `AlreadyExists` (`read_cache.rs:617`), which keeps the
block's class, depth and links. I found no path that evicts a pinned,
leased, transfer-locked or mid-save block.

Links are only made between two resident blocks, under the same lock:
- `get_prefix_blocks` links `keys[i-1] -> keys[i]` only when `keys[i]`
  hits. The scan breaks at the first miss, so `keys[i-1]` is resident
  (`read_cache.rs:224-236`).
- `link_group_members` checks both ends (`read_cache.rs:250-261`).

Links are only dropped in two places: `finish_eviction` -> `unlink`
(`read_cache.rs:798`), and `remove_all`, which resets the map
(`read_cache.rs:397`). `TinyLfuCache::remove` has no other caller. So the
link maps are bounded by the resident count.

Out-of-order eviction does not leak:
- Evicting a parent first clears each child's parent pointer
  (`read_cache.rs:131-133`).
- Evicting a child first removes it from its parent's list, and drops the
  list once it is empty (`read_cache.rs:112-119`).

Forks work. After each victim the walk returns to the parent and goes down
the next branch, and the root goes last. Re-linking a child to a new parent
moves it cleanly (`link`, `read_cache.rs:99-110`). A client that sends a
cycle is bounded by `steps` and then broken by `unlink`.

The invariant only covers links the server has learned. These cases have
no parent link, so their parent can go first and leave them stranded:
- a block saved but not yet hit by a group 0 prefix query (the newest
  turn, decode tails, SSD/RDMA prefetch inserts);
- a head that was evicted and saved again. Its resident children lost
  their link when it left and stay unlinked until the next query hits
  through them.

H5's attributed stranded share of 0.04-0.10 is this. That is expected. The
doc (evict-tail.md:17-18) and the metric text (`metrics.rs:75-78`) should
say "known dependent" (N1).

Class handling is wrong across classes. See M1.

## 2. Model-agnosticism and the group > 0 rule

The diff has no model names and no Kind branches. The server only uses
group ids and `raw != encoded` (`lib.rs:633-641`).

The rule as coded: a group > 0 block is a child of its group 0 position and
takes the position's depth.
- In a subtree walk, `deepest_child` picks the chain child (depth p+1) before
  the member (depth p) (`read_cache.rs:820-829`). Resume points therefore
  shrink from the tail: deeper positions and their members go first, then
  p's member, then p.
- A linked member never outlives its position.
- An unlinked member (never membership-queried) has UNKNOWN_DEPTH and ranks
  first. It can outlive its position only if that position goes inside an
  ancestor's subtree in the same round. That is fine.

81fdb6d against the doc: the doc's claim is accurate, including what it
does not cover (evict-tail.md:80-81). A member can still leave alone, as a
window candidate, while its position and deeper positions stay:
- For a Full group > 0 that breaks every resume point at or past p.
- The deeper group 0 chain and the deeper members then become bytes that no
  resume point can use, and nothing evicts them early.
- This happens whenever members age faster than their positions, i.e. when
  the client refreshes group 0 over a wider range than it membership-queries
  group g.

The earlier finding makes this likely. Blocks under vLLM's local hit are
queried but never loaded. If the connector refreshes their group 0 positions
but not their group g members, those members go first, while their positions
are kept by the descendant rule. The resume point is gone once vLLM drops
its local copy.

The server cannot fix this without kinds, so it has to be a written client
contract: "a membership query must refresh every group block a resume point
needs at least as often as its position" (S4). I did not read the connector
(out of scope), so I cannot say whether it meets this. No hybrid model has
run on H5.

## 3. Performance of the window scan

Each pressure round, per class, under the cache lock (`read_cache.rs:839-885`):
- walk `w = 25% of len` LRU entries (a hashlink linked list, so pointer
  chasing);
- build a `Vec` of 24-byte tuples;
- `select_nth` to `4 x batch`, sort those, and clone them.

`batch = clamp(resident / 64, 1, 512)` (`mod.rs:792`). Below 32k resident
blocks the window is 16 x batch, which is cheap. Above that the batch stays
at 512 and the window keeps growing.

Estimates for a 160 GiB pool, all retained:

| block size | resident | window | entries scanned per evicted block | per round (5-15 ns/entry) |
|---|---|---|---|---|
| 2.25 MiB (Qwen3-4B, as gated) | ~73k | ~18k | ~36 | ~0.1-0.3 ms |
| 2 MiB | ~82k | ~20k | ~40 | ~0.1-0.3 ms |
| 64 KiB | ~2.6M | ~655k | ~1280 | ~3-10 ms, plus a ~16 MB Vec |

H4's LRU costs O(batch), a few µs. Every query, insert and backup scan takes
the same lock, so at small blocks query latency gets ms-scale stalls for
each round. The reclaim loop (`mod.rs:596-634`) has no round cap. It stops
only when the allocator fits or a round evicts nothing, so an allocation on
a fragmented pool can run many rounds back to back.

The pathological case is the fallback scan (`read_cache.rs:874-883`, M2).
When the candidates do not fill the batch, it walks the whole class with no
cap and clones every leaf key. Pinned leaves stay in the list too (the
filter only checks for children). The realistic trigger is the oldest
retained blocks being pinned:
- decode backup reports the oldest retained blocks
  (`backup_candidates`), and the decode node RDMA-pulls them under transfer
  locks;
- P2P serving locks the blocks it is reading.

That is the window. At 2.6M blocks one fallback clones about 2.6M keys
(namespace String plus hash Vec, about 200 MB of allocation churn), roughly
0.1-0.5 s with the lock held. If a round frees a few blocks but not enough,
the next round does it again.

Fixes:
- Cap the window at an absolute count, e.g.
  `min(pct x len, max(4096, 8 x batch))`. The sim supports this: the oldest
  4096 blocks scored 0.762 / 0.726 at 0.5, against 0.764 / 0.734 for 25%
  (evict-tail.md:104-106).
- Cap the fallback the way `BACKUP_SCAN_FACTOR` caps the backup scan, skip
  pinned leaves without cloning them, and stop at the first `batch` victims.

An index (a depth-ordered heap or per-depth buckets) is not needed once the
scan is capped.

## 4. Evidence and hit-rate judgment

I read all eight `summary.json` files. Provenance:
- the H5 server sha256 is ace342eb… in every H5 run;
- `build.log` shows it was built from rev 7791c305… (and H4 b2411b45… from
  d45a366…);
- `-w0` and `-w10` are `exec` wrappers around the same binary.

Every E_C / hit / H\*_C / H_lru cell in evict-tail.md:183-196 matches the
summaries (warm_timed). The w10 numbers (0.152 / 0.161, 0.734 / 0.735) also
match.

Window 0 is back at H4's level:

| C/U | arm | H4 (2 runs) | H5 w0 | H5 25% (2 runs) |
|---|---|---|---|---|
| 0.25 | old | 0.174 / 0.179 | 0.179 | 0.129 / 0.147 |
| 0.25 | v2 | 0.194 / 0.170 | 0.180 | 0.159 / 0.163 |
| 0.5 | old | 0.699 / 0.702 | 0.698 | 0.763 / 0.769 |
| 0.5 | v2 | 0.719 / 0.713 | 0.706 | 0.763 / 0.768 |

v2 at 0.5 is 0.007 below H4's two runs. w0 ran once, so that is inside
noise. The 0.25 regression is the policy, not the build.

The "38-58k evictions with resident children" for w0 is the sum of the four
per-pass counter values in each w0 `result.json` (old cap025 38.3k, v2 cap
57.7k). The H5 25% runs export no such counter. An OTel counter that is
never added is not exported, and the same binary exports it at w0, so "0 on
every H5 run" holds, by absence.

The "unknown-depth blocks evicted first" explanation is not supported. The
data point the other way. I re-ran `sim/unknown.py` for old at cap025 and
split its share into absolute counts:

| run | credited blocks | never-hit share | never-hit served | previously-hit served |
|---|---|---|---|---|
| H4 `cap3-h4` | 21978 | 0.074 | ~1.6k | ~20.4k |
| H5 `cap3-h5` | 16121 | 0.217 | ~3.5k | ~12.6k |
| H4 `cap3-h4-b` | 22958 | 0.112 | ~2.6k | ~20.4k |
| H5 `cap3-h5-b` | 18736 | 0.159 | ~3.0k | ~15.8k |
| H5 w0 | 22769 | 0.082 | ~1.9k | ~20.9k |

H5 serves more never-hit (unknown-depth) blocks than H4, not fewer. The
whole loss is in previously hit, known-depth blocks (-4.6k to -7.8k). If
unknown depth were evicted first and that were the cause, the never-hit
count would fall. The doc also cites only the first pair ("22% against
6-7%"). The second pair is 0.159 against 0.112, a much weaker gap. For v2
the share does not differ (H5 0.061 / 0.070, H4 0.070 / 0.055).

A hypothesis that fits better, which I have not verified: on old, the heads
under vLLM's local hit are queried and never loaded. H5 keeps them by design
(`queried_never_loaded_head_outlives_its_resident_descendants`) and pays
for them with deep, known-depth tails. At C/U 0.25 that capacity is the
difference. LRU drops those heads for free while vLLM still holds them
locally. The fix is to rewrite the regression section (S3), not the code.

Hit-rate judgment (evict-tail.md:270-294): mostly honest.
- The 0.5 and 0.75 statements match the data: 0.888-0.889 against H\*_C
  0.929-0.930 and H_lru 0.857-0.858; 0.930-0.935 against 0.971.
- E_C (0.763-0.769) is stated as below G5's 0.9.
- The own-local E_C (0.927-0.928) supports the local-hit explanation.
- One overreach: "hit does not move beyond run-to-run spread" at 0.25 pools
  both arms (0.430-0.464). For old alone, H4 is 0.451 / 0.458 and w0 0.459,
  against H5 0.430 / 0.457. The first H5 old run is 0.02 below every LRU
  run. Say "old hit drops up to 0.02 in one of two runs" (N3).

The judgment "close enough at 0.5 and 0.75, capacity-bound at 0.25" is
reasonable for Qwen3-4B.

## 5. Tests

`~/kexi/logs/rsi-server-reclaim-verify-81fdb6d/` came from 81fdb6d's
export:
- `env.txt` has rev=81fdb6d0…;
- the source dir's `.rsi-rev` is 81fdb6d0…;
- I hashed `read_cache.rs` (72d06d0a…), `lib.rs` (3397af76…) and
  `storage/mod.rs` (2e30b090…) on the host, and they equal
  `git show 81fdb6d:<file>`;
- the log has the post-81fdb6d test name
  `group_blocks_leave_before_their_position_and_pins_hold_the_chain`, and
  the 7791c30 name (`tail_first_keeps_resume_point_groups_together…`) does
  not appear.

Results:
- `tail_first_evicts_deepest_block_of_oldest_window_first`,
  `queried_never_loaded_head_outlives_its_resident_descendants` and the
  group/pin test all report `ok`. pegaflow-core lib: 180 passed, 4 ignored.
- The only failures are the 11 `ssd_cache` tests, all panicking at
  `harness.rs:379` "pinned pool exhausted". The same 11 fail in the 67922b5
  and 6a99d43 verify logs on this host.
- I found no master run log on .44. "As on master" is plausible but I did
  not verify it.
- `rc.txt`: clippy cuda13 0, ignored 0 / 0, check-rest 0, check.sh 1 (typos
  missing, ruff format on untouched files, per the doc).

Missing tests: cross-class descent (with M1), and a check that the link maps
are empty once every block has been evicted in arbitrary order (S5).

## 6. Gate runs on 7791c30 vs 81fdb6d

`git show 81fdb6d` removes `ChainLinks::members`, `eviction_root` and the
`members` insert/remove calls.
- `eviction_root(key)` returned `key` unless `key` was in `members`.
- `members` was only filled by `link_group_members`, which only group > 0
  queries reach.
- The rest of 81fdb6d is comments and a test rename.

Qwen3-4B is one group, so both commits evict identically on it, and the
gate runs on the 7791c30 binary are valid for 81fdb6d. 4a7b4e0 only changes
docs and help text.

## 7. Default recommendation

Ship with `--tail-window-percent` defaulting to 0 (opt-in), and pass 25
explicitly in RSI gate runs. Reasons:
- M1 changes the production decode-backup / P2P-demotion contract, and no
  gate run covers it (no MetaServer; reclaimable evictions are 0 in every
  run).
- The uncapped scans (M2) put lock stalls on every query at small block
  sizes.
- Old at C/U 0.25 loses 0.03-0.05 E_C, and the doc's explanation does not
  hold, so the cause is unknown.
- The group > 0 rule depends on a client contract (§2) that no hybrid model
  has exercised.

Revisit 25 as the default once M1 and M2 are fixed and one hybrid model has
passed G5 cap with the window on. The 0.5 gain (+0.06 E_C, +0.03 hit) is
real and repeated across two GPU-swapped pairs.

## Findings

### Must-fix

- M1 `read_cache.rs:839-873`, `906-937`: the reclaimable pass descends into
  retained descendants. This breaks reclaimable-first and stalls decode
  backup.
  - Scenario: a prefill node with MetaServer and decode backup. Chain
    c0..c9. `backup_candidates` reports the oldest retained blocks c0..c4,
    the decode node copies them, and BackupHint demotes them to reclaimable.
    c5..c9 stay retained and exist nowhere else.
  - Under pressure the reclaimable pass picks c4. `evict_subtree` evicts
    c9, c8, …, c5 (retained, cluster-unique), then c4. The cluster loses
    c5..c9 and keeps c0..c4 twice.
  - Freed bytes come from retained, so reclaimable stays at its watermark.
    `backup_report_budget` (the gap to the watermark) stays near 0, and
    backup stops.
  - The same applies to TransferRelease and ReclaimHint demotions. H4
    evicted c0..c4 first.
  - Fix: keep plain LRU in the reclaimable class. Its blocks have a copy on
    another node, so evicting a head there strands nothing that RDMA
    prefetch cannot refill. Run tail-first only in the retained class, and
    never descend across a class in the reclaimable pass. Add a table case
    for this.
- M2 `read_cache.rs:874-883` (and `851-852`): the fallback leaf scan walks
  and clones the whole class under the cache lock with no cap, and the
  window grows linearly with the class.
  - Scenario: 160 GiB of 64 KiB blocks (2.6M resident). Decode backup pulls,
    or P2P transfer locks, pin the oldest retained blocks. Every candidate
    subtree ends at a pinned leaf, and each round clones about 2.6M keys
    (0.1-0.5 s with the lock held).
  - `reclaim_until_allocator_can_allocate` repeats this every round that
    frees anything.
  - Fix: cap the fallback (like `BACKUP_SCAN_FACTOR`), skip pinned leaves
    without cloning them, and cap the window at an absolute count (§3).
  - This is Must while the default is 25. It drops to Should if the default
    becomes 0.

### Should-fix

- S1 `pegaflow-server/src/lib.rs:195` and `storage/mod.rs:114`: default 25
  → 0 (§7).
- S2 `read_cache.rs:851-855`: cap the window at an absolute count even with
  the default off. At 64 KiB blocks one round scans about 1280 entries per
  evicted block (§3).
- S3 `evict-tail.md:248-263`: the 0.25 cause contradicts the data. H5
  serves more unknown-depth blocks than H4 (about 3.5k against 1.6k), and
  the loss is in known-depth blocks. Rewrite the section with both pairs
  and state that the cause is unknown or give the head hypothesis (§4).
  Otherwise the next iteration tunes the wrong rule.
- S4 `lib.rs:629-641` and `evict-tail.md:63-87`: write the client contract
  for group > 0 refresh (§2), and say that a Full group member leaving early
  strands its position's deeper chain. Run one hybrid model before the
  default goes on.
- S5 `read_cache.rs` tests: add (a) the M1 class-crossing case, and (b)
  "link maps empty after evicting every block in arbitrary / parent-first
  order", which backs the no-leak claim. Merge them into the existing
  eviction-order table where they fit.

### Nit

- N1 `metrics.rs:75-78`, `evict-tail.md:17-18`, `32-34`:
  - The invariant is "no resident known dependent". Unlinked blocks (new
    turn, decode tails, prefetch inserts, children of a re-saved head) are
    not covered.
  - "except for … full cleanup" is wrong: `remove_all` never increments the
    counter.
- N2 `verify.sh` `env.txt` hashes only 4 files, none of them
  `read_cache.rs` or `lib.rs`. Hash the files the commit changes. The doc's
  "env.txt sha256s match" proves nothing about the changed code; I checked
  it separately.
- N3 `evict-tail.md:202-204`, `265-266`: the 0.25 hit spread pools both
  arms. Old H5 run 1 is 0.430, against 0.451-0.459 for every LRU run (§4).
- N4 `lib.rs:625`: `query_group_membership` takes the cache lock a second
  time and builds two `BlockKey`s per position, misses included. Link
  inside the same lock as `get_membership`, and only for hits.
- N5 `read_cache.rs:250-261`: a member linked to a position of
  UNKNOWN_DEPTH takes UNKNOWN_DEPTH too. That is fine, but it is a reason a
  hybrid model's members may rank first. Worth a sentence in §Group
  coherence.

## Verdict

accept after must-fix (M1, M2)
