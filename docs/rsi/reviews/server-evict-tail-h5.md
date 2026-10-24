# Review: server H5 tail-first eviction (d45a366..4a7b4e0)

Reviewer: independent agent, fresh context. Scope: 7791c30, 81fdb6d, d9ea4d1,
4a7b4e0. Line numbers are at 4a7b4e0. Status: in progress (mid-way push).

## 1. Correctness under the cache lock

Pins: every victim goes through `evict_subtree`, which checks
`is_cache_owned_only` on the leaf it is about to evict
(`read_cache.rs:924`). Query leases, transfer locks (`transfer_lock.rs`
holds `Arc<SealedBlock>`) and backup copies all hold a strong ref, so they
are skipped. A pinned leaf aborts the whole subtree, so an ancestor of a
pinned block is never taken. A block mid-save is not in the read cache yet,
and a re-save of a resident block is `AlreadyExists` (`read_cache.rs:617`),
which keeps metadata and links. Nothing here can evict a pinned block.

Links: created only between two resident blocks under the same lock
(`get_prefix_blocks` links `keys[i-1]` only when `keys[i]` hits, and the
scan stops at the first miss, so `keys[i-1]` is resident;
`link_group_members` checks both ends, `read_cache.rs:250-261`). The only
removal paths are `finish_eviction` -> `unlink` (`read_cache.rs:798`) and
`remove_all`, which resets the map (`read_cache.rs:397`). `cache.remove` has
no other caller. So the link map is bounded by the resident count and is
freed on out-of-order eviction: evicting a parent first drops the child's
parent pointer (`read_cache.rs:131-133`), and evicting a child first drops
it from the parent's list. No leak found.

Forks: one parent with many children works. The walk goes back to the
parent after each victim and descends into the next branch, and the root
goes last. A parent change (`link` re-insert) moves the child cleanly.

The invariant only covers learned links. A block saved but not yet hit by a
prefix query (the newest turn's blocks, decode tails) has no parent link,
so its parent can go first and leave it stranded. The attribution's
0.04-0.10 stranded share for H5 is this. Doc line 17-18 ("never evicted
while a block that depends on it is resident") and the metric description
(`metrics.rs:75-78`) should say "known dependent".

Class crossing: see M1.

## 2. Model-agnosticism and the group > 0 rule

No model names and no Kind branches in the diff. The server uses only
group ids and `raw != encoded` (`lib.rs:633-641`).

The rule as coded: a group > 0 block is a child of its group 0 position,
with the position's depth. During a subtree walk the deeper chain child
(depth p+1) is picked before the member (depth p) (`deepest_child`,
`read_cache.rs:820-829`), so resume points shrink from the tail: all deeper
positions and their members go, then p's member, then p. A linked member
never outlives its position. An unlinked member is never refreshed, so it
is at least as old as its position and ranks first (UNKNOWN_DEPTH). It can
outlive the position by up to one round when the position goes as part of
an ancestor's subtree. That is acceptable but not "never".

Open hole, which the doc admits (doc:80-81): a member can leave alone as a
window candidate while its position and deeper positions stay. For a Full
group > 0 that breaks every resume point at or past p, and the deeper group
0 chain plus deeper members become bytes no resume point can use. It happens
when the member ages out before its position: the client refreshes group 0
over a range (e.g. from block 0, or the vLLM-local-hit region that is
queried but never loaded) but issues group g membership queries over a
narrower range. The server cannot see the kind, so this has to be a client
contract, written down: "every group block a resume point needs must be
refreshed by a membership query at least as often as its position". See
S5. No hybrid model has run on H5, so this is unverified.

## 3. Performance of the window scan

(in progress)

## 4. Evidence and hit-rate judgment

(in progress)

## 5. Tests

(in progress)

## 6. Gate runs on 7791c30 vs 81fdb6d

`git show 81fdb6d` removes `ChainLinks::members`, `eviction_root` and the
`members` insert/remove. `eviction_root(key)` returned `key` unless `key`
was in `members`, and `members` was filled only by `link_group_members`,
which only group > 0 queries reach. Qwen3-4B is one group, so 7791c30 and
81fdb6d evict identically on it. The rest of 81fdb6d is comments and a
test rename. The gate runs on the 7791c30 binary (`build.log`:
rev=7791c305…, sha256 ace342eb…, matching every H5 `summary.json`
provenance) are valid for 81fdb6d.

## Findings (draft)

### Must-fix

- M1 (draft) cross-class descent breaks reclaimable-first and the decode
  backup contract. Details to follow.

### Should-fix

### Nit

## Verdict

(pending)
