# Complexity review: connector_v2 at e1bab80 (PR #473)

Scope: `python/pegaflow/connector_v2/{core/scheduler.py, core/worker.py,
connector.py, policy.py, models/default.toml}` and
`python/tests/test_connector_v2_core.py`, diff `88f0078..e1bab80` (41
non-merge commits). Evidence comes from commit messages, `models/*/ITERATIONS.md`
and `PROGRESS.md`. vLLM behaviour is cited from those notes. No GPU run, no
SSH, no code branch touched.

Size (lines, total / code without comments, docstrings and blanks):

| file | 88f0078 | e1bab80 | code at e1bab80 |
|---|---|---|---|
| core/scheduler.py | 552 | 1014 | 651 |
| core/worker.py | 473 | 832 | 573 |
| connector.py | 354 | 461 | 331 |
| policy.py + default.toml | 124 | 156 | |
| test_connector_v2_core.py | 1333 (27 tests) | 2793 (64 tests) | |
| README.md | 161 | 301 | |

The reading this review starts from comes from the TP2 stop record
(Qwen3.5-4B ITERATIONS "TPOT split", "Qwen3-4B TP2 stop"). v2's direct
per-step connector cost is about 0.30 ms, native's about 0.67 ms. The gap
left over is vLLM's PIECEWISE step shape plus server load latency. So any
mechanism whose only claim is lower connector CPU earns nothing at the
current bar.

Markers in the evidence column:
- [old] measured on a rev before the fences (ebe69db), pins (5667cd7) or
  per-request entries (a59a254) changed the picture;
- [model] measured on another model or TP than where the code runs;
- [base] measured against `baseline` (probe only), not native;
- [noise] effect inside the ~3 pp TP2 run spread, or recorded as neutral;
- [never] never isolated by an A/B.

## 1. Mechanism inventory

Lines are approximate and include each mechanism's comments and docstrings
in scheduler, worker and connector. Tests are excluded.

| # | mechanism | lines | state it adds | category | commits | evidence (run, effect) | must know about |
|---|---|---|---|---|---|---|---|
| M1 | hit core: query, resolve, accept, `_assign`, `_recut`, `_fill_targets`, lease release | 230 | `_hits`, `_Lease`/`_Hit`/`_Lookup` | correctness | pre-88f0078, c4c1615 | G2 digests 0 mismatched in every official run | M8, M9, M10 |
| M2 | prefetch wait on `QueryLoading` | 15 | `_loading_since`, `loading_wait_ms` | perf (server API) | pre | none in range; the gate never exercises SSD/RDMA prefetch [never] | M9 (clock reset, 02b6767) |
| M3 | checkpoint saves from vLLM boundary offers | 35 | `offers` plumbing, `_offered_ids` | model support (Qwen3.5) | c4c1615 | `G2.digest_groups` [0..3], it2 official c5d3f04 | M19 |
| M4 | one storage rank and one writer per replicated shard | 10 (+67 layout) | `_storage_rank`, `_writer` | model support (DSV2 MLA TP2) | 0a956a1 | DSV2 it2 slice | none |
| M5 | sync short loads, `covered`/invalid blocks | 85 | `LoadPlan.sync/covered`, `_invalid`, `sync_load_tokens` | perf | pre (2ff447e) | Qwen3-4B it2 gsm8k warm TTFT sum 282 → 175 s, G4 fail → pass; H9(a) `qwen3-4b-p5-h9-tp2-*`: 0 and 64 no better than 256 | M6, M7, M18 (load-target fence) |
| M6 | fail closed after a failed sync load | 45 | `_unconfirmed`, `confirm_steps`, `_saves_disabled` | correctness | pre (it2d), 5c3ab63 | review-driven; path count 0 in gate runs | M5, M18 |
| M7 | sync loads sent first (H5 part 1) | 4 + stat | `sync_first_steps` | perf | b8501b2 | `qwen3-4b-p5-h5-tp1-{cf67695,daa5e11}`: "no measurable effect", 1-4 reordered steps per pass [noise] | M5 |
| M8 | H2 lookup skip (`cannot_beat_local`) | 35 | none | perf | 2efc6da, 800630d | `qwen3.5-4b-p5-it2-s3` → `-s4b`: gsm8k +5.3% → -0.1% vs baseline, server queries 6046 → 23 [base]; full-only: equal to "no block past the local hit" | M1, server LRU recency |
| M9 | H4 defer behind in-flight async loads | 117 | `_inflight`, `_inflight_refs`, `_deferred_since`, `_defer_inflight` | perf | e985938, 02b6767, dc4a4e0 | `qwen3-4b-p5-it4-defer-e985938` TP1: rps +2.5/+3.0% [noise], p50 -6.5/-12.4%, turn-0 ext 54k → 15k; `qwen3-4b-p5-it4-tp2-{c8a8842,02b6767}`: rps vs native -9.3% → -2.6% (one pair) [old]; Qwen3.5: never fires; native has the same wait | M1, M2, M10, `request_finished`, `update_connector_output` |
| M10 | H8 reuse an unallocated hit | 45 | `_Hit.at`, `HIT_REUSE_MAX_S`, `lookup_cost.reused` | perf | ee249b7, dc4a4e0 | `qwen3-4b-p5-h8-tp1-ee249b7`: own verdict "perf-neutral", lookup µs/call 1200 → 314; TP2 and cap075: `reused` 0 [noise] | M1, M9 (leak ordering bug fixed in dc4a4e0), `Policy.accept` (skipped) |
| M11 | async saves on a saver thread | 80 | `_jobs`, `_cv`, `_submitted`, `_completed`, `_SaveJob` | perf + correctness | pre (it2a) | Qwen3-4B it2a: step-thread save 10.4 s → 14 ms per 500-step window | all save mechanisms |
| M12 | cross-step save batching | 70 | `_held`, `_held_steps`, `_held_since`, `_held_bytes`, `_batch_steps`, `SAVE_HOLD_MAX_S`, knob `save_batch_steps` | perf | 72bffa6, 3e6f7a3, 253dd4c | DSV2 TP2 `dsv2-lite-p4-it4-rep-01`, `-it5-rep-01`: cold p50 KOff +5..+12% → +0.7..+2.4% vs baseline [old][base]; Qwen3-4B TP1 `qwen3-4b-p4-it4-qrep-b`: neutral (KOff at baseline); Qwen3-4B TP2: hold 258-274 ms, 0 pressure sends, "saves are not the cause" | M13, M14, M15, M18 (it created the hold H3/H7 fixed), M20 (idle stepping), M19 (off when pinned) |
| M13 | KV-pressure send | 30 | `send_saves`, `kv_usage_unknown`, `_kv_usage_warned`, knob `save_batch_max_kv_usage` | perf (repairs M12) | 3e6f7a3, 01dd020 | `dsv2-lite-p4-it4-press-02` (`--num-gpu-blocks 1200`): cold p50 0.454 → 0.29-0.32, KOff 0.30 [old]; Qwen3-4B G5: 3988 of 4076 RPCs are pressure sends | M12 |
| M14 | 16 MiB merged-RPC cap | 25 | `SAVE_RPC_MAX_BYTES`, `_wire_bytes` | correctness (M12 only) | bc0d267 | "no change at the gate's settings" [never] | M12 |
| M15 | one entry per held step in a merged RPC | 11 | none | server compensation | 43a3759 | `qwen3-4b-p4-ab-{a,b}` G5 0.5: 0.794 → 0.837 (it undoes M12's damage) | M12, server region pinning |
| M16 | one save entry per (group, request) | 8 | none | server compensation | a59a254 | `dsv2-lite-p4-m2-a59a254`: resident/C 0.81 → 0.92, hit at 0.75 +0.036; `qwen3-4b-p4-m2-a59a254`: +0.016..+0.030 | server region pinning |
| M17 | preemption flush and window flush-every-step | 35 | `StepPlan.flush`, `_flush_always`, `Worker.flush` waits | correctness | pre (it2a) | Qwen3-4B it5 preempt pass: ~34 flushes, 0 mismatched; window branch never run (no window model yet; gemma-4 has one) [never] | M18, M19 |
| M18 | block fences (H3 then H7): finished requests return False, writes to fenced blocks wait | 170 | sched `_saved_blocks`, `_fenced`, `_fence_steps`, `_fence_floor`, `_saved_through`, `_last_save_step`, `_positional_bt`; worker `_step_seqs`, `_done_seq`; `StepPlan.fence_step`; connector table extraction | perf + correctness | d201ac3, c8a8842, ebe69db, 5c3ab63 | `qwen3-4b-p5-it5-tp1-ebe69db`: p50 vs native behind → -10.5%, rps +13.3%, finished-held blocks 126-135/step → 0; TP2 `-it5-tp2-5c3ab63`: neutral, 0 fences. The hold it removed was mostly M12's: unbatched hold 17-23 ms mean vs 250-376 ms batched (DSV2 it4/it5) | M5/M6 (load and recompute targets), M12 (held jobs inside a fence), M17, M19, M20, M22 |
| M19 | pinned saves (H1) | 65 | `_pin`, `_unpin`, `_pins`, `_pin_saves`, `_job_steps`, `PegaWorkerMetadata` | correctness + perf | 5667cd7 | `qwen3.5-4b-p5-it2-s1` → `-s3`: cold TTFT +27.5% → +3.8%, req/s -6.2% → -1.7%, flush_ms 1518 → 0 [base] | M3, M12 (forces batch 1), M17, M18, M20, M22 |
| M20 | `saved_through` worker meta and `has_pending_saves` | 30 | `_job_steps`, `_last_step` | correctness | 5667cd7, ebe69db | unit-tested; needed so pins clear and an idle engine sends held saves | M12, M18, M19 |
| M21 | save outcome classification, fail closed | 45 | `_saver_dead`, `over`/`raised` in `_run_job`, `_save_settled` | correctness | 163f0b8, 809a0c6, 12ce60a, e49c584 | review findings; server limitation: a raise does not prove the copy is over | M18, M19, M22 |
| M22 | quiesce window and backoff after a raised save | 70 | `_quiescing`, `_quiesce_s`, `_stop`, `SAVE_QUIESCE_S`, `SAVE_QUIESCE_MAX_S`, stall log level | failure handling, server compensation | e4a2373, 33666ce | 48 Internal "pinned pool exhausted" in 45 s in one run (`qwen3-4b-tp2-slice-4bb7d8d` cap025); 0 save errors in the other 528 server logs | M18, M19, M21, shutdown |
| M23 | worker `save stats` | 150 | `SaveStats` (33 fields), `_Batch.polls/issued_at`, `report_save_stats`, counters on 6 paths under `_cv` | diagnostic | a31c073, baa8b49, b8501b2, daa5e11, ebe69db, 01dd020, 33666ce | consumers were one-off scripts archived at the TP2 stop; `rsi-gates` does not parse the line | all worker paths |
| M24 | scheduler `hit stats`, miss attribution, `lookup_cost` | 70 | `hit_stats`, `lookup_cost`, `_counted`, `_hit_dirty`, `_hit_logged_at`, `HIT_LOG_S` | diagnostic | 863cc99, 25fcccb, bc9c38a, baa8b49 | `missing=` attributed the Qwen3.5 G5 losses; `lookup_cost` fed H8 | M1, M10, `request_finished` |

## 2. Verdicts

| # | verdict | correctness need | perf claim |
|---|---|---|---|
| M1-M4 | keep | yes (G2) / model support | none |
| M2 | keep | Loading would otherwise be a miss | unmeasured, 15 lines |
| M5 | keep | none | real: hard gsm8k bar, re-screened by H9(a) on the current tree |
| M6 | keep | yes: stops wrong KV entering the store | none |
| M7 | simplify: keep the 2-line sort, drop `sync_first_steps` | no | none measured; the FIFO-loader argument is sound and it costs a sort |
| M8 | keep | no | beyond the 2% hard bar on Qwen3.5 gsm8k |
| M9 | keep; simplify its state (section 3) | no | one TP2 pair beyond spread, TP1 inside it. Kept because native does the same wait and turn-0 ext matches native, not on the rps number alone |
| M10 | delete | no | its own record says perf-neutral; it cuts connector CPU, which the intro shows is not the gap |
| M11 | keep | yes | hard gsm8k bar |
| M12 | ablate-then-decide | no | only measured on DSV2 TP2 cold vs baseline before pins, fences and per-request entries; neutral on Qwen3-4B; caused one G5 regression and the hold that needed M13 and M18 |
| M13 | ablate (rides on M12) | no | exists only to repair M12 |
| M14 | ablate (rides on M12) | only with M12 | none |
| M15 | ablate (rides on M12) | server compensation | moot with one step per RPC |
| M16 | keep (8 lines); flag server | server compensation | beyond spread on G5 0.75 |
| M17 | ablate (with M18, replaced by M19) | yes for unpinned layouts | none |
| M18 | ablate-then-decide | yes for unpinned layouts | the TP1 gain was measured against the batched hold; untested against pins without batching |
| M19 | keep; candidate to cover every layout | yes for zeroing/COW/checkpoint | beyond spread on Qwen3.5 cold |
| M20 | keep | yes | none |
| M21 | keep | yes | none |
| M22 | simplify now (drop the backoff, 33666ce); delete after the server change | compensates server behaviour | none |
| M23 | simplify: keep `rpcs`, `failed`, `raised`, `flushes`/`flush_ms`, `fences`/`fence_ms`, `sync_wait_ms`, `max_job_ms`; the rest goes to the probe or behind `PEGA_DIAG` | `failed`/`raised` show a fail-closed path fired | none |
| M24 | simplify: keep `lookups`, `missing=`, `lost_tokens`, `deferred`; drop `lookup_cost` and the timing | `missing=` is the hybrid miss signal | none |
| dead | delete `Worker.pinned_saves` property (no caller) | no | none |

Server-compensating mechanisms, and what each reduces to if the server
changed (the server is human-owned; record these in PROGRESS "Server
limitations"):

| mechanism | server behaviour it compensates | server change | connector after the change |
|---|---|---|---|
| M22 quiesce + backoff | pinned-pool exhaustion is raised as Internal before any copy is issued (offload.rs phase-2 alloc), indistinguishable from a client-side Internal | return `ok=false` (or InvalidArgument) for every failure raised before a copy was submitted, and reply to an error only once no copy of that request is in flight | M21 alone: `ok=false` is a future miss, a raise fails closed for good. About -70 lines and 3 tests |
| M21 "raise may still copy" | the server gives no bound on a copy after an error | as above | `_save_settled` becomes trivial |
| M15, M16 | one pinned region per layer entry, freed only when all its blocks are evicted | per-block region accounting | M16 becomes optional; M15 disappears with M12 anyway |
| M8 side effect | a prefix query refreshes LRU recency of every hit block | none needed | none |

## 3. Structural issues

| issue | where | fix |
|---|---|---|
| Request state is spread over 9 maps keyed by `req_id` (`_requests`, `_hits`, `_loading_since`, `_inflight`, `_deferred_since`, `_saved_upto`, `_saved_blocks`, `_counted`, and `_unconfirmed` by block). `request_finished` pops 8 of them by hand. dc4a4e0 fixed a lease leak caused by the pop order. | scheduler | One `_Req` record per request (request, hit, wait start and reason, in-flight keys, saved_upto, saved blocks). Finish drops one entry and releases its leases. |
| Two step-to-job maps in the worker: `_step_seqs`/`_done_seq`/`_fence_of`/`_forget_done_steps` for fences, `_job_steps`/`saved_through` for pins. On the scheduler side `_saved_through`, `_fence_floor`, `_last_save_step`, `_pins` by step and `_fence_steps` by step track overlapping "done" points. | worker, scheduler | One deque of (step, last seq) in the worker and one `saved_through` in the scheduler. The pin-everywhere design (section 4) needs only `saved_through` and `_pins`. |
| Two block-lifetime protocols. Pinned layouts use `pin`/`saved_through`, unpinned ones use fences, flush and window flush. `StepPlan` carries `fence_step`, `flush`, `send_saves`, `kv_usage_unknown` and `step`, and each layout reads half of them. | all three files | Pins for every layout if A1 passes. |
| Knobs that never vary. `models/` holds only `default.toml`; no profile overrides any of the 4 knobs. `test_policy_defaults_match_shipped_toml` pins them. | policy | Make them constants until a profile needs one. The `accept`/`choose_saves` hooks stay as the RSI surface. M12 removal takes two knobs with it. |
| Diagnostics on the hot path. 33 `SaveStats` fields are updated under `_cv` on `save`, `_start_loads`, `_wait_sync`, `poll` and `_run_job`; one 33-field log line. The gate (`rsi-gates`) reads none of it. | worker, scheduler | Keep the fail-closed and fence counters; per-step timing belongs in `rsi/gates/rsi_probe.py` or the `PEGA_DIAG_CONN` hook, which already timed every connector method in the TPOT split. |
| Tests pin internals. 53 private accesses: `sched._fenced` (11), `worker._cv` (12), `_quiesce_wait`, `_batch_steps`, `_completed`, `_held`, `_settled`, `_plan`. Tests doubled to 2793 lines. | tests | Assert on contracts: plan fields, `poll()`, `saved_through()`, `take_invalid()`, and the fake RPC's call log. Inject the quiesce window and batch size through the constructor, not by patching attributes. |
| History in docstrings. `HIT_REUSE_MAX_S` cites a dated run, `SAVE_QUIESCE_S` cites "277 v2 runs, 3853 windows", `_release_held` cites an A/B. About 30% of scheduler and worker lines are comments or docstrings. The rsi-notes rule is that code branches carry current-state docs only. | scheduler, worker, README | Move evidence to rsi-notes; keep the invariant each constant protects. |
| `_pin_saves` is computed in both roles; the worker loads a `Policy` only for `batch_steps`. | connector | Goes away with M12. |

## 4. Target design

Mechanisms kept: M1-M9, M11, M16, M19 for every layout, M20, M21. M22 is
kept as a fixed window until the server change, then deleted. M23/M24
shrink to the counters listed above.

State model:
- Scheduler: `_reqs: dict[str, _Req]`, `_inflight_refs`, `_unconfirmed`,
  `_pins: dict[step, token]`, `_saved_through`, `_saves_disabled`,
  `_plan`.
- Worker: `_jobs` queue, `_submitted`, `_completed`, `_job_steps` deque,
  `_saver_dead`, `_quiescing` (until the server change), `_batches`,
  `_failed`, `_done`, `_invalid`.
- `StepPlan`: `loads`, `saves`, `step`. No `fence_step`, `flush`,
  `send_saves` or `kv_usage_unknown`.
- `handle_preemptions` becomes a no-op. `Worker.flush` stays for shutdown
  only.

Estimated size if the A1 pin-everywhere arm passes (section 5):

| file | e1bab80 | target | after the server change |
|---|---|---|---|
| core/scheduler.py | 1014 | ~780 | ~780 |
| core/worker.py | 832 | ~560 | ~490 |
| connector.py | 461 | ~420 | ~420 |
| total | 2307 | ~1760 | ~1690 |

Moving history out of the docstrings saves roughly another 100 lines. If A1
keeps the fences and only batching goes, the total is ~1950.

Invariants the simplification must keep, each a unit test against public
behaviour:

1. A GPU block in a plan's saves stays pinned in vLLM's block pool until
   `saved_through >= plan.step` on every rank. A raise that does not prove
   the copy is over keeps it pinned for the window, or for good once the
   saver is dead.
2. `saved_through` is the min over ranks, never decreases, and never passes
   a job whose RPC has not returned (ok, `ok=false`, `ValueError`) or whose
   compute wait has not failed.
3. No save is planned for a request whose blocks touch an unconfirmed sync
   load, and none at all after vLLM reports a failed sync load.
4. A sync load is reported only after the server state is final. A failed
   one reports exactly its covered blocks as invalid, a failed async load
   reports `failed_recving`, and every async load is reported exactly once.
5. `accept` can only lower `p`. `choose_saves` can only drop items, and a
   forged item raises.
6. Every lease is released exactly once: on a miss, a stale hit, a deferral,
   a plan failure, finish and shutdown.
7. A deferral holds no lease and ends on `finished_recving`/`failed_recving`,
   on finish, or after `loading_wait_ms`.
8. Checkpoint groups save only from vLLM's offer for the boundary this
   step's forward computed. Only the writer replica saves.
9. `has_pending_saves` is true while any pin or planned save is unreported.
10. One save entry per (storage group, request).

## 5. Ablation plan

Default workload: Qwen3-4B TP1 and TP2 replay with in-run native. gsm8k
stays the hard ≤2% bar vs native. GPU-hour figures are estimates from the
durations of past runs of the same shape.

Accept-delete rule: delete the mechanism unless the default arm beats the
ablated arm by more than the ~3 pp spread (rps, TTFT p50, or a G5 hit at
the same C/U) and that margin holds on a repeat. The ablated arm must
still pass every G2/G3 verdict and the gsm8k bar.

| order | change | toggle or patch | workload, model/TP | lines at stake | GPU-h (est.) |
|---|---|---|---|---|---|
| Q0 | delete M10, the dead property, the M7 stat, and the M23/M24 trims; knobs to constants | code only. M10's deletion returns to the measured ebe69db state | CI, source-only gate, vLLM e2e correctness | ~200 | 0 |
| Q1 | screen M12+M13+M14+M15 and M17+M18 in one run per config. Arms: `native`, `v2` (e1bab80), `v2-nobatch` (profile `save_batch_steps = 1`, no code), `v2-pinall` (`connector._pin_saves` returns True: batch 1, no fences, no flush) | arm `v2-pinall` is a one-line patch on a diag branch | (a) Qwen3-4B TP1 replay + gsm8k + preempt, the KV-pressured case where pins can hurt; (b) Qwen3-4B TP2 replay; (c) DSV2-Lite TP2 replay with the cold pass, where M12's gain was measured. Qwen3.5 already runs pinall. | ~340 | ~5 |
| Q1-G5 | same arms, cap only at C/U 0.5 | as Q1 | Qwen3-4B TP1 cap 0.5 (KV usage runs near 99% there, so a pinned block that stays out of vLLM's pool until its copy lands costs the most; batching off did better on G5 before) | (decides Q1) | ~1.5 |
| Q2 | repeat the winning Q1 arm against `v2` | as Q1 | the config where the margin was closest | | ~2 |
| Q3 | M22 backoff to a fixed window now; delete M22 after the server change | code only | CI; correctness under raises is unit-tested | ~70 | 0 |

Q1 decision table:

| outcome | delete | keep |
|---|---|---|
| `v2-pinall` within spread of `v2` on (a)-(c) and Q1-G5 | M12-M15, M17, M18 (~340 lines, ~470 test lines) | M19 for every layout |
| `v2-pinall` loses beyond spread on (a), `v2-nobatch` within spread everywhere | M12-M15 (~140) | M17, M18 |
| `v2-nobatch` loses beyond spread on (c) cold, confirmed by Q2 | nothing from M12 | M12 for unpinned layouts; re-ask whether M13 still fires after M18 |

Not queued: M5 and M8 (re-screened or beyond the hard bar on the current
tree), M9 (native parity; ablate only if a later change touches load
timing), M16 (8 lines, beyond spread on G5).

## 6. Guardrail for future RSI iterations

- Net-line budget. A perf hypothesis states its net code lines in
  scheduler + worker + connector, tests and docs excluded. It may add at
  most 40. More than that needs a gain of at least twice the spread (6 pp)
  or a hard-bar flip, confirmed on a repeat.
- Neutral means delete. A perf mechanism whose gate result is neutral or
  inside the spread is deleted, not kept because it "costs nothing" or
  "removes server load". M10 and M7 were kept under those reasons.
- Re-ablation. At every model acceptance, and after every 3 accepted perf
  hypotheses, run one screen with each older perf mechanism toggled off as
  its own arm, using the accept-delete rule from section 5. Each perf
  mechanism keeps a documented one-line toggle until that screen has run.
- Repairs count against the original. A mechanism added to repair
  another's side effect (M13, M14, M15 for M12; M18 partly for M12's hold)
  is listed under the original's row, and the next screen ablates them
  together.
- Server compensation is labelled. A connector mechanism that works around
  server behaviour names the PROGRESS "Server limitations" entry and the
  server change that would delete it.
- Diagnostics stay out of the core. New counters go into the gate probe or
  behind `PEGA_DIAG`. Hot-path counters are allowed only for fail-closed
  and fence or pin waits.
- Total cap. scheduler + worker + connector stay at or below 1800 lines
  after this cleanup. A change that exceeds it deletes the same amount.
- Reviewer checklist item, mandatory: "Which existing mechanisms does this
  interact with, and was each re-measured on this tree with in-run native?
  Net lines? Does any test touch a private attribute? Can an existing
  mechanism be deleted instead?"
