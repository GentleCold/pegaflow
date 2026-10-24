# Review: P5 Qwen3.5-4B it0 + it1

Scope: `feat/rsi-qwen3.5-4b` cc12f5e..4ec49a9 (spec + mapping, `adapters/mamba.py`
+ `LayerDecl.layout`, it1 core change, it1 docs). The it2 plan (3379a66) landed
during the review and is cited only where a finding affects it. No official
gate exists for this model yet (hybrid gate pending), so this covers code,
semantics and the agent's own run records.

What I checked:
- the diff;
- vLLM v0.32.0 source, fetched with curl: `v1/core/sched/{scheduler,output}.py`,
  `v1/core/{kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager}.py`,
  `v1/kv_cache_interface.py`, `v1/worker/{utils,kv_connector_model_runner_mixin}.py`,
  `model_executor/layers/mamba/{abstract,mamba_utils}.py`,
  `model_executor/layers/mamba/gdn/{base,qwen_gdn_linear_attn}.py`, `config/cache.py`, `envs.py`;
- the run records and event logs under `~/kexi/rsi-runs/` (read-only), and
  `~/kexi/rsi-tools/p5/*`;
- the local default tests.

I did not read the old connector, the pd/nixl connectors, the model branches
in pegaflow-server or pegaflow-core, or git history of model adaptations.

## Summary

The mapping and the core change are correct against vLLM 0.32. GDN maps to
`Checkpoint(528)`. Saves take the exact block vLLM offers for the boundary
this step computes, and they finish synchronously in that step's
`wait_for_save`, before any later step can zero, copy into or reuse the
block. Loads are group-coherent by construction. Semantics, key derivation
and existing namespaces are unchanged.

Most of the agent's numbers reproduce from the run records. The main
exception is the order3 attribution: the p4 divergence happens before p4's
preemption. The ordering check also ran on a tree that was not the committed
revision. Neither is a code defect, and neither is a must-fix.

## Verified

Semantics (`pegaflow-semantics/src/groups.rs`, untouched in the range):
- `needed(Checkpoint, p) = [p/bt - 1, p/bt)` and `saveable = [after/bt - 1]`
  only when `after % bt == 0`. Both match vLLM's state column
  `(p - 1) // 528 = p/528 - 1` (`MambaManager`, `remove_skipped_blocks`;
  `abstract.py` `bind_kv_cache`).
- Boundary cases (`resolve` caps at `P - 1` and steps by lcm 528):
  - P < 528 and P = 528 give 0. vLLM caches nothing there either.
  - P = 529 gives p = 528, needing block 0. vLLM stops the chunk at
    `get_mamba_prefill_checkpoint_position(529) = 528`, so that state is
    computed and offered.
  - P = 1056 (len % 528 == 0) gives p = 528, not 1056, because one token
    must be computed. vLLM's forced stop is also `floor(1055/528)*528 = 528`.
  - No off-by-one found.
- `git diff --stat 6545352..4ec49a9 -- pegaflow-semantics pegaflow-server pegaflow-core python/src` is empty.

Namespace:
- `LayerDecl.layout` is only appended to the connector's namespace
  fingerprint (`core/layout.py:156-159`). `Worker.register` never reads it,
  and nothing about it reaches the server.
- Run logs show the namespaces did not move. Qwen3-4B is `83e5c20d` in both
  P3 it2e and P5 nonreg-b. DSV2 is `65acd040` in both the P4 it2 slice and
  P5 nonreg-b.
- "Different bytes give a different namespace" holds for every variation
  vLLM 0.32 offers:
  - `mamba_cache_dtype` / `mamba_ssm_cache_dtype` change `dtypes`;
  - `VLLM_SSM_CONV_STATE_LAYOUT=DS` turns conv `(3, 8192)` into `(8192, 3)`
    (`_orient_conv_shape`);
  - a different padding changes the segment size.
- The ssm state shape `(v_heads/tp, v_dim, k_dim)` is fixed by
  `gated_delta_net_state_shape` for every backend. The prefill and decode
  kernels share the tensor in place, so there is no hidden transposed
  variant at equal shapes.

Adapter (`adapters/mamba.py`):
- The view check matches `bind_kv_cache` (`[B, 1, 1, real]` int8, pages
  `page_size_bytes` apart).
- Copying the padded page is safe. Group tensors alias at offset 0 and block
  ids come from one pool, so the 16 KiB tail of a GDN-owned block belongs to
  no one else.
- TP: conv channels and v-heads both divide by tp
  (`gated_delta_net_state_shape`), so `Heads(rank, tp)` / `Replicated`
  (`tp_replicated`) is right. Only `seal_plan` covers TP2; there is no TP2 run.

Ordering against vLLM 0.32:
- `cache_blocks` runs in `allocate_slots` with
  `num_tokens_to_cache = computed + new`. The offer for boundary `after`
  therefore arrives in the scheduler output of the step whose forward
  writes it.
- `wait_for_save` runs after that forward (`kv_connector_model_runner_mixin.py:92`).
  The worker marks the compute stream and blocks until the server's copy is
  done (`worker.py:317-332`).
- The next step's Mamba precopy and attention zeroing are enqueued only
  after that.
- A request freed in the same `schedule()` has its pending offers dropped
  by vLLM (`pop_blocks_for_free`), and it is not in `scheduled`, so no save
  can read a reallocated block.
- Offers drained for other reasons (a loaded state cached by
  `_update_waiting_for_remote_kv`, older boundaries) never match `saveable`
  for that step and are ignored (`_offered_ids`, `scheduler.py:99-108`).
- Event logs: every checkpoint save in R, A and C used a block offered in
  the same step. Checks: order2 42/66, order3 39/78, R 36/36.

Loads and group coherence:
- For an external hit at p, base `add_local_computed_blocks` /
  `allocate_external_computed_blocks` (not overridden by `MambaManager`)
  null out the local Mamba blocks. They allocate one fresh block per Mamba
  group at `p/528 - 1`.
- `_assign` (`scheduler.py:266-303`) loads it for all three GDN groups plus
  attention blocks `[local, p/528)`. `skip_zeroing_group_ids` covers all
  four groups.
- `resolve` picks only a p where every group holds its needed blocks.
  `_assign` re-checks that, and returns None if any group's lease lacks the
  block.
- On a partial hit across groups 0-3 the hit drops to the longest common p,
  or to 0.
- A per-request load failure is reported via `failed_recving`. vLLM
  (recompute) resets `num_computed_tokens = 0` and frees the blocks, so a
  half-applied multi-group load is never cached. Sync loads (which vLLM
  cannot map with more than one group) are off for this layout
  (`connector.py:144`).

Non-regression (summary.json, read directly):
- Qwen3-4B `2026-10-24-qwen3-4b-p5-it1-nonreg-b` at rev c4c1615:
  - every v2 verdict passes;
  - E_ext 1.0 / 1.0;
  - warm TTFT p50 0.0930 / 0.0917 vs 0.2099 / 0.2125.
  The `from=` lines point at `~/kexi/pegaflow-rsi-qwen35`.
- DSV2 `2026-10-23-dsv2-lite-p5-it1-nonreg-b` (merged tree 7f51197, its
  `mamba.py` md5-identical to the branch):
  - loads are 10390 / 358088 matched, 0 mismatched;
  - split-only failures equal P4's;
  - cold replay TTFT p50 is 0.1883 vs 0.1709 (+10.2%), the only non-split
    fail. The P4 slice failed the same check at +6.4% and also failed cold
    req/s.

  The MLA path sees only `offers=None` and the `_count_miss` bookkeeping, so
  I accept "no regression". The cold-TTFT check is noise-bound across GPU
  pairs, so this is not proof.

Policy rules:
- `policy.py` has no model or spec class names. It is 110 lines with 2
  knobs, and this range adds no knob and no `models/*.toml`.
- Checkpoint behaviour is Kind-driven in core (`g.sem.kind == "checkpoint"`,
  `scheduler.py:493`, `connector.py:172-174`).

Isolation:
- The diff touches only `connector_v2/**`, `tests/test_connector_v2_core.py`
  and docs.
- Everything in the adapter and core change is derivable from the vLLM 0.32
  source (`MambaSpec` fields, `bind_kv_cache`, `boundary_state_offloads`
  docstrings) and `spec.json`.

Tests:
- Locally (`/tmp/rsi-venv`, stub cases): `tests/test_connector_v2_core.py`
  gave 111 passed, 63 skipped (native semantics absent).
- The new tests are table-driven with ids, have no GPU or vLLM dependency,
  and protect real contracts: offer selection, state-at-p load, namespace
  layout, the V1 namespace vector, `seal_plan` at TP1/TP2.
- The checkpoint edge math lives in the untouched semantics crate's vectors.
  No Testing Principles violation.

## Must-fix

None.

## Should-fix

1. **The order3 p4 divergence is misattributed** (`docs/rsi/models/Qwen3.5-4B/ITERATIONS.md:291-294`;
   it2 plan `:417-433` repeats it as cause (a) "after preemption").

   p4 in C is req `11-93212282`. Timeline from the order3 event log:
   - external hit at 1584, loaded around C step 3059;
   - decode;
   - preempted at C step 3572;
   - reloaded the same 1584 hit, with the same shas;
   - recomputed 1584→2112 (step 3688, 528 tokens).

   A resumed 528-token chunk ending at 2112, with a save of the 2112 block
   hash, needs `num_tokens >= 2112` (prompt 1900). So p4 had produced at
   least 212 output tokens before its preemption. The divergence is at output token 152, so
   it happened before p4 was preempted. In A, p4 (req 4) was never
   preempted (A preempted reqs 1, 3 and 5).

   Token 152 sits at position 2052, inside `[1584, 2112)`. So the only
   connector inputs to p4's C computation up to the divergence were:
   - the three GDN states at 1584 (byte-equal to A's saves);
   - full blocks 0-2. Blocks 0 and 1 load with identical layer-0 shas in
     all 7 loads of those hashes (`bdbbc750…`, `6f8bebe0…`). Block
     `a2bb744a` loads identically in both C loads (`eff3fdff…`), but A's
     save of it was not captured.

   Batch non-invariance during decode is still plausible: p4 decoded beside
   reqs 10 and 12 in C, beside reqs 3 and 5 in A, and GDN has no
   batch-invariant mode. But that cause is batch composition from the start
   of decode, not "after preemption".

   - Failure scenario: it2 leg 1's "batch control under the order3 tight
     config, so preemption happens" reproduces a divergence. The team
     concludes "preemption non-invariance" and declares `batch_invariant`
     on that basis. A real full-attention save error on `a2bb744a`
     (hypothesis (c)) would then never be tested, because leg 2 is skipped.
   - Fix:
     - correct the attribution;
     - make leg 1's batch control a plain batched-vs-alone decode
       comparison, no preemption needed;
     - always run leg 2's full-attention byte comparison for `[0, 1584)`.
       It is cheap and the tool already captures the C side.
     - Also log the top-2 logit margin at the divergent token. A
       near-tie (< ~1e-2 in fp32) supports rounding.

   Would the hybrid gate catch a real bug here? `G2.digest_self` would
   catch wrong bytes (stale checkpoint, bad full block) only if the Mamba
   digest is taken in the offer step (as the it2 plan says) and the
   workload actually preempts. With an unlimited pool and 481 GPU blocks,
   gsm8k and replay may never preempt, so ask the gate owner for a
   preemption-forcing slice or the G5 `cap` pass. Output-level divergence
   from non-invariance shows only in `G2.*_outputs`, which needs the
   model's invariance flags in gate.json.

   Same item: `PROGRESS.md:101` says "ordering check passed". It should say
   "passed, one open output divergence (order3 p4)".

2. **The ordering check did not run on the committed revision** (`ITERATIONS.md:267-299`).

   The order and order2 runs finished at 22:27:50 UTC, and order3 at
   22:34:21 (`status` mtime). On `~/kexi/pegaflow-rsi-qwen35`, though:
   - `connector.py`, `core/layout.py`, `core/scheduler.py`,
     `adapters/{__init__,mamba}.py` have mtimes 22:35:21-22:35:42;
   - `.rsi-rev` (c4c1615) was written at 22:35:52;
   - c4c1615's commit time is 06:35:07+08:00 = 22:35:07 UTC.

   So the runs used the pre-commit working copy, and the record names no
   revision. The `mamba.py` pyc header shows source mtime 21:41:46 at the
   same size. The scheduler pyc was rebuilt at 23:35 from the later source,
   so its run-time content is unknown.

   My reading of vLLM 0.32 (Verified, ordering) supports the conclusion
   independently, so this is provenance, not a known defect.
   - Failure scenario: a late edit to `_offered_ids` or `plan_saves` before
     the commit changed which block is saved. The byte-equality evidence
     would then describe code that is not on the branch.
   - Fix: rerun order3 at the committed revision (`.rsi-rev` written first)
     as part of it2 leg 1, and record the revision with every tool run.

3. **`hit stats` counts calls, not requests** (`core/scheduler.py:258`, `:527-553`).

   `_count_miss` runs on every `get_num_new_matched_tokens`. vLLM calls
   that every scheduling pass while a waiting request cannot be allocated.
   order3 logged `lookups: 1445` for 12 workload requests plus a few 1-token
   reset requests. Under block pressure, `missing=` counts and `lost_tokens`
   multiply by the number of retries.

   - Failure scenario: the it2 plan (`ITERATIONS.md:409-413`) cross-checks
     connector `missing=` / `lost_tokens` against the gate's per-group
     fallback count in the G5 `cap` pass, and treats disagreement as
     blocking. That pass is exactly where requests wait for blocks and get
     re-queried, so the counts disagree with no bug, and the slice blocks.
   - Fix: count once per request. For example, count at
     `update_state_after_alloc`, or key the last counted `(req_id, p_full)`
     and skip repeats. Extend `test_miss_attribution_names_the_missing_group`
     with a repeated call for the same request.

4. **Sync checkpoint saves are an unmeasured G4 risk, and the it2 plan does
   not mention it** (`connector.py:172-174`; `ITERATIONS.md` it2 plan).

   With retention 528, every prefill chunk end saves 24 GDN pages (about
   52 MB) plus the attention block, synchronously, before the next step.
   order3 `save stats` show 8.6-19 ms per save step: `flush_ms` 42.9 over
   5 steps and 57.8 over 3, with up to 41 ms of the latter spent waiting on
   compute.
   - Failure scenario: the cold-replay `G4.replay_cold.*` checks (≤ 2%) fail
     on Qwen3.5-4B because every 528-token chunk stalls on a 50-70 MB copy,
     and it2 reads that as noise, as with DSV2.
   - Fix: add to the it2 plan that the cold-replay G4 delta is attributed
     using `save stats` (`flush_ms` vs step time). If it fails, the next
     hypothesis is to fence the offered block id, as vLLM's
     OffloadingConnector does (`ITERATIONS.md:194-195`), and save
     asynchronously.

## Nit

1. `adapters/mamba.py:23-25`: the docstring says the core still rejects
   checkpoint groups. it1 removed that check (`core/layout.py`). A reader
   would look for a rejection that is not there.
2. `ITERATIONS.md:143`: the it0 mapping table says stored bytes 2146304
   (unpadded). The adapter stores the padded 2162688-byte page, as
   `ITERATIONS.md:207-212` explains. Update the table so the record has one
   number.
3. `tests/test_connector_v2_core.py:1552-1572`: the namespace test varies
   only the dtype. Add the `VLLM_SSM_CONV_STATE_LAYOUT=DS` variant (conv
   `(8192, 3)`, same page size) as a second case. It is the real-world
   same-size byte-order variant, and it is covered today only by the shapes
   in `layout`.
4. `adapters/mamba.py:138-157`: `cache_config.use_replayssm` (written back
   "only on flush", Mamba2 decode) is not visible in `MambaSpec`, so the
   adapter cannot reject it. GDN does not use it today. Note it in the
   module docstring so a Mamba2 model does not inherit this adapter
   silently.

Verdict: accept it1
