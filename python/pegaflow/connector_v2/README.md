# connector_v2

vLLM KV connector driven by `pegaflow.semantics`. Load it with

```
--kv-transfer-config '{"kv_connector": "PegaKVConnectorV2",
  "kv_connector_module_path": "pegaflow.connector_v2", "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute"}'
```

Use `kv_load_failure_policy=recompute`. Under vLLM's default `fail`, any failed
load (for example an expired lease) errors the request. The scheduler logs a
warning at init when it sees `fail`.

Endpoint: `pegaflow_host` / `pegaflow_port` in the extra config, then
`PEGAFLOW_HOST` / `PEGAFLOW_PORT`, default `127.0.0.1:50055`.

## Layering

| Layer | Knows | Must not know |
|-------|-------|---------------|
| `connector.py` | vLLM connector API | models, policy |
| `core/` | semantics + server contract (query, lease, load, save) | vLLM imports, spec classes |
| `adapters/` | one vLLM `KVCacheSpec` class each | requests, server |
| `policy.py` | choices inside the semantic range | GPU layout, server |

- **Hits.** Each transferable group is queried. Storage group 0 is a full
  group and uses the server's prefix query, which can prefetch. The other
  groups use membership queries (local only for now). A model without a full
  group is `Unsupported`.
  - vLLM re-queries a waiting request every scheduling pass, so queries
    that cannot beat vLLM's local hit are skipped. The bound is
    `semantics.resolve` with every unqueried group's blocks counted as
    present, checked before any query and again after group 0. If it is
    not past the local hit, the remaining groups are not queried and
    create no lease (group 0's lease is released). A full-only layout
    skips only when no full block lies past the local hit.
  - A hit vLLM did not allocate (for example, its KV pool was short) keeps
    its leases until vLLM asks again for the request (every scheduling
    pass): then they are released and the request is looked up again (or
    waits, see below).
  - `semantics.resolve` gives the longest legal `p`. `policy.accept` may lower it.
  - Only the blocks in `semantics.needed(g, p)` that are past vLLM's local hit get loaded.
  - Leases that cover none of those blocks are released immediately. Unwanted
    blocks inside a kept lease get `None` targets.
  - If vLLM allocates a shorter hit than planned, and that hit is still a legal
    resume point, the load is re-planned. Otherwise the load is reported failed.
  - Each request gets its own `load` RPC, so one bad lease fails one request.
  - vLLM prefix-caches an async load's blocks only once the load lands. A
    request whose blocks another request is still loading asynchronously
    (with prefix caching on) is not looked up: `get_num_new_matched_tokens`
    returns `None` until those loads finish or fail (`finished_recving` /
    `failed_recving` in `update_connector_output`, or the loader finishes),
    so the shared blocks become vLLM's local hit and only the tail is
    loaded. It waits only if the in-flight blocks give a resume point past
    the local hit (`semantics.resolve`), holds no lease meanwhile, and gives
    up after `loading_wait_ms` (0 disables the wait). Counted as `deferred`
    in `hit stats`.
  - Short hits (at most `sync_load_tokens`, default 256) are loaded
    synchronously: vLLM schedules the request at once and `start_load_kv`
    waits for the copy before the forward (~1 ms). The wait has no deadline:
    until the server signals, its copy may still land in the blocks, so the
    load is neither failed nor done before that. Longer hits load
    asynchronously between steps, so one big copy does not stall the batch.
    A failed sync load is reported as `invalid_block_ids` (the blocks holding
    the hit's tokens), which vLLM recomputes. Sync loads are only used when
    vLLM can recover them: one KV cache group and no block zeroing
    (`needs_kv_cache_zeroing`); otherwise every load is asynchronous.
    The server copies a device's loads one at a time in arrival order, so
    the workers send a step's sync loads before its async ones.
    The server keeps the first copy of a key, so nothing computed on a load
    that may have failed is saved:
    - a request whose blocks include a sync-loaded block saves nothing until
      vLLM has processed that step's output (`max_concurrent_batches` plans
      later); the held range is then offered again;
    - once vLLM reports a failed sync load, the scheduler plans no save for
      the rest of the process (logged as an error). Under the recompute
      policy vLLM evicts nothing: the failed blocks and every block computed
      on them stay in its prefix cache under their correct hashes, so any
      later request (a sharer's next turn, a hit on a preempted loader's
      freed blocks) may extend bad KV. From then on every load is also
      asynchronous, so no further failed load enters vLLM's prefix cache.

    Residual risk: this protects the store, not vLLM's own outputs. Until the
    rewound holder recomputes, requests in this process that hit the failed
    blocks in vLLM's prefix cache (a sharer vLLM did not rewind, a hit on a
    preempted or aborted loader's freed blocks) can produce wrong tokens.
    vLLM 0.32 has no connector hook to evict them.

    An async load needs neither: vLLM schedules the request only after it
    processed the failure and reset its computed tokens.
- **Saves.** Each step, `semantics.saveable(g, before, after)` gives the blocks
  that became storable. `policy.choose_saves` filters them. The saves are
  asynchronous: `wait_for_save` records a CUDA event on the compute stream and
  queues the job; a saver thread waits on the event, then issues the save RPC.
  The step thread never waits on the GPU. Block lifetime: every block a plan
  saves is pinned in vLLM's `BlockPool` (`touch`, one reference per saved
  block) when the scheduler plans the save, and unpinned (`free_blocks`) once
  every worker reports that step's saves done (`PegaWorkerMetadata.saved_through`,
  min over ranks). A save is done once its RPC returned (the server replies
  after the GPU->CPU copy completed) or failed; with one FIFO saver per worker
  that covers every earlier job. A pinned block is never handed out again, so
  vLLM cannot zero, copy into or reuse it while it is read, whether its request
  finished, was preempted, or slid it out of a window. So:
  - `request_finished` always returns `False`: vLLM frees the blocks at once,
    as with its own `OffloadingConnector`, and the pins keep the ones a save
    still reads;
  - there is no flush and no fence in `handle_preemptions`, and saves are not
    batched across steps (each step's saves are one RPC, pins stay short);
  - `has_pending_push_work` keeps vLLM stepping until every pin is released;
  - without a bound block pool nothing is saved (fail closed);
  - a pin raises the block's `ref_cnt`; vLLM 0.32 reads `ref_cnt == 1`
    (`is_block_writable`) only for speculative Mamba blocks, so check this
    before enabling spec decode on a checkpoint layout.
  - a save the server reports failed still releases its blocks (a failed
    save is a future miss). The save handler reports engine errors as a
    gRPC status, so the RPC raises. Only the binding's `ValueError` proves
    no copy runs. The binding raises it for InvalidArgument, or before any
    RPC is sent; tonic's client never makes InvalidArgument, and the
    handler returns it before it copies or after its copy synchronized. So
    it counts as a failed save.
  - any other raise quiesces the saver (fail closed, then bounded
    recovery). For `Worker.SAVE_QUIESCE_S` (10 s) no save RPC is sent, new
    saves are dropped, and the raised job's pins (and those of every later
    job) hold. Then the job counts as done, the jobs queued behind it are
    dropped without an RPC, and saving resumes. Shutdown ends a window
    early. Each raise is logged and counted as `raised` in `save stats`.
    No step waits for the window, but the pins of its planned saves stay
    in vLLM's block pool until it ends. This covers server
    errors such as an exhausted pinned pool (Internal) or an unknown
    instance (FailedPrecondition). The binding raises those as the same
    class as transport errors (`PegaFlowError`), and the client also
    reports its own decode and HTTP/2 errors as Internal. The window is a
    heuristic, not a proof: no reply bounds a copy the server may still
    run, so it is kept well above the longest save job seen in gate logs.
    Any other BaseException out of the RPC (a panic in the binding) counts
    as a raise too. A saver thread that dies with no copy possibly running
    (its compute wait raised) stops saving for good: its pins are
    released and later saves are dropped.
    Server limitations:
    - The server seals a job's blocks only after its copy synchronized. But
      the job whose RPC raised may still be copying, and vLLM may reuse its
      blocks once the window ends. A GPU copy that fails part-way returns
      Internal with earlier copies still queued. A call the client drops (no
      reply) is cancelled on the server only when the connection notices.
      Closing that needs a server that answers only once no copy of the job
      runs.
    - An exhausted pinned pool fails in allocation, before any copy of the
      request is issued, so it needs no window. The server reports it as
      Internal, like a part-way copy failure, so the connector cannot tell
      them apart. A distinct reply for errors raised before the copy (for
      example `SaveResponse.status.ok = false`) would make it a plain failed
      save.
    - A save RPC that hangs keeps its pins (and every later job's) for
      good: the client has no per-call deadline, and the server may still
      be copying.
- **Checkpoint groups** (Mamba / GDN state, vLLM `mamba_cache_mode="align"`).
  vLLM's block table for such a group is null except the running state
  block, so positions do not say which block holds which state. Saves come
  from `SchedulerOutput.kv_connector_block_state.boundary_state_offloads`
  (`(group, block id, boundary tokens)`): `saveable` still decides which
  boundary a step computed, and only an offer for exactly that boundary is
  saved. A boundary vLLM did not offer (it retains only what
  `--prefix-cache-retention-interval` keeps) is not saved. An offered block
  holds its state only until vLLM frees or reuses it; the pin (see Saves)
  keeps it until the copy is done. Loads need nothing
  special: the needed block `[p - bt, p)` is always past the local hit, and
  vLLM allocates a fresh state block for an external hit.
- **Hit stats.** The scheduler logs `hit stats` (cumulative) at most every
  30 s while lookups run, whenever the last live request finishes, and at
  shutdown (vLLM may kill the engine core without calling it): lookups, and per set of storage groups missing at the
  longest resume point the full group backs, how often and how many tokens
  were lost, counted once per request and resume point (vLLM re-queries a
  waiting request every scheduling pass) (`missing=` with no group: the policy lowered `p`). This is the
  group-coherence signal for server eviction. Load counters:
  `deferred` counts requests that waited for another request's async load.
- **Namespace.** It is a hash of:
  - the adapters' declarations: group kinds, block sizes, per-rank segments
    and shards, transfer backend, page_first, and `LayerDecl.layout` when an
    adapter declares it (e.g. the Mamba state shapes and dtypes);
  - vLLM's `kv_cache_layout` (the page byte order, e.g. LBNHC `[N, H, C]` vs
    LBHNC `[H, N, C]`);
  - the parallel shape;
  - the model: path, revision, quantization;
  - the dtypes and vLLM's prefix hash algorithm.

  Declarations carry sizes, not byte order. So an adapter must reject any byte
  order that neither its declaration nor `kv_cache_layout` names. For example,
  `full_attention` rejects kernel-split HNC pages.
- **Save stats.** Every 500 `wait_for_save` calls (and at shutdown) each
  worker logs `save stats`: `steps`, save `rpcs` sent, `failed` saves,
  `raised` (save RPCs that raised with a copy possibly running; each starts
  a quiesce window), the step-thread wait `sync_wait_ms`, and `max_job_ms` (compute wait plus RPC
  of one save job). Per-step timing belongs in the gate's probe, not here.
- **Init checks.** These run on both roles:
  - unknown spec classes are `Unsupported`;
  - every TP rank's declarations must pass `semantics.seal_plan` (V0).

  At registration, every declared segment must equal `LayerView.bytes_per_block`,
  the size the server copies per segment.

## Adding a model

1. Start vLLM with this connector. An unknown spec class fails at init with
   `Unsupported("no adapter for <Class> (group i) ...")` on both roles.
2. Add `adapters/<name>.py` exposing `ADAPTER` with `group`, `layer` and `view`
   (see the `Adapter` protocol in `adapters/__init__.py`).
3. Add one line to `_REGISTRY`: `"<Class>": "<name>"`.

Registered adapters:

| Spec class | Module | Declares | Onboarded by |
|---|---|---|---|
| `FullAttentionSpec` | `full_attention` | `Group.full(block_size)`; one segment = the whole page (K/V interleaved per cell); `Heads` shard, replicas beyond `max_tp_shards` (checked against `kv_tp_replicas`); dense layer-compact pages only, no kernel-split HNC pages; rejects window, chunked, padded, packed, quantized and `non_causal` specs | Qwen3-4B |
| `MambaSpec` | `mamba` | `Group.checkpoint(block_size)`; one segment = the whole (padded) state page; state shapes/dtypes in `LayerDecl.layout`; `Heads` shard (or `Replicated` for `tp_replicated`); `align` mode only, rejects speculative/prefill-checkpoint blocks, per-token states, DCP | Qwen3.5-4B (in progress) |

The shared rules live in the semantics crate, so an adapter only declares
facts: the dependency kind, the tokens per block, the bytes per segment, and
the shard. It never decides what to load or save.

## Policy (the RSI surface)

`policy.py` is the place to tune behaviour. Its knobs are constants shared by
every model. The policy can only narrow what semantics allows:

- it sees immutable copies (`frozenset`s, a tuple of `SaveItem`s);
- `accept` is capped at the legal `p`, and the result is re-checked against
  the server answers;
- `choose_saves` may only return offered items. Anything else raises.

So a bad policy costs hit rate, never correctness.

## Tests

`python/tests/test_connector_v2_core.py` is part of the default (source-only)
unit gate. It uses a stub adapter, a fake server, and a pure-Python semantics
stub that is injected through the `semantics=` argument of `Scheduler` and
`build_layout`. Cases parametrized `native` re-run against the real
`pegaflow.semantics`, and `test_stub_matches_native` cross-checks the stub.
Both skip without the native extension.
