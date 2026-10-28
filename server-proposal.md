# Server proposal: limits that connector_v2 works around

Status: proposed 2026-10-28, waiting on the user's decision. Until the
user decides, connector_v2 gets no new mechanism that compensates for any
item below. If an item blocks a gate or a model, the RSI loop stops and
reports here.

Evidence comes from the official and diag runs named in
`PROGRESS.md` "Server limitations", `models/Qwen3.5-4B/ITERATIONS.md`
and `reviews/complexity-e1bab80.md`. The connector mechanism ids
(M12-M22) come from that review.

## Summary

| # | server behaviour | effect today | proposed change | connector code it removes | priority |
|---|---|---|---|---|---|
| S1 | save errors are not classified by copy state: "pinned pool exhausted" (raised before any copy) and a part-way d2h failure both come back as gRPC Internal | the connector must assume a copy may still run after any Internal, so it holds fences and pins and pauses saves for a 10 s window (a heuristic) | return `SaveResponse{status.ok=false}` for every failure raised before a copy is submitted; for a failure after submission, reply only once no copy of that request is in flight | M22 quiesce window (~70 lines, 3 tests); M21 shrinks to "raise = fail closed" | high: simplest change, largest deletion per line |
| S2 | one pinned region per layer entry of a save call, freed only when every block in it is evicted | dead bytes stay pinned (pool used − resident up to 0.83 GiB on Qwen3.5-4B G5 0.75); the connector splits RPCs per step and per request to keep regions small | make per-block allocation the default (`--blockwise-alloc` already exists; diag runs show pool used − resident 0.00 GiB) or account regions per block | M16 (one entry per request) becomes optional; M15 goes with M12 regardless | high: option exists, needs a default flip plus a perf check |
| S3 | reclaim frees a fixed 512 blocks per round and can over-evict to an empty read cache | Qwen3.5-4B G5 0.75 loses the shared prefix for a whole pass in most runs (dc4a4e0 4 of 5, 33666ce 1 of 3, save_batch_steps=1 1 of 2, blockwise 1 of 2); fragmentation is only a contributing factor | size reclaim to the need (`clamp(resident / 64, 1, 512)` was H1) and find the over-eviction trigger (candidates: two ranks racing in reclaim; regions still held by in-flight copies) with the server's per-call debug log | none directly; it caps the hit rate at small capacity, so no connector change can recover it | medium: hit-rate only, root cause still open |
| S4 | loads run serially on one load worker per device, FIFO, one CUDA event sync per task; one bad lease fails a whole multi-request load call | async loads land over 2-12 steps vs native's 2-4; H2D 14.8 GB/s vs native ~22 GB/s; a sync load waits behind older async copies (0.34-0.82 s per warm pass at TP2) | batch copies per step across requests, give sync loads priority (or a separate queue), fail per lease instead of per call | M9 defer-behind-in-flight (117 lines) and the sync/async demotion rules lose their reason; enables one load RPC per step | medium: main remaining perf gap vs native at TP2 |
| S5 | eviction is per (hash, group) key, not group-coherent | a resume point needs every group; measured negligible on Qwen3.5-4B (≤ 4 of ~7400 lookups) | evict a block's groups together | none today; may matter for gemma-4's sliding window | low: watch on gemma |
| S6 | server copies run in their own CUDA context and time-slice with vLLM | ~0.85 ms/step extra at TP2 while loads run | copy engine / stream priority choice on the server | none (the connector can only copy fewer bytes) | low |

## Existing server options to rule out first

Before any item above counts as a server limit, the options the server
already has must be measured on that layout. Not done yet:

- S4: the `kernel` transfer backend was measured only on Qwen3-4B (64 KiB
  pages, +26% per load, rejected). DSV2 (MLA) and Qwen3.5-4B (Mamba
  checkpoint pages) inherited `direct` without a measurement. Queued after
  Q1.
- S2: `--blockwise-alloc` has two diag runs on Qwen3.5-4B (fragmentation
  gone, prefix loss not). No perf measurement yet.

## What the connector does meanwhile

- Existing compensations stay as they are; Q1 may still delete those
  that measure as unneeded (it removes code, it does not add any).
- No new mechanism whose reason is an item above.
- A gate or model blocked only by an item above stops the loop and is
  reported to the user with the run and the item number.
