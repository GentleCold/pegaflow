# gemma-4-E4B-it plan (P6)

Scope (2026-10-28): adapter + correctness only. No perf iteration, and no
connector mechanism that compensates for a server limit (`server-proposal.md`).

Model facts (`config.json` text_config): 42 layers, `sliding_window` 512, a
`full_attention` layer every 6th layer, the rest `sliding_attention`. It needs a
window adapter (sliding-window spec) next to `full_attention`.

## First window layout on pinned saves

Since the refactor (connector_v2 `ff6706a`), every layout pins its saved
blocks in vLLM's block pool until all workers report them saved. Before that,
window layouts would have flushed every step (M17). That branch was never run,
because no earlier model has a window group. gemma is the first real run of
pinned saves on a window layout:

- vLLM frees window blocks that slide out while the request runs. The pin must
  keep any such block a pending save still reads out of the free queue.
- Check G2 digests (`digest_self`, `digest_vs_baseline`) on the window group
  specifically. Watch long prompts (> 512 tokens), where blocks slide out
  mid-request, and the preempt workload.
- Watch the pin count against the KV pool. Window blocks are freed early, so
  pins may hold a larger share of a small pool than on full-attention layouts.
  Report it if it stalls scheduling; do not add a workaround.
- S5 (eviction per (hash, group), not group-coherent) may show up here: watch
  `hit stats` `missing=` per group.

## Server options to measure before settling

Per the server-options rule (rsi-gates SKILL steps 3 and 5;
`server-proposal.md` "Existing server options to rule out first"):

| Option | Arm | Applicable | Result |
|---|---|---|---|
| `transfer_backend` direct | v2 (default) | yes | todo |
| `transfer_backend` kernel | `PegaKVConnectorV2Kernel` (diag) | yes | todo |
| `page_first` | `PegaKVConnectorV2PageFirst` (diag) | only if the layout has a single storage group; full + window is 2 groups, so expected "not applicable" | todo |
| `--blockwise-alloc` | base server + flag (wrapper, noted in provenance) | yes | todo |
