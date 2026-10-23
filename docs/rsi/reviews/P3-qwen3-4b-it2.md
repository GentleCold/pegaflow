# Review: P3 Qwen3-4B it2 (async saves + sync short loads)

Reviewer: independent review agent (fresh context), 2026-10-23.
Scope: `0bb44bf`, `2ff447e`, `b783f5d`. Verdict: **APPROVE-WITH-FIXES**.
The measured numbers stand (no load failure occurred in any run); the fixes
are on failure paths and future layouts.

## Must-fix

- **M1** A failed sync load still saves blocks computed on top of it
  (`core/worker.py:208-230, 276-288`, `core/scheduler.py:358-430`). vLLM runs
  the step's forward, the new full blocks attend to garbage and are queued
  under correct chained hashes; the server keeps the first copy of a key
  (`pegaflow-core/src/storage/write_path.rs:144`), so later readers get a
  wrong hit. Fix: drop saves of a request whose sync load failed until the
  rewind is visible; test that no save goes out.

## Should-fix

- **S1** On vLLM's V2 model runner the preemption flush runs after block
  zeroing / CoW (`v1/worker/gpu/model_runner.py:1736, 1261, 1265, 2003`;
  V1 is ordered correctly). Safe for Qwen3-4B (no zeroing, hash block ==
  block); unsafe for layouts with zeroing or `hash_block_size < block_size`.
  Docs (README, ITERATIONS it2a, SKILL lesson) overstate the guarantee.
- **S2** A sync load that times out (30 s) is reported invalid while the
  server copy may still land into reused blocks. Wait without a deadline or
  fail hard.
- **S3** Held-out replay slice required by the policy rules not run; the
  "replay hits p50 ≈ 720 tokens" claim has no artifact.
- **S4** A dead saver thread or hung save RPC hangs `flush()` forever and
  leaks held blocks; advance `_completed` in `try/finally`.

## Nits

- `.rsi-rev` one commit behind the synced tests (harmless).
- `_wait_sync` spins with `time.sleep(0)`.
- `policy.py:12` docstring names a model class (review grep rule).

## Verified by the reviewer

Hold/release of finished requests, idle-engine draining, preemption bookkeeping,
prefix-cache sharing, event/stream ordering, RPC client thread safety,
MultiConnector + RsiProbe interplay, sync-load ordering and fallback condition,
isolation (no diff outside connector_v2/tests/docs/SKILL), unit tests (121
passed on .44), and every slice number in ITERATIONS/PROGRESS against
`summary.json`.

Not verified: runtime preemption flush (0 flushes), `Event.synchronize`
releasing the GIL, runtime behaviour of failed/timed-out sync loads.

## Resolution

Filled in by it2c (see `models/Qwen3-4B/ITERATIONS.md`).

| ID | Resolution |
|---|---|
| M1 | pending (it2c) |
| S1 | pending (it2c) |
| S2 | pending (it2c) |
| S3 | pending (it2c) |
| S4 | pending (it2c) |
| nits | pending (it2c) |
