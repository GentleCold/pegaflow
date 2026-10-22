---
name: pegaflow-new-model
description: Onboard a new model to the PegaFlow vLLM connector in RSI mode (spec dump -> adapter/policy -> gates -> review). Use when adding support for a new model or KV layout, or when resuming the RSI effort in docs/rsi/.
---

# PegaFlow new-model onboarding (RSI loop)

This skill is the main process for supporting a model. Keep improving it: when a
step was missing, wrong, or slow, edit this file in the same iteration and log
the change in `docs/rsi/PROGRESS.md`.

## 0. Resume first

1. Read `docs/rsi/PROGRESS.md` (state, last iteration, open questions).
2. Read `docs/rsi/GATES.md` (what "done" means). Never relax a gate.
3. Read `docs/rsi/models/<model>/ITERATIONS.md` if it exists.
4. Write what you are about to do into PROGRESS.md before doing it.

## 1. Isolation

Do not read, grep, or copy:
- `python/pegaflow/connector/**` (old connector)
- model branches in `pegaflow-server/src/service.rs`, `pegaflow-core/src/instance.rs`,
  `pegaflow-core/src/offload.rs`
- git history of past model adaptations

Derive everything from: the vLLM 0.32 source, the model's spec dump, the
`pegaflow-semantics` API, the new connector package, and gate output. The old
connector is a black-box baseline (run it, read its numbers, not its code).

## 2. Loop for one model

1. **Spec dump** (G0): start vLLM with the dump hook, write
   `docs/rsi/models/<model>/spec.json`. Note groups, spec classes, block sizes,
   page bytes, TP shard info.
2. **Map to semantics**: each group -> (Kind, Shard, unit). If no existing Kind
   fits, propose a new one in `pegaflow-semantics` with V1 vectors, log it in
   PROGRESS.md "Semantics changelog" (no need to ask the user), and have the
   review agent audit it.
3. **Adapter/policy**: add the minimum adapter code for unknown spec classes;
   put model knobs in `models/<model>`. Prefer policy changes over adapter changes.
4. **Gates**: run G0-G4 on the remote box (skill `remote-server-dev`), save the
   run record under `~/kexi/rsi-runs/`, paste the summary into ITERATIONS.md.
5. **Optimize**: if hit efficiency E or perf is below target, form a hypothesis
   from metrics (fallbacks per group, unused loads, write amplification), change
   policy, rerun. One hypothesis per iteration, record result even if negative.
6. **Review**: spawn an independent review agent (fresh context) with the diff,
   gate summary, and isolation list. It checks: correctness of semantics changes,
   gate integrity (nothing relaxed), copying from the old connector, regressions
   on earlier models. Fix or justify every finding in ITERATIONS.md.
7. **Accept**: mark the model done in PROGRESS.md with the accepted numbers.

## 3. Lessons (append)

- vLLM 0.32 + flashinfer-python 0.7.0.post1: remove stale `flashinfer-cubin`
  (version check aborts startup; no 0.7 cubin on PyPI).
- On .3, ldconfig resolves `libcudart.so` to CUDA 12.5; a cuda-13 server build
  aborts in preflight. Export
  `LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib` for gate runs.
