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

1. **Spec dump + gate self-test** (G0): any gate run writes the dump (the
   `RsiProbe` stacked in every arm). Run the self-test command from GATES.md
   ("Self-test at onboarding"): plain, baseline and the four `selftest:` arms on
   128 gsm8k questions. Copy `RUN_DIR/baseline/gsm8k/probe/spec.json` to
   `docs/rsi/models/<model>/spec.json`, and have the gate owner write
   `docs/rsi/models/<model>/gate.json` (batch invariance, backend,
   `mamba_cache_mode`). Accept the model's gate only if the baseline controls
   pass, honest passes and every negative arm fails both digest checks;
   otherwise the gate is blind on this model and must be fixed first. Note
   groups, spec classes, block sizes, page bytes, TP shard info. If
   `oracle.py` raises "no Kind mapping", or the self-test connector raises
   NotImplementedError (non-full groups), stop: the gate owner maps the spec
   class first (gates are not the agent's to edit).
2. **Map to semantics**: each group -> (Kind, Shard, unit). If no existing Kind
   fits, propose a new one in `pegaflow-semantics` with V1 vectors, log it in
   PROGRESS.md "Semantics changelog" (no need to ask the user), and have the
   review agent audit it.
3. **Adapter/policy**: add the minimum adapter code for unknown spec classes;
   put model knobs in `models/<model>`. Prefer policy changes over adapter changes.
4. **Gates**: on .3 from `~/kexi/pegaflow-semantics-rsi`, run
   `python -m rsi.gates.run_gate --model ... --tag <model>-it<N> --arms baseline,old,<module>:<Class>`
   (full command in GATES.md "Implementation"; always include `plain`). Run
   record lands in `~/kexi/rsi-runs/<date>-<tag>/summary.json`; paste verdicts +
   hit table into ITERATIONS.md. Iterate on small slices first
   (`--gsm8k-limit 64 --replay-convs 8 --replay-turns 6`); such runs are
   `official: false` and exit 1 by design. Accept only an official run
   (`summary.json` "official": true, empty `not_official_because`).
   The connector under test must read `PEGAFLOW_PORT` and tolerate an empty
   `kv_connector_extra_config`; vLLM is started with
   `kv_load_failure_policy=recompute`.
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
- `pegaflow-server` (PyO3-linked) needs the venv's libpython: `run_gate` adds
  `sysconfig LIBDIR` to `LD_LIBRARY_PATH` and sets `PYO3_PYTHON`/`PYTHONHOME`.
  Start it by hand the same way.
- No `uv`/`ruff` on the local Mac: sync with
  `COPYFILE_DISABLE=1 tar czf - rsi | ssh ... 'cd ~/kexi/pegaflow-semantics-rsi && tar xzf -'`
  (without COPYFILE_DISABLE macOS ships `._*` files that break ruff), run
  `uvx --index-url https://pypi.tuna.tsinghua.edu.cn/simple ruff format/check --config python/pyproject.toml`
  remotely, tar the result back.
- The feature dir on .3 is not a git repo: write `.rsi-rev` before a gate run so
  the run record has a revision.
- Long ssh commands: launch gate runs with `setsid nohup ... < /dev/null &` so
  the ssh session returns; poll the log instead of `sleep` > 2 min in one call.
- Cross-run byte equality of computed KV only holds with
  `VLLM_BATCH_INVARIANT=1` + FLASH_ATTN (default for gsm8k in `run_gate`).
  Without it ~2.5% of layer-blocks differ by rounding.
- Pegaflow-server NUMA split: by default the pool is divided across NUMA
  nodes, so one GPU sees `pool_size / nodes` (7.5 GiB on .3) and evicts.
  `run_gate` passes `--disable-numa-affinity --pool-size 160gb`; G3.no_eviction
  fails any run that evicts (the oracle assumes unlimited capacity).
- G3 scores only what the connector can add over vLLM's local cache (`E_ext`).
  Total hit rate looks fine for a connector that only serves the first wave.
- Digest references must be exhaustive to call a mismatch: gsm8k digests every
  request; replay digests every 4th conversation, so many replay loads are
  `uncertain`, which is expected.
- 4090 / Qwen3-4B defaults (gpu-mem 0.8, 8192 len): 4745 GPU blocks of 16
  tokens; gsm8k warm oracle hit rate 0.99, local 0.93, so E_ext is decided by
  the ~7% the connector must serve.
- `rsi_poison` runs through `POST /collective_rpc`; kwargs arrive as strings over
  HTTP, so call methods without kwargs (or cast inside).
- Garbage KV produces garbage text: workload parsers must not crash on it (the
  liar arm hit `float('1e999')`).
- Self-test arm signatures on Qwen3-4B (use them to judge a new model's
  self-test): liar = all external blocks unreferenced in self and mismatched vs
  baseline, acc 0; bitflip = exactly one mismatched block per request, outputs
  only partly differ (accuracy alone would miss it); offbyone = most external
  blocks mismatched, acc ~0.2.
- The local vLLM checkouts are not 0.32 (no `transfer_group_ids`,
  `KVCacheTensor.layer_stride`, `KVConnectorBlockState`). Read the real source
  with `curl https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/<path>`
  into `/tmp`. Key files: `v1/kv_cache_interface.py` (specs,
  `create_kv_cache_views`), `v1/worker/utils.py` (`allocate_kv_cache`),
  `v1/core/sched/scheduler.py`, `distributed/kv_transfer/kv_connector/v1/{base,multi_connector}.py`.
- Derive segments from the spec dump's `kv_cache_layout` and the layer view
  stride, not from "K then V". In vLLM 0.32 the page is `[H|N, N|H, C]` with
  `C = K | V`, so K and V are interleaved and a page is ONE segment. The
  server's gRPC register path has no block stride: block `b` sits at
  `data_ptr + b * (segments * bytes_per_block)`. Only layer-compact layouts
  (block stride == page) can be registered as is.
- The worker gets manager-block views (vLLM keeps `kv_caches` for connectors),
  but with kernel-block splitting dim 0 counts kernel blocks. Divide by
  `block_size / N`.
- Local default gate without uv: `python3 -m venv /tmp/rsi-venv && /tmp/rsi-venv/bin/pip install pytest ruff`.
  This runs the stub cases only; the native cases need the remote venv.
- Adapter declarations carry sizes, not byte order. Two layouts with equal
  sizes share a namespace unless something names the difference. The core
  hashes `kv_cache_layout`; an adapter must reject any other byte-order
  variation it accepts but cannot name (e.g. kernel-split HNC pages). Add a
  "different bytes give a different namespace" test with each new adapter.
- Make costs you defer observable in the first gate run (e.g. the worker
  `save stats` line), so the next iteration decides from data.
