---
name: pegaflow-new-model
description: Onboard a new model to the PegaFlow vLLM connector in RSI mode (spec dump -> adapter/policy -> gates -> review). Use when adding support for a new model or KV layout, or when resuming the RSI effort (working memory on the rsi-notes branch, gates in docs/rsi/GATES.md).
---

# PegaFlow new-model onboarding (RSI loop)

This skill is the main process for supporting a model. Keep improving it: when a
step was missing, wrong, or slow, edit this file in the same iteration and add
one line to `PROGRESS.md` on the `rsi-notes` branch.

## 0. Resume first

1. `git -C .worktrees/rsi-notes pull --rebase` (create the worktree with
   `git worktree add .worktrees/rsi-notes rsi-notes` if missing).
2. Read the notes' `PROGRESS.md` (state, last iteration, open questions,
   decisions) and `models/<model>/ITERATIONS.md` if it exists.
3. Read `docs/rsi/GATES.md` (what "done" means). Never relax a gate.
4. Write what you are about to do into the notes' `PROGRESS.md`, commit and
   push it before doing it.

## Notes and code branches

- Working memory (progress, plans, iteration journals, hit-rate judgments,
  review outcomes, run tags, dated user decisions) lives only on the
  orphan branch `rsi-notes`, never in a code branch. Commit there in small
  `docs(rsi): ...` commits; `git pull --rebase` before every push, never
  force-push (several agents share it).
- Code branches carry current-state docs only (`docs/rsi/GATES.md`,
  `docs/rsi/models/<model>/{spec.json,spec-tp<N>.json,gate.json}`, this skill, package
  READMEs): no dated history, change logs, journals or review files. State
  a decision as a rule; its date and quote go to the notes' "Decisions".
- Compaction (notes `README.md`): `PROGRESS.md` <= 300 lines, each
  `ITERATIONS.md` <= 400, `gates-evidence.md` <= 300. When over budget, and
  at every model acceptance, rewrite instead of appending: current state
  first, each closed iteration one table row (iteration, hypothesis, result,
  run tag, commit), dead ends one line each, superseded numbers dropped. A
  review file on the notes branch lives only until its findings are
  resolved, then becomes one line in the model's `ITERATIONS.md`.

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
   ("Self-test at onboarding"): plain, baseline and the `selftest:` arms on
   128 gsm8k questions, at each TP the model is gated at (`--tp N`). Copy
   `RUN_DIR/baseline/gsm8k/probe/spec.json` to
   `docs/rsi/models/<model>/spec.json` for TP 1 and to `spec-tp<N>.json` for
   `--tp N` (the gate reads only the dump of the run's `--tp`; a TP without
   one is never official), and have the gate owner write
   `docs/rsi/models/<model>/gate.json` (batch / split invariance, backend,
   `vllm_args`, `gsm8k_vllm_args`, `lmcache_dist`). The self-test's `plain`
   arm runs the onboarding controls (`G2.plain_equals_baseline`,
   `G2.split_control`, `G2.batch_control`) on the full gsm8k set; when they
   pass, the gate owner copies `summary.controls_record` (run, rev,
   `gsm8k_limit` 0, the three verdicts, `inputs`) into gate.json
   `controls["tp<N>"]` (GATES.md "Onboarding controls"). A subset or failing
   run writes no record (`summary.controls_record_withheld` says why).
   Routine runs have no `plain` and are official only while that record is
   current: a probe, workload-driver, control-code, flag, dataset or
   gate.json setting change makes it stale, and the self-test with `plain`
   is rerun. Accept the model's gate only if the
   baseline controls pass, honest passes and every negative arm is caught;
   otherwise the gate is blind on this model and must be fixed first. Note groups, spec classes, block sizes, page bytes, TP shard info.
   If `oracle.py` raises "no Kind mapping", or the self-test connector raises
   NotImplementedError, stop: the gate owner maps the spec class first
   (gates are not the agent's to edit).
2. **Map to semantics**: each group -> (Kind, Shard, unit). If no existing Kind
   fits, propose a new one in `pegaflow-semantics` with V1 vectors, log it in
   the notes' "Semantics changelog" (no need to ask the user), and have the
   review agent audit it.
3. **Adapter/policy**: add the minimum adapter code for unknown spec classes;
   put model knobs in `models/<model>`. Prefer policy changes over adapter changes.
4. **Gates**: on the 4090 host (10.1.0.44) from the feature dir, run the
   command in GATES.md "Usage" with `--arms baseline,native,<module>:<Class>`
   and `--under-test-path`. `baseline` is the correctness reference, `native`
   every perf reference (both required). `old` and `lmcache` are optional
   reference arms: add them only when their numbers are wanted (lmcache then
   needs the model's overlay, gate.json `lmcache_dist`: Qwen3-4B
   `~/kexi/overlays/lmcache-pr4467`, DeepSeek-V2-Lite `~/kexi/overlays/lmcache`
   (0.5.6; PR #4467 crashes on MLA); a model without `lmcache_dist` cannot
   run it officially). `--cap-curve` adds the reported G5 curve (0.25 / 0.75)
   and stays official. Never use a wrong-KV reference number (e.g. a curve
   point whose `digest_self` fails, or lmcache where GATES.md "lmcache
   status per model" says not comparable). Add `,preempt` to
   `--workloads` on models with a Mamba state group. The run record lands in
   `~/kexi/rsi-runs/<date>-<tag>/summary.json`; put verdicts + hit table into
   the notes' ITERATIONS.md. Iterate on slices first (`--gsm8k-limit 64
   --replay-convs 8 --replay-turns 6`); they are `official: false` and exit 1
   by design. Accept only an official run (`"official": true`, empty
   `not_official_because`). The connector under test must read
   `PEGAFLOW_PORT` and tolerate an empty `kv_connector_extra_config`; vLLM
   runs with `kv_load_failure_policy=recompute`.
5. **Optimize**: hit-rate levels (E_ext, E_C) and replay perf vs native
   are reported, not gated (GATES.md "Hit rate", G4); gsm8k seconds within
   2% of native is hard. Stopping rule: iterate until you judge that no
   connector-side optimization is left, for hit rate and for perf; "close
   enough to native" is not a stop. For every accepted run write, in the
   notes' ITERATIONS.md:
   - a "hit-rate judgment": measured hit rate vs H* / H*_C and the
     reference arms (H_lru, native; old / lmcache only if run), every gap
     split into connector headroom, server, Kind / vLLM structure and
     noise;
   - a "perf vs native judgment" (GATES.md G4): the remaining gap in % for
     warm replay, cold replay and gsm8k, each split into connector, server
     (known server limits, not ours to fix), vLLM / native structural
     differences, and noise, with numbers;
   - the connector-side hypotheses tried (results, negative ones too) and
     why each remaining one is not worth trying, with numbers.
   While connector-side headroom remains, form a hypothesis from metrics
   (fallbacks per group, unused loads, write amplification, stranded bytes
   per group, save/load time), change the lowest layer that fixes it,
   rerun. One hypothesis per iteration, record the result even if
   negative. Hypothesis checklist (extend it):
   - window groups: save only the trailing window, not every block;
   - checkpoint groups: save only at boundaries that requests resume from;
   - state that cannot be resumed (`prefix_cacheable=False`, ring buffers): never save;
   - eviction must be group-coherent: a resume point needs every group's data,
     so evict a prefix's groups together and prefer evicting the tail;
   - scan resistance: one-pass sweeps must not flush reused prefixes;
   - save cost: no synchronous wait on the model step.
   Do not block a model on hit-rate level alone; correctness checks stay hard.
   Do not chase the highest hit rate: H*_C is a ceiling, not a target.
   The server is out of scope: gates run on the base pegaflow-server built
   from `feat/semantics-rsi`, and agents do not change or tune it. When the
   evidence points at the server (eviction, admission, pools, allocator,
   query), attribute the gap with numbers (GATES.md "Known server limits";
   new ones go to the notes' "Server limitations") and go on: explained,
   not fixed, never blocking. Connector code stays model-agnostic: per-model
   differences live only in `connector_v2/models/<model>.toml`.
   Policy complexity rules (keep policy growing with Kinds, not with models):
   - no model names and no vLLM spec class names in `policy.py`; decide only
     on Kind and numbers (block tokens, window, sizes);
   - every knob needs gate evidence that it helps; a toml sets only values
     that differ from `default.toml`; a knob no accepted model uses is deleted;
   - budget: about 150 lines of policy per Kind and at most 12 knobs in
     total; past that, refactor first (a refactor-only iteration);
   - a simplification is a valid iteration: accept it when every accepted
     model's gate numbers do not regress;
   - no tuning to the trace: report the accepted numbers on a held-out replay
     slice too (`--replay-skip` past the conversations you tuned on);
   - if policy logic keeps growing, it is in the wrong layer: propose a Kind
     (semantics, human-reviewed) or a server change (separate PR) instead.
6. **Review**: spawn an independent review agent (fresh context) with the diff,
   gate summary, and isolation list. It checks: correctness of semantics changes,
   gate integrity (nothing relaxed), the hit-rate and perf vs native
   judgments (numbers against `summary.hit_report` / `native_report`; the
   four-way attribution holds: nothing connector-side labeled server,
   structural or noise without evidence; the reasons for stopping are
   backed by numbers), copying from the old connector,
   regressions on earlier models, the policy complexity rules above (grep
   `policy.py` for model / spec class names, count knobs and lines, unused
   knobs), and that the connector does not touch the probe (GATES.md
   "Boundary", review checklist). It reads every `summary.scan` hit (flagged
   and allowlisted). The reviewer returns its report to you; it writes no
   review file in a code branch. Fix or justify every finding, then add one
   line to the notes' ITERATIONS.md (verdict, key findings, fix commits).
7. **Accept**: mark the model done in the notes' PROGRESS.md with the
   accepted numbers: the hit-rate table (local / external / total hit,
   oracle H*, E_ext; at capacity H*_C, H_lru, E_C) next to the G4 perf
   table, one row each for the arm under test (v2) and `native`, plus
   `baseline` for the correctness columns (gsm8k accuracy, preempt /
   resume, cap reference local hit). `old` and `lmcache` get a row only if
   run, else "not run (optional)" (or "not comparable" with the GATES.md
   reason). The G4 table shows v2 next to `native`: gsm8k seconds (hard, at
   most 2% worse), cold and warm replay TTFT p50 and req/s (reported), with
   the perf vs native judgment and its attribution. Accept only once the
   judgments say no connector-side optimization is left. Do not wait for the user: the
   review agent's pass is the acceptance; the user does one acceptance pass
   at the end. Compact the model's ITERATIONS.md (section "Notes and code
   branches").

## 3. Lessons

Keep this list current-state and short: when it passes ~40 items, condense
it (merge related items, drop ones the gate or code now enforces).

Host and build:
- 4090 host: 10.1.0.44 (jump host route), shared venv
  `~/kexi/pegaflow/python/.venv`. Export `CUDA_HOME=~/kexi/cuda13`, the venv
  `bin` and `~/kexi/cuda13/bin` on `PATH`, `LD_LIBRARY_PATH=~/kexi/cuda13-lib`
  (unversioned symlinks to the venv's `nvidia/cu13/lib`): the system toolkit
  is 12.x, a cuda-13 server build aborts when `libcudart.so` resolves to
  12.x, and FlashInfer JIT needs a 13.0 nvcc matching the 13.0 headers
  (`~/kexi/cuda13` comes from the venv wheels `nvidia-cuda-nvcc`,
  `nvidia-cuda-crt` and `nvidia-nvvm`, pinned to 13.0.x).
- A tree for `--under-test-path`, and the gate / feature tree the run
  starts from, must hold no `.pyc` and carry `.rsi-rev` (else provenance is
  unverified and the run not official): extract with `git archive <rev> | tar -x` (+ `echo <rev> > .rsi-rev`) or
  sync with `tar --exclude=__pycache__ --exclude='*.pyc'`; never import from
  it before the run.
- vLLM 0.32 + flashinfer-python 0.7.0.post1: remove stale `flashinfer-cubin`
  (its version check aborts startup).
- Build pegaflow-server with `PYO3_PYTHON` = the venv python (a uv venv has
  no pip: `maturin develop --uv`). `run_gate` adds the venv's libpython dir
  to `LD_LIBRARY_PATH` and sets `PYO3_PYTHON` / `PYTHONHOME`; start the
  server by hand the same way.
- No `uv` / `ruff` on the local Mac: sync with `COPYFILE_DISABLE=1 tar czf -`
  (else `._*` files break ruff), run `uvx --index-url
  https://pypi.tuna.tsinghua.edu.cn/simple ruff format/check --config
  python/pyproject.toml` remotely, tar the result back. Local tests without uv:
  `python3 -m venv /tmp/rsi-venv && /tmp/rsi-venv/bin/pip install pytest
  ruff`; that runs the stub cases only, the native cases need the remote
  venv.
- Long ssh commands: launch gate runs with `setsid nohup ... < /dev/null &`
  and poll the log instead of sleeping > 2 min in one call.
- The local vLLM checkouts are not 0.32. Read the real source with
  `curl https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/<path>`:
  `v1/kv_cache_interface.py`, `v1/worker/utils.py`,
  `v1/core/sched/scheduler.py`,
  `distributed/kv_transfer/kv_connector/v1/{base,multi_connector}.py`.

Gate reading:
- G3 scores only what the connector adds over vLLM's local cache (`E_ext`);
  a total hit rate looks fine for a connector that only serves the first
  wave. Qwen3-4B on the 4090 (gpu-mem 0.8, 8192 len): 4745 GPU blocks of 16
  tokens, gsm8k warm oracle 0.99, local 0.93, so E_ext is decided by the ~7%
  the connector must serve.
- The server pool is per NUMA node by default; `run_gate` passes
  `--disable-numa-affinity --pool-size 64gb`, and any eviction fails
  `G3.no_eviction`.
- Replay digests every 4th conversation, so many replay loads are
  `uncertain`; that is expected.
- Self-test signatures on Qwen3-4B: liar = all external blocks unreferenced
  in self and mismatched vs baseline, acc 0; bitflip = one mismatched block
  per request, outputs only partly differ (accuracy alone would miss it); offbyone = most external blocks
  mismatched, acc ~0.2.
- Garbage KV produces garbage text: workload parsers must not crash on it
  (`float('1e999')`).
- `rsi_poison` goes through `POST /collective_rpc`; kwargs arrive as strings
  over HTTP, so call methods without kwargs (or cast inside).
- G4's reference is the in-run `native`, never baseline (baseline is the
  correctness reference; its replay runs `cold` only). Hard: gsm8k seconds
  at most 2% worse than native (the regression floor). Read every `G4.*`
  verdict; the save path's cost shows in gsm8k and in the reported cold
  replay gap. A 64-question gsm8k slice cannot resolve 2% (~10 s): use
  the full 1319 when a G4 gsm8k check is the target.
- G4 replay cold and warm vs `native` are judged, not gated (`*_vs_native`
  reported, `summary.native_report`). vLLM's OffloadingConnector is the
  reference, not the finish line (Qwen3.5-4B: native warm TTFT p50 0.172
  s, v2 0.300 s, vLLM without a store 1.12 s): keep iterating while the
  connector-side part of the gap is not exhausted; compare v2's and native's load path
  first (sync vs async load, load round trip in steps, per-request load
  size) before touching policy. A native replay that loads nothing makes
  the run non-official: fix the native arm's setup or have the gate owner
  list the model under "native" in GATES.md.

Adapter and connector:
- Derive segments from the spec dump's `kv_cache_layout` and the layer view
  stride, not from "K then V". In vLLM 0.32 the page is `[H|N, N|H, C]` with
  `C = K | V`, so a page is ONE segment. The server's register path has no
  block stride (block `b` at `data_ptr + b * segments * bytes_per_block`):
  only layer-compact layouts register as is.
- With kernel-block splitting the worker's views count kernel blocks in
  dim 0: divide by `block_size / N`.
- Adapter declarations carry sizes, not byte order: two layouts with equal
  sizes share a namespace unless something names the difference. The core
  hashes `kv_cache_layout`; reject any other byte-order variation you cannot
  name (e.g. kernel-split HNC pages), and add a "different bytes give a
  different namespace" test with each new adapter.
- Make deferred costs observable in the first gate run (e.g. the worker
  `save stats` line), so the next iteration decides from data.
- Async saves need three hooks: `wait_for_save` only queues (CUDA event +
  saver thread); `request_finished` returns True and
  `get_transfer_results.finished_sending` releases the blocks after the copy;
  `handle_preemptions` (before the forward) flushes when the plan saw
  `preempted_req_ids`. On the V2 model runner it runs after block zeroing and
  copy-on-write, so layouts with `needs_kv_cache_zeroing` or
  `hash_block_size < block_size` must finish saves inside `wait_for_save`.
  Report a request in `finished_sending` even when its save failed or the
  saver died (complete jobs in `finally`), or vLLM leaks its blocks.
- On short-prefix workloads the async load round trip, not the copy, is the
  cost (a warm gsm8k request waited ~7.6 steps for a ~1 ms load): read the
  probe's `episodes` vs first `progress` step before blaming saves. A sync
  load (`load_kv_async=False`, wait in `start_load_kv`) fixes it, but vLLM
  only recovers failed sync loads via `invalid_block_ids` with one KV cache
  group and no block zeroing. A failed sync load has already fed the forward and the server keeps a key's first
  copy: hold every save of a request touching sync-loaded blocks until vLLM
  processed that step's output (`max_concurrent_batches` plans), and after
  any reported failure stop all saves for the process: under the recompute
  policy vLLM evicts nothing, so KV computed on the bad blocks stays
  prefix-cached under correct hashes and any later request can extend it
  (tracking "tainted" requests misses sharers and preempted loaders). Never
  give a sync load a deadline: the server's copy may still land.

PR:
- PR CI must pass. Before reporting a branch done, run `./scripts/check.sh`
  (fmt, typos, clippy, cargo check) on the remote and the default Python unit
  gate. CI runs `cz check` on every commit, merges included: use git's
  default merge message or `chore(rsi): merge ...`, never `merge(...)`.
  Locally, `typos` and `commitizen` install from PyPI into a temp venv.
