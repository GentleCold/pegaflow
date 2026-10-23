"""RSI gate orchestrator (human-owned, see docs/rsi/GATES.md).

Runs arms on one model and writes a run record:

    RUN_DIR/
      summary.json            numbers, verdicts, ``official`` flag
      <arm>/<workload>/       probe dir, vllm.log, pegaflow.log, pass results

Arms:
  plain                    vLLM without any KV connector (gsm8k only; control)
  baseline                 vLLM + RsiProbe only (no PegaFlow)
  old                      reference connector (pegaflow.connector:PegaKVConnector), black box
  native                   reference: vLLM's OffloadingConnector (CPU offload), record only
  lmcache                  reference: LMCacheConnectorV1, local CPU backend only, record only
  selftest:<mode>          gate self-test connector (rsi.gates.selftest), gsm8k only
  <module>:<Class>         connector under test

Every arm/workload pair gets fresh servers. Passes: ``cold``, then before each
warm pass ``/reset_prefix_cache`` (local only) and ``rsi_poison`` (every GPU KV
byte overwritten), so warm reuse can only come from the connector. Replay runs
``--replay-repeats`` warm passes; the first carries digest tags (G2/G3), the
others are timed for G4. The ``cap`` workload (G5) is its own, larger
replay (``--cap-convs`` x ``--cap-turns``): the baseline arm runs it first
(``cap_cold``, ``cap_warm``) to fix ``U`` and the reference local hits, then
every store arm runs ``cap_cold``, ``cap_warm``, ``cap_warm2..N`` per
fraction ``f`` on a fresh store of ``C = f x U`` bytes; warm pass ``j``
starts the conversations rotated by ``j/4`` of their count.

Per-model gate settings (attention backend, batch invariance, extra vLLM args)
come from ``docs/rsi/models/<model>/gate.json``; the committed spec dump is
``docs/rsi/models/<model>/spec.json``.

A run is ``official`` only with the official profile (constants below, all
required arms and workloads). Every required verdict that cannot be evaluated
is a failure. Exit 0 only for an official run where every verdict passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import sysconfig
import time
from contextlib import ExitStack, contextmanager, suppress
from datetime import date
from pathlib import Path
from typing import Any

import requests

from rsi.gates import digest_compare, oracle, workloads

REPO = Path(__file__).resolve().parents[2]
OLD = "pegaflow.connector:PegaKVConnector"
SELFTEST = "rsi.gates.selftest.connector:SelftestConnector"
NEGATIVE_SELFTESTS = ("liar", "bitflip", "offbyone")

# Official profile: thresholds and workload sizes (GATES.md is the source).
OFFICIAL = {
    "gsm8k_limit": 0,  # full test split
    "shots": 8,
    "replay_convs": 32,
    "replay_turns": 12,
    "replay_out_tokens": 64,
    "replay_repeats": 3,
    "replay_digest_every": 4,
    "cap_convs": 64,
    "cap_turns": 12,
    "cap_fractions": "0.25,0.5,0.75",
    "concurrency": 32,
    "max_model_len": 8192,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 256,
}
ACC_TOL = 0.01
MIN_EFFICIENCY = 0.95
ORACLE_FLOOR = {"gsm8k": 0.5, "replay": 0.5}  # warm oracle hit rate, else E is vacuous
MIN_JUDGED_GSM8K = 0.9  # every gsm8k request is digested: nearly all loads are judged
# G4: passes with nothing to reuse from PegaFlow (replay cold: vLLM's own
# prefix cache serves the earlier turns; gsm8k: the shared few-shot prefix) can
# only tie baseline, so they bound the connector's overhead; warm replay must win.
MAX_OVERHEAD = 0.02
# G5: store capacity as a fraction of the cap trace's unique KV bytes; only
# CAPACITY_FRACTION is gated, the others are a reported curve. The bar is the
# clairvoyant oracle at that capacity.
CAPACITY_FRACTION = 0.5
MIN_EFFICIENCY_C = 0.9
MIN_CAPACITY_GPU_KV = 2.0  # official: gated C >= 2 x vLLM GPU KV bytes
WORKLOADS = ("gsm8k", "replay", "cap")
SELFTEST_LIMIT = 128  # gsm8k questions per self-test arm


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, proc: subprocess.Popen, timeout: float, log: Path) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            tail = log.read_text()[-3000:] if log.exists() else ""
            raise RuntimeError(f"process exited {proc.returncode} before {url} was ready\n{tail}")
        with suppress(requests.RequestException):
            if requests.get(url, timeout=2).status_code == 200:
                return
        time.sleep(2)
    raise TimeoutError(f"{url} not ready after {timeout}s")


@contextmanager
def launched(cmd: list[str], env: dict[str, str], log: Path, health: str, timeout: float):
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True
        )
        try:
            wait_http(health, proc, timeout, log)
            yield proc
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            # Children (vLLM's EngineCore) outlive the API server while they
            # free GPU / CPU-offload memory; the next stage needs that memory.
            if not _group_gone(proc.pid, 120):
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                _group_gone(proc.pid, 60)


def _group_gone(pgid: int, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(1)
    return False


def pegaflow_server(args, log: Path, pool_size: str):
    grpc, http = free_port(), free_port()
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYO3_PYTHON"] = sys.executable
    env["PYTHONHOME"] = sys.base_prefix
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (env.get("LD_LIBRARY_PATH"), libdir) if x)
    env["PYTHONPATH"] = f"{REPO / 'python'}:{sysconfig.get_path('purelib')}"
    cmd = [
        args.pegaflow_server,
        "--addr",
        f"127.0.0.1:{grpc}",
        "--http-addr",
        f"127.0.0.1:{http}",
        "--pool-size",
        pool_size,
        "--devices",
        args.gpu,
        "--enable-prometheus",
        # One pool for the whole size: with per-NUMA pools a single GPU only
        # sees pool_size / numa_nodes, and the oracle assumes no eviction.
        "--disable-numa-affinity",
    ]
    return grpc, http, launched(cmd, env, log, f"http://127.0.0.1:{http}/health", 300)


_REFERENCE = ("plain", "baseline", "old", "native", "lmcache")
# Reference arms: numbers recorded, never required, a crash is recorded.
_RECORD_ONLY = ("old", "native", "lmcache")
# Reference stores without a PegaFlow server (a CPU cache inside vLLM).
_OFFLOAD = ("native", "lmcache")
LMCACHE_OVERLAY = Path.home() / "kexi" / "overlays" / "lmcache"


def lmcache_env(cpu_bytes: int) -> dict[str, str]:
    """LMCache config (env, read by ``LMCacheEngineConfig.from_env``): local
    CPU backend only, sized to exactly ``cpu_bytes``. ``max_local_cpu_size``
    is GiB and LMCache allocates ``int(GiB * 1024**3)`` bytes; ``bytes / 2**30``
    is exact in a float and ``repr`` round-trips it. Chunk size stays at
    LMCache's default (256 tokens); telemetry is off. Disk and remote stay at
    their defaults (None); the caller drops any inherited ``LMCACHE_*``."""
    return {
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": repr(cpu_bytes / (1 << 30)),
        "LMCACHE_MAX_LOCAL_DISK_SIZE": "0",
        "LMCACHE_TRACK_USAGE": "false",
        "DO_NOT_TRACK": "1",
    }


def lmcache_config(log: Path, cpu_bytes: int) -> dict[str, Any]:
    """What the lmcache arm ran with: the env we set and LMCache's own
    ``LMCache Configuration:`` log line (chunk size, backends, CPU GiB)."""
    line = next((x for x in log.read_text().splitlines() if "LMCache Configuration:" in x), None)
    return {
        "cpu_bytes": cpu_bytes,
        "env": lmcache_env(cpu_bytes),
        "logged": line.split("LMCache Configuration:", 1)[1].strip()[:4000] if line else None,
    }


def arm_kind(arm: str) -> str:
    if arm in _REFERENCE:
        return arm
    return "selftest" if arm.startswith("selftest:") else "under_test"


def arm_label(arm: str) -> str:
    if arm.startswith("selftest:"):
        return "selftest-" + arm.split(":", 1)[1]
    return arm if arm in _REFERENCE else arm.split(":")[-1]


def parse_size(s: str) -> int:
    """Bytes of a ``--pool-size`` string (same units as pegaflow-server)."""
    s = s.strip().lower()
    for unit, mult in (("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * mult)
    return int(float(s.removesuffix("b")))


def kv_transfer_config(
    arm: str, probe_dir: Path, digest: str, cpu_bytes: int = 0
) -> dict[str, Any] | None:
    kind = arm_kind(arm)
    if kind == "plain":
        return None
    probe = {
        "kv_connector": "RsiProbe",
        "kv_connector_module_path": "rsi.gates.rsi_probe",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"probe_dir": str(probe_dir), "digest": digest},
    }
    if kind == "baseline":
        return probe
    if kind == "native":
        # vLLM's own CPU offload, as ``--kv-offloading-size`` would set it up.
        under_test = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
        }
    elif kind == "lmcache":
        # LMCache's vLLM connector; sized by env (``lmcache_env``).
        under_test = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    else:
        if kind == "selftest":
            module, cls = SELFTEST.split(":")
            extra = {"mode": arm.split(":", 1)[1]}
        else:
            module, cls = (OLD if kind == "old" else arm).split(":")
            extra = {}
        under_test = {
            "kv_connector": cls,
            "kv_connector_module_path": module,
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        # A failed load must be recomputed, never served from garbage.
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {"connectors": [under_test, probe]},
    }


def vllm_server(
    args,
    gate: dict[str, Any],
    arm: str,
    probe_dir: Path,
    invariant: bool,
    pf_port,
    log,
    cpu_bytes: int = 0,
):
    port = free_port()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["VLLM_SERVER_DEV_MODE"] = "1"  # /reset_prefix_cache, /collective_rpc
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = ":".join(x for x in (str(REPO), env.get("PYTHONPATH")) if x)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    backend: list[str] = []
    if gate.get("attention_backend"):
        backend = ["--attention-backend", gate["attention_backend"]]
    if invariant:
        # Batch invariance makes computed KV independent of batch composition,
        # so loaded blocks can be compared byte-for-byte with a baseline run.
        env["VLLM_BATCH_INVARIANT"] = "1"
        if gate.get("invariant_attention_backend"):
            backend = ["--attention-backend", gate["invariant_attention_backend"]]
    else:
        env.pop("VLLM_BATCH_INVARIANT", None)
    if pf_port is not None:
        env["PEGAFLOW_PORT"] = str(pf_port)  # contract: the connector under test reads it
    for k in [k for k in env if k.startswith("LMCACHE_")]:
        del env[k]
    if arm_kind(arm) == "lmcache":
        # LMCache lives in an overlay (its pins would downgrade the shared
        # venv); only this arm's vLLM sees it.
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{args.lmcache_overlay}"
        env |= lmcache_env(cpu_bytes)
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(port),
        "--served-model-name",
        "m",
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(args.gpu_mem),
        "--generation-config",
        "vllm",
        "--seed",
        "42",
        "--worker-extension-cls",
        "rsi.gates.rsi_probe.RsiWorkerExt",
        *backend,
        *gate.get("vllm_args", []),
        *args.vllm_arg,
    ]
    cfg = kv_transfer_config(arm, probe_dir, "tagged", cpu_bytes)
    if cfg is not None:
        cmd += ["--kv-transfer-config", json.dumps(cfg)]
    if args.num_gpu_blocks:
        cmd += ["--num-gpu-blocks-override", str(args.num_gpu_blocks)]
    url = f"http://127.0.0.1:{port}"
    return url, launched(cmd, env, log, f"{url}/health", 900)


def reset_and_poison(url: str) -> dict[str, Any]:
    """Drop the local prefix cache, then overwrite all GPU KV bytes."""
    for _ in range(30):
        r = requests.post(f"{url}/reset_prefix_cache", timeout=30)
        if r.ok and r.json().get("success", True) is not False:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"reset_prefix_cache kept failing: {r.status_code} {r.text}")
    r = requests.post(f"{url}/collective_rpc", json={"method": "rsi_poison"}, timeout=120)
    r.raise_for_status()
    results = r.json()["results"]
    if not results or not all(isinstance(x, dict) and x.get("bytes") for x in results):
        raise RuntimeError(f"rsi_poison did not cover the KV cache: {results}")
    return {"poisoned": results}


def cap_names(repeats: int) -> list[str]:
    """G5 pass labels: the replay passes, prefixed ``cap_``."""
    return [f"cap_{n}" for n in ["cold", "warm"] + [f"warm{k}" for k in range(2, repeats + 1)]]


def cap_order(name: str, n: int) -> list[int]:
    """Conversation start order of a cap pass. Warm pass ``j`` (``cap_warm`` =
    1) rotates by ``j * n // 4``, so a store frozen after the cold pass does
    not see the same arrival order again and recency differs between passes."""
    stage = name.removeprefix("cap_")
    j = 0 if stage == "cold" else 1 if stage == "warm" else int(stage.removeprefix("warm"))
    k = (j * n // 4) % n if n else 0
    return list(range(k, n)) + list(range(k))


def cap_label(f: float) -> str:
    """Workload dir / result key of capacity fraction ``f``."""
    return "cap" if f == CAPACITY_FRACTION else f"cap{round(f * 100):03d}"


def parse_fractions(s: str) -> list[float]:
    out = sorted({float(x) for x in s.split(",") if x.strip()})
    if CAPACITY_FRACTION not in out:
        raise ValueError(f"--cap-fractions must include the gated {CAPACITY_FRACTION}")
    if any(not 0 < f <= 1 for f in out):
        raise ValueError(f"--cap-fractions must be in (0, 1]: {out}")
    return out


def gpu_kv_bytes(spec: dict[str, Any]) -> int:
    """vLLM's GPU KV bytes: ``num_blocks`` x per-block bytes over all groups."""
    return spec["kv_cache_config"]["num_blocks"] * sum(oracle.block_bytes(spec))


def _pos(r: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    if r.get("conv") is None or r.get("turn") is None:
        return None
    return (r.get("pass"), r["conv"], r["turn"])


def reference_locals(reqs: list[dict[str, Any]]) -> dict[str, int]:
    """Baseline local hit per workload position ``pass/conv/turn``."""
    return {"/".join(map(str, p)): r["local_hit"] or 0 for r in reqs if (p := _pos(r))}


def match_locals(reqs: list[dict[str, Any]], ref: dict[str, int] | None) -> list[int | None]:
    """Reference local hit per request: the baseline's at the same
    conversation / turn; ``cap_cold`` -> baseline ``cap_cold``, every warm pass
    -> baseline ``cap_warm`` (local hits after reset + poison). None if absent."""
    out: list[int | None] = []
    for r in reqs:
        p = _pos(r)
        if ref is None or p is None:
            out.append(None)
            continue
        label = "cap_cold" if p[0] == "cap_cold" else "cap_warm"
        out.append(ref.get(f"{label}/{p[1]}/{p[2]}"))
    return out


def run_arm_workload(
    args,
    gate,
    arm: str,
    workload: str,
    out: Path,
    capacity: int = 0,
    ref_local: dict[str, int] | None = None,
) -> dict[str, Any]:
    """``capacity`` (``cap*`` workloads): store bytes, for PegaFlow's pool or
    the native arm's CPU cache. Other workloads use ``--pool-size``.
    ``ref_local``: baseline cap local hits (``reference_locals``)."""
    out.mkdir(parents=True, exist_ok=True)
    probe = out / "probe"
    if probe.exists():
        shutil.rmtree(probe)
    kind = arm_kind(arm)
    gsm = workload == "gsm8k"
    cap = workload.startswith("cap")
    if cap and capacity <= 0 and kind != "baseline":
        raise ValueError("cap workload needs a capacity")
    store_bytes = capacity if cap else parse_size(args.pool_size)
    invariant = gsm and bool(gate.get("batch_invariant"))
    result: dict[str, Any] = {
        "arm": arm,
        "workload": workload,
        "invariant": invariant,
        "passes": {},
    }
    if cap:
        result["capacity_bytes"] = capacity
    with ExitStack() as stack:
        pf_port = pf_http = None
        if kind in ("old", "under_test"):
            pf_port, pf_http, cm = pegaflow_server(
                args, out / "pegaflow.log", str(capacity) if cap else args.pool_size
            )
            stack.enter_context(cm)
        url, cm = vllm_server(
            args,
            gate,
            arm,
            probe,
            invariant,
            pf_port,
            out / "vllm.log",
            store_bytes if kind in _OFFLOAD else 0,
        )
        stack.enter_context(cm)
        if kind == "lmcache":
            result["lmcache"] = lmcache_config(out / "vllm.log", store_bytes)
        if cap and pf_http:
            # What the server actually allocated: must not exceed C.
            m = asyncio.run(workloads.scrape(f"http://127.0.0.1:{pf_http}", ("pegaflow",)))
            pool = [n for k, n in m.items() if k.startswith("pegaflow_pool_capacity_bytes")]
            result["pool_capacity_bytes"] = int(sum(pool)) if pool else None

        if gsm:
            limit = SELFTEST_LIMIT if kind == "selftest" else args.gsm8k_limit
            prompts = workloads.gsm8k_prompts(Path(args.gsm8k_dir), args.shots, limit)
            names = ["cold", "warm"]
        else:
            convs = workloads.build_turns(
                Path(args.replay_file),
                args.model,
                args.cap_convs if cap else args.replay_convs,
                args.max_model_len - args.replay_out_tokens - 8,
                args.cap_turns if cap else args.replay_turns,
            )
            names = ["cold", "warm"] + [f"warm{k}" for k in range(2, args.replay_repeats + 1)]
            if cap:
                # Baseline: U (cold) and reference local hits (cold, warm).
                names = cap_names(args.replay_repeats)[: 2 if kind == "baseline" else None]
        for name in names:
            stage = name.removeprefix("cap_")
            if stage != "cold":
                time.sleep(args.settle)  # let async saves land before the warm pass
                result["passes"].setdefault("_poison", {})[name] = reset_and_poison(url)
            before = asyncio.run(workloads.scrape(url))
            pf_metrics = f"http://127.0.0.1:{pf_http}" if pf_http else None
            pf_before = (
                asyncio.run(workloads.scrape(pf_metrics, ("pegaflow",))) if pf_metrics else {}
            )
            if gsm:
                # Digest every request: under batch invariance every copy is a
                # reference, so nothing a connector loads goes unjudged.
                xargs = {"rsi_pass": name, "rsi_digest": "full"}
                res = asyncio.run(
                    workloads.run_gsm8k(url, "m", prompts, args.concurrency, xargs=xargs)
                )
                (out / f"{name}-outputs.json").write_text(json.dumps(res.pop("outputs")))
            else:
                # Digest a conversation sample in the untimed passes; all turns
                # of a sampled conversation, so its references are digested too.
                timed = stage not in ("cold", "warm")
                xs = [
                    {"rsi_pass": name}
                    | ({} if timed or c % args.replay_digest_every else {"rsi_digest": "full"})
                    for c in range(len(convs))
                ]
                res = asyncio.run(
                    workloads.run_replay(
                        url,
                        "m",
                        convs,
                        args.concurrency,
                        args.replay_out_tokens,
                        xargs=xs,
                        order=cap_order(name, len(convs)) if cap else None,
                    )
                )
            time.sleep(2)  # late finish records and metrics land in this pass
            res["vllm_metrics"] = workloads.diff(asyncio.run(workloads.scrape(url)), before)
            if pf_metrics:
                pf_after = asyncio.run(workloads.scrape(pf_metrics, ("pegaflow",)))
                res["pegaflow_metrics"] = workloads.diff(pf_after, pf_before)
                if cap:
                    # Absolute occupancy at pass end vs C: shows fragmentation.
                    res["pegaflow_end"] = {
                        k.split("{")[0]: v
                        for k, v in pf_after.items()
                        if k.startswith(
                            ("pegaflow_cache_resident_bytes{", "pegaflow_pool_used_bytes")
                        )
                    }
            result["passes"][name] = res
            print(f"[{arm}/{workload}/{name}] {summary_line(res)}", flush=True)

    if kind == "plain":
        (out / "result.json").write_text(json.dumps(result, indent=1))
        return result
    spec = json.loads((probe / "spec.json").read_text())
    reqs = oracle.load_requests(probe / "requests.jsonl")
    result["hit"] = oracle.summarize(spec, reqs, {n: {n} for n in names})
    if cap and kind == "baseline":
        result["unique_bytes"] = oracle.unique_bytes(spec, reqs, {"cap_cold"})
        result["gpu_kv_bytes"] = gpu_kv_bytes(spec)
        result["ref_local"] = reference_locals(reqs)
    elif cap:
        timed = {n for n in names if n.removeprefix("cap_") not in ("cold", "warm")}
        result["capacity"] = oracle.summarize_capacity(
            spec,
            reqs,
            capacity,
            {"warm_timed": timed} | {n: {n} for n in names},
            match_locals(reqs, ref_local),
        )
        result["external_load_bytes"] = {
            n: oracle.external_load_bytes(spec, reqs, {n}) for n in names
        }
    errs = probe / "load_errors.jsonl"
    result["load_errors"] = (
        [json.loads(x) for x in errs.read_text().splitlines()] if errs.exists() else []
    )
    result["digest_self"] = digest_compare.compare(probe, probe, invariant)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def summary_line(res: dict[str, Any]) -> str:
    keys = ("accuracy", "ttft_p50", "ttft_p99", "req_per_s", "seconds")
    return " ".join(f"{k}={res[k]:.4g}" for k in keys if k in res)


# ------------------------------------------------------------------- G0

# Depend on GPU memory / CLI, not on the model layout.
_VOLATILE = {"num_blocks", "size", "layer_stride", "model", "max_model_len", "probe"}


def _strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in _VOLATILE}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _worker_layout(probe: Path) -> dict[str, Any]:
    out = {}
    for f in sorted(probe.glob("spec-*.json")):
        d = json.loads(f.read_text())
        out[d["rank"]] = {
            name: [
                {"shape": t["shape"][1:], "stride": t["stride"], "dtype": t["dtype"]}
                for t in layer["tensors"]
            ]
            for name, layer in d["layers"].items()
        }
    return out


def spec_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += spec_diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in spec_diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def g0(run_dir: Path, label: str, committed: Path) -> dict[str, Any]:
    """Arm layout == baseline layout (gsm8k), baseline layout == committed dump."""
    base = run_dir / "baseline" / "gsm8k" / "probe"
    mine = run_dir / label / "gsm8k" / "probe"
    if not (base / "spec.json").exists() or not (mine / "spec.json").exists():
        return {"pass": False, "reason": "spec dump missing"}
    diffs = spec_diff(
        _strip(json.loads((base / "spec.json").read_text())),
        _strip(json.loads((mine / "spec.json").read_text())),
    ) + spec_diff(_worker_layout(base), _worker_layout(mine), ".worker")
    if committed.exists():
        diffs += [
            "committed" + d
            for d in spec_diff(
                _strip(json.loads(committed.read_text())),
                _strip(json.loads((base / "spec.json").read_text())),
            )
        ]
    else:
        diffs.append(f"committed spec dump {committed} missing")
    return {"pass": not diffs, "diffs": diffs[:20], "num_diffs": len(diffs)}


# ------------------------------------------------------------- verdicts


def _outputs(run_dir: Path, label: str, name: str) -> list[str] | None:
    f = run_dir / label / "gsm8k" / f"{name}-outputs.json"
    return json.loads(f.read_text()) if f.exists() else None


def _differ(ref: list[str] | None, got: list[str] | None) -> int | None:
    if ref is None or got is None or len(got) > len(ref):
        return None
    return sum(a != b for a, b in zip(ref[: len(got)], got, strict=True))


def _digest_verdict(d: dict[str, Any] | None, min_judged: float | None) -> dict[str, Any]:
    """``min_judged``: fraction of loaded external blocks that must be judged
    (matched or mismatched, not uncertain); None = no external load required."""
    if d is None:
        return {"pass": False, "reason": "not evaluated"}
    judged = d["external_matched"] + d["external_mismatched"]
    seen = d["external_compared"] + d["external_uncertain"] + d["external_unreferenced"]
    ok = (
        d["local_mismatched"] == 0
        and d["external_mismatched"] == 0
        and d["local_unreferenced"] == 0
        and d["external_unreferenced"] == 0
    )
    if min_judged is not None:
        ok = ok and judged > 0 and judged >= min_judged * seen
    if d["invariant"]:
        ok = ok and d["computed_exact"] == d["computed_compared"]
    keys = [k for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {
        "pass": ok,
        **{k: d[k] for k in keys},
        "first_bad": next((g["first_bad"] for g in d["groups"].values() if g["first_bad"]), None),
    }


def _hit_verdict(r: dict[str, Any] | None, workload: str) -> dict[str, Any]:
    if r is None:
        return {"pass": False, "reason": "not evaluated"}
    w = r["hit"]["warm"]
    floor = ORACLE_FLOOR[workload]
    e = w["efficiency"]
    ok = (
        e is not None
        and e >= MIN_EFFICIENCY
        and w["oracle_hit_rate"] >= floor
        and w["excess_tokens"] <= w["excess_limit"]
        and not r["load_errors"]
    )
    return {
        "pass": ok,
        "efficiency": e,
        "oracle_hit_rate": w["oracle_hit_rate"],
        "oracle_floor": floor,
        "external_hit_rate": w["external_hit_rate"],
        "excess_tokens": w["excess_tokens"],
        "excess_limit": w["excess_limit"],
        "load_errors": len(r["load_errors"]),
    }


def _metric(metrics: dict[str, float], name: str) -> float:
    return sum(n for k, n in metrics.items() if k.split("{")[0] == name)


def _evictions(metrics: dict[str, float]) -> float:
    return _metric(metrics, "pegaflow_cache_block_evictions_total")


def _capacity_verdict(r: dict[str, Any] | None, pegaflow: bool, gated: bool) -> dict[str, Any]:
    """G5 for one cap stage. ``pegaflow``: the arm stores in a PegaFlow server
    (pool size and load bytes must be proven). ``gated``: the gated fraction
    (else the numbers are a reported curve point, ``pass`` informational)."""
    if r is None or "capacity" not in r:
        reason = (r or {}).get("error", "not evaluated")
        return {"pass": False, "reason": reason}
    c = r["capacity"]
    w = c["warm_timed"]
    cold = c.get("cap_cold", {})
    e = w["efficiency_c"]
    passes = {n: p for n, p in r["passes"].items() if not n.startswith("_")}
    pm = {n: p.get("pegaflow_metrics", {}) for n, p in passes.items()}
    evicted = {n: _evictions(m) for n, m in pm.items()}
    resident_end = {n: p["pegaflow_end"] for n, p in passes.items() if "pegaflow_end" in p}
    io = {
        n: {
            "saved_bytes": _metric(m, "pegaflow_save_bytes_total"),
            "loaded_bytes": _metric(m, "pegaflow_load_bytes_total"),
            "external_hit_bytes": r.get("external_load_bytes", {}).get(n),
        }
        for n, m in pm.items()
    }
    pool = r.get("pool_capacity_bytes")
    # A server that grants itself more memory than C is not measured at C;
    # a PegaFlow arm that cannot show its pool size fails.
    pool_ok = pool <= r["capacity_bytes"] if pool is not None else not pegaflow
    # Every external hit must have been loaded from the C-byte server: load
    # bytes over the scored passes cover the external-hit blocks.
    scored = [n for n in passes if n.removeprefix("cap_") not in ("cold", "warm")]
    need = sum(io[n]["external_hit_bytes"] or 0 for n in scored)
    loaded = sum(io[n]["loaded_bytes"] for n in scored)
    loads_ok = not pegaflow or (
        all(io[n]["external_hit_bytes"] is not None for n in scored) and loaded >= need
    )
    ok = (
        e is not None
        and e >= MIN_EFFICIENCY_C
        and w["excess_tokens"] <= w["excess_limit"]
        and not r["load_errors"]
        and pool_ok
        and loads_ok
        and w["unmatched_requests"] == 0
    )
    diag = c["diagnostics"]
    return {
        "pass": ok,
        "gated": gated,
        "efficiency_c": e,
        # Within-pass multi-turn reuse (turn k+1 after GPU eviction): reported.
        "efficiency_c_cold": cold.get("efficiency_c"),
        "cold_ext_oracle_c_share": cold.get("ext_oracle_c_share"),
        "freeze_vs_oracle_c_cold": cold.get("freeze_vs_oracle_c"),
        "efficiency_c_own_local": w["efficiency_c_own_local"],
        "min_efficiency_c": MIN_EFFICIENCY_C,
        "unique_bytes": r.get("unique_bytes"),
        "capacity_bytes": r["capacity_bytes"],
        "capacity_fraction": r.get("capacity_fraction"),
        "pool_capacity_bytes": pool,
        "pool_ok": pool_ok,
        "loads_ok": loads_ok,
        "loaded_bytes": loaded,
        "external_hit_bytes": need,
        "unmatched_requests": w["unmatched_requests"],
        "hit_rate": w["hit_rate"],
        "oracle_hit_rate": w["oracle_hit_rate"],
        "oracle_c_hit_rate": w["oracle_c_hit_rate"],
        "lru_hit_rate": w["lru_hit_rate"],
        "freeze_hit_rate": w["freeze_hit_rate"],
        "lru_vs_oracle_c": w["lru_vs_oracle_c"],
        "freeze_vs_oracle_c": w["freeze_vs_oracle_c"],
        "excess_tokens": w["excess_tokens"],
        "excess_limit": w["excess_limit"],
        "load_errors": len(r["load_errors"]),
        "evicted": evicted,
        "server_end_bytes": resident_end,
        "server_io_bytes": io,
        "oracle_c_groups": diag["oracle_c"]["groups"],
        "lru_groups": diag["lru"]["groups"],
        "oracle_c_passes": diag["oracle_c"]["passes"],
        "lru_passes": diag["lru"]["passes"],
    }


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _cmp(mine: list[float], theirs: list[float], lower: bool, slack: float) -> dict[str, Any]:
    """Median of ``mine`` vs median of ``theirs``. ``slack`` 0: strictly better;
    otherwise at most ``slack`` (fraction) worse."""
    a, b = _median(mine), _median(theirs)
    if a is None or b is None:
        ok = False
    elif slack == 0:
        ok = a < b if lower else a > b
    else:
        ok = a <= b * (1 + slack) if lower else a >= b * (1 - slack)
    return {"pass": ok, "value": mine, "baseline": theirs, "slack": slack}


def _perf_verdicts(
    rep: dict[str, Any] | None,
    base_rep: dict[str, Any] | None,
    gsm: dict[str, Any] | None,
    base_gsm: dict[str, Any] | None,
    replay_ok: bool,
) -> dict[str, Any]:
    """G4 vs baseline vLLM. Replay numbers only count when the replay KV was
    correct (``replay_ok``)."""

    def passes(r: dict[str, Any] | None, names: tuple[str, ...], key: str) -> list[float]:
        ps = (r or {}).get("passes", {})
        return [ps[n][key] for n in names if n in ps and key in ps[n]]

    def timed(r: dict[str, Any] | None) -> tuple[str, ...]:
        ps = (r or {}).get("passes", {})
        return tuple(n for n in ps if n.startswith("warm") and n != "warm")

    g: dict[str, Any] = {}
    for key, lower in (("ttft_p50", True), ("req_per_s", False)):
        for name, names, slack in (
            ("replay_warm", timed(rep), 0.0),
            ("replay_cold", ("cold",), MAX_OVERHEAD),
        ):
            mine = passes(rep, names, key)
            theirs = passes(base_rep, timed(base_rep) if name == "replay_warm" else names, key)
            v = _cmp(mine, theirs, lower, slack)
            v["pass"] = v["pass"] and replay_ok
            g[f"G4.{name}.{key}"] = v
    for name in ("cold", "warm"):
        g[f"G4.gsm8k_{name}.seconds"] = _cmp(
            passes(gsm, (name,), "seconds"),
            passes(base_gsm, (name,), "seconds"),
            True,
            MAX_OVERHEAD,
        )
    return g


def verdicts(results: dict[str, dict[str, Any]], run_dir: Path, committed: Path) -> dict[str, Any]:
    v: dict[str, Any] = {}
    base = results.get("baseline", {})
    ref_cold = _outputs(run_dir, "baseline", "cold")
    ref_acc = base.get("gsm8k", {}).get("passes", {}).get("cold", {}).get("accuracy")
    base_probe = run_dir / "baseline" / "gsm8k" / "probe"
    invariant = base.get("gsm8k", {}).get("invariant", False)

    # Controls: the probe changes nothing (plain == baseline), poisoning and
    # the local reset change nothing (baseline warm == cold), local hits are
    # byte-identical to their computed copies.
    v["baseline"] = {
        "G2.plain_equals_baseline": {
            "pass": _differ(ref_cold, _outputs(run_dir, "plain", "cold")) == 0,
            "differ": _differ(ref_cold, _outputs(run_dir, "plain", "cold")),
        },
        # Output equality needs batch invariance; gate.json declares when a
        # model cannot have it, and then only accuracy and digests count.
        "G2.warm_equals_cold": {
            "pass": not invariant or _differ(ref_cold, _outputs(run_dir, "baseline", "warm")) == 0,
            "differ": _differ(ref_cold, _outputs(run_dir, "baseline", "warm")),
            "invariant": invariant,
        },
        "G2.digest_self.gsm8k": _digest_verdict(base.get("gsm8k", {}).get("digest_self"), None),
        "G2.digest_self.cap": _digest_verdict(base.get("cap", {}).get("digest_self"), None)
        if "cap" in base
        else {"pass": True, "na": "no cap workload"},
    }
    base_errors = {wl: r["error"] for wl, r in base.items() if "error" in r}
    if base_errors:
        v["baseline"]["errors"] = base_errors

    for label, per_all in results.items():
        kind = "selftest" if label.startswith("selftest-") else label
        if kind in ("plain", "baseline"):
            continue
        # A crashed stage is recorded under "errors" and evaluates as missing.
        per_wl = {wl: r for wl, r in per_all.items() if "error" not in r}
        g: dict[str, Any] = {}
        gsm = per_wl.get("gsm8k")
        g["G0.spec"] = g0(run_dir, label, committed)
        g["G2.digest_self.gsm8k"] = _digest_verdict(gsm and gsm["digest_self"], MIN_JUDGED_GSM8K)
        g["G2.digest_vs_baseline.gsm8k"] = _digest_verdict(
            digest_compare.compare(base_probe, run_dir / label / "gsm8k" / "probe", invariant)
            if gsm and base_probe.exists()
            else None,
            MIN_JUDGED_GSM8K,
        )
        diffs = {p: _differ(ref_cold, _outputs(run_dir, label, p)) for p in ("cold", "warm")}
        g["G2.gsm8k_outputs"] = {
            "pass": not invariant or all(d == 0 for d in diffs.values()),
            "invariant": invariant,
            **{f"{p}_differ": d for p, d in diffs.items()},
        }
        if kind == "selftest":
            # Negative arms must be caught by both digest checks (a single
            # flipped byte need not change a greedy output, so outputs are
            # only reported); the honest arm must pass every G2 check.
            digests = ("G2.digest_self.gsm8k", "G2.digest_vs_baseline.gsm8k")
            if label == "selftest-honest":
                g["pass"] = all(g[k]["pass"] for k in (*digests, "G2.gsm8k_outputs"))
                g["expect"] = "pass every G2 check"
            else:
                g["pass"] = not any(g[k]["pass"] for k in digests)
                g["expect"] = "fail both G2 digest checks"
            v[label] = g
            continue
        accs = {p: gsm["passes"][p]["accuracy"] for p in ("cold", "warm")} if gsm else {}
        g["G2.gsm8k"] = {
            "pass": ref_acc is not None
            and len(accs) == 2
            and all(abs(a - ref_acc) <= ACC_TOL for a in accs.values()),
            "baseline": ref_acc,
            **{
                f"{p}_delta_pp": (a - ref_acc) * 100 for p, a in accs.items() if ref_acc is not None
            },
        }
        rep = per_wl.get("replay")
        # Replay shares prefixes with untagged conversations, so many loads stay
        # uncertain; only require that some were judged and none mismatched.
        g["G2.digest_self.replay"] = _digest_verdict(rep and rep["digest_self"], 0.0)
        g["G3.gsm8k_warm"] = _hit_verdict(gsm, "gsm8k")
        # G3/G4 replay numbers only count when the replay KV was correct.
        g["G3.replay_warm"] = _hit_verdict(rep, "replay")
        if not g["G2.digest_self.replay"]["pass"]:
            g["G3.replay_warm"]["pass"] = False
        evicted = {
            f"{wl}.{p}": _evictions(r["passes"][p].get("pegaflow_metrics", {}))
            for wl, r in per_wl.items()
            if not wl.startswith("cap") and "passes" in r
            for p in r["passes"]
            if not p.startswith("_")
        }
        # The oracle has unlimited capacity: an evicting pool measures its size.
        if kind in _OFFLOAD:
            # No PegaFlow server: the CPU cache exports no eviction counter.
            g["G3.no_eviction"] = {"pass": True, "na": f"{kind} arm has no PegaFlow server"}
        else:
            g["G3.no_eviction"] = {
                "pass": bool(evicted) and not any(evicted.values()),
                "evicted": evicted,
            }
        g |= _perf_verdicts(
            rep, base.get("replay"), gsm, base.get("gsm8k"), g["G2.digest_self.replay"]["pass"]
        )
        cap = per_wl.get("cap")
        pegaflow = kind not in _OFFLOAD
        g["G2.digest_self.cap"] = _digest_verdict(cap and cap.get("digest_self"), 0.0)
        g["G5.replay_warm"] = _capacity_verdict(cap, pegaflow, gated=True)
        if not g["G2.digest_self.cap"]["pass"]:
            g["G5.replay_warm"]["pass"] = False
        # Other fractions: a reported curve, never gated.
        curve = {}
        for wl in sorted(per_wl):
            if wl.startswith("cap") and wl != "cap":
                cv = _capacity_verdict(per_wl[wl], pegaflow, gated=False)
                cv["digest_self"] = _digest_verdict(per_wl[wl].get("digest_self"), 0.0)["pass"]
                curve[wl] = {
                    k: cv.get(k)
                    for k in (
                        "pass",
                        "reason",
                        "capacity_fraction",
                        "capacity_bytes",
                        "efficiency_c",
                        "oracle_c_hit_rate",
                        "lru_hit_rate",
                        "freeze_vs_oracle_c",
                        "lru_vs_oracle_c",
                        "hit_rate",
                        "evicted",
                        "loads_ok",
                        "digest_self",
                    )
                }
        g["G5.curve"] = curve
        errors = {wl: r["error"] for wl, r in per_all.items() if "error" in r}
        if errors:
            g["errors"] = errors
        g["pass"] = not errors and all(
            x["pass"] for x in g.values() if isinstance(x, dict) and "pass" in x
        )
        if kind in _RECORD_ONLY:
            # Reference arms: every number is recorded, nothing is gated.
            g["required"] = False
        v[label] = g
    v["baseline"]["pass"] = not base_errors and all(
        x["pass"] for x in v["baseline"].values() if isinstance(x, dict) and "pass" in x
    )
    return v


def git_rev() -> str:
    with suppress(Exception):
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    return (REPO / ".rsi-rev").read_text().strip() if (REPO / ".rsi-rev").exists() else "unknown"


def capacity_floor(base_cap: dict[str, Any] | None) -> dict[str, Any]:
    """Gated ``C`` vs ``MIN_CAPACITY_GPU_KV`` x vLLM GPU KV bytes (baseline cap)."""
    if not base_cap or "unique_bytes" not in base_cap:
        return {"pass": False, "reason": "no baseline cap result"}
    u, kv = base_cap["unique_bytes"], base_cap["gpu_kv_bytes"]
    c = int(CAPACITY_FRACTION * u)
    return {
        "pass": c >= MIN_CAPACITY_GPU_KV * kv,
        "unique_bytes": u,
        "capacity_bytes": c,
        "gpu_kv_bytes": kv,
        "capacity_over_gpu_kv": c / kv if kv else None,
        "min_capacity_over_gpu_kv": MIN_CAPACITY_GPU_KV,
    }


def official_problems(
    args, arms: list[str], wls: list[str], gate_file: Path, committed: Path
) -> list[str]:
    out = [
        f"{k}={getattr(args, k)} (official {v})"
        for k, v in OFFICIAL.items()
        if getattr(args, k) != v
    ]
    kinds = {arm_kind(a) for a in arms}
    for need in ("plain", "baseline", "under_test"):
        if need not in kinds:
            out.append(f"arm {need} missing")
    if set(wls) != set(WORKLOADS):
        out.append(f"workloads {wls} != {','.join(WORKLOADS)}")
    if args.num_gpu_blocks or args.vllm_arg:
        out.append("--num-gpu-blocks / --vllm-arg override")
    for f in (gate_file, committed):
        if not f.exists():
            out.append(f"{f} missing")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="<model>-<iter>, used in the run dir name")
    ap.add_argument(
        "--arms",
        default="plain,baseline,old",
        help="comma list; selftest arms: selftest:honest,selftest:liar,...",
    )
    ap.add_argument(
        "--workloads",
        default=",".join(WORKLOADS),
        help="cap (G5) needs the baseline arm in the same run",
    )
    ap.add_argument("--runs-root", default=str(Path.home() / "kexi" / "rsi-runs"))
    ap.add_argument("--gpu", default="0", help="CUDA device ids (comma list, len = tp)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=OFFICIAL["max_model_len"])
    ap.add_argument(
        "--max-num-batched-tokens", type=int, default=OFFICIAL["max_num_batched_tokens"]
    )
    ap.add_argument("--max-num-seqs", type=int, default=OFFICIAL["max_num_seqs"])
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--vllm-arg", action="append", default=[], help="extra vllm serve arg")
    ap.add_argument("--pool-size", default="160gb")
    ap.add_argument(
        "--pegaflow-server", default=str(REPO / "target" / "release" / "pegaflow-server")
    )
    ap.add_argument(
        "--lmcache-overlay",
        default=str(LMCACHE_OVERLAY),
        help="dir with lmcache + its missing pure deps, on PYTHONPATH of the lmcache arm only",
    )
    ap.add_argument("--concurrency", type=int, default=OFFICIAL["concurrency"])
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--gsm8k-dir", default=str(Path.home() / "kexi" / "datasets" / "gsm8k-main"))
    ap.add_argument("--shots", type=int, default=OFFICIAL["shots"])
    ap.add_argument("--gsm8k-limit", type=int, default=OFFICIAL["gsm8k_limit"], help="0 = full")
    ap.add_argument(
        "--replay-file", default=str(Path.home() / "kexi" / "datasets" / "r2egym-256.jsonl")
    )
    ap.add_argument("--replay-convs", type=int, default=OFFICIAL["replay_convs"])
    ap.add_argument("--replay-turns", type=int, default=OFFICIAL["replay_turns"])
    ap.add_argument("--replay-out-tokens", type=int, default=OFFICIAL["replay_out_tokens"])
    ap.add_argument("--replay-repeats", type=int, default=OFFICIAL["replay_repeats"])
    ap.add_argument("--replay-digest-every", type=int, default=OFFICIAL["replay_digest_every"])
    ap.add_argument("--cap-convs", type=int, default=OFFICIAL["cap_convs"])
    ap.add_argument("--cap-turns", type=int, default=OFFICIAL["cap_turns"])
    ap.add_argument(
        "--cap-fractions",
        default=OFFICIAL["cap_fractions"],
        help=f"C/U list; {CAPACITY_FRACTION} is gated, the rest are reported",
    )
    args = ap.parse_args(argv)
    try:
        fractions = parse_fractions(args.cap_fractions)
    except ValueError as e:
        ap.error(str(e))
    args.model = str(Path(args.model).expanduser())
    if len(args.gpu.split(",")) != args.tp:
        ap.error(f"--gpu {args.gpu} does not have --tp {args.tp} devices")
    if args.replay_repeats < 2:
        ap.error("--replay-repeats must be >= 2 (the first warm pass is digested, not timed)")

    model_dir = REPO / "docs" / "rsi" / "models" / Path(args.model).name
    gate_file, committed = model_dir / "gate.json", model_dir / "spec.json"
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else {}
    run_dir = Path(args.runs_root) / f"{date.today().isoformat()}-{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    wls = [w for w in args.workloads.split(",") if w]
    if "baseline" in arms:  # every comparison references the baseline
        arms = ["baseline"] + [a for a in arms if a != "baseline"]
    if "cap" in wls:  # G5 sizes every store from the baseline cap trace
        wls = [w for w in wls if w != "cap"] + ["cap"]
    results: dict[str, dict[str, Any]] = {}

    def stage(arm: str, wl: str, out: Path, **kw) -> dict[str, Any]:
        """Run one stage; a reference arm's crash is recorded, not fatal."""
        try:
            return run_arm_workload(args, gate, arm, wl, out, **kw)
        except Exception as e:
            if arm_kind(arm) not in _RECORD_ONLY:
                raise
            print(f"[{arm}/{wl}] reference arm failed: {e!r}", flush=True)
            return {"arm": arm, "workload": wl, "error": f"{type(e).__name__}: {e}"[:2000]}

    for arm in arms:
        kind = arm_kind(arm)
        label = arm_label(arm)
        for wl in wls:
            if wl != "gsm8k" and kind in ("plain", "selftest"):
                continue
            if wl != "cap":
                results.setdefault(label, {})[wl] = stage(arm, wl, run_dir / label / wl)
                continue
            if kind == "baseline":
                results.setdefault(label, {})[wl] = stage(arm, wl, run_dir / label / wl)
                continue
            base_cap = results.get("baseline", {}).get("cap", {})
            u = base_cap.get("unique_bytes")
            if not u:
                results.setdefault(label, {})[wl] = {
                    "arm": arm,
                    "workload": wl,
                    "error": "no baseline cap trace in this run to size the store",
                }
                continue
            for f in fractions:
                name = cap_label(f)
                out = run_dir / label / name
                r = stage(arm, name, out, capacity=int(f * u), ref_local=base_cap.get("ref_local"))
                r["unique_bytes"] = u
                r["capacity_fraction"] = f
                if out.exists():
                    (out / "result.json").write_text(json.dumps(r, indent=1))
                results.setdefault(label, {})[name] = r

    def strip(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if k not in ("digest_self", "ref_local")}

    problems = official_problems(args, arms, wls, gate_file, committed)
    v = verdicts(results, run_dir, committed)
    floor = None
    if "cap" in wls:
        # Not a verdict: a store smaller than this measures vLLM's GPU cache,
        # so the run cannot be official (G5 is then a flow + digest smoke).
        floor = capacity_floor(results.get("baseline", {}).get("cap"))
        if not floor["pass"]:
            problems.append(f"capacity below {MIN_CAPACITY_GPU_KV:g}x GPU KV")
        # The bar must separate a store frozen after its first fill.
        for label, g in v.items():
            if label in _REFERENCE or label.startswith("selftest-"):
                continue
            fz = g.get("G5.replay_warm", {}).get("freeze_vs_oracle_c")
            if fz is None or fz >= MIN_EFFICIENCY_C:
                problems.append("G5 trace does not separate freeze from H*_C")
    summary = {
        "tag": args.tag,
        "rev": git_rev(),
        "args": vars(args),
        "gate": gate,
        "official": not problems,
        "not_official_because": problems,
        "capacity_floor": floor,
        "results": {a: {w: strip(r) for w, r in per.items()} for a, per in results.items()},
        "verdicts": v,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(
        json.dumps(
            {"official": not problems, "problems": problems, "verdicts": v}, indent=1, default=str
        )
    )
    print(f"run record: {run_dir}")
    required = [x for x in v.values() if x.get("required", True)]
    return 0 if not problems and all(x["pass"] for x in required) else 1


if __name__ == "__main__":
    sys.exit(main())
