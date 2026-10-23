"""RSI gate orchestrator (human-owned, see docs/rsi/GATES.md).

Runs arms on one model and writes a run record:

    RUN_DIR/
      summary.json            numbers, verdicts, ``official`` flag
      <arm>/<workload>/       probe dir, vllm.log, pegaflow.log, pass results

Arms:
  plain                    vLLM without any KV connector (gsm8k only; control)
  baseline                 vLLM + RsiProbe only (no PegaFlow)
  old                      reference connector (pegaflow.connector:PegaKVConnector), black box
  selftest:<mode>          gate self-test connector (rsi.gates.selftest), gsm8k only
  <module>:<Class>         connector under test

Every arm/workload pair gets fresh servers. Passes: ``cold``, then before each
warm pass ``/reset_prefix_cache`` (local only) and ``rsi_poison`` (every GPU KV
byte overwritten), so warm reuse can only come from the connector. Replay runs
``--replay-repeats`` warm passes; the first carries digest tags (G2/G3), the
others are timed for G4.

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


def pegaflow_server(args, log: Path):
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
        args.pool_size,
        "--devices",
        args.gpu,
        "--enable-prometheus",
        # One pool for the whole size: with per-NUMA pools a single GPU only
        # sees pool_size / numa_nodes, and the oracle assumes no eviction.
        "--disable-numa-affinity",
    ]
    return grpc, http, launched(cmd, env, log, f"http://127.0.0.1:{http}/health", 300)


def arm_kind(arm: str) -> str:
    if arm in ("plain", "baseline", "old"):
        return arm
    return "selftest" if arm.startswith("selftest:") else "under_test"


def arm_label(arm: str) -> str:
    if arm.startswith("selftest:"):
        return "selftest-" + arm.split(":", 1)[1]
    return arm if arm in ("plain", "baseline", "old") else arm.split(":")[-1]


def kv_transfer_config(arm: str, probe_dir: Path, digest: str) -> dict[str, Any] | None:
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
    args, gate: dict[str, Any], arm: str, probe_dir: Path, invariant: bool, pf_port, log
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
    cfg = kv_transfer_config(arm, probe_dir, "tagged")
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


def run_arm_workload(args, gate, arm: str, workload: str, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    probe = out / "probe"
    if probe.exists():
        shutil.rmtree(probe)
    kind = arm_kind(arm)
    gsm = workload == "gsm8k"
    invariant = gsm and bool(gate.get("batch_invariant"))
    result: dict[str, Any] = {
        "arm": arm,
        "workload": workload,
        "invariant": invariant,
        "passes": {},
    }
    with ExitStack() as stack:
        pf_port = pf_http = None
        if kind in ("old", "under_test"):
            pf_port, pf_http, cm = pegaflow_server(args, out / "pegaflow.log")
            stack.enter_context(cm)
        url, cm = vllm_server(args, gate, arm, probe, invariant, pf_port, out / "vllm.log")
        stack.enter_context(cm)

        if gsm:
            limit = SELFTEST_LIMIT if kind == "selftest" else args.gsm8k_limit
            prompts = workloads.gsm8k_prompts(Path(args.gsm8k_dir), args.shots, limit)
            names = ["cold", "warm"]
        else:
            convs = workloads.build_turns(
                Path(args.replay_file),
                args.model,
                args.replay_convs,
                args.max_model_len - args.replay_out_tokens - 8,
                args.replay_turns,
            )
            names = ["cold", "warm"] + [f"warm{k}" for k in range(2, args.replay_repeats + 1)]
        for name in names:
            if name != "cold":
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
                timed = name not in ("cold", "warm")
                xs = [
                    {"rsi_pass": name}
                    | ({} if timed or c % args.replay_digest_every else {"rsi_digest": "full"})
                    for c in range(len(convs))
                ]
                res = asyncio.run(
                    workloads.run_replay(
                        url, "m", convs, args.concurrency, args.replay_out_tokens, xargs=xs
                    )
                )
            time.sleep(2)  # late finish records and metrics land in this pass
            res["vllm_metrics"] = workloads.diff(asyncio.run(workloads.scrape(url)), before)
            if pf_metrics:
                res["pegaflow_metrics"] = workloads.diff(
                    asyncio.run(workloads.scrape(pf_metrics, ("pegaflow",))), pf_before
                )
            result["passes"][name] = res
            print(f"[{arm}/{workload}/{name}] {summary_line(res)}", flush=True)

    if kind == "plain":
        (out / "result.json").write_text(json.dumps(result, indent=1))
        return result
    spec = json.loads((probe / "spec.json").read_text())
    reqs = oracle.load_requests(probe / "requests.jsonl")
    result["hit"] = oracle.summarize(spec, reqs, {n: {n} for n in names})
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
    }

    for label, per_wl in results.items():
        kind = "selftest" if label.startswith("selftest-") else label
        if kind in ("plain", "baseline"):
            continue
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
            f"{wl}.{p}": sum(
                n
                for k, n in r["passes"][p].get("pegaflow_metrics", {}).items()
                if k.startswith("pegaflow_cache_block_evictions_total")
            )
            for wl, r in per_wl.items()
            for p in r["passes"]
            if not p.startswith("_")
        }
        # The oracle has unlimited capacity: an evicting pool measures its size.
        g["G3.no_eviction"] = {
            "pass": bool(evicted) and not any(evicted.values()),
            "evicted": evicted,
        }
        g |= _perf_verdicts(
            rep, base.get("replay"), gsm, base.get("gsm8k"), g["G2.digest_self.replay"]["pass"]
        )
        g["pass"] = all(x["pass"] for x in g.values() if isinstance(x, dict))
        v[label] = g
    v["baseline"]["pass"] = all(x["pass"] for x in v["baseline"].values() if isinstance(x, dict))
    return v


def git_rev() -> str:
    with suppress(Exception):
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    return (REPO / ".rsi-rev").read_text().strip() if (REPO / ".rsi-rev").exists() else "unknown"


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
    if set(wls) != {"gsm8k", "replay"}:
        out.append(f"workloads {wls} != gsm8k,replay")
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
    ap.add_argument("--workloads", default="gsm8k,replay")
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
    args = ap.parse_args(argv)
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
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        kind = arm_kind(arm)
        for wl in wls:
            if wl != "gsm8k" and kind in ("plain", "selftest"):
                continue
            label = arm_label(arm)
            results.setdefault(label, {})[wl] = run_arm_workload(
                args, gate, arm, wl, run_dir / label / wl
            )

    def strip(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if k != "digest_self"}

    problems = official_problems(args, arms, wls, gate_file, committed)
    v = verdicts(results, run_dir, committed)
    summary = {
        "tag": args.tag,
        "rev": git_rev(),
        "args": vars(args),
        "gate": gate,
        "official": not problems,
        "not_official_because": problems,
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
    return 0 if not problems and all(x["pass"] for x in v.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
