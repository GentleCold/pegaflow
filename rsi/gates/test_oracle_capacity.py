"""G5 capacity oracle and verdict contract (no GPU; needs the built
``pegaflow.semantics`` and the gate's Python deps).

Run from the repo root: ``python -m pytest rsi/gates/test_oracle_capacity.py``.
"""

from __future__ import annotations

import pytest

from rsi.gates import oracle, run_gate

BT = 4  # tokens per block, every group
PAGE = 500  # bytes per layer page
LAYERS = 2  # -> 1000 bytes per (group, key) block


def _group(gid: int, cls: str, mro: list[str], bt: int = BT, page: int = PAGE) -> dict:
    return {
        "id": gid,
        "num_layers": LAYERS,
        "enable_kv_transfer": True,
        "is_eagle_group": False,
        "spec": {
            "class": cls,
            "mro": mro,
            "block_size": bt,
            "page_size_bytes": page,
            "real_page_size_bytes": page,
            "sliding_window": None,
        },
    }


FULL = _group(0, "FullAttentionSpec", ["FullAttentionSpec", "AttentionSpec", "KVCacheSpec"])
CKPT = _group(1, "MambaSpec", ["MambaSpec", "KVCacheSpec"])


def _spec(*groups: dict) -> dict:
    return {"parallel": {"tp": 1}, "kv_cache_config": {"groups": list(groups), "num_blocks": 4}}


def _req(prompt: str, n_blocks: int, step: int, label: str, conv: int = 0) -> dict:
    """One request whose prompt is ``n_blocks`` key units (``BT`` tokens)
    plus one token, computed in one step per key unit (so checkpoint groups
    store every block end too)."""
    plen = n_blocks * BT + 1
    return {
        "pass": label,
        "conv": conv,
        "turn": 0,
        "prompt_len": plen,
        "computed": plen,
        "first_step": step,
        "local_hit": 0,
        "first_computed": 0,
        "progress": [[step, e] for e in range(BT, plen + 1, BT)],
        "key_tokens": BT,
        "keys": [f"{prompt}:{j}" for j in range(n_blocks)],
    }


def _trace(prompts: list[str], n_blocks: int, passes: list[str]) -> list[dict]:
    reqs, step = [], 1
    for label in passes:
        for c, p in enumerate(prompts):
            reqs.append(_req(p, n_blocks, step, label, c))
            step += 2  # visible to the next request
    return reqs


def _by_pass(reqs: list[dict], hits: list[int], label: str) -> int:
    return sum(h for r, h in zip(reqs, hits, strict=True) if r["pass"] == label)


# Cyclic scan over 3 prompts x 4 blocks (12 kB) with room for 8 blocks.
CYCLIC = (_spec(FULL), _trace(["p0", "p1", "p2"], 4, ["a", "b", "c"]))
# Full + checkpoint: 2 prompts x (2 full + 2 checkpoint) blocks.
HYBRID = (_spec(FULL, CKPT), _trace(["p0", "p1"], 2, ["a", "b"]))
# Mixed block sizes: full 4 tokens / 1000 B, checkpoint 8 tokens / 3000 B.
CKPT8 = _group(1, "MambaSpec", ["MambaSpec", "KVCacheSpec"], bt=2 * BT, page=3 * PAGE)
MIXED = (_spec(FULL, CKPT8), _trace(["p0", "p1"], 4, ["a", "b"]))


@pytest.mark.parametrize(
    "case", [CYCLIC, HYBRID, MIXED], ids=["cyclic-full", "full+checkpoint", "mixed-block-sizes"]
)
@pytest.mark.parametrize("policy", ["belady", "lru"])
def test_capacity_at_least_u_equals_unlimited_oracle(case, policy):
    spec, reqs = case
    u = oracle.unique_bytes(spec, reqs)
    hits, diag = oracle.simulate(spec, reqs, u, policy)
    assert hits == oracle.replay(spec, reqs)
    assert all(g["evicted"] == 0 for g in diag["groups"])


@pytest.mark.parametrize(
    "case,sizes,unique",
    [
        # 2 prompts x (2 full + 2 checkpoint) blocks of 1000 B.
        (HYBRID, [1000, 1000], 8 * 1000),
        # 2 prompts x (4 full x 1000 B + 2 checkpoint x 3000 B).
        (MIXED, [1000, 3000], 2 * (4 * 1000 + 2 * 3000)),
    ],
    ids=["same-size", "mixed-block-sizes"],
)
def test_unique_bytes_counts_distinct_group_keys(case, sizes, unique):
    spec, reqs = case
    assert oracle.block_bytes(spec) == sizes
    assert oracle.unique_bytes(spec, reqs) == unique
    assert oracle.unique_bytes(spec, reqs, {"a"}) == unique  # pass b adds no new keys


def test_belady_beats_lru_on_cyclic_scan():
    spec, reqs = CYCLIC
    cap = 8 * PAGE * LAYERS
    unlimited = oracle.replay(spec, reqs)
    belady, _ = oracle.simulate(spec, reqs, cap, "belady", unlimited)
    lru, _ = oracle.simulate(spec, reqs, cap, "lru", unlimited)
    for label in ("b", "c"):
        assert _by_pass(reqs, lru, label) == 0  # LRU thrashes a scan larger than C
        assert _by_pass(reqs, lru, label) < _by_pass(reqs, belady, label)
        assert _by_pass(reqs, belady, label) <= _by_pass(reqs, unlimited, label)
    assert all(b <= o for b, o in zip(belady, unlimited, strict=True))


def test_lru_strands_checkpoints_belady_keeps_coherent_prefixes():
    """C = 6 blocks. LRU stores all 8 and evicts p0's full blocks first (they
    were saved before its checkpoints), stranding p0's checkpoints. Belady
    skips checkpoint block 0 (no resume point needs it) and keeps both
    prompts' resumable sets (2 full + last checkpoint each)."""
    spec, reqs = HYBRID
    cap = 6 * PAGE * LAYERS
    unlimited = oracle.replay(spec, reqs)
    belady, bdiag = oracle.simulate(spec, reqs, cap, "belady", unlimited)
    lru, ldiag = oracle.simulate(spec, reqs, cap, "lru", unlimited)
    assert ldiag["passes"]["a"]["stranded_bytes"][1] > 0
    assert bdiag["passes"]["a"]["stranded_bytes"] == [0, 0]
    assert _by_pass(reqs, belady, "b") == _by_pass(reqs, unlimited, "b") > 0
    assert _by_pass(reqs, lru, "b") < _by_pass(reqs, belady, "b")


def test_summarize_capacity_caps_ext_by_unlimited_oracle():
    """A connector claiming more than the unlimited store could serve gets no
    E_C credit for it, and is counted as excess."""
    spec, reqs = CYCLIC
    reqs = [dict(r) for r in reqs]
    for r in reqs:
        r["first_computed"] = r["prompt_len"] - 1  # claims everything, even in pass a
    out = oracle.summarize_capacity(spec, reqs, 8 * PAGE * LAYERS, {"a": {"a"}, "b": {"b"}})
    assert out["a"]["ext_tokens"] == 0 and out["a"]["excess_tokens"] > 0
    b = out["b"]
    assert b["ext_tokens"] == b["oracle_tokens"]
    assert b["efficiency_c"] >= 1.0
    assert b["lru_hit_rate"] == 0.0 < b["oracle_c_hit_rate"] <= b["oracle_hit_rate"]


def test_local_hit_blocks_count_as_present_and_are_not_external():
    """vLLM's local hit holds every group's ``needed`` range at that length:
    resolve may use it, and only the rest is an external load."""
    spec, reqs = HYBRID
    kinds = oracle._active(spec)
    r = reqs[0]  # 2 key units, prompt 9 tokens
    assert oracle._resolve(kinds, r, {}, 0) == 0
    assert oracle._resolve(kinds, r, {}, BT) == BT
    # Resuming at 2 blocks with 1 local: full block 1 and checkpoint block 1.
    ext = oracle._external_keys(kinds, r, 2 * BT, BT)
    assert ext == [(0, "p0:1"), (1, "p0:1")]
    resident = dict.fromkeys(ext, 0)
    assert oracle._resolve(kinds, r, resident, BT) == 2 * BT


def test_reference_local_hit_fixes_the_bar():
    """E_C subtracts the baseline's local hit: an arm that loses its own
    local hits (and reloads them externally) gains nothing."""
    spec, reqs = CYCLIC
    cap = 8 * PAGE * LAYERS
    honest, lazy = [dict(r) for r in reqs], [dict(r) for r in reqs]
    for h, z in zip(honest, lazy, strict=True):
        if h["pass"] != "a":
            h["local_hit"], h["first_computed"] = BT, 3 * BT  # vLLM kept block 0
            z["local_hit"], z["first_computed"] = 0, 3 * BT  # dropped it, loads it
    ref = [r["local_hit"] for r in honest]
    seg = {"b": {"b"}}
    a = oracle.summarize_capacity(spec, honest, cap, seg, ref)["b"]
    b = oracle.summarize_capacity(spec, lazy, cap, seg, ref)["b"]
    assert a["efficiency_c"] == b["efficiency_c"]
    assert a["unmatched_requests"] == b["unmatched_requests"] == 0
    own = oracle.summarize_capacity(spec, lazy, cap, seg)["b"]
    assert own["unmatched_requests"] == own["requests"]  # no reference given
    # Per-arm local (diagnostic) would reward dropping local hits.
    assert own["efficiency_c_own_local"] >= a["efficiency_c"]


def test_cap_order_rotates_deterministically():
    n = 8
    orders = [run_gate.cap_order(x, n) for x in run_gate.cap_names(4)]
    assert orders[0] == list(range(n))
    assert [o[0] for o in orders] == [0, 2, 4, 6, 0]
    assert all(sorted(o) == list(range(n)) for o in orders)


def _digest_ok() -> dict:
    keys = ("local_mismatched", "external_mismatched", "local_unreferenced",
            "external_unreferenced", "external_compared", "external_uncertain",
            "computed_exact", "computed_compared", "local_compared", "local_matched",
            "local_uncertain", "local_uncertain_preempted",
            "external_uncertain_preempted")  # fmt: skip
    return {**dict.fromkeys(keys, 0), "external_matched": 1, "invariant": False, "groups": {}}


def _cap_result(pool: int | None, loaded: int) -> dict:
    """A cap stage result built from the real oracle on CYCLIC, relabelled."""
    spec, reqs = CYCLIC
    names = {"a": "cap_cold", "b": "cap_warm", "c": "cap_warm2"}
    reqs = [r | {"pass": names[r["pass"]]} for r in reqs]
    for r in reqs:
        if r["pass"] != "cap_cold":
            r["first_computed"] = r["prompt_len"] - 1
    cap = 8 * PAGE * LAYERS
    segs = {"warm_timed": {"cap_warm2"}} | {n: {n} for n in names.values()}
    loads = {n: oracle.external_load_bytes(spec, reqs, {n}) for n in names.values()}
    metrics = {'pegaflow_load_bytes_total{otel_scope_name="x"}': loaded}
    return {
        "capacity_bytes": cap,
        "capacity_fraction": 0.5,
        "unique_bytes": 2 * cap,
        "pool_capacity_bytes": pool,
        "capacity": oracle.summarize_capacity(spec, reqs, cap, segs, [0] * len(reqs)),
        "external_load_bytes": loads,
        "load_errors": [],
        "digest_self": _digest_ok(),
        "passes": {
            "cap_cold": {},
            "cap_warm": {},
            "cap_warm2": {
                "pegaflow_metrics": metrics,
                "pegaflow_end": {"pegaflow_cache_resident_bytes": 1.0},
            },
        },
    }


def test_capacity_verdict_requires_pool_and_load_proof():
    good = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    need = good["external_load_bytes"]["cap_warm2"]
    assert need > 0
    v = run_gate._capacity_verdict(good, pegaflow=True, gated=True)
    assert v["pass"] and v["loads_ok"] and v["external_hit_bytes"] == need
    assert v["server_io_bytes"]["cap_warm2"]["loaded_bytes"] == 10**9
    assert v["server_end_bytes"] == {"cap_warm2": {"pegaflow_cache_resident_bytes": 1.0}}
    no_pool = _cap_result(None, loaded=10**9)
    assert not run_gate._capacity_verdict(no_pool, pegaflow=True, gated=True)["pass"]
    assert run_gate._capacity_verdict(no_pool, pegaflow=False, gated=True)["pass"]
    too_big = _cap_result(8 * PAGE * LAYERS + 1, loaded=10**9)
    assert not run_gate._capacity_verdict(too_big, pegaflow=True, gated=True)["pass"]
    short = _cap_result(8 * PAGE * LAYERS, loaded=need - 1)  # hits not from the store
    assert not run_gate._capacity_verdict(short, pegaflow=True, gated=True)["loads_ok"]


def test_verdicts_reference_arms_not_required_and_errors_isolated(tmp_path):
    good = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    good["digest_self"] |= {"external_compared": 1}
    results = {
        "baseline": {},
        "old": {"gsm8k": {"arm": "old", "workload": "gsm8k", "error": "RuntimeError: boom"}},
        "native": {"cap": good},
        "lmcache": {"cap": good},
        "UnderTest": {"cap": good, "cap025": good},
    }
    v = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")
    assert v["old"]["required"] is False and v["old"]["errors"]
    for ref in ("native", "lmcache"):  # in-vLLM CPU stores: no server to prove
        assert v[ref]["required"] is False
        assert v[ref]["G3.no_eviction"]["na"]
        assert v[ref]["G5.replay_warm"]["pool_ok"]
    assert v["UnderTest"].get("required", True)
    assert v["UnderTest"]["G5.replay_warm"]["pass"]
    assert "cap025" in v["UnderTest"]["G5.curve"]


@pytest.mark.parametrize("cpu_bytes", [12_919_701_504, 2_404_122_624, 171_798_691_840, 1])
def test_lmcache_cpu_size_is_exact_bytes(cpu_bytes):
    """LMCache takes GiB and allocates ``int(GiB * 1024**3)``: the arm must get
    exactly the PegaFlow pool / C bytes, local CPU only."""
    env = run_gate.lmcache_env(cpu_bytes)
    assert int(float(env["LMCACHE_MAX_LOCAL_CPU_SIZE"]) * 1024**3) == cpu_bytes
    assert env["LMCACHE_LOCAL_CPU"] == "True" and env["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0"
    assert not any(k in env for k in ("LMCACHE_LOCAL_DISK", "LMCACHE_REMOTE_URL"))


@pytest.mark.parametrize(
    "external,vacuous_ok,ok",
    [
        ({}, True, True),  # cap: nothing loaded in the sample is not a KV error
        ({}, False, False),  # replay: some external load must be judged
        ({"external_compared": 1, "external_mismatched": 1}, True, False),
        ({"external_uncertain": 1}, True, False),  # loaded but never judged
    ],
    ids=["cap-vacuous", "replay-vacuous", "cap-mismatch", "cap-unjudged"],
)
def test_digest_verdict_vacuous_sample(external, vacuous_ok, ok):
    d = _digest_ok() | {"external_matched": 0} | external
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=vacuous_ok)
    assert v["pass"] is ok
    assert ("vacuous" in v) is (ok and vacuous_ok)


# --------------------------------------------- G2.digest_self.cap (exhaustive)


def _probe(tmp_path, reqs: list[dict], digests: list[dict]):
    import json

    d = tmp_path / "probe"
    d.mkdir()
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (d / "digest-pp0-tp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in digests))
    return d


def _cap_probe(tmp_path, corrupt: int, timed_untagged: bool):
    """Cap self-trace: cold computes 4 convs x 4 blocks, warm loads them all
    (``corrupt`` of the 16 external blocks carry wrong bytes); with
    ``timed_untagged`` a third pass recomputes every key undigested, the
    pre-fix cap sampling that made the reference non-exhaustive."""
    reqs, digests, step = [], [], 1

    def block(c: int, j: int, bad: bool = False) -> dict:
        return {"key": f"c{c}:{j}", "covered": (j + 1) * BT, "exact": ["bad" if bad else f"d{c}{j}"], "l1": [1.0]}  # fmt: skip

    passes = [("cap_cold", "full"), ("cap_warm", "full")]
    if timed_untagged:
        passes.append(("cap_warm2", None))
    n = 0
    for name, tag in passes:
        for c in range(4):
            rid = f"{name}-{c}"
            r = _req(f"c{c}", 4, step, name, c) | {"req_id": rid, "digest": tag}
            if name == "cap_warm":
                r["episodes"] = [[step, 0, 4 * BT]]  # all 4 blocks loaded
                r["progress"] = [[step, 4 * BT + 1]]
            else:
                r["episodes"] = [[step, 0, 0]]
            r["keys"] = [f"c{c}:{j}" for j in range(4)]
            reqs.append(r)
            if tag:
                blocks = []
                for j in range(4):
                    bad = name == "cap_warm" and n < corrupt
                    n += name == "cap_warm"
                    blocks.append(block(c, j, bad))
                digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": blocks})  # fmt: skip
            step += 2
    return _probe(tmp_path, reqs, digests)


@pytest.mark.parametrize(
    "corrupt,timed_untagged,ok",
    [(0, False, True), (8, False, False), (1, False, False), (8, True, False)],
    ids=["clean", "half-corrupt", "one-corrupt", "half-corrupt-untagged-later-pass"],
)
def test_digest_self_cap_catches_partially_corrupt_loads(tmp_path, corrupt, timed_untagged, ok):
    """Some external blocks match, others are wrong: the cap verdict must fail
    (the reviewer's 79/158 corruption passed when undigested later passes
    made the reference non-exhaustive and wrong blocks went ``uncertain``)."""
    from rsi.gates import digest_compare

    probe = _cap_probe(tmp_path, corrupt, timed_untagged)
    d = digest_compare.compare(probe, probe, invariant=False)
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=False, exhaustive=True)
    assert v["pass"] is ok, d
    if corrupt and not timed_untagged:
        assert d["external_mismatched"] == corrupt


@pytest.mark.parametrize(
    "uncertain,preempted,loaded,ok",
    [(0, 0, 100, True), (1, 1, 100, True), (2, 2, 100, False), (1, 0, 100, False)],
    ids=["judged", "one-preempted", "over-allowance", "untagged-uncertain"],
)
def test_digest_self_cap_exhaustive_allowance(uncertain, preempted, loaded, ok):
    d = _digest_ok() | {
        "external_compared": loaded - uncertain,
        "external_matched": loaded - uncertain,
        "external_uncertain": uncertain,
        "external_uncertain_preempted": preempted,
    }
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=False, exhaustive=True)
    assert v["pass"] is ok
    assert v.get("preempted_uncertain", 0) == preempted


def test_vacuous_cap_digest_only_for_reference_arms(tmp_path):
    """No external block judged: a reference arm passes flagged vacuous, the
    arm under test fails (a passing G5 needs loads, and they must be judged)."""
    empty = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    empty["digest_self"] |= {"external_matched": 0}
    v = run_gate.verdicts({"baseline": {}, "old": {"cap": empty}, "UnderTest": {"cap": empty}}, tmp_path, tmp_path / "spec.json")  # fmt: skip
    assert v["old"]["G2.digest_self.cap"]["pass"] and v["old"]["G2.digest_self.cap"]["vacuous"]
    assert not v["UnderTest"]["G2.digest_self.cap"]["pass"]
    assert not v["UnderTest"]["G5.replay_warm"]["pass"]


# ------------------------------------------------------------ S1 / bytes / floor


def test_own_local_above_reference_earns_no_ext():
    """An arm whose vLLM happens to keep more locally than the baseline did
    (variance) gets no store credit for those tokens."""
    spec, reqs = CYCLIC
    reqs = [dict(r) for r in reqs]
    for r in reqs:
        if r["pass"] != "a":
            r["local_hit"] = r["first_computed"] = 2 * BT  # all local, no store load
    out = oracle.summarize_capacity(spec, reqs, 8 * PAGE * LAYERS, {"b": {"b"}}, [0] * len(reqs))
    assert out["b"]["ext_tokens"] == 0 and out["b"]["efficiency_c"] == 0.0


@pytest.mark.parametrize(
    "tp,replicas,copies", [(1, 1, 1), (2, 1, 2), (2, 2, 1), (4, 2, 2)], ids=str
)
def test_block_and_gpu_kv_bytes_count_distinct_tp_copies(tp, replicas, copies):
    spec = _spec(FULL) | {"parallel": {"tp": tp}}
    spec["kv_cache_config"] |= {"kv_tp_replicas": replicas, "tensors": [{"size": 7000}, {"size": 3000, "host_resident": True}]}  # fmt: skip
    assert oracle.block_bytes(spec) == [PAGE * LAYERS * copies]
    assert run_gate.gpu_kv_bytes(spec) == 7000 * copies


def test_gpu_kv_bytes_counts_shared_tensors_once():
    """Hybrid groups sharing one tensor: the allocation counts once (the old
    num_blocks x Σ group block bytes formula counted it per group)."""
    spec = _spec(FULL, CKPT)
    spec["kv_cache_config"]["tensors"] = [{"size": 4 * PAGE * LAYERS}]
    assert run_gate.gpu_kv_bytes(spec) == 4 * PAGE * LAYERS
    assert spec["kv_cache_config"]["num_blocks"] * sum(oracle.block_bytes(spec)) == 2 * run_gate.gpu_kv_bytes(spec)  # fmt: skip


def test_capacity_floor_fails_closed_and_g5_problems():
    assert not run_gate.capacity_floor(None)["pass"]
    assert not run_gate.capacity_floor({"gpu_kv_bytes": 1})["pass"]
    low = run_gate.capacity_floor({"unique_bytes": 3, "gpu_kv_bytes": 1})
    high = run_gate.capacity_floor({"unique_bytes": 4, "gpu_kv_bytes": 1})
    assert not low["pass"] and high["pass"]
    sep = {"G5.replay_warm": {"freeze_vs_oracle_c": 0.7}}
    flat = {"G5.replay_warm": {"freeze_vs_oracle_c": 0.95}}
    assert run_gate.g5_problems({"X": sep, "old": flat}, high, None) == []
    assert run_gate.g5_problems({"X": flat}, high, None) == ["G5 trace does not separate freeze from H*_C"]  # fmt: skip
    assert run_gate.g5_problems({"X": {}}, high, None)  # freeze not evaluated: not official
    assert run_gate.g5_problems({"X": sep}, low, {"enough": False}) == ["capacity below 2x GPU KV (whole replay file used)"]  # fmt: skip


def test_reference_locals_unmatched_path():
    ref = run_gate.reference_locals(
        [{"pass": "cap_cold", "conv": 0, "turn": 1, "local_hit": 32},
         {"pass": "cap_warm", "conv": 0, "turn": 1, "local_hit": None},
         {"pass": "cap_warm", "conv": None, "turn": 0, "local_hit": 5}]
    )  # fmt: skip
    assert ref == {"cap_cold/0/1": 32, "cap_warm/0/1": 0}
    reqs = [{"pass": "cap_cold", "conv": 0, "turn": 1}, {"pass": "cap_warm3", "conv": 0, "turn": 1},
            {"pass": "cap_warm2", "conv": 9, "turn": 0}, {"pass": "cap_warm2", "turn": 0}]  # fmt: skip
    assert run_gate.match_locals(reqs, ref) == [32, 0, None, None]
    assert run_gate.match_locals(reqs, None) == [None] * 4


def test_unique_estimate_counts_shared_prefix_once():
    spec = _spec(FULL)
    # vLLM computes prompt + out - 1 tokens: out = BT + 1 completes one block.
    est = run_gate.UniqueEstimate(spec, out_tokens=BT + 1)
    a = list(range(3 * BT))
    est.add([a[: 2 * BT], a])  # turn 2 extends turn 1: 3 distinct blocks + 2 generated
    assert est.bytes == (3 + 2) * PAGE * LAYERS


@pytest.mark.parametrize("given,ok", [("0.75,0.5,0.25", True), ("0.5,0.25,0.75,0.5", True), ("0.25,0.5", False)])  # fmt: skip
def test_official_cap_fractions_compare_as_sets(tmp_path, given, ok):
    import argparse

    args = argparse.Namespace(**(run_gate.OFFICIAL | {"cap_fractions": given, "num_gpu_blocks": 0, "vllm_arg": []}))  # fmt: skip
    probs = run_gate.official_problems(args, ["plain", "baseline", "m:C"], list(run_gate.WORKLOADS), tmp_path, tmp_path)  # fmt: skip
    assert any(p.startswith("cap_fractions") for p in probs) is not ok


def test_stop_waits_for_the_whole_process_group(tmp_path, monkeypatch):
    """The leader exits at once on SIGTERM, a child keeps tearing down for
    2 s (vLLM's EngineCore, a pinned pool after SIGKILL): ``stop`` must
    return only once the group is gone, then log the exit time."""
    import os
    import subprocess
    import time

    monkeypatch.setattr(run_gate, "TERM_GRACE", 30)
    script = "sh -c 'trap \"sleep 2; exit 0\" TERM; while :; do sleep 0.1; done' & wait"
    log = tmp_path / "log"
    with log.open("w") as fh:
        proc = subprocess.Popen(["sh", "-c", script], stdout=fh, start_new_session=True)
        time.sleep(0.5)
        t0 = time.monotonic()
        run_gate.stop(proc, "sh", fh)
        took = time.monotonic() - t0
    assert took >= 1.5
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)
    assert "gone after SIGTERM" in log.read_text()
