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
            "computed_exact", "computed_compared")  # fmt: skip
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
    results = {
        "baseline": {},
        "old": {"gsm8k": {"arm": "old", "workload": "gsm8k", "error": "RuntimeError: boom"}},
        "native": {"cap": good},
        "UnderTest": {"cap": good, "cap025": good},
    }
    v = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")
    assert v["old"]["required"] is False and v["old"]["errors"]
    assert v["native"]["required"] is False
    assert v["native"]["G3.no_eviction"]["na"]
    assert v["UnderTest"].get("required", True)
    assert v["UnderTest"]["G5.replay_warm"]["pass"]
    assert "cap025" in v["UnderTest"]["G5.curve"]
