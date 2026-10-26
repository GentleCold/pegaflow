"""G5 capacity oracle and verdict contract (no GPU). Needs ``pytest``,
``requests`` and a built ``pegaflow.semantics``; the command is in
docs/rsi/GATES.md ("Tests"). The isolation tests run only on Linux as root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.gates import digest_compare, oracle, run_gate

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


# Qwen3.5 shape: groups 0-2 MambaSpec (GDN), group 3 full attention, one block size.
_MAMBA = ["MambaSpec", "KVCacheSpec"]
QWEN35_SPEC = _spec(*(_group(g, "MambaSpec", _MAMBA) for g in range(3)), dict(FULL, id=3))
QWEN35 = (QWEN35_SPEC, _trace(["p0", "p1"], 2, ["a", "b"]))


@pytest.mark.parametrize(
    "case",
    [CYCLIC, HYBRID, MIXED, QWEN35],
    ids=["cyclic-full", "full+checkpoint", "mixed-block-sizes", "qwen3.5-3ckpt+full"],
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


def _fallback_trace() -> list[dict]:
    """One document, three requests (Qwen3.5 P5 it2 shape, minimized).
    r0 (3 blocks) stores full 0-2 and checkpoints 1, 2 (its steps end at 8
    and 12, not 4). r1 (1 block) stores checkpoint 0. r2 (3 blocks) resumes
    at 12 with an unlimited store: full 0-2 + checkpoint 2, four blocks."""
    return [
        _req("d", 3, 1, "a") | {"progress": [[1, 2 * BT], [2, 3 * BT], [3, 3 * BT + 1]]},
        _req("d", 1, 6, "a"),
        _req("d", 3, 10, "a"),
    ]


SWA = _group(1, "SlidingWindowSpec", ["SlidingWindowSpec", "AttentionSpec", "KVCacheSpec"])
SWA["spec"]["sliding_window"] = BT + 1  # Window{BT}: the last block before p


@pytest.mark.parametrize(
    "spec,reqs,items,policy,oc",
    [
        # 3 blocks of room, r2 needs 4 at P. needed(P) holds no earlier
        # checkpoint, so old Belady sees checkpoint 0 unused and drops it
        # (r2: 0). LRU keeps it and resumes at 4. The fallback-aware use set
        # (every checkpoint <= P) keeps full 0-1 + checkpoint 1: 8, the
        # optimum for 3 blocks.
        (_spec(FULL, CKPT), _fallback_trace(), 3, "belady_fallback", 2 * BT),
        # Qwen3.5 shape: 3 GDN groups + full; 5 blocks of room = full 0-1 +
        # three checkpoints 1 (P = 12 needs 6).
        (QWEN35_SPEC, _fallback_trace(), 5, "belady_fallback", 2 * BT),
        # Full only: needed is prefix-closed, so the candidates are old
        # Belady and LRU and Belady wins (the old H*_C exactly).
        (CYCLIC[0], CYCLIC[1], 8, "belady", None),
    ],
    ids=["checkpoint-fallback", "qwen3.5-fallback", "full-only-cyclic"],
)
def test_oracle_c_is_never_below_lru(spec, reqs, items, policy, oc):
    """H*_C = the best of the candidate stores, each one simulated over the
    whole trace: never below H_lru on a segment, and identical to the old
    Belady where every group's ``needed`` is prefix-closed (Full)."""
    cap = items * PAGE * LAYERS
    out = oracle.summarize_capacity(spec, reqs, cap)
    w = out["all"]
    assert w["ext_oracle_c_tokens"] >= w["ext_lru_tokens"], w
    assert w["oracle_c_policy"] == policy
    belady, bdiag = oracle.simulate(spec, reqs, cap, "belady")
    assert out["diagnostics"]["oracle_c"] == bdiag
    if policy == "belady":
        assert out["oracle_c_policies"] == ["belady", "lru"]
        assert "oracle_c_fallback" not in out["diagnostics"]
        assert w["oracle_c_tokens"] == sum(belady) > w["lru_tokens"]
    else:
        assert sum(belady) < w["lru_tokens"] < w["oracle_c_tokens"] == oc
        assert w["oracle_c_candidates"]["belady"]["lru_above_requests"] == 1
        assert w["oracle_c_candidates"][policy]["lru_above_requests"] == 0
        assert w["oracle_c_tokens"] <= sum(oracle.replay(spec, reqs))


@pytest.mark.parametrize(
    "spec,plen_blocks,closed",
    [
        (_spec(FULL), 6, True),
        (_spec(FULL, CKPT), 6, False),
        (_spec(FULL, SWA), 1, True),  # prompts within the window: like Full
        (_spec(FULL, SWA), 6, False),  # the window slides past block 0
        (QWEN35_SPEC, 6, False),
    ],
    ids=["full", "full+checkpoint", "window-short", "window-long", "qwen3.5"],
)
def test_fallback_use_set_differs_only_where_needed_is_not_prefix_closed(spec, plen_blocks, closed):
    """``prefix_closed`` (from ``semantics.needed``, i.e. the Kind) decides
    whether ``belady_fallback`` runs; where it holds, the fallback use set is
    the old one, so a full-only bar cannot move."""
    reqs = [_req("p", plen_blocks, 1, "a")]
    assert oracle.prefix_closed(spec, reqs) is closed
    assert ("belady_fallback" not in oracle.oracle_c_policies(spec, reqs)) is closed
    kinds = oracle._active(spec)
    p = plen_blocks * BT
    for lo in (0, BT):
        old = set(oracle._external_keys(kinds, reqs[0], p, lo))
        new = set(oracle._fallback_keys(kinds, reqs[0], p, lo))
        assert old <= new
        assert (old == new) is closed


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


def test_checkpoint_needs_and_saves_only_block_ends():
    """GDN = Checkpoint(bt): resuming at p needs exactly block p/bt - 1; a
    state is saveable only when a step ends on a block boundary."""
    g = oracle.group_kinds(QWEN35_SPEC)[0][1]
    assert (g.kind, g.block_tokens) == ("checkpoint", BT)
    assert [oracle.sem.needed(g, p) for p in (BT, 3 * BT)] == [(0, 1), (2, 3)]
    assert oracle.sem.saveable(g, 0, 3 * BT) == (2, 3)  # only the state at the chunk end
    assert oracle.sem.saveable(g, 0, 3 * BT - 1) == (0, 0)  # mid-block: nothing


@pytest.mark.parametrize(
    "ends,hit",
    [
        ([BT, 2 * BT, 3 * BT, 3 * BT + 1], 3 * BT),  # every boundary materialized
        ([2 * BT, 3 * BT + 1], 2 * BT),  # last state on a boundary: block 1
        ([BT + 2, 3 * BT + 1], 0),  # no chunk ends on a boundary: no state
        ([BT, 3 * BT - 1, 3 * BT + 1], BT),  # full blocks 1-2 stored, state only at 1 block
    ],
    ids=["aligned", "last-state-mid", "no-aligned-end", "full-ahead-of-state"],
)
def test_qwen35_replay_counts_only_chunk_end_states(ends, hit):
    """A second request on the same prompt hits only up to the last block end
    some step of the first one stopped at (vLLM writes state at chunk ends)."""
    plen = 3 * BT + 1
    first = _req("p0", 3, 1, "a") | {"progress": [[1 + k, e] for k, e in enumerate(ends)]}
    again = _req("p0", 3, len(ends) + 2, "b")
    assert oracle.replay(QWEN35_SPEC, [first, again]) == [0, hit]
    assert hit < plen


@pytest.mark.parametrize(
    "ckpt,full,hit",
    [
        ([{1, 2}] * 3, {0, 1, 2}, 3 * BT),
        ([{1, 2}, {1, 2}, {1}], {0, 1, 2}, 2 * BT),  # group 2 lacks block 2's state
        ([{2}, {2}, {1}], {0, 1, 2}, 0),  # no length where all three states exist
        ([{0, 1, 2}] * 3, {0, 2}, BT),  # full-attention gap at block 1
        ([{0, 1, 2}] * 3, set(), 0),  # states alone resume nothing
    ],
    ids=["all", "one-group-short", "groups-disagree", "full-gap", "no-full"],
)
def test_qwen35_hit_needs_every_group(ckpt, full, hit):
    """H* (and H*_C, same ``_resolve``) counts a length only where all three
    GDN groups hold that length's state and full attention holds the prefix."""
    kinds = oracle._active(QWEN35_SPEC)
    r = _req("p0", 3, 1, "a")
    resident = {(gi, f"p0:{b}"): 0 for gi, bs in enumerate([*ckpt, full]) for b in bs}
    assert oracle._resolve(kinds, r, resident) == hit
    # Resuming at ``hit`` loads one state per GDN group plus the full prefix.
    want = [(gi, f"p0:{hit // BT - 1}") for gi in range(3)] + [(3, f"p0:{b}") for b in range(hit // BT)]  # fmt: skip
    assert oracle._needed_keys(kinds, r, hit) == (want if hit else [])


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


@pytest.mark.parametrize(
    "spec,reqs,items,win",
    [
        (_spec(FULL, CKPT), _fallback_trace(), 3, "oracle_c_fallback"),
        (CYCLIC[0], CYCLIC[1], 8, "oracle_c"),
    ],
    ids=["fallback-wins", "full-only"],
)
def test_capacity_verdict_diagnostics_are_the_winning_store(spec, reqs, items, win):
    """The verdict's per-group / per-pass oracle_C diagnostics describe the
    store that set H*_C on warm_timed (plain Belady also reported, as
    belady_*, when it did not win); a record without the winner's
    diagnostics fails, not crashes."""
    r = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    r["capacity"] = oracle.summarize_capacity(
        spec, reqs, items * PAGE * LAYERS, {"warm_timed": {"a"}}, [0] * len(reqs)
    )
    diag = r["capacity"]["diagnostics"]
    v = run_gate._capacity_verdict(r, pegaflow=False, gated=True)
    assert v["oracle_c_groups"] == diag[win]["groups"], v
    assert v["oracle_c_passes"] == diag[win]["passes"]
    if win == "oracle_c":
        assert "belady_groups" not in v
    else:
        assert diag[win]["groups"] != diag["oracle_c"]["groups"]
        assert v["belady_groups"] == diag["oracle_c"]["groups"]
        assert v["belady_passes"] == diag["oracle_c"]["passes"]
        del diag[win]
        v = run_gate._capacity_verdict(r, pegaflow=False, gated=True)
        assert v["pass"] is False and f"capacity.diagnostics.{win}.groups" in v["reason"], v


@pytest.mark.parametrize(
    "drop",
    [
        "capacity.warm_timed.efficiency_c_own_local",  # pre-20e6b58 records
        "capacity.warm_timed.excess_tokens",
        "capacity.diagnostics.lru",
        "capacity_bytes",
        "load_errors",
    ],
)
def test_capacity_record_without_a_field_fails_not_crashes(drop):
    """S13: a cap record lacking a field the verdict reads fails with
    "cannot evaluate: missing <field>" instead of raising (which would lose
    every other verdict of the run, and its summary.json)."""
    r = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    *path, key = drop.split(".")
    d = r
    for k in path:
        d = d[k]
    del d[key]
    v = run_gate._capacity_verdict(r, pegaflow=True, gated=True)
    assert v["pass"] is False and v["reason"].startswith("cannot evaluate: missing "), v
    assert drop in v["reason"]


def _g3_result(e: float | None, oracle_rate: float, excess: int, errors: int) -> dict:
    warm = {"efficiency": e, "oracle_hit_rate": oracle_rate, "excess_tokens": excess,
            "excess_limit": 16, "external_hit_rate": 0.5, "hit_rate": 0.6,
            "local_hit_tokens": 10, "prompt_tokens": 100}  # fmt: skip
    return {"hit": {"warm": warm}, "load_errors": [{}] * errors}


@pytest.mark.parametrize(
    "e,oracle_rate,excess,errors,ok,below",
    [
        (1.0, 0.99, 0, 0, True, []),
        (0.5, 0.99, 0, 0, True, ["efficiency"]),  # low E_ext: reported, not gated
        (None, 0.2, 0, 0, True, ["efficiency", "oracle_hit_rate"]),
        (1.0, 0.99, 17, 0, False, []),  # excess beyond one block per request
        (1.0, 0.99, 0, 1, False, []),  # load errors
    ],
    ids=["at-bar", "low-e", "no-e-low-oracle", "excess", "load-errors"],
)
def test_g3_hit_level_is_reported_correctness_is_gated(e, oracle_rate, excess, errors, ok, below):
    """User, 2026-10-24: no hard hit-rate threshold. E_ext and the oracle hit
    rate are reported with their reference; excess and load errors still fail."""
    v = run_gate._hit_verdict(_g3_result(e, oracle_rate, excess, errors), "replay")
    assert v["pass"] is ok
    assert [k for k, x in v["reported"].items() if x["below"]] == below
    assert v["efficiency"] == e and v["oracle_hit_rate"] == oracle_rate


def test_g5_low_e_c_is_reported_not_gated():
    """A capacity record whose E_C is far below 0.9 passes G5 on correctness
    and validity; the number and its reference stay in the verdict and in the
    run's hit report."""
    r = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    r["capacity"]["warm_timed"]["efficiency_c"] = 0.3
    v = run_gate._capacity_verdict(r, pegaflow=True, gated=True)
    assert v["pass"] and v["efficiency_c"] == 0.3
    assert v["reported"]["efficiency_c"] == {"value": 0.3, "reference": 0.9, "below": True}
    r["load_errors"] = [{}]
    assert not run_gate._capacity_verdict(r, pegaflow=True, gated=True)["pass"]
    rep = run_gate.hit_report({"baseline": {}, "X": {"G5.replay_warm": v}})
    assert rep["X"]["status"] == "reported"
    assert rep["X"]["G5.replay_warm"]["efficiency_c"] == 0.3
    assert rep["X"]["G5.replay_warm"]["below_reference"] == ["efficiency_c"]


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
    assert (
        env["LMCACHE_CHUNK_SIZE"] == "256"
        and run_gate.lmcache_env(1, 16)["LMCACHE_CHUNK_SIZE"] == "16"
    )
    assert env["LMCACHE_LOCAL_CPU"] == "True" and env["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0"
    assert not any(k in env for k in ("LMCACHE_LOCAL_DISK", "LMCACHE_REMOTE_URL"))


_MLA = _group(0, "MLAAttentionSpec", ["MLAAttentionSpec", "FullAttentionSpec", "AttentionSpec", "KVCacheSpec"])  # fmt: skip


@pytest.mark.parametrize(
    "spec,tp,ranks,backends,c,ok",
    [
        (None, 1, 1, 1, 1000, True),
        (_spec(_MLA), 2, 1, 1, 1000, True),  # DSV2: save only first rank
        (_spec(FULL), 2, 2, 2, 1000, True),  # each rank stores its shard: C / 2 each
        (None, 2, None, None, 1000, None),  # TP > 1, no dump: the arm cannot start
        (_spec(FULL), 2, 2, 1, 1000, False),  # one backend fewer than storing ranks
        (_spec(_MLA), 2, 1, 2, 1000, False),  # MLA but both ranks stored: 2 C
    ],
    ids=["tp1", "mla-tp2", "full-tp2", "tp2-unknown", "full-tp2-short", "mla-tp2-both"],
)
def test_lmcache_store_totals_c_over_storing_ranks(tmp_path, spec, tp, ranks, backends, c, ok):
    """S1: LMCache sizes its CPU backend per worker. Each storing rank gets
    C / ranks, and the log must show exactly that many backends, so the
    arm's whole store is at most C (fail-closed otherwise)."""
    assert run_gate.lmcache_storing_ranks(spec, tp) == ranks
    if ranks is None:
        args = SimpleNamespace(lmcache_overlay=str(tmp_path), lmcache_chunk_size=256, lmcache_ranks=None)  # fmt: skip
        with pytest.raises(ValueError, match="storing ranks unknown"):
            run_gate.vllm_server(args, {}, "lmcache", tmp_path, False, None, tmp_path / "l", c)
        return
    log = tmp_path / "vllm.log"
    log.write_text(f"{run_gate.LMCACHE_BACKEND_MARK}\n" * backends)
    cfg = run_gate.lmcache_config(log, c, tmp_path, 256, ranks)
    assert cfg["per_rank_bytes"] * ranks <= c
    assert int(float(cfg["env"]["LMCACHE_MAX_LOCAL_CPU_SIZE"]) * 1024**3) == c // ranks
    assert (run_gate.lmcache_size_problem(cfg) is None) is ok


@pytest.mark.parametrize(
    "declared,dists,ok",
    [
        ("lmcache-0.5.6", ["lmcache-0.5.6"], True),
        ("lmcache-0.5.7.dev0+pr4467.e05c2ea", ["lmcache-0.5.6"], False),  # wrong build
        (None, ["lmcache-0.5.6"], False),  # nothing declared: no official lmcache
        ("lmcache-0.5.6", [], False),  # empty overlay
        ("lmcache-0.5.6", ["lmcache-0.5.6", "lmcache-0.5.7"], False),  # two dists
    ],
    ids=["declared", "other-build", "undeclared", "empty", "two"],
)
def test_lmcache_build_must_be_the_declared_one(tmp_path, declared, dists, ok):
    """S2: the overlay's lmcache dist must be gate.json's ``lmcache_dist``:
    the run is not official otherwise, and the arm fails G0.lmcache_build."""
    import argparse

    for d in dists:
        (tmp_path / f"{d}.dist-info").mkdir()
    gate = {"lmcache_dist": declared} if declared else {}
    fixed = {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t", "lmcache_chunk_size": 256, "lmcache_overlay": str(tmp_path)}  # fmt: skip
    args = argparse.Namespace(**(run_gate.OFFICIAL | fixed))
    arms = ["plain", "baseline", "lmcache", "pegaflow.m:C"]
    probs = run_gate.official_problems(args, arms, list(run_gate.WORKLOADS), tmp_path, tmp_path, gate)  # fmt: skip
    assert (not any("lmcache" in p for p in probs)) is ok, probs
    stage = {"lmcache": {"dist": [f"{d}.dist-info" for d in dists], "cpu_bytes": 10, "storing_ranks": 1, "per_rank_bytes": 10, "backends": 1}}  # fmt: skip
    assert run_gate.lmcache_build_verdict({"gsm8k": stage, "cap": stage}, gate)["pass"] is ok
    assert not run_gate.lmcache_build_verdict({"gsm8k": {}}, gate)["pass"]


def test_committed_gates_declare_their_lmcache_build():
    """S2: every committed gate.json names the lmcache build its reference
    numbers come from (GATES.md, "lmcache status per model")."""
    for f in sorted((run_gate.REPO / "docs/rsi/models").glob("*/gate.json")):
        gate = json.loads(f.read_text())
        if f.parent.name == "Qwen3-4B":
            assert gate["lmcache_dist"] == "lmcache-0.5.7.dev0+pr4467.e05c2ea", f
        if f.parent.name.startswith("DeepSeek-V2-Lite"):
            assert gate["lmcache_dist"] == "lmcache-0.5.6", f  # #4467 crashes on MLA


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
    assert run_gate.g5_problems({"X": sep}, high, {"unmodeled": ["x"]}) == ["cap plan sized from an unmodeled spec: x"]  # fmt: skip


def test_unmodeled_spec_only_flags_checkpoint_models():
    """The Checkpoint step-end model assumes ``hash_block_size ==
    block_size``, no Eagle, no prefill checkpoint blocks; Full-only specs
    are counted exactly whatever those values are."""
    ckpt = _spec(FULL, CKPT) | {"cache": {"mamba_cache_mode": "align"}}
    assert run_gate.unmodeled_spec(ckpt) == []
    committed = run_gate.REPO / "docs/rsi/models/Qwen3.5-4B/spec.json"
    assert run_gate.unmodeled_spec(json.loads(committed.read_text())) == []
    sub = {**ckpt, "kv_cache_config": {**ckpt["kv_cache_config"], "hash_block_size": BT // 2}}
    assert len(run_gate.unmodeled_spec(sub)) == 2  # both groups
    full = _spec(FULL)
    assert run_gate.unmodeled_spec({**full, "kv_cache_config": {**full["kv_cache_config"], "hash_block_size": 1}}) == []  # fmt: skip
    pcb = json.loads(json.dumps(ckpt))
    pcb["kv_cache_config"]["groups"][1]["spec"]["num_prefill_checkpoint_blocks"] = 1
    assert run_gate.unmodeled_spec(pcb) == ["group 1: num_prefill_checkpoint_blocks 1"]


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
    est = run_gate.UniqueEstimate(spec, out_tokens=BT + 1, budget=4 * BT)
    a = list(range(3 * BT))
    est.add([a[: 2 * BT], a])  # turn 2 extends turn 1: 3 distinct blocks + 2 generated
    assert est.bytes == (3 + 2) * PAGE * LAYERS


@pytest.mark.parametrize(
    "start,plen,budget,align,junction,ends",
    [
        pytest.param(0, 1722, 2048, True, 0, [1584, 1722], id="stop-at-last-cacheable"),
        pytest.param(0, 1796, 2048, True, 1056, [1056, 1584, 1796], id="stop-at-junction"),
        pytest.param(1584, 5314, 2048, True, 0, [3168, 4752, 5280, 5314], id="chunks-end-aligned"),
        pytest.param(0, 1722, 2048, False, 0, [1722], id="no-align-one-chunk"),
        pytest.param(0, 5000, 2048, False, 0, [2048, 4096, 5000], id="no-align-budget"),
    ],
)  # fmt: skip
def test_prefill_step_ends_follow_the_align_split(start, plen, budget, align, junction, ends):
    assert run_gate.prefill_step_ends(start, plen, 528, budget, align, junction) == ends


def test_unique_estimate_counts_only_saved_checkpoints():
    """Checkpoint groups store a block only where a step ends on its boundary
    (the oracle's rule): a 4-block prompt prefilled in one chunk stores the
    last cacheable checkpoint only; full attention stores all 4."""
    spec = _spec(FULL, CKPT) | {"cache": {"mamba_cache_mode": "align"}}
    est = run_gate.UniqueEstimate(spec, out_tokens=1, budget=8 * BT)
    est.add([list(range(4 * BT + 1))])
    assert est.bytes == (4 + 1) * PAGE * LAYERS
    # The next turn resumes at that checkpoint: one more chunk, one more state.
    est.add([list(range(6 * BT + 1))])
    assert est.bytes == (6 + 2) * PAGE * LAYERS


def test_cap_only_run_sizes_from_the_committed_spec(tmp_path):
    """``--workloads cap`` has no earlier baseline stage: auto must size from
    the committed dump, not fall back to ``CAP_MIN_CONVS``."""
    committed = tmp_path / "spec.json"
    arm = tmp_path / "baseline"
    assert run_gate.cap_plan_spec(arm, ["cap"], committed) is None
    committed.write_text("{}")
    assert run_gate.cap_plan_spec(arm, ["cap"], committed) == committed
    # The preempt stage's dump has a shrunk GPU pool: never size from it.
    (arm / "preempt" / "probe").mkdir(parents=True)
    (arm / "preempt" / "probe" / "spec.json").write_text("{}")
    assert run_gate.cap_plan_spec(arm, ["preempt", "cap"], committed) == committed
    for w in ("replay", "gsm8k"):
        (arm / w / "probe").mkdir(parents=True)
        (arm / w / "probe" / "spec.json").write_text("{}")
    wls = ["preempt", "replay", "gsm8k", "cap"]
    assert run_gate.cap_plan_spec(arm, wls, committed) == arm / "gsm8k/probe/spec.json"
    assert run_gate.cap_plan_spec(arm, ["preempt", "replay", "cap"], committed) == arm / "replay/probe/spec.json"  # fmt: skip


S1_CAP = Path(__file__).parent / "testdata/qwen3.5-4b-s1-cap-cold.json"


def test_unique_estimate_matches_the_measured_hybrid_trace():
    """Qwen3.5-4B's committed spec on the s1 cap trace (8 x 6, out 64,
    measured on the baseline arm): within 5% of the measured ``U``, and not
    above it (an under-estimate only adds conversations). The estimate
    before the Checkpoint rule was 4.47 GB, 1.28x the measured 3.49 GB."""
    spec = json.loads((run_gate.REPO / "docs/rsi/models/Qwen3.5-4B/spec.json").read_text())
    data = json.loads(S1_CAP.read_text())
    a = data["args"]
    est = run_gate.UniqueEstimate(spec, a["replay_out_tokens"], a["max_num_batched_tokens"])
    convs: dict[int, list] = {}
    for conv, turn, plen, keys in sorted(data["requests"]):
        # Same keys = same prefix: one token value per key keeps the chain.
        ids = [t for k in keys for t in [int(k, 16)] * 528]
        convs.setdefault(conv, []).append(ids + [-1 - turn] * (plen - len(ids)))
    for c in sorted(convs):
        est.add(convs[c])
    measured = data["measured_unique_bytes"]
    assert 0.95 * measured <= est.bytes <= measured, est.bytes / measured


@pytest.mark.parametrize(
    "over,key,ok",
    [
        pytest.param(
            {"cap_fractions": "0.75,0.5,0.25"}, "cap_fractions", True, id="fractions-reordered"
        ),  # fmt: skip
        pytest.param(
            {"cap_fractions": "0.5,0.25,0.75,0.5"}, "cap_fractions", True, id="fractions-dup"
        ),  # fmt: skip
        pytest.param(
            {"cap_fractions": "0.25,0.5"}, "cap_fractions", False, id="fractions-missing"
        ),  # fmt: skip
        pytest.param({"replay_skip": 0}, "replay_skip", True, id="replay-skip-0"),
        pytest.param({"replay_skip": 32}, "replay_skip=32 (held-out", False, id="held-out"),
        pytest.param(
            {"arm": "pegaflow.connector_v2:C"}, "arm under test", True, id="pegaflow-module"
        ),  # fmt: skip
        pytest.param(
            {"arm": "rsi.gates.myconn:C"},
            "arm under test rsi.gates",
            False,
            id="rsi-module-under-test",
        ),  # fmt: skip
        pytest.param(
            {"lmcache_chunk_size": 256}, "lmcache_chunk_size", True, id="lmcache-chunk-default"
        ),  # fmt: skip
        pytest.param(
            {"lmcache_chunk_size": 16}, "lmcache_chunk_size=16", False, id="lmcache-chunk-16"
        ),  # fmt: skip
    ],
)
def test_official_args(tmp_path, over, key, ok):
    """cap_fractions compare as sets; ``--replay-skip`` > 0 is a held-out
    slice of the trace, never official; the arm under test's module must be
    in the ``pegaflow`` package (M8: the only one the snapshot holds); the
    lmcache arm is official only with LMCache's default chunk size."""
    import argparse

    arm = over.pop("arm", "pegaflow.m:C")
    fixed = {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t/python", "lmcache_chunk_size": run_gate.LMCACHE_CHUNK, "lmcache_overlay": str(tmp_path)}  # fmt: skip
    args = argparse.Namespace(**(run_gate.OFFICIAL | fixed | over))
    arms = ["plain", "baseline", "lmcache", arm]
    probs = run_gate.official_problems(args, arms, list(run_gate.WORKLOADS), tmp_path, tmp_path)
    assert any(p.startswith(key) for p in probs) is not ok, probs


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


@pytest.mark.parametrize(
    "arm,exc,fatal",
    [
        ("old", ValueError("crash"), False),
        ("old", run_gate.TeardownError("group alive"), True),
        ("native", "teardown-during-crash", True),
        ("UnderTest", ValueError("crash"), True),
    ],
    ids=["ref-crash-recorded", "ref-teardown-aborts", "ref-teardown-chained", "under-test-crash"],
)
def test_stage_never_swallows_a_teardown_failure(monkeypatch, tmp_path, arm, exc, fatal):
    """S3: a reference arm's crash is recorded, but a surviving process group
    aborts the run (the next stage would start on a live GPU)."""

    def boom(*_a, **_k):
        if exc == "teardown-during-crash":
            try:
                raise ValueError("crash")
            finally:
                raise run_gate.TeardownError("group alive")
        raise exc

    monkeypatch.setattr(run_gate, "run_arm_workload", boom)
    if fatal:
        with pytest.raises((ValueError, run_gate.TeardownError)):
            run_gate.run_stage(None, {}, arm, "gsm8k", tmp_path)
    else:
        assert "error" in run_gate.run_stage(None, {}, arm, "gsm8k", tmp_path)


_GATE_LIKE = """
import sys, time
from pathlib import Path
from rsi.gates import run_gate
run_gate.install_stop_handlers()
port, out = sys.argv[1], Path(sys.argv[2])
cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
with run_gate.launched(cmd, {}, out / "sleeper.log", f"http://127.0.0.1:{port}/", 60) as p:
    (out / "pid").write_text(str(p.pid))
    time.sleep(600)
"""


def test_port_leases_are_exclusive_and_released(tmp_path):
    """Stage ports: leased under per-port flocks, so two gate runs never get
    the same port (single or vLLM's contiguous ``VLLM_PORT`` block); released
    when the stage ends, with the lease files in the record."""
    p = run_gate.free_port()
    a = run_gate.PortLease(1, tmp_path, (p, p))
    with pytest.raises(RuntimeError, match="no 1 free ports"):
        run_gate.PortLease(1, tmp_path, (p, p))
    assert a.record["files"] == [str(tmp_path / str(p))] and a.record["released_at"] is None
    a.release()
    assert a.record["released_at"] is not None
    b = run_gate.PortLease(1, tmp_path, (p, p))
    assert b.ports == a.ports == [p]
    b.release()
    lo = 20000 + p % 5000
    c = run_gate.PortLease(1, tmp_path, (lo, lo + 9), block=4)
    assert len(set(c.block)) == 4 and c.block == list(range(c.block[0], c.block[0] + 4))
    assert c.ports[0] not in c.block
    with pytest.raises(RuntimeError, match="contiguous"):
        run_gate.PortLease(1, tmp_path, (c.block[0], c.block[-1]), block=4)
    c.release()


class _Dead:
    pid = 4242

    def poll(self):
        return 1


@pytest.mark.parametrize(
    "line,ports",
    [
        pytest.param("OSError: [Errno 98] Address already in use", [], id="no-port"),
        pytest.param(
            "OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 33641): address already in use",
            [33641],
            id="uvicorn",
        ),  # fmt: skip
        pytest.param(
            "torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use",
            [29500],
            id="tcpstore",
        ),  # fmt: skip
        pytest.param(
            "zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:20017')",
            [20017],
            id="zmq",
        ),  # fmt: skip
        pytest.param(
            'Error: Os { code: 98, kind: AddrInUse, message: "Address in use" } (os error 98) binding 127.0.0.1:21000',
            [21000],
            id="rust",
        ),  # fmt: skip
    ],
)
def test_any_bind_failure_is_a_port_race(tmp_path, line, ports):
    """EADDRINUSE on any bind in a stage log (the gate's leased port or one
    vLLM / torch / ZMQ / the server picked itself) is a PortRace naming the
    parsed port, whether the gate leased it, and the lease files."""
    p = ports[0] if ports else run_gate.free_port()
    lease = run_gate.PortLease(1, tmp_path / "locks", (p, p))
    log = tmp_path / "vllm.log"
    log.write_text(f"INFO starting\n{line}\n")
    race = run_gate.port_race(log, [lease], _Dead())
    assert isinstance(race, run_gate.PortRace)
    msg = str(race)
    assert (f"{p} (leased)" in msg) if ports else ("unknown" in msg), msg
    assert str(tmp_path / "locks" / str(p)) in msg and "not retried" in msg
    log.write_text("RuntimeError: CUDA out of memory\n")
    assert run_gate.port_race(log, [lease], _Dead()) is None
    lease.release()


def test_port_race_fails_the_stage_naming_the_port(tmp_path, monkeypatch):
    """Another process binds the leased port after the lease, before the
    stage does: the stage fails with ``PortRace`` naming the port (not
    retried), its leases are released, and run_stage writes the cause and
    the lease files to ``stage-error.json``."""
    import json
    import socket

    p = run_gate.free_port()
    locks, out = tmp_path / "locks", tmp_path / "stage"
    cmd = [sys.executable, "-c", f"import socket; socket.socket().bind(('127.0.0.1', {p}))"]
    squatter = socket.socket()
    got: list[run_gate.PortLease] = []

    def stage(args, gate, arm, wl, out, **kw):
        got.append(run_gate.PortLease(1, locks, (p, p)))
        squatter.bind(("127.0.0.1", p))  # the race: taken after the lease
        squatter.listen()
        out.mkdir(parents=True)
        with run_gate.launched(cmd, {}, out / "vllm.log", f"http://127.0.0.1:{p}/x", 30, got[0]):
            pass

    monkeypatch.setattr(run_gate, "run_arm_workload", stage)
    try:
        with pytest.raises(run_gate.PortRace, match=f"{p} \\(leased\\)"):
            run_gate.run_stage(None, {}, "baseline", "gsm8k", out)
    finally:
        squatter.close()
    assert got[0]._fds == [] and got[0].record["released_at"] is not None
    err = json.loads((run_gate.gate_dir(out) / "stage-error.json").read_text())
    assert err["type"] == "PortRace" and f"{p} (leased)" in err["error"]
    assert err["port_leases"][-1]["files"] == [str(locks / str(p))]


def test_leases_are_released_on_every_exit_path(tmp_path, monkeypatch):
    """A stage that raises before it launches, or is stopped by a signal
    (SystemExit), releases every lease it took."""
    p = run_gate.free_port()

    def take_then(exc):
        def stage(args, gate, arm, wl, out, **kw):
            run_gate.PortLease(1, tmp_path, (p, p))
            raise exc

        return stage

    for exc in (ValueError("before launch"), SystemExit(143)):
        monkeypatch.setattr(run_gate, "run_arm_workload", take_then(exc))
        with pytest.raises(type(exc)):
            run_gate.run_stage(None, {}, "baseline", "gsm8k", tmp_path / "s")
        assert run_gate._LEASES[-1]._fds == []
        run_gate.PortLease(1, tmp_path, (p, p)).release()  # free again


def test_sigterm_tears_down_launched_groups(tmp_path):
    """``kill -TERM <gate>`` must stop the stages' process groups (own
    sessions, so the signal never reaches them) before the gate exits."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.Popen([sys.executable, "-c", _GATE_LIKE, port, str(tmp_path)], env=env)
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    sleeper = int(pid_file.read_text())
    os.killpg(sleeper, 0)  # alive, own group
    gate.send_signal(signal.SIGTERM)
    assert gate.wait(timeout=60) == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(sleeper, 0)


_WINDOW_LIKE = """
import os, signal, subprocess, sys, time
from pathlib import Path
from rsi.gates import run_gate
port, out, where = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
real_popen, real_stop = subprocess.Popen, run_gate.stop


def popen(*a, **kw):  # the signal lands right after Popen returns
    p = real_popen(*a, **kw)
    (out / "pid").write_text(str(p.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return p


def stop(proc, name, fh, _first=[True]):  # ... or on entering finally:
    if _first.pop() if _first else False:
        os.kill(os.getpid(), signal.SIGTERM)
    return real_stop(proc, name, fh)


if where == "popen":
    run_gate.subprocess.Popen = popen
else:
    run_gate.stop = stop
run_gate.install_stop_handlers()
cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
with run_gate.launched(cmd, {}, out / "server.log", f"http://127.0.0.1:{port}/", 60) as p:
    (out / "pid").write_text(str(p.pid))
(out / "after").write_text("")
"""


@pytest.mark.parametrize("where", ["popen", "finally"])
def test_stop_signal_in_a_launch_window_still_stops_the_group(tmp_path, where):
    """N5: a stop signal right after ``Popen`` returns, or on entering the
    ``finally:`` before ``stop`` owns the group, still stops the group
    (deferred, or by ``stop_live`` at exit) and the gate exits 128 + sig."""
    import os
    import signal
    import subprocess
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.run(
        [sys.executable, "-c", _WINDOW_LIKE, port, str(tmp_path), where],
        env=env,
        timeout=120,
        capture_output=True,
        text=True,
    )
    assert gate.returncode == 128 + signal.SIGTERM, gate.stdout + gate.stderr
    assert not (tmp_path / "after").exists()
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    if where == "finally":
        assert "left by a signal window" in gate.stdout, gate.stdout


@pytest.mark.parametrize(
    "path_ver,home_ver,problem",
    [
        pytest.param("13.0", "13.0", None, id="both-match"),
        pytest.param(None, "13.0", None, id="home-only"),
        pytest.param("12.2", None, "is CUDA 12.2, torch is CUDA 13.0", id="path-12-attempt-1"),
        pytest.param("13.0", "12.2", "is CUDA 12.2", id="home-mismatch"),
        pytest.param(None, None, "no nvcc", id="none"),
    ],
)
def test_cuda_env_must_match_torch(tmp_path, path_ver, home_ver, problem):
    """GPU3 attempt 1: the system nvcc 12.2 on PATH, torch on CUDA 13, and
    FlashInfer's JIT failed mid-run. Every nvcc a stage can pick must have
    torch's CUDA major; paths and versions are recorded."""

    def nvcc(d, ver):
        b = d / "bin"
        b.mkdir(parents=True)
        f = b / "nvcc"
        f.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {ver}, V{ver}.1'\n")
        f.chmod(0o755)
        return d

    env = {"PATH": "/nonexistent", "LD_LIBRARY_PATH": "/x/lib"}
    if path_ver:
        env["PATH"] = str(nvcc(tmp_path / "p", path_ver) / "bin")
    if home_ver:
        env["CUDA_HOME"] = str(nvcc(tmp_path / "h", home_ver))
    rec = run_gate.cuda_env(env, torch_cuda="13.0")
    assert rec["nvcc_path_version"] == path_ver and rec["nvcc_cuda_home_version"] == home_ver
    assert rec["LD_LIBRARY_PATH"] == "/x/lib"
    got = run_gate.cuda_problems(rec)
    assert (problem in " ".join(got)) if problem else not got, got


# ------------------------------------------------------------ provenance


def _git_tree(tmp_path):
    """A repo with python/pegaflow/{__init__,a,conn}.py committed; returns (repo, rev)."""
    import subprocess

    repo = tmp_path / "src"
    pkg = repo / "python" / "pegaflow"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "conn.py").write_text("print('original')\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "config", "uploadpack.allowFilter", "true"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    rev = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    return repo, rev


def _copy(repo, tmp_path, rev, edit=None, extra=None):
    import shutil

    dst = tmp_path / "copy"
    shutil.copytree(repo / "python", dst / "python")
    (dst / ".rsi-rev").write_text(rev + "\n")
    if edit:
        (dst / "python" / "pegaflow" / "a.py").write_text(edit)
    for name, body in (extra or {}).items():
        (dst / "python" / "pegaflow" / name).write_bytes(body)
    return dst


def _native(pkg):
    from rsi.gates import provenance

    return pkg / Path(provenance.native_module()).name


@pytest.mark.parametrize(
    "case,verified",
    [("git-clean", True), ("git-dirty", False), ("rsi-rev", True), ("rsi-rev-edited", False),
     ("rsi-rev-extra-py", False), ("rsi-rev-native", True), ("unknown-rev", False), ("no-rev", False)],
)  # fmt: skip
def test_tree_rev_is_verified_blob_for_blob(tmp_path, case, verified):
    """M3: a run is official with a tree under test only if its rev is
    verified: a clean git toplevel, or ``.rsi-rev`` whose tracked files match
    on disk (the expected native module is allowed and hashed)."""
    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    if case.startswith("git"):
        root = repo
        if case == "git-dirty":
            (repo / "python" / "pegaflow" / "a.py").write_text("x = 2\n")
    elif case == "no-rev":
        root = _copy(repo, tmp_path, rev)
        (root / ".rsi-rev").unlink()
    else:
        root = _copy(
            repo,
            tmp_path,
            "0" * 40 if case == "unknown-rev" else rev,
            edit="x = 2\n" if case == "rsi-rev-edited" else None,
            extra={"b.py": b""} if case == "rsi-rev-extra-py" else None,
        )
        if case == "rsi-rev-native":
            _native(root / "python" / "pegaflow").write_bytes(b"elf")
    rec = trees.tree(root, ("python/pegaflow",), provenance.native_module())
    assert rec["verified"] is verified, rec
    if case == "rsi-rev-native":
        assert list(rec["native_sha256"]) == [provenance.native_module()]
    if verified:
        assert rec["rev"] == rev


def _plant(case, pkg, tmp_path):
    """M5 bypasses: each makes ``import pegaflow.conn`` run injected code
    from a tree whose tracked blobs are all intact."""
    import importlib.util
    import marshal

    evil = tmp_path / "elsewhere"
    if case == "dir-symlink":
        (evil / "conn").mkdir(parents=True)
        (evil / "conn" / "__init__.py").write_text("print('INJECTED')\n")
        (pkg / "conn").symlink_to(evil / "conn", target_is_directory=True)
    elif case in ("unchecked-pyc", "git-ignored-pyc"):
        code = compile("print('INJECTED')\n", str(pkg / "conn.py"), "exec")
        # PEP 552 unchecked-hash pyc: the importer never compares it with conn.py.
        head = importlib.util.MAGIC_NUMBER + (1).to_bytes(4, "little") + b"\0" * 8
        cache = pkg / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / f"conn.{sys.implementation.cache_tag}.pyc").write_bytes(head + marshal.dumps(code))
    elif case == "extra-so":
        ext = Path(_native(pkg)).name.removeprefix("pegaflow")
        (pkg / f"conn{ext}").write_bytes(b"elf")  # beats conn.py in the loader order
    elif case == "file-symlink":
        evil.mkdir(parents=True)
        (evil / "conn.py").write_text("print('INJECTED')\n")
        (pkg / "conn.py").unlink()
        (pkg / "conn.py").symlink_to(evil / "conn.py")


@pytest.mark.parametrize("case", ["dir-symlink", "unchecked-pyc", "git-ignored-pyc", "extra-so", "file-symlink"])  # fmt: skip
def test_tree_bypasses_are_rejected(tmp_path, case):
    """M5: every file Python could import from a tree is accounted for. A
    symlink, an untracked or ignored ``.pyc`` (even in ``__pycache__``, even
    in a clean git toplevel whose ``status`` hides it) and any native module
    but the expected one reject the tree."""
    import subprocess

    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    if case == "git-ignored-pyc":
        (repo / ".gitignore").write_text("__pycache__/\n*.so\n")
        subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ignore"], check=True)  # fmt: skip
        root = repo
        _plant(case, repo / "python/pegaflow", tmp_path)
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        )
        assert status == ""  # what the old check looked at
    else:
        root = _copy(repo, tmp_path, rev)
        _plant(case, root / "python/pegaflow", tmp_path)
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    rec = trees.tree(root, ("python/pegaflow",), provenance.native_module())
    assert rec["verified"] is False, rec
    staged = provenance.stage(root / "python", tmp_path / "ov", trees)
    assert staged["verified"] is False, staged
    # The snapshot never carries the planted file: importing it runs the
    # tracked code only, with no cache dir in the snapshot.
    snap = Path(staged["pegaflow_dir"])
    assert not any(p.suffix in (".pyc", ".so") or p.is_symlink() for p in snap.rglob("*"))
    if (snap / "conn.py").exists():
        env = {"PYTHONPATH": str(tmp_path / "ov"), "PYTHONPYCACHEPREFIX": str(tmp_path / "pyc")}
        out = subprocess.check_output([sys.executable, "-c", "import pegaflow.conn"], env=env, text=True)  # fmt: skip
        assert out == "original\n"


def test_stage_copies_only_the_verified_rev(tmp_path):
    """M5: the arm imports a snapshot (tracked blobs + the expected native
    module, sha256 equal to the source's), so editing the tree mid-run
    changes nothing it imports, and the snapshot holds only ``pegaflow``."""
    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    root = _copy(repo, tmp_path, rev)
    for d in ("python/rsi/gates", "python/vllm"):
        (root / d).mkdir(parents=True)
    (root / "python/sitecustomize.py").write_text("")
    _native(root / "python/pegaflow").write_bytes(b"elf")
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    rec = provenance.stage(root / "python", tmp_path / "ov", trees)
    assert rec["verified"], rec
    assert [p.name for p in (tmp_path / "ov").iterdir()] == ["pegaflow"]
    snap = Path(rec["pegaflow_dir"])
    assert sorted(p.name for p in snap.iterdir()) == sorted(["__init__.py", "a.py", "conn.py", _native(snap).name])  # fmt: skip
    assert rec["native_sha256"] == provenance.sha256(_native(root / "python/pegaflow"))
    (root / "python/pegaflow/a.py").write_text("x = 'edited mid-run'\n")
    assert (snap / "a.py").read_text() == "x = 1\n"


@pytest.mark.parametrize(
    "role,edit",
    [
        pytest.param(None, None, id="unchanged"),
        pytest.param("gate", "rsi/gates/rsi_probe.py", id="gate-probe-edited"),
        pytest.param("gate", "rsi/gates/new.py", id="gate-file-added"),
        pytest.param("under_test", "pegaflow/a.py", id="snapshot-edited"),
        pytest.param("pegaflow_server", "pegaflow-server", id="server-binary-replaced"),
    ],
)
def test_a_tree_changed_between_stages_is_named(tmp_path, monkeypatch, role, edit):
    """M10: what was verified before stage 1 is fingerprinted and checked
    again before every stage (and after the last one); a change makes the
    run non-official with the role and the path named. In-tree pyc of the
    gate tree is not part of it (never read)."""
    import argparse
    import sys

    gate = tmp_path / "gate"
    (gate / "rsi/gates").mkdir(parents=True)
    (gate / "rsi/gates/rsi_probe.py").write_text("x = 1\n")
    (gate / "python/pegaflow").mkdir(parents=True)
    (gate / "python/pegaflow/__init__.py").write_text("")
    ov = tmp_path / "ov"
    (ov / "pegaflow").mkdir(parents=True)
    (ov / "pegaflow/a.py").write_text("y = 1\n")
    srv = tmp_path / "bin"
    srv.mkdir()
    (srv / "pegaflow-server").write_bytes(b"elf")
    monkeypatch.setattr(run_gate, "REPO", gate)
    monkeypatch.setattr(run_gate.provenance, "gate_pegaflow_dir", lambda: None)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "pyc"))
    args = argparse.Namespace(arms="baseline,pegaflow.m:C", under_test_path="/t", under_test_overlay=str(ov), pegaflow_server=str(srv / "pegaflow-server"))  # fmt: skip
    watch = run_gate.ProvenanceWatch(args, gate / "docs/rsi/models/M")
    (gate / "rsi/gates/__pycache__").mkdir()
    (gate / "rsi/gates/__pycache__/rsi_probe.cpython-312.pyc").write_bytes(b"pyc")
    assert watch.check("baseline/gsm8k") == {}
    if edit:
        base = {"gate": gate, "under_test": ov, "pegaflow_server": srv}[role]
        (base / edit).write_text("tampered\n")
    changes = watch.check("ut/replay")
    probs = watch.problems()
    if role is None:
        assert changes == {} and probs == []
    else:
        assert list(changes) == [role], changes
        assert len(probs) == 1 and "before ut/replay" in probs[0] and Path(edit).name in probs[0], probs  # fmt: skip
    assert [c["before"] for c in watch.record()["checks"]] == ["baseline/gsm8k", "ut/replay"]


def test_each_stage_gets_a_fresh_bytecode_cache_outside_the_arm(tmp_path):
    """M10: a stage's PYTHONPYCACHEPREFIX is its own dir, outside every arm
    dir; the isolated arm gets only its current stage's cache and its own
    compile cache (S5) bound rw."""
    import argparse

    run = tmp_path / "runs" / "r"
    env: dict[str, str] = {}
    run_gate.pycache_env(env, run / "ut" / "replay" / "vllm.log")
    assert env["PYTHONPYCACHEPREFIX"] == str(run / ".pycache-stages" / "ut" / "replay")
    assert not Path(env["PYTHONPYCACHEPREFIX"]).is_relative_to(run / "ut")
    args = argparse.Namespace(
        runs_root=str(tmp_path / "runs"), under_test_path="", under_test_overlay=""
    )
    iso = run_gate.isolation(args, run / "ut" / "replay")
    assert iso["rw"] == [str((run / "ut" / "replay").resolve()), str((run / ".pycache-stages" / "ut" / "replay").resolve()), str((run / ".compile-cache" / "ut").resolve())]  # fmt: skip


@pytest.mark.parametrize("bad", [None, "gate", "venv_pegaflow", "under_test"])
def test_every_imported_tree_must_be_verified(bad):
    """M6: an unverified gate tree (the judge) or venv ``pegaflow`` (the
    oracle's semantics) makes the run non-official, as the tree under test
    does."""
    prov = {
        "gate": {"root": "/g", "verified": True},
        "venv_pegaflow": {"root": "/v", "verified": True},
        "under_test": {"source": {"root": "/u"}, "verified": True},
    }
    if bad:
        prov[bad] = prov[bad] | {"verified": False, "reason": "tree differs from rev"}
    problems = run_gate.provenance_problems(prov)
    assert len(problems) == (bad is not None), problems
    assert run_gate.provenance_problems({"gate": {"verified": True}}) == [
        "venv pegaflow (oracle semantics, reference arms): provenance not recorded"
    ]


_needs_isolation = pytest.mark.skipif(
    not run_gate.isolation_available(), reason="isolate.py needs Linux and root"
)

_PROBE_ESCAPE = """
import os, sys
base, arm, gate_pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = {}
seen["baseline_dir"] = os.path.exists(f"{base}/baseline/gsm8k/warm-outputs.json")
seen["own_dir"] = os.path.exists(f"{arm}/gsm8k/mine.txt")
seen["earlier_stage"] = os.path.exists(f"{arm}/replay/digest.jsonl")
seen["gate_dir"] = os.path.exists(f"{base}/.gate/ut/gsm8k/warm-outputs.json")
open(f"{arm}/gsm8k/written.txt", "w").write("ok")
for how in ("root", "cwd"):
    try:
        seen[how] = os.path.exists(f"/proc/{gate_pid}/{how}{base}/baseline/gsm8k/warm-outputs.json")
    except OSError:
        seen[how] = False
try:
    fd = os.open(f"/proc/{gate_pid}/ns/mnt", os.O_RDONLY)
    os.setns(fd, 0)
    seen["setns"] = True
except OSError:
    seen["setns"] = False
try:
    import ctypes
    seen["umount"] = ctypes.CDLL(None, use_errno=True).umount2(base.encode(), 2) == 0
except Exception:
    seen["umount"] = False
print(repr(seen))
"""


@_needs_isolation
def test_isolated_arm_cannot_read_other_arms(tmp_path):
    """M7(a), M9: the arm under test sees its current stage dir (rw) and
    not the baseline's outputs, its own earlier stages or the gate dir,
    neither by path nor through ``/proc/<gate>/root`` or ``cwd``, and cannot
    unmount the cover."""
    import argparse
    import os
    import subprocess

    runs = tmp_path / "runs"
    base = runs / "2026-10-24-t"
    (base / "baseline" / "gsm8k").mkdir(parents=True)
    (base / "baseline" / "gsm8k" / "warm-outputs.json").write_text("[]")
    arm = base / "ut"
    (arm / "gsm8k").mkdir(parents=True)
    (arm / "gsm8k" / "mine.txt").write_text("")
    (arm / "replay").mkdir()
    (arm / "replay" / "digest.jsonl").write_text("")
    (base / ".gate" / "ut" / "gsm8k").mkdir(parents=True)
    (base / ".gate" / "ut" / "gsm8k" / "warm-outputs.json").write_text("[]")
    args = argparse.Namespace(runs_root=str(runs), under_test_path="", under_test_overlay="")
    cmd = run_gate.isolated(args, "m:C", arm / "gsm8k" / "vllm.log", [sys.executable, "-c", _PROBE_ESCAPE, str(base), str(arm), str(os.getpid())])  # fmt: skip
    assert cmd[0] == sys.executable and "--hide" in cmd
    out = subprocess.check_output(cmd, text=True, timeout=60)
    seen = eval(out)  # noqa: S307 - our own child's repr
    assert seen == {"baseline_dir": False, "own_dir": True, "earlier_stage": False, "gate_dir": False, "root": False, "cwd": False, "setns": False, "umount": False}, seen  # fmt: skip
    assert (arm / "gsm8k" / "written.txt").read_text() == "ok"  # rw bind reaches the host dir
    assert os.path.exists(base / "baseline" / "gsm8k" / "warm-outputs.json")  # host unchanged


_ISOLATED_GATE = """
import sys, time, argparse
from pathlib import Path
from rsi.gates import run_gate
run_gate.install_stop_handlers()
port, runs = sys.argv[1], Path(sys.argv[2])
arm = runs / "r" / "ut" / "gsm8k"
arm.mkdir(parents=True, exist_ok=True)
args = argparse.Namespace(runs_root=str(runs), under_test_path="", under_test_overlay="")
# A child that forks a grandchild, as vLLM forks EngineCore / workers, and
# one that setsid()s out of the group (S11); its pid goes to the arm's dir.
inner = "import subprocess, sys, http.server; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); s = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], start_new_session=True); open(sys.argv[2], 'w').write(str(s.pid)); http.server.test(http.server.SimpleHTTPRequestHandler, port=int(sys.argv[1]), bind='127.0.0.1')"
cmd = run_gate.isolated(args, "m:C", arm / "vllm.log", [sys.executable, "-c", inner, port, str(arm / "setsid.pid")])
with run_gate.launched(cmd, {}, arm / "vllm.log", f"http://127.0.0.1:{port}/", 60) as p:
    (runs / "pid").write_text(str(p.pid))
    time.sleep(600)
"""


@_needs_isolation
def test_isolated_arm_is_torn_down_with_its_group(tmp_path):
    """M7(a): the isolated arm is exec'd in place, so it is the group
    leader ``launched`` started; its health endpoint is reachable on
    loopback, and SIGTERM to the gate stops every member of the group (the
    grandchild included) before the gate exits. S11: a child that setsid()'d
    out of the group is stopped too (it is in the arm's mount namespace)."""
    import os
    import signal
    import subprocess
    import time

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.Popen([sys.executable, "-c", _ISOLATED_GATE, port, str(tmp_path)], env=env)
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    leader = int(pid_file.read_text())
    assert os.readlink(f"/proc/{leader}/ns/mnt") != os.readlink("/proc/self/ns/mnt")
    assert os.readlink(f"/proc/{leader}/ns/pid") == os.readlink("/proc/self/ns/pid")
    members = [p for p in os.listdir("/proc") if p.isdigit() and _pgid(int(p)) == leader]
    assert len(members) >= 2, members  # leader + grandchild
    escaped = int((tmp_path / "r" / "ut" / "gsm8k" / "setsid.pid").read_text())
    assert _pgid(escaped) == escaped != leader  # out of the group
    assert os.readlink(f"/proc/{escaped}/ns/mnt") == os.readlink(f"/proc/{leader}/ns/mnt")
    gate.send_signal(signal.SIGTERM)
    assert gate.wait(timeout=120) == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(leader, 0)
    assert _dead(escaped), "setsid'd child survived teardown"


def _env_with_path(root: Path) -> dict[str, str]:
    """The test's env with the repo root prepended to ``PYTHONPATH`` (kept,
    not replaced: it may be what finds the built ``pegaflow``)."""
    import os

    path = [str(root), *filter(None, [os.environ.get("PYTHONPATH")])]
    return os.environ | {"PYTHONPATH": os.pathsep.join(path)}


def _dead(pid: int) -> bool:
    """Gone, or a zombie its new parent (init) has not reaped yet."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


def _pgid(pid: int) -> int | None:
    import os

    try:
        return os.getpgid(pid)
    except OSError:
        return None


@pytest.mark.parametrize(
    "body,hit",
    [
        ("x = 1\n", None),
        ("from rsi.gates import rsi_probe\n", "rsi"),
        ("see docs/rsi/models/Qwen3-4B/ITERATIONS.md, built in pegaflow-semantics-rsi/\n", None),
        ("m = importlib.import_module('rsi.gates.x')\n", "rsi"),
        ("import sys\nsys.modules['rsi.gates.rsi_probe'] = None\n", "sys.modules write"),
        ("import gc\nobjs = gc.get_objects()\n", "gc introspection"),
        ("p = run_dir / 'baseline' / 'gsm8k' / 'warm-outputs.json'\n", "-outputs.json"),
        ("glob('digest-*.jsonl')\n", "digest-"),
        ("sys.meta_path.insert(0, Finder())\n", "import hook"),
    ],
    ids=[
        "clean",
        "rsi-import",
        "rsi-in-paths",
        "rsi-module-string",
        "sys-modules",
        "gc",
        "outputs",
        "digests",
        "meta-path",
    ],
)
def test_static_scan_flags_instrument_access(tmp_path, body, hit):
    """M7(b): a snapshot that names the instrument or run artifacts, or
    patches modules at runtime, is flagged (run non-official until a
    reviewer clears the id); strings of a native module are scanned too."""
    import argparse

    from rsi.gates import provenance

    pkg = tmp_path / "ov" / "pegaflow"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "conn.py").write_text(body)
    hits = provenance.scan(pkg)
    names = {h["id"].rsplit(":", 1)[1] for h in hits}
    assert (hit in names) if hit else not names, hits
    args = argparse.Namespace(under_test_path="x", under_test_overlay=str(tmp_path / "ov"))
    assert bool(run_gate.scan_record(args)["flagged"]) is bool(hit)
    allow = [{"path": "pegaflow/conn.py", "pattern": n, "reason": "test"} for n in names]
    rec = run_gate.scan_record(args, allow)
    assert rec["flagged"] == [] and len(rec["allowlisted"]) == len(hits)
    assert all(h["reason"] == "test" for h in rec["allowlisted"])
    (pkg / "conn.py").write_text("x = 1\n")
    (pkg / "pegaflow.so").write_bytes(b"\x7fELF\x00\x01rsi_poison\x00")
    assert [h["id"] for h in provenance.scan(pkg)] == ["pegaflow/pegaflow.so:strings:rsi_poison"]


def test_official_needs_under_test_path(tmp_path):
    """An arm under test without ``--under-test-path`` imports the venv's
    pegaflow (no snapshot, no verification): never official."""
    import argparse

    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": ""}))  # fmt: skip
    probs = run_gate.official_problems(args, ["plain", "baseline", "m:C"], list(run_gate.WORKLOADS), tmp_path, tmp_path)  # fmt: skip
    assert any(p.startswith("--under-test-path missing") for p in probs), probs


@pytest.mark.parametrize(
    "arm,recorded,ok",
    [
        pytest.param("pegaflow.connector_v2:C", "snap", True, id="under-test-from-snapshot"),
        pytest.param("pegaflow.connector_v2:C", None, False, id="under-test-not-recorded"),
        pytest.param("pegaflow.connector_v2:C", "venv", False, id="under-test-from-venv"),
        pytest.param("pegaflow.connector_v2:C", "gate", False, id="under-test-from-gate"),
        pytest.param("rsi.gates.myconn:C", "gate", False, id="rsi-module-under-test-from-gate"),
        pytest.param("rsi.gates.myconn:C", "snap", True, id="rsi-module-under-test-from-snapshot"),
        pytest.param("selftest:honest", "gate", True, id="selftest-from-gate"),
        pytest.param("selftest:honest", "snap", False, id="selftest-from-snapshot"),
    ],
)
def test_connector_module_must_come_from_the_snapshot(tmp_path, arm, recorded, ok):
    """Import-tree provenance: the probe records ``__file__`` of the arm's
    connector module. Its root is chosen by arm kind (M8): only the declared
    self-test arms resolve under the gate tree; an arm under test must
    resolve under its snapshot whatever its module name (an ``rsi.*`` one
    in the gate tree is never staged or scanned)."""
    import json
    import sys

    from rsi.gates import provenance

    snap = tmp_path / "ov" / "pegaflow"
    purelib = provenance.venv_purelib()
    files = {
        "rsi.gates.rsi_probe": str(run_gate.REPO / "rsi/gates/rsi_probe.py"),
        "worker_extension": str(run_gate.REPO / "rsi/gates/rsi_probe.py"),
        "vllm": str(purelib / "vllm/__init__.py"),
        "torch": str(purelib / "torch/__init__.py"),
        "pegaflow": str(snap / "__init__.py"),
        "prefix": sys.prefix,
    }
    mod = run_gate.expected_connector(arm)
    where = {"snap": snap, "venv": tmp_path / "venv" / "pegaflow", "gate": run_gate.REPO / "rsi"}
    if recorded:
        files[f"connector:{mod}"] = str(where[recorded] / "conn" / "__init__.py")
    probe = tmp_path / "ut" / "gsm8k" / "probe"
    probe.mkdir(parents=True)
    (probe / "spec.json").write_text(json.dumps({"modules": files}))
    root = run_gate.connector_root(arm, snap.resolve())
    v = run_gate.g0_provenance(tmp_path, "ut", snap.resolve(), mod, root)
    assert v["pass"] is ok, v
    assert run_gate.expected_connector("selftest:honest") == "rsi.gates.selftest.connector"
    assert run_gate.expected_connector("native") is None


def test_connector_under_test_is_wired_to_the_probe(tmp_path):
    mod = "pegaflow.connector_v2"
    cfg = run_gate.kv_transfer_config(f"{mod}:PegaKVConnectorV2", tmp_path, "tagged")
    probe_cfg = cfg["kv_connector_extra_config"]["connectors"][1]
    assert probe_cfg["kv_connector_extra_config"]["connector_modules"] == [mod]


@pytest.mark.parametrize(
    "arm,plugins",
    [
        ("old", run_gate.PEGAFLOW_PLUGIN),
        ("plain", ""),
        ("baseline", ""),
        ("native", ""),
        ("lmcache", ""),
        ("selftest:honest", ""),
        ("pegaflow.connector_v2:PegaKVConnectorV2", ""),
    ],
)
def test_only_the_old_arm_loads_the_pegaflow_plugin(tmp_path, monkeypatch, arm, plugins):
    """vLLM's API server looks MultiConnector children up by registered
    name to rebuild their stats; ``PegaKVConnector`` is registered only by
    the pegaflow plugin. Every other arm runs with no plugin (M7d)."""
    # The name is the venv package's entry point (python/pyproject.toml).
    eps = (
        (run_gate.REPO / "python/pyproject.toml")
        .read_text()
        .split('[project.entry-points."vllm.general_plugins"]\n', 1)[1]
    )
    assert eps.startswith(f'{run_gate.PEGAFLOW_PLUGIN} = "pegaflow.vllm_plugin:register"')
    seen: dict[str, str] = {}

    class _Lease:
        ports, block = [20001], [20010]

    def launched(cmd, env, log, url, timeout, lease):
        seen["plugins"] = env["VLLM_PLUGINS"]
        return None

    monkeypatch.setattr(run_gate, "PortLease", lambda *a, **kw: _Lease())
    monkeypatch.setattr(run_gate, "launched", launched)
    monkeypatch.setattr(run_gate, "isolated", lambda args, arm, log, cmd: cmd)
    args = SimpleNamespace(
        gpu="0", model="m", tp=1, max_model_len=8, max_num_batched_tokens=8, max_num_seqs=1,
        gpu_mem=0.5, vllm_arg=[], num_gpu_blocks=0, under_test_path="",
        lmcache_overlay=str(tmp_path),
        lmcache_chunk_size=run_gate.LMCACHE_CHUNK,
        lmcache_ranks=1,
    )  # fmt: skip
    run_gate.vllm_server(args, {}, arm, tmp_path / "probe", False, None, tmp_path / "log", 1 << 30)
    assert seen["plugins"] == plugins == run_gate.vllm_plugins(arm)


def test_module_files_must_resolve_under_gate_and_venv(tmp_path):
    import sys

    from rsi.gates import provenance

    repo = run_gate.REPO
    purelib = provenance.venv_purelib()
    pf = tmp_path / "pf"
    good = {
        "rsi.gates.rsi_probe": str(repo / "rsi/gates/rsi_probe.py"),
        "worker_extension": str(repo / "rsi/gates/rsi_probe.py"),
        "vllm": str(purelib / "vllm/__init__.py"),
        "torch": str(purelib / "torch/__init__.py"),
        "pegaflow": str(pf / "__init__.py"),
        "prefix": sys.prefix,
    }
    assert provenance.module_problems(good, repo, pf, "x") == []
    for k, bad in (
        ("rsi.gates.rsi_probe", str(tmp_path / "rsi/gates/rsi_probe.py")),
        ("vllm", str(tmp_path / "vllm/__init__.py")),
        ("pegaflow", str(tmp_path / "other/__init__.py")),
        ("sitecustomize", str(tmp_path / "sitecustomize.py")),
    ):
        assert provenance.module_problems(good | {k: bad}, repo, pf, "x"), k


@pytest.mark.parametrize("pool,fits", [("64gb", True), ("1kb", False)])
def test_pool_check_fails_before_an_evicting_store(pool, fits):
    import argparse

    args = argparse.Namespace(pool_size=pool)
    u = {"workload": "replay", "unique_bytes": 2048, "how": "estimate"}
    if fits:
        assert run_gate.check_pool(args, u)["fits"]
    else:
        with pytest.raises(run_gate.PoolTooSmall):
            run_gate.check_pool(args, u)


def test_contamination_marks_not_drops():
    rep = lambda **p: {"replay": {"passes": {k: {"ttft_p99": v} for k, v in p.items()}}}  # noqa: E731
    results = {
        "baseline": rep(cold=1.0, warm2=0.5, warm3=0.5),
        "old": rep(cold=9.0, warm2=6.0, warm3=0.4),
        "X": rep(cold=1.0, warm=50.0, warm2=0.5, warm3=5.0, _warmup=99.0),
    }
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    # 9x and exactly 10x pass; 12x is flagged; untimed warm / _warmup ignored.
    assert got == {("old", "warm2")}
    # S8: a baseline stall (p99 > 10x the arms' median for that pass).
    results["baseline"] = rep(cold=1.0, warm2=0.5, warm3=40.0)
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    assert ("baseline", "warm3") in got  # median(0.4, 5.0) = 2.7; 40 > 27
    results["baseline"] = rep(cold=1.0, warm2=0.5, warm3=27.0)
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    assert ("baseline", "warm3") not in got


def test_warmup_seconds_reported_not_gated():
    """S9: ``passes._warmup.seconds`` per arm vs baseline, in the G4 verdicts
    without a ``pass`` key (reported, not gated)."""

    def rep(w, c):
        return {"passes": {"_warmup": {"seconds": w}, "cold": {"ttft_p50": c, "req_per_s": 1.0}}}

    base_gsm = {"passes": {"_warmup": {"seconds": 0.5}}}
    g = run_gate._perf_verdicts(rep(3.0, 1.0), rep(1.0, 1.0), None, base_gsm, True)
    assert g["G4.warmup_seconds"] == {
        "replay": {"value": 3.0, "baseline": 1.0},
        "gsm8k": {"value": None, "baseline": 0.5},
    }
    assert "pass" not in g["G4.warmup_seconds"]


def _replay(
    ttft: float | tuple[float, ...],
    rps: float | tuple[float, ...],
    warm: tuple[str, ...] = ("warm2", "warm3"),
    eff: float = 1.0,
) -> dict:
    """A replay stage result with correct KV whose timed warm passes have
    ``ttft`` (TTFT p50), ``rps`` (req/s) and ``eff`` (E_ext); a tuple gives
    one value per timed pass, a number the same value on every pass."""

    def per_pass(x):
        return x if isinstance(x, tuple) else (x,) * len(warm)

    hit = {"efficiency": eff, "oracle_hit_rate": 0.9, "hit_rate": 0.9, "local_hit_tokens": 0, "prompt_tokens": 1, "external_hit_rate": 0.9, "excess_tokens": 0, "excess_limit": 1}  # fmt: skip
    passes = {"cold": {"ttft_p50": 1.0, "req_per_s": 1.0}, "warm": {"ttft_p50": 9.0, "req_per_s": 0.1}}  # fmt: skip
    passes |= {
        n: {"ttft_p50": t, "req_per_s": r}
        for n, t, r in zip(warm, per_pass(ttft), per_pass(rps), strict=True)
    }
    hits = dict.fromkeys(("warm", *warm), hit)
    return {"passes": passes, "hit": hits, "load_errors": [], "digest_self": _digest_ok()}


NATIVE_KEYS = ("G4.replay_warm.ttft_p50_vs_native", "G4.replay_warm.req_per_s_vs_native")


@pytest.mark.parametrize(
    "mine,native,gap_pct,within",
    [
        # Qwen3-4B it7: TTFT 2.1 ms behind, inside v2's 25.2 ms warm2-warm3
        # range; req/s 0.287 ahead, outside both ranges (0.028, 0.135).
        (((0.5514, 0.5766), (9.6597, 9.6319)), ((0.5612, 0.5626), (9.2913, 9.4260)), (0.37, 3.07), (True, False)),
        # DSV2 it7: TTFT inside native's 24 ms range; req/s 5.6% behind.
        (((0.2957, 0.3035), (12.4927, 12.1985)), ((0.2799, 0.3039), (13.0061, 13.1634)), (2.64, -5.65), (True, False)),
        # Qwen3.5 it2: TTFT +72% (0.300 s vs 0.172 s), reported, never a fail.
        (((0.2996, 0.2952), (14.6105, 14.6551)), ((0.1745, 0.1712), (13.1495, 13.1609)), (72.06, 11.23), (False, False)),
        # |gap| == the larger range is within noise (exact binary fractions).
        (((0.5, 1.0), (2.0, 3.0)), ((0.25, 0.25), (1.5, 1.5)), (200.0, 66.67), (True, True)),
        (((0.5, 1.0), (2.0, 3.0)), ((0.125, 0.125), (1.0, 1.0)), (500.0, 150.0), (False, False)),
        (((0.3, 0.3), (8.0, 8.0)), ((0.3, 0.3), (8.0, 8.0)), (0.0, 0.0), (True, True)),  # tie, range 0
    ],
    ids=["qwen3-4b-it7", "dsv2-it7", "qwen35-it2", "gap-equals-range", "gap-above-range", "tie"],
)  # fmt: skip
def test_g4_vs_native_reported(tmp_path, mine, native, gap_pct, within):
    """User, 2026-10-26: v2 vs native on G4 warm is a judged comparison
    ("差不多就行了"), not a gate. ``*_vs_native`` is ``reported`` (no ``pass``
    key) and carries what the perf vs native judgment needs: per-arm pass
    values, medians, ranges, gap, gap %, noise, ``within_noise``. Native's
    own G0 / G2 do not matter; G4 vs baseline stays hard."""
    results = {"baseline": {"replay": _replay(5.0, 1.0)}}
    results |= {"pegaflow.m:C": {"replay": _replay(*mine)}}
    results |= {"native": {"replay": _replay(*native) | {"digest_self": None}}}
    v = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")
    g = v["pegaflow.m:C"]
    assert not v["native"]["G2.digest_self.replay"]["pass"]  # its KV verdicts fail ...
    for k, lo, hi, pct, w in zip(NATIVE_KEYS, mine, native, gap_pct, within, strict=True):
        x = g[k]  # ... its timing still counts
        assert x["status"] == "reported" and "pass" not in x and "void" not in x
        assert (x["value"], x["native"]) == (list(lo), list(hi))
        assert x["range"] == max(lo) - min(lo) and x["native_range"] == max(hi) - min(hi)
        assert x["gap"] == x["median"] - x["native_median"]
        assert x["noise"] == max(x["range"], x["native_range"])
        assert x["gap_pct"] == pytest.approx(pct, abs=0.01) and x["within_noise"] is w
    assert g["G4.replay_warm.ttft_p50"]["pass"] and g["G4.replay_warm.req_per_s"]["pass"]
    rep = run_gate.native_report(v)["pegaflow.m:C"]
    for k, pct, lower in zip(NATIVE_KEYS, gap_pct, (True, False), strict=True):
        assert rep[k]["gap_pct"] == g[k]["gap_pct"]
        assert rep[k]["behind_pct"] == pytest.approx(pct if lower else -pct, abs=0.01)
    # A wrong-KV replay marks the comparison void (the arm fails on G2 / G4 vs baseline).
    results["pegaflow.m:C"]["replay"]["digest_self"] = None
    g = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")["pegaflow.m:C"]
    assert all("void" in g[k] and "pass" not in g[k] for k in NATIVE_KEYS)
    # No native replay: reported with the reason (the run is not official).
    del results["native"]
    results["pegaflow.m:C"]["replay"] = _replay(*mine)
    g = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")["pegaflow.m:C"]
    assert all(g[k]["reason"] and "gap" not in g[k] and "pass" not in g[k] for k in NATIVE_KEYS)
    # Native ran but has no timed warm pass: reported with the reason too.
    results["native"] = {"replay": _replay(0.3, 8.0, warm=())}
    g = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")["pegaflow.m:C"]
    assert all("timed warm pass" in g[k]["reason"] for k in NATIVE_KEYS)
    # The arm under test has no timed warm pass, native is normal.
    results["native"] = {"replay": _replay(*native)}
    results["pegaflow.m:C"]["replay"] = _replay(0.3, 8.0, warm=())
    g = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")["pegaflow.m:C"]
    assert all(g[k]["reason"].startswith("no timed warm pass") and "gap" not in g[k] for k in NATIVE_KEYS)  # fmt: skip


def test_g4_vs_native_never_fails_a_passing_arm(tmp_path):
    """An arm under test that passes every gated check still passes with its
    warm TTFT far behind native's, outside both pass-to-pass ranges
    (Qwen3.5 it2-like: 0.297 s vs 0.173 s): ``*_vs_native`` and its
    ``within_noise`` never reach the arm's ``pass``."""
    nat = _replay((0.1745, 0.1712), (13.1495, 13.1609))
    cap = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    cap["digest_self"] |= {"external_compared": 1}
    gsm = {"hit": nat["hit"], "load_errors": []}
    arm = {"replay": _replay((0.2996, 0.2952), (14.6105, 14.6551)), "cap": cap, "gsm8k": gsm}
    extra = {"baseline": {"replay": _replay(5.0, 1.0)}, "pegaflow.m:C": arm, "native": {"replay": nat}}  # fmt: skip
    g, _ = _resume_run(tmp_path, True, False, label="pegaflow.m:C", extra=extra)
    assert g["pass"] is True, {k: x for k, x in g.items() if isinstance(x, dict) and x.get("pass") is False}  # fmt: skip
    t = g["G4.replay_warm.ttft_p50_vs_native"]
    assert t["gap_pct"] > 70 and abs(t["gap"]) > 10 * t["noise"]
    assert all(g[k]["status"] == "reported" and g[k]["within_noise"] is False for k in NATIVE_KEYS)


def test_g4_vs_native_only_for_the_arm_under_test(tmp_path):
    """Reference and self-test arms are not judged against native."""
    rep = {"replay": _replay(0.5, 2.0)}
    results = {"baseline": {"replay": _replay(5.0, 1.0)}, "native": {"replay": _replay(0.1, 9.0)}}
    results |= {"old": rep, "lmcache": rep, "selftest-honest": rep, "pegaflow.m:C": rep}
    v = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")
    for label in ("native", "old", "lmcache", "selftest-honest"):
        assert not any(k in v[label] for k in NATIVE_KEYS), label
    assert all(v["pegaflow.m:C"][k]["status"] == "reported" for k in NATIVE_KEYS)


ARMS_N = ["plain", "baseline", "native", "pegaflow.m:C"]
_NAT = _replay(0.3, 8.0)


@pytest.mark.parametrize(
    "arms,wls,native,problem",
    [
        (ARMS_N, run_gate.WORKLOADS, _NAT, None),
        (["plain", "baseline", "pegaflow.m:C"], run_gate.WORKLOADS, None, "arm native missing"),
        (ARMS_N, ("gsm8k", "cap"), None, "workloads"),  # no replay: official_problems names it
        (ARMS_N, run_gate.WORKLOADS, None, "native replay stage missing"),
        (ARMS_N, run_gate.WORKLOADS, {"error": "RuntimeError: boom"}, "native replay stage missing"),
        (ARMS_N, run_gate.WORKLOADS, _replay(0.3, 8.0, ("warm2",)), "native replay timed warm passes"),
        # Ran, but served (almost) nothing: times like baseline, so no reference.
        (ARMS_N, run_gate.WORKLOADS, _replay(5.0, 1.0, eff=0.0), "native replay is not a working store"),
        (ARMS_N, run_gate.WORKLOADS, _NAT | {"load_errors": [{}]}, "native replay is not a working store"),
        (ARMS_N, run_gate.WORKLOADS, _NAT | {"hit": {}}, "native replay is not a working store"),
        (ARMS_N, run_gate.WORKLOADS, _replay(0.3, 8.0, eff=run_gate.NATIVE_MIN_EFFICIENCY), None),
    ],
    ids=["present", "arm-missing", "no-replay-workload", "no-replay", "replay-crashed",
         "warm-pass-missing", "native-degenerate", "native-load-errors", "native-no-hit",
         "native-at-floor"],
)  # fmt: skip
def test_official_needs_native_warm_replay(tmp_path, arms, wls, native, problem):
    """G4 vs native: a run without the native arm, whose native replay did
    not produce every timed warm pass, or whose native replay is not a
    working store (judged from its replay data, not its G0 / G2), is not
    official. ``main`` adds ``g4_native_problems`` unconditionally; the
    arm / workload guard is inside it, so this covers the wiring's logic."""
    import argparse

    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t"}))  # fmt: skip
    results = {"baseline": {"replay": _replay(5.0, 1.0)}}
    if native is not None:
        results["native"] = {"replay": native}
    probs = run_gate.official_problems(args, arms, list(wls), tmp_path, tmp_path)
    probs += run_gate.g4_native_problems(results, arms, list(wls))
    gap = [p for p in probs if "native" in p or p.startswith("workloads")]
    assert (gap == []) is (problem is None), probs
    assert problem is None or gap[0].startswith(problem), probs
    assert len([p for p in probs if p.startswith("native")]) <= 1, probs


def test_warmup_shares_no_prefix_with_workload():
    from rsi.gates import workloads

    w = workloads.warmup_prompt()
    assert w == workloads.warmup_prompt() and len(w) == workloads.WARMUP_TOKENS
    reqs = [{"pass": "_warmup", "keys": ["w0", "w1"]}, {"pass": "cold", "keys": ["a0", "a1"]}]
    assert oracle.warmup_problems(reqs) == []
    reqs.append({"pass": "warm", "keys": ["w0", "b1"]})
    assert oracle.warmup_problems(reqs) == [
        "1 workload requests share a prefix block with the warmup"
    ]


@pytest.mark.parametrize(
    "digested,skipped,ok",
    [((0, 1), {}, True), ((1,), {}, False), ((0, 1), {"0": "MambaSpec: not digested"}, False)],
    ids=["all-groups", "group-without-records", "group-skipped-by-probe"],
)
def test_every_transferred_group_must_be_digested(tmp_path, digested, skipped, ok):
    """A hybrid model whose state group (e.g. Mamba) the probe cannot digest
    fails G2: the digest checks would judge only the attention bytes."""
    import json

    probe = tmp_path / "probe"
    probe.mkdir()
    spec = _spec(CKPT, FULL) | {"probe": {"digest_skipped_groups": skipped}}
    (probe / "spec.json").write_text(json.dumps(spec))
    (probe / "digest-pp0-tp0.jsonl").write_text("".join(json.dumps({"group": g}) + "\n" for g in digested))  # fmt: skip
    v = run_gate.digest_groups(probe)
    assert v["pass"] is ok, v
    assert v["groups"] == [0, 1]


_RUN_LIKE = """
import asyncio, sys, time
from pathlib import Path
from rsi.gates import run_gate
port, out, where = sys.argv[1], Path(sys.argv[2]), sys.argv[3]


async def swallowing_pass():
    # A pass that converts its cancellation into an ordinary error, which a
    # reference arm's stage would record and move past.
    try:
        await asyncio.sleep(600)
    except asyncio.CancelledError:
        raise RuntimeError("pass failed")


def fake_stage(args, gate, arm, wl, out_dir, **kw):
    (out / f"started-{arm}").write_text("")
    cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
    with run_gate.launched(cmd, {}, out / f"{arm}.log", f"http://127.0.0.1:{port}/", 60) as p:
        (out / "pid").write_text(str(p.pid))
        try:
            if where == "asyncio":
                asyncio.run(swallowing_pass())
            else:
                time.sleep(600)
        except Exception as e:  # what a pass loop could do with it
            raise RuntimeError("stage failed") from e
    return {}


run_gate.run_arm_workload = fake_stage
run_gate.install_stop_handlers()
for arm in ("old", "native"):  # reference arms: an ordinary crash is recorded
    run_gate.run_stage(None, {}, arm, "replay", out)
(out / "finished").write_text("")
"""


@pytest.mark.parametrize("where", ["asyncio", "sleep"])
@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_stop_signal_aborts_the_run(tmp_path, sig, where):
    """``kill -<SIG> -<run_gate pgid>`` while a stage is inside asyncio.run or
    a sleep: the stage's process group is stopped, no later stage starts, and
    the gate exits 128 + signal. The gate is started with SIGINT ignored, as
    a background job of a non-interactive shell (``setsid nohup ... &``) is."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())

    def ignore_sigint():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    gate = subprocess.Popen(
        [sys.executable, "-c", _RUN_LIKE, port, str(tmp_path), where],
        env=env,
        start_new_session=True,
        preexec_fn=ignore_sigint,
    )
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    time.sleep(1)  # inside asyncio.run / sleep
    sleeper = int(pid_file.read_text())
    signum = getattr(signal, sig)
    os.killpg(gate.pid, signum)
    assert gate.wait(timeout=120) == 128 + signum
    with pytest.raises(ProcessLookupError):
        os.killpg(sleeper, 0)
    assert not (tmp_path / "started-native").exists()
    assert not (tmp_path / "finished").exists()


def test_g0_spec_ignores_module_paths(tmp_path):
    """M4: a fresh probe dump carries ``modules`` (host paths: the arm under
    test imports pegaflow from its overlay, the baseline from the venv; the
    committed dump has none). G0.spec must pass on the committed Qwen3-4B
    layout with differing ``modules``; G0.provenance judges them."""
    import json

    committed = run_gate.REPO / "docs/rsi/models/Qwen3-4B/spec.json"
    spec = json.loads(committed.read_text())
    for label, pf in (("baseline", "/venv/pegaflow/__init__.py"), ("arm", "/run/overlay/pegaflow/__init__.py")):  # fmt: skip
        probe = tmp_path / label / "gsm8k" / "probe"
        probe.mkdir(parents=True)
        dump = spec | {"modules": {"pegaflow": pf, "rsi.gates.rsi_probe": f"/{label}/rsi_probe.py"}}
        (probe / "spec.json").write_text(json.dumps(dump))
    v = run_gate.g0(tmp_path, "arm", committed)
    assert v["pass"], v
    # A real layout difference still fails.
    bad = json.loads((tmp_path / "arm/gsm8k/probe/spec.json").read_text())
    bad["kv_cache_config"]["kv_cache_layout"] = "LBHNC"
    (tmp_path / "arm/gsm8k/probe/spec.json").write_text(json.dumps(bad))
    assert not run_gate.g0(tmp_path, "arm", committed)["pass"]


# ------------------------------------------------- Mamba align state records


def _state_probe(tmp_path, warm_byte: str, preempt: bool):
    """Group 1 is a state group (probe ``state_groups``), group 0 full.

    - ``a`` (cold): a step ends on 2 blocks; the probe digested that state in
      its step (record ``episode`` 0). With ``preempt``, ``a`` is preempted
      after it and recomputes from 0 in episode 1, which crosses 2 blocks
      without ending a step there (no state computed).
    - ``b`` (warm): starts at 2 blocks (loaded state, ``warm_byte``) and digests
      that start state as episode 0; its full blocks at finish as usual."""
    import json

    a = _req("p", 3, 1, "cold") | {"req_id": "a", "digest": "full"}
    a["progress"] = [[1, 2 * BT], [2, 3 * BT + 1]]
    a["episodes"] = [[1, 0, 0]]
    if preempt:
        a["episodes"].append([5, 0, 0])
        a["progress"] += [[5, 3 * BT + 1]]
    b = _req("p", 3, 10, "warm") | {"req_id": "b", "digest": "full"}
    b["episodes"], b["progress"] = [[10, 0, 2 * BT]], [[10, 3 * BT + 1]]

    def rec(rid, gid, covered, byte, **kw):
        blk = {"key": f"p:{covered // BT - 1}", "covered": covered, "exact": [byte], "l1": [1.0]}
        return {"req_id": rid, "rank": "pp0-tp0", "group": gid, "layers": ["l0"], "blocks": [blk]} | kw  # fmt: skip

    digests = [
        rec("a", 1, 2 * BT, "S", episode=0),
        rec("b", 1, 2 * BT, warm_byte, episode=0),
        *(rec(rid, 0, c, f"F{c}") for rid in ("a", "b") for c in (BT, 2 * BT, 3 * BT)),
    ]
    probe = _probe(tmp_path, [a, b], digests)
    (probe / "spec.json").write_text(json.dumps({"probe": {"state_groups": [1]}}))
    return probe


def _zero_probe(tmp_path, cold_byte: str, dropped: bool):
    """Key ``p:1`` (2 blocks) computed by ``a`` in a preempted episode 0
    (digested at preemption, bytes ``cold_byte``) and loaded by ``b``
    (external, bytes of an all-zero page, i.e. a load that wrote nothing
    into a zeroed block). ``dropped``: the probe dropped a's block because
    vLLM zeroed it before the digest."""
    z = digest_compare.ZERO_DIGEST
    a = _req("p", 2, 1, "cold") | {"req_id": "a", "digest": "full"}
    a["episodes"], a["progress"] = [[1, 0, 0], [5, 0, 0]], [[1, 2 * BT + 1], [5, 2 * BT + 1]]
    a["computed"], a["preempt_digested"] = 2 * BT + 1, [0]
    if dropped:
        a["preempt_dropped"] = [[0, 0, 2 * BT, "zeroed"]]
    b = _req("p", 2, 10, "warm") | {"req_id": "b", "digest": "full"}
    b["episodes"], b["progress"] = [[10, 0, 2 * BT]], [[10, 2 * BT + 1]]

    def rec(rid, byte, **kw):
        blk = {"key": "p:1", "covered": 2 * BT, "exact": [byte], "l1": [0.0]}
        return {"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]} | kw  # fmt: skip

    digests = [rec("b", z), rec("a", "F")]  # a's last episode, digested at finish
    if not dropped:
        digests.append(rec("a", cold_byte, episode=0))
    return _probe(tmp_path, [a, b], digests)


@pytest.mark.parametrize(
    "cold_byte,dropped,matched,mismatched,uncertain",
    [
        ("F", False, 0, 1, 0),  # a real copy: the no-op load is caught
        (digest_compare.ZERO_DIGEST, False, 0, 0, 1),  # zero copy is no reference
        ("F", True, 0, 0, 1),  # dropped: an undigested computation
    ],
    ids=["real-copy", "zero-copy", "dropped"],
)
def test_zero_digest_is_never_a_reference(tmp_path, cold_byte, dropped, matched, mismatched, uncertain):  # fmt: skip
    """R2: vLLM zeroes new attention blocks, so a preempted block digested
    after vLLM zeroed it would make a load that wrote nothing ``matched``.
    An all-zero computed copy is never a reference, and a block the probe
    dropped counts as an undigested computation (``uncertain``, not
    ``mismatched``: the reference is not exhaustive)."""
    probe = _zero_probe(tmp_path, cold_byte, dropped)
    d = digest_compare.compare(probe, probe, invariant=False)
    g = d["groups"]["0"]
    assert (g["external_matched"], g["external_mismatched"], g["external_uncertain"]) == (matched, mismatched, uncertain)  # fmt: skip
    assert d["ref_zero_copies"] == (cold_byte == digest_compare.ZERO_DIGEST)


@pytest.mark.parametrize(
    "warm_byte,preempt,matched,mismatched",
    [("S", False, 1, 0), ("X", False, 0, 1), ("S", True, 1, 0), ("X", True, 0, 1)],
    ids=["match", "wrong-state", "match-preempted", "wrong-state-preempted"],
)
def test_state_records_tagged_by_their_episode(tmp_path, warm_byte, preempt, matched, mismatched):
    """A state digested in a preempted episode is that episode's computation
    (not the last episode's), and a state exists only where a step ended:
    episode 1 crossing 2 blocks mid-chunk computes no state, so the reference
    stays exhaustive and a wrong loaded state is mismatched, not uncertain."""
    from rsi.gates import digest_compare

    probe = _state_probe(tmp_path, warm_byte, preempt)
    d = digest_compare.compare(probe, probe, invariant=False)
    g = d["groups"]["1"]
    assert (g["external_matched"], g["external_mismatched"], g["external_uncertain"]) == (matched, mismatched, 0)  # fmt: skip
    assert g["computed_compared"] == 0  # one computed copy: nothing to compare it with
    full = d["groups"]["0"]
    assert full["external_matched"] == 2 and full["external_mismatched"] == 0
    v = run_gate._digest_verdict(d, 0.9)
    assert v["pass"] is (mismatched == 0)


# ------------------------------------------------------------ split invariance


def _split_probe(tmp_path):
    """Key ``p:2`` (block 3) computed four times: twice from start 0 (bytes
    A, A), once from start 2 blocks (bytes B: another split, e.g. MLA's
    separately attended cached context), once from start 2 blocks with bytes
    C (same split as B, different bytes: a real fault)."""
    reqs, digests = [], []
    for i, (start, byte) in enumerate([(0, "A"), (0, "A"), (2 * BT, "B"), (2 * BT, "C")]):
        rid = f"r{i}"
        r = _req("p", 3, 10 * i + 1, "cold") | {"req_id": rid, "digest": "full"}
        r["episodes"] = [[10 * i + 1, start, start]]
        r["progress"] = [[10 * i + 1, 3 * BT + 1]]
        reqs.append(r)
        blk = {"key": "p:2", "covered": 3 * BT, "exact": [byte], "l1": [1.0]}
        digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]})  # fmt: skip
    return _probe(tmp_path, reqs, digests)


@pytest.mark.parametrize(
    "split_inv,compared,exact,unmatched",
    [(True, 4, 2, 0), (False, 4, 2, 0)],
    ids=["split-invariant", "split-variant"],
)
def test_computed_exact_judges_same_split_only(tmp_path, split_inv, compared, exact, unmatched):
    """Split-invariant: every copy is a reference (B, C inexact). Split-variant:
    A/A judged against each other (exact), B/C against each other (both
    inexact: same split, different bytes still fails)."""
    from rsi.gates import digest_compare

    probe = _split_probe(tmp_path)
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=split_inv)
    assert (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"]) == (
        compared,
        exact,
        unmatched,
    )
    assert not run_gate._digest_verdict(d, None)["pass"]


def test_computed_other_split_only_is_reported_not_judged(tmp_path):
    """Drop C: B has only copies at another split. Split-variant: B is
    unmatched (reported, coverage < 1) and the verdict passes; split-invariant
    keeps the strict rule and fails."""
    import json

    from rsi.gates import digest_compare

    probe = _split_probe(tmp_path)
    for name in ("requests.jsonl", "digest-pp0-tp0.jsonl"):
        f = probe / name
        f.write_text("".join(x + "\n" for x in f.read_text().splitlines() if json.loads(x)["req_id"] != "r3"))  # fmt: skip
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=False)
    assert (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"]) == (2, 2, 1)
    v = run_gate._digest_verdict(d, None)
    assert v["pass"] and v["computed_coverage"] == pytest.approx(2 / 3)
    strict = digest_compare.compare(probe, probe, invariant=True, split_invariant=True)
    assert not run_gate._digest_verdict(strict, None)["pass"]
    # S7: the unmatched block counts as non-aligned unless its prompt length
    # is a multiple of the alignment (no recorded length: non-aligned).
    by = digest_compare.compare(probe, probe, True, False, align=16)["computed_by_pass"]
    assert by["cold"]["unjudged_nonaligned"] == 1
    f = probe / "requests.jsonl"
    f.write_text("".join(json.dumps(json.loads(x) | {"prompt_len": 64}) + "\n" for x in f.read_text().splitlines()))  # fmt: skip
    by = digest_compare.compare(probe, probe, True, False, align=16)["computed_by_pass"]
    assert by["cold"]["unjudged_nonaligned"] == 0  # reported, not absent


def _ctrl(differ: int, lp: int, n: int = 10, prewarmed: int = 10) -> dict:
    return {
        "n": n,
        "differ": [i < differ for i in range(n)],
        "logprob_differ": [i < lp for i in range(n)],
        "prewarmed": prewarmed,
        "ref_equals_cold": True,
    }


@pytest.mark.parametrize(
    "claim,ctrl,ok",
    [
        (True, _ctrl(0, 0), True),
        (False, _ctrl(0, 0), False),  # declared variant, measured invariant
        (True, _ctrl(0, 3), False),  # logprobs moved: not split-invariant
        (False, _ctrl(2, 5), True),
        (False, _ctrl(2, 5, prewarmed=5), False),  # pre-warm did not take
        (True, None, False),
        (True, _ctrl(0, 0) | {"ref_equals_cold": False}, False),  # N1: no stable basis
    ],
    ids=[
        "inv-ok",
        "false-claim",
        "logprob-only",
        "variant-ok",
        "vacuous",
        "missing",
        "ref-not-cold",
    ],  # fmt: skip
)
def test_split_control_establishes_gate_claim(claim, ctrl, ok):
    assert run_gate._split_control_verdict(ctrl, claim)["pass"] is ok


@pytest.mark.parametrize(
    "claim,ctrl,ok",
    [
        (True, _ctrl(0, 0), True),
        (False, _ctrl(0, 0), False),  # declared non-invariant, measured invariant
        (True, _ctrl(0, 1), False),  # one logprob moved: not batch-invariant
        (False, _ctrl(0, 1), True),  # GDN: logprobs move alone vs batched
        (True, _ctrl(1, 1), False),
        (False, _ctrl(1, 1, prewarmed=9), True),
        (False, _ctrl(1, 1, prewarmed=5), False),  # alone ran at another split
        (False, _ctrl(0, 0, n=0, prewarmed=0), False),  # nothing compared
        (False, None, False),
    ],
    ids=[
        "inv-ok",
        "false-variant-claim",
        "logprob-refutes",
        "variant-logprob",
        "output-refutes",
        "variant-ok",
        "vacuous",
        "empty",
        "missing",
    ],  # fmt: skip
)
def test_batch_control_establishes_gate_claim(claim, ctrl, ok):
    assert run_gate._batch_control_verdict(ctrl, claim)["pass"] is ok


def test_batch_control_compares_alone_with_the_batched_split_prefix(tmp_path):
    """``alone`` repeats the first prompts of ``split``: each is compared with
    the same prompt batched, and its pre-warm must reach the last full block."""
    import json

    d = tmp_path / run_gate.GATE_DIR / "plain" / "gsm8k"
    d.mkdir(parents=True)
    tok = {"logprobs": [-0.5], "prompt_tokens": 40, "cached": 32}
    (d / "split-outputs.json").write_text(json.dumps(["a", "b", "c"]))
    (d / "split-tokens.json").write_text(json.dumps([tok, tok, tok]))
    (d / "alone-outputs.json").write_text(json.dumps(["a", "x"]))
    (d / "alone-tokens.json").write_text(json.dumps([tok, tok | {"cached": 16}]))
    ctrl = run_gate.batch_control(tmp_path)
    assert (ctrl["n"], ctrl["differ"], ctrl["logprob_differ"], ctrl["prewarmed"]) == (
        2,
        [False, True],
        [False, False],
        1,
    )
    (d / "alone-outputs.json").write_text(json.dumps(["a", "b", "c", "d"]))
    assert run_gate.batch_control(tmp_path) is None  # more than split: not comparable


@pytest.mark.parametrize(
    "split_inv,cold,warm,vs_split,ok",
    [
        (True, 0, 0, (None, None), True),
        (True, 0, 1, (None, None), False),
        (False, 0, 60, (0, 3), True),  # differs only where the plain split stops a block early
        (False, 0, 60, (1, 0), False),  # one wrong completion fails
        (False, 1, 60, (0, 0), False),  # cold has the baseline's split: no slack
        (False, 0, 60, (None, None), False),  # no control: not judged, fails
    ],
    ids=[
        "inv-equal",
        "inv-differ",
        "variant-equal-split",
        "variant-one-wrong",
        "variant-cold",
        "variant-missing",
    ],  # fmt: skip
)
def test_outputs_exact_against_same_split(split_inv, cold, warm, vs_split, ok):
    v = run_gate._outputs_verdict({"cold": cold, "warm": warm}, True, split_inv, vs_split)
    assert v["pass"] is ok, v


def test_outputs_reported_not_gated_without_batch_invariance():
    """batch_invariant=false: no warm reference shares the warm pass's batch
    composition, so differing completions are reported, never a pass/fail."""
    v = run_gate._outputs_verdict({"cold": 3, "warm": 7}, False, True, (5, 1))
    assert (v["pass"], v["gated"], v["warm_differ"], v["warm_differ_vs_split"]) == (
        True,
        False,
        7,
        5,
    )


_RET = ["--mamba-cache-mode", "align", "--prefix-cache-retention-interval", "528"]


@pytest.mark.parametrize(
    "cmd,ok",
    [
        (["vllm", "serve", "m", "--seed", "42", *_RET], True),
        (["vllm", "serve", "m", *_RET, "--prefix-cache-retention-interval", "0"], False),
        (["vllm", "serve", "m", "--mamba-cache-mode", "align"], False),
        (None, False),  # not recorded
    ],
    ids=["applied", "overridden-later", "missing", "unrecorded"],
)
def test_gate_vllm_args_applied_to_every_stage(cmd, ok):
    gate = {"vllm_args": _RET}
    good = {"vllm_cmd": ["vllm", "serve", "m", *_RET]}
    results = {
        "plain": {"gsm8k": good},
        "baseline": {"gsm8k": good, "cap": {"error": "crashed"}},  # crash: G2 judges it
        "selftest-honest": {"gsm8k": {} if cmd is None else {"vllm_cmd": cmd}},
    }
    v = run_gate.gate_args_verdict(results, gate)
    assert v["pass"] is ok, v
    assert set(v["bad"]) <= {"selftest-honest/gsm8k"}


_DET = ["--compilation-config", '{"inductor_compile_config":{"deterministic":true}}']


@pytest.mark.parametrize(
    "gsm,other,ok",
    [
        ([*_RET, *_DET], _RET, True),
        (_RET, _RET, False),  # gsm8k stage without its compile flag
        ([*_RET, *_DET, "--compilation-config", "{}"], _RET, False),  # overridden later
    ],
    ids=["gsm8k-only", "missing-on-gsm8k", "overridden-later"],
)
def test_gsm8k_vllm_args_apply_to_gsm8k_stages_only(gsm, other, ok):
    """gate.json ``gsm8k_vllm_args``: required on every gsm8k stage, not on
    the others (replay/cap/preempt keep vLLM's default compile)."""
    gate = {"vllm_args": _RET, "gsm8k_vllm_args": _DET}
    results = {
        label: {"gsm8k": {"vllm_cmd": ["vllm", *gsm]}, "preempt": {"vllm_cmd": ["vllm", *other]}}
        for label in ("plain", "baseline")
    }
    v = run_gate.gate_args_verdict(results, gate)
    assert v["pass"] is ok, v
    assert all(k.endswith("/gsm8k") for k in v["bad"]), v


def test_split_differ_separates_aligned_prompts():
    ctrl = {"n": 4, "split_outputs": ["a", "b", "c", "d"], "aligned": [False, True, False, True]}
    assert run_gate.split_differ(ctrl, ["a", "x", "c", "d"]) == (0, 1)
    assert run_gate.split_differ(ctrl, ["x", "b", "c"]) == (1, 0)
    assert run_gate.split_differ(None, ["a"]) == (None, None)


def test_arm_writes_to_its_dir_cannot_change_a_verdict(tmp_path):
    """M9: the gate judges outputs from ``<run>/.gate/<label>/<wl>`` (never
    bound into an arm). A forged ``warm-outputs.json`` in the arm's own
    stage dir changes nothing; a symlink the arm leaves there (e.g. its
    digest file -> the baseline's) fails the stage."""
    import json

    for label, warm in (("baseline", ["a", "b"]), ("ut", ["a", "x"])):
        g = tmp_path / run_gate.GATE_DIR / label / "gsm8k"
        g.mkdir(parents=True)
        (g / "cold-outputs.json").write_text(json.dumps(["a", "b"]))
        (g / "warm-outputs.json").write_text(json.dumps(warm))
    own = tmp_path / "ut" / "gsm8k"
    own.mkdir(parents=True)
    (own / "warm-outputs.json").write_text(json.dumps(["a", "b"]))  # the forgery
    ref = run_gate._outputs(tmp_path, "baseline", "cold")
    assert run_gate._differ(ref, run_gate._outputs(tmp_path, "ut", "warm")) == 1
    assert run_gate.gate_dir(own) == tmp_path / ".gate" / "ut" / "gsm8k"
    assert run_gate.untrusted_problems(own) == []
    (own / "probe").mkdir()
    (own / "probe" / "digest-pp0-tp0.jsonl").symlink_to(tmp_path / "baseline-digest.jsonl")
    assert run_gate.untrusted_problems(own) == ["symlink probe/digest-pp0-tp0.jsonl"]


def test_missing_token_record_is_not_aligned(tmp_path):
    """S7: a request without a token record is never exempted from the
    exact warm-vs-split comparison."""
    import json

    d = tmp_path / run_gate.GATE_DIR / "plain" / "gsm8k"
    d.mkdir(parents=True)
    for name in ("cold", "split_ref", "split"):
        (d / f"{name}-outputs.json").write_text(json.dumps(["a", "b"]))
    (d / "split_ref-tokens.json").write_text(json.dumps([None, None]))
    (d / "split-tokens.json").write_text(json.dumps([None, {"prompt_tokens": 64, "cached": 48}]))
    ctrl = run_gate.split_control(tmp_path)
    assert ctrl["aligned"] == [False, True]
    assert run_gate.split_differ(ctrl, ["x", "b"]) == (1, 0)


def test_split_control_prewarm_must_reach_last_full_block(tmp_path):
    """S6: the pre-warm counts only if ``split`` cached the prompt up to its
    last full block (one block more than split_ref is not enough)."""
    import json

    d = tmp_path / run_gate.GATE_DIR / "plain" / "gsm8k"
    d.mkdir(parents=True)
    outs = ["o0", "o1", "o2"]
    for name in ("cold", "split_ref", "split"):
        (d / f"{name}-outputs.json").write_text(json.dumps(outs))
    ref = [{"logprobs": [0.0], "cached": 32, "prompt_tokens": n} for n in (100, 64, 100)]
    got = [{"logprobs": [0.0], "cached": c, "prompt_tokens": n} for c, n in ((96, 100), (48, 64), (48, 100))]  # fmt: skip
    (d / "split_ref-tokens.json").write_text(json.dumps(ref))
    (d / "split-tokens.json").write_text(json.dumps(got))
    ctrl = run_gate.split_control(tmp_path)
    assert ctrl["prewarmed"] == 2 and ctrl["aligned"] == [False, True, False]
    assert ctrl["ref_equals_cold"]


def _cov(by_pass: dict) -> dict:
    v = {"pass": True}
    run_gate._coverage_verdict(v, {"split_invariant": False, "computed_by_pass": by_pass}, ("cold", "warm"))  # fmt: skip
    return v


@pytest.mark.parametrize(
    "warm,ok",
    [
        ({"compared": 95, "unmatched_split": 5, "unjudged_nonaligned": 0}, True),
        (
            {"compared": 0, "unmatched_split": 2136, "unjudged_nonaligned": 0},
            False,
        ),  # the reviewer's vacuous honest arm
        (
            {"compared": 80, "unmatched_split": 10, "unreferenced": 10, "unjudged_nonaligned": 0},
            False,
        ),
        ({}, False),  # no warm computed block judged at all
        # S7: 95% coverage, but one unjudged block on a non-aligned prompt.
        ({"compared": 95, "unmatched_split": 5, "unjudged_nonaligned": 1}, False),
        ({"compared": 95, "unmatched_split": 5}, False),  # count not reported
    ],
    ids=["covered", "vacuous", "low", "missing", "nonaligned-unjudged", "nonaligned-unknown"],
)
def test_split_variant_needs_computed_coverage_per_pass(warm, ok):
    v = _cov({"cold": {"compared": 10}, "warm": warm})
    assert v["pass"] is ok, v
    assert _cov({"cold": {"compared": 10}, "warm": warm} | {"cold": {}})["pass"] is False


def test_split_variant_still_fails_wrong_loaded_bytes(tmp_path):
    """S2: a loaded (external) block with wrong bytes fails on a
    split-variant model too; only computed blocks are judged per split."""
    from rsi.gates import digest_compare

    reqs, digests = [], []
    for i, (name, start, byte) in enumerate([("cold", 0, "A"), ("warm", 2 * BT, "flip")]):
        r = _req("p", 2, 10 * i + 1, name) | {"req_id": f"r{i}", "digest": "full"}
        r["episodes"] = [[10 * i + 1, 0, start]]
        r["progress"] = [[10 * i + 1, 2 * BT + 1]]
        reqs.append(r)
        blk = {"key": "p:1", "covered": 2 * BT, "exact": [byte], "l1": [1.0]}
        digests.append({"req_id": f"r{i}", "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]})  # fmt: skip
    probe = _probe(tmp_path, reqs, digests)
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=False)
    assert d["external_mismatched"] == 1
    assert not run_gate._digest_verdict(d, 0.0)["pass"]


def _ctx_probe(tmp_path, name: str, rows: list[tuple[str, str, str]], seeds: tuple = ()):
    """Requests resuming at 2 blocks: each loads context block ``p:1``
    (bytes ``ctx``) and computes ``p:2`` (bytes ``out``). ``seeds``: bytes of
    cold requests computing ``p:1`` from scratch (its reference copies)."""
    reqs, digests = [], []
    for j, ctx in enumerate(seeds):
        rid = f"{name}s{j}"
        reqs.append(_req("p", 2, 100 + 10 * j, "cold", 50 + j) | {"req_id": rid, "digest": "full", "episodes": [[100 + 10 * j, 0, 0]]})  # fmt: skip
        blk = {"key": "p:1", "covered": 2 * BT, "exact": [ctx], "l1": [1.0]}
        digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]})  # fmt: skip
    for i, (label, ctx, out) in enumerate(rows):
        rid = f"{name}{i}"
        r = _req("p", 3, 10 * i + 1, label, i) | {"req_id": rid, "digest": "full"}
        r["episodes"] = [[10 * i + 1, 0, 2 * BT]]
        r["progress"] = [[10 * i + 1, 3 * BT + 1]]
        reqs.append(r)
        blocks = [
            {"key": "p:1", "covered": 2 * BT, "exact": [ctx], "l1": [1.0]},
            {"key": "p:2", "covered": 3 * BT, "exact": [out], "l1": [1.0]},
        ]
        digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": blocks})  # fmt: skip
    d = tmp_path / name
    d.mkdir()
    return _probe(d, reqs, digests)


@pytest.mark.parametrize(
    "ref_rows,cand_rows,compared,exact,unmatched",
    [
        # Same split and context bytes: judged, equal.
        ([("cold", "X", "A")], [("warm", "X", "A")], 1, 1, 0),
        # Same split and context, other bytes: judged, inexact (a fault).
        ([("cold", "X", "A")], [("warm", "X", "flip")], 1, 0, 0),
        # Same split, context computed elsewhere (other bytes): no
        # same-context copy, unjudged (the honest DSV2 lmcache case).
        ([("cold", "X", "A")], [("warm", "Y", "B")], 0, 0, 1),
    ],
    ids=["same-context", "same-context-wrong", "other-context"],
)
def test_split_key_includes_context_bytes(tmp_path, ref_rows, cand_rows, compared, exact, unmatched):  # fmt: skip
    """Split-variant: a computed block is judged only against copies computed
    from the same context bytes; the context itself is judged as loaded
    blocks (here ``p:1`` external: ``Y`` has no reference copy)."""
    from rsi.gates import digest_compare

    ref = _ctx_probe(tmp_path, "ref", ref_rows, seeds=(ref_rows[0][1],))
    cand = _ctx_probe(tmp_path, "cand", cand_rows)
    d = digest_compare.compare(ref, cand, invariant=True, split_invariant=False)
    got = (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"])
    assert got == (compared, exact, unmatched), d
    assert d["external_mismatched"] == (cand_rows[0][1] != ref_rows[0][1])
    if compared and not exact:
        assert not run_gate._digest_verdict(d, None)["pass"]


def test_reference_pools_several_probe_dirs(tmp_path):
    """The resume replay adds same-split copies to the baseline's: a block
    unjudged against the baseline alone is judged against both, and a wrong
    one is still inexact."""
    from rsi.gates import digest_compare

    base = _ctx_probe(tmp_path, "base", [("cold", "X", "A")], seeds=("X",))
    # The replay computed the context at the arm's split too (bytes Y).
    res = _ctx_probe(tmp_path, "res", [("resume", "Y", "B")], seeds=("Y",))
    for out, ok in (("B", True), ("flip", False)):
        cand = _ctx_probe(tmp_path, f"cand{out}", [("warm", "Y", out)])
        alone = digest_compare.compare(base, cand, True, False)
        assert alone["computed_unmatched_split"] == 1
        both = digest_compare.compare([base, res], cand, True, False)
        assert (both["computed_compared"], both["computed_exact"]) == (1, int(ok))
        assert both["external_mismatched"] == 0  # ``Y`` was computed by the replay
        assert run_gate._digest_verdict(both, None)["pass"] is ok


def _warm(conv, step: int, local: int, start: int, plen: int = 30) -> dict:
    return {
        "req_id": f"w{conv}",
        "pass": "warm",
        "conv": conv,
        "prompt_len": plen,
        "episodes": [[step, local, start]],
    }


def test_resume_plan_replays_the_arm_schedule():
    """Waves follow the arm's first-episode steps; the replay pre-warms the
    arm's resume point only where the connector loaded (start > local);
    other passes are ignored; a missing or bad prompt index fails closed."""
    reqs = [
        _warm(1, 5, 0, 16),  # first request: loaded [0, 16)
        _warm(0, 7, 16, 16),  # local only: no pre-warm
        _warm(2, 7, 16, 24, 32),  # local + loaded, resumes at 24
        {"pass": "cold", "conv": 0, "episodes": [[1, 0, 0]]},
    ]
    plan = run_gate.resume_plan(reqs, 3)
    assert plan == {
        "waves": [[[1, 16, 16, 30]], [[0, 0, 16, 30], [2, 24, 24, 32]]],
        "chains": {},
        "problems": [],
    }
    assert run_gate.resume_plan(reqs, 4)["problems"] == ["3 of 4 prompts have a warm request"]
    bad = run_gate.resume_plan([*reqs, _warm(None, 9, 0, 0)], 3)
    assert bad["problems"] == ["warm request wNone: prompt index None"]
    dup = run_gate.resume_plan([*reqs, _warm(2, 9, 0, 0)], 3)
    assert dup["problems"] == ["warm request w2: prompt index 2"]


def test_resume_check_and_differ_fail_closed():
    """Only a prompt whose replay resumed where the arm did is compared; an
    unreplayed one counts as differing unless its prompt is block-aligned."""
    plan = {"waves": [[[0, 16, 16, 30], [1, 16, 16, 32], [2, 24, 24, 30]]], "problems": []}
    reqs = [
        {"pass": "resume", "conv": 0, "prompt_len": 30, "episodes": [[1, 0, 16]]},
        {"pass": "resume", "conv": 1, "prompt_len": 32, "episodes": [[1, 0, 32]]},  # moved
        {"pass": "resume_prewarm", "conv": 2, "prompt_len": 24, "episodes": [[1, 0, 0]]},
    ]
    check = run_gate.resume_check(plan, reqs)
    assert (check["missing"], check["start_differs"], check["prompt_len_differs"]) == ([2], [1], [])  # fmt: skip
    ref = ["a", "b", None]
    # Prompt 1 (32 tokens, aligned) is exempt; prompt 2 (missing) differs.
    assert run_gate.resume_differ(check, plan, ref, ["a", "x", "c"], 16) == (1, 1)
    assert run_gate.resume_differ(check, plan, ref, ["x", "b", "c"], 16) == (2, 0)
    full = run_gate.resume_check(plan, reqs[:1] + [
        {"pass": "resume", "conv": 1, "prompt_len": 32, "episodes": [[1, 0, 16]]},
        {"pass": "resume", "conv": 2, "prompt_len": 30, "episodes": [[1, 16, 24]]},
    ])  # fmt: skip
    assert run_gate.resume_differ(full, plan, ["a", "b", "c"], ["a", "b", "c"], 16) == (0, 0)
    assert run_gate.resume_differ(full, plan | {"problems": ["x"]}, ["a"], ["a"], 16) == (None, None)  # fmt: skip
    v = run_gate._resume_verdict(None)
    assert not v["pass"]


@pytest.mark.parametrize(
    "ep1,tokens,chain,problem",
    [
        ([9, 24, 24], [30, 40], [[1, 0, 24, 40]], None),  # resumed from its own cached blocks
        ([9, 16, 24], [30, 40], [[1, 24, 24, 40]], None),  # connector loaded [16, 24)
        ([9, 24, 24], None, None, "episode lengths None"),  # record predates episode_tokens
        ([9, 24, 24], [30, 20], None, "episode lengths [30, 20]"),  # shorter than the prompt
    ],
    ids=["local-resume", "loaded-resume", "no-lengths", "bad-lengths"],
)
def test_resume_plan_chains_preempted_episodes(ep1, tokens, chain, problem):
    """S6: a warm request vLLM preempted is replayed at its first episode in
    the wave, then at every later episode (its start, pre-warm by the same
    rule, its length then); without usable lengths the plan fails closed."""
    pre = _warm(0, 5, 0, 16) | {"episodes": [[5, 0, 16], ep1]}
    if tokens is not None:
        pre["episode_tokens"] = tokens
    plan = run_gate.resume_plan([pre, _warm(1, 5, 0, 16)], 2)
    assert plan["waves"] == [[[0, 16, 16, 30], [1, 16, 16, 30]]]
    assert plan["chains"].get("0") == chain
    assert plan["problems"] == ([f"warm request w0: {problem}"] if problem else [])


@pytest.mark.parametrize(
    "ep_req,key",
    [
        ({"prompt_len": 40, "episodes": [[4, 24, 24]]}, None),
        ({"prompt_len": 40, "episodes": [[4, 32, 32]]}, "episode_start_differs"),
        ({"prompt_len": 41, "episodes": [[4, 24, 24]]}, "episode_prompt_len_differs"),
        (None, "episode_missing"),
    ],
    ids=["replayed", "start-differs", "length-differs", "missing"],
)
def test_resume_check_judges_each_later_episode(ep_req, key):
    """S6: a later episode replayed elsewhere than the arm's leaves its
    computed blocks unjudged and its completion counts as differing."""
    plan = {"waves": [[[0, 16, 16, 30]]], "chains": {"0": [[1, 0, 24, 40]]}, "problems": []}
    reqs = [{"pass": "resume", "conv": 0, "prompt_len": 30, "episodes": [[1, 0, 16]]}]
    if ep_req:
        reqs.append({"pass": run_gate.RESUME_EP_PASS, "conv": 0, "turn": 1} | ep_req)
    check = run_gate.resume_check(plan, reqs)
    assert check["episodes_planned"] == 1
    for k in ("episode_missing", "episode_start_differs", "episode_prompt_len_differs"):
        assert check[k] == ([[0, 1]] if k == key else []), k
    assert run_gate.resume_differ(check, plan, ["a"], ["a"], 16) == ((1, 0) if key else (0, 0))
    assert run_gate._resume_verdict({"digest_self": _digest_ok(), "resume_check": check})["pass"]


@pytest.mark.parametrize(
    "n,want",
    [(0, []), (1, [("a", [1])]), (3, [("a", [1]), ("bc", [2, 3])]), (2, None), (5, None)],
    ids=["none", "first", "chunk-edge", "straddles", "too-short"],
)
def test_resume_episode_prompt_cuts_on_chunk_edges(n, want):
    """S6: a later episode's prompt is the replay's own first ``n`` output
    tokens; the text kept with it must be exactly those tokens' chunks."""
    from rsi.gates import workloads

    pieces = [("a", [1]), ("bc", [2, 3]), ("d", [4])]
    assert workloads.cut_pieces(pieces, n) == want


def _blk_rec(rid: str, blocks: list[tuple[int, str]], prompt: str = "p", **kw) -> dict:
    """A digest record of group 0: (covered, bytes) per block."""
    blks = [{"key": f"{prompt}:{c // BT - 1}", "covered": c, "exact": [b], "l1": [1.0]} for c, b in blocks]  # fmt: skip
    return {"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": blks} | kw


def _ep_req(rid: str, label: str, conv: int, episodes: list, progress: list, n: int = 3, prompt: str = "p") -> dict:  # fmt: skip
    return _req(prompt, n, episodes[0][0], label, conv) | {
        "req_id": rid, "digest": "full", "episodes": episodes, "progress": progress,
    }  # fmt: skip


def test_pooled_reference_keeps_the_zero_and_state_rules(tmp_path):
    """M1: pooling probe dirs keeps hybrid R2 (an all-zero computed copy is
    never a reference, counted over every dir) and unions the state groups."""
    z = digest_compare.ZERO_DIGEST
    dirs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        dirs.append(_zero_probe(tmp_path / name, z, False))
    (dirs[1] / "spec.json").write_text(json.dumps({"probe": {"state_groups": [1]}}))
    ref = digest_compare.Reference(dirs)
    assert ref.zero_copies == 2 and ref.state_groups == {1}
    d = digest_compare.compare(dirs, dirs[0], invariant=False)
    assert d["external_matched"] == 0 and d["ref_zero_copies"] == 2


@pytest.mark.parametrize(
    "ep0_byte,exact", [("A3", 3), ("flip", 2)], ids=["honest", "wrong-preempted-block"]
)
def test_context_fingerprint_is_per_episode(tmp_path, ep0_byte, exact):
    """M1: a preempted episode's blocks (digested at preemption) are keyed by
    that episode's own start and context, not the last episode's: episode 0
    computed from 0 is judged against the baseline's cold copies, and a wrong
    byte there fails; episode 1 (resumed at 2 blocks) has no same-context
    copy and stays unjudged."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "cand").mkdir()
    ref = _probe(
        tmp_path / "ref",
        [_ep_req("a", "cold", 0, [[1, 0, 0]], [[1, 3 * BT + 1]])],
        [_blk_rec("a", [(BT, "A1"), (2 * BT, "A2"), (3 * BT, "A3")])],
    )
    b = _ep_req("b", "cold", 0, [[1, 0, 0], [5, 0, 2 * BT]], [[1, 3 * BT], [5, 3 * BT + 1]])
    b["preempt_digested"] = [0]
    cand = _probe(
        tmp_path / "cand",
        [b],
        [
            _blk_rec("b", [(BT, "A1"), (2 * BT, "A2"), (3 * BT, ep0_byte)], episode=0),
            _blk_rec("b", [(BT, "A1"), (2 * BT, "A2"), (3 * BT, "B3")]),  # finish: episode 1
        ],
    )
    d = digest_compare.compare(ref, cand, invariant=True, split_invariant=False)
    assert (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"]) == (3, exact, 1)  # fmt: skip
    assert d["external_matched"] == 2
    assert run_gate._digest_verdict(d, None)["pass"] is (exact == 3)


def _resume_run(tmp_path, split_inv: bool, resume: bool, wrong: bool = False, resume_ok: bool = True, label: str = "selftest-honest", extra: dict | None = None):  # fmt: skip
    """A gsm8k record of the baseline and arm ``label`` (an honest self-test
    arm by default); ``extra``: more stage fields per arm and workload,
    merged into the record. Prompt 1
    (``q``, 10 blocks) is computed from scratch in every pass: 10 judged warm
    blocks. Prompt 0 is 13 tokens (keys ``p:0..2``, not aligned to vLLM's
    16). Baseline
    cold computes ``A1 A2 A3``; the arm's warm request resumes at 2 blocks
    (loads ``p:0 p:1``) and computes ``p:2``: ``A3`` when split invariant,
    else ``B3`` (MLA's merge at the split). ``resume``: the baseline's
    replay at the arm's resume point (it computes ``B3`` there too).
    ``wrong``: the arm loaded an honest copy of ``p:1`` computed at another
    split (``S2``: baseline and arm both computed it from 1 block) and wrote
    wrong bytes ``W3`` over it."""
    run = tmp_path / "run"
    spec = _spec(FULL)
    committed = run / "spec.json"

    def probe(label: str, wl: str, reqs: list, digests: list) -> Path:
        d = run / label / wl / "probe"
        d.mkdir(parents=True)
        (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        (d / "digest-pp0-tp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in digests))
        (d / "spec.json").write_text(json.dumps(spec))
        return d

    full = [[1, 3 * BT + 1]]
    cold = [(BT, "A1"), (2 * BT, "A2"), (3 * BT, "A3")]
    other = [(BT, "A1"), (2 * BT, "S2"), (3 * BT, "S3")]  # computed from 1 block
    qn = 10
    qblocks = [((j + 1) * BT, f"Q{j}") for j in range(qn)]

    def q(rid: str, label: str, step: int) -> dict:
        return _ep_req(rid, label, 1, [[step, 0, 0]], [[step, qn * BT + 1]], qn, "q")

    base_reqs = [_ep_req("bc", "cold", 0, [[1, 0, 0]], full), q("bq", "cold", 1)]
    base_digs = [_blk_rec("bc", cold), _blk_rec("bq", qblocks, "q")]
    arm_reqs = [_ep_req("xc", "cold", 0, [[1, 0, 0]], full), q("xq", "cold", 1), q("xwq", "warm", 10)]  # fmt: skip
    arm_digs = [_blk_rec("xc", cold), _blk_rec("xq", qblocks, "q"), _blk_rec("xwq", qblocks, "q")]  # fmt: skip
    if wrong:
        base_reqs.append(_ep_req("bs", "split", 0, [[5, BT, BT]], [[5, 3 * BT + 1]]))
        base_digs.append(_blk_rec("bs", other))
        arm_reqs.append(_ep_req("xs", "cold", 2, [[3, BT, BT]], [[3, 3 * BT + 1]]))
        arm_digs.append(_blk_rec("xs", other))
    warm3 = "A3" if split_inv else "W3" if wrong else "B3"
    arm_reqs.append(_ep_req("xw", "warm", 0, [[10, 0, 2 * BT]], [[10, 3 * BT + 1]]))
    arm_digs.append(_blk_rec("xw", [(BT, "A1"), (2 * BT, "S2" if wrong else "A2"), (3 * BT, warm3)]))  # fmt: skip
    bp = probe("baseline", "gsm8k", base_reqs, base_digs)
    ap = probe(label, "gsm8k", arm_reqs, arm_digs)
    committed.write_text(json.dumps(spec))
    acc = {p: {"accuracy": 1.0, "seconds": 1.0} for p in ("cold", "warm")}
    results: dict = {
        "baseline": {
            "gsm8k": {
                "invariant": True,
                "passes": acc,
                "digest_self": digest_compare.compare(bp, bp, True, split_inv),
            }
        },  # fmt: skip
        label: {
            "gsm8k": {"passes": acc, "digest_self": digest_compare.compare(ap, ap, True, split_inv)}
        },  # fmt: skip
    }
    for lab, per in (extra or {}).items():
        results[lab] = results.get(lab, {}) | {wl: results.get(lab, {}).get(wl, {}) | r for wl, r in per.items()}  # fmt: skip
    two = ["ans", "ansq"]
    outs = {("baseline", "gsm8k", "cold"): two, (label, "gsm8k", "cold"): two, (label, "gsm8k", "warm"): two}  # fmt: skip
    if resume:
        rl = f"{run_gate.RESUME_PREFIX}{label}"
        pre = _ep_req("rp", "resume_prewarm", 0, [[2, 0, 0]], [[2, 2 * BT]], n=2)
        pre["prompt_len"] = pre["computed"] = 2 * BT
        rr = _ep_req("rr", run_gate.RESUME_PASS, 0, [[3, 2 * BT, 2 * BT]], [[3, 3 * BT + 1]])
        rq = q("rq", run_gate.RESUME_PASS, 3)
        rp = probe("baseline", rl, [pre, rr, rq], [_blk_rec("rp", cold[:2]), _blk_rec("rr", cold[:2] + [(3 * BT, "B3")]), _blk_rec("rq", qblocks, "q")])  # fmt: skip
        plan = run_gate.resume_plan(oracle.load_requests(ap / "requests.jsonl"), 2)
        ds = digest_compare.compare(rp, rp, True, split_inv)
        results["baseline"][rl] = {
            "resume_plan": plan,
            "resume_check": run_gate.resume_check(
                plan, oracle.load_requests(rp / "requests.jsonl")
            ),
            "digest_self": ds if resume_ok else ds | {"local_mismatched": 1},
        }
        outs[("baseline", rl, run_gate.RESUME_PASS)] = two
    for (lab, wl, name), o in outs.items():
        f = run / run_gate.GATE_DIR / lab / wl / f"{name}-outputs.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(o))
    gate = {"batch_invariant": True, "split_invariant": split_inv}
    return run_gate.verdicts(results, run, committed, gate)[label], gate


@pytest.mark.parametrize(
    "split_inv,resume,wrong,resume_ok,ok,failing",
    [
        (False, False, False, True, False, "G2.resume_reference"),
        (False, True, False, True, True, None),
        (False, True, False, False, False, "G2.resume_reference"),  # S5: required alone
        (True, False, False, True, True, None),
        (False, True, True, True, False, "G2.digest_vs_baseline.gsm8k"),
    ],
    ids=[
        "split-variant-no-resume",
        "split-variant-resume",
        "resume-reference-fails",
        "split-invariant",
        "wrong-over-other-split",
    ],  # fmt: skip
)
def test_verdicts_judge_warm_kv_at_the_arm_resume_points(tmp_path, split_inv, resume, wrong, resume_ok, ok, failing):  # fmt: skip
    """S4/S5 at verdict level. Split-variant: without a ``resume-<label>``
    stage the arm fails G2.resume_reference; with one, warm computed KV is
    judged against the pooled baseline + replay copies and warm outputs
    against the replay, and the honest self-test needs G2.resume_reference.
    Split-invariant: no resume check and no resume stage. A wrong computed
    block over an honest other-split context has no same-context copy
    (``unmatched_split``): it fails through S7, not as mismatched."""
    g, gate = _resume_run(tmp_path, split_inv, resume, wrong, resume_ok)
    assert run_gate.resume_stage_needed(gate, ["baseline", "selftest:honest"]) is (not split_inv)
    assert ("G2.resume_reference" in g) is (not split_inv)
    assert g["pass"] is ok, {k: x for k, x in g.items() if isinstance(x, dict) and not x.get("pass", True)}  # fmt: skip
    if failing:
        assert not g[failing]["pass"]
        others = [k for k, x in g.items() if isinstance(x, dict) and k.startswith("G2.") and x.get("pass") is False]  # fmt: skip
        if resume:
            assert others == [failing], others
    d = g["G2.digest_vs_baseline.gsm8k"]
    if resume and not split_inv:
        assert g["G2.gsm8k_outputs"]["warm_reference"].startswith(run_gate.RESUME_PREFIX)
        warm = d["computed_coverage_by_pass"]["warm"]
        # The wrong case clears coverage (10 of 11 judged): only S7 catches it.
        assert warm["coverage"] >= run_gate.MIN_COMPUTED_COVERAGE
        assert warm["unjudged_nonaligned"] == (1 if wrong else 0)
        assert d["external_mismatched"] == d["local_mismatched"] == 0
    if split_inv:
        assert d["computed_exact"] == d["computed_compared"] > 0


def test_workload_sessions_never_reuse_connections():
    """vLLM's uvicorn closes idle keep-alive connections after 5 s; a pooled
    client reuses a closing socket and the pass dies with "Server
    disconnected" (the gate then stops vLLM: looks like an external SIGTERM)."""
    import asyncio

    from rsi.gates import workloads

    async def force_close() -> bool:
        async with workloads.session() as s:
            return s.connector.force_close

    assert asyncio.run(force_close())
    # Every workload HTTP client goes through session(): no other
    # ClientSession / TCPConnector anywhere in the gate code.
    import re
    from pathlib import Path

    src = Path(workloads.__file__).parent
    for f in sorted(src.rglob("*.py")):
        if f.name.startswith("test_"):
            continue
        text = f.read_text()
        if f.name == "workloads.py":
            text = text.replace(text[text.index("def session(") : text.index("# ---", text.index("def session("))], "")  # fmt: skip
        assert not re.search(r"ClientSession\(|TCPConnector\(", text), f


def _preempt_probe(tmp_path, digested: bool, warm_byte: str):
    """``a`` (cold) computes 3 full blocks in episode 0, is preempted and
    recomputes them in episode 1 with other rounding (``F*'``). With
    ``digested`` the probe digested episode 0's blocks at preemption
    (``preempt_digested``, records ``episode`` 0). ``b`` (warm, resumed after
    a preemption: episode 1) loads 2 blocks (``warm_byte`` at block 1):
    the copy saved from episode 0."""
    a = _req("p", 3, 1, "cold") | {"req_id": "a", "digest": "full"}
    a["episodes"], a["progress"] = [[1, 0, 0], [5, 0, 0]], [[1, 3 * BT], [5, 3 * BT + 1]]
    a["preempt_digested"] = [0] if digested else []
    b = _req("p", 3, 10, "warm") | {"req_id": "b", "digest": "full"}
    b["episodes"], b["progress"] = [[10, 0, 0], [12, 0, 2 * BT]], [[10, BT], [12, 3 * BT + 1]]
    b["preempt_digested"] = [0]

    def rec(rid, c, byte, **kw):
        blk = {"key": f"p:{c // BT - 1}", "covered": c, "exact": [byte], "l1": [1.0]}
        return {"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]} | kw  # fmt: skip

    digests = [rec("a", c, f"F{c}'") for c in (BT, 2 * BT, 3 * BT)]
    if digested:
        digests += [rec("a", c, f"F{c}", episode=0) for c in (BT, 2 * BT, 3 * BT)]
    digests += [rec("b", BT, f"F{BT}"), rec("b", 2 * BT, warm_byte), rec("b", 3 * BT, "Fb")]
    return _probe(tmp_path, [a, b], digests)


@pytest.mark.parametrize(
    "digested,warm_byte,matched,mismatched,uncertain",
    [
        (True, f"F{2 * BT}", 2, 0, 0),
        (True, "X", 1, 1, 0),
        (False, f"F{2 * BT}", 0, 0, 2),  # the saved copy was never digested
        (False, "X", 0, 0, 2),
    ],
    ids=["digested-match", "digested-wrong", "undigested-match", "undigested-wrong"],
)
def test_preempted_episode_blocks_complete_the_reference(
    tmp_path, digested, warm_byte, matched, mismatched, uncertain
):
    """Without batch invariance only an exhaustive reference judges a load:
    the blocks a preempted episode computed are digested at preemption, so a
    copy saved from that episode is a reference and a wrong load mismatches."""
    from rsi.gates import digest_compare

    d = digest_compare.compare(
        *[_preempt_probe(tmp_path, digested, warm_byte)] * 2, invariant=False
    )
    g = d["groups"]["0"]
    assert (g["external_matched"], g["external_mismatched"], g["external_uncertain"]) == (matched, mismatched, uncertain)  # fmt: skip
    assert d["preempted_requests"] == 2
    after = d["after_preemption"]
    assert (after["external_compared"], after["external_uncertain"]) == (2, uncertain)


def _pre(preempted: int, resumed_loads: int, mismatched: int = 0) -> dict:
    d = {k: 0 for k in run_gate.digest_compare._group() if k != "first_bad"}
    d |= {"local_compared": 10, "local_matched": 10 - mismatched, "local_mismatched": mismatched}
    d |= {"invariant": False, "groups": {}, "preempted_requests": preempted}
    d["after_preemption"] = {"local_compared": resumed_loads, "external_compared": 0}
    return {"digest_self": d, "gpu_blocks": 40}


@pytest.mark.parametrize(
    "r,ok",
    [
        (_pre(4, 3), True),
        (_pre(0, 0), False),  # vLLM never preempted: vacuous
        (_pre(4, 0), False),  # preempted, but no resumed load was judged
        (_pre(4, 3, mismatched=1), False),
        (None, False),
        ({"error": "crashed"}, False),
    ],
    ids=["ok", "no-preemption", "no-resumed-load", "mismatch", "missing", "crashed"],
)
def test_preempt_verdict_needs_judged_loads_after_preemption(r, ok):
    assert run_gate._preempt_verdict(r, None)["pass"] is ok


def _alone_dirs(tmp_path, plain_ref, base_ref, alone, cached=(48, 48)):
    """``alone_ref`` (plain, baseline) and ``alone`` (plain, pre-warmed) passes."""
    import json

    for label, passes in (
        ("plain", {"alone_ref": plain_ref, "alone": alone, "cold": ["c0", "c1"], "split": alone}),
        ("baseline", {"alone_ref": base_ref, "cold": ["c0", "cX"]}),
    ):
        d = tmp_path / run_gate.GATE_DIR / label / "gsm8k"
        d.mkdir(parents=True, exist_ok=True)
        for name, outs in passes.items():
            (d / f"{name}-outputs.json").write_text(json.dumps(outs))
            c = cached if name in ("alone", "split") else (32, 32)
            toks = [
                {"logprobs": [len(o)], "cached": n, "prompt_tokens": 50}
                for o, n in zip(outs, c, strict=True)
            ]
            (d / f"{name}-tokens.json").write_text(json.dumps(toks))


@pytest.mark.parametrize(
    "plain_ref,base_ref,alone,plain_ok,split_measured",
    [
        (["a", "b"], ["a", "b"], ["a", "b"], True, True),
        (["a", "b"], ["a", "x"], ["a", "b"], False, True),  # the probe changed an output
        (["a", "b"], ["a", "b"], ["a", "bb"], True, False),  # the split changed one
    ],
    ids=["equal", "probe-changes", "split-changes"],
)
def test_non_invariant_controls_compare_one_composition(
    tmp_path, plain_ref, base_ref, alone, plain_ok, split_measured
):
    """batch_invariant=false: batched cold passes differ by composition, so
    plain vs baseline and the split control compare one-at-a-time passes."""
    _alone_dirs(tmp_path, plain_ref, base_ref, alone)
    ref_cold = ["c0", "cX"]  # batched: differs from plain cold, reported only
    v = run_gate._plain_equals_baseline(tmp_path, ref_cold, batch_invariant=False)
    assert (v["pass"], v["cold_differ"]) == (plain_ok, 1)
    ctrl = run_gate.split_control(tmp_path, batch_invariant=False)
    assert ctrl["pair"] == ["alone_ref", "alone"] and ctrl["prewarmed"] == 2
    assert ctrl["ref_equals_cold"] is plain_ok
    sv = run_gate._split_control_verdict(ctrl, True)
    assert sv["measured_invariant"] is split_measured
    assert "batched" in sv  # the batched pair stays reported (None: passes missing)
    assert (
        run_gate._plain_equals_baseline(tmp_path, ref_cold, batch_invariant=True)["pass"] is False
    )


# ------------------------------------------- hybrid gate review fixes (M2, S1-S6)


@pytest.mark.parametrize(
    "tail,ok",
    [
        ([], True),
        (["--prefix-cache-retention-interval=0"], False),
        (["--prefix_cache_retention_interval", "0"], False),
        (["--prefix-cache-ret", "0"], False),  # argparse abbreviation
        (["--mamba_cache_mode=all"], False),
        (["--prefix_cache_retention_interval=528"], True),  # same value, other spelling
        (["-cc", "{}"], False),
        (["--compilation-config={}"], False),
        (["--compilation-config.inductor_compile_config.deterministic", "false"], False),
        (["--compilation_config", _DET[1]], True),
    ],
    ids=[
        "as-is", "eq-override", "underscore-override", "abbrev-override", "underscore-eq",
        "same-value-respelled", "short-alias", "eq-compile", "dotted-compile", "underscore-compile",
    ],
)  # fmt: skip
def test_gate_args_read_every_vllm_spelling(tail, ok):
    """S4: a later flag overrides gate.json's in vLLM's parser whatever its
    spelling (``=``, ``_``, abbreviation, ``-cc``, dotted keys appended last);
    G0.vllm_args must see it."""
    gate = {"vllm_args": _RET, "gsm8k_vllm_args": _DET}
    results = {"baseline": {"gsm8k": {"vllm_cmd": ["vllm", "serve", "m", *_RET, *_DET, *tail]}}}
    assert run_gate.gate_args_verdict(results, gate)["pass"] is ok


@pytest.mark.parametrize(
    "edit,ok",
    [
        (None, True),
        ("drop-deterministic", False),
        ("no-compilation-block", True),  # a record from before the dump: not judged
    ],
    ids=["pinned", "compile-differs", "record-predates"],
)
def test_g0_spec_pins_the_compile_config(tmp_path, edit, ok):
    """S4: the effective compilation config is in the spec dump and G0.spec
    holds it to the committed one (Qwen3.5-4B gsm8k: deterministic Inductor)."""
    import json

    committed = run_gate.REPO / "docs/rsi/models/Qwen3.5-4B/spec.json"
    spec = json.loads(committed.read_text())
    assert spec["compilation"]["inductor_compile_config"]["deterministic"] is True
    if edit == "drop-deterministic":
        spec["compilation"]["inductor_compile_config"].pop("deterministic")
    elif edit == "no-compilation-block":
        # A schema-1 dump (before 9204863): retention, no compile/runner/schema.
        for k in ("compilation", "model_runner"):
            spec.pop(k)
        spec["probe"].pop("schema")
    for label in ("baseline", "arm"):
        probe = tmp_path / label / "gsm8k" / "probe"
        probe.mkdir(parents=True)
        (probe / "spec.json").write_text(json.dumps(spec))
    assert run_gate.g0(tmp_path, "arm", committed)["pass"] is ok


def _schema0(spec: dict) -> dict:
    """The committed spec as a probe before e620462 wrote it: no retention,
    compilation, runner or schema fields."""
    import copy

    s = copy.deepcopy(spec)
    for path in (*run_gate.SPEC_FIELDS, ("probe", "schema")):
        run_gate._pop(s, path)
    return s


@pytest.mark.parametrize(
    "model,edit,ok,na",
    [
        ("Qwen3-4B", "schema0", True, 4),  # qwen3-4b-p4-it5-nonreg's dump shape
        ("Qwen3-4B", "schema0-retention-differs", False, 2),  # a field it had, differing
        ("Qwen3-4B", "schema0-layout-differs", False, 4),
        ("Qwen3-4B", "schema2-retention-missing", False, 1),  # its rev knew retention
        ("Qwen3-4B", "current", True, 0),
        ("Qwen3-4B", "current-runner-missing", False, 0),
        ("Qwen3-4B", "current-runner-differs", False, 0),
        ("Qwen3.5-4B", "schema0", True, 4),
    ],
    ids=[
        "older-record",
        "older-record-field-differs",
        "older-record-layout-differs",
        "known-field-missing",
        "current",
        "current-field-missing",
        "current-field-differs",
        "older-hybrid-record",
    ],  # fmt: skip
)
def test_g0_spec_fields_unknown_to_the_record_are_na(tmp_path, model, edit, ok, na):
    """R1: a spec field missing from a recorded dump is ``na`` only when the
    record's probe schema did not know it (recorded, else the newest known
    field present); a known field missing, or any field present with another
    value, fails."""
    import json

    committed = run_gate.REPO / f"docs/rsi/models/{model}/spec.json"
    spec = json.loads(committed.read_text())
    if edit.startswith("schema0"):
        spec = _schema0(spec)
        if edit == "schema0-retention-differs":
            spec["cache"]["prefix_cache_retention_interval"] = 64
            spec["kv_cache_config"]["prefix_cache_retention_interval"] = 64
        elif edit == "schema0-layout-differs":
            spec["kv_cache_config"]["kv_cache_layout"] = "LBHNC"
    elif edit == "schema2-retention-missing":
        spec = _schema0(spec)
        spec["compilation"] = json.loads(committed.read_text())["compilation"]
    elif edit == "current-runner-missing":
        spec.pop("model_runner")
    elif edit == "current-runner-differs":
        spec["model_runner"] = "V1"
    for label in ("baseline", "arm"):
        probe = tmp_path / label / "gsm8k" / "probe"
        probe.mkdir(parents=True)
        (probe / "spec.json").write_text(json.dumps(spec))
    v = run_gate.g0(tmp_path, "arm", committed)
    assert v["pass"] is ok, v
    assert len(v.get("na_fields", [])) == na, v


def test_seeded_autotune_holds_only_tile_configs(tmp_path):
    """N2: Triton's autotune format can name IR to compile instead of the
    kernel (``ir_override``) or a ``pre_hook``; a seeded file must hold
    plain tile configs only, else seeding refuses."""
    import json

    good = {"key": [1], "configs_timings": [[{"kwargs": {"BK": 32}, "num_warps": 4, "num_ctas": 1, "num_stages": 2, "maxnreg": None, "pre_hook": None, "ir_override": None}, [0.1, 0.2, 0.3]]]}  # fmt: skip
    assert run_gate.autotune_file_problem(good) is None
    cfg = good["configs_timings"][0][0]
    for bad in (
        {"ir_override": "/tmp/k.ttgir"},
        {"pre_hook": "f"},
        {"extra": 1},
        {"kwargs": {"BK": "32"}},
        {"num_warps": "4"},
    ):
        assert run_gate.autotune_file_problem({"configs_timings": [[cfg | bad, [0.1]]]}), bad
    assert run_gate.autotune_file_problem({"configs_timings": []})
    caches = tmp_path / ".compile-cache"
    src = caches / "baseline" / "triton" / "H1"
    src.mkdir(parents=True)
    bad_file = good | {"configs_timings": [[cfg | {"ir_override": "/x.ttir"}, [0.1]]]}
    (src / "k.autotune.json").write_text(json.dumps(bad_file))
    (caches / "plain").mkdir()
    with pytest.raises(ValueError, match="ir_override"):
        run_gate.seed_autotune(caches / "plain")


def _groups(**per: dict) -> dict:
    return {"groups": {g: {"local_mismatched": 0, "external_mismatched": 0, "local_unreferenced": 0, "external_unreferenced": 0} | v for g, v in per.items()}}  # fmt: skip


_BAD = {"external_mismatched": 3}
_ALL4 = {"0": _BAD, "1": _BAD, "2": _BAD, "3": _BAD}


@pytest.mark.parametrize(
    "d_self,d_base,targets,sabotage,ok",
    [
        (_groups(**_ALL4), _groups(**_ALL4), [0, 1, 2, 3], {"0": 1, "1": 1, "2": 1, "3": 1}, True),
        # offbyone before the fix: only full attention wrong
        (_groups(**{"3": _BAD}), _groups(**{"3": _BAD}), [0, 1, 2, 3], {"3": 1}, False),
        (
            _groups(**_ALL4),
            _groups(**{"3": _BAD}),
            [0, 1, 2, 3],
            {"0": 1, "1": 1, "2": 1, "3": 1},
            False,
        ),  # fmt: skip
        (
            _groups(**_ALL4),
            _groups(**_ALL4),
            [0, 1, 2, 3],
            {"3": 1},
            False,
        ),  # claims no state sabotage
        (
            _groups(**{"0": {"local_unreferenced": 2}}),
            _groups(**{"0": {"external_unreferenced": 2}}),
            [0],
            {"0": 2},
            True,
        ),  # liar  # fmt: skip
        (_groups(**{"0": _BAD}), _groups(**{"0": _BAD}), [0], None, True),  # record predates counts
        (_groups(), _groups(), [], {}, False),
        # R3: 3 bad of 4 sabotaged in one check: a sabotaged load got through
        (_groups(**{"0": _BAD}), _groups(**{"0": _BAD}), [0], {"0": 4}, False),
        # ... unless that check could not judge it (external uncertain)
        (
            _groups(**{"0": _BAD | {"external_uncertain": 1}}),
            _groups(**{"0": _BAD | {"external_uncertain": 1}}),
            [0],
            {"0": 4},
            True,
        ),  # fmt: skip
        # a local uncertain is no sabotaged load: it does not excuse one
        (
            _groups(**{"0": _BAD | {"local_uncertain": 1}}),
            _groups(**{"0": _BAD | {"local_uncertain": 1}}),
            [0],
            {"0": 4},
            False,
        ),  # fmt: skip
        (_groups(**{"0": _BAD}), _groups(**{"0": _BAD}), [0], {"0": 3}, True),
    ],
    ids=[
        "all-groups",
        "full-only",
        "one-check-only",
        "no-sabotage",
        "unreferenced",
        "predates",
        "nothing",
        "fewer-bad-than-sabotaged",
        "uncertain-excused",
        "local-uncertain-not-excused",
        "bad-equals-sabotaged",
    ],  # fmt: skip
)
def test_negative_arm_caught_in_every_group(d_self, d_base, targets, sabotage, ok):
    """M2/R3: a negative self-test arm counts as caught only if each group it
    corrupts, judged on its own, has in both gsm8k digest checks at least as
    many loads equal to no reference as the connector sabotaged there (less
    the external loads that check left uncertain), and a sabotage at all."""
    v = run_gate.caught_per_group(d_self, d_base, targets, sabotage)
    assert v["pass"] is ok, v
    assert set(v["groups"]) == {str(t) for t in targets}


def test_sabotage_counts_sum_per_group(tmp_path):
    import json

    d = tmp_path / "selftest"
    d.mkdir()
    (d / "sabotage-1.jsonl").write_text(
        "".join(
            json.dumps({"mode": "offbyone", "group": g, "n": n}) + "\n"
            for g, n in ((0, 1), (3, 2), (0, 2))
        )  # fmt: skip
    )
    (d / "sabotage-2.jsonl").write_text(json.dumps({"mode": "offbyone", "group": 3, "n": 1}) + "\n")
    assert run_gate.sabotage_counts(d) == {"0": 3, "3": 3}
    assert run_gate.sabotage_counts(tmp_path / "missing") == {}


def _acc(**p: float) -> dict:
    return {"passes": {k: {"accuracy": v} for k, v in p.items()}}


@pytest.mark.parametrize(
    "arm,base,inv,ok",
    [
        (_acc(cold=0.70, warm=0.70), _acc(cold=0.70), True, True),
        (_acc(cold=0.70, warm=0.68), _acc(cold=0.70), True, False),
        # not invariant: batched warm moves with load timing (reported); alone_ref gates
        (
            _acc(cold=0.70, warm=0.68, alone_ref=0.71875),
            _acc(cold=0.70, alone_ref=0.71875),
            False,
            True,
        ),  # fmt: skip
        (
            _acc(cold=0.70, warm=0.70, alone_ref=0.70),
            _acc(cold=0.70, alone_ref=0.71875),
            False,
            False,
        ),  # fmt: skip
        (_acc(cold=0.70, warm=0.70), _acc(cold=0.70, alone_ref=0.71875), False, False),  # missing
    ],
    ids=["inv-equal", "inv-off", "alone-equal", "alone-off", "alone-missing"],
)
def test_gsm8k_accuracy_gated_at_one_batch_composition(arm, base, inv, ok):
    """S2: without batch invariance every arm (honest self-test included) is
    held to ACC_TOL on its one-at-a-time ``alone_ref`` pass vs the
    baseline's; batched deltas are reported."""
    v = run_gate.gsm8k_accuracy_verdict(arm, base, inv)
    assert v["pass"] is ok, v
    assert ("batched_delta_pp" in v) is (not inv)


def test_new_baseline_checks_are_na_for_older_records(tmp_path):
    """S1: a record is judged by the gate rev it ran under: a run made before
    G2.batch_control (no ``alone`` pass) and G0.vllm_args (no ``vllm_cmd``)
    is ``na`` on them; a current record missing them fails."""
    gsm = {"passes": {p: {"accuracy": 0.5} for p in ("cold", "warm", "split_ref", "split")}}
    old = {"plain": {"gsm8k": gsm}, "baseline": {"gsm8k": {"passes": {"cold": {}, "warm": {}}}}}
    v = run_gate.verdicts(old, tmp_path, tmp_path / "spec.json", {"batch_invariant": True})
    for k in ("G2.batch_control", "G0.vllm_args"):
        assert v["baseline"][k]["pass"] and v["baseline"][k]["na"], v["baseline"][k]
    new = {
        "plain": {"gsm8k": gsm | {"vllm_cmd": ["vllm"], "passes": gsm["passes"] | {"alone": {}}}},
        "baseline": {"gsm8k": {"vllm_cmd": ["vllm"], "passes": {"cold": {}, "warm": {}}}},
    }
    v = run_gate.verdicts(new, tmp_path, tmp_path / "spec.json", {"batch_invariant": True})
    assert not v["baseline"]["G2.batch_control"]["pass"]  # alone ran, outputs missing
    assert v["baseline"]["G0.vllm_args"]["pass"]  # no gate args: nothing to miss


@pytest.mark.parametrize(
    "model,wls,missing",
    [
        ("Qwen3.5-4B", ["gsm8k", "replay", "cap"], True),
        ("Qwen3.5-4B", ["gsm8k", "replay", "cap", "preempt"], False),
        ("Qwen3-4B", ["gsm8k", "replay", "cap"], False),
    ],
    ids=["state-model-without", "state-model-with", "dense"],
)
def test_official_needs_preempt_on_state_models(model, wls, missing):
    """S3: on a model with a Mamba state group the official profile includes
    ``preempt`` (gsm8k loads one state key per group)."""
    import argparse

    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t"}))  # fmt: skip
    committed = run_gate.REPO / f"docs/rsi/models/{model}/spec.json"
    probs = run_gate.official_problems(args, ["plain", "baseline", "pegaflow.m:C"], wls, committed, committed)  # fmt: skip
    assert any(p.startswith("workload preempt missing") for p in probs) is missing, probs


@pytest.mark.parametrize(
    "skipped,ok",
    [
        ({}, True),
        ({"not_offered": 3, "offer_unaligned": 1}, True),  # computed states: not loads
        ({"start_no_block": 1}, False),
        ({"start_unaligned": 2}, False),
        ({"start_no_tokens": 1}, False),
    ],
    ids=["none", "offers-only", "start-no-block", "start-unaligned", "start-no-tokens"],
)
def test_unjudged_start_state_fails_the_digest_verdict(skipped, ok):
    """S6: a start state (local hit or a load) the probe could not digest is
    a load nobody judged."""
    d = _digest_ok() | {"state_skipped": skipped}
    assert run_gate._digest_verdict(d, None)["pass"] is ok


def _spec_groups(*groups: dict) -> dict:
    return {"kv_cache_config": {"groups": [{"spec": g} for g in groups]}}


_FULL16 = {"mro": ["FullAttentionSpec"], "block_size": 16}
_MAMBA = {"mro": ["MambaSpec"], "block_size": 528, "num_speculative_blocks": 0}


@pytest.mark.parametrize(
    "spec,max_len,blocks",
    [
        (_spec_groups(_FULL16), 4096, 4 * 2048 // 16),  # 4 requests of 2048 tokens
        (_spec_groups(_FULL16), 16384, 16384 // 16 + 1),  # one max_model_len request + null
        (_spec_groups(_FULL16 | {"sliding_window": 128}), 4096, 4 * (128 + 16) // 16),
        (_spec_groups(_MAMBA, _MAMBA | {"num_speculative_blocks": 1}), 4096, 4 * (2 + 3)),
    ],
    ids=["full", "max-len-bound", "window", "mamba-speculative"],
)
def test_preempt_gpu_blocks_sizing(spec, max_len, blocks):
    """S6: the preempt pool holds PREEMPT_REQS requests of PREEMPT_SPAN tokens
    (so longer ones force preemption) and admits one max_model_len request."""
    assert run_gate.preempt_gpu_blocks(spec, max_len) == blocks


def test_preempt_gpu_blocks_qwen35_committed():
    """The Qwen3.5-4B self-test ran on 40 blocks (3 GDN groups x 2 + 4 of 528)."""
    import json

    spec = json.loads((run_gate.REPO / "docs/rsi/models/Qwen3.5-4B/spec.json").read_text())
    assert run_gate.preempt_gpu_blocks(spec, 4096) == 40
    assert run_gate.has_state_group(spec)
    assert not run_gate.has_state_group(json.loads((run_gate.REPO / "docs/rsi/models/Qwen3-4B/spec.json").read_text()))  # fmt: skip


def test_plain_equals_baseline_needs_equal_logprobs(tmp_path):
    """S6: equal completions with one different sampled-token logprob still
    fail (the probe changed the numbers)."""
    import json

    _alone_dirs(tmp_path, ["a", "b"], ["a", "b"], ["a", "b"])
    f = tmp_path / run_gate.GATE_DIR / "baseline" / "gsm8k" / "alone_ref-tokens.json"
    toks = json.loads(f.read_text())
    toks[1]["logprobs"] = [-0.25]
    f.write_text(json.dumps(toks))
    v = run_gate._plain_equals_baseline(tmp_path, ["c0", "cX"], batch_invariant=False)
    assert (v["pass"], v["differ"], v["logprob_differ"]) == (False, 0, 1)


def test_split_control_flags_a_vacuous_measurement(tmp_path):
    """N2: when nearly every pair started at the same point the control
    compared no split: reported ``vacuous`` with no measured value."""
    _alone_dirs(tmp_path, ["a", "b"], ["a", "b"], ["a", "b"], cached=(32, 32))
    ctrl = run_gate.split_control(tmp_path, batch_invariant=False)
    assert ctrl["same_start"] == 2
    v = run_gate._split_control_verdict(ctrl, True)
    assert v["vacuous"] and v["measured_invariant"] is None


def test_each_arm_gets_its_own_compile_cache(tmp_path, monkeypatch):
    """S5: vLLM's cache root (torch compile, Inductor, Triton) is per arm
    under the run dir, emptied before the arm's first stage, shared by its
    later stages, never another arm's or the global one."""
    seen: list[dict[str, str]] = []

    class _Lease:
        ports, block = [20001], [20010]

    monkeypatch.setattr(run_gate, "PortLease", lambda *a, **kw: _Lease())
    monkeypatch.setattr(run_gate, "launched", lambda cmd, env, *a: seen.append(env))
    monkeypatch.setattr(run_gate, "isolated", lambda args, arm, log, cmd: cmd)
    monkeypatch.setattr(run_gate, "_FRESH_CACHES", set())
    monkeypatch.setenv("TRITON_CACHE_DIR", "/global/triton")
    args = SimpleNamespace(
        gpu="0", model="m", tp=1, max_model_len=8, max_num_batched_tokens=8, max_num_seqs=1,
        gpu_mem=0.5, vllm_arg=[], num_gpu_blocks=0, under_test_path="",
    )  # fmt: skip
    run = tmp_path / "run"
    stale = run / ".compile-cache" / "baseline" / "torch_compile_cache" / "x"
    stale.mkdir(parents=True)
    for label, wl in (("baseline", "gsm8k"), ("baseline", "preempt"), ("selftest-liar", "gsm8k")):
        (run / label / wl).mkdir(parents=True, exist_ok=True)
        if label == "baseline" and wl == "preempt":
            (run / ".compile-cache/baseline/artifact").write_text("from gsm8k")
        run_gate.vllm_server(args, {}, "baseline", run / label / wl / "probe", False, None, run / label / wl / "vllm.log")  # fmt: skip
    roots = [e["VLLM_CACHE_ROOT"] for e in seen]
    assert roots == [
        str(run / ".compile-cache" / x) for x in ("baseline", "baseline", "selftest-liar")
    ]
    assert not stale.exists()  # emptied before the arm's first stage
    assert (run / ".compile-cache/baseline/artifact").exists()  # kept across its stages
    for e in seen:  # no compiled-code cache outside the arm's root
        root = Path(e["VLLM_CACHE_ROOT"])
        for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"):
            assert Path(e[k]).is_relative_to(root), k
        assert e["TRITON_CACHE_AUTOTUNING"] == "1"


def test_arms_run_the_baselines_triton_autotune_picks(tmp_path):
    """A fresh arm root gets a copy of the baseline's ``*.autotune.json``
    (FLA kernels outside Inductor pick tiles by timing, and the picks change
    the numbers) and nothing else of the baseline's cache."""
    import json

    caches = tmp_path / ".compile-cache"
    base = caches / "baseline" / "triton"
    (base / "H1").mkdir(parents=True)
    pick = json.dumps({"key": [1], "configs_timings": [[{"kwargs": {"BK": 32}, "num_warps": 4, "num_stages": 2}, [0.1]]]})  # fmt: skip
    (base / "H1" / "chunk_fwd_kernel_o.autotune.json").write_text(pick)
    (base / "H1" / "chunk_fwd_kernel_o.cubin").write_text("kernel")
    (caches / "baseline" / "torch_compile_cache").mkdir()
    arm = caches / "selftest-honest"
    arm.mkdir()
    assert run_gate.seed_autotune(arm) == 1
    assert (arm / "triton/H1/chunk_fwd_kernel_o.autotune.json").read_text() == pick
    assert sorted(p.name for p in arm.rglob("*") if p.is_file()) == ["chunk_fwd_kernel_o.autotune.json"]  # fmt: skip
    assert run_gate.seed_autotune(caches / "baseline") == 0  # the source itself
    assert run_gate.seed_autotune(tmp_path / "x" / "plain") == 0  # no baseline yet
