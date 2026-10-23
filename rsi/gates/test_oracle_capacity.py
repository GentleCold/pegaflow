"""G5 capacity oracle contract (no GPU; needs the built ``pegaflow.semantics``).

Run from the repo root: ``python -m pytest rsi/gates/test_oracle_capacity.py``.
"""

from __future__ import annotations

import pytest

from rsi.gates import oracle

BT = 4  # tokens per block, every group
PAGE = 500  # bytes per layer page
LAYERS = 2  # -> 1000 bytes per (group, key) block


def _group(gid: int, cls: str, mro: list[str]) -> dict:
    return {
        "id": gid,
        "num_layers": LAYERS,
        "enable_kv_transfer": True,
        "is_eagle_group": False,
        "spec": {
            "class": cls,
            "mro": mro,
            "block_size": BT,
            "page_size_bytes": PAGE,
            "real_page_size_bytes": PAGE,
            "sliding_window": None,
        },
    }


FULL = _group(0, "FullAttentionSpec", ["FullAttentionSpec", "AttentionSpec", "KVCacheSpec"])
CKPT = _group(1, "MambaSpec", ["MambaSpec", "KVCacheSpec"])


def _spec(*groups: dict) -> dict:
    return {"parallel": {"tp": 1}, "kv_cache_config": {"groups": list(groups)}}


def _req(prompt: str, n_blocks: int, step: int, label: str) -> dict:
    """One request whose prompt is ``n_blocks`` full blocks plus one token,
    computed in one step that crosses every block boundary (so checkpoint
    groups store every block too)."""
    plen = n_blocks * BT + 1
    return {
        "pass": label,
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
        for p in prompts:
            reqs.append(_req(p, n_blocks, step, label))
            step += 2  # visible to the next request
    return reqs


def _by_pass(reqs: list[dict], hits: list[int], label: str) -> int:
    return sum(h for r, h in zip(reqs, hits, strict=True) if r["pass"] == label)


# Cyclic scan over 3 prompts x 4 blocks (12 kB) with room for 8 blocks.
CYCLIC = (_spec(FULL), _trace(["p0", "p1", "p2"], 4, ["a", "b", "c"]))
# Full + checkpoint: 2 prompts x (2 full + 2 checkpoint) blocks.
HYBRID = (_spec(FULL, CKPT), _trace(["p0", "p1"], 2, ["a", "b"]))


@pytest.mark.parametrize("case", [CYCLIC, HYBRID], ids=["cyclic-full", "full+checkpoint"])
@pytest.mark.parametrize("policy", ["belady", "lru"])
def test_capacity_at_least_u_equals_unlimited_oracle(case, policy):
    spec, reqs = case
    u = oracle.unique_bytes(spec, reqs)
    hits, diag = oracle.simulate(spec, reqs, u, policy)
    assert hits == oracle.replay(spec, reqs)
    assert all(g["evicted"] == 0 for g in diag["groups"])


def test_unique_bytes_counts_distinct_group_keys():
    spec, reqs = HYBRID
    assert oracle.block_bytes(spec) == [PAGE * LAYERS, PAGE * LAYERS]
    # 2 prompts x (2 full + 2 checkpoint) distinct blocks; later passes add none.
    assert oracle.unique_bytes(spec, reqs) == 8 * PAGE * LAYERS
    assert oracle.unique_bytes(spec, reqs, {"a"}) == 8 * PAGE * LAYERS


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
