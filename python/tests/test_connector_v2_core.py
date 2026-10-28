"""Contract of ``pegaflow.connector_v2.core`` (default unit gate, source-only).

Drives the scheduler/worker state machines through a stub adapter, a fake
server client and ``StubSemantics``, a small pure-Python model of
``pegaflow.semantics`` injected at the boundary, so no vLLM, torch, native
extension or server is needed. Cases marked ``native`` re-run against the real
``pegaflow.semantics`` and skip without the extension;
``test_stub_matches_native`` keeps the stub honest.
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
import types
from dataclasses import dataclass, field

import pytest

from pegaflow.connector_v2 import adapters
from pegaflow.connector_v2.adapters import LayerDecl, LayerView, Parallel, Unsupported
from pegaflow.connector_v2.core import (
    Scheduler,
    StepPlan,
    Worker,
    build_layout,
)
from pegaflow.connector_v2.core.layout import StorageRanks
from pegaflow.connector_v2.core.scheduler import GroupSave, LoadPlan
from pegaflow.connector_v2.policy import Policy, SaveItem

BT = 4


# --------------------------------------------------------------- semantics stub
@dataclass(frozen=True)
class _Group:
    kind: str
    block_tokens: int
    window_tokens: int | None = None

    @staticmethod
    def full(bt: int) -> _Group:
        return _Group("full", bt)

    @staticmethod
    def window(bt: int, tokens: int) -> _Group:
        return _Group("window", bt, tokens)

    @staticmethod
    def checkpoint(bt: int) -> _Group:
        return _Group("checkpoint", bt)

    @staticmethod
    def scratch(bt: int) -> _Group:
        return _Group("scratch", bt)


@dataclass(frozen=True)
class _Shard:
    index: int = 0
    count: int = 1

    @staticmethod
    def replicated() -> _Shard:
        return _Shard()

    @staticmethod
    def heads(index: int, count: int) -> _Shard:
        return _Shard(index, count)


@dataclass(frozen=True)
class _Region:
    layer: str
    group: int
    segments: list[int]
    shard: _Shard


def _range(lo: int, hi: int) -> tuple[int, int]:
    return (lo, hi) if lo < hi else (0, 0)


def _needed(g: _Group, p: int) -> tuple[int, int]:
    n = p // g.block_tokens
    if g.kind == "full":
        return _range(0, n)
    if g.kind == "window":
        return _range(max(p - g.window_tokens, 0) // g.block_tokens, n)
    if g.kind == "checkpoint":
        return _range(max(n - 1, 0), n)
    return (0, 0)


def _resolve(groups, present, prompt_len: int) -> int:
    steps = [g.block_tokens for g in groups if g.kind != "scratch"]
    if not steps:
        return 0
    have = [set(s) for s in present]
    best = 0
    for p in range(math.lcm(*steps), prompt_len, math.lcm(*steps)):
        if all(set(range(*_needed(g, p))) <= h for g, h in zip(groups, have, strict=True)):
            best = p
    return best


def _saveable(g: _Group, before: int, after: int) -> tuple[int, int]:
    bt = g.block_tokens
    if after <= before:
        return (0, 0)
    if g.kind in ("full", "window"):
        return _range(before // bt, after // bt)
    if g.kind == "checkpoint" and after % bt == 0:
        return _range(after // bt - 1, after // bt)
    return (0, 0)


@dataclass(frozen=True)
class _Fragment:
    layer: str
    part: int
    holders: list[int]

    @property
    def writer(self) -> int:
        return self.holders[0]


@dataclass(frozen=True)
class _SealPlan:
    plan: dict[int, list[_Fragment]]

    @property
    def groups(self) -> list[int]:
        return sorted(self.plan)

    def fragments(self, group: int) -> list[_Fragment]:
        return self.plan[group]

    def slot_of(self, rank: int, layer: str) -> tuple[int, int] | None:
        for g, frags in self.plan.items():
            for slot, f in enumerate(frags):
                if f.layer == layer and rank in f.holders:
                    return g, slot
        return None


def _seal_plan(ranks):
    """The checks the core relies on (equal segments, every part held) and
    the fragment order: layer name, then part; holders ascending."""
    layers: dict[str, tuple[int, list[int], int, dict[int, set[int]]]] = {}
    for rank, regions in ranks:
        for r in regions:
            g, segs, count, parts = layers.setdefault(
                r.layer, (r.group, r.segments, r.shard.count, {})
            )
            if segs != r.segments or count != r.shard.count:
                raise ValueError(f"layer {r.layer}: segment sizes differ")
            parts.setdefault(r.shard.index, set()).add(rank)
    plan: dict[int, list[_Fragment]] = {}
    for layer, (g, _, count, parts) in sorted(layers.items()):
        if set(parts) != set(range(count)):
            raise ValueError(f"layer {layer}: missing part")
        plan.setdefault(g, []).extend(_Fragment(layer, p, sorted(parts[p])) for p in range(count))
    return _SealPlan(plan)


StubSemantics = types.SimpleNamespace(
    Group=_Group,
    Shard=_Shard,
    Region=_Region,
    needed=_needed,
    resolve=_resolve,
    saveable=_saveable,
    seal_plan=_seal_plan,
)


def _native():
    ext = pytest.importorskip("pegaflow.pegaflow", exc_type=ImportError)
    if getattr(ext, "__file__", None) is None:  # file-less stub from tests/unit_stubs.py
        pytest.skip("native extension not built (unit stub installed)")
    from pegaflow import semantics

    return semantics


@pytest.fixture
def sem(request):
    return _native() if getattr(request, "param", "stub") == "native" else StubSemantics


BOTH = pytest.mark.parametrize("sem", ["stub", "native"], indirect=True)


# ------------------------------------------------------------- adapter + fakes
@dataclass(frozen=True)
class StubFullSpec:
    block_size: int = BT
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class StubWindowSpec:
    block_size: int = BT
    window: int = 7
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class StubCheckpointSpec:
    block_size: int = BT
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class MysterySpec:
    block_size: int = BT


class _StubAdapter:
    def __init__(self, semantics):
        self.sem = semantics

    def group(self, spec):
        if isinstance(spec, StubWindowSpec):
            return self.sem.Group.window(spec.block_size, spec.window)
        if isinstance(spec, StubCheckpointSpec):
            return self.sem.Group.checkpoint(spec.block_size)
        return self.sem.Group.full(spec.block_size)

    def layer(self, spec, layer_name, par):
        return LayerDecl(
            spec.segments, self.sem.Shard.heads(par.tp_rank, par.tp_size), spec.backend
        )

    def view(self, spec, layer_name, kv):
        return LayerView(kv, num_blocks=16, bytes_per_block=64, kv_stride_bytes=1024)


@pytest.fixture(autouse=True)
def stub_adapter(monkeypatch, sem):
    module = types.ModuleType(f"{adapters.__name__}._test_stub")
    module.ADAPTER = _StubAdapter(sem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("StubFullSpec", "StubWindowSpec", "StubCheckpointSpec"):
        monkeypatch.setitem(adapters._REGISTRY, name, "_test_stub")


def _kv_cache_config(*specs):
    groups = [
        types.SimpleNamespace(kv_cache_spec=s, layer_names=[f"layers.{i}.attn"])
        for i, s in enumerate(specs)
    ]
    return types.SimpleNamespace(
        transfer_group_ids=tuple(range(len(specs))),
        kv_cache_groups=groups,
        hash_block_size=BT,
    )


def _layout(sem, *specs, par=None):
    specs = specs or (StubWindowSpec(), StubFullSpec())
    par = par or Parallel()
    return build_layout(_kv_cache_config(*specs), par, {"model": "m"}, semantics=sem)


@dataclass
class Ready:  # QueryReady
    num_hit_blocks: int
    lease: bytes
    hit_positions: list[int] = field(default_factory=list)


class Loading:  # QueryLoading
    pass


class FakeRpc:
    """``answers[sid]`` is a Ready/Loading or an exception to raise."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.released: list[bytes] = []
        self.queried: list[int] = []  # group_id of every query_prefetch call
        self.loads: list[list] = []
        self.load_ok: dict[bytes, bool] = {}
        self.registered: dict | None = None

    def query_prefetch(self, instance_id, hashes, req_id, wait_for_full_prefix, group_id):
        self.queried.append(group_id)
        ans = self.answers[group_id]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def release(self, lease):
        self.released.append(lease)

    def load(self, instance_id, tp_rank, device_id, shm, layers, loads):
        self.loads.append(loads)
        if not all(self.load_ok.get(lease, True) for lease, _ in loads):
            return False, "query lease is unknown or expired"
        return True, ""

    def register_context_batch(self, *args):
        names = ["instance_id", "namespace", "tp_rank", "pp_rank", "tp_size", "world_size"]
        names += ["device_id", "names", "wrappers", "num_blocks", "bytes_per_block"]
        names += ["kv_strides", "segments", "backend", "page_first", "group_ids"]
        self.registered = dict(zip(names, args, strict=True))
        return True, ""

    def save(self, *args):
        return True, ""


class FakePool:
    """vLLM BlockPool references by block id: a pinned block stays out of the
    free queue (never reallocated, zeroed or copied into) until unpinned."""

    def __init__(self):
        self.refs: dict[int, int] = {}

    def pin(self, block_ids):
        for b in block_ids:
            self.refs[b] = self.refs.get(b, 0) + 1
        return list(block_ids)

    def unpin(self, token):
        for b in token:
            self.refs[b] -= 1
            if not self.refs[b]:
                del self.refs[b]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _req(num_tokens=13, n_hashes=3, req_id="r"):
    hashes = [f"h{i}".encode() for i in range(n_hashes)]
    return types.SimpleNamespace(request_id=req_id, num_tokens=num_tokens, block_hashes=hashes)


def _sched(sem, answers, policy=None, layout=None, pool=None, **kw):
    rpc = FakeRpc(answers)
    layout = layout or _layout(sem)
    sched = Scheduler(layout, policy or Policy(), rpc, "inst", clock=Clock(), semantics=sem, **kw)
    pool = pool or FakePool()
    sched.bind_pins(pool.pin, pool.unpin)
    return sched, rpc


# Full group (sid 0, prefix query) + window(4, 7) group (sid 1, membership).
# 13-token prompt: 3 hashed blocks; resume at 12 needs full {0,1,2}, window {1,2}.
HIT = {0: Ready(3, b"L0"), 1: Ready(3, b"L1", [0, 1, 2])}


class _Policy(Policy):
    def __init__(self, accept=None, choose=None):
        super().__init__(loading_wait_ms=0)
        self._accept, self._choose = accept, choose

    def accept(self, p, present):
        return self._accept(p, present) if self._accept else p

    def choose_saves(self, items):
        return self._choose(items) if self._choose else items


def _mutate_present(p, present):
    present[1].add(0)  # "assume the block is there"
    return p


def _raise(*_):
    raise RuntimeError("policy bug")


# ----------------------------------------------------------------------- hits
@BOTH
@pytest.mark.parametrize(
    "answers,policy,num_computed,expected,released",
    [
        pytest.param(HIT, Policy(), 4, (8, True), [], id="hit"),
        pytest.param(
            {0: Loading(), 1: Ready(3, b"L1", [0, 1, 2])},
            Policy(loading_wait_ms=100),
            0,
            (None, False),
            [b"L1"],
            id="loading-retries",
        ),
        pytest.param(
            {0: Ready(3, b"L0"), 1: RuntimeError("server down")},
            Policy(),
            0,
            (0, False),
            [b"L0"],
            id="server-error-is-miss",
        ),
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(1, b"L1", [2])},
            Policy(),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="window-gap-is-miss",
        ),
        # Window only has block 0: resolve lowers p to 4, both leases kept.
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(1, b"L1", [0])},
            Policy(),
            0,
            (4, True),
            [],
            id="window-short-lowers-p",
        ),
        # M1: a policy cannot widen the hit by mutating what it is shown.
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(2, b"L1", [1, 2])},
            _Policy(accept=_mutate_present),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="policy-mutates-present",
        ),
        # Returning more than semantics allows is capped at the legal p.
        pytest.param(
            HIT, _Policy(accept=lambda p, _: 10**6), 0, (12, True), [], id="policy-widens"
        ),
        # S6: a raising policy is a miss with every lease released, not a crash.
        pytest.param(
            HIT, _Policy(accept=_raise), 0, (0, False), [b"L0", b"L1"], id="policy-raises"
        ),
    ],
)
def test_matched_tokens(sem, answers, policy, num_computed, expected, released):
    sched, rpc = _sched(sem, answers, policy)
    assert sched.get_num_new_matched_tokens(_req(), num_computed) == expected
    assert sorted(rpc.released) == released


def test_semantics_error_releases_leases():
    def boom(*_):
        raise OverflowError("bad group")

    sem = types.SimpleNamespace(**{**vars(StubSemantics), "resolve": boom})
    sched, rpc = _sched(sem, HIT, layout=_layout(StubSemantics))
    assert sched.get_num_new_matched_tokens(_req(), 0) == (0, False)
    assert sorted(rpc.released) == [b"L0", b"L1"]


def test_hit_lifecycle(sem):
    """Lease needed blocks only -> load plan; an unallocated hit is released on finish."""
    # vLLM group 0 is the window group: the full group must still become sid 0.
    layout = _layout(sem)
    assert [g.gid for g in layout.groups] == [1, 0]
    sched, rpc = _sched(sem, HIT, layout=layout)
    req = _req()

    assert sched.get_num_new_matched_tokens(req, 4) == (8, True)
    assert sched.loaded_group_ids("r") == (0, 1)
    window_ids, full_ids = [0, 21, 22, 23], [10, 11, 12, 13]
    sched.update_state_after_alloc(req, (window_ids, full_ids), 8)
    (plan,) = sched.take_plan().loads
    assert not plan.fail
    # Block 0 of both groups is already on the GPU (num_computed=4): not loaded.
    assert plan.loads == [
        (b"L0", [[None, 11, 12], [None, None, None]]),
        (b"L1", [[None, None, None], [None, 21, 22]]),
    ]
    assert rpc.released == []  # leases are consumed by the workers' load

    # Hit that never gets allocated: vLLM asks again for a head it could not
    # allocate (held leases released, looked up afresh), then aborts it.
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    assert sorted(rpc.released) == [b"L0", b"L1"]
    sched.request_finished("r")
    assert sorted(rpc.released) == [b"L0", b"L0", b"L1", b"L1"]


def _other(num_tokens=13, n_hashes=3, req_id="b", diverge_at=None):
    req = _req(num_tokens, n_hashes, req_id)
    if diverge_at is not None:
        req.block_hashes[diverge_at:] = [b"x%d" % i for i in range(diverge_at, n_hashes)]
    req.skip_reading_prefix_cache = False
    return req


def test_waits_for_inflight_loads_of_its_blocks():
    """vLLM caches a's blocks only once its async load lands: b waits (no
    query, no lease) instead of loading its own copy of the same blocks.
    Mirrors vLLM's OffloadingConnector ``_chunks_being_loaded``."""
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"F")})
    sched = Scheduler(
        layout, Policy(sync_load_tokens=0), rpc, "inst", clock=Clock(), semantics=StubSemantics
    )
    a = _req(req_id="a")
    sched.update_state_after_alloc(a, (FULL_IDS,), sched.get_num_new_matched_tokens(a, 0)[0])
    assert not any(lp.fail for lp in sched.take_plan().loads)
    rpc.queried.clear()
    assert sched.get_num_new_matched_tokens(_other(), 0) == (None, False)
    assert rpc.queried == []


FULL_IDS, WINDOW_IDS = [10, 11, 12, 13], [20, 21, 22, 23]


@pytest.mark.parametrize(
    "ext,ids,fail,loads,released",
    [
        pytest.param(0, (WINDOW_IDS, FULL_IDS), None, None, [b"L0", b"L1"], id="no-external"),
        pytest.param(16, (WINDOW_IDS, FULL_IDS), True, [], [b"L0", b"L1"], id="more-than-planned"),
        # vLLM trimmed the hit to 8 tokens: re-plan from the held leases (S1).
        pytest.param(
            8,
            (WINDOW_IDS, FULL_IDS),
            False,
            [
                (b"L0", [[10, 11, None], [None, None, None]]),
                (b"L1", [[None, None, None], [20, 21, None]]),
            ],
            [],
            id="trimmed-replans",
        ),
        pytest.param(12, ([20, 0, 22, 23], FULL_IDS), True, [], [b"L0", b"L1"], id="null-block"),
        pytest.param(12, (WINDOW_IDS, [10, 11]), True, [], [b"L0", b"L1"], id="short-block-table"),
    ],
)
def test_alloc(ext, ids, fail, loads, released):
    sched, rpc = _sched(StubSemantics, HIT)
    req = _req()
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    sched.update_state_after_alloc(req, ids, ext)
    plans = sched.take_plan().loads
    if fail is None:
        assert plans == []
    else:
        assert [(p.fail, p.loads) for p in plans] == [(fail, loads)]
    assert sorted(rpc.released) == released
    # A second call (allowed by vLLM) finds no hit and does nothing.
    sched.update_state_after_alloc(req, ids, ext)
    assert sched.take_plan().loads == []


@pytest.mark.parametrize(
    "sync_loads,num_computed,load_state,expected",
    [
        # S2: no deadline. Until the server signals, its copy may still land,
        # so a slow load is neither failed (vLLM would recompute into the
        # blocks) nor done; the worker clock jumps 60 s per read here.
        pytest.param(True, 0, (5, 1), ((12, False), set(), set()), id="sync-slow-ok"),
        # A failed sync load cannot be a failed receive (vLLM already counts
        # the tokens): every GPU block holding hit tokens is reported invalid.
        pytest.param(True, 4, -1, ((8, False), set(), {11, 12}), id="sync-failed"),
        # A sync load whose RPC failed is invalid, never waited for.
        pytest.param(True, 4, "rpc-failed", ((8, False), set(), {11, 12}), id="sync-rpc-failed"),
        # Layouts vLLM cannot recover a failed sync load for stay async.
        pytest.param(False, 0, 1, ((12, True), {"r"}, set()), id="layout-async"),
    ],
)
def test_sync_loads(sync_loads, num_computed, load_state, expected):
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"L0")})
    sched = Scheduler(
        layout,
        Policy(sync_load_tokens=256),
        rpc,
        "inst",
        clock=Clock(),
        semantics=StubSemantics,
        sync_loads=sync_loads,
    )
    req = _req()
    matched = sched.get_num_new_matched_tokens(req, num_computed)
    sched.update_state_after_alloc(req, ([10, 11, 12, 13],), matched[0])
    if load_state == "rpc-failed":
        rpc.load_ok[b"L0"], load_state = False, 1
    polls, load_state = load_state if isinstance(load_state, tuple) else (0, load_state)
    state = FakeLoadState(state=load_state, polls=polls)
    worker = _worker(StubSemantics, rpc, [state], layout=layout, clock=JumpClock(60.0))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.start(sched.take_plan())
    _, recving, _ = worker.poll()
    assert (matched, recving, worker.take_invalid()) == expected
    worker.shutdown()


# Step 2 onwards, after vLLM processed step 0's outcome. "X" is shared's next
# turn: a new request whose prefix hit covers shared's blocks 10, 11, 20, 21.
OK_NEXT = [[("r", 17, 1), ("shared", 17, 1)], [("X", 16, 5)]]
FAIL_NEXT = [[("r", 4, 13), ("shared", 17, 1)], [("X", 16, 5)]]


@pytest.mark.parametrize(
    "load_state,steps,expected",
    [
        pytest.param(1, OK_NEXT, [[13, 20, 21], [22]], id="sync-ok"),
        # vLLM (recompute policy) evicts nothing: r is rewound, but shared
        # (not rewound) built 20-21 on the bad block 11 and they stay cached
        # under correct hashes, so its next turn X would extend bad KV.
        pytest.param(-1, FAIL_NEXT, [[], []], id="sync-failed-sharer-next-turn"),
    ],
)
def test_sync_load_outcome_gates_saves(load_state, steps, expected):
    """M1: the server keeps the first copy of a key, so KV computed on a sync
    load is not saved before vLLM processed the load's outcome
    (``confirm_steps``: vLLM's max_concurrent_batches), and after a failed one
    nothing is saved at all (vLLM may serve KV built on it from its prefix
    cache to any later request).

    Step 0: r sync-loads blocks 1-2 and prefills to 17; other is unrelated.
    Step 1 (scheduled before step 0's output is processed): r decodes, shared
    prefills on a local prefix hit on r's blocks 0-1.
    """
    layout = _layout(StubSemantics, StubFullSpec())
    pool = FakePool()
    sched, rpc = _sched(
        StubSemantics, {0: Ready(3, b"L0")}, layout=layout, pool=pool, sync_loads=True
    )
    tables = {
        "r": [10, 11, 12, 13],
        "shared": [10, 11, 20, 21],
        "other": [30, 31, 32, 33],
        "X": [10, 11, 20, 21, 22, 23],
    }
    reqs = {rid: _req(17, 4, rid) for rid in ("r", "shared", "other")}
    for req in reqs.values():
        sched.on_new_request(req)
    req = reqs["r"]
    n, load_async = sched.get_num_new_matched_tokens(req, 4)
    assert (n, load_async) == (8, False)
    sched.update_state_after_alloc(req, (tables["r"],), n)

    def step(scheduled):
        sched.plan_saves(scheduled, lambda rid: (tables[rid],))
        return [b for s in sched.take_plan().saves for b in s.block_ids]

    # The load plan and step 0's saves ship in one StepPlan.
    sched.plan_saves([("r", 12, 5), ("other", 0, 17)], lambda rid: (tables[rid],))
    load_step = sched.take_plan()
    saved0 = [b for s in load_step.saves for b in s.block_ids]
    worker = _worker(StubSemantics, rpc, [FakeLoadState(state=load_state)], layout=layout)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.start(load_step)
    saved1 = step([("r", 17, 1), ("shared", 8, 9)])
    sched.on_load_errors(worker.take_invalid())  # vLLM processes step 0's output
    later = []
    for i, scheduled in enumerate(steps):
        if i == 1:
            sched.on_new_request(_req(21, 6, "X"))
        later.append(step(scheduled))
    assert (saved0, saved1, later) == ([30, 31, 32, 33], [], expected)
    # A save held for an unconfirmed load pins nothing: only chosen saves do.
    assert set(pool.refs) == {30, 31, 32, 33, *(b for blocks in later for b in blocks)}
    worker.shutdown()


@pytest.mark.parametrize("event", ["finish", "new-hit"])
def test_lifecycle_after_saves_disabled(event):
    """After a failed sync load no save is planned, but "other"'s save from
    before it may still read its blocks: they stay pinned until it is done,
    also once other finishes. S1: later short hits load async (a failed
    async load is never cached)."""
    layout = _layout(StubSemantics, StubFullSpec())
    pool = FakePool()
    sched, _ = _sched(
        StubSemantics, {0: Ready(3, b"L0")}, layout=layout, pool=pool, sync_loads=True
    )
    tables = {"r": [10, 11, 12, 13], "other": [30, 31, 32, 33]}
    reqs = {rid: _req(17, 4, rid) for rid in ("r", "other")}
    for req in reqs.values():
        sched.on_new_request(req)
    n, _ = sched.get_num_new_matched_tokens(reqs["r"], 4)
    sched.update_state_after_alloc(reqs["r"], (tables["r"],), n)
    sched.plan_saves([("r", 12, 5), ("other", 0, 17)], lambda rid: (tables[rid],))
    assert [b for s in sched.take_plan().saves for b in s.block_ids] == [30, 31, 32, 33]
    sched.on_load_errors({11, 12})  # r's sync load failed
    if event == "finish":
        assert sched.request_finished("other") is False
        assert set(pool.refs) == {30, 31, 32, 33}
        sched.on_saved_through(0)
        assert pool.refs == {}
    else:
        assert sched.get_num_new_matched_tokens(_req(17, 4, "late"), 4) == (8, True)
    sched.plan_saves([("other", 17, 1)], lambda rid: (tables[rid],))
    assert sched.take_plan().saves == []


def test_shutdown_releases_unsent_plans_and_held_hits():
    sched, rpc = _sched(StubSemantics, HIT)
    req = _req()
    sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, (WINDOW_IDS, FULL_IDS), 12)
    rpc.answers = {0: Ready(3, b"H0"), 1: Ready(3, b"H1", [0, 1, 2])}
    # Never allocated; other blocks than r's in-flight load (no deferral).
    assert sched.get_num_new_matched_tokens(_other(req_id="held", diverge_at=0), 0) == (12, True)
    sched.shutdown()
    assert sorted(rpc.released) == [b"H0", b"H1", b"L0", b"L1"]


# ---------------------------------------------------------------------- saves
def _step_saves(sched, steps, block_ids):
    out = []
    for step in steps:
        sched.plan_saves(step, lambda rid: block_ids.get(rid))
        out.append([(s.sid, s.block_ids, s.block_hashes) for s in sched.take_plan().saves])
    return out


H = [f"h{i}".encode() for i in range(4)]


@BOTH
@pytest.mark.parametrize(
    "num_tokens,steps,block_ids,expected",
    [
        # Prefill 13 tokens in one step: full {0,1,2} and window {0,1,2}.
        pytest.param(
            13,
            [[("r", 0, 13)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [[(0, [10, 11, 12], H[:3]), (1, [20, 21, 22], H[:3])]],
            id="prefill",
        ),
        # Chunked prefill 6 + 7: block 1 becomes saveable only in the 2nd chunk.
        pytest.param(
            13,
            [[("r", 0, 6)], [("r", 6, 7)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [
                [(0, [10], H[:1]), (1, [20], H[:1])],
                [(0, [11, 12], H[1:3]), (1, [21, 22], H[1:3])],
            ],
            id="chunked",
        ),
        # Window block 0 already freed (null id): skipped, full still saved.
        pytest.param(
            13,
            [[("r", 0, 13)]],
            {"r": ([0, 21, 22, 23], FULL_IDS)},
            [[(0, [10, 11, 12], H[:3]), (1, [21, 22], H[1:3])]],
            id="null-id",
        ),
        # Async scheduling: 8 known tokens, 4 scheduled + spec tokens beyond:
        # `after` is capped at num_tokens, block 2 waits for a later step.
        pytest.param(
            10,
            [[("r", 0, 12)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [[(0, [10, 11], H[:2]), (1, [20, 21], H[:2])]],
            id="async-capped",
        ),
        # No block ids this step: the range is offered again next step.
        pytest.param(
            13,
            [[("r", 0, 8)], [("r", 8, 5)]],
            {},
            [[], []],
            id="no-block-ids",
        ),
    ],
)
def test_saves(sem, num_tokens, steps, block_ids, expected):
    sched, _ = _sched(sem, {})
    sched.on_new_request(_req(num_tokens=num_tokens, n_hashes=num_tokens // BT))
    assert _step_saves(sched, steps, block_ids) == expected


# Checkpoint group (vLLM gid 0) + full group (gid 1, sid 0). Like vLLM's Mamba
# align table, the checkpoint row is null except the running state block.
CKPT_IDS = [0, 0, 0, 33]


@BOTH
@pytest.mark.parametrize(
    "steps,offers,expected",
    [
        # Prefill to 12 in chunks of 4: each chunk end's state is offered with
        # its block; the checkpoint saves exactly the offered block, never the
        # positional one (CKPT_IDS[b]).
        pytest.param(
            [("r", 0, 4), ("r", 4, 4), ("r", 8, 4)],
            [{"r": [(0, 31, 4)]}, {"r": [(0, 32, 8)]}, {"r": [(0, 33, 12)]}],
            [
                [(0, [10], H[:1]), (1, [31], H[:1])],
                [(0, [11], H[1:2]), (1, [32], H[1:2])],
                [(0, [12], H[2:3]), (1, [33], H[2:3])],
            ],
            id="offer-per-boundary",
        ),
        # vLLM did not retain (offer) the boundary at 4: no checkpoint save.
        pytest.param(
            [("r", 0, 4), ("r", 4, 4)],
            [{}, {"r": [(0, 32, 8)]}],
            [[(0, [10], H[:1])], [(0, [11], H[1:2]), (1, [32], H[1:2])]],
            id="unoffered-skipped",
        ),
        # Offers for another group, another request, an older boundary or an
        # unaligned one never stand in for this step's state.
        pytest.param(
            [("r", 0, 4), ("r", 4, 4)],
            [{"r": [(1, 31, 4)], "x": [(0, 39, 4)]}, {"r": [(0, 31, 4), (0, 35, 6)]}],
            [[(0, [10], H[:1])], [(0, [11], H[1:2])]],
            id="foreign-offers-ignored",
        ),
        # A step that ends off a boundary computes no checkpoint state.
        pytest.param([("r", 0, 6)], [{"r": [(0, 31, 4)]}], [[(0, [10], H[:1])]], id="off-boundary"),
    ],
)
def test_checkpoint_saves_from_offers(sem, steps, offers, expected):
    layout = _layout(sem, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(sem, {}, layout=layout)
    sched.on_new_request(_req(num_tokens=13, n_hashes=3))
    out = []
    for step, offer in zip(steps, offers, strict=True):
        sched.plan_saves([step], lambda _: (CKPT_IDS, FULL_IDS), offers=offer)
        out.append([(s.sid, s.block_ids, s.block_hashes) for s in sched.take_plan().saves])
    assert out == expected


@BOTH
@pytest.mark.parametrize(
    "num_computed,ckpt_ans,expected,loads",
    [
        # Cold: full blocks 0-2 and the checkpoint state at 12 (block 2).
        pytest.param(
            0,
            Ready(2, b"C", [0, 2]),
            (12, True),
            [(b"F", [[10, 11, 12], [None] * 3]), (b"C", [[None, None], [None, 33]])],
            id="cold",
        ),
        # Local hit covers 8 tokens: full loads only block 2; the state at 12
        # is loaded (vLLM drops its own local Mamba state for an external hit).
        pytest.param(
            8,
            Ready(2, b"C", [0, 2]),
            (4, True),
            [(b"F", [[None, None, 12], [None] * 3]), (b"C", [[None, None], [None, 33]])],
            id="after-local-hit",
        ),
        # Only the state at 4 is stored: resolve falls back to p=4.
        pytest.param(
            0,
            Ready(1, b"C", [0]),
            (4, True),
            [(b"F", [[10, None, None], [None] * 3]), (b"C", [[None], [33]])],
            id="older-checkpoint",
        ),
    ],
)
def test_checkpoint_loads_state_at_p(sem, num_computed, ckpt_ans, expected, loads):
    layout = _layout(sem, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(sem, {0: Ready(3, b"F"), 1: ckpt_ans}, layout=layout)
    req = _req()
    matched = sched.get_num_new_matched_tokens(req, num_computed)
    assert matched == expected
    p = num_computed + matched[0]
    # vLLM allocates the external hit's state block at (p - 1) // bt.
    ckpt_ids = [0] * (p // BT - 1) + [33]
    sched.update_state_after_alloc(req, (ckpt_ids, FULL_IDS), matched[0])
    (plan,) = sched.take_plan().loads
    assert (plan.fail, plan.loads) == (False, loads)


def test_held_hit_released_when_deferred():
    """A held hit whose request starts waiting for another request's async
    load of its blocks is released, not reused (the wait holds no lease)."""
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"B")})
    policy = Policy(sync_load_tokens=0)
    sched = Scheduler(layout, policy, rpc, "inst", clock=Clock(), semantics=StubSemantics)
    b = _other()
    assert sched.get_num_new_matched_tokens(b, 0) == (12, True)  # held: not allocated
    rpc.answers = {0: Ready(3, b"A")}
    a = _req(req_id="a")
    sched.update_state_after_alloc(a, (FULL_IDS,), sched.get_num_new_matched_tokens(a, 0)[0])
    sched.take_plan()
    rpc.queried.clear()
    assert sched.get_num_new_matched_tokens(b, 0) == (None, False)
    assert (rpc.queried, rpc.released) == ([], [b"B"])


def _forge(items):
    a, b = items[0], items[1]
    return [SaveItem(a.req_id, a.group, a.kind, a.block, b.block_id, a.block_hash)]


def test_choose_saves_rejects_a_forged_item():
    """M2: a policy cannot store one block's KV under another block's hash."""
    sched, _ = _sched(StubSemantics, {}, _Policy(choose=_forge))
    sched.on_new_request(_req())
    with pytest.raises(ValueError):
        sched.plan_saves([("r", 0, 13)], lambda _: (WINDOW_IDS, FULL_IDS))


# --------------------------------------------------------------------- worker
class FakeLoadState:
    """``polls``: is_ready() calls answering False before it turns ready."""

    def __init__(self, ready=True, state=1, polls=0):
        self.ready, self.state, self.polls = ready, state, polls

    def shm_name(self):
        return "shm"

    def is_ready(self):
        if self.polls:
            self.polls -= 1
            return False
        return self.ready

    def get_state(self):
        return self.state


class FakeMark:
    """Compute-stream mark: ``synchronize`` blocks until the test releases it."""

    def __init__(self, gate=None):
        self.gate = gate

    def synchronize(self):
        if self.gate is not None:
            assert self.gate.wait(5), "test never released the compute mark"


class JumpClock:
    """Advances ``step`` seconds per read: any deadline in the code expires."""

    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _worker(sem, rpc, states, layout=None, mark=FakeMark, clock=None, par=None):
    it = iter(states)
    kwargs = {"clock": clock} if clock is not None else {}
    return Worker(
        layout or _layout(sem),
        rpc,
        "inst",
        par or Parallel(),
        0,
        lambda t: b"ipc",
        lambda: next(it),
        mark,
        **kwargs,
    )


def test_worker_loads_and_poll():
    rpc = FakeRpc()
    rpc.load_ok[b"bad"] = False
    pending = FakeLoadState(ready=False)
    # One state per load RPC: ok, bad-lease (unused), gpu-error, later.
    states = [FakeLoadState(), FakeLoadState(), FakeLoadState(state=-1), pending]
    worker = _worker(StubSemantics, rpc, states)
    plan = StepPlan(
        loads=[
            LoadPlan("fail", fail=True),
            LoadPlan("empty"),
            LoadPlan("ok", [(b"L0", [[1]])]),
            LoadPlan("bad-lease", [(b"bad", [[2]])]),  # S1: does not fail "ok"
            LoadPlan("gpu-error", [(b"L2", [[3]])]),
            LoadPlan("later", [(b"L3", [[4]])]),
        ]
    )
    worker.start(plan)
    assert len(rpc.loads) == 4  # one RPC per request with something to load
    _, finished, failed = worker.poll()
    assert failed == {"fail", "bad-lease", "gpu-error"}
    assert finished == {"fail", "empty", "ok", "bad-lease", "gpu-error"}
    pending.ready = True
    assert worker.poll() == (set(), {"later"}, set())
    assert worker.poll() == (set(), set(), set())  # each load is reported once


@pytest.mark.parametrize(
    "spec,error",
    [
        pytest.param(StubFullSpec(), None, id="ok"),
        # S2: a declaration must equal what the server copies per segment.
        pytest.param(StubFullSpec(segments=(64, 32)), "!= server copy size", id="short-segment"),
    ],
)
def test_worker_register(spec, error):
    rpc = FakeRpc()
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, spec))
    kv = {"layers.0.attn": "tensor"}
    if error:
        with pytest.raises(Unsupported, match=error):
            worker.register(kv, {"layers.0.attn": spec})
        return
    worker.register(kv, {"layers.0.attn": spec})
    reg = rpc.registered
    assert (reg["names"], reg["segments"], reg["group_ids"]) == (["layers.0.attn"], [2], [0])
    assert (reg["backend"], reg["page_first"]) == (spec.backend, False)


def test_worker_save_by_storage_group():
    saved = []
    rpc = FakeRpc()
    rpc.save = lambda *args: saved.append(args[-1]) or (True, "")
    worker = _worker(StubSemantics, rpc, [])
    worker.register(
        {"layers.0.attn": "w", "layers.1.attn": "f"},
        {"layers.0.attn": StubWindowSpec(), "layers.1.attn": StubFullSpec()},
    )
    worker.save(StepPlan(saves=[GroupSave(1, [21], [b"h1"]), GroupSave(0, [11], [b"h1"])]))
    worker.save(StepPlan())
    assert worker.flush(5)
    assert saved == [[("layers.0.attn", [21], [b"h1"]), ("layers.1.attn", [11], [b"h1"])]]
    worker.shutdown()


class _Panic(BaseException):
    """Like pyo3's PanicException: escapes ``except Exception``."""


def _save_outcome(outcome):
    def save(calls, saves):
        calls.append(saves)
        if outcome == "rejected":  # InvalidArgument: the binding raises ValueError
            raise ValueError("save RPC failed: code: 'Client specified an invalid argument'")
        if outcome == "raises":  # any other status: a copy may still run
            raise RuntimeError("save RPC failed: code: 'Internal error'")
        if outcome == "panics":  # a binding panic, possibly after the request was sent
            raise _Panic("rust panic in save")
        return outcome == "saved", ""

    return save


@pytest.mark.parametrize("outcome", ["saved", "save-failed", "rejected", "raises", "panics"])
def test_freed_block_is_not_reused_while_a_save_reads_it(outcome):
    """vLLM gets a finished request's blocks back at once (request_finished is
    always False, no finished_sending hold), but a block a save still reads
    stays pinned in vLLM's block pool, so vLLM cannot hand it to another
    request (or zero or copy into it) until every worker reports the save done.

    The step thread never waits on the GPU for a save: the save RPC runs
    after the compute mark. The pin clears once the save is done, stored or
    not (a reported failure, or a rejected request, is a future miss).
    S4: any other raised save RPC, a binding panic (BaseException)
    included, is not a finished copy (the server may still copy): the pin
    holds for the quiesce window, saves submitted meanwhile are dropped,
    then the job counts done and saving resumes.
    """
    import threading

    gate, window = threading.Event(), threading.Event()
    rpc = FakeRpc()
    calls = []
    save = _save_outcome(outcome)
    rpc.save = lambda *args: save(calls, args[-1])
    layout = _layout(StubSemantics, StubFullSpec())
    worker = _worker(StubSemantics, rpc, [], layout=layout, mark=lambda: FakeMark(gate))
    worker._quiesce_wait = lambda s: window.wait(5)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})

    pool = FakePool()
    sched, _ = _sched(StubSemantics, {}, layout=layout, pool=pool)
    for rid in ("saver", "idle"):
        sched.on_new_request(_req(req_id=rid))
    tables = {"saver": [10, 11, 12, 13], "idle": [20]}
    sched.plan_saves([("saver", 0, 13), ("idle", 0, 3)], lambda rid: (tables[rid],))
    step0 = sched.take_plan()
    worker.save(step0)  # returns although the compute mark is not reached
    assert calls == []
    assert sched.has_pending_saves()  # vLLM keeps stepping until it is reported

    # Both finish: neither is held, only the saved blocks stay pinned.
    assert sched.request_finished("saver") is False
    assert sched.request_finished("idle") is False
    sched.on_saved_through(worker.saved_through())
    assert pool.refs == {10: 1, 11: 1, 12: 1}  # the copy has not run: not reusable

    gate.set()
    if outcome in ("raises", "panics"):
        with worker._cv:
            assert worker._cv.wait_for(lambda: worker._quiescing, 5)
        sched.on_saved_through(worker.saved_through())
        assert pool.refs == {10: 1, 11: 1, 12: 1}  # the window holds the pin
        worker.save(StepPlan(saves=step0.saves, step=1))  # dropped: the raised copy may run
        window.set()
    assert worker.flush(5) and len(calls) == 1
    sched.on_saved_through(worker.saved_through())
    assert pool.refs == {} and not sched.has_pending_saves()
    worker.save(StepPlan(saves=step0.saves, step=2))  # saving resumed: the RPC is sent
    assert worker.flush(5) and len(calls) == 2
    worker.shutdown()


def test_preempted_request_finishing_while_reloading_is_not_held():
    """A preempted request re-admitted with an async load and then aborted must
    not be reported in finished_sending: vLLM would free the blocks the load
    still copies into, then fail on the load's finished_recving."""
    layout = _layout(StubSemantics, StubFullSpec())
    sched, _ = _sched(StubSemantics, {0: Ready(3, b"L0")}, layout=layout)
    pending = FakeLoadState(ready=False)
    worker = _worker(StubSemantics, FakeRpc(), [pending], layout=layout)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    req = _req()
    sched.on_new_request(req)
    polls = []

    def step(scheduled, tables=lambda rid: None):
        sched.plan_saves(scheduled, tables)
        plan = sched.take_plan()
        worker.start(plan)
        worker.save(plan)
        polls.append(worker.poll())

    step([("r", 0, 13)], lambda rid: ([10, 11, 12, 13],))  # r saves
    step([])  # r is preempted
    n, is_async = sched.get_num_new_matched_tokens(req, 0)
    assert n and is_async
    sched.update_state_after_alloc(req, ([20, 21, 22, 23],), n)
    step([])  # the load is issued
    assert sched.request_finished("r") is False  # aborted while loading
    step([])
    pending.ready = True
    step([])
    assert [sending for sending, _, _ in polls] == [set()] * 5
    assert polls[-1][1] == {"r"}
    worker.shutdown()


def test_pinned_saves_hold_blocks_until_every_worker_saved():
    """Every saved block (checkpoint state blocks included) is pinned in
    vLLM's pool from the plan until the workers report that step saved
    (min over ranks)."""
    pool = FakePool()
    layout = _layout(StubSemantics, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(StubSemantics, {}, layout=layout, pool=pool)
    sched.on_new_request(_req())
    ids = lambda _: (CKPT_IDS, FULL_IDS)  # noqa: E731
    sched.plan_saves([("r", 0, 4)], ids, offers={"r": [(0, 31, 4)]})
    p0 = sched.take_plan()
    assert (p0.step, pool.refs) == (0, {10: 1, 31: 1})
    sched.plan_saves([("r", 4, 4)], ids, offers={"r": [(0, 32, 8)]})
    p1 = sched.take_plan()
    assert (p1.step, pool.refs) == (1, {10: 1, 31: 1, 11: 1, 32: 1})
    assert sched.request_finished("r") is False
    sched.on_saved_through(0)
    assert pool.refs == {11: 1, 32: 1} and sched.has_pending_saves()
    sched.on_saved_through(5)
    assert pool.refs == {} and not sched.has_pending_saves()
    # No pool bound: nothing is saved (fail closed).
    sched = Scheduler(
        _layout(StubSemantics, StubFullSpec()), Policy(), FakeRpc(), "inst", semantics=StubSemantics
    )
    sched.on_new_request(_req())
    sched.plan_saves([("r", 0, 13)], lambda _: (FULL_IDS,))
    assert sched.take_plan().saves == [] and not sched.has_pending_saves()


def test_worker_reports_saved_through():
    """saved_through (the only pin release) is the last plan step with every
    save before it done: never past a pending job, and a step without saves
    counts once the earlier ones are done."""
    import threading

    gates = [threading.Event(), threading.Event()]
    marks = iter(gates)
    worker = _worker(StubSemantics, FakeRpc(), [], layout=_layout(StubSemantics, StubFullSpec()))
    worker._mark_compute = lambda: FakeMark(next(marks))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    for plan in (_save_plan(0), StepPlan(step=1), _save_plan(2)):
        worker.save(plan)
    assert worker.saved_through() == -1
    gates[0].set()
    with worker._cv:
        assert worker._cv.wait_for(lambda: worker._completed >= 1, 5)
    assert worker.saved_through() == 1  # job 2 still pending
    gates[1].set()
    assert worker.flush(5) and worker.saved_through() == 2
    worker.shutdown()


def _save_plan(step):
    return StepPlan(saves=[GroupSave(0, [step], [b"h%d" % step])], step=step)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize("error", [RuntimeError, _Panic], ids=["failed", "saver-dies"])
def test_failed_compute_wait_sends_no_save(error):
    """The forward failed: its KV is not valid, so no save RPC is sent and the
    job completes as failed (its blocks are released). A BaseException there
    ends the saver thread; no copy runs, so holds are released too and later
    saves are dropped."""

    class BadMark:
        def synchronize(self):
            raise error("CUDA error: an illegal memory access")

    rpc, calls = FakeRpc(), []
    rpc.save = lambda *args: calls.append(args) or (True, "")
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker._mark_compute = BadMark
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.save(_save_plan(0))
    assert worker.flush(5) and calls == [] and worker.saved_through() == 0
    worker.save(_save_plan(1))  # dropped once the saver died
    assert worker.flush(5) and calls == [] and worker.saved_through() == 1
    worker._saver.join(0 if error is RuntimeError else 5)
    assert worker._saver.is_alive() is (error is RuntimeError)
    worker.shutdown()


def test_worker_metadata_aggregates_the_min_over_ranks():
    """vLLM folds every worker's metadata into one: pins clear only
    once every rank saved the step."""
    try:  # a real vLLM: other test modules install partial vllm stubs
        from pegaflow.connector_v2.connector import PegaWorkerMetadata
    except ImportError:
        pytest.skip("needs vLLM")

    got = PegaWorkerMetadata(5).aggregate(PegaWorkerMetadata(3)).aggregate(PegaWorkerMetadata(7))
    assert got.saved_through == 3


# --------------------------------------------------------------------- layout
@BOTH
@pytest.mark.parametrize(
    "specs,par,match",
    [
        pytest.param(
            (StubFullSpec(), MysterySpec()),
            Parallel(),
            r"no adapter for MysterySpec \(group 1\)",
            id="unknown-spec",
        ),
        pytest.param((StubWindowSpec(),), Parallel(), "no full-attention group", id="window-only"),
        pytest.param(
            (StubFullSpec(), StubWindowSpec(backend="kernel")),
            Parallel(),
            r"disagree on \(transfer_backend, page_first\)",
            id="mixed-backend",
        ),
        pytest.param(
            (StubFullSpec(backend="dma"),), Parallel(), "unknown transfer_backend", id="bad-backend"
        ),
    ],
)
def test_layout_rejects(sem, specs, par, match):
    with pytest.raises(Unsupported, match=match):
        _layout(sem, *specs, par=par)


class _RankSkewedSpec(StubFullSpec):
    pass


class _RankSkewedAdapter(_StubAdapter):
    def layer(self, spec, layer_name, par):
        segs = (64, 64) if par.tp_rank == 0 else (64, 32)
        return LayerDecl(segs, self.sem.Shard.heads(par.tp_rank, par.tp_size))


@BOTH
def test_layout_runs_seal_plan(sem, monkeypatch):
    """S2 / V0: TP ranks declaring different segment sizes fail seal_plan."""
    module = types.ModuleType(f"{adapters.__name__}._test_skewed")
    module.ADAPTER = _RankSkewedAdapter(sem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(adapters._REGISTRY, "_RankSkewedSpec", "_test_skewed")
    with pytest.raises(Unsupported, match="fail seal_plan"):
        _layout(sem, _RankSkewedSpec(), par=Parallel(tp_size=2))


# ----------------------------------------------------------- stub vs native
def test_stub_matches_native():
    native = _native()
    rng = random.Random(0)
    kinds = [("full", ()), ("window", (7,)), ("window", (12,)), ("checkpoint", ()), ("scratch", ())]
    for _ in range(300):
        picks = [rng.choice(kinds) + (rng.choice([2, 4, 8]),) for _ in range(rng.randint(1, 3))]
        stub_g = [getattr(_Group, k)(bt, *a) for k, a, bt in picks]
        nat_g = [getattr(native.Group, k)(bt, *a) for k, a, bt in picks]
        present = [{b for b in range(16) if rng.random() < 0.75} for _ in picks]
        prompt_len = rng.randrange(0, 64)
        assert _resolve(stub_g, present, prompt_len) == native.resolve(nat_g, present, prompt_len)
        for s, n in zip(stub_g, nat_g, strict=True):
            assert (s.kind, s.block_tokens, s.window_tokens) == (
                n.kind,
                n.block_tokens,
                n.window_tokens,
            )
            before = rng.randrange(0, 40)
            after = before + rng.randrange(0, 20)
            assert _needed(s, after) == tuple(native.needed(n, after))
            assert _saveable(s, before, after) == tuple(native.saveable(n, before, after))


# ------------------------------------------------- adapters/full_attention.py
@dataclass(frozen=True)
class FullAttentionSpec:  # duck-typed vLLM 0.32 spec; the registry keys on the class name
    """Defaults match Qwen3-4B's full-attention spec: bf16, block 16, 8 KV heads x 128."""

    block_size: int = 16
    num_kv_heads: int = 8
    head_size: int = 128
    page_size_bytes: int = 65536
    max_tp_shards: int | None = None
    dcp_sharded: bool = True
    has_layer_views: bool = True
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    page_size_padded: int | None = None
    num_head_slots: int | None = None
    state_content_bytes: int | None = None
    tokens_per_state: int = 1
    kv_quant_mode: int = 0
    non_causal: bool = False


@dataclass
class FakeKV:
    """The bits of a torch.Tensor the adapter reads; strides in elements."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    elem: int = 2

    def stride(self):
        return self.strides

    def element_size(self):
        return self.elem


QWEN3_BLOCKS = 4745
# vLLM 0.32 LBNHC layer view [B, H, N, C/2] in bf16 (probe spec dump, 4090 run).
QWEN3_KV = FakeKV((QWEN3_BLOCKS, 8, 16, 256), (32768, 256, 2048, 1))


@pytest.fixture
def full_attention(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import full_attention as mod

    adapter = mod.FullAttentionAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


def _attn_cfg(spec, n_layers, kv_cache_layout="LBNHC"):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(n_layers)]
    group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=names)
    return names, types.SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[group],
        hash_block_size=16,
        kv_cache_layout=kv_cache_layout,
    )


# Per-rank page for 4 of the 8 kv heads (TP2, or TP4 over 2 head shards).
HALF_SPEC = FullAttentionSpec(num_kv_heads=4, page_size_bytes=32768)
HALF_KV = FakeKV((QWEN3_BLOCKS, 4, 16, 256), (16384, 256, 1024, 1))


@BOTH
@pytest.mark.parametrize(
    "spec,par,kv,shards",
    [
        # Qwen3-4B as run by the gate: 36 layers, TP1, LBNHC.
        pytest.param(FullAttentionSpec(), Parallel(), QWEN3_KV, [(0, 1)], id="tp1"),
        pytest.param(HALF_SPEC, Parallel(tp_size=2), HALF_KV, [(0, 2), (1, 2)], id="tp2"),
    ],
)
def test_full_attention_layout(sem, full_attention, spec, par, kv, shards):
    names, cfg = _attn_cfg(spec, 36)
    layout = build_layout(cfg, par, {"model": "Qwen3-4B"}, semantics=sem)  # runs seal_plan
    assert [(g.sem.kind, g.sem.block_tokens, g.hash_scale) for g in layout.groups] == [
        ("full", 16, 1)
    ]
    ranks = [dataclasses.replace(par, tp_rank=r) for r in range(par.tp_size)]
    decls = [full_attention.layer(spec, names[0], r) for r in ranks]
    assert [d.shard for d in decls] == [sem.Shard.heads(i, n) for i, n in shards]
    # One storage rank per head shard; its first holder writes it.
    first = {i: r for r, (i, _) in reversed(list(enumerate(shards)))}
    assert layout.storage == StorageRanks(
        shards[0][1],
        tuple(i for i, _ in shards),
        tuple(first[i] == r for r, (i, _) in enumerate(shards)),
    )
    # Rank 0 registration as the server sees it: one page-sized segment per layer.
    rpc = FakeRpc()
    _worker(sem, rpc, [], layout=layout).register(
        dict.fromkeys(names, kv), dict.fromkeys(names, spec)
    )
    reg = rpc.registered
    page = spec.page_size_bytes
    assert reg["names"] == names
    assert set(reg["segments"]) == {1}
    assert set(reg["bytes_per_block"]) == set(reg["kv_strides"]) == {page}
    assert set(reg["num_blocks"]) == {QWEN3_BLOCKS}
    assert (reg["backend"], reg["page_first"], set(reg["group_ids"])) == ("direct", False, {0})


@BOTH
@pytest.mark.parametrize(
    "change,par,kv",
    [
        ({"sliding_window": 4096}, Parallel(), QWEN3_KV),
        ({"page_size_padded": 69632}, Parallel(), QWEN3_KV),
        ({"kv_quant_mode": 1}, Parallel(), QWEN3_KV),
        ({"num_head_slots": 4}, Parallel(), QWEN3_KV),
        ({"non_causal": True}, Parallel(), QWEN3_KV),
        ({"has_layer_views": False}, Parallel(), QWEN3_KV),
        # vLLM says ranks replicate (kv_tp_replicas) but the spec declares none.
        pytest.param(
            {"max_tp_shards": None},
            Parallel(tp_size=4, kv_tp_replicas=2),
            QWEN3_KV,
            id="undeclared-replicas",
        ),
        # M1: a split [H, N, C] page is stored as 2 x [H, N/2, C], which the
        # namespace (kv_cache_layout) does not tell apart from the unsplit page.
        pytest.param(
            {}, Parallel(), FakeKV((20, 8, 8, 256), (16384, 2048, 256, 1)), id="LBHNC-split"
        ),
        # Block-outer layouts put the other layers' pages between our blocks.
        pytest.param(
            {}, Parallel(), FakeKV((10, 8, 16, 256), (36 * 32768, 256, 2048, 1)), id="BLNHC"
        ),
        pytest.param({}, Parallel(), FakeKV((10, 8, 16, 256), (32768, 256, 4096, 1)), id="holes"),
        pytest.param(
            {}, Parallel(), FakeKV((10, 8, 16, 128), (16384, 128, 1024, 1)), id="wrong-page"
        ),
    ],
)
def test_full_attention_rejects_unvalidated_specs(sem, full_attention, change, par, kv):
    spec = dataclasses.replace(FullAttentionSpec(), **change)
    with pytest.raises(Unsupported):
        _layout_of(sem, spec, par)
        full_attention.view(spec, "l0", kv)


def _layout_of(sem, spec, par=None, kv_cache_layout="LBNHC"):
    _, cfg = _attn_cfg(spec, 2, kv_cache_layout)
    return build_layout(cfg, par or Parallel(), {"model": "Qwen3-4B"}, semantics=sem)


# ---------------------------------------------------------------- adapters/mamba.py
@dataclass(frozen=True)
class MambaSpec:  # duck-typed vLLM 0.32 spec; the registry keys on the class name
    """Defaults match Qwen3.5-4B's GDN spec: align mode, page padded."""

    block_size: int = 528
    shapes: tuple[tuple[int, ...], ...] = ((3, 8192), (32, 128, 128))
    dtypes: tuple[str, ...] = ("bfloat16", "float32")
    page_size_bytes: int = 2162688
    mamba_cache_mode: str = "align"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None
    num_heads: int = 1
    tokens_per_state: int = -1
    tp_replicated: bool = False
    has_layer_views: bool = True


QWEN35_BLOCKS = 481
QWEN35_REAL = 2146304
# vLLM 0.32 raw Mamba layer view [B, 1, 1, real] int8, pages padded to 2162688.
QWEN35_KV = FakeKV((QWEN35_BLOCKS, 1, 1, QWEN35_REAL), (2162688, QWEN35_REAL, QWEN35_REAL, 1), 1)


@pytest.fixture
def mamba(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import mamba as mod

    adapter = mod.MambaAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


@BOTH
def test_mamba_declares_one_padded_page_checkpoint(sem, mamba):
    spec = MambaSpec()
    g = mamba.group(spec)
    assert (g.kind, g.block_tokens) == ("checkpoint", 528)
    decl = mamba.layer(spec, "layers.0.linear_attn", Parallel())
    # The copy unit is the block stride: the server addresses data_ptr + b * bytes_per_block.
    assert decl.segments == (2162688,)
    assert decl.layout == (((3, 8192), "bfloat16"), ((32, 128, 128), "float32"))
    view = mamba.view(spec, "layers.0.linear_attn", QWEN35_KV)
    assert (view.num_blocks, view.bytes_per_block, view.kv_stride_bytes) == (
        QWEN35_BLOCKS,
        2162688,
        2162688,
    )
    tp2 = [mamba.layer(spec, "l", Parallel(tp_rank=r, tp_size=2)).shard for r in (0, 1)]
    assert tp2 == [sem.Shard.heads(0, 2), sem.Shard.heads(1, 2)]
    replicated = dataclasses.replace(spec, tp_replicated=True)
    assert mamba.layer(replicated, "l", Parallel(tp_size=2)).shard == sem.Shard.replicated()


def _ssm_kv(shape, strides):
    return FakeKV(shape, strides, 1)


@BOTH
@pytest.mark.parametrize(
    "change,kv",
    [
        ({"mamba_cache_mode": "none"}, QWEN35_KV),
        ({"num_speculative_blocks": 1}, QWEN35_KV),
        ({"num_prefill_checkpoint_blocks": 2}, QWEN35_KV),
        ({"tokens_per_state": 16}, QWEN35_KV),
        ({"has_layer_views": False}, QWEN35_KV),
        pytest.param({"page_size_bytes": 2146303}, QWEN35_KV, id="states-overflow-page"),
        # Block-outer layouts put other layers' pages between our blocks.
        pytest.param(
            {},
            _ssm_kv((10, 1, 1, QWEN35_REAL), (8 * 2162688, QWEN35_REAL, QWEN35_REAL, 1)),
            id="block-outer",
        ),
        pytest.param(
            {}, _ssm_kv((10, 1, 1, 2097152), (2162688, 2097152, 2097152, 1)), id="ssm-only"
        ),
        pytest.param({}, _ssm_kv((10, 3, 8192), (2162688, 8192, 1)), id="not-4d"),
    ],
)
def test_mamba_rejects_unvalidated_specs(sem, mamba, change, kv):
    spec = dataclasses.replace(MambaSpec(), **change)
    with pytest.raises(Unsupported):
        mamba.group(spec)
        mamba.view(spec, "l0", kv)


def _qwen35_cfg(tp=1, **change):
    """Qwen3.5-4B layout: 3 GDN groups + 1 attention group, block 528."""
    attn = FullAttentionSpec(
        block_size=528, num_kv_heads=4 // tp, head_size=256, page_size_bytes=2162688 // tp
    )
    groups = [
        types.SimpleNamespace(
            kv_cache_spec=dataclasses.replace(MambaSpec(), **change),
            layer_names=[f"layers.{i}.linear_attn" for i in range(g, 32, 4)],
        )
        for g in range(3)
    ]
    groups.append(
        types.SimpleNamespace(
            kv_cache_spec=attn, layer_names=[f"layers.{i}.self_attn.attn" for i in range(3, 32, 4)]
        )
    )
    return types.SimpleNamespace(
        transfer_group_ids=(0, 1, 2, 3),
        kv_cache_groups=groups,
        hash_block_size=528,
        kv_cache_layout="LBNHC",
    )


@BOTH
@pytest.mark.parametrize("tp", [1, 2])
def test_qwen35_layout(sem, mamba, full_attention, tp):
    """Full group first (prefix query), then the 3 GDN checkpoint groups; step
    528. V0: every rank's GDN + attention regions seal (build_layout runs
    seal_plan over all TP ranks)."""
    par = Parallel(tp_size=tp)
    layout = build_layout(_qwen35_cfg(tp), par, {"model": "Qwen3.5-4B"}, semantics=sem)
    assert [(g.gid, g.sem.kind) for g in layout.groups] == [
        (3, "full"),
        (0, "checkpoint"),
        (1, "checkpoint"),
        (2, "checkpoint"),
    ]
    assert layout.step == 528


# -------------------------------------------------- adapters/mla_attention.py
@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):  # registry keys on the class name
    """Defaults match DeepSeek-V2-Lite-Chat's MLA spec (bf16 latent
    kv_lora_rank 512 + rope 64 = 576, one head, replicated over TP)."""

    num_kv_heads: int = 1
    head_size: int = 576
    head_size_v: int = 0
    page_size_bytes: int = 18432
    max_tp_shards: int | None = 1
    cache_dtype_str: str | None = "auto"
    alignment: int | None = None
    model_version: str | None = None
    cache_role: str = "sparse"
    is_index_group_leader: bool = False
    non_causal_multi_token_decode: bool = False


DSV2_BLOCKS = 6222
# vLLM 0.32 LBNHC layer view [B, H=1, N, C=576] in bf16 (probe dump, both TP2 ranks).
DSV2_KV = FakeKV((DSV2_BLOCKS, 1, 16, 576), (9216, 576, 576, 1))


@pytest.fixture
def mla_attention(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import mla_attention as mod

    adapter = mod.MLAAttentionAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


@BOTH
@pytest.mark.parametrize(
    "par",
    [
        pytest.param(Parallel(), id="tp1"),
        # DeepSeek-V2-Lite as run by the gate: TP2, vLLM kv_tp_replicas=2.
        pytest.param(Parallel(tp_size=2, kv_tp_replicas=2), id="tp2"),
    ],
)
def test_mla_attention_layout(sem, mla_attention, par):
    spec = MLAAttentionSpec()
    names, cfg = _attn_cfg(spec, 27)
    layout = build_layout(cfg, par, {"model": "DeepSeek-V2-Lite-Chat"}, semantics=sem)
    assert [(g.sem.kind, g.sem.block_tokens) for g in layout.groups] == [("full", 16)]
    ranks = [dataclasses.replace(par, tp_rank=r) for r in range(par.tp_size)]
    # Every rank holds the same latent page: one fragment per layer after seal_plan.
    assert {mla_attention.layer(spec, names[0], r).shard for r in ranks} == {sem.Shard.replicated()}
    # Replicas share one storage rank: one slot per layer, only rank 0 saves.
    n = par.tp_size
    assert layout.storage == StorageRanks(1, (0,) * n, (True,) + (False,) * (n - 1))
    for r in ranks:
        rpc = FakeRpc()
        saved = []
        rpc.save = lambda *args, saved=saved: saved.append(args) or (True, "")
        loaded = []
        rpc.load = lambda *args, loaded=loaded: loaded.append(args[1]) or (True, "")
        worker = _worker(sem, rpc, [FakeLoadState()], layout=layout, par=r)
        worker.register(dict.fromkeys(names, DSV2_KV), dict.fromkeys(names, spec))
        reg = rpc.registered
        assert (reg["tp_rank"], reg["tp_size"]) == (0, 1)
        assert set(reg["segments"]) == {1}
        assert set(reg["bytes_per_block"]) == set(reg["kv_strides"]) == {18432}
        assert set(reg["num_blocks"]) == {DSV2_BLOCKS}
        assert (reg["backend"], reg["page_first"]) == ("direct", False)
        worker.save(StepPlan(saves=[GroupSave(0, [11], [b"h1"])]))
        assert worker.flush(5)
        assert [(a[1], len(a[-1])) for a in saved] == ([(0, 27)] if r.tp_rank == 0 else [])
        worker.start(StepPlan(loads=[LoadPlan("q", [(b"L0", [[3]])], sync=False)]))
        assert loaded == [0]  # every holder loads its own GPU from storage rank 0
        worker.shutdown()


@BOTH
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tp_shards", None),
        ("max_tp_shards", 2),
        ("alignment", 576),
        ("model_version", "deepseek_v4"),
        ("cache_role", "indexer"),
        ("is_index_group_leader", True),
        ("cache_dtype_str", "fp8_ds_mla"),
        ("state_content_bytes", 656),
        ("kv_quant_mode", 1),
        ("page_size_padded", 18944),
        ("non_causal_multi_token_decode", True),
        ("sliding_window", 4096),
        ("head_size_v", 128),
    ],
)
def test_mla_attention_rejects_unvalidated_specs(sem, mla_attention, field, value):
    spec = dataclasses.replace(MLAAttentionSpec(), **{field: value})
    _, cfg = _attn_cfg(spec, 2)
    with pytest.raises(Unsupported):
        build_layout(cfg, Parallel(tp_size=2, kv_tp_replicas=2), {"model": "m"}, semantics=sem)


# ------------------------------------------------------------------ namespace
def _qwen35_layout(sem, **change):
    return build_layout(_qwen35_cfg(**change), Parallel(), {"model": "Qwen3.5-4B"}, semantics=sem)


@BOTH
@pytest.mark.parametrize(
    "a,b",
    [
        pytest.param(
            lambda sem: _layout(sem, StubFullSpec()),
            lambda sem: _layout(sem, StubFullSpec(backend="kernel")),
            id="transfer-mode",
        ),
        # M1: [N, H, C] and [H, N, C] pages hold different bytes for one declaration.
        pytest.param(
            lambda sem: _layout_of(sem, FullAttentionSpec(), kv_cache_layout="LBNHC"),
            lambda sem: _layout_of(sem, FullAttentionSpec(), kv_cache_layout="LBHNC"),
            id="page-byte-order",
        ),
        # Equal page sizes, different bytes: the spec class is part of the namespace.
        pytest.param(
            lambda sem: _layout_of(sem, MLAAttentionSpec()),
            lambda sem: _layout_of(
                sem, FullAttentionSpec(num_kv_heads=1, head_size=288, page_size_bytes=18432)
            ),
            id="mla-vs-full",
        ),
        # The Mamba state layout enters the namespace: another state dtype, or
        # VLLM_SSM_CONV_STATE_LAYOUT=DS (same page size, other byte order).
        pytest.param(
            lambda sem: _qwen35_layout(sem),
            lambda sem: _qwen35_layout(sem, dtypes=("bfloat16", "bfloat16")),
            id="mamba-state-dtype",
        ),
        pytest.param(
            lambda sem: _qwen35_layout(sem),
            lambda sem: _qwen35_layout(sem, shapes=((8192, 3), (32, 128, 128))),
            id="mamba-conv-layout",
        ),
    ],
)
def test_namespace_separates_incompatible_bytes(sem, full_attention, mla_attention, mamba, a, b):
    assert a(sem).namespace == a(sem).namespace
    assert a(sem).namespace != b(sem).namespace
