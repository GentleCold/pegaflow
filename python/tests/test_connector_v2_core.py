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
from pegaflow.connector_v2.core import Scheduler, StepPlan, Worker, build_layout
from pegaflow.connector_v2.core.layout import _tp_ranks
from pegaflow.connector_v2.core.scheduler import GroupSave, LoadPlan
from pegaflow.connector_v2.policy import Policy, SaveItem, load_policy

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


def _seal_plan(ranks):
    """Only the checks the core relies on: equal segments, every part held."""
    layers: dict[str, tuple[list[int], int, set[int]]] = {}
    for _, regions in ranks:
        for r in regions:
            segs, count, parts = layers.setdefault(r.layer, (r.segments, r.shard.count, set()))
            if segs != r.segments or count != r.shard.count:
                raise ValueError(f"layer {r.layer}: segment sizes differ")
            parts.add(r.shard.index)
    for layer, (_, count, parts) in layers.items():
        if parts != set(range(count)):
            raise ValueError(f"layer {layer}: missing part")
    return object()


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
    pytest.importorskip("pegaflow.pegaflow", exc_type=ImportError)
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
        self.loads: list[list] = []
        self.load_ok: dict[bytes, bool] = {}
        self.registered: dict | None = None

    def query_prefetch(self, instance_id, hashes, req_id, wait_for_full_prefix, group_id):
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


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _req(num_tokens=13, n_hashes=3, req_id="r"):
    hashes = [f"h{i}".encode() for i in range(n_hashes)]
    return types.SimpleNamespace(request_id=req_id, num_tokens=num_tokens, block_hashes=hashes)


def _sched(sem, answers, policy=None, layout=None):
    rpc = FakeRpc(answers)
    layout = layout or _layout(sem)
    sched = Scheduler(layout, policy or Policy(), rpc, "inst", clock=Clock(), semantics=sem)
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
        pytest.param(HIT, Policy(min_hit_tokens=16), 0, (0, False), [b"L0", b"L1"], id="min-hit"),
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
        # Lowering p is re-checked: at p=8 the window needs block 0 (absent).
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(2, b"L1", [1, 2])},
            _Policy(accept=lambda p, _: 8),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="policy-lowers-into-gap",
        ),
        # Lowering to a legal p keeps only the leases that p still needs.
        pytest.param(HIT, _Policy(accept=lambda p, _: 8), 0, (8, True), [], id="policy-lowers"),
        pytest.param(
            HIT, _Policy(accept=lambda p, _: 6), 0, (0, False), [b"L0", b"L1"], id="policy-off-step"
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

    # Hit that never gets allocated (e.g. aborted while waiting).
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    sched.request_finished("r")
    assert sorted(rpc.released) == [b"L0", b"L1"]

    # Loading wait expiry falls back to a miss.
    rpc.answers = {0: Loading(), 1: HIT[1]}
    rpc.released.clear()
    sched = Scheduler(
        layout, Policy(loading_wait_ms=100), rpc, "inst", clock=Clock(), semantics=sem
    )
    assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
    sched._clock.now += 0.2
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


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
        pytest.param(6, (WINDOW_IDS, FULL_IDS), True, [], [b"L0", b"L1"], id="trimmed-off-step"),
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


def test_shutdown_releases_unsent_plans():
    sched, rpc = _sched(StubSemantics, HIT)
    req = _req()
    sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, (WINDOW_IDS, FULL_IDS), 12)
    sched.shutdown()
    assert sorted(rpc.released) == [b"L0", b"L1"]


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
        pytest.param(13, [[("gone", 0, 13)]], {}, [[]], id="unknown-request"),
    ],
)
def test_saves(sem, num_tokens, steps, block_ids, expected):
    sched, _ = _sched(sem, {})
    sched.on_new_request(_req(num_tokens=num_tokens, n_hashes=num_tokens // BT))
    assert _step_saves(sched, steps, block_ids) == expected


def test_saves_reoffer_after_missing_block_ids():
    sched, _ = _sched(StubSemantics, {})
    sched.on_new_request(_req())
    ids: dict = {}
    assert _step_saves(sched, [[("r", 0, 8)]], ids) == [[]]
    ids["r"] = (WINDOW_IDS, FULL_IDS)
    assert _step_saves(sched, [[("r", 8, 5)]], ids) == [
        [(0, [10, 11, 12], H[:3]), (1, [20, 21, 22], H[:3])]
    ]


def _forge(items):
    a, b = items[0], items[1]
    return [SaveItem(a.req_id, a.group, a.kind, a.block, b.block_id, a.block_hash)]


def _append(items):
    items.append(items[0])
    return items


@pytest.mark.parametrize(
    "choose,expected",
    [
        pytest.param(lambda items: [], [], id="save-none"),
        pytest.param(
            lambda items: [i for i in items if i.kind == "full"], [(0, [10, 11, 12])], id="subset"
        ),
        pytest.param(lambda items: [items[0], items[0]], [(0, [10])], id="duplicate-once"),
        # M2: a forged (block, hash) pair or an in-place edit fails loudly.
        pytest.param(_forge, ValueError, id="forged-item"),
        pytest.param(_append, AttributeError, id="mutates-items"),
    ],
)
def test_choose_saves_is_validated(choose, expected):
    sched, _ = _sched(StubSemantics, {}, _Policy(choose=choose))
    sched.on_new_request(_req())
    plan = lambda: sched.plan_saves([("r", 0, 13)], lambda _: (WINDOW_IDS, FULL_IDS))  # noqa: E731
    if isinstance(expected, type):
        with pytest.raises(expected):
            plan()
        return
    plan()
    assert [(s.sid, s.block_ids) for s in sched.take_plan().saves] == expected


# --------------------------------------------------------------------- worker
class FakeLoadState:
    def __init__(self, ready=True, state=1):
        self.ready, self.state = ready, state

    def shm_name(self):
        return "shm"

    def is_ready(self):
        return self.ready

    def get_state(self):
        return self.state


def _worker(sem, rpc, states, layout=None):
    it = iter(states)
    return Worker(
        layout or _layout(sem),
        rpc,
        "inst",
        Parallel(),
        0,
        lambda t: b"ipc",
        lambda: next(it),
        lambda: None,
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
    worker.start_loads(plan)
    assert len(rpc.loads) == 4  # one RPC per request with something to load
    finished, failed = worker.poll()
    assert failed == {"fail", "bad-lease", "gpu-error"}
    assert finished == {"fail", "empty", "ok", "bad-lease", "gpu-error"}
    pending.ready = True
    assert worker.poll() == ({"later"}, set())
    assert worker.poll() == (set(), set())


@pytest.mark.parametrize(
    "spec,error",
    [
        pytest.param(StubFullSpec(), None, id="ok"),
        # S2: a declaration must equal what the server copies per segment.
        pytest.param(StubFullSpec(segments=(64, 32)), "!= server copy size", id="short-segment"),
        pytest.param(StubFullSpec(segments=(128,)), "!= server copy size", id="whole-block"),
        pytest.param(StubFullSpec(backend="kernel"), None, id="kernel-backend"),
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
    assert saved == [[("layers.0.attn", [21], [b"h1"]), ("layers.1.attn", [11], [b"h1"])]]


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
        # M3: fail closed until the core handles boundary_state_offloads (P5).
        pytest.param(
            (StubFullSpec(), StubCheckpointSpec()),
            Parallel(),
            "boundary_state_offloads",
            id="checkpoint",
        ),
        pytest.param((StubWindowSpec(),), Parallel(), "no full-attention group", id="window-only"),
        # S2 / V0: TP2 ranks declaring different segment sizes fail seal_plan.
        pytest.param(
            (StubFullSpec(segments=(64, 64)),),
            Parallel(tp_size=2),
            None,
            id="tp2-ok",
        ),
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
    if match is None:
        _layout(sem, *specs, par=par)
        return
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
    module = types.ModuleType(f"{adapters.__name__}._test_skewed")
    module.ADAPTER = _RankSkewedAdapter(sem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(adapters._REGISTRY, "_RankSkewedSpec", "_test_skewed")
    with pytest.raises(Unsupported, match="fail seal_plan"):
        _layout(sem, _RankSkewedSpec(), par=Parallel(tp_size=2))


def test_namespace_covers_transfer_mode():
    a = _layout(StubSemantics, StubFullSpec())
    b = _layout(StubSemantics, StubFullSpec(backend="kernel"))
    assert (a.transfer_backend, b.transfer_backend) == ("direct", "kernel")
    assert a.namespace != b.namespace
    assert a.namespace == _layout(StubSemantics, StubFullSpec()).namespace


# --------------------------------------------------------------------- policy
def test_policy_defaults_match_shipped_toml():
    assert load_policy(None) == Policy()


@pytest.mark.parametrize("profile", ["../x", "a/b", "x y"])
def test_policy_profile_name_is_checked(profile):
    with pytest.raises(ValueError, match="model_profile"):
        load_policy(profile)


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
    """Defaults = docs/rsi/models/Qwen3-4B/spec.json (bf16, 8 kv heads x 128)."""

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


@BOTH
@pytest.mark.parametrize(
    "kv,num_blocks",
    [
        pytest.param(QWEN3_KV, QWEN3_BLOCKS, id="qwen3-4b-LBNHC"),
        pytest.param(FakeKV((10, 8, 16, 256), (32768, 4096, 256, 1)), 10, id="LBHNC"),
        # Kernel blocks of 8 tokens split each 16-token manager block in two.
        pytest.param(FakeKV((20, 8, 8, 256), (16384, 256, 2048, 1)), 10, id="kernel-split"),
        # Block-outer layouts put the other layers' pages between our blocks.
        pytest.param(FakeKV((10, 8, 16, 256), (36 * 32768, 256, 2048, 1)), None, id="BLNHC"),
        pytest.param(FakeKV((10, 8, 16, 256), (32768, 256, 4096, 1)), None, id="holes"),
        pytest.param(FakeKV((10, 8, 16, 128), (16384, 128, 1024, 1)), None, id="wrong-page"),
    ],
)
def test_full_attention_view(sem, full_attention, kv, num_blocks):
    spec = FullAttentionSpec()
    if num_blocks is None:
        with pytest.raises(Unsupported):
            full_attention.view(spec, "l0", kv)
        return
    view = full_attention.view(spec, "l0", kv)
    # One segment = the whole page: LBNHC interleaves K and V per token/head.
    assert (view.num_blocks, view.bytes_per_block, view.kv_stride_bytes) == (
        num_blocks,
        65536,
        65536,
    )
    assert view.tensor is kv


@BOTH
@pytest.mark.parametrize(
    "spec,par,shards",
    [
        pytest.param(FullAttentionSpec(), Parallel(), [(0, 1)], id="tp1"),
        pytest.param(FullAttentionSpec(), Parallel(tp_size=2), [(0, 2), (1, 2)], id="tp2"),
        # 4 ranks over 2 head shards: consecutive ranks hold replicas.
        pytest.param(
            FullAttentionSpec(max_tp_shards=2),
            Parallel(tp_size=4),
            [(0, 2), (0, 2), (1, 2), (1, 2)],
            id="tp4-replicas",
        ),
    ],
)
def test_full_attention_layout(sem, full_attention, spec, par, shards):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(3)]
    cfg = types.SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=spec, layer_names=names)],
        hash_block_size=16,
    )
    layout = build_layout(cfg, par, {"model": "Qwen3-4B"}, semantics=sem)  # runs seal_plan
    assert [(g.sem.kind, g.sem.block_tokens, g.hash_scale) for g in layout.groups] == [
        ("full", 16, 1)
    ]
    decls = [full_attention.layer(spec, names[0], r) for r in _tp_ranks(par)]
    assert {d.segments for d in decls} == {(65536,)}
    assert [d.shard for d in decls] == [sem.Shard.heads(i, n) for i, n in shards]


@BOTH
@pytest.mark.parametrize(
    "field,value",
    [
        ("sliding_window", 4096),
        ("page_size_padded", 69632),
        ("kv_quant_mode", 1),
        ("num_head_slots", 4),
        ("has_layer_views", False),
    ],
)
def test_full_attention_rejects_unvalidated_specs(sem, full_attention, field, value):
    spec = dataclasses.replace(FullAttentionSpec(), **{field: value})
    with pytest.raises(Unsupported):
        full_attention.group(spec)


@BOTH
def test_full_attention_registers_one_page_segment(sem, full_attention):
    """Qwen3-4B worker registration as the server sees it (TP1, 36 layers)."""
    spec = FullAttentionSpec()
    names = [f"model.layers.{i}.self_attn.attn" for i in range(36)]
    cfg = types.SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=spec, layer_names=names)],
        hash_block_size=16,
    )
    layout = build_layout(cfg, Parallel(), {"model": "Qwen3-4B"}, semantics=sem)
    rpc = FakeRpc()
    worker = _worker(sem, rpc, [], layout=layout)
    worker.register(dict.fromkeys(names, QWEN3_KV), dict.fromkeys(names, spec))
    reg = rpc.registered
    assert reg["names"] == names
    assert set(reg["segments"]) == {1}
    assert set(reg["bytes_per_block"]) == set(reg["kv_strides"]) == {65536}
    assert set(reg["num_blocks"]) == {QWEN3_BLOCKS}
    assert (reg["backend"], reg["page_first"], set(reg["group_ids"])) == ("direct", False, {0})
