"""RsiProbe and SelftestConnector step logic on synthetic scheduler outputs
(S6): what each step queues, digests and records. CPU only; needs the venv's
vllm and torch (imported by the probe), no GPU, no server."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from rsi.gates import rsi_probe  # noqa: E402
from rsi.gates.selftest import connector as selftest  # noqa: E402

BT = 4  # every group's block size
STATE, FULL = 0, 1  # group ids: an align-mode state group and full attention


def _probe() -> rsi_probe.RsiProbe:
    """A scheduler-side probe without vLLM config: groups STATE and FULL."""
    p = object.__new__(rsi_probe.RsiProbe)
    spec = SimpleNamespace(block_size=BT)
    p._groups = [SimpleNamespace(kv_cache_spec=spec), SimpleNamespace(kv_cache_spec=spec)]
    p._state_groups, p._skip_groups = {STATE}, {}
    p._digest, p._key_tokens, p._step = "tagged", BT, 0
    p._live, p._local_hit, p._pending, p._expect = {}, {}, [], []
    return p


def _request(rid: str, n_tokens: int):
    return SimpleNamespace(
        request_id=rid,
        all_token_ids=list(range(100, 100 + n_tokens)),
        sampling_params=SimpleNamespace(extra_args={"rsi_digest": "full"}),
    )


def _out(sched, new=(), cached=(), resumed=(), offers=None, preempted=()):
    """A SchedulerOutput stand-in: ``new`` = (rid, computed, block_ids);
    ``cached`` = (rid, computed, new_block_ids)."""
    return SimpleNamespace(
        num_scheduled_tokens=dict(sched),
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, num_computed_tokens=c, block_ids=b) for r, c, b in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _, _ in cached],
            num_computed_tokens=[c for _, c, _ in cached],
            new_block_ids=[b for _, _, b in cached],
            resumed_req_ids=set(resumed),
        ),
        preempted_req_ids=set(preempted),
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads=offers or {}),
    )


def _admit(p, rid: str, n_tokens: int, local: int, start: int, sched: int, blocks, offers=None):
    p.get_num_new_matched_tokens(_request(rid, n_tokens), local)
    return p.build_connector_meta(_out({rid: sched}, new=[(rid, start, blocks)], offers=offers))


_BLOCKS = ([10, 11, 12, 13], [20, 21, 22, 23])


@pytest.mark.parametrize(
    "start,blocks,n_tokens,what",
    [
        (2 * BT, _BLOCKS, 13, "start"),
        (2 * BT + 1, _BLOCKS, 13, "start_unaligned"),
        (2 * BT, ([10], [20]), 13, "start_no_block"),
        (2 * BT, ([10, 0, 12], [20, 21, 22]), 13, "start_no_block"),  # null block
    ],
    ids=["loaded", "unaligned", "short-table", "null-block"],
)
def test_start_state_is_queued_for_before_the_forward(start, blocks, n_tokens, what):
    """M1: the start state (block ``start / bt - 1``) goes to ``start_states``,
    which the worker digests in ``handle_preemptions`` (before the precopy),
    never to ``states`` (``wait_for_save``); a start state that cannot be
    digested is recorded as a ``start_*`` skip."""
    p = _probe()
    meta = _admit(p, "a", n_tokens, local=BT, start=start, sched=1, blocks=blocks)
    assert meta.states == []
    jobs = [(j.req_id, j.episode, j.group, j.block_id, j.covered) for j in meta.start_states]
    assert jobs == ([("a", 0, STATE, 11, 2 * BT)] if what == "start" else [])
    assert p._live["a"].states == [[0, start, STATE, what]]


def test_offers_are_computed_states_and_late_offers_are_not_skips():
    """An offer above the episode start is a computation, digested after the
    forward (``states``); vLLM may offer a boundary one step late, so a step
    ending on a boundary is ``not_offered`` only if neither that step nor the
    next one offered it."""
    p = _probe()
    meta = _admit(p, "a", 30, local=0, start=BT, sched=2 * BT, blocks=_BLOCKS,
                  offers={"a": [(STATE, 12, 3 * BT), (STATE, 11, BT)]})  # fmt: skip
    # BT is the start state (not a computation); 3 BT ended this step.
    assert [(j.block_id, j.covered) for j in meta.states] == [(12, 3 * BT)]
    assert [(j.block_id, j.covered) for j in meta.start_states] == [(10, BT)]
    # Step 2 ends on 4 BT, offered only in step 3; step 3 ends on 5 BT, never offered.
    p.build_connector_meta(_out({"a": BT}, cached=[("a", 3 * BT, ([], []))]))
    assert [s[3] for s in p._live["a"].states] == ["start", "offer"]
    p.build_connector_meta(
        _out({"a": BT}, cached=[("a", 4 * BT, ([], []))], offers={"a": [(STATE, 13, 4 * BT)]})
    )
    p.build_connector_meta(_out({"a": 1}, cached=[("a", 5 * BT, ([], []))]))
    p.build_connector_meta(_out({"a": 1}, cached=[("a", 5 * BT + 1, ([], []))]))
    states = p._live["a"].states
    assert [s for s in states if s[3] == "not_offered"] == [[0, 5 * BT, STATE, "not_offered"]]
    assert [0, 4 * BT, STATE, "offer"] in states


def test_preempted_episode_digests_its_computed_full_blocks():
    """``_track`` keeps the block table (appended for running requests,
    replaced on resume); a preemption queues the previous episode's fully
    computed full-attention blocks (state groups are digested per step) and
    a resumed episode gets its own start state from the new table."""
    p = _probe()
    _admit(p, "a", 30, local=0, start=0, sched=2 * BT, blocks=([10, 11], [20, 21]))
    p.build_connector_meta(_out({"a": BT + 1}, cached=[("a", 2 * BT, ([12, 13], [22, 23]))]))
    live = p._live["a"]
    assert live.blocks == [[10, 11, 12, 13], [20, 21, 22, 23]] and live.computed == 3 * BT + 1
    p.get_num_new_matched_tokens(live.request, 2 * BT)
    meta = p.build_connector_meta(
        _out(
            {"a": 1},
            cached=[("a", 2 * BT, ([30, 31, 32], [40, 41, 42]))],
            resumed=["a"],
            preempted=["a"],
        )  # fmt: skip
    )
    (job,) = meta.preempted
    assert job.episode == 0 and job.blocks[STATE] == []
    assert [(b[1], b[3]) for b in job.blocks[FULL]] == [(20, BT), (21, 2 * BT), (22, 3 * BT)]
    assert live.preempt_digested == [0] and len(live.episodes) == 2
    assert [(j.episode, j.block_id) for j in meta.start_states] == [(1, 31)]
    assert live.blocks == [[30, 31, 32], [40, 41, 42]]


def test_worker_digests_start_states_before_the_forward(monkeypatch):
    """M1: start states are digested in ``handle_preemptions`` (vLLM calls it
    before ``_update_states``, the align precopy and ``start_load_kv``),
    offers in ``wait_for_save`` (after the forward wrote them)."""
    p = object.__new__(rsi_probe.RsiProbe)
    p._groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=BT))]
    calls: list[tuple[str, str, int | None]] = []
    monkeypatch.setattr(p, "_open_digest_log", lambda: None)
    monkeypatch.setattr(
        p,
        "_digest_group",
        lambda rid, gid, blocks, episode=None: calls.append(
            (where, blocks[0][1], episode)
        ),  # fmt: skip
    )
    job = rsi_probe._StateJob
    meta = rsi_probe.ProbeMetadata(
        states=[job("a", 0, 0, 12, "k12", 3 * BT)], start_states=[job("b", 1, 0, 7, "k8", 2 * BT)]
    )
    p._connector_metadata = meta
    where = "handle_preemptions"
    p.handle_preemptions(meta)
    where = "wait_for_save"
    p.wait_for_save()
    assert calls == [("handle_preemptions", 7, 1), ("wait_for_save", 12, 0)]


# ---------------------------------------------------------- selftest connector


def _selftest(mode: str, tmp_path, n_state: int = 3):
    """Worker side: ``n_state`` state groups (one layer, 2 x int8 state per
    block) then one full-attention group (fp16)."""
    c = object.__new__(selftest.SelftestConnector)
    c._mode = mode
    c._state = set(range(n_state))
    state_spec = SimpleNamespace(block_size=BT, dtypes=(torch.float16,))
    full_spec = SimpleNamespace(block_size=BT)
    c._groups = [SimpleNamespace(kv_cache_spec=state_spec)] * n_state + [SimpleNamespace(kv_cache_spec=full_spec)]  # fmt: skip
    c._views = [[torch.zeros(4, 2, dtype=torch.float16)] for _ in c._groups]
    c._store = {}
    for gid in range(len(c._groups)):
        for k in ("k1", "k2"):
            c._store[(gid, k)] = [
                torch.full((2,), float(10 * gid + int(k[1])), dtype=torch.float16)
            ]
    c._sending, c._recving, c._lock = set(), set(), threading.Lock()
    c._report = str(tmp_path / "sabotage.jsonl")
    return c


def _load(c, n_state: int = 3):
    """One hit at 2 blocks: a state per state group (key k2, block 1), the
    full blocks k1, k2 into blocks 0, 1."""
    blocks = [[("k2", 1)] for _ in range(n_state)] + [[("k1", 0), ("k2", 1)]]
    c._connector_metadata = selftest.SelftestMetadata(loads=[selftest._Job("a", blocks)])
    c.start_load_kv(None)


@pytest.mark.parametrize(
    "mode,state_rows,full_rows,sabotaged,async_done",
    [
        ("honest", [2, 12, 22], [31, 32], {}, True),
        ("offbyone", [12, 22, 2], [32, 32], {0: 1, 1: 1, 2: 1, 3: 1}, True),
        ("sync", [2, 12, 22], [31, 32], {0: 1, 1: 1, 2: 1}, False),
        ("liar", [0, 0, 0], [0, 0], {0: 1, 1: 1, 2: 1, 3: 2}, True),
    ],
    ids=["honest", "offbyone", "sync", "liar"],
)
def test_selftest_sabotage_hits_every_group(tmp_path, mode, state_rows, full_rows, sabotaged, async_done):  # fmt: skip
    """M2: offbyone loads the next state group's state of the same prefix
    (not another saved prefix, which may not exist), the full blocks shifted
    by one; sync loads honest bytes synchronously (no async completion);
    each mode reports what it corrupted, per group."""
    import json

    c = _selftest(mode, tmp_path)
    _load(c)
    got_state = [int(c._views[g][0][1][0]) for g in range(3)]
    got_full = [int(c._views[3][0][b][0]) for b in (0, 1)]
    assert (got_state, got_full) == (state_rows, full_rows)
    counts: dict[int, int] = {}
    if (tmp_path / "sabotage.jsonl").exists():
        for line in (tmp_path / "sabotage.jsonl").read_text().splitlines():
            r = json.loads(line)
            counts[r["group"]] = counts.get(r["group"], 0) + r["n"]
    assert counts == sabotaged
    assert (c._recving == {"a"}) is async_done


def test_selftest_bitflip_corrupts_every_group(tmp_path):
    c = _selftest("bitflip", tmp_path)
    _load(c)
    assert all(float(c._views[g][0][1][0]) != float(10 * g + 2) for g in range(3))
    assert float(c._views[3][0][0][0]) != 31.0 and float(c._views[3][0][1][0]) == 32.0


@pytest.mark.parametrize("mode,is_async", [("honest", True), ("sync", False), ("liar", True)])
def test_selftest_load_mode(mode, is_async):
    """State-group models load async except ``sync`` (vLLM's precopy runs
    before a synchronous ``start_load_kv``)."""
    c = object.__new__(selftest.SelftestConnector)
    c._mode, c._state, c._unit = mode, {0}, BT
    c._local, c._reqs, c._saved, c._state_saved = {}, {}, set(), {}
    req = SimpleNamespace(request_id="a", num_prompt_tokens=3 * BT + 1, prompt_token_ids=list(range(3 * BT + 1)))  # fmt: skip
    keys = rsi_probe.prefix_keys(req.prompt_token_ids, [BT, 2 * BT, 3 * BT])
    c._saved.update(keys)
    c._state_saved.update({k: {0} for k in keys})
    assert c.get_num_new_matched_tokens(req, 0) == (3 * BT, is_async)
