"""Per-TP committed spec dump (no GPU): ``--tp`` selects ``spec.json`` (TP 1)
or ``spec-tp<N>.json``, every consumer in ``run_gate.main`` reads that file,
and a TP without a committed dump is never official. Command: docs/rsi/GATES.md
("Tests")."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.gates import run_gate

QWEN3 = run_gate.REPO / "docs/rsi/models/Qwen3-4B"


def test_committed_dumps_record_their_tp():
    """Every committed dump is the one its file name selects: a dump copied
    to the wrong file would judge G0.spec against another TP's layout."""
    files = sorted((run_gate.REPO / "docs/rsi/models").glob("*/spec*.json"))
    assert QWEN3 / "spec-tp2.json" in files
    for f in files:
        tp = json.loads(f.read_text())["parallel"]["tp"]
        assert run_gate.committed_spec(f.parent, tp) == f, f


def _gate_repo(tmp_path: Path) -> Path:
    """A gate repo with Qwen3-4B's committed gate.json and both dumps."""
    repo = tmp_path / "repo"
    model = repo / "docs/rsi/models/Qwen3-4B"
    model.mkdir(parents=True)
    for name in ("gate.json", "spec.json", "spec-tp2.json"):
        shutil.copy(QWEN3 / name, model / name)
    return repo


def _stub_main(monkeypatch, repo: Path, stage=None) -> None:
    """Everything in ``main`` that would touch real trees, the GPU or the
    toolchain; ``stage``: the ``run_stage`` stub (default: fail fast)."""

    def unexpected(*_a, **_k):
        raise AssertionError("main ran a stage")

    monkeypatch.setattr(run_gate, "REPO", repo)
    monkeypatch.setattr(run_gate, "install_stop_handlers", lambda: None)
    monkeypatch.setattr(run_gate, "record_provenance", lambda *_: {})
    monkeypatch.setattr(run_gate, "cuda_env", lambda: {})
    monkeypatch.setattr(run_gate, "cuda_problems", lambda _: [])
    monkeypatch.setattr(run_gate, "scan_record", lambda _: {"flagged": []})
    monkeypatch.setattr(run_gate, "run_stage", stage or unexpected)
    monkeypatch.setattr(
        run_gate,
        "ProvenanceWatch",
        lambda *_: SimpleNamespace(check=lambda _w: {}, problems=lambda: [], record=lambda: {}),
    )


@pytest.mark.parametrize(
    "tp,selected",
    [(1, "spec.json"), (2, "spec-tp2.json"), (4, None)],
    ids=["tp1", "tp2", "tp4-no-dump"],
)
def test_main_feeds_every_consumer_the_selected_dump(tmp_path, monkeypatch, tp, selected):
    """``main`` with the stages stubbed: lmcache storing ranks, the
    unique-bytes estimate, the cap plan fallback, ``official_problems`` and
    the verdicts (G0.spec) all get this TP's file; the preempt pool is
    sized from it (``preempt_gpu_blocks``); the cap stage runs with the
    plan's ``workers``. Without one: no estimate, no
    storing ranks, and the run names the missing dump."""
    repo = _gate_repo(tmp_path)
    model_dir = repo / "docs/rsi/models/Qwen3-4B"
    want = model_dir / selected if selected else model_dir / f"spec-tp{tp}.json"
    want_spec = json.loads(want.read_text()) if selected else None
    seen: dict[str, list] = {}

    def spy(name, fn=None, ret=None, arg=0):
        def f(*a, **kw):
            seen.setdefault(name, []).append(a[arg])
            return fn(*a, **kw) if fn else ret

        monkeypatch.setattr(run_gate, name, f)

    spy("lmcache_storing_ranks", run_gate.lmcache_storing_ranks)
    spy("preempt_gpu_blocks", run_gate.preempt_gpu_blocks)
    spy("estimate_unique", lambda _a, _s, w: {"workload": w, "unique_bytes": 0, "how": "stub"}, arg=1)  # fmt: skip
    spy("cap_plan_spec", run_gate.cap_plan_spec, arg=2)
    spy("plan_cap_convs", ret={"workers": 77}, arg=1)
    spy("official_problems", run_gate.official_problems, arg=4)
    spy("verdicts", ret={}, arg=2)
    workers: dict[str, set[int]] = {}

    def stage(a, _g, arm, wl, _o, **_k):
        workers.setdefault(wl, set()).add(a.concurrency)
        return {"arm": arm, "workload": wl}

    _stub_main(monkeypatch, repo, stage)
    wls = "gsm8k,replay,cap" + (",preempt" if selected else "")
    argv = ["--model", "/m/Qwen3-4B", "--tag", "t", "--runs-root", str(tmp_path / "runs"),
            "--arms", "plain,baseline,old", "--workloads", wls,
            "--tp", str(tp), "--gpu", ",".join(map(str, range(tp)))]  # fmt: skip
    assert run_gate.main(argv) == 1  # never official: sliced arms, provenance stubbed

    (summary,) = (tmp_path / "runs").glob("*-t/summary.json")
    problems = json.loads(summary.read_text())["not_official_because"]
    assert seen["lmcache_storing_ranks"] == [want_spec]
    assert seen["official_problems"] == [want]
    assert seen["verdicts"] == [want]
    assert seen["cap_plan_spec"] == [want]
    # The cap plan's workers drive the cap stages only.
    assert workers.pop("cap") == {77}
    assert set().union(*workers.values()) == {run_gate.OFFICIAL["concurrency"]}
    if selected:
        assert seen["preempt_gpu_blocks"] == [want_spec]
        assert seen["estimate_unique"] == [want_spec, want_spec]  # gsm8k, replay
        assert seen["plan_cap_convs"] == [want_spec]  # no baseline dump: committed
        assert not any("committed spec dump" in p for p in problems), problems
    else:
        assert "preempt_gpu_blocks" not in seen and "estimate_unique" not in seen
        assert seen["plan_cap_convs"] == [None]
        assert f"committed spec dump {want} missing (no dump for --tp {tp})" in problems


def test_preempt_without_this_tps_dump_does_not_start(tmp_path, monkeypatch, capsys):
    """The preempt pool is sized from this TP's dump only: none, no run (an
    argument error before the run dir, provenance or any stage)."""
    repo = _gate_repo(tmp_path)
    _stub_main(monkeypatch, repo)

    def no_sizing(*_a):
        raise AssertionError("preempt pool sized without this TP's dump")

    monkeypatch.setattr(run_gate, "preempt_gpu_blocks", no_sizing)
    monkeypatch.setattr(run_gate, "record_provenance", lambda *_: pytest.fail("ran past the preempt guard"))  # fmt: skip
    argv = ["--model", "/m/Qwen3-4B", "--tag", "t", "--runs-root", str(tmp_path / "runs"),
            "--workloads", "gsm8k,preempt", "--tp", "4", "--gpu", "0,1,2,3"]  # fmt: skip
    with pytest.raises(SystemExit) as e:
        run_gate.main(argv)
    assert e.value.code == 2
    assert "docs/rsi/models/Qwen3-4B/spec-tp4.json (missing)" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "tp,file,dump_tp,problem",
    [
        (1, "spec.json", 1, None),
        (2, "spec-tp2.json", 2, None),
        (2, "spec-tp2.json", 1, "records parallel.tp=1, not --tp 2"),  # TP1 dump copied over
        (2, None, None, "missing (no dump for --tp 2)"),
    ],
    ids=["tp1", "tp2", "tp2-wrong-dump", "tp2-missing"],
)
def test_official_needs_this_tps_dump(tmp_path, tp, file, dump_tp, problem):
    import argparse

    if file:
        (tmp_path / file).write_text(json.dumps({"parallel": {"tp": dump_tp}}))
    (tmp_path / "gate.json").write_text("{}")
    committed = run_gate.committed_spec(tmp_path, tp)
    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t", "tp": tp}))  # fmt: skip
    probs = run_gate.official_problems(args, ["plain", "baseline", "native", "pegaflow.m:C"], list(run_gate.WORKLOADS), tmp_path / "gate.json", committed)  # fmt: skip
    hits = [p for p in probs if p.startswith("committed spec dump")]
    assert hits == ([f"committed spec dump {committed} {problem}"] if problem else []), probs
