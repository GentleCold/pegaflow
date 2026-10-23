"""Server teardown contract of the gate (no GPU; needs the gate's Python deps).

Run from the repo root: ``python -m pytest rsi/gates/test_teardown.py``.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from rsi.gates import run_gate


def test_stop_waits_for_the_whole_process_group(tmp_path, monkeypatch):
    """The leader exits at once on SIGTERM, a child keeps tearing down for
    2 s (vLLM's EngineCore, a pinned pool after SIGKILL): ``stop`` must
    return only once the group is gone, then log the exit time."""
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
