"""Worker-side glue: register KV caches, run the step's loads and saves.

vLLM- and torch-free: tensors are opaque, and the connector injects ``wrap``
(tensor -> CUDA IPC bytes), ``new_load_state`` and ``mark_compute`` (returns a
handle whose ``synchronize()`` waits until the work enqueued so far on the
compute stream is done, e.g. a recorded CUDA event), so unit tests can drive it
with fakes.

Saves are asynchronous: ``save`` only marks the compute stream and queues the
job; a saver thread waits for the mark and then issues the save RPC (the server
copies GPU->CPU on its own stream and replies once the copy is done). The
source blocks must stay untouched until the copy is done: the scheduler keeps
every block it plans to save pinned in vLLM's block pool until all workers
report the step's saves done (``saved_through``), so no such block is
reallocated, zeroed or copied into meanwhile.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..adapters import Parallel, Unsupported, adapter_for
from .layout import Layout
from .scheduler import StepPlan

logger = logging.getLogger(__name__)


def _save_settled(e: BaseException) -> bool:
    """True if a raised save RPC proves that no copy of the job runs.

    Only an InvalidArgument status does: the binding raises ``ValueError``
    for it, tonic's client never makes it, and the save handler returns it
    before it copies or after its copy synchronized. Every other error may
    leave a copy reading the job's blocks:

    - Internal: the handler also returns it when a GPU copy failed
      part-way with earlier copies still queued, and the client itself
      reports decode and HTTP/2 errors as Internal;
    - no reply (Unavailable, Cancelled, DeadlineExceeded, Unknown): the
      handler may still run;
    - FailedPrecondition (unknown instance) raises the same exception
      class as those, so it fails closed too;
    - a Rust panic in the binding (pyo3 ``PanicException``, a
      BaseException) may come after the request was sent.
    """
    return isinstance(e, ValueError)


@dataclass
class _Batch:
    state: Any  # PyLoadState
    req_ids: list[str]


@dataclass
class _SaveJob:
    seq: int
    ready: Any  # mark_compute() handle
    saves: list[tuple[str, list[int], list[bytes]]]


@dataclass
class SaveStats:
    """Save counters since the last report (grep "save stats" in vllm.log):
    the fail-closed paths, and the waits on the step's critical path."""

    steps: int = 0  # wait_for_save calls
    rpcs: int = 0  # save RPCs sent
    failed: int = 0  # jobs that did not store their blocks
    raised: int = 0  # save RPCs that raised with a copy possibly running (each quiesces)
    sync_wait_s: float = 0.0  # waiting for sync-load copies
    max_job_s: float = 0.0  # saver thread: compute wait + RPC of one job
    _since: float = field(default=0.0, repr=False)


class Worker:
    # wait_for_save calls between save-cost log lines.
    SAVE_LOG_EVERY = 500
    SHUTDOWN_FLUSH_S = 30.0
    # Waits that never give up (sync loads) log this often.
    STALL_LOG_S = 10.0
    # Sync-load poll interval: doubles from the first value up to the cap.
    SYNC_POLL_S = (20e-6, 200e-6)
    # Quiesce window after a save RPC raised without proving its copy over
    # (``_save_settled``): no save RPC is sent and the job's pins (and every
    # later job's) hold for this long, then the job counts as done and saving
    # resumes. A heuristic, not a proof: the server reports no bound on a copy
    # it may still run. It must stay well above the longest save job (compute
    # wait + RPC; evidence in rsi-notes).
    SAVE_QUIESCE_S = 10.0

    def __init__(
        self,
        layout: Layout,
        rpc: Any,
        instance_id: str,
        par: Parallel,
        device_id: int,
        wrap: Callable[[Any], bytes],
        new_load_state: Callable[[], Any],
        mark_compute: Callable[[], Any],
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._par = par
        # Server-side rank: TP replicas share one (layout.StorageRanks).
        storage = layout.storage
        self._storage_rank = storage.rank[par.tp_rank]
        self._writer = storage.writer[par.tp_rank]
        self._device_id = device_id
        self._wrap = wrap
        self._new_load_state = new_load_state
        self._mark_compute = mark_compute
        self._clock = clock
        self.save_stats = SaveStats(_since=clock())
        # Registered layer names per storage group (index into layout.groups).
        self._layers: list[list[str]] = [[] for _ in layout.groups]
        self._batches: list[_Batch] = []
        self._failed: set[str] = set()  # failed, not yet reported
        self._done: set[str] = set()  # finished without a load batch, not yet reported
        self._invalid: set[int] = set()  # GPU blocks of failed sync loads, not yet reported
        # Async saves. Jobs complete in submit order (one saver thread), so a
        # sequence number is enough to know what has landed.
        self._cv = threading.Condition()
        self._submitted = 0  # step thread only
        self._completed = 0  # saver thread, under _cv
        # (job seq, plan step) of submitted jobs not known done, and the last
        # plan step seen by ``save`` (``saved_through``).
        self._job_steps: collections.deque[tuple[int, int]] = collections.deque()
        self._last_step = -1
        # The saver thread exited with no copy of its jobs possibly running:
        # every hold is released and later saves are dropped (future misses).
        self._saver_dead = False
        # A save RPC raised and its window (SAVE_QUIESCE_S) runs: new saves
        # are dropped (future misses). Set and cleared by the saver thread.
        self._quiescing = False
        self._stop = threading.Event()  # shutdown: ends a quiesce window early
        self._jobs: queue.SimpleQueue[_SaveJob | None] = queue.SimpleQueue()
        self._saver = threading.Thread(
            target=self._save_loop, name="pegaflow-v2-saver", daemon=True
        )
        self._saver.start()

    def register(self, kv_caches: dict[str, Any], specs: dict[str, Any]) -> None:
        """``specs``: layer name -> vLLM KVCacheSpec for every transferable layer."""
        names: list[str] = []
        wrappers: list[bytes] = []
        num_blocks: list[int] = []
        bytes_per_block: list[int] = []
        kv_strides: list[int] = []
        segments: list[int] = []
        group_ids: list[int] = []
        for sid, g in enumerate(self._layout.groups):
            for name in g.layers:
                if name not in kv_caches:
                    continue  # another PP rank owns this layer
                spec = specs[name]
                adapter = adapter_for(spec, g.gid)
                decl = adapter.layer(spec, name, self._par)
                view = adapter.view(spec, name, kv_caches[name])
                # The server copies bytes_per_block per segment: a declaration
                # that disagrees would store bytes other than the declared ones.
                if not decl.segments or any(s != view.bytes_per_block for s in decl.segments):
                    raise Unsupported(
                        f"layer {name}: declared segments {decl.segments} != server copy "
                        f"size bytes_per_block={view.bytes_per_block} per segment"
                    )
                if len(decl.segments) > 1 and view.kv_stride_bytes < view.bytes_per_block:
                    raise Unsupported(
                        f"layer {name}: kv_stride_bytes={view.kv_stride_bytes} < "
                        f"bytes_per_block={view.bytes_per_block}: segments overlap"
                    )
                names.append(name)
                wrappers.append(self._wrap(view.tensor))
                num_blocks.append(view.num_blocks)
                bytes_per_block.append(view.bytes_per_block)
                kv_strides.append(view.kv_stride_bytes)
                segments.append(len(decl.segments))
                group_ids.append(sid)
                self._layers[sid].append(name)
        if not names:
            raise Unsupported("no transferable KV cache layer on this worker")
        par = self._par
        ok, msg = self._rpc.register_context_batch(
            self._instance_id,
            self._layout.namespace,
            self._storage_rank,
            par.pp_rank,
            self._layout.storage.tp_size,
            par.world_size,
            self._device_id,
            names,
            wrappers,
            num_blocks,
            bytes_per_block,
            kv_strides,
            segments,
            self._layout.transfer_backend,
            self._layout.page_first,
            group_ids,
        )
        if not ok:
            raise RuntimeError(f"PegaFlow register_context_batch failed: {msg}")
        logger.info(
            "[PegaKVConnectorV2] registered %d layers instance=%s namespace=%s device=%d "
            "storage_rank=%d writer=%s",
            len(names),
            self._instance_id,
            self._layout.namespace,
            self._device_id,
            self._storage_rank,
            self._writer,
        )

    def start(self, plan: StepPlan) -> None:
        """Start this step's loads."""
        self._start_loads(plan)

    def _start_loads(self, plan: StepPlan) -> None:
        """One ``load`` RPC per request, the step's sync loads first.

        The server consumes leases one by one and fails the whole call on the
        first bad one (e.g. an expired lease), so batching requests would fail
        every request of the step for one bad lease.

        Sync loads are waited for before returning (vLLM calls this before
        the forward when the step has any); a failed one reports its blocks
        as invalid. Async loads report through ``poll``. The server copies a
        device's loads one at a time in arrival order, so a sync load sent
        after the step's async loads would stall the step for their copies
        too.
        """
        sync: list[tuple[Any, list[int]]] = []
        for lp in sorted(plan.loads, key=lambda lp: not lp.sync):
            if lp.fail:
                if lp.sync:
                    self._invalid.update(lp.covered)
                else:
                    self._failed.add(lp.req_id)
            elif not lp.loads:
                if not lp.sync:
                    self._done.add(lp.req_id)  # nothing to copy, still owes a completion
            else:
                state = self._start_load(lp.req_id, lp.loads)
                if state is None:
                    if lp.sync:
                        self._invalid.update(lp.covered)
                    else:
                        self._failed.add(lp.req_id)
                elif lp.sync:
                    sync.append((state, lp.covered))
                else:
                    self._batches.append(_Batch(state, [lp.req_id]))
        if sync:
            self._wait_sync(sync)

    def _start_load(self, req_id: str, loads: list[tuple[bytes, list[list[int | None]]]]) -> Any:
        """Issue the load RPC; the load state, or None if the RPC failed."""
        try:
            state = self._new_load_state()
            ok, msg = self._rpc.load(
                self._instance_id,
                self._storage_rank,
                self._device_id,
                state.shm_name(),
                self._layers,
                loads,
            )
        except Exception as e:
            ok, msg = False, repr(e)
        if not ok:
            logger.warning("[PegaKVConnectorV2] load failed req=%s: %s", req_id, msg)
            return None
        return state

    def _wait_sync(self, loads: list[tuple[Any, list[int]]]) -> None:
        """Wait until every sync load reached a final state.

        Short by construction (policy ``sync_load_tokens``): ~1 ms per load.
        There is no deadline: until the server signals, its copy may still
        land in the blocks, so neither "failed" (vLLM would recompute into
        them) nor "done" is safe to report. Polls with a growing sleep.
        """
        t0 = self._clock()
        next_log = t0 + self.STALL_LOG_S
        for state, covered in loads:
            delay, cap = self.SYNC_POLL_S
            while not state.is_ready():
                time.sleep(delay)
                delay = min(delay * 2, cap)
                if self._clock() >= next_log:
                    next_log += self.STALL_LOG_S
                    logger.warning(
                        "[PegaKVConnectorV2] sync load still pending after %.0f s",
                        self._clock() - t0,
                    )
            if state.get_state() != 1:
                logger.warning(
                    "[PegaKVConnectorV2] sync load failed state=%s blocks=%d",
                    state.get_state(),
                    len(covered),
                )
                self._invalid.update(covered)
        with self._cv:
            self.save_stats.sync_wait_s += self._clock() - t0

    def take_invalid(self) -> set[int]:
        """GPU blocks whose sync load failed since the last call."""
        out, self._invalid = self._invalid, set()
        return out

    # ----------------------------------------------------------------- saves
    def save(self, plan: StepPlan) -> None:
        """Queue this step's saves behind a mark on the compute stream. Never
        blocks. vLLM calls this every engine step, also steps without a
        forward, so ``saved_through`` advances on idle steps too."""
        saves: list[tuple[str, list[int], list[bytes]]] = []
        # A replica that is not its part's writer stores nothing: the writer's
        # copy seals the block and every holder loads from it.
        for gs in plan.saves if self._writer else ():
            for name in self._layers[gs.sid]:
                saves.append((name, gs.block_ids, gs.block_hashes))
        if saves and (self._saver_dead or self._quiescing):
            with self._cv:
                self.save_stats.failed += 1
            saves = []
        if saves:
            # The mark covers this step's forward (and draft forward): the
            # saver only reads the blocks after the KV writes completed.
            ready = self._mark_compute()
            self._submitted += 1
            self._job_steps.append((self._submitted, plan.step))
            self._jobs.put(_SaveJob(self._submitted, ready, saves))
        self._last_step = max(self._last_step, plan.step)
        with self._cv:
            self.save_stats.steps += 1
            report = self.save_stats.steps >= self.SAVE_LOG_EVERY
        if report:
            self.report_save_stats()

    def saved_through(self) -> int:
        """The latest plan step whose saves, and every earlier one's, are done
        (failed or not: a failed save is a future miss, its blocks may go)."""
        with self._cv:
            completed = self._submitted if self._saver_dead else self._completed
        steps = self._job_steps
        while steps and steps[0][0] <= completed:
            steps.popleft()
        return steps[0][1] - 1 if steps else self._last_step

    def _settled(self, target: int) -> bool:
        return self._saver_dead or self._completed >= target

    def flush(self, timeout: float) -> bool:
        """Wait up to ``timeout`` s until every submitted save job is done
        (shutdown, tests); False on timeout."""
        target = self._submitted
        with self._cv:
            return self._cv.wait_for(lambda: self._settled(target), timeout)

    def _save_loop(self) -> None:
        try:
            while (job := self._jobs.get()) is not None:
                if not self._run_job(job) and not self._quiesce(job):
                    break
        finally:
            with self._cv:
                # Dying while a raised job's copy may run: its holds stay for good.
                if not self._quiescing:
                    self._saver_dead = True
                self._cv.notify_all()

    def _quiesce(self, raised: _SaveJob) -> bool:
        """After ``raised``'s RPC raised: send nothing for the window
        (``SAVE_QUIESCE_S``) while its pins hold (its copy may still run),
        then count it done and resume. Jobs queued behind it are completed as
        failed without an RPC (no copy of theirs started): under the pressure
        that made the RPC raise they would likely raise too, each holding
        pins for another window.
        Shutdown ends the window early. False if shutdown came during it."""
        window = self.SAVE_QUIESCE_S
        self._quiesce_wait(window)
        last, dropped, stop = raised.seq, 0, False
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is None:
                stop = True
                break
            last, dropped = job.seq, dropped + 1
        with self._cv:
            self._completed = last
            self.save_stats.failed += dropped
            self._quiescing = False
            self._cv.notify_all()
        logger.warning(
            "[PegaKVConnectorV2] saves resume after %.0f s quiesce (%d queued jobs dropped)",
            window,
            dropped,
        )
        return not stop

    def _quiesce_wait(self, seconds: float) -> None:  # tests replace it
        self._stop.wait(seconds)

    def _run_job(self, job: _SaveJob) -> bool:
        """Run one save job; False if its copy may still run.

        A save RPC that raises (``_save_settled`` false) is not a finished
        save: the server may still copy the job's blocks. The job is not
        counted as completed, so its pins (and every later job's) hold, and ``_quiescing`` drops new saves; ``_quiesce`` then waits
        out the window and resumes. Any BaseException out of the RPC (e.g.
        a Rust panic in the binding) counts as such a raise. Only the
        raised job's copy may race a reuse, and only if it outlasts the
        window; closing that needs a server that reports such a copy only
        once it stopped (a server limitation). A compute wait that raises
        past ``except Exception`` ends the saver thread (``_saver_dead``):
        no RPC was sent, so every hold is released and later saves are
        dropped.
        A failed reply, a rejected request (``_save_settled``), or a failed
        compute wait (no RPC sent) completes the job as a failed save: a
        future miss.
        """
        t0 = self._clock()
        # over: no copy of this job can still be running
        ok = over = raised = sent = False
        try:
            try:
                job.ready.synchronize()
            except Exception as e:  # the forward failed: its KV is not valid
                ok, over, msg = False, True, f"compute wait failed: {e!r}"
            else:
                sent = True
                try:
                    ok, msg = self._rpc.save(
                        self._instance_id,
                        self._storage_rank,
                        self._par.pp_rank,
                        self._device_id,
                        job.saves,
                    )
                    over = True
                except BaseException as e:  # incl. a binding panic: the copy may run
                    ok, over, msg = False, _save_settled(e), repr(e)
                    raised = not over
            if not over:
                logger.error(
                    "[PegaKVConnectorV2] save RPC raised, saves pause %.0f s (fail closed): %s",
                    self.SAVE_QUIESCE_S,
                    msg,
                )
            elif not ok:
                # Best effort: a dropped save is a future miss, not a wrong hit.
                logger.warning("[PegaKVConnectorV2] save failed: %s", msg)
        finally:
            with self._cv:
                stats = self.save_stats
                stats.max_job_s = max(stats.max_job_s, self._clock() - t0)
                stats.failed += not ok
                stats.rpcs += sent
                if over:
                    # Failed or not, the copy is over: its pins are released.
                    self._completed = job.seq
                elif raised:  # the thread lives on and quiesces
                    stats.raised += 1
                    self._quiescing = True
                else:  # the compute wait raised a BaseException: no RPC was sent
                    self._saver_dead = True
                self._cv.notify_all()
        return over

    def report_save_stats(self) -> None:
        """Log and reset the save-cost window (grep "save stats" in vllm.log)."""
        now = self._clock()
        with self._cv:
            s, self.save_stats = self.save_stats, SaveStats(_since=now)
        if not s.steps:
            return
        logger.info(
            "[PegaKVConnectorV2] save stats tp=%d window_s=%.1f steps=%d rpcs=%d failed=%d "
            "raised=%d sync_wait_ms=%.1f max_job_ms=%.2f",
            self._par.tp_rank,
            now - s._since,
            s.steps,
            s.rpcs,
            s.failed,
            s.raised,
            s.sync_wait_s * 1e3,
            s.max_job_s * 1e3,
        )

    def shutdown(self) -> None:
        self._stop.set()  # a quiesce window ends now: its queued jobs are dropped
        if not self.flush(self.SHUTDOWN_FLUSH_S):
            logger.warning("[PegaKVConnectorV2] saves still pending at shutdown")
        self._jobs.put(None)
        self._saver.join(timeout=1.0)
        self.report_save_stats()

    # ------------------------------------------------------------------ poll
    def poll(self) -> tuple[set[str], set[str], set[str]]:
        """Returns (finished_sending, finished_recving, failed_recving).

        Failed receives are in both receive sets. Nothing is ever sending:
        finished requests are not held (pins cover their blocks).
        """
        sending: set[str] = set()
        finished, self._done = self._done, set()
        pending: list[_Batch] = []
        for b in self._batches:
            if not b.state.is_ready():
                pending.append(b)
                continue
            finished.update(b.req_ids)
            state = b.state.get_state()
            if state != 1:
                logger.warning("[PegaKVConnectorV2] load state=%s reqs=%s", state, b.req_ids)
                self._failed.update(b.req_ids)
        self._batches = pending
        failed, self._failed = self._failed, set()
        return sending, finished | failed, failed
