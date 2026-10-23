"""Worker-side glue: register KV caches, run the step's loads and saves.

vLLM- and torch-free: tensors are opaque, and the connector injects ``wrap``
(tensor -> CUDA IPC bytes), ``new_load_state`` and ``sync`` (wait for the
compute stream before saving), so unit tests can drive it with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..adapters import Parallel, Unsupported, adapter_for
from .layout import Layout
from .scheduler import StepPlan

logger = logging.getLogger(__name__)


@dataclass
class _Batch:
    state: Any  # PyLoadState
    req_ids: list[str]


class Worker:
    def __init__(
        self,
        layout: Layout,
        rpc: Any,
        instance_id: str,
        par: Parallel,
        device_id: int,
        wrap: Callable[[Any], bytes],
        new_load_state: Callable[[], Any],
        sync: Callable[[], None],
    ):
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._par = par
        self._device_id = device_id
        self._wrap = wrap
        self._new_load_state = new_load_state
        self._sync = sync
        # Registered layer names per storage group (index into layout.groups).
        self._layers: list[list[str]] = [[] for _ in layout.groups]
        self._batches: list[_Batch] = []
        self._failed: set[str] = set()  # failed, not yet reported
        self._done: set[str] = set()  # finished without a load batch, not yet reported

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
            par.tp_rank,
            par.pp_rank,
            par.tp_size,
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
            "[PegaKVConnectorV2] registered %d layers instance=%s namespace=%s device=%d",
            len(names),
            self._instance_id,
            self._layout.namespace,
            self._device_id,
        )

    def start_loads(self, plan: StepPlan) -> None:
        """One ``load`` RPC per request.

        The server consumes leases one by one and fails the whole call on the
        first bad one (e.g. an expired lease), so batching requests would fail
        every request of the step for one bad lease.
        """
        for lp in plan.loads:
            if lp.fail:
                self._failed.add(lp.req_id)
            elif not lp.loads:
                self._done.add(lp.req_id)  # nothing to copy, still owes a completion
            else:
                self._start_load(lp.req_id, lp.loads)

    def _start_load(self, req_id: str, loads: list[tuple[bytes, list[list[int | None]]]]) -> None:
        try:
            state = self._new_load_state()
            ok, msg = self._rpc.load(
                self._instance_id,
                self._par.tp_rank,
                self._device_id,
                state.shm_name(),
                self._layers,
                loads,
            )
        except Exception as e:
            ok, msg = False, repr(e)
        if not ok:
            logger.warning("[PegaKVConnectorV2] load failed req=%s: %s", req_id, msg)
            self._failed.add(req_id)
            return
        self._batches.append(_Batch(state, [req_id]))

    def save(self, plan: StepPlan) -> None:
        saves: list[tuple[str, list[int], list[bytes]]] = []
        for gs in plan.saves:
            for name in self._layers[gs.sid]:
                saves.append((name, gs.block_ids, gs.block_hashes))
        if not saves:
            return
        # The server copies with its own stream: this step's KV must be written.
        self._sync()
        try:
            ok, msg = self._rpc.save(
                self._instance_id, self._par.tp_rank, self._par.pp_rank, self._device_id, saves
            )
        except Exception as e:
            ok, msg = False, repr(e)
        if not ok:
            # Best effort: a dropped save is a future miss, not a wrong hit.
            logger.warning("[PegaKVConnectorV2] save failed: %s", msg)

    def poll(self) -> tuple[set[str], set[str]]:
        """Returns (finished_recving, failed_recving); failed ids are in both."""
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
        return finished | failed, failed
