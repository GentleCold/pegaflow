"""Gate self-test arms (M8).

``SelftestConnector`` is a minimal in-process KV store with sabotage modes. It
is not PegaFlow and not a reference for onboarding agents: it exists so every
gate run on a new model can prove the gate catches a wrong connector.

``kv_connector_extra_config.mode``:

- ``honest``: at finish, copies every complete computed prompt block (all
  layers, through the layer views) to CPU; serves the longest saved prefix.
  Positive control: must pass the G2 digest and output gates.
- ``liar``: reports every complete prompt block as a hit and loads nothing.
- ``bitflip``: loads correctly, then flips one exponent bit in one element of
  layer 0 of the first loaded block of every group.
- ``offbyone``: loads block ``i + 1``'s bytes into slot ``i``; a state group
  (one block per load) gets the state saved just before the right one in
  save order (the one after it if the right one was saved first).

The negative modes must fail ``G2.digest_self`` and ``G2.digest_vs_baseline``.

Groups: full attention (incl. MLA) and Mamba state in
``mamba_cache_mode=align`` (no speculative blocks). A state group saves the
states vLLM offers (``boundary_state_offloads``) in ``wait_for_save`` of the
step whose forward wrote them, and a hit at ``p`` loads one state per state
group into block ``p / block_size - 1`` plus the full blocks past the local
hit. Loads on such models are async: vLLM copies the previous state column
forward before a synchronous ``start_load_kv`` would write it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, group_kernel_blocks

from rsi.gates.rsi_probe import _parts, prefix_keys

MODES = ("honest", "liar", "bitflip", "offbyone")


@dataclass
class _Job:
    req_id: str
    # Per group: (prefix key of the source, block id).
    blocks: list[list[tuple[str, int]]]


@dataclass
class SelftestMetadata(KVConnectorMetadata):
    saves: list[_Job] = field(default_factory=list)
    loads: list[_Job] = field(default_factory=list)
    # State saves: copied in wait_for_save (after the forward that wrote them).
    state_saves: list[_Job] = field(default_factory=list)


class SelftestConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._mode = str(extra.get("mode", "honest"))
        if self._mode not in MODES:
            raise ValueError(f"SelftestConnector mode must be one of {MODES}, got {self._mode!r}")
        self._groups = kv_cache_config.kv_cache_groups
        mode = getattr(vllm_config.cache_config, "mamba_cache_mode", None)
        self._state: set[int] = set()
        for gid, g in enumerate(self._groups):
            spec = g.kv_cache_spec
            if isinstance(spec, MambaSpec) and mode == "align" and not spec.num_speculative_blocks:
                self._state.add(gid)
            elif not isinstance(spec, FullAttentionSpec) or getattr(spec, "sliding_window", None):
                raise NotImplementedError(
                    f"SelftestConnector: group {gid} is {type(spec).__name__} "
                    f"(mamba_cache_mode={mode}); only full attention and align-mode "
                    "Mamba state are supported"
                )
        self._bs = [g.kv_cache_spec.block_size for g in self._groups]
        self._unit = math.lcm(*self._bs)
        if role == KVConnectorRole.SCHEDULER:
            # Prefix keys at unit ends: full blocks saved in every full group,
            # and states saved in every state group.
            self._saved: set[str] = set()
            self._state_saved: dict[str, set[int]] = {}
            self._state_order: list[str] = []  # save order (offbyone's neighbour)
            self._local: dict[str, int] = {}
            self._reqs: dict[str, object] = {}
            self._saves: list[_Job] = []
            self._loads: list[_Job] = []
        else:
            self._views: list[list[torch.Tensor]] = []
            self._store: dict[tuple[int, str], list[torch.Tensor]] = {}
            self._sending: set[str] = set()
            self._recving: set[str] = set()
            self._lock = threading.Lock()

    # ------------------------------------------------------- scheduler side
    def _job(self, req_id: str, tokens: list[int], lo: int, hi: int, block_ids) -> _Job:
        """Full groups: blocks covering ``(lo, hi]``; state groups: the state at ``hi``."""
        per_group = []
        for gid, bs in enumerate(self._bs):
            if gid in self._state:
                key = prefix_keys(tokens, [hi])[0]
                per_group.append([(key, block_ids[gid][hi // bs - 1])] if hi > lo else [])
                continue
            idx = list(range(lo // bs, hi // bs))
            keys = prefix_keys(tokens, [(i + 1) * bs for i in idx])
            per_group.append([(k, block_ids[gid][i]) for k, i in zip(keys, idx, strict=True)])
        return _Job(req_id, per_group)

    def _hit_units(self, keys: list[str]) -> int:
        """Longest n: full blocks of the first n units saved and, with state
        groups, the state at n units saved in every state group."""
        n = next((k for k, key in enumerate(keys) if key not in self._saved), len(keys))
        if not self._state:
            return n
        full = self._state
        return next((k for k in range(n, 0, -1) if self._state_saved.get(keys[k - 1]) == full), 0)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._local[request.request_id] = num_computed_tokens
        self._reqs[request.request_id] = request
        units = (request.num_prompt_tokens - 1) // self._unit
        if self._mode == "liar":
            n = units
        else:
            tokens = list(request.prompt_token_ids or [])
            n = self._hit_units(prefix_keys(tokens, [self._unit * (k + 1) for k in range(units)]))
        ext = max(0, n * self._unit - num_computed_tokens)
        return ext, bool(self._state) and ext > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens <= 0:
            return
        if self._mode == "liar":
            if self._state:  # async: report the (empty) load as done
                self._loads.append(_Job(request.request_id, [[] for _ in self._groups]))
            return
        local = self._local.get(request.request_id, 0)
        tokens = list(request.prompt_token_ids or [])
        job = self._job(
            request.request_id, tokens, local, local + num_external_tokens, blocks.get_block_ids()
        )
        if self._mode == "offbyone":
            for gid in self._state:
                if job.blocks[gid]:
                    key, bid = job.blocks[gid][0]
                    job.blocks[gid] = [(self._neighbour(key), bid)]
        self._loads.append(job)

    def _neighbour(self, key: str) -> str:
        """Another state saved in every state group (the right one if none)."""
        done = [k for k in self._state_order if self._state_saved.get(k) == self._state]
        i = done.index(key)
        return done[i - 1] if i > 0 else done[min(1, len(done) - 1)]

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = SelftestMetadata(saves=self._saves, loads=self._loads)
        self._saves, self._loads = [], []
        if self._state and self._mode != "liar":
            meta.state_saves = self._offered(scheduler_output)
        return meta

    def _offered(self, scheduler_output) -> list[_Job]:
        """Prompt states this step's forward computes (offered by vLLM): not
        a state at or below where the request started this step (a loaded or
        locally cached state vLLM re-offers once it caches it)."""
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        offers = getattr(block_state, "boundary_state_offloads", None) or {}
        before = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        before |= dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        jobs = []
        for rid, entries in offers.items():
            request = self._reqs.get(rid)
            if request is None or rid not in before:
                continue
            tokens = list(request.prompt_token_ids or [])
            per_group: list[list[tuple[str, int]]] = [[] for _ in self._groups]
            for gid, bid, boundary in entries:
                if gid not in self._state or boundary % self._unit or not before[rid] < boundary <= len(tokens):  # fmt: skip
                    continue
                key = prefix_keys(tokens, [boundary])[0]
                per_group[gid].append((key, bid))
                got = self._state_saved.setdefault(key, set())
                if not got:
                    self._state_order.append(key)
                got.add(gid)
            if any(per_group):
                jobs.append(_Job(rid, per_group))
        return jobs

    def request_finished(self, request, block_ids):
        return self._finish(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        return self._finish(request, block_ids)

    def _finish(self, request, block_ids):
        self._local.pop(request.request_id, None)
        self._reqs.pop(request.request_id, None)
        if self._mode == "liar":
            return False, None
        end = min(request.num_computed_tokens, request.num_prompt_tokens)
        end = end // self._unit * self._unit
        tokens = list(request.prompt_token_ids or [])
        keys = prefix_keys(tokens, list(range(self._unit, end + 1, self._unit)))
        # Saves always cover a whole prefix, so the unsaved units are a suffix.
        first = next((k for k, key in enumerate(keys) if key not in self._saved), len(keys))
        if first == len(keys):
            return False, None
        self._saved.update(keys[first:])
        job = self._job(request.request_id, tokens, first * self._unit, end, block_ids)
        for gid in self._state:
            job.blocks[gid] = []  # states come from offers, not the finish block table
        self._saves.append(job)
        # Keep the blocks until the worker has copied them (next step).
        return True, None

    # ---------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        num_blocks = {}
        for t in self._kv_cache_config.kv_cache_tensors:
            for name in t.layers:
                num_blocks[name] = self._kv_cache_config.num_blocks_of(t)
        self._views = [
            [
                group_kernel_blocks(x, num_blocks[name])
                for name in g.layer_names
                for x in _parts(kv_caches[name])
            ]
            for g in self._groups
        ]

    def _save(self, jobs: list[_Job]) -> None:
        for job in jobs:
            for gid, blocks in enumerate(job.blocks):
                if not blocks:
                    continue
                views = self._views[gid]
                idx = torch.tensor([b for _, b in blocks], device=views[0].device)
                copies = [v.index_select(0, idx).cpu() for v in views]
                for j, (key, _) in enumerate(blocks):
                    self._store[(gid, key)] = [c[j] for c in copies]

    def start_load_kv(self, forward_context, **kwargs) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, SelftestMetadata):
            return
        # Saves first: a load in the same step may read what was just saved.
        self._save(meta.saves)
        with self._lock:
            self._sending.update(job.req_id for job in meta.saves)
        for job in meta.loads:
            for gid, blocks in enumerate(job.blocks):
                if not blocks:
                    continue
                keys = [k for k, _ in blocks]
                if self._mode == "offbyone" and len(keys) > 1:
                    keys = keys[1:] + keys[-1:]
                views = self._views[gid]
                idx = torch.tensor([b for _, b in blocks], device=views[0].device)
                for li, v in enumerate(views):
                    src = torch.stack([self._store[(gid, k)][li] for k in keys])
                    if self._mode == "bitflip" and li == 0:
                        raw = src[0].reshape(-1).view(torch.uint8)
                        # Highest byte of element 0 (little endian): an exponent
                        # bit. State views are raw int8 pages: element 0 of the
                        # first packed state.
                        spec = self._groups[gid].kv_cache_spec
                        size = spec.dtypes[0].itemsize if gid in self._state else src.element_size()
                        raw[size - 1] ^= 0x40
                    v.index_copy_(0, idx, src.to(v.device, non_blocking=False))
            if self._state:
                with self._lock:
                    self._recving.add(job.req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        meta = self._get_connector_metadata()
        if isinstance(meta, SelftestMetadata):
            self._save(meta.state_saves)

    def get_finished(self, finished_req_ids: set[str]):
        with self._lock:
            done, self._sending = self._sending, set()
            recv, self._recving = self._recving, set()
        return (done or None), (recv or None)
