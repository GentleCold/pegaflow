"""Gate-owned vLLM KV connector probe (human-owned, see docs/rsi/GATES.md).

The probe never loads or saves KV. It is attached next to the connector under
test (MultiConnector) or alone (baseline) and records:

- G0 spec dump: ``spec.json`` (scheduler: kv_cache_config, specs, groups) and
  ``spec-pp{p}-tp{t}.json`` (worker: per-layer view shape/stride/dtype).
- G2 V3 digests: for finished sampled requests, a fingerprint of every
  complete GPU block (per group, per layer, per rank) keyed by the hash of the
  token prefix the block covers. Bytes are addressed with ``KVCacheTensor``
  geometry (``offset + l*layer_stride + b*block_stride``, unpadded page);
  ``register_kv_caches`` checks that geometry against every layer view and
  refuses to start when they disagree.
- ``requests.jsonl``: per request prompt length, computed tokens at finish,
  schedule *episodes* (first schedule and every resume after preemption:
  step, local prefix-cache hit, computed tokens at start), *progress* (the
  scheduler steps at which computed tokens crossed a ``key_tokens`` boundary,
  with the computed count after that step) and the prefix-key chain at
  ``key_tokens`` granularity (gcd of group block sizes). ``rsi.gates.oracle``
  replays this offline.
- ``load_errors.jsonl``: invalid block ids / failed receives reported to the
  scheduler (``update_connector_output``).
- Mamba state groups in ``mamba_cache_mode=align`` (vLLM keeps the state at
  a block end only until compute moves on, so none is left at finish): a
  state is digested in the step that holds it. Computed states: the
  ``boundary_state_offloads`` vLLM offers in the step whose forward writes
  them (``kv_connector_block_state``), digested in ``wait_for_save`` (after
  that forward, before any later step can reuse the block). Loaded states:
  the episode's start state, block ``start / block_size - 1`` of the
  episode's first step (the local hit or what a connector loaded), digested
  in ``handle_preemptions``. On vLLM 0.32's V2 runner that is after the
  align precopy (which only reads the start block into a later one) and
  before a sync ``start_load_kv`` and the forward, so it digests the bytes
  the forward computes from (a sync load landing after it is too late for
  the forward and shows up as a mismatch). These records carry ``episode`` (index into the request's
  episodes), so a state computed in a preempted episode is tagged by that
  episode. Outside align mode Mamba groups are skipped (running state).

Config via ``kv_connector_extra_config``: ``probe_dir`` (required),
``digest``: ``"off"`` | ``"all"`` | ``"tagged"`` (only requests sent with
``vllm_xargs={"rsi_digest": "full" | "ref" | "hit"}``: every complete block,
computed blocks only, loaded blocks only). Requests may also carry
``rsi_pass`` (a label copied into the record) and ``rsi_conv`` / ``rsi_turn``
(workload position, copied as ``conv`` / ``turn``). Digests run in ``start_load_kv`` and add GPU
work and syncs: only tag requests whose timing is not measured.

``RsiWorkerExt`` (``--worker-extension-cls``) adds ``rsi_poison`` for the
gate's KV poisoning between passes.
"""

from __future__ import annotations

import dataclasses
import enum
import fractions
import hashlib
import json
import math
import os
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

_CHUNK_BYTES = 64 << 20  # bound on the uint8 chunk; int64 temporaries are 8x this
_MASK = 2**64 - 1


def module_files(connector_modules: list[str] = ()) -> dict[str, Any]:
    """Files this process resolved its instrument and runtime from (G0
    provenance): the probe, the worker extension, vLLM, torch, ``pegaflow``
    (imported, or where it would resolve) and its native extension if
    loaded, ``site/usercustomize`` if loaded, the interpreter prefix, and
    ``connector:<module>`` for each module the gate names in
    ``connector_modules`` (the arm's connector, stacked before the probe in
    ``MultiConnector``, so already imported)."""
    import importlib.util
    import sys

    def file_of(name: str) -> str | None:
        m = sys.modules.get(name)
        return getattr(m, "__file__", None) if m is not None else None

    out: dict[str, Any] = {
        "rsi.gates.rsi_probe": __file__,
        "worker_extension": file_of(RsiWorkerExt.__module__),
        "prefix": sys.prefix,
    }
    for name in ("vllm", "torch", "pegaflow"):
        f = file_of(name)
        if f is None:
            spec = importlib.util.find_spec(name)
            f = spec.origin if spec else None
        out[name] = f
    for name in ("pegaflow.pegaflow", "sitecustomize", "usercustomize"):
        if file_of(name):
            out[name] = file_of(name)
    for name in connector_modules:
        out[f"connector:{name}"] = file_of(name)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, enum.Enum):
        return value.value if isinstance(value.value, (str, int)) else value.name
    if isinstance(value, fractions.Fraction):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _spec_dict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


_DERIVED = (
    "page_size_bytes",
    "real_page_size_bytes",
    "num_heads",
    "state_content_size_bytes",
    "tokens_per_state",
    "num_states",
    "prefix_cacheable",
    "prefix_replay_tokens",
    "has_layer_views",
    "uses_slot_mapping",
    "block_table_token_alignment",
)


def _spec_dict(spec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"class": type(spec).__name__}
    out["mro"] = [c.__name__ for c in type(spec).__mro__ if c.__name__ != "object"]
    for f in dataclasses.fields(spec):
        out[f.name] = _jsonable(getattr(spec, f.name))
    for name in _DERIVED:
        try:
            out[name] = _jsonable(getattr(spec, name))
        except Exception as e:  # property may assert on some specs
            out[name] = f"<error: {type(e).__name__}>"
    return out


def _hf_summary(hf_config: Any) -> dict[str, Any]:
    keys = (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "sliding_window",
        "layer_types",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "num_kv_shared_layers",
        "full_attention_interval",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    out: dict[str, Any] = {}
    for cfg_name, cfg in (
        ("", hf_config),
        ("text_config.", getattr(hf_config, "text_config", None)),
    ):
        if cfg is None:
            continue
        for k in keys:
            v = getattr(cfg, k, None)
            if v is not None:
                out[cfg_name + k] = _jsonable(v)
    return out


def _compilation_summary(vllm_config: Any) -> dict[str, Any]:
    """The compile settings that decide the model's numbers: mode, backend,
    Inductor config and CUDA graph mode (the cache dir is volatile)."""
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        return {}
    return {
        k: _jsonable(getattr(cc, k, None))
        for k in ("mode", "backend", "inductor_compile_config", "cudagraph_mode")
    }


# Spec dump schema (``probe.schema``); bump when a field is added, and add it
# to ``run_gate.SPEC_FIELDS`` so older records are not judged on it.
SPEC_SCHEMA = 3


def _model_runner(vllm_config: Any) -> str:
    """vLLM's model runner: V2 calls ``handle_preemptions`` after zeroing,
    COW copies and the Mamba precopy, V1 before them."""
    return "V2" if getattr(vllm_config, "use_v2_model_runner", False) else "V1"


def _clobbered_blocks(scheduler_output: Any) -> dict[int, str]:
    """Block ids vLLM overwrites this step before the forward: freshly
    allocated blocks it zeroes (``new_block_ids_to_zero``) and COW copy
    destinations (``kv_cache_block_copies``). On the V2 runner both happen
    before ``handle_preemptions``, so a freed preempted block reallocated in
    the same step no longer holds the preempted episode's bytes."""
    out = {int(b): "zeroed" for b in getattr(scheduler_output, "new_block_ids_to_zero", None) or ()}
    for c in getattr(scheduler_output, "kv_cache_block_copies", None) or ():
        dst = c.dst_block_id if hasattr(c, "dst_block_id") else c[1]  # KVCacheBlockCopy
        out[int(dst)] = "cow"
    return out


def prefix_keys(tokens: list[int], covered: list[int]) -> list[str]:
    """Hex prefix hash for each coverage length in ``covered`` (ascending)."""
    out: list[str] = []
    h = hashlib.blake2b(digest_size=8)
    pos = 0
    for end in covered:
        h.update(array("I", tokens[pos:end]).tobytes())
        pos = end
        out.append(h.copy().hexdigest())
    return out


def _parts(t: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(t, (list, tuple)):
        return tuple(x for p in t for x in _parts(p))
    return (t,)


def _span(v: torch.Tensor, base_ptr: int) -> tuple[int, int]:
    """Byte range [lo, hi) of a strided view relative to ``base_ptr``."""
    lo = v.data_ptr() - base_ptr
    extent = 1 + sum((d - 1) * abs(s) for d, s in zip(v.shape, v.stride(), strict=True) if d)
    return lo, lo + extent * v.element_size()


def _float_parts(spec: Any, page: int) -> list[tuple[int, int, torch.dtype]]:
    """Byte ranges ``[lo, hi)`` of a page read as floats for the L1 report:
    the whole page as the spec's dtype, or a state spec's packed states
    (``shapes`` / ``dtypes``, contiguous from byte 0, as vLLM binds them)."""
    dtype = getattr(spec, "dtype", None)
    if isinstance(dtype, torch.dtype):
        ok = dtype.is_floating_point and page % dtype.itemsize == 0
        return [(0, page, dtype)] if ok else []
    out, lo = [], 0
    for shape, dt in zip(getattr(spec, "shapes", ()), getattr(spec, "dtypes", ()), strict=False):
        n = math.prod(shape) * dt.itemsize
        if dt.is_floating_point and lo + n <= page:
            out.append((lo, lo + n, dt))
        lo += n
    return out


def request_tags(request: Any) -> dict[str, Any]:
    sp = getattr(request, "sampling_params", None)
    return dict(getattr(sp, "extra_args", None) or {})


@dataclass
class _DigestJob:
    req_id: str
    # Per group: list of (block_index, block_id, prefix_key, covered_tokens).
    blocks: list[list[tuple[int, int, str, int]]]
    # Schedule episode the bytes belong to: set for a preempted request
    # (digested in ``handle_preemptions``); None = finish (last episode).
    episode: int | None = None


@dataclass
class _StateJob:
    """One Mamba state block to digest in this step: an offer in
    ``wait_for_save``, a start state in ``handle_preemptions``."""

    req_id: str
    episode: int
    group: int
    block_id: int
    key: str
    covered: int


@dataclass
class ProbeMetadata(KVConnectorMetadata):
    jobs: list[_DigestJob] = field(default_factory=list)
    states: list[_StateJob] = field(default_factory=list)
    # Episode start states, digested as the forward reads them (V2: after
    # the Mamba precopy, which only reads them, before any sync load).
    start_states: list[_StateJob] = field(default_factory=list)
    # Requests preempted this step: digested before this step's forward can
    # reuse their freed blocks.
    preempted: list[_DigestJob] = field(default_factory=list)


@dataclass
class _Live:
    tags: dict[str, Any]
    episodes: list[list[Any]] = field(default_factory=list)  # [step, local_hit, start]
    progress: list[list[int]] = field(default_factory=list)  # [step, computed_after]
    request: Any = None  # vLLM's Request (token ids for state keys)
    # Mamba state digests: [episode, covered, group, "offer" | "start" | skip reason]
    states: list[list[Any]] = field(default_factory=list)
    # Block table per group (scheduler output: new, appended, resumed) and
    # computed tokens after the last scheduled step: what a preemption frees.
    blocks: list[list[int]] = field(default_factory=list)
    computed: int = 0
    preempt_digested: list[int] = field(default_factory=list)  # episodes
    # Preempted blocks not digested: [episode, group, covered, "zeroed" | "cow"]
    preempt_dropped: list[list[Any]] = field(default_factory=list)


class RsiProbe(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        probe_dir = extra.get("probe_dir") or os.environ.get("RSI_PROBE_DIR")
        if not probe_dir:
            raise ValueError("RsiProbe needs kv_connector_extra_config.probe_dir")
        self._dir = Path(probe_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._digest = str(extra.get("digest", "off"))
        if self._digest not in ("off", "all", "tagged"):
            raise ValueError(f"RsiProbe digest must be off|all|tagged, got {self._digest!r}")
        self._groups = kv_cache_config.kv_cache_groups
        self._state_groups = self._align_state_groups(vllm_config)
        self._skip_groups = self._undigestable_groups(vllm_config)
        self._connector_modules = [str(m) for m in extra.get("connector_modules", [])]

        if role == KVConnectorRole.SCHEDULER:
            self._live: dict[str, _Live] = {}
            self._local_hit: dict[str, int] = {}
            self._step = 0
            self._key_tokens = math.gcd(*(g.kv_cache_spec.block_size for g in self._groups))
            self._pending: list[_DigestJob] = []
            # States a step ended on, not offered in that step: vLLM may offer
            # them one step later (a decode token's hash needs the sampled
            # token). (req_id, group, covered, episode).
            self._expect: list[tuple[str, int, int, int]] = []
            self._req_log = open(self._dir / "requests.jsonl", "a", buffering=1)  # noqa: SIM115
            self._err_log = None
            self._dump_scheduler_spec(vllm_config, kv_cache_config)
        else:
            self._raw: torch.Tensor | None = None
            self._regions: dict[int, list[tuple[str, int, int, int, Any, int]]] = {}
            self._done: set[str] = set()
            self._lock = threading.Lock()
            self._rank = self._rank_label()
            self._digest_log = None
            self._weights: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _align_state_groups(self, vllm_config) -> set[int]:
        """Mamba groups in align mode: digested per step (offers, episode
        starts), never at finish."""
        from vllm.v1.kv_cache_interface import MambaSpec

        if getattr(vllm_config.cache_config, "mamba_cache_mode", None) != "align":
            return set()
        return {
            gid
            for gid, g in enumerate(self._groups)
            if isinstance(g.kv_cache_spec, MambaSpec)
            and not getattr(g.kv_cache_spec, "num_speculative_blocks", 0)
        }

    def _undigestable_groups(self, vllm_config) -> dict[int, str]:
        """Groups whose block contents are not a function of the prefix alone."""
        from vllm.v1.kv_cache_interface import MambaSpec

        mode = getattr(vllm_config.cache_config, "mamba_cache_mode", None)
        out = {}
        for gid, g in enumerate(self._groups):
            if isinstance(g.kv_cache_spec, MambaSpec) and gid not in self._state_groups:
                # Outside align mode a state block is the running state of the
                # request, not the state at the block end (and speculative
                # blocks shift it): not digested (the gate fails an arm whose
                # transferable group is undigested).
                out[gid] = f"MambaSpec with mamba_cache_mode={_jsonable(mode)}: not digested"
        return out

    # ------------------------------------------------------------------ G0
    def _dump_scheduler_spec(self, vllm_config, kv_cache_config) -> None:
        import vllm

        mc, cc, pc = vllm_config.model_config, vllm_config.cache_config, vllm_config.parallel_config
        groups = []
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            groups.append(
                {
                    "id": gid,
                    "spec": _spec_dict(g.kv_cache_spec),
                    "num_layers": len(g.layer_names),
                    "layer_names": list(g.layer_names),
                    "is_eagle_group": g.is_eagle_group,
                    "host_resident": g.host_resident,
                    "enable_kv_transfer": g.enable_kv_transfer,
                    "role": _jsonable(g.role),
                }
            )
        tensors = [
            {
                "size": t.size,
                "layers": list(t.layers),
                "layer_stride": t.layer_stride,
                "block_stride": t.block_stride,
                "offset": t.offset,
                "host_resident": t.host_resident,
            }
            for t in kv_cache_config.kv_cache_tensors
        ]
        dump = {
            "vllm_version": vllm.__version__,
            "model": mc.model,
            "dtype": _jsonable(mc.dtype),
            "max_model_len": mc.max_model_len,
            "hf": _hf_summary(mc.hf_config),
            "parallel": {
                "tp": pc.tensor_parallel_size,
                "pp": pc.pipeline_parallel_size,
                "dcp": getattr(pc, "decode_context_parallel_size", 1),
            },
            "cache": {
                "block_size": cc.block_size,
                "cache_dtype": _jsonable(cc.cache_dtype),
                "enable_prefix_caching": cc.enable_prefix_caching,
                "mamba_cache_mode": _jsonable(getattr(cc, "mamba_cache_mode", None)),
                # Which Mamba / sliding-window states vLLM retains (and offers).
                "prefix_cache_retention_interval": _jsonable(
                    getattr(cc, "prefix_cache_retention_interval", None)
                ),
            },
            "kv_cache_config": {
                "num_blocks": kv_cache_config.num_blocks,
                "hash_block_size": kv_cache_config.hash_block_size,
                "cache_hit_alignment_tokens": kv_cache_config.cache_hit_alignment_tokens,
                "kv_cache_layout": kv_cache_config.kv_cache_layout,
                "kv_tp_replicas": kv_cache_config.kv_tp_replicas,
                "prefix_cache_retention_interval": _jsonable(
                    getattr(kv_cache_config, "prefix_cache_retention_interval", None)
                ),
                "transfer_group_ids": list(kv_cache_config.transfer_group_ids),
                "prefix_cacheable_group_ids": list(kv_cache_config.prefix_cacheable_group_ids),
                "groups": groups,
                "tensors": tensors,
            },
            # Effective compile settings (G0.spec pins them): an arm whose
            # compile differs from the baseline's computes other numbers.
            "compilation": _compilation_summary(vllm_config),
            # Decides the worker step order the probe's digests rely on.
            "model_runner": _model_runner(vllm_config),
            "probe": {
                "schema": SPEC_SCHEMA,
                "digest_skipped_groups": {str(k): v for k, v in self._skip_groups.items()},
                "state_groups": sorted(self._state_groups),
            },
            "modules": module_files(self._connector_modules),
        }
        (self._dir / "spec.json").write_text(json.dumps(dump, indent=1, sort_keys=True))

    def _rank_label(self) -> str:
        # Digest keys include the rank; a fallback label would silently make
        # cross-run comparisons find no reference.
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        return f"pp{get_pp_group().rank_in_group}-tp{get_tp_group().rank_in_group}"

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Every layer view (and raw-buffer specs) aliases one int8 backing
        # allocation; digests address it with KVCacheTensor geometry.
        first = next((p for t in kv_caches.values() for p in _parts(t)), None)
        if first is not None:
            storage = first.untyped_storage()
            self._raw = torch.empty(0, dtype=torch.uint8, device=first.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            self._regions = {gid: self._layer_regions(gid) for gid in range(len(self._groups))}
            self._check_geometry(kv_caches)
        layer_group = {n: gid for gid, g in enumerate(self._groups) for n in g.layer_names}
        base = self._raw.data_ptr() if self._raw is not None else 0
        layers = {}
        for name, t in kv_caches.items():
            layers[name] = {
                "group": layer_group.get(name),
                "tensors": [
                    {
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "dtype": _jsonable(x.dtype),
                        "element_size": x.element_size(),
                        "offset_bytes": x.data_ptr() - base,
                    }
                    for x in _parts(t)
                ],
            }
        out = {
            "rank": self._rank,
            "device": str(first.device) if first is not None else None,
            "raw_bytes": self._raw.numel() if self._raw is not None else 0,
            "layers": layers,
            "modules": module_files(self._connector_modules),
        }
        (self._dir / f"spec-{self._rank}.json").write_text(
            json.dumps(out, indent=1, sort_keys=True)
        )

    def _check_geometry(self, kv_caches: dict[str, Any]) -> None:
        """Fail fast when the digest address math does not match the views."""
        from vllm.v1.kv_cache_interface import group_kernel_blocks

        assert self._raw is not None
        ptr0, size = self._raw.data_ptr(), self._raw.numel()
        for name, t in kv_caches.items():
            for x in _parts(t):
                if x.untyped_storage().data_ptr() != ptr0:
                    raise RuntimeError(
                        f"RsiProbe: layer {name} is not in the shared KV allocation; "
                        "digest geometry would read other memory"
                    )
        for gid, regions in self._regions.items():
            for name, base, bstride, page, spec, nb in regions:
                if base < 0 or base + (nb - 1) * bstride + page > size:
                    raise RuntimeError(f"RsiProbe: layer {name} region exceeds the KV allocation")
                if not getattr(spec, "has_layer_views", True) or name not in kv_caches:
                    continue  # raw-buffer spec: the backend lays it out itself
                for b in sorted({1, nb - 1} - {0}) if nb > 1 else ():
                    lo, hi, total = None, None, 0
                    for x in _parts(kv_caches[name]):
                        try:
                            v = group_kernel_blocks(x, nb)[b]
                        except Exception as e:
                            raise RuntimeError(
                                f"RsiProbe: layer {name} view {tuple(x.shape)} has no block "
                                f"dim of {nb} blocks: {e}"
                            ) from e
                        s, e2 = _span(v, ptr0)
                        lo = s if lo is None else min(lo, s)
                        hi = e2 if hi is None else max(hi, e2)
                        total += v.numel() * v.element_size()
                    start = base + b * bstride
                    if (
                        lo is None
                        or lo < start
                        or hi > start + spec.page_size_bytes
                        or total != page
                    ):
                        raise RuntimeError(
                            f"RsiProbe: group {gid} layer {name} block {b}: view bytes "
                            f"[{lo}, {hi}) total {total} != digest region [{start}, "
                            f"{start + page}) (block_stride {bstride}, page {page})"
                        )

    # ------------------------------------------------------- scheduler side
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # Called with the local prefix-cache hit before any external lookup;
        # the last call before a (re)schedule is the one that counts.
        rid = request.request_id
        self._local_hit[rid] = num_computed_tokens
        if rid not in self._live:
            self._live[rid] = _Live(tags=request_tags(request))
        self._live[rid].request = request
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        self._step += 1
        sched = scheduler_output.num_scheduled_tokens
        states: list[_StateJob] = []
        starts: list[_StateJob] = []
        # First: a request preempted (and possibly resumed) this step is
        # digested from the block table of its previous episode.
        clobbered = _clobbered_blocks(scheduler_output)
        preempted = [
            job
            for rid in sorted(getattr(scheduler_output, "preempted_req_ids", None) or ())
            if (job := self._preempt_job(rid, clobbered)) is not None
        ]
        for req in scheduler_output.scheduled_new_reqs:
            self._episode(req.req_id, req.num_computed_tokens)
            self._advance(req.req_id, req.num_computed_tokens, sched.get(req.req_id, 0))
            self._start_states(req.req_id, req.block_ids, req.num_computed_tokens, starts)
            self._track(req.req_id, req.block_ids, True, req.num_computed_tokens, sched)
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            before = cached.num_computed_tokens[i]
            resumed = rid in cached.resumed_req_ids
            if resumed:
                self._episode(rid, before)
                # A resumed request's new_block_ids replace its block table.
                self._start_states(rid, cached.new_block_ids[i], before, starts)
            self._advance(rid, before, sched.get(rid, 0))
            self._track(rid, cached.new_block_ids[i], resumed, before, sched)
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        offers = getattr(block_state, "boundary_state_offloads", None) or {}
        for rid, entries in offers.items():
            # Only a request this step computes: vLLM also offers a state it
            # caches after an async load landed (not a computation).
            if sched.get(rid, 0) > 0:
                self._offered_states(rid, entries, states)
        if self._state_groups:
            before = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
            before |= dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
            self._unoffered(before, sched, offers)
        jobs, self._pending = self._pending, []
        return ProbeMetadata(jobs=jobs, states=states, start_states=starts, preempted=preempted)

    def _track(self, rid: str, block_ids, replace: bool, before: int, sched) -> None:
        """Keep the block table (``replace``: new or resumed request) and the
        computed tokens after this step."""
        live = self._live.get(rid)
        if live is None:
            return
        groups = [list(b or []) for b in (block_ids or [None] * len(self._groups))]
        if replace or not live.blocks:
            live.blocks = groups
        else:
            for table, new in zip(live.blocks, groups, strict=True):
                table.extend(new)
        live.computed = before + sched.get(rid, 0)

    def _preempt_job(self, rid: str, clobbered: dict[int, str] | None = None) -> _DigestJob | None:
        """The bytes a preempted episode leaves (its fully computed blocks),
        tagged by that episode; without it its computations stay undigested
        and its loads unjudged. ``clobbered`` (block id -> ``zeroed`` |
        ``cow``): freed blocks vLLM reallocated this step and overwrites
        before ``handle_preemptions`` on the V2 runner; they are dropped (no
        reference from them) and recorded in ``preempt_dropped``."""
        live = self._live.get(rid)
        if live is None or not live.episodes or live.request is None:
            return None
        digest = self._digest_mode(live)
        if digest is None:
            return None
        ep = len(live.episodes) - 1
        tokens = list(live.request.all_token_ids)
        computed = min(live.computed, len(tokens))
        per_group = self._entries(tokens, computed, live.episodes[-1][2], digest, live.blocks)
        if clobbered:
            for gid, blocks in enumerate(per_group):
                for b in blocks:
                    if b[1] in clobbered:
                        live.preempt_dropped.append([ep, gid, b[3], clobbered[b[1]]])
                per_group[gid] = [b for b in blocks if b[1] not in clobbered]
        live.preempt_digested.append(ep)
        live.blocks, live.computed = [], 0
        return _DigestJob(rid, per_group, episode=ep) if any(per_group) else None

    def _unoffered(self, before: dict[str, int], sched: dict[str, int], offers) -> None:
        """A step ending exactly on a state boundary computes that state; one
        vLLM offered neither in that step nor in the next is never digested
        (recorded ``not_offered``)."""
        got = {(rid, gid, b) for rid, entries in offers.items() for gid, _, b in entries}
        late, self._expect = self._expect, []
        for rid, gid, after, ep in late:
            live = self._live.get(rid)
            if live is not None and (rid, gid, after) not in got:
                live.states.append([ep, after, gid, "not_offered"])
        for rid, n in sched.items():
            live = self._live.get(rid)
            if n <= 0 or live is None or not live.episodes or rid not in before:
                continue
            if self._digest_mode(live) not in ("full", "ref"):
                continue
            after = before[rid] + n
            for gid in sorted(self._state_groups):
                bt = self._groups[gid].kv_cache_spec.block_size
                if after % bt == 0 and (rid, gid, after) not in got:
                    self._expect.append((rid, gid, after, len(live.episodes) - 1))

    @staticmethod
    def _state_key(live: _Live, covered: int) -> str | None:
        """Prefix key of the first ``covered`` tokens (None: not known yet)."""
        tokens = live.request.all_token_ids if live.request is not None else []
        if len(tokens) < covered:
            return None
        return prefix_keys(list(tokens[:covered]), [covered])[0]

    def _start_states(self, rid: str, block_ids, start: int, out: list[_StateJob]) -> None:
        """The state an episode starts from (local hit or loaded), per state
        group: block ``start / bt - 1`` of the episode's block table."""
        live = self._live.get(rid)
        if not self._state_groups or start <= 0 or live is None:
            return
        if self._digest_mode(live) not in ("full", "hit"):
            return
        ep = len(live.episodes) - 1
        for gid in sorted(self._state_groups):
            bt = self._groups[gid].kv_cache_spec.block_size
            ids = block_ids[gid] if block_ids is not None and gid < len(block_ids) else []
            idx = start // bt - 1
            bid = ids[idx] if start % bt == 0 and idx < len(ids) else None
            key = self._state_key(live, start)
            if start % bt:
                what = "start_unaligned"
            elif not bid:
                what = "start_no_block"
            elif key is None:
                what = "start_no_tokens"
            else:
                what = "start"
                out.append(_StateJob(rid, ep, gid, bid, key, start))
            live.states.append([ep, start, gid, what])

    def _offered_states(self, rid: str, entries, out: list[_StateJob]) -> None:
        """States vLLM offers this step (``(group, block_id, boundary)``):
        written by this step's forward. A boundary at or below the episode
        start is the start state (``_start_states``), not a computation."""
        live = self._live.get(rid)
        if not self._state_groups or live is None or not live.episodes:
            return
        if self._digest_mode(live) not in ("full", "ref"):
            return
        ep, start = len(live.episodes) - 1, live.episodes[-1][2]
        for gid, bid, boundary in entries:
            if gid not in self._state_groups or boundary <= start:
                continue
            bt = self._groups[gid].kv_cache_spec.block_size
            key = self._state_key(live, boundary)
            if boundary % bt:
                what = "offer_unaligned"
            elif key is None:
                what = "offer_no_tokens"
            else:
                what = "offer"
                out.append(_StateJob(rid, ep, gid, bid, key, boundary))
            live.states.append([ep, boundary, gid, what])

    def _episode(self, rid: str, start: int) -> None:
        live = self._live.setdefault(rid, _Live(tags={}))
        live.episodes.append([self._step, self._local_hit.pop(rid, None), start])

    def _advance(self, rid: str, before: int, n: int) -> None:
        live = self._live.get(rid)
        kt = self._key_tokens
        if live is not None and (before + n) // kt > before // kt:
            live.progress.append([self._step, before + n])

    def update_connector_output(self, connector_output):
        invalid = getattr(connector_output, "invalid_block_ids", None) or set()
        failed = getattr(connector_output, "failed_recving", None) or set()
        if not invalid and not failed:
            return
        if self._err_log is None:
            self._err_log = open(self._dir / "load_errors.jsonl", "a", buffering=1)  # noqa: SIM115
        rec = {"step": self._step, "invalid_blocks": len(invalid), "failed_recving": sorted(failed)}
        self._err_log.write(json.dumps(rec) + "\n")

    def request_finished(self, request, block_ids):
        return self._finish(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        return self._finish(request, block_ids)

    def _finish(self, request, block_ids) -> tuple[bool, dict[str, Any] | None]:
        rid = request.request_id
        tokens = list(request.all_token_ids)
        computed = min(request.num_computed_tokens, len(tokens))
        live = self._live.pop(rid, None) or _Live(tags=request_tags(request))
        self._local_hit.pop(rid, None)
        if self._state_groups:
            # vLLM drops a finished request's pending offers.
            for x in [x for x in self._expect if x[0] == rid]:
                live.states.append([x[3], x[2], x[1], "not_offered"])
            self._expect = [x for x in self._expect if x[0] != rid]
        first = live.episodes[0] if live.episodes else [None, None, None]
        kt = self._key_tokens
        digest = self._digest_mode(live)
        self._req_log.write(
            json.dumps(
                {
                    "req_id": rid,
                    "pass": live.tags.get("rsi_pass"),
                    # Workload position (replay: conversation, turn) for
                    # cross-arm matching; None when the workload sets none.
                    "conv": live.tags.get("rsi_conv"),
                    "turn": live.tags.get("rsi_turn"),
                    "prompt_len": request.num_prompt_tokens,
                    "num_tokens": len(tokens),
                    "computed": computed,
                    "first_step": first[0],
                    "local_hit": first[1],
                    "first_computed": first[2],
                    "episodes": live.episodes,
                    "progress": live.progress,
                    **({"states": live.states} if self._state_groups else {}),
                    "preempt_digested": live.preempt_digested,
                    "preempt_dropped": live.preempt_dropped,
                    "finish_step": self._step,
                    "digest": digest,
                    "key_tokens": kt,
                    "keys": prefix_keys(tokens, list(range(kt, computed + 1, kt))),
                }
            )
            + "\n"
        )
        if digest is None:
            return False, None

        # Bytes held at finish come from the last episode: blocks ending at
        # or below its start were loaded (local or external), the rest computed.
        per_group = self._entries(tokens, computed, live.episodes[-1][2], digest, block_ids)
        if not any(per_group):
            return False, None
        self._pending.append(_DigestJob(rid, per_group))
        return True, None

    def _entries(self, tokens, computed: int, start: int, digest: str, block_ids):
        """Per group: (index, block id, key, covered) of the blocks to digest."""
        per_group = []
        for gid in range(len(self._groups)):
            ids = block_ids[gid] if gid < len(block_ids) else []
            if gid in self._skip_groups or gid in self._state_groups:
                # State groups are digested in their steps (_digest_states).
                per_group.append([])
                continue
            bt = self._groups[gid].kv_cache_spec.block_size
            # Only blocks whose token span is fully computed are comparable
            # across runs: a partial attention page depends on where the
            # request stopped.
            n = computed // bt
            entries = [
                (i, bid)
                for i, bid in enumerate(ids[:n])
                if bid and (digest == "full" or (digest == "ref") == ((i + 1) * bt > start))
            ]
            covered = [(i + 1) * bt for i, _ in entries]
            keys = prefix_keys(tokens, covered)
            per_group.append(
                [(i, bid, k, c) for (i, bid), k, c in zip(entries, keys, covered, strict=True)]
            )
        return per_group

    def _digest_mode(self, live: _Live) -> str | None:
        """``full`` (every complete block), ``ref`` (computed blocks only, i.e.
        reference material) or ``hit`` (loaded blocks only); None = no digest."""
        if not live.episodes or self._digest == "off":
            return None
        if self._digest == "all":
            return "full"
        tag = live.tags.get("rsi_digest")
        if not tag:
            return None
        mode = "full" if tag in (1, True, "1", "full") else str(tag)
        if mode not in ("full", "ref", "hit"):
            raise ValueError(f"RsiProbe: rsi_digest must be full|ref|hit, got {tag!r}")
        return mode

    # ---------------------------------------------------------- worker side
    def start_load_kv(self, forward_context, **kwargs) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, ProbeMetadata) or not meta.jobs:
            return
        self._open_digest_log()
        for job in meta.jobs:
            for gid, blocks in enumerate(job.blocks):
                if blocks:
                    self._digest_group(job.req_id, gid, blocks)
            with self._lock:
                self._done.add(job.req_id)

    def _open_digest_log(self) -> None:
        if self._digest_log is None:
            self._digest_log = open(  # noqa: SIM115
                self._dir / f"digest-{self._rank}.jsonl", "a", buffering=1
            )

    def _layer_regions(self, gid: int) -> list[tuple[str, int, int, int, Any, int]]:
        """(layer, base, block_stride, page_bytes, spec, num_blocks) per layer of a group."""
        group = self._groups[gid]
        names = set(group.layer_names)
        out = []
        for t in self._kv_cache_config.kv_cache_tensors:
            if t.host_resident:
                continue
            nb = self._kv_cache_config.num_blocks_of(t)
            for li, name in enumerate(t.layers):
                if name not in names:
                    continue
                spec = group.kv_cache_spec
                if hasattr(spec, "kv_cache_specs"):  # UniformTypeKVCacheSpecs
                    spec = spec.kv_cache_specs[name]
                # Unpadded bytes only: hybrid groups share pages and the
                # padding may hold another group's stale data.
                page = getattr(spec, "real_page_size_bytes", None) or spec.page_size_bytes
                out.append((name, t.offset + li * t.layer_stride, t.block_stride, page, spec, nb))
        return out

    def _weights_for(self, device: torch.device, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, n)
        if key not in self._weights:
            i = torch.arange(n, dtype=torch.int64, device=device)
            w1 = (i * 2654435761 + 97) % 2147483647 | 1
            w2 = (i * 40503 + 2246822519) % 2147483629 | 1
            self._weights[key] = (w1, w2)
        return self._weights[key]

    def _digest_group(self, req_id: str, gid: int, blocks, episode: int | None = None) -> None:
        """``episode``: a state record (digested in its step) or a preempted
        episode's blocks, tagged by that schedule episode; None = a
        finish-time record (last episode)."""
        assert self._raw is not None
        raw = self._raw
        ids = [b[1] for b in blocks]
        rows: dict[int, dict[str, Any]] = {
            b[0]: {"key": b[2], "covered": b[3], "exact": [], "l1": []} for b in blocks
        }
        regions = self._regions[gid]
        for _name, base, bstride, page, spec, nb in regions:
            pages = raw.as_strided((nb, page), (bstride, 1), base)
            parts = _float_parts(spec, page)
            chunk = max(1, _CHUNK_BYTES // page)
            for s in range(0, len(ids), chunk):
                idx = torch.tensor(ids[s : s + chunk], device=raw.device)
                g = pages.index_select(0, idx)
                u = g.to(torch.int64)
                w1, w2 = self._weights_for(u.device, page)
                h1 = (u * w1).sum(-1).tolist()
                h2 = (u * w2).sum(-1).tolist()
                del u
                l1 = [0.0] * len(idx)
                for lo, hi, dtype in parts:
                    part = g[:, lo:hi].contiguous().view(dtype).float().abs().sum(-1).tolist()
                    l1 = [a + b for a, b in zip(l1, part, strict=True)]
                for j, b in enumerate(blocks[s : s + chunk]):
                    rows[b[0]]["exact"].append(f"{h1[j] & _MASK:016x}{h2[j] & _MASK:016x}")
                    rows[b[0]]["l1"].append(l1[j])
        assert self._digest_log is not None
        rec: dict[str, Any] = {
            "req_id": req_id,
            "rank": self._rank,
            "group": gid,
            "layers": [r[0] for r in regions],
            "blocks": [dict(index=i, **r) for i, r in sorted(rows.items())],
        }
        if episode is not None:
            rec["episode"] = episode
        self._digest_log.write(json.dumps(rec) + "\n")

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def handle_preemptions(self, kv_connector_metadata) -> None:
        # vLLM 0.32's V2 runner (what runs; ``spec.json`` ``model_runner``)
        # calls this from ``kv_connector.pre_forward``, after
        # ``update_requests`` (zeroes new blocks, applies COW copies) and the
        # Mamba align precopy, and before a sync ``start_load_kv`` and the
        # forward. Freed preempted blocks still hold their episode's bytes
        # except the ones reallocated and overwritten this step, which
        # ``build_connector_meta`` already dropped (``preempt_dropped``). A
        # start-state block holds what the forward computes from: the
        # precopy only reads it (its destination is a later block), and a
        # sync load lands after this digest.
        meta = kv_connector_metadata
        if not isinstance(meta, ProbeMetadata):
            return
        if meta.preempted:
            self._open_digest_log()
        for job in meta.preempted:
            for gid, blocks in enumerate(job.blocks):
                if blocks:
                    self._digest_group(job.req_id, gid, blocks, episode=job.episode)
        self._digest_states(meta.start_states)

    def wait_for_save(self):
        # Mamba states this step's forward wrote (same stream), before a
        # later step's forward can reuse the block.
        meta = self._get_connector_metadata()
        if isinstance(meta, ProbeMetadata):
            self._digest_states(meta.states)

    def _digest_states(self, states: list[_StateJob]) -> None:
        if not states:
            return
        self._open_digest_log()
        by: dict[tuple[str, int, int], list[tuple[int, int, str, int]]] = {}
        for s in states:
            bt = self._groups[s.group].kv_cache_spec.block_size
            by.setdefault((s.req_id, s.episode, s.group), []).append(
                (s.covered // bt - 1, s.block_id, s.key, s.covered)
            )
        for (rid, ep, gid), blocks in by.items():
            self._digest_group(rid, gid, blocks, episode=ep)

    def get_finished(self, finished_req_ids: set[str]):
        with self._lock:
            done, self._done = self._done, set()
        return (done or None), None

    def shutdown(self):
        for name in ("_req_log", "_err_log", "_digest_log"):
            f = getattr(self, name, None)
            if f is not None:
                f.close()


class RsiWorkerExt:
    """``--worker-extension-cls`` hooks, called through ``POST /collective_rpc``."""

    def rsi_poison(self, value: int = 0x3F) -> dict[str, int]:
        """Fill every KV cache allocation with one byte value.

        The gate calls this right after a successful ``/reset_prefix_cache``
        (no block is held), so the warm pass can only read KV that was
        computed or loaded after this point; stale cold-pass bytes that
        happen to be correct cannot rescue a connector that loads nothing.
        0x3F is finite and non-zero as bf16/fp16/fp32/fp8, so masked
        attention lanes stay finite for honest runs.
        """
        raws: dict[int, torch.Tensor] = {}
        for t in getattr(self.model_runner, "kv_caches", None) or []:
            for x in _parts(t):
                st = x.untyped_storage()
                if st.data_ptr() not in raws:
                    raws[st.data_ptr()] = torch.empty(0, dtype=torch.uint8, device=x.device).set_(
                        st, 0, (st.nbytes(),), (1,)
                    )
        if not raws:
            raise RuntimeError("rsi_poison: model_runner.kv_caches is empty")
        for raw in raws.values():
            raw.fill_(value)
        torch.cuda.synchronize()
        return {"allocations": len(raws), "bytes": sum(r.numel() for r in raws.values())}
