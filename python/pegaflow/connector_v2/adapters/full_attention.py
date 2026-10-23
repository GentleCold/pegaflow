"""vLLM ``FullAttentionSpec``: standard K/V attention, every past token attended.

Stored unit: one manager block of one layer = one KV page, copied as ONE
segment. vLLM 0.32 lays a page out as ``[H, N, C]`` or ``[N, H, C]`` with
``C = K | V`` per (head, token) cell, so K and V are interleaved inside the
page and cannot be split into separate segments; a page is the natural copy
unit and is what the attention kernel reads back.

Supported geometry: every layer view stores its blocks as dense pages
``page_size_bytes`` apart (layer-compact layouts such as LBNHC / LBHNC). The
server's register path addresses block ``b`` at ``data_ptr + b * page``, so a
block-outer layout (e.g. BLNHC, cross-layer pages) is ``Unsupported`` here.
"""

from __future__ import annotations

from typing import Any

from . import LayerDecl, LayerView, Parallel, Unsupported


class FullAttentionAdapter:
    def __init__(self, semantics: Any | None = None):
        self._semantics = semantics

    @property
    def sem(self) -> Any:
        if self._semantics is None:
            from pegaflow import semantics

            self._semantics = semantics
        return self._semantics

    # ------------------------------------------------------------------ group
    def group(self, spec: Any) -> Any:
        _check_spec(spec)
        return self.sem.Group.full(spec.block_size)

    # ------------------------------------------------------------------ layer
    def layer(self, spec: Any, layer_name: str, par: Parallel) -> LayerDecl:
        _check_spec(spec)
        return LayerDecl((spec.page_size_bytes,), self._shard(spec, par))

    def _shard(self, spec: Any, par: Parallel) -> Any:
        if par.dcp_size > 1 and getattr(spec, "dcp_sharded", True):
            raise Unsupported("FullAttentionSpec: DCP token sharding is not declared yet")
        # vLLM splits KV heads across TP; ranks beyond max_tp_shards hold
        # replicas of consecutive ranks (KVCacheConfig.kv_tp_replicas).
        shards = min(par.tp_size, spec.max_tp_shards or par.tp_size)
        if par.tp_size % shards:
            raise Unsupported(
                f"FullAttentionSpec: tp_size {par.tp_size} not a multiple of "
                f"max_tp_shards {spec.max_tp_shards}"
            )
        return self.sem.Shard.heads(par.tp_rank // (par.tp_size // shards), shards)

    # ------------------------------------------------------------------- view
    def view(self, spec: Any, layer_name: str, kv: Any) -> LayerView:
        """``kv``: vLLM's per-layer view ``[B, H, N, C]`` in the spec dtype.

        ``B`` counts kernel blocks when vLLM splits a manager block for the
        attention backend; vLLM only allows that for dense pages, so grouping
        them back is a reshape of dim 0.
        """
        page = spec.page_size_bytes
        shape, stride, elem = tuple(kv.shape), tuple(kv.stride()), kv.element_size()
        if len(shape) != 4:
            raise Unsupported(f"{layer_name}: expected a [B, H, N, C] view, got shape {shape}")
        # One kernel block's [H, N, C] cells must tile a dense byte range, and
        # kernel splitting only shortens N: block_size / N of them make a page.
        cells = shape[1] * shape[2] * shape[3] * elem
        ratio = spec.block_size // shape[2] if shape[2] else 0
        if not _dense(shape[1:], stride[1:]) or ratio * shape[2] != spec.block_size:
            raise Unsupported(
                f"{layer_name}: page cells are not dense (shape {shape}, stride {stride})"
            )
        if cells * ratio != page:
            raise Unsupported(
                f"{layer_name}: {ratio} x {cells} B kernel blocks != page_size_bytes {page}"
            )
        if stride[0] * elem != cells or shape[0] % ratio:
            raise Unsupported(
                f"{layer_name}: blocks are {stride[0] * elem} B apart, need dense "
                f"{cells} B kernel blocks tiling {page} B pages (layer-compact KV "
                f"layout, e.g. VLLM_KV_CACHE_LAYOUT=LBNHC)"
            )
        return LayerView(kv, shape[0] // ratio, page, page)


# Fields that change the page layout or meaning in ways this adapter has not
# been validated on; each needs its own declaration when a model brings it.
_DEFAULT_FIELDS = {
    "sliding_window": None,
    "attention_chunk_size": None,
    "page_size_padded": None,
    "num_head_slots": None,
    "state_content_bytes": None,
    "tokens_per_state": 1,
    "kv_quant_mode": 0,
}


def _dense(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """Elements of ``shape``/``stride`` (in elements) cover 0..prod-1 exactly once."""
    expect = 1
    for n, s in sorted(
        (d for d in zip(shape, stride, strict=True) if d[0] != 1), key=lambda d: d[1]
    ):
        if s != expect:
            return False
        expect *= n
    return True


def _check_spec(spec: Any) -> None:
    odd = {
        name: getattr(spec, name)
        for name, default in _DEFAULT_FIELDS.items()
        if getattr(spec, name, default) != default
    }
    if odd:
        raise Unsupported(f"FullAttentionSpec with {odd} is not supported yet")
    if not getattr(spec, "has_layer_views", True):
        raise Unsupported("FullAttentionSpec without per-layer views is not supported")


ADAPTER = FullAttentionAdapter()
