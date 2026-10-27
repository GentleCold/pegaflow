"""Policy: choices inside the legal range computed by ``pegaflow.semantics``.

This is the main RSI surface. A policy can only lower what semantics allows
(accept a shorter hit, save fewer blocks). It receives immutable copies, and
the core validates every answer against what it offered: ``accept`` is capped
at the legal ``p`` and re-checked against the server answers, and
``choose_saves`` may only return offered items (anything else raises). So a
bad policy costs hit rate, never correctness.

The knobs are constants shared by every model (no model overrides one).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SaveItem:
    """One block the core may save this step."""

    req_id: str
    group: int  # vLLM kv_cache_group index
    kind: str  # semantics Group kind: full / window / checkpoint
    block: int  # logical block index within the group
    block_id: int  # GPU block id
    block_hash: bytes


@dataclass
class Policy:
    # How long a request may wait while the server prefetches, and, on its own
    # clock, while another request's async load of its blocks lands; 0 = never.
    loading_wait_ms: int = 1000
    # Hits of at most this many tokens load before the step's forward (where the
    # layout allows it), longer ones asynchronously between steps.
    sync_load_tokens: int = 256
    # Merge consecutive steps' saves into one save RPC: send when the oldest
    # held step has seen save_batch_steps steps (<= 1 sends every step), when
    # the GPU KV cache is under pressure, or after the worker's staleness cap
    # (Worker.SAVE_HOLD_MAX_S). Preemption flushes, fences on a held save
    # and shutdown send at once.
    save_batch_steps: int = 32
    # Send held saves at once while vLLM's GPU KV cache usage (0..1, after
    # the step's allocations) is at or above this: freed blocks are reused
    # soon, and a fence on a held save would wait for the batch. > 1 disables it.
    save_batch_max_kv_usage: float = 0.9

    def accept(self, p: int, present: Sequence[frozenset[int]]) -> int:
        """Hit length to use, <= ``p``; ``present[sid]`` = blocks the server has."""
        return p  # default: the longest legal hit

    def choose_saves(self, items: Sequence[SaveItem]) -> Sequence[SaveItem]:
        """Subset of ``items`` to save this step."""
        return items  # default: save everything semantics allows
