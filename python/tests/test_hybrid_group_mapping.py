"""Regression coverage for mixed logical block-size save/query mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import CacheGroupLayout, ConnectorContext
from pegaflow.connector.scheduler import SchedulerConnector
from pegaflow.connector.tp_shards import ShardedQueryReady


def _layout() -> CacheGroupLayout:
    return CacheGroupLayout(
        layer_names=(("full",), ("sliding",)),
        hash_group_index=0,
        has_recurrent_state=False,
        recurrent_group_indices=frozenset(),
        recurrent_layer_names=frozenset(),
        sliding_window_group_indices=frozenset({1}),
        group_sliding_windows=(None, 32),
        storage_group_ids=(0, 1),
        group_block_sizes=(32, 16),
        layer_block_sizes=((('full', 32),), (('sliding', 16),)),
    )


def _scheduler() -> SchedulerConnector:
    context = ConnectorContext(
        instance_id="test",
        namespace="ns",
        block_size=32,
        hash_block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
    )
    scheduler = SchedulerConnector(context)
    scheduler._cache_groups = _layout()
    return scheduler


def test_save_maps_one_full_block_to_two_sliding_blocks():
    scheduler = _scheduler()
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(
        request_id="r1",
        num_tokens=128,
        num_prompt_tokens=128,
        block_hashes=list(hashes),
    )
    scheduler._requests["r1"] = request
    scheduler._block_hashes["r1"] = scheduler._request_block_hashes(request)
    scheduler._allocated_blocks["r1"] = [
        list(range(10, 14)),
        list(range(20, 28)),
    ]
    scheduler._scheduled_tokens["r1"] = 128
    scheduler._next_stored_block_idx["r1"] = 0

    intent = scheduler._consume_full_block_saves("r1")

    assert intent is not None
    assert intent.block_ids_by_group == (
        (10, 11, 12, 13),
        (20, 21, 22, 23, 24, 25, 26, 27),
    )
    assert intent.block_hashes_by_group == (
        (hashes[1], hashes[3], hashes[5], hashes[7]),
        hashes,
    )


def test_sliding_query_uses_retained_window_suffix():
    scheduler = _scheduler()
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(request_id="r1", num_tokens=128, block_hashes=list(hashes))
    results = []

    def query_group_membership(_instance, block_hashes, _req_id, _group_id):
        results.append(tuple(block_hashes))
        return [(tuple(range(len(block_hashes))), b"lease")]

    scheduler._tp_shard_client.query_group_membership = query_group_membership
    ready = ShardedQueryReady(num_hit_blocks=1, leases=(b"dense",))

    attached = scheduler._attach_sliding_group_queries(
        request, computed_blocks=2, full_hashes=list(hashes[2:]), ready=ready, req_id="r1"
    )

    assert results == [hashes[4:6]]
    assert attached.block_starts_by_group == (0, 4)
    assert attached.hit_positions_by_group == ((0,), (0, 1))
    assert attached.leases_by_group == ((b"dense",), (b"lease",))
