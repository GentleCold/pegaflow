import argparse
import asyncio
import hashlib
import io
import json
import tarfile
from collections import UserDict
from pathlib import Path
from typing import Any

import pytest

from pegaflow.benchmarks.cache_eviction_replay import (
    CompletedRequest,
    MetricSample,
    PreparedRequest,
    RequestResult,
    TraceEntry,
    _extract_token_ids,
    _normalize_text_messages,
    adapt_request,
    compare_command,
    counter_delta,
    iter_trace_entries,
    metric_sum,
    parse_prometheus,
    prepare_entry,
    public_result,
    replay_prepared,
    summarize_metrics,
    theoretical_stats,
)


class FakeBlockHasher:
    block_size = 2
    fingerprint: dict[str, Any] = {"test": True}

    def hash_payload(self, payload: dict[str, Any]) -> tuple[int, tuple[bytes, ...]]:
        count = len(payload["messages"])
        return count * 2, tuple(
            hashlib.sha256(f"block-{index}".encode()).digest() for index in range(count)
        )


def write_trace(path: Path, bodies: list[dict[str, Any]]) -> None:
    lines = []
    for body in bodies:
        lines.append(json.dumps({"request_body": json.dumps(body)}).encode() + b"\n")
    content = b"".join(lines)
    info = tarfile.TarInfo("export/trace.json")
    info.size = len(content)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(content))


def request_result(index: int, *, ok: bool = True) -> RequestResult:
    return RequestResult(
        index=index,
        request_id=f"trace-replay-{index}",
        ok=ok,
        status_code=200 if ok else 400,
        failure_kind=None if ok else "http_4xx",
        prompt_tokens=4 if ok else None,
        full_blocks=2 if ok else 0,
        completion_tokens=1 if ok else None,
        ttft_ms=10.0 if ok else None,
        e2e_ms=20.0 if ok else None,
        dispatch_offset_ms=float(index),
    )


def test_trace_reader_offset_allowlist_and_output_redaction(tmp_path: Path) -> None:
    trace = tmp_path / "trace.tar.gz"
    bodies = [
        {
            "model": "source-model",
            "messages": [{"role": "user", "content": f"private-{index}"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "stream": True,
            "stream_options": {"continuous_usage_stats": True},
            "reasoning_effort": "high",
            "enable_thinking": True,
            "chat_template_kwargs": {"thinking": True},
        }
        for index in range(4)
    ]
    write_trace(trace, bodies)

    entries = list(iter_trace_entries(trace, start_offset=1, max_records=2))

    assert [entry.index for entry in entries] == [1, 2]
    payload = adapt_request(entries[0].body or {}, "served-model", max_tokens=1)
    assert payload["model"] == "served-model"
    assert payload["max_tokens"] == 1
    assert payload["stream_options"] == {
        "continuous_usage_stats": True,
        "include_usage": True,
    }
    assert "reasoning_effort" not in payload
    assert "enable_thinking" not in payload
    assert "chat_template_kwargs" not in payload

    prepared = prepare_entry(entries[0], "served-model", 1, FakeBlockHasher())
    assert isinstance(prepared, PreparedRequest)
    persisted = json.dumps(public_result(request_result(entries[0].index)))
    assert "private-1" not in persisted
    assert "messages" not in persisted
    assert "block_hash" not in persisted


def test_trace_reader_reports_bad_rows_without_silently_skipping(tmp_path: Path) -> None:
    content = b'{"missing":"request_body"}\nnot-json\n'
    info = tarfile.TarInfo("trace.json")
    info.size = len(content)
    trace = tmp_path / "bad.tar.gz"
    with tarfile.open(trace, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(content))

    entries = list(iter_trace_entries(trace, start_offset=0, max_records=2))

    assert [entry.index for entry in entries] == [0, 1]
    assert all(entry.body is None for entry in entries)
    assert entries[0].error_kind == "trace_KeyError"
    assert entries[1].error_kind == "trace_JSONDecodeError"


def test_non_stream_request_drops_stream_options() -> None:
    payload = adapt_request(
        {
            "messages": [{"role": "user", "content": "private"}],
            "stream": False,
            "stream_options": {"include_usage": True},
        },
        served_model="model",
        max_tokens=1,
    )

    assert payload["stream"] is False
    assert "stream_options" not in payload


def test_text_content_normalization_matches_vllm_string_format() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        },
        {"role": "assistant", "content": None, "tool_calls": []},
    ]

    normalized = _normalize_text_messages(messages)

    assert normalized == [
        {"role": "user", "content": "first\nsecond"},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    assert isinstance(messages[0]["content"], list)


def test_text_content_normalization_rejects_non_text_parts() -> None:
    with pytest.raises(ValueError, match="only text chat content parts"):
        _normalize_text_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        )


def test_token_ids_support_non_dict_mapping_results() -> None:
    tokenized = UserDict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})

    assert _extract_token_ids(tokenized) == [1, 2, 3]


def test_dispatcher_starts_requests_in_export_order_and_caps_concurrency() -> None:
    prepared = [
        PreparedRequest(
            index=index,
            payload={"stream": False},
            prompt_tokens=2,
            block_hashes=(hashlib.sha256(str(index).encode()).digest(),),
        )
        for index in range(8)
    ]
    starts: list[int] = []
    active = 0
    max_active = 0

    async def sender(item: PreparedRequest) -> CompletedRequest:
        nonlocal active, max_active
        starts.append(item.index)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.001 * (3 - item.index % 3))
        active -= 1
        return CompletedRequest(request_result(item.index), item.block_hashes)

    completed = asyncio.run(replay_prepared(prepared, concurrency=3, sender=sender))

    assert starts == list(range(8))
    assert max_active == 3
    assert sorted(item.result.index for item in completed) == list(range(8))


def test_theoretical_ratios_use_prefix_first_miss_and_successes_only() -> None:
    hashes = {
        name: hashlib.sha256(name.encode()).digest() for name in ("a", "b", "c", "d", "x")
    }
    completed = [
        CompletedRequest(request_result(0), (hashes["a"], hashes["b"], hashes["c"])),
        CompletedRequest(request_result(1), (hashes["a"], hashes["b"], hashes["d"])),
        CompletedRequest(request_result(2), (hashes["a"], hashes["x"], hashes["c"])),
        CompletedRequest(request_result(3, ok=False), (hashes["a"], hashes["b"])),
    ]

    stats = theoretical_stats(completed, bucket_bits=14)

    assert stats["successful_indices"] == [0, 1, 2]
    assert stats["total_full_block_references"] == 9
    assert stats["exact_distinct_full_blocks"] == 5
    assert stats["prefix_aware_ideal_hit_blocks"] == 3
    assert stats["prefix_aware_ideal_hit_ratio"] == pytest.approx(3 / 9)
    assert stats["hll_theoretical_hit_ratio"] == pytest.approx(4 / 9, abs=0.01)


def test_prometheus_parser_preserves_labels_and_computes_deltas() -> None:
    before_text = """
# HELP ignored ignored
pegaflow_cache_block_hits_total{otel_scope_name="pegaflow-core"} 10
pegaflow_cache_tier_block_requests_total{tier="ram"} 7
pegaflow_cache_tier_block_requests_total{tier="miss"} 3
pegaflow_cache_resident_bytes{class="retained"} 1024
pegaflow_cache_resident_blocks{class="retained"} 2
"""
    after_text = """
pegaflow_cache_block_hits_total{otel_scope_name="pegaflow-core"} 16
pegaflow_cache_tier_block_requests_total{tier="ram"} 11
pegaflow_cache_tier_block_requests_total{tier="miss"} 5
pegaflow_cache_block_evictions_total 2
pegaflow_cache_block_evictions_by_class_total{class="reclaimable"} 2
pegaflow_cache_resident_bytes{class="retained"} 2048
pegaflow_cache_resident_blocks{class="retained"} 4
"""
    before = {"p0": parse_prometheus(before_text)}
    after = {"p0": parse_prometheus(after_text)}

    assert metric_sum(after["p0"], "pegaflow_cache_tier_block_requests", {"tier": "ram"}) == 11
    assert counter_delta(before, after, "pegaflow_cache_block_hits") == 6

    summary = summarize_metrics(before, after)
    assert summary["tier_block_request_deltas"] == {
        "ram": 4,
        "rdma": 0,
        "ssd": 0,
        "miss": 2,
    }
    assert summary["eviction_deltas_by_class"]["reclaimable"] == 2
    assert summary["occupancy"]["p0"]["end_bytes"] == 2048
    assert summary["mean_resident_block_bytes"] == 512


def write_comparison_run(
    run_dir: Path,
    strategy: str,
    actual_ratio: float,
    evictions: int,
    successful_indices: list[int] | None = None,
) -> None:
    run_dir.mkdir()
    config = {
        "trace_sha256": "trace",
        "start_offset": 0,
        "max_records": 2,
        "model": "model",
        "served_model": "model",
        "max_tokens": 1,
        "concurrency": 8,
        "request_rate": "inf",
        "block_size": 128,
        "prefix_caching_hash_algo": "sha256",
        "python_hash_seed": "0",
        "tokenizer": {"tokenizer_sha256": "tokenizer"},
        "git": {"branch": strategy, "commit": strategy},
    }
    block_stats = {
        "successful_indices": successful_indices or [0, 1],
        "total_full_block_references": 10,
        "hll_theoretical_hit_ratio": 0.5,
        "prefix_aware_ideal_hit_ratio": 0.4,
    }
    summary = {
        "strategy": strategy,
        "requests": {"successful": 2},
        "actual_external_hit_ratio": actual_ratio,
        "successful_request_throughput": 1.0,
        "estimated_unique_footprint_bytes": 100,
        "ttft": {"mean_ms": 10, "p99_ms": 20},
        "e2e": {"mean_ms": 20, "p99_ms": 30},
        "metrics": {
            "counter_deltas": {
                "pegaflow_cache_block_evictions": evictions,
                "pegaflow_cache_block_admission_rejections": 0,
                "pegaflow_load_failures": 0,
            },
            "occupancy": {"p0": {"end_bytes": 100}},
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "block_stats.json").write_text(json.dumps(block_stats), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_compare_sorts_by_actual_ratio_and_requires_eviction(tmp_path: Path) -> None:
    first = tmp_path / "lru"
    second = tmp_path / "s3fifo"
    output = tmp_path / "comparison"
    write_comparison_run(first, "lru", 0.3, evictions=2)
    write_comparison_run(second, "s3fifo", 0.4, evictions=0)

    compare_command(
        argparse.Namespace(
            run_dir=[str(first), str(second)],
            output_dir=str(output),
        )
    )

    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "inconclusive_no_eviction_pressure"
    assert [row["strategy"] for row in comparison["rows"]] == ["s3fifo", "lru"]


def test_compare_rejects_different_success_sets(tmp_path: Path) -> None:
    first = tmp_path / "lru"
    second = tmp_path / "s3fifo"
    write_comparison_run(first, "lru", 0.3, evictions=2)
    write_comparison_run(second, "s3fifo", 0.4, evictions=2, successful_indices=[0])

    with pytest.raises(ValueError, match="successful request set mismatch"):
        compare_command(
            argparse.Namespace(
                run_dir=[str(first), str(second)],
                output_dir=str(tmp_path / "comparison"),
            )
        )


def test_metric_sample_shape_is_stable() -> None:
    assert MetricSample("metric", (("class", "retained"),), 1.0).labels == (
        ("class", "retained"),
    )


def test_prepare_error_does_not_persist_exception_text() -> None:
    class FailingHasher:
        block_size = 128
        fingerprint: dict[str, Any] = {}

        def hash_payload(self, payload: dict[str, Any]) -> tuple[int, tuple[bytes, ...]]:
            raise RuntimeError(payload["messages"][0]["content"])

    outcome = prepare_entry(
        TraceEntry(0, {"messages": [{"role": "user", "content": "private-value"}]}),
        "model",
        1,
        FailingHasher(),
    )

    assert isinstance(outcome, RequestResult)
    persisted = json.dumps(public_result(outcome))
    assert outcome.failure_kind == "prepare_RuntimeError"
    assert "private-value" not in persisted
