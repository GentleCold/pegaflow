"""Replay exported chat traces and compare PegaFlow cache replacement policies.

The replay output is intentionally metadata-only. Prompts, messages, token IDs,
block hashes, response bodies, and server error bodies are never persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

import httpx

DEFAULT_BLOCK_SIZE = 128
DEFAULT_CONCURRENCY = 8
DEFAULT_HASH_ALGO = "sha256"
DEFAULT_HLL_BUCKET_BITS = 14

ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "min_tokens",
        "logprobs",
        "top_logprobs",
        "n",
        "best_of",
        "use_beam_search",
        "length_penalty",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
        "priority",
    }
)

COUNTER_METRICS = (
    "pegaflow_cache_block_hits",
    "pegaflow_cache_block_misses",
    "pegaflow_cache_block_insertions",
    "pegaflow_cache_block_admission_rejections",
    "pegaflow_cache_block_evictions",
    "pegaflow_cache_block_evictions_still_referenced",
    "pegaflow_cache_eviction_reclaimed_bytes",
    "pegaflow_load_failures",
)


class BlockHasher(Protocol):
    block_size: int
    fingerprint: dict[str, Any]

    def hash_payload(self, payload: dict[str, Any]) -> tuple[int, tuple[bytes, ...]]: ...


@dataclass(frozen=True)
class TraceEntry:
    index: int
    body: dict[str, Any] | None
    error_kind: str | None = None


@dataclass(frozen=True)
class PreparedRequest:
    index: int
    payload: dict[str, Any]
    prompt_tokens: int
    block_hashes: tuple[bytes, ...]


@dataclass
class RequestResult:
    index: int
    request_id: str
    ok: bool
    status_code: int | None
    failure_kind: str | None
    prompt_tokens: int | None
    full_blocks: int
    completion_tokens: int | None
    ttft_ms: float | None
    e2e_ms: float | None
    dispatch_offset_ms: float | None


@dataclass(frozen=True)
class CompletedRequest:
    result: RequestResult
    block_hashes: tuple[bytes, ...] = field(repr=False)


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


class HyperLogLog:
    """Small HLL implementation matching ``pegaflow-common`` semantics."""

    def __init__(self, bucket_bits: int = DEFAULT_HLL_BUCKET_BITS) -> None:
        if not 4 <= bucket_bits <= 18:
            raise ValueError("bucket_bits must be in 4..=18")
        self.bucket_bits = bucket_bits
        self.registers = [0] * (1 << bucket_bits)

    def insert(self, value: bytes) -> None:
        if not value:
            raise ValueError("HLL input cannot be empty")
        bit_count = len(value) * 8
        number = int.from_bytes(value, "big")
        index = number >> (bit_count - self.bucket_bits)
        remainder_bits = bit_count - self.bucket_bits
        remainder = number & ((1 << remainder_bits) - 1)
        rho = remainder_bits - remainder.bit_length() + 1 if remainder else remainder_bits + 1
        self.registers[index] = max(self.registers[index], rho)

    def cardinality(self) -> float:
        register_count = len(self.registers)
        alpha = {
            16: 0.673,
            32: 0.697,
            64: 0.709,
        }.get(register_count, 0.7213 / (1 + 1.079 / register_count))
        harmonic_sum = sum(2.0**-register for register in self.registers)
        raw = alpha * register_count * register_count / harmonic_sum
        if raw <= 2.5 * register_count:
            zeros = self.registers.count(0)
            if zeros:
                return register_count * math.log(register_count / zeros)
        return raw


def _normalize_text_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match vLLM's string-format normalization for text-only chat content."""

    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if content is None:
            item["content"] = ""
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])
                else:
                    raise ValueError("only text chat content parts are supported")
            item["content"] = "\n".join(text_parts)
        normalized.append(item)
    return normalized


def _extract_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    if not isinstance(tokenized, list) or any(
        not isinstance(item, int) for item in tokenized
    ):
        raise TypeError("chat template did not return a flat token ID list")
    return tokenized


class VllmBlockHasher:
    """Tokenize chat payloads and reproduce vLLM's chained block hashes."""

    def __init__(
        self,
        model: str,
        block_size: int,
        hash_algo: str,
        trust_remote_code: bool,
    ) -> None:
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise RuntimeError("PYTHONHASHSEED=0 is required for reproducible vLLM hashes")

        try:
            from transformers import AutoTokenizer
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core import kv_cache_utils
        except ImportError as exc:
            raise RuntimeError("transformers and vLLM are required for trace replay") from exc

        self.block_size = block_size
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        self._hash_fn = get_hash_fn_by_name(hash_algo)
        self._kv_cache_utils = kv_cache_utils
        kv_cache_utils.init_none_hash(self._hash_fn)

        fingerprint_payload = {
            "name_or_path": self._tokenizer.name_or_path,
            "vocab_size": self._tokenizer.vocab_size,
            "chat_template": self._tokenizer.chat_template,
            "special_tokens_map": self._tokenizer.special_tokens_map,
            "commit_hash": self._tokenizer.init_kwargs.get("_commit_hash"),
        }
        encoded = json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ).encode()
        self.fingerprint = {
            "model": model,
            "hash_algo": hash_algo,
            "block_size": block_size,
            "chat_content_format": "string",
            "tokenizer_sha256": hashlib.sha256(encoded).hexdigest(),
            "tokenizer_name_or_path": self._tokenizer.name_or_path,
            "model_revision": self._tokenizer.init_kwargs.get("_commit_hash"),
        }

    def hash_payload(self, payload: dict[str, Any]) -> tuple[int, tuple[bytes, ...]]:
        template_args: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if payload.get("tools"):
            template_args["tools"] = payload["tools"]
        messages = _normalize_text_messages(payload["messages"])
        token_ids = _extract_token_ids(
            self._tokenizer.apply_chat_template(messages, **template_args)
        )

        hashes: list[bytes] = []
        parent: bytes | None = None
        full_tokens = len(token_ids) // self.block_size * self.block_size
        for start in range(0, full_tokens, self.block_size):
            parent = bytes(
                self._kv_cache_utils.hash_block_tokens(
                    self._hash_fn,
                    parent,
                    token_ids[start : start + self.block_size],
                    None,
                )
            )
            hashes.append(parent)
        return len(token_ids), tuple(hashes)


def iter_trace_entries(
    archive_path: Path,
    start_offset: int,
    max_records: int,
) -> Iterator[TraceEntry]:
    """Stream records from tar.gz without extracting the JSONL member."""

    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    selected = 0
    line_index = 0
    found_file = False
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            found_file = True
            source = archive.extractfile(member)
            if source is None:
                continue
            for raw_line in source:
                current_index = line_index
                line_index += 1
                if current_index < start_offset:
                    continue
                if selected >= max_records:
                    return
                selected += 1
                try:
                    row = json.loads(raw_line)
                    raw_body = row["request_body"]
                    body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
                    if not isinstance(body, dict):
                        raise TypeError("request_body must decode to an object")
                    yield TraceEntry(index=current_index, body=body)
                except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
                    yield TraceEntry(
                        index=current_index,
                        body=None,
                        error_kind=f"trace_{type(exc).__name__}",
                    )
    if not found_file:
        raise ValueError("trace archive contains no regular file")


def adapt_request(
    body: dict[str, Any],
    served_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {key: body[key] for key in ALLOWED_REQUEST_FIELDS if key in body}
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    payload["model"] = served_model
    payload["max_tokens"] = max_tokens
    payload["stream"] = bool(payload.get("stream", True))
    if payload["stream"]:
        stream_options = payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        payload["stream_options"] = {**stream_options, "include_usage": True}
    else:
        payload.pop("stream_options", None)
    return payload


def prepare_entry(
    entry: TraceEntry,
    served_model: str,
    max_tokens: int,
    block_hasher: BlockHasher,
) -> PreparedRequest | RequestResult:
    request_id = f"trace-replay-{entry.index}"
    if entry.body is None:
        return RequestResult(
            index=entry.index,
            request_id=request_id,
            ok=False,
            status_code=None,
            failure_kind=entry.error_kind or "trace_error",
            prompt_tokens=None,
            full_blocks=0,
            completion_tokens=None,
            ttft_ms=None,
            e2e_ms=None,
            dispatch_offset_ms=None,
        )
    try:
        payload = adapt_request(entry.body, served_model, max_tokens)
        prompt_tokens, hashes = block_hasher.hash_payload(payload)
    except Exception as exc:
        return RequestResult(
            index=entry.index,
            request_id=request_id,
            ok=False,
            status_code=None,
            failure_kind=f"prepare_{type(exc).__name__}",
            prompt_tokens=None,
            full_blocks=0,
            completion_tokens=None,
            ttft_ms=None,
            e2e_ms=None,
            dispatch_offset_ms=None,
        )
    return PreparedRequest(
        index=entry.index,
        payload=payload,
        prompt_tokens=prompt_tokens,
        block_hashes=hashes,
    )


def _usage(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        int(completion_tokens) if isinstance(completion_tokens, int) else None,
    )


async def _consume_sse(
    response: httpx.Response,
    started: float,
) -> tuple[int | None, int | None, float | None, str | None]:
    prompt_tokens = None
    completion_tokens = None
    ttft_ms = None
    saw_done = False
    try:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                continue
            chunk = json.loads(data)
            if ttft_ms is None and chunk.get("choices"):
                ttft_ms = (time.perf_counter() - started) * 1000
            chunk_prompt, chunk_completion = _usage(chunk)
            if chunk_prompt is not None:
                prompt_tokens = chunk_prompt
            if chunk_completion is not None:
                completion_tokens = chunk_completion
    except (json.JSONDecodeError, UnicodeDecodeError, httpx.HTTPError) as exc:
        return prompt_tokens, completion_tokens, ttft_ms, f"stream_{type(exc).__name__}"
    if not saw_done:
        return prompt_tokens, completion_tokens, ttft_ms, "stream_incomplete"
    return prompt_tokens, completion_tokens, ttft_ms, None


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    prepared: PreparedRequest,
    replay_started: float,
) -> CompletedRequest:
    started = time.perf_counter()
    request_id = f"trace-replay-{prepared.index}"
    status_code = None
    prompt_tokens = None
    completion_tokens = None
    ttft_ms = None
    failure_kind = None
    try:
        async with client.stream(
            "POST",
            endpoint,
            json=prepared.payload,
            headers={"X-Request-Id": request_id},
        ) as response:
            status_code = response.status_code
            if not response.is_success:
                await response.aread()
                failure_kind = f"http_{response.status_code // 100}xx"
            elif prepared.payload["stream"]:
                prompt_tokens, completion_tokens, ttft_ms, failure_kind = await _consume_sse(
                    response, started
                )
            else:
                raw = await response.aread()
                parsed = json.loads(raw)
                prompt_tokens, completion_tokens = _usage(parsed)
                ttft_ms = (time.perf_counter() - started) * 1000
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        failure_kind = f"request_{type(exc).__name__}"

    if failure_kind is None and prompt_tokens is None:
        failure_kind = "missing_usage"
    if failure_kind is None and prompt_tokens != prepared.prompt_tokens:
        failure_kind = "prompt_token_mismatch"
    e2e_ms = (time.perf_counter() - started) * 1000
    ok = failure_kind is None
    result = RequestResult(
        index=prepared.index,
        request_id=request_id,
        ok=ok,
        status_code=status_code,
        failure_kind=failure_kind,
        prompt_tokens=prompt_tokens if prompt_tokens is not None else prepared.prompt_tokens,
        full_blocks=len(prepared.block_hashes) if ok else 0,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        dispatch_offset_ms=(started - replay_started) * 1000,
    )
    return CompletedRequest(result, prepared.block_hashes if ok else ())


async def replay_prepared(
    prepared_requests: Iterable[PreparedRequest],
    concurrency: int,
    sender: Callable[[PreparedRequest], Any],
) -> list[CompletedRequest]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    queue: asyncio.Queue[PreparedRequest | None] = asyncio.Queue()
    for prepared in prepared_requests:
        queue.put_nowait(prepared)
    for _ in range(concurrency):
        queue.put_nowait(None)

    completed: list[CompletedRequest] = []

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                completed.append(await sender(item))
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await queue.join()
    await asyncio.gather(*workers)
    return completed


async def run_replay(
    prepared_requests: Sequence[PreparedRequest],
    endpoint: str,
    concurrency: int,
    timeout_s: float,
    api_key: str | None,
) -> tuple[list[CompletedRequest], float]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(timeout_s if timeout_s > 0 else None)
    replay_started = time.perf_counter()
    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:

        async def sender(prepared: PreparedRequest) -> CompletedRequest:
            return await send_request(client, endpoint, prepared, replay_started)

        completed = await replay_prepared(prepared_requests, concurrency, sender)
    return completed, time.perf_counter() - replay_started


_METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([\d.eE+-]+)(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse_prometheus(text: str) -> list[MetricSample]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = tuple(sorted(_LABEL_RE.findall(raw_labels or "")))
        samples.append(MetricSample(name=name, labels=labels, value=float(raw_value)))
    return samples


def metric_sum(
    samples: Sequence[MetricSample],
    name: str,
    required_labels: dict[str, str] | None = None,
) -> float:
    names = {name, f"{name}_total"} if not name.endswith("_total") else {name}
    required_labels = required_labels or {}
    total = 0.0
    for sample in samples:
        if sample.name not in names:
            continue
        labels = dict(sample.labels)
        if all(labels.get(key) == value for key, value in required_labels.items()):
            total += sample.value
    return total


def fetch_metrics(endpoints: dict[str, str]) -> tuple[dict[str, str], dict[str, list[MetricSample]]]:
    raw = {}
    parsed = {}
    for name, url in endpoints.items():
        request = Request(url, headers={"Accept": "text/plain"})
        with urlopen(request, timeout=10) as response:
            content = response.read().decode("utf-8", errors="replace")
        raw[name] = content
        parsed[name] = parse_prometheus(content)
    return raw, parsed


def write_raw_metrics(path: Path, raw: dict[str, str], endpoints: dict[str, str]) -> None:
    chunks = []
    for name in sorted(raw):
        chunks.append(f"# PEGAFLOW_REPLAY_ENDPOINT name={name} url={endpoints[name]}\n")
        chunks.append(raw[name].rstrip() + "\n")
    path.write_text("".join(chunks), encoding="utf-8")


def counter_delta(
    before: dict[str, list[MetricSample]],
    after: dict[str, list[MetricSample]],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    return sum(
        metric_sum(after[endpoint], name, labels) - metric_sum(before[endpoint], name, labels)
        for endpoint in before
    )


def summarize_metrics(
    before: dict[str, list[MetricSample]],
    after: dict[str, list[MetricSample]],
) -> dict[str, Any]:
    counters = {name: counter_delta(before, after, name) for name in COUNTER_METRICS}
    tiers = {
        tier: counter_delta(
            before,
            after,
            "pegaflow_cache_tier_block_requests",
            {"tier": tier},
        )
        for tier in ("ram", "rdma", "ssd", "miss")
    }
    classes = {
        cache_class: counter_delta(
            before,
            after,
            "pegaflow_cache_block_evictions_by_class",
            {"class": cache_class},
        )
        for cache_class in ("reclaimable", "retained")
    }
    occupancy = {}
    resident_block_bytes = []
    for endpoint in before:
        start_bytes = metric_sum(before[endpoint], "pegaflow_cache_resident_bytes")
        end_bytes = metric_sum(after[endpoint], "pegaflow_cache_resident_bytes")
        start_blocks = metric_sum(before[endpoint], "pegaflow_cache_resident_blocks")
        end_blocks = metric_sum(after[endpoint], "pegaflow_cache_resident_blocks")
        occupancy[endpoint] = {
            "start_bytes": start_bytes,
            "end_bytes": end_bytes,
            "start_blocks": start_blocks,
            "end_blocks": end_blocks,
        }
        if end_bytes > 0 and end_blocks > 0:
            resident_block_bytes.append(end_bytes / end_blocks)
    return {
        "counter_deltas": counters,
        "tier_block_request_deltas": tiers,
        "eviction_deltas_by_class": classes,
        "occupancy": occupancy,
        "mean_resident_block_bytes": (
            sum(resident_block_bytes) / len(resident_block_bytes)
            if resident_block_bytes
            else None
        ),
    }


def theoretical_stats(
    completed: Sequence[CompletedRequest],
    bucket_bits: int,
) -> dict[str, Any]:
    successful = sorted((item for item in completed if item.result.ok), key=lambda item: item.result.index)
    hll = HyperLogLog(bucket_bits)
    exact: set[bytes] = set()
    seen: set[bytes] = set()
    ideal_hits = 0
    references = 0
    for item in successful:
        references += len(item.block_hashes)
        for block_hash in item.block_hashes:
            hll.insert(block_hash)
            exact.add(block_hash)
        for block_hash in item.block_hashes:
            if block_hash not in seen:
                break
            ideal_hits += 1
        seen.update(item.block_hashes)
    estimated_distinct = hll.cardinality() if references else 0.0
    return {
        "successful_indices": [item.result.index for item in successful],
        "total_full_block_references": references,
        "exact_distinct_full_blocks": len(exact),
        "hll_bucket_bits": bucket_bits,
        "hll_estimated_distinct_full_blocks": estimated_distinct,
        "hll_theoretical_hit_ratio": (
            max(0.0, min(1.0, (references - estimated_distinct) / references))
            if references
            else None
        ),
        "prefix_aware_ideal_hit_blocks": ideal_hits,
        "prefix_aware_ideal_hit_ratio": ideal_hits / references if references else None,
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_stats(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo: Path) -> dict[str, str | None]:
    def command(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "branch": command("branch", "--show-current"),
        "commit": command("rev-parse", "HEAD"),
        "upstream_master": command("rev-parse", "upstream/master"),
    }


def public_result(result: RequestResult) -> dict[str, Any]:
    return asdict(result)


def build_summary(
    strategy: str,
    results: Sequence[CompletedRequest],
    preparation_failures: Sequence[RequestResult],
    replay_duration_s: float,
    block_stats: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    all_results = [item.result for item in results] + list(preparation_failures)
    successful = [result for result in all_results if result.ok]
    failures: dict[str, int] = {}
    for result in all_results:
        if not result.ok:
            key = result.failure_kind or "unknown"
            failures[key] = failures.get(key, 0) + 1
    ttfts = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    e2es = [result.e2e_ms for result in successful if result.e2e_ms is not None]
    denominator = block_stats["total_full_block_references"]
    actual_hits = metrics["counter_deltas"]["pegaflow_cache_block_hits"]
    mean_block_bytes = metrics["mean_resident_block_bytes"]
    exact_distinct = block_stats["exact_distinct_full_blocks"]
    return {
        "strategy": strategy,
        "requests": {
            "selected": len(all_results),
            "successful": len(successful),
            "failed": len(all_results) - len(successful),
            "failure_kinds": failures,
        },
        "block_stats": block_stats,
        "actual_external_hit_blocks": actual_hits,
        "actual_external_hit_ratio": actual_hits / denominator if denominator else None,
        "actual_ratio_valid": bool(denominator and 0 <= actual_hits <= denominator),
        "estimated_unique_footprint_bytes": (
            exact_distinct * mean_block_bytes if mean_block_bytes is not None else None
        ),
        "replay_duration_s": replay_duration_s,
        "successful_request_throughput": (
            len(successful) / replay_duration_s if replay_duration_s > 0 else None
        ),
        "ttft": latency_stats(ttfts),
        "e2e": latency_stats(e2es),
        "metrics": metrics,
    }


def parse_named_urls(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for index, value in enumerate(values):
        if "=" in value:
            name, url = value.split("=", 1)
        else:
            name, url = f"server{index}", value
        if not name or not url:
            raise ValueError(f"invalid named URL: {value}")
        if name in parsed:
            raise ValueError(f"duplicate metrics endpoint name: {name}")
        parsed[name] = url
    if not parsed:
        raise ValueError("at least one --metrics-url is required")
    return parsed


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def replay_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_empty_output_dir(output_dir)
    trace_path = Path(args.trace).expanduser().resolve()
    metrics_endpoints = parse_named_urls(args.metrics_url)
    block_hasher = VllmBlockHasher(
        model=args.model,
        block_size=args.block_size,
        hash_algo=args.prefix_caching_hash_algo,
        trust_remote_code=args.trust_remote_code,
    )

    prepared = []
    preparation_failures = []
    for entry in iter_trace_entries(trace_path, args.start_offset, args.max_records):
        outcome = prepare_entry(entry, args.served_model, args.max_tokens, block_hasher)
        if isinstance(outcome, PreparedRequest):
            prepared.append(outcome)
        else:
            preparation_failures.append(outcome)

    raw_start, parsed_start = fetch_metrics(metrics_endpoints)
    write_raw_metrics(output_dir / "metrics_start.prom", raw_start, metrics_endpoints)
    completed, replay_duration_s = asyncio.run(
        run_replay(
            prepared,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        )
    )
    raw_end, parsed_end = fetch_metrics(metrics_endpoints)
    write_raw_metrics(output_dir / "metrics_end.prom", raw_end, metrics_endpoints)

    completed.sort(key=lambda item: item.result.index)
    preparation_failures.sort(key=lambda item: item.index)
    all_public_results = sorted(
        [item.result for item in completed] + preparation_failures,
        key=lambda item: item.index,
    )
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as destination:
        for result in all_public_results:
            destination.write(json.dumps(public_result(result), sort_keys=True) + "\n")

    block_stats = theoretical_stats(completed, args.hll_bucket_bits)
    (output_dir / "block_stats.json").write_text(
        json.dumps(block_stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics = summarize_metrics(parsed_start, parsed_end)
    summary = build_summary(
        args.strategy,
        completed,
        preparation_failures,
        replay_duration_s,
        block_stats,
        metrics,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    repo = Path(__file__).resolve().parents[3]
    run_config = {
        "strategy": args.strategy,
        "trace_sha256": sha256_file(trace_path),
        "start_offset": args.start_offset,
        "max_records": args.max_records,
        "endpoint": args.endpoint,
        "metrics_endpoints": metrics_endpoints,
        "model": args.model,
        "served_model": args.served_model,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "request_rate": "inf",
        "block_size": args.block_size,
        "prefix_caching_hash_algo": args.prefix_caching_hash_algo,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "tokenizer": block_hasher.fingerprint,
        "git": git_metadata(repo),
        "api_key_configured": bool(args.api_key_env and os.environ.get(args.api_key_env)),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["requests"]["successful"] else 2


def comparison_row(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    block_stats = json.loads((run_dir / "block_stats.json").read_text(encoding="utf-8"))
    return config, summary, block_stats


def compare_command(args: argparse.Namespace) -> int:
    if len(args.run_dir) < 2:
        raise ValueError("compare requires at least two run directories")
    runs = [(Path(value).expanduser().resolve(), *comparison_row(Path(value).expanduser())) for value in args.run_dir]
    contract_fields = (
        "trace_sha256",
        "start_offset",
        "max_records",
        "model",
        "served_model",
        "max_tokens",
        "concurrency",
        "request_rate",
        "block_size",
        "prefix_caching_hash_algo",
        "python_hash_seed",
        "tokenizer",
    )
    reference_config = runs[0][1]
    reference_blocks = runs[0][3]
    for run_dir, config, _summary, blocks in runs[1:]:
        for field_name in contract_fields:
            if config.get(field_name) != reference_config.get(field_name):
                raise ValueError(f"run contract mismatch for {field_name}: {run_dir}")
        if blocks.get("successful_indices") != reference_blocks.get("successful_indices"):
            raise ValueError(f"successful request set mismatch: {run_dir}")
        if blocks.get("total_full_block_references") != reference_blocks.get(
            "total_full_block_references"
        ):
            raise ValueError(f"block denominator mismatch: {run_dir}")

    rows = []
    for run_dir, config, summary, blocks in runs:
        metrics = summary["metrics"]
        counters = metrics["counter_deltas"]
        occupancy = metrics["occupancy"]
        rows.append(
            {
                "strategy": summary["strategy"],
                "run_dir": str(run_dir),
                "branch": config["git"].get("branch"),
                "commit": config["git"].get("commit"),
                "successful_requests": summary["requests"]["successful"],
                "total_full_blocks": blocks["total_full_block_references"],
                "hll_theoretical_hit_ratio": blocks["hll_theoretical_hit_ratio"],
                "prefix_aware_ideal_hit_ratio": blocks["prefix_aware_ideal_hit_ratio"],
                "actual_external_hit_ratio": summary["actual_external_hit_ratio"],
                "mean_ttft_ms": summary["ttft"]["mean_ms"],
                "p99_ttft_ms": summary["ttft"]["p99_ms"],
                "mean_e2e_ms": summary["e2e"]["mean_ms"],
                "p99_e2e_ms": summary["e2e"]["p99_ms"],
                "request_throughput": summary["successful_request_throughput"],
                "evictions": counters["pegaflow_cache_block_evictions"],
                "admission_rejections": counters[
                    "pegaflow_cache_block_admission_rejections"
                ],
                "load_failures": counters["pegaflow_load_failures"],
                "end_occupancy_bytes": sum(item["end_bytes"] for item in occupancy.values()),
                "estimated_unique_footprint_bytes": summary[
                    "estimated_unique_footprint_bytes"
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            row["actual_external_hit_ratio"] is not None,
            row["actual_external_hit_ratio"] or -1,
        ),
        reverse=True,
    )
    enough_pressure = all(row["evictions"] > 0 for row in rows)
    comparison = {
        "status": "ranked" if enough_pressure else "inconclusive_no_eviction_pressure",
        "contract": {field: reference_config.get(field) for field in contract_fields},
        "successful_indices": reference_blocks["successful_indices"],
        "rows": rows,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="replay one policy run")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--endpoint", required=True)
    replay.add_argument("--metrics-url", action="append", default=[], metavar="NAME=URL")
    replay.add_argument("--model", required=True)
    replay.add_argument("--served-model", required=True)
    replay.add_argument("--strategy", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--start-offset", type=int, default=0)
    replay.add_argument("--max-records", type=int, default=1000)
    replay.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    replay.add_argument("--max-tokens", type=int, default=1)
    replay.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    replay.add_argument("--prefix-caching-hash-algo", default=DEFAULT_HASH_ALGO)
    replay.add_argument("--hll-bucket-bits", type=int, default=DEFAULT_HLL_BUCKET_BITS)
    replay.add_argument("--timeout-s", type=float, default=600)
    replay.add_argument("--api-key-env", default="")
    replay.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    replay.set_defaults(handler=replay_command)

    compare = subparsers.add_parser("compare", help="validate and compare policy runs")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("run_dir", nargs="+")
    compare.set_defaults(handler=compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
