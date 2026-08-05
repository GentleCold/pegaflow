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
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    completion_offset_ms: float | None = None
    attempts: int = 1


@dataclass(frozen=True)
class CompletedRequest:
    result: RequestResult
    block_hashes: tuple[bytes, ...] = field(repr=False)


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


@dataclass
class CompletionLedger:
    """Successful request progress visible to the metrics sampler."""

    successful_requests: int = 0
    successful_full_blocks: int = 0

    def record(self, completed: CompletedRequest) -> None:
        if completed.result.ok:
            self.successful_requests += 1
            self.successful_full_blocks += completed.result.full_blocks

    def snapshot(self) -> dict[str, int]:
        return {
            "successful_requests": self.successful_requests,
            "successful_full_blocks": self.successful_full_blocks,
        }


@dataclass
class TimeSeriesCollector:
    """Collect periodic metrics and completed-request progress for one replay."""

    output_dir: Path
    endpoints: dict[str, str]
    baseline: dict[str, list[MetricSample]]
    replay_started: float
    interval_s: float
    ledger: CompletionLedger
    samples: list[dict[str, Any]] = field(default_factory=list)

    def add_sample(
        self,
        *,
        kind: str,
        raw: dict[str, str],
        parsed: dict[str, list[MetricSample]],
        errors: dict[str, str] | None = None,
    ) -> None:
        sample_index = len(self.samples)
        elapsed_s = max(0.0, time.perf_counter() - self.replay_started)
        progress = self.ledger.snapshot()
        sample_errors = errors or {}
        metrics: dict[str, Any] | None = None
        hit_blocks_delta: float | None = None
        ram_hit_blocks_delta: float | None = None
        rdma_hit_blocks_delta: float | None = None
        ratio: float | None = None
        ram_ratio: float | None = None
        rdma_ratio: float | None = None
        invalid_reason: str | None = None
        if parsed and set(parsed) == set(self.baseline):
            metrics = summarize_metrics(self.baseline, parsed)
            tier_deltas = metrics["tier_block_request_deltas"]
            ram_hit_blocks_delta = tier_deltas.get("ram", 0.0)
            rdma_hit_blocks_delta = tier_deltas.get("rdma", 0.0)
            hit_blocks_delta = ram_hit_blocks_delta + rdma_hit_blocks_delta
            denominator = progress["successful_full_blocks"]
            if ram_hit_blocks_delta < 0 or rdma_hit_blocks_delta < 0:
                invalid_reason = "hit_counter_reset"
            elif denominator <= 0:
                invalid_reason = "no_completed_successful_full_blocks"
            elif hit_blocks_delta > denominator:
                invalid_reason = "hits_exceed_completed_full_blocks"
            else:
                ratio = hit_blocks_delta / denominator
                ram_ratio = ram_hit_blocks_delta / denominator
                rdma_ratio = rdma_hit_blocks_delta / denominator
        elif parsed:
            invalid_reason = "incomplete_metrics_endpoints"
        elif not sample_errors:
            invalid_reason = "metrics_empty"

        sample = {
            "sample_index": sample_index,
            "sample_kind": kind,
            "elapsed_sec": elapsed_s,
            "sample_wall_time": datetime.now(timezone.utc).isoformat(),
            "interval_s": self.interval_s,
            **progress,
            "hit_blocks_delta": hit_blocks_delta,
            "cumulative_actual_hit_ratio": ratio,
            "ram_hit_blocks_delta": ram_hit_blocks_delta,
            "cumulative_ram_hit_ratio": ram_ratio,
            "rdma_hit_blocks_delta": rdma_hit_blocks_delta,
            "cumulative_rdma_hit_ratio": rdma_ratio,
            "ratio_invalid_reason": invalid_reason,
            "metrics_errors": sample_errors,
            "metrics": metrics,
        }
        self.samples.append(sample)

        if raw:
            sample_dir = self.output_dir / "metrics_samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_raw_metrics(sample_dir / f"{sample_index:04d}_{kind}.prom", raw, self.endpoints)

    async def run(self) -> None:
        if self.interval_s <= 0:
            return
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                raw, parsed, errors = await asyncio.to_thread(fetch_metrics_partial, self.endpoints)
                self.add_sample(kind="interval", raw=raw, parsed=parsed, errors=errors)
        except asyncio.CancelledError:
            raise


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
    if not isinstance(tokenized, list) or any(not isinstance(item, int) for item in tokenized):
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


def exceeds_model_len(
    prepared: PreparedRequest,
    max_tokens: int,
    max_model_len: int,
) -> bool:
    return bool(max_model_len and prepared.prompt_tokens + max_tokens > max_model_len)


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
) -> tuple[int | None, int | None, float | None, str | None, bool]:
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
        return (
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            f"stream_{type(exc).__name__}",
            isinstance(exc, httpx.ReadError),
        )
    if not saw_done:
        return prompt_tokens, completion_tokens, ttft_ms, "stream_incomplete", False
    return prompt_tokens, completion_tokens, ttft_ms, None, False


async def _send_request_once(
    client: httpx.AsyncClient,
    endpoint: str,
    prepared: PreparedRequest,
    request_id: str,
    started: float,
) -> tuple[
    int | None,
    int | None,
    int | None,
    float | None,
    str | None,
    bool,
]:
    status_code = None
    prompt_tokens = None
    completion_tokens = None
    ttft_ms = None
    failure_kind = None
    retryable = False
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
                retryable = 500 <= response.status_code < 600
            elif prepared.payload["stream"]:
                (
                    prompt_tokens,
                    completion_tokens,
                    ttft_ms,
                    failure_kind,
                    retryable,
                ) = await _consume_sse(response, started)
            else:
                raw = await response.aread()
                parsed = json.loads(raw)
                prompt_tokens, completion_tokens = _usage(parsed)
                ttft_ms = (time.perf_counter() - started) * 1000
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        failure_kind = f"request_{type(exc).__name__}"
        retryable = isinstance(exc, httpx.ReadError)
    return (
        status_code,
        prompt_tokens,
        completion_tokens,
        ttft_ms,
        failure_kind,
        retryable,
    )


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    prepared: PreparedRequest,
    replay_started: float,
    max_request_retries: int = 0,
) -> CompletedRequest:
    if max_request_retries < 0:
        raise ValueError("max_request_retries must be non-negative")
    started = time.perf_counter()
    base_request_id = f"trace-replay-{prepared.index}"
    attempts = 0
    while True:
        attempts += 1
        request_id = base_request_id if attempts == 1 else f"{base_request_id}-retry-{attempts - 1}"
        (
            status_code,
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            failure_kind,
            retryable,
        ) = await _send_request_once(
            client,
            endpoint,
            prepared,
            request_id,
            started,
        )
        if failure_kind is None or not retryable or attempts > max_request_retries:
            break
        await asyncio.sleep(min(0.1 * attempts, 0.5))

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
        completion_offset_ms=(time.perf_counter() - replay_started) * 1000,
        attempts=attempts,
    )
    return CompletedRequest(result, prepared.block_hashes if ok else ())


async def replay_prepared(
    prepared_requests: Iterable[PreparedRequest],
    concurrency: int,
    sender: Callable[[PreparedRequest], Any],
    on_complete: Callable[[CompletedRequest], None] | None = None,
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
                result = await sender(item)
                completed.append(result)
                if on_complete is not None:
                    on_complete(result)
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
    max_request_retries: int = 0,
    replay_started: float | None = None,
    on_complete: Callable[[CompletedRequest], None] | None = None,
) -> tuple[list[CompletedRequest], float]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(timeout_s if timeout_s > 0 else None)
    replay_started = replay_started if replay_started is not None else time.perf_counter()
    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:

        async def sender(prepared: PreparedRequest) -> CompletedRequest:
            return await send_request(
                client,
                endpoint,
                prepared,
                replay_started,
                max_request_retries=max_request_retries,
            )

        completed = await replay_prepared(
            prepared_requests,
            concurrency,
            sender,
            on_complete=on_complete,
        )
    return completed, time.perf_counter() - replay_started


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([\d.eE+-]+)(?:\s+\d+)?$")
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


def fetch_metrics(
    endpoints: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[MetricSample]]]:
    raw = {}
    parsed = {}
    for name, url in endpoints.items():
        request = Request(url, headers={"Accept": "text/plain"})
        with urlopen(request, timeout=10) as response:
            content = response.read().decode("utf-8", errors="replace")
        raw[name] = content
        parsed[name] = parse_prometheus(content)
    return raw, parsed


def fetch_metrics_partial(
    endpoints: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[MetricSample]], dict[str, str]]:
    """Fetch each endpoint independently for best-effort timeline samples."""

    raw: dict[str, str] = {}
    parsed: dict[str, list[MetricSample]] = {}
    errors: dict[str, str] = {}
    for name, url in endpoints.items():
        try:
            request = Request(url, headers={"Accept": "text/plain"})
            with urlopen(request, timeout=10) as response:
                content = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - preserve endpoint-local failures in samples
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        raw[name] = content
        parsed[name] = parse_prometheus(content)
    return raw, parsed, errors


def write_raw_metrics(path: Path, raw: dict[str, str], endpoints: dict[str, str]) -> None:
    chunks = []
    for name in sorted(raw):
        chunks.append(f"# PEGAFLOW_REPLAY_ENDPOINT name={name} url={endpoints[name]}\n")
        chunks.append(raw[name].rstrip() + "\n")
    path.write_text("".join(chunks), encoding="utf-8")


TIMESERIES_FIELDS = (
    "sample_index",
    "sample_kind",
    "elapsed_sec",
    "sample_wall_time",
    "interval_s",
    "successful_requests",
    "successful_full_blocks",
    "hit_blocks_delta",
    "cumulative_actual_hit_ratio",
    "ram_hit_blocks_delta",
    "cumulative_ram_hit_ratio",
    "rdma_hit_blocks_delta",
    "cumulative_rdma_hit_ratio",
    "ratio_invalid_reason",
    "evictions_delta",
    "admission_rejections_delta",
    "load_failures_delta",
    "ram_block_requests_delta",
    "rdma_block_requests_delta",
    "ssd_block_requests_delta",
    "miss_block_requests_delta",
    "end_resident_bytes",
    "metrics_errors",
)


def _timeseries_row(sample: dict[str, Any]) -> dict[str, Any]:
    metrics = sample.get("metrics") or {}
    counters = metrics.get("counter_deltas") or {}
    tiers = metrics.get("tier_block_request_deltas") or {}
    occupancy = metrics.get("occupancy") or {}
    return {
        "sample_index": sample["sample_index"],
        "sample_kind": sample["sample_kind"],
        "elapsed_sec": sample["elapsed_sec"],
        "sample_wall_time": sample["sample_wall_time"],
        "interval_s": sample["interval_s"],
        "successful_requests": sample["successful_requests"],
        "successful_full_blocks": sample["successful_full_blocks"],
        "hit_blocks_delta": sample["hit_blocks_delta"],
        "cumulative_actual_hit_ratio": sample["cumulative_actual_hit_ratio"],
        "ram_hit_blocks_delta": sample["ram_hit_blocks_delta"],
        "cumulative_ram_hit_ratio": sample["cumulative_ram_hit_ratio"],
        "rdma_hit_blocks_delta": sample["rdma_hit_blocks_delta"],
        "cumulative_rdma_hit_ratio": sample["cumulative_rdma_hit_ratio"],
        "ratio_invalid_reason": sample["ratio_invalid_reason"],
        "evictions_delta": counters.get("pegaflow_cache_block_evictions"),
        "admission_rejections_delta": counters.get("pegaflow_cache_block_admission_rejections"),
        "load_failures_delta": counters.get("pegaflow_load_failures"),
        "ram_block_requests_delta": tiers.get("ram"),
        "rdma_block_requests_delta": tiers.get("rdma"),
        "ssd_block_requests_delta": tiers.get("ssd"),
        "miss_block_requests_delta": tiers.get("miss"),
        "end_resident_bytes": sum(item.get("end_bytes", 0) for item in occupancy.values()),
        "metrics_errors": json.dumps(sample.get("metrics_errors") or {}, sort_keys=True),
    }


def write_timeseries(
    output_dir: Path,
    samples: Sequence[dict[str, Any]],
    block_stats: dict[str, Any],
) -> None:
    """Persist non-sensitive timeline samples and theoretical references."""

    timeline_path = output_dir / "metrics_timeseries.jsonl"
    with timeline_path.open("w", encoding="utf-8") as destination:
        for sample in samples:
            payload = {
                **sample,
                "hll_theoretical_hit_ratio": block_stats["hll_theoretical_hit_ratio"],
                "prefix_aware_ideal_hit_ratio": block_stats["prefix_aware_ideal_hit_ratio"],
            }
            destination.write(json.dumps(payload, sort_keys=True) + "\n")

    csv_path = output_dir / "hit_rate_timeseries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        fields = [
            *TIMESERIES_FIELDS,
            "hll_theoretical_hit_ratio",
            "prefix_aware_ideal_hit_ratio",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = _timeseries_row(sample)
            row["hll_theoretical_hit_ratio"] = block_stats["hll_theoretical_hit_ratio"]
            row["prefix_aware_ideal_hit_ratio"] = block_stats["prefix_aware_ideal_hit_ratio"]
            writer.writerow(row)


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
            sum(resident_block_bytes) / len(resident_block_bytes) if resident_block_bytes else None
        ),
    }


def theoretical_stats(
    completed: Sequence[CompletedRequest],
    bucket_bits: int,
) -> dict[str, Any]:
    successful = sorted(
        (item for item in completed if item.result.ok), key=lambda item: item.result.index
    )
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


def build_workload_manifest(
    *,
    selected_indices: Sequence[int],
    prepared_requests: Sequence[PreparedRequest],
    preparation_failures: Sequence[RequestResult],
    filtered_requests: Sequence[dict[str, Any]],
    max_model_len: int,
    max_tokens: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "selected_indices": list(selected_indices),
        "eligible_indices": [request.index for request in prepared_requests],
        "preparation_failure_indices": [result.index for result in preparation_failures],
        "filtered_requests": list(filtered_requests),
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {**manifest, "sha256": hashlib.sha256(encoded).hexdigest()}


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


async def run_replay_with_metrics(
    prepared: Sequence[PreparedRequest],
    *,
    endpoint: str,
    concurrency: int,
    timeout_s: float,
    api_key: str | None,
    max_request_retries: int,
    metrics_endpoints: dict[str, str],
    output_dir: Path,
    metrics_interval_s: float,
) -> tuple[
    list[CompletedRequest],
    float,
    dict[str, str],
    dict[str, list[MetricSample]],
    dict[str, str],
    dict[str, list[MetricSample]],
    TimeSeriesCollector,
]:
    raw_start, parsed_start = await asyncio.to_thread(fetch_metrics, metrics_endpoints)
    write_raw_metrics(output_dir / "metrics_start.prom", raw_start, metrics_endpoints)

    replay_started = time.perf_counter()
    ledger = CompletionLedger()
    collector = TimeSeriesCollector(
        output_dir=output_dir,
        endpoints=metrics_endpoints,
        baseline=parsed_start,
        replay_started=replay_started,
        interval_s=metrics_interval_s,
        ledger=ledger,
    )
    collector.add_sample(kind="baseline", raw=raw_start, parsed=parsed_start)
    sampler_task = asyncio.create_task(collector.run())
    try:
        completed, replay_duration_s = await run_replay(
            prepared,
            endpoint=endpoint,
            concurrency=concurrency,
            timeout_s=timeout_s,
            api_key=api_key,
            max_request_retries=max_request_retries,
            replay_started=replay_started,
            on_complete=ledger.record,
        )
    finally:
        sampler_task.cancel()
        with suppress(asyncio.CancelledError):
            await sampler_task

    raw_end, parsed_end = await asyncio.to_thread(fetch_metrics, metrics_endpoints)
    write_raw_metrics(output_dir / "metrics_end.prom", raw_end, metrics_endpoints)
    collector.add_sample(kind="final", raw=raw_end, parsed=parsed_end)
    return (
        completed,
        replay_duration_s,
        raw_start,
        parsed_start,
        raw_end,
        parsed_end,
        collector,
    )


def build_summary(
    strategy: str,
    results: Sequence[CompletedRequest],
    preparation_failures: Sequence[RequestResult],
    replay_duration_s: float,
    block_stats: dict[str, Any],
    metrics: dict[str, Any],
    *,
    selected_count: int | None = None,
    filtered_count: int = 0,
    eligible_count: int | None = None,
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
    tier_hits = metrics["tier_block_request_deltas"]
    ram_hits = tier_hits.get("ram", 0.0)
    rdma_hits = tier_hits.get("rdma", 0.0)
    actual_hits = ram_hits + rdma_hits
    external_counter_hits = metrics["counter_deltas"]["pegaflow_cache_block_hits"]
    mean_block_bytes = metrics["mean_resident_block_bytes"]
    exact_distinct = block_stats["exact_distinct_full_blocks"]
    selected_count = len(all_results) if selected_count is None else selected_count
    eligible_count = len(all_results) if eligible_count is None else eligible_count
    retried = [result for result in all_results if result.attempts > 1]
    return {
        "strategy": strategy,
        "requests": {
            "selected": selected_count,
            "filtered": filtered_count,
            "eligible": eligible_count,
            "successful": len(successful),
            "failed": eligible_count - len(successful),
            "retried": len(retried),
            "retry_attempts": sum(result.attempts - 1 for result in retried),
            "failure_kinds": failures,
        },
        "block_stats": block_stats,
        "actual_external_hit_blocks": actual_hits,
        "actual_external_hit_ratio": actual_hits / denominator if denominator else None,
        "external_hit_counter_delta": external_counter_hits,
        "actual_ram_hit_blocks": ram_hits,
        "actual_ram_hit_ratio": ram_hits / denominator if denominator else None,
        "actual_rdma_hit_blocks": rdma_hits,
        "actual_rdma_hit_ratio": rdma_hits / denominator if denominator else None,
        "actual_ratio_valid": bool(
            denominator
            and 0 <= ram_hits <= denominator
            and 0 <= rdma_hits <= denominator
            and actual_hits <= denominator
            and actual_hits == external_counter_hits
        ),
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
    if args.metrics_interval_s < 0:
        raise ValueError("metrics interval must be non-negative")
    if args.max_model_len < 0:
        raise ValueError("max model len must be non-negative")
    if args.max_request_retries < 0:
        raise ValueError("max request retries must be non-negative")
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
    filtered_requests = []
    selected_indices = []
    for entry in iter_trace_entries(trace_path, args.start_offset, args.max_records):
        selected_indices.append(entry.index)
        outcome = prepare_entry(entry, args.served_model, args.max_tokens, block_hasher)
        if isinstance(outcome, PreparedRequest):
            total_tokens = outcome.prompt_tokens + args.max_tokens
            if exceeds_model_len(outcome, args.max_tokens, args.max_model_len):
                filtered_requests.append(
                    {
                        "index": outcome.index,
                        "prompt_tokens": outcome.prompt_tokens,
                        "total_tokens": total_tokens,
                        "reason": "max_model_len_exceeded",
                    }
                )
            else:
                prepared.append(outcome)
        else:
            preparation_failures.append(outcome)

    workload_manifest = build_workload_manifest(
        selected_indices=selected_indices,
        prepared_requests=prepared,
        preparation_failures=preparation_failures,
        filtered_requests=filtered_requests,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )
    (output_dir / "workload_manifest.json").write_text(
        json.dumps(workload_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (
        completed,
        replay_duration_s,
        raw_start,
        parsed_start,
        raw_end,
        parsed_end,
        collector,
    ) = asyncio.run(
        run_replay_with_metrics(
            prepared,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
            max_request_retries=args.max_request_retries,
            metrics_endpoints=metrics_endpoints,
            output_dir=output_dir,
            metrics_interval_s=args.metrics_interval_s,
        )
    )

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
    write_timeseries(output_dir, collector.samples, block_stats)
    summary = build_summary(
        args.strategy,
        completed,
        preparation_failures,
        replay_duration_s,
        block_stats,
        metrics,
        selected_count=len(selected_indices),
        filtered_count=len(filtered_requests),
        eligible_count=len(prepared) + len(preparation_failures),
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
        "max_model_len": args.max_model_len,
        "max_request_retries": args.max_request_retries,
        "workload_manifest_sha256": workload_manifest["sha256"],
        "concurrency": args.concurrency,
        "request_rate": "inf",
        "metrics_interval_s": args.metrics_interval_s,
        "timeseries_schema_version": 1,
        "timeseries_time_base": "monotonic_elapsed_sec",
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
    runs = [
        (Path(value).expanduser().resolve(), *comparison_row(Path(value).expanduser()))
        for value in args.run_dir
    ]
    contract_fields = (
        "trace_sha256",
        "start_offset",
        "max_records",
        "model",
        "served_model",
        "max_tokens",
        "max_model_len",
        "max_request_retries",
        "workload_manifest_sha256",
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

    for run_dir, _config, summary, _blocks in runs:
        requests = summary["requests"]
        if requests.get("successful") != requests.get("eligible"):
            raise ValueError(f"eligible request did not complete successfully: {run_dir}")
        if not summary.get("actual_ratio_valid"):
            raise ValueError(f"actual hit ratio is invalid: {run_dir}")
        load_failures = summary["metrics"]["counter_deltas"].get("pegaflow_load_failures", 0)
        if load_failures:
            raise ValueError(f"PegaFlow load failures are non-zero: {run_dir}")

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
                "actual_ram_hit_blocks": summary["actual_ram_hit_blocks"],
                "actual_ram_hit_ratio": summary["actual_ram_hit_ratio"],
                "actual_rdma_hit_blocks": summary["actual_rdma_hit_blocks"],
                "actual_rdma_hit_ratio": summary["actual_rdma_hit_ratio"],
                "actual_external_hit_blocks": summary["actual_external_hit_blocks"],
                "actual_external_hit_ratio": summary["actual_external_hit_ratio"],
                "mean_ttft_ms": summary["ttft"]["mean_ms"],
                "p99_ttft_ms": summary["ttft"]["p99_ms"],
                "mean_e2e_ms": summary["e2e"]["mean_ms"],
                "p99_e2e_ms": summary["e2e"]["p99_ms"],
                "request_throughput": summary["successful_request_throughput"],
                "evictions": counters["pegaflow_cache_block_evictions"],
                "admission_rejections": counters["pegaflow_cache_block_admission_rejections"],
                "load_failures": counters["pegaflow_load_failures"],
                "end_occupancy_bytes": sum(item["end_bytes"] for item in occupancy.values()),
                "estimated_unique_footprint_bytes": summary["estimated_unique_footprint_bytes"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["actual_ram_hit_ratio"] is not None,
            row["actual_ram_hit_ratio"] or -1,
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
    replay.add_argument(
        "--metrics-interval-s",
        type=float,
        default=0.0,
        help="periodic metrics sample interval; 0 disables interval samples",
    )
    replay.add_argument("--max-tokens", type=int, default=1)
    replay.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="filter requests whose prompt plus output tokens exceed this limit; 0 disables",
    )
    replay.add_argument(
        "--max-request-retries",
        type=int,
        default=0,
        help="retry transient HTTP 5xx and connection read failures in place",
    )
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
