#!/usr/bin/env python3
"""Two-node RDMA-only integration benchmark for the P/D transfer path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RANK_MAP = {
    "0": {"cuda_device": 0, "nic": "mlx5_1", "worker_cpu": 16},
    "1": {"cuda_device": 1, "nic": "mlx5_1", "worker_cpu": 30},
    "2": {"cuda_device": 2, "nic": "mlx5_2", "worker_cpu": 60},
    "3": {"cuda_device": 3, "nic": "mlx5_2", "worker_cpu": 90},
    "4": {"cuda_device": 4, "nic": "mlx5_3", "worker_cpu": 120},
    "5": {"cuda_device": 5, "nic": "mlx5_3", "worker_cpu": 150},
    "6": {"cuda_device": 6, "nic": "mlx5_4", "worker_cpu": 180},
    "7": {"cuda_device": 7, "nic": "mlx5_4", "worker_cpu": 210},
}


@dataclass(frozen=True)
class TransferRegionLayout:
    region_idx: int
    base_addr: int
    block_len: int


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    block_ids: tuple[int, ...]
    regions: tuple[TransferRegionLayout, ...]
    mr_desc: object | None = None


@dataclass(frozen=True)
class RankContext:
    rank: int
    cuda_device: int
    nic: str
    worker_cpu: int
    engine: Any
    buffer: Any
    layers: tuple[LayerRemoteLayout, ...]


def load_native() -> Any:
    repo = Path(__file__).resolve().parents[1]
    native = repo / "target" / "release" / "libpegaflow.so"
    if not native.exists():
        native = repo / "target" / "debug" / "libpegaflow.so"
    if native.exists():
        spec = importlib.util.spec_from_file_location("pegaflow.pegaflow", native)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["pegaflow.pegaflow"] = module
        spec.loader.exec_module(module)
        return module
    from pegaflow import pegaflow

    return pegaflow


def parse_rank_map(value: str | None) -> dict[str, dict[str, int | str]]:
    if value is None:
        return DEFAULT_RANK_MAP
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("rank map must be a JSON object")
    out: dict[str, dict[str, int | str]] = {}
    for rank, config in loaded.items():
        if not isinstance(config, dict):
            raise ValueError(f"rank_map[{rank}] must be an object")
        rank_key = str(rank)
        cuda_device = int(config.get("cuda_device", rank_key))
        out[rank_key] = {
            "cuda_device": cuda_device,
            "nic": str(config["nic"]),
            "worker_cpu": int(config["worker_cpu"]),
        }
    return out


def selected_ranks(rank_map: dict[str, dict[str, int | str]], ranks: int) -> list[int]:
    out = sorted(int(rank) for rank in rank_map)
    if ranks > len(out):
        raise ValueError(
            f"requested ranks={ranks}, but rank_map only has {len(out)} entries"
        )
    return out[:ranks]


def build_layers(
    base_addr: int,
    *,
    layer_count: int,
    block_count: int,
    block_bytes: int,
) -> tuple[LayerRemoteLayout, ...]:
    layer_bytes = block_count * block_bytes
    block_ids = tuple(range(block_count))
    return tuple(
        LayerRemoteLayout(
            layer_name=f"layer.{layer_idx}",
            layer_idx=layer_idx,
            block_ids=block_ids,
            regions=(
                TransferRegionLayout(
                    region_idx=0,
                    base_addr=base_addr + layer_idx * layer_bytes,
                    block_len=block_bytes,
                ),
            ),
        )
        for layer_idx in range(layer_count)
    )


def layer_to_native(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "block_ids": list(layer.block_ids),
        "regions": [
            {
                "region_idx": region.region_idx,
                "base_addr": region.base_addr,
                "block_len": region.block_len,
            }
            for region in layer.regions
        ],
        "mr_desc": layer.mr_desc,
    }


def layer_from_native(layer: dict[str, Any]) -> LayerRemoteLayout:
    return LayerRemoteLayout(
        layer_name=str(layer["layer_name"]),
        layer_idx=int(layer["layer_idx"]),
        block_ids=tuple(int(block_id) for block_id in layer["block_ids"]),
        regions=tuple(
            TransferRegionLayout(
                region_idx=int(region["region_idx"]),
                base_addr=int(region["base_addr"]),
                block_len=int(region["block_len"]),
            )
            for region in layer["regions"]
        ),
        mr_desc=layer.get("mr_desc"),
    )


def register_local_layers(
    engine: Any, layers: tuple[LayerRemoteLayout, ...]
) -> tuple[LayerRemoteLayout, ...]:
    registered = engine.register_local_layers(
        [layer_to_native(layer) for layer in layers]
    )
    return tuple(layer_from_native(layer) for layer in registered)


def build_coalesced_layer_blocks(
    block_count: int, block_bytes: int
) -> list[dict[str, Any]]:
    return [
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 0,
                    "src_offset_bytes": 0,
                    "bytes": block_count * block_bytes,
                }
            ]
        }
    ]


def create_rank_contexts(
    native: Any,
    rank_map: dict[str, dict[str, int | str]],
    ranks: list[int],
    *,
    layer_count: int,
    block_count: int,
    block_bytes: int,
    device: str,
    fill_byte: int,
) -> list[RankContext]:
    contexts: list[RankContext] = []
    size = layer_count * block_count * block_bytes
    for rank in ranks:
        config = rank_map[str(rank)]
        cuda_device = int(config["cuda_device"])
        nic = str(config["nic"])
        worker_cpu = int(config["worker_cpu"])
        stage(
            "create rank engine "
            f"rank={rank} cuda={cuda_device} nic={nic} worker_cpu={worker_cpu}"
        )
        engine = native.PdRdmaEngine(
            cuda_device=cuda_device,
            domains=[nic],
            device=device,
            pin_worker_cpu=worker_cpu,
        )
        if device != "cuda":
            raise ValueError("pd_rdma_two_node_it currently requires CUDA test buffers")
        buffer = native.PdRdmaTestBuffer(size=size, cuda_device=cuda_device)
        buffer.fill(fill_byte)
        layers = register_local_layers(
            engine,
            build_layers(
                buffer.ptr(),
                layer_count=layer_count,
                block_count=block_count,
                block_bytes=block_bytes,
            ),
        )
        contexts.append(
            RankContext(
                rank=rank,
                cuda_device=cuda_device,
                nic=nic,
                worker_cpu=worker_cpu,
                engine=engine,
                buffer=buffer,
                layers=layers,
            )
        )
    return contexts


def handshake(
    *,
    request_id: str,
    engine_id: str,
    rank: int,
    ranks: int,
    block_size: int,
    layers: tuple[LayerRemoteLayout, ...],
    imm_id: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "engine_id": engine_id,
        "tp_rank": rank,
        "tp_size": ranks,
        "block_size": block_size,
        "layers": [layer_to_native(layer) for layer in layers],
        "imm_id": imm_id,
    }


def register_remote(engine: Any, req_id: str, remote_handshake: dict[str, Any]) -> None:
    native_handshake = {
        **remote_handshake,
        "layers": [layer_dict_to_native(layer) for layer in remote_handshake["layers"]],
    }
    engine.register_remote(req_id, native_handshake)


def layer_dict_to_native(layer: dict[str, Any]) -> dict[str, Any]:
    return {
        **layer,
        "mr_desc": mr_desc_to_native(layer.get("mr_desc")),
    }


def mr_desc_to_native(mr_desc: Any | None) -> Any | None:
    if not isinstance(mr_desc, dict):
        return mr_desc
    addr_rkey_list = mr_desc.get("addr_rkey_list")
    if addr_rkey_list is None:
        return mr_desc
    return {
        **mr_desc,
        "addr_rkey_list": [
            (str(addr_rkey[0]), int(addr_rkey[1])) for addr_rkey in addr_rkey_list
        ],
    }


def send_json(sock_file: Any, payload: dict[str, Any]) -> None:
    sock_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sock_file.flush()


def recv_json(sock_file: Any) -> dict[str, Any]:
    line = sock_file.readline()
    if not line:
        raise EOFError("control socket closed")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("control payload must be a JSON object")
    return payload


def nic_counters(nics: set[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for nic in sorted(nics):
        counters = Path("/sys/class/infiniband") / nic / "ports" / "1" / "counters"
        out[nic] = {
            "xmit": int((counters / "port_xmit_data").read_text().strip()),
            "rcv": int((counters / "port_rcv_data").read_text().strip()),
        }
    return out


def nic_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
    elapsed_s: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for nic in sorted(before):
        xmit_bytes = (after[nic]["xmit"] - before[nic]["xmit"]) * 4
        rcv_bytes = (after[nic]["rcv"] - before[nic]["rcv"]) * 4
        out[nic] = {
            "xmit_GB": xmit_bytes / 1e9,
            "rcv_GB": rcv_bytes / 1e9,
            "xmit_gbps": xmit_bytes * 8 / elapsed_s / 1e9 if elapsed_s > 0 else 0.0,
            "rcv_gbps": rcv_bytes * 8 / elapsed_s / 1e9 if elapsed_s > 0 else 0.0,
        }
    return out


def expected_nic_bytes(
    contexts: list[RankContext],
    *,
    iterations: int,
    block_count: int,
    block_bytes: int,
) -> dict[str, int]:
    bytes_per_rank = iterations * len(contexts[0].layers) * block_count * block_bytes
    out = {ctx.nic: 0 for ctx in contexts}
    for ctx in contexts:
        out[ctx.nic] += bytes_per_rank
    return out


def buffer_sample_ranges(size: int, sample_bytes: int) -> tuple[tuple[int, int], ...]:
    if sample_bytes <= 0:
        return ()
    length = min(size, sample_bytes)
    return tuple(
        (offset, length)
        for offset in sorted({0, max(0, (size - length) // 2), size - length})
    )


def sample_pattern(rank: int, offset: int, length: int, seed: int) -> bytes:
    rng_seed = ((seed & 0xFF) << 56) ^ ((rank & 0xFFFF) << 40) ^ offset
    return random.Random(rng_seed).randbytes(length)


def write_source_patterns(
    contexts: list[RankContext], *, sample_bytes: int, seed: int
) -> dict[str, list[tuple[int, int]]]:
    written_ranges: dict[str, list[tuple[int, int]]] = {}
    for ctx in contexts:
        ranges = buffer_sample_ranges(ctx.buffer.size(), sample_bytes)
        written_ranges[str(ctx.rank)] = list(ranges)
        for offset, length in ranges:
            ctx.buffer.write_bytes(
                offset, sample_pattern(ctx.rank, offset, length, seed)
            )
    return written_ranges


def reset_decode_sample_ranges(
    contexts: list[RankContext], *, sample_bytes: int, value: int
) -> None:
    if sample_bytes <= 0:
        return
    for ctx in contexts:
        for offset, length in buffer_sample_ranges(ctx.buffer.size(), sample_bytes):
            ctx.buffer.write_bytes(offset, bytes([value]) * length)


def verify_decode_buffers(
    contexts: list[RankContext], *, seed: int, sample_bytes: int
) -> dict[str, Any]:
    started = time.perf_counter()
    sampled_bytes = 0
    checked_ranges: dict[str, list[tuple[int, int]]] = {}
    for ctx in contexts:
        ranges = buffer_sample_ranges(ctx.buffer.size(), sample_bytes)
        checked_ranges[str(ctx.rank)] = list(ranges)
        for offset, length in ranges:
            data = ctx.buffer.to_bytes_range(offset, length)
            wanted = sample_pattern(ctx.rank, offset, length, seed)
            if data != wanted:
                mismatch = next(
                    idx
                    for idx, (actual, want) in enumerate(zip(data, wanted, strict=True))
                    if actual != want
                )
                raise AssertionError(
                    "RDMA payload mismatch "
                    f"rank={ctx.rank} cuda={ctx.cuda_device} offset={offset + mismatch} "
                    f"expected_seed=0x{seed:02x} actual=0x{data[mismatch]:02x}"
                )
            sampled_bytes += length
    return {
        "enabled": sample_bytes > 0,
        "sample_bytes_per_range": sample_bytes,
        "sampled_bytes": sampled_bytes,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "ranges": checked_ranges,
    }


def summarize_verify_results(
    verify_results: list[dict[str, Any]], *, sample_bytes: int
) -> dict[str, Any]:
    return {
        "enabled": sample_bytes > 0,
        "sample_bytes_per_range": sample_bytes,
        "sampled_bytes": sum(int(result["sampled_bytes"]) for result in verify_results),
        "elapsed_ms": sum(float(result["elapsed_ms"]) for result in verify_results),
        "iterations": verify_results,
    }


def result_check_errors(
    result: dict[str, Any],
    *,
    min_bandwidth_gbps: float,
    min_nic_gbps: float,
    min_nic_byte_ratio: float,
) -> list[str]:
    errors: list[str] = []
    bandwidth = float(result["bandwidth_gbps"])
    if min_bandwidth_gbps > 0 and bandwidth < min_bandwidth_gbps:
        errors.append(
            f"aggregate bandwidth {bandwidth:.2f}Gbps is below {min_bandwidth_gbps:.2f}Gbps"
        )
    if min_nic_gbps > 0:
        direction = "xmit_gbps" if result["role"] == "prefill" else "rcv_gbps"
        for nic, delta in sorted(result["nic_delta"].items()):
            nic_gbps = float(delta[direction])
            if nic_gbps < min_nic_gbps:
                errors.append(
                    f"{nic} {direction} {nic_gbps:.2f}Gbps is below {min_nic_gbps:.2f}Gbps"
                )
    if min_nic_byte_ratio > 0:
        byte_direction = "xmit_GB" if result["role"] == "prefill" else "rcv_GB"
        expected_by_nic = result["expected_nic_bytes"]
        for nic, expected_bytes in sorted(expected_by_nic.items()):
            actual_bytes = float(result["nic_delta"][nic][byte_direction]) * 1e9
            min_bytes = int(expected_bytes) * min_nic_byte_ratio
            if actual_bytes < min_bytes:
                errors.append(
                    f"{nic} {byte_direction} {actual_bytes / 1e9:.2f}GB is below "
                    f"{min_nic_byte_ratio:.2f}x expected {int(expected_bytes) / 1e9:.2f}GB"
                )
    return errors


def apply_result_checks(
    result: dict[str, Any],
    *,
    min_bandwidth_gbps: float,
    min_nic_gbps: float,
    min_nic_byte_ratio: float,
    extra_errors: list[str] | None = None,
) -> list[str]:
    errors = result_check_errors(
        result,
        min_bandwidth_gbps=min_bandwidth_gbps,
        min_nic_gbps=min_nic_gbps,
        min_nic_byte_ratio=min_nic_byte_ratio,
    )
    if extra_errors:
        errors.extend(extra_errors)
    if errors:
        result["ok"] = False
        result["error"] = "; ".join(errors)
    return errors


def finish_result(
    result: dict[str, Any],
    *,
    json_out: str | None,
    min_bandwidth_gbps: float,
    min_nic_gbps: float,
    min_nic_byte_ratio: float,
    extra_errors: list[str] | None = None,
) -> None:
    errors = apply_result_checks(
        result,
        min_bandwidth_gbps=min_bandwidth_gbps,
        min_nic_gbps=min_nic_gbps,
        min_nic_byte_ratio=min_nic_byte_ratio,
        extra_errors=extra_errors,
    )
    print_result(result, json_out)
    if errors:
        raise RuntimeError(result["error"])


def run_rank_push(
    ctx: RankContext,
    *,
    requests: list[dict[str, Any]],
    block_count: int,
    block_bytes: int,
) -> None:
    blocks = build_coalesced_layer_blocks(block_count, block_bytes)
    for request in requests:
        prefill_req = request["prefill_req"]
        for layer in ctx.layers:
            ctx.engine.push_layer(prefill_req, layer.layer_idx, blocks)
        ctx.engine.wait_for_pushes(prefill_req)
        ctx.engine.push_done(prefill_req)


def run_prefill(args: argparse.Namespace) -> None:
    native = load_native()
    rank_map = parse_rank_map(args.rank_map)
    ranks = selected_ranks(rank_map, args.ranks)
    contexts = create_rank_contexts(
        native,
        rank_map,
        ranks,
        layer_count=args.layers,
        block_count=args.blocks,
        block_bytes=args.block_bytes,
        device=args.device,
        fill_byte=args.src_byte,
    )
    source_patterns = write_source_patterns(
        contexts, sample_bytes=args.verify_sample_bytes, seed=args.src_byte
    )
    rank_by_id = {ctx.rank: ctx for ctx in contexts}
    nics = {ctx.nic for ctx in contexts}

    with socket.create_connection(
        (args.decode_host, args.port), timeout=args.connect_timeout_s
    ) as sock:
        sock_file = sock.makefile("rw", encoding="utf-8", newline="\n")
        send_json(
            sock_file,
            {
                "event": "prefill_ready",
                "ranks": ranks,
                "layers": args.layers,
                "blocks": args.blocks,
                "block_bytes": args.block_bytes,
                "iterations": args.iterations,
                "src_byte": args.src_byte,
            },
        )
        setup = recv_json(sock_file)
        assert setup["event"] == "setup"
        requests_by_iteration_by_rank: dict[int, dict[int, dict[str, Any]]] = {
            iteration: {} for iteration in range(args.iterations)
        }
        for request in setup["requests"]:
            rank = int(request["rank"])
            iteration = int(request["iteration"])
            if iteration not in requests_by_iteration_by_rank:
                raise ValueError(f"unexpected setup iteration {iteration}")
            ctx = rank_by_id[rank]
            register_remote(ctx.engine, request["prefill_req"], request["handshake"])
            requests_by_iteration_by_rank[iteration][rank] = request
        for iteration, requests_by_rank in requests_by_iteration_by_rank.items():
            missing_ranks = sorted(set(ranks) - set(requests_by_rank))
            if missing_ranks:
                raise ValueError(
                    f"setup missing ranks for iteration {iteration}: {missing_ranks}"
                )
        send_json(sock_file, {"event": "registered"})

        before = nic_counters(nics)
        elapsed_s = 0.0

        def run_checked(ctx: RankContext, request: dict[str, Any]) -> None:
            try:
                run_rank_push(
                    ctx,
                    requests=[request],
                    block_count=args.blocks,
                    block_bytes=args.block_bytes,
                )
            except BaseException as err:
                thread_errors.append(err)

        for iteration in range(args.iterations):
            start = recv_json(sock_file)
            assert start["event"] == "start"
            assert int(start["iteration"]) == iteration
            thread_errors: list[BaseException] = []
            started = time.perf_counter()
            requests_by_rank = requests_by_iteration_by_rank[iteration]
            threads = [
                threading.Thread(
                    target=run_checked,
                    args=(ctx, requests_by_rank[ctx.rank]),
                    name=f"rdma-it-prefill-rank-{ctx.rank}-iter-{iteration}",
                )
                for ctx in contexts
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            elapsed_s += time.perf_counter() - started
            if thread_errors:
                raise RuntimeError(
                    f"{len(thread_errors)} prefill push thread(s) failed"
                ) from (thread_errors[0])
        after = nic_counters(nics)

        bytes_total = (
            args.ranks * args.iterations * args.layers * args.blocks * args.block_bytes
        )
        result = {
            "role": "prefill",
            "ok": True,
            "ranks": args.ranks,
            "layers": args.layers,
            "blocks": args.blocks,
            "block_bytes": args.block_bytes,
            "iterations": args.iterations,
            "bytes": bytes_total,
            "elapsed_ms": elapsed_s * 1000,
            "bandwidth_gbps": bytes_total * 8 / elapsed_s / 1e9,
            "rank_domains": {
                str(ctx.rank): {
                    "cuda_device": ctx.cuda_device,
                    "nic": ctx.nic,
                    "worker_cpu": ctx.worker_cpu,
                    "domains": ctx.engine.num_domains(),
                    "groups": ctx.engine.num_groups(),
                    "link_speed": ctx.engine.aggregated_link_speed(),
                }
                for ctx in contexts
            },
            "nic_delta": nic_delta(before, after, elapsed_s),
            "expected_nic_bytes": expected_nic_bytes(
                contexts,
                iterations=args.iterations,
                block_count=args.blocks,
                block_bytes=args.block_bytes,
            ),
            "source_patterns": source_patterns,
        }
        errors = apply_result_checks(
            result,
            min_bandwidth_gbps=args.min_bandwidth_gbps,
            min_nic_gbps=args.min_nic_gbps,
            min_nic_byte_ratio=args.min_nic_byte_ratio,
        )
        send_json(sock_file, {"event": "prefill_done", "result": result})
        print_result(result, args.json_out)
        if errors:
            raise RuntimeError(result["error"])


def run_decode(args: argparse.Namespace) -> None:
    native = load_native()
    rank_map = parse_rank_map(args.rank_map)
    ranks = selected_ranks(rank_map, args.ranks)
    contexts = create_rank_contexts(
        native,
        rank_map,
        ranks,
        layer_count=args.layers,
        block_count=args.blocks,
        block_bytes=args.block_bytes,
        device=args.device,
        fill_byte=args.dst_byte,
    )
    rank_by_id = {ctx.rank: ctx for ctx in contexts}
    nics = {ctx.nic for ctx in contexts}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.listen_host, args.port))
        server.listen(1)
        stage(f"decode listening on {args.listen_host}:{args.port}")
        sock, addr = server.accept()
        stage(f"decode accepted prefill control connection from {addr}")
        with sock, sock.makefile("rw", encoding="utf-8", newline="\n") as sock_file:
            ready = recv_json(sock_file)
            assert ready["event"] == "prefill_ready"
            assert ready["ranks"] == ranks
            assert ready["layers"] == args.layers
            assert ready["blocks"] == args.blocks
            assert ready["block_bytes"] == args.block_bytes
            assert ready["iterations"] == args.iterations
            src_byte = int(ready["src_byte"])

            requests: list[dict[str, Any]] = []
            requests_by_iteration: dict[int, list[dict[str, Any]]] = {
                iteration: [] for iteration in range(args.iterations)
            }
            for iteration in range(args.iterations):
                for rank in ranks:
                    ctx = rank_by_id[rank]
                    decode_req = f"rdma-it-d-r{rank}-i{iteration}"
                    prefill_req = f"rdma-it-p-r{rank}-i{iteration}"
                    imm_id = ((iteration + 1) << 8) + rank + 1
                    hs = handshake(
                        request_id=decode_req,
                        engine_id="decode",
                        rank=rank,
                        ranks=args.ranks,
                        block_size=args.block_size,
                        layers=ctx.layers,
                        imm_id=imm_id,
                    )
                    register_remote(ctx.engine, decode_req, hs)
                    request = {
                        "rank": rank,
                        "iteration": iteration,
                        "decode_req": decode_req,
                        "prefill_req": prefill_req,
                        "handshake": hs,
                    }
                    requests.append(request)
                    requests_by_iteration[iteration].append(request)
            send_json(sock_file, {"event": "setup", "requests": requests})
            registered = recv_json(sock_file)
            assert registered["event"] == "registered"

            before = nic_counters(nics)
            elapsed_s = 0.0
            verify_results: list[dict[str, Any]] = []

            def wait_checked(ctx: RankContext, req_id: str) -> None:
                try:
                    ctx.engine.wait_done(req_id)
                except BaseException as err:
                    wait_errors.append(err)

            for iteration in range(args.iterations):
                reset_decode_sample_ranges(
                    contexts,
                    sample_bytes=args.verify_sample_bytes,
                    value=args.dst_byte,
                )
                send_json(sock_file, {"event": "start", "iteration": iteration})
                wait_errors: list[BaseException] = []
                started = time.perf_counter()
                wait_threads = [
                    threading.Thread(
                        target=wait_checked,
                        args=(ctx, request["decode_req"]),
                        name=f"rdma-it-decode-rank-{ctx.rank}-iter-{iteration}",
                    )
                    for request in requests_by_iteration[iteration]
                    for ctx in (rank_by_id[int(request["rank"])],)
                ]
                for thread in wait_threads:
                    thread.start()
                for thread in wait_threads:
                    thread.join()
                elapsed_s += time.perf_counter() - started
                if wait_errors:
                    raise RuntimeError(
                        f"{len(wait_errors)} decode wait thread(s) failed"
                    ) from (wait_errors[0])
                verify_result = verify_decode_buffers(
                    contexts,
                    seed=src_byte,
                    sample_bytes=args.verify_sample_bytes,
                )
                verify_result["iteration"] = iteration
                verify_results.append(verify_result)
            after = nic_counters(nics)
            done = recv_json(sock_file)
            assert done["event"] == "prefill_done"

    bytes_total = (
        args.ranks * args.iterations * args.layers * args.blocks * args.block_bytes
    )
    result = {
        "role": "decode",
        "ok": True,
        "ranks": args.ranks,
        "layers": args.layers,
        "blocks": args.blocks,
        "block_bytes": args.block_bytes,
        "iterations": args.iterations,
        "bytes": bytes_total,
        "elapsed_ms": elapsed_s * 1000,
        "bandwidth_gbps": bytes_total * 8 / elapsed_s / 1e9,
        "verify": summarize_verify_results(
            verify_results, sample_bytes=args.verify_sample_bytes
        ),
        "prefill_result": done["result"],
        "rank_domains": {
            str(ctx.rank): {
                "cuda_device": ctx.cuda_device,
                "nic": ctx.nic,
                "worker_cpu": ctx.worker_cpu,
                "domains": ctx.engine.num_domains(),
                "groups": ctx.engine.num_groups(),
                "link_speed": ctx.engine.aggregated_link_speed(),
            }
            for ctx in contexts
        },
        "nic_delta": nic_delta(before, after, elapsed_s),
        "expected_nic_bytes": expected_nic_bytes(
            contexts,
            iterations=args.iterations,
            block_count=args.blocks,
            block_bytes=args.block_bytes,
        ),
    }
    extra_errors = []
    if not result["prefill_result"].get("ok", False):
        extra_errors.append(f"prefill failed: {result['prefill_result'].get('error')}")
    finish_result(
        result,
        json_out=args.json_out,
        min_bandwidth_gbps=args.min_bandwidth_gbps,
        min_nic_gbps=args.min_nic_gbps,
        min_nic_byte_ratio=args.min_nic_byte_ratio,
        extra_errors=extra_errors,
    )


def print_result(result: dict[str, Any], json_out: str | None) -> None:
    text = json.dumps(result, sort_keys=True)
    print(text)
    if json_out is not None:
        Path(json_out).write_text(text + "\n")


def stage(message: str) -> None:
    print(f"[pd-rdma-two-node-it] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="role", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--rank-map")
        subparser.add_argument("--ranks", type=int, default=8)
        subparser.add_argument("--layers", type=int, default=61)
        subparser.add_argument("--blocks", type=int, default=1024)
        subparser.add_argument("--block-size", type=int, default=16)
        subparser.add_argument("--block-bytes", type=int, default=18432)
        subparser.add_argument("--iterations", type=int, default=1)
        subparser.add_argument("--device", choices=("cuda",), default="cuda")
        subparser.add_argument("--json-out")
        subparser.add_argument("--min-bandwidth-gbps", type=float, default=0.0)
        subparser.add_argument("--min-nic-gbps", type=float, default=0.0)
        subparser.add_argument("--min-nic-byte-ratio", type=float, default=0.98)
        subparser.add_argument("--verify-sample-bytes", type=int, default=1024 * 1024)

    decode = subparsers.add_parser("decode")
    add_common(decode)
    decode.add_argument("--listen-host", default="0.0.0.0")
    decode.add_argument("--port", type=int, default=19190)
    decode.add_argument("--dst-byte", type=lambda x: int(x, 0), default=0x00)

    prefill = subparsers.add_parser("prefill")
    add_common(prefill)
    prefill.add_argument("--decode-host", required=True)
    prefill.add_argument("--port", type=int, default=19190)
    prefill.add_argument("--connect-timeout-s", type=float, default=120.0)
    prefill.add_argument("--src-byte", type=lambda x: int(x, 0), default=0x5A)

    args = parser.parse_args()
    if (
        args.ranks <= 0
        or args.layers <= 0
        or args.blocks <= 0
        or args.block_size <= 0
        or args.block_bytes <= 0
    ):
        raise ValueError(
            "ranks, layers, blocks, block-size, and block-bytes must be positive"
        )
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if (
        args.min_bandwidth_gbps < 0
        or args.min_nic_gbps < 0
        or args.min_nic_byte_ratio < 0
    ):
        raise ValueError("bandwidth thresholds must be nonnegative")
    if args.verify_sample_bytes < 0:
        raise ValueError("verify-sample-bytes must be nonnegative")
    if args.role == "decode":
        run_decode(args)
    else:
        run_prefill(args)


if __name__ == "__main__":
    main()
