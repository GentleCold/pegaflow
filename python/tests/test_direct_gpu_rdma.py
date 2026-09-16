"""Owner process -> CUDA IPC/FD registration -> gRPC direct load -> GPU bytes.

Run with PEGAFLOW_IB_DEVICE=mlx5_0 and CUDA_VISIBLE_DEVICES limited to test GPUs:
    pytest -m integration tests/test_direct_gpu_rdma.py
Requires release server/metaserver binaries and the matching native extension.
"""

import hashlib
import os
import pickle
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from .conftest import find_available_port, wait_for_server_ready

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _stop(process):
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def test_ipc_owner_export_direct_rdma_bytes(tmp_path):
    import torch

    from pegaflow.dma_buf import DmaBufExports
    from pegaflow.ipc_wrapper import CudaIPCWrapper
    from pegaflow.pegaflow import EngineRpcClient, PyLoadState

    nic = os.environ.get("PEGAFLOW_IB_DEVICE")
    if not nic:
        pytest.skip("set PEGAFLOW_IB_DEVICE to explicitly select test NICs")
    assert torch.cuda.device_count() >= 2, "select two non-production GPUs"
    root = Path(__file__).resolve().parents[2]
    meta_port, meta_http, *ports = [find_available_port() for _ in range(6)]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    with ExitStack() as stack:

        def start(name, argv):
            log = stack.enter_context((tmp_path / f"{name}.log").open("w"))
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            stack.callback(_stop, process)
            return process

        meta = start(
            "meta",
            [
                str(root / "target/release/pegaflow-metaserver"),
                "--addr",
                f"127.0.0.1:{meta_port}",
                "--http-addr",
                f"127.0.0.1:{meta_http}",
            ],
        )
        deadline = time.monotonic() + 30
        while True:
            assert meta.poll() is None, (tmp_path / "meta.log").read_text()
            try:
                with urlopen(f"http://127.0.0.1:{meta_http}/health", timeout=1):
                    break
            except URLError:
                assert time.monotonic() < deadline, "MetaServer did not become ready"
                time.sleep(0.05)
        clients = []
        for index in range(2):
            start(
                f"server{index}",
                [
                    str(root / "target/release/pegaflow-server"),
                    "--addr",
                    f"127.0.0.1:{ports[index]}",
                    "--http-addr",
                    f"127.0.0.1:{ports[2 + index]}",
                    "--pool-size",
                    "64mb",
                    "--devices",
                    str(index),
                    "--nics",
                    nic,
                    "--metaserver-addr",
                    f"http://127.0.0.1:{meta_port}",
                    "--disable-numa-affinity",
                ],
            )
            endpoint = f"http://127.0.0.1:{ports[index]}"
            assert wait_for_server_ready(endpoint), (tmp_path / f"server{index}.log").read_text()
            clients.append(EngineRpcClient(endpoint))

        namespace = f"ipc-dma-buf-{uuid.uuid4()}"
        names = ["layer0", "layer1"]
        tensors = []
        for device, client in enumerate(clients):
            # Two split-K/V layers share one allocation, with an unaligned view offset.
            allocation = torch.zeros(
                2 * 2 * 8 * 1024 + 8192, dtype=torch.uint8, device=f"cuda:{device}"
            )
            layers = [
                allocation[128 + index * 16384 : 128 + (index + 1) * 16384].view(2, 8, 1024)
                for index in range(2)
            ]
            tensors.append(layers)
            if device == 0:
                for layer_index, layer in enumerate(layers):
                    for segment in range(2):
                        for block in range(8):
                            layer[segment, block].fill_(1 + layer_index * 32 + segment * 8 + block)
            torch.cuda.synchronize(device)
            with DmaBufExports() as exports:
                wrappers = [
                    pickle.dumps(CudaIPCWrapper(layer, exports if device else None))
                    for layer in layers
                ]
                ok, message = client.register_context_batch(
                    f"instance{device}",
                    namespace,
                    0,
                    0,
                    1,
                    1,
                    device,
                    names,
                    wrappers,
                    [8, 8],
                    [1024, 1024],
                    [8192, 8192],
                    [2, 2],
                    "direct",
                    False,
                )
                assert ok, message

        hashes = [hashlib.sha256(f"{namespace}-{index}".encode()).digest() for index in range(8)]
        ok, message = clients[0].save(
            "instance0", 0, 0, 0, [(name, list(range(8)), hashes) for name in names]
        )
        assert ok, message
        deadline = time.monotonic() + 30
        while True:
            ready = clients[1].query_prefetch("instance1", hashes, "owner-ready", direct_gpu=True)
            if getattr(ready, "num_hit_blocks", 0) == 8:
                clients[1].release(ready.lease)
                break
            assert time.monotonic() < deadline, "remote owner never became queryable"
            time.sleep(0.05)

        for case, destinations in enumerate(([3], [0, 1, 2], [6, 1, 4])):
            for layer in tensors[1]:
                layer.zero_()
            torch.cuda.synchronize(1)
            ready = clients[1].query_prefetch(
                "instance1", hashes[: len(destinations)], f"direct-{case}", direct_gpu=True
            )
            assert ready.num_hit_blocks == len(destinations)
            state = PyLoadState()
            ok, message = clients[1].load(
                "instance1", 0, 1, state.shm_name(), [names], [(ready.lease, [destinations])]
            )
            assert ok, message
            deadline = time.monotonic() + 30
            while not state.is_ready():
                assert time.monotonic() < deadline, "direct RDMA did not complete"
                time.sleep(0.01)
            assert state.get_state() == 1
            for source, target in zip(tensors[0], tensors[1], strict=True):
                expected = torch.zeros_like(target, device="cpu")
                expected[:, destinations] = source[:, : len(destinations)].cpu()
                assert torch.equal(target.cpu(), expected), f"byte mismatch in case {case}"
        for index, client in enumerate(clients):
            ok, message = client.unregister_context(f"instance{index}")
            assert ok, message
