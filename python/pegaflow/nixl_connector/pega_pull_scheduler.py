# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Scheduler-side Pega RDMA v1 hooks for the NIXL pull connector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from pegaflow.nixl_connector.metadata import NixlHandshakePayload
from pegaflow.nixl_connector.pega_rdma_v1 import (
    PEGA_RDMA_V1_ACCEPT_ENDPOINT,
    PEGA_RDMA_V1_EXTENSION,
    PegaRdmaV1Config,
    accept_handshake_via_zmq,
    reverse_peer_key,
)
from pegaflow.nixl_connector.pull_scheduler import NixlPullConnectorScheduler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class PegaNixlPullConnectorScheduler(NixlPullConnectorScheduler):
    """Fold Pega RDMA v1 accept metadata into NIXL's GET_META response."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._pega_rdma_config = PegaRdmaV1Config.from_extra_config(extra_config)

    def _handle_handshake_extensions(
        self,
        target_tp_rank: int,
        payload: NixlHandshakePayload,
        request_extensions: dict[str, Any],
    ) -> NixlHandshakePayload:
        """Accept D-side RDMA metadata before replying to NIXL GET_META."""
        rdma_request = request_extensions.get(PEGA_RDMA_V1_EXTENSION)
        if rdma_request is None:
            return payload
        if not isinstance(rdma_request, dict):
            raise ValueError("Pega RDMA v1 request extension must be a dict")

        peer_key = rdma_request.get("peer_key")
        metadata = rdma_request.get("metadata")
        if not isinstance(peer_key, str) or not isinstance(metadata, bytes):
            raise ValueError("Pega RDMA v1 request extension missing peer_key/metadata")

        endpoint = (payload.extensions or {}).get(PEGA_RDMA_V1_ACCEPT_ENDPOINT)
        if not isinstance(endpoint, str):
            raise RuntimeError(
                "Pega RDMA v1 accept endpoint missing from target rank handshake payload"
            )

        timeout_ms = int(self._pega_rdma_config.handshake_timeout_s * 1000)
        response_extensions = dict(payload.extensions or {})
        try:
            response_metadata = accept_handshake_via_zmq(
                endpoint,
                reverse_peer_key(peer_key),
                metadata,
                timeout_ms,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "metadata": response_metadata,
            }
        except Exception as exc:
            logger.warning(
                "Failed to accept Pega RDMA v1 handshake target_rank=%s peer=%s: %s",
                target_tp_rank,
                peer_key,
                exc,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "error": str(exc),
            }
        logger.debug(
            "Accepted Pega RDMA v1 handshake target_rank=%s peer=%s",
            target_tp_rank,
            peer_key,
        )
        return NixlHandshakePayload(
            compatibility_hash=payload.compatibility_hash,
            agent_metadata_bytes=payload.agent_metadata_bytes,
            extensions=response_extensions,
        )
