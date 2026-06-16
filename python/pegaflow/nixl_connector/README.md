# vLLM Source

The initial contents of this package were copied from vLLM:

- Commit: `c69c73418ab0ad13e28022ed16573019653a9bf7`
- Source path: `vllm/distributed/kv_transfer/kv_connector/v1/nixl`

The original vLLM source headers and Apache-2.0 license notices were preserved.
PegaFlow changes in this package rename/expose the connector as
`PegaNixlConnector` and replace the NIXL data-plane transport with the
PegaFlow RDMA transport.
