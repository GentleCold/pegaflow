#!/usr/bin/env bash
# Verification script for issue #185: PCIe link speed in topology log
set -euo pipefail

echo "=== Step 1: Build pegaflow-transfer ==="
cargo build -p pegaflow-transfer

echo ""
echo "=== Step 2: Run rdma_topo unit tests ==="
cargo test -p pegaflow-transfer -- rdma_topo --nocapture

# do ci check in local
# echo ""
# echo "=== Step 3: CI lint checks ==="
# ./scripts/check.sh

echo ""
echo "=== All checks passed ==="
echo ""
echo "Optional: run the following on a server with GPU + RDMA NICs to verify the topology log:"
echo "  RUST_LOG=info cargo run -p pegaflow-server -- --nics mlx5_0"
