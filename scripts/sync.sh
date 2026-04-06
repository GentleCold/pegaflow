#!/usr/bin/env bash
# sync.sh — Sync code to jumpserver (exclude build artifacts and large files)
# Usage: ./scripts/sync.sh [target]
#
# Targets:
#   5090   - 192.168.172.86  (default)
#   h100   - 60.217.65.245
#   4090   - 10.121.196.3

set -euo pipefail

# ──────────────────────────── Targets ─────────────────────────────
declare -A TARGETS=(
    [5090]="kexi@root@192.168.172.86"
    [h100]="kexi@root@60.217.65.245"
    [4090]="kexi@root@10.121.196.3"
)
DEFAULT_TARGET="5090"

# ──────────────────────────── Parse args ──────────────────────────
TARGET="${1:-$DEFAULT_TARGET}"

if [[ -z "${TARGETS[$TARGET]+x}" ]]; then
    echo "Unknown target: $TARGET"
    echo "Available targets: ${!TARGETS[*]}"
    exit 1
fi

# ──────────────────────────── Config ──────────────────────────────
JUMP_HOST="jumpcg.ppio.cloud"
JUMP_PORT="2222"
JUMP_USER="${TARGETS[$TARGET]}"
REMOTE_DIR="~/kexi/pegaflow"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH_OPTS="-p ${JUMP_PORT} -l ${JUMP_USER}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  target: ${TARGET} (${JUMP_USER})"
echo "    from: ${LOCAL_DIR}/"
echo "      to: ${JUMP_HOST}:${REMOTE_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create remote directory if needed
ssh ${SSH_OPTS} "${JUMP_HOST}" "mkdir -p ${REMOTE_DIR}"

rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    --exclude='.git/' \
    --exclude='target/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.so' \
    --exclude='*.egg-info/' \
    --exclude='.pytest_cache/' \
    --exclude='examples/bench_results/' \
    --exclude='examples/pd_logs/' \
    --exclude='assets/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='analyse/' \
    --exclude='CLAUDE.local.md' \
    "${LOCAL_DIR}/" \
    "${JUMP_HOST}:${REMOTE_DIR}/"

echo ""
echo "✓ Sync to ${TARGET} complete"
