#!/usr/bin/env bash
# remote_build.sh — Run this ON the jumpserver to set up Python env and build PegaFlow
# Usage: bash ~/kexi/pegaflow/scripts/remote_build.sh
#
# Idempotent: safe to run multiple times. Skips steps that are already done.
#
# Build workflow:
#   1. `maturin develop -r` in python/ compiles EVERYTHING:
#      - PyO3 cdylib (pegaflow.so) for Python import
#      - pegaflow-server-py / pegaflow-metaserver-py Rust binaries (bundled in wheel)
#      - Recursively compiles the entire Rust workspace (pegaflow-core, pegaflow-server, etc.)
#   2. Standalone `cargo build -r` only needed for binaries NOT in the python crate
#      (e.g., pegaflow-router)

set -euo pipefail

PROJ_DIR="$HOME/kexi/pegaflow"
VENV_DIR="$PROJ_DIR/python/.venv"

# ── Mirror sources (for faster downloads in China) ──────────────
export UV_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Building PegaFlow on $(hostname)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── [1/6] Check tools ────────────────────────────────────────────
echo ""
echo "[1/6] Checking tools..."

# Check / install uv
if ! command -v uv &>/dev/null; then
    echo "  uv not found, installing..."
    pip install uv --quiet || python3 -m pip install uv --quiet
fi
echo "  uv: $(uv --version)"

# Check Rust toolchain
if ! command -v cargo &>/dev/null; then
    echo "  cargo not found — installing rustup (stable)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
echo "  cargo: $(cargo --version)"
echo "  rustc: $(rustc --version)"

# ── [2/6] Create or reuse virtual environment ───────────────────
echo ""
cd "$PROJ_DIR/python"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "[2/6] Reusing existing virtual environment at $VENV_DIR"
else
    echo "[2/6] Creating virtual environment (.venv) in $PROJ_DIR/python/ ..."
    uv venv .venv --python python3
fi
source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)"

# ── Set up Python environment for PyO3 ──────────────────────────
# PYO3_PYTHON: tells PyO3 build script which interpreter to link against.
# Without this, it may pick the system Python instead of the venv one.
export PYO3_PYTHON="$(which python)"

# LD_LIBRARY_PATH: the compiled Rust binaries (pegaflow-server-py, etc.) are
# dynamically linked against libpython3.x.so. At runtime the dynamic linker
# needs to find this .so; LIBDIR tells us where the venv's Python keeps it.
PYTHON_LIBDIR="$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
export LD_LIBRARY_PATH="${PYTHON_LIBDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "  PYO3_PYTHON: $PYO3_PYTHON"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ── [3/6] Install maturin (skips if already installed) ───────────
echo ""
echo "[3/6] Ensuring maturin is installed..."
uv pip install "maturin>=1.0,<2.0"

# ── [4/6] Install Python dev dependencies (skips if satisfied) ──
echo ""
echo "[4/6] Ensuring Python dependencies (vllm + dev tools)..."
uv pip install vllm pytest pytest-cov pytest-xdist numpy

# ── [5/6] Build Python package (includes all Rust binaries) ─────
echo ""
echo "[5/6] Building with maturin develop (compiles PyO3 lib + server/metaserver binaries)..."
maturin develop -r

# ── [6/6] Build Rust binaries ────────────────────────────────────
# maturin develop only compiles the cdylib (pegaflow.so), NOT [[bin]] targets.
# Binaries are only bundled when building wheels (maturin build).
# So we must compile them separately with cargo.
echo ""
echo "[6/6] Building Rust binaries..."
cd "$PROJ_DIR"
cargo build --release \
    --bin pegaflow-server-py \
    --bin pegaflow-metaserver-py \
    --bin pegaflow-router

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Build complete!"
echo ""
echo "Activate the environment (LD_LIBRARY_PATH needed for Rust binaries to find libpython):"
echo "  cd $PROJ_DIR/python && source .venv/bin/activate"
echo "  export LD_LIBRARY_PATH=\$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))'):\$LD_LIBRARY_PATH"
echo ""
echo "Quick smoke test:"
echo "  python -c 'import pegaflow; print(\"OK\")'  # Python bindings"
echo "  pegaflow-server --help                      # Server binary (via Python wrapper)"
echo ""
echo "Standalone binaries:"
echo "  $PROJ_DIR/target/release/pegaflow-router --help"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
