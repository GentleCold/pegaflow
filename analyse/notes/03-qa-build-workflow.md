# 构建流程 Q&A

## Q1: Python 和 Rust 的交互与构建关系

### 整体数据流

```
vLLM/SGLang (Python)
    │
    ├── import pegaflow.connector (纯 Python)
    │       │
    │       └── from pegaflow.pegaflow import EngineRpcClient, PyLoadState (PyO3 .so)
    │                   │
    │                   └── gRPC ──> pegaflow-server (Rust 二进制)
    │                                     │
    │                                     └── PegaEngine (Rust 核心引擎)
    │
    └── pegaflow-server 命令 (Python wrapper → 调用 Rust 二进制 pegaflow-server-py)
```

### 构建步骤

**只需要一步 `maturin develop -r`**（在 `python/` 目录下），它同时完成：

1. 编译 PyO3 cdylib → `pegaflow/pegaflow.so`（Python 可 import）
2. 编译 `pegaflow-server-py` 二进制（Rust 二进制，打包进 wheel）
3. 编译 `pegaflow-metaserver-py` 二进制（同上）
4. 安装到 `.venv`

```toml
# python/Cargo.toml
[lib]
name = "pegaflow"
crate-type = ["cdylib"]       # → pegaflow.so

[[bin]]
name = "pegaflow-server-py"   # → Rust 二进制，通过 Python wrapper 调用
path = "src/bin/server.rs"

[[bin]]
name = "pegaflow-metaserver-py"
path = "src/bin/metaserver.rs"
```

### pegaflow-server 命令的启动链

```
用户执行: pegaflow-server --port 50055
    │
    └── python/pegaflow/_server.py (entry point, 定义在 pyproject.toml [project.scripts])
            │
            └── find_binary("pegaflow-server-py")  → 找到 maturin 打包的 Rust 二进制
                    │
                    └── subprocess.run([binary] + sys.argv[1:])  → 执行 Rust 二进制
```

### 何时需要单独 `cargo build`？

- **不需要**：日常开发 pegaflow-server/metaserver，`maturin develop` 全搞定
- **需要**：
  - 调试 `pegaflow-router`（在 `pegaflow-server/src/bin/` 下，不在 python crate 里）
  - 只想编译 Rust 不涉及 Python（如跑 `cargo test`、`cargo bench`）
  - CI 中单独跑 `cargo clippy` / `cargo check`

### 关键理解

- `python/Cargo.toml` 的 `pegaflow-py` crate 依赖了 `pegaflow-server` 和 `pegaflow-metaserver`，所以 maturin 编译时会递归编译整个 Rust workspace
- Python 侧的 `pegaflow-server` 命令本质是个 thin wrapper，真正逻辑在 Rust 二进制里
