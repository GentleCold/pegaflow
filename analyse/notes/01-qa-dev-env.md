# 开发环境 & 部署 Q&A

## Q1: patchelf 警告

**现象**：运行 `maturin develop` 时出现：
```
⚠️ Warning: Failed to set rpath for .../libpegaflow.so: Failed to execute 'patchelf'
```

**原因**：maturin 尝试用 `patchelf` 修改 `.so` 的 rpath（动态库搜索路径），但系统未安装。

**结论**：`maturin develop` 开发模式下完全无害，可忽略。patchelf 只在 `maturin build` 打 wheel 发布包时才真正需要。

```bash
# 如需打 wheel 再装
pip install maturin[patchelf]
```

---

## Q2: pegaflow-server 报找不到 libpython3.13.so

**现象**：
```
/root/kexi/pegaflow/target/debug/pegaflow-server-py: error while loading shared libraries:
libpython3.13.so.1.0: cannot open shared object file: No such file or directory
```

**原因**：`pegaflow-server-py`（`python/src/bin/server.rs`）是通过 PyO3 构建的，动态链接了 uv 管理的 Python 3.13，但 uv 的 Python 不在系统库搜索路径里。

**解决**：直接用纯 Rust 的服务进程，不依赖 Python：
```bash
cargo run --release --bin pegaflow-server
# 或
./target/release/pegaflow-server
```

两个 binary 对比：

| Binary | 源码 | 链接 Python | 用途 |
|--------|------|------------|------|
| `pegaflow-server`（pegaflow-server crate）| `pegaflow-server/src/main.rs` | ❌ | 正式服务进程 |
| `pegaflow-server-py`（python crate）| `python/src/bin/server.rs` | ✅ | pip install 场景的包装入口 |

---

## Q3: benchmark 报无法连接 50055

**现象**：
```
pegaflow.PegaFlowServiceError: failed to connect to engine server at
http://127.0.0.1:50055: transport error
```

**原因**：PegaFlow gRPC 服务未启动，vLLM 初始化时找不到服务端。

**解决**：先启动服务再跑 benchmark：
```bash
# 终端 1：启动服务
cargo run --release --bin pegaflow-server

# 等看到以下日志后
# INFO pegaflow_server: gRPC server listening on 0.0.0.0:50055

# 终端 2：跑 benchmark
cd python && uv run python ../examples/bench_kv_cache.py \
    --model /data/Qwen3-8B --num-prompts 10
```

验证服务健康：
```bash
curl http://127.0.0.1:8080/health  # 返回 {"status":"ok"}
```

---

## Q4: cargo run -r 和 maturin develop -r 的 `-r` 参数

两者都是 `--release` 的缩写，区别在于作用对象：

| 命令 | 作用 |
|------|------|
| `cargo run -r` | release 模式编译 Rust 二进制，输出到 `target/release/` |
| `maturin develop -r` | release 模式编译 PyO3 `.so` 扩展，安装到 `.venv` |

- **debug 模式**：编译快，运行慢，适合日常开发
- **release 模式**：编译慢，运行快，适合性能测试和生产

---

## Q5: 代码同步到 jumpserver

**连接信息**：
```
ssh -p 2222 -l kexi@root@192.168.172.86 jumpcg.ppio.cloud
```

**一键同步**：
```bash
./scripts/sync.sh
```

排除了：`target/`、`.venv/`、`__pycache__/`、`*.so`、`assets/`、`analyse/` 等大文件/无关目录。

**上传单个文件**：
```bash
scp -P 2222 本地文件 kexi@root@192.168.172.86@jumpcg.ppio.cloud:~/kexi/pegaflow/目标路径
```
