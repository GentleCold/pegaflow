# 20 — 开发指南

**面向**：希望在 PegaFlow 项目上进行实际开发的工程师

---

## 1. 环境搭建

### 1.1 必要依赖

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Rust | stable (edition 2024) | 核心语言 |
| CUDA Toolkit | 12.4+ | GPU 内存操作（cudarc 0.19.3） |
| Python | 3.10+ | vLLM/SGLang 集成 |
| maturin | 1.0+ | Python 扩展构建工具 |
| protobuf / grpc tools | 任意版本 | .proto 编译（tonic/prost 自动处理） |

**RDMA 可选依赖**（如需测试 RDMA 功能）：
- `libibverbs-dev`（ibverbs 用户空间库）
- `librdmacm-dev`（RDMA 连接管理）
- InfiniBand 或 RoCE 网卡

### 1.2 Rust 工具链

```bash
# 安装 Rust（推荐 rustup）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 确认版本（需要 edition 2024，即 Rust 1.85+）
rustc --version
cargo --version

# 推荐安装 cargo-watch（开发时自动重编译）
cargo install cargo-watch

# 推荐安装 cargo-nextest（更快的测试运行器）
cargo install cargo-nextest

# typos（CI 用，检查拼写错误）
cargo install typos-cli
```

### 1.3 构建项目

```bash
# 克隆项目
git clone <repository>
cd pegaflow

# Debug 构建（快速，用于开发）
cargo build

# Release 构建（优化，用于性能测试和生产部署）
cargo build --release

# 只构建特定 crate
cargo build -p pegaflow-core
cargo build -p pegaflow-server
```

### 1.4 Python 开发环境

```bash
# 创建虚拟环境（项目约定用 .venv）
python -m venv .venv
source .venv/bin/activate

# 安装 maturin（Python-Rust 混合构建工具）
pip install maturin

# 开发模式安装（修改 Rust 代码后重新 build 即可，无需 reinstall）
cd python
maturin develop

# 安装开发依赖
pip install -e ".[dev]"
```

**为什么需要 maturin？**

PegaFlow 的 Python 包（`pegaflow-llm`）包含 Rust 扩展模块（`.so` 文件）。maturin 负责：
1. 调用 `cargo build` 编译 Rust 代码
2. 将 `.so` 文件放到正确位置
3. 生成 `wheel` 包

---

## 2. 运行测试

### 2.1 Rust 单元测试

```bash
# 运行所有测试
cargo test

# 运行特定 crate 的测试
cargo test -p pegaflow-core
cargo test -p pegaflow-metaserver

# 运行特定测试函数（支持名称前缀匹配）
cargo test prefix_hit          # 运行所有包含 "prefix_hit" 的测试
cargo test -p pegaflow-core -- --nocapture  # 显示 println! 输出

# 使用 nextest（更快，更好的输出格式）
cargo nextest run
cargo nextest run -p pegaflow-core
```

### 2.2 Python 测试

```bash
cd python
source ../.venv/bin/activate

# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_connector.py

# 排除集成测试（需要运行中的 server）
pytest -m "not integration"
```

### 2.3 基准测试

```bash
# 固定内存拷贝基准（测试 PCIe 带宽利用率）
cargo bench --bench pinned_copy

# UDS（Unix Domain Socket）延迟基准
cargo bench --bench uds_latency
```

---

## 3. CI 检查

本地运行与 CI 相同的检查（提交前必做）：

```bash
./scripts/check.sh
```

该脚本依次运行：

```bash
# 1. 代码格式化检查
cargo fmt --check

# 2. 拼写检查（检查注释/文档中的错别字）
typos

# 3. Lint 检查
cargo clippy --all-targets -- -D warnings

# 4. 编译检查（不产生二进制）
cargo check
```

**提交信息格式**（Commitizen 规范）：

```
feat: 添加跨节点 P/D 数据传输支持
fix: 修复 RDMA 握手超时导致连接泄漏
refactor: 重构 WritePipeline 减少锁竞争
chore: 升级 tonic 到 0.14
docs: 补充 MetaServer TTL 设计说明
test: 添加 ReadCache pin/unpin 集成测试
ci: 修复 GitHub Actions CUDA 环境配置
```

⚠️ **不要直接 commit 到 master**，请创建 `feat/`, `fix/`, `chore/` 等分支。

---

## 4. 项目结构速查

```
pegaflow/
├── Cargo.toml                  # Workspace 根配置（版本、共享依赖）
├── scripts/check.sh            # CI 本地检查脚本
│
├── pegaflow-common/            # 公共工具（logging, numa, hll）
│   └── src/
│       ├── lib.rs
│       ├── logging.rs          # 日志初始化（logforth）
│       ├── numa.rs             # NUMA 拓扑检测
│       └── hll.rs              # HyperLogLog 算法
│
├── pegaflow-core/              # 核心存储引擎
│   └── src/
│       ├── lib.rs              # PegaEngine 主入口
│       ├── block.rs            # Block 类型体系
│       ├── cache.rs            # TinyLFU 缓存
│       ├── gpu_worker.rs       # GPU worker 线程
│       ├── offload.rs          # Save 路径（GPU→CPU）
│       ├── storage/
│       │   ├── mod.rs          # StorageEngine 整合
│       │   ├── read_cache.rs   # Pin/unpin/consume
│       │   ├── write_path.rs   # Insert worker（Actor）
│       │   ├── prefetch.rs     # SSD+RDMA 预取调度
│       │   ├── remote_fetch.rs # 跨节点 RDMA fetch
│       │   └── transfer_lock.rs # RDMA 期间防驱逐锁
│       ├── backing/
│       │   ├── ssd.rs          # SSD 后端协调器
│       │   ├── ssd_cache.rs    # SSD Ring Buffer
│       │   └── uring.rs        # io_uring 异步 I/O
│       ├── internode/
│       │   ├── metaserver_client.rs  # MetaServer 客户端
│       │   └── service_discovery.rs  # K8s Pod 发现
│       ├── pinned_pool.rs      # 固定内存池
│       └── sync_state.rs       # LoadState 共享内存
│
├── pegaflow-proto/             # Protobuf 定义
│   └── proto/engine.proto      # Engine + MetaServer gRPC 服务
│
├── pegaflow-server/            # gRPC 服务
│   └── src/
│       ├── service.rs          # RPC handler 实现
│       ├── registry.rs         # CUDA Tensor 注册表
│       └── http_server.rs      # /metrics /health /ready
│
├── pegaflow-metaserver/        # 跨节点 Block Hash 注册中心
│   └── src/
│       ├── service.rs          # gRPC 服务
│       └── store.rs            # moka LRU 缓存
│
├── pegaflow-transfer/          # RDMA 传输引擎
│   └── src/
│       ├── engine.rs           # MooncakeTransferEngine
│       └── sideway_backend.rs  # UD 控制面 + RC 数据面
│
└── python/                     # Python 包
    ├── Cargo.toml              # PyO3 Rust 侧配置
    ├── src/lib.rs              # PyO3 绑定
    └── pegaflow/
        ├── pegaflow.pyi        # Python 类型存根
        ├── ipc_wrapper.py      # CUDA IPC 封装
        └── connector/
            ├── scheduler.py    # vLLM Scheduler 侧连接器
            ├── worker.py       # vLLM Worker 侧连接器
            └── common.py       # 共享数据结构
```

---

## 5. 常见开发任务

### 5.1 新增 gRPC RPC 方法

**步骤**：

1. **修改 proto 文件**（`pegaflow-proto/proto/engine.proto`）：
   ```protobuf
   // 在 service Engine 中新增
   rpc MyNewRpc(MyRequest) returns (MyResponse);
   
   // 定义消息类型
   message MyRequest { string instance_id = 1; }
   message MyResponse { Status status = 1; }
   ```

2. **重新生成代码**（Tonic 在编译时自动生成，无需手动运行 protoc）：
   ```bash
   cargo build -p pegaflow-proto
   ```

3. **在 service.rs 实现 handler**（`pegaflow-server/src/service.rs`）：
   ```rust
   async fn my_new_rpc(
       &self,
       request: Request<MyRequest>,
   ) -> Result<Response<MyResponse>, Status> {
       let req = request.into_inner();
       
       // 调用引擎
       self.engine.my_new_operation(&req.instance_id)
           .map_err(Self::map_engine_error)?;
       
       Ok(Response::new(MyResponse { status: Some(Self::ok_status()) }))
   }
   ```

4. **在 PegaEngine 实现业务逻辑**（`pegaflow-core/src/lib.rs`）：
   ```rust
   pub fn my_new_operation(&self, instance_id: &str) -> Result<(), EngineError> {
       let instance = self.get_instance(instance_id)?;
       // ... 实现逻辑
       Ok(())
   }
   ```

5. **在 Python 绑定中暴露**（如需要，`python/src/lib.rs`）：
   ```rust
   #[pymethods]
   impl EngineRpcClient {
       fn my_new_rpc(&self, instance_id: &str) -> PyResult<()> {
           self.call_rpc("my_new_rpc", self.client.clone().my_new_rpc(MyRequest {
               instance_id: instance_id.to_string(),
           }))
       }
   }
   ```

### 5.2 修改存储策略（如调整 TinyLFU 参数）

TinyLFU 配置在 `StorageConfig`（`pegaflow-core/src/storage/mod.rs`）：

```rust
pub struct StorageConfig {
    pub enable_lfu_admission: bool,  // 是否启用 TinyLFU 准入控制
    pub hint_value_size_bytes: Option<usize>,  // LFU 滑动窗口大小提示
    pub max_prefetch_blocks: usize,  // 最大并发预取 block 数
    // ...
}
```

Python 侧配置（`python/src/lib.rs` 中的 `PegaEngineConfig`）：

```rust
#[pyclass]
struct PegaEngineConfig {
    pub pinned_pool_bytes: usize,   // 固定内存池大小
    pub enable_lfu: bool,           // TinyLFU 开关
    pub max_prefetch_blocks: usize, // 预取并发上限
    // ...
}
```

### 5.3 新增 Prometheus 指标

**Rust 侧**（`pegaflow-core/src/metrics.rs`）：

```rust
pub struct CoreMetrics {
    // 已有指标...
    pub my_new_counter: Counter<u64>,  // 新增计数器
}

fn init_metrics() -> CoreMetrics {
    CoreMetrics {
        my_new_counter: meter.u64_counter("pegaflow_my_new_total")
            .with_description("My new metric description")
            .build(),
    }
}
```

**使用**：

```rust
core_metrics().my_new_counter.add(1, &[]);
```

**查看**：启动后访问 `http://localhost:9090/metrics`（HTTP 服务端口）。

### 5.4 修改写路径（添加新的插入逻辑）

写路径入口：`pegaflow-core/src/storage/write_path.rs`

```rust
// insert_worker_loop() 中的 InsertWorkerCommand 枚举
pub(crate) enum InsertWorkerCommand {
    RawInsert(RawSaveBatch),
    Gc { max_age: Duration, reply: oneshot::Sender<usize> },
    // 新增命令：
    MyNewCommand { ... },
}
```

> **注意**：修改 insert worker 时要确保不阻塞（所有操作应该 `O(n)` 或更好，不能有长时间的锁持有）。

---

## 6. 调试技巧

### 6.1 日志级别控制

```bash
# 全局 info 级别
RUST_LOG=info ./pegaflow-server

# 对特定 crate 开启 debug
RUST_LOG=info,pegaflow_core=debug,pegaflow_server=debug ./pegaflow-server

# 只看存储引擎的 trace 级别
RUST_LOG=pegaflow_core::storage=trace ./pegaflow-server

# 关闭某个噪音模块
RUST_LOG=debug,pegaflow_core::cache=warn ./pegaflow-server
```

**日志格式**（logforth，带颜色）：

```
2026-03-31T10:23:45.123Z INFO  pegaflow_core::storage::write_path insert_worker: inserted hash=abc123
2026-03-31T10:23:45.124Z DEBUG pegaflow_core::backing::ssd submit_prefix: keys=5 found=3
```

### 6.2 环境变量配置

```bash
# gRPC 服务地址（客户端配置）
PEGAFLOW_ENGINE_ENDPOINT=127.0.0.1:50055

# 实例 ID 覆盖（测试时有用）
PEGAFLOW_INSTANCE_ID=test-instance-1

# MetaServer 地址
PEGAFLOW_METASERVER_ENDPOINT=127.0.0.1:50056

# Connector 调优参数
PEGA_BYPASS_BLOCKS=4           # 短于 4 块的请求在高负载时跳过远端查询
PEGA_HIGH_LOAD_THRESHOLD=10    # pending_prefetches >= 10 视为高负载
PEGA_MAX_PENDING_SAVE_REQUESTS=100  # 最多 100 个请求同时 Save
```

### 6.3 Prometheus 指标监控

启动 PegaFlow Server 后，访问：

```
http://localhost:8080/health    # {"status":"ok"}（liveness probe）
http://localhost:8080/ready     # 就绪检查（readiness probe）
http://localhost:8080/metrics   # Prometheus 格式指标
```

**常用指标**：

| 指标名 | 类型 | 含义 |
|--------|------|------|
| `pegaflow_cache_block_hits_total` | Counter | 内存缓存命中次数 |
| `pegaflow_cache_block_misses_total` | Counter | 缓存未命中次数 |
| `pegaflow_cache_resident_bytes` | Gauge | 当前内存中 block 占用字节 |
| `pegaflow_save_bytes_total` | Counter | 累计 Save 字节数 |
| `pegaflow_save_duration_seconds` | Histogram | Save RPC 耗时分布 |
| `pegaflow_pool_alloc_failures_total` | Counter | 固定内存分配失败次数 |
| `pegaflow_inflight_bytes` | Gauge | 正在等待 TP slot 的 InflightBlock 字节数 |
| `pegaflow_ssd_prefetch_bytes_total` | Counter | SSD 预取字节数 |
| `pegaflow_load_failures_total` | Counter | GPU Load 失败次数 |
| `pegaflow_metaserver_registered_total` | Counter | 注册到 MetaServer 的 block 数 |

### 6.4 常见问题排查

**问题：Save RPC 报 `WorkerMissing`**

原因：指定的 `device_id` 未注册（`RegisterContextBatch` 未完成）。

检查：
```bash
# 查看 server 日志，确认 RegisterContextBatch 成功
grep "Registered context batch" server.log

# 或检查指标
curl http://localhost:8080/metrics | grep registered
```

**问题：固定内存池耗尽（`Storage: pinned pool exhausted`）**

原因：并发 Save 请求过多，或内存池配置太小。

解决：
1. 增大 `pinned_pool_bytes`（默认 30GB）
2. 减少并发 Save 请求（设置 `PEGA_MAX_PENDING_SAVE_REQUESTS`）
3. 检查 `pegaflow_pool_alloc_failures_total` 指标

**问题：RDMA 握手失败**

原因可能：
- 网卡名称配置错误（`rdma_nic_names` 配置项）
- 防火墙阻止 RDMA 端口
- 客户端和服务端 RDMA 驱动版本不兼容

检查：
```bash
# 查看 RDMA 拓扑
pegaflow_topo_cli  # CLI 工具，显示 NUMA-GPU-NIC 拓扑

# 运行 RDMA benchmark（验证连通性）
pegaflow_cpu_bench --remote <server_ip>
```

---

## 7. 架构扩展建议

### 7.1 新增存储后端

当前存储层次：内存 → SSD（io_uring）→ RDMA 远端节点

如需新增远端对象存储（如 S3）：

1. 在 `pegaflow-core/src/backing/` 新增 `s3.rs`，实现类似 `SsdBackingStore` 的接口
2. 在 `StorageEngine` 中集成
3. 在 `PrefetchScheduler::full_prefix_scan()` 中添加第四条预取路径

### 7.2 高可用 MetaServer

当前 MetaServer 是单点。高可用方案：
- 在 MetaServer 前加 L7 负载均衡（如 Nginx）
- MetaServer 是无状态的（丢失索引后，下次注册会重建）
- 分片（Sharding）：按 namespace hash 分片到多个 MetaServer 实例

### 7.3 服务端 Save 压缩

当前数据以未压缩格式存储在 SSD。可以在 `insert_worker_loop()` 中，在 `ssd_store.ingest_batch()` 之前添加压缩（LZ4/Zstd）：
- 优点：减少 SSD 写入量，提高有效容量
- 缺点：增加 CPU 开销，读取时需要解压

---

## 8. 关键 Rust 模式总结

### Arc + RwLock 并发模式

```rust
// PegaEngine::instances 的访问模式
let instances = self.instances.read().expect("...");  // 多读者并发
let instances = self.instances.write().expect("...");  // 独占写入
```

### oneshot 通道（请求-响应）

```rust
// GPU worker 完成通知
let (tx, rx) = oneshot::channel();
self.save_tx.send(SaveTask { ..., reply: tx })?;
rx.await?  // 等待 GPU worker 完成后回复
```

### Arc::new_cyclic（循环引用）

```rust
// StorageEngine 持有对自身的弱引用（见文档 07）
let storage = Arc::new_cyclic(|weak_self| {
    StorageEngine { self_ref: weak_self.clone(), ... }
});
```

### fire-and-forget MPSC

```rust
// 发送后不等待（适用于写路径、MetaServer 注册等）
match self.insert_tx.try_send(batch) {
    Ok(()) => {},
    Err(mpsc::error::TrySendError::Full(_)) => warn!("queue full, dropping"),
    Err(mpsc::error::TrySendError::Closed(_)) => {},  // 关机中
}
```

### OnceLock 全局单例

```rust
// 线程安全的一次性初始化（Python PyO3 中的 Tokio runtime）
static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    TOKIO_RUNTIME.get_or_init(|| Runtime::new().unwrap())
}
```

### unsafe NonNull + Send

```rust
// 裸指针跨线程传递（GPU SaveBlock）
pub struct SaveBlock {
    pub k_dst_ptr: *mut u8,  // 裸指针
}
// 手动声明 Send 安全性（指针不会并发访问）
unsafe impl Send for SaveBlock {}
```

---

## 9. 依赖库速查

| 库 | 版本 | 用途 |
|----|------|------|
| `tokio` | 1.50 | 异步运行时（全功能） |
| `tonic` | 0.14 | gRPC 服务端/客户端 |
| `prost` | 0.14 | Protobuf 编解码 |
| `pyo3` | 0.28 | Python-Rust 互操作 |
| `cudarc` | 0.19.3 | CUDA 绑定（cuda-12040） |
| `parking_lot` | 0.12 | 高性能 Mutex/RwLock |
| `dashmap` | 6.1 | 并发哈希表 |
| `moka` | latest | 异步 LRU/TTL 缓存 |
| `shared_memory` | 0.12 | POSIX 共享内存 |
| `sideway` | 0.4.1 | RDMA 传输（UD/RC） |
| `logforth` | 0.29 | 结构化日志（带颜色） |
| `fastrace` | 0.7 | 分布式链路追踪 |
| `axum` | 0.8 | HTTP 服务（metrics/health） |
| `prometheus` | 0.14 | Prometheus 指标 |
| `uuid` | 1.22 | UUID v4 生成 |
| `clap` | 4.5 | 命令行参数解析 |
| `offset-allocator` | 0.2 | 高效连续内存偏移分配 |
| `hashlink` | 0.11 | 有序哈希表（LRU 顺序） |
| `maturin` | 1.x | Python 扩展构建 |

---

## 10. 开发工作流

```
1. 理解任务
   阅读相关 analyse/ 文档
   阅读对应源码

2. 创建分支
   git checkout -b feat/my-feature

3. 编写代码
   cargo watch -x check  # 实时检查编译错误

4. 运行测试
   cargo nextest run -p pegaflow-core

5. 本地 CI 检查
   ./scripts/check.sh

6. 提交（Commitizen 格式）
   git add -p  # 选择性暂存（避免提交 .env 等敏感文件）
   git commit -m "feat: 添加新功能描述"

7. Push 并创建 PR
   git push -u origin feat/my-feature
```
