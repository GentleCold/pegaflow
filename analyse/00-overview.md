# 00 — PegaFlow 项目总览

## 1. 项目定位

PegaFlow 是一个**高性能 KV 缓存传输系统**，专为大语言模型（LLM）推理场景设计，与 **vLLM** 和 **SGLang** 集成。其核心价值：

- **GPU → CPU 卸载**：将 GPU 显存中的 KV cache（Key-Value 注意力缓存）搬运到 CPU 固定内存（pinned memory），当 GPU 显存不足时继续服务
- **SSD 二级缓存**：当 CPU 内存也不足时，进一步将 KV 块写入 SSD（通过 io_uring）
- **RDMA 跨节点传输**：多机部署时，通过 RDMA（远程直接内存访问）在节点间传输 KV 块，避免重复计算
- **内容寻址**：每个 KV 块用内容哈希（SHA-256 等）标识，相同内容的块天然去重

---

## 2. 七个 Crate 的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    pegaflow (workspace)                         │
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────────────────┐     │
│  │ pegaflow-common  │◄────│ pegaflow-core                │     │
│  │  - logging       │     │  - PegaEngine (主引擎)        │     │
│  │  - numa          │     │  - StorageEngine (存储引擎)   │     │
│  │  - hll           │     │  - ReadCache, WritePipeline   │     │
│  └──────────────────┘     │  - PinnedPool (内存池)        │     │
│           ▲               │  - GpuWorker                  │     │
│           │               └──────────────┬───────────────┘     │
│  ┌────────┴─────────┐                    │uses                  │
│  │ pegaflow-proto   │◄───────────────────┤                      │
│  │  - engine.proto  │     ┌──────────────▼───────────────┐     │
│  │  (gRPC 定义)     │     │ pegaflow-transfer             │     │
│  └──────────────────┘     │  - TransferEngine (RDMA)     │     │
│           ▲               │  - RcBackend (RC 队列对)      │     │
│           │               └──────────────────────────────┘     │
│  ┌────────┴─────────┐     ┌──────────────────────────────┐     │
│  │ pegaflow-server  │     │ pegaflow-metaserver           │     │
│  │  - gRPC 服务端   │     │  - block hash 注册中心        │     │
│  │  - HTTP 健康检查 │     │  - LRU + TTL store            │     │
│  └──────────────────┘     └──────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ python (pyo3 + Python)                                   │  │
│  │  - PyO3 绑定：PegaEngine + EngineClient                  │  │
│  │  - vLLM v1 KV Connector (scheduler + worker)             │  │
│  │  - SGLang radix cache 集成                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**依赖方向**（A → B 表示 A 依赖 B）：
- `pegaflow-core` → `pegaflow-common`, `pegaflow-transfer`
- `pegaflow-server` → `pegaflow-core`, `pegaflow-proto`, `pegaflow-common`
- `pegaflow-metaserver` → `pegaflow-proto`, `pegaflow-common`
- `python` → `pegaflow-core`, `pegaflow-proto`

---

## 3. 核心概念词典

| 术语 | 含义 |
|------|------|
| **Instance** | 一个模型实例，由 `instance_id` + `namespace` 标识，具有固定的 `num_layers`（层数）和 `tp_size`（张量并行度） |
| **Worker** | 张量并行中的一个 rank（编号），每个 worker 对应一块 GPU（`device_id`） |
| **Block** | KV cache 的存储单元，内容寻址，同一内容的 block 全局唯一 |
| **BlockKey** | `namespace` + `hash` 的组合键，用于在存储层唯一标识一个 block |
| **Slot** | 在 TP（张量并行）场景下，一个 block 有 `tp_size` 个 slot，每个 GPU worker 存一份 |
| **SealedBlock** | 所有 slot 都已填充完毕的不可变 block，可安全读取 |
| **InflightBlock** | 正在被写入的 block，部分 slot 可能还未到达 |
| **Pinned Memory** | CUDA 固定内存（页锁定内存），物理地址不变，DMA 传输不需要额外拷贝 |
| **Namespace** | 用于隔离不同模型/实例的命名空间（如模型名称），防止不同模型的 block 互相污染 |
| **TP rank** | 张量并行的 rank 编号，从 0 到 `tp_size - 1` |
| **KV stride** | 一个 KV block 中 K 段（或 V 段）的步长，用于从大缓冲区中定位单个 block 的数据 |
| **RDMA** | Remote Direct Memory Access，无需 CPU 参与的网络内存访问技术 |
| **RC QP** | Reliable Connected Queue Pair，RDMA 可靠连接队列对 |
| **MetaServer** | 跨节点的 block hash 注册中心，记录"哪个节点有哪些 block" |

---

## 4. 完整数据流

### 4.1 Save 流程（GPU → CPU → SSD）

```
vLLM Worker
  │
  │ 1. RegisterContext (gRPC)
  │    传递 CUDA IPC handle（GPU 显存地址）、block 配置
  │
  ▼
pegaflow-server::GrpcEngineService
  │
  │ 2. 调用 PegaEngine::register_context_layer_batch()
  │    → 存储 GPU 显存指针到 CudaTensorRegistry
  │
  │ 3. Save RPC：传入 block_ids + block_hashes（每层）
  │
  ▼
PegaEngine::batch_save_kv_blocks_from_ipc()
  │
  │ 4. 通过 CUDA IPC 打开 GPU 显存（cudaIpcOpenMemHandle）
  │ 5. 在 CPU 侧分配 Pinned Memory（RAII: PinnedAllocation）
  │ 6. 用 CUDA 异步拷贝（cudaMemcpyAsync）从 GPU 拷贝到 CPU
  │
  ▼
StorageEngine::send_raw_insert()
  │
  │ 7. 通过 MPSC 通道发送到 insert worker 线程
  │
  ▼
insert_worker_loop() [专用 OS 线程]
  │
  │ 8. 按 BlockKey 聚合 InflightBlock（多 slot 场景等待所有 slot）
  │ 9. 所有 slot 到齐 → seal() → SealedBlock
  │ 10. 插入 ReadCache（TinyLFU 准入）
  │ 11. 异步写入 SsdBackingStore（via io_uring）
  │ 12. 向 MetaServer 注册 block hash（fire-and-forget）
```

### 4.2 Query + Load 流程（CPU → GPU）

```
vLLM Scheduler
  │
  │ 1. QueryPrefetch RPC（传入 block_hashes 前缀列表）
  │
  ▼
PegaEngine::count_prefix_hit_blocks_with_prefetch()
  │
  │ 2. 检查 ReadCache（内存命中？）
  │    命中：返回 PrefetchStatus::Done { hit, missing: 0 }
  │    部分命中：触发 SSD 预取
  │             返回 PrefetchStatus::Loading { hit, loading }
  │    未命中：触发 RDMA 远端预取
  │
  ▼ (调用方轮询直到 Done)
  │
  │ 3. Load RPC（传入 block_ids + block_hashes）
  │
  ▼
PegaEngine::batch_load_kv_blocks_multi_layer()
  │
  │ 4. consume_pinned_blocks()：取出已 pin 的 SealedBlock
  │ 5. 提交 LoadTask 到 GPU worker pool（fire-and-forget）
  │ 6. 通过共享内存 LoadState 通知 vLLM Worker 完成
  │
  ▼
GpuWorker（专用线程，绑定到 CUDA 设备）
  │
  │ 7. 从 SealedBlock 读取 K/V 指针
  │ 8. cudaMemcpyAsync 从 CPU pinned memory → GPU HBM
  │ 9. 更新 LoadState.completed
```

### 4.3 跨节点 RDMA 流程

```
节点 A（请求方）                     节点 B（提供方）
  │                                       │
  │ 1. QueryPrefetch → RDMA fetch 触发     │
  │                                       │
  │ 2. MetaServer 查询："block X 在哪？"   MetaServer
  │    ← 返回 node_B:50055               │
  │                                       │
  │ 3. RdmaHandshake RPC → node_B         │
  │    交换 QP 信息（GID/LID/QP_num）     │
  │    ← 返回 server 的 HandshakeMetadata │
  │                                       │
  │ 4. QueryBlocksForTransfer RPC → B     │
  │    ← 返回 k_ptr/v_ptr/rkey            │
  │    + 锁定 block（TransferLock）        │
  │                                       │
  │ 5. RDMA READ：直接读取 B 的内存       │
  │    （无需 B 的 CPU 参与）              │
  │                                       │
  │ 6. ReleaseTransferLock RPC → B        │
  │    释放锁，B 可以驱逐该 block          │
  │                                       │
  │ 7. 写入本地 ReadCache                 │
```

---

## 5. 目录结构速览

```
pegaflow/
├── Cargo.toml                  # Workspace 根配置（版本 0.17.0，edition 2024）
├── pegaflow-common/            # 公共工具库（日志、NUMA、HLL）
│   └── src/
│       ├── lib.rs
│       ├── logging.rs
│       ├── numa.rs
│       └── hll.rs
├── pegaflow-proto/             # Protobuf 协议定义
│   ├── proto/engine.proto      # 核心 RPC 接口定义
│   └── src/lib.rs
├── pegaflow-core/              # 核心存储引擎
│   ├── src/
│   │   ├── lib.rs              # PegaEngine 主入口
│   │   ├── block.rs            # Block 类型体系
│   │   ├── cache.rs            # TinyLFU 缓存
│   │   ├── allocator.rs        # 偏移量分配器
│   │   ├── pinned_mem.rs       # CUDA 固定内存封装
│   │   ├── pinned_pool.rs      # 内存池（NUMA 感知）
│   │   ├── gpu_worker.rs       # GPU 传输 worker
│   │   ├── transfer.rs         # CUDA memcpy 操作
│   │   ├── offload.rs          # Save 路径：GPU → CPU
│   │   ├── seal_offload.rs     # Save 路径：槽位元数据
│   │   ├── instance.rs         # 实例/GPU 上下文
│   │   ├── metrics.rs          # Prometheus 指标
│   │   ├── trace.rs            # 分布式追踪
│   │   ├── sync_state.rs       # LoadState（共享内存通知）
│   │   ├── storage/            # 存储子系统
│   │   │   ├── mod.rs          # StorageEngine 入口
│   │   │   ├── read_cache.rs   # 读缓存（pin/consume）
│   │   │   ├── write_path.rs   # 写路径 worker
│   │   │   ├── prefetch.rs     # 预取调度器
│   │   │   └── transfer_lock.rs # RDMA 传输锁
│   │   ├── backing/            # 后端存储（SSD/RDMA）
│   │   │   ├── ssd.rs          # SSD 协调器
│   │   │   ├── ssd_cache.rs    # SSD 块存储
│   │   │   ├── uring.rs        # io_uring 异步 I/O
│   │   │   ├── rdma.rs         # RDMA 传输封装
│   │   │   └── rdma_fetch.rs   # RDMA 拉取 store
│   │   └── internode/          # 跨节点通信
│   │       ├── metaserver_client.rs  # MetaServer 客户端
│   │       └── mod.rs
│   ├── benches/                # 性能基准测试
│   └── tests/                  # 集成测试
├── pegaflow-server/            # gRPC 服务进程
│   └── src/
│       ├── main.rs             # 服务入口
│       ├── service.rs          # RPC 实现
│       ├── registry.rs         # CUDA tensor 注册表
│       ├── http_server.rs      # HTTP 健康检查
│       ├── metric.rs           # 指标上报
│       └── bin/pegaflow-router.rs  # P/D 路由器
├── pegaflow-metaserver/        # MetaServer 进程
│   └── src/
│       ├── lib.rs, main.rs     # 入口
│       ├── service.rs          # gRPC 实现
│       └── store.rs            # Block hash LRU 存储
├── pegaflow-transfer/          # RDMA 传输库
│   └── src/
│       ├── engine.rs           # TransferEngine 公开 API
│       ├── rc_backend/         # RC 队列对后端
│       └── rdma_topo.rs        # NUMA/RDMA 拓扑检测
├── python/                     # Python 包
│   ├── src/lib.rs              # PyO3 Rust 绑定
│   └── pegaflow/
│       ├── connector/          # vLLM v1 连接器
│       │   ├── scheduler.py    # 调度器端逻辑
│       │   ├── worker.py       # Worker 端逻辑
│       │   └── state_manager.py
│       └── sglang/             # SGLang 集成
└── examples/                   # 使用示例
```

---

## 6. 关键设计决策

### 6.1 Split-Storage（K/V 分段存储）

vLLM 使用"层优先"布局：所有 K 连续，然后所有 V 连续：
```
[Layer0_K | Layer0_V | Layer1_K | Layer1_V | ...]
```

PegaFlow 的 Load 操作（CPU → GPU）需要将 K 和 V 分开批量拷贝，以最大化 PCIe 带宽利用率。因此每个 `RawBlock` 最多有 2 个 `Segment`（K 段 + V 段），而不是 1 个连续段。

### 6.2 内容寻址

所有 block 用内容哈希标识（由 vLLM/SGLang 计算并传入）：
- **去重**：相同 token 序列的 KV cache 只存一份
- **前缀语义**：vLLM 查询时传入前缀 hash 列表，PegaFlow 从头扫描到第一个 miss 停止

### 6.3 Pin 协议（防 LRU 驱逐）

Load 期间存在竞争：block 正在被复制到 GPU，同时可能被 LRU 驱逐。PegaFlow 用显式 pin 协议解决：
1. `QueryPrefetch` 命中时 → `pin_blocks()` 增加引用计数
2. 被 pin 的 block 不会被 LRU 驱逐
3. `Load` 完成后 → `consume_pinned_blocks()` 消费一次 pin
4. `Load` 取消时 → `unpin_blocks()` 释放 pin

### 6.4 TinyLFU 准入策略

新 block 进入缓存时，PegaFlow 用 TinyLFU（基于 Count-Min Sketch）比较候选 block 和当前 LRU 尾部的访问频率。如果候选 block 比 LRU 尾部更"冷"，则拒绝准入（防止扫描攻击）。

---

## 7. 版本信息

- 当前版本：`0.17.0`（Cargo.toml workspace.package.version）
- Rust edition：`2024`（最新稳定特性）
- 主要依赖：
  - `tokio 1.50`：异步运行时
  - `tonic 0.14` + `prost 0.14`：gRPC 框架
  - `cudarc 0.19.3`：CUDA 绑定（cuda-12040 特性）
  - `sideway 0.4.1`：RDMA 底层库
  - `pyo3 0.28`：Python 绑定
  - `axum 0.8`：HTTP 服务器
  - `moka`（metaserver 中）：高并发 LRU 缓存
