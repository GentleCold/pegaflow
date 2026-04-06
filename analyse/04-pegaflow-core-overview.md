# 04 — pegaflow-core 总览

**文件位置**：`pegaflow-core/src/`  
**这是 PegaFlow 最核心的 crate**，包含所有存储逻辑、内存管理、GPU 传输和跨节点通信的实现。

---

## 1. 模块结构

```
pegaflow-core/src/
├── lib.rs              ← PegaEngine（对外接口）
├── block.rs            ← Block 类型体系（见文档 05）
├── cache.rs            ← TinyLFU 缓存（见文档 08）
├── allocator.rs        ← ScaledOffsetAllocator（底层位图分配器）
├── pinned_mem.rs       ← CUDA 固定内存封装（见文档 06）
├── pinned_pool.rs      ← 内存池管理（见文档 06）
├── gpu_worker.rs       ← GPU 传输 worker（见文档 13）
├── transfer.rs         ← CUDA memcpy 操作
├── offload.rs          ← Save 路径：GPU → CPU 数据复制
├── seal_offload.rs     ← Save 路径：槽位元数据构建
├── instance.rs         ← InstanceContext + GpuContext
├── metrics.rs          ← Prometheus 指标定义
├── trace.rs            ← 分布式追踪宏
├── sync_state.rs       ← LoadState（共享内存通知机制）
├── storage/            ← 存储子系统（见文档 07-10）
│   ├── mod.rs          ← StorageEngine
│   ├── read_cache.rs   ← ReadCache（pin/consume）
│   ├── write_path.rs   ← WritePipeline + insert worker
│   ├── prefetch.rs     ← PrefetchScheduler
│   └── transfer_lock.rs ← TransferLockManager
├── backing/            ← 后端存储
│   ├── mod.rs          ← 后端工厂函数（new_ssd, new_rdma）
│   ├── ssd.rs          ← SsdBackingStore
│   ├── ssd_cache.rs    ← SSD 块映射
│   ├── uring.rs        ← io_uring 异步 I/O
│   ├── rdma.rs         ← RdmaTransport 封装
│   └── rdma_fetch.rs   ← RdmaFetchStore
└── internode/          ← 跨节点通信
    ├── mod.rs
    └── metaserver_client.rs
```

---

## 2. PegaEngine：对外接口

`PegaEngine` 是 `pegaflow-core` 的**公开 API 入口**，所有 gRPC 处理器都通过它与存储层交互。

```rust
// pegaflow-core/src/lib.rs:133
pub struct PegaEngine {
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    // ↑ 实例注册表：instance_id → 实例上下文（含 GPU 信息）
    
    storage: Arc<StorageEngine>,
    // ↑ 存储引擎：内存池、缓存、SSD、RDMA 的统一入口
    
    topology: Arc<NumaTopology>,
    // ↑ GPU-NUMA 拓扑：确定每块 GPU 对应的 NUMA 节点
}
```

### 主要方法一览

| 方法 | 对应 gRPC | 功能描述 |
|------|-----------|---------|
| `register_context_layer_batch` | `RegisterContextBatch` | 注册 GPU 的 KV cache 层 |
| `batch_save_kv_blocks_from_ipc` | `Save` | GPU → CPU 数据卸载 |
| `batch_load_kv_blocks_multi_layer` | `Load` | CPU → GPU 数据恢复 |
| `count_prefix_hit_blocks` | `Query` | 纯内存命中查询 |
| `count_prefix_hit_blocks_with_prefetch` | `QueryPrefetch` | 带 SSD/RDMA 预取的查询 |
| `unpin_blocks` | `Unpin` | 释放 pin 引用（Load 取消） |
| `unregister_instance` | `UnregisterContext` | 清理实例资源 |
| `query_blocks_for_transfer` | `QueryBlocksForTransfer` | 跨节点：查块+加锁 |
| `release_transfer_lock` | `ReleaseTransferLock` | 跨节点：释放锁 |
| `rdma_accept_handshake` | `RdmaHandshake` | RDMA 握手（服务端） |

---

## 3. InstanceContext 与 GpuContext

```rust
// pegaflow-core/src/instance.rs
pub struct InstanceContext {
    instance_id: String,
    namespace: String,
    num_layers: usize,    // 总层数
    tp_size: usize,       // 张量并行度
    world_size: usize,    // 总 world size
    layer_ids: HashMap<String, usize>,  // layer_name → layer_id
    slot_map: Vec<Vec<usize>>,  // [layer_id][tp_rank] → slot_id
    gpus: RwLock<HashMap<i32, Arc<GpuContext>>>,  // device_id → GPU 上下文
}
```

**Slot 概念**：在 TP=4 的场景，同一个 block 有 4 份（每个 GPU 存一份）。`slot_map[layer_id][tp_rank]` 给出这个 GPU worker 在 `SealedBlock` 的 slots 数组中的索引位置。

```rust
pub struct GpuContext {
    device_id: i32,
    numa_node: NumaNode,
    registrations: HashMap<String, Arc<KVCacheRegistration>>,  // layer_name → 注册信息
    worker_pool: Arc<GpuWorkerPool>,  // 异步任务执行器
}

pub struct KVCacheRegistration {
    pub data_ptr: u64,           // GPU 缓冲区基地址
    pub size_bytes: usize,       // 总大小
    pub num_blocks: usize,       // 总块数
    pub bytes_per_block: usize,  // 每块原始字节数
    pub padded_bytes_per_block: usize, // SSD 对齐后的字节数
    pub kv_stride_bytes: usize,  // 块间步长（= bytes_per_block 或更大）
    pub segments: usize,         // 1（连续）或 2（K/V 分段）
}
```

---

## 4. 错误类型设计

```rust
// pegaflow-core/src/lib.rs:83
pub enum EngineError {
    InstanceMissing(String),      // instance_id 未找到
    WorkerMissing(String, i32),   // device_id 未找到
    InvalidArgument(String),      // 参数非法
    CudaInit(String),             // CUDA 初始化失败
    Storage(String),              // 存储层错误
    Poisoned(&'static str),       // 内部锁中毒（线程 panic）
    TopologyMismatch(String),     // 注册时拓扑不匹配
}
```

在 `pegaflow-server/src/service.rs` 中，这些错误被映射为 gRPC Status 码：

```rust
fn map_engine_error(err: EngineError) -> Status {
    match err {
        EngineError::InvalidArgument(_) => Status::invalid_argument(...),
        EngineError::InstanceMissing(_) | EngineError::WorkerMissing(_, _) => 
            Status::failed_precondition(...),
        // ...
    }
}
```

---

## 5. StorageConfig：完整配置项解读

```rust
// pegaflow-core/src/storage/mod.rs:31
pub struct StorageConfig {
    // TinyLFU 准入策略（默认开启）
    // 关闭后：所有块都准入缓存（适合小缓存，避免 CM-Sketch 开销）
    pub enable_lfu_admission: bool,

    // 缓存块大小提示（字节）
    // 影响 TinyLFU CM-Sketch 的大小估算（更精确的预算分配）
    pub hint_value_size_bytes: Option<usize>,

    // SSD 预取的最大并发块数（背压控制）
    // 防止 SSD 预取队列无限增长，默认 512
    pub max_prefetch_blocks: usize,

    // SSD 缓存配置（None = 不使用 SSD）
    pub ssd_cache_config: Option<SsdCacheConfig>,

    // RDMA NIC 名称列表（None = 不使用 RDMA）
    // 示例：Some(vec!["mlx5_0".to_string(), "mlx5_1".to_string()])
    pub rdma_nic_names: Option<Vec<String>>,

    // NUMA 感知内存分配（默认开启）
    // 开启时：每个 NUMA 节点分配独立内存池，GPU 数据放到对应节点
    pub enable_numa_affinity: bool,

    // 逐块分配模式（默认关闭）
    // 关闭（默认）：批量分配大块内存，内部用偏移分配器切分（性能好但有碎片）
    // 开启：每个 block 单独 mmap，碎片少但每次分配有系统调用开销
    pub blockwise_alloc: bool,

    // RDMA 传输锁超时（默认 120 秒）
    // 超时后，被锁定的 block 可以被 LRU 驱逐
    pub transfer_lock_timeout: Duration,

    // MetaServer 地址（None = 单节点模式，不使用 MetaServer）
    pub metaserver_addr: Option<String>,

    // 本节点对外暴露的地址（用于 MetaServer 注册和 RDMA 连接标识）
    pub advertise_addr: Option<String>,

    // MetaServer 注册队列深度（控制 fire-and-forget 队列大小）
    pub metaserver_queue_depth: usize,

    // 内存池分片数（减少多线程分配时的锁竞争）
    pub pool_shards: usize,
}
```

---

## 6. 初始化顺序

`PegaEngine::new_with_config()` 初始化顺序：

```
1. NumaTopology::detect()
   ├── 调用 nvidia-smi 获取 GPU-NUMA 映射
   └── 读取 /sys/devices/system/node/ 获取 NUMA 节点列表

2. StorageEngine::new_with_config()
   ├── 确定 NUMA 节点列表
   ├── PinnedAllocator 初始化
   │   ├── 全局模式：单个 PinnedMemoryPool
   │   └── NUMA 模式：每节点一个 PinnedMemoryPool
   ├── ReadCache 初始化（TinyLFU + LRU）
   ├── WritePipeline 初始化（创建 MPSC 通道）
   ├── RdmaTransport 初始化（注册 pinned memory 到 RDMA NIC）
   ├── SsdBackingStore 初始化（若配置了 SSD）
   ├── RdmaFetchStore 初始化（若配置了 MetaServer + RDMA）
   ├── PrefetchScheduler 初始化
   ├── TransferLockManager 初始化
   └── 启动 insert worker OS 线程

3. PegaEngine 就绪，等待 gRPC 请求
```

---

## 7. 关键并发模型

PegaFlow 使用以下几种并发机制：

```
┌─────────────────────────────────────────────────────────────┐
│                    线程模型                                   │
│                                                             │
│  Tokio 线程池（异步 I/O 密集型任务）                         │
│  ├── gRPC 服务处理线程（每个 RPC 一个 task）                 │
│  ├── SSD 预取 task                                          │
│  └── RDMA 远端拉取 task                                     │
│                                                             │
│  专用 OS 线程（CPU 密集型或阻塞型任务）                       │
│  ├── pegaflow-insert：insert worker（处理 Save 批次）        │
│  └── numa{N}-init：NUMA 节点内存初始化线程                   │
│                                                             │
│  GPU worker 线程（每块 GPU 一个，CUDA 上下文绑定）            │
│  └── 负责 GPU ↔ CPU 的 cudaMemcpyAsync                      │
└─────────────────────────────────────────────────────────────┘
```

数据在线程间通过以下方式共享：
- `Arc<T>`：共享不可变状态（分配器、缓存）
- `parking_lot::Mutex<T>`：共享可变状态（缓存内部）
- `std::sync::mpsc`：save 路径从 gRPC handler → insert worker
- `tokio::sync::oneshot`：insert worker → GC 请求者

---

## 8. metrics.rs — Prometheus 指标

```rust
// 主要指标（counter/gauge/histogram）：
pub struct CoreMetrics {
    // 缓存命中/缺失
    pub cache_block_hits: Counter,
    pub cache_block_misses: Counter,
    pub cache_block_insertions: Counter,
    pub cache_block_admission_rejections: Counter,  // TinyLFU 拒绝
    
    // 内存使用
    pub cache_resident_bytes: Gauge,
    pub inflight_bytes: Gauge,           // inflight block 占用
    pub pinned_for_load_unique_bytes: Gauge,
    
    // 驱逐
    pub cache_block_evictions: Counter,
    pub cache_block_evictions_still_referenced: Counter,
    pub cache_eviction_reclaimed_bytes: Counter,
    
    // 内存分配
    pub pool_alloc_failures: Counter,    // 内存耗尽
    
    // GC
    pub inflight_gc_cleaned: Counter,
}
```

指标通过 `pegaflow-server/src/http_server.rs` 暴露为 `/metrics` HTTP 接口（Prometheus 格式）。

---

## 9. trace.rs — 分布式追踪

```rust
// 使用 fastrace 库，支持 OpenTelemetry 导出
// 使用宏封装，feature flag 控制（默认关闭，避免生产开销）

#[macro_export]
macro_rules! trace_root {
    ($name:expr, $root:ident, || [$($key:expr => $val:expr),*]) => {
        // 创建 trace span 根节点
    };
}

#[macro_export]
macro_rules! trace_scope {
    ($name:expr, $span:ident) => {
        // 创建子 span，离开作用域时自动结束
    };
}
```

在关键路径（Save/Load/Query）的 gRPC handler 中使用，配合 Jaeger 等工具可以查看端到端请求延迟分布。

---

## 10. sync_state.rs — LoadState 异步通知

**场景**：Load RPC 是"fire-and-forget"的，gRPC handler 立即返回，实际的 GPU 传输在 GPU worker 线程异步执行。vLLM Worker 需要知道何时传输完成。

解决方案：**共享内存（POSIX shared memory）**

```rust
// pegaflow-core/src/sync_state.rs
pub struct LoadState {
    shm: shared_memory::Shmem,  // POSIX 共享内存对象
}

// PegaFlow（写入端）：
load_state.set_completed();  // 传输完成后设置 completed=true

// vLLM Worker（读取端，通过 Python 绑定）：
while not load_state.is_completed():
    time.sleep(0.001)  # 轮询（spin-wait）
```

共享内存名称由 vLLM Worker 在 Load RPC 的 `load_state_shm` 字段传入。这种方式避免了额外的 gRPC 回调，减少延迟。
