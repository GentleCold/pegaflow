# 12 — RDMA 传输引擎

**核心文件**：
- `pegaflow-transfer/src/engine.rs`（264 行）— 传输引擎公开 API
- `pegaflow-core/src/backing/rdma.rs`（85 行）— RDMA 传输包装器
- `pegaflow-core/src/backing/rdma_fetch.rs`（535 行）— RDMA 远端拉取

---

## 1. RDMA 基本概念

**什么是 RDMA？**

RDMA（Remote Direct Memory Access，远程直接内存访问）允许计算机直接访问另一台机器的内存，**绕过 CPU 和操作系统**：

```
传统网络数据传输：
节点 A            网络              节点 B
└── CPU 读内存 → 操作系统 → 网卡 → 网络 → 网卡 → 操作系统 → CPU 写内存

RDMA 数据传输：
节点 A            高速网络（InfiniBand/RoCE）    节点 B
└── RDMA NIC 直接读取节点 B 内存 → 写入节点 A 内存（CPU 不参与！）
```

**核心优势**：
- 极低延迟（微秒级）
- 高带宽（100 Gbps+）
- CPU 利用率极低

**关键术语**：
| 术语 | 含义 |
|------|------|
| **NIC** | RDMA 网卡（如 Mellanox mlx5）|
| **QP（Queue Pair）** | 用于通信的队列对（发送队列 + 接收队列）|
| **RC QP** | Reliable Connected QP，可靠连接型 |
| **GID** | 全局标识符（类似 IP 地址，128位）|
| **LID** | 本地标识符（InfiniBand 的层 2 地址）|
| **PSN** | 包序列号（用于可靠传输）|
| **rkey** | 远端键（允许对方访问注册内存区域的令牌）|
| **MR** | 内存区域（Memory Region，已向 NIC 注册的内存）|

---

## 2. TransferEngine — 传输引擎

```rust
// pegaflow-transfer/src/engine.rs:103
pub struct TransferEngine {
    backend: RcBackend,  // RC QP 后端（实际的 RDMA 实现）
}
```

### 内存注册

RDMA NIC 必须预先知道哪些内存可以被远端访问。这叫做"注册内存"：

```rust
pub fn register_memory(&self, regions: &[MemoryRegion]) -> Result<()> {
    for region in regions {
        // 调用 ibv_reg_mr() 向 NIC 注册内存区域
        // NIC 获得物理地址，生成 lkey（本地键）和 rkey（远端键）
        self.backend.register_memory(region.ptr, region.len)?;
    }
    Ok(())
}
```

PegaFlow 在初始化时注册所有固定内存（通过 `RdmaTransport::new()`）：

```rust
// pegaflow-core/src/backing/rdma.rs:32
fn new(nic_names: &[String], allocator: &PinnedAllocator) -> Result<Self, String> {
    let engine = TransferEngine::new(nic_names)?;
    
    // 获取所有 pinned memory pool 的 (base_ptr, size)
    let regions: Vec<(NonNull<u8>, usize)> = allocator.memory_regions();
    let mr_descs: Vec<MemoryRegion> = regions.iter()
        .map(|&(ptr, len)| MemoryRegion { ptr, len })
        .collect();
    
    // 注册给 RDMA NIC
    engine.register_memory(&mr_descs)?;
    
    Ok(Self { engine, registered_ptrs: ... })
}
```

---

## 3. 握手（Handshake）流程

RC QP 连接需要双方交换 QP 元数据才能建立可靠连接：

```
节点 A（客户端）                      节点 B（服务端）
    │                                      │
    │ engine.get_or_prepare("node-B")      │
    │ → 创建本地 QP，生成 HandshakeMetadata │
    │   { gid, lid, qp_num, psn, memory_regions } │
    │                                      │
    │──── gRPC RdmaHandshake(local_meta) ──→│
    │                                      │ 服务端创建对应 QP
    │←─── 返回 server_meta ────────────────│
    │                                      │
    │ engine.complete_handshake(           │
    │   "node-B", local_meta, server_meta) │
    │ → 将本地 QP 连接到远端 QP           │
    │   QP 状态：INIT → RTR → RTS         │
    │                                      │
    │ 连接建立！                            │
```

### ConnectionStatus 状态机

```rust
// pegaflow-transfer/src/engine.rs:93
pub enum ConnectionStatus {
    Existing,            // 已连接，直接发起 RDMA 传输
    Connecting,          // 握手正在进行（并发情况）
    Prepared(HandshakeMetadata),  // 未连接，准备好了本地 QP，需要通过 gRPC 交换元数据
}

pub fn get_or_prepare(&self, remote_addr: &str) -> Result<ConnectionStatus> {
    match self.backend.get_or_prepare(remote_addr)? {
        GetOrPrepareResult::Existing       => Ok(ConnectionStatus::Existing),
        GetOrPrepareResult::AlreadyConnecting => Ok(ConnectionStatus::Connecting),
        GetOrPrepareResult::NeedHandshake(nics) => Ok(ConnectionStatus::Prepared(...)),
    }
}
```

### HandshakeMetadata 序列化

```rust
// pegaflow-transfer/src/engine.rs:59
pub struct HandshakeMetadata {
    pub(crate) nics: Vec<NicHandshake>,  // 每个 NIC 一个 NicHandshake
}

pub(crate) struct NicHandshake {
    pub(crate) endpoint: RcEndpoint,              // QP 端点信息
    pub(crate) memory_regions: Vec<RegisteredMemoryRegion>,  // 内存区域（含 rkey）
}

pub(crate) struct RcEndpoint {
    pub(crate) gid: [u8; 16],  // 全局标识符（128位）
    pub(crate) lid: u16,        // 本地标识符（InfiniBand）
    pub(crate) qp_num: u32,     // QP 编号
    pub(crate) psn: u32,        // 初始包序列号
}
```

序列化使用 `bincode`（二进制格式，紧凑高效），通过 gRPC `handshake_metadata` 字段传输。

---

## 4. 批量 RDMA 传输

```rust
// pegaflow-transfer/src/engine.rs:175
pub fn batch_transfer_async(
    &self,
    op: TransferOp,      // Read 或 Write
    remote_addr: &str,   // 远端节点地址
    descs: &[TransferDesc],  // 传输描述符列表
) -> Result<Vec<mea::oneshot::Receiver<Result<usize>>>>
```

```rust
pub struct TransferDesc {
    pub local_ptr: NonNull<u8>,   // 本节点内存地址
    pub remote_ptr: NonNull<u8>,  // 远端内存地址（来自 TransferSlotInfo.k_ptr）
    pub len: usize,               // 传输字节数
}
```

**NUMA 感知分配**：`batch_transfer_async` 内部将传输分配给最近的 RDMA NIC（根据 `slot.numa_node` 选择同一 NUMA 节点的 NIC），避免跨 NUMA 访问降低带宽。

---

## 5. RdmaFetchStore — 远端拉取完整流程

```rust
// pegaflow-core/src/backing/rdma_fetch.rs:43
pub(crate) struct RdmaFetchStore {
    metaserver_client: Arc<MetaServerClient>,  // 查询块所在节点
    rdma_transport: Arc<RdmaTransport>,         // RDMA 引擎
    allocate_fn: AllocateFn,                    // 分配本地固定内存
    advertise_addr: String,                     // 本节点地址（用于 MetaServer 过滤自身）
    grpc_channels: Arc<DashMap<String, EngineClient<Channel>>>,  // gRPC 连接缓存
    connect_group: Arc<Group<String, ()>>,  // Singleflight：防止并发握手竞争
}
```

### submit_remote_fetch() 流程

```rust
pub(crate) async fn submit_remote_fetch(
    &self, namespace: &str, keys: Vec<BlockKey>
) -> (usize, oneshot::Receiver<PrefetchResult>) {
    // 1. 查询 MetaServer：哪些节点有这些 block？
    let node_blocks = self.metaserver_client.query(namespace, &hashes).await?;
    
    // 2. 选择最优节点（拥有最多目标 block 的非本节点）
    let best = node_blocks.iter()
        .filter(|nb| nb.node != self.advertise_addr)  // 排除自身
        .max_by_key(|nb| nb.block_hashes.len());       // 选最多的
    
    // 3. 在后台 tokio task 中执行 RDMA 传输
    tokio::spawn(async move {
        let result = rdma_fetch_task(...).await;
        done_tx.send(result);
    });
    
    (found, done_rx)
}
```

### rdma_fetch_task() — 完整 4 步流程

```rust
async fn rdma_fetch_task(...) -> PrefetchResult {
    // 步骤 1：确保 RDMA 连接已建立（Singleflight 防并发握手冲突）
    ensure_connected(connect_group, rdma, grpc_channels, remote_addr, advertise_addr).await?;
    
    // 步骤 2：gRPC QueryBlocksForTransfer（锁定远端 block，获取内存地址）
    let (client, response) = query_remote_blocks(...).await?;
    let transfer_session_id = response.transfer_session_id;
    
    // 步骤 3：RDMA READ 所有 block（并设置超时）
    let result = fetch_blocks_via_rdma(rdma, allocate_fn, namespace, remote_addr,
                                        &response.blocks, transfer_timeout).await?;
    
    // 步骤 4：释放传输锁（fire-and-forget）
    spawn_release_lock(client, transfer_session_id);
    
    result
}
```

### Singleflight — 防止并发握手冲突

```rust
// 场景：并发 10 个请求同时需要连接节点 B
// 没有 Singleflight：10 个并发握手 → 服务端创建 10 套 QP → 互相竞争 → 混乱
// 有 Singleflight：只有第一个请求执行握手，其余等待

connect_group.try_work(remote_addr.to_string(), async || {
    // 只有第一个 task 进入这里
    let local_meta = match rdma.engine().get_or_prepare(remote_addr) {
        Ok(ConnectionStatus::Existing) => return Ok(()),  // 已连接，立即返回
        Ok(ConnectionStatus::Prepared(m)) => m,
        _ => return Err(...),
    };
    
    // 通过 gRPC 交换握手元数据
    let response = client.rdma_handshake(RdmaHandshakeRequest {
        requester_id: advertise_addr.to_string(),
        handshake_metadata: local_meta.to_bytes(),
    }).await?;
    
    // 完成连接
    rdma.engine().complete_handshake(remote_addr, &local_meta, &remote_meta)
}).await
```

### fetch_blocks_via_rdma() — 核心 RDMA READ

```rust
async fn fetch_blocks_via_rdma(rdma, allocate_fn, namespace, remote_addr,
                                 blocks, transfer_timeout) {
    let mut all_descs: Vec<TransferDesc> = Vec::new();
    
    for block_info in blocks {
        for slot in &block_info.slots {
            let numa = NumaNode(slot.numa_node);
            
            // 分配本地固定内存（NUMA 亲和：选与远端 slot 相同 NUMA 节点的本地内存）
            // K 段
            if slot.k_size > 0 {
                let alloc = allocate_fn(slot.k_size, Some(numa))?;
                all_descs.push(TransferDesc {
                    local_ptr: alloc.as_non_null(),          // 本地目标
                    remote_ptr: NonNull::new(slot.k_ptr as *mut u8)?,  // 远端源
                    len: slot.k_size as usize,
                });
            }
            
            // V 段（如果是分段存储）
            if slot.v_size > 0 && slot.v_ptr != 0 {
                let alloc = allocate_fn(slot.v_size, Some(numa))?;
                all_descs.push(TransferDesc { ... });
            }
        }
    }
    
    // 提交所有 RDMA READ 操作（NUMA 感知，分配给最近的 NIC）
    let receivers = rdma.engine().batch_transfer_async(
        TransferOp::Read, remote_addr, &all_descs
    )?;
    
    // 等待所有 RDMA 操作完成（带超时）
    tokio::time::timeout(transfer_timeout, async {
        for rx in receivers {
            rx.await??;  // 等待每个 NIC 的传输完成
        }
    }).await??;
    
    // 用分配的内存构建 SealedBlock
    let sealed = Arc::new(SealedBlock::from_slots(slots));
    result.push((key, sealed));
}
```

> **重要**：`all_descs` 包含 `NonNull<u8>`（不可跨 `.await` 边界），所以用 `{}` 块将它限制在 `batch_transfer_async` 调用前就 drop，确保不违反 Rust 的 `!Send` 限制。

---

## 6. 传输超时与传输锁

```rust
// 服务端设置传输锁超时（默认 120 秒）
// 客户端必须在此时间内完成 RDMA 传输

fn transfer_timeout_from_server(lock_timeout_secs: u32) -> Duration {
    let server = Duration::from_secs(lock_timeout_secs as u64);
    // 留 60 秒安全边距（客户端比服务端锁早 60 秒超时）
    server.saturating_sub(LOCK_TIMEOUT_MARGIN).max(MIN_TRANSFER_TIMEOUT)
}

// 例：服务端 lock_timeout=120s
// 客户端 timeout = 120s - 60s = 60s
// 客户端 60s 内没完成 → 超时 → 服务端 120s 后强制释放锁
```

**释放传输锁（fire-and-forget）**：

```rust
fn spawn_release_lock(mut client: EngineClient<Channel>, transfer_session_id: String) {
    tokio::spawn(async move {
        // 不阻塞调用者，在后台释放锁
        client.release_transfer_lock(ReleaseTransferLockRequest {
            transfer_session_id,
        }).await;
    });
}
```

---

## 7. 完整跨节点传输时序

```
节点 A（请求方）                 MetaServer              节点 B（数据方）
    │                               │                         │
    │ [QueryPrefetch → RDMA miss]   │                         │
    │                               │                         │
    │──── query(namespace, hashes) ─→│                         │
    │←─── [node_blocks: {B: [h1,h2]}]│                         │
    │                               │                         │
    │ [选择节点 B]                  │                         │
    │                               │                         │
    │ [ensure_connected(B)]         │                         │
    │────── RdmaHandshake ─────────────────────────────────→ │
    │←───── server HandshakeMetadata ─────────────────────── │
    │ [complete_handshake: QP connected]                      │
    │                               │                         │
    │────── QueryBlocksForTransfer ────────────────────────→ │
    │       (namespace, [h1, h2], requester_id)               │
    │←───── {blocks: [{h1, slots:[k_ptr,v_ptr]}, ...]        │
    │       transfer_session_id: "sess-1"                     │
    │       lock_timeout_secs: 120}                           │
    │                               │                         │
    │ [allocate local pinned memory]                          │
    │                               │                         │
    │════════ RDMA READ (h1.k) ════════════════════════════→ │
    │════════ RDMA READ (h1.v) ════════════════════════════→ │
    │════════ RDMA READ (h2.k) ════════════════════════════→ │
    │                               │                         │
    │ [所有 RDMA 完成]               │                         │
    │                               │                         │
    │ [构建 SealedBlock]             │                         │
    │ [ReadCache::batch_insert()]   │                         │
    │                               │                         │
    │────── ReleaseTransferLock ───────────────────────────→ │
    │       (transfer_session_id: "sess-1")                   │
    │                               │                         B 可以 LRU 驱逐该块了
```

---

## 8. gRPC 连接复用

```rust
fn get_or_create_channel(cache: &DashMap<String, EngineClient<Channel>>, addr: &str) -> ... {
    if let Some(client) = cache.get(addr) {
        return Ok(client.clone());  // 复用！tonic Channel clone 是廉价的（引用计数）
    }
    
    let channel = Endpoint::from_shared(url)?
        .connect_timeout(Duration::from_secs(5))
        .connect_lazy();  // 延迟连接（首次使用时才建立）
    
    let client = EngineClient::new(channel);
    cache.insert(addr.to_string(), client.clone());
    Ok(client)
}
```

**DashMap**：无锁并发 HashMap（来自 `dashmap` crate），比 `Mutex<HashMap>` 性能更好，适合高并发读取场景。

---

## 9. RDMA 内存可见性

RDMA NIC 可以直接读取注册内存中的数据。PegaFlow 注册的是 CUDA **固定内存**（`cudaHostAlloc` 或 `mmap(MAP_HUGETLB)` 注册），这些内存：
1. 物理地址固定（不会被 OS 换出）
2. CPU 和 RDMA NIC 都可以访问
3. RDMA NIC 注册后获得 `rkey`，远端节点凭 `rkey` 读写

因此 `RdmaTransport` 在 `Drop` 时必须注销内存：

```rust
impl Drop for RdmaTransport {
    fn drop(&mut self) {
        if let Err(e) = self.engine.unregister_memory(&self.registered_ptrs) {
            error!("Failed to unregister RDMA memory regions: {e}");
        }
    }
}
```

**TransferLock 的必要性**：

RDMA READ 是异步的，从节点 A 发出 READ 到接收数据有时间差。这段时间内节点 B 的 block 不能被 LRU 驱逐（否则物理内存被释放/复用，RDMA READ 读到垃圾数据）。

TransferLock 通过持有 `Arc<SealedBlock>` 引用来防止内存释放：只要 TransferSession 存在，`Arc` 引用计数 > 1，PinnedAllocation 不会释放。
