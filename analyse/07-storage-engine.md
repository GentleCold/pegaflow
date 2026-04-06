# 07 — StorageEngine：存储引擎总览

**核心文件**：`pegaflow-core/src/storage/mod.rs`（736 行）  
**作用**：管理 PegaFlow 所有存储相关子系统的协调者

---

## 1. StorageEngine 结构

```rust
// pegaflow-core/src/storage/mod.rs:78
pub(crate) struct StorageEngine {
    allocator: Arc<PinnedAllocator>,         // 固定内存分配器（NUMA 感知）
    read_cache: Arc<ReadCache>,              // 内存读缓存（LRU + TinyLFU）
    prefetch: PrefetchScheduler,             // 预取调度器（SSD + RDMA）
    write_pipeline: Arc<WritePipeline>,      // 写路径（MPSC → insert worker）
    ssd_store: Option<Arc<SsdBackingStore>>, // SSD 后端（可选）
    rdma_transport: Option<Arc<RdmaTransport>>, // RDMA 传输（可选）
    blockwise_alloc: bool,                   // 逐块分配模式开关
    metaserver_client: Option<Arc<MetaServerClient>>, // MetaServer 客户端（可选）
    transfer_lock: Arc<TransferLockManager>, // RDMA 传输锁管理器
}
```

**设计思路**：各子系统通过 `Arc` 共享，`StorageEngine` 是统一的访问入口，但各子系统之间耦合度低，可以独立启用/禁用。

---

## 2. 初始化流程（Arc::new_cyclic 技巧）

### 问题：循环引用

`StorageEngine` 中的 SSD、RDMA 后端需要调用 `StorageEngine::allocate()` 来分配内存。但 `StorageEngine` 尚未构建完成，如何让后端持有引擎的引用？

### 解决方案：`Arc::new_cyclic`

```rust
// pegaflow-core/src/storage/mod.rs:174
let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
    // 此时 Arc<Self> 尚不存在，只有 Weak 弱引用
    
    let alloc_weak = weak_engine.clone();
    let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
        alloc_weak
            .upgrade() // Weak → Option<Arc>
            .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
    });
    
    // 用 allocate_fn 创建 SSD/RDMA 后端
    let ssd_store = ssd_cache_config.and_then(|cfg| 
        crate::backing::new_ssd(cfg, allocate_fn.clone(), is_numa)
    );
    
    // 构建并返回 Self（此时 Arc 创建完成，Weak 可以 upgrade）
    Self { allocator, read_cache, ssd_store, ... }
});
```

> **Rust 新手提示**：`Arc::new_cyclic` 是标准库提供的特殊构造函数，专门解决 Arc 内部需要引用自身的场景。闭包接收的 `Weak<Self>` 弱引用此时无法 upgrade（Arc 尚未创建），但可以克隆保存，等 `new_cyclic` 返回后就可以 upgrade 了。

### 启动 insert worker 线程

```rust
// pegaflow-core/src/storage/mod.rs:222
std::thread::Builder::new()
    .name("pegaflow-insert".into())
    .spawn(move || {
        let _keep_alive = deps; // 持有 deps Arc，防止过早销毁
        write_path::insert_worker_loop(insert_rx, weak_deps);
    })
    .expect("failed to spawn insert worker thread");
```

`insert_worker_loop` 使用 `Weak<InsertDeps>` 而不是 `Arc<InsertDeps>`，原因：
- 如果用 `Arc`，insert worker 线程持有 `Arc<StorageEngine>`（通过 `read_cache`），StorageEngine 永远无法被销毁
- 用 `Weak`：StorageEngine 销毁时，`Weak::upgrade()` 返回 `None`，worker 自然退出

---

## 3. 内存分配与 LRU 回收

### allocate() — 带 LRU 驱逐的分配

```rust
// pegaflow-core/src/storage/mod.rs:257
pub(crate) fn allocate(
    &self,
    size: NonZeroU64,
    numa_node: Option<NumaNode>,
) -> Option<Arc<PinnedAllocation>> {
    let requested_bytes = size.get();
    
    loop {
        // 1. 先尝试直接分配
        if let Some(alloc) = self.allocator.allocate(size, node) {
            return Some(alloc);
        }
        
        // 2. 分配失败 → 驱逐 LRU 块
        let (freed_blocks, _freed_bytes, largest_free) =
            self.reclaim_until_allocator_can_allocate(requested_bytes);
        
        // 3. 如果无法驱逐更多，或驱逐后仍不够大，放弃
        if freed_blocks == 0 || largest_free < requested_bytes {
            break;
        }
        // 4. 继续尝试分配
    }
    
    // 真正耗尽：记录指标，返回 None
    core_metrics().pool_alloc_failures.add(1, &[]);
    None
}
```

### reclaim_until_allocator_can_allocate() — LRU 驱逐

```rust
// 每批驱逐 RECLAIM_BATCH_SIZE = 64 个块
fn reclaim_until_allocator_can_allocate(&self, required_bytes: u64) -> (usize, u64, u64) {
    let mut freed_blocks = 0;
    let mut freed_bytes = 0u64;
    
    while self.allocator.largest_free_allocation() < required_bytes {
        // 从 ReadCache 移除最旧的 64 个块（LRU 尾部）
        let evicted = self.read_cache.remove_lru_batch(RECLAIM_BATCH_SIZE);
        if evicted.is_empty() { break; } // 没有更多可驱逐的块
        
        // 检查仍被引用的块（正在 Load 的块不会立即释放内存）
        for (_key, block) in &evicted {
            if Arc::strong_count(block) > 1 {
                // 这个块还被 pinned_for_load 持有，内存不会立即释放
                core_metrics().cache_block_evictions_still_referenced.add(1, &[]);
            }
        }
        
        drop(evicted); // Arc 引用计数减一，如果降到 0 则内存释放
        freed_blocks += evicted.len();
    }
    
    (freed_blocks, freed_bytes, self.allocator.largest_free_allocation())
}
```

---

## 4. 写路径接口

```rust
// 接收来自 GPU worker 的 Save 批次：
pub(crate) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
    self.write_pipeline.send_raw_insert(batch);
    // 非阻塞：发送到 MPSC 通道，insert worker 线程异步处理
}
```

---

## 5. 读路径接口

### 纯内存查询（不触发预取）

```rust
// 前缀语义：遇到第一个 miss 就停止
pub(crate) fn check_prefix_memory_only(
    &self,
    namespace: &str,
    hashes: &[Vec<u8>],
) -> (usize, usize) { // (hit, missing)
    self.read_cache.check_prefix_memory_only(namespace, hashes)
}
```

### 带预取的查询

```rust
// 异步：可能触发 SSD/RDMA 预取，返回 Loading 状态
pub(crate) async fn check_prefix_and_prefetch(
    &self,
    instance_id: &str,
    req_id: &str,
    namespace: &str,
    hashes: &[Vec<u8>],
    num_workers: usize,
) -> PrefetchStatus {
    self.prefetch.check_and_prefetch(
        &self.read_cache, instance_id, req_id, namespace, hashes, num_workers,
    ).await
}
```

### Pin / Consume / Unpin 协议

```rust
// Load 时从 pinned 池取出 block（消耗一次 pin 引用）：
pub(crate) fn consume_pinned_blocks(
    &self,
    instance_id: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> Result<Vec<Arc<SealedBlock>>, String> {
    self.read_cache.consume_pinned_blocks(instance_id, namespace, block_hashes)
}

// Load 取消时释放 pin：
pub(crate) fn unpin_blocks(&self, ...) -> usize {
    self.read_cache.unpin_blocks(...)
}
```

---

## 6. 跨节点 RDMA 相关接口

```rust
// 查找特定 block（非前缀语义）：
pub(crate) fn get_blocks_for_transfer(&self, keys: &[BlockKey]) 
    -> Vec<(BlockKey, Arc<SealedBlock>)>

// 锁定 block 防止 LRU 驱逐（返回 session_id）：
pub(crate) fn lock_blocks_for_transfer(&self, ...) -> String

// 释放锁：
pub(crate) fn release_transfer_lock(&self, session_id: &str) -> usize

// 查询 GC 过期的锁：
pub(crate) fn gc_expired_transfer_locks(&self) -> usize

// 返回固定内存区域（用于 RDMA NIC 注册）：
pub(crate) fn pinned_memory_regions(&self) -> Vec<(u64, usize)>
```

---

## 7. transfer_lock.rs — RDMA 传输锁管理

**为什么需要传输锁？**

跨节点 RDMA 传输过程：
1. 节点 A 查询节点 B 的 block 位置（得到内存地址）
2. 节点 A 发起 RDMA READ（直接读 B 的内存）
3. 步骤 1-2 之间，B 可能 LRU 驱逐该 block → A 读到垃圾数据

传输锁防止在 RDMA 传输期间驱逐 block：

```rust
// pegaflow-core/src/storage/transfer_lock.rs（概念）
pub(crate) struct TransferLockManager {
    sessions: Mutex<HashMap<String, TransferSession>>,
    lock_timeout: Duration,
}

struct TransferSession {
    locked_blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    created_at: Instant,
    requester_id: String,
}
```

```rust
// 锁定：
pub(crate) fn lock_blocks(&self, requester_id: &str, blocks: Vec<...>) -> String {
    let session_id = uuid::Uuid::new_v4().to_string();
    let session = TransferSession { locked_blocks: blocks, ... };
    self.sessions.lock().insert(session_id.clone(), session);
    session_id
}

// 释放：
pub(crate) fn release(&self, session_id: &str) -> usize {
    // 移除 session → Arc<SealedBlock> 引用计数减一 → 可以被 LRU 驱逐
    self.sessions.lock().remove(session_id)
        .map(|s| s.locked_blocks.len())
        .unwrap_or(0)
}

// 超时 GC（防止客户端崩溃导致锁永不释放）：
pub(crate) fn gc_expired(&self) -> usize {
    let timeout = self.lock_timeout;
    let now = Instant::now();
    let mut sessions = self.sessions.lock();
    let before = sessions.len();
    sessions.retain(|_, s| now.duration_since(s.created_at) < timeout);
    before - sessions.len()
}
```

---

## 8. 子系统关系图

```
StorageEngine
│
├── PinnedAllocator
│   └── PinnedMemoryPool(s) [NUMA 感知 or 全局]
│       └── CUDA fixed memory (30GB)
│
├── ReadCache
│   ├── TinyLfuCache (LRU + LFU 准入)
│   └── pinned_for_load map (pin 引用计数)
│
├── PrefetchScheduler
│   ├── ssd_store → SsdBackingStore [可选]
│   └── rdma_fetch → RdmaFetchStore [可选]
│
├── WritePipeline
│   └── MPSC channel → insert_worker_thread
│       ├── ReadCache.batch_insert()
│       ├── SsdBackingStore.ingest_batch() [可选]
│       └── MetaServerClient.try_register() [可选]
│
├── RdmaTransport [可选]
│   └── TransferEngine (RC QP 连接池)
│
├── MetaServerClient [可选]
│   └── background registration_loop (tokio task)
│
└── TransferLockManager
    └── sessions map (UUID → locked blocks)
```

---

## 9. 测试中的辅助方法

`StorageEngine` 在 `#[cfg(test)]` 下提供了额外的测试辅助方法：

```rust
#[cfg(test)]
impl StorageEngine {
    // 直接插入 block 到内存缓存（绕过 GPU 拷贝流程）：
    pub(crate) fn test_insert_cache(&self, key: BlockKey, block: Arc<SealedBlock>)

    // 查询某个 block 的 pin 引用计数：
    pub(crate) fn test_pin_count(&self, instance_id: &str, key: &BlockKey) -> usize
}
```

这两个方法让单元测试可以在没有 GPU 的环境下测试缓存逻辑。
