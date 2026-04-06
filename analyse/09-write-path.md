# 09 — 写路径：WritePipeline + Insert Worker

**核心文件**：`pegaflow-core/src/storage/write_path.rs`（454 行）

---

## 1. 写路径概览

Save 请求的完整路径：

```
vLLM Worker（GPU 端）
        │
        │ [Save RPC]
        ▼
GrpcEngineService::save()                     （gRPC 线程）
        │
        │ batch_save_kv_blocks_from_ipc()
        ▼
offload.rs: GPU → CPU cudaMemcpyAsync()       （GPU worker 线程）
        │
        │ RawSaveBatch（CPU 内存中的原始数据）
        ▼
WritePipeline::send_raw_insert()              （MPSC 发送，非阻塞）
        │
        │ ───── MPSC 通道 ─────
        ▼
insert_worker_loop()                          （专用 OS 线程）
        │
        ├─→ 组装 InflightBlock（多 TP slot 等待）
        ├─→ seal() → SealedBlock（所有 slot 完成）
        ├─→ ReadCache::batch_insert()（放入内存缓存）
        ├─→ SsdBackingStore::ingest_batch()（写入 SSD，可选）
        └─→ MetaServerClient::try_register()（注册到 MetaServer，可选）
```

---

## 2. WritePipeline — 写入管道

```rust
// pegaflow-core/src/storage/write_path.rs:25
pub(super) struct WritePipeline {
    insert_tx: Sender<InsertWorkerCommand>,
    // 标准库 MPSC 通道的发送端
    // MPSC = Multi-Producer Single-Consumer
    // 多个 gRPC handler（多线程）可以同时发送，insert worker 单线程接收
}
```

### 命令类型

```rust
// pegaflow-core/src/storage/write_path.rs:17
pub(super) enum InsertWorkerCommand {
    // 主命令：插入一批 KV 数据
    RawInsert(crate::offload::RawSaveBatch),
    
    // 维护命令：GC（清理超时的 inflight blocks）
    Gc {
        max_age: std::time::Duration,  // 超过此时间的 inflight block 将被清理
        reply: oneshot::Sender<usize>, // 返回清理了多少个
    },
}
```

> **Rust 新手提示**：`oneshot::Sender` 是 tokio 的一次性通道。与普通通道不同，oneshot 只能发送一条消息，适合"请求-响应"模式。这里 `Gc` 命令需要知道清理了多少个 block，所以携带了一个 reply 通道。

### 发送接口

```rust
// 非阻塞发送（save 路径）：
pub(super) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
    let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    // 忽略发送错误（insert worker 已退出时 send 会失败）
}

// 异步 GC（等待结果）：
pub(super) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
    let (reply_tx, reply_rx) = oneshot::channel();
    if self.insert_tx.send(InsertWorkerCommand::Gc { max_age, reply: reply_tx }).is_err() {
        return 0;
    }
    reply_rx.await.unwrap_or(0)  // 等待 insert worker 处理完毕
}
```

**为什么 SaveRPC 是非阻塞的？**

vLLM Worker 完成前向计算后立即发起 Save，不能阻塞等待存储完成（否则影响推理吞吐）。MPSC 通道让 Save 路径完全异步，insert worker 在后台处理。

---

## 3. InsertDeps — 依赖注入

```rust
// pegaflow-core/src/storage/write_path.rs:55
pub(super) struct InsertDeps {
    pub(super) read_cache: Arc<ReadCache>,               // 内存缓存
    pub(super) ssd_store: Option<Arc<SsdBackingStore>>,  // SSD 后端（可选）
    pub(super) metaserver_client: Option<Arc<MetaServerClient>>, // MetaServer（可选）
}
```

`InsertDeps` 将 insert worker 依赖的资源打包，通过 `Weak<InsertDeps>` 传入 worker 线程。

**为什么用 `Weak` 而不是 `Arc`？**

```
如果用 Arc<InsertDeps>：
  insert_worker_thread 持有 Arc<InsertDeps>
  InsertDeps 持有 Arc<ReadCache>
  ReadCache 被 StorageEngine 持有
  → 循环引用：StorageEngine 永不销毁（因为 insert worker 持有引用）

如果用 Weak<InsertDeps>：
  StorageEngine 销毁时，Arc<InsertDeps> 引用计数归零
  → deps 销毁
  → Weak::upgrade() 返回 None
  → insert worker 自然退出
```

---

## 4. insert_worker_loop — 核心处理循环

```rust
// pegaflow-core/src/storage/write_path.rs:61
pub(super) fn insert_worker_loop(rx: Receiver<InsertWorkerCommand>, deps: Weak<InsertDeps>) {
    let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();
    // ↑ 本地状态：记录尚未完成的 block（等待所有 TP slot）
    
    while let Ok(cmd) = rx.recv() {  // 阻塞等待下一条命令
        // 批量读取：有多条命令时一次性全取（减少锁竞争）
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {  // 非阻塞尝试取更多
            cmds.push(more);
        }
        
        for cmd in cmds {
            match cmd {
                InsertWorkerCommand::RawInsert(batch) => {
                    process_raw_save_batch(&mut inflight, &deps, batch);
                }
                InsertWorkerCommand::Gc { max_age, reply } => {
                    let cleaned = gc_inflight(&mut inflight, max_age);
                    let _ = reply.send(cleaned);  // 回复结果
                }
            }
        }
    }
    
    // rx.recv() 返回 Err 说明所有发送端已关闭（StorageEngine 已销毁）
    info!("Insert worker shutting down, {} inflight blocks remaining", inflight.len());
}
```

**批量读取的优化**：

`rx.recv()` 阻塞等待，但 `rx.try_recv()` 立即返回（无论是否有数据）。组合使用可以在有大量积压命令时，一次性取出所有待处理命令，减少循环开销。

---

## 5. process_raw_save_batch — 批次处理

```rust
// pegaflow-core/src/storage/write_path.rs:89
fn process_raw_save_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    batch: crate::offload::RawSaveBatch,
) {
    let start = std::time::Instant::now();
    let namespace = &batch.namespace;
    let numa_node = batch.numa_node;
    let total_slots = batch.total_slots;  // 该 namespace 的 TP 大小
    
    // 从 GPU offload 数据中解包出插入条目
    let (entries, total_bytes, total_blocks) = crate::offload::build_insert_entries(&batch);
    
    process_insert_batch(inflight, deps, entries, total_slots, numa_node, namespace);
    
    debug!("insert_worker: batch sealed blocks={} bytes={} ms={:.2}", ...);
}
```

---

## 6. process_insert_batch — InflightBlock 组装

这是写路径的核心逻辑：

```rust
// pegaflow-core/src/storage/write_path.rs:111
fn process_insert_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    entries: InsertEntries,  // Vec<(BlockKey, Vec<(slot_id, Arc<RawBlock>)>)>
    total_slots: usize,      // TP 大小（期望的 slot 总数）
    numa_node: NumaNode,
    namespace: &str,
) {
    let mut sealed_blocks = Vec::new();

    for (key, slots) in entries {
        // 获取或创建 InflightBlock
        let inflight_block = match inflight.entry(key.clone()) {
            Entry::Vacant(v) => v.insert(InflightBlock::new(total_slots)),
            Entry::Occupied(o) => {
                let ib = o.into_mut();
                // 安全检查：slot 数量不一致说明配置有问题
                if ib.total_slots() != total_slots {
                    error!("insert worker: slot count mismatch ...");
                    continue;
                }
                ib
            }
        };

        let mut completed = false;
        for (slot_id, block) in slots {
            match inflight_block.insert_slot(slot_id, block, numa_node) {
                SlotInsertResult::Inserted { completed: c, footprint_added } => {
                    // 更新 inflight_bytes 指标
                    inflight_bytes_added += footprint_added;
                    completed = c;
                    if completed { break; }  // 所有 slot 都填充完毕，可以 seal
                }
                SlotInsertResult::Duplicate => {}  // 重复插入，忽略
            }
        }

        if completed {
            let inflight_block = inflight.remove(&key).expect("just inserted");
            let total_footprint = inflight_block.footprint();
            inflight_bytes_removed += total_footprint;
            
            let sealed = Arc::new(inflight_block.seal());  // InflightBlock → SealedBlock

            // 立即插入到内存缓存
            if let Some(deps) = deps.upgrade() {
                deps.read_cache.batch_insert(vec![(key.clone(), Arc::clone(&sealed))]);
            }

            sealed_blocks.push((key, sealed));
        }
    }
    
    // 更新指标
    // ...
    
    // 将 sealed blocks 发送给 SSD 和 MetaServer
    if !sealed_blocks.is_empty() && let Some(deps) = deps.upgrade() {
        send_backing_batches(&deps, &sealed_blocks);
    }
}
```

### InflightBlock 状态机（单个 block 的生命周期）

```
TP=3 的场景（一个 block 需要 3 个 GPU 的 slot 数据）：

[第一批：来自 GPU 0，slot_id=0]
inflight: {key → InflightBlock { slots: [Some(block0), None, None], remaining: 2 }}

[第二批：来自 GPU 1，slot_id=1]
inflight: {key → InflightBlock { slots: [Some(block0), Some(block1), None], remaining: 1 }}

[第三批：来自 GPU 2，slot_id=2]
→ remaining = 0 → completed!
→ inflight.remove(key)
→ inflight_block.seal()
→ SealedBlock { slots: [block0, block1, block2] }
→ ReadCache::batch_insert()
```

---

## 7. send_backing_batches — 同步到 SSD 和 MetaServer

```rust
// pegaflow-core/src/storage/write_path.rs:191
fn send_backing_batches(deps: &InsertDeps, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
    if blocks.is_empty() { return; }
    
    // 转换为 Weak 引用（SSD 层只需要异步写入，不需要阻止内存释放）
    let weak_blocks: Vec<(BlockKey, Weak<SealedBlock>)> = blocks
        .iter()
        .map(|(k, b)| (k.clone(), Arc::downgrade(b)))
        .collect();
    
    // 发送给 SSD 存储（异步 io_uring 写入）
    if let Some(ssd) = &deps.ssd_store {
        ssd.ingest_batch(weak_blocks);
    }
    
    // 注册到 MetaServer（跨节点 block 发现）
    if let Some(client) = &deps.metaserver_client {
        register_block_hashes(client, blocks);
    }
}
```

**为什么传 `Weak<SealedBlock>` 给 SSD？**

SSD 写入是异步的（io_uring），写入期间 block 可能已被 LRU 驱逐（内存释放）。`Weak<SealedBlock>` 允许在 SSD 实际执行 I/O 时检查 block 是否还存在：

```rust
// 在 SSD 写入执行时：
if let Some(block) = weak_block.upgrade() {
    // block 还活着，执行写入
    io_uring_write(block.data(), ...);
} else {
    // block 已被 LRU 驱逐，跳过写入
    // （内存中没有了，写入 SSD 也没意义）
}
```

**MetaServer 注册：fire-and-forget**

```rust
fn register_block_hashes(client: &MetaServerClient, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
    let entries: Vec<(String, Vec<u8>)> = blocks
        .iter()
        .map(|(key, _)| (key.namespace.clone(), key.hash.clone()))
        .collect();
    client.try_register(entries);  // 非阻塞，放入异步队列
}
```

`try_register` 将注册请求放入 MetaServer 客户端的异步队列，不阻塞 insert worker。即使 MetaServer 暂时不可用，也不影响本地的内存/SSD 存储。

---

## 8. GC 机制 — 清理超时 InflightBlock

```rust
// pegaflow-core/src/storage/write_path.rs:215
fn gc_inflight(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    max_age: std::time::Duration,
) -> usize {
    let before = inflight.len();
    
    inflight.retain(|key, block| {
        let age = block.age();  // Instant::now() - created_at
        if age > max_age {
            warn!(
                "GC: removing stale inflight block: namespace={} hash_len={} \
                 filled={}/{} age_secs={}",
                key.namespace, key.hash.len(),
                block.filled_count(), block.total_slots(),
                age.as_secs()
            );
            core_metrics().inflight_bytes.add(-(block.footprint() as i64), &[]);
            false  // retain 返回 false → 从 HashMap 中移除
        } else {
            true   // 保留
        }
    });
    
    let cleaned = before - inflight.len();
    if cleaned > 0 {
        core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
        info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
    }
    cleaned
}
```

**为什么需要 GC？**

考虑这个场景：TP=4 的部署，其中 GPU 2 崩溃重启：

```
期望：4 个 slot 数据全部到达 → seal
实际：slot 0, 1, 3 到达，slot 2 永远不会来（GPU 崩溃）

结果：InflightBlock 永远卡在 remaining=1，占用内存资源
```

GC 定期清理超时的 InflightBlock（默认超过 120 秒），释放内存。

---

## 9. 线程安全保障

`insert_worker_loop` 在专用 OS 线程中运行，`inflight` HashMap 是该线程的**私有**状态，不需要加锁。

```
┌──────────────────────────────────────────────────────────────────┐
│ gRPC 线程 1 ──→ insert_tx.send()  ┐                              │
│ gRPC 线程 2 ──→ insert_tx.send()  ├──→ MPSC 通道 → insert worker │
│ gRPC 线程 3 ──→ insert_tx.send()  ┘     │                        │
│                                         │ (单线程处理)            │
│                                         ▼                        │
│                             HashMap<BlockKey, InflightBlock>     │
│                             （无锁！只有 insert worker 访问）     │
└──────────────────────────────────────────────────────────────────┘
```

这种设计称为 **Actor 模式**：将可变状态封装在单个线程中，通过消息传递而非锁来保护状态。

---

## 10. 测试案例解读

```rust
// 测试：单 slot 立即 seal
#[tokio::test]
async fn single_slot_seals_immediately() {
    // TP=1，只有一个 slot，insert 后立即 seal
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block)])];
    let mut inflight = HashMap::new();
    
    process_insert_batch(&mut inflight, &weak_deps, entries, 1, ...);
    
    assert!(inflight.is_empty(), "block should have been sealed");
    assert!(engine.read_cache.contains_keys(...)[0], "in cache");
}

// 测试：多 slot 需要多次插入才能 seal
#[tokio::test]
async fn multi_slot_partial_then_complete() {
    // TP=3，需要 3 次插入
    // 第 1 次：slot 0 → inflight.len() == 1，缓存中没有
    // 第 2 次：slot 1 → 仍然 inflight
    // 第 3 次：slot 2 → sealed！缓存中有了
}

// 测试：重复插入同一 slot 是幂等的
#[tokio::test]
async fn duplicate_slot_is_idempotent() {
    // 同一 slot 插入两次，只有第一次生效
    let entries = vec![(key.clone(), vec![(0, block_a), (0, block_b)])];
    process_insert_batch(..., 2, ...);
    
    // TP=2，只有 slot 0 填充了（slot 1 还没有）
    assert_eq!(inflight_block.filled_count(), 1);
}

// 测试：GC 清理超时的 inflight block
#[tokio::test]
async fn gc_inflight_removes_old_blocks() {
    inflight.insert(key, InflightBlock::new(2));
    
    // max_age=60s，刚创建的 block 不会被清理
    let cleaned = gc_inflight(&mut inflight, Duration::from_secs(60));
    assert_eq!(cleaned, 0);
    
    // max_age=0，所有 block 都超时
    let cleaned = gc_inflight(&mut inflight, Duration::ZERO);
    assert_eq!(cleaned, 1);
    assert!(inflight.is_empty());
}
```

---

## 11. 性能特性

| 特性 | 实现方式 | 效果 |
|------|---------|------|
| 非阻塞发送 | MPSC 通道 | Save RPC 立即返回，不等待磁盘 I/O |
| 批量处理 | 一次性取所有积压命令 | 减少循环开销，提高吞吐 |
| 无锁 inflight | Actor 模式（单线程状态）| 避免锁竞争 |
| SSD 异步 I/O | io_uring + Weak 引用 | 不阻塞 insert worker |
| MetaServer 异步注册 | fire-and-forget 队列 | 不影响本地写入路径 |
