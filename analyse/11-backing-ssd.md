# 11 — SSD 后端（io_uring + SsdCache）

**核心文件**：
- `pegaflow-core/src/backing/ssd.rs`（234 行）— SSD 协调器
- `pegaflow-core/src/backing/ssd_cache.rs`（~600 行）— 环形缓冲区 + 写/读 worker
- `pegaflow-core/src/backing/uring.rs`（353 行）— io_uring 封装

---

## 1. SSD 后端的作用

内存层（固定内存）容量有限（通常 30-100 GB），而 SSD 层容量大（512 GB+）、成本低。SSD 后端提供**二级缓存**：

```
内存缓存（30-100 GB）
    ↕ 驱逐 / 预取
SSD 缓存（512 GB+，O_DIRECT，io_uring）
    ↕ 预取
磁盘（或远端节点）
```

写路径（Save → SSD）：block seal 后，insert worker 异步通知 SSD 写入  
读路径（SSD → 内存）：QueryPrefetch 触发，io_uring 异步读取到固定内存

---

## 2. SsdBackingStore 结构

```rust
// pegaflow-core/src/backing/ssd.rs:25
pub(crate) struct SsdBackingStore {
    _file: std::fs::File,          // 保持文件描述符存活（io_uring 需要）
    io: Arc<UringIoEngine>,        // io_uring 引擎（实际执行 I/O）
    write_tx: mpsc::Sender<SsdWriteBatch>,     // 写入队列（tokio MPSC）
    prefetch_tx: mpsc::Sender<PrefetchBatch>,  // 预取队列（tokio MPSC）
    inner: Mutex<SsdInner>,        // 包含 SsdRingBuffer（带锁）
    allocate_fn: AllocateFn,       // 用于分配固定内存（预取时使用）
    is_numa: bool,                 // 是否启用 NUMA 感知分配
}
```

### 初始化

```rust
// pegaflow-core/src/backing/ssd.rs:37
pub(super) fn new(config: SsdCacheConfig, allocate_fn: AllocateFn, is_numa: bool) -> io::Result<Arc<Self>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT)  // 重要：绕过系统页缓存
        .open(&config.cache_path)?;
    
    file.set_len(config.capacity_bytes)?;  // 预分配文件空间（避免碎片）
    
    let io = Arc::new(UringIoEngine::new(file.as_raw_fd(), UringConfig::default())?);
    
    // 写和预取各一个 tokio MPSC 通道
    let (write_tx, write_rx) = mpsc::channel(config.write_queue_depth);     // 默认深度 8
    let (prefetch_tx, prefetch_rx) = mpsc::channel(config.prefetch_queue_depth); // 默认深度 2
    
    // 启动两个 tokio worker task
    tokio::spawn(async move { ssd_writer_loop(store_weak, write_rx, io, write_inflight).await });
    tokio::spawn(async move { ssd_prefetch_loop(store_weak, prefetch_rx, io, capacity, prefetch_inflight).await });
    
    Ok(Arc::new(store))
}
```

**为什么使用 `O_DIRECT`？**

`O_DIRECT` 绕过操作系统的页缓存（Page Cache），数据直接在应用程序缓冲区（固定内存）和磁盘之间传输：

```
普通 I/O（经过页缓存）：
磁盘 → 内核页缓存 → 用户空间内存     （额外一次拷贝，额外内存压力）

O_DIRECT：
磁盘 → 用户空间固定内存              （零内核缓存，避免双重缓存）
```

PegaFlow 已有自己的缓存层（ReadCache），不需要内核重复缓存。

> **限制**：`O_DIRECT` 要求 I/O 大小和偏移量对齐到 512 字节（`SSD_ALIGNMENT = 512`）。这是为什么 KV block 大小需要 padding 到对齐值。

---

## 3. SsdRingBuffer — 环形缓冲区

SSD 文件被当作一个循环使用的**环形缓冲区**，新数据写入时自动覆盖最旧的数据（FIFO 淘汰）。

```rust
// pegaflow-core/src/backing/ssd_cache.rs:125
pub(super) struct SsdRingBuffer {
    capacity: u64,       // 环形缓冲区容量（字节）
    head: u64,           // 下一次写入的逻辑位置（单调递增）
    tail: u64,           // 最旧有效数据的逻辑位置
    order: VecDeque<BlockKey>,          // 按插入顺序排列的 key（FIFO 淘汰用）
    entries: HashMap<BlockKey, SsdEntryState>,  // 快速查找
}
```

**逻辑偏移 vs 物理偏移**：

```
逻辑偏移（head）：单调递增，永不回绕
物理偏移 = head % capacity

例：capacity = 100GB
  write block A: head = 0       → file_offset = 0
  write block B: head = 10GB    → file_offset = 10GB
  ...（写了 120GB 数据）
  write block X: head = 120GB   → file_offset = 20GB（物理回绕）
  
tail = head - capacity = 20GB
  → 逻辑偏移 < 20GB 的数据已被覆盖（过期）
```

### 两阶段提交

```
prepare_batch() → 分配空间，插入 Writing 状态
       │
       │ io_uring 异步写入
       ▼
commit_write(key, success=true) → Writing → Committed
       │
       └─ success=false → 删除条目（写入失败）
```

```rust
// 分配连续空间（处理环形回绕）
fn allocate_contiguous(&mut self, size: u64) -> (u64, u64) {
    let phys = self.head % self.capacity;
    let space_until_end = self.capacity - phys;
    
    if size > space_until_end {
        // 当前位置到文件末尾放不下 → 跳过剩余空间，从文件开头写
        self.head += space_until_end;
    }
    
    let begin = self.head;
    self.head += size;
    
    // 推进 tail（确保 head - tail <= capacity）
    let new_tail = self.head.saturating_sub(self.capacity);
    self.advance_tail(new_tail);  // 淘汰过期条目
    
    (begin, begin % self.capacity)  // 返回 (逻辑偏移, 物理偏移)
}
```

```rust
// 推进 tail，淘汰过期条目
fn advance_tail(&mut self, new_tail: u64) {
    self.tail = new_tail;
    
    // 从队列头部清理过期的 key
    while let Some(key) = self.order.front() {
        match self.entries.get(key) {
            Some(state) if state.begin() >= new_tail => break,  // 还在有效范围内
            _ => {
                let key = self.order.pop_front().unwrap();
                self.entries.remove(&key);  // 清理过期条目
            }
        }
    }
}
```

---

## 4. 写路径：ingest_batch → ssd_writer_loop

```rust
// fire-and-forget 写入
pub(crate) fn ingest_batch(&self, blocks: Vec<(BlockKey, Weak<SealedBlock>)>) {
    if blocks.is_empty() { return; }
    let len = blocks.len();
    let batch = SsdWriteBatch { blocks };
    
    if self.write_tx.try_send(batch).is_ok() {
        core_metrics().ssd_write_queue_pending.add(len as i64, &[]);
    } else {
        warn!("SSD write queue full, dropping {} blocks", len);
        core_metrics().ssd_write_queue_full.add(len as u64, &[]);
        // 队列满时直接丢弃！SSD 是尽力而为（best-effort）的
    }
}
```

`ssd_writer_loop`（在 tokio task 中运行）：
1. 从通道接收 `SsdWriteBatch`
2. 将 `Weak<SealedBlock>` 升级为 `Arc`（如果块已被 LRU 驱逐则跳过）
3. 调用 `prepare_batch()` 分配 SSD 空间
4. 构建 `iovec` 数组（block 的所有 segment）
5. 调用 `io.writev_at_async()` 提交 io_uring 写请求
6. 等待完成，调用 `commit_write(key, success)`

---

## 5. 读路径：submit_prefix → ssd_prefetch_loop

```rust
// 预取：前缀语义扫描 SSD 索引，提交读取
pub(crate) fn submit_prefix(
    &self,
    keys: Vec<BlockKey>,
) -> (usize, oneshot::Receiver<PrefetchResult>) {
    let (done_tx, done_rx) = oneshot::channel();
    
    // 前缀扫描 SSD 索引（遇 miss 停止）
    let requests: Vec<PrefetchRequest> = {
        let inner = self.inner.lock();
        keys.into_iter()
            .map_while(|key| {
                let entry = inner.ring.get(&key)?.clone();  // 只有 Committed 状态的才算
                Some(PrefetchRequest { key, entry })
            })
            .collect()
    };
    
    let found = requests.len();
    if found == 0 {
        let _ = done_tx.send(Vec::new());  // 立即返回空结果
        return (0, done_rx);
    }
    
    // 发送到预取队列
    let batch = PrefetchBatch { requests, done_tx };
    if let Err(e) = self.prefetch_tx.try_send(batch) {
        warn!("SSD prefetch queue full, dropping {} reads", e.into_inner().requests.len());
        // 队列满时：放弃预取，调用方会在 PrefetchScheduler 的下次轮询中重试
    }
    
    (found, done_rx)
}
```

`ssd_prefetch_loop`（在 tokio task 中运行）：
1. 从通道接收 `PrefetchBatch`
2. 为每个 block 分配固定内存（`allocate_prefetch()`）
3. 构建 `iovec` 读目标
4. 调用 `io.readv_at_async()` 提交 io_uring 读请求
5. 等待完成，重建 `SealedBlock`（用 `SlotMeta` 中的段大小信息）
6. 通过 `done_tx.send(results)` 通知 PrefetchScheduler

---

## 6. io_uring 引擎详解

### 什么是 io_uring？

`io_uring` 是 Linux 5.1+ 引入的异步 I/O 框架，通过**共享内存环形队列**实现零系统调用的批量 I/O：

```
应用程序                     内核
┌────────────────────┐     ┌────────────────────┐
│ 提交队列（SQ）      │────→│ 处理 I/O 请求       │
│ (Submission Queue) │     │                    │
│                    │     │                    │
│ 完成队列（CQ）      │←────│ 写入完成事件        │
│ (Completion Queue) │     │                    │
└────────────────────┘     └────────────────────┘
```

传统 `pread/pwrite` 每次都需要系统调用（用户态→内核态切换），`io_uring` 一次提交多个操作，大幅减少系统调用次数。

### UringIoEngine 结构

```rust
// pegaflow-core/src/backing/uring.rs:199
pub(super) struct UringIoEngine {
    txs: Vec<mpsc::SyncSender<IoCtx>>,   // 每个 shard 一个发送端
    handles: Vec<JoinHandle<()>>,          // 每个 shard 一个 OS 线程
}
```

每个 `UringShard` 在独立 OS 线程中运行，管理一个 `IoUring` 实例：

```rust
impl UringShard {
    fn run(mut self, fd: RawFd) {
        let mut inflight = 0usize;
        
        loop {
            // 填充提交队列：批量读取待处理的 IoCtx
            while inflight < self.io_depth {
                let ctx = if inflight == 0 {
                    self.rx.recv()     // 阻塞等待（无 I/O 时省 CPU）
                } else {
                    self.rx.try_recv() // 非阻塞取更多（已有 I/O 时批量处理）
                };
                
                // 构建 SQE（提交队列条目）
                let sqe = match ctx.io_type {
                    IoType::Readv  => opcode::Readv::new(fd, iovecs_ptr, count).offset(offset).build(),
                    IoType::Writev => opcode::Writev::new(fd, iovecs_ptr, count).offset(offset).build(),
                };
                
                // 将 IoCtx 的所有权通过 Box::into_raw 传入 SQE 的 user_data
                let data = Box::into_raw(Box::new(ctx)) as u64;
                unsafe { self.uring.submission().push(&sqe.user_data(data)) };
                inflight += 1;
            }
            
            // 等待至少一个完成（系统调用）
            self.uring.submit_and_wait(1);
            
            // 处理完成队列
            for cqe in self.uring.completion() {
                let ctx = unsafe { Box::from_raw(cqe.user_data() as *mut IoCtx) };
                let result = if cqe.result() < 0 {
                    Err(io::Error::from_raw_os_error(-cqe.result()))
                } else {
                    Ok(cqe.result() as usize)
                };
                let _ = ctx.complete.send(result);  // 通过 oneshot 通知调用者
            }
        }
    }
}
```

### 关键 unsafe 说明

```rust
// 为什么 IoCtx 需要 unsafe impl Send？
unsafe impl Send for IoCtx {}
// IoCtx 包含裸指针（iovecs 中的 *mut u8）
// 这些指针指向固定内存（PinnedAllocation），物理地址不会变化
// 在 io_uring 操作期间，caller 保证内存不会被释放（通过 Arc 或 blocking）

// 为什么用 Box::into_raw 传递 IoCtx？
let data = Box::into_raw(Box::new(ctx)) as u64;
// SQE 的 user_data 是 u64，用来在完成时找回对应的 IoCtx
// Box::into_raw 将所有权"泄漏"给内核，完成时 Box::from_raw 回收
// 如果 push 失败，需要手动 Box::from_raw 避免内存泄漏
let push_result = unsafe { self.uring.submission().push(&sqe) };
if push_result.is_err() {
    let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };  // 防止泄漏
    ...
}
```

### 分片（Shard）设计

```rust
fn pick_tx(&self, offset: u64) -> &mpsc::SyncSender<IoCtx> {
    let idx = (offset as usize / 4096) % self.txs.len();
    &self.txs[idx]  // 按文件偏移选择 shard
}
```

多个 shard 并行处理不同偏移区域的 I/O，提高并发度。默认 `threads = 1`（单 shard），高吞吐场景可配置多 shard。

---

## 7. 性能关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `write_queue_depth` | 8 | 写队列深度（满时丢弃新块）|
| `prefetch_queue_depth` | 2 | 预取队列深度（低值 = 低尾延迟）|
| `write_inflight` | 2 | 最大并发写 I/O 数 |
| `prefetch_inflight` | 16 | 最大并发读 I/O 数 |
| `io_depth` | 128 | io_uring 队列深度 |
| `SSD_ALIGNMENT` | 512 | O_DIRECT 对齐要求 |

**预取并发 > 写并发的原因**：读取路径在请求的关键路径上（影响推理延迟），需要更高并发来减少等待时间。写入是异步 fire-and-forget，不在关键路径上。

---

## 8. SSD 的 SlotMeta：重建 SealedBlock

SSD 存储的是原始内存数据（各 slot 的 segment 字节流），读取时需要知道如何重建 `SealedBlock`。

```rust
// SlotMeta：SSD 索引中记录每个 slot 的元数据
pub struct SlotMeta {
    segment_sizes: SmallVec<[u64; 2]>,  // 各 segment 的大小（1或2段）
    numa_node: NumaNode,                  // slot 的 NUMA 亲和性
}
```

预取完成时，用 `SlotMeta` 中的 `segment_sizes` 来切分读取的内存，重建 `SealedBlock { slots: [RawBlock { segments: [...] }] }`。

---

## 9. 完整数据流图

```
写路径（Save → SSD）：

insert_worker  →  ingest_batch(Weak<SealedBlock>)
                        │
                        ▼ tokio MPSC
              ssd_writer_loop (tokio task)
                        │
                        ├── Weak::upgrade() → 获取 Arc<SealedBlock>
                        ├── prepare_batch() → 分配环形缓冲区空间
                        ├── build iovecs （segment 指针 + 大小）
                        └── writev_at_async() → io_uring SQE
                                    │
                                    ▼ io_uring 完成
                        commit_write(key, true)  → Writing → Committed


读路径（SSD → 内存）：

QueryPrefetch → PrefetchScheduler
                        │
                        ▼
              ssd.submit_prefix(keys)
                        │
                        ├── 扫描 SsdRingBuffer.get(key)（只有 Committed 才算）
                        └── PrefetchBatch → prefetch_tx
                                    │
                                    ▼ tokio MPSC
              ssd_prefetch_loop (tokio task)
                        │
                        ├── allocate_prefetch() → Arc<PinnedAllocation>
                        ├── build iovecs（目标内存指针）
                        └── readv_at_async() → io_uring SQE
                                    │
                                    ▼ io_uring 完成
                        重建 SealedBlock（用 SlotMeta）
                        done_tx.send(results)
                                    │
                                    ▼
              PrefetchScheduler::poll_existing()
                        │
                        └── ReadCache::batch_insert()
```
