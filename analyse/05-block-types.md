# 05 — Block 类型体系

**文件位置**：`pegaflow-core/src/block.rs`  
**代码行数**：约 370 行  
**核心作用**：定义 KV cache 数据的存储抽象层次

---

## 1. 类型层次全图

```
用途            类型               说明
──────────────────────────────────────────────────────────────
标识层    BlockKey           namespace + content_hash
          BlockHash          = Vec<u8>（哈希字节串）

存储层    Segment            单个连续内存片段（指针+大小+生命周期）
          RawBlock           Segment 列表（布局无关）
          InflightBlock      正在写入的 Block（有 N 个槽位，逐步填充）
          SealedBlock        所有槽位填充完毕的不可变 Block

视图层    LayerBlock         将 RawBlock 解释为 KV 层（K/V 分段访问）

状态层    BlockStatus        块在存储层次中的位置（内存/SSD/缺失）
          PrefetchStatus     QueryPrefetch 的结果状态机
          SlotInsertResult   插入槽位的结果（新插入/重复/是否完成）

输入层    LayerSave          Save RPC 的单层输入数据
```

---

## 2. BlockKey — 内容寻址键

```rust
// pegaflow-core/src/block.rs:21
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub namespace: String,   // 模型命名空间（如 "llama3-70b"）
    pub hash: Vec<u8>,       // 内容哈希（由 vLLM 计算，通常 32 字节 SHA-256）
}
```

**为什么用 `namespace` + `hash` 组合？**

不同模型（不同 `namespace`）的 KV cache 格式不同（不同的 head_dim、num_heads），即使 token 序列相同，也不能共用同一份 KV 数据。`namespace` 隔离确保数据正确性。

```rust
// 内存大小估算：
pub fn estimated_size(&self) -> u64 {
    (self.namespace.capacity() + self.hash.capacity() + 48) as u64
    // 48 字节 = 两个 Vec 的 metadata（ptr + len + cap）+ 结构体 padding
}
```

> **Rust 新手提示**：`#[derive(Hash)]` 让 Rust 自动实现哈希函数，使 `BlockKey` 可以用作 `HashMap` 的键。`PartialEq + Eq` 是 Hash 的前提（相等的值必须有相同的哈希）。

---

## 3. Segment — 最底层内存片段

```rust
// pegaflow-core/src/block.rs:84
pub(crate) struct Segment {
    ptr: NonNull<u8>,                    // 指向固定内存的裸指针
    size: usize,                          // 片段字节数
    _allocation: Arc<PinnedAllocation>,   // RAII：持有分配对象防止释放
}
```

**生命周期管理**：

`_allocation` 字段用下划线前缀表示"不使用但需要持有"。`Arc<PinnedAllocation>` 的引用计数确保：
- 只要有 `Segment` 存在，底层 CUDA 固定内存就不会被释放
- 当 `Segment` 被 drop 时，`Arc` 引用计数减一
- 当所有持有该 `PinnedAllocation` 的 `Segment` 都被 drop，内存才真正释放

```rust
// SAFETY 注释解释了为什么手动实现 Send/Sync 是安全的：
// CUDA pinned memory 物理地址固定，线程安全
unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}
```

---

## 4. RawBlock — 存储层块（布局无关）

```rust
// pegaflow-core/src/block.rs:110
pub struct RawBlock {
    segments: Box<[Segment]>,  // 定长数组（分配后大小不变）
    total_size: usize,          // 所有片段的总字节数
}
```

**为什么是 `Box<[Segment]>` 而不是 `Vec<Segment>`？**

`Box<[T]>`（装箱切片）vs `Vec<T>`：
- `Vec<T>` 有 `capacity`（预分配），占 24 字节（ptr + len + cap）
- `Box<[T]>` 恰好占用元素数量，占 16 字节（ptr + len）
- `RawBlock` 一旦创建不再增删 segment，用 `Box<[T]>` 更省内存

```rust
// 创建时从 Vec 转换：
pub(crate) fn new(segments: Vec<Segment>) -> Self {
    let total_size = segments.iter().map(|s| s.size).sum();
    Self {
        segments: segments.into_boxed_slice(), // Vec → Box<[T]>
        total_size,
    }
}
```

### RawBlock 提供的接口

```rust
// 按索引访问片段：
pub(crate) fn segment_ptr(&self, index: usize) -> Option<NonNull<u8>>
pub(crate) fn segment_size(&self, index: usize) -> Option<usize>

// 迭代所有片段（用于 SSD I/O 构造 iovec）：
pub(crate) fn segment_iovecs(&self) -> impl Iterator<Item = (NonNull<u8>, usize)>

// 内存占用统计（用于缓存大小计算）：
pub(crate) fn memory_footprint(&self) -> u64
```

---

## 5. LayerBlock — KV 层视图

`LayerBlock` 是 `RawBlock` 的**语义包装**，将存储层的"片段列表"解释为"KV 缓存层的 K 段和 V 段"。

```rust
// pegaflow-core/src/block.rs:162
pub struct LayerBlock {
    raw: Arc<RawBlock>,
}

// 不变量：raw.num_segments() >= 1
```

```rust
// K 段：始终是 segment[0]
pub fn k_ptr(&self) -> *const u8 {
    self.raw.segment_ptr(0).unwrap().as_ptr()
}
pub fn k_size(&self) -> usize { self.raw.segment_size(0).unwrap() }

// V 段：可能是 segment[1]（分段存储），也可能不存在（连续存储）
pub fn v_ptr(&self) -> Option<*const u8> {
    self.raw.segment_ptr(1).map(|p| p.as_ptr() as *const u8)
}
pub fn v_size(&self) -> Option<usize> { self.raw.segment_size(1) }
```

**两种存储模式**：
```
连续存储（segments=1）：
┌──────────────────────────────────┐
│ K[0]K[1]...K[n] V[0]V[1]...V[n] │
└──────────────────────────────────┘
segment[0]：整块（K+V 连续）

分段存储（segments=2）：
┌───────────────────┐  ┌───────────────────┐
│ K[0]K[1]...K[n]   │  │ V[0]V[1]...V[n]   │
└───────────────────┘  └───────────────────┘
segment[0]：K 段         segment[1]：V 段
```

**分段存储的优势**：Load（CPU → GPU）时可以将所有块的 K 段合并为一次 `cudaMemcpy`，V 段合并为另一次，比逐块 K+V 传输更高效。

---

## 6. SealedBlock — 不可变的完成块

```rust
// pegaflow-core/src/block.rs:208
pub struct SealedBlock {
    slots: Box<[Arc<RawBlock>]>,  // 每个 TP slot 一个 RawBlock
    footprint: u64,                // 总内存占用（用于缓存大小跟踪）
    slot_numas: Vec<NumaNode>,    // 每个 slot 的 NUMA 亲和性（用于 RDMA NIC 选择）
}
```

**Slot 概念详解**：

在张量并行（TP）场景，同一个 KV block 在不同 GPU 上有不同的内容（每个 GPU 负责不同的 attention heads）。`SealedBlock` 的 `slots` 数组存储所有 TP rank 的数据：

```
TP=4 的场景（一个 SealedBlock 有 4 个 slot）：
slots[0] = GPU 0 负责的 attention heads 的 KV 数据
slots[1] = GPU 1 负责的 attention heads 的 KV 数据
slots[2] = GPU 2 负责的 attention heads 的 KV 数据
slots[3] = GPU 3 负责的 attention heads 的 KV 数据
```

Load 时，GPU 1 只需要 `slots[1]`（通过 `slot_map` 查到对应的 slot_id）。

```rust
pub(crate) fn get_slot(&self, slot_id: usize) -> Option<&Arc<RawBlock>> {
    self.slots.get(slot_id)
}
```

**`slot_numas` 的用途**：
RDMA 传输时，服务端需要告知客户端每个 slot 的内存所在的 NUMA 节点，客户端据此选择最近的 RDMA NIC（避免跨 NUMA 的内存访问影响 RDMA 带宽）。

---

## 7. InflightBlock — 写路径中间状态

`InflightBlock` 是 `SealedBlock` 的"建造者"，在所有 TP slot 写入完毕前保持 inflight 状态。

```rust
// pegaflow-core/src/block.rs:279
pub(crate) struct InflightBlock {
    slots: Vec<Option<Arc<RawBlock>>>,  // None = 尚未填充
    remaining: usize,                    // 剩余未填充的 slot 数
    total_slots: usize,
    footprint: u64,                      // 已填充部分的内存占用
    created_at: Instant,                 // 创建时间（用于 GC 超时检测）
    slot_numas: Vec<NumaNode>,           // 每个 slot 的 NUMA 信息
}
```

### 插入槽位

```rust
// pegaflow-core/src/block.rs:323
pub(crate) fn insert_slot(
    &mut self,
    slot_id: usize,
    block: Arc<RawBlock>,
    numa_node: NumaNode,
) -> SlotInsertResult {
    // 幂等性：重复插入同一 slot 是 no-op（防止网络重传等）
    if self.slots[slot_id].is_some() {
        return SlotInsertResult::Duplicate;
    }
    
    self.slots[slot_id] = Some(block);
    self.slot_numas[slot_id] = numa_node;
    self.remaining -= 1;
    
    SlotInsertResult::Inserted {
        completed: self.remaining == 0,  // 是否所有 slot 都已填充？
        footprint_added: ...,
    }
}
```

### 密封（Seal）

```rust
pub(crate) fn seal(self) -> SealedBlock {
    let slots: Vec<Arc<RawBlock>> = self.slots
        .into_iter()
        .map(|opt| opt.expect("所有 slot 必须已填充"))
        .collect();
    SealedBlock::from_slots_with_footprint(slots.into_boxed_slice(), self.footprint, self.slot_numas)
}
```

### 状态机

```
InflightBlock::new(total_slots)
         │
         ▼
    [slot_0 = None, slot_1 = None, ...]
         │
         │ insert_slot(0, block_0, numa)
         ▼
    [slot_0 = Some(block_0), slot_1 = None, ...]  remaining = total_slots - 1
         │
         │ insert_slot(1, block_1, numa)
         ▼
    [slot_0 = Some(block_0), slot_1 = Some(block_1), ...] remaining = 0
         │
         │ completed = true → seal()
         ▼
    SealedBlock { slots: [block_0, block_1, ...] }
```

---

## 8. BlockStatus — 块在存储层次中的位置

```rust
// pegaflow-core/src/block.rs:57
pub enum BlockStatus {
    Cached,      // 在内存缓存（ReadCache）中，可直接访问
    Inflight,    // 正在写入（尚未 seal），不可读
    Prefetching, // 正在从 SSD 或远端读取，稍后可用
    InSsd,       // 在 SSD 中，可触发预取
    Miss,        // 完全缺失（内存和 SSD 都没有）
}
```

---

## 9. PrefetchStatus — QueryPrefetch 的结果

```rust
// pegaflow-core/src/block.rs:71
pub enum PrefetchStatus {
    // 过渡状态：部分块正在预取，调用方应轮询重试
    Loading {
        hit: usize,     // 已在内存中的块数
        loading: usize, // 正在从 SSD/远端加载的块数
    },
    // 终止状态：确定结果（missing=0 表示全命中）
    Done {
        hit: usize,     // 在内存中的块数
        missing: usize, // 完全缺失的块数（0 = 全命中）
    },
}
```

**前缀语义**：查询 `[hash_0, hash_1, hash_2, hash_3]` 时，如果 `hash_1` 缺失，则后续的 `hash_2`, `hash_3` 即使在缓存中也算 miss（因为 vLLM 需要连续的前缀才能复用 KV cache）。

```
查询序列：[h0, h1, h2, h3]
缓存状态：h0=Cached, h1=Miss, h2=Cached, h3=Cached

结果：Done { hit: 1, missing: 3 }
（h1 缺失导致 h2, h3 也算 missing，即使它们在缓存中）
```

---

## 10. LayerSave — Save RPC 的输入结构

```rust
// pegaflow-core/src/block.rs:44
pub struct LayerSave {
    pub layer_name: String,          // 如 "model.layers.31"
    pub block_ids: Vec<i32>,         // GPU 缓冲区中的物理槽位编号
    pub block_hashes: Vec<Vec<u8>>,  // 每个 slot 对应的内容哈希
}
```

`block_ids[i]` 和 `block_hashes[i]` 一一对应：第 `i` 个 GPU 物理槽位包含哈希为 `block_hashes[i]` 的 KV 数据。

---

## 11. 设计模式小结

| 模式 | 应用 |
|------|------|
| Newtype（新类型） | `NumaNode(u32)`, `BlockHash = Vec<u8>` |
| RAII | `PinnedAllocation::drop()` 自动释放内存 |
| Builder | `InflightBlock` → `SealedBlock::seal()` |
| 内部可变性 | `Arc<ReadCache>` 中的 `Mutex<ReadCacheInner>` |
| 不变量断言 | `debug_assert!` 检查 slot_id 范围、segment 数量 |
