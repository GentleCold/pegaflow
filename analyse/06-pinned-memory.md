# 06 — 固定内存管理

**核心文件**：
- `pegaflow-core/src/pinned_mem.rs` — CUDA 固定内存底层封装
- `pegaflow-core/src/allocator.rs` — 字节级偏移分配器
- `pegaflow-core/src/pinned_pool.rs` — 内存池管理（RAII + NUMA）

---

## 1. 为什么需要固定内存？

**普通内存的问题**：操作系统可以随时将普通内存页面换到磁盘（swap），导致物理地址不固定。GPU 的 DMA（直接内存访问）控制器需要稳定的**物理地址**来传输数据，如果页面被换出，DMA 会失败。

**固定内存（Pinned / Page-locked Memory）**：
- 物理地址永久固定，操作系统不会换出
- GPU 可以直接 DMA 读写，无需 CPU 参与数据拷贝
- 带宽更高（PCIe 理论带宽 ~64 GB/s），延迟更低
- 代价：占用物理内存，系统可分配的普通内存减少

```
普通内存（CPU → GPU 拷贝的流程）：
CPU RAM 页面 --（可能换出）--> 固定缓冲区 --DMA--> GPU HBM
                              （额外一次拷贝）

固定内存（CPU → GPU 拷贝的流程）：
固定内存（物理地址固定）------DMA直接---------> GPU HBM
                            （零拷贝）
```

---

## 2. pinned_mem.rs — 底层 CUDA 内存封装

### 两种分配策略

**策略 1：Write-Combined（默认）**

```rust
// pegaflow-core/src/pinned_mem.rs（概念示意）
// cudaHostAlloc(ptr, size, cudaHostAllocWriteCombined)
// Write-Combined: CPU 写入时绕过 L1/L2 缓存，直接写入内存总线
// 优点：CPU→GPU 写入更快
// 缺点：CPU 读取较慢（但 PegaFlow 主要是写入场景）
```

**策略 2：HugePages + cudaHostRegister**

```bash
# 系统预配置（一次性）：
sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'
# 预留 15360 × 2MB = 30GB 大页
```

```rust
// 1. mmap(MAP_HUGETLB) 分配 2MB 大页内存
// 2. cudaHostRegister() 将其注册为固定内存
// 优点：单次 mmap 系统调用分配大块连续内存，分配速度更快
//       减少 TLB 压力（2MB 页 vs 4KB 页，TLB 条目减少 512×）
// 缺点：需要预先配置，失败时 fallback 到 Write-Combined
```

### PinnedMemory 结构体

```rust
pub struct PinnedMemory {
    ptr: NonNull<u8>,
    size: usize,
    strategy: AllocationStrategy,  // WriteCombined 或 HugePages
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        match self.strategy {
            AllocationStrategy::WriteCombined => {
                // cudaFreeHost(ptr)  — 解注册 + 释放
            }
            AllocationStrategy::HugePages => {
                // cudaHostUnregister(ptr)  — 解注册
                // munmap(ptr, size)        — 释放内存映射
            }
        }
    }
}
```

> **Rust 新手提示**：`Drop` trait 类似 C++ 的析构函数。实现 `Drop::drop()` 后，当值离开作用域时 Rust 自动调用它。这是 RAII 模式的核心。

---

## 3. allocator.rs — 字节级偏移分配器

### 问题背景

PegaFlow 一次性分配一大块固定内存（如 30GB），然后按需切分给各个 block。需要一个高效的**子分配器**。

### ScaledOffsetAllocator

```rust
// pegaflow-core/src/allocator.rs:66
pub struct ScaledOffsetAllocator {
    unit_size: NonZeroU64,    // 最小分配单位（如 4KB）
    total_units: u32,          // 总单位数
    inner: RawAllocator,       // 底层 offset-allocator（u32 偏移量）
}
```

**工作原理**：

```
大内存块（30GB）：
┌────────────────────────────────────────────────────────┐
│ unit 0 │ unit 1 │ unit 2 │ ... │ unit N               │
│ 4KB    │ 4KB    │ 4KB    │     │ 4KB                  │
└────────────────────────────────────────────────────────┘

分配请求：8KB → 分配 2 个 unit，返回 Allocation { offset_bytes: 0, size_bytes: 8KB }
分配请求：12KB → 分配 3 个 unit，返回 Allocation { offset_bytes: 8KB, size_bytes: 12KB }
```

```rust
// Allocation 结构体：
pub struct Allocation {
    pub offset_bytes: u64,    // 从大内存块开始的字节偏移量
    pub size_bytes: NonZeroU64, // 实际分配大小（向上取整到 unit_size）
    raw: RawAllocation,        // 底层 u32 令牌（用于释放）
}
```

**释放**：`RawAllocator::free(raw)` 将单位标记为可复用。**注意**：这个分配器本身不线程安全，外部需加锁（见 `PinnedMemoryPool`）。

### 分配单位大小的选择

- 太小（如 1 字节）：`RawAllocator` 的 u32 偏移量会溢出（30GB / 1B = 30G > u32::MAX）
- 太大（如 1MB）：分配粒度太粗，浪费内存
- 实践中：通常设为 `bytes_per_block`（一个 KV block 的大小），使 block 精确占用整数个 unit

---

## 4. pinned_pool.rs — 内存池管理

### PinnedMemoryPool

```rust
// pegaflow-core/src/pinned_pool.rs:60（概略）
pub(crate) struct PinnedMemoryPool {
    base_ptr: NonNull<u8>,        // 大内存块的基地址
    allocator: parking_lot::Mutex<ScaledOffsetAllocator>,
    pinned_mem: Arc<PinnedMemory>, // 底层 CUDA 固定内存（保持存活）
    // 统计信息：
    used_bytes: AtomicUsize,
    total_bytes: usize,
}
```

**分配流程**：
```rust
fn allocate(&self, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
    let mut alloc = self.allocator.lock();
    let allocation = alloc.allocate(size)?; // 返回字节偏移量
    
    // 计算实际指针：
    let ptr = unsafe {
        NonNull::new(self.base_ptr.as_ptr().add(allocation.offset_bytes as usize))
            .expect("offset within pool")
    };
    
    Some(Arc::new(PinnedAllocation {
        allocation,
        ptr,
        pool: Arc::clone(&self.self_arc), // 持有 pool 引用，防止 pool 先于 allocation 销毁
    }))
}
```

### PinnedAllocation — RAII 分配守卫

```rust
// pegaflow-core/src/pinned_pool.rs:22
pub struct PinnedAllocation {
    allocation: Allocation,      // 偏移量令牌（用于释放）
    ptr: NonNull<u8>,            // 指向分配内存的指针
    pool: Arc<PinnedMemoryPool>, // 持有 pool 防止先于 allocation 销毁
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        self.pool.free_internal(&self.allocation); // 归还给分配器
    }
}
```

**生命周期链**：

```
PinnedMemory（大块固定内存）
    ↑ Arc 持有（alive as long as pool alive）
PinnedMemoryPool（分配器 + 基地址）
    ↑ Arc 持有（alive as long as allocation alive）
PinnedAllocation（分配令牌 + 指针）
    ↑ Arc 持有
Segment（裸指针 + 大小）
    ↑ Box/Arc 持有
RawBlock（Segment 数组）
    ↑ Arc 持有
SealedBlock（RawBlock 数组）
    ↑ Arc 持有（在 ReadCache 中）

当 SealedBlock 被 LRU 驱逐，Arc 引用计数归零：
SealedBlock drop → RawBlock drop → Segment drop →
PinnedAllocation drop → 归还给分配器 → 内存可复用
```

---

## 5. PinnedAllocator — NUMA 感知统一分配器

`PinnedAllocator` 是对一个或多个 `PinnedMemoryPool` 的统一管理器：

```rust
// 两种模式：
// 1. 全局模式（单节点，无 NUMA 优化）：
pub(crate) fn new_global(
    capacity_bytes: usize,
    pool_shards: usize,    // 分片数，减少锁竞争
    use_hugepages: bool,
    ssd_enabled: bool,
    unit_hint: Option<NonZeroU64>,
) -> Self

// 2. NUMA 感知模式（多路服务器，性能最佳）：
pub(crate) fn new_numa(
    capacity_bytes: usize,    // 总容量（均分到各 NUMA 节点）
    numa_nodes: &[NumaNode],
    pool_shards: usize,
    use_hugepages: bool,
    ssd_enabled: bool,
    unit_hint: Option<NonZeroU64>,
) -> Self
```

**NUMA 感知分配**：

```rust
pub(crate) fn allocate(&self, size: NonZeroU64, numa_node: NumaNode) -> Option<Arc<PinnedAllocation>> {
    match &self.inner {
        Inner::Global(pools) => {
            // 轮询（round-robin）选择分片
            let shard = self.next_shard.fetch_add(1, Ordering::Relaxed) % pools.len();
            pools[shard].allocate(size)
        }
        Inner::Numa(numa_pools) => {
            // 优先从目标 NUMA 节点的 pool 分配
            if let Some(pools) = numa_pools.get(&numa_node) {
                let shard = ...;
                pools[shard].allocate(size)
                    .or_else(|| self.allocate_from_any(size)) // fallback
            }
        }
    }
}
```

### 内存区域查询（用于 RDMA 注册）

```rust
pub(crate) fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
    // 返回所有 pool 的 (base_ptr, total_bytes)
    // RDMA NIC 需要预先注册这些内存区域，才能直接读写
}
```

---

## 6. 分配器性能考量

| 参数 | 影响 | 建议 |
|------|------|------|
| `pool_shards` | 多线程分配时的锁竞争 | 设为 CPU 核心数或 2-4 |
| `unit_hint` | 分配粒度（影响碎片率） | 设为 `bytes_per_block` |
| `use_hugepages` | 分配速度 + TLB 效率 | 预配置后开启 |
| `blockwise_alloc` | 碎片 vs 系统调用开销 | 不规则大小时开启 |

---

## 7. 内存回收：LRU 驱逐

当分配失败时（内存耗尽），`StorageEngine::allocate()` 触发 LRU 驱逐：

```rust
// pegaflow-core/src/storage/mod.rs:266
fn allocate(&self, size: NonZeroU64, ...) -> Option<Arc<PinnedAllocation>> {
    loop {
        if let Some(alloc) = self.allocator.allocate(size, node) {
            return Some(alloc); // 成功
        }
        
        // 驱逐最旧的 block（一批 64 个）
        let (freed_blocks, _freed_bytes, largest_free) =
            self.reclaim_until_allocator_can_allocate(size.get());
        
        if freed_blocks == 0 || largest_free < size.get() {
            break; // 无法释放足够空间
        }
    }
    
    // 分配失败：记录指标，返回 None
    core_metrics().pool_alloc_failures.add(1, &[]);
    None
}
```

**被 pin 的 block 不会被驱逐**：`ReadCache::remove_lru_batch()` 只移除 LRU 缓存条目，但 `pinned_for_load` 中的 block（`Arc` 引用计数 > 1）的内存不会立即释放——它们的内存在 `consume_pinned_blocks()` 消费并且 `Arc` 引用归零后才释放。

---

## 8. 使用示例（完整 block 生命周期）

```rust
// 1. 分配固定内存（用于存储一个 KV block）
let size = NonZeroU64::new(bytes_per_block as u64).unwrap();
let allocation: Arc<PinnedAllocation> = engine.storage.allocate(size, Some(numa_node))?;

// 2. 获取可写指针，执行 CUDA 拷贝
let ptr = {
    let alloc = Arc::get_mut(&mut allocation).unwrap();
    NonNull::new(alloc.as_mut_ptr()).unwrap()
};
// cudaMemcpyAsync(ptr, gpu_src, size, D2H, stream);

// 3. 包装为 Segment
let segment = Segment::new(ptr, bytes_per_block, allocation);

// 4. 包装为 RawBlock
let raw_block = Arc::new(RawBlock::new(vec![segment]));

// 5. 插入到 InflightBlock → 最终 seal 为 SealedBlock → 放入 ReadCache
// （内存生命周期由 Arc 链自动管理）
```
