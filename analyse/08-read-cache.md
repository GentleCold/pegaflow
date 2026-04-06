# 08 — ReadCache + TinyLFU 缓存

**核心文件**：
- `pegaflow-core/src/storage/read_cache.rs`（376 行）
- `pegaflow-core/src/cache.rs`（276 行）

---

## 1. 整体架构

ReadCache 是 PegaFlow 内存层的核心数据结构，负责：
1. **存储**已密封的 KV 块（`SealedBlock`）
2. **准入控制**（TinyLFU 策略）
3. **Pin/Consume/Unpin 协议**（防止 Load 期间块被驱逐）
4. **LRU 驱逐**（内存压力时释放最旧的块）

```
ReadCache
│
├── TinyLfuCache（LRU + TinyLFU 准入策略）
│   ├── LruCache<BlockKey, Arc<SealedBlock>>  —— 主存储
│   └── TinyLfu（可选）                        —— 频率估计器
│
└── pinned_for_load                            —— Pin 引用计数
    ├── pinned_for_load: HashMap<(instance_id, BlockKey), (block, refcount)>
    └── pinned_for_load_by_key: HashMap<BlockKey, (footprint_bytes, total_refcount)>
```

---

## 2. ReadCache 结构详解

```rust
// pegaflow-core/src/storage/read_cache.rs:11
pub(super) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
    // 用 parking_lot::Mutex 包裹内部状态，确保线程安全
}

// 内部状态（被锁保护）
struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    
    // Pin 映射：键 = (实例ID, 块键)，值 = (块引用, 引用计数)
    // 语义：某个实例的某个块被 pin 住了多少次
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
    
    // 聚合引用计数（按块键，不区分实例）
    // 用于指标统计：记录被 pin 的唯一块占用多少字节
    pinned_for_load_by_key: HashMap<BlockKey, (u64, usize)>,
}
```

> **Rust 新手提示**：`parking_lot::Mutex` 是第三方库提供的互斥锁，比标准库 `std::sync::Mutex` 性能更好（使用自旋锁 + 系统调用的混合策略），且不会在 panic 时"中毒"（poisoning），简化了错误处理。

---

## 3. Pin / Consume / Unpin 三步协议

这是 ReadCache 最关键的设计，解决的问题：

**问题**：vLLM 发起 `QueryPrefetch` 查到块在内存中，随后发起 `Load`。但从 Query 到 Load 之间有时间差，块可能被 LRU 驱逐。

**解决方案**：三步协议

```
QueryPrefetch 命中 → pin_blocks()
       │
       ▼
   块被固定（Arc 引用计数 > 1），LRU 驱逐时虽然从 cache 中移除，
   但内存不会释放（Arc 还活着）
       │
       ▼
Load RPC 到来 → consume_pinned_blocks()
       │
       ▼
   返回 Arc<SealedBlock>，pin 引用计数减 1
       │
如果 Load 被取消（请求被抢占）↓
       ▼
Unpin RPC → unpin_blocks()
       │
       ▼
   pin 引用计数减 1，如果归零，内存可被 LRU 驱逐
```

### 3.1 pin_blocks() — 增加引用计数

```rust
// pegaflow-core/src/storage/read_cache.rs:140
pub(super) fn pin_blocks(
    &self,
    instance_id: &str,
    num_workers: usize,   // 每个 GPU worker 消费一次，共需 num_workers 次引用
    blocks: &[(BlockKey, Arc<SealedBlock>)],
) {
    let mut inner = self.inner.lock();
    for (key, block) in blocks {
        let pin_key = (instance_id.to_string(), key.clone());
        let footprint = block.memory_footprint();

        // 如果已存在，增加引用计数；否则新建条目
        match inner.pinned_for_load.entry(pin_key) {
            Entry::Occupied(mut o) => {
                o.get_mut().1 += num_workers;  // 增加引用计数
            }
            Entry::Vacant(v) => {
                v.insert((Arc::clone(block), num_workers));  // 新建
            }
        }
        
        // 同步更新聚合统计（用于 Prometheus 指标）
        match inner.pinned_for_load_by_key.entry(key.clone()) {
            Entry::Occupied(mut o) => { o.get_mut().1 += num_workers; }
            Entry::Vacant(v) => {
                v.insert((footprint, num_workers));
                core_metrics().pinned_for_load_unique_bytes.add(footprint as i64, &[]);
            }
        }
    }
}
```

**关键细节**：`num_workers` 为什么不是 1？

在张量并行（TP=4）场景，同一个 block 需要被 4 个 GPU worker 分别 Load。每个 worker 会调用一次 `consume_pinned_blocks`，所以需要 4 个引用。

### 3.2 consume_pinned_blocks() — 消费一次引用

```rust
// pegaflow-core/src/storage/read_cache.rs:182
pub(super) fn consume_pinned_blocks(
    &self,
    instance_id: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> Result<Vec<Arc<SealedBlock>>, String> {
    let mut result = Vec::new();
    let mut inner = self.inner.lock();

    // 阶段 1：验证所有块都存在（atomic 验证，失败则整体回滚）
    for (idx, pin_key) in pin_keys.iter().enumerate() {
        if let Some((block, _)) = inner.pinned_for_load.get(pin_key) {
            result.push(Arc::clone(block));
        } else {
            return Err(format!("missing pinned KV block at index {}", idx));
        }
    }

    // 阶段 2：减少所有 pin 引用计数
    for pin_key in &pin_keys {
        inner.decrement_pin(pin_key);
    }

    Ok(result)
}
```

**两阶段设计的重要性**：先验证所有 pin 都存在，再统一扣减。避免"部分成功"导致的不一致状态。

### 3.3 decrement_pin() — 引用计数减一

```rust
// pegaflow-core/src/storage/read_cache.rs:28
fn decrement_pin(&mut self, pin_key: &(String, BlockKey)) -> bool {
    let Some((_, count)) = self.pinned_for_load.get_mut(pin_key) else {
        return false;  // pin 不存在，返回 false（Unpin 时无害）
    };
    
    *count = count.saturating_sub(1);  // 减一，防止下溢到 u64::MAX
    
    if *count == 0 {
        self.pinned_for_load.remove(pin_key);  // 引用计数归零，删除条目
    }
    
    // 同步更新聚合统计
    if let Some((bytes, total)) = self.pinned_for_load_by_key.get_mut(&pin_key.1) {
        *total = total.saturating_sub(1);
        if *total == 0 {
            let bytes_val = *bytes;
            self.pinned_for_load_by_key.remove(&pin_key.1);
            // 当这个块完全没有 pin 引用时，从指标中减去占用字节
            core_metrics().pinned_for_load_unique_bytes.add(-(bytes_val as i64), &[]);
        }
    }
    true
}
```

> **Rust 新手提示**：`saturating_sub(1)` 是"饱和减法"，当结果本来会下溢（usize 减到负数）时，返回 0 而不是 panic。这是防御性编程的好习惯。

---

## 4. LRU 驱逐机制

```rust
// pegaflow-core/src/storage/read_cache.rs:271
pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    let mut inner = self.inner.lock();
    (0..batch_size)
        .map_while(|_| inner.cache.remove_lru())  // 每次取一个 LRU 尾部元素
        .collect()
}
```

这个方法被 `StorageEngine::reclaim_until_allocator_can_allocate()` 调用，每次驱逐最多 64 个块。

**被 pin 的块的内存为何不立即释放？**

```
remove_lru_batch() 调用后：
┌─────────────────────────────────────────────────────┐
│  LruCache 中的 Arc 引用被移除 → block.arc_count - 1  │
│                                                       │
│  如果 block 被 pinned_for_load 持有（arc_count > 1）: │
│  → 内存仍被引用，不会释放                              │
│                                                       │
│  如果 block 没有 pin（arc_count == 1）:               │
│  → Arc 引用归零 → drop → PinnedAllocation::drop()    │
│  → 内存归还给分配器                                    │
└─────────────────────────────────────────────────────┘
```

这正是代码注释所说的 `strong_count(block) > 1` 的情况——`StorageEngine` 会记录这类指标：

```rust
// pegaflow-core/src/storage/mod.rs
if Arc::strong_count(block) > 1 {
    core_metrics().cache_block_evictions_still_referenced.add(1, &[]);
}
```

---

## 5. 查询接口对比

ReadCache 提供三种查询接口，语义各不同：

| 方法 | 前缀语义？ | 更新频率？ | 用途 |
|------|-----------|-----------|------|
| `check_prefix_memory_only` | 是（遇 miss 停止）| 是 | Query RPC（纯命中统计）|
| `get_prefix_blocks` | 是（遇 miss 停止）| 是 | QueryPrefetch（获取块用于 pin）|
| `get_blocks` | 否（跳过 missing）| 是 | 跨节点 transfer（按需获取）|

```rust
// check_prefix_memory_only：只统计，不返回块
pub(super) fn check_prefix_memory_only(&self, namespace: &str, hashes: &[Vec<u8>]) -> (usize, usize) {
    let mut hit = 0;
    let mut inner = self.inner.lock();
    for hash in hashes {
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        if inner.cache.get(&key).is_some() { // get() 会更新 LRU 位置和频率
            hit += 1;
        } else {
            break;  // 前缀语义：遇 miss 停止
        }
    }
    (hit, hashes.len() - hit)
}
```

```rust
// get_blocks：跳过 missing，不停止
pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    let mut inner = self.inner.lock();
    let mut found = Vec::new();
    for key in keys {
        if let Some(block) = inner.cache.get(key) {
            found.push((key.clone(), Arc::clone(&block)));
            // 注意：没有 break！遇 miss 继续查下一个
        }
    }
    found
}
```

---

## 6. TinyLFU 详解

### 6.1 为什么需要 TinyLFU？

纯 LRU 的问题：**扫描攻击（Scan Attack）**

```
场景：对象存储系统执行全表扫描
扫描序列：[A, B, C, D, E, F, G, H, ...（100 万个块）]

LRU 行为：缓存中本来有热点块 [X, Y, Z]
扫描执行后：热点块全被驱逐，缓存被扫描块占满
结果：缓存命中率骤降，直到热点块重新加载
```

TinyLFU 解决：**准入控制**

```
对于每个新加入的块（候选），与最旧的块（受害者）比较频率：
if freq(候选) >= freq(受害者) → 准入
if freq(候选) < freq(受害者)  → 拒绝（候选是冷数据）
```

对于扫描块（只访问一次），其频率为 1，而热点块频率远高于 1，因此扫描块永远无法驱逐热点块。

### 6.2 Count-Min Sketch（CM-Sketch）实现

```rust
// pegaflow-core/src/cache.rs:109
struct Estimator {
    // 二维数组：d 行 × w 列
    // 每个元素是 AtomicU8（0-255）
    estimator: Box<[(Box<[AtomicU8]>, RandomState)]>,
    // ↑ 每个元组 = 一行计数器 + 独立随机哈希函数
}
```

**CM-Sketch 工作原理**：

```
插入 key K 时：
┌──────────────────────────────────────────────────┐
│ 行 0（哈希函数 h0）: h0(K) % w → 列索引 → 计数器++ │
│ 行 1（哈希函数 h1）: h1(K) % w → 列索引 → 计数器++ │
│ 行 2（哈希函数 h2）: h2(K) % w → 列索引 → 计数器++ │
└──────────────────────────────────────────────────┘

查询 key K 的频率时：
freq(K) = min(行0对应格子, 行1对应格子, 行2对应格子)
                          ↑
                     取最小值（防止哈希碰撞导致高估）
```

**为什么取最小值？**

哈希碰撞时，同一个格子可能被多个 key 共享，计数器会被高估。取所有行的最小值，能最小化高估的概率。

```rust
// 插入（递增计数器）
fn incr<T: Hash>(&self, key: T) -> u8 {
    let mut min = u8::MAX;
    for (slot, hasher) in &self.estimator {
        let hash = hasher.hash_one(&key) as usize;
        let counter = &slot[hash % slot.len()];
        let (_, new) = incr_no_overflow(counter);  // 原子递增，不溢出
        min = std::cmp::min(min, new);
    }
    min  // 返回估计频率
}

// 查询频率
fn get<T: Hash>(&self, key: T) -> u8 {
    let mut min = u8::MAX;
    for (slot, hasher) in &self.estimator {
        let hash = hasher.hash_one(&key) as usize;
        let counter = &slot[hash % slot.len()];
        min = std::cmp::min(min, counter.load(Ordering::Relaxed));
    }
    min
}
```

### 6.3 时间衰减（老化）机制

**问题**：计数器只增不减，历史热点会永久占据高频率。一个昨天热但今天冷的块，其频率仍然很高。

**解决方案**：定期将所有计数器右移 1 位（除以 2）

```rust
// pegaflow-core/src/cache.rs:226
fn incr<T: Hash>(&self, key: T) -> u8 {
    let window_size = self.window_counter.fetch_add(1, Ordering::Relaxed);
    
    // 每 window_limit 次访问触发一次老化
    if window_size == self.window_limit || window_size > self.window_limit * 2 {
        self.window_counter.store(0, Ordering::Relaxed);
        self.estimator.age(AGE_SHIFT_BITS);  // 所有计数器 >> 1
    }
    
    self.estimator.incr(key)
}
```

**window_limit 计算**：

```rust
// 缓存有 N 个块，窗口大小 = N * 8
// 每访问 8N 次触发一次老化
fn new(cache_size: usize) -> Self {
    Self {
        estimator: Estimator::optimal(cache_size),
        window_counter: Default::default(),
        window_limit: cache_size * WINDOW_LIMIT_MULTIPLIER,  // × 8
    }
}
```

**老化效果图**：

```
时间流逝：
t=0：所有计数器 = [100, 5, 80, 3, 60, ...]
               （历史热点）（扫描块）
t=老化后：     [50,  2, 40, 1, 30, ...]
               历史频率减半，最近访问的块的优势被放大
```

### 6.4 准入判断

```rust
// pegaflow-core/src/cache.rs:76
pub(crate) fn insert(&mut self, key: BlockKey, value: ArcSealedBlock) -> CacheInsertOutcome {
    // 先记录访问（无论是否准入，都提高频率）
    if let Some(freq) = &self.freq {
        freq.incr(&key);
    }
    
    // 内容寻址：同一哈希的块不重复存储
    if self.lru.contains_key(&key) {
        return CacheInsertOutcome::AlreadyExists;
    }
    
    // 准入决策：候选 vs 受害者（LRU 链表头部 = 最旧的块）
    if let Some(freq) = &self.freq {
        let candidate_freq = freq.get(&key);
        if let Some((victim_key, _)) = self.lru.iter().next() {
            let victim_freq = freq.get(victim_key);
            if candidate_freq < victim_freq {
                return CacheInsertOutcome::Rejected;  // 拒绝冷数据
            }
        }
    }
    
    self.lru.insert(key, value);
    CacheInsertOutcome::InsertedNew
}
```

> **注意**：`lru.iter().next()` 返回的是 LRU 链表的**最旧**元素（LRU head），也就是下一个会被驱逐的受害者。

### 6.5 Optimal vs Compact 模式

```rust
// 两种大小的 CM-Sketch：

// Optimal：精确估计，内存较大
// w = ceil(e / ε)，d = ceil(ln(1-δ) / ln(0.5))
// ε = 错误率 = 1/items，δ = 失败概率 = 1/items
fn optimal(items: usize) -> Self { ... }

// Compact：1/100 大小的 CM-Sketch，内存节省但精度略低
fn compact(items: usize) -> Self {
    let (slots, hashes) = Self::optimal_paras(items / COMPACT_ESTIMATOR_DIVISOR);
    Self::new(hashes, slots, ...)
}
```

实际使用的是 `Estimator::optimal`（通过 `TinyLfu::new`），`compact` 目前标注为 `#[allow(dead_code)]`，是备选方案。

---

## 7. TinyLfuCache 的批量插入

```rust
// pegaflow-core/src/storage/read_cache.rs:121
pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
    let mut inner = self.inner.lock();
    for (key, block) in blocks {
        let footprint_bytes = block.memory_footprint();
        match inner.cache.insert(key, block) {
            CacheInsertOutcome::InsertedNew => {
                let m = core_metrics();
                m.cache_block_insertions.add(1, &[]);
                m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            }
            CacheInsertOutcome::AlreadyExists => {}
            CacheInsertOutcome::Rejected => {
                core_metrics().cache_block_admission_rejections.add(1, &[]);
            }
        }
    }
}
```

这个方法被 `insert_worker_loop` 在 SealedBlock 密封后调用，将新块批量加入缓存。

---

## 8. 完整数据流图

```
[QueryPrefetch RPC]
        │
        ▼
ReadCache::get_prefix_blocks()  ← 前缀扫描 + 更新 LRU
        │
        │ 返回命中的块
        ▼
ReadCache::pin_blocks(instance_id, num_workers, &blocks)
        │                        ↑
        │                    每个 GPU worker 一个引用
        ▼
  [pinned_for_load] 更新引用计数


[Load RPC] — GPU worker 准备好接收数据
        │
        ▼
ReadCache::consume_pinned_blocks(instance_id, namespace, hashes)
        │
        ├── 返回 Vec<Arc<SealedBlock>>（GPU worker 用来 cudaMemcpy）
        └── 引用计数减 1


[Unpin RPC] — Load 被取消
        │
        ▼
ReadCache::unpin_blocks(instance_id, namespace, hashes)
        │
        └── 引用计数减 1，归零则可被 LRU 驱逐


[内存不足时]
        │
        ▼
StorageEngine::reclaim_until_allocator_can_allocate()
        │
        ▼
ReadCache::remove_lru_batch(64)
        │
        ├── 从 LRU 链表移除最旧的 64 个块
        └── Arc 引用计数减 1
            ├── 若 count == 1（没有 pin）→ drop → 内存立即释放
            └── 若 count > 1（有 pin）→ 暂时保留，pin 释放后才 drop
```

---

## 9. 测试案例解读

```rust
// get_blocks 和 get_prefix_blocks 的关键区别：
#[test]
fn get_blocks_does_not_break_at_first_miss() {
    // 缓存：keys 0, 2, 4 存在；keys 1, 3 缺失
    
    // get_blocks: 返回 0, 2, 4（跳过 1, 3，不停止）
    let result = cache.get_blocks(&keys);
    assert_eq!(result.len(), 3);  // 返回 3 个
    
    // get_prefix_blocks: 遇到 key 1（miss）就停止
    let (prefix_hit, _) = cache.get_prefix_blocks(&keys);
    assert_eq!(prefix_hit, 1);   // 只返回 key 0
}
```

**为什么 Query RPC 需要前缀语义？**

vLLM 的 prefix caching 需要**连续的前缀**才能复用 KV cache。如果 token 序列 `[t0, t1, t2]` 中 `t1` 缺失，`t2` 即使命中也没有意义（GPU 无法只加载 t2 而跳过 t1）。

**为什么跨节点 transfer 不需要前缀语义？**

跨节点 transfer（RDMA）是按需获取特定的 block，不依赖连续性。节点 A 可以单独获取节点 B 的某几个块，不需要连续前缀。
