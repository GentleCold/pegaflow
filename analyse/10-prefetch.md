# 10 — 预取调度器（PrefetchScheduler）

**核心文件**：`pegaflow-core/src/storage/prefetch.rs`（265 行）

---

## 1. 预取的背景和必要性

**问题**：内存命中率不够高时，需要从 SSD 或远端节点预先加载 KV 块到内存，以减少 Load 阶段的等待时间。

**两级预取**：
1. **SSD 预取**：KV 块在 SSD 上 → 触发 io_uring 异步读取到固定内存
2. **RDMA 远端预取**：KV 块在另一台节点的内存中 → 通过 MetaServer 发现 + RDMA READ 传输到本节点

**触发时机**：`QueryPrefetch` RPC（与 `Query` 不同，`QueryPrefetch` 会主动触发预取）

---

## 2. 核心数据结构

```rust
// pegaflow-core/src/storage/prefetch.rs:50
pub(super) struct PrefetchScheduler {
    state: Mutex<PrefetchState>,
    ssd_store: Option<Arc<SsdBackingStore>>,    // SSD 后端（可选）
    rdma_fetch: Option<Arc<RdmaFetchStore>>,    // RDMA 远端拉取（可选）
    max_prefetch_blocks: usize,                  // 最大并发预取块数（背压控制）
}

struct PrefetchState {
    // 活跃预取请求：req_id → 预取条目
    active: HashMap<String, PrefetchEntry>,
    
    // 不变量：inflight_count == active.values().map(|e| e.loading_count).sum()
    inflight_count: usize,
    
    // 已知失败的远端请求：req_id → 失败时间
    // 防止对同一请求重复触发 RDMA（远端已驱逐该块）
    failed_remote: HashMap<String, Instant>,
}

struct PrefetchEntry {
    // 预取完成后通过 oneshot 通道接收结果
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    loading_count: usize,     // 正在加载的块数
    source: PrefetchSource,   // Ssd 或 Rdma
}

// PrefetchResult = Vec<(BlockKey, Arc<SealedBlock>)>
```

---

## 3. 主入口：check_and_prefetch()

```rust
// pegaflow-core/src/storage/prefetch.rs:75
pub(super) async fn check_and_prefetch(
    &self,
    read_cache: &ReadCache,
    instance_id: &str,
    req_id: &str,       // 请求 ID（跨 QueryPrefetch 轮询的追踪键）
    namespace: &str,
    hashes: &[Vec<u8>],
    num_workers: usize,
) -> PrefetchStatus {
    // 步骤 1：检查是否有已有的预取任务（轮询模式）
    if let Some(status) = self.poll_existing(read_cache, req_id) {
        match status {
            PollResult::StillLoading => {
                // 预取还未完成，通知调用方等待
                return PrefetchStatus::Loading { hit: 0, loading: 1 };
            }
            PollResult::Completed => {
                // 预取完成，重新扫描（fall through to full_prefix_scan）
            }
        }
    }

    // 步骤 2：完整前缀扫描（可能触发新的预取）
    self.full_prefix_scan(read_cache, instance_id, req_id, namespace, hashes, num_workers).await
}
```

### 调用方轮询模式

vLLM 会用同一个 `req_id` 多次调用 `QueryPrefetch`（轮询直到预取完成）：

```
调用 1：QueryPrefetch(req_id="req-1")
  → 内存 miss，触发 SSD 预取
  → 返回 PrefetchStatus::Loading { hit: 0, loading: 5 }

调用 2：QueryPrefetch(req_id="req-1")  （几毫秒后）
  → poll_existing("req-1") → 预取还在进行
  → 返回 PrefetchStatus::Loading { hit: 0, loading: 1 }

调用 3：QueryPrefetch(req_id="req-1")  （预取完成）
  → poll_existing("req-1") → 预取完成，blocks 插入内存
  → full_prefix_scan → 全部命中
  → 返回 PrefetchStatus::Done { hit: 5, missing: 0 }
```

---

## 4. poll_existing() — 检查已有预取任务

```rust
// pegaflow-core/src/storage/prefetch.rs:106
fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> Option<PollResult> {
    let mut state = self.state.lock();
    let entry = state.active.get_mut(req_id)?;  // 没有则返回 None

    match entry.blocks_rx.try_recv() {
        // 还没收到结果 → 继续等待
        Err(oneshot::TryRecvError::Empty) => Some(PollResult::StillLoading),
        
        // 收到结果！
        Ok(prefetched_blocks) => {
            let expected = entry.loading_count;
            let source = entry.source;
            state.remove_entry(req_id);  // 从 active 中移除，释放 inflight_count
            
            // RDMA 特殊处理：远端节点可能已驱逐该块，返回的数量少于预期
            if source == PrefetchSource::Rdma
                && prefetched_blocks.len() < expected
                && expected > 0
            {
                // 记录失败，避免下次重复触发 RDMA
                state.failed_remote.insert(req_id.to_string(), Instant::now());
                info!("RDMA prefetch returned fewer blocks than expected: ...");
            }
            
            drop(state);  // 先释放锁再 batch_insert（避免持锁做 I/O）
            read_cache.batch_insert(prefetched_blocks);  // 将预取结果放入内存缓存
            Some(PollResult::Completed)
        }
        
        // 发送端已关闭（预取任务崩溃）→ 重新扫描
        Err(oneshot::TryRecvError::Disconnected) => {
            warn!("Backing prefetch sender dropped for req_id={}", req_id);
            state.remove_entry(req_id);
            Some(PollResult::Completed)
        }
    }
}
```

---

## 5. full_prefix_scan() — 完整前缀扫描

这是预取的核心逻辑，分为三个阶段：

```rust
// pegaflow-core/src/storage/prefetch.rs:147
async fn full_prefix_scan(...) -> PrefetchStatus {
    // --- 阶段 1：内存前缀扫描 ---
    let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);
    let remaining = &keys[hit..];  // 内存 miss 的 keys
    
    // --- 阶段 2：SSD 预取（如果有 miss 且 SSD 可用）---
    if !remaining.is_empty() && has_backing {
        // 背压控制：只预取到 max_prefetch_blocks 上限
        let available = max_prefetch_blocks.saturating_sub(inflight_count);
        let check_limit = remaining.len().min(available);
        
        let (found, rx) = ssd.submit_prefix(check_keys);
        if found > 0 {
            state.inflight_count += found;
            loading = found;
            blocks_rx = Some(rx);
            entry_source = Some(PrefetchSource::Ssd);
        }
    }
    
    // --- 阶段 3：RDMA 远端预取（SSD 也没有 AND 内存也 miss 时）---
    if loading == 0
        && hit == 0
        && !remaining.is_empty()
        && rdma_fetch.is_some()
        && !state.failed_remote.contains_key(req_id)  // 未曾失败过
    {
        let (found, rx) = rdma_fetch.submit_remote_fetch(namespace, remaining).await;
        if found > 0 {
            state.inflight_count += found;
            loading = found;
            blocks_rx = Some(rx);
            entry_source = Some(PrefetchSource::Rdma);
        } else {
            // 远端也没有，记录失败避免重试
            state.failed_remote.insert(req_id.to_string(), Instant::now());
        }
    }
    
    let missing = keys.len() - hit - loading;
    
    // --- 返回结果 ---
    if loading > 0 {
        // 注册预取条目，等待后续 poll
        state.active.insert(req_id, PrefetchEntry { blocks_rx: rx, loading_count: loading, ... });
        PrefetchStatus::Loading { hit, loading }
    } else {
        // 所有块已在内存中或完全 miss
        read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
        PrefetchStatus::Done { hit, missing }
    }
}
```

### 三阶段决策树

```
full_prefix_scan() 执行流程：

内存前缀扫描
├── 全命中 → pin_blocks → Done { hit: N, missing: 0 }
└── 有 miss
    ├── SSD 可用 AND 有配额
    │   ├── SSD 有这些块 → submit_prefix → Loading { hit, loading }
    │   └── SSD 也没有
    │       └── RDMA 可用 AND 未曾失败
    │           ├── 远端有这些块 → submit_remote_fetch → Loading { hit, loading }
    │           └── 远端也没有 → failed_remote 记录 → Done { hit, missing }
    └── SSD 不可用 OR 超配额
        └── 尝试 RDMA（同上）
        └── Done { hit, missing }
```

---

## 6. 背压控制（Backpressure）

```rust
// 计算可用预取配额
let available = {
    let state = self.state.lock();
    self.max_prefetch_blocks.saturating_sub(state.inflight_count)
};

if available > 0 {
    let check_limit = remaining.len().min(available);  // 不超过配额
    let backpressure_skipped = remaining.len() - check_limit;  // 跳过的块
    if backpressure_skipped > 0 {
        core_metrics().ssd_prefetch_backpressure_blocks.add(backpressure_skipped as u64, &[]);
    }
    Some(remaining[..check_limit].to_vec())
} else {
    // 完全超配额，记录指标
    core_metrics().ssd_prefetch_backpressure_blocks.add(remaining.len() as u64, &[]);
    None
}
```

**为什么需要背压？**

SSD 读取速度有限（通常 5-10 GB/s），如果同时有大量预取请求，队列会无限增长，导致：
1. 内存占用失控（预取的块占用固定内存）
2. 单个请求等待时间变长（队列太长）

`max_prefetch_blocks`（默认 512）限制了同时处于"正在从 SSD 读取"状态的块数。

---

## 7. RDMA 失败处理

```rust
// 记录失败：
state.failed_remote.insert(req_id.to_string(), Instant::now());

// 检查是否失败过：
!state.failed_remote.contains_key(req_id)

// GC 清理过期的失败记录：
pub(super) fn gc_failed_remote(&self, max_age: std::time::Duration) -> usize {
    let mut state = self.state.lock();
    let before = state.failed_remote.len();
    state.failed_remote.retain(|_, ts| ts.elapsed() < max_age);
    before - state.failed_remote.len()
}
```

**场景说明**：

```
节点 B 有 blocks [h1, h2, h3]，本节点发起 RDMA 预取

情况 1：RDMA 成功，返回 3 个块 → 正常
情况 2：RDMA 触发了，但返回 2 个块（h3 被 LRU 驱逐了）
  → prefetched_blocks.len() (2) < expected (3)
  → 记录 failed_remote[req_id] = now
  → 下次 QueryPrefetch 不再尝试 RDMA
  → 避免无谓的网络请求（远端块已蒸发）
```

---

## 8. PrefetchStatus 状态机

```rust
// pegaflow-core/src/block.rs:71
pub enum PrefetchStatus {
    Loading {
        hit: usize,     // 已在内存中的块数
        loading: usize, // 正在从 SSD/RDMA 加载的块数
    },
    Done {
        hit: usize,     // 内存命中块数
        missing: usize, // 完全缺失块数（所有层次都没有）
    },
}
```

**与 gRPC 的映射**（`QueryResponse.prefetch_state`）：

| PrefetchStatus | gRPC PrefetchState | 含义 |
|----------------|-------------------|------|
| `Loading` | `PREFETCH_LOADING = 1` | 还在加载，应重试 |
| `Done { missing: 0 }` | `PREFETCH_DONE = 0` | 全命中，可以 Load |
| `Done { missing > 0 }` | `PREFETCH_DONE = 0` | 有缺失，只能用命中部分 |

---

## 9. 完整时序图

```
vLLM Scheduler 调用 QueryPrefetch：

时间 →
t=0  QueryPrefetch(req_id="R1", hashes=[h0,h1,h2,h3,h4])
     → 内存：h0 命中
     → SSD：h1,h2 找到，提交 io_uring 读取
     → RDMA：不触发（SSD 有结果）
     → 返回 Loading { hit: 1, loading: 2 }

t=5ms QueryPrefetch(req_id="R1")
     → poll_existing("R1") → try_recv → Empty（IO 还在进行）
     → 返回 Loading { hit: 0, loading: 1 }

t=15ms SSD io_uring 读取完成，h1+h2 已在内存
     （通过 oneshot 通道通知 PrefetchScheduler）

t=20ms QueryPrefetch(req_id="R1")
     → poll_existing("R1") → try_recv → Ok([h1_block, h2_block])
     → batch_insert([h1, h2])
     → remove_entry("R1")
     → full_prefix_scan 重新扫描
     → 内存：h0,h1,h2 命中，h3 miss
     → SSD：h3 不在 SSD，尝试 RDMA
     → RDMA：h3 在节点 B，提交 remote_fetch
     → 返回 Loading { hit: 3, loading: 1 }

t=50ms RDMA 传输完成

t=55ms QueryPrefetch(req_id="R1")
     → h3 已在内存
     → Done { hit: 4, missing: 1 }  （h4 完全缺失）
```

---

## 10. 与 StorageEngine 的集成

```rust
// pegaflow-core/src/storage/mod.rs
pub(crate) async fn check_prefix_and_prefetch(
    &self,
    instance_id: &str,
    req_id: &str,
    namespace: &str,
    hashes: &[Vec<u8>],
    num_workers: usize,
) -> PrefetchStatus {
    self.prefetch.check_and_prefetch(
        &self.read_cache,
        instance_id,
        req_id,
        namespace,
        hashes,
        num_workers,
    ).await
}
```

`StorageEngine` 将 `ReadCache` 传入 `PrefetchScheduler`，使预取完成后可以直接将块插入缓存并 pin 住（用于随后的 Load）。
