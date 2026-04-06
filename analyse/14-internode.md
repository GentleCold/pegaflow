# 14 — 跨节点通信（MetaServerClient + RDMA fetch）

**核心文件**：
- `pegaflow-core/src/internode/metaserver_client.rs` — MetaServer 客户端
- `pegaflow-core/src/backing/rdma_fetch.rs` — RDMA 远端拉取（见文档 12）
- `pegaflow-metaserver/src/store.rs` — MetaServer 存储层

---

## 1. 跨节点 KV cache 共享

多节点集群场景：

```
节点 A                 MetaServer                节点 B
(GPU 推理)             (块哈希注册中心)           (KV cache 已存在)
    │                      │                          │
    │ [QueryPrefetch]       │                          │
    │ 本地内存 miss          │                          │
    │ SSD 也没有            │                          │
    │                      │                          │
    │──── query(ns, hashes) ─→                         │
    │←─── {B: [h1, h2, h3]} ─                         │
    │                      │                          │
    │────── QueryBlocksForTransfer(h1,h2,h3) ─────────→│
    │←───── {内存地址, rkey, session_id} ──────────────│
    │                      │                          │
    │══════ RDMA READ ════════════════════════════════→│
    │                      │                          │
    │────── ReleaseTransferLock(session_id) ───────────→│
```

---

## 2. MetaServerClient

### 2.1 结构

```rust
// pegaflow-core/src/internode/metaserver_client.rs:61
pub struct MetaServerClient {
    insert_tx: mpsc::Sender<RegistrationBatch>,  // fire-and-forget 注册队列
    query_client: MetaServerGrpcClient<Channel>, // 直接查询客户端（懒连接）
}
```

**两种操作，两种设计**：
- **注册（Insert）**：不在关键路径上（block 写入后异步注册），用 MPSC 通道 + 后台 task
- **查询（Query）**：在 QueryPrefetch 关键路径上，直接 RPC 调用（异步 await）

### 2.2 fire-and-forget 注册

```rust
// pegaflow-core/src/internode/metaserver_client.rs:102
pub(crate) fn try_register(&self, entries: Vec<(String, Vec<u8>)>) {
    let batch = RegistrationBatch { entries };
    match self.insert_tx.try_send(batch) {
        Ok(()) => {
            core_metrics().metaserver_registration_blocks.add(count as u64, &[]);
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            // 队列满时直接丢弃（容忍注册延迟，不影响写路径）
            warn!("MetaServer registration queue full, dropping {} hashes", count);
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            // 后台 task 已退出（正常关机流程）
        }
    }
}
```

**为什么注册可以丢弃？**

MetaServer 是辅助索引（best-effort），不是写路径的核心。即使某个 block 未注册到 MetaServer，也只是其他节点无法通过 MetaServer 发现它（不影响本节点的功能）。

### 2.3 registration_loop — 后台注册任务

```rust
async fn registration_loop(
    mut rx: mpsc::Receiver<RegistrationBatch>,
    metaserver_addr: String,
    advertise_addr: String,
) {
    let mut backoff_ms = INITIAL_BACKOFF_MS;  // 100ms
    
    loop {
        // 建立 gRPC 连接（带重试）
        let client = loop {
            match connect(&metaserver_addr).await {
                Ok(c) => { backoff_ms = INITIAL_BACKOFF_MS; break c; }
                Err(e) => {
                    warn!("MetaServer connect failed: {e}, retry in {backoff_ms}ms");
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);  // 指数退避
                }
            }
        };
        
        // 接收批次并发送
        while let Some(batch) = rx.recv().await {
            // 按 namespace 分组，每个 namespace 一个 gRPC 请求
            let mut by_namespace: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
            for (ns, hash) in batch.entries {
                by_namespace.entry(ns).or_default().push(hash);
            }
            
            for (namespace, hashes) in by_namespace {
                let request = InsertBlockHashesRequest {
                    namespace,
                    block_hashes: hashes,
                    node: advertise_addr.clone(),
                };
                if let Err(e) = client.insert_block_hashes(request).await {
                    warn!("MetaServer insert failed: {e}");
                    break;  // 连接断了，重新建立
                }
            }
        }
    }
}
```

**指数退避（Exponential Backoff）**：

```
首次失败：等待 100ms
第 2 次：等待 200ms
第 3 次：等待 400ms
...
最大：等待 30000ms（30 秒）

目的：避免雪崩效应（MetaServer 重启时所有节点同时重连）
```

### 2.4 直接查询

```rust
pub(crate) async fn query(
    &self,
    namespace: &str,
    hashes: &[Vec<u8>],
) -> Result<Vec<NodeBlockHashes>, ClientError> {
    let response = self.query_client.clone()
        .query_block_hashes(QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: hashes.to_vec(),
        })
        .await
        .map_err(|e| ClientError::RpcFailed(e.to_string()))?
        .into_inner();
    
    // 返回按节点分组的结果
    Ok(response.node_blocks)
}
```

注意 `self.query_client.clone()`：Tonic 的 `EngineClient<Channel>` clone 是廉价的（Arc 语义），多个并发查询可以共享同一个 HTTP/2 连接。

---

## 3. MetaServer 服务端

### 3.1 BlockHashStore（moka 缓存）

```rust
// pegaflow-metaserver/src/store.rs:22
pub struct BlockHashStore {
    cache: Arc<Cache<BlockKey, Arc<str>>>,
    // BlockKey（namespace + hash） → 节点地址
}
```

使用 [moka](https://github.com/moka-rs/moka) crate（高性能并发缓存，类似 Java Caffeine）：

```rust
pub fn with_capacity_and_ttl(max_capacity_bytes: u64, ttl_minutes: u64) -> Self {
    let cache = Cache::builder()
        .max_capacity(max_capacity_bytes)   // 默认 512MB
        .weigher(|key: &BlockKey, node: &Arc<str>| {
            // 按实际内存占用计算权重（而非条目数）
            (key.estimated_size() + node.len() as u64 + 16) as u32
        })
        .time_to_live(Duration::from_secs(ttl_minutes * 60))  // 默认 120 分钟
        .build();
    
    Self { cache: Arc::new(cache) }
}
```

**大小感知淘汰（Size-Aware Eviction）**：

标准 LRU 按条目数计算容量（每个条目权重相同）。moka 的 weigher 允许按实际大小（字节数）计算，使缓存容量更精确。

### 3.2 insert_hashes()

```rust
pub async fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
    let node: Arc<str> = Arc::from(node);  // 节点地址用 Arc<str> 共享（节省内存）
    let mut inserted = 0;
    for hash in hashes {
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        self.cache.insert(key, Arc::clone(&node)).await;
        inserted += 1;
    }
    inserted
}
```

**节点覆盖**：如果同一 block hash 被多个节点注册，后来的覆盖之前的（只记录最新注册的节点）。这是可接受的，因为任意一个持有该 block 的节点都可以服务请求。

### 3.3 query_hashes()

```rust
pub async fn query_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<CrossNodeBlock> {
    let mut existing = Vec::new();
    for hash in hashes {
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        if let Some(node) = self.cache.get(&key).await {
            existing.push(CrossNodeBlock { block_hash: hash.clone(), node });
        }
    }
    existing
}
```

注意：查询**不是**前缀语义——每个 hash 独立查询，找到的返回，找不到的跳过。这与 ReadCache 的 `get_blocks()` 语义一致（跨节点 transfer 不需要连续前缀）。

---

## 4. TTL 的必要性

**问题**：KV cache 在内存中有 LRU 驱逐，但 MetaServer 不知道某个 block 何时被驱逐。如果 block 在内存中已被驱逐，MetaServer 还记录着它，会导致无效的 RDMA 请求。

**解决方案**：TTL（Time To Live，默认 120 分钟）

```
t=0:     节点 A 写入 block H → 注册到 MetaServer
t=30min: 节点 A 的 block H 被 LRU 驱逐（内存不足）
t=60min: 节点 B 查询 H → MetaServer 返回 {A: [H]}
t=60min: 节点 B 向 A 发起 QueryBlocksForTransfer
         A 的 ReadCache 找不到 H
         A 返回空结果
         B 标记 remote_fetch 失败（failed_remote）

t=120min: MetaServer TTL 到期，H 被自动清除
```

TTL 确保最终一致性：即使 MetaServer 数据暂时过时，120 分钟后自动清除，不会永久占用索引空间。

---

## 5. 节点发现（Kubernetes 环境）

除了 MetaServer 的基于 block hash 的发现，PegaFlow 还支持基于 Kubernetes Pod 的节点发现（`service_discovery.rs`）：

```
Kubernetes API Server
        │
        ▼
service_discovery.rs（Watch Pods）
        │ 发现新的 PegaFlow Pod
        ▼
注册到本地节点列表
        │
        ▼
P/D 模式：Prefill 节点将 KV 传输给 Decode 节点
```

这用于 P/D 解耦（Prefill-Decode Disaggregation）场景，是更进阶的部署模式，与 MetaServer 模式互补。

---

## 6. 组件关系

```
跨节点数据流：

本节点                              MetaServer              远端节点
  │                                    │                       │
  │ [insert_worker seal block B]       │                       │
  │ MetaServerClient::try_register(B)  │                       │
  │──────── InsertBlockHashes(B) ─────→│ store: {B → "node-A"} │
  │                                    │                       │
  │ [QueryPrefetch B miss locally]     │                       │
  │ RdmaFetchStore::submit_remote_fetch(B)                     │
  │──────── QueryBlockHashes(B) ──────→│                       │
  │←─────── [{node: "node-B", hashes: [B]}]                   │
  │                                    │                       │
  │ [选择最优节点：node-B]              │                       │
  │──────── QueryBlocksForTransfer(B) ─────────────────────→  │
  │←─────── {blocks: [{B, slots}], session_id, timeout} ──── │
  │                                    │                       │
  │═════════ RDMA READ ════════════════════════════════════→  │
  │                                    │                       │
  │──────── ReleaseTransferLock ───────────────────────────→  │
```
