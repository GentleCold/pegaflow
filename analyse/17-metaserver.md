# 17 — MetaServer（跨节点块哈希注册中心）

**核心文件**：
- `pegaflow-metaserver/src/store.rs`（250 行）— 块哈希存储
- `pegaflow-metaserver/src/service.rs` — gRPC 服务实现
- `pegaflow-metaserver/src/lib.rs` — MetaServer 入口和 CLI

---

## 1. MetaServer 的作用

MetaServer 是 PegaFlow 集群中的**独立服务**，维护"哪个节点有哪些块"的全局索引：

```
节点 A                    MetaServer                  节点 B
  │ Save block H          │                              │
  │──InsertBlockHashes(H)─→│ store: {H → "node-A"}        │
  │                       │                              │
  │                       │                   Save block H
  │                       │←─InsertBlockHashes(H)─────── │
  │                       │ store: {H → "node-B"} (覆盖)  │
  │                       │                              │
节点 C 查询 H:            │                              │
  │──QueryBlockHashes(H)──→│                              │
  │←─{node: "node-B", [H]} │                              │
  │                       │                              │
  │── RDMA → 节点 B ──────────────────────────────────→  │
```

---

## 2. BlockHashStore — moka 缓存

```rust
// pegaflow-metaserver/src/store.rs:22
pub struct BlockHashStore {
    cache: Arc<Cache<BlockKey, Arc<str>>>,
    // Key: BlockKey（namespace + hash）
    // Value: 节点地址（如 "10.0.0.1:50055"）
}
```

**moka 缓存特性**：
- **异步**（`moka::future::Cache`）：`insert` 和 `get` 都是 async，适合 tokio
- **大小感知**（size-aware）：用 `weigher` 按实际内存占用分配容量
- **TTL**：每个条目 120 分钟后自动过期
- **LRU**：容量满时驱逐最久未使用的条目

```rust
let cache = Cache::builder()
    .max_capacity(512 * 1024 * 1024)   // 512MB
    .weigher(|key: &BlockKey, node: &Arc<str>| {
        (key.estimated_size() + node.len() as u64 + 16) as u32
        // key.estimated_size() = namespace.capacity + hash.capacity + 48
        // node.len() = 节点地址字节数（如 "10.0.0.1:50055" = 15 字节）
        // +16 = Arc 开销
    })
    .time_to_live(Duration::from_secs(120 * 60))
    .build();
```

### 插入

```rust
pub async fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
    let node: Arc<str> = Arc::from(node);  // 共享同一节点字符串（引用计数，节省内存）
    for hash in hashes {
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        self.cache.insert(key, Arc::clone(&node)).await;
    }
    hashes.len()
}
```

### 查询

```rust
pub async fn query_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<CrossNodeBlock> {
    let mut existing = Vec::new();
    for hash in hashes {
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        if let Some(node) = self.cache.get(&key).await {
            existing.push(CrossNodeBlock { block_hash: hash.clone(), node });
        }
        // 注意：没有前缀语义，找不到就跳过，不停止
    }
    existing
}
```

### 维护任务

```rust
// moka 的惰性淘汰：不立即删除过期条目，需要显式触发
pub async fn run_pending_tasks(&self) {
    self.cache.run_pending_tasks().await;
}
// 应定期调用，确保 TTL 到期条目被清理
```

---

## 3. MetaServer gRPC 服务

MetaServer 实现了 `MetaServer` trait（protobuf 生成），提供 4 个 RPC：

### InsertBlockHashes

```rust
async fn insert_block_hashes(&self, request: ...) -> ... {
    let req = request.into_inner();
    let inserted = self.store.insert_hashes(
        &req.namespace,
        &req.block_hashes,
        &req.node,
    ).await;
    
    Ok(Response::new(InsertBlockHashesResponse {
        status: Some(ok_status()),
        inserted_count: inserted as u64,
    }))
}
```

### QueryBlockHashes

```rust
async fn query_block_hashes(&self, request: ...) -> ... {
    let req = request.into_inner();
    let found = self.store.query_hashes(&req.namespace, &req.block_hashes).await;
    
    // 按节点分组结果
    let mut node_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
    for cb in &found {
        node_map.entry(cb.node.to_string())
            .or_default()
            .push(cb.block_hash.clone());
    }
    
    let node_blocks: Vec<NodeBlockHashes> = node_map.into_iter().map(|(node, hashes)| {
        NodeBlockHashes { node, block_hashes: hashes }
    }).collect();
    
    Ok(Response::new(QueryBlockHashesResponse {
        existing_hashes: found.iter().map(|cb| cb.block_hash.clone()).collect(),
        found_count: found.len() as u64,
        total_queried: req.block_hashes.len() as u64,
        node_blocks,
        ...
    }))
}
```

---

## 4. MetaServer CLI

MetaServer 以独立进程运行，提供命令行界面：

```bash
# 启动 MetaServer
pegaflow-metaserver \
    --bind 0.0.0.0:50056 \
    --capacity 536870912 \    # 512MB
    --ttl-minutes 120
```

**单点 vs 高可用**：

当前实现是单点 MetaServer（单个进程）。对于高可用需求，可以在 MetaServer 前面加 L7 负载均衡（数据可容忍短暂不一致，因为 TTL 最终会清理过期条目）。

---

## 5. 为什么 MetaServer 不是强一致的？

PegaFlow 的 MetaServer 设计为**最终一致**（eventually consistent）：

1. **写入延迟**：block 被写入内存后，通过异步 fire-and-forget 注册到 MetaServer，可能有延迟
2. **节点覆盖**：同一 block 可能被多个节点注册，后来的会覆盖之前的
3. **TTL 过期**：block 被 LRU 驱逐后，MetaServer 的记录还会存在 120 分钟

这些不一致都是可接受的：
- **读取时验证**：RDMA 传输前会通过 `QueryBlocksForTransfer` 验证 block 是否还在
- **失败处理**：如果远端 block 不存在，`RdmaFetchStore` 会记录 `failed_remote`，下次不再重试
- **降级**：RDMA 失败只是缓存 miss，推理系统会降级处理（重新计算或仅使用本地缓存）
