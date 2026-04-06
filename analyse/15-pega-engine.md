# 15 — PegaEngine 主入口

**核心文件**：
- `pegaflow-core/src/lib.rs`（~986 行）— 主引擎及 Save/Load/Query 流程
- `pegaflow-core/src/offload.rs`（~537 行）— Save 路径：分配 + GPU 拷贝 + 批量构建

---

## 1. PegaEngine 结构

```rust
// pegaflow-core/src/lib.rs:133
pub struct PegaEngine {
    /// 活跃实例注册表（instance_id → InstanceContext）
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    /// 存储引擎（固定内存 + 缓存 + SSD + RDMA 等所有子系统）
    storage: Arc<StorageEngine>,
    /// GPU-NUMA 拓扑（用于选择最优 NUMA 节点分配内存）
    topology: Arc<NumaTopology>,
}
```

**初始化流程**：

```rust
// lib.rs:158
pub fn new_with_config(pool_size, use_hugepages, storage_config) -> Self {
    let topology = Arc::new(NumaTopology::detect());  // 探测 NUMA 拓扑
    topology.log_summary();                           // 打印 NUMA 信息到日志

    // 如果启用 NUMA 亲和性 且 多 NUMA 节点 → 按节点创建多个固定内存池
    let numa_nodes: Vec<NumaNode> = if config.enable_numa_affinity && topology.is_multi_numa() {
        topology.numa_nodes().to_vec()  // 例：[NUMA(0), NUMA(1)]
    } else {
        vec![]  // 单 NUMA 或禁用 → 统一内存池
    };

    let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes);
    PegaEngine { instances: RwLock::new(HashMap::new()), storage, topology }
}
```

**`EngineError` 错误枚举**（`lib.rs:83`）：

| 变体 | 含义 |
|------|------|
| `InstanceMissing(String)` | instance_id 未注册 |
| `WorkerMissing(String, i32)` | 该 device 未注册 |
| `InvalidArgument(String)` | 参数错误 |
| `CudaInit(String)` | CUDA 初始化失败 |
| `Storage(String)` | 存储引擎错误 |
| `Poisoned(&'static str)` | Mutex 被毒化（线程 panic） |
| `TopologyMismatch(String)` | TP 拓扑与现有实例不一致 |

---

## 2. RegisterContext — 注册 GPU KV cache 上下文

```
vLLM Worker RegisterContextBatch RPC
        │
        ▼ pegaflow-server/registry.rs（打开 CUDA IPC handle）
        │
        ▼ PegaEngine::register_context_layer_batch()
        │
        ├── get_or_create_instance()  ← 首次注册时创建 InstanceContext
        ├── topology.numa_for_gpu(device_id)  ← 查 GPU 所在 NUMA 节点
        └── instance.register_new_gpu(device_id, numa_node, kv_caches)
```

**`register_context_layer_batch()` 源码解析**（`lib.rs:242`）：

```rust
pub fn register_context_layer_batch(
    &self,
    instance_id, namespace, device_id, tp_rank, tp_size, world_size, num_layers,
    layer_names, data_ptrs, size_bytes_list, num_blocks_list,
    bytes_per_block_list, kv_stride_bytes_list, segments_list,
) -> Result<(), EngineError> {
    // 1. 校验 device_id >= 0
    if device_id < 0 { return Err(InvalidArgument(...)); }

    // 2. 为每个 layer 构建 KVCacheRegistration
    let ssd_enabled = self.storage.is_ssd_enabled();
    for i in 0..batch_size {
        let mut registration = KVCacheRegistration::new(
            data_ptrs[i],        // GPU 虚拟地址（由 CUDA IPC 打开得到）
            size_bytes_list[i],  // 总字节数
            num_blocks_list[i],  // block 数量
            bytes_per_block_list[i],  // 每个 block 字节数（非填充）
            kv_stride_bytes_list[i],  // K→V 步长（分段存储用）
            segments_list[i],         // 1（连续）或 2（K/V 分段）
        )?;

        // 3. 如果启用了 SSD，对 bytes_per_block 进行 SSD 对齐填充
        if ssd_enabled {
            registration = registration.with_ssd_padding(SSD_ALIGNMENT);
        }

        kv_caches.insert(layer_name.clone(), registration);
    }

    // 4. 获取或创建 InstanceContext（检查 TP 拓扑一致性）
    let instance = self.get_or_create_instance(instance_id, namespace, num_layers, tp_size, world_size)?;

    // 5. 查询 GPU 所在 NUMA 节点
    let numa_node = self.topology.numa_for_gpu(device_id);

    // 6. 向 InstanceContext 注册新 GPU
    instance.register_new_gpu(device_id, numa_node, kv_caches)
}
```

**`get_or_create_instance()` 幂等性**（`lib.rs:191`）：

```rust
fn get_or_create_instance(&self, instance_id, namespace, num_layers, tp_size, world_size)
    -> Result<Arc<InstanceContext>, EngineError>
{
    // 写锁保护（防止并发创建）
    let mut instances = self.instances.write().expect("...");
    
    if let Some(instance) = instances.get(instance_id) {
        // 已存在：验证拓扑（num_layers/tp_size/world_size 必须一致）
        instance.verify_topology(num_layers, tp_size, world_size)
            .map_err(|e| EngineError::TopologyMismatch(...))?;
        return Ok(Arc::clone(instance));
    }
    
    // 不存在：创建新实例
    let instance = Arc::new(InstanceContext::new(...)?);
    instances.insert(instance_id.to_string(), Arc::clone(&instance));
    Ok(instance)
}
```

---

## 3. Save 路径：`batch_save_kv_blocks_from_ipc()`

Save 路径分 4 个 Phase，核心在 `offload.rs`：

```
vLLM Worker save() RPC
        │
        ▼ PegaEngine::batch_save_kv_blocks_from_ipc()
              │
              ├── Phase 0: 解析元数据（layer 信息、slot_id）
              ├── Phase 1: Hash 过滤（已在缓存中的 block 跳过）
              ├── Phase 2: 分配固定内存（K/V 分段 or 连续）
              ├── Phase 3: 提交 GPU → CPU 拷贝（等待完成）
              └── Phase 4: 构建 RawSaveBatch → 发送给 insert worker（fire-and-forget）
```

### Phase 0：解析元数据

```rust
// offload.rs:196
let gpu = instance.get_gpu(device_id)?;  // 查找已注册的 GPU 上下文

for LayerSave { layer_name, block_ids, block_hashes } in saves {
    let layer_id = instance.get_layer_id(&layer_name)?;
    let registration = gpu.get_registration(&layer_name)?;
    let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
    
    // 过滤无效 block_id（< 0 或越界）
    let blocks_to_save: Vec<(usize, Vec<u8>)> = block_ids.iter().zip(block_hashes)
        .filter(|(id, _)| *id >= 0 && *id < registration.num_blocks)
        .map(|(id, hash)| (*id as usize, hash))
        .collect();
}
```

### Phase 1：Hash 过滤

```rust
// offload.rs:269
// 所有 layer 的 hash 求并集
let mut hashes_to_save: HashSet<Vec<u8>> = HashSet::new();
for layer in &layers {
    for (_, hash) in &layer.blocks_to_save {
        hashes_to_save.insert(hash.clone());
    }
}

// 一次性过滤掉已在缓存中的 hash（避免重复写入）
self.storage.filter_hashes_not_in_cache_inplace(&namespace, &mut hashes_to_save);

// 如果所有 hash 都已缓存 → 跳过（零拷贝！）
if hashes_to_save.is_empty() { return Ok(()); }
```

**优化点**：先过滤再分配，避免为已缓存的 block 做无效的内存分配和 GPU 拷贝。

### Phase 2：分配固定内存

根据存储模式选择分配策略：

```
分段存储（segments=2, kv_stride_bytes > bytes_per_block）：
  K 段: 一次分配 num_blocks × padded_segment_size 字节
  V 段: 同上（两个独立的 PinnedAllocation）

连续存储（segments=1）：
  一次分配 num_blocks × padded_block_size 字节
```

```rust
// offload.rs:334
let is_split = registration.segments == 2
    && registration.kv_stride_bytes > registration.bytes_per_block;

if is_split {
    let k_allocation = self.storage.allocate(alloc_size, numa_node)?;
    let v_allocation = self.storage.allocate(alloc_size, numa_node)?;
    
    layer.allocs.push(LayerAlloc::Split {
        k_allocation, v_allocation,
        k_base, v_base,
        padded_segment_size,
    });
} else {
    let allocation = self.storage.allocate(alloc_size, numa_node)?;
    layer.allocs.push(LayerAlloc::Contiguous { allocation, base_addr });
}
```

**`blockwise_alloc` 选项**：每个 block 独立分配（避免碎片），默认批量分配（节省分配次数）。

### Phase 3：GPU → CPU 拷贝

```rust
// offload.rs:474
// 将所有 layer 的 SaveBlock 批量提交给 GPU worker（单次 CUDA stream 同步）
gpu.worker_pool().batch_save(gpu_save_layers).await?;
// ↑ 等待所有 cudaMemcpyAsync 完成（SaveTask + oneshot 通道）
```

gRPC handler 在这里等待完成，完成后数据已在 CPU 固定内存中。

### Phase 4：发送给 insert worker

```rust
// offload.rs:527
self.storage.send_raw_insert(RawSaveBatch {
    namespace,
    total_slots,
    numa_node: gpu.preferred_numa(),
    layers: raw_layers,  // 包含 PinnedAllocation 引用 + block hashes
});
// fire-and-forget：立即返回，insert worker 在后台处理
```

**insert worker 收到 `RawSaveBatch` 后**（`write_path.rs`）：
1. `build_insert_entries()` — 构建 `(BlockKey → Vec<(slot_id, Arc<RawBlock>)>)` 映射
2. 组装 `InflightBlock` → 等待所有 TP slot → seal → 插入 ReadCache
3. SSD 异步写入
4. MetaServer 注册

---

## 4. Load 路径：`batch_load_kv_blocks_multi_layer()`

Load 路径是 **fire-and-forget** 模式：gRPC handler 立即返回，vLLM Worker 轮询共享内存等待完成。

```
vLLM Worker load() RPC
        │
        ▼ PegaEngine::batch_load_kv_blocks_multi_layer()
              │
              ├── LoadState::attach(shm_name)  ← 附加到已存在的共享内存
              │
              ├── consume_pinned_blocks()  ← 从 ReadCache 消费 pin 的 block
              │   （阻止 LRU 驱逐：已在 query_prefetch 阶段 pin 住）
              │
              ├── 为每个 layer 构建 LayerLoadData
              │   layer_name → registration（GPU 内存布局）
              │   block_id  → slot_id → Arc<RawBlock>（CPU 固定内存地址）
              │
              └── gpu.worker_pool().submit_load(LoadTask)  ← 非阻塞提交

        Load RPC 立即返回 Ok
              │
              ▼ 后台（OS 线程：gpu{N}-load）
              ├── process_load_task()
              │   批量 cudaMemcpyAsync（K 段全部 → V 段全部）
              └── stream.synchronize()
                  LoadState::set_completed()  ← 写共享内存标志

vLLM Worker Python 侧轮询:
while not load_state.is_completed(): sleep(1ms)
```

**`batch_load_kv_blocks_multi_layer_inner()` 核心逻辑**（`lib.rs:492`）：

```rust
fn batch_load_kv_blocks_multi_layer_inner(&self, ...) -> Result<(), EngineError> {
    // 1. 查找 GPU 上下文
    let gpu = instance.get_gpu(device_id)?;
    
    // 2. 消费 pin 的 block（查 ReadCache 获取 Arc<SealedBlock>）
    let block_cache = self.storage.consume_pinned_blocks(instance_id, namespace, block_hashes)?;
    // ↑ 返回 Vec<Option<Arc<SealedBlock>>>，对应 block_hashes 中的每个 hash
    
    // 3. 为每个 layer 构建 LoadBlock 列表
    for layer_name in layer_names {
        let layer_id = instance.get_layer_id(layer_name)?;
        let registration = gpu.get_registration(layer_name)?;
        let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
        
        let blocks: Vec<LoadBlock> = block_ids.iter().zip(block_cache.iter())
            .filter_map(|(block_id, block_entry)| {
                let block_idx = usize::try_from(*block_id).ok()?;
                let block = block_entry.get_slot(slot_id)?.clone();
                // ↑ 从 SealedBlock 中取出对应 slot（TP rank）的 RawBlock
                Some(LoadBlock { block_idx, block })
            })
            .collect();
        
        if !blocks.is_empty() {
            layers.push(LayerLoadData { layer_name, registration, blocks });
        }
    }
    
    // 4. 如果没有需要 Load 的 block → 直接标记完成
    if layers.is_empty() {
        LoadState::attach(load_state_shm)?.set_completed();
        return Ok(());
    }
    
    // 5. 提交到 GPU worker（非阻塞）
    gpu.worker_pool().submit_load(LoadTask { layers, load_state_shm })
}
```

---

## 5. Query 路径

### 纯内存查询（Query RPC）

```rust
// lib.rs:366
pub fn count_prefix_hit_blocks(
    &self,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
) -> Result<(usize, usize), EngineError> {
    let instance = self.get_instance(instance_id)?;
    let namespace = instance.namespace();
    
    let (hit, missing) = self.storage.check_prefix_memory_only(namespace, block_hashes);
    
    // 记录 Prometheus 指标
    core_metrics().cache_block_hits.add(hit as u64, &[]);
    core_metrics().cache_block_misses.add(missing as u64, &[]);
    
    Ok((hit, missing))
}
```

- **前缀语义**：从 `block_hashes[0]` 开始，遇到第一个 miss 就停止
- **无副作用**：不 pin block，不触发预取
- **用途**：vLLM `query` RPC（轻量级，不涉及 Load 路径）

### 带预取的查询（QueryPrefetch RPC）

```rust
// lib.rs:397
pub async fn count_prefix_hit_blocks_with_prefetch(
    &self,
    instance_id: &str,
    req_id: &str,     // 每个请求唯一 ID（用于 PrefetchScheduler 跟踪状态）
    block_hashes: &[Vec<u8>],
) -> Result<PrefetchStatus, EngineError> {
    // 空 req_id 视为无效请求（防御性保护）
    if req_id.is_empty() {
        return Ok(PrefetchStatus::Done { hit: 0, missing: block_hashes.len() });
    }
    
    let instance = self.get_instance(instance_id)?;
    let namespace = instance.namespace();
    let world_size = instance.world_size();
    
    // 调用存储引擎（含预取逻辑）
    let status = self.storage
        .check_prefix_and_prefetch(instance_id, req_id, namespace, block_hashes, world_size)
        .await;
    
    // 记录指标
    match &status {
        PrefetchStatus::Done { hit, missing } => {
            core_metrics().cache_block_hits.add(*hit as u64, &[]);
            core_metrics().cache_block_misses.add(*missing as u64, &[]);
        }
        PrefetchStatus::Loading { hit, .. } => {
            core_metrics().cache_block_hits.add(*hit as u64, &[]);
        }
    }
    
    Ok(status)
}
```

**返回值语义**：

| `PrefetchStatus` | 含义 | 行动 |
|------------------|------|------|
| `Done { hit=N, missing=0 }` | 全部命中 | 直接 Load |
| `Loading { hit=N, loading=M }` | 预取进行中 | gRPC 返回 Loading，让 vLLM 稍后重试 |
| `Done { hit=N, missing=M }` | 部分 miss（无法预取）| Load 命中部分 |

---

## 6. 跨节点传输：`query_blocks_for_transfer()`

```rust
// lib.rs:586
pub fn query_blocks_for_transfer(
    &self,
    namespace: &str,
    block_hashes: &[Vec<u8>],
    requester_id: &str,
) -> (String, Vec<(BlockKey, Arc<SealedBlock>)>) {
    let keys: Vec<BlockKey> = block_hashes.iter()
        .map(|h| BlockKey::new(namespace.to_string(), h.clone()))
        .collect();
    
    // 1. 从 ReadCache 查找 block（非前缀语义，找哪个返回哪个）
    let found = self.storage.get_blocks_for_transfer(&keys);
    
    // 2. 为找到的 block 创建传输锁（防止 RDMA 期间被 LRU 驱逐）
    let session_id = self.storage.lock_blocks_for_transfer(requester_id, &found);
    // ↑ session_id 用于后续 release_transfer_lock()
    
    (session_id, found)
}
```

**RDMA 完整流程**：

```
远端节点 A 发起请求（正在预取 block H）
        │
        │ QueryBlocksForTransfer(namespace, [H], "node-A")
        ▼
本节点（持有 block H）
  get_blocks_for_transfer([H]) → [(BlockKey, Arc<SealedBlock>)]
  lock_blocks_for_transfer("node-A", found) → session_id
        │
        │ 返回 {blocks: [{H, slots: [{k_ptr, k_size, v_ptr, v_size, numa}]}], session_id}
        ▼
远端节点 A
  RDMA READ（将本节点内存内容读到远端固定内存）
        │
        │ ReleaseTransferLock(session_id)
        ▼
本节点
  release_transfer_lock(session_id)  ← 解锁，block 可以被 LRU 驱逐
```

---

## 7. RDMA 握手：`rdma_accept_handshake()`

```rust
// lib.rs:637
pub fn rdma_accept_handshake(
    &self,
    client_addr: &str,
    client_handshake_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let rdma = self.storage.rdma_transport()?;
    
    if client_handshake_bytes.is_empty() {
        // 客户端认为已连接 → 返回本地缓存的握手元数据（复用已有连接）
        return Ok(rdma.engine().local_meta_for(client_addr).map(|m| m.to_bytes()).unwrap_or_default());
    }
    
    // 客户端发来握手数据 → 建立新连接
    let client_meta = HandshakeMetadata::from_bytes(client_handshake_bytes)?;
    
    // 先作废旧连接（防止客户端重启导致 QP 状态不一致）
    rdma.engine().invalidate_connection(client_addr);
    
    // 准备本地 QP
    let server_meta = match rdma.engine().get_or_prepare(client_addr)? {
        ConnectionStatus::Prepared(m) => m,
        ConnectionStatus::Existing => unreachable!(),  // 刚作废，不可能已存在
        ConnectionStatus::Connecting => return Err("already in progress".into()),
    };
    
    // 完成握手：连接本地 QP 和客户端 QP
    rdma.engine().complete_handshake(client_addr, &server_meta, &client_meta)?;
    
    Ok(server_meta.to_bytes())  // 返回本地握手元数据
}
```

---

## 8. 实例生命周期

```
vLLM Worker 启动
        │
        │ RegisterContextBatch（每个 GPU 调用一次）
        ▼
PegaEngine::register_context_layer_batch()
  └── InstanceContext（首次创建）
      └── GpuContext（每个 GPU 一个）
              └── KVCacheRegistration（每个 Layer 一个）
              └── GpuWorkerPool（load/save 各一个 OS 线程）

推理服务运行中：
  Save（每次 forward 完成后）
  Load（每次 cache hit + prefetch 后）
  Query / QueryPrefetch（每次请求调度时）

vLLM Worker 退出
        │
        │ UnregisterContext
        ▼
PegaEngine::unregister_instance()
  └── 从 instances 移除（Arc 引用计数归零时自动释放 GpuContext/WorkerPool）
```

---

## 9. 组件依赖关系

```
PegaEngine
  ├── instances: RwLock<HashMap<String, Arc<InstanceContext>>>
  │   └── InstanceContext
  │       ├── layer_ids, slot_layout（TP rank → slot_id）
  │       └── gpus: HashMap<i32, Arc<GpuContext>>
  │           ├── KVCacheRegistration（每层 GPU 内存布局）
  │           └── GpuWorkerPool（CUDA save/load 线程）
  │
  └── storage: Arc<StorageEngine>（见文档 07）
      ├── 固定内存池（PinnedAllocator）
      ├── 读缓存（ReadCache + TinyLFU）
      ├── 预取调度（PrefetchScheduler：SSD + RDMA）
      ├── 写管道（WritePipeline + insert worker）
      ├── SSD 后端（SsdBackingStore + io_uring）
      └── RDMA 传输（RdmaTransport + RdmaFetchStore）
```
