# 16 — gRPC 服务（pegaflow-server）

**核心文件**：
- `pegaflow-server/src/service.rs`（~859 行）— gRPC 服务实现
- `pegaflow-server/src/registry.rs` — CUDA Tensor 注册表
- `pegaflow-server/src/http_server.rs` — HTTP 健康检查 + Prometheus 指标

---

## 1. GrpcEngineService 结构

```rust
// pegaflow-server/src/service.rs:24
#[derive(Clone)]
pub struct GrpcEngineService {
    engine: Arc<PegaEngine>,                               // 核心引擎
    registry: Arc<Mutex<CudaTensorRegistry>>,              // CUDA IPC 注册表
    shutdown: Arc<Notify>,                                 // 关机信号
    hll_tracker: Arc<std::sync::Mutex<HllTracker>>,       // 命中率估计器
}
```

`GrpcEngineService` 实现了 `Engine` trait（由 protobuf 代码生成），所有 gRPC handler 都在这里实现。

---

## 2. 错误映射策略

```rust
// pegaflow-server/src/service.rs:57
fn map_engine_error(err: EngineError) -> Status {
    match err {
        // 参数错误 → 客户端应修复请求
        EngineError::InvalidArgument(_) => 
            Status::invalid_argument(err.to_string()),
        
        // 实例/Worker 未找到 → 客户端应先注册
        EngineError::InstanceMissing(_) | EngineError::WorkerMissing(_, _) =>
            Status::failed_precondition(err.to_string()),
        
        // 拓扑不匹配 → 配置问题
        EngineError::TopologyMismatch(_) => 
            Status::failed_precondition(err.to_string()),
        
        // 服务端内部错误
        EngineError::CudaInit(_) | EngineError::Storage(_) | EngineError::Poisoned(_) =>
            Status::internal(err.to_string()),
    }
}
```

| EngineError | gRPC Status | 含义 |
|------------|-------------|------|
| InvalidArgument | INVALID_ARGUMENT | 请求参数错误，客户端问题 |
| InstanceMissing/WorkerMissing | FAILED_PRECONDITION | 需要先调用 RegisterContext |
| TopologyMismatch | FAILED_PRECONDITION | TP rank/size 配置不一致 |
| CudaInit/Storage | INTERNAL | 服务端内部错误 |

---

## 3. RegisterContextBatch — 注册 GPU KV cache 上下文

```rust
#[async_trait]
impl Engine for GrpcEngineService {
    async fn register_context_batch(
        &self,
        request: Request<RegisterContextRequest>,
    ) -> Result<Response<RegisterContextResponse>, Status> {
        let req = request.into_inner();
        
        // 通过 Python（PyO3）打开 CUDA IPC handle
        // wrapper_bytes 是 vLLM 传来的 CUDA IPC handle 序列化字节
        let registrations = Python::attach(|py| {
            self.registry.lock().register_context(py, &req)
        }).map_err(|e| Self::map_py_error("register_context", e))?;
        
        // 注册到 PegaEngine（每个 layer 的内存布局）
        self.engine.register_context_layer_batch(
            &req.instance_id,
            &req.namespace,
            req.tp_rank as usize,
            req.tp_size as usize,
            req.world_size as usize,
            req.device_id,
            req.num_layers as usize,
            &req.layer_names,
            registrations,  // KVCacheRegistration 列表
        ).map_err(Self::map_engine_error)?;
        
        Ok(Response::new(RegisterContextResponse { status: Some(Self::ok_status()) }))
    }
}
```

**CUDA IPC 打开（Python 侧）**：

`wrapper_bytes` 是 vLLM Python 代码序列化的 CUDA IPC handle（`cudaIpcGetMemHandle`），PegaFlow 用 PyO3 调用 Python 代码的 `torch.cuda._shared_memory` 来打开它，获得 GPU 内存基地址。

---

## 4. Save — GPU → CPU KV cache 卸载

```rust
async fn save(&self, request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
    let req = request.into_inner();
    
    let saves: Vec<LayerSave> = req.saves.iter().map(|s| LayerSave {
        layer_name: s.layer_name.clone(),
        block_ids: s.block_ids.clone(),
        block_hashes: s.block_hashes.clone(),
    }).collect();
    
    // 异步：触发 CUDA D2H 传输，等待完成后将数据放入写管道
    self.engine.batch_save_kv_blocks_from_ipc(
        &req.instance_id,
        req.tp_rank as usize,
        req.device_id,
        saves,
    ).await.map_err(Self::map_engine_error)?;
    
    // gRPC 返回（数据已在 CPU 内存中，异步写入 SSD 和 MetaServer）
    Ok(Response::new(SaveResponse { status: Some(Self::ok_status()) }))
}
```

---

## 5. Query/QueryPrefetch — 查询命中 + 预取

```rust
async fn query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
    let req = request.into_inner();
    
    // 获取 namespace（从实例 ID 到命名空间的映射）
    let namespace = self.engine.get_namespace(&req.instance_id)
        .map_err(Self::map_engine_error)?;
    
    let (hit, missing) = self.engine.count_prefix_hit_blocks(
        &namespace,
        &req.block_hashes,
    );
    
    // 记录 HLL 命中率指标
    if !req.block_hashes.is_empty() {
        self.hll_tracker.lock().unwrap().record_hit(&req.block_hashes[..hit.min(1)]);
    }
    
    Ok(Response::new(QueryResponse {
        status: Some(Self::ok_status()),
        hit_blocks: hit as u64,
        missing_blocks: missing as u64,
        prefetch_state: PrefetchState::PrefetchDone as i32,
        loading_blocks: 0,
    }))
}

async fn query_prefetch(&self, request: Request<QueryRequest>) -> ... {
    let req = request.into_inner();
    let namespace = ...;
    let num_workers = self.engine.get_num_workers(&req.instance_id, ...)? as usize;
    
    let status = self.engine.count_prefix_hit_blocks_with_prefetch(
        &req.instance_id,
        &req.req_id,
        &namespace,
        &req.block_hashes,
        num_workers,
    ).await;
    
    let (hit, loading, missing, prefetch_state) = match status {
        PrefetchStatus::Loading { hit, loading } => 
            (hit, loading, 0, PrefetchState::PrefetchLoading),
        PrefetchStatus::Done { hit, missing } => 
            (hit, 0, missing, PrefetchState::PrefetchDone),
    };
    
    Ok(Response::new(QueryResponse {
        hit_blocks: hit as u64,
        loading_blocks: loading as u64,
        missing_blocks: missing as u64,
        prefetch_state: prefetch_state as i32,
        ...
    }))
}
```

---

## 6. QueryBlocksForTransfer — 跨节点传输

```rust
async fn query_blocks_for_transfer(
    &self,
    request: Request<QueryBlocksForTransferRequest>,
) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
    let req = request.into_inner();
    
    // 查找块 + 加锁（防止 RDMA 期间被 LRU 驱逐）
    let (locked_blocks, session_id) = self.engine.query_blocks_for_transfer(
        &req.namespace,
        &req.block_hashes,
        &req.requester_id,
    );
    
    // 构建 TransferBlockInfo（每个 block 的每个 slot 的内存地址）
    let blocks: Vec<TransferBlockInfo> = locked_blocks.iter().map(|(key, sealed)| {
        let slots: Vec<TransferSlotInfo> = sealed.slots().iter().zip(sealed.slot_numas()).map(|(raw_block, numa)| {
            Self::build_transfer_slot_info(raw_block, numa)
        }).collect();
        
        TransferBlockInfo {
            block_hash: key.hash.clone(),
            slots,
        }
    }).collect();
    
    Ok(Response::new(QueryBlocksForTransferResponse {
        blocks,
        transfer_session_id: session_id,
        lock_timeout_secs: self.engine.transfer_lock_timeout_secs(),
        ...
    }))
}
```

### build_transfer_slot_info() — 构建内存描述符

```rust
fn build_transfer_slot_info(
    raw_block: &Arc<RawBlock>,
    numa_node: NumaNode,
) -> TransferSlotInfo {
    let layer_block = LayerBlock::new(Arc::clone(raw_block));
    
    if let Some(v_ptr) = layer_block.v_ptr() {
        // 分段存储：K 和 V 在不同内存区域
        TransferSlotInfo {
            k_ptr: layer_block.k_ptr() as u64,  // K 段虚拟地址
            k_size: layer_block.k_size() as u64,
            v_ptr: v_ptr as u64,                // V 段虚拟地址
            v_size: layer_block.v_size().unwrap_or(0) as u64,
            numa_node: numa_node.0,
        }
    } else {
        // 连续存储：K+V 在同一块内存，v_ptr = 0 表示连续
        TransferSlotInfo {
            k_ptr: layer_block.k_ptr() as u64,
            k_size: layer_block.k_size() as u64,
            v_ptr: 0,   // 表示连续存储
            v_size: 0,
            numa_node: numa_node.0,
        }
    }
}
```

---

## 7. RdmaHandshake — RDMA 握手

```rust
async fn rdma_handshake(&self, request: Request<RdmaHandshakeRequest>) -> ... {
    let req = request.into_inner();
    
    let metadata = self.engine.rdma_accept_handshake(
        &req.requester_id,
        &req.handshake_metadata,
    ).map_err(|e| Status::internal(e))?;
    
    Ok(Response::new(RdmaHandshakeResponse {
        handshake_metadata: metadata,  // bincode 序列化的本地 NicHandshake
    }))
}
```

握手采用"服务端连接重用"模式：
- 如果 `req.handshake_metadata` **非空**：执行完整握手（创建 RC QP）
- 如果 `req.handshake_metadata` **为空**：返回本地元数据（复用现有连接）

---

## 8. HTTP 服务

```
/metrics     → Prometheus 格式的指标（用于监控系统抓取）
/health      → {"status": "ok"}（用于 Kubernetes liveness probe）
/ready       → 就绪检查（用于 Kubernetes readiness probe）
```

**指标示例**：
```
# HELP pegaflow_cache_block_hits_total Total number of cache hit blocks
# TYPE pegaflow_cache_block_hits_total counter
pegaflow_cache_block_hits_total 12345

# HELP pegaflow_cache_resident_bytes Bytes currently in the read cache
# TYPE pegaflow_cache_resident_bytes gauge
pegaflow_cache_resident_bytes 3.145728e+10

# HELP pegaflow_pool_alloc_failures_total Number of failed allocation attempts
# TYPE pegaflow_pool_alloc_failures_total counter
pegaflow_pool_alloc_failures_total 0
```

---

## 9. 线程安全说明

`GrpcEngineService` 实现了 `Clone`（`#[derive(Clone)]`），Tonic gRPC 框架会为每个请求 clone 一个 service 实例。所有字段都是 `Arc<T>` 或 `Arc<Mutex<T>>`，clone 只是增加引用计数，代价极低。

```
tokio 线程池中的多个线程
    ├── 线程 1 处理 Save RPC → GrpcEngineService.clone() → Arc<PegaEngine>
    ├── 线程 2 处理 Query RPC → GrpcEngineService.clone() → Arc<PegaEngine>
    └── 线程 3 处理 Load RPC → GrpcEngineService.clone() → Arc<PegaEngine>
                                           ↕ 共享同一个 PegaEngine 实例
```
