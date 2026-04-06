# 13 — GPU Worker（CUDA IPC + 异步传输）

**核心文件**：
- `pegaflow-core/src/gpu_worker.rs`（~430 行）— GPU worker 线程
- `pegaflow-core/src/transfer.rs` — CUDA memcpy 操作
- `pegaflow-core/src/offload.rs` — Save 路径：GPU → CPU
- `pegaflow-core/src/sync_state.rs` — Load 完成通知（共享内存）

---

## 1. 为什么需要专用 GPU Worker 线程？

CUDA 操作有以下特性：
1. **线程上下文绑定**：每个 CPU 线程需要一个 CUDA 上下文，上下文切换开销大
2. **流（Stream）顺序**：同一 CUDA 流上的操作严格顺序执行，不同流可以并行
3. **异步传输**：`cudaMemcpyAsync` 是非阻塞的，需要 `cudaStreamSynchronize` 等待完成

PegaFlow 的设计：**每块 GPU 一个专用 OS 线程**（Load worker + Save worker），绑定到对应 NUMA 节点：

```
GPU 0 (NUMA 0)     GPU 1 (NUMA 1)
├── gpu0-load       ├── gpu1-load
└── gpu0-save       └── gpu1-save
```

---

## 2. GpuWorkerPool 结构

```rust
// pegaflow-core/src/gpu_worker.rs:65
pub struct GpuWorkerPool {
    device_id: i32,
    load_tx: mpsc::UnboundedSender<LoadTask>,  // 向 Load worker 发任务
    save_tx: mpsc::UnboundedSender<SaveTask>,  // 向 Save worker 发任务
}
```

**初始化**：

```rust
pub(crate) fn spawn(device_id: i32, numa_node: NumaNode) -> Result<Self, EngineError> {
    let (load_tx, load_rx) = mpsc::unbounded_channel();  // tokio 无界通道
    let (save_tx, save_rx) = mpsc::unbounded_channel();
    
    // 启动 Load worker 线程
    std::thread::Builder::new()
        .name(format!("gpu{}-load", device_id))
        .spawn(move || {
            if numa_node.is_valid() {
                pin_thread_to_numa_node(numa_node);  // 绑定到正确的 NUMA 节点
            }
            load_worker_loop(device_id, load_rx);  // 进入主循环
        })?;
    
    // 启动 Save worker 线程（类似）
    std::thread::Builder::new()
        .name(format!("gpu{}-save", device_id))
        .spawn(move || {
            if numa_node.is_valid() { pin_thread_to_numa_node(numa_node); }
            save_worker_loop(device_id, save_rx);
        })?;
    
    Ok(Self { device_id, load_tx, save_tx })
}
```

**为什么 Load 和 Save 各用一个线程？**

Load（CPU→GPU）和 Save（GPU→CPU）是反向传输，可以**并行执行**（PCIe 双向带宽）。分开两个线程 + 两个 CUDA 流，最大化 PCIe 利用率。

---

## 3. Save 路径：GPU → CPU

### SaveTask 结构

```rust
pub struct SaveTask {
    pub layers: Vec<SaveLayerData>,
    pub reply: oneshot::Sender<Result<(), EngineError>>,  // 完成后回复 gRPC handler
}

pub struct SaveLayerData {
    pub registration: KVCacheRegistration,  // GPU 缓冲区信息
    pub blocks: Vec<SaveBlock>,
}

pub struct SaveBlock {
    pub block_idx: usize,          // GPU 缓冲区中的物理槽位
    pub k_dst_ptr: *mut u8,        // 预分配的固定内存目标（K 段）
    pub v_dst_ptr: Option<*mut u8>, // 预分配的固定内存目标（V 段，可选）
}

// 包含裸指针，需要手动声明 Send
unsafe impl Send for SaveBlock {}
unsafe impl Send for SaveLayerData {}
```

### save_worker_loop()

```rust
fn save_worker_loop(device_id: i32, mut rx: mpsc::UnboundedReceiver<SaveTask>) -> Result<()> {
    let ctx = CudaContext::new(device_id as usize)?;  // 创建 CUDA 上下文
    let stream = ctx.new_stream()?;                    // 创建 CUDA 流
    
    while let Some(task) = rx.blocking_recv() {
        let result = process_save_task(&task, &stream);
        let _ = task.reply.send(result);  // 回复结果给 gRPC handler
    }
}
```

**process_save_task()**：

```
对每一层（layer）：
  对每个 block：
    GPU 地址 = base_ptr + block_idx × kv_stride_bytes
    cudaMemcpyAsync(cpu_dst, gpu_src, size, DeviceToHost, stream)

所有 cudaMemcpyAsync 提交后：
  cudaStreamSynchronize(stream)  ← 等待所有拷贝完成

返回 Ok(())
```

**关键点：批量提交 + 一次同步**

```rust
// 所有层的所有 block 先提交，再统一同步：
// 这比每个 block 单独 sync 快得多（GPU 可以流水线执行多个 DMA）
for layer_data in &task.layers {
    for block in &layer_data.blocks {
        cudaMemcpyAsync(dst, src, size, stream);  // 非阻塞提交
    }
}
stream.synchronize()?;  // 一次性等待所有完成
```

### batch_save() — 调用方接口

```rust
// pegaflow-core/src/gpu_worker.rs:147
pub(crate) async fn batch_save(&self, layers: Vec<SaveLayerData>) -> Result<(), EngineError> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let task = SaveTask { layers, reply: reply_tx };
    
    self.save_tx.send(task)?;  // 发送给 save worker（非阻塞）
    
    reply_rx.await?  // 等待 save worker 完成（异步 await，不阻塞 tokio 线程）
}
```

> **Rust 新手提示**：`oneshot::channel()` 是 tokio 的一次性通道（发送一条消息后关闭）。Save worker 在完成 CUDA 传输后调用 `reply.send(result)`，gRPC handler 中的 `reply_rx.await` 就会收到结果并继续执行。

---

## 4. Load 路径：CPU → GPU

### LoadTask 结构

```rust
pub struct LoadTask {
    pub layers: Vec<LayerLoadData>,
    pub load_state_shm: String,  // 共享内存名（完成通知）
}

pub struct LayerLoadData {
    pub layer_name: String,
    pub registration: KVCacheRegistration,
    pub blocks: Vec<LoadBlock>,
}

pub struct LoadBlock {
    pub block_idx: usize,        // 目标 GPU 槽位
    pub block: Arc<RawBlock>,    // CPU 固定内存中的数据
}
```

### load_worker_loop()

```rust
fn load_worker_loop(device_id: i32, mut rx: mpsc::UnboundedReceiver<LoadTask>) -> Result<()> {
    let ctx = CudaContext::new(device_id as usize)?;
    let stream = ctx.new_stream()?;
    
    while let Some(task) = rx.blocking_recv() {
        let result = process_load_task(&task, &stream);
        
        // 通过共享内存通知 vLLM Worker 传输完成
        match LoadState::attach(&task.load_state_shm) {
            Ok(load_state) => match result {
                Ok(()) => load_state.set_completed(),  // 标记完成
                Err(e) => {
                    core_metrics().load_failures.add(1, &[]);
                    load_state.set_error();             // 标记失败
                }
            },
            Err(e) => error!("Failed to attach to LoadState: {e}"),
        }
    }
}
```

**process_load_task() — K/V 分段优化**：

```rust
fn process_load_task(task: &LoadTask, stream: &CudaStream) -> Result<()> {
    for layer_data in &task.layers {
        let registration = &layer_data.registration;
        
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block {
            // 分段存储（K/V 分开）+ 层优先布局：K 段批量 → V 段批量
            let mut k_transfers = Vec::new();
            let mut v_transfers = Vec::new();
            
            for block in &layer_data.blocks {
                let k_gpu_offset = segment_offset(registration, block.block_idx, 0)?;
                let v_gpu_offset = segment_offset(registration, block.block_idx, 1)?;
                
                let k_cpu_ptr = block.block.segment_ptr(0).unwrap().as_ptr() as *const u8;
                let v_cpu_ptr = block.block.segment_ptr(1)  // 分段存储有 segment[1]
                    .map(|p| p.as_ptr() as *const u8)
                    .unwrap_or_else(|| unsafe { k_cpu_ptr.add(segment_size) });  // 连续存储
                
                k_transfers.push((k_gpu_offset, k_cpu_ptr));
                v_transfers.push((v_gpu_offset, v_cpu_ptr));
            }
            
            // 先批量提交所有 K 段，再批量提交所有 V 段
            batch_copy_segments_to_gpu(&k_transfers, segment_size, registration, stream)?;
            batch_copy_segments_to_gpu(&v_transfers, segment_size, registration, stream)?;
        } else {
            // 连续存储或非层优先：逐块传输
            for block in &layer_data.blocks { ... }
        }
    }
    
    stream.synchronize()?;
    Ok(())
}
```

**分段批量传输的优势**：

```
连续传输（K+V 交织）：
GPU 内存：[layer0.block0.K] [layer0.block0.V] [layer0.block1.K] [layer0.block1.V] ...

分段批量传输：
K 批次：[layer0.block0.K] → [layer0.block1.K] → [layer0.block2.K] → ...（一次 DMA）
V 批次：[layer0.block0.V] → [layer0.block1.V] → [layer0.block2.V] → ...（一次 DMA）

优势：减少 DMA 启动次数，GPU 可以更好地流水线执行
```

---

## 5. LoadState — 共享内存完成通知

**问题**：Load RPC 是 fire-and-forget 的（gRPC handler 立即返回），但 vLLM Worker 需要知道何时传输完成。

**解决方案**：POSIX 共享内存（Shared Memory）

```
vLLM Python Worker                    PegaFlow GPU Worker
      │                                      │
      │ [创建共享内存 "shm-load-1234"]        │
      │                                      │
      │──── Load RPC (load_state_shm="shm-load-1234") ────→│
      │                                      │
      │ [轮询共享内存 is_completed?]          │
      │ while not shm.is_completed():        │
      │     time.sleep(0.001)                │
      │                                      │
      │                               [CUDA 传输完成]
      │                               load_state.set_completed()
      │                               ↓ 写入共享内存标志位
      │ [共享内存标志 = true]          │
      │ [继续推理...]                 │
```

**为什么不用 gRPC 回调？**

gRPC 回调需要额外的网络往返（GPU worker → gRPC server → vLLM client），而共享内存是本地操作（同一台机器上的进程间通信），延迟更低（纳秒级 vs 微秒级）。

```rust
// pegaflow-core/src/sync_state.rs
pub struct LoadState {
    shm: shared_memory::Shmem,  // POSIX 共享内存对象
}

impl LoadState {
    pub fn attach(name: &str) -> Result<Self, ...> {
        // 打开已存在的共享内存（由 vLLM 创建）
        let shm = ShmemConf::new().os_id(name).open()?;
        Ok(Self { shm })
    }
    
    pub fn set_completed(&self) {
        // 写入完成标志（原子操作）
        unsafe { *(self.shm.as_ptr() as *mut u8) = 1; }
    }
    
    pub fn set_error(&self) {
        unsafe { *(self.shm.as_ptr() as *mut u8) = 2; }
    }
}
```

---

## 6. CUDA IPC Handle（跨进程 GPU 内存访问）

**背景**：vLLM Worker 和 PegaFlow 是不同的进程。PegaFlow 需要访问 vLLM 的 GPU 显存（用于 GPU→CPU 拷贝）。

**解决方案**：CUDA IPC（进程间通信）Handle

```
vLLM Worker Process                PegaFlow Process
    │                                    │
    │ [分配 GPU KV 缓冲区]               │
    │ cuda_ptr = cudaMalloc(30GB)         │
    │                                    │
    │ handle = cudaIpcGetMemHandle(ptr)   │
    │ # handle 是一个不透明的 64 字节标识符│
    │                                    │
    │──── RegisterContextBatch(wrapper_bytes=handle) ─→│
    │                                    │
    │                    cudaIpcOpenMemHandle(handle)
    │                    # 获得访问 vLLM GPU 缓冲区的指针
    │                                    │
    │                    [Save 时：]
    │                    cudaMemcpyAsync(cpu_dst,
    │                         vllm_gpu_ptr + offset,
    │                         size, DeviceToHost, stream)
```

**CUDA IPC 的限制**：
- 只能在同一台机器的进程间使用（不支持跨节点）
- 需要两个进程在同一 CUDA 设备上
- PegaFlow 打开 IPC handle 后获得一个本地的 GPU 虚拟地址，该地址映射到 vLLM 的实际显存

---

## 7. 完整 Save 流程（GPU → CPU → SSD）

```
vLLM Worker 完成前向计算
        │
        │ [Save RPC]
        ▼
GrpcEngineService::save()         （tokio 线程）
        │
        ▼
PegaEngine::batch_save_kv_blocks_from_ipc()
  ├── 为每个 block 分配 CPU 固定内存
  ├── 构建 SaveBlock { block_idx, k_dst_ptr, v_dst_ptr }
  └── GpuWorkerPool::batch_save(layers) → await
              │
              ▼ (oneshot 通道 + tokio 异步等待)
         save_worker_loop()         （OS 线程：gpu{N}-save）
              │
              ├── cudaMemcpyAsync(cpu_dst, gpu_src, ..., DeviceToHost)
              │   (批量：所有 layer × 所有 block)
              ├── stream.synchronize()
              └── reply.send(Ok(()))
              │
              ▼
        batch_save() 返回 Ok(())    （tokio 线程继续）
              │
              ▼
  offload.rs::build_raw_save_batch()
  → RawSaveBatch { 已在 CPU 内存的数据 }
        │
        ▼
  WritePipeline::send_raw_insert(batch)  （MPSC 非阻塞）
        │
        ▼ ─────── MPSC 通道 ──────────
              │
              ▼
         insert_worker_loop()       （OS 线程：pegaflow-insert）
              ├── 组装 InflightBlock（等待所有 TP slot）
              ├── seal → SealedBlock
              ├── ReadCache::batch_insert()
              ├── SsdBackingStore::ingest_batch()
              └── MetaServerClient::try_register()

        gRPC Save RPC 返回 Ok       （已在上面一步完成）
```

---

## 8. 完整 Load 流程（SSD/内存 → CPU → GPU）

```
vLLM Worker 需要 KV cache
        │
        │ [Load RPC]
        ▼
GrpcEngineService::load()         （tokio 线程）
        │
        ▼
PegaEngine::batch_load_kv_blocks_multi_layer()
  ├── ReadCache::consume_pinned_blocks() → Vec<Arc<SealedBlock>>
  ├── 构建 LoadBlock { block_idx, block: Arc<RawBlock> }
  └── GpuWorkerPool::submit_load(task) → 非阻塞返回
              │
              ▼ (UnboundedSender：立即返回)
         load_worker_loop()         （OS 线程：gpu{N}-load）

        Load RPC 立即返回 Ok        （gRPC handler 不等待 CUDA）
              │
              ▼
  [CUDA 传输在后台进行...]
  
  process_load_task():
  ├── 分层批量 cudaMemcpyAsync (HostToDevice)
  │   K 段全部 → V 段全部（减少 DMA 启动次数）
  └── stream.synchronize()
        │
        ▼
  LoadState::set_completed()
  → 写入共享内存标志位

        vLLM Worker Python 侧轮询:
        while not shm.is_completed(): sleep(1ms)
        [继续推理...]
```

---

## 9. 线程模型总结

```
每块 GPU 的线程结构（以 GPU 0 为例）：

OS 线程 "gpu0-load"（NUMA 0）：
  ├── CUDA context 0
  ├── CUDA stream（Load 专用）
  └── 循环等待 LoadTask（blocking_recv）

OS 线程 "gpu0-save"（NUMA 0）：
  ├── CUDA context 0
  ├── CUDA stream（Save 专用）
  └── 循环等待 SaveTask（blocking_recv）
  
通信方式：
  Tokio 线程（gRPC handler）→ mpsc::UnboundedSender → GPU worker OS 线程
  GPU worker OS 线程 → oneshot::Sender → Tokio 线程（Save 等待完成）
  GPU worker OS 线程 → shared memory → vLLM Worker Python（Load 完成通知）
```

**关键设计原则**：CUDA 操作在专用 OS 线程中执行，Tokio 运行时永远不会被 CUDA 阻塞操作（如 `stream.synchronize()`）所阻塞。
