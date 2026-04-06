# 19 — vLLM v1 Connector（Python 连接器）

**核心文件**：
- `python/pegaflow/connector/scheduler.py` — 调度器侧连接器
- `python/pegaflow/connector/worker.py` — Worker 侧连接器
- `python/pegaflow/connector/common.py` — 共享数据结构
- `python/pegaflow/sglang/pegaflow_radix_cache.py` — SGLang 集成

---

## 1. vLLM v1 KV Connector 架构

vLLM v1 引入了 `KVConnector` 接口，将调度器侧（Scheduler）和 Worker 侧分为两个独立组件：

```
vLLM Scheduler（主进程）                    vLLM Worker（GPU 进程）
┌───────────────────────────────┐            ┌─────────────────────────────┐
│  SchedulerConnector           │            │  WorkerConnector            │
│  - get_num_new_matched_tokens │  元数据    │  - register_kv_caches       │
│  - update_state_after_alloc   │─────────→ │  - start_load_kv            │
│  - build_connector_meta       │            │  - save_kv_layer            │
│  - update_connector_output    │            │  - wait_for_save            │
│  - request_finished           │            │  - get_finished             │
└───────────────────────────────┘            └─────────────────────────────┘
         │                                              │
         │ PegaFlowEngineRpcClient(gRPC)                │ PegaFlowEngineRpcClient(gRPC)
         ▼                                              ▼
    QueryPrefetch RPC                         RegisterContext / Save / Load RPC
```

**分工**：
- **SchedulerConnector**：决策层，负责 Query/Prefetch 和 Save 意图的调度决策
- **WorkerConnector**：执行层，负责实际的 GPU 数据传输（Save/Load）

---

## 2. ConnectorContext — 共享上下文

两个 Connector 都持有同一个 `ConnectorContext`（共享配置和客户端）：

```python
# connector/common.py
class ConnectorContext:
    engine_client: EngineRpcClient   # gRPC 客户端（Rust 侧 PyO3 绑定）
    instance_id: str                  # 实例 ID（每个 vLLM 实例唯一）
    namespace: str                    # 命名空间（隔离不同模型/实验）
    virtual_block_size: int           # 虚拟块大小（token 数）
    tp_rank: int                      # Tensor Parallel rank
    tp_size: int                      # Tensor Parallel 大小
    world_size: int                   # 总进程数（含 TP+PP）
    device_id: int | None             # CUDA device ID
    state_manager: ServiceStateManager  # 服务可用性管理器
```

---

## 3. SchedulerConnector — 调度器侧

### 3.1 关键状态

```python
# scheduler.py:42
class SchedulerConnector:
    # 调优参数（从环境变量读取）
    BYPASS_BLOCKS: int = parse_env_int("PEGA_BYPASS_BLOCKS", 0)           # 短请求绕过阈值
    HIGH_LOAD_THRESHOLD: int = parse_env_int("PEGA_HIGH_LOAD_THRESHOLD", 0)  # 高负载阈值
    MAX_PENDING_SAVE_REQUESTS: int = parse_env_int("PEGA_MAX_PENDING_SAVE_REQUESTS", 0)
    
    # Load 状态
    _pending_load_intents: dict[str, LoadIntent]       # 待执行的 Load 意图
    _prefetch_start_times: dict[str, float]             # 预取开始时间（用于统计）
    _prefetch_tracker: PrefetchTracker                  # 预取并发计数器
    
    # Save 状态（每请求）
    _block_hashes: dict[str, tuple[bytes, ...]]         # 请求的 block hash 序列
    _allocated_blocks: dict[str, list[int]]             # 已分配的 GPU block ID
    _scheduled_tokens: dict[str, int]                   # 已调度的 token 数
    _stored_blocks: dict[str, int]                      # 已 Save 的 block 数
    
    _pending_saves: set[str]                            # 正在 Save 的请求 ID
```

### 3.2 `get_num_new_matched_tokens()` — 查询缓存命中

vLLM Scheduler 在调度每个新请求时调用此方法，决定是否可以从 KV cache 加载：

```python
# scheduler.py:73
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    block_hashes = request.block_hashes
    computed_blocks = num_computed_tokens // self._ctx.virtual_block_size
    remaining_hashes = block_hashes[computed_blocks:]  # 跳过已本地计算的块
    
    # ── 短路绕过逻辑 ──
    # 当系统高负载（pending_prefetches >= HIGH_LOAD_THRESHOLD）时，
    # 短请求（blocks < BYPASS_BLOCKS）直接跳过远端查询，避免雪崩
    if len(remaining_hashes) < BYPASS_BLOCKS and pending >= HIGH_LOAD_THRESHOLD:
        return (0, False)
    
    # ── 调用 QueryPrefetch RPC ──
    hit_blocks = self._count_available_block_prefix(remaining_hashes, req_id)
    
    if hit_blocks is None:  # 预取进行中，告诉调度器稍后重试
        return (None, False)
    
    num_hit_tokens = hit_blocks * self._ctx.virtual_block_size
    return (num_hit_tokens, True) if num_hit_tokens > 0 else (0, False)
```

### 3.3 `_count_available_block_prefix()` — QueryPrefetch 封装

```python
# scheduler.py:379
def _count_available_block_prefix(self, block_hashes, req_id) -> int | None:
    # 服务不可用时快速失败（避免阻塞调度）
    if not self._ctx.state_manager.is_available():
        return 0
    
    try:
        result = self._ctx.engine_client.query_prefetch(
            self._ctx.instance_id, block_hash_list, req_id=req_id
        )
    except PegaFlowServiceError as e:
        self._ctx.state_manager.mark_unavailable(str(e))  # 标记不可用，后续跳过
        return 0
    except PegaFlowBusinessError as e:
        raise  # 客户端参数错误，向上传播
    
    prefetch_state = result.get("prefetch_state", "done")
    hit_blocks = result.get("hit_blocks", 0)
    
    if prefetch_state == "loading":
        # 首次进入 loading 状态：记录开始时间，计数器 +1
        if req_id not in self._prefetch_start_times:
            self._prefetch_start_times[req_id] = time.perf_counter()
            self._prefetch_tracker.on_prefetch_start()
        return None  # 信号：稍后重试
    
    # 预取完成：记录统计（持续时间、命中数）
    if req_id in self._prefetch_start_times:
        duration_ms = (time.perf_counter() - self._prefetch_start_times.pop(req_id)) * 1000
        self._prefetch_tracker.on_prefetch_complete(duration_ms, hit_blocks)
    
    return hit_blocks
```

### 3.4 `build_connector_meta()` — 构建 Save 意图

每个调度周期（batched forward pass）结束后调用，决定哪些 block 需要 Save：

```python
# scheduler.py:185
def build_connector_meta(self, scheduler_output) -> PegaConnectorMetadata:
    # ── 收集所有可能的 Save 意图 ──
    potential_saves = {}
    
    for req in scheduler_output.scheduled_new_reqs:
        # 新请求：计算本轮新完成的 block
        if save_intent := self._consume_save_intent(req_id):
            potential_saves[req_id] = save_intent
    
    for req_id in cached_reqs.req_ids:
        # 续跑请求：刷新 block_hashes（decode 阶段新增的 block）
        req = self._requests.get(req_id)
        if req:
            self._block_hashes[req_id] = tuple(req.block_hashes)  # 刷新！
        if save_intent := self._consume_save_intent(req_id):
            potential_saves[req_id] = save_intent
    
    # ── 应用 Save 限额（MAX_PENDING_SAVE_REQUESTS）──
    if MAX_PENDING_SAVE_REQUESTS <= 0:
        save_intents = potential_saves  # 无限制
    else:
        # 优先保留块数多的请求（长请求更值得 Save）
        # 短请求超出限额时被丢弃，记录 save_dropped_count
        available_slots = MAX_PENDING_SAVE_REQUESTS - len(self._pending_saves)
        new_saves.sort(key=lambda x: x[2], reverse=True)  # 按 block 数降序
        # ... 选择前 available_slots 个
    
    self._pending_saves.update(save_intents.keys())
    return PegaConnectorMetadata(load_intents=load_intents, save_intents=save_intents)
```

### 3.5 `_consume_save_intent()` — 计算新增 block

```python
# scheduler.py:297
def _consume_save_intent(self, req_id) -> SaveIntent | None:
    block_hashes = self._block_hashes.get(req_id)
    allocated = self._allocated_blocks.get(req_id, [])
    scheduled = self._scheduled_tokens.get(req_id, 0)
    stored = self._stored_blocks.get(req_id, 0)
    
    # 可以 Save 的 block 数（三者取最小）：
    # - block_hashes 长度（已有 hash 的 block 数）
    # - allocated 长度（已分配 GPU slot 的 block 数）  
    # - scheduled // virtual_block_size（已调度 token 对应的 block 数）
    saveable = min(len(block_hashes), len(allocated), scheduled // virtual_block_size)
    new_blocks = saveable - stored  # 本轮新增的 block
    
    if new_blocks <= 0:
        return None
    
    self._stored_blocks[req_id] = stored + new_blocks
    return SaveIntent(
        block_ids=tuple(allocated[stored : stored + new_blocks]),
        block_hashes=block_hashes[stored : stored + new_blocks],
    )
```

---

## 4. WorkerConnector — Worker 侧

### 4.1 异步 Save 架构

Worker 侧使用独立的 Save 线程异步处理 Save，不阻塞 forward pass：

```python
# worker.py:47
class WorkerConnector:
    def __init__(self, context):
        # ── Save 组件 ──
        self._save_queue = queue.Queue()         # Save 任务队列
        self._save_thread = threading.Thread(    # 独立 Save 线程
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()
        
        self._req_pending_layers: dict[str, int]   # 每个请求还有多少层未 Save
        self._completed_saves: set[str]            # 已完成 Save 的请求 ID
        
        # ── Load 组件 ──
        self._pending_loads: dict[str, PyLoadState]      # shm_name → LoadState
        self._pending_load_reqs: dict[str, set[str]]     # shm_name → 请求 ID 集合
```

### 4.2 `register_kv_caches()` — 注册 GPU KV 缓冲区

Worker 启动时调用，将 vLLM 的 GPU KV 缓冲区注册到 PegaFlow：

```python
# worker.py:95
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    for layer_name, kv_cache in kv_caches.items():
        # 获取 CUDA IPC handle（序列化为字节）
        wrapper = CudaIPCWrapper(kv_cache)
        wrapper_bytes = pickle.dumps(wrapper)
        
        # 解析 tensor 形状，判断存储布局
        if kv_cache.shape[0] == 2:
            # KV-first 布局：shape = [2, num_blocks, ...]
            num_blocks = shape[1]
            bytes_per_block = stride[1] * element_size
            kv_stride_bytes = stride[0] * element_size
            segments = 2  # K 和 V 分段存储
        else:
            # blocks-first 布局：shape = [num_blocks, ...]
            num_blocks = shape[0]
            bytes_per_block = stride[0] * element_size
            kv_stride_bytes = 0
            segments = 1  # 连续存储
    
    # 批量注册（单次 gRPC 调用，包含所有 layer）
    self._ctx.engine_client.register_context_batch(
        instance_id, namespace, tp_rank, tp_size, world_size, device_id, num_layers,
        layer_names, ipc_wrappers, num_blocks_list, bytes_per_block_list,
        kv_stride_bytes_list, layer_segments,
    )
```

### 4.3 `start_load_kv()` — 触发异步 Load

vLLM 在每个 forward pass 开始前调用，触发 CPU → GPU 数据传输：

```python
# worker.py:265
def start_load_kv(self, metadata, forward_context, **kwargs):
    if not metadata.load_intents:
        return
    
    # 收集所有请求的 block_ids 和 block_hashes（合并成一批）
    for req_id, load_intent in metadata.load_intents.items():
        all_block_ids.extend(load_intent.block_ids)
        all_block_hashes.extend(load_intent.block_hashes)
    
    # 创建共享内存 LoadState（用于接收完成通知）
    load_state = PyLoadState()
    shm_name = load_state.shm_name()
    
    # 触发 Load RPC（立即返回，PegaFlow 后台完成传输）
    self._ctx.engine_client.load(
        instance_id, tp_rank, device_id, shm_name,
        target_layers, all_block_ids, all_block_hashes,
    )
    
    # 注册等待状态（get_finished 中轮询）
    with self._load_completion_lock:
        for req_id in request_ids:
            self._pending_loads[req_id] = load_state
        self._pending_load_reqs[shm_name] = set(request_ids)
```

### 4.4 `save_kv_layer()` — 触发异步 Save

vLLM 每处理完一层（layer）的 forward pass 后调用：

```python
# worker.py:344
def save_kv_layer(self, metadata, layer_name, kv_layer, attn_metadata, **kwargs):
    request_ids = list(metadata.save_intents.keys())
    if not request_ids:
        return
    
    # 初始化新请求的层计数器
    with self._save_completion_lock:
        for req_id in request_ids:
            if req_id not in self._req_pending_layers:
                # 每个请求需要 Save len(self._registered_layers) 层
                self._req_pending_layers[req_id] = len(self._registered_layers)
    
    # 提交到 Save 队列（非阻塞）
    self._save_queue.put(SaveTask(
        layer_name=layer_name,
        attn_metadata=attn_metadata,
        metadata=metadata,
        request_ids=request_ids,
    ))
```

### 4.5 `_save_worker()` — 后台 Save 线程

```python
# worker.py:439
def _save_worker(self):
    while True:
        task = self._save_queue.get()
        if task is None: break  # 关机信号
        
        # 批量收集（不等待，尽量合并）
        batch = [task]
        while True:
            try:
                t = self._save_queue.get_nowait()
                batch.append(t)
            except queue.Empty:
                break
        
        self._process_save_batch(batch)

def _process_save_batch(self, batch):
    # 按 layer 汇总 block_ids 和 block_hashes
    saves_by_layer = {}
    for task in batch:
        for save_intent in task.metadata.save_intents.values():
            saves_by_layer[task.layer_name][0].extend(save_intent.block_ids)
            saves_by_layer[task.layer_name][1].extend(save_intent.block_hashes)
    
    if saves_by_layer:
        # !! 关键：等待 GPU 计算完成（否则 KV cache 可能包含未初始化数据）
        torch.cuda.synchronize(self._torch_device)
        
        # 调用 Save RPC
        self._ctx.engine_client.save(
            instance_id, tp_rank, device_id, saves_list
        )
    
    # 递减层计数器（即使 Save 失败也要递减，避免永久阻塞）
    self._decrement_layer_counter(all_request_ids)
```

### 4.6 层计数器与 Save 完成通知

```python
# worker.py:536
def _decrement_layer_counter(self, request_ids):
    completed_reqs = []
    with self._save_completion_lock:
        for req_id in request_ids:
            if req_id in self._req_pending_layers:
                self._req_pending_layers[req_id] -= 1
                if self._req_pending_layers[req_id] == 0:
                    # 所有层已 Save 完毕！
                    self._completed_saves.add(req_id)
                    del self._req_pending_layers[req_id]
                    completed_reqs.append(req_id)
                    event = self._save_completion_events.pop(req_id, None)
                    if event: event.set()  # 通知 handle_preemptions()
```

**计数器工作原理**：

```
vLLM 模型有 32 层，每层 forward 完后调用 save_kv_layer()

req_1 首次出现（在 layer_0 的 save）：
  _req_pending_layers["req_1"] = 32  （初始化为总层数）

layer_0 Save Worker 处理：
  _req_pending_layers["req_1"] = 31  （-1）

...经过 32 次 Save...

layer_31 Save Worker 处理：
  _req_pending_layers["req_1"] = 0
  → _completed_saves.add("req_1")   （标记 Save 完成）
```

### 4.7 抢占处理（`handle_preemptions()`）

当 vLLM 决定抢占某个请求（重用其 GPU block）前，必须等待 PegaFlow 完成 Save，否则会读取到脏数据：

```python
# worker.py:574
def handle_preemptions(self, preempted_req_ids):
    events_to_wait = []
    with self._save_completion_lock:
        for req_id in preempted_req_ids:
            event = self._save_completion_events.get(req_id)
            if event:
                events_to_wait.append((req_id, event))
    
    # 阻塞等待所有被抢占请求的 Save 完成
    for req_id, event in events_to_wait:
        event.wait()  # 阻塞直到 _decrement_layer_counter() 触发 event.set()
```

---

## 5. `get_finished()` — 轮询完成状态

vLLM Scheduler 每轮调度后调用，检查哪些请求的 Save/Load 已完成：

```python
# worker.py:186
def get_finished(self, finished_req_ids) -> tuple[set | None, set | None]:
    # ── 检查 Save 完成 ──
    with self._save_completion_lock:
        done_saves = self._completed_saves & self._finished_requests
        done_saves.update(self._completed_saves & finished_req_ids)
        if done_saves:
            self._completed_saves -= done_saves
            finished_sending = done_saves  # 告诉 Scheduler 这些请求 Save 完了
    
    # ── 检查 Load 完成（轮询共享内存）──
    with self._load_completion_lock:
        for shm_name, req_ids in self._pending_load_reqs.items():
            sample_req_id = next(iter(req_ids))
            load_state = self._pending_loads.get(sample_req_id)
            
            if load_state.is_ready():  # 检查共享内存标志位
                state = load_state.get_state()
                if state >= 0:
                    completed_reqs.update(req_ids)  # Load 成功
                else:
                    logger.error("async_load_failed: state=%d", state)  # Load 失败
    
    return (finished_sending, finished_recving)
```

---

## 6. SaveIntent 和 LoadIntent 数据结构

```python
# connector/common.py
@dataclass
class SaveIntent:
    block_ids: tuple[int, ...]     # GPU pool 中的 block 槽位索引
    block_hashes: tuple[bytes, ...]  # 对应的内容哈希

@dataclass
class LoadIntent:
    block_ids: tuple[int, ...]     # 目标 GPU 槽位
    block_hashes: tuple[bytes, ...]  # 要加载的 block 哈希
    num_tokens: int                  # 对应的 token 数

@dataclass
class PegaConnectorMetadata:
    load_intents: dict[str, LoadIntent]   # req_id → LoadIntent
    save_intents: dict[str, SaveIntent]   # req_id → SaveIntent
```

---

## 7. 完整请求生命周期

```
1. 新请求到来（vLLM Scheduler）
        │
        ▼ SchedulerConnector.get_num_new_matched_tokens()
        QueryPrefetch RPC → PegaFlow（查询 + 触发预取）
        ├── 返回 loading → 告知 Scheduler 稍后重试
        └── 返回 hit_blocks → 继续分配

2. Scheduler 分配 GPU block
        │
        ▼ SchedulerConnector.update_state_after_alloc()
        保存 block_ids/block_hashes/load_intent

3. Scheduler 生成调度输出
        │
        ▼ SchedulerConnector.build_connector_meta()
        生成 PegaConnectorMetadata（load_intents + save_intents）

4. Worker 执行 forward pass（每个 GPU process）
        │
        ├── WorkerConnector.start_load_kv()
        │   Load RPC → PegaFlow（CPU → GPU 传输，fire-and-forget）
        │   创建 PyLoadState（共享内存）
        │
        ├── forward pass 计算...（此时 CUDA 传输在后台进行）
        │
        ├── WorkerConnector.save_kv_layer()（每层调用一次）
        │   将 SaveTask 入队（非阻塞）
        │
        └── WorkerConnector.wait_for_save()（forward pass 结束）
            处理 CUDA graph 跳过的 Save 请求

5. 后台 Save 线程（PegaSaveWorker）
        │
        ├── torch.cuda.synchronize()（确保 GPU 计算完成）
        ├── Save RPC → PegaFlow（GPU → CPU 传输）
        └── _decrement_layer_counter()（所有层完成后标记 Save 完成）

6. Scheduler 收到完成通知
        │
        ▼ SchedulerConnector.update_connector_output()
        从 _pending_saves 移除已完成请求

7. 请求结束
        │
        ▼ SchedulerConnector.request_finished()
        ├── 如有 pending saves → held_requests（等待 Save 完成再释放 block）
        └── 无 pending saves → 立即释放资源
```

---

## 8. SGLang 集成（简介）

SGLang 的 radix cache 需要基于前缀树（radix tree）的 KV cache 管理。PegaFlow 通过继承 SGLang 的 `RadixCache` 类提供集成：

```python
# sglang/pegaflow_radix_cache.py
class PegaFlowRadixCache(RadixCache):
    """在 SGLang radix cache 基础上增加 PegaFlow 远端 KV cache 支持。"""
    
    def match_prefix(self, key, **kwargs):
        # 先查本地 radix tree
        local_result = super().match_prefix(key)
        
        if local_result.missing:
            # 向 PegaFlow 查询（QueryPrefetch）
            hit_blocks = self._query_pegaflow(missing_hashes)
            # 如果命中 → 触发 Load 流程
        
        return result
```

SGLang 与 vLLM 的主要区别：
- SGLang 使用 prefix tree 做 token 粒度匹配
- vLLM 使用 block 粒度匹配
- PegaFlow 对两者提供统一的后端接口
