# 18 — Python 绑定（PyO3）

**核心文件**：
- `python/src/lib.rs`（~700 行）— PyO3 Rust 绑定
- `python/pegaflow/pegaflow.pyi` — Python 类型存根
- `python/pegaflow/ipc_wrapper.py` — CUDA IPC handle 封装

---

## 1. PyO3 是什么？

PyO3 是 Rust 和 Python 之间的互操作库，允许：
- 用 Rust 编写 Python 扩展模块（`.so` 文件）
- 在 Rust 代码中调用 Python 函数
- 在 Python 中使用 Rust 结构体作为 Python 类

PegaFlow 使用 PyO3 将 Rust 的 `PegaEngine` 和 gRPC 客户端暴露给 Python（vLLM/SGLang）。

---

## 2. 全局 Tokio Runtime

**问题**：Rust async 代码需要在 Tokio runtime 中运行，但 Python 不懂 Rust 的 future。

**解决方案**：全局单例 Tokio runtime

```rust
// python/src/lib.rs:27
static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> PyResult<&'static Runtime> {
    if let Some(rt) = TOKIO_RUNTIME.get() {
        return Ok(rt);  // 快速路径：已初始化
    }
    
    let rt = Runtime::new().map_err(runtime_creation_error)?;
    let _ = TOKIO_RUNTIME.set(rt);  // 只有第一个成功（竞争安全）
    
    TOKIO_RUNTIME.get().ok_or_else(|| ...)
}
```

> **Rust 新手提示**：`OnceLock<T>` 是线程安全的单次初始化容器（类似 Go 的 `sync.Once`）。`set()` 只有第一次调用成功，后续调用会返回 `Err`（值被忽略）。这保证了全局只有一个 Tokio runtime。

**在 Python 调用中执行 async 代码**：

```rust
fn call_rpc<F, T>(&self, future: F) -> PyResult<T>
where F: Future<Output = Result<T, Status>> {
    let rt = get_runtime()?;
    rt.block_on(future).map_err(|e| rpc_status_error("rpc", e))
}
```

`block_on()` 在当前线程中同步运行 async future（阻塞直到完成）。这样 Python 调用者看到的是普通同步函数。

---

## 3. PegaEngine Python 类

```rust
// python/src/lib.rs:108
#[pyclass]
struct PegaEngine {
    engine: CoreEngine,  // Rust 内部的 pegaflow_core::PegaEngine
}

#[pymethods]
impl PegaEngine {
    #[new]
    fn new(config: PegaEngineConfig) -> PyResult<Self> {
        let rt = get_runtime()?;
        let engine = rt.block_on(CoreEngine::new(config.into()))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create engine: {e}")))?;
        Ok(Self { engine })
    }
    
    // Python 调用示例：engine.save(instance_id, tp_rank, device_id, saves)
    fn save(&self, instance_id: &str, tp_rank: u32, device_id: i32, saves: Vec<SaveLayer>) 
        -> PyResult<()> 
    {
        let rt = get_runtime()?;
        rt.block_on(self.engine.batch_save_kv_blocks_from_ipc(...))
            .map_err(|e| map_engine_error(e))
    }
}
```

**`#[pyclass]` 和 `#[pymethods]`**：这两个宏告诉 PyO3 将 Rust 结构体和其方法暴露为 Python 类和方法。

---

## 4. 错误类型体系

```rust
// python/src/lib.rs:23
create_exception!(pegaflow, PegaFlowError, PyException);         // 基类
create_exception!(pegaflow, PegaFlowServiceError, PegaFlowError); // 服务错误（服务端问题）
create_exception!(pegaflow, PegaFlowBusinessError, PegaFlowError); // 业务错误（客户端问题）

fn is_service_error(code: Code) -> bool {
    matches!(code,
        Code::Unavailable | Code::DeadlineExceeded | Code::Internal |
        Code::Aborted | Code::Cancelled
    )
}

fn rpc_status_error(method: &str, err: GrpcStatus) -> PyErr {
    let msg = format!("{method} RPC failed: {err}");
    if is_service_error(err.code()) {
        PegaFlowServiceError::new_err(msg)   // 服务端问题，应该重试或检查健康
    } else {
        PegaFlowBusinessError::new_err(msg)  // 请求参数问题，应该修复请求
    }
}
```

**Python 侧使用**：

```python
from pegaflow.pegaflow import PegaFlowServiceError, PegaFlowBusinessError

try:
    client.save(instance_id, ...)
except PegaFlowServiceError:
    # 服务端问题，检查健康状态，可能需要重连
    reconnect()
except PegaFlowBusinessError:
    # 请求问题，记录错误
    logger.error("Invalid request")
```

---

## 5. EngineRpcClient — gRPC 客户端封装

```rust
#[pyclass]
struct EngineRpcClient {
    endpoint: String,
    client: EngineClient<Channel>,
    rt_handle: Handle,  // Tokio runtime 句柄（用于 block_on）
}

impl EngineRpcClient {
    fn call_rpc<F, T>(&self, method: &str, future: F) -> PyResult<T>
    where F: Future<Output = Result<Response<T>, Status>> {
        self.rt_handle.block_on(future)
            .map(|r| r.into_inner())
            .map_err(|e| rpc_status_error(method, e))
    }
}

#[pymethods]
impl EngineRpcClient {
    #[new]
    fn new(endpoint: &str) -> PyResult<Self> {
        let rt = get_runtime()?;
        let channel = Endpoint::from_shared(endpoint)?
            .connect_timeout(Duration::from_secs(10))
            .connect_lazy();  // 延迟连接
        Ok(Self {
            endpoint: endpoint.to_string(),
            client: EngineClient::new(channel),
            rt_handle: rt.handle().clone(),
        })
    }
    
    fn query(&self, instance_id: &str, block_hashes: Vec<Vec<u8>>, req_id: &str) 
        -> PyResult<(u64, u64, u64, i32)>  // (hit, loading, missing, prefetch_state)
    {
        let response = self.call_rpc("query", self.client.clone().query(QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes,
            req_id: req_id.to_string(),
        }))?;
        
        Ok((response.hit_blocks, response.loading_blocks, 
            response.missing_blocks, response.prefetch_state))
    }
}
```

---

## 6. PyLoadState — 共享内存 Load 完成通知

```rust
#[pyclass]
struct PyLoadState {
    inner: LoadState,
}

#[pymethods]
impl PyLoadState {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        // 创建新的共享内存（由 Python 侧创建，Rust 侧写入）
        let inner = LoadState::new(name)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create LoadState: {e}")))?;
        Ok(Self { inner })
    }
    
    fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }
    
    fn is_error(&self) -> bool {
        self.inner.is_error()
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
}
```

**Python 侧使用**：

```python
from pegaflow.pegaflow import PyLoadState

# 创建共享内存
shm_name = f"pegaflow-load-{request_id}"
load_state = PyLoadState(shm_name)

# 发起 Load RPC（Rust 端会在传输完成后写入共享内存）
client.load(instance_id, tp_rank, device_id, shm_name, layer_names, block_ids, block_hashes)

# 轮询等待完成（不阻塞 Python GIL）
while not load_state.is_completed() and not load_state.is_error():
    time.sleep(0.001)  # 1ms 轮询间隔
```

---

## 7. Python 模块注册

```rust
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册异常类
    m.add("PegaFlowError", m.py().get_type::<PegaFlowError>())?;
    m.add("PegaFlowServiceError", m.py().get_type::<PegaFlowServiceError>())?;
    m.add("PegaFlowBusinessError", m.py().get_type::<PegaFlowBusinessError>())?;
    
    // 注册类
    m.add_class::<PegaEngine>()?;
    m.add_class::<EngineRpcClient>()?;
    m.add_class::<PyLoadState>()?;
    
    Ok(())
}
```

编译后生成 `pegaflow/pegaflow.so`（Linux）或 `pegaflow/pegaflow.pyd`（Windows），Python 直接 `import pegaflow.pegaflow` 即可使用。

---

## 8. ipc_wrapper.py — CUDA IPC Handle

```python
# python/pegaflow/ipc_wrapper.py
class CudaIPCWrapper:
    """Wraps a torch.Tensor to expose its CUDA IPC handle."""
    
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        # 获取 CUDA IPC handle（字节串）
        self._handle = torch.cuda._shared_memory.get_ipc_handle(tensor)
    
    @property
    def wrapper_bytes(self) -> bytes:
        return self._handle.serialize()
    
    @property
    def data_ptr(self) -> int:
        return self._tensor.data_ptr()
    
    @property
    def num_blocks(self) -> int:
        return self._tensor.shape[0]
```

vLLM 调用 `CudaIPCWrapper.wrapper_bytes` 获取 IPC handle，通过 `RegisterContextBatch` RPC 传给 PegaFlow。PegaFlow 在 `registry.rs` 中用 PyO3 打开这个 handle，获得访问 GPU 显存的权限。
