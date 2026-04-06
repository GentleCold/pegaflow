# Python 扩展机制 Q&A

## Q1: Maturin 和 PyO3 是什么，为什么要用？

**问题背景**：vLLM 是 Python 程序，PegaFlow 核心是 Rust，需要桥梁。

### PyO3 — Python ↔ Rust 互操作库

PyO3 让你用 Rust 写出 Python 可以直接 `import` 的模块（`.so` 文件）。原理是 Python 扩展模块本质上是符合 Python C API 的动态库，PyO3 自动生成胶水代码：

```rust
#[pyclass]
struct EngineRpcClient { ... }

#[pymethods]
impl EngineRpcClient {
    fn save(&self, instance_id: &str, ...) -> PyResult<()> { ... }
}

#[pymodule]
fn pegaflow(m: &PyModule) -> PyResult<()> {
    m.add_class::<EngineRpcClient>()?;
    Ok(())
}
```

编译后 Python 直接用：
```python
from pegaflow.pegaflow import EngineRpcClient
client = EngineRpcClient("127.0.0.1:50055")
client.save(...)
```

### Maturin — 构建工具

| 工具 | 职责 | 类比 |
|------|------|------|
| PyO3 | 解决"怎么写"（Rust/Python 互操作注解）| 类似 `cgo` |
| Maturin | 解决"怎么编译打包"（调 cargo build，放 .so 到 venv）| 类似 `setuptools` |

```
maturin develop
    ├── 1. cargo build 编译 Rust
    ├── 2. 生成 pegaflow/pegaflow.so
    └── 3. 安装到 .venv，Python 可 import
```

### 为什么不直接用 ctypes/cffi？

PyO3 直接操作 Python 对象（GIL、引用计数、异常），比 ctypes 更安全、更符合 Python 习惯，且能直接抛 Python 异常。

---

## Q2: EngineRpcClient 为什么不用纯 Python + gRPC 实现？

`EngineRpcClient` 功能上**可以用纯 Python 实现**，但放在 Rust 里有工程原因：

### 1. PyO3 已经是必须的
有些东西只能用 PyO3：
- `PyLoadState`：操作 POSIX 共享内存，纯 Python 做不到
- `PegaEngine`：进程内嵌入 Rust 引擎，不走网络

既然 PyO3 已存在，`EngineRpcClient` 顺手放进去复用基础设施。

### 2. 共享同一个 Tokio runtime
```rust
static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
// EngineRpcClient 和 PyLoadState 共用同一套异步调度
```
若改成纯 Python（grpcio），两套异步系统并存，增加复杂度。

### 3. 避免引入 grpcio 重型依赖
纯 Python gRPC 需要 `grpcio`（重型 C 扩展，安装常出问题）+ `protobuf`。用 PyO3 包 tonic，Python 侧零额外依赖。

### 4. 统一错误类型
```rust
PegaFlowServiceError  // 服务端问题
PegaFlowBusinessError // 客户端参数问题
```
一处定义，所有调用共用，不需要在 Python 侧维护额外的错误映射。

### 总结

| 组件 | 能否纯 Python 实现 | 为何用 Rust |
|------|------------------|------------|
| `EngineRpcClient` | ✅ 能 | 工程一致性，共享 runtime 和错误类型 |
| `PyLoadState` | ❌ 不能 | 需要直接操作共享内存 |
| `PegaEngine` | ❌ 不能 | 进程内嵌 Rust 引擎 |

---

## Q3: .pyi 文件有什么用？

`.pyi` 是**类型存根（Type Stub）**，只写类型签名，不写实现。

**为什么需要**：`pegaflow.so` 是二进制，IDE 和 mypy 看不进去，不知道里面有什么类和方法。`.pyi` 作为说明书告诉工具链接口长什么样：

```python
# pegaflow.pyi — 只有签名
class EngineRpcClient:
    def __new__(cls, endpoint: str) -> "EngineRpcClient": ...
    def save(self, instance_id: str, tp_rank: int, ...) -> None: ...
    def load(self, instance_id: str, ...) -> None: ...
```

| 场景 | 没有 .pyi | 有 .pyi |
|------|----------|--------|
| IDE 补全 | ❌ | ✅ |
| mypy 类型检查 | ❌ 报 `Module has no attribute` | ✅ |
| 运行时 | ✅（不影响）| ✅（运行时忽略）|

类比：就像 C 的 `.h` 头文件，实现在 `.so` 里，声明在 `.pyi` 里。
