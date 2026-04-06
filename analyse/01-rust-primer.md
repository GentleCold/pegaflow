# 01 — Rust 核心概念速查（专为 PegaFlow 定制）

> 本文不是 Rust 完整教程，而是专注于 PegaFlow 代码中**反复出现**的 Rust 特性，结合项目代码解释每个概念的实际用途。

---

## 1. 所有权与 `Arc<T>`

### 基本概念

Rust 的核心特性是**所有权（Ownership）**：每个值在任意时刻只有一个"所有者"，所有者离开作用域时值被自动销毁。

```rust
// 普通所有权：engine 是唯一所有者
let engine = PegaEngine::new();
// engine 离开作用域时自动调用 Drop::drop()，释放内存
```

### `Arc<T>`：多所有者共享

当需要多个地方"拥有"同一份数据时，用 `Arc<T>`（Atomically Reference Counted）：

```rust
// pegaflow-core/src/storage/mod.rs:79
pub(crate) struct StorageEngine {
    allocator: Arc<PinnedAllocator>,  // ← 多处共享同一个分配器
    read_cache: Arc<ReadCache>,       // ← gRPC handler 和 insert worker 都需要访问
    ...
}
```

`Arc<T>` 的工作原理：
- 克隆 `Arc` 时只递增引用计数（原子操作，线程安全）
- 最后一个 `Arc` 被销毁时，引用计数归零，真正释放内存
- `Arc::clone(&x)` 等价于 `x.clone()`（只是语义更清晰）

```rust
// 示例：StorageEngine 初始化时共享 read_cache
let read_cache = Arc::new(ReadCache::new(...));
// ...
let engine = StorageEngine {
    read_cache: read_cache.clone(), // 引用计数从 1 → 2
    // ...
};
// engine 和 read_cache 都持有同一个 ReadCache
```

### `Weak<T>`：弱引用（不阻止销毁）

```rust
// pegaflow-core/src/storage/mod.rs:174
let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
    let alloc_weak = weak_engine.clone(); // Weak 不增加引用计数
    let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
        alloc_weak
            .upgrade() // Weak → Option<Arc>，引擎已销毁时返回 None
            .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
    });
    // ...
});
```

**为什么用 `Weak`？** 解决循环引用：`StorageEngine` 内部的闭包需要引用 `StorageEngine` 自身。如果用 `Arc`，就会形成 `StorageEngine → Arc<StorageEngine>` 的循环，导致引用计数永远不为零（内存泄漏）。`Weak` 不增加引用计数，循环被打破。

---

## 2. 锁：`Mutex<T>` 与 `RwLock<T>`

### 标准库 vs parking_lot

PegaFlow 大量使用 `parking_lot::Mutex`，而不是标准库的 `std::sync::Mutex`。原因：
- parking_lot 实现更快（无系统调用，使用用户态自旋锁）
- API 更简洁：`.lock()` 返回 `MutexGuard`（不返回 `Result`，不会"中毒"）

```rust
// pegaflow-core/src/storage/read_cache.rs:11
pub(super) struct ReadCache {
    inner: Mutex<ReadCacheInner>, // parking_lot::Mutex
}

// 使用：
let mut inner = self.inner.lock(); // 自动获取锁
inner.cache.insert(key, block);   // 访问内部数据
// inner 离开作用域 → 自动释放锁（RAII）
```

### `RwLock`：读写分离

允许多个读者同时访问，但写者独占：

```rust
// pegaflow-core/src/lib.rs:135
pub struct PegaEngine {
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    // ↑ 查询实例（读）远比注册/注销（写）频繁，用 RwLock 提升并发
}

// 读访问：
let instances = self.instances.read().expect("lock poisoned");
let inst = instances.get(instance_id);

// 写访问：
let mut instances = self.instances.write().expect("lock poisoned");
instances.insert(id, arc_instance);
```

---

## 3. 生命周期与 `'static`

Rust 要求所有引用都有明确的生命周期（编译期确保引用不会悬空）。

```rust
// pegaflow-core/src/numa.rs:224
pub fn run_on_numa<T, F>(node: NumaNode, f: F) -> Result<T, String>
where
    T: Send + 'static,    // T 必须是 'static：不含对栈上数据的引用
    F: FnOnce() -> T + Send + 'static,
```

**`'static` 意味着**：值可以存活整个程序，不含临时引用。线程闭包必须是 `'static`，因为新线程的生命周期可能超过创建它的函数。

---

## 4. `unsafe` 代码与裸指针

### 为什么需要 `unsafe`？

Rust 的安全性检查在某些低层操作时必须手动保证：
1. 解引用裸指针（`*const T`/`*mut T`）
2. 调用 C 函数（FFI，如 `libc`、CUDA API）
3. 实现 `Send`/`Sync`

### `NonNull<u8>`：非空裸指针

```rust
// pegaflow-core/src/block.rs:84
pub(crate) struct Segment {
    ptr: NonNull<u8>,  // 比 *mut u8 更安全：保证非空
    size: usize,
    _allocation: Arc<PinnedAllocation>,
}
```

`NonNull<u8>` 相当于 `*mut u8` 但编译器保证不为 NULL。用 `NonNull` 的好处：
- 启用空指针优化：`Option<NonNull<u8>>` 不需要额外字节
- 表达"这个指针不应该为空"的意图

```rust
// 使用 NonNull：
let ptr: NonNull<u8> = NonNull::new(raw_ptr).expect("pointer must not be null");
unsafe {
    let slice = std::slice::from_raw_parts(ptr.as_ptr(), size);
}
```

### `Send` + `Sync` 手动实现

```rust
// pegaflow-core/src/block.rs:103-104
// Safety: Segment 持有的指针指向 CUDA pinned memory（物理地址固定）
// Arc<PinnedAllocation> 管理生命周期，Send+Sync 由调用方保证使用正确
unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}
```

Rust 的 `Send`（可安全移动到其他线程）和 `Sync`（可从多线程共享引用）是编译期安全保证。当你持有裸指针时，Rust 无法自动推断，需要手动用 `unsafe impl` 声明（并在注释中说明为何安全）。

---

## 5. 枚举与模式匹配

Rust 的枚举（enum）非常强大，可以携带数据：

```rust
// pegaflow-core/src/block.rs:71
pub enum PrefetchStatus {
    Loading { hit: usize, loading: usize },  // 带命名字段
    Done { hit: usize, missing: usize },     // 带命名字段
}

// 使用 match 解构：
match status {
    PrefetchStatus::Done { hit, missing } => {
        println!("命中 {} 块，缺失 {} 块", hit, missing);
    }
    PrefetchStatus::Loading { hit, loading } => {
        println!("命中 {} 块，正在加载 {} 块", hit, loading);
    }
}
```

类比：这相当于其他语言中"带类型标签的联合体"，但 Rust 会在编译期强制你处理所有情况。

---

## 6. `Option<T>` 与 `Result<T, E>`

Rust 没有 `null`，用 `Option<T>` 表示"可能没有值"：

```rust
// pegaflow-core/src/lib.rs:234
fn get_instance(&self, instance_id: &str) -> Result<Arc<InstanceContext>, EngineError> {
    let instances = self.instances.read().expect("lock poisoned");
    instances
        .get(instance_id)     // 返回 Option<&Arc<InstanceContext>>
        .cloned()             // Option<&T> → Option<T>（克隆内部值）
        .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))
        // Option<T> → Result<T, E>：None 变成 Err(...)
}
```

常用转换方法：
| 方法 | 作用 |
|------|------|
| `.unwrap()` | 取值，None/Err 时 panic（只在测试中用） |
| `.expect("msg")` | 同上，但 panic 时显示自定义消息 |
| `.unwrap_or(default)` | None/Err 时返回默认值 |
| `.ok_or(err)` | `Option<T>` → `Result<T, E>` |
| `.map(f)` | 对内部值应用函数，不影响 None/Err |
| `?` 运算符 | Err 时提前返回，相当于 `try!` |

```rust
// ? 运算符示例：
fn save(...) -> Result<(), EngineError> {
    let instance = self.get_instance(instance_id)?; // Err 时直接返回
    // 继续执行...
    Ok(())
}
```

---

## 7. Trait：接口/协议

Trait 类似其他语言的接口（Interface）或协议（Protocol）：

```rust
// pegaflow-server/src/service.rs:134
#[async_trait]
impl Engine for GrpcEngineService {
    // 实现 Engine trait（由 protobuf 生成）的所有 RPC 方法
    async fn save(&self, request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
        // ...
    }
}
```

`Display` trait 用于自定义打印格式：
```rust
// pegaflow-core/src/lib.rs:100
impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InstanceMissing(ctx) => write!(f, "instance {ctx} not found"),
            // ...
        }
    }
}
```

---

## 8. 异步编程：`async`/`await` 与 Tokio

Rust 的异步模型基于 **Future**（惰性计算）和 **运行时**（执行 Future）。PegaFlow 使用 Tokio 作为运行时。

```rust
// async 函数返回 Future（懒执行，直到被 await）
pub async fn count_prefix_hit_blocks_with_prefetch(
    &self,
    instance_id: &str,
    req_id: &str,
    block_hashes: &[Vec<u8>],
) -> Result<PrefetchStatus, EngineError> {
    // await 在这里暂停，让出 CPU 给其他任务
    let status = self.storage
        .check_prefix_and_prefetch(instance_id, req_id, namespace, block_hashes, world_size)
        .await;   // ← 等待 Future 完成
    Ok(status)
}
```

**关键点**：
- `async fn` 编译成返回 `Future` 的普通函数
- `.await` 暂停当前异步任务，不阻塞 OS 线程
- Tokio 可以在一个 OS 线程上运行成千上万个异步任务

**与线程的区别**：
```rust
// 专用 OS 线程（CPU 密集型任务）：
std::thread::Builder::new()
    .name("pegaflow-insert".into())
    .spawn(move || {
        write_path::insert_worker_loop(insert_rx, weak_deps); // 阻塞循环
    })

// Tokio 异步任务（I/O 密集型任务）：
tokio::spawn(async {
    some_async_operation().await;
});
```

---

## 9. 通道（Channels）

### `std::sync::mpsc`：多生产者单消费者

```rust
// pegaflow-core/src/storage/write_path.rs:30
pub(super) fn new() -> (Self, Receiver<InsertWorkerCommand>) {
    let (insert_tx, insert_rx) = std::sync::mpsc::channel();
    //   发送端（多个 clone）   接收端（唯一）
    (Self { insert_tx }, insert_rx)
}

// 发送（非阻塞）：
self.insert_tx.send(InsertWorkerCommand::RawInsert(batch)).ok();

// 接收（阻塞直到有消息）：
while let Ok(cmd) = rx.recv() {
    // 处理命令...
}
```

### `tokio::sync::oneshot`：一次性通道

```rust
// 用于请求-响应模式：
let (reply_tx, reply_rx) = oneshot::channel();
self.insert_tx.send(InsertWorkerCommand::Gc {
    max_age,
    reply: reply_tx, // 发送端传给 worker
}).ok();
let result = reply_rx.await.unwrap_or(0); // 等待 worker 回复
```

---

## 10. 闭包

Rust 的闭包（Closure）是可以捕获环境变量的匿名函数：

```rust
// pegaflow-core/src/storage/mod.rs:177
let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
    //                                  ^^^^ 捕获 alloc_weak（move 语义）
    alloc_weak
        .upgrade()
        .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
});
```

**`move` 关键字**：将捕获的变量移动进闭包（所有权转移），而不是借用。必须在闭包需要比创建它的函数活得更长时使用（如线程、异步任务）。

---

## 11. 迭代器与函数式操作

Rust 的迭代器非常高效（编译期展开，零开销抽象）：

```rust
// pegaflow-core/src/numa.rs:323-326
(0..count)
    .map(|device_id| (device_id, get_device_numa_node(device_id)))
    .collect() // 收集成 Vec<(u32, NumaNode)>

// 等价于：
let mut result = Vec::new();
for device_id in 0..count {
    result.push((device_id, get_device_numa_node(device_id)));
}
result
```

常用迭代器方法：
| 方法 | 作用 |
|------|------|
| `.map(f)` | 逐元素变换 |
| `.filter(p)` | 按条件过滤 |
| `.collect()` | 收集成集合 |
| `.zip(iter)` | 并行迭代两个集合 |
| `.enumerate()` | 带索引迭代 |
| `.fold(init, f)` | 归约（相当于 reduce） |
| `.any(p)` / `.all(p)` | 存在/全称量词 |
| `.flatten()` | 嵌套迭代器展平 |

---

## 12. 模块系统

```rust
// 模块声明（在 lib.rs 或 mod.rs 中）：
pub mod block;          // block.rs 或 block/mod.rs（public）
mod cache;              // cache.rs（private，只在本 crate 可见）
pub(crate) mod storage; // storage/mod.rs（crate 内可见）

// 可见性说明：
pub fn foo()        // 公开（crate 外可见）
pub(crate) fn bar() // crate 内可见
fn baz()            // 私有（仅本模块）
pub(super) fn qux() // 父模块可见
```

**文件 vs 模块的对应关系**：
- `mod storage;` → 找 `storage.rs` 或 `storage/mod.rs`
- `storage/mod.rs` 中的 `mod read_cache;` → `storage/read_cache.rs`

---

## 快速参考：常见 Rust 语法

```rust
// 结构体
struct Foo { field: u32 }
impl Foo {
    fn new(val: u32) -> Self { Self { field: val } }
    fn get(&self) -> u32 { self.field }
    fn set(&mut self, val: u32) { self.field = val; }
}

// 枚举
enum Bar { A, B(u32), C { x: i32, y: i32 } }

// 泛型
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// 特征对象（动态分发）
fn call(obj: &dyn Display) { println!("{}", obj); }

// 生命周期
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 宏
vec![1, 2, 3]
println!("hello {}", name)
format!("{}:{}", host, port)
assert_eq!(a, b)
todo!()
unreachable!()
```
