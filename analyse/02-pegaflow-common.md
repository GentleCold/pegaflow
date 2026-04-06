# 02 — pegaflow-common：公共工具库

**文件位置**：`pegaflow-common/src/`  
**被依赖于**：所有其他 crate（common 是最底层，无重依赖）

---

## 1. 库结构（lib.rs）

```rust
// pegaflow-common/src/lib.rs
pub mod hll;        // HyperLogLog 命中率估计
pub mod logging;    // 统一日志配置
pub mod numa;       // NUMA 拓扑 + 线程亲和性

// 重导出（re-export）最常用的 NUMA 函数，让其他 crate 直接 use pegaflow_common::NumaNode
pub use numa::{
    NumaNode, NumaTopology, format_cpu_list, pin_thread_to_numa_node,
    query_pages_numa, read_cpu_topology_from_sysfs, run_on_numa,
};
```

---

## 2. logging.rs — 统一日志配置

### 设计目标

PegaFlow 需要在多种场景下运行：
- gRPC 服务器（输出到 stdout，带颜色，方便 k8s 日志收集）
- Python 绑定（输出到 stderr，不带颜色，防止与 Python 日志混淆）
- 路由器（输出到 stderr）

`logging.rs` 提供两个简单入口，隐藏所有配置细节。

### 核心数据结构

```rust
// pegaflow-common/src/logging.rs:8
static INIT: Once = Once::new();
// Once 确保日志只初始化一次（即使多次调用 init()）
// 这在 Python 绑定中很重要：Python 可能多次导入 pegaflow 模块
```

```rust
// 日志输出目标枚举
#[derive(Debug, Clone, Copy, Default)]
pub enum LogOutput {
    #[default]  // ← Default trait 自动实现，默认值为 Stderr
    Stderr,
    Stdout,
}

// 完整配置结构体
pub struct LoggingConfig {
    pub level: String,   // 如 "info" 或 "info,pegaflow_core=debug"
    pub output: LogOutput,
    pub colored: bool,
}
```

### 噪声模块静音

```rust
// pegaflow-common/src/logging.rs:70-80
const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 9] = [
    ("h2", "warn"),          // HTTP/2 协议库（tonic 底层）
    ("hyper", "warn"),       // HTTP 客户端库
    ("hyper_util", "warn"),
    ("tonic", "warn"),       // gRPC 框架
    ("tower", "warn"),       // gRPC 中间件
    ("opentelemetry", "info"),
    ("opentelemetry_otlp", "info"),
    ("opentelemetry_sdk", "info"),
    ("offset_allocator", "info"), // 内存分配器调试日志
];
```

这些第三方库在 `debug` 级别会产生大量日志，干扰调试。`apply_default_module_levels` 函数在用户未显式配置时自动压制这些日志。

### 公开 API

```rust
// 服务器使用（彩色 stdout）：
pub fn init_stdout_colored(level: &str) {
    init(LoggingConfig::new(level).stdout().colored());
}

// Python 绑定/路由器使用（无色 stderr）：
pub fn init_stderr(level: &str) {
    init(LoggingConfig::new(level).stderr());
}
```

### 使用示例

```rust
// 在 pegaflow-server/src/main.rs 中：
pegaflow_common::logging::init_stdout_colored("info");
// 等价于设置 RUST_LOG=info（但 RUST_LOG 环境变量优先级更高）

// 调试模式：
RUST_LOG=info,pegaflow_core=debug cargo run
```

---

## 3. numa.rs — NUMA 拓扑检测

### 什么是 NUMA？

NUMA（Non-Uniform Memory Access，非一致内存访问）是多路服务器的内存架构。每个 CPU 组（Socket）有自己的本地内存（NUMA 节点），访问本地内存比访问远端内存快 2-5 倍。

```
典型双路服务器 NUMA 拓扑：
┌─────────────────────────────────────────────────┐
│  NUMA Node 0                NUMA Node 1          │
│  ┌─────────────┐            ┌─────────────┐     │
│  │ CPU 0-15    │            │ CPU 16-31   │     │
│  │ (Core 0-7   │            │ (Core 8-15  │     │
│  │ + HT 8-15)  │            │ + HT 24-31) │     │
│  └──────┬──────┘            └──────┬──────┘     │
│         │                          │             │
│  ┌──────▼──────┐            ┌──────▼──────┐     │
│  │ 本地内存    │            │ 本地内存    │     │
│  │ 128 GB      │◄──QPI/UPI──►│ 128 GB      │     │
│  └─────────────┘            └─────────────┘     │
│                                                  │
│  GPU 0 ──── 通过 PCIe ──── NUMA 0 亲和          │
│  GPU 1 ──── 通过 PCIe ──── NUMA 1 亲和          │
└─────────────────────────────────────────────────┘
```

PegaFlow 的优化：在 GPU 0 的 NUMA 节点上分配内存，GPU 0 的 DMA 传输就走本地内存，带宽高、延迟低。

### NumaNode 结构体

```rust
// pegaflow-common/src/numa.rs:18
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNode(pub u32);  // 简单的 newtype 包装 u32

impl NumaNode {
    pub const UNKNOWN: NumaNode = NumaNode(u32::MAX); // 哨兵值
    pub fn is_unknown(&self) -> bool { self.0 == u32::MAX }
    pub fn is_valid(&self) -> bool { self.0 != u32::MAX }
}

// Display 实现：
// NumaNode(0) → "NUMA0"
// NumaNode::UNKNOWN → "UNKNOWN"
```

> **Rust 新手提示**：`NumaNode(pub u32)` 是 Tuple Struct（元组结构体），只有一个未命名字段。`pub u32` 使字段公开可访问（`node.0`）。这是常见的 newtype 模式，避免将 `u32` 和 NUMA node 编号混用。

### 从 sysfs 读取 CPU 拓扑

Linux 通过 `/sys/devices/system/node/` 暴露 NUMA 拓扑信息：

```
/sys/devices/system/node/
├── node0/
│   └── cpulist    # 内容例如 "0-15,32-47"
└── node1/
    └── cpulist    # 内容例如 "16-31,48-63"
```

```rust
// pegaflow-common/src/numa.rs:88
pub fn read_cpu_topology_from_sysfs() -> Result<HashMap<u32, Vec<usize>>, String> {
    let node_dir = std::path::Path::new("/sys/devices/system/node");
    // 遍历所有 node* 目录，读取 cpulist 文件
    // 解析格式如 "0-3,8-11" → [0,1,2,3,8,9,10,11]
}
```

`parse_cpulist` 函数解析 Linux CPU 列表格式（`"0-3,8,16-17"` → `[0,1,2,3,8,16,17]`）。

### 线程 CPU 亲和性绑定

```rust
// pegaflow-common/src/numa.rs:178
pub fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    let node_to_cpus = read_cpu_topology_from_sysfs()?;
    let cpus = node_to_cpus.get(&node.0).ok_or_else(|| ...)?;

    // SAFETY: cpu_set_t 是 C 结构体，安全零初始化
    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();
        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut cpu_set); // 设置允许的 CPU
        }
        libc::sched_setaffinity(
            0,  // 0 = 当前线程
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );
    }
}
```

**为什么要绑定线程？** Linux 的"首次接触"（first-touch）内存分配策略：物理内存页在**第一次写入时**分配，分配在**执行写入的 CPU 对应的 NUMA 节点**上。为了让内存分配在正确的 NUMA 节点，必须先将线程 pin 到目标节点的 CPU，再分配内存。

### `run_on_numa`：在目标 NUMA 节点运行闭包

```rust
// pegaflow-common/src/numa.rs:224
pub fn run_on_numa<T, F>(node: NumaNode, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(format!("numa{}-init", node.0))
        .spawn(move || {
            pin_thread_to_numa_node(node)?; // 1. 绑定线程
            let result = f();               // 2. 执行闭包（内存在正确的 NUMA 节点上分配）
            tx.send(Ok(result))
        })?;
    // 等待结果...
}
```

**使用场景**：在 `PinnedAllocator` 初始化时，为每个 NUMA 节点创建一个线程，在该线程上 `mmap` + CUDA pinned memory 分配，确保物理页面落在正确的 NUMA 节点。

### GPU NUMA 亲和性检测

```rust
// 通过 nvidia-smi 查询 GPU 的 NUMA 亲和节点：
// nvidia-smi topo --get-numa-id-of-nearby-cpu -i 0
// 输出示例："GPU 00000000:3B:00.0: 0"  ← GPU 0 亲和 NUMA 0
```

```rust
// NumaTopology 结构体：
pub struct NumaTopology {
    gpu_numa_map: HashMap<i32, NumaNode>,  // device_id → NUMA node
    numa_nodes: Vec<NumaNode>,             // 系统中所有 NUMA 节点
}

// 使用：
let topo = NumaTopology::detect();
let numa = topo.numa_for_gpu(0); // GPU 0 对应的 NUMA 节点
```

### `query_pages_numa`：查询内存页 NUMA 位置

```rust
// pegaflow-common/src/numa.rs:428
pub fn query_pages_numa(addrs: &[*const u8]) -> Vec<NumaNode> {
    // 使用 move_pages(2) 系统调用（nodes=NULL = 只查询不移动）
    libc::syscall(libc::SYS_move_pages, 0, count, pages.as_ptr(), NULL, status.as_mut_ptr(), 0)
    // status[i] = 第 i 个地址所在的 NUMA 节点编号（或负数表示错误）
}
```

用于**验证**内存确实分配在了预期的 NUMA 节点（调试和测试使用）。

---

## 4. hll.rs — HyperLogLog 命中率估计

### 用途

`HllTracker` 在 `pegaflow-server` 中被使用（`service.rs:28`），在 Query 和 QueryPrefetch RPC 中记录所有请求的 block hash，周期性估计：
- **独特 block 数量**（基数）
- **理论最大命中率**：`(总请求数 - 不重复块数) / 总请求数`

这个指标帮助运维判断缓存容量是否充足。

### HyperLogLog 算法简介

HyperLogLog 是一种**概率性基数估计**算法，用少量内存（几 KB）估计数十亿不重复元素的数量，误差约 1%。

**基本思想**：
- 对每个元素计算哈希值，统计哈希中**前导零的个数**
- 前导零越多，说明见过的不同元素越多
- 用多个"桶"（registers）并行统计，取调和平均

```rust
// pegaflow-common/src/hll.rs:25
pub struct HyperLogLog {
    registers: Vec<u8>,   // m = 2^bucket_bits 个寄存器
    bucket_bits: u8,      // 通常 14，即 16384 个桶
    lz_mask: u32,         // 用于快速计算前导零
}

// bucket_bits=14 的精度：标准误差 ≈ 1.04 / sqrt(16384) ≈ 0.81%
```

```
哈希值（256位）结构：
┌──────────────────┬──────────────────────────────┐
│ 前 14 位         │  剩余 242 位                   │
│ = 桶索引(0-16383)│  统计前导零个数                │
└──────────────────┴──────────────────────────────┘
```

### 滑动窗口 HllTracker

```rust
// pegaflow-common/src/hll.rs:235
pub struct HllTracker {
    slots: VecDeque<WindowSlot>,  // 时间槽队列
    merged: HyperLogLog,          // 所有已结束槽的合并 HLL（缓存，避免重复计算）
    merged_dirty: bool,           // merged 是否需要重新计算
    slot_duration: Duration,      // 每个时间槽长度（如 1 小时）
    window_duration: Duration,    // 总窗口长度（如 24 小时）
}
```

工作原理：
1. 时间轴被划分为等长的槽（slot），每个槽有一个独立的 HLL
2. 新请求写入当前（最新）槽
3. 超过 `window_duration` 的槽被丢弃
4. 计算指标时，合并所有活跃槽的 HLL 得到总基数

```rust
// 查询命中率：
let metric = tracker.metric();
println!(
    "过去24小时：不重复块数={:.0}, 总请求数={}, 估计命中率={:.1}%",
    metric.cardinality,
    metric.total_requests,
    metric.estimated_hit_rate * 100.0
);
```

---

## 5. 小结

| 模块 | 功能 | 在 PegaFlow 中的使用位置 |
|------|------|--------------------------|
| `logging` | 统一日志初始化 | 所有服务进程的 main() |
| `numa` | NUMA 拓扑检测 + 线程绑定 | PinnedAllocator 初始化 |
| `hll` | 命中率统计 | GrpcEngineService（服务端指标） |
