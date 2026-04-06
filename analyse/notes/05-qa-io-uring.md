# Q&A: io_uring

## io_uring 是什么？

io_uring 是 Linux 5.1（2019年）引入的**异步 I/O 框架**，设计目标是彻底解决老式 AIO (`libaio`) 的性能缺陷。

### 核心思想：共享环形队列

```
用户空间                    内核空间
┌─────────────────────┐    ┌─────────────────────┐
│  SQ (提交队列)       │───▶│  处理 I/O 操作       │
│  [req][req][req]...  │    │                     │
├─────────────────────┤    ├─────────────────────┤
│  CQ (完成队列)       │◀───│  写入完成结果        │
│  [res][res][res]...  │    │                     │
└─────────────────────┘    └─────────────────────┘
        共享内存（mmap，无需系统调用拷贝）
```

用户空间和内核**共享同一块内存**（通过 `mmap`），避免了每次 I/O 都陷入内核的开销。

### 两个核心队列

| 队列 | 全称 | 作用 |
|------|------|------|
| **SQE** | Submission Queue Entry | 用户往这里写"我要做什么 I/O" |
| **CQE** | Completion Queue Entry | 内核往这里写"那个 I/O 完成了，结果是 X" |

提交：`io_uring_submit()` — 一次系统调用可以批量提交多个请求
收割：直接轮询 CQ 或 `io_uring_wait_cqe()` 等待 — 甚至可以**零系统调用**

### 三种操作模式

1. **中断驱动（默认）**：提交后内核异步处理，完成通知用户，正常系统调用开销
2. **内核轮询（`IORING_SETUP_IOPOLL`）**：内核持续轮询 I/O 完成，延迟更低，适合 NVMe 等高速设备
3. **SQ 轮询（`IORING_SETUP_SQPOLL`）**：内核线程持续监视 SQ，提交时**完全不需要系统调用**，适合超高 IOPS 场景

### 与其他方案的对比

| 方案 | 系统调用开销 | 真正异步 | 支持操作类型 |
|------|------------|---------|------------|
| `read`/`write` | 每次都陷入内核 | 否（阻塞线程） | 有限 |
| `libaio` | 每批次 | 部分（O_DIRECT 才真正异步） | 有限 |
| **io_uring** | 可以零次 | 是 | 极广（网络、文件、定时器…） |

## io_uring 在 PegaFlow 中的使用

在 `pegaflow-core/src/backing/uring.rs` 中用于 **SSD 缓存的异步读写**：

```
sealed block → 写入队列 → io_uring SQE → NVMe 异步写入 → CQE 回调
prefetch 请求 → io_uring SQE → NVMe 异步读取 → CQE → 解压/返回
```

对应 `pegaflow-server` 的 SSD 参数：

| 参数 | 与 io_uring 的关系 |
|------|-------------------|
| `--ssd-write-inflight` | 同时飞行(in-flight)的写入 SQE 数量上限 |
| `--ssd-prefetch-inflight` | 同时飞行的预取 SQE 数量上限 |
| `--ssd-write-queue-depth` | 排队等待提交的写入批次深度 |
| `--ssd-prefetch-queue-depth` | 排队等待提交的预取批次深度 |

本质是用 io_uring 将 SSD I/O 做成流水线：CPU 提交请求后立即返回处理其他事情，
NVMe 完成后通过 CQ 通知，实现 CPU 与 I/O 设备的真正并行。
