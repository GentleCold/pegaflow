# 存储技术：RAID 与 NVMe

---

## 1. RAID

**RAID = Redundant Array of Independent Disks**，将多块磁盘组合成一个逻辑卷，目标是**性能**或**冗余**或两者兼得。

### 常见级别

**RAID 0 — 条带化（纯性能）**
```
数据块：[A1][A2][A3][A4]
         ↓    ↓    ↓    ↓
磁盘1: [A1]  [A3]
磁盘2: [A2]  [A4]

读写速度 = 单盘 × N
容量 = 单盘 × N
冗余 = 无（任意一块坏 → 全部数据丢失）
```

**RAID 1 — 镜像（纯冗余）**
```
磁盘1: [A1][A2]   ← 完全相同的两份
磁盘2: [A1][A2]

读速度 ≈ 单盘 × N（可并行读不同块）
写速度 = 单盘（需同时写所有盘）
容量 = 单盘（N 块盘只得 1 份空间）
冗余 = 允许 N-1 块盘损坏
```

**RAID 5 — 条带 + 分布式奇偶校验**
```
磁盘1: [A1] [B1] [C1] [P_D]
磁盘2: [A2] [B2] [P_C] [D2]
磁盘3: [A3] [P_B] [C3] [D3]
磁盘4: [P_A] [B4] [C4] [D4]
            ↑ 奇偶校验块轮流分布在各盘

P_A = A1 XOR A2 XOR A3（任意一块坏了可由其余盘重算）

读速度快（条带并行）
写速度有开销（每次写需更新校验块）
容量 = (N-1) × 单盘
冗余 = 允许 1 块盘损坏
```

**RAID 6** — 同 RAID 5 但双校验块，允许 2 块同时损坏。

**RAID 10** — 先镜像(RAID 1)再条带(RAID 0)，兼顾性能与冗余，最少 4 块盘。

### 对比表

| 级别 | 最少盘数 | 读性能 | 写性能 | 可用容量 | 允许坏盘数 |
|------|---------|--------|--------|---------|-----------|
| RAID 0 | 2 | N× | N× | 100% | 0 |
| RAID 1 | 2 | N× | 1× | 50% | N-1 |
| RAID 5 | 3 | N× | 中等 | (N-1)/N | 1 |
| RAID 6 | 4 | N× | 较慢 | (N-2)/N | 2 |
| RAID 10 | 4 | N× | N/2× | 50% | 每镜像组 1 块 |

---

## 2. NVMe

**NVMe（Non-Volatile Memory Express）** 是专为 SSD 设计的协议，直接走 PCIe 总线，绕过 SATA/SCSI 的协议栈。

### 与 SATA 对比

```
传统路径（SATA SSD）：
  应用 → 文件系统 → SCSI/ATA → SATA 控制器 → SSD
  最大队列深度：32，最大队列数：1

NVMe 路径：
  应用 → 文件系统 → NVMe 驱动 → PCIe → SSD
  最大队列深度：65535，最大队列数：65535（每 CPU 核一个独立队列）
```

| 指标 | SATA SSD | NVMe SSD |
|------|---------|---------|
| 接口 | SATA III | PCIe 4.0 x4 |
| 顺序读 | ~550 MB/s | ~7000 MB/s |
| 随机读 IOPS | ~100K | ~1000K |
| 延迟 | ~500μs | ~100μs |
| 队列深度 | 32 | 65535 |

NVMe 延迟更低的核心原因：协议栈更短 + 多队列无锁竞争。

---

## 3. NVMe-oF（NVMe over Fabrics）

把 NVMe 协议**跑在网络上**，让远端服务器像访问本地盘一样访问另一台机器的 NVMe SSD。

```
主机 A                          存储节点 B
┌──────────┐    RDMA/TCP       ┌──────────────┐
│  应用    │                   │ NVMe SSD × 4 │
│ NVMe驱动 │◀─────────────────▶│ NVMe-oF 目标  │
│  发起端  │   (Fabric 网络)   │              │
└──────────┘                   └──────────────┘
  主机 A 看到的就是一块本地 NVMe 盘
```

**传输层选项**：
- **RoCE / InfiniBand**：RDMA，延迟最低（~10μs），高性能存储首选
- **TCP**：部署简单，延迟稍高，Linux 5.0+ 内核原生支持
- **FC（光纤通道）**：传统企业存储

NVMe-oF 保留本地 NVMe 的多队列特性：
```
CPU 核0 → NVMe 队列0 ─┐
CPU 核1 → NVMe 队列1  ├──▶ Fabric 网络 ──▶ 远端 SSD
CPU 核2 → NVMe 队列2 ─┘

每核独立队列，无锁竞争，性能线性扩展
```

---

## 4. RAID 在 NVMe 场景下的变化

传统软件 RAID（md/LVM）或 RAID 卡有额外 CPU 开销，NVMe 高 IOPS 场景下开销占比更明显。现代做法：

- **NVMe RAID 0**：多块 NVMe 条带，顺序读可达 10+ GB/s
- **硬件 RAID 控制器**：企业 NVMe 阵列内置，卸载 CPU 压力
- **分布式存储替代 RAID**：Ceph、MinIO 等，通过副本/纠删码实现冗余，比 RAID 6 更灵活，横向扩展能力更强

---

## 5. 与 PegaFlow 的关系

PegaFlow 当前 SSD 缓存为**本地 NVMe**（`pegaflow-core/src/backing/uring.rs` 用 io_uring 读写本地文件）。

多节点场景的演进路径：
```
当前：
  pegaflow-server → io_uring → 本地 NVMe

扩展方向（NVMe-oF）：
  pegaflow-server → io_uring → NVMe-oF 客户端 → RDMA → 共享 NVMe 存储池

好处：多个 pegaflow-server 节点共享同一大容量存储池，
      无需每台机器单独配置本地 SSD
```

RAID 与 PegaFlow：
- 生产部署中 SSD 缓存盘通常做 RAID 0（纯性能，KV 数据可从内存重建，不需冗余）
- 或直接用多块裸 NVMe + 应用层分片（`--ssd-cache-path` 当前只支持单路径）
