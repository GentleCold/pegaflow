# GPUDirect Storage (GDS) — 环境要求与核心概念

## 什么是 GDS

GDS 允许存储设备（NVMe SSD、网络存储）与 GPU 显存之间**直接传输数据**，绕过 CPU 和系统内存（bounce buffer）。

```
传统路径：GPU ← CPU/系统内存 ← 内核缓冲区 ← NVMe
GDS 路径：GPU ← PCIe ← NVMe（直传）
```

---

## 硬件要求

### GPU
- NVIDIA Ampere 及以上（A100、H100 等）为完整支持
- RTX 4090（Ada Lovelace）、RTX 5090（Blackwell）可用，但属消费级，有限制（见下文）
- GPU 必须支持 GPUDirect 技术

### 存储设备
- 本地 NVMe SSD（PCIe 直连）
- 企业级网络存储（InfiniBand/RoCE，需支持 GPUDirect Storage over network）

### PCIe 拓扑
- GPU 和 NVMe 在**同一 PCIe switch** 下效果最优（P2P 路径最短）
- 跨 CPU socket 会有额外延迟

---

## 软件要求

| 组件 | 要求 |
|------|------|
| Linux 内核 | ≥ 4.15，推荐 5.4+ |
| NVIDIA 驱动 | ≥ 450.80，推荐 470+ |
| CUDA Toolkit | ≥ 11.4（cuFile API 正式版） |
| nvidia-fs 内核模块 | 必须加载，随 CUDA Toolkit 附带 |
| libcufile.so | GDS 用户态 API 库 |

### 支持的文件系统
- ✅ ext4、XFS（原生支持）
- ✅ GPFS、Lustre、WekaFS、VAST（企业存储，需各自驱动）
- ❌ tmpfs、NFS（不支持）

---

## 消费级 GPU（4090 / 5090）的限制

| 功能 | GeForce 4090/5090 | 数据中心 A100/H100 |
|------|------------------|-------------------|
| GDS 本地 SSD | ✅ 支持 | ✅ 支持 |
| GPUDirect RDMA（远端存储） | ❌ 驱动禁用 | ✅ 支持 |
| NVLink P2P | ❌ 无硬件接口 | ✅ 支持 |
| ECC 内存 | ⚠️ 可软件开启，损失 ~6% 显存 | ✅ 默认开启 |

RTX 5090（Blackwell，2025年初发布）需驱动 560+ 确保 nvidia-fs 成熟支持。

---

## 使用约束

- 必须使用 `cuFileRead` / `cuFileWrite` API，不能用普通 `read()`/`write()` 或 `io_uring`
- GPU 内存必须通过 `cuFileBufRegister` 注册，否则自动降级为 bounce buffer 路径
- 文件偏移需 **4KB 对齐**，非对齐访问自动降级
- 同一 PCIe switch 下 GPU + NVMe 性能最优

---

## 环境检查命令

```bash
# 1. GPU 型号和驱动版本
nvidia-smi

# 2. PCIe 拓扑（确认 GPU 和 NVMe 在同一 switch）
nvidia-smi topo -m

# 3. 驱动版本（需 ≥ 450.80）
nvidia-smi | grep "Driver Version"

# 4. nvidia-fs 内核模块（最关键）
lsmod | grep nvidia_fs
modinfo nvidia_fs
sudo modprobe nvidia_fs   # 如未加载

# 5. CUDA 版本（需 ≥ 11.4）
nvcc --version
cat /usr/local/cuda/version.json

# 6. cuFile 库是否存在
ls /usr/local/cuda/lib64/libcufile*
cat /etc/cufile.json

# 7. 官方 GDS 环境检查工具（最全面）
/usr/local/cuda/gds/tools/gdscheck -p

# 8. GDS 读写功能测试
/usr/local/cuda/gds/tools/gdsio -f /tmp/test_gds -d 0 -w 4 -s 1M -x 0
# -d 0 = GPU 0, -w 4 = 4 workers, -s 1M = 1MB, -x 0 = GDS 模式

# 9. 文件系统类型（ext4/xfs 支持，nfs/tmpfs 不支持）
df -Th
stat -f /path/to/storage

# 10. ECC 状态查询与开关
nvidia-smi --query-gpu=ecc.mode.current --format=csv
sudo nvidia-smi -e 1   # 开启 ECC（需重启）
sudo nvidia-smi -e 0   # 关闭 ECC
```

---

## NVLink P2P 详解

### NVLink 带宽对比

| 互联方式 | 双向带宽 |
|---------|---------|
| PCIe 4.0 x16 | ~64 GB/s |
| PCIe 5.0 x16 | ~128 GB/s |
| NVLink 4.0（H100） | ~900 GB/s |

### P2P（Peer-to-Peer）含义

GPU 之间**直接**互传数据，不经过 CPU 或系统内存。

```
无 NVLink：GPU A → PCIe → CPU/内存 → PCIe → GPU B
有 NVLink：GPU A ←→ GPU B（直接，极低延迟）
```

消费级 GeForce 系列在驱动层面禁用了 GPUDirect RDMA，且物理上没有 NVLink 接口，因此多 GPU 场景只能走 PCIe P2P，带宽受限。

---

## ECC 内存详解

### 原理：Hamming Code

ECC 在存储数据时额外写入校验位，利用**不同校验位覆盖不同位组合**的方式，在读取时不仅能检测错误，还能**定位到具体哪一位出错**。

以 Hamming(7,4) 为例，4 位数据 + 3 位校验 = 7 位：

```
位置:  1   2   3   4   5   6   7
内容: p1  p2  d1  p3  d2  d3  d4

p1 覆盖位置 1,3,5,7  → p1 = d1⊕d2⊕d4
p2 覆盖位置 2,3,6,7  → p2 = d1⊕d3⊕d4
p3 覆盖位置 4,5,6,7  → p3 = d2⊕d3⊕d4
```

读取时重算校验位，得到校验综合值 `s3 s2 s1`（3位二进制数）：
- `000` = 无错误
- `001`~`111` = 对应位置出错，**直接翻转该位修复**

### 关键洞察

> n 个校验位可以编码 2ⁿ 种位置信息。3 个校验位 → 定位 7 个位中任意 1 个错误（2³-1=7）。

### 纠错能力边界

| 错误数 | 能力 |
|--------|------|
| 1-bit | ✅ 定位并修复 |
| 2-bit | ⚠️ 检测到有错，但无法正确定位 |
| 3-bit | ❌ 可能被误判为 1-bit，静默损坏 |

GPU 实际使用 **SECDED**（Single Error Correct, Double Error Detect）扩展方案，比基础 Hamming 多一个校验位。

### 比特翻转的来源
- 宇宙射线（高能粒子撞击）
- 高温环境
- 长时间高负载运行

对 AI 推理场景：KV cache 中 1 个比特翻转可能导致输出错误，数据中心长期运行时 ECC 是必要保障。
