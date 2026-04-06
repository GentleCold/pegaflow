# Q&A: io_uring / HLL / TTL Store / TinyLFU

## Q: io_uring 是什么？

Linux 5.1 引入的异步 I/O 框架。用户空间和内核通过 mmap 共享两个环形队列（SQ 提交队列 / CQ 完成队列），避免每次 I/O 都陷入内核。

三种模式：中断驱动（默认）/ 内核轮询 IOPOLL / SQ 轮询（零系统调用，适合高 IOPS）。

PegaFlow 用途：`pegaflow-core/src/backing/uring.rs` 实现 SSD 缓存的异步读写，让 CPU 不阻塞在等待 NVMe 响应上。
相关参数：`--ssd-write-inflight`、`--ssd-prefetch-inflight`（飞行中请求上限）。

详见 `05-qa-io-uring.md`。

---

## Q: IOPS 是什么？高 IOPS 场景为什么需要 io_uring？

IOPS = Input/Output Operations Per Second，衡量存储设备每秒处理的 I/O 请求次数。
区别于带宽（吞吐量），IOPS 反映随机小块 I/O 能力。

NVMe SSD 可达 100万+ IOPS，但每次传统 `read()`/`write()` 系统调用本身耗时 1~3μs，
100万次 × 2μs = 2秒 全耗在系统调用上，硬件能力被软件栈拖累。

io_uring 的 SQ 轮询模式：内核线程持续盯 SQ，用户写入共享内存即可，无需系统调用，
彻底解除系统调用对 IOPS 的限制。代价：持续占用一个 CPU 核做 busy-poll。

---

## Q: HLL 是什么？

HyperLogLog，概率数据结构，用几 KB 固定内存估计数据流中**不同元素的数量**（基数），误差约 0.8%。

原理：对每个元素哈希后统计二进制前导零数，前导零越多说明见过的元素越多（概率论）。
用 m 个桶分摊统计，调和平均降低方差。桶数 = 2^bucket_bits，误差 ≈ 1.04/√m。

PegaFlow 实现了**滑动窗口 HLL**：按时间槽分片，保留窗口内所有槽，合并时取各桶最大值。
用途：统计过去 24h 内访问过多少个唯一 block hash，暴露为 Prometheus 指标。

详细算法原理见 `06-algorithms-hll-cms-tinylfu.md`。

---

## Q: MetaServer 的 TTL Store 是什么？

MetaServer (`pegaflow-metaserver/src/store.rs`) 用 moka 库实现 LRU + TTL 双策略缓存，
存储 `block_hash → 节点地址` 映射。

两种淘汰策略：
- **LRU**：容量满时驱逐最久未访问的条目（控制内存上限）
- **TTL**：条目插入超过 ttl_minutes 自动失效（防止节点下线后残留"僵尸记录"）

为什么需要 TTL：节点崩溃后 block 已不存在，但 MetaServer 仍指向该节点，
TTL 到期后自动清除，避免其他节点被导向死亡节点。

相关参数：`--max-capacity-mb 512`、`--ttl-minutes 120`。

---

## Q: TinyLFU 是什么？解决了 LRU 的什么问题？

LRU 只看最近是否被访问，不看访问频率。一次性扫描请求（如遍历不同 prompt）会将热数据驱逐出缓存，称为**缓存污染**。

TinyLFU 在驱逐前加准入过滤：新候选项的历史访问频率必须高于被驱逐项，否则直接拒绝。
频率统计用 Count-Min Sketch（矩阵 + 多哈希），内存极小，且定期 Aging（右移衰减）防止历史永久压制新热点。

W-TinyLFU 三区结构解决新 key 频率为 0 进不来的问题：
Window LRU(1%) → TinyLFU 门卫 → Probationary(19%) → Protected(80%)。

PegaFlow 参数：`--enable-lfu-admission`（默认关闭）。适合多用户共享 system prompt 等有明显热点的场景。

详细算法原理见 `06-algorithms-hll-cms-tinylfu.md`。
