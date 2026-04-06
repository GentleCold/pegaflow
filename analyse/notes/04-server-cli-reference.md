# PegaFlow Server CLI 参数详解

PegaFlow 提供三个可执行的服务端二进制文件，各自有不同的 CLI 参数。

---

## 1. pegaflow-server（核心 KV 缓存服务）

主入口: `pegaflow-server/src/lib.rs` (Cli struct)

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--addr` | SocketAddr | `127.0.0.1:50055` | gRPC 服务监听地址 |
| `--log-level` | String | `info` | 日志级别: trace/debug/info/warn/error |
| `--devices` | Vec\<i32\> | 自动检测全部 GPU | CUDA 设备列表，逗号分隔，如 `--devices 0,1,2,3` |

### 内存池参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool-size` | 带单位字符串 | `30gb` | 钉扎内存池(pinned memory)总大小。支持 kb/mb/gb/tb 单位 |
| `--pool-shards` | usize | `1` | 内存池分片数量。多分片可降低分配器锁竞争，以 round-robin 方式分配 |
| `--use-hugepages` | bool | `false` | 启用大页(huge pages)分配钉扎内存。需事先通过 `/proc/sys/vm/nr_hugepages` 配置系统大页 |
| `--disable-numa-affinity` | bool | `false` | 禁用 NUMA 感知内存分配，改为使用单一内存池。默认按 NUMA 节点分配独立池 |
| `--blockwise-alloc` | bool | `false` | 逐块分配而非批量连续分配。当块释放顺序不同时可减少内存碎片化 |
| `--hint-value-size` | 带单位字符串 | 无 | 典型 value 大小的提示值，用于调优缓存和分配器参数 |

### 缓存策略参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-lfu-admission` | bool | `false` | 启用 TinyLFU 准入策略（默认纯 LRU）。TinyLFU 通过频率过滤低价值条目，提升缓存命中率 |

### SSD 缓存参数

SSD 缓存作为内存缓存的扩展层，使用 io_uring 异步 I/O 实现。设置 `--ssd-cache-path` 后整个 SSD 缓存功能才会启用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ssd-cache-path` | String | 无(不启用) | SSD 缓存文件路径。设置此参数即启用 SSD 缓存 |
| `--ssd-cache-capacity` | 带单位字符串 | `512gb` | SSD 缓存总容量 |
| `--ssd-write-queue-depth` | usize | `8` | SSD 写入队列深度（最大排队写入批次数） |
| `--ssd-prefetch-queue-depth` | usize | `2` | SSD 预取队列深度（最大排队预取批次数） |
| `--ssd-write-inflight` | usize | `2` | SSD 写入并发数（最大同时写入块数） |
| `--ssd-prefetch-inflight` | usize | `16` | SSD 预取并发数（最大同时读取块数） |
| `--max-prefetch-blocks` | usize | `800` | 允许处于预取状态的最大块数（SSD 预取背压控制） |

### RDMA / 跨节点传输参数

这些参数用于多节点部署（P2P RDMA 传输），`--nics` 和 `--metaserver-addr` 应当一起设置。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--nics` | Vec\<String\> | 无 | RDMA NIC 名称列表，如 `--nics mlx5_0 mlx5_1`。设置后钉扎内存会注册 RDMA 访问 |
| `--metaserver-addr` | String | 无 | MetaServer 地址，如 `http://127.0.0.1:50056`。设置后本节点会自动向 MetaServer 注册 block hash |
| `--metaserver-queue-depth` | usize | `256` | MetaServer 注册队列深度（最大排队注册批次数） |
| `--transfer-lock-timeout-secs` | u64 | `120` | 传输锁超时（秒）。跨节点 RDMA 传输期间的块锁定时长上限，超时后强制释放（崩溃恢复用） |

### 监控参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--http-addr` | SocketAddr | `0.0.0.0:9091` | HTTP 服务地址，提供健康检查和 Prometheus 指标 |
| `--enable-prometheus` | bool | `true` | 启用 `/metrics` Prometheus 端点 |
| `--metrics-otel-endpoint` | String | 无 | OTLP 指标导出 gRPC 端点，如 `http://127.0.0.1:4317` |
| `--metrics-period-secs` | u64 | `10` | OTLP 指标导出周期（秒） |
| `--trace-sample-rate` | f64 | `1.0` | 链路追踪采样率，0.0~1.0。如 0.01 表示 1% |

### HyperLogLog 统计参数

用于统计滑动窗口内的唯一 block 数量（基数估计）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--metric-hll-slot-secs` | u64 | `3600` (1小时) | HLL 时间槽轮转间隔 |
| `--metric-hll-window-secs` | u64 | `86400` (24小时) | HLL 滑动窗口持续时间 |
| `--metric-hll-bucket-bits` | u8 | `14` (16384桶, ~0.8%误差) | HLL 桶索引位数(4~18) |

### 典型使用示例

```bash
# 最简单启动（单节点开发）
cargo run -r --bin pegaflow-server

# 生产环境：指定 GPU、大内存池、开启 SSD 缓存
cargo run -r --bin pegaflow-server -- \
    --addr 0.0.0.0:50055 \
    --devices 0,1,2,3 \
    --pool-size 60gb \
    --use-hugepages \
    --pool-shards 4 \
    --ssd-cache-path /data/pegaflow/ssd_cache \
    --ssd-cache-capacity 1tb

# 多节点 P2P：开启 RDMA + MetaServer
cargo run -r --bin pegaflow-server -- \
    --addr 10.0.0.1:50055 \
    --pool-size 30gb \
    --nics mlx5_0 mlx5_1 \
    --metaserver-addr http://10.0.0.100:50056
```

---

## 2. pegaflow-metaserver（跨节点块索引注册中心）

主入口: `pegaflow-metaserver/src/lib.rs`

MetaServer 维护一个 block hash → 节点地址的映射表，供多节点间发现可远程获取的 KV 缓存块。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--addr` | SocketAddr | `127.0.0.1:50056` | gRPC 监听地址 |
| `--log-level` | String | `info` | 日志级别 |
| `--max-capacity-mb` | u64 | `512` | 缓存最大容量 (MB)，基于 moka 的 LRU 缓存 |
| `--ttl-minutes` | u64 | `120` | 缓存条目 TTL（分钟）。过期条目自动清除 |

```bash
# 启动 MetaServer
cargo run -r --bin pegaflow-metaserver -- \
    --addr 0.0.0.0:50056 \
    --max-capacity-mb 1024 \
    --ttl-minutes 60
```

---

## 3. pegaflow-router（P/D 分离路由器）

主入口: `pegaflow-server/src/bin/pegaflow-router.rs`

P/D 分离(Prefill/Decode disaggregation)路由器。将推理请求先发到 Prefill 节点（max_tokens=1，触发 KV 缓存生成），然后转发给 Decode 节点完成实际生成。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | String | `0.0.0.0` | HTTP 监听地址 |
| `--port` | u16 | `8080` | HTTP 监听端口 |
| `--prefill` | Vec\<String\> | **必填** | Prefill 节点端点列表（至少一个），如 `--prefill http://p1:8000 http://p2:8000` |
| `--decode` | Vec\<String\> | **必填** | Decode 节点端点列表（至少一个），如 `--decode http://d1:8000 http://d2:8000` |

路由器暴露两个 API 端点:
- `POST /v1/chat/completions` — 兼容 OpenAI Chat Completions API
- `POST /v1/completions` — 兼容 OpenAI Completions API

支持流式(SSE)和非流式响应。使用 round-robin 负载均衡。

```bash
# 1P1D 部署
cargo run -r --bin pegaflow-router -- \
    --port 8080 \
    --prefill http://10.0.0.1:8000 \
    --decode http://10.0.0.2:8000

# 2P2D 部署
cargo run -r --bin pegaflow-router -- \
    --prefill http://p1:8000 http://p2:8000 \
    --decode http://d1:8000 http://d2:8000
```

---

## 4. pegaflow-cpu-bench（RDMA 基准测试工具）

主入口: `pegaflow-transfer/src/bin/cpu_bench.rs`

CPU 内存 RDMA 延迟基准测试工具，模拟真实的 block 传输工作负载。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--block-size` | String | `4mb` | 块大小 |
| `--blocks-per-task` | String | `150` | 每个任务传输的块数，可为范围如 `100-200` |
| `--tasks` | usize | `50` | 测量的任务数量 |
| `--warmup-tasks` | usize | `5` | 热身任务数量（不计入统计） |
| `--mode` | String | `both` | 基准模式: `read`/`write`/`both` |
| `--nic` | String | 无 | 限定单个 NIC，如 `--nic mlx5_0` |
| `--exclude-nic` | String | 无 | 排除指定 NIC |
| `--numa` | u32 | 无 | 限定单个 NUMA 节点 |
