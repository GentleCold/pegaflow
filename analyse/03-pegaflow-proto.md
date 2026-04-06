# 03 — pegaflow-proto：gRPC 协议定义

**文件位置**：`pegaflow-proto/proto/engine.proto`  
**用途**：定义 PegaFlow 服务端与客户端（vLLM/SGLang）之间的 gRPC 接口契约

---

## 1. gRPC 与 Protobuf 简介

**Protocol Buffers（protobuf）**：Google 开发的二进制序列化格式，比 JSON 体积小 3-10 倍，序列化/反序列化更快。

**gRPC**：基于 protobuf 的高性能 RPC 框架：
- 使用 HTTP/2 多路复用（一个连接可并行多个请求）
- 支持双向流
- 代码自动生成（Rust 使用 `prost` 生成结构体，`tonic` 生成客户端/服务端代码）

**构建流程**：
```
engine.proto
    │
    ▼ prost/tonic 代码生成（build.rs）
    │
    ├── Rust 结构体（SaveRequest, LoadResponse, etc.）
    ├── Engine trait（服务端需实现此 trait）
    └── EngineClient（客户端调用）
```

---

## 2. Engine 服务详解

### 2.1 公共消息

```protobuf
// 所有响应都包含此状态
message ResponseStatus {
  bool ok = 1;       // 是否成功
  string message = 2; // 错误消息（失败时）
}
```

### 2.2 RegisterContextBatch — 注册 GPU KV 缓存上下文

**触发时机**：vLLM Worker 初始化时，每个 GPU 调用一次。

```protobuf
message RegisterContextRequest {
  string instance_id = 1;        // 模型实例 ID（如 "llama3-70b-instance-0"）
  string namespace = 2;          // 命名空间（通常是模型名）
  uint32 tp_rank = 3;            // 当前 GPU 的张量并行 rank（0 到 tp_size-1）
  uint32 tp_size = 4;            // 张量并行总度（如 4 表示 4 GPU）
  uint32 world_size = 5;         // 总 world size（通常等于 tp_size）
  int32 device_id = 6;           // CUDA 设备编号（如 0,1,2,3）
  uint32 num_layers = 7;         // 模型层数（如 Llama3-70B 有 80 层）

  // 以下字段每层一个（长度均为 num_layers）
  repeated string layer_names = 8;    // 层名（如 "model.layers.0"）
  repeated bytes wrapper_bytes = 9;   // CUDA IPC handle（序列化后的 GPU 内存句柄）
  repeated uint64 num_blocks = 10;    // 每层 KV 块的总数
  repeated uint64 bytes_per_block = 11; // 每块字节数
  repeated uint64 kv_stride_bytes = 12; // K/V 步长（从基地址到下一块的偏移）
  repeated uint32 segments = 13;      // 每块的段数（1=连续, 2=K/V 分开）
}
```

**重要字段解释**：
- `wrapper_bytes`：CUDA IPC handle（进程间共享 GPU 内存的"钥匙"），vLLM Worker 序列化后通过 gRPC 传给 PegaFlow，PegaFlow 用它打开对 GPU 显存的访问权限
- `segments`：决定每个 block 的存储布局（1 段 = K 和 V 连续存储；2 段 = K 和 V 分开存储，支持更高效的批量传输）

### 2.3 Save — 将 GPU KV 缓存保存到 CPU

**触发时机**：vLLM Worker 完成一批 token 的前向计算后，将 KV cache 异步卸载。

```protobuf
message SaveLayer {
  string layer_name = 1;          // 要保存的层名
  repeated int32 block_ids = 2;   // GPU KV 缓冲区中的块 ID（物理位置）
  repeated bytes block_hashes = 3; // 每块的内容哈希（用于内容寻址）
}

message SaveRequest {
  string instance_id = 1;
  uint32 tp_rank = 2;
  int32 device_id = 3;
  repeated SaveLayer saves = 4;  // 多层同时保存（一次 RPC 减少 N 次往返）
}
```

`block_ids` 是 GPU 缓冲区中的物理槽位编号（vLLM 内部的 block 管理器分配的），PegaFlow 用它计算实际的 GPU 内存地址：

```
GPU 内存地址 = base_ptr + block_id * kv_stride_bytes
```

### 2.4 Load — 从 CPU 恢复 KV 缓存到 GPU

**触发时机**：vLLM Scheduler 决定一个请求需要 KV cache（之前已卸载），Worker 发起 Load。

```protobuf
message LoadRequest {
  string instance_id = 1;
  uint32 tp_rank = 2;
  int32 device_id = 3;
  string load_state_shm = 4;    // 共享内存名（用于异步通知完成）
  repeated string layer_names = 5;  // 要加载的层列表
  repeated int32 block_ids = 6;    // 目标 GPU 槽位
  repeated bytes block_hashes = 7; // 要加载的块哈希
}
```

`load_state_shm`：共享内存名称，PegaFlow 写入完成状态，vLLM Worker 轮询此共享内存，避免阻塞 gRPC 连接。详见文档 `sync_state` 部分。

### 2.5 Query — 纯内存命中查询

```protobuf
message QueryRequest {
  string instance_id = 1;
  repeated bytes block_hashes = 2;  // 按前缀顺序排列
  string req_id = 3;                 // 请求 ID（用于 SSD 预取状态跟踪）
}

enum PrefetchState {
  PREFETCH_DONE = 0;    // 终止状态：已确定命中/缺失
  PREFETCH_LOADING = 1; // 过渡状态：正在从 SSD/远端加载，调用方应重试
}

message QueryResponse {
  ResponseStatus status = 1;
  uint64 hit_blocks = 2;           // 内存中命中的块数
  PrefetchState prefetch_state = 3; // 预取状态
  uint64 loading_blocks = 4;       // 正在从 SSD/RDMA 加载的块数
  uint64 missing_blocks = 5;       // 完全缺失的块数（SSD 和远端都没有）
}
```

**Query vs QueryPrefetch 的区别**：
| RPC | 触发 SSD 预取？ | 触发 RDMA 远端预取？ | 返回时机 |
|-----|----------------|----------------------|---------|
| `Query` | 否 | 否 | 立即返回内存命中数 |
| `QueryPrefetch` | 是 | 是 | 返回当前状态（可能是 LOADING） |

### 2.6 Unpin — 释放预查询的块

```protobuf
message UnpinRequest {
  string instance_id = 1;
  repeated bytes block_hashes = 2;  // 要取消 pin 的块
}
```

在 `QueryPrefetch` 命中后，块被 pin 住（防止 LRU 驱逐）。如果 Load 被取消（请求被抢占），需要 Unpin 释放引用。

### 2.7 跨节点传输相关 RPC

```protobuf
// TransferSlotInfo：单个 TP slot 的 RDMA 内存描述符
message TransferSlotInfo {
  uint64 k_ptr = 1;    // K 段在远端的内存地址（虚拟地址）
  uint64 k_size = 2;   // K 段字节数
  uint64 v_ptr = 3;    // V 段地址（0 表示与 K 连续）
  uint64 v_size = 4;   // V 段字节数（0 表示连续）
  uint32 numa_node = 5; // 内存所在 NUMA 节点（用于选择最近的 RDMA NIC）
}

// 每个 block 包含所有 TP slot 的描述符
message TransferBlockInfo {
  bytes block_hash = 1;
  repeated TransferSlotInfo slots = 2;  // 长度 = tp_size
}

message QueryBlocksForTransferRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;
  string requester_id = 3;  // 请求方标识（用于日志追踪和锁管理）
}

message QueryBlocksForTransferResponse {
  ResponseStatus status = 1;
  repeated TransferBlockInfo blocks = 2;
  string transfer_session_id = 3;  // 用于 ReleaseTransferLock
  uint32 lock_timeout_secs = 4;    // 客户端必须在此超时前完成 RDMA 传输
}
```

**跨节点 RDMA 传输流程**：
1. 节点 A 调用 `QueryBlocksForTransfer` → 节点 B 锁定 block，返回内存地址
2. 节点 A 用 `RdmaHandshake` 建立 RC 连接
3. 节点 A 发起 RDMA READ，直接读取节点 B 的内存
4. 节点 A 调用 `ReleaseTransferLock` 释放锁

### 2.8 RDMA 握手

```protobuf
message RdmaHandshakeRequest {
  string requester_id = 1;        // 客户端的可路由地址（作为连接标识）
  bytes handshake_metadata = 2;   // 客户端的 HandshakeMetadata（bincode 序列化）
}
// HandshakeMetadata 包含：QP 编号、GID、内存区域注册信息（rkey）
```

握手采用"服务端连接重用"模式：如果客户端发送空的 `handshake_metadata`，服务端返回缓存的本地元数据（避免重复建立 QP）。

---

## 3. MetaServer 服务详解

MetaServer 是独立进程，负责维护"哪个节点有哪些块"的全局索引。

```protobuf
service MetaServer {
  rpc InsertBlockHashes(InsertBlockHashesRequest) returns (InsertBlockHashesResponse);
  rpc QueryBlockHashes(QueryBlockHashesRequest) returns (QueryBlockHashesResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc Shutdown(ShutdownRequest) returns (ShutdownResponse);
}
```

### InsertBlockHashes — 注册块哈希

```protobuf
message InsertBlockHashesRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;   // 本节点新写入的 block 哈希列表
  string node = 3;                   // 本节点地址（如 "10.0.0.1:50055"）
}
```

PegaFlow 在 `insert_worker` 写入新 block 后，通过 `MetaServerClient::try_register()` 异步注册（fire-and-forget，不阻塞写路径）。

### QueryBlockHashes — 查询块所在节点

```protobuf
message QueryBlockHashesResponse {
  ResponseStatus status = 1;
  repeated bytes existing_hashes = 2;  // 存在的块哈希
  uint64 total_queried = 3;
  uint64 found_count = 4;
  repeated NodeBlockHashes node_blocks = 5;  // 按节点分组的结果
}

message NodeBlockHashes {
  string node = 1;           // 节点地址
  repeated bytes block_hashes = 2;  // 该节点拥有的块
}
```

节点 A 查询某些 block hash 时，MetaServer 返回"哪些块在哪些节点上"，节点 A 据此决定从哪个节点发起 RDMA 拉取。

---

## 4. Rust 代码生成

```rust
// pegaflow-proto/build.rs：
tonic_build::configure()
    .compile_protos(&["proto/engine.proto"], &["proto/"])
```

生成的代码存在 Rust 的构建缓存中（`target/` 目录），在其他 crate 中通过以下方式引用：

```rust
// pegaflow-server/src/proto.rs（或 lib.rs）：
pub mod engine {
    tonic::include_proto!("pegaflow");
    // 展开为所有生成的结构体和服务 trait
}
```

---

## 5. 与 HTTP/REST API 的对比

| 特性 | gRPC（PegaFlow 使用）| REST JSON |
|------|---------------------|-----------|
| 序列化 | Protobuf（二进制，紧凑）| JSON（文本，冗长）|
| 传输 | HTTP/2（多路复用）| HTTP/1.1 |
| 类型安全 | 编译期 schema 验证 | 运行时验证 |
| 代码生成 | 自动（client + server）| 手动或 OpenAPI |
| 流式支持 | 原生支持（双向流）| 需要 WebSocket |
| 性能 | 高（延迟低）| 中（overhead 大）|
