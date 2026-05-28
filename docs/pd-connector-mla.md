# PegaFlow PD Connector MLA Support

## 背景

`pegaflow.pd_connector` 现有 PD push 路径主要面向 FlashAttention HND KV cache：

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

这个 layout 里每个 layer 有 K/V 两段，P/D TP rank 也是一一对应。MLA 模型通常走
FlashMLA，KV cache 是 blocks-first 3D：

```text
[num_blocks, block_size, head_size]
```

MLA 的 KV cache 在 TP ranks 之间是 replicated，不需要按 head dimension split。
因此在 `prefill_tp > decode_tp` 时，一个 D rank 只需要接收一个代表 P rank 推来的
完整 MLA KV。

DeepSeek-V3.2 / DeepSeek-V4 这类模型还可能有 indexer cache layer。indexer layer 也
是 3D blocks-first cache，但它的 page size、head size、alignment 或压缩比例可能和主
MLA layer 不同。PD connector 不能把 `is_mla=True` 理解成“所有 layer 共用一个全局
layout”；layout、region 数量和 block_len 都必须是 per-layer / per-tensor 的。

NIXL 已经支持 logical block size 和 kernel physical block size 不一致，例如
logical=128、physical=64，并用 `physical_blocks_per_logical_kv_block` 把 scheduler
logical block 展开成 kernel physical block。PD connector MLA 第一版先不实现这层 split：
如果 MLA KV cache 的 `logical_block_size != physical_block_size`，直接抛异常。这样先把
layout/message/TP mapping 跑通，后续再把内部进度单位扩展到 physical sub-block。

## 目标

- 支持 MLA / FlashMLA KV cache 的 PD push。
- 支持带 indexer cache layer 的 MLA model；layout 按每个 layer/tensor 识别。
- 第一版支持 `prefill_tp >= decode_tp`，且 `prefill_tp % decode_tp == 0`。
- MLA 第一版要求 `logical_block_size == physical_block_size`。
- 复用 NIXL 的 transfer-region / block-len / TP-mapping 语义；PD 只增加 push 和 IMM 完成信号。
- 保持非 MLA FlashAttention HND 路径语义不变。
- 错误走异常，配置、layout、TP mapping 不满足约束时 fail fast。
- 保持 P 端 layer-wise RDMA push + IMM done 的时序，不退回 NIXL pull 模型。

## 非目标

- 第一版不支持 `prefill_tp < decode_tp`。这个方向需要一个 P rank push 到多个 D rank，
  native request/remote state 要扩展为一对多。
- 第一版不支持 MLA logical/physical block split。`logical=128, physical=64` 这类
  配置直接抛异常。NIXL 支持这类配置，但 PD push 先不承担这部分复杂度。
- 第一版不把 SWA layer 作为必须支持项。若模型暴露出当前 schema 无法表达的 SWA cache
  layout，注册 KV cache 时直接抛异常；本设计的必达范围是 MLA + indexer layer。
- 第一版不支持 DCP/PCP 与 MLA PD push 叠加。检测到 `dcp_world_size != 1` 或
  `pcp_world_size != 1` 时在启动或注册 KV cache 阶段抛异常。
- 不支持非 MLA 的 heterogeneous TP。非 MLA 需要 head split，当前 PD push 继续要求
  P/D TP 一致。
- 不做 CPU/SSD 中转回退。RDMA 注册或传输失败时由 D 本地重算。

## 现有实现需要弱化的假设

MLA 方案不需要推翻 PD push 主流程。D 侧仍然先注册 remote layout 并异步等 IMM，P 侧仍然
在 `save_kv_layer()` 里 layer-wise push，最后在所有 WRITE 完成后发 IMM。

需要从当前实现里抽出来的是三个局部假设：

- 5D HND layout：当前 `FlashAttnHndLayout` 只描述 `[2, B, block, H, D]`。
- K/V 两段：metadata 和 native RDMA schema 现在用 `k_block_addrs/v_block_addrs` 表达。
- TP 1:1：P worker 现在按 `handshake.tp_rank == local_tp_rank` 选目标 D rank。

新的设计以 NIXL 的 transfer cache region 为核心描述每个 layer，用 per-layer layout 决定
本地 slice 和 remote address，再用 NIXL 等价的 rank plan 处理 P TP > D TP。

`PdConnector.get_required_kvcache_layout()` 对 MLA 返回 `None`，让 vLLM / FlashMLA 使用默认
backend layout；非 MLA 继续返回 `"HND"`：

```python
if model_config.use_mla:
    return None
return "HND"
```

`is_mla` 只控制这两个行为：

- `get_required_kvcache_layout()` 是否返回 `None`。
- P/D TP mapping 是否走 MLA replicated-cache 规则。

`is_mla` 不控制具体 layer 选什么 layout class。layout class 必须由每个 cache tensor 的
shape/spec 决定。

## Layout 结构

### NIXL 对齐点

第一版 PD MLA 不重新发明 KV cache metadata。下面这些语义直接向 NIXL 靠齐：

- `register_kv_caches(kv_caches: dict[str, Tensor])` 是注册入口，但每个 layer 还要读取
  `layer_spec`。NIXL 在 DSv32 indexer 场景会把 `UniformTypeKVCacheSpecs` 展开到具体
  layer spec。
- 每个 layer 拆成若干 transfer cache regions。非 blocks-first HND 是 K/V 两个 regions；
  MLA 和 indexer 是一个 region。
- 每个 region 用 `base_addr + block_id * block_len` 定位 block，不传每个 block 的地址矩阵。
- 每层可以有不同 `block_len`。NIXL 用 `block_len_per_layer` 支持这一点；PD 对应为
  per-layer/per-region `block_len`。
- `physical_blocks_per_logical_kv_block` 沿用 NIXL 语义。PD MLA 第一版只接受 `1`，否则
  注册 KV cache 阶段抛异常。
- TP mapping 复用 NIXL `compute_tp_mapping()` 的数学语义；PD push 方向只是把
  “D rank 读 P rank”反过来变成“代表 P rank 写 D rank”。

### KvCacheLayout

每个 layer 注册一个 `KvCacheLayout` 实例，不维护全局 region 数量或全局 `block_len`：

```python
@dataclass(frozen=True)
class TransferRegionLayout:
    region_idx: int
    base_addr: int
    block_len: int


class KvCacheLayout(Protocol):
    layer_name: str
    regions: tuple[TransferRegionLayout, ...]
    logical_block_size: int
    physical_block_size: int

    @property
    def num_blocks(self) -> int: ...

    def block_slices(self, block_id: int) -> LayerBlockSlices: ...

    def remote_layout(
        self,
        layer_idx: int,
        block_ids: set[int] | None = None,
    ) -> LayerRemoteLayout: ...

    def touched_blocks_from_slot_mapping(self, slot_mapping: Any) -> set[int]: ...
```

`block_id` 在第一版里始终是 scheduler logical block id。MLA 第一版要求 logical block 和
physical block 一致，所以不需要 sub-block id。

### Layout 选择

`PdWorkerConnector.register_kv_caches()` 对每个 layer/tensor 单独选择 layout。这里和
NIXL 对齐：入口拿到的是 `dict[layer_name, tensor]`，但不能只靠 tensor shape 推导全部信息；
需要同时使用 vLLM 的 per-layer cache spec。NIXL 在注册时会读取
`self._layer_specs[layer_name]`，并在 DSv32 indexer 场景下把 `UniformTypeKVCacheSpecs`
展开到具体 layer spec，再用 `layer_spec.page_size_bytes` 记录 per-layer block length。

`register_kv_caches()` 不需要改 vLLM 接口。vLLM 已经在 connector 构造函数里传入
`kv_cache_config`，当前 PD connector 只是没有继续使用它。P0 改法对齐 NIXL：

```python
class PdConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        if kv_cache_config is None:
            raise ValueError("PdConnector requires kv_cache_config")

        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = PdSchedulerConnector(vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self._worker = PdWorkerConnector(vllm_config, kv_cache_config)
```

worker 初始化时保留 NIXL 同款映射：

```python
self.kv_cache_config = kv_cache_config
self.num_blocks = kv_cache_config.num_blocks
self._layer_specs = {
    layer_name: group.kv_cache_spec
    for group in kv_cache_config.kv_cache_groups
    for layer_name in group.layer_names
}
```

`register_kv_caches()` 只做 lookup：

```python
layer_spec = self._layer_specs[layer_name]
layout = layout_from_tensor(layer_name, tensor, layer_spec, ...)
```

如果某个 `kv_caches` layer 不在 `_layer_specs`，说明 vLLM cache config 和实际注册 tensor
已经不一致，直接抛异常。

```python
def layout_from_tensor(
    layer_name: str,
    cache_tensor: torch.Tensor,
    layer_spec: KVCacheSpec,
    *,
    logical_block_size: int,
    physical_blocks_per_logical: int,
) -> KvCacheLayout:
    if isinstance(layer_spec, UniformTypeKVCacheSpecs):
        layer_spec = layer_spec.kv_cache_specs[layer_name]

    if physical_blocks_per_logical != 1:
        raise ValueError(
            "PD MLA requires physical_blocks_per_logical_kv_block == 1 in the first version"
        )

    shape = tuple(cache_tensor.shape)

    if len(shape) == 3:
        return MlaBlocksLayout.from_tensor(
            layer_name,
            cache_tensor,
            layer_spec,
            logical_block_size,
        )

    if len(shape) == 5 and shape[0] == 2:
        return FlashAttentionHndLayout.from_tensor(
            layer_name,
            cache_tensor,
            layer_spec,
            logical_block_size,
        )

    raise ValueError(f"unsupported PD KV cache layout for {layer_name}: shape={shape}")
```

这样主 MLA layer、indexer cache layer 都走 3D `MlaBlocksLayout`，且各自保留自己的
`block_len`。shape 只负责选择 layout class；`layer_spec` 负责校验 page bytes、indexer
这类 per-layer 差异和后续 logical/physical split。第一版仍然要求
`physical_blocks_per_logical_kv_block == 1`，否则注册 KV cache 时抛异常。如果后续需要支持
SWA，也应新增 layout class 或明确复用现有 class，而不是靠全局 `is_mla` 分支。

### FlashAttentionHndLayout

现有 `FlashAttnHndLayout` 改成实现 `KvCacheLayout`：

- `regions = (K region, V region)`
- `logical_block_size == physical_block_size == shape[2]`
- 每个 region 的 `block_len = block_size * num_kv_heads * head_size * element_size`
- `touched_blocks_from_slot_mapping()` 使用 `slot // block_size`

这对应 NIXL 的 `split_k_and_v=True`：HND tensor 的 dim0 被拆成两个 transfer regions，
分别注册 K 和 V。

### MlaBlocksLayout

新增 `MlaBlocksLayout`。它同时覆盖主 MLA cache 和 DeepSeek indexer cache，只要求 tensor 是
3D blocks-first：

```text
shape   = [num_blocks, block_size, head_size]
strides = [block_stride, row_stride, 1]
```

校验：

- `len(shape) == 3`
- `block_size > 0`
- `logical_block_size == shape[1]`
- `stride[2] == 1`
- `stride[1] == head_size`

`block_len` 优先使用 `layer_spec.page_size_bytes`，并用 tensor stride 和 dtype 做一致性校验：

```python
physical_page_size = layer_spec.page_size_bytes // physical_blocks_per_logical
region_block_len = physical_page_size // len(regions)
block_len_from_tensor = cache_tensor.stride(0) * cache_tensor.element_size()
assert region_block_len == block_len_from_tensor
```

这样 indexer layer 即便和主 MLA layer 的 `head_size`、alignment 或 page bytes 不同，也能按
自己的 tensor layout 注册。这和 NIXL 的 `block_len_per_layer.append(physical_page_size)` 是
同一个信息。

这个公式刻意照着 NIXL 的两步除法：

```python
physical_page_size = layer_spec.page_size_bytes // physical_blocks_per_logical
physical_page_size = physical_page_size // len(cache_list)
```

对 MLA/indexer，`physical_blocks_per_logical=1` 且 `len(regions)=1`，所以 `region_block_len`
就是 `layer_spec.page_size_bytes`。对 HND K/V 分 region 时，`layer_spec.page_size_bytes`
表示 K+V 整页，单个 K 或 V region 的 `block_len` 要除以 2。

如果 `logical_block_size != shape[1]`，直接抛异常。这个条件会拒绝
logical=128、FlashMLA physical=64 的配置：

```text
PdConnector MLA requires logical block size to match FlashMLA block size:
logical=128 physical=64
```

MLA / indexer 的 `block_slices(block_id)` 返回一个 region slice：

```text
region_idx        = 0
block_id          = block_id
src_offset_bytes  = block_id * block_len
bytes             = block_len
```

### num_blocks 不变量

PD 第一版也沿用 NIXL 的 `num_blocks` 检查：所有参与 transfer 的 cache tensor 必须和
`kv_cache_config.num_blocks` 对齐。

NIXL 在注册阶段要求每个 cache tensor 的 `shape[0] == num_blocks`。PD 对应规则：

- 3D MLA/indexer：`shape[0] == kv_cache_config.num_blocks`
- 5D HND：`shape[1] == kv_cache_config.num_blocks`
- handshake 中的 `block_ids` 必须对每个 layer 都满足 `0 <= block_id < layout.num_blocks`

第一版不支持同一个 request 对不同 layer 使用不同 block-id range。如果真实 indexer 模型里
indexer layer 的 `num_blocks` 和主 MLA layer 不一致，注册 KV cache 阶段直接抛异常；P0 要用
真实 MLA+indexer 模型打印并确认 `cache_tensor.shape[0]` 是否一致。

## 控制面消息

PD connector 有两类 `kv_transfer_params`。

### Router 到 D: ConsumerKvParams

```json
{
  "do_remote_prefill": true,
  "prefill_url": "http://prefill:8001",
  "remote_request_id": "req-p",
  "done_request_id": "req-d"
}
```

D scheduler 收到后：

1. `get_num_new_matched_tokens()` 返回需要 remote prefill 的 token 数。
2. `update_state_after_alloc()` 记录 D 侧 block ids。
3. `build_connector_meta()` 把 `WaitReqMeta` 发给 D worker。

### D 到 P: ProducerKvParams

D rank0 dispatch prefill request 时，把所有 D rank handshakes 放进 P 请求：

```json
{
  "do_remote_prefill_sender": true,
  "target_engine_id": "decode-engine",
  "target_request_id": "req-d",
  "pd_handshakes": [
    {
      "request_id": "req-d",
      "engine_id": "decode-engine",
      "tp_rank": 0,
      "tp_size": 4,
      "block_size": 64,
      "layers": []
    },
    {
      "request_id": "req-d",
      "engine_id": "decode-engine",
      "tp_rank": 1,
      "tp_size": 4,
      "block_size": 64,
      "layers": []
    }
  ]
}
```

这里 `tp_rank` 和 `tp_size` 表示 D 侧 rank，而不是 P 侧 rank。P worker 必须用
`PushRankPlan` 选择目标 handshake，不能把 D rank 当成本地 P rank。

### PdHandshake

```python
@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int        # decode-side rank
    tp_size: int        # decode-side TP size
    block_size: int     # scheduler logical block size
    layers: tuple[LayerRemoteLayout, ...]
    imm_id: int | None = None
```

### LayerRemoteLayout

wire format 第一版切到 NIXL-style regions，不再在协议层保留 `k_block_addrs/v_block_addrs`。
旧 K/V 字段可以在 `FlashAttentionHndLayout` 内部作为便利方法存在，但不进入 P/D 控制面消息。

NIXL 的 handshake metadata 发的是 `kv_caches_base_addr` 和 `block_lens`，后续按
`base_addr + block_id * block_len` 建 transfer descriptors。PD handshake 是 request-scoped，
但保持同一套地址语义：每个 layer 发本 request 的 `block_ids`，再发每个 region 的
`base_addr/block_len`。

```python
@dataclass(frozen=True)
class RemoteTransferRegion:
    region_idx: int
    base_addr: int
    block_len: int


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    block_ids: tuple[int, ...]
    regions: tuple[RemoteTransferRegion, ...]
    mr_desc: Any | None = None
```

不变量：

- `regions` 里的 `region_idx` 必须覆盖 `[0, len(regions))`
- `block_ids` 是 scheduler logical block ids
- 同一个 request 中不同 layer 可以有不同的 region 数量和 `block_len`

示例：

```text
FlashAttention HND layer:
  regions[0] = K region
  regions[1] = V region

MLA / indexer layer:
  regions[0] = MLA/indexer region
```

native remote state 按 layer 存，不允许假设 request-level global region count：

```rust
struct PdRemoteRegion {
    base_addr: u64,
    block_len: u64,
}

struct PdRemoteLayer {
    mr_desc: MemoryRegionDescriptor,
    allowed_block_ids: HashSet<u64>,
    regions: Vec<PdRemoteRegion>,
}
```

`regions.len()` 是这个 layer 自己的 region count。`push_layer()` 处理同一个 request 的不同
layer 时，每层都从自己的 `PdRemoteLayer` 读取 region 信息。remote address 直接按 NIXL
规则计算：

```text
remote_addr = region.base_addr + block_id * region.block_len
```

### LayerBlockSlices

P worker 提交给 native RDMA 的本地 slice 用 region id 做 key，避免 tuple 位置和
`region_idx` 双重编码：

```python
@dataclass(frozen=True)
class BlockRegionSlice:
    block_id: int
    src_offset_bytes: int
    bytes: int


@dataclass(frozen=True)
class LayerBlockSlices:
    regions_by_idx: Mapping[int, BlockRegionSlice]
```

序列化给 native 时按 `region_idx` 排序展开：

```python
for region_idx, region in sorted(slices.regions_by_idx.items()):
    ...
```

native `push_layer()` 对每个 `(region_idx, BlockRegionSlice)`：

1. 校验 `region_idx < remote_layer.regions.len()`。
2. 校验 `block_id` 在 `allowed_block_ids` 中。
3. 用 `region.base_addr + block_id * region.block_len` 计算远端地址。
4. 用 `src_offset_bytes` 找本地 MR offset。
5. 提交一个 RDMA WRITE target。

### Local memory registration

native `register_local_layers()` 不能再假设每层有 K/V 两个 region。它应从每个
`KvCacheLayout` 的 regions 计算本 layer 需要注册的连续 MR range，和 NIXL 的
`caches_data = (base_addr, num_blocks * block_len, device_id, "")` 语义一致：

- HND layer：region 0/1 分别覆盖 K/V blocks，可共享一个 tensor MR，也可以按实际地址范围合并。
- MLA / indexer layer：只有 region 0，一个 layer 一个连续 3D tensor range。

注册结果仍按 layer 存，`push_layer()` 用 `src_offset_bytes` 计算本地 offset。这样主 MLA
和 indexer layer 的 bytes/page 不同也不会互相影响。

## P TP > D TP

### Rank Plan

新增 `python/pegaflow/pd_connector/tp_mapping.py`，但它只是 NIXL MLA 规则的 push 侧投影，
不是一套新的 TP mapping。NIXL pull 侧在 `compute_tp_mapping()` 里对 MLA 使用：

```python
attn_ranks = [tp_rank * remote_tp_size // tp_size]
```

PD push 侧把变量反过来：D rank `d` 会读 P rank
`d * prefill_tp_size // decode_tp_size`，所以 P rank `p` 只有在
`p % (prefill_tp_size // decode_tp_size) == 0` 时才是 representative rank。

```python
@dataclass(frozen=True)
class PushRankPlan:
    should_push: bool
    target_decode_rank: int | None


def build_mla_push_rank_plan(
    *,
    prefill_tp_rank: int,
    prefill_tp_size: int,
    decode_tp_size: int,
) -> PushRankPlan:
    if prefill_tp_size < decode_tp_size:
        raise ValueError("MLA PD push requires prefill_tp >= decode_tp")
    if prefill_tp_size % decode_tp_size != 0:
        raise ValueError("prefill_tp must be divisible by decode_tp")
    ratio = prefill_tp_size // decode_tp_size
    if prefill_tp_rank % ratio != 0:
        return PushRankPlan(should_push=False, target_decode_rank=None)
    return PushRankPlan(
        should_push=True,
        target_decode_rank=prefill_tp_rank // ratio,
    )
```

非 MLA 第一版：

```python
def build_non_mla_push_rank_plan(
    *,
    prefill_tp_rank: int,
    prefill_tp_size: int,
    decode_tp_size: int,
) -> PushRankPlan:
    if prefill_tp_size != decode_tp_size:
        raise ValueError("non-MLA PD push requires matching P/D TP size")
    return PushRankPlan(should_push=True, target_decode_rank=prefill_tp_rank)
```

### P8/D4 示例

```text
prefill_tp = 8
decode_tp  = 4
ratio      = 2
```

| P rank | plan | target D rank |
| ---: | --- | ---: |
| 0 | push | 0 |
| 1 | skip | - |
| 2 | push | 1 |
| 3 | skip | - |
| 4 | push | 2 |
| 5 | skip | - |
| 6 | push | 3 |
| 7 | skip | - |

等价于 NIXL 读路径：

```text
D0 reads P0  <=>  P0 pushes D0
D1 reads P2  <=>  P2 pushes D1
D2 reads P4  <=>  P4 pushes D2
D3 reads P6  <=>  P6 pushes D3
```

### P worker 状态

`PrefillHandler.process_push_reqs()` 不再无条件 open RDMA request：

```python
plan = build_mla_push_rank_plan(...)
if plan.should_push:
    handshake = select_decode_handshake(req.handshakes, plan.target_decode_rank)
    rdma.open_request(req_id, handshake)
    state = ActivePush(req=req, plan=plan, handshake=handshake)
else:
    state = ActivePush(req=req, plan=plan, handshake=None)
```

规则：

- `PUSHING` rank 执行 layer-wise RDMA WRITE，并在所有 WRITE 完成后发 IMM。
- `SKIPPED` rank 不调用 `rdma.open_request()`，不提交 layer push，不发 IMM。
- `SKIPPED` rank 仍然保留 request lifecycle，直到本地 producer forward 完成。

### Completion

vLLM `KVOutputAggregator` 默认期待每个 TP worker 都上报 completion。P TP > D TP 时，只有
representative ranks 做 RDMA，但所有 P ranks 都参与 producer request forward。

skipped rank 可以在本地 producer forward 完成后直接上报 `finished_sending`，不需要等待
representative rank 的 RDMA 完成：

- `PUSHING` rank：`local_done = producer_finished & rdma_completed`
- `SKIPPED` rank：`local_done = producer_finished`

原因有两个：

- representative rank RDMA 读取的是它自己的 local KV cache，不读取 skipped rank 的 KV cache。
  MLA cache 是 replicated，但每个 TP rank 的实际 tensor/MR 是本 rank 私有的。
- scheduler 不会因为某一个 skipped worker 先上报就释放 request blocks。`request_finished()`
  对 producer request 返回 `delay_free_blocks=True`，真正 free blocks 发生在 scheduler 收到
  聚合后的 `finished_sending` 之后；而 `KVOutputAggregator` 默认等所有 TP workers 都上报同一个
  req id，representative rank 没完成时，scheduler 看不到这个 req 的 `finished_sending`。

所以第一版继续沿用 vLLM 的 all-rank completion 契约，不新增 TP-group collective。skipped rank
提前清理 connector-local push state 没问题；这不会释放 block allocator 里的 blocks。

后续可以优化成只让 representative ranks 上报，并把 expected finished count 改成
`decode_tp_size`。但当前 `get_finished()` 接口只返回 `(finished_sending, finished_recving)`，
动态 `expected_finished_count` 没有从 connector 传出的干净通路；第一版不做这个优化。

### D worker 状态

D 侧每个 rank 只等自己的 IMM：

1. D worker 构造本 rank handshake。
2. D rank0 dispatch 所有 D rank handshakes 给 P。
3. D rank i 只 open/wait 本地 request。
4. 代表 P rank 写入 D rank i 的 KV cache 后发 IMM。
5. D rank i 上报 `finished_recving`。

D 不需要知道哪些 P ranks skipped。skipped 是 P 侧 rank plan 的内部决策。

## Block ID

第一版 block id 始终表示 scheduler logical block id。

### MLA 第一版

MLA 第一版要求：

```text
logical_block_size == physical_block_size
```

所以：

```text
remote block id == local block id == slot // block_size
```

`MlaBlocksLayout.touched_blocks_from_slot_mapping()`：

```python
return {int(slot) // self.logical_block_size for slot in slots if int(slot) >= 0}
```

`slot_mapping` 语义必须在 P0 用真实 MLA/indexer 模型确认并写入实现注释或测试记录，不能留到
P1 之后再发现需要返工。第一版假设它仍然表示 token slot；如果 FlashMLA/indexer 暴露的是
kernel block slot，`touched_blocks_from_slot_mapping()` 要单独分支处理。

### 后续支持 logical/physical split

NIXL 的做法是记录 `physical_blocks_per_logical_kv_block`，并把 logical block ids 展开成
kernel physical block ids。PD push 后续要支持这件事，需要一起改：

- `LayerRemoteLayout` 带 physical sub-block 地址。
- `LayerBlockSlices` 使用 physical sub-block id。
- `_remote_block_id_map()` 按 sub-block 映射。
- `ChunkTracker` 从 `(layer_idx, block_id)` 升级到
  `(layer_idx, logical_block_id, physical_subblock_idx)`。
- IMM 只能在所有 required sub-block 都完成后发送。

这部分不放进 MLA 第一版。

## 启动期校验

fail fast 的检查尽量放在 `PdConnector.__init__` 或 `register_kv_caches()`：

- `is_mla and dcp_world_size != 1`：抛异常。
- `is_mla and pcp_world_size != 1`：抛异常。
- `is_mla and prefill_tp_size < decode_tp_size`：P 收到 producer params 后抛异常。
- `is_mla and prefill_tp_size % decode_tp_size != 0`：P 收到 producer params 后抛异常。
- `not is_mla and prefill_tp_size != decode_tp_size`：P 收到 producer params 后抛异常。
- layer tensor shape 不是 3D MLA/indexer 或 5D HND：注册 KV cache 时抛异常。
- layer 不存在于 `kv_cache_config.kv_cache_groups`：注册 KV cache 时抛异常。
- layer `num_blocks` 和 `kv_cache_config.num_blocks` 不一致：注册 KV cache 时抛异常。
- MLA/indexer `physical_blocks_per_logical_kv_block != 1`：注册 KV cache 时抛异常。
- MLA/indexer `logical_block_size != physical_block_size`：注册 KV cache 时抛异常。

DCP/PCP 不应等到第一个 push request 才报错；否则用户会在服务启动成功后才遇到运行期失败。

## 实现步骤

### P0: layout 和 message

- `PdConnector.get_required_kvcache_layout()` 对 MLA 返回 `None`。
- `PdConnector.__init__()` 必须接收并保存 `kv_cache_config`；WORKER/SCHEDULER 构造都传下去。
- `PdWorkerConnector.__init__()` 像 NIXL 一样从 `kv_cache_config.kv_cache_groups` 构造
  `_layer_specs`。
- 新增 NIXL-style transfer region dataclasses：`base_addr/block_len/region_idx`。
- `FlashAttnHndLayout` 实现 `KvCacheLayout`。
- 新增 `MlaBlocksLayout`，覆盖主 MLA layer 和 indexer cache layer。
- `PdWorkerConnector.register_kv_caches()` 按每个 tensor 的 shape/spec 选择 layout。
- 按 NIXL 两步除法校验 `region_block_len = page_size_bytes / physical_blocks_per_logical / region_count`。
- 所有 layer 的 `num_blocks` 必须等于 `kv_cache_config.num_blocks`。
- MLA/indexer `physical_blocks_per_logical_kv_block != 1` 时抛异常。
- MLA/indexer `logical_block_size != physical_block_size` 时抛异常。
- Rust `PdRdmaEngine` 支持 regions-only schema。
- native `register_local_layers()` 支持每层不同 region count、`block_len` 和 MR range。

### P1: rank plan 和 completion

- 新增 `pd_connector/tp_mapping.py`。
- `PrefillHandler.process_push_reqs()` 构造 `ActivePush`。
- `PUSHING` rank 选择目标 D handshake。
- `SKIPPED` rank 不 open RDMA request。
- `get_finished_sending()` 对 `PUSHING` rank 等 `producer_finished & rdma_completed`。
- `get_finished_sending()` 对 `SKIPPED` rank 只等 `producer_finished`。
- 不改 `get_finished_count()`，让 `KVOutputAggregator` 继续按 all-rank completion 聚合。

### P2: scheduler/worker metadata

- `ProducerKvParams` 继续携带 all-D-rank `pd_handshakes`。
- `PushReqMeta` 不需要新增 target rank 字段，P worker 可用本地 TP 信息和 D handshakes 推导。
- 日志里打印 `prefill_tp_rank/prefill_tp_size/decode_tp_size/plan/target_decode_rank`。
- 增加 skipped rank counter，至少包含 `engine_id/request_id/prefill_tp_rank/decode_tp_rank`。
- 增加 per-layer layout debug 日志：`layer_name/layout_kind/region_count/block_len/block_size`。

### P3: 正确性测试

默认 Python tests 不依赖 GPU、vLLM、native extension。新增测试：

- MLA 3D tensor layout：logical=64、physical=64、head=576。
- indexer 3D tensor layout：同一个 request 中 `block_len` 与主 MLA layer 不同。
- HND K/V region 的 `block_len` 校验使用 `page_size_bytes / 2`。
- indexer 和主 MLA layer 的 `shape[0]` 都等于 `kv_cache_config.num_blocks`。
- indexer `shape[0]` 与 `kv_cache_config.num_blocks` 不一致时抛异常。
- MLA 3D tensor layout：logical=128、physical=64，抛异常。
- MLA/indexer `physical_blocks_per_logical_kv_block=2`，注册 KV cache 阶段抛异常。
- MLA/indexer remote layout：一个 block 产生一个 region slice。
- mixed per-layer region count：同一个 request 内 layer A region=1、layer B region=2，native adapter
  按 layer 展开。
- regions-only schema roundtrip：wire format 不包含 `k_block_addrs/v_block_addrs`。
- P8/D4 mapping：P0/P2/P4/P6 push，P1/P3/P5/P7 skip。
- TP mapping 与 NIXL `compute_tp_mapping(is_mla=True)` 的读侧结果互为反向映射。
- skipped P ranks 在 producer forward 完成后上报 `finished_sending`。
- scheduler 只有在 aggregator 收到所有 P ranks 的 `finished_sending` 后才释放 producer blocks。
- 不引入 TP-group collective，避免 `get_finished_sending()` 路径出现额外同步。
- 非 MLA P8/D4 直接报错。
- DCP/PCP guard 在 connector 初始化或 KV cache 注册阶段报错。
- `slot_mapping` 在 MLA/indexer 下的语义 probe，有测试覆盖 token-slot 到 block-id 的计算。

### P4: GPU E2E

触发条件：connector-visible cache semantics、MLA、indexer layer、PD push、heterogeneous TP。

建议 E2E：

```bash
cd python
uv run --extra test pytest -m e2e tests/test_vllm_pd_mla_e2e.py \
  --model /data/models/<mla-model-with-indexer> \
  --prefill-tp 8 \
  --decode-tp 4 \
  --max-model-len 4096
```

验收：

- D 输出与 direct vLLM baseline 一致。
- 主 MLA layer 和 indexer layer 都完成 remote write。
- 主 MLA layer 和 indexer layer 注册日志里的 `num_blocks` 一致。
- P representative ranks 有 RDMA push 日志。
- P skipped ranks 无 RDMA push，本地 producer forward 完成后有 `finished_sending`。
- D 每个 rank 都收到 IMM。
- `logical_block_size != physical_block_size` 时启动或注册 KV cache 阶段抛异常。

## 失败路径

- layout 不匹配：注册 KV cache 时抛异常。
- MLA/indexer `physical_blocks_per_logical_kv_block != 1`：注册 KV cache 时抛异常。
- MLA/indexer logical/physical block size 不一致：注册 KV cache 时抛异常。
- 同一 layer 的 region count 与 native remote state 不一致：open remote request 时抛异常。
- P/D TP 不满足第一版约束：P worker 收到 producer params 时抛异常。
- representative P rank RDMA WRITE 失败：上报 transfer failure，D 本地重算。
- skipped P rank 不能吞掉异常：只有明确判定 `should_push=False` 后才能跳过 RDMA。
- D rank 等 IMM 超时：与现有 wait_done 失败路径一致，标记 load failure。

## 后续方向

1. 支持 MLA logical/physical split，按 NIXL 的 `physical_blocks_per_logical_kv_block` 语义扩展。
2. 评估 SWA layer 是否需要作为 PD connector MLA 路径的一等支持对象。
3. DCP + MLA 后续可以借鉴普通 `PegaKVConnector` 的 effective TP rank/size 语义，但 PD push 的
   P/D rank mapping 要重新审一遍。
4. 一对多 push 支持 `prefill_tp < decode_tp` 时，native `PdRemoteRequest` 需要支持一个
   producer request 绑定多个 remote handshakes。
