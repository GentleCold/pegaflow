# 算法原理：TLSF（Two-Level Segregated Fit）

TLSF 是一种**实时系统内存分配算法**，核心目标：**O(1) 时间复杂度**的 malloc/free，且碎片率极低（< 25%）。

---

## 1. 问题背景

```
glibc malloc（dlmalloc/ptmalloc）：
  - 平均 O(1)，但最坏情况 O(n)（需遍历 free list）
  - 不适合实时系统（中断处理、音视频编解码等）

TLSF 目标：
  - 严格 O(1)，有确定性上界
  - 碎片率 < 25%（实测）
  - 适合嵌入式/实时系统，也适合高性能服务器
```

---

## 2. 核心数据结构：两级位图 + 空闲链表

### 两级分类

```
第一级（FL，First Level）：按 2 的幂次划分大小区间
  [2^i, 2^(i+1)) → FL = i

第二级（SL，Second Level）：每个 FL 区间再等分为 2^SL_INDEX_COUNT 个子格
  假设 SL_INDEX_COUNT=4（16 个子格）

例：FL=6（[64, 128)），SL 把它等分成 16 份：
  [64, 68)   → SL=0
  [68, 72)   → SL=1
  ...
  [124, 128) → SL=15
```

### 整体结构

```
fl_bitmap:     一个整数，bit[i]=1 表示 FL=i 下有空闲块
sl_bitmap[FL]: 每个 FL 对应一个整数，bit[j]=1 表示该 (FL,SL=j) 下有空闲块
free_list[FL][SL]: 双向链表，存放该大小类的所有空闲块

┌────────────────────────────────────────────────┐
│ fl_bitmap:  0 0 1 1 0 1 0 0 ...                │
│                 ↑ ↑   ↑                        │
│                 │ │   └── FL=5 有空闲块          │
│                 │ └────── FL=4 有空闲块          │
│                 └──────── FL=3 有空闲块          │
├────────────────────────────────────────────────┤
│ sl_bitmap[4]:  0 1 0 0 1 0 0 0 ...             │
│ sl_bitmap[5]:  1 0 0 1 0 0 0 0 ...             │
├────────────────────────────────────────────────┤
│ free_list[4][1]: block_A ↔ block_C             │
│ free_list[4][4]: block_B                       │
│ free_list[5][0]: block_D ↔ block_E             │
│ free_list[5][3]: block_F                       │
└────────────────────────────────────────────────┘
```

---

## 3. 关键操作：mapping

**mapping_insert(size) → (fl, sl)**：给定块大小，计算它应进入哪个 (fl, sl)

```
fl = floor(log2(size))           ← 最高有效位位置，用 CLZ 指令 O(1)
sl = (size >> (fl - SL_INDEX_COUNT)) & (SL_COUNT - 1)
                                 ← 取次高 SL_INDEX_COUNT 位
```

图示（SL_INDEX_COUNT=4，size=100）：
```
100 = 0b1100100
        ↑
        fl = 6（最高位在第 6 位）

取第 6 位之后的 4 位: 0b1001 = 9 → sl = 9

所以 100 字节的块放入 free_list[6][9]
```

**mapping_search(size) → (fl, sl)**：查找时需向上取整，确保找到的块一定够用：

```
size_rounded_up = size + (1 << (fl - SL_INDEX_COUNT)) - 1
再对 rounded_up 做 mapping_insert
```

---

## 4. malloc 流程（严格 O(1)）

```
1. mapping_search(size) → (fl, sl)

2. 在 sl_bitmap[fl] 中找 ≥ sl 的第一个置 1 的 bit：
   suitable_sl = FFS(sl_bitmap[fl] >> sl) + sl   ← FFS 指令，O(1)

3. 若 sl_bitmap[fl] 的 sl 之后没有置 1 的 bit：
   suitable_fl = FFS(fl_bitmap >> (fl+1)) + fl+1
   suitable_sl = FFS(sl_bitmap[suitable_fl])      ← 仍是 O(1)

4. 从 free_list[suitable_fl][suitable_sl] 取链表头块
   若链表变空，清零对应 bitmap bit

5. 若取出的块比需求大 → split：
   多余部分插入对应 free_list（O(1)）

总步骤数有严格上界，整体 O(1)
```

---

## 5. free 流程（严格 O(1)）

```
1. 检查物理相邻的前一个块是否空闲（块头存有前块大小）
   → 若是，合并（coalesce），从 free_list 摘出

2. 检查物理相邻的后一个块是否空闲
   → 若是，同样合并

3. mapping_insert(merged_size) → (fl, sl)
   将合并后的块插入对应 free_list，更新 bitmap

所有操作均为指针操作 + 位运算，O(1)
```

---

## 6. 块的内存布局（Boundary Tag）

```
┌──────────────────────────┐  ← 前一个物理块末尾
│ prev_phys_block (指针)    │  用于 O(1) 找到前邻块做合并
├──────────────────────────┤
│ size | free_bit | prev_free │  大小 + 2 个状态标志位
├──────────────────────────┤
│ prev_free_block (指针)    │  ┐ 空闲时才有效
│ next_free_block (指针)    │  ┘ 双向链表节点
├──────────────────────────┤
│       用户数据区           │
└──────────────────────────┘
```

Boundary Tag 技术：块尾部也存有块大小，使得从任意块出发都能在 O(1) 内定位物理相邻块。

---

## 7. 碎片分析

**内部碎片**（分配块比请求大）：
```
最坏情况：请求 size 落在 SL 子区间下边界，
          分配到该子区间上边界的块
          浪费 ≤ 1/SL_COUNT = 6.25%（SL_INDEX_COUNT=4 时）
```

**外部碎片**（零散空闲块无法利用）：
```
TLSF 采用立即合并（immediate coalescing）策略：
  free() 时立刻与相邻空闲块合并，不延迟
  外部碎片率理论上界 < 25%
```

---

## 8. 与其他分配器对比

| 分配器 | malloc | free | 碎片率 | 适用场景 |
|--------|--------|------|--------|---------|
| dlmalloc | 平均 O(1)，最坏 O(n) | 平均 O(1) | 低 | 通用 |
| jemalloc | O(1) | O(1) | 低 | 多线程服务器 |
| tcmalloc | O(1) | O(1) | 低 | Google 服务 |
| **TLSF** | **严格 O(1)** | **严格 O(1)** | < 25% | **实时/高性能** |
| buddy system | O(log n) | O(log n) | 高 | OS 内核页分配 |

---

## 9. 与 PegaFlow 的关系

PegaFlow 的 pinned memory pool（`pegaflow-core/src/pinned_pool.rs`）需求与 TLSF 高度契合：

- **高频分配/释放**：每个 block seal/evict 都要分配/释放固定大小内存
- **低延迟确定性**：不能在关键路径上出现不确定延迟
- **NUMA 感知**：在指定 NUMA 节点上分配钉扎内存

`--pool-shards N` 参数将整个内存池分成 N 个独立子池（每个子池一套 TLSF 结构），以 round-robin 分配，降低多线程对单一分配器的锁竞争。
