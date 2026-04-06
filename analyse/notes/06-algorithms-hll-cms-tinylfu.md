# 算法原理：HyperLogLog / Count-Min Sketch / TinyLFU

---

## 1. HyperLogLog (HLL)

### 问题

统计一个数据流中**不同元素的数量**（基数估计，Cardinality Estimation）。

朴素方案：HashSet 精确计数，内存 O(n)，1000万元素 ≈ 数百 MB。  
HLL 目标：用 **固定几 KB 内存**，误差约 0.8%。

### 直觉：前导零与概率

对任意哈希函数 `h`，其输出均匀分布在 [0, 2^32)：

```
前导零数 k 出现的概率 = 1 / 2^(k+1)

k=0 → 概率 1/2   (最常见)
k=1 → 概率 1/4
k=2 → 1/8
...
k=9 → 1/1024

观察到最大前导零数为 k_max
→ 直觉上大约见过了 2^(k_max+1) 个不同元素
```

单个估计方差极大，HLL 通过**分桶取调和平均**大幅降低误差。

### 算法步骤

```
初始化：m 个桶（m = 2^b，b 是 bucket_bits），每桶存一个计数器 M[j] = 0

对每个新元素 x：
  1. 计算 h = hash(x)
  2. 取 h 的前 b 位作为桶编号 j
  3. 取 h 的剩余位，计算前导零数 + 1，记为 ρ
  4. M[j] = max(M[j], ρ)

估计基数：
  Z = 1 / Σ(2^(-M[j]))    ← 调和平均
  E = α_m × m² × Z

  其中 α_m 是修正常数（消除系统性偏差）
```

图示：

```
hash(x) = 0101 | 00011010...
           ↑↑↑↑   ↑↑↑↑↑↑↑↑
           桶编号   剩余位，前导零 = 2，ρ = 3
           j = 5

M[5] = max(M[5], 3)
```

### 误差与内存

```
标准误差 ≈ 1.04 / √m

m = 2^14 = 16384 桶 → 误差 ≈ 0.81%
内存 = 16384 × 6bit ≈ 12KB（每桶最多记到 64，需 6bit）
```

### 滑动窗口 HLL（PegaFlow 实现）

`pegaflow-common/src/hll.rs` 实现了时间分片的滑动窗口：

```
时间轴（每格 = 1 slot = 1小时）：

│ slot_t-23 │ ... │ slot_t-1 │ slot_t(当前) │
└──────────────────────────────────────────┘
              window = 24小时

每隔 slot_secs 创建新槽，丢弃窗口外的旧槽
查询 = 所有槽的 HLL 做 union（取每桶最大值）后估计基数
```

**Union 操作**：两个 HLL 合并只需对应桶取 max，合并后误差不增加。

用途：统计过去 24h 内访问过多少个**唯一 block hash**，作为 Prometheus 监控指标。

对应参数：
```
--metric-hll-slot-secs   3600   # 每槽时长
--metric-hll-window-secs 86400  # 窗口总长
--metric-hll-bucket-bits 14     # 桶数 = 2^14，误差 ~0.8%
```

---

## 2. Count-Min Sketch (CMS)

### 问题

估计数据流中**每个元素出现的频率**，同样要求内存远小于 O(n)。

用途：TinyLFU 中统计 block 访问频率，作为准入决策依据。

### 核心结构

一个 `d 行 × w 列` 的整数矩阵，配合 `d` 个独立哈希函数：

```
         列 0   列 1   列 2  ...  列 w-1
行 0  [  0      3      1    ...    2   ]   ← 哈希函数 h0
行 1  [  1      0      4    ...    0   ]   ← 哈希函数 h1
行 2  [  2      1      0    ...    3   ]   ← 哈希函数 h2
```

### 操作

**更新**（元素 x 出现一次）：
```
for i in 0..d:
    col = h_i(x) % w
    matrix[i][col] += 1
```

**查询**（估计 x 的频率）：
```
freq(x) = min over i of matrix[i][h_i(x) % w]
```

图示：
```
x = "block_abc_hash"

h0(x) = 3 → matrix[0][3] += 1
h1(x) = 7 → matrix[1][7] += 1
h2(x) = 1 → matrix[2][1] += 1

查询 freq(x) = min(matrix[0][3], matrix[1][7], matrix[2][1])
```

### 为什么取 min？

哈希碰撞会导致计数**虚高**（只会多不会少），取所有行的最小值能得到最接近真实值的**上界估计**：

```
真实频率 ≤ 估计频率（取 min 尽量压低上界偏差）

误差界：P(估计误差 > ε·N) < δ
其中 w = ⌈e/ε⌉，d = ⌈ln(1/δ)⌉
（e ≈ 2.718，N = 总元素数）
```

### Aging（频率衰减）

若不衰减，历史高频 key 永久占据优势，新热点无法进入缓存。  
TinyLFU 的做法：当计数器总和达到阈值（通常 = 缓存容量的 10 倍），**所有计数器右移 1 位（除以 2）**。

```
衰减前：A=100, B=80, C=60
衰减后：A=50,  B=40, C=30
新热点 D 从 0 开始，几次访问后就能与老数据竞争
```

---

## 3. TinyLFU / W-TinyLFU

### 问题

LRU 的缺陷：一次性扫描请求会将热数据驱逐出缓存（缓存污染）。

### TinyLFU 准入过滤器

在 LRU 驱逐前加一道"门槛"：

```
新候选项 candidate（即将进缓存）
被驱逐项 victim（即将出缓存）

if freq(candidate) > freq(victim):
    允许替换
else:
    candidate 被拒，victim 留在缓存
```

频率查询依赖上面的 Count-Min Sketch。

### W-TinyLFU 三区结构

纯 TinyLFU 问题：新 key 历史频率为 0，永远进不来。  
解法：加一个 **Window 区**（小型 LRU）作为缓冲：

```
容量分配（以 100 为例）：
  Window LRU:      1%  = 1 个槽
  Probationary:   19%  = 19 个槽  ┐
  Protected:      80%  = 80 个槽  ┘ 合称 Main Cache

数据流向：
新请求 ──▶ Window LRU
              │ 满了，淘汰 candidate_w
              ▼
         TinyLFU 门卫
         比较 freq(candidate_w) vs freq(victim_main)
              │ candidate 胜
              ▼
         Probationary LRU（观察期）
              │ 再次命中
              ▼
         Protected LRU（热数据区，不易被驱逐）
```

### 各区的作用

| 区域 | 容量 | 作用 |
|------|------|------|
| Window LRU | ~1% | 新 key 缓冲区，避免频率为 0 就被拒 |
| Probationary | ~19% | 观察期，只有二次命中才晋升 |
| Protected | ~80% | 热数据保护区，频繁访问的 key 集中于此 |

### 与 LRU 的命中率对比（Caffeine 论文数据）

```
工作负载         LRU 命中率    W-TinyLFU 命中率
─────────────────────────────────────────────
搜索引擎 trace      45%           62%
数据库 trace        35%           55%
均匀随机            33%           33%  ← 无优势（无热点）
```

结论：存在热点访问规律时 W-TinyLFU 显著优于 LRU；纯随机场景两者相当。

### PegaFlow 中的适用场景

```
--enable-lfu-admission 开启 W-TinyLFU 准入

适合：
  - 多用户共享系统 prompt prefix（高频 block）
  - 少量热点 token 序列被反复访问
  - 内存紧张需最大化命中率

不适合 / 无需开启：
  - 每个请求的 KV 都完全独特（无热点）
  - 访问模式已经接近顺序（LRU 本来就好）
  - 调试阶段追求行为可预测
```

---

## 三者的关系

```
W-TinyLFU（缓存策略）
    └── 依赖 Count-Min Sketch 统计访问频率
            └── 利用哈希函数多行取 min 估计频率

HyperLogLog（监控指标）
    └── 独立使用，统计滑动窗口内唯一 block 数
            └── 利用哈希前导零数 + 分桶调和平均估计基数
```

两者都是**概率数据结构**，以可控的误差换取极低的内存占用，是大规模缓存系统的常用基础组件。
