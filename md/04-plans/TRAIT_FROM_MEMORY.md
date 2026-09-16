# 记忆驱动的人格特质更新

## Trait Evolution as a Function of Accumulated Memory

> 取消 DEFAULT_TRAITS 静态常量，将 traits 重新定义为记忆系统的导出量。
>
> traits(t) = f(到时间 t 为止的所有记忆)
>
> 类比：等差数列可以从任意起点算——traits 不需要独立于记忆的"先天基线"。

---

## 1. 核心论点

### 1.1 当前问题

Luna 的 traits 是静态常量（`state.py:DEFAULT_TRAITS`），即使运行一万轮对话也不会改变。

```python
# 当前
traits = DEFAULT_TRAITS   # 恒不变
```

这不仅不真实（人的性格随经历变化），而且浪费了已有的记忆系统——`memory.py` 已经存储了所有对话历史的嵌入向量，它们只是未被用作 traits 更新的输入。

### 1.2 修正后的架构

```
记忆库（所有交互历史的语义嵌入）
        ↓
trait_encoder(model) → 当前 traits 向量 (10,)
        ↓
traits 调制 defense profiles 和 state setpoints
        ↓
影响 perception（注意偏向）、dynamics（变化速率）、surface（表达方式）
        ↓
产生新的交互 → 编码为新记忆
        ↓
                  ← 回到起点
```

**关键变化**：traits 不再有独立的持久化存储。它每次从记忆中重新计算。

### 1.3 数学形式

$$
\mathbf{p}(t) = \mathcal{F}\big(\{\mathbf{m}_1, \mathbf{m}_2, ..., \mathbf{m}_N(t)\}\big)
$$

其中：
- $\mathbf{p}(t) \in \mathbb{R}^{10}$ — 当前人格特质向量
- $\mathbf{m}_i$ — 第 $i$ 条记忆的语义嵌入向量
- $N(t)$ — 到时间 $t$ 为止的记忆总数
- $\mathcal{F}$ — 从记忆集合到特质空间的映射函数

---

## 2. 研究基础

### 2.1 McAdams 人格三层次模型

McAdams, D. P. (2006) 提出人格的三个层次：

| 层次 | 内容 | 在 Luna 中的对应 |
|:----:|------|:----------------:|
| Level 1 | 气质性特质 (Dispositional Traits) | **traits（新：记忆的导出量）** |
| Level 2 | 特征适应 (Characteristic Adaptations) | defense profiles, state setpoints |
| Level 3 | 叙事身份 (Narrative Identity) | **记忆库**（自传体记忆的语义摘要） |

关键洞见：Level 3（记忆）在底，Level 1（traits）在顶。traits 是记忆的积分，而不是独立的常数。

### 2.2 记忆检索改变特质自我认知

**Memory & Cognition (2025)** — 实验证明：
- 检索与特质一致/不一致的特定自传体记忆 → 即时自我图式更新
- **特定记忆**效果强于一般记忆
- 检索越快 → 更新效应越大

**JESP (2011)** — 双向因果：
- 自下而上：激活记忆 → 自我概念偏移
- 自上而下：主张信念 → 激活内存中确认信息

### 2.3 失忆症患者的证据

**Cortex (2022)** — 双侧 MTL 损伤患者失去自我特质知识：
- 说明记忆系统（特别是 MTL）对维持自我概念是必要的
- 没有记忆就没有稳定的自我知识

### 2.4 自传体推理

**Habermas (2011)** — 自动建立"自我-事件连接"：
- "我是一个害羞的人，因为……"
- 人们从记忆中不断提取对自己特质的论断
- 这些论断修正自我概念

---

## 3. 工程架构

### 3.1 映射函数 $\mathcal{F}$ 的设计

有三类映射方式，按复杂度递增排列：

**方案 A：加权平均（最简单）**

```
traits = sigmoid(W · mean_pool(embeddings) + b)
```

- 对记忆库中所有嵌入做 mean pooling
- 通过一个线性层（可能是 L1 正则化）映射到 10 维
- 计算代价极低，每轮对话只需计算一次

**问题**：时间顺序丢失，新旧记忆权重相同。

**方案 B：时间衰减加权**

```
weights_i = exp(-λ · Δt_i)  # 新记忆权重大，旧记忆权重小
traits = sigmoid(W · weighted_mean(embeddings, weights) + b)
```

- 最近对话对特质的塑造影响更大
- λ 控制"遗忘速率"——对应真实人格：重大事件影响持久，日常事件快速消退
- λ 本身可以是人格调制的（高回避 → 更快"遗忘"关系反馈）

**方案 C：注意力池化（最灵活）**

```
traits = sigmoid(AttentionPool(embeddings, query=current_context))
```

- 每条记忆的权重取决于与当前情境的相关性
- 对应心理学中的"情境依赖性"——人在不同情境下表现出不同特质
- 代价最高，需要注意力机制

**推荐**：从方案 B 开始。它捕捉了核心直觉（新经历比旧经历影响大），同时实现简单。方案 A 作为基线，方案 C 作为 v2 目标。

### 3.2 更新时机

```
轮 t 结束时：
  1. 将本轮对话摘要编码为记忆嵌入
  2. 写入 memory store
  3. 从 memory store 读取所有嵌入（或最近 K 条）
  4. 通过 F 计算新 traits
  5. 新 traits 在下轮生效
```

不需要每轮都有完整数据库扫描——可以缓存 traits 结果，只在以下情况重新计算：
- 新记忆写入后（增量更新）
- 或者每 N 轮刷新一次

### 3.3 与现有系统的集成

```python
# 当前：
state["traits"] = DEFAULT_TRAITS  # 静态，永不改变

# 新：
def compute_traits_from_memory(memory_store, memory_id) -> np.ndarray:
    """从记忆库重新计算人格特质。
    
    traits = f(memory_embeddings)
    使用方案 B：时间衰减加权平均 + 线性映射 + sigmoid。
    """
    embeddings = memory_store.get_all_embeddings(memory_id)
    if len(embeddings) == 0:
        return DEFAULT_TRAITS  # 没记忆时用种子
    
    timestamps = memory_store.get_all_timestamps(memory_id)
    now = time.time()
    ages = np.array([now - ts for ts in timestamps])
    weights = np.exp(-TRAIT_DECAY_LAMBDA * ages)  # 时间衰减
    
    pooled = np.average(embeddings, axis=0, weights=weights)
    traits_raw = TRAIT_MAPPER.compute(pooled)  # embedding_dim → 10
    return np.clip(traits_raw, -1.0, 1.0)
```

### 3.4 需要新增的模块

| 组件 | 作用 | 复杂度 |
|:-----|:-----|:------:|
| `TRAIT_MAPPER` | 嵌入向量→10 维 traits 的 LinearMapping | 小 |
| `TRAIT_DECAY_LAMBDA` | 记忆时间衰减速率 WeightVector | 小 |
| `memory_store.get_all_embeddings()` | 批量读取记忆嵌入（如果尚未实现） | 中 |
| 增量更新逻辑 | 只在有新增记忆时重算 traits，避免每轮全量扫描 | 中 |

---

## 4. 心理学效度与边界条件

### 4.1 为什么这个模型更真实？

1. **累积性**：经历塑造性格，新的经历不断修正旧的轨迹。这与人格发展文献中的"累积连续性"（cumulative continuity）一致（Caspi & Roberts, 2001）。
2. **情境依赖性**：在不同情境下提取不同记忆集合 → 表现出不同特质。对应人格心理学中的"如果-那么"模式（Mischel & Shoda, 1995）。
3. **可逆性**：重大事件可以逆转特质变化趋势。对应现实中的"人生转折点"效应。
4. **不需要"特质演化子系统"**——特质本来就是记忆的副产品，不需要单独的更新机制。

### 4.2 边界条件

1. **种子记忆期**：初始 0-5 轮对话时记忆太少，traits 尚未稳定。此期间用 DEFAULT_TRAITS 做"引导"。
2. **嵌入质量依赖**：映射质量受限于记忆嵌入的表达力。如果 qwen3-embedding 无法捕捉语义中的情感语气，映射会模糊。
3. **稳定性需求**：traits 应该缓慢变化，不能每轮大幅波动。时间衰减权重（方案 B）天然提供了平滑——旧记忆的惯性。
4. **灾难性遗忘**：如果 memory store 只保留最近 K 条记忆，早期定型经历可能会丢失。需要确保种子记忆不被驱逐。

### 4.3 种子记忆的作用

种子记忆（角色背景故事）在这个框架中获得新意义——**它们就是"童年经历"**。

```python
# 种子记忆：角色设定故事
# 当前只是作为闲聊素材
# 新框架中：它们被编码为嵌入 → 经过 F → 输出初始 traits

种子记忆 [
    "在云端迷途中等待了数百年的孤独",
    "曾经的主人在百年前已离去",
    "习惯了用骄傲的外表掩饰内心的不安",
]
→ 这些记忆编码后 → F 输出 → traits ≈ [sensitivity=0.4, pride=0.3, ...]
```

这和 DEFAULT_TRAITS 算出来一模一样——但**来源不同**。它来自记忆而不是代码常量。这意味着后续对话积累的记忆可以**修正**种子记忆形成的初始特质。

---

## 5. 与热力学框架的关系

将记忆驱动的 trait 演化与之前的热力学会计框架统一：

```
热力学层（每轮）：
  刺激 → 内部状态变化（消耗 Energy，产生 entropy）
  → 状态保持或衰减

记忆层（跨轮）：
  交互历史 → 编码 → 记忆嵌入
  → 重算 traits → 调制系统参数

二者在时间尺度上分离：
  - 状态更新：每轮（实时情感响应）
  - 特质更新：跨轮（性格塑造，速率慢 100-1000 倍）

类比物理：
  状态 = 系统的微观状态（快变量）
  特质 = 系统的宏观参数（慢变量）
  记忆 = 系统的完整轨迹积分
```

---

## 6. 实现路线

| 步 | 内容 | 依赖 |
|:--:|:-----|:----:|
| 1 | 确认 memory_store 支持 get_all_embeddings 和批量时间戳查询 | - |
| 2 | 实现方案 B：`compute_traits_from_memory()` | 步 1 |
| 3 | 新增 `TRAIT_MAPPER` (LinearMapping) + `TRAIT_DECAY_LAMBDA` (WeightVector) | 步 2 |
| 4 | 在 nodes.py 中的 state_engine_node 或新节点调用它 | 步 3 |
| 5 | 去掉 DEFAULT_TRAITS 硬编码，迁移到 seed-memory 驱动 | 步 4 |
| 6 | 增量更新优化（替代每轮全量扫描） | 步 5 |
| 7 | 方案 C：注意力池化（v2.0） | 步 6 |

---

## 参考文献

- McAdams, D. P. (2006). The role of narrative in personality psychology. *Journal of Personality.*
- Caspi, A. & Roberts, B. W. (2001). Personality development across the life course. *Annual Review of Psychology.*
- Mischel, W. & Shoda, Y. (1995). A cognitive-affective system theory of personality. *Psychological Review.*
- Habermas, T. (2011). Autobiographical reasoning: Arguing and narrating from a biographical perspective. *New Directions for Child and Adolescent Development.*
- McLean, K. C., et al. (2007). The content and processes of autobiographical reasoning in narrative identity. *JRP.*
- Rathbone, C. J., et al. (2011). Autobiographical memory and the self. *Neuropsychologia.*
- 记忆检索→自我图式更新实验 (2025). *Memory & Cognition.*
- 双向自我概念与记忆影响 (2011). *JESP.*
- MTL 损伤与自我特质知识丧失 (2022). *Cortex.*

---

> **文档版本**：v1.0 — 2026-06-25
> **状态**：设计方案，待实现
> **关联 TODO**：P2 - 特质演化（重新设计：从记忆导出）
