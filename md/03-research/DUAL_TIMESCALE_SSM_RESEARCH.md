# 双速 SSM：多时间尺度情感动力学研究综述

> **日期**: 2026-06-24 | **范围**: 基于 ~30 篇论文/框架的跨学科综合研究
> **目的**: 为 Luna 状态引擎从"扁平残差动力学"升级到"快慢双速 SSM"提供学术依据

---

## 摘要

Luna 当前的状态引擎采用统一时间尺度的残差动力学：

$$
h_t = h_{t-1} + dt \cdot (\alpha \cdot \Delta_{\text{coupling}} + \Delta_{\text{stimulus}})
$$

所有状态维度在同一时间尺度上更新，带来三个无法回避的问题：

1. **维度灾难**：11 维扁平向量中语义边界模糊，PC1 解释 52%+ 方差
2. **无天然低通滤波**：关系状态对单轮刺激的响应度与内部状态在同一量级
3. **时间结构缺失**：无法区分"当下心跳加速"和"长期暧昧感"

本综述基于神经科学（iEEG）、计算情感建模（EMA/HED/LSEF/MATE）、心理学理论（Helson AL/Solomon 对立过程）的跨学科调研，为双速 SSM 架构提供完整的理论基础。

---

## 目录

1. [神经科学证据：快慢情感的独立通路](#1-神经科学证据快慢情感的独立通路)
2. [计算模型：EMA 的单一评价层](#2-计算模型ema-的单一评价层)
3. [加权平均：Helson AL + HED 模型](#3-加权平均helson-al--hed-模型)
4. [低秩稀疏分解：LSEF 框架](#4-低秩稀疏分解lsef-框架)
5. [对立过程：Solomon B-process](#5-对立过程solomon-b-process)
6. [MATE 架构：双过程习惯化 + O-U 调节](#6-mate-架构双过程习惯化--o-u-调节)
7. [综合设计映射](#7-综合设计映射)
8. [参考文献](#8-参考文献)

---

## 1. 神经科学证据：快慢情感的独立通路

### 1.1 核心论文

**Kakusa, M., et al.** (2025). *Distinct neural temporal architectures encode rapid social expressions and sustained internal mood states.* bioRxiv.

**方法**：连续多天颅内 iEEG 记录（2,037 个电极触点），同时自动监测面部表情和定期心境评估。

### 1.2 关键发现

| 通路 | 时间尺度 | 脑区 | 神经编码 | 解码准确率 |
|:-----|:--------:|:-----|:---------|:----------:|
| **快速表达** (面部表情) | 秒级 | 外侧颞叶皮层 | 非周期性活动 (aperiodic) | 79.5% |
| **持续心境** | 分钟-小时 | 边缘系统 (limbic) | 低 γ 功率 | 个体差异大（5/12） |

**最重要的结论**：优化用于解码面部表情的神经特征**完全无法预测**持续心境状态，反之亦然——两个系统使用**计算上独立的神经机制**。

### 1.3 对 Luna 设计的直接影响

既然大脑已经使用两套独立的神经系统处理快/慢信号，Luna 的 internal/surface（快）和 relationship（慢）每轮接收**同源但不同加工深度**的刺激是完全合理的：

| 层 | 对应的神经通路 | 更新频率 | 加工深度 |
|:---|:-------------|:--------:|:---------|
| internal / surface | 外侧颞叶（快速表达） | 每轮 | 浅加工，原始刺激直接驱动 |
| relationship | 边缘系统（持续心境） | N轮一次 | 深加工，加权积累后驱动 |

*Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12724620/)*

---

## 2. 计算模型：EMA 的单一评价层

### 2.1 核心论文

**Marsella, S.C. & Gratch, J.** (2009). *EMA: A process model of appraisal dynamics.* Cognitive Systems Research, 10(1), 70-90.

### 2.2 核心论点

EMA 提出了与"多层评价"（快速模式匹配 + 慢速推理评价）不同的架构：

> **反对多层评价假设，提出单一自动评价过程**。快慢差异不来自评价机制本身，而来自评价之前的**信息处理深度**。

### 2.3 架构

```
感知(快) ──→ causal interpretation ──→ appraisal(自动) ──→ emotion
推理(慢) ──↗      ↑
              coping(调节)
```

- **Causal Interpretation**（因果解释）：agent 对"自己与环境关系"的主观认知，包含信念、欲望、意图、计划
- **Appraisal**（评价）：一组持续活跃的特征检测器，将因果解释映射到评价变量（合意性、可能性、可控制性等）
- **Coping**（应对）：评价的逆过程——修改因果解释的某些特征

### 2.4 对 Luna 设计的直接影响

EMA 支持"**不需要专门的慢速评价模块，快慢来自输入处理深度**"的架构：

```
每轮:
  inner_stimuli(7D)                    ← 浅加工（原始感知）
    ├──→ internal[t] += Δ(inner)       ← EMA "快路径"：每轮更新
    │
    └──→ s_rel += α·(inner - s_rel)    ← EMA "慢路径"：加权积累
          if cnt%N==0:
            rel += Δ(s_rel, dt=N)       ← 深加工，低频更新
```

这与 EMA 的"同一评价机制处理不同时间尺度的输入"完全一致。

*Source: [PDF](https://people.ict.usc.edu/~gratch/papers/COGSYS-RS-EMOTION-2008-6.pdf)*

---

## 3. 加权平均：Helson AL + HED 模型

### 3.1 Adaptation-Level Theory

**Helson, H.** (1964). *Adaptation-Level Theory.* Harper & Row.

核心公式：

$$
AL = (\bar{X})^p \cdot B^q \cdot R^r
$$

其中：
- $\bar{X}$ = 焦点刺激的几何均数
- $B$ = 背景/情境刺激
- $R$ = 残余刺激（过往经验、生理状态、记忆痕迹）
- $p + q + r = 1$ = 经验确定的权重系数

**情绪强度** = 刺激相对于 AL 的偏差：

$$
u_t = X_t - AL_t
$$

### 3.2 HED 计算模型

**Steephen, J.E.** (2013). *HED: A Computational Model of Affective Adaptation and Emotion Dynamics.* IEEE Transactions on Affective Computing, 4(2), 197-210.

将 AL 理论计算化——**递推加权平均**：

$$
AL_t = \alpha \cdot X_{t-1} + (1 - \alpha) \cdot AL_{t-1}
$$

$$
u_t = X_t - AL_t
$$

其中 $\alpha$ 控制适应速度：
- $\alpha \to 1$：快速适应（只关注最新刺激）
- $\alpha \to 0$：慢速适应（积累全部历史）

### 3.3 HED-ID 扩展

**Steephen et al.** (2020). *HED-ID: An Affective Adaptation Model Explaining the Intensity-Duration Relationship of Emotion.* IEEE TAC, 11, 736-750.

情绪呈**负加速指数衰减**：

$$
I(t) = I_0 \cdot e^{-\lambda t}
$$

解释了"相同的刺激重复发生时情绪响应越来越小"——即你的加权积累方案中**权重 `w_t` 应随重复递减**的数学形式。

### 3.4 评价条件化加权平均

**Ingendahl, M., et al.** (2024). *The Interplay of Multiple Unconditioned Stimuli in Evaluative Conditioning: A Weighted Averaging Framework.* Journal of Personality and Social Psychology, 127(5), 964-985.

用 10 个实验证明态度的形成遵循严格的加权平均规则：

$$
Attitude = \frac{\Sigma(w_i \cdot v_i)}{\Sigma(w_j)}
$$

其中：
- $v_i$ = 每次刺激的效价
- $w_i$ = 刺激权重（由强度/显著性决定）
- **负性刺激权重大于正性刺激**（消极偏见 —— 匹配 Luna 已有的 FAB 非对称衰减 ×1.8）

### 3.5 对 Luna 设计的直接影响

```
关系刺激缓冲区 = AL 的递推加权平均实现:

s_rel[t] = τ · s_rel[t-1] + (1-τ) · inner_stimuli[t]

其中 τ = α 控制适应速度:
  τ=0.3: 快速适应，关系层灵敏
  τ=0.7: 慢速适应，关系层惯性大

事件权重 w_t:
  w_t = ‖inner_stimuli‖₁ / 7 + γ · ST_EMOTIONAL_WEIGHT[t]
  
  与 Ingendahl 一致: 高强度刺激 → 高权重 → 对关系影响更大
```

*Source: [IEEE TAC 2013](https://dl.acm.org/doi/10.1109/T-AFFC.2013.2) | [JPSP 2024](https://www.ovid.com/journals/jpspy/abstract/10.1037/pspa0000401)*

---

## 4. 低秩稀疏分解：LSEF 框架

### 4.1 核心论文

**Cui, F.-Q., et al.** (2025). *Robust Low-Rank Sparse Framework for Video-Based Affective Computing.* arXiv:2511.11406.

### 4.2 核心公式

将情感动态分解为低秩基底 + 稀疏瞬态：

$$
E(t) = L(t) + S(t)
$$

| 分量 | 数学性质 | 时间尺度 | 心理学对应 |
|:-----|:---------|:--------:|:-----------|
| $L(t)$ 低秩情感基底 | 稳定、光滑 | 分钟-小时 | 持续心境、关系累积 |
| $S(t)$ 稀疏瞬态信号 | 高频、判别性 | 秒-分钟 | 即时情绪反应 |

### 4.3 核心模块

| 模块 | 功能 | 数学操作 |
|:-----|:-----|:---------|
| **SEM** (Stability Encoding) | 提取低秩基底 | Gaussian 低通滤波 |
| **DDM** (Dynamic Decoupling) | 分离稀疏瞬态 | 时域门控 + 图正交化 |
| **CIM** (Consistency Integration) | 重建多尺度信号 | $E = L + S$ |
| **RAO** (Rank-Aware Optimization) | 平衡梯度平滑/敏感度 | 动态秩调节 |

### 4.4 对 Luna 设计的直接影响

**直接映射到双速 SSM**：

```
每轮 t:
  inner_stimuli = E_raw(t)                    # 原始情感信号
  
  # SEM 等价: 关系缓冲区 = 低通滤波
  s_rel[t] = τ·s_rel[t-1] + (1-τ)·inner[t]   # L(t) 低秩基底
  
  # DDM 等价: 内部态 = 瞬态信号
  internal[t] += Δ(inner[t])                  # S(t) 稀疏瞬态
  
  # CIM 等价: 表面 = 重建
  surface[t] = project(internal, rel)          # E(t) = L + S 重建
```

*Source: [arXiv:2511.11406](https://arxiv.org/abs/2511.11406)*

---

## 5. 对立过程：Solomon B-process

### 5.1 经典理论

**Solomon, R.L. & Corbit, J.D.** (1974). *Opponent-Process Theory of Acquired Motivation.*

每个情感事件触发两个过程：

| 过程 | 时间常数 | 方向 | 峰值 | 衰减 |
|:-----|:--------:|:----|:----:|:----:|
| **A-process** | 快（秒） | 与刺激同向 | 高 | 快速 |
| **B-process** | 慢（分-时） | 与刺激反向 | 低 | 缓慢（~4× A） |

### 5.2 MATE 的实现

**Lobozov (2026)** 在 MATE 中将其量化为：每个情感尖峰触发延迟的反向摆动，B-process 衰减比 A-process 慢 4 倍。

### 5.3 对 Luna 设计的意义

对立过程解释了为什么要分离快慢层：
- **内部态** = A-process（快速响应刺激，快速恢复）
- **关系态** = B-process（延迟响应，缓慢恢复 —— 通过加权积累缓冲实现自然的"延迟反向摆动"）

Luna 的 `rel_counter` 缓冲 + 刺激加权平均已经部分实现 B-process 的延迟特性。

---

## 6. MATE 架构：双过程习惯化 + O-U 调节

### 6.1 核心论文

**Lobozov, S.** (2026). *MATE: A Deterministic Affective Middleware for LLM-Based Companions with Emergent Character and Persistent Internal State.* v8.0. Zenodo.

### 6.2 双过程习惯化（模块 #2）

MATE 实现了两阶段情绪处理：

| 阶段 | 时间尺度 | 功能 | 数学形式 |
|:-----|:--------:|:-----|:---------|
| 快状态 | 每轮 | 瞬态情感响应 | identity mapping |
| 慢状态 | 累积 | 习惯化基线漂移 | 低通滤波 |

这正好对应 Luna 的：
- 每轮 internal 更新
- `s_rel` 缓冲区积累后驱动 relationship

### 6.3 稳态情绪调节（模块 #7）

Ornstein-Uhlenbeck 过程：

$$
dX(t) = \theta(\mu - X(t))dt + \sigma dW(t)
$$

| 参数 | 功能 | Luna 对应 |
|:-----|:-----|:-----------|
| $\theta$ | 回归速率 | SELF_DECAY（已有差异化实现） |
| $\mu$ | 回归基线 | setpoint（compute_setpoint） |
| $\sigma$ | 扩散强度 | Langevin 噪声 σ=0.015（06-24 新增） |
| $dW$ | Wiener 过程 | 每维独立高斯噪声 |

### 6.4 对 Luna 的验证

Luna 已有的设计完全落入 O-U 框架：
- 时间衰减 = $\theta$ 项（lambda 衰减率）
- Langevin 噪声 = $\sigma dW$ 项（06-24 新增）
- setpoint = $\mu$ 项（compute_setpoint）
- 差异化 SELF_DECAY = 维度特异性的 $\theta_i$

*Source: [Zenodo v8.0](https://zenodo.org/records/20400530)*

---

## 7. 综合设计映射

### 7.1 学术支撑全景

| Luna 设计元素 | 学术支撑 | 来源 |
|:--------------|:---------|:-----|
| **internal 每轮更新** | 快速情感通路的神经独立性 | Kakusa 2025 |
| **surface 从 internal+rel 投影** | EMA 的单一评价层 | Marsella & Gratch 2009 |
| **relationship 加权积累 N轮一次** | Helson AL + HED 加权平均 | Helson 1964; Steephen 2013 |
| **刺激缓冲区 s_rel = α·s + (1-α)·s_rel** | HED 递推 AL | Steephen 2013 |
| **事件触发更新** | Episode-Contingent Sampling | Revol 2025 |
| **Langevin 噪声** | O-U 过程扩散项 | Lobozov 2026 |
| **低秩 + 稀疏分解** | LSEF 框架 | Cui 2025 |
| **快慢分离** | 双过程习惯化 | Lobozov 2026 |
| **负性权重偏见** | 评价条件化加权平均 | Ingendahl 2024 |
| **非对称衰减** | 对立过程 B-process | Solomon 1974 |
| **差异化 SELF_DECAY** | O-U 维度特异性 θ | Steephen 2013 |

### 7.2 最终设计的数学形式

```
每轮 t:

  ① 感知:  inner_stimuli = apply_defenses(stimuli, profiles)   # EMA "因果解释"

  ② 内部态(每轮, EMA 快路径):  
     internal[t] = internal[t-1] + Δ(inner, coupling) + N(0, σ²)
                  ↑ A-process                     ↑ O-U 扩散

  ③ 事件权重计算:
     w_t = ‖inner‖₁ / 7 + γ·stimuli[EW]                       # Ingendahl 加权
     push(buf_w, w_t)
     push(buf_s, inner)

  ④ 关系缓冲区(递推加权平均, HED AL):
     s_rel[t] = τ·s_rel[t-1] + (1-τ)·inner                    # L(t) 低秩基底

  ⑤ 关系态(N轮一次, EMA 慢路径):
     if len(buf_w) == N:
       s_avg = Σ(buf_w·buf_s) / Σ(buf_w)                       # 加权平均
       rel += Δ(s_avg, dt=N) + N(0, σ_rel²)                    # B-process
       clear(buf)

  ⑥ 表面(每轮, LSEF 重建):
     surface[t] = project(internal[t], rel[t], outer)          # E(t) = L+S
```

### 7.3 关键参数

| 参数 | 范围 | 功能 | 默认值 | 来源 |
|:-----|:----:|:-----|:-----:|:-----|
| `τ` | [0.3, 0.9] | 关系缓冲区衰减率 | 0.6 | HED AL |
| `γ` | [0.2, 0.8] | EW 显著性 bonus | 0.3 | Ingendahl |
| `N` | [3, 8] | 关系更新间隔 | 5 | Revol 事件触发 |
| `σ` | [0.005, 0.03] | Langevin 扩散 | 0.015 | MATE O-U |
| `θ_rel` | [0.05, 0.20] | 关系回归速率 | 0.10 | O-U 过程 |

---

## 8. 参考文献

### 神经科学

1. **Kakusa, M., et al.** (2025). Distinct neural temporal architectures encode rapid social expressions and sustained internal mood states. *bioRxiv*. doi:10.64898/2025.12.20.692681
   - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12724620/) | 连续 iEEG 快慢通路分离

### 计算情感建模

2. **Marsella, S.C. & Gratch, J.** (2009). EMA: A process model of appraisal dynamics. *Cognitive Systems Research*, 10(1), 70-90.
   - [PDF](https://people.ict.usc.edu/~gratch/papers/COGSYS-RS-EMOTION-2008-6.pdf) | 单一评价层架构

3. **Steephen, J.E.** (2013). HED: A Computational Model of Affective Adaptation and Emotion Dynamics. *IEEE Transactions on Affective Computing*, 4(2), 197-210.
   - [ACM](https://dl.acm.org/doi/10.1109/T-AFFC.2013.2) | ALₜ = αXₜ₋₁ + (1-α)ALₜ₋₁

4. **Steephen, J.E., et al.** (2020). HED-ID: An Affective Adaptation Model Explaining the Intensity-Duration Relationship of Emotion. *IEEE TAC*, 11, 736-750.
   - 负加速指数衰减

5. **Cui, F.-Q., et al.** (2025). Robust Low-Rank Sparse Framework for Video-Based Affective Computing. *arXiv:2511.11406*.
   - [arXiv](https://arxiv.org/abs/2511.11406) | E(t) = L(t) + S(t) 低秩稀疏分解

6. **Lobozov, S.** (2026). MATE: A Deterministic Affective Middleware for LLM-Based Companions. v8.0. *Zenodo*.
   - [Zenodo](https://zenodo.org/records/20400530) | 双过程习惯化 + O-U 调节 + 67KLOC 实现

### 心理学理论

7. **Helson, H.** (1964). *Adaptation-Level Theory*. Harper & Row.
   - AL = 加权几何均数，情绪 = 刺激 - AL

8. **Solomon, R.L. & Corbit, J.D.** (1974). Opponent-Process Theory of Acquired Motivation.
   - A-process(快) + B-process(慢，衰减 4×)

9. **Ingendahl, M., et al.** (2024). The Interplay of Multiple Unconditioned Stimuli in Evaluative Conditioning: A Weighted Averaging Framework. *Journal of Personality and Social Psychology*, 127(5), 964-985.
   - [Ovid](https://www.ovid.com/journals/jpspy/abstract/10.1037/pspa0000401) | 态度 = 加权平均 + 负性偏见

### 方法论

10. **Revol, J., et al.** (2025). Episode-contingent experience-sampling designs for accurate estimates of autoregressive dynamics. *Psychological Methods*.
    - [PubMed](https://pubmed.ncbi.nlm.nih.gov/40354252/) | 事件触发 > 固定间隔

### 情感动力学

11. **Kuppens, P. & Verduyn, P.** (2015). Emotion Dynamics. *Current Opinion in Psychology*, 3, 22-26.
    - VAR(1) 模型: s(t) = Φ·s(t-1) + ε(t)，惯性 = Φ 对角线

12. **Loossens, T., et al.** (2020). The Affective Ising Model: A computational account of human affect dynamics. *Emotion*.
    - 非线性多吸引子情感景观

13. **Oravecz, Z., et al.** (2009-2011). Hierarchical OU process for modeling core affect.
    - [PubMed](https://pubmed.ncbi.nlm.nih.gov/36107656/) | 情感动力学的 O-U 标准建模

### 相关架构参考

14. **Fu, C., et al.** (2026). Sentipolis: Emotion-Aware Agents for Social Simulations. *arXiv:2601.18027*.
    - [arXiv](https://export.arxiv.org/abs/2601.18027) | 双速情感动力学 + PAD + poignancy score

15. **Koley, G.** (2025). SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation. *arXiv:2505.09081*.
    - [arXiv](https://arxiv.org/abs/2505.09081) | ‖pₜ₊ₖ - pₜ‖ ≤ 0.08·log(k) + 0.12

---

> 本文档会根据后续搜索和设计迭代持续更新。
