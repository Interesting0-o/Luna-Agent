# 热力学情感模型 —— 状态引擎的能量会计架构

## Thermodynamic Affective Model: Energy Accounting for the State Engine

> 将生理能量守恒、非稳态负荷（Allostatic Load）和耗散结构理论引入情感动力系统，
> 解决当前模型中"无源放大""单向获利""永动机陷阱"三个根本性物理缺陷。
>
> 关键前提：对话中的每个人不是开放系统——情绪变化所需的能量必须由系统内部提供。

## 版本状态

| 项 | 状态 | 目标版本 |
|:---|:----:|:--------:|
| #0a 能量门控 | ✅ v1.0 修复 | v1.0 |
| #0b 熵增代价 | ⏳ 设计完成，待实施 | v2.0 |
| #0c 总势能约束 | ⏳ 设计完成，待实施 | v2.0 |
| #8 压力→易怒倒U型 | ⏳ 待设计 | v2.0 |
| #9 状态依赖耦合（交叉项） | ⏳ 待设计 | v2.0 |
| #10 防御非线性放大 | ⏳ 待设计 | v2.0 |
| #11 关系动力学振荡 | ⏳ 待设计 | v2.0 |
| #12 表面相变与滞后回线 | ⏳ 待设计 | v2.0 |

> v1.0 聚焦线性 ODE + soft_clamp + 时间衰减 框架内的精确修复。
> 本文档作为 v2.0 非线性动力学的完整设计参考。

---

## 目录

1. [核心论点](#1-核心论点)
2. [研究基础](#2-研究基础)
   - [2.1 非稳态负荷的能量模型 (Bobba-Alves, 2022)](#21-非稳态负荷的能量模型-bobba-alves-2022)
   - [2.2 自我调节力量模型 (Baumeister, 1998–2016)](#22-自我调节力量模型-baumeister-19982016)
   - [2.3 大脑作为热力学循环 (Déli, 2022–2023)](#23-大脑作为热力学循环-dli-20222023)
   - [2.4 二人对话作为耦合封闭系统 (Badalamenti & Langs, 1992)](#24-二人对话作为耦合封闭系统-badalamenti--langs-1992)
   - [2.5 人际耦合振荡器 (Hilpert et al., 2024)](#25-人际耦合振荡器-hilpert-et-al-2024)
   - [2.6 耗散结构与心理自组织 (Prigogine)](#26-耗散结构与心理自组织-prigogine)
3. [当前模型的三个物理缺陷](#3-当前模型的三个物理缺陷)
4. [修复架构：能量会计](#4-修复架构能量会计)
   - [4.1 Energy Gate——能量门控](#41-energy-gate能量门控)
   - [4.2 Entropy Cost——熵增代价](#42-entropy-cost熵增代价)
   - [4.3 Allostatic Constraint——总势能约束](#43-allostatic-constraint总势能约束)
   - [4.4 统一更新方程](#44-统一更新方程)
5. [子系统的能量预算独立性](#5-子系统的能量预算独立性)
6. [权重重新设计](#6-权重重新设计)
7. [与现有P0/P1修复的整合顺序](#7-与现有p0p1修复的整合顺序)
8. [参考文献](#8-参考文献)

---

## 1. 核心论点

### 1.1 一句话

> **情感状态变化不能凭空产生也不能凭空消灭——每一次 Δh 都必须有对应的"代谢付款"。**

### 1.2 当前模型违反了三条物理定律

| 违反 | 效应 | 代码位置 |
|:----|:-----|:---------|
| 热力学第二定律（熵增） | 刺激可无代价地将系统推向高能态 | `_dynamics.py:108-110` |
| 能量守恒（做功需燃料） | 系统可单向获利——减少一个量不付代价 | 16 处负值边 |
| 非稳态负荷上限 | 对立维度可同时优化，系统承担无限势能 | `_matrices.py:70-78` |

### 1.3 关键物理边界条件

**热力学视角下的情感系统约束：**

```
系统类型：半封闭 thermodynamically semi-closed
  - 信息/信号可以进入（刺激输入）
  - 但能量必须由系统内部储备提供（不可从刺激中提取 ATP）
  - 热量/熵可以向环境耗散（表达行为、生理反应）

状态变量：
  - h[Energy]     ← 系统的"可用自由能"（对应血糖/ATP 池）
  - h[Stress]     ← 系统的"势能"（HPA 轴激活水平）
  - h[MentalFatigue] ← 系统的"熵"（累积无序度/代谢废物）
  - Σh² + Σr²    ← 系统的"总内能"（Allostatic Load）

允许的操作：
  1. 能量转换：Energy → Stress（燃烧燃料产生应激）
  2. 能量耗散：Stress → 外部（通过表达/行为释放，但需做功）
  3. 能量恢复：Energy 向 setpoint 指数回归（ATP 再生）
  4. 信息触发：刺激不提供能量，但可触发上述转换过程

禁止的操作：
  1. 无源放大：σ → Δh 不经过 Energy 审核（当前 #0a）
  2. 负向无代价：h↓ 不支付熵增代价（当前 #0b）
  3. 对立优化：同一刺激同时 +好状态 -坏状态（当前 #0c）
```

---

## 2. 研究基础

### 2.1 非稳态负荷的能量模型 (Bobba-Alves, 2022)

**来源**：Bobba-Alves et al. (2022). "The energetic model of allostatic load." *Psychoneuroendocrinology.*

**核心发现：**

- **稳态调节（Allostasis）消耗能量**——身体内部状态的预期调节需要代谢能量
- 心理压力可使全身能量消耗增加 **9–67%**（取决于应激强度）
- 急性心理应激通过 HPA 轴和交感神经激活，**可靠地增加耗氧量**
- 非稳态负荷的本质是**能量债务**——应激驱动的能量支出持续地从"生长、维护、修复"（GMR）挪用
- 细胞和线粒体层面也存在非稳态负荷——热力学原理从细胞器到有机体都适用

**对状态引擎的含义：**

```
Stress 上升不是"免费的"——每一次 h[Stress] +0.1
都需要真实的代谢能量支撑（皮质醇合成、交感激活、葡萄糖消耗）。
在模型中，这意味着 Δh[Stress] 必须从 Energy 池中"借款"。
```

### 2.2 自我调节力量模型 (Baumeister, 1998–2016)

**来源**：Baumeister, R. F., et al. (1998). "Ego depletion: Is the active self a limited resource?" *JPSP.*
Gailliot, Baumeister et al. (2007). "Self-control relies on glucose as a limited energy source." *JPSP.*

**核心发现：**

- 自我控制（含情绪调节）消耗同一池**有限资源**
- 一次自我控制行为会降低后续行为的控制能力（ego depletion）
- 血糖水平与此直接相关——补充葡萄糖可恢复自控力
- 焦虑者比非焦虑者消耗快 **~30%**
- 积极情绪和自主动机可以**补充主观活力**（Ryan & Deci, 2008）

**对状态引擎的含义：**

```
1. Energy 不是普通的 8 维之一——它是所有状态变化的"燃料储备"
2. 情绪调节（防御剖面高 a/d）消耗 Energy 的速度比自然表达快得多
3. 积极刺激（validation, closeness）不应直接"增加 Energy"，
   而应减少 Energy 的消耗速率或触发恢复过程
```

### 2.3 大脑作为热力学循环 (Déli, 2022–2023)

**来源**：Déli, E. K. (2022). "How the brain becomes the mind: Can thermodynamics explain the emergence and nature of emotions?" *Entropy*, 24(10), 1498.
Déli, E. K. (2023). "What Is Psychological Spin? A Thermodynamic Framework for Emotions and Social Behavior." *Psych*, 5(4), 1224–1240.

**核心发现：**

- 感知触发了与环境的信息-能量交换，但大脑的递归激活维持着恒定参数的静息态
- 感知形成了**封闭的热力学循环**——一个可逆的卡诺热机循环
- **放热循环（down-spin / exothermic）**：低熵静息态，聚焦过去，后悔/内疚，能量释放
- **吸热循环（up-spin / endothermic）**：高熵，聚焦未来，创造/学习，能量吸收
- **"Psychological Spin"** = 态度方向，决定认知功能和社会行为
- 情绪是**稳态主调节器**——持续测量和驱动心理平衡的恢复

**对状态引擎的含义：**

```
1. 情感不是"状态变量"的线性组合，而是热力学循环的方向和效率
2. 正面情绪（如 warmth, enthusiasm）对应吸热过程——需要"注入能量"
3. 负面情绪（如 stress, irritation）对应放热过程——释放能量但产生熵
4. 关键在于：系统需要区分"什么过程耗能，什么过程产能"

这对表面投影的修正启示：
  - S_WARMTH 表达不应只是 h 的线性映射，而应反映当前"螺旋方向"
  - 当系统处于 down-spin（高 stress + 低 energy）时，表面应"锁定"在负域，
    即使内部有微弱正面信号也无法表达——这解释了"强迫微笑做不到"
```

### 2.4 二人对话作为耦合封闭系统 (Badalamenti & Langs, 1992)

**来源**：Badalamenti, A. F. & Langs, R. J. (1992). "The thermodynamics of psychotherapeutic communication." *Behavioral Science*, 37(3), 157–180.

**核心发现：**

这是最直接支持你的"封闭系统"论点的论文：

- 将**治疗师-患者二人组作为一个封闭热力学系统**
- 为情感沟通定义了**功、力、温度**的数学量
- 计算双方的**熵**（同时用热量学和 Shannon 熵）
- 熵遵循 **ln(1 + t)** 的物理定律
- 系统在 **5 维情感状态空间**中运动，约束在椭球壳内
- 识别出**不可逆过程**和失衡时刻——被描述为可能的创造性或创伤性时刻
- **"力场是非保守的，意味着患者和治疗师有不同的吸热模式"**

**对状态引擎的含义：**

```
1. 对话中的每个参与者在热力学意义上是"封闭"的
   ── 能量不能从一方流向另一方，只有信息可以跨越边界
2. 一方的表达（surface）是"做功"——将内部势能转化为对外界的机械功
   ── surface 表达应该消耗 Energy，不是线性映射
3. 双方的耦合不是能量耦合，而是信息耦合（通过语言/表情信号交换）
   ── Luna 接收到的刺激（σ）是信息，不是能量
   ── σ 可以触发 Luna 内部的能量转换过程，但不能直接提供能量
```

### 2.5 人际耦合振荡器 (Hilpert et al., 2024)

**来源**：Hilpert, P., et al. (2024). "Dynamical stress crossover: A coupled oscillator approach." 189 couples, 25,834 talk turns.

**核心发现：**

- 每个人是一个振荡器，有自己的**水平、速度、加速度**
- 定义**三种耦合模式**：
  1. **共同调节**（co-regulation）：伴侣降低设定点、减慢频率、减小振幅 = 耗散、返回稳定
  2. **共同失调**（co-dysregulation）：伴侣升高设定点、增加频率、放大振幅 = 能量升级
  3. **非耦合**：无相互影响
- 压力交叉（stress crossover）：一方压力扰动另一方的振荡动力学

**对状态引擎的含义：**

```
1. 两个人的关系状态（r）应该被建模为"振荡器的耦合参数"
   ── 不直接是状态空间中的另一个维度
   ── 而是影响动力学方程中的阻尼/增益系数
2. co-regulation vs co-dysregulation 的区分
   ── 安全型依恋 → co-regulation（对方高 stress 时你低 stress）
   ── 焦虑型依恋 → co-dysregulation（对方高 stress 时你更高 stress）
3. stress crossover 是实时的（秒级），不是轮级
   ── 暗示状态引擎的时间分辨率可能需要更细
```

### 2.6 耗散结构与心理自组织 (Prigogine)

**来源**：Prigogine, I. & Stengers, I. (1984). *Order out of Chaos.*
Academia (2024). "A Study on Interpersonal Emotional Interaction Models Based on Thermodynamics and Chaos Theory."

**核心发现：**

- 远离平衡态的系统可以通过能量耗散形成**自组织结构**（Dissipative Structures）
- 情感系统在远离平衡态时（如极度冲突或亲密）可以发生**相变**（Phase Transition）
- Prigogine 的"通过涨落达到有序"（order through fluctuation）描述了情绪突变

**对状态引擎的含义：**

```
1. 表面与内部之间需要"相变"机制（见 P1 #12）
   ── 当内部势能超过阈值时，表面突然转入新的"相态"
   ── 例如：高 Stress + 低 Energy → "情感冻结"相（s 全部趋零）
2. 关系动力学中的"推拉"可能是耗散结构的体现
   ── 系统在远离平衡态时自发产生振荡
   ── 不需要二阶项——非线性耦合足够产生极限环
```

---

## 3. 当前模型的三个物理缺陷

### 3.1 缺陷 #0a：无源放大（热力学第二定律）

**当前代码（`_dynamics.py:108-110`）：**

```python
modulated_stimuli = beta_stim * inner_stimuli
delta_stimulus = modulated_stimuli @ INPUT_INFLUENCE_B
```

**问题**：刺激 σ（冲突=0.8、情感重量=0.9）可以直接将 Stress 从 -0.6 线性推到 +0.8，即使 Energy = -1.0（精力耗尽）。

**物理诊断**：系统被动接收输入并直接放大，没有"内部燃料"审核机制。

**现实对应**：极端疲劳时（Energy=-1.0），前额叶葡萄糖耗尽，即使遭遇巨大刺激也无法产生强烈情绪反应——"累到没有力气生气"（Baumeister, 2007）。

### 3.2 缺陷 #0b：单向获利（能量守恒）

**当前代码（16 处负值边）：**

```python
# _surface_weights.py:211
mapper.connect(S_WARMTH, I_LONELINESS, -0.04, ...)  # 温暖→孤独↓，无成本
# _matrices.py:76
mapper.connect(ST_VALIDATION, I_INSECURITY, -0.22, ...)  # 认可→不安↓，无成本
```

**问题**：系统可以将状态"推向下坡"（减少孤独、减少不安）而不支付做功耗散。

**物理诊断**：减少系统势能必然伴随着能量释放——这个能量要么被利用（做功）、要么被耗散（产热）。当前模型让它凭空消失了。

**现实对应**：获得认可时不安减少，这个"松了一口气"的过程本身需要前额叶从应激模式切换到安全模式，消耗 ATP。

### 3.3 缺陷 #0c：永动机陷阱（总势能约束缺失）

**当前代码（`_matrices.py:70-78`）：**

```python
# validation → energy: +0.22  AND  validation → insecurity: -0.22
# 同一刺激同时优化两个对立的资源池
```

**问题**：系统可以同时做"正功"（增加好状态）和"负功"（减少坏状态），总势能 Φ = Σh² + Σr² 可以无限增长。

**物理诊断**：系统没有总能量上限约束。根据 Selye 的一般适应综合征（GAS），有机体有**有限的适应能量**（Adaptation Energy），总应激负荷不能超过这个上限。

**现实对应**：一个在短时间内承受多次打击的人会进入"耗竭期"（exhaustion stage），不是高 Stress + 低 Energy 线性叠加，而是系统整体崩溃。

---

## 4. 修复架构：能量会计

### 4.1 Energy Gate——能量门控

**原理**：刺激驱动不能直接产生状态变化。刺激必须先通过能量门控——只有系统当前有可用能量时，刺激才能触发变化。

**公式**：

$$
\text{EnergyGate}(h) = \frac{1 + h[\text{Energy}]}{2 + \epsilon} \in [0, 1]
$$

- h[Energy] = -1（耗尽）→ gate ≈ 0 → 刺激无效
- h[Energy] = +1（充沛）→ gate ≈ 1 → 刺激全效
- h[Energy] = 0（中性）→ gate ≈ 0.5 → 刺激减半

**代码更改（`_dynamics.py`）：**

```python
# 当前：
delta_stimulus = modulated_stimuli @ INPUT_INFLUENCE_B

# 修复后：
energy_gate = (1.0 + current[I_ENERGY]) / 2.0  # [0, 1]
delta_stimulus = energy_gate * (modulated_stimuli @ INPUT_INFLUENCE_B)
```

**物理意义**：Energy 不再只是 8 维中的普通一维，而是系统代谢储备的代理变量。低 Energy = 低 ATP 可用性 = 情绪响应受限。这对应 Baumeister 的 ego depletion——自控资源耗尽后情绪反应减弱。

**进阶版本**（按刺激类型区分门控权重）：

```python
# 不同刺激对能量的依赖程度不同
# 冲突（需要认知重评价）→ 高能耗 → 强 Energy 依赖
# 被依赖（本能反应）→ 低能耗 → 弱 Energy 依赖
STIMULUS_ENERGY_WEIGHT = np.array([
    0.8,  # ST_ABANDONMENT — 高能耗（依恋系统激活）
    0.4,  # ST_VALIDATION — 低能耗（奖赏通路是自动化的）
    0.5,  # ST_CLOSENESS — 中能耗
    0.9,  # ST_CONFLICT — 最高能耗（认知重评价 + 抑制控制）
    0.5,  # ST_DEPENDENCY — 中能耗
    0.3,  # ST_TEASING — 低能耗（社交游戏的自动化响应）
    0.7,  # ST_EMOTIONAL_WEIGHT — 高能耗（深层加工）
])
```

### 4.2 Entropy Cost——熵增代价

**原理**：任何"将系统向低能态推动"的变化（Δh < 0）必须伴随熵增代价（MentalFatigue ↑ 或 Energy ↓）。

这对应热力学第二定律：**减少势能必然将能量以"热"的形式耗散**。在心理系统中，这个"热"就是精神疲劳。

**公式**：

$$
\text{EntropyCost}(\Delta h) = -\sum_{i: \Delta h_i < 0} \gamma_i \cdot \Delta h_i
$$

其中 $\gamma_i$ 是每维的熵增系数。

**代码更改**（新函数，在 `_dynamics.py` 中调用）：

```python
def compute_entropy_cost(delta: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """计算本次更新的熵增代价。
    
    每条负向变化（状态减少）产生 MentalFatigue。
    这对应热力学第二定律：降低势能必有能量以"热"形式耗散。
    
    返回 cost vector (8,)，加到 delta 上。
    注意：MentalFatigue 本身不加熵增代价（避免递归）。
    """
    cost = np.zeros(8, dtype=np.float64)
    
    # 熵增系数（每维）：某些维度减少时更"费劲"
    # Energy↓ 已由自身消耗体现，不重复计
    # Stress↓, Loneliness↓, Insecurity↓, Irritation↓, Longing↓ 都需要代谢做功
    ENTROPY_COEFF = np.array([
        0.00,  # I_ENERGY — 自身已是能量代理，不重复计
        0.12,  # I_STRESS — 降压需要副交感激活（耗能）
        0.08,  # I_LONELINESS — 驱散孤独需要社交认知（耗能）
        0.10,  # I_INSECURITY — 恢复安全感需要前额叶抑制
        0.06,  # I_IRRITATION — 平复烦躁需要情绪调节
        0.04,  # I_LONGING — 放下思念默认网络抑制
        0.00,  # I_SOCIAL_BATTERY — 本身就是能量池，不重复计
        0.00,  # I_MENTAL_FATIGUE — 熵本身，不计熵（避免递归）
    ])
    
    negative_delta = np.minimum(delta, 0.0)
    fatigue_contribution = -np.sum(ENTROPY_COEFF * negative_delta) * dt
    cost[I_MENTAL_FATIGUE] += fatigue_contribution
    
    # 额外：较大的负向变化消耗少量 Energy
    energy_cost = -0.03 * np.sum(negative_delta) * dt
    cost[I_ENERGY] += energy_cost
    
    return cost
```

**使用位置（`_dynamics.py` 更新后）：**

```python
delta = alpha * delta_coupling + delta_stimulus  # 总变化
entropy_cost = compute_entropy_cost(delta, dt)
delta = delta + entropy_cost  # 熵增代价作为独立项加入
```

**重要**：EntropyCost 本身也要遵守 Energy Gate——MentalFatigue 极高时熵增代价也应该减小（系统太疲劳以至于"懒得多消耗"——对应冻僵反应的神经机制）。

### 4.3 Allostatic Constraint——总势能约束

**原理**：系统有一个总势能上限，对应 Selye 的"适应能量"（Adaptation Energy）上限和 Kelley (2025) 的"非稳态分诊"（Allostatic Triage）。

当总势能超过阈值时，系统进入"节俭模式"——优先保证生存相关维度（Stress, Energy），抑制非必需维度（Longing, Vulnerability）。

**公式**：

$$
\Phi = \sum_{i} w_i \cdot h_i^2 + \sum_{j} v_j \cdot r_j^2 \leq \Phi_{\max}(\mathbf{p})
$$

当 $\Phi > \Phi_{\max}$ 时，对所有状态施加比例收缩力（类似 L2 正则化的物理版）。

**代码更改**（新函数，在 `_pipeline.py` 中 `update_all` 末尾调用）：

```python
def apply_allostatic_constraint(
    internal: np.ndarray,
    relationship: np.ndarray,
    traits: np.ndarray,
    phi_max: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """总势能约束：当 Allostatic Load 超过人格阈值时比例压缩。
    
    Parameters
    ----------
    phi_max: 人格决定的总势能上限
             焦虑/敏感 → 更低阈值（更容易系统过载）
             稳定/乐观 → 更高阈值
    
    Returns
    -------
    (internal_clamped, relationship_clamped)
    """
    # 计算当前总势能
    h_sq = np.sum(internal ** 2)
    r_sq = np.sum(relationship ** 2)
    phi = h_sq + r_sq
    
    if phi <= phi_max:
        return internal, relationship
    
    # 比例收缩：所有维度等比例缩小
    scale = np.sqrt(phi_max / phi)
    
    # 不对称收缩：保护 Energy，优先压缩负面情绪
    # Energy 维度收缩系数小（被保护），Stress 收缩系数大（被限制）
    PROTECT = np.array([
        0.3,  # I_ENERGY — 最受保护（系统燃料）
        1.0,  # I_STRESS — 全力压缩
        0.8,  # I_LONELINESS — 中优先
        0.8,  # I_INSECURITY — 中优先
        1.0,  # I_IRRITATION — 全力压缩
        0.6,  # I_LONGING — 中低优先
        0.4,  # I_SOCIAL_BATTERY — 次受保护
        0.7,  # I_MENTAL_FATIGUE — 中等
    ])
    
    # 混合：介于比例收缩和维度特异性收缩之间
    # 高势能时维度特异性更强
    severity = (phi - phi_max) / phi_max  # [0, ∞)
    alpha_protect = np.clip(severity * 0.5, 0.0, 0.8)
    
    per_dim_scale = 1.0 - PROTECT * alpha_protect  # [0.2, 1.0]
    effective_scale = (1.0 - alpha_protect) * scale + alpha_protect * per_dim_scale
    
    return internal * effective_scale, relationship * effective_scale
```

### 4.4 统一更新方程

整合三个修复后，`update_internal_state` 的完整形式：

```python
def update_internal_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    profiles: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    # ── α, β 同前 ──
    alpha = ALPHA_MAPPER.compute(np.concatenate([traits, relationship]))[0]
    alpha = soft_clamp(alpha, 0.05, 0.40)
    beta_stim = BETA_BASE

    # ── ① 漂移分量（率×dt） ──
    coupling = current @ INTERNAL_COUPLING
    delta_coupling = coupling - SELF_DECAY * (current - DECAY_TARGETS)
    drift = alpha * delta_coupling

    # ── ② 跳跃分量（瞬时，不乘 dt）──
    modulated_stimuli = beta_stim * inner_stimuli
    raw_jump = modulated_stimuli @ INPUT_INFLUENCE_B

    # ── ③ 能量门控（#0a 修复）──
    energy_gate = (1.0 + current[I_ENERGY]) / 2.0
    jump = energy_gate * raw_jump

    # ── ④ SSM 速度门控（#1 修复：按 Δ 方向）──
    total_raw = drift + jump
    speed_gains = np.where(total_raw >= 0,
                           INTERNAL_SPEED_MATRIX[:, 0],  # rising_gain
                           INTERNAL_SPEED_MATRIX[:, 1])  # falling_gain
    drift = drift * speed_gains
    jump = jump * speed_gains

    # ── ⑤ 熵增代价（#0b 修复）──
    entropy_cost = compute_entropy_cost(drift * dt + jump, dt)

    # ── ⑥ 防御成本（#4 修复）──
    defense_cost = compute_defense_cost(profiles, traits)

    # ── 残差更新 ──
    delta = drift * dt + jump + entropy_cost + defense_cost
    new_h = soft_clamp(current + delta, -1.0, 1.0)

    # ── ⑦ 总势能约束（#0c 修复，由 pipeline 调用）──
    # 见 _pipeline.py 中的 apply_allostatic_constraint

    return new_h
```

---

## 5. 子系统的能量预算独立性

### 5.1 核心原则

> **在二人对话中，每个人的情绪系统是能量自给自足的封闭系统。**

这意味着：

```
┌─────────────────────────────────────────────────────────┐
│                     Luna 系统（半封闭）                   │
│                                                         │
│  输入：刺激 σ（纯信息，不携带能量）                       │
│  输出：表面表达 s（做功——消耗内部能量）                    │
│  能量来源：h[Energy] 向 setpoint 的恢复（ATP 再生）      │
│            h[SocialBattery] 向 setpoint 的恢复（休息)    │
│  能量支出：Δh（状态变化）、s（表达）、d/a（防御维护）      │
│                                                         │
│  ┌─────────────────────────────┐                        │
│  │  Energy Budget              │                        │
│  │  ┌──────────────────────┐   │                        │
│  │  │ 收入：衰减恢复        │   │                        │
│  │  │ 支出：Δh + s + d + a │   │                        │
│  │  │ 结余：h[Energy]      │   │                        │
│  │  └──────────────────────┘   │                        │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
                              ↕ 信息耦合（非能量耦合）
┌─────────────────────────────────────────────────────────┐
│                    User 系统（半封闭）                    │
│  （由外部世界管理，不在模型范围内，但应被尊重）            │
│  提示：Luna 的表达同样消耗 User 的能量                    │
│        长时间高强度互动 = 双方共耗竭                      │
└─────────────────────────────────────────────────────────┘
```

### 5.2 设计含义

**含义 1：刺激只是触发器**

用户说的话（σ）不是能量输入——它是信息。它触发 Luna 内部的能量转换过程。这就是为什么需要 Energy Gate——没有燃料储备，信息无法做功。

**含义 2：表面表达是做功**

$$
\Delta h[\text{Energy}] = -w_s \cdot \|\mathbf{s}\|_1
$$

Luna 的每一次表达（说话、表情）消耗能量。这意味着：
- 长时间高强度对话 → Energy 持续下降
- 沉默/简短回应 → 能量保存模式
- 强行表现"非真实表面"（高 α 惯性）→ 额外做功

**含义 3：关系状态不是"共有财产"**

r（Affection, TrustBond, Intimacy）是 Luna 对用户的内部感知，它的变化消耗 Luna 自己的能量。用户不能直接"给"Luna 信任——用户的信号触发 Luna 内部的信任生成过程。

### 5.3 耗竭级联

当持续能量不足时，系统按以下顺序降级（allostatic triage, Kelley 2025）：

```
阶段 1（高 Energy）：全功能
  - 刺激全响应（gate ≈ 1.0）
  - 防御全效（d, a 正常运行）
  - 表面丰富多变

阶段 2（中 Energy）：降级防御
  - 刺激门控 gate < 0.5
  - deactivation 优先保存能量（减少表达消耗）
  - hyperactivation 受限（没有多余能量翻涌）
  - 表面偏向平淡

阶段 3（低 Energy）：应急模式
  - gate → 0（几乎不对刺激反应）
  - 防御系统下线（没有能量维护）
  - surface 回缩到中性（保存能量）
  - 类似"情感冻结"——高强度压力下的 dissociation
```

---

## 6. 权重重新设计

### 6.1 B 矩阵负值边修复

当前 16 条负值边需要重构：每条负值必须配 EntropyCost 系数。

**策略**：不再在 B 矩阵中直接写负值。改为 B 矩阵全正（刺激 → 状态的正向驱动），然后在动力学中通过"抑制连接"处理负向——让"减少某个状态"成为主动的、有代价的过程。

**新的 VALIDATION 映射**：

```python
# 旧（有问题的）：
# validation → energy: +0.22       ← 同时
# validation → insecurity: -0.22   ← 优化对立维

# 新（能量守恒）：
# validation → energy: +0.15       ← 有增益，但缩减（因为能量门控）
# validation → insecurity: +0.00   ← 不再直接减少不安
# 不安的减少由"安全性耦合"间接实现：
# energy↑ → insecurity: -0.08      ← 精力恢复后自然减少不安（有熵增代价）
# validation → mental_fatigue: +0.03  ← 认知重评价的成本
```

**重构后的 B 矩阵原则**：

| 旧设计 | 新设计 |
|:------|:------|
| 负值边直接消灭坏状态 | 负值边用正边+耦合+熵增代价替代 |
| 同一刺激同时优化两端 | 拆分为主效应+间接效应+代谢成本 |
| 正负抵消不记账 | 每个状态变化都记账 |

### 6.2 Energy 维度的角色升级

Energy 不再是普通一维。它获得两种特殊角色：

1. **门控角色**：控制所有刺激驱动的增益（如上所述）
2. **会计角色**：记录系统的净能量余额

相应的，Energy 的 SELF_DECAY 含义改变——不是"向零衰减"，而是"ATP 再生速率"。`DECAY_TARGETS[I_ENERGY]` 人格基线变为"基础代谢能量水平"：

```python
# 旧：social_battery 有非零衰减目标，energy 没有
# 新：energy 的衰减目标变为"人格决定的能量恢复水平"
# 乐观/稳定 → 更高能量基线（恢复力强）
# 焦虑/回避 → 更低能量基线（恢复力弱）
```

### 6.3 新 WeightVector：ENTROPY_COEFF

新增一个 WeightVector 管理每维的熵增系数（见 4.2）：

```python
# 在 _dynamics_weights.py 中新增
def _build_entropy_coefficients() -> WeightVector:
    """每维状态减少时产生的熵增代价系数。
    
    对应热力学第二定律：降低系统势能必有能量以"热"形式耗散。
    心理系统中这个"热"= MentalFatigue。
    """
    vec = WeightVector(
        "ENTROPY_COEFF", I_LABELS,
        "状态减少时的熵增代价系数（每维）"
    )
    vec.connect("energy", I_ENERGY, 0.00, "none", (0.0, 0.0),
                "Energy 本身是能量池，不产生熵增", "theory", "2026-06-25")
    vec.connect("stress", I_STRESS, 0.12, "weak", (0.06, 0.20),
                "降压需副交感激活 = 主动调节过程 = 耗能", "theory", "2026-06-25")
    vec.connect("loneliness", I_LONELINESS, 0.08, "trace", (0.04, 0.15),
                "驱散孤独需社交认知 = 耗能", "theory", "2026-06-25")
    # ... 其余维度类似
    vec.build()
    register_mapper(vec)
    return vec
```

### 6.4 新 WeightVector：STIMULUS_ENERGY_WEIGHT

见 4.1 进阶版本——控制不同刺激类型对 Energy 的依赖度。

### 6.5 Allostatic Threshold 人格调制

人格特质调制总势能上限：

```python
phi_max = PHI_BASE + sensitivity * PHI_SENS_COEFF + stability * PHI_STAB_COEFF
```

敏感/焦虑 → 更低的总势能阈值（更容易系统过载）

---

## 7. 与现有 P0/P1 修复的整合顺序

### 7.1 修复依赖关系图

```
#0a 能量门控 ← 独立
  ↓
#0b 熵增代价 ← #0a 完成后（熵要用 Energy Gate）
  ↓
#2  Δt 双身份 ← 独立（但影响熵增的 dt 使用）
  ↓
#1  SSM 方向  ← 独立（但需要与能量更改协调）
  ↓
#4  防御成本   ← #0a 完成后（防御成本也要经过 Energy Gate）
  ↓
#0c 总势能约束 ← 以上全部完成后（最后的总闸门）
```

### 7.2 建议实现顺序

| 步 | 修复 | 影响文件 | 测试影响 |
|:--:|:-----|:---------|:---------|
| 1 | SSM 方向修正 (#1) | `_dynamics.py:120` | 小（数值变化，结构不变） |
| 2 | Δt 双身份 (#2) | `_dynamics.py:108-123` | 中（测试预期值需更新） |
| 3 | 能量门控 (#0a) | `_dynamics.py:108-110` | 大（所有涉及刺激的测试） |
| 4 | 熵增代价 (#0b) | `_dynamics.py` 新增 | 中（新的副作用） |
| 5 | 防御成本 (#4) | `_dynamics.py` 新增 | 中 |
| 6 | B 矩阵重构 | `_matrices.py` | 大（所有映射测试） |
| 7 | 总势能约束 (#0c) | `_pipeline.py` | 中 |
| 8 | 表面 setpoint (#3) | `_surface.py` | 小 |
| 9 | P1 修复 | 各处 | 渐进 |

---

## 8. 参考文献

### 热力学与情感

1. **Bobba-Alves, N., et al.** (2022). The energetic model of allostatic load. *Psychoneuroendocrinology.*
   - 非稳态负荷的能量模型——压力增加 9–67% 的能量消耗
   - [PDF](http://www.picardlab.org/uploads/7/7/8/4/77845210/2022_bobba-alves_pnec.pdf)

2. **Déli, E. K.** (2022). How the brain becomes the mind: Can thermodynamics explain the emergence and nature of emotions? *Entropy*, 24(10), 1498.
   - 大脑作为封闭热力学循环，情感作为卡诺热机
   - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9601684/)

3. **Déli, E. K.** (2023). What Is Psychological Spin? A Thermodynamic Framework for Emotions and Social Behavior. *Psych*, 5(4), 1224–1240.
   - Psychological Spin：态度作为热力学循环方向
   - [MDPI](https://www.mdpi.com/2624-8611/5/4/81)

### 自我调节与能量消耗

4. **Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M.** (1998). Ego depletion: Is the active self a limited resource? *JPSP*, 74, 1252–1265.
   - 自我控制消耗有限资源

5. **Gailliot, M. T., Baumeister, R. F., et al.** (2007). Self-control relies on glucose as a limited energy source. *JPSP*, 92, 325–336.
   - 自控力依赖血糖作为有限能源

6. **Baumeister, R. F. & Vohs, K. D.** (2016). Strength model of self-regulation as limited resource. *Advances in Experimental Social Psychology*, 54, 67–127.
   - 力量模型的全面回顾与更新

### 二人对话的热力学

7. **Badalamenti, A. F. & Langs, R. J.** (1992). The thermodynamics of psychotherapeutic communication. *Behavioral Science*, 37(3), 157–180.
   - 二人组作为封闭热力学系统，熵遵循 ln(1+t) 定律
   - [PubMed](https://pubmed.ncbi.nlm.nih.gov/1497564/)

8. **Hilpert, P., et al.** (2024). Dynamical stress crossover: A coupled oscillator approach. *Journal of Social and Personal Relationships.*
   - 189 对夫妻、25,834 话轮——压力交叉的耦合振荡模型

### 非稳态负荷与分诊

9. **McEwen, B. S.** (1998). Stress, adaptation, and disease: Allostasis and allostatic load. *Annals of the New York Academy of Sciences*, 840, 33–44.
   - 非稳态负荷的奠基性论文

10. **Kelley, D. P., et al.** (2025). The allostatic triage model of psychopathology. *Neuroscience & Biobehavioral Reviews.*
    - 大脑在应激下重新分配有限能量资源——"从心智到线粒体"的分诊
    - [PubMed](https://pubmed.ncbi.nlm.nih.gov/41093260/)

### 耗散结构

11. **Prigogine, I. & Stengers, I.** (1984). *Order out of Chaos: Man's New Dialogue with Nature.*
    - 远离平衡态的自组织

### 关系渗透与社会渗透

12. **Altman, I. & Taylor, D. A.** (1973). *Social Penetration: The Development of Interpersonal Relationships.*
    - 亲密与信任的共生关系

### 情绪调节

13. **Gross, J. J.** (2015). Emotion regulation: Current status and future prospects. *Psychological Inquiry*, 26(1), 1–26.
    - 情绪调节的过程模型

14. **Richards, J. M. & Gross, J. J.** (2000). Emotion regulation and memory: The cognitive costs of keeping one's cool. *JPSP*, 79, 410–424.
    - 压抑（Suppression）比表达消耗更多认知资源

### 补充

15. **Ryan, R. M. & Deci, E. L.** (2008). From ego depletion to vitality: Theory and findings concerning the facilitation of energy available to the self. *Social and Personality Psychology Compass*, 2, 702–717.
    - 自主动机可以维持甚至增强可用能量

16. **Selye, H.** (1950). Stress and the general adaptation syndrome. *British Medical Journal*, 1(4667), 1383–1392.
    - 一般适应综合征——有限的适应能量

---

> **文档版本**：v1.0 — 2026-06-25
> **状态**：设计方案，待实现
> **关联 TODO**：#0a, #0b, #0c, #1, #2, #4
