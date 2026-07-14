# 状态引擎数学模型

## State Engine Mathematical Model

> Luna 连续人格动力系统的完整数学描述。
> 基于 Bowlby 依恋理论、残差式情感动力学 (Oravecz et al., 2009) 与 SSM 双速门控。

---

## 1. 符号系统

### 1.1 状态空间

| 符号 | 维度 | 值域 | 描述 |
|------|:----:|:----:|------|
| $\mathbf{h}_t$ | 8 | $[-1, 1]$ | 内部状态 (Internal State) |
| $\mathbf{r}_t$ | 3 | $[-1, 1]$ | 关系状态 (Relationship State) |
| $\mathbf{s}_t$ | 7 | $[-1, 1]$ | 表面状态 (Surface State) |
| $\mathbf{p}$ | 10 | $[-1, 1]$ | 人格特质 (Traits, **静态**) |
| $\boldsymbol{\sigma}_t$ | 7 | $[0, 1]$ | 心理刺激 (Stimulus Vector, 每轮临时) |

### 1.2 索引约定

**内部状态** $\mathbf{h}$:
$$
h[0]=\text{Energy},\;
h[1]=\text{Stress},\;
h[2]=\text{Loneliness},\;
h[3]=\text{Insecurity}
$$
$$
h[4]=\text{Irritation},\;
h[5]=\text{Longing},\;
h[6]=\text{SocialBattery},\;
h[7]=\text{MentalFatigue}
$$

**关系状态** $\mathbf{r}$:
$$
r[0]=\text{Affection},\;
r[1]=\text{TrustBond},\;
r[2]=\text{Intimacy}
$$

**表面状态** $\mathbf{s}$:
$$
s[0]=\text{Expressiveness},\;
s[1]=\text{Warmth},\;
s[2]=\text{Sharpness},\;
s[3]=\text{Softness}
$$
$$
s[4]=\text{Enthusiasm},\;
s[5]=\text{Restraint},\;
s[6]=\text{Vulnerability}
$$

**刺激** $\boldsymbol{\sigma}$:
$$
\sigma[0]=\text{Abandonment},\;
\sigma[1]=\text{Validation},\;
\sigma[2]=\text{Closeness},\;
\sigma[3]=\text{Conflict}
$$
$$
\sigma[4]=\text{Dependency},\;
\sigma[5]=\text{Teasing},\;
\sigma[6]=\text{EmotionalWeight}
$$

---

## 2. 管线总览 (每轮对话)

每轮 $t$ 的执行顺序（**认知评价优先于反馈**，2026-06-25 修复 #5）：

```
                      ┌── 辅助增益回路 ──┐
                      ↓                  │
   ┌─────────────────────────────────────┴─────────────────┐
   │ ④ Surface → Internal Feedback (auxiliary)            │
   │    s_{t-1} → h_{t-1} 微量调制，得到 h'_{t-1}         │
   │    + 防御代谢成本 (高 d/a 时的独立能耗)               │
   └──────────────────┬──────────────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────────────┐
   │ ① Defense Profiles — 主路径起点                     │
   │    (deactivation, hyperactivation) = f(p, r_{t-1},  │
   │     h_{t-1})                                       │
   │     → inner_σ, outer_σ                              │
   └──────────────────┬──────────────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────────────┐
   │ ② Residual Dynamics — 跳-扩散混合                   │
   │    h_t = h_{t-1} + dt×漂移 + 跳跃 + 噪声             │
   │    r_t = r_{t-1} + dt×漂移 + 跳跃                    │
   └──────────────────┬──────────────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────────────┐
   │ ③ Surface Projection — 主路径终点                   │
   │    s_t = α·proj(h_t, r_t, outer_σ) + (1-α)·s_{t-1} │
   └─────────────────────────────────────────────────────┘
                      ↓
                      └── s_t 进入 LLM，同时存入下一轮的反馈源 ──→
```

对话间隔中的时间衰减（不参与每轮动态）：

```
   ┌─────────────────────────────────────────────────────┐
   │ ⑤ Time Decay                                        │
   │    h' = setpoint + (h - setpoint)·exp(-λ·Δt)        │
   │    r' = setpoint + (r - setpoint)·exp(-λ_rel·Δt)    │
   └─────────────────────────────────────────────────────┘
```

---

## 3. ① 防御剖面 (Defense Profiles)

基于 Bowlby (1980) 依恋防御二分法，7 维逐类型敏感度。

### 3.1 秩-1 分解结构

PCA 审计发现 7 维有效秩仅 ~1.4–1.9，故采用秩-1 分解：

$$
\begin{aligned}
\tilde{\mathbf{d}}(\mathbf{p}, \mathbf{r}, \mathbf{h}) &= \mathbf{b}_d + I_d(\mathbf{p}, \mathbf{r}, \mathbf{h}) \cdot \mathbf{v}_d \\[4pt]
\tilde{\mathbf{a}}(\mathbf{p}, \mathbf{r}) &= \mathbf{b}_a + I_a(\mathbf{p}, \mathbf{r}) \cdot \mathbf{v}_a + \mathbf{m}_a(\mathbf{h})
\end{aligned}
$$

其中：
- $\mathbf{b}_d, \mathbf{b}_a \in \mathbb{R}^7$ — 基线向量（pre-sigmoid 均值）
- $\mathbf{v}_d, \mathbf{v}_a \in \mathbb{R}^7$ — PC1 方向（单位向量）
- $I_d, I_a \in \mathbb{R}$ — 标量强度，由特质和状态线性调制
- $\mathbf{m}_a(\mathbf{h}) \in \mathbb{R}^7$ — 状态调制项（hyperactivation 专属，允许交叉模式）

### 3.2 强度函数

**Deactivation 强度**（21 维输入 → 标量）：

$$
I_d = \mathbf{w}_d^{\mathsf{T}} \begin{bmatrix} \mathbf{p} \\ \mathbf{r} \\ \mathbf{h} \end{bmatrix}
$$

关键贡献：
- `pride` $+2.00$ — 高自尊增加防御
- `attachment_avoidance` $+1.30$ — 疏离策略核心驱动
- `emotional_stability` $-1.20$ — 安全型减少防御
- `trust_bond` $-0.40$ — 安全基地效应
- `stress` $+0.70$, `insecurity` $+0.65$ — 急性状态增强防御

**Hyperactivation 强度**（14 维输入 → 标量，仅特质 + 关系，状态调制单独处理）：

$$
I_a = \mathbf{w}_a^{\mathsf{T}} \begin{bmatrix} \mathbf{p} \\ \mathbf{r} \end{bmatrix}
$$

关键贡献：
- `attachment_anxiety` $+2.40$ — 核心驱动
- `attachment_avoidance` $-2.10$ — 核心抑制
- `sensitivity` $+0.55$, `jealousy_sensitivity` $+0.45$
- `intimacy` $+0.40$, `affection` $+0.30$

### 3.3 状态调制（Hyperactivation 专属）

$$
\mathbf{m}_a(\mathbf{h}) = \mathbf{W}_{hm} \mathbf{h}, \quad \mathbf{W}_{hm} \in \mathbb{R}^{7 \times 8}
$$

示例连接：
- `insecurity` $\rightarrow$ `ST_ABANDONMENT` $+0.50$ — 不安全感放大被抛弃恐惧
- `irritation` $\rightarrow$ `ST_CONFLICT` $+0.30$ — 烦躁降低冲突触发阈值
- `loneliness` $\rightarrow$ `ST_CLOSENESS` $+0.20$ — 孤独增强对亲密的渴望
- `stress` $\rightarrow$ `ST_CLOSENESS` $-0.12$ — 压力减少对亲近信号的接收

### 3.4 Sigmoid 映射

$$
\begin{aligned}
\mathbf{d} &= \text{sigmoid}\big((\tilde{\mathbf{d}} - \gamma_d) \cdot \kappa_d\big) \\[4pt]
\mathbf{a} &= \text{sigmoid}\big((\tilde{\mathbf{a}} - \gamma_a) \cdot \kappa_a\big)
\end{aligned}
$$

其中 $\gamma_d = 0.35, \gamma_a = 0.38$ 为 sigmoid 偏移，$\kappa_d = \kappa_a = 5.0$ 为缩放倍数。

输出约束：$\mathbf{d}, \mathbf{a} \in [0, 1]^7$

### 3.5 防御应用

防御剖面将原始刺激 $\boldsymbol{\sigma}$ 分为"内心感受"和"外在表达"两层：

$$
\begin{aligned}
\sigma^{\text{inner}}[i] &= \sigma[i] \cdot (1 + 0.50 \cdot \mathbf{a}[i]) \\[4pt]
\sigma^{\text{outer}}[i] &= \sigma^{\text{inner}}[i] \cdot (1 - 0.70 \cdot \mathbf{d}[i])
\end{aligned}
$$

其中：
- $\mathbf{a}[i] \uparrow$ → hyperactivation 放大内心感受（"我比看起来更在意"）
- $\mathbf{d}[i] \uparrow$ → deactivation 削减外在表达（"我不想让人看出来"）
- 两者独立：可 `高a + 高d`（内心翻涌但表面平静）或 `低a + 低d`（表里如一）

输出范围：$\sigma^{\text{inner}}, \sigma^{\text{outer}} \in [0, 1]^7$

### 3.6 防御代谢成本（2026-06-25 修复 #4）

防御剖面本身消耗能量，不依赖表面表达通路。高 a（过度激活/内心翻涌）和高 d（去激活/压抑）都产生直接代谢成本：

$$
\Delta\mathbf{h}^{\text{defcost}} = \text{cost}_{\text{deact}}(\mathbf{d}) + \text{cost}_{\text{hyper}}(\mathbf{a})
$$

- 去激活（压抑）：$\text{Energy} \downarrow$, $\text{MentalFatigue} \uparrow$（前额叶抑制消耗）
- 过度激活（翻涌）：$\text{Stress} \uparrow$, $\text{MentalFatigue} \uparrow$（情绪放大消耗）
- 高依恋焦虑者成本更高（情绪调节效率更低，$\times (1 + 0.3\cdot\text{anxiety})$）

应用时机：在表面反馈之后、动力学之前。这样即使 $\mathbf{s}=0$（面无表情），高防御状态仍产生代谢债务。


---

## 4. ④ 表面→内部反馈 (Surface→Internal Feedback)

**延迟因果**：上一轮的表面表达 $\mathbf{s}_{t-1}$ 反馈调制本轮的初始内部状态 $\mathbf{h}_{t-1}$，得到修正值 $\mathbf{h}'_{t-1}$，再由动力学更新为 $\mathbf{h}_t$。

$$
\mathbf{h}_{t-1} \xrightarrow{\;+\;\Delta\mathbf{h}^{\text{fb}}(\mathbf{s}_{t-1})\;} \mathbf{h}'_{t-1} \xrightarrow{\text{动力学}} \mathbf{h}_t
$$

该时序对应"先笑 → 然后感觉变好"的心理过程（surface[t-1] → internal[t] 而非同轮即时反馈）。

### 4.1 反馈计算

$$
\Delta\mathbf{h}^{\text{fb}} = \mathbf{W}_{\text{fb}}^{+} \cdot \max(\mathbf{s}_{t-1}, \mathbf{0}) \;+\; \mathbf{W}_{\text{fb}}^{-} \cdot \min(\mathbf{s}_{t-1}, \mathbf{0})
$$

其中：
- $\mathbf{W}_{\text{fb}}^{+} \in \mathbb{R}^{7 \times 8}$ — 正值反馈矩阵（主动表达 → 失调/反馈/消耗）
- $\mathbf{W}_{\text{fb}}^{-} \in \mathbb{R}^{7 \times 8}$ — 负值反馈矩阵（压抑/伪装 → 代谢成本）

### 4.2 三类反馈机制

**① 情绪失调成本** — 表里不一产生内部压力：
$$
\begin{aligned}
\text{RESTRAINT} &\rightarrow \text{STRESS}: +0.06 \\
\text{WARMTH} &\rightarrow \text{STRESS}: +0.04
\end{aligned}
$$

**② 面部/躯体反馈** — 表达改变内在感受：
$$
\begin{aligned}
\text{SHARPNESS} &\rightarrow \text{IRRITATION}: +0.05 \\
\text{WARMTH} &\rightarrow \text{LONELINESS}: -0.04 \\
\text{ENTHUSIASM} &\rightarrow \text{ENERGY}: +0.05 \\
\text{VULNERABILITY} &\rightarrow \text{LONGING}: +0.04
\end{aligned}
$$

**③ 表达消耗成本** — 压抑远大于表达（Richards & Gross, 2000）：
$$
\begin{aligned}
\text{EXPRESSIVENESS} &\rightarrow \text{ENERGY}: -0.03 \quad\text{(本能反射，代价低)} \\
\text{EXPRESSIVENESS} &\rightarrow \text{MENTAL\_FATIGUE}: +0.02 \quad\text{(脸酸)} \\
\text{RESTRAINT} &\rightarrow \text{MENTAL\_FATIGUE}: +0.10 \quad\text{(前额叶持续抑制)} \\
\text{RESTRAINT} &\rightarrow \text{STRESS}: +0.08 \quad\text{(表里不一的失调成本)}
\end{aligned}
$$
### 4.3 反馈后的调制

$$
\mathbf{h}'_{t-1} = \texttt{soft\_clamp}(\mathbf{h}_{t-1} + \Delta\mathbf{h}^{\text{fb}}, -1, 1)
$$

`soft_clamp` 在区间内恒等，边界外 tanh 软饱和。

---

## 5. ② 残差动力学 (Residual Dynamics)

**核心思想**：无内建稳态恢复（稳态拉回由时间衰减处理），每轮只做刺激驱动 + 耦合驱动 的残差增量。

### 5.1 内部状态更新

$$
\mathbf{h}_t = \mathbf{h}_{t-1} + \Delta t \cdot \underbrace{\big(
    \alpha \cdot \Delta\mathbf{h}^{\text{cpl}} \big)}_{\text{漂移(率×dt)}}
    + \underbrace{\vphantom{\big(}\text{EnergyGate}(\boldsymbol{\beta} \odot \boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}}\mathbf{W}_B}_{\text{跳跃(点过程，不乘dt)}}
    + \underbrace{\vphantom{\big(}\Delta\mathbf{h}^{\text{defcost}}}_{\text{防御成本}} + \boldsymbol{\epsilon}
$$
#### α: 跨维度耦合速率

$$
\alpha = \text{clamp}\big(\mathbf{w}_\alpha^{\mathsf{T}}[\mathbf{p}; \mathbf{r}_{t-1}],\; 0.05,\; 0.40\big)
$$

受 `emotional_openness` $(+0.15)$、`emotional_stability` $(-0.075)$、`trust_bond` $(+0.06)$ 调制。
基础偏置 $0.285$。

#### Δ_coupling: 跨维度耦合

$$
\Delta\mathbf{h}^{\text{cpl}} = \mathbf{h}'_{t-1}^{\mathsf{T}}\mathbf{W}_{cc} - \boldsymbol{\lambda} \odot (\mathbf{h}'_{t-1} - \mathbf{h}^{\text{target}})
$$

- $\mathbf{W}_{cc} \in \mathbb{R}^{8 \times 8}$ — 内部耦合矩阵（14 条稀疏规则，~17.2% 密度）
- $\boldsymbol{\lambda} \in \mathbb{R}^8$ — 每维自阻尼率（0.06–0.22）
- $\mathbf{h}^{\text{target}}$ — 自阻尼收敛目标（仅 `social_battery` 非零 $+0.20$）

代表性耦合规则：
$$
\begin{aligned}
\text{Energy} &\rightarrow \text{Stress}: &-0.05 &\quad\text{(精力充沛→压力降低)} \\
\text{Insecurity} &\rightarrow \text{Stress}: &+0.10 &\quad\text{(不安全感→压力上升)} \\
\text{Stress} &\rightarrow \text{Irritation}: &+0.15 &\quad\text{(压力积累→易怒)} \\
\text{Loneliness} &\rightarrow \text{Longing}: &+0.15 &\quad\text{(孤独→思念)} \\
\text{SocialBattery} &\rightarrow \text{Irritation}: &-0.08 &\quad\text{(电量低→烦躁)}
\end{aligned}
$$

#### Δ_stimulus: 刺激驱动

$$
\Delta\mathbf{h}^{\text{stim}} = (\boldsymbol{\beta} \odot \boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}} \mathbf{W}_{B}
$$

- $\boldsymbol{\beta} \in \mathbb{R}^7$ — 常数接受率，每维 $0.05$（方案 B：防御不调制 $\beta$）
- $\mathbf{W}_{B} \in \mathbb{R}^{7 \times 8}$ — 输入影响矩阵（28.6% 密度）

代表性连接（每维唯一起源签名，约束⑩）：
$$
\begin{aligned}
\text{abandonment} &\rightarrow \text{insecurity}+0.28,\; \text{loneliness}+0.22,\; \text{longing}+0.18 \\
\text{validation} &\rightarrow \text{energy}+0.22,\; \text{insecurity}-0.22 \\
#### Energy Gate——能量门控（2026-06-25 修复 #0a）

刺激驱动的振幅受限于当前可用能量。低电量时情绪响应被生理性压制（前额叶葡萄糖耗尽效应）：

$$
\text{EnergyGate} = \frac{1 + h[\text{Energy}]}{2} \in [0, 1], \quad
h[\text{Energy}] = -1 \to 0, \; h[\text{Energy}] = +1 \to 1
$$

最低保留 10% 响应（社会脚本的最低能耗）：

$$
\Delta\mathbf{h}^{\text{stim}} = \max(0.1,\; \text{EnergyGate}) \cdot (\boldsymbol{eta} \odot \boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}} \mathbf{W}_B
$$

\text{conflict} &\rightarrow \text{stress}+0.35,\; \text{irritation}+0.30,\; \text{energy}-0.20 \\
\text{emotional\_weight} &\rightarrow \text{stress}+0.22,\; \text{mental\_fatigue}+0.18
\end{aligned}
$$

#### SSM 值相关双速门控

$$
\Delta\mathbf{h}^{\text{gate}}[i] = \Delta\mathbf{h}[i] \cdot G[i, \, \mathbb{1}_{\Delta\mathbf{h}[i] \geq 0}]
$$

$$
G[i, j] = \begin{cases}
G[i, 0] & \text{if } \Delta\mathbf{h}[i] \geq 0 \quad (\text{上升增益}) \\
G[i, 1] & \text{if } \Delta\mathbf{h}[i] < 0 \quad (\text{下降增益})
\end{cases}
$$

$\mathbf{G} \in \mathbb{R}^{8 \times 2}$ 实现情感不对称：

| 维度 | 上升增益 | 下降增益 | 效应 |
|:----:|:--------:|:--------:|------|
| Stress | 1.0 | 0.6 | 快起慢消 |
| Irritation | 1.0 | 0.6 | 一触即炸、平息慢 |
| Loneliness | 0.3 | 0.5 | 持久不散 |
| Longing | 0.2 | 0.5 | 极持久牵挂 |
| Energy | 0.6 | 0.4 | 耗尽恢复慢 |
| SocialBattery | 0.6 | 0.3 | 低电恢复极慢 |

#### Langevin 噪声

$$
\mathbf{h}_t = \texttt{soft\_clamp}(\mathbf{h}_t + \boldsymbol{\epsilon}, -1, 1), \quad
\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}),\; \sigma = 0.015
$$

每维独立高斯噪声，打破维度同步（O-U 过程标准做法）。

### 5.2 关系状态更新

$$
\mathbf{r}_t = \mathbf{r}_{t-1} + \Delta t \cdot \big(
    \alpha_r \cdot \Delta\mathbf{r}^{\text{cpl}} + \Delta\mathbf{r}^{\text{stim}}
\big)
$$

#### α_rel: 关系耦合速率（比内部慢 5–10×）

$$
\alpha_r = \text{clamp}\big(\mathbf{w}_{\alpha_r}^{\mathsf{T}}[\mathbf{p}; \mathbf{r}_{t-1}],\; 0.005,\; 0.08\big)
$$

基础偏置 $0.045$，受 `emotional_openness` $(+0.02)$ 和 `trust_bond` $(+0.015)$ 调制。

#### β_rel: 关系刺激接受率

$$
\beta_r = \text{clamp}\big(\mathbf{w}_{\beta_r}^{\mathsf{T}}\mathbf{p},\; 0.002,\; 0.06\big)
$$

基础偏置 $0.013$，受 `attachment_anxiety` $(+0.0075)$ 调制。

#### Δ_coupling: 关系耦合 + 跨尺度耦合

$$
\Delta\mathbf{r}^{\text{cpl}} = \mathbf{r}_{t-1}^{\mathsf{T}}\mathbf{W}_{rc} + \mathbf{h}_t^{\mathsf{T}}\mathbf{W}_{cs} - \boldsymbol{\lambda}_r \odot \mathbf{r}_{t-1}
$$

- $\mathbf{W}_{rc} \in \mathbb{R}^{3 \times 3}$ — 关系耦合矩阵（44.4% 密度，3×3 特例）
- $\mathbf{W}_{cs} \in \mathbb{R}^{8 \times 3}$ — 跨尺度耦合（内部 → 关系，5 条规则）
- $\boldsymbol{\lambda}_r \in \mathbb{R}^3$ — 关系自阻尼率（0.10–0.12）

关系耦合级联设计：
$$
\text{Affection} \xrightarrow{+0.10} \text{TrustBond} \xrightarrow{+0.08} \text{Intimacy}
$$

末端反馈防止 intimacy 成为死胡同：
$$
\begin{aligned}
\text{Intimacy} &\rightarrow \text{Affection}: +0.03 \\
\text{Intimacy} &\rightarrow \text{TrustBond}: (\text{已移除 2026-06-25 \#7}) \\
\end{aligned}
$$

跨尺度耦合：
$$
\begin{aligned}
\text{Stress} &\rightarrow \text{TrustBond}: -0.03 \\
\text{Stress} &\rightarrow \text{Intimacy}: +0.015 \\
\text{Energy} &\rightarrow \text{Affection}: +0.015
\end{aligned}
$$

#### 刺激驱动

$$
\Delta\mathbf{r}^{\text{stim}} = \beta_r \cdot (\boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}}\mathbf{W}_{B_r}
$$

$\mathbf{W}_{B_r} \in \mathbb{R}^{7 \times 3}$ 28.6% 密度，每维关系态有完全不重叠的刺激来源：

$$
\begin{aligned}
\text{Affection} &\leftarrow \text{validation}\; (+0.18),\; \text{closeness}\; (+0.10) \\
\text{TrustBond} &\leftarrow \text{conflict}\; (-0.25),\; \text{abandonment}\; (-0.10) \\
\text{Intimacy} &\leftarrow \text{dependency}\; (+0.15),\; \text{teasing}\; (+0.10)
\end{aligned}
$$

---

## 6. ③ 表面投影 (Surface Projection)

### 6.1 线性基线

$$
\mathbf{s}^{\text{raw}} = \mathbf{W}_s \begin{bmatrix} \mathbf{h}_t \\ \mathbf{r}_t \\ \boldsymbol{\sigma}^{\text{outer}} \end{bmatrix} + \mathbf{b}_s
$$

- $\mathbf{W}_s \in \mathbb{R}^{18 \times 7}$ — 表面映射矩阵（带偏置的线性映射）
- traits **不直连** surface（约束①），通过 defense/dynamics 间接影响

代表性连接：

内部态 → 表面（核心："真实感受的外泄"）：
$$
\begin{aligned}
\text{energy} &\rightarrow \text{expressiveness}+0.4,\; \text{enthusiasm}+0.5 \\
\text{irritation} &\rightarrow \text{sharpness}+0.5 \\
\text{stress} &\rightarrow \text{restraint}+0.2,\; \text{warmth}-0.15 \\
\text{loneliness} &\rightarrow \text{vulnerability}+0.3 \\
\text{insecurity} &\rightarrow \text{restraint}+0.3
\end{aligned}
$$

关系态 → 表面：
$$
\begin{aligned}
\text{affection} &\rightarrow \text{warmth}+0.4 \\
\text{trust\_bond} &\rightarrow \text{softness}+0.2
\end{aligned}
$$

外刺激 → 表面（压抑后的残余）：
$$
\begin{aligned}
\text{validation} &\rightarrow \text{warmth}+0.3,\; \text{enthusiasm}+0.15 \\
\text{conflict} &\rightarrow \text{sharpness}+0.25 \\
\text{closeness} &\rightarrow \text{softness}+0.2
\end{aligned}
$$

### 6.2 惯性混合系数

$$
\alpha = \text{clamp}\big(0.5 - 0.3 \cdot \max(0, h[\text{Stress}]) + 0.2 \cdot \max(0, h[\text{Energy}]),\; 0.1,\; 0.9\big)
$$

效应：
- 高压力 → $\alpha \downarrow$（表面更"僵"，惯性更强，不易变化）
- 高精力 → $\alpha \uparrow$（表面更"灵"，响应更快）
- 默认 $\alpha \approx 0.5$（压力、精力中性时）

### 6.3 时间衰减混合

当 $\Delta t > 0.01$ 小时（对话间隔存在时）：

$$
\mathbf{s}'_{t-1} = \begin{cases}
\mathbf{s}^{\text{raw}} + (\mathbf{s}_{t-1} - \mathbf{s}^{\text{raw}}) \cdot e^{-0.5 \cdot \Delta t}, & \Delta t < 1.0\,\text{h} \\[4pt]
\mathbf{s}^{\text{sp}} + (\mathbf{s}_{t-1} - \mathbf{s}^{\text{sp}}) \cdot e^{-0.3 \cdot \Delta t}, & \Delta t \geq 1.0\,\text{h}
\end{cases}
$$

短间隔（<1h）半衰期约 $t_{1/2} = \ln 2 / 0.5 \approx 1.4$ 小时，向 raw（当前内心投影）回归。
长间隔（≥1h）向人格基线 $\mathbf{s}^{\text{sp}}$ 回归——独处久了表面不应残留上轮情绪映射（修复 2026-06-25 #3）。
表面基线 $\mathbf{s}^{\text{sp}}$ 由 SURFACE_MAPPER 的偏置项决定（中性"静息脸"，≈ [-0.3, -0.2, -0.1, -0.1, -0.2, -0.1, -0.5]）。

### 6.4 最终输出

$$
\mathbf{s}_t = \texttt{soft\_clamp}\big(
    \alpha \cdot \mathbf{s}^{\text{raw}} + (1 - \alpha) \cdot \mathbf{s}'_{t-1},
    -1, 1
\big)
$$

首帧（$\mathbf{s}_{t-1} = \text{None}$）无极惯性混合，$\mathbf{s}_t = \mathbf{s}^{\text{raw}}$。

---

## 7. ⑤ 时间衰减 (Time Decay)

**不参与每轮动态**，仅在对话间隔中执行。将状态拉到人格决定的基线。

### 7.1 有效衰减率

$$
\lambda_{\text{eff}}[i] = \frac{\lambda_{\text{base}}[i] \cdot M_p(\mathbf{p})}{1 + k \cdot \Delta t}
$$

- $\lambda_{\text{base}}$ — 基础衰减率（每小时）
- $M_p$ — 人格调制因子
- $k$ — 时间曲线参数（幂律尾拟合）

**人格调制**：
$$
M_p^{\text{int}} = \text{clamp}\big(1.0 + 0.15\cdot\text{stability} + 0.075\cdot\text{optimism} - 0.125\cdot\text{anxiety},\; 0.3,\; 2.0\big)
$$

**时间曲线** $1/(1 + k \cdot \Delta t)$：
- $\Delta t \to 0$: $\lambda_{\text{eff}} \approx \lambda_{\text{base}} \cdot M_p$（全速衰减）
- $\Delta t \to \infty$: $\lambda_{\text{eff}} \to 0$（幂律长尾）

### 7.2 内部状态衰减

$$
\mathbf{h}^{\text{decay}}[i] = \mathbf{h}^{\text{sp}}[i] + \big(\mathbf{h}[i] - \mathbf{h}^{\text{sp}}[i]\big) \cdot e^{-\lambda_{\text{eff}}[i] \cdot \Delta t}
$$

其中 $\mathbf{h}^{\text{sp}} = \mathbf{W}_{sp} \mathbf{p} + \mathbf{b}_{sp}$（人格决定的基线）。

基础衰减率 $\lambda_{\text{base}}$（每小时）：

| 维度 | $\lambda$ | 半衰期 |
|:----:|:---------:|:------:|
| Irritation | 0.69 | ~1 h |
| Energy | 0.35 | ~2 h |
| SocialBattery | 0.35 | ~2 h |
| Stress | 0.23 | ~3 h |
| Loneliness | 0.17 | ~4 h |
| Insecurity | 0.14 | ~5 h |
| Longing | 0.12 | ~6 h |

### 7.3 关系状态衰减

同构结构，但衰减率慢得多（天级半衰期）：

| 维度 | $\lambda$ | 半衰期 |
|:----:|:---------:|:------:|
| Intimacy | 0.0041 | ~7 d |
| Affection | 0.0021 | ~14 d |
| TrustBond | 0.0021 | ~14 d |

### 7.4 非对称衰减 (Fading Affect Bias)

关系状态中负向偏离 $(r[i] < 0 \land r[i] < \text{sp}[i])$ 加速衰减：

$$
\lambda_{\text{eff}}[i] \gets \lambda_{\text{eff}}[i] \times 1.8
$$

负面关系印象消退比正面快 $1.8\times$（心理学 FAB 效应）。

### 7.5 长时间间隔收敛保障

当 $\Delta t > 168$ 小时（> 1 周）：

$$
\mathbf{h}^{\text{decay}} \gets \mathbf{h}^{\text{decay}} + \big(1 - e^{-0.01 \cdot (\Delta t - 168)}\big) \cdot (\mathbf{h}^{\text{sp}} - \mathbf{h}^{\text{decay}})
$$

消除幂律残余（$1/(1 + k \cdot \Delta t)$ 在 $\Delta t \to \infty$ 时的渐近残余）。

---

## 8. 初始化

首次运行时，所有状态从人格特质推算：

$$
\begin{aligned}
\mathbf{h}_0 &= \texttt{clamp}(\mathbf{W}_{sp}\mathbf{p} + \mathbf{b}_{sp},\; -0.9,\; 0.9) \\[4pt]
\mathbf{r}_0 &= \texttt{clamp}(\mathbf{W}_{rp}\mathbf{p} + \mathbf{b}_{rp},\; -0.96,\; 0.96) \\[4pt]
\mathbf{s}_0 &= \texttt{project\_surface}(\mathbf{h}_0, \mathbf{r}_0, \mathbf{0}, \text{prev}=\text{None})
\end{aligned}
$$

---

## 9. 完整的单轮更新方程

综合 4 步管线，每轮 $t$ 的完整变换：

$$
$$
\boxed{
\begin{aligned}
& \text{① 防御剖面（认知评价优先）} \\[4pt]
\mathbf{d} &= \text{sigmoid}\big((\mathbf{b}_d + I_d(\mathbf{p}, \mathbf{r}_{t-1}, \mathbf{h}_{t-1}) \cdot \mathbf{v}_d - \gamma_d) \cdot \kappa_d\big) \\[4pt]
\mathbf{a} &= \text{sigmoid}\big((\mathbf{b}_a + I_a(\mathbf{p}, \mathbf{r}_{t-1}) \cdot \mathbf{v}_a + \mathbf{m}_a(\mathbf{h}_{t-1}) - \gamma_a) \cdot \kappa_a\big) \\[4pt]
\boldsymbol{\sigma}^{\text{inner}} &= \boldsymbol{\sigma}' \odot (\mathbf{1} + 0.5 \cdot \mathbf{a}) \\[4pt]
\boldsymbol{\sigma}^{\text{outer}} &= \boldsymbol{\sigma}^{\text{inner}} \odot (\mathbf{1} - 0.7 \cdot \mathbf{d}) \\[4pt]
& \text{④ 表面反馈（辅助增益）+ 防御成本} \\[4pt]
\Delta\mathbf{h}^{\text{fb}} &= \mathbf{W}_{\text{fb}}^{+}\max(\mathbf{s}_{t-1}, \mathbf{0}) + \mathbf{W}_{\text{fb}}^{-}\min(\mathbf{s}_{t-1}, \mathbf{0}) \\[4pt]
\Delta\mathbf{h}^{\text{defcost}} &= \mathbf{c}_{\text{deact}}(\mathbf{d}) + \mathbf{c}_{\text{hyper}}(\mathbf{a}) \\[4pt]
\mathbf{h}' &= \texttt{clamp}\big(\mathbf{h}_{t-1} + \Delta\mathbf{h}^{\text{fb}} + \Delta\mathbf{h}^{\text{defcost}}, -1, 1\big) \\[4pt]
& \text{② 跳-扩散动力学} \\[4pt]
\text{漂移} &= \alpha \big( \mathbf{h}'^{\mathsf{T}}\mathbf{W}_{cc} - \boldsymbol{\lambda} \odot (\mathbf{h}' - \mathbf{h}^{\text{target}}) \big) \\[4pt]
\text{跳跃} &= \max\big(0.1,\; \tfrac{1 + h'[\text{Energy}]}{2}\big) \cdot (\boldsymbol{\beta} \odot \boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}}\mathbf{W}_B \\[4pt]
\Delta\mathbf{h} &= \text{漂移} \cdot \Delta t + \text{跳跃} \\[4pt]
\Delta\mathbf{h} &= \Delta\mathbf{h} \odot \mathbf{G}[\;,\; \mathbb{1}_{\Delta\mathbf{h} \geq 0}] \\[4pt]
\mathbf{h}_t &= \texttt{clamp}(\mathbf{h}' + \Delta\mathbf{h} + \boldsymbol{\epsilon}, -1, 1) \\[4pt]
\Delta\mathbf{r} &= \alpha_r\big( \mathbf{r}_{t-1}^{\mathsf{T}}\mathbf{W}_{rc} + \mathbf{h}_t^{\mathsf{T}}\mathbf{W}_{cs} - \boldsymbol{\lambda}_r \odot \mathbf{r}_{t-1} \big) \cdot \Delta t + \beta_r(\boldsymbol{\sigma}^{\text{inner}})^{\mathsf{T}}\mathbf{W}_{B_r} \\[4pt]
\mathbf{r}_t &= \texttt{clamp}(\mathbf{r}_{t-1} + \Delta\mathbf{r}, -1, 1) \\[4pt]
& \text{③ 表面投影（带惯性混合 + 基线回归）} \\[4pt]
\mathbf{s}^{\text{raw}} &= \mathbf{W}_s [\mathbf{h}_t; \mathbf{r}_t; \boldsymbol{\sigma}^{\text{outer}}] + \mathbf{b}_s \\[4pt]
\alpha_s &= \text{clamp}\big(0.5 - 0.3 \cdot \max(0, h_t[\text{Stress}]) + 0.2 \cdot \max(0, h_t[\text{Energy}]),\; 0.1,\; 0.9\big) \\[4pt]
\mathbf{s}'_{t-1} &= \begin{cases} \mathbf{s}^{\text{raw}} + (\mathbf{s}_{t-1} - \mathbf{s}^{\text{raw}}) e^{-0.5\Delta t}, & \Delta t < 1\text{h} \\[4pt]
& \mathbf{s}^{\text{sp}} + (\mathbf{s}_{t-1} - \mathbf{s}^{\text{sp}}) e^{-0.3\Delta t}, & \Delta t \geq 1\text{h} \end{cases} \\[4pt]
\mathbf{s}_t &= \texttt{clamp}\big(\alpha_s \cdot \mathbf{s}^{\text{raw}} + (1 - \alpha_s) \cdot \mathbf{s}'_{t-1}, -1, 1\big)
\end{aligned}
}
$$
}
$$

---

## 10. 约束体系

| # | 约束 | 数学表达 | 验证 |
|:-:|------|----------|:----:|
| ③ | 低秩耦合 | $\text{rank}_{\text{eff}}(\mathbf{W}_{cc}) \ll 8$, $\text{rank}_{\text{eff}}(\mathbf{W}_{rc}) \ll 3$ | B_int 28.6%, B_rel 28.6% |
| ⑥ | 正交稀疏 | $\text{density}(\mathbf{W}) \leq 30\%$, $\max_{i \neq j} |\mathbf{G}_{\text{norm}}[i,j]| < 0.3$ | B_int ✅, B_rel ✅, 关系耦合 44.4% ⚠️(3×3特例) |
| ⑦ | 谱半径 | $\rho(\mathbf{W}_{cc}) < 0.95$, $\rho(\mathbf{W}_{rc}) < 0.95$ | 非线性 soft_clamp 保证 |
| ⑨ | 全局 Jacobian | $\text{density}(\nabla_{\text{total}}) \approx 19.9\%$ | 稀疏连接保证 |
| ⑩ | 刺激正交 | $\forall i \neq j: |\text{supp}(\mathbf{W}_{B}[i,:]) \cap \text{supp}(\mathbf{W}_{B}[j,:])| \leq 1$ | 每对 ≤1 共享目标 |

---

## 11. 参数规模

| 模块 | 参数类型 | 数量 | 审计状态 |
|:----:|----------|:----:|:---------:|
| 输入 B 矩阵 | WeightMapper | 16 条目 | ✅ 全部 reviewed |
| 关系 B 矩阵 | WeightMapper | 6 条目 | ✅ 全部 reviewed |
| 内部耦合 | WeightMapper | 14 规则 | ✅ 全部 reviewed |
| 关系耦合 | WeightMapper | 4 规则 | ✅ 全部 reviewed |
| 跨尺度耦合 | WeightMapper | 5 规则 | ✅ 全部 reviewed |
| 内部自阻尼 | WeightVector | 8 维 | ✅ 全部 reviewed |
| 关系自阻尼 | WeightVector | 3 维 | ✅ 全部 reviewed |
| α 速率 | LinearMapping | 4 参数 | ✅ 全部 reviewed |
| α_rel 速率 | LinearMapping | 3 参数 | ✅ 全部 reviewed |
| β_rel 速率 | LinearMapping | 2 参数 | ✅ 全部 reviewed |
| Setpoint 内部 | LinearMapping | 18 参数 | ✅ 全部 reviewed |
| Setpoint 关系 | LinearMapping | 9 参数 | ✅ 全部 reviewed |
| 防御强度 | 2× LinearMapping | 17 参数 | ✅ 全部 reviewed |
| 防御状态调制 | LinearMapping | 16 参数 | ✅ 全部 reviewed |
| 衰减率 | 2× WeightVector | 11 维 | ✅ 全部 reviewed |
| 人格调制 | 2× LinearMapping | 9 参数 | ✅ 全部 reviewed |
| 表面映射 | LinearMapping | 40+ 参数 | ✅ 全部 reviewed |
| 表面反馈 | 2× WeightMapper | 17 条 | ✅ 全部 reviewed |
| SSM 速度门控 | WeightMapper | 16 条目 | ✅ 全部 reviewed |
| **总计** | | **250+ 参数** | **无 `origin=legacy`** |

---

## 参考文献

- Bowlby, J. (1980). *Attachment and Loss, Vol. 3: Loss, Sadness and Depression.*
- Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2009). *A hierarchical Ornstein-Uhlenbeck model for continuous repeated measurement data.* Psychometrika.
- Mikulincer, M. & Shaver, P. R. (2003). *The attachment behavioral system in adulthood.* JPSP.
- Cacioppo, J. T. & Patrick, W. (2008). *Loneliness: Human Nature and the Need for Social Connection.*
- Gross, J. J. (2015). *Emotion regulation: Current status and future prospects.* Psychological Inquiry.
- Lazarus, R. S. & Folkman, S. (1984). *Stress, Appraisal, and Coping.*
- Berkowitz, L. (1989). *Frustration-aggression hypothesis.* Psychological Review.
