# TODO

> Luna 项目待办事项与已知问题清单。
> 更新于 2026-07-14。非线性动力学扩展与热力学会计已剥离至 `md/04-plans/THERMODYNAMIC_AFFECTIVE_MODEL.md`。
> v1.0 聚焦线性 ODE + soft_clamp + 时间衰减 框架内的精确修复。
> 完整分析见 `md/STATE_ENGINE_REVIEW_20260625.md`（待生成）。

---

## 已完成 ✅

- [x] **约束② StimulusMetadata** — 感知节点返回 confidence/source/decay_modulator/timestamp 结构
- [x] **约束⑪ StateFormatter 连续投影** — `_desc()` 替换为 9 区连续投影，消除硬阈值离散
- [x] **decay_modulator 持久化 + 时间衰减接入管线** — `state_engine_node` 先 `apply_time_decay` 再 `update_all`
- [x] **表面惯性时间衰减** — `project_surface` 新增 `delta_hours`，`prev_surface` 向 raw 回归
- [x] **表面负值反馈** — `SURFACE_FEEDBACK_NEG` 矩阵处理压抑/伪装的代谢成本
- [x] **State Formatter 重写**：`_desc()` 连续投影替代 5 级硬阈值 ✅ 06-22
- [x] **参数集中管理**：250+ 参数通过 WeightMapper/WeightVector/LinearMapping 管理，全 provenance

### 06-23 修复批 ✅
- [x] **非对称衰减正负判断** — `deviation < 0` → `(current < 0) & (deviation < 0)`（`_decay.py`）
- [x] **α/α_rel 裁剪边界放宽** — α: [0.02,0.35]→[0.05,0.40]; α_rel: [0.005,0.06]→[0.005,0.08]
- [x] **β_stim 乘法公式** — 加性→乘法 `β = max(ε, BASE+hyper·GAIN) · (1-deact·0.5)`
- [x] **vulnerability 入边增强** — 新增 stress(+0.10) + energy(-0.05) → vulnerability
- [x] **表面→内部反馈因果延迟** — surface[t-1] 影响 internal[t] 而非同轮即时反馈
- [x] **双速 rel_buffer** — 关系态每 3 轮更新一次（已移除，改为 β_rel 减半 + 每轮更新）
- [x] **B 矩阵秩验证** — 新增 `test_matrices.py` 中 `TestMatrixRank`（INPUT_INFLUENCE_B 秩≥6, REL 秩=3）
- [x] **关系级联双向** — 新增 intimacy→affection(+0.03) + intimacy→trust_bond(-0.01)
- [x] **感知层注入状态上下文** — internal/relationship 摘要注入 perception prompt
- [x] **深度分析测试** — 新增 `test_deep_analysis.py`（20 项追踪测试）

---

## ⚠️ 待检验（已知未修复，小影响）

- [ ] **时间衰减渐近收敛** 🟢：
  - 理论残余 exp(-λ_base/k) 最大 9.07%（longing 维度），实际影响很小（Δt>72h 已<5%）
  - if 分支在 Δt=168h 处的非平滑仅 0.13% 突变
  - **评估**：影响低，待大版本时用 γ<1 公式级修复

- [ ] **状态空间维度冗余（旧）** 🔴：PCA 14维有效自由度仅6维。B矩阵去相关化已完成（密度28.6%），但全局雅可比密度19.9%，仍偏高。

---

## 🔴 P0 — 数学结构缺陷（公式错误或遗漏，直接影响行为正确性）

> 完整热力学会计框架（含 #0b 熵增代价、#0c 总势能约束）已移入 `md/THERMODYNAMIC_AFFECTIVE_MODEL.md`，待 v2.0 非线性阶段设计。

### #0a 刺激驱动无能量约束（违反热力学第二定律——无源放大）
- **位置**：`state_engine/_dynamics.py:108-110`（`delta_stimulus` 直接无约束加性映射）
- **物理问题**：系统可凭空产生高振幅情绪——连续 σ=1.0 刺激可将 Irritation/Stress 线性推到 +1.0，不消耗任何"系统内部燃料"
- **真实物理/心理**：大脑产生强烈情绪需消耗葡萄糖和神经递质（多巴胺、肾上腺素），是"化学能→电能→情绪势能"的转化过程。社交电量低时情绪振幅必须被严重钳制——极度疲劳的人连发火都没力气
- **修正思路**：刺激驱动项乘以当前可用能量系数 `Energy_factor = (1 + h[Energy]) / 2`，低电量时外界再大刺激也掀不起浪

### #1 SSM 速度门控的方向判断逻辑错误
- **位置**：`state_engine/_dynamics.py:119-121`
- **问题**：`np.where(current >= 0, ...)` 按当前值符号选增益，但"快起慢消"应基于变化方向 `sign(Δh)` 而非 `sign(h)`
  - 当前 h>0 且 Δh<0（正在消退）→ 误用正区增益（快），违背"慢消"
  - 当前 h<0 且 Δh>0（正在上升）→ 误用负区增益（慢），违背"快起"
- **修正思路**：改为 `np.where(delta >= 0, rising_gain, falling_gain)`，列名改为 `rising_gain`/`falling_gain`
- **关联权重**：`_dynamics_weights.py:641` `SPEED_LABELS = ["positive_gain", "negative_gain"]`

### #2 Δt 双重身份谬误（积分步长 = 物理时间）
- **位置**：`state_engine/_dynamics.py:64,130`（`dt: float = 1.0` 硬编码）、`_pipeline.py:107-116`（未传 dt，始终用默认值 1.0）
- **问题**：
  - 动力学中使用 `h_t = h_{t-1} + dt × Δ`，此处 dt 被当作积分步长（微积分 dt）
  - 时间衰减中 Δt 是真正的对话间隔物理时间
  - 若传实际时间给 dt，刺激驱动会被放大成千上万倍溢出
  - 若 dt=1.0，则刺激响应与间隔时长无关，物理错误
- **修正思路**：刺激驱动不应乘 Δt（心理冲击是毫秒级即时响应），仅衰减乘 Δt

### #3 表面状态无 Setpoint（基线回归）
- **位置**：`state_engine/_surface.py:78-91`
- **问题**：长时间间隔时 s → raw（内心投影），而非归零。h 和 r 都有人格基线，s 没有
- **场景**：独处一周后内心已平静，但表面残留 Vulnerability/Longing 映射
- **修正思路**：为 s 引入人格基线（默认中性），Δt→∞ 时 s → neutral

### #4 防御剖面操作不消耗系统资源
- **位置**：`state_engine/_defenses.py:117-131`
- **问题**：高 a（内心翻涌）+ 高 d（表面压抑）在 s=0（面无表情）时，反馈 Δh=0，系统认为无能耗
- **场景**："面无表情但内心惊涛骇浪"——现实中消耗巨大，模型中零成本
- **修正思路**：防御剖面本身应产生能量成本（独立于表面反馈通路）

---

## 🟠 P1 — 心理学效度缺陷（行为与真实情绪/认知规律不符）

### #5 管线顺序违反拉扎勒斯认知评价理论
- **位置**：`state_engine/_pipeline.py:94-112`（当前顺序：反馈→防御→动力学→表面）
- **问题**：反馈先于防御，但认知评价（Appraisal）始终优先于生理反馈。人先判断"有没有威胁"（防御），然后才允许面部反馈信号调情绪
- **场景**：强扯微笑时，若防御判断"社交危险"，微笑的反馈信号根本来不及让你变开心
- **修正思路**：顺序应为 ①防御→④反馈→②动力学，或 ①→②→④→③（反馈在动力学之后）

### #6 表达成本倒置（压抑 < 表达）
- **位置**：`state_engine/_surface_weights.py:218-224`（SURFACE_FEEDBACK_MATRIX ③）
- **问题**：当前 EXPRESSIVENESS→ENERGY -0.06（表达消耗大），RESTRAINT→MENTAL_FATIGUE +0.04（压抑消耗小）
- **心理学依据**：Richards & Gross (2000) 实验证明压抑（Suppression）比表达消耗大得多，因需持续动用前额叶抑制冲动
- **场景**：强忍怒火一天→虚脱；开怀大笑一天→只是脸酸
- **修正思路**：交换量级，压抑成本应数倍于表达成本

### #7 Intimacy → TrustBond 负反馈违反社会渗透理论
- **位置**：`state_engine/_dynamics_weights.py:192-195`
- **问题**：Intimacy → TrustBond: -0.01，亲密增加时信任微降
- **心理学依据**：Altman & Taylor (1973) 社会渗透理论中，亲密与信任是共生关系。健康依恋中亲密只会加强信任
- **场景**：伴侣越交心→信任越牢不可破。该 -0.01 是为数学稳定杜撰的心理谬误
- **修正思路**：移除负向边（或转为 PTSD 模式特例）

---

> **以下 P1 项已移入 `md/THERMODYNAMIC_AFFECTIVE_MODEL.md`，待 v2.0 非线性阶段设计：**
> - #8 压力→易怒倒U型（Yerkes-Dodson）
> - #9 状态依赖耦合（交叉项）
> - #10 防御非线性放大
> - #11 关系动力学振荡（二阶项）
> - #12 表面相变与滞后回线

## 🟡 P2 — 架构增强

- [ ] **权重生成化 Phase 2**：从"文档化硬编码"升级到"生成化权重（Generator 对象，修改原理参数自动重计算）"。
- [ ] **记忆系统集成**：`memory_inject_node` + `memory_summery_node` 接入 `graph/_builder.py`。
- [ ] **特质演化（重新设计）**：不再作为独立子系统，而是由记忆库导出的函数。见 `md/TRAIT_FROM_MEMORY.md`。
- [ ] **刺激维度扩展**：增加 `anticipation`、`guilt`、`disappointment`。
- [ ] **UserModel / Theory of Mind**：独立用户心理剖面。
- [ ] **FastAPI 服务化**：完善 `main.py`。
- [ ] **双速缓冲区（已实现）**

## P3 — 代码质量

- [x] **速度增益 (8, 2) 矩阵化**：删除 `INTERNAL_SPEED_CLASS`(索引数组) + `INTERNAL_SPEED_GAIN`(查找表) 两层间接，改用 `INTERNAL_SPEED_MATRIX` (8, 2) WeightMapper。每维独立正/负值增益，实现情感不对称（Fading Affect Bias）。去掉 `state.py` 中的 `SPEED_FAST/MEDIUM/SLOW`、`SPEED_LABELS`、`INTERNAL_SPEED_CLASS`。✅

- [x] **β 常数化（方案 B, 2026-06-24）**：防御不再调制 β_stim。HYPER_BETA_GAIN 和 DEACT_SUPPRESSION_RATIO 已移除，β = BETA_BASE（常量 0.05）。防御路径重新分配：hyper 仅放大 inner_stimuli（感受强度），deact 仅压抑 outer_stimuli（表达压抑），β 回归纯架构速率常数。因果链从"乘法级联"变为"一条直线"。

---

## 已归档（见 md/ROADMAP.md）

- 权重外部化 Phase 1 → 已完成（WeightMapper 替代裸数值）
- 内部驱力系统 → 见 `md/INTERNAL_DRIVE_SYSTEM.md`
