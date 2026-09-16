# Luna — 混合架构 AI 角色扮演引擎：情感因果骨架 + LLM 表达层

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.13+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="license">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status">
</p>

**Luna** 是一个基于 LangGraph 的 AI 角色扮演引擎，采用**混合架构**：数学引擎提供跨时间一致的情感因果骨架（External Causal Skeleton），LLM 负责情感表达的自然语言生成。两者互补而非竞争。

角色原型为《崩坏3》「月下誓约·予爱以心」——一个带有依恋焦虑、高自尊、暗中在意但嘴上不说的吸血鬼少女。

## 核心特性

- **混合架构定位**：数学引擎不是情感模拟器，而是 LLM 的"剧本"——提供跨时间的因果一致性和参数化的行为控制面板。详见 [`md/01-vision/`](md/01-vision/) 的叙事转向分析
- **三层心理状态模型**：Internal State（真实感受）→ Relationship State（对用户的关系感知）→ Surface State（外显表达），三者解耦
- **"口是心非"的计算实现**：基于 Bowlby (1980) 依恋防御二分法——Deactivation（去激活）压抑外在表达，Hyperactivation（过度激活）放大内心感受。高 Pride + 高 Attachment Anxiety 的角色内心翻江倒海但表面波澜不惊
- **残差动力学驱动**：所有状态更新使用 `h_t = h_{t-1} + dt · (耦合 + 刺激 + 稳态)` 形式，没有 if-else 决策树。耦合关系通过显式命名规则定义，每条附带心理学注释
- **人格稳定但状态可变**：10 维 Traits 决定角色的"底色"，通过调制速率参数（α, β, γ）而非直接叠加来影响状态演化
- **时间感知衰减**：基于真实时间间隔（Δt 小时）的指数衰减，不同维度有独立的衰减速率和人格化目标基线
- **LLM 感知层**：自动从用户输入中提取 7 维心理刺激（被抛弃感、被认可感、亲密靠近、冲突、依赖、调侃、情感重量），3 轮重试 + JSON 验证
- **约束宪法 —— 防止黑盒化**：所有矩阵必须通过 11 条严格约束才能存在（详见下文"状态引擎约束框架"）
- **SQLite 持久化**：通过 LangGraph Checkpointer，状态跨会话保持

## 环境要求

- **Python** ≥ 3.13
- **Ollama** 运行中，已拉取 `nomic-embed-text`（用于记忆系统的嵌入检索）
- **DeepSeek API Key**（用于主对话 LLM 和感知模型）
- **uv** 包管理器（推荐）或 pip

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Luna

# 2. 安装依赖（uv）
uv sync

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 确保 Ollama 运行并拉取嵌入模型
ollama pull nomic-embed-text
```

## 快速开始

```bash
# 启动交互式对话（TUI）
uv run python agent.py
```

首次运行会自动创建 `db/Luna.db`（SQLite 状态持久化）。在 TUI 中输入消息即可与角色对话。

## 架构概览

```
START → inject_system → perception → state_engine → state_formatter → llm → END
                              │ error=True → END（跳过 state_engine 和 formatter）
```

| 节点 | 职责 | 模型 |
|------|------|------|
| `inject_system` | 首次运行时注入角色人设 + Traits（仅执行一次） | — |
| `perception` | 从用户输入提取 7 维心理刺激（3 轮重试 + 递增强调） | DeepSeek |
| `state_engine` | **4 步管道**：① Bowlby 防御剖面 → ② 表面→内部反馈 → ③ 残差动力学(内部+关系) → ④ 表面投影 | —（纯数学） |
| `state_formatter` | 数值状态向量 → 中文"导演笔记"（离散 5 级阈值） | —（纯规则） |
| `llm` | 注入角色人设 + 状态描述，生成回复 | DeepSeek |

## 状态引擎（4 步管道）

> 旧版定位：模拟情感 → 新版定位：外部因果骨架。详见 [`md/01-vision/HYBRID_ARCHITECTURE_STRATEGY.md`](md/01-vision/HYBRID_ARCHITECTURE_STRATEGY.md)。

旧版 7 层管线（含门控系统）已重构为基于 Bowlby 依恋防御理论的 4 步管道（反馈-延迟顺序排列）：

```
① Defense Profiles（防御剖面）
   compute_defense_profiles() → profiles (2, 7)
   apply_defenses() → inner_stimuli, outer_stimuli
   
   基于 Bowlby (1980) 的依恋防御二分法：
   - Deactivation（去激活）：高回避→压抑外在表达
   - Hyperactivation（过度激活）：高焦虑→放大内心感受

② Surface→Internal Feedback（表面反馈，延迟）
   compute_surface_feedback(prev_surface, current_internal) → delta_internal (8,)
   
   上轮表面状态 → 本轮内部状态修正（延迟效应）
   - Emotional Labor：压抑表达消耗心理能量
   - Facial Feedback：表情反作用于感受
   - Expression Cost：伪装的精神成本

③ Residual Dynamics（残差动力学）
   update_internal_state() → new_internal (8,)
   update_relationship_state() → new_relationship (3,) [每 3 轮]

   公式: h_t = h_{t-1} + dt · (α·耦合 + β·刺激 + SSM 速度门控)
   - 耦合通过显式命名规则定义（每行附带心理学注释）
   - 防御剖面调制 β 速率
   - SSM 速度门控：FAST(1.0)=stress/irritation, MEDIUM(0.6)=energy/social_battery/mental_fatigue, SLOW(0.5)=loneliness/insecurity/longing
   - 关系态使用 β_rel 减半 + 缓冲轮次实现双速近似

④ Surface Projection（表面投影）
   project_surface() → surface_state (7,)
   
   内部状态 + 关系状态 + 外显刺激 → 可观测的 7 维表达
   - 惯性混合: s(t) = α·raw + (1-α)·s(t-1)
   - α 由 stress/energy 动态调制
```

对比旧版门控系统（已被移除）：

对比旧版门控系统（已被移除）：

| 旧版 | 新版 | 原因 |
|------|------|------|
| 4 门并行（suppression/vulnerability/attachment/leakage） | 2 维防御剖面（deactivation/hyperactivation） | Bowlby 理论依据 + PCA 验证有效维度从 ~1.5 提升至 ~4+ |
| 全局标量调制 | 刺激特异性逐维权重 | 避免维度同步漂移塌缩 |
| 独立门控逻辑 | 统一的 `compute_defense_profiles()` | 简化认知负担 |

## 状态引擎约束框架

> Luna 的约束宪法——防止系统退化为不可解释的黑盒。

状态引擎的所有参数和矩阵必须通过以下约束才能存在。约束在 `state_engine/_validator.py` 的 `ConstraintRegistry` 中集中执行，每次 `build_matrix()` 调用时自动全量检查，失败则抛出 `ConstraintViolationError`。

### 语义架构层（保证信息流的意图透明）

| # | 约束 | 含义 |
|---|------|------|
| ① | **Trait 不直接影响状态** | Trait 只调制速率参数（α, β, γ），不参与状态更新方程的主项 |
| ② | **刺激携带元属性** | 每维刺激携带置信度、来源编码、衰减调节因子 |
| ④ | **禁止跨层直接连线** | Surface 只看 Relationship State（不跨层读 Internal 或 Traits） |
| ⑤ | **语义映射层** | 禁止裸数值 `B[i,j] = 0.25`——所有参数通过 `WeightMapper` 声明语义关系 |

### 数学保证层（保证系统的结构透明）

| # | 约束 | 含义 |
|---|------|------|
| ③ | **矩阵低秩** | 有效秩远小于名义维度，耦合由少量潜在因子驱动 |
| ⑥ | **正交稀疏** | 密度 ≤ 30%，行 Gram 矩阵非对角元素 < 0.3 |
| ⑦ | **谱半径 ρ < 0.95** | 系统是收缩映射，不会发散 |
| ⑨ | **全局雅可比稀疏** | 组合矩阵的传播路径数 ≤ 5/对——这是最重要的约束，防止"单个矩阵都通过，乘在一起变黑盒" |

### 流程透明层（保证参数的历史透明）

| # | 约束 | 含义 |
|---|------|------|
| ⑧ | **参数审计** | 每个参数有 provenance（来源、依据、审查日期），无 `origin=legacy` 参数 |
| ⑩ | **刺激正交性** | 7 个 ST_* 维度每对最多共享 1 个激活场景 |
| ⑪ | **状态格式化连续性** | 9 区连续投影替代离散 5 级阈值 |

> 详细定义见 [`md/02-architecture/STATE_ENGINE_CONSTRAINTS.md`](md/02-architecture/STATE_ENGINE_CONSTRAINTS.md)。

## 项目结构

```
Luna/
├── agent.py                 # TUI 交互入口
├── nodes.py                 # LangGraph 节点函数（5 个活跃 + 2 个存根）
├── perception.py            # 感知层：心理刺激提取 + JSON 验证 + 3 轮重试
├── state.py                 # 状态类型定义（向量索引常量 + TypedDict + 默认值）
├── state_formatter.py       # 状态格式化：数值 → "导演笔记"
├── llm.py                   # LLM 模型初始化（DeepSeek）
├── config.py                # 运行时配置（感知重试参数等）
├── main.py                  # FastAPI 入口（开发中）
├── graph/                   # LangGraph 图定义与路由
│   ├── _builder.py          #   图构造
│   ├── _routing.py          #   条件路由
│   └── __init__.py          #   编译后的导出
├── state_engine/            # 3 步心理状态管道
│   ├── _pipeline.py         #   管道编排（update_all/initialize_all）
│   ├── _defenses.py         #   防御剖面（Bowlby 二分法）← 替代旧门控系统
│   ├── _dynamics.py         #   内部 & 关系动力系统（残差形式）
│   ├── _decay.py            #   时间感知衰减（真实 Δt 驱动）
│   ├── _surface.py          #   表面投影（Internal + Relationship → 表达）
│   ├── _matrices.py         #   耦合矩阵工厂（将被 WeightMapper + Mapper 替代）
│   ├── _validator.py        #   [规划中] 约束检查注册表 + 全局雅可比验证
│   ├── _mapper.py           #   [规划中] 语义映射层（WeightMapper）
│   └── _utils.py            #   数值工具（soft_clamp, sigmoid）
├── prompts/                 # Prompt 数据
│   ├── character.py         #   角色人设 SYSTEM_PROMPT
│   ├── perception.py        #   感知系统 prompt
│   └── memory_summery.py    #   记忆系统 prompt（部分实现）
├── memory.py                # 记忆系统（向量/嵌入/混合检索）
├── tests/                   # 8 个测试文件 + conftest.py
├── db/                      # SQLite 持久化
└── md/                      # 设计文档索引与分类
    ├── index.md                        # 文档入口 & 阅读路线
    ├── 01-vision/                      # 战略定位 — 新叙事与批判性反思
    │   ├── CRITIQUE.md                 #   首席批判官报告
    │   ├── HYBRID_ARCHITECTURE_STRATEGY.md  # 混合架构路线转向
    │   └── EXPERIMENT_A_B.md           #   A/B 验证方案
    ├── 02-architecture/                # 架构设计 — 核心管线与约束
    │   ├── ARCHITECTURE.md             #   系统总览
    │   ├── STATE_ENGINE_MATH.md        #   数学形式化
    │   ├── STATE_ENGINE_CONSTRAINTS.md #   约束宪法全文
    │   ├── DEFENSE_PROFILE_METHODOLOGY.md  # Bowlby 防御剖面设计
    │   ├── SURFACE_PROJECTION_RESEARCH.md  # 表面投影层设计
    │   └── WEIGHT_MAPPER_IMPLEMENTATION.md # 语义映射层实现
    ├── 03-research/                    # 研究报告 — 学术调研与理论依据
    │   ├── AFFECTIVE_GEOMETRY_RESEARCH.md  # 情感几何数学表示
    │   ├── SPARSE_ANTAGONIST_ANALYSIS.md   # 稀疏耦合与拮抗对
    │   ├── DUAL_TIMESCALE_SSM_RESEARCH.md  # 双速 SSM 调研
    │   └── DEEP_RESEARCH_REPORT.md     #   综合可行性评估
    └── 04-plans/                       # 未来方向 — 路线图与待设计方案
        ├── ROADMAP.md                  #   路线图 & 执行计划
        ├── STATE_ENGINE_TEST_REPORT.md #   测试报告
        ├── THERMODYNAMIC_AFFECTIVE_MODEL.md # 热力学情感模型 v2
        ├── INTERNAL_DRIVE_SYSTEM.md    #   内部驱力系统
        ├── MEMORY_SYSTEM.md            #   记忆系统设计
        ├── TRAIT_FROM_MEMORY.md        #   Traits 记忆驱动更新
        └── lunar_memory_drive_design.md #  记忆+驱力联合设计
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph |
| 主对话 + 感知模型 | DeepSeek (`deepseek-v4-pro`) |
| 嵌入模型（记忆检索） | Ollama (`nomic-embed-text`) |
| 状态向量 | NumPy `ndarray` |
| 持久化 | SQLite (`langgraph-checkpoint-sqlite`) + JSON |
| 包管理 | uv (Python 3.13) |
| 约束执行 | `ConstraintRegistry`（权重参数全生命周期审计）|

## 与常规 prompt-based 角色扮演的对比

| 维度 | 常规方案 | Luna |
|------|---------|-------|
| 人设维持 | 依赖 system prompt + 模型 adherence | 数学模型保证（Traits + 动力学） |
| 情绪连贯性 | 高 prompt 长度下劣化 | 状态向量自然连续演化 |
| "口是心非" | 需要显式描述 | 防御机制 + Surface 投影自动产生 |
| 长期行为一致性 | 随着对话长度指数劣化 | 时间衰减 + Setpoint 约束稳定域 |
| 参数可解释性 | 黑盒（prompt 中的隐性 bias） | 11 条约束保证每个参数可追溯 |
| 可控性 | prompt 工程（试错） | 调 Traits/矩阵系数，行为变化可推理 |

## 已知限制

- 🔴 记忆系统虽作为库完整实现（三路检索 + MemoryStore），但未接入 LangGraph 管道（`memory_inject_node` / `memory_summery_node` 是存根）
- 🔴 特质永远不变（`DEFAULT_TRAITS` 静态常量），角色"长不大"
- 🔴 无目标/意图系统，角色纯被动响应，缺乏内部驱力
- 🟡 刺激向量仅 7 类关系性信号，缺少 Anticipation / Guilt / Disappointment 等关键情绪类别
- 🟡 8 维内部 + 3 维关系 + 7 维表面的有效自由度约 6（PCA 验证 73%+ 方差由前 2 主成分解释）
- 🟢 感知层的刺激提取依赖 DeepSeek 而非 Ollama，存在外部 API 依赖延迟

## 贡献

本项目处于早期开发阶段。欢迎提 Issue 或 PR。

改进方向和已知问题见 [md/04-plans/ROADMAP.md](md/04-plans/ROADMAP.md)。约束框架的完整定义见 [`md/02-architecture/STATE_ENGINE_CONSTRAINTS.md`](md/02-architecture/STATE_ENGINE_CONSTRAINTS.md)。设计文档索引见 [`md/index.md`](md/index.md)。

## 许可证

MIT © 2025 Luna Dev
