# Luna 设计文档索引

> **定位**：Luna 的数学引擎不是情感模拟器，而是**情感的外部因果骨架（External Causal Skeleton）**——为 LLM 的生成提供跨时间一致的情感因果结构。
>
> **路线**：放弃"数学引擎独立模拟情感"的叙事，转向"数学引擎为 LLM 提供情感的外部因果骨架"。
>
> 详见 [`01-vision/HYBRID_ARCHITECTURE_STRATEGY.md`](01-vision/HYBRID_ARCHITECTURE_STRATEGY.md)。

---

## 📂 文档结构

```
md/
├── 01-vision/           # 战略定位 — 新叙事、批判性反思、验证方案
├── 02-architecture/     # 架构设计 — 核心管线、数学模型、约束宪法
├── 03-research/         # 研究报告 — 跨学科学术调研与理论依据
└── 04-plans/            # 未来方向 — 路线图、实验报告、待设计方案
```

---

## 01 — 战略定位（Vision）

| 文档 | 摘要 |
|------|------|
| [CRITIQUE.md](01-vision/CRITIQUE.md) | 首席批判官报告：三条致命假设与项目存续条件。**必须首先阅读。** 论证了纯数学情感模拟路线的不可通性，提出了混合架构的必要性 |
| [HYBRID_ARCHITECTURE_STRATEGY.md](01-vision/HYBRID_ARCHITECTURE_STRATEGY.md) | 路线转向声明：从"数学引擎独立模拟情感" → "数学引擎为 LLM 提供外部因果骨架"。基于学术证据定位混合架构 |
| [EXPERIMENT_A_B.md](01-vision/EXPERIMENT_A_B.md) | A/B 实验方案：量化数学引擎的边际价值 |
| [EXPERIMENT_REPORT.md](01-vision/EXPERIMENT_REPORT.md) | **A/B 多变量实验综合报告**：5条件对比的定量+定性结论（2026-07-14） |

## 02 — 架构设计（Architecture）

| 文档 | 摘要 |
|------|------|
| [ARCHITECTURE.md](02-architecture/ARCHITECTURE.md) | 系统总览：5 节点 LangGraph 管线、状态空间定义、约束框架概述 |
| [STATE_ENGINE_MATH.md](02-architecture/STATE_ENGINE_MATH.md) | 状态引擎的完整数学形式化（符号系统、更新方程、耦合结构） |
| [STATE_ENGINE_CONSTRAINTS.md](02-architecture/STATE_ENGINE_CONSTRAINTS.md) | 约束宪法全文：11 条约束的推导依据与验证方法 |
| [DEFENSE_PROFILE_METHODOLOGY.md](02-architecture/DEFENSE_PROFILE_METHODOLOGY.md) | Bowlby 依恋防御剖面的心理学依据与参数设计 |
| [SURFACE_PROJECTION_RESEARCH.md](02-architecture/SURFACE_PROJECTION_RESEARCH.md) | 表面投影层的跨学科调研（30+ 心理学/计算框架综述） |
| [WEIGHT_MAPPER_IMPLEMENTATION.md](02-architecture/WEIGHT_MAPPER_IMPLEMENTATION.md) | 语义映射层的实现模式：WeightMapper / WeightVector / LinearMapping |

## 03 — 研究报告（Research）

| 文档 | 摘要 |
|------|------|
| [AFFECTIVE_GEOMETRY_RESEARCH.md](03-research/AFFECTIVE_GEOMETRY_RESEARCH.md) | 情感几何的数学表示研究：PAD 空间、Schwartz 价值环、OCEAN 人格的统一 |
| [SPARSE_ANTAGONIST_ANALYSIS.md](03-research/SPARSE_ANTAGONIST_ANALYSIS.md) | 稀疏耦合与拮抗对方案分析：解决耦合矩阵过密导致的维度同步漂移 |
| [DUAL_TIMESCALE_SSM_RESEARCH.md](03-research/DUAL_TIMESCALE_SSM_RESEARCH.md) | 双速 SSM 调研：为状态引擎的快慢双速动力学提供神经科学/计算建模依据 |
| [DEEP_RESEARCH_REPORT.md](03-research/DEEP_RESEARCH_REPORT.md) | 综合调研报告：混合架构在 2024-2025 学术共识下的可行性评估 |

## 04 — 未来方向（Plans）

| 文档 | 摘要 |
|------|------|
| [ROADMAP.md](04-plans/ROADMAP.md) | 路线图与执行计划：问题清单、修复优先级、四象限战略 |
| [STATE_ENGINE_TEST_REPORT.md](04-plans/STATE_ENGINE_TEST_REPORT.md) | 状态引擎测试报告：约束验证、蒙特卡洛仿真、心理场景测试 |
| [THERMODYNAMIC_AFFECTIVE_MODEL.md](04-plans/THERMODYNAMIC_AFFECTIVE_MODEL.md) | 热力学情感模型 v2.0 设计：能量会计、非稳态负荷、耗散结构 |
| [INTERNAL_DRIVE_SYSTEM.md](04-plans/INTERNAL_DRIVE_SYSTEM.md) | 内部驱力系统设计：让状态引擎从纯被动响应变为自驱动情感系统 |
| [MEMORY_SYSTEM.md](04-plans/MEMORY_SYSTEM.md) | 记忆系统设计：向量/嵌入/混合三路检索 |
| [TRAIT_FROM_MEMORY.md](04-plans/TRAIT_FROM_MEMORY.md) | 记忆驱动的人格特质更新：traits(t) = f(到 t 为止的所有记忆) |
| [lunar_memory_drive_design.md](04-plans/lunar_memory_drive_design.md) | 记忆 + 内部驱力联合设计方案（设计评审稿） |

---

## 阅读顺序建议

### 🆕 新读者优先读（理解项目定位）
1. `01-vision/CRITIQUE.md` — 先看批判，理解为什么纯数学模拟路线有问题
2. `01-vision/HYBRID_ARCHITECTURE_STRATEGY.md` — 再看转向后的混合架构战略

### 🔧 要理解代码（看架构）
3. `02-architecture/ARCHITECTURE.md` — 系统总览
4. `02-architecture/STATE_ENGINE_MATH.md` — 数学形式化

### 📊 要验证设计合理性（看研究）
5. `03-research/DUAL_TIMESCALE_SSM_RESEARCH.md` — SSM 快慢双速依据
6. `02-architecture/STATE_ENGINE_CONSTRAINTS.md` — 约束宪法

### 🚀 要看下一步（看规划）
7. `04-plans/ROADMAP.md` — 路线图
8. `04-plans/EXPERIMENT_A_B.md` — 验证实验

---

> **注意**：文档中引用的代码路径（如 `state_engine/_defenses.py`）均相对于项目根目录。所有文档遵循 [语义映射层约束](../state_engine/_validator.py) 确保参数可审计。
