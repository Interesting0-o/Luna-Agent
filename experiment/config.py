"""A/B 实验配置 — 扩展版。

支持 5 条件、感知模型选择、时间衰减子实验。
"""

from dataclasses import dataclass, field
from typing import Optional


# ── 条件元数据表 ──
CONDITION_META = {
    "A_full": {
        "label": "A_full",
        "name": "完整数学引擎",
        "description": "当前生产管线：防御→反馈→动力学→表面投影",
    },
    "A_no_def": {
        "label": "A_no_def",
        "name": "引擎无防御",
        "description": "跳过 Bowlby 防御剖面，刺激直接进入动力学",
    },
    "B_scratch": {
        "label": "B_scratch",
        "name": "LLM 自报（无上下文）",
        "description": "LLM 每轮从零推断状态，无上一轮状态信息",
    },
    "B_context": {
        "label": "B_context",
        "name": "LLM 自报（有上下文）",
        "description": "LLM 推断时注入上一轮状态描述作为上下文",
    },
    "C_none": {
        "label": "C_none",
        "name": "无状态基线",
        "description": "无状态注入，仅角色 prompt + 对话历史",
    },
}

DEFAULT_CONDITIONS = ["A_full", "B_scratch", "C_none"]
"""默认运行条件（与旧版 A/B/C 兼容）"""

ALL_CONDITIONS = list(CONDITION_META.keys())
"""全部 5 条件"""


@dataclass
class ExperimentConfig:
    """全局实验配置"""

    # ── 运行控制 ──
    conditions: list[str] = field(default_factory=lambda: DEFAULT_CONDITIONS)
    """运行的实验条件列表（来自 CONDITION_META 的 label）"""

    scenarios_file: str = ""
    """场景文件路径（空=使用内置场景）"""

    num_seed_turns: int = 3
    """种子对话轮数（所有条件共享）"""

    num_test_turns: int = 2
    """测试轮数（条件分支后）"""

    random_seed: int = 42
    """全局随机种子"""

    # ── 感知 ──
    perception_provider: str = "deepseek"
    """感知提供者: "deepseek" | "ollama" """

    # ── 时间衰减 ──
    time_decay_hours: float = 0.0
    """对话间隔时间（小时），0=连续对话"""

    # ── 模型 ──
    generation_model: str = "deepseek-v4-pro"
    """回复生成模型"""

    state_inference_temperature: float = 0.3
    """状态推断温度（低=稳定推断）"""

    judge_model: str = "deepseek-v4-pro"
    """LLM-as-Judge 模型"""

    judge_temperature: float = 0.0
    """评判温度（0=确定性评分）"""

    judge_num_repeats: int = 3
    """每轮评分重复次数（去偏）"""

    # ── 路径 ──
    output_dir: str = "experiment/outputs"
    """结果输出目录"""

    run_tag: str = ""
    """运行标识（自动生成）"""

    # ── 数据收集 ──
    collect_state_trajectories: bool = True
    """是否收集完整状态轨迹"""


def condition_label(code: str) -> str:
    """返回条件的人类可读标签。"""
    meta = CONDITION_META.get(code)
    if meta:
        return f"{code} ({meta['name']})"
    return code


def condition_summary() -> str:
    """返回条件汇总表。"""
    lines = [f"{'代码':<12} {'名称':<20} {'说明':<40}", "-" * 72]
    for code, meta in CONDITION_META.items():
        lines.append(f"{code:<12} {meta['name']:<20} {meta['description']:<40}")
    return "\n".join(lines)
