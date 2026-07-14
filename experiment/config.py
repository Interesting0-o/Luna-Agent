"""A/B 实验配置。

控制运行哪些条件、模型选择、日志路径等。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """全局实验配置"""

    # ── 运行控制 ──
    conditions: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    """运行的实验条件列表"""

    scenarios_file: str = ""
    """场景文件路径（空=使用内置场景）"""

    num_seed_turns: int = 3
    """种子对话轮数（所有条件共享）"""

    num_test_turns: int = 2
    """测试轮数（条件分支后）"""

    random_seed: int = 42
    """全局随机种子"""

    # ── 模型 ──
    generation_model: str = "deepseek-v4-pro"
    """回复生成模型"""

    perception_model: str = "qwen2.5:7b"
    """感知模型（Ollama）"""

    state_inference_model: str = "deepseek-v4-pro"
    """条件 B 状态推断模型（与生成用不同实例）"""

    state_inference_temperature: float = 0.3
    """状态推断温度（低=稳定推断）"""

    judge_model: str = "deepseek-v4-pro"
    """LLM-as-Judge 模型（与生成模型不同）"""

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

    collect_raw_responses: bool = True
    """是否收集原始回复文本"""

    # ── 条件 C 的刺激开关 ──
    condition_c_skip_perception: bool = True
    """条件 C 是否完全跳过感知（True=无刺激提取）"""


@dataclass
class ScenarioConfig:
    """单个场景配置"""

    name: str
    """场景名称"""

    category: str
    """场景类别：关系升温/冲突对抗/日常闲聊/情感重负/矛盾信号"""

    traits: list[float]
    """角色人格特质（10 维）"""

    seed_user_inputs: list[str]
    """种子对话的用户输入（3 轮）"""

    test_user_inputs: list[str]
    """测试轮的用户输入（2 轮）"""

    description: str = ""
    """场景描述/期望的情感弧线"""
