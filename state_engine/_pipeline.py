"""State Engine Pipeline —— 4 步管线编排。

① 防御剖面 (deactivation / hyperactivation) → inner/outer 刺激
② 残差动力学 → 内部 + 关系状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）
③ 表面投影 → 可观测表达（带惯性混合 + 时间衰减）
④ 表面→内部反馈 → 情绪失调成本 + 面部反馈 + 表达消耗 + 压抑成本

稳态恢复（拉到人格基线）由时间衰减（_decay.py）在对话间隔中处理，
不参与每轮动态。

防御剖面基于 Bowlby (1980) 依恋防御二分法:
  - Deactivation (去激活):  高回避 → 削减外在表达
  - Hyperactivation (过度激活): 高焦虑 → 放大内心感受
二者独立，形成 4 种防御模式: 铁壁、玻璃心、真淡定、纸墙。
"""

from typing import Optional
import numpy as np
from state import ST_SIZE, StimulusMetadata
from ._defenses import compute_defense_profiles, apply_defenses, compute_defense_metabolic_cost
from ._dynamics import (
    update_internal_state,
    update_relationship_state,
    compute_setpoint,
    compute_rel_setpoint,
)
from ._dynamics_weights import INTERNAL_NOISE_SIGMA
from ._surface import project_surface, compute_surface_feedback
from ._utils import soft_clamp

# Langevin 噪声 RNG（情感动力学扩散项）
# 每轮每维独立噪声打破维度同步，模拟情感随机性
_NOISE_RNG = np.random.default_rng(42)


def initialize_all(traits: np.ndarray) -> dict:
    """首次运行：用 Traits 初始化所有状态层。"""
    internal = compute_setpoint(traits)
    relationship = compute_rel_setpoint(traits)
    outer_zero = np.zeros(ST_SIZE, dtype=np.float64)
    surface = project_surface(internal, relationship, outer_zero, prev_surface=None)

    return {
        "internal_state": internal,
        "relationship_state": relationship,
        "surface_state": surface,
    }


def update_all(
    current_internal: Optional[np.ndarray],
    current_relationship: Optional[np.ndarray],
    traits: np.ndarray,
    stimuli: np.ndarray,
    prev_surface: Optional[np.ndarray] = None,
    stimulus_metadata: Optional[StimulusMetadata] = None,
    delta_hours: float = 0.0,
    noise_sigma: Optional[float] = None,
    skip_defenses: bool = False,
) -> dict:
    """State Engine 主入口：4 步管线（反馈延迟版，无关系态 buffer）。

    关系态现在每轮更新（buffer 已移除），通过极低的 α_rel / β_rel
    实现天然慢速动力学，无需人工冻结。

    步骤:
      ① 防御剖面 → (inner_stimuli, outer_stimuli)
         deactivation 控制 outer 削减，hyperactivation 控制 inner 放大
      ④ 表面→内部反馈（延迟：上一轮 surface → 本轮 internal 调制）
         情绪失调成本 + 面部反馈 + 表达消耗（trace 量级）
      ② 残差动力学（内部 + 关系，使用反馈调制后的 internal）
      ③ 表面投影（带惯性混合 + 时间衰减）

    Args:
        current_internal: 当前内部状态 h_{t-1} (8,) 或 None
        current_relationship: 当前关系状态 r_{t-1} (3,) 或 None
        traits: 人格特质 (10,)
        stimuli: 原始心理刺激 (7,)
        prev_surface: 前一帧表面状态 (7,)，None 表示首帧
        stimulus_metadata: 刺激元属性（约束②），含置信度/来源/衰减因子
        delta_hours: 自上次更新以来的时间（小时），用于 surface 惯性衰减

    Returns:
        {"internal_state": (8,), "relationship_state": (3,), "surface_state": (7,)}
    """
    if current_internal is None:
        return initialize_all(traits)

    # 约束②：应用刺激元属性 — 置信度缩放 + missing 维度清零
    if stimulus_metadata is not None:
        stimuli = stimuli * stimulus_metadata.confidence
        missing_mask = stimulus_metadata.source == 3
        stimuli[missing_mask] = 0.0

    # ① 防御剖面 → inner / outer（先认知评价）
    # 拉扎勒斯认知评价理论（Lazarus & Folkman, 1984）：大脑先判断"有没有威胁"，
    # 然后才整合生理反馈信号。当前顺序：防御先于反馈。
    if skip_defenses:
        # 实验条件：跳过 Bowlby 防御剖面，原始刺激直接进入动力学
        inner_stimuli = stimuli.copy()
        outer_stimuli = stimuli.copy()
        profiles = None
    else:
        profiles = compute_defense_profiles(traits, current_relationship, current_internal)
        inner_stimuli, outer_stimuli = apply_defenses(stimuli, profiles)

    # ④ 表面→内部反馈（延迟调制：上一轮 surface[t-1] 影响本轮 internal[t-1]）
    # 在防御评价之后整合——面部反馈信号（"先笑→然后感觉变好"）作为辅助增益回路，
    # 微调已评估过的内部状态。所有权重为 trace 量级 (0.01–0.10)。
    # 修复 2026-06-25：从 ④→①→② 改为 ①→④→②，符合评价优先于生理信号的因果时序。
    if prev_surface is not None:
        feedback = compute_surface_feedback(prev_surface, current_internal)
        current_internal = soft_clamp(current_internal + feedback, -1.0, 1.0)

    # 防御代谢成本（#4 修复）：高 a/d 时的独立能耗
    # 即便 s=0（表面平静），内心翻涌 + 压抑仍消耗能量。
    # 注意：防御成本应用在 feedback 之后、动力学之前，
    # 这样能耗后的内部状态进入本轮动力学积分。
    if profiles is not None:
        defense_cost = compute_defense_metabolic_cost(profiles, traits)
        current_internal = soft_clamp(current_internal + defense_cost, -1.0, 1.0)

    # ② 残差动力学（使用反馈调制 + 防御成本后的 current_internal）
    # dt=1.0 代表一轮认知周期。刺激跳跃不乘 dt（#2 修复），
    # 只有耦合漂移乘 dt。物理时间衰减由 _decay.py 处理。
    new_internal = update_internal_state(
        current_internal, inner_stimuli, traits, current_relationship, profiles,
        dt=1.0,
    )
    new_relationship = update_relationship_state(
        current_relationship, inner_stimuli, traits,
        dt=1.0,
        current_internal=new_internal,
    )

    # Langevin 噪声：每维独立高斯扩散项，打破完美同步
    # 心理学基础：O-U 过程的情感建模标准做法 (Oravecz et al., 2009)
    sigma = noise_sigma if noise_sigma is not None else INTERNAL_NOISE_SIGMA
    if sigma > 0:
        noise = _NOISE_RNG.normal(0, sigma, size=new_internal.shape)
        new_internal = soft_clamp(new_internal + noise, -1.0, 1.0)

    # ③ 表面投影（带惯性混合 + 时间衰减）
    surface = project_surface(
        new_internal, new_relationship, outer_stimuli, prev_surface,
        delta_hours=delta_hours,
    )

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "surface_state": surface,
    }
