"""State Dynamics —— 残差式状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）。

核心公式:
  h_t = h_{t-1} + Δt · (α·Δ_coupling + Δ_stimulus_modulated)

两个参数 + 一个逐维度调制:
  α — 跨维度耦合速率 (traits + relationship)
  β — 刺激接受速率，逐刺激维度 (defense profiles: hyper↑每维, deact↓每维)

稳态恢复（拉到人格基线）已移除——职责转移到 _decay.py 的时间衰减。

约束合规:
  - SELF_DECAY / REL_SELF_DECAY / DECAY_TARGETS → WeightVector in _dynamics_weights.py
  - 耦合系数 (internal/relationship/cross-scale) → WeightMapper in _dynamics_weights.py
  - 所有参数带完整 provenance
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
)
from ._utils import soft_clamp
from ._matrices import INPUT_INFLUENCE_B, REL_INPUT_INFLUENCE_B
from ._dynamics_weights import (
    SELF_DECAY, DECAY_TARGETS, REL_SELF_DECAY,
    INTERNAL_COUPLING, RELATIONSHIP_COUPLING, CROSS_SCALE_COUPLING,
    ALPHA_MAPPER, ALPHA_REL_MAPPER, BETA_REL_MAPPER,
    BETA_BASE,
    SETPOINT_MAPPER, REL_SETPOINT_MAPPER,
    INTERNAL_SPEED_MATRIX,
)


def compute_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的内部情绪稳态基线。

    通过 SETPOINT_MAPPER(LinearMapping) 计算:
      sp = DEFAULT_INTERNAL + traits @ W  (W 带 provenance)
    """
    return np.clip(SETPOINT_MAPPER.compute(traits), -0.9, 0.9)


def compute_rel_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的关系稳态基线（3 维版）。

    通过 REL_SETPOINT_MAPPER(LinearMapping) 计算:
      sp = DEFAULT_RELATIONSHIP + traits @ W  (W 带 provenance)
    """
    return np.clip(REL_SETPOINT_MAPPER.compute(traits), -0.96, 0.96)


def update_internal_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    profiles: np.ndarray,  # (2, 7) defense profiles
    dt: float = 1.0,  # 积分步长；每轮 = 1.0（调用协定，传真实 Δt 会溢出刺激）
) -> np.ndarray:
    """残差式内部状态更新（跳-扩散混合动力学）。

    核心分离:
      h_t = h_{t-1} + dt × 漂移 + 跳跃

    漂移（drift = α·Δ_coupling）— 率过程，乘 dt。
      跨维度耦合 + 自阻尼是时间连续过程，积累量正比于间隔时长。
      如：压力持续积累烦躁。

    跳跃（jump = EnergyGate(β·σ @ B)）— 点过程，不乘 dt。
      刺激冲击是毫秒级即时响应，与间隔时长无关。
      如：一句话扎心瞬间暴怒。

    修复 2026-06-25 (#2)：旧公式 h_t = h' + dt·(α·Δcpl + Δstim)
      将刺激冲击也乘以 dt，导致 Δt 小时刺激无效、Δt 大时溢出。
      现在是唯一正确的跳-扩散分离。

    修复 2026-06-25 (#0a)：跳跃受能量门控
      EnergyGate = (1 + h[Energy]) / 2，低电量时情绪响应受限。

    Args:
        current: 当前内部状态 h_{t-1} (8,)
        inner_stimuli: 防御过滤后的"里"刺激 (7,)
        traits: 人格特质 (10,)
        relationship: 关系状态 (3,)
        profiles: 防御剖面 (2, 7)
        dt: 积分步长（每轮 = 1.0，不允许省略）

    Returns:
        更新后的内部状态 h_t (8,)
    """
    # ── α: 跨维度耦合速率 ──
    alpha = ALPHA_MAPPER.compute(np.concatenate([traits, relationship]))[0]
    if alpha < 0.05 or alpha > 0.40:
        logger.warning("ALPHA_MAPPER 输出 %.4f 被截断到 [0.05, 0.40]", alpha)
    alpha = soft_clamp(alpha, 0.05, 0.40)

    # ── β: 刺激接受速率（常数，不防御调制）──
    # 方案 B (2026-06-24): 防御只负责 inner/outer 分离,
    # β_stim = BETA_BASE 为纯架构常数（每维 0.05）。
    beta_stim = BETA_BASE  # (7,) 每维 0.05

    # ── ① 漂移：跨维度耦合 + 自阻尼（率过程，需乘 dt）──
    coupling = current @ INTERNAL_COUPLING  # (8,)
    delta_coupling = coupling - SELF_DECAY * (current - DECAY_TARGETS)
    drift = alpha * delta_coupling

    # ── ② 跳跃：刺激驱动（点过程，不乘 dt）+ 能量门控 ──
    # 能量门控 (#0a)：Energy ∈ [-1, 1] → gate ∈ [0, 1]
    # 低电量 = 前额叶葡萄糖耗尽 = 情绪响应被生理性压制
    energy_factor = (1.0 + current[I_ENERGY]) / 2.0  # [-1, 1] → [0, 1]
    energy_factor = np.clip(energy_factor, 0.1, 1.0)  # 保留最低 10% 响应
    modulated_stimuli = beta_stim * inner_stimuli  # (7,)
    jump = energy_factor * (modulated_stimuli @ INPUT_INFLUENCE_B)  # (8,)

    # ── ③ SSM 双速门控：按总变化方向选择增益 ──
    #   delta >= 0 → rising_gain（上升快/慢）
    #   delta < 0  → falling_gain（消退快/慢）
    # 修复 2026-06-25 (#1)：之前误用 sign(current) 而非 sign(delta)
    delta = drift * dt + jump  # 漂移×dt + 跳跃（不乘 dt）
    speed_gains = np.where(delta >= 0, INTERNAL_SPEED_MATRIX[:, 0], INTERNAL_SPEED_MATRIX[:, 1])
    delta *= speed_gains

    return soft_clamp(current + delta, -1.0, 1.0)


def update_relationship_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    dt: float = 1.0,  # 积分步长；每轮 = 1.0
    current_internal: np.ndarray | None = None,
) -> np.ndarray:
    """残差式关系状态更新（跳-扩散混合，时间常数比内部慢 5-10 倍）。

    与 update_internal_state 同构的跳-扩散分离:
      r_t = r_{t-1} + dt × 耦合漂移 + 刺激跳跃

    与内部状态更新同构，但:
      - α_rel 更小（关系变化极慢）
      - β_rel 更小（刺激对关系的影响有缓冲）
      - 可选的 current_internal 参数提供跨尺度耦合（内→关）
    """
    # ── α_rel: 关系跨维度耦合速率 ──
    alpha = ALPHA_REL_MAPPER.compute(np.concatenate([traits, current]))[0]
    if alpha < 0.005 or alpha > 0.08:
        logger.warning("ALPHA_REL_MAPPER 输出 %.4f 被截断到 [0.005, 0.08]", alpha)
    alpha = soft_clamp(alpha, 0.005, 0.08)

    # ── β_rel: 关系刺激接受速率 ──
    beta = BETA_REL_MAPPER.compute(traits)[0]
    if beta < 0.002 or beta > 0.06:
        logger.warning("BETA_REL_MAPPER 输出 %.4f 被截断到 [0.002, 0.06]", beta)
    beta = soft_clamp(beta, 0.002, 0.06)

    # ── 漂移：关系耦合 + 自阻尼 + 跨尺度（率过程，乘 dt）──
    rel_coupling = current @ RELATIONSHIP_COUPLING  # (3,)
    if current_internal is not None:
        cross = current_internal @ CROSS_SCALE_COUPLING  # (8,) @ (8,3) → (3,)
        rel_coupling = rel_coupling + cross
    delta_coupling = rel_coupling - REL_SELF_DECAY * current
    drift = alpha * delta_coupling

    # ── 跳跃：刺激输入（点过程，不乘 dt）──
    delta_stimulus = (inner_stimuli @ REL_INPUT_INFLUENCE_B) * beta

    # ── 更新 ──
    return soft_clamp(current + drift * dt + delta_stimulus, -1.0, 1.0)
