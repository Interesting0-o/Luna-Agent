"""Defense Profiles —— 秩-1 分解：基线 + 强度 × PC1 方向。

基于 Bowlby (1980) 的依恋防御二分法 + Richardson et al. (2023, 2025) 的实证验证:

  profiles[0, :] — Deactivation (去激活): 削减外在表达
    高回避 → 情感疏离、压抑表达。关联特质: pride↑, avoidance↑, stability↓
    由 suppression + 逆 vulnerability 合并而来。

  profiles[1, :] — Hyperactivation (过度激活): 放大内心感受
    高焦虑 → 放大关系威胁/亲近信号。关联特质: attachment_anxiety↑, jealousy↑

每个剖面是 7 维敏感度向量 ∈ [0, 1]。

═══ 秩-1 架构（2026-06-22 重构）═══

  100,000 随机样本 PCA 审计证实：7 维有效秩仅 ~1.4-1.9。
  故将 ~225 参数的逐维度调制模型降为秩-1 分解:

    profile_raw = BASELINE + INTENSITY × PC1_DIR        # (7,)
    profile     = sigmoid((profile_raw - SHIFT) × SCALE)

  其中 INTENSITY = w·[traits, relationship, internal]   # 标量
  BASELINE 固定为 7 维均值，PC1_DIR 固定为第一主成分方向。

验证:
  去激活:   秩-1 R² = 84.3%（vs 完全模型）
  过度激活: 秩-1 R² = 94.7%（vs 完全模型）

扩展方式: 增加新防御维度（如 boundary）需验证:
  1. 概念判别效度 — 与 deactivation/hyperactivation 不是同一构念
  2. 因子/相关性检查 — |r| < 0.3 才值得独立成维度
  3. 交互机制 — 在 apply_defenses 中有不同于现有维度的数学操作
"""

import numpy as np
from state import ST_SIZE
from ._utils import soft_clamp, _sigmoid
from ._defense_weights import (
    DEACT_BASELINE_RAW, DEACT_PC1_DIR, DEACT_INTENSITY,
    HYPER_BASELINE_RAW, HYPER_PC1_DIR, HYPER_INTENSITY,
    HYPER_STATE_MODULATION,
    DEACT_SIGMOID_SHIFT, DEACT_SIGMOID_SCALE,
    HYPER_SIGMOID_SHIFT, HYPER_SIGMOID_SCALE,
    HYPER_APPLY_GAIN, DEACT_APPLY_GAIN,
)


def compute_defense_profiles(
    traits: np.ndarray,
    relationship: np.ndarray,
    internal: np.ndarray,
) -> np.ndarray:
    """秩-1 防御剖面计算: baseline + intensity × PC1 方向。

    Parameters
    ----------
    traits: (10,)       人格特质
    relationship: (3,)  关系状态
    internal: (8,)      内部状态

    Returns
    -------
    profiles: (2, 7)
        profiles[0, :] — deactivation: 削减外在表达 [0, 1]
        profiles[1, :] — hyperactivation: 放大内心感受 [0, 1]
    """
    profiles = np.zeros((2, ST_SIZE), dtype=np.float64)
    inputs_full = np.concatenate([traits, relationship, internal])  # (21,) for deact
    inputs_tr = np.concatenate([traits, relationship])              # (13,) for hyper

    # ── Profile 0: Deactivation ──
    # baseline(7,) + intensity(scalar) × PC1_dir(7,)
    d_intensity = DEACT_INTENSITY.compute(inputs_full)[0]
    deact_raw = DEACT_BASELINE_RAW + d_intensity * DEACT_PC1_DIR
    profiles[0] = _sigmoid((deact_raw - DEACT_SIGMOID_SHIFT) * DEACT_SIGMOID_SCALE)

    # ── Profile 1: Hyperactivation ──
    # 人格基线（秩-1: 仅 traits+relationship → 标量 × PC1_DIR）
    h_intensity = HYPER_INTENSITY.compute(inputs_tr)[0]
    hyper_raw_base = HYPER_BASELINE_RAW + h_intensity * HYPER_PC1_DIR

    # 状态调制（维度特异性: internal → hyper_delta，可产生交叉模式）
    h_state_delta = HYPER_STATE_MODULATION.compute(internal)  # (7,)

    # 合并：人格基线 + 状态调制
    hyper_raw = hyper_raw_base + h_state_delta
    profiles[1] = _sigmoid((hyper_raw - HYPER_SIGMOID_SHIFT) * HYPER_SIGMOID_SCALE)

    return soft_clamp(profiles, 0.0, 1.0)


def apply_defenses(
    stimuli: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """应用二维防御剖面，生成 inner / outer 刺激。

    每一类心理刺激经过其对应的防御剖面维度:

      inner[s]  = stimuli[s] × (1 + hyperactivation[s] × 0.50)
        → 过度激活放大内心感受: "我比看起来更在意"

      outer[s]  = inner[s] × (1 − deactivation[s] × 0.70)
        → 去激活削减外在表达: "我不想让人看出来"

    deactivation 控制 outer 削减，hyperactivation 控制 inner 放大。
    两者独立——可以内心翻江倒海但表面波澜不惊（高 hyper + 高 deact），
    也可以内心平静且表里如一（低 hyper + 低 deact）。

    Args:
        stimuli: 原始心理刺激 (7,)
        profiles: 防御剖面 (2, 7)

    Returns:
        (inner_stimuli, outer_stimuli) — 均为 (7,)
    """
    deact = profiles[0]  # (7,)
    hyper = profiles[1]  # (7,)

    # Inner: 过度激活放大内心感受
    # HYPER_APPLY_GAIN 来自 _defense_weights.py（WeightVector 带 provenance）
    inner = stimuli * (1.0 + hyper * HYPER_APPLY_GAIN)

    # Outer: 去激活削减外在表达
    # DEACT_APPLY_GAIN 来自 _defense_weights.py（WeightVector 带 provenance）
    outer = inner * (1.0 - deact * DEACT_APPLY_GAIN)

    inner = soft_clamp(inner, 0.0, 1.0)
    outer = soft_clamp(outer, 0.0, 1.0)

    return inner, outer


def compute_defense_metabolic_cost(
    profiles: np.ndarray,
    traits: np.ndarray,
) -> np.ndarray:
    """防御剖面的即时代谢成本（独立于表面反馈通路）。

    高 a（过度激活/内心翻涌）和 高 d（去激活/压抑）都消耗能量：
      - hyperactivation: 情绪放大需要认知资源 → MentalFatigue↑, Stress↑
      - deactivation: 压抑需要前额叶抑制控制 → Energy↓, MentalFatigue↑

    这在表面反馈（s→h 通路）中未被覆盖——当 s=0 时反馈通路无成本，
    但高 a + 高 d 时"面无表情但内心惊涛骇浪"的现实消耗巨大。

    修复 2026-06-25 (#4)：在动力学之前直接扣除防御的代谢成本。

    Args:
        profiles: 防御剖面 (2, 7) — [0]=deactivation, [1]=hyperactivation
        traits: 人格特质 (10,)

    Returns:
        cost (8,) 加到 current_internal 上，在动力学之前生效
    """
    from state import I_ENERGY, I_STRESS, I_MENTAL_FATIGUE

    cost = np.zeros(8, dtype=np.float64)
    deact = np.mean(profiles[0])  # 平均去激活水平 [0, 1]
    hyper = np.mean(profiles[1])  # 平均过度激活水平 [0, 1]

    # ① 去激活成本：压抑消耗执行控制资源
    # 前额叶持续抑制冲动 → Energy↓ + MentalFatigue↑
    cost[I_ENERGY] -= 0.04 * deact
    cost[I_MENTAL_FATIGUE] += 0.06 * deact

    # ② 过度激活成本：情绪放大消耗认知资源
    # 持续的反刍/放大 → Stress↑ + MentalFatigue↑
    cost[I_STRESS] += 0.04 * hyper
    cost[I_MENTAL_FATIGUE] += 0.06 * hyper

    # ③ 高焦虑者成本更高（情绪调节效率更低）
    anxiety_mod = 1.0 + max(0.0, float(traits[8])) * 0.3  # T_ATTACHMENT_ANXIETY
    cost *= anxiety_mod

    return cost
