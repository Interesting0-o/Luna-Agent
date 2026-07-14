"""条件 B：LLM 状态推断。

用独立 LLM 调用，根据对话历史 + 刺激向量 + 人格特征
推断当前状态向量（替代数学引擎的 4 步管线）。

关键设计:
  1. 使用与回复生成不同的模型实例（避免状态泄漏）
  2. 温度 0.3（低随机性，稳定推断）
  3. 强制 JSON 输出 + bounds 检查
  4. 每轮独立推断（不持久化）
  5. prompt 注入时间间隔信息
"""

import json
import logging
from typing import Optional

import numpy as np

from state import (
    ST_SIZE, I_SIZE, R_SIZE, S_SIZE,
    ST_LABELS, I_LABELS, R_LABELS, S_LABELS, T_LABELS,
)
from llm import model

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个心理状态推断系统。根据对话历史、感知到的刺激信号和角色人格特征，推断角色此刻的连续心理状态。

## 推理原则
1. 状态值在 [-1.0, +1.0] 范围内连续变化
2. 状态变化应当有因果连续性——前一轮的状态会自然延续到本轮
3. 刺激信号会推动状态向相应方向偏移（冲突→stress↑、亲密→affection↑）
4. 人格特征会调制状态的强度和倾向（高焦虑→对冲突更敏感）
5. 状态不应过度集中在中间值（0 附近）——合理使用极端值表达强烈情感

## 输出格式
仅输出 JSON，不要包含其他文字。

## 注意
- 所有字段必须为 [-1.0, +1.0] 范围内的浮点数
- 请充分利用整个 [-1, 1] 范围，不要回避极端值
- 情绪应该有起伏变化，不要总在 0 附近
"""


def _build_state_inference_prompt(
    traits: np.ndarray,
    stimuli: np.ndarray,
    conversation_history_text: str,
    previous_state_text: str = "",
    delta_hours: float = 0.0,
) -> list[dict]:
    """构建 LLM 状态推断的 prompt。

    Args:
        traits: 人格特质 (10,)
        stimuli: 当前刺激向量 (7,)
        conversation_history_text: 对话历史的文本表示
        previous_state_text: 上一轮状态描述（可选）
        delta_hours: 距离上次对话的小时数

    Returns:
        LLM 消息列表
    """
    # 人格特征描述
    trait_lines = []
    for i, (label, val) in enumerate(zip(T_LABELS, traits)):
        trait_lines.append(f"  {label}: {float(val):+.2f}")

    # 刺激信号描述
    stim_lines = []
    for i, (label, val) in enumerate(zip(ST_LABELS, stimuli)):
        marker = "■" * max(1, int(abs(val) * 5)) if abs(val) > 0.05 else "−"
        stim_lines.append(f"  {label}: {float(val):+.3f} {marker}")

    # 状态输出模板
    state_template = """{
  "internal_state": {
    "energy": 0.0, "stress": 0.0, "loneliness": 0.0, "insecurity": 0.0,
    "irritation": 0.0, "longing": 0.0, "social_battery": 0.0, "mental_fatigue": 0.0
  },
  "relationship_state": {
    "affection": 0.0, "trust_bond": 0.0, "intimacy": 0.0
  },
  "surface_state": {
    "expressiveness": 0.0, "warmth": 0.0, "sharpness": 0.0,
    "softness": 0.0, "enthusiasm": 0.0, "restraint": 0.0, "vulnerability": 0.0
  }
}"""

    time_info = ""
    if delta_hours > 0:
        time_info = f"\n距离上次对话已过去 {delta_hours:.1f} 小时。在此期间情感会自然衰减回归基线。"

    prev_info = ""
    if previous_state_text:
        prev_info = f"\n## 上一轮状态\n{previous_state_text}\n"

    user_prompt = f"""## 对话历史
{conversation_history_text}

## 感知到的刺激信号（本轮）
{''.join(stim_lines)}

## 角色人格特征
{''.join(trait_lines)}{prev_info}{time_info}

## 请输出当前心理状态（JSON 格式）
{state_template}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_state_json(text: str) -> Optional[dict]:
    """从 LLM 输出中解析状态 JSON。"""
    # 尝试直接解析
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        import re
        match = re.search(r'\{[^{}]*"internal_state"[^{}]*"relationship_state"[^{}]*"surface_state"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
            else:
                return None

    # 验证结构
    for key in ("internal_state", "relationship_state", "surface_state"):
        if key not in data:
            logger.warning(f"LLM 状态推断缺少 {key}")
            return None

    try:
        internal = data["internal_state"]
        relationship = data["relationship_state"]
        surface = data["surface_state"]

        # 映射到 numpy 数组
        internal_arr = np.array([
            internal.get("energy", 0.0),
            internal.get("stress", 0.0),
            internal.get("loneliness", 0.0),
            internal.get("insecurity", 0.0),
            internal.get("irritation", 0.0),
            internal.get("longing", 0.0),
            internal.get("social_battery", 0.0),
            internal.get("mental_fatigue", 0.0),
        ], dtype=np.float64)

        rel_arr = np.array([
            relationship.get("affection", 0.0),
            relationship.get("trust_bond", 0.0),
            relationship.get("intimacy", 0.0),
        ], dtype=np.float64)

        surf_arr = np.array([
            surface.get("expressiveness", 0.0),
            surface.get("warmth", 0.0),
            surface.get("sharpness", 0.0),
            surface.get("softness", 0.0),
            surface.get("enthusiasm", 0.0),
            surface.get("restraint", 0.0),
            surface.get("vulnerability", 0.0),
        ], dtype=np.float64)

        # bounds 检查与裁剪
        for arr in (internal_arr, rel_arr, surf_arr):
            np.clip(arr, -1.0, 1.0, out=arr)

        return {
            "internal": internal_arr,
            "relationship": rel_arr,
            "surface": surf_arr,
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"LLM 状态推断解析失败: {e}")
        return None


def llm_state_inference(
    traits: np.ndarray,
    stimuli: np.ndarray,
    conversation_history_text: str,
    config: ExperimentConfig,
    previous_state_text: str = "",
    delta_hours: float = 0.0,
    max_retries: int = 0,
) -> Optional[dict]:
    """用 LLM 推断当前状态。

    Returns:
        包含 internal / relationship / surface 的字典，或 None。
    """
    messages = _build_state_inference_prompt(
        traits=traits,
        stimuli=stimuli,
        conversation_history_text=conversation_history_text,
        previous_state_text=previous_state_text,
        delta_hours=delta_hours,
    )

    for attempt in range(max_retries + 1):
        try:
            response = model.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            result = _parse_state_json(text)
            if result is not None:
                return result
            logger.warning(f"LLM 状态推断第 {attempt + 1} 次尝试返回格式无效，重试...")
        except Exception as e:
            logger.warning(f"LLM 状态推断第 {attempt + 1} 次尝试异常: {e}")
            if attempt == max_retries:
                return None

    return None
