"""DeepSeek 感知模块 —— 替代 Ollama 做心理刺激提取。

Ollama qwen2.5:7b 的 JSON 输出不稳定（pilot中100%失败），
使用 DeepSeek v4 Pro 做感知提取以获取稳定的刺激向量。

与原 perception.py 接口兼容：
  call_perception_with_retry() → {"user_stimuli": ndarray(7,), ...}
"""

import json
import logging
import numpy as np
from typing import Optional
from langchain.messages import SystemMessage

from llm import model
from state import ST_SIZE, ST_LABELS, ST_LABEL_IDX, stimuli_from_dict, StimulusMetadata

logger = logging.getLogger(__name__)

# 精简版感知 prompt——DeepSeek 比 Ollama 强大得多，不需要冗长的逐步引导
PERCEPTION_SYSTEM_PROMPT = """你是一个心理刺激提取系统。分析用户的输入，提取角色感受到的 7 维心理刺激。

## 7 维刺激定义
1. abandonment_stimulus [0.0-1.0]: 被抛弃/被冷落感
2. validation_stimulus [0.0-1.0]: 被认可/被重视感
3. closeness_stimulus [0.0-1.0]: 亲密靠近/温暖感
4. conflict_stimulus [0.0-1.0]: 冲突/对抗感
5. dependency_stimulus [0.0-1.0]: 被依赖感/责任感
6. teasing_stimulus [0.0-1.0]: 调侃/轻松氛围感
7. emotional_weight_stimulus [0.0-1.0]: 情感重量/冲击感

不要同时激活多个维度——一段话通常只传达 1-3 种主要信号。
充分使用 [0,1] 全程，强烈信号打高分(0.8+)，微弱信号打低分(0.2以下)。

输出 JSON 格式：
{"user_stimuli": {"abandonment_stimulus": 0.0, "validation_stimulus": 0.0, ...}}
"""


def call_perception_with_retry(
    user_context: list,
    cfg: dict,
    internal_state: Optional[np.ndarray] = None,
    relationship_state: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """用 DeepSeek 提取心理刺激。

    与 perception.py:call_perception_with_retry 接口完全兼容。

    Args:
        user_context: 对话上下文消息列表
        cfg: 配置字典（只用 max_retries）
        internal_state: 当前内部状态 (8,)
        relationship_state: 当前关系状态 (3,)

    Returns:
        {"user_stimuli": ndarray(7,)} 或 None
    """
    max_attempts = cfg.get("max_retries", 2)
    last_error = None

    # 构建 prompt：提取用户最新输入
    last_user_msg = ""
    for msg in reversed(user_context):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_msg = msg.content
            break
        elif isinstance(getattr(msg, "content", ""), str) and getattr(msg, "content", ""):
            # Try to extract from dict-style messages too
            pass

    # 注入状态上下文
    prompt = PERCEPTION_SYSTEM_PROMPT
    if internal_state is not None:
        i_labels = ["energy", "stress", "loneliness", "insecurity",
                    "irritation", "longing", "social_battery", "mental_fatigue"]
        i_summary = " | ".join(f"{l}={internal_state[i]:+.2f}" for i, l in enumerate(i_labels))
        prompt += f"\n## 角色当前内部状态\n{i_summary}\n（高压力→易感被抛弃，高好感→易感被认可）\n"
    if relationship_state is not None:
        r_labels = ["affection", "trust_bond", "intimacy"]
        r_summary = " | ".join(f"{l}={relationship_state[i]:+.2f}" for i, l in enumerate(r_labels))
        prompt += f"\n## 角色对用户的关系状态\n{r_summary}\n"

    for attempt in range(max_attempts):
        try:
            msgs = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"用户说: {last_user_msg}\n\n请提取刺激向量。"},
            ]
            raw = model.invoke(msgs)
            text = raw.content.strip() if hasattr(raw, "content") else str(raw).strip()

            # 剥 JSON fence
            import re
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
            text = text.strip()

            data = json.loads(text)
            stimuli = data.get("user_stimuli", data)  # 兼容直接返回或嵌套返回

            # 验证
            if isinstance(stimuli, dict):
                arr = np.zeros(ST_SIZE, dtype=np.float64)
                valid = False
                for key, idx in ST_LABEL_IDX.items():
                    if key in stimuli and isinstance(stimuli[key], (int, float)):
                        arr[idx] = np.clip(stimuli[key], 0.0, 1.0)
                        valid = True
                if valid:
                    logger.info(f"DeepSeek 感知成功: { {k: f'{arr[ST_LABEL_IDX[k]]:.2f}' for k in list(ST_LABEL_IDX)[:3]} }...")
                    return {"user_stimuli": arr}

            last_error = f"无效响应格式: {text[:100]}"
            logger.warning(f"DeepSeek 感知第 {attempt+1} 次格式无效: {last_error}")

        except (json.JSONDecodeError, Exception) as e:
            last_error = str(e)
            logger.warning(f"DeepSeek 感知第 {attempt+1} 次尝试失败: {last_error}")

    logger.error(f"DeepSeek 感知全部 {max_attempts} 次尝试失败。最后错误: {last_error}")
    return None
