"""实验管道 —— 5 条件路由。

条件:
  A_full    完整数学引擎（4步管线）
  A_no_def  跳过 Bowlby 防御剖面
  B_scratch LLM 每轮从零推断状态
  B_context LLM 推断 + 上一轮状态上下文
  C_none    无状态注入基线
"""

import json
import logging
from typing import Optional

import numpy as np

from state import (
    DEFAULT_TRAITS, ST_SIZE, ST_LABELS, I_LABELS, R_LABELS, S_LABELS, T_LABELS,
    StimulusMetadata,
)
from state_engine import update_all, initialize_all
from state_engine._decay import apply_time_decay
from state_formatter import format_state_for_node
from llm import model

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _extract_context_text(messages: list) -> str:
    """将消息列表转换为 LLM 可读的对话历史文本。"""
    lines = []
    for i, m in enumerate(messages):
        content = getattr(m, "content", str(m))
        role = getattr(m, "type", "unknown")
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"
        lines.append(f"[轮次 {i}] {role}: {content}")
    return "\n".join(lines)


def _make_stimulus_metadata(stimuli: np.ndarray) -> StimulusMetadata:
    return StimulusMetadata.with_confidence(float(np.mean(np.abs(stimuli))))


def _get_perception_fn(config: ExperimentConfig):
    """根据配置选择感知提供者。"""
    if config.perception_provider == "deepseek":
        from .llm_perception import call_perception_with_retry
        logger.info("使用 DeepSeek 感知")
        return call_perception_with_retry
    from perception import call_perception_with_retry
    logger.info("使用 Ollama 感知")
    return call_perception_with_retry


def _state_desc_prompt(state_desc: str) -> str:
    """用 format_state 输出构建状态注入 prompt。"""
    return f"## 当前心理状态\n{state_desc}\n"


# ═══════════════════════════════════════════════════════════════
# 感知
# ═══════════════════════════════════════════════════════════════

def _run_perception(
    conversation_history: list,
    internal: np.ndarray,
    relationship: np.ndarray,
    condition: str,
    config: ExperimentConfig,
) -> tuple[np.ndarray, StimulusMetadata]:
    """运行感知提取。条件 C 和 mock 模式跳过。"""
    if condition == "C_none":
        return np.zeros(ST_SIZE, dtype=np.float64), _make_stimulus_metadata(np.zeros(ST_SIZE))

    # Check if this condition uses perception
    if condition.startswith("B_") or condition.startswith("A_"):
        perception_fn = _get_perception_fn(config)
        cfg = {"max_retries": 2, "context_window": 10, "retry_emphases": []}
        stim_result = perception_fn(
            conversation_history, cfg,
            internal_state=internal,
            relationship_state=relationship,
        )
        if stim_result is not None:
            return stim_result["user_stimuli"], _make_stimulus_metadata(stim_result["user_stimuli"])

    logger.warning(f"[{condition}] 感知失败，使用零刺激")
    return np.zeros(ST_SIZE, dtype=np.float64), _make_stimulus_metadata(np.zeros(ST_SIZE))


# ═══════════════════════════════════════════════════════════════
# 状态更新路由
# ═══════════════════════════════════════════════════════════════

def _update_state(
    condition: str,
    internal: np.ndarray,
    relationship: np.ndarray,
    surface: np.ndarray,
    prev_surface: Optional[np.ndarray],
    stimuli: np.ndarray,
    traits_array: np.ndarray,
    stimulus_metadata: StimulusMetadata,
    config: ExperimentConfig,
    previous_state_text: str = "",
    conversation_history_text: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据条件更新状态。"""

    # ── C_none: 无状态更新 ──
    if condition == "C_none":
        return internal, relationship, surface

    # ── A_full: 完整引擎 ──
    if condition == "A_full":
        decayed = apply_time_decay(internal, relationship, traits_array, delta_hours=config.time_decay_hours)
        internal = decayed["internal_state"]
        relationship = decayed["relationship_state"]
        result = update_all(
            current_internal=internal,
            current_relationship=relationship,
            traits=traits_array,
            stimuli=stimuli,
            prev_surface=prev_surface,
            stimulus_metadata=stimulus_metadata,
            delta_hours=config.time_decay_hours,
            noise_sigma=0.005,
            skip_defenses=False,
        )
        return result["internal_state"], result["relationship_state"], result["surface_state"]

    # ── A_no_def: 无防御剖面 ──
    if condition == "A_no_def":
        decayed = apply_time_decay(internal, relationship, traits_array, delta_hours=config.time_decay_hours)
        internal = decayed["internal_state"]
        relationship = decayed["relationship_state"]
        result = update_all(
            current_internal=internal,
            current_relationship=relationship,
            traits=traits_array,
            stimuli=stimuli,
            prev_surface=prev_surface,
            stimulus_metadata=stimulus_metadata,
            delta_hours=config.time_decay_hours,
            noise_sigma=0.005,
            skip_defenses=True,  # ← 关键差异
        )
        return result["internal_state"], result["relationship_state"], result["surface_state"]

    # ── B_scratch / B_context: LLM 状态推断 ──
    if condition.startswith("B_"):
        from .llm_inference import llm_state_inference

        # B_scratch: 无对话历史 + 无上一轮状态（完全从零推断）
        # B_context: 有对话历史 + 有上一轮状态描述（含context）
        has_context = (condition == "B_context")
        inferred = llm_state_inference(
            traits=traits_array,
            stimuli=stimuli,
            conversation_history_text=conversation_history_text if has_context else "",
            config=config,
            previous_state_text=previous_state_text if has_context else "",
            delta_hours=config.time_decay_hours,
        )
        if inferred is not None:
            return inferred["internal"], inferred["relationship"], inferred["surface"]
        logger.warning(f"[{condition}] LLM 状态推断失败，保持上一轮状态")
        return internal, relationship, surface

    raise ValueError(f"Unknown condition: {condition}")


def _generate_response(
    state_desc: str,
    conversation_history: list,
    user_text: str,
    condition: str,
) -> str:
    """生成角色回复。"""
    from langchain.messages import HumanMessage, AIMessage

    sys_prompt = (
        f"你是月下誓约·予爱以心。{state_desc}\n\n"
        "请根据对话历史和当前心理状态，以角色的身份回复。"
    )
    fill_msgs = [
        {"role": "system", "content": sys_prompt},
        *[{"role": "user" if isinstance(m, HumanMessage) else "assistant",
           "content": m.content} for m in conversation_history],
    ]
    try:
        response = model.invoke(fill_msgs)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[{condition}] 回复生成失败: {e}", exc_info=True)
        return f"[生成错误] {e}"


# ═══════════════════════════════════════════════════════════════
# 场景运行器
# ═══════════════════════════════════════════════════════════════

def run_scenario(
    scenario_name: str,
    seed_inputs: list[str],
    test_inputs: list[str],
    traits_array: np.ndarray,
    condition: str,
    config: ExperimentConfig,
    seed: int = 42,
) -> dict:
    """在指定条件下完整运行一个场景。"""
    from langchain.messages import HumanMessage, AIMessage

    result = {
        "condition": condition,
        "scenario": scenario_name,
        "seed_turns": [],
        "test_turns": [],
        "state_trajectory": [],
        "stimulus_trajectory": [],
        "state_desc_trajectory": [],
        "error": None,
    }

    # 初始化状态
    initial = initialize_all(traits_array)
    internal = initial["internal_state"].copy()
    relationship = initial["relationship_state"].copy()
    surface = initial["surface_state"].copy()
    prev_surface = None
    conversation_history = []
    previous_state_text = ""  # for B_context

    state_desc = ""

    # ── 种子轮 ──
    for turn_idx, user_text in enumerate(seed_inputs):
        user_msg = HumanMessage(content=user_text)
        conversation_history.append(user_msg)

        # 感知
        stimuli, stim_meta = _run_perception(
            conversation_history, internal, relationship, condition, config)

        # 状态更新
        internal, relationship, surface = _update_state(
            condition, internal, relationship, surface, prev_surface,
            stimuli, traits_array, stim_meta, config,
            previous_state_text=previous_state_text,
            conversation_history_text=_extract_context_text(conversation_history))

        # 格式化状态
        state_desc = format_state_for_node({
            "internal_state": internal,
            "relationship_state": relationship,
            "surface_state": surface,
            "traits": traits_array,
        })

        # 保存上一轮状态描述（供 B_context 使用）
        previous_state_text = state_desc

        # 生成回复
        response_text = _generate_response(state_desc, conversation_history, user_text, condition)

        ai_msg = AIMessage(content=response_text)
        conversation_history.append(ai_msg)
        prev_surface = surface.copy()

        result["seed_turns"].append({
            "turn": turn_idx,
            "user_input": user_text,
            "response": response_text,
            "stimuli": stimuli.tolist(),
            "internal": internal.tolist(),
            "relationship": relationship.tolist(),
            "surface": surface.tolist(),
        })
        result["state_trajectory"].append({
            "internal": internal.copy(), "relationship": relationship.copy(), "surface": surface.copy()})
        result["stimulus_trajectory"].append(stimuli.copy())

    # ── 测试轮 ──
    for turn_idx, user_text in enumerate(test_inputs):
        user_msg = HumanMessage(content=user_text)
        conversation_history.append(user_msg)

        stimuli, stim_meta = _run_perception(
            conversation_history, internal, relationship, condition, config)

        internal, relationship, surface = _update_state(
            condition, internal, relationship, surface, prev_surface,
            stimuli, traits_array, stim_meta, config,
            previous_state_text=previous_state_text)

        state_desc = format_state_for_node({
            "internal_state": internal,
            "relationship_state": relationship,
            "surface_state": surface,
            "traits": traits_array,
        })
        previous_state_text = state_desc

        response_text = _generate_response(state_desc, conversation_history, user_text, condition)

        ai_msg = AIMessage(content=response_text)
        conversation_history.append(ai_msg)
        prev_surface = surface.copy()

        result["test_turns"].append({
            "turn": len(seed_inputs) + turn_idx,
            "user_input": user_text,
            "response": response_text,
            "stimuli": stimuli.tolist(),
            "internal": internal.tolist(),
            "relationship": relationship.tolist(),
            "surface": surface.tolist(),
        })
        result["state_trajectory"].append({
            "internal": internal.copy(), "relationship": relationship.copy(), "surface": surface.copy()})
        result["stimulus_trajectory"].append(stimuli.copy())

    return result


def run_all_conditions(
    seed_inputs: list[str],
    test_inputs: list[str],
    traits_array: np.ndarray,
    scenario_name: str,
    config: ExperimentConfig,
) -> dict:
    """在配置的所有条件下运行同一场景。"""
    results = {}
    for cond in config.conditions:
        logger.info(f"    运行条件 {cond}...")
        r = run_scenario(
            scenario_name=scenario_name,
            seed_inputs=seed_inputs,
            test_inputs=test_inputs,
            traits_array=traits_array,
            condition=cond,
            config=config,
        )
        results[cond] = r
    return results
