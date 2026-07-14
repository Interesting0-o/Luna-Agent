"""实验管道：三个条件的独立运行器。

条件 A / B / C 共享种子对话生成，仅在测试轮分支。
"""

import json
import logging
import copy
from typing import Optional
from datetime import datetime, timezone

import numpy as np

from state import (
    State, DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    ST_SIZE, ST_LABELS, I_LABELS, R_LABELS, S_LABELS, T_LABELS,
    I_STRESS, I_ENERGY, I_LONELINESS, I_INSECURITY, I_IRRITATION, I_LONGING,
    I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    StimulusMetadata,
)
from perception import call_perception_with_retry
from state_engine import update_all, initialize_all
from state_engine._decay import apply_time_decay
from state_formatter import format_state_for_node
from llm import model, perception_model

from .config import ExperimentConfig
from .llm_inference import llm_state_inference

logger = logging.getLogger(__name__)


def _mock_generate(conversation_history: list, condition: str, user_text: str) -> str:
    """无 API 时生成模拟回复。"""
    condition_signatures = {
        "A": "(轻轻叹了口气) 又是这样...你总是这样突然出现，说些让人措手不及的话。",
        "B": "哼，你这话说得...让我怎么接啊。",
        "C": "嗯。",
    }
    base = condition_signatures.get(condition, condition_signatures["C"])
    return f"{base} [模拟回复-条件{condition}]"


def _mock_infer_state(
    internal: np.ndarray, relationship: np.ndarray, surface: np.ndarray,
    stimuli: np.ndarray, condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模拟状态推断（无 API 时用噪声代替）。"""
    noise_scale = 0.05
    new_internal = np.clip(internal + rng.normal(0, noise_scale, internal.shape), -1, 1)
    new_rel = np.clip(relationship + rng.normal(0, noise_scale * 0.3, relationship.shape), -1, 1)
    new_surface = np.clip(surface + rng.normal(0, noise_scale, surface.shape), -1, 1)
    return new_internal, new_rel, new_surface
    """将消息列表转换为 LLM 可读的对话历史文本。"""
    lines = []
    idx = 0
    for m in messages:
        content = getattr(m, "content", str(m))
        role = getattr(m, "type", "unknown")
        # Truncate very long messages for context
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"
        lines.append(f"[轮次 {idx}] {role}: {content}")
        idx += 1
    return "\n".join(lines)


def _make_stimulus_metadata(stimuli: np.ndarray) -> StimulusMetadata:
    """从刺激向量创建 StimulusMetadata。"""
    return StimulusMetadata.with_confidence(float(np.mean(np.abs(stimuli))))


def _format_state_vector(name: str, values: np.ndarray, labels: list[str]) -> str:
    """格式化状态向量为人类可读描述。"""
    parts = []
    for i, (label, val) in enumerate(zip(labels, values)):
        parts.append(f"  {label}: {float(val):+.3f}")
    return f"{name} ({len(values)}d):\n" + "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Run one full scenario under a given condition
# ═══════════════════════════════════════════════════════════════


def run_scenario(
    scenario_name: str,
    seed_inputs: list[str],
    test_inputs: list[str],
    traits_array: np.ndarray,
    condition: str,
    config: ExperimentConfig,
    seed: int = 42,
    mock_mode: bool = False,
) -> dict:
    """在指定条件下完整运行一个场景。

    Args:
        scenario_name: 场景名称
        seed_inputs: 种子对话用户输入（N 轮，所有条件共享）
        test_inputs: 测试轮用户输入（M 轮，条件分支后）
        traits_array: 人格特质 (10,)
        condition: "A" / "B" / "C"
        config: 实验配置
        seed: 随机种子

    Returns:
        结果字典:
          - condition: 条件标识
          - scenario: 场景名称
          - seed_turns: 种子轮状态/回复记录
          - test_turns: 测试轮状态/回复记录
          - state_trajectory: 完整状态轨迹（测试轮）
          - error: 错误信息（如有）
    """
    rng = np.random.default_rng(seed)
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

    # ── 初始化状态 ──
    initial = initialize_all(traits_array)
    internal = initial["internal_state"].copy()
    relationship = initial["relationship_state"].copy()
    surface = initial["surface_state"].copy()
    prev_surface = None  # First turn has no previous surface
    current_stimulus_metadata = _make_stimulus_metadata(np.zeros(ST_SIZE))

    # ── 对话历史 (LangChain 消息格式) ──
    from langchain.messages import HumanMessage, AIMessage
    conversation_history = []

    # ── 1. 种子对话（所有条件共享） ──
    for turn_idx, user_text in enumerate(seed_inputs):
        user_msg = HumanMessage(content=user_text)
        conversation_history.append(user_msg)

        # Percept
        if condition in ("A", "B"):
            stim_result = call_perception_with_retry(
                conversation_history,
                {"max_retries": 1, "context_window": 10, "retry_emphases": ["仔细分析用户输入的情感色彩"]},
                internal_state=internal,
                relationship_state=relationship,
            )
            if stim_result is None:
                stimuli = np.zeros(ST_SIZE, dtype=np.float64)
                logger.warning(f"[{scenario_name}/{condition}] 感知失败，使用零刺激")
            else:
                stimuli = stim_result["user_stimuli"]
                current_stimulus_metadata = _make_stimulus_metadata(stimuli)
        else:
            # Condition C: no perception
            stimuli = np.zeros(ST_SIZE, dtype=np.float64)

        # State update
        if condition == "A":
            # 应用时间衰减
            decayed = apply_time_decay(
                internal, relationship, traits_array, delta_hours=0.0,
            )
            internal = decayed["internal_state"]
            relationship = decayed["relationship_state"]
            update_result = update_all(
                current_internal=internal,
                current_relationship=relationship,
                traits=traits_array,
                stimuli=stimuli,
                prev_surface=prev_surface,
                stimulus_metadata=current_stimulus_metadata,
                delta_hours=0.0,
                noise_sigma=0.005,
            )
            internal = update_result["internal_state"]
            relationship = update_result["relationship_state"]
            surface = update_result["surface_state"]
        elif condition == "B":
            if mock_mode:
                internal, relationship, surface = _mock_infer_state(
                    internal, relationship, surface, stimuli, condition, rng)
            else:
                # LLM state inference
                state_desc_text = _extract_context_text(conversation_history)
                inferred = llm_state_inference(
                    traits=traits_array,
                    stimuli=stimuli,
                    conversation_history_text=state_desc_text,
                    config=config,
                )
                if inferred is not None:
                    internal = inferred["internal"]
                    relationship = inferred["relationship"]
                    surface = inferred["surface"]
            # If inference fails, keep previous state
        else:
            # Condition C: no state update
            pass

        # Format state
        state_desc = format_state_for_node({
            "internal_state": internal,
            "relationship_state": relationship,
            "surface_state": surface,
            "traits": traits_array,
        })

        # Generate response
        if mock_mode:
            response_text = _mock_generate(conversation_history, condition, user_text)
        else:
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
                response_msg = model.invoke(fill_msgs)
                response_text = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
            except Exception as e:
                response_text = f"[生成错误] {e}"
                logger.error(f"[{scenario_name}/{condition}] 回复生成失败: {e}", exc_info=True)

        ai_msg = AIMessage(content=response_text)
        conversation_history.append(ai_msg)
        prev_surface = surface.copy()

        seed_turn = {
            "turn": turn_idx,
            "user_input": user_text,
            "response": response_text,
            "stimuli": stimuli.tolist(),
            "internal": internal.tolist(),
            "relationship": relationship.tolist(),
            "surface": surface.tolist(),
        }
        result["seed_turns"].append(seed_turn)
        result["state_trajectory"].append({
            "internal": internal.copy(),
            "relationship": relationship.copy(),
            "surface": surface.copy(),
        })
        result["stimulus_trajectory"].append(stimuli.copy())
        result["state_desc_trajectory"].append(str(state_desc)[:200])

    # ── 2. 测试轮（条件分支后） ──
    for turn_idx, user_text in enumerate(test_inputs):
        user_msg = HumanMessage(content=user_text)
        conversation_history.append(user_msg)

        # Percept
        if condition in ("A", "B"):
            stim_result = call_perception_with_retry(
                conversation_history,
                {"max_retries": 1, "context_window": 10, "retry_emphases": ["仔细分析用户输入的情感色彩"]},
                internal_state=internal,
                relationship_state=relationship,
            )
            if stim_result is None:
                stimuli = np.zeros(ST_SIZE, dtype=np.float64)
                logger.warning(f"[{scenario_name}/{condition}] 测试轮感知失败")
            else:
                stimuli = stim_result["user_stimuli"]
                current_stimulus_metadata = _make_stimulus_metadata(stimuli)
        else:
            stimuli = np.zeros(ST_SIZE, dtype=np.float64)

        # State update
        if condition == "A":
            decayed = apply_time_decay(
                internal, relationship, traits_array, delta_hours=0.0,
            )
            internal = decayed["internal_state"]
            relationship = decayed["relationship_state"]
            update_result = update_all(
                current_internal=internal,
                current_relationship=relationship,
                traits=traits_array,
                stimuli=stimuli,
                prev_surface=prev_surface,
                stimulus_metadata=current_stimulus_metadata,
                delta_hours=0.0,
                noise_sigma=0.005,
            )
            internal = update_result["internal_state"]
            relationship = update_result["relationship_state"]
            surface = update_result["surface_state"]
        elif condition == "B":
            if mock_mode:
                internal, relationship, surface = _mock_infer_state(
                    internal, relationship, surface, stimuli, condition, rng)
            else:
                state_desc_text = _extract_context_text(conversation_history)
                inferred = llm_state_inference(
                    traits=traits_array,
                    stimuli=stimuli,
                    conversation_history_text=state_desc_text,
                    config=config,
                )
                if inferred is not None:
                    internal = inferred["internal"]
                    relationship = inferred["relationship"]
                    surface = inferred["surface"]
        else:
            pass

        # Format state
        state_desc = format_state_for_node({
            "internal_state": internal,
            "relationship_state": relationship,
            "surface_state": surface,
            "traits": traits_array,
        })

        # Generate response
        if mock_mode:
            response_text = _mock_generate(conversation_history, condition, user_text)
        else:
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
                response_msg = model.invoke(fill_msgs)
                response_text = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
            except Exception as e:
                response_text = f"[生成错误] {e}"
                logger.error(f"[{scenario_name}/{condition}] 测试轮回复生成失败: {e}", exc_info=True)

        ai_msg = AIMessage(content=response_text)
        conversation_history.append(ai_msg)
        prev_surface = surface.copy()

        test_turn = {
            "turn": len(seed_inputs) + turn_idx,
            "user_input": user_text,
            "response": response_text,
            "stimuli": stimuli.tolist(),
            "internal": internal.tolist(),
            "relationship": relationship.tolist(),
            "surface": surface.tolist(),
        }
        result["test_turns"].append(test_turn)
        result["state_trajectory"].append({
            "internal": internal.copy(),
            "relationship": relationship.copy(),
            "surface": surface.copy(),
        })
        result["stimulus_trajectory"].append(stimuli.copy())
        result["state_desc_trajectory"].append(str(state_desc)[:200])

    return result


def run_all_conditions(
    seed_inputs: list[str],
    test_inputs: list[str],
    traits_array: np.ndarray,
    scenario_name: str,
    config: ExperimentConfig,
    mock_mode: bool = False,
) -> dict:
    """在三种条件下运行同一场景，返回聚合结果。"""
    results = {}
    for cond in config.conditions:
        r = run_scenario(
            scenario_name=scenario_name,
            seed_inputs=seed_inputs,
            test_inputs=test_inputs,
            traits_array=traits_array,
            condition=cond,
            config=config,
            mock_mode=mock_mode,
        )
        results[cond] = r
    return results
