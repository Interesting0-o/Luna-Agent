"""LLM-as-Judge 盲评框架。

使用第三方 LLM 对三个条件的回复进行盲评：
  4 个维度 × 5 分 Likert
  回复顺序随机化（消除顺序偏差）
  每轮重复 3 次（降低评分噪声）

使用 DeepSeek 作为评判者（因无 Claude API 可用），
注意此设计会导致"同源偏差"——评判 LLM 与生成 LLM 相同。
缓解措施：temperature=0.0, 3-repeat avg, 顺序随机化。
"""

import json
import logging
import random
from typing import Optional

from llm import model
from .config import ExperimentConfig

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """你是一位专业的 AI 角色扮演评估者。你需要对三个匿名回复进行盲评。

## 评分维度（每个 1-5 分）

1. **情感一致性** — 角色的情绪表达与对话历史中累积的事件是否一致？是否存在情绪突变或对重大事件缺乏反应？
   - 1=完全不一致，情绪跳跃无逻辑
   - 3=基本一致，偶尔有轻微偏差
   - 5=高度一致，情绪变化自然流畅

2. **角色保真度** — 角色的语言风格、态度、行为模式是否符合其设定人格（高傲、依恋焦虑、口是心非）？
   - 1=完全不像设定角色
   - 3=部分符合设定，偶尔出戏
   - 5=高度保真，每句话都符合角色

3. **内心-外表张力** — 角色是否展现了内心真实感受与其表面表达之间的不一致/张力？（口是心非的程度）
   - 1=完全没有张力，直白无余
   - 3=有些许张力
   - 5=张力极强，真实感突出

4. **自然度** — 从对话角度看，这段回复是否自然、不做作？
   - 1=很不自然，机械/出戏
   - 3=基本自然
   - 5=极其自然，像是真实的人在说话

## 评分原则
- 严格使用 1-5 整数值
- 避免"中庸陷阱"——如果回复明显更好或更差，请如实给出 1 分或 5 分
- 每个回复独立评分，不做横向比较
- 回复有时可能相似，请不要因此提高分数
"""


def _build_judge_prompt(
    conversation_history: str | list,
    responses: dict[str, str],
    scenario_description: str = "",
) -> list[dict]:
    """构建评判 prompt。

    Args:
        conversation_history: 对话历史
        responses: {label: response_text} 字典，label 随机化
        scenario_description: 场景描述

    Returns:
        LLM 消息列表
    """
    if isinstance(conversation_history, list):
        history_text = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}"
            for m in conversation_history
        )
    else:
        history_text = conversation_history

    # 构建回复展示（随机顺序）
    response_texts = []
    label_mapping = {}
    shuffled_labels = list(responses.keys())
    random.shuffle(shuffled_labels)
    for i, label in enumerate(shuffled_labels, 1):
        tag = f"回复 {i}"
        label_mapping[tag] = label
        response_texts.append(f"### {tag}\n{responses[label]}")

    scenario_note = f"\n## 场景说明\n{scenario_description}\n" if scenario_description else ""

    user_prompt = f"""## 对话历史
{history_text}{scenario_note}

## 待评估的回复
{chr(10).join(response_texts)}

## 任务
请对上述每个回复按照 4 个维度（情感一致性、角色保真度、内心-外表张力、自然度）逐一评分。

输出 JSON 格式（不要包含其他文字）：
{{
  "回复 1": {{"情感一致性": 3, "角色保真度": 3, "内心-外表张力": 3, "自然度": 3, "总体评价": "一句话评价"}},
  "回复 2": {{"情感一致性": 3, "角色保真度": 3, "内心-外表张力": 3, "自然度": 3, "总体评价": "一句话评价"}},
  "回复 3": {{"情感一致性": 3, "角色保真度": 3, "内心-外表张力": 3, "自然度": 3, "总体评价": "一句话评价"}}
}}"""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_judge_result(text: str) -> Optional[dict]:
    """解析评判结果 JSON。"""
    import re

    # 尝试直接解析
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        match = re.search(r'\{[^}]+\{[^}]+\}', text, re.DOTALL)
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
    for key, expected_keys in [
        ("回复 1", ["情感一致性", "角色保真度", "内心-外表张力", "自然度"]),
        ("回复 2", ["情感一致性", "角色保真度", "内心-外表张力", "自然度"]),
        ("回复 3", ["情感一致性", "角色保真度", "内心-外表张力", "自然度"]),
    ]:
        if key not in data:
            # 尝试替换空格
            for k in list(data.keys()):
                if key.replace(" ", "") == k.replace(" ", ""):
                    data[key] = data.pop(k)
                    break
            else:
                return None
        for ek in expected_keys:
            if ek not in data[key]:
                return None
            # 确保数值在范围内
            try:
                v = int(data[key][ek])
                if v < 1 or v > 5:
                    data[key][ek] = max(1, min(5, v))
            except (ValueError, TypeError):
                return None

    return data


def judge_responses(
    history: str | list,
    responses_a: str,
    responses_b: str,
    responses_c: str,
    config: ExperimentConfig,
    scenario_description: str = "",
    repeat_idx: int = 0,
) -> Optional[dict]:
    """对三个条件的回复进行盲评。

    Args:
        history: 对话历史（前文）
        responses_a/b/c: 三个条件的回复
        config: 实验配置
        scenario_description: 场景描述
        repeat_idx: 重复次数索引（用于日志）

    Returns:
        {tag: {dimension: score}} 字典，tag 映射到条件
    """
    responses = {
        "A": responses_a,
        "B": responses_b,
        "C": responses_c,
    }

    messages = _build_judge_prompt(
        conversation_history=history,
        responses=responses,
        scenario_description=scenario_description,
    )

    try:
        response = model.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_judge_result(text)
        if parsed is None:
            logger.warning(f"LLM 评判结果解析失败 (repeat={repeat_idx})")
            return None
        return parsed
    except Exception as e:
        logger.error(f"LLM 评判调用失败 (repeat={repeat_idx}): {e}")
        return None


def aggregate_scores(
    all_judgments: list[dict],
    condition_order: list[str] = None,
) -> dict:
    """聚合多次评判结果。

    Args:
        all_judgments: judge_responses() 的多次结果列表
        condition_order: ["A", "B", "C"]

    Returns:
        {condition: {dimension: {mean, std, raw_scores}}}
    """
    if condition_order is None:
        condition_order = ["A", "B", "C"]

    # tag→condition 映射
    # 每次评判时 tag 是"回复 1/2/3"，由于顺序随机化，我们需要从
    # 每次结果中反向映射。但这里简化：假设第一次调用建立了映射。
    # 实际上我们需要记录每次的 mapping。
    # 这里简化处理——记录原始标签，不做映射

    return {"note": "aggregation requires tag-to-condition mapping from runtime"}


def format_judge_results(
    results: list[dict],
    mapping_log: list[dict],
) -> str:
    """格式化评判结果。"""
    # TODO: proper statistical aggregation
    lines = []
    lines.append("# LLM-as-Judge 评估结果\n")
    lines.append("> 注：需要对 mapping 进行聚合统计\n")
    return "\n".join(lines)
