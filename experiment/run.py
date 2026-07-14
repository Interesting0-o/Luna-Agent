"""A/B 实验主入口。

用法:
  uv run python -m experiment.run                  # 全部 20 场景
  uv run python -m experiment.run --scenarios 3    # 前 3 个场景（pilot）
  uv run python -m experiment.run --conditions A,B # 仅运行 A/B
  uv run python -m experiment.run --pilot           # 快速 pilot（2 场景）
  uv run python -m experiment.run --analyze-only    # 仅分析已有结果

每个场景在三种条件下运行，保存结果到 experiment/outputs/。
自动计算定量指标并调用 LLM-as-Judge 评估。
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import DEFAULT_TRAITS
from experiment.config import ExperimentConfig
from experiment.pipeline import run_all_conditions, run_scenario
from experiment import scenarios as scenarios_module
from experiment.scenarios import get_scenario_summary
from experiment.auto_metrics import (
    compute_all_metrics,
    format_metrics_comparison,
    generate_summary_stats,
)
from experiment.llm_judge import judge_responses

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, run_tag: str):
    """配置日志。"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_tag}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    logger.info(f"日志文件: {log_file}")


def run_experiment(config: ExperimentConfig, mock_mode: bool = False) -> dict:
    """运行完整实验。

    Args:
        config: 实验配置

    Returns:
        结果汇总字典
    """
    run_tag = config.run_tag or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(config.output_dir, exist_ok=True)

    # 选择场景
    if config.scenarios_file:
        with open(config.scenarios_file) as f:
            scenarios = json.load(f)
        logger.info(f"从文件加载了 {len(scenarios)} 个场景")
    else:
        scenarios = scenarios_module.SCENARIOS
        logger.info(f"使用内置 {len(scenarios)} 个场景")

    logger.info(f"运行条件: {', '.join(config.conditions)}")
    logger.info(f"每个场景 {config.num_seed_turns} 种子轮 + {config.num_test_turns} 测试轮")

    all_results = {}
    traits = DEFAULT_TRAITS.copy()

    for idx, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{len(scenarios)}] 场景: {name} ({scenario['category']})")
        logger.info(f"  描述: {scenario['description']}")

        try:
            t0 = time.time()
            results = run_all_conditions(
                seed_inputs=scenario["seed_user_inputs"][:config.num_seed_turns],
                test_inputs=scenario["test_user_inputs"][:config.num_test_turns],
                traits_array=traits,
                scenario_name=name,
                config=config,
                mock_mode=mock_mode,
            )
            elapsed = time.time() - t0
            logger.info(f"  ✓ 完成 ({elapsed:.1f}s)")

            # 记录每个条件的回复数量
            for cond, r in results.items():
                n_seed = len(r.get("seed_turns", []))
                n_test = len(r.get("test_turns", []))
                logger.info(f"  条件 {cond}: {n_seed} 种子轮 + {n_test} 测试轮")

            all_results[name] = results

            # 每 5 个场景保存一次中间结果
            if idx % 5 == 0:
                _save_intermediate(all_results, config.output_dir, run_tag)
                logger.info(f"  中间结果已保存 ({idx}/{len(scenarios)})")

        except Exception as e:
            logger.error(f"  ✗ 场景 '{name}' 失败: {e}", exc_info=True)
            all_results[name] = {"error": str(e)}

    # 保存完整结果
    _save_final(all_results, config.output_dir, run_tag, config, traits)

    return all_results


def _save_intermediate(results: dict, output_dir: str, run_tag: str):
    """保存中间结果（仅 JSON 兼容部分）。"""
    serializable = _make_serializable(results)
    path = os.path.join(output_dir, f"{run_tag}_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    logger.debug(f"中间结果保存至: {path}")


def _save_final(
    results: dict,
    output_dir: str,
    run_tag: str,
    config: ExperimentConfig,
    traits: np.ndarray,
):
    """保存最终结果并计算指标。"""
    serializable = _make_serializable(results)

    # 1. 原始结果
    raw_path = os.path.join(output_dir, f"{run_tag}_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"原始结果: {raw_path}")

    # 2. 自动指标
    for scenario_name, scenario_results in results.items():
        if "error" in scenario_results:
            continue
        metrics = compute_all_metrics(scenario_results)
        scenario_results["auto_metrics"] = metrics

    # 3. 指标汇总
    try:
        summary = _compute_summary_metrics(results)
        summary_path = os.path.join(output_dir, f"{run_tag}_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"指标摘要: {summary_path}")
    except Exception as e:
        logger.error(f"指标摘要生成失败: {e}", exc_info=True)

    # 4. 保存配置
    config_path = os.path.join(output_dir, f"{run_tag}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "conditions": config.conditions,
            "run_tag": run_tag,
            "num_scenarios": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"配置: {config_path}")


def _make_serializable(obj):
    """递归将对象转换为 JSON 可序列化格式。"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _compute_summary_metrics(all_results: dict) -> dict:
    """跨场景汇总自动指标。"""
    summary = {
        "by_condition": {c: [] for c in ["A", "B", "C"]},
        "by_category": {},
        "overall": {},
    }

    # Collect per-scenario metrics by condition
    for scenario_name, scenario_results in all_results.items():
        if "error" in scenario_results:
            continue
        # auto_metrics was set at scenario level: scenario_results["auto_metrics"][cond][...]
        auto_m = scenario_results.get("auto_metrics", {})
        for cond in ["A", "B", "C"]:
            metrics = auto_m.get(cond, {})
            if metrics:
                summary["by_condition"][cond].append(metrics)

    # Average across scenarios
    for cond, cond_metrics in summary["by_condition"].items():
        if not cond_metrics:
            continue
        summary["overall"][cond] = _average_metrics(cond_metrics)

    return summary


def _average_metrics(metrics_list: list[dict]) -> dict:
    """对多个场景的指标取均值。"""
    if not metrics_list:
        return {}
    avg = {}
    for key in ["cross_turn_continuity", "surface_internal_divergence",
                "extreme_value_frequency", "hurst_exponents"]:
        layer_values = {}
        count = 0
        for m in metrics_list:
            layer_data = m.get(key, {})
            if not layer_data:
                continue
            for layer, vals in layer_data.items():
                if isinstance(vals, dict):
                    for subkey, val in vals.items():
                        if isinstance(val, (int, float)):
                            layer_values.setdefault(f"{layer}.{subkey}", []).append(val)
            count += 1
        avg[key] = {
            k: float(np.mean(v)) for k, v in layer_values.items()
        } if layer_values else {}
    return avg


# _generate_report 已移除，使用 experiment/outputs/ 中的独立分析脚本生成报告


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Luna A/B 实验 — 验证数学引擎的边际价值"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="快速 pilot 模式（仅 2 场景，1 测试轮）"
    )
    parser.add_argument(
        "--scenarios", type=int, default=None,
        help="运行的场景数（前 N 个）"
    )
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="运行的条件，逗号分隔（如 'A,B'）"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiment/outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="运行标识"
    )
    parser.add_argument(
        "--analyze-only", type=str, default=None,
        help="仅分析已有结果文件（文件路径）"
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="列出所有场景"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="mock 模式（无 API 调用，使用合成数据验证框架）"
    )
    return parser.parse_args()


def main():
    """实验主入口。"""
    args = parse_args()

    if args.list_scenarios:
        print(get_scenario_summary())
        return

    if args.analyze_only:
        # 仅分析已有结果
        print(f"分析模式: {args.analyze_only}")
        print("（待实现：加载已有结果 → 重新计算指标 → 生成报告）")
        return

    config = ExperimentConfig(
        output_dir=args.output_dir,
        run_tag=args.tag or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    if args.conditions:
        config.conditions = [c.strip().upper() for c in args.conditions.split(",")]

    if args.pilot:
        config.scenarios_file = ""  # 使用内置场景，但只取前 2 个
        config.num_test_turns = 1

    run_tag = config.run_tag
    setup_logging(config.output_dir, run_tag)

    logger.info("=" * 60)
    logger.info("Luna A/B 实验")
    logger.info(f"条件: {config.conditions}")
    logger.info(f"运行标识: {run_tag}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"Mock模式: {'开启' if args.mock else '关闭 (需要API key)'}")
    logger.info(f"Pilot模式: {'开启' if args.pilot else '关闭'}")
    logger.info("=" * 60)

    # 限制场景数
    if args.scenarios:
        m = min(args.scenarios, len(scenarios_module.SCENARIOS))
        scenarios_module.SCENARIOS = scenarios_module.SCENARIOS[:m]
        logger.info(f"限制运行前 {m} 个场景（--scenarios={args.scenarios}）")
    elif args.pilot:
        scenarios_module.SCENARIOS = scenarios_module.SCENARIOS[:2]
        logger.info("Pilot 模式：运行前 2 个场景，1 测试轮")

    # 场景确认
    logger.info(f"\n{get_scenario_summary()}")

    # 运行实验
    results = run_experiment(config, mock_mode=args.mock)

    # 打印摘要
    n_ok = sum(1 for v in results.values() if "error" not in v)
    n_err = sum(1 for v in results.values() if "error" in v)
    logger.info(f"\n{'='*60}")
    logger.info(f"实验完成: {n_ok} ✓ / {n_err} ✗")
    logger.info(f"结果目录: {config.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
