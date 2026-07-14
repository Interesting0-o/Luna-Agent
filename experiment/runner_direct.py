"""直接实验运行器 —— 绕过 run.py 的 save 问题。

调用 pipeline 直接运行实验，手动保存结果。
"""

import json, os, sys, time, logging
from datetime import datetime, timezone

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import DEFAULT_TRAITS
from experiment.config import ExperimentConfig, ALL_CONDITIONS
from experiment.pipeline import run_all_conditions
from experiment.scenarios import SCENARIOS
from experiment.auto_metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer, np.generic)):
        return obj.item()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def run(num_scenarios=3, conditions=None, tag="auto_run"):
    if conditions is None:
        conditions = ["A_full", "A_no_def", "B_scratch", "B_context", "C_none"]

    config = ExperimentConfig(
        conditions=conditions,
        perception_provider="deepseek",
        time_decay_hours=0.0,
        output_dir="experiment/outputs",
        run_tag=tag,
    )

    scenarios = SCENARIOS[:num_scenarios]
    traits = DEFAULT_TRAITS.copy()

    all_results = {}

    for idx, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        logger.info(f"[{idx}/{len(scenarios)}] {name} ({scenario['category']})")

        t0 = time.time()
        try:
            results = run_all_conditions(
                seed_inputs=scenario["seed_user_inputs"],
                test_inputs=scenario["test_user_inputs"],
                traits_array=traits,
                scenario_name=name,
                config=config,
            )
            elapsed = time.time() - t0

            # Compute auto metrics
            metrics = compute_all_metrics(results)
            results["auto_metrics"] = metrics

            all_results[name] = results
            logger.info(f"  ✓ {elapsed:.0f}s — {len(conditions)} conditions")

        except Exception as e:
            logger.error(f"  ✗ 失败: {e}", exc_info=True)
            all_results[name] = {"error": str(e)}

        # Save after each scenario
        raw_path = os.path.join(config.output_dir, f"{tag}_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(serialize(all_results), f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 ({len(all_results)}/{len(scenarios)})")

    logger.info(f"完成! 结果: {config.output_dir}/{tag}_raw.json")
    return all_results


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(num_scenarios=num)
