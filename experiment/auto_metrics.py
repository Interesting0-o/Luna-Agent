"""自动评估指标。

计算四种定量指标:
  1. 跨轮状态连续性 — 相邻轮次状态向量的余弦相似度方差
  2. 表面-内部分歧度 — surface vs internal 的 L2 距离
  3. 状态极端值频率 — 落在 [-1, -0.8] ∪ [0.8, 1] 的比例
  4. 状态随机游走检验 — Hurst 指数

所有指标在三种条件间直接比较。
"""

import numpy as np
from typing import Optional


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-12:
        return 0.0
    return dot / norm


def _hurst_exponent(ts: np.ndarray, max_lag: int = 8) -> float:
    """计算时间序列的 Hurst 指数。

    H < 0.5 → 均值回归（白噪声/回复性）
    H = 0.5 → 随机游走（布朗运动）
    H > 0.5 → 趋势增强（有结构的演化）

    使用经典 R/S 分析方法。
    """
    if len(ts) < 4:
        return 0.5

    ts = ts[~np.isnan(ts)]
    if len(ts) < 4:
        return 0.5

    max_lag = min(max_lag, len(ts) // 2)

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        n = len(ts) // lag * lag
        if n < lag:
            continue
        segments = ts[:n].reshape(-1, lag)
        # R/S per segment
        mean = segments.mean(axis=1, keepdims=True)
        deviations = segments - mean
        cumsum = deviations.cumsum(axis=1)
        R = cumsum.max(axis=1) - cumsum.min(axis=1)
        S = segments.std(axis=1, ddof=1)
        S = np.where(S < 1e-12, 1e-12, S)
        rs = (R / S).mean()
        rs_values.append(rs)

    if len(rs_values) < 2:
        return 0.5

    # Log-log regression: log(R/S) = H * log(lag) + C
    log_lags = np.log(list(lags[:len(rs_values)]))
    log_rs = np.log(np.array(rs_values))

    H, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(H, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# 主评估函数
# ═══════════════════════════════════════════════════════════════


def compute_all_metrics(
    condition_results: dict[str, dict],
) -> dict[str, dict]:
    """对所有条件计算全部自动指标。

    Args:
        condition_results: {condition: run_scenario() result} 字典

    Returns:
        {condition: { metric_name: value }} 字典
    """
    metrics = {}
    for cond, result in condition_results.items():
        metrics[cond] = _compute_condition_metrics(result)
    return metrics


def _compute_condition_metrics(result: dict) -> dict:
    """对单个条件的结果计算全部自动指标。"""
    trajectory = result["state_trajectory"]

    if len(trajectory) < 2:
        return {
            "cross_turn_continuity": {},
            "surface_internal_divergence": {},
            "extreme_value_frequency": {},
            "hurst_exponents": {},
        }

    # ── 1. 跨轮状态连续性 ──
    # 分别对 internal / relationship / surface 计算
    continuity = {}
    for layer in ("internal", "relationship", "surface"):
        vecs = np.array([t[layer] for t in trajectory])
        if len(vecs) < 2:
            continuity[layer] = {"mean": 0.0, "var": 0.0}
            continue
        sims = []
        for i in range(len(vecs) - 1):
            sim = _cosine_similarity(vecs[i], vecs[i + 1])
            sims.append(sim)
        continuity[layer] = {
            "mean": float(np.mean(sims)),
            "var": float(np.var(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
        }

    # ── 2. 表面-内部分歧度 ──
    # 表面状态 vs 内部状态的 L2 距离（每轮）
    s_vs_i_norms = []
    s_vs_r_norms = []
    for t in trajectory:
        s = t["surface"]
        i = t["internal"]
        r = t["relationship"]
        s_vs_i_norms.append(float(np.linalg.norm(s - i[:7])))  # internal 8→7维
        s_vs_r_norms.append(float(np.linalg.norm(s - np.pad(r, (0, 4), mode='constant'))))  # rel 3→7维
    divergence = {
        "surface_vs_internal": {
            "mean": float(np.mean(s_vs_i_norms)),
            "var": float(np.var(s_vs_i_norms)),
            "max": float(np.max(s_vs_i_norms)),
        },
        "surface_vs_relationship": {
            "mean": float(np.mean(s_vs_r_norms)),
            "var": float(np.var(s_vs_r_norms)),
            "max": float(np.max(s_vs_r_norms)),
        },
    }

    # ── 3. 状态极端值频率 ──
    extreme = {}
    for layer in ("internal", "relationship", "surface"):
        vecs = np.array([t[layer] for t in trajectory])
        total = vecs.size
        extreme_count = np.sum(np.abs(vecs) >= 0.8)
        near_zero = np.sum(np.abs(vecs) < 0.1)
        extreme[layer] = {
            "extreme_ratio": float(extreme_count / total) if total > 0 else 0.0,
            "near_zero_ratio": float(near_zero / total) if total > 0 else 0.0,
        }

    # ── 4. Hurst 指数 ──
    hurst = {}
    for layer in ("internal", "relationship", "surface"):
        vecs = np.array([t[layer] for t in trajectory])
        # 每维计算 Hurst，取均值
        dim_hursts = []
        for dim in range(vecs.shape[1]):
            h = _hurst_exponent(vecs[:, dim])
            dim_hursts.append(h)
        hurst[layer] = {
            "mean": float(np.mean(dim_hursts)),
            "per_dim": [float(h) for h in dim_hursts],
        }

    return {
        "cross_turn_continuity": continuity,
        "surface_internal_divergence": divergence,
        "extreme_value_frequency": extreme,
        "hurst_exponents": hurst,
    }


def format_metrics_comparison(
    metrics: dict[str, dict],
    condition_order: list[str] = None,
) -> str:
    """格式化三种条件的指标对比表格。"""
    if condition_order is None:
        condition_order = ["A", "B", "C"]

    lines = []
    lines.append("# 自动指标对比\n")
    lines.append(f"| 指标 | 维度 | {' | '.join(f'条件 {c}' for c in condition_order)} |")
    sep = f"|------|------|{'|'.join('-' * 12 for _ in condition_order)}|"
    lines.append(sep)

    # Helper
    def val_str(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # ── 1. 连续性 ──
    lines.append("\n## 1. 跨轮状态连续性\n")
    cond_data = {}
    for cond in condition_order:
        if cond in metrics:
            cond_data[cond] = metrics[cond].get("cross_turn_continuity", {})

    for layer in ("internal", "relationship", "surface"):
        for metric in ("mean", "var"):
            row = [f"continuity_{metric}", layer]
            for cond in condition_order:
                cd = cond_data.get(cond, {})
                v = cd.get(layer, {}).get(metric, "N/A")
                row.append(val_str(v) if isinstance(v, float) else v)
            lines.append("| " + " | ".join(row) + " |")

    # ── 2. 分歧度 ──
    lines.append("\n## 2. 表面-内部分歧度\n")
    for layer_key in ("surface_vs_internal", "surface_vs_relationship"):
        for metric in ("mean", "max"):
            row = [f"divergence_{metric}", layer_key.replace("surface_vs_", "s_vs_")]
            for cond in condition_order:
                div = metrics.get(cond, {}).get("surface_internal_divergence", {})
                v = div.get(layer_key, {}).get(metric, "N/A")
                row.append(val_str(v) if isinstance(v, float) else v)
            lines.append("| " + " | ".join(row) + " |")

    # ── 3. 极端值频率 ──
    lines.append("\n## 3. 极端值与中间值频率\n")
    for layer in ("internal", "relationship", "surface"):
        for metric in ("extreme_ratio", "near_zero_ratio"):
            row = [f"{metric}", layer]
            for cond in condition_order:
                ext = metrics.get(cond, {}).get("extreme_value_frequency", {})
                v = ext.get(layer, {}).get(metric, "N/A")
                row.append(val_str(v) if isinstance(v, float) else v)
            lines.append("| " + " | ".join(row) + " |")

    # ── 4. Hurst ──
    lines.append("\n## 4. Hurst 指数（>0.5=有结构，<0.5=均值回归）\n")
    for layer in ("internal", "relationship", "surface"):
        row = ["hurst_mean", layer]
        for cond in condition_order:
            h = metrics.get(cond, {}).get("hurst_exponents", {})
            v = h.get(layer, {}).get("mean", "N/A")
            row.append(val_str(v) if isinstance(v, float) else v)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_summary_stats(metrics: dict[str, dict]) -> dict:
    """生成关键摘要统计。"""
    summary = {}

    for cond in sorted(metrics.keys()):
        m = metrics[cond]
        s = {}

        # Average continuity across layers
        cont = m.get("cross_turn_continuity", {})
        cont_means = [v.get("mean", 0) for v in cont.values() if isinstance(v, dict) and "mean" in v]
        s["avg_continuity"] = float(np.mean(cont_means)) if cont_means else 0.0

        # Surface-internal divergence
        div = m.get("surface_internal_divergence", {})
        s_vs_i = div.get("surface_vs_internal", {})
        s["surface_internal_divergence"] = s_vs_i.get("mean", 0.0)

        # Extreme value frequency
        ext = m.get("extreme_value_frequency", {})
        ext_ratios = [
            v.get("extreme_ratio", 0) for v in ext.values()
            if isinstance(v, dict) and "extreme_ratio" in v
        ]
        s["extreme_ratio"] = float(np.mean(ext_ratios)) if ext_ratios else 0.0

        zero_ratios = [
            v.get("near_zero_ratio", 0) for v in ext.values()
            if isinstance(v, dict) and "near_zero_ratio" in v
        ]
        s["near_zero_concentration"] = float(np.mean(zero_ratios)) if zero_ratios else 0.0

        # Average Hurst
        hurst = m.get("hurst_exponents", {})
        hurst_means = [v.get("mean", 0.5) for v in hurst.values() if isinstance(v, dict) and "mean" in v]
        s["avg_hurst"] = float(np.mean(hurst_means)) if hurst_means else 0.5

        summary[cond] = s

    return summary
