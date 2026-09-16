"""实验结果分析器。

从 raw JSON 结果重新生成完整分析报告。
独立于实验运行，可对任意已有结果进行分析。

用法:
  uv run python -m experiment.analyze experiment/outputs/<tag>_raw.json
"""

import argparse
import json
import sys
import os
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(path: str) -> dict:
    """加载实验结果，恢复 numpy 数组。"""
    with open(path) as f:
        raw = json.load(f)

    # 恢复数组
    for sname, sdata in raw.items():
        if "error" in sdata:
            continue
        for cond in list(sdata.keys()):
            cd = sdata.get(cond, {})
            if not isinstance(cd, dict):
                continue
            for turn_key in ("state_trajectory",):
                traj = cd.get(turn_key, [])
                for t in traj:
                    for k in ("internal", "relationship", "surface"):
                        if k in t and isinstance(t[k], list):
                            t[k] = np.array(t[k], dtype=np.float64)
    return raw


def compute_metrics(raw: dict) -> dict:
    """对所有场景/条件计算自动指标。"""
    from experiment.auto_metrics import compute_all_metrics, generate_summary_stats

    for sname, sdata in raw.items():
        if "error" in sdata:
            continue
        # 只传入条件键（排除 auto_metrics 等非条件键）
        cond_data = {k: v for k, v in sdata.items()
                     if k.startswith(("A_", "B_", "C_"))}
        if not cond_data:
            continue
        metrics = compute_all_metrics(cond_data)
        sdata["auto_metrics"] = metrics

    # 跨条件汇总
    summary = _cross_condition_summary(raw)
    return summary


def _cross_condition_summary(raw: dict) -> dict:
    """跨场景按条件汇总指标。"""
    # 收集所有条件
    all_conds = set()
    for sdata in raw.values():
        if "error" in sdata:
            continue
        all_conds.update(k for k in sdata.keys() if k != "error" and k != "auto_metrics")

    # 按条件收集指标值
    metrics_by_cond = {c: [] for c in sorted(all_conds)}
    for sdata in raw.values():
        if "error" in sdata:
            continue
        am = sdata.get("auto_metrics", {})
        for cond in sorted(all_conds):
            cm = am.get(cond, {})
            if cm:
                metrics_by_cond[cond].append(cm)

    # 聚合
    def avg(lst):
        if not lst:
            return {}
        merged = {}
        for m in lst:
            for cat, vals in m.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if isinstance(v, (int, float, np.floating)):
                            merged.setdefault(f"{cat}.{k}", []).append(float(v))
                        elif isinstance(v, dict):
                            for k2, v2 in v.items():
                                if isinstance(v2, (int, float, np.floating)):
                                    merged.setdefault(f"{cat}.{k}.{k2}", []).append(float(v2))
        return {k: float(np.mean(v)) for k, v in merged.items() if v}

    return {cond: avg(vals) for cond, vals in metrics_by_cond.items() if vals}


def cohens_d(a: list[float], b: list[float]) -> float:
    """计算 Cohen's d 效应量。"""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1, n2 = len(a), len(b)
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def generate_report(raw: dict, summary: dict, tag: str = "") -> str:
    """生成完整分析报告。"""
    from experiment.scenarios import SCENARIOS

    lines = []
    lines.append(f"# 多变量实验分析报告\n")
    lines.append(f"> **结果文件**: {tag}")
    lines.append(f"> **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> **场景数**: {len(raw)}")
    conditions = sorted(set(
        k for s in raw.values() if "error" not in s
        for k in s.keys() if k not in ("error", "auto_metrics")
    ))
    lines.append(f"> **条件**: {', '.join(conditions)}\n")

    # ── 1. 核心指标对比表 ──
    lines.append("## 1. 核心指标对比\n")
    indicators = [
        ("cross_turn_continuity.internal.mean", "内态连续性", "越高越好"),
        ("cross_turn_continuity.surface.mean", "表面连续性", "越高越好"),
        ("surface_internal_divergence.surface_vs_internal.mean", "表-内分歧度", "越高口是心非越强"),
        ("extreme_value_frequency.internal.extreme_ratio", "内态极端值率", "越高状态越丰富"),
        ("extreme_value_frequency.surface.near_zero_ratio", "表面中性率", "越低表达越明确"),
        ("hurst_exponents.internal.mean", "内态Hurst", ">0.5=有结构"),
    ]

    lines.append(f"| {'指标':<26} | {' | '.join(f'{c:>10}' for c in conditions)} |")
    lines.append(f"|{'—'*28}|{'|'.join('—'*12 for _ in conditions)}|")

    for key, label, note in indicators:
        row = [f"**{label}**"]
        for cond in conditions:
            cm = summary.get(cond, {})
            v = cm.get(key, None)
            if v is not None:
                row.append(f"{v:>10.4f}")
            else:
                row.append(f"{'N/A':>10}")
        lines.append("| " + " | ".join(row) + " |")

    # ── 2. 效应量矩阵 ──
    lines.append("\n## 2. 条件间效应量 (Cohen's d)\n")
    lines.append("> 正值表示前者 > 后者，|d|>0.3=显著，|d|>0.8=强烈\n")

    # 收集每个场景的 continuity 值（按条件）
    continuity_by_cond = {c: [] for c in conditions}
    for sdata in raw.values():
        if "error" in sdata:
            continue
        am = sdata.get("auto_metrics", {})
        for cond in conditions:
            cm = am.get(cond, {})
            v = cm.get("cross_turn_continuity", {}).get("internal", {}).get("mean", None)
            if v is not None:
                continuity_by_cond[cond].append(v)

    lines.append(f"| 对比 | Δ连续性 | Cohen's d | 解读 |")
    lines.append(f"|------|---------|-----------|------|")
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            v1 = continuity_by_cond.get(c1, [])
            v2 = continuity_by_cond.get(c2, [])
            if v1 and v2:
                d = cohens_d(v1, v2)
                delta = np.mean(v1) - np.mean(v2)
                interp = "强烈差异" if abs(d) > 0.8 else ("显著差异" if abs(d) > 0.3 else "微弱差异")
                lines.append(f"| {c1} vs {c2} | {delta:+.4f} | {d:+.3f} | {interp} |")

    # ── 3. 场景类别聚合 ──
    lines.append("\n## 3. 按场景类别聚合\n")
    categories = {}
    for s in SCENARIOS:
        categories.setdefault(s["category"], []).append(s["name"])

    for cat, names in categories.items():
        lines.append(f"### {cat}\n")
        cat_metrics = {c: [] for c in conditions}
        for sname in names:
            sdata = raw.get(sname, {})
            if "error" in sdata:
                continue
            am = sdata.get("auto_metrics", {})
            for cond in conditions:
                cm = am.get(cond, {})
                v = cm.get("cross_turn_continuity", {}).get("internal", {}).get("mean", None)
                if v is not None:
                    cat_metrics[cond].append(v)

        lines.append(f"| {'条件':<12} | {'连续性均值':<12} | {'样本数':<8} |")
        lines.append(f"|{'—'*14}|{'—'*14}|{'—'*10}|")
        for cond in conditions:
            vals = cat_metrics[cond]
            if vals:
                lines.append(f"| {cond:<12} | {np.mean(vals):>10.4f}  | {len(vals):<8} |")
            else:
                lines.append(f"| {cond:<12} | {'N/A':>10}  | {'0':<8} |")
        lines.append("")

    # ── 4. 逐场景摘要 ──
    lines.append("\n## 4. 逐场景详情\n")
    for sname, sdata in raw.items():
        if "error" in sdata:
            lines.append(f"- **{sname}** ✗ 错误: {sdata['error']}\n")
            continue
        traj_lens = {}
        for cond in conditions:
            cd = sdata.get(cond, {})
            traj = cd.get("state_trajectory", [])
            traj_lens[cond] = len(traj)

        lines.append(f"### {sname}\n")
        lines.append(f"| {'条件':<12} | {'轮数':<6} | {'continuity':<12} | {'divergence':<12} | {'extreme':<10} | {'Hurst':<8} |")
        lines.append(f"|{'—'*14}|{'—'*8}|{'—'*14}|{'—'*14}|{'—'*12}|{'—'*10}|")
        for cond in conditions:
            cd = sdata.get(cond, {})
            traj = cd.get("state_trajectory", [])
            am = sdata.get("auto_metrics", {}).get(cond, {})
            cont = am.get("cross_turn_continuity", {}).get("internal", {}).get("mean", None)
            div = am.get("surface_internal_divergence", {}).get("surface_vs_internal", {}).get("mean", None)
            ext = am.get("extreme_value_frequency", {}).get("internal", {}).get("extreme_ratio", None)
            hurst = am.get("hurst_exponents", {}).get("internal", {}).get("mean", None)

            def f(v):
                return f"{v:>10.4f}" if v is not None else "  N/A"

            lines.append(f"| {cond:<12} | {len(traj):<6} | {f(cont):<12} | {f(div):<12} | {f(ext):<10} | {f(hurst):<8} |")
        lines.append("")

        # 样本回复
        lines.append("**回复样例（测试轮）**\n")
        for cond in conditions:
            cd = sdata.get(cond, {})
            test_turns = cd.get("test_turns", [])
            if test_turns:
                resp = test_turns[0].get("response", "")
                lines.append(f"- **{cond}**: {resp[:150]}...\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="实验结果分析器")
    parser.add_argument("result_file", nargs="?", default="", help="raw JSON 结果文件路径")
    parser.add_argument("--tag", default="", help="输出标识")
    args = parser.parse_args()

    if not args.result_file:
        # 找最新的结果文件
        output_dir = "experiment/outputs"
        candidates = [f for f in os.listdir(output_dir) if f.endswith("_raw.json")]
        if not candidates:
            print("未找到结果文件。使用: uv run python -m experiment.analyze <path>")
            return
        args.result_file = os.path.join(output_dir, sorted(candidates)[-1])
        print(f"自动选择最新结果: {args.result_file}")

    tag = args.tag or os.path.basename(args.result_file).replace("_raw.json", "")
    print(f"加载: {args.result_file}")
    raw = load_results(args.result_file)
    print(f"场景数: {len(raw)}")

    summary = compute_metrics(raw)

    report = generate_report(raw, summary, tag)

    report_path = os.path.join("experiment/outputs", f"{tag}_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
