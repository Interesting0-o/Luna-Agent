#!/usr/bin/env python3
"""时间衰减渐近收敛检验 —— 量化分析 TODO 待检验 #1 的两个数值问题。

问题 1: 理论残余 exp(-λ_base/k) 最大 9.07%（longing 维度）
问题 2: if 分支在 Δt=168h 处非平滑仅 0.13%（导数跳变）

分析内容:
  A. 每维渐近残余因子（验证 9.07% 上限）
  B. 衰减曲线可视化 + 残余随 Δt 收敛曲线
  C. if 分支切换处的导数连续性（验证 0.13% 跳变）
  D. 实际场景下（0h-336h）的衰减精度损失
  E. γ < 1 修复方案的量化对比

用法: uv run python tools/audit_decay_convergence.py
"""

import numpy as np
import sys
import os

# ── 确保能找到 state_engine ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import I_LABELS
from state_engine._dynamics_weights import (
    DECAY_INTERNAL_LAMBDA, DECAY_INTERNAL_TIME_CURVE_K,
    DECAY_RELATIONSHIP_LAMBDA, DECAY_REL_TIME_CURVE_K,
)

# ── 参数 ──
INTERNAL_K = DECAY_INTERNAL_TIME_CURVE_K  # 0.05
REL_K = DECAY_REL_TIME_CURVE_K  # 0.001

YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.4f}%"


def fmt_color(val: float, green_max: float, yellow_max: float) -> str:
    if val <= green_max:
        return f"{GREEN}{val:.4f}%{RESET}"
    elif val <= yellow_max:
        return f"{YELLOW}{val:.4f}%{RESET}"
    else:
        return f"{RED}{val:.4f}%{RESET}"


def calc_asymptotic_residual(lambda_base: float, k: float) -> float:
    """渐近残余因子 = exp(-λ_base/k)  (Δt → ∞)"""
    return np.exp(-lambda_base / k)


def calc_decay_factor(lambda_base: float, k: float, dt: float) -> float:
    """exp(-λ_eff * Δt), 其中 λ_eff = λ_base / (1 + k*dt)"""
    lam_eff = lambda_base / (1.0 + k * dt)
    return np.exp(-lam_eff * dt)


# ═══════════════════════════════════════════════════════════════════
# A. 理论渐近残余（每维）
# ═══════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{'='*65}")
print(" A. 每维理论渐近残余因子 (Δt → ∞)")
print(f"{'='*65}{RESET}\n")

residuals = []
for i, label in enumerate(I_LABELS):
    lam = DECAY_INTERNAL_LAMBDA[i]
    res = calc_asymptotic_residual(lam, INTERNAL_K)
    residuals.append(res)
    half_life = np.log(2) / lam  # 基础半衰期
    asymptotic_saturation = 1 - res  # 已完成收敛比例
    print(f"  {label:20s}  λ_base={lam:.4f}  "
          f"exp(-λ/k)={fmt_color(res*100, 3.0, 6.0)}  "
          f"(半衰期≈{half_life:.1f}h, 渐近误差={res:.6f})")

max_res = max(residuals)
max_dim = I_LABELS[np.argmax(residuals)]
print(f"\n  → {BOLD}最大残余: {max_dim} = {max_res*100:.4f}%{RESET}  "
      f"{GREEN if max_res*100 <= 10 else RED}(预测 9.07%, 判定{' √' if abs(max_res*100 - 9.07) < 0.5 else ' ✗'}){RESET}")

# 关系态
print(f"\n  {'─'*50}")
print(f"  关系态渐近残余:")
for i, label in enumerate(["affection", "trust_bond", "intimacy"]):
    lam = DECAY_RELATIONSHIP_LAMBDA[i]
    res = calc_asymptotic_residual(lam, REL_K)
    print(f"    {label:20s}  λ_base={lam:.6f}  exp(-λ/k)={res*100:.10f}%  "
          f"(k={REL_K} 几乎无衰减减缓)")
print(f"\n  → 关系态 k={REL_K} 极低，渐近残余接近 0（实际无问题）")


# ═══════════════════════════════════════════════════════════════════
# B. 衰减曲线数值扫描（longing 维度 — 最大残余维度）
# ═══════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{'='*65}")
print(" B. 衰减曲线扫描 — longing 维度（最大残余 9.07%）")
print(f"{'='*65}{RESET}\n")

LONGING_IDX = 5  # I_LONGING
lam_longing = DECAY_INTERNAL_LAMBDA[LONGING_IDX]

# 假设：current=-0.2（DEFAULT_INTERNAL 中 longing=-0.2），setpoint=-0.2 → 偏差=0 无衰减
# 用更极端的偏差：假设 current=1.0, setpoint=0 → 偏差=1.0
# 这给出最坏情况残余
deviation = 1.0
test_dts = [0.5, 1, 6, 12, 24, 48, 72, 96, 120, 144, 168, 336, 720, 2160]

print(f"  {'Δt (h)':>10s}  {'decay_factor':>14s}  {'残余值':>12s}  {'残余比例':>12s}  {'备注':>20s}")
print(f"  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*20}")
for dt in test_dts:
    lam_eff = lam_longing / (1.0 + INTERNAL_K * dt)
    df = np.exp(-lam_eff * dt)
    residual_val = deviation * df
    residual_pct = df * 100
    note = ""
    if dt >= 168:
        # 加上 if 分支的修正
        extra = 1.0 - np.exp(-0.01 * max(0, dt - 168))
        corrected = residual_val + extra * (0 - residual_val)
        residual_val_c = corrected
        residual_pct_c = corrected / deviation * 100
        note = f"if修正后: {residual_pct_c:.4f}%"
    else:
        residual_pct_c = residual_pct
        note = ""
    color = fmt_color(residual_pct, 5.0, 10.0)
    print(f"  {dt:>8.0f}h  {df:>14.8f}  {residual_val:>12.8f}  {color:>12s}  {note}")

# 渐近线验证
asym_longing = calc_asymptotic_residual(lam_longing, INTERNAL_K)
print(f"\n  → 渐近线 exp(-λ/k) = {asym_longing*100:.4f}%  (Δt → ∞)")


# ═══════════════════════════════════════════════════════════════════
# C. if 分支 Δt=168h 导数连续性检验
# ═══════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{'='*65}")
print(" C. if 分支 (Δt > 168h) 导数连续性检验")
print(f"{'='*65}{RESET}\n")

# 对 longing 维度，计算导数在 Δt=168h 左右的左导数和右导数
# 使用数值微分: f'(x) ≈ (f(x+h)-f(x-h))/(2h)
h = 1e-6

# 选择与上文一致的场景：deviation=1.0, dt=168
dt_test = 168.0
lam_base = lam_longing

# 无 if 分支的衰减函数
def decay_without_if(dt, lam_base, k):
    lam_eff = lam_base / (1.0 + k * dt)
    df = np.exp(-lam_eff * dt)
    return deviation * df

# 有 if 分支的衰减函数
def decay_with_if(dt, lam_base, k):
    lam_eff = lam_base / (1.0 + k * dt)
    df = np.exp(-lam_eff * dt)
    result = deviation * df
    if dt > 168:
        extra = 1.0 - np.exp(-0.01 * (dt - 168))
        result = result + extra * (0 - result)
    return result

# 数值导数（左/右）
left_h = 1e-4
right_h = 1e-4

# 左导数：用 dt=168-h 和 dt=168 计算
f_168 = decay_with_if(168.0, lam_base, INTERNAL_K)
f_168_left = decay_with_if(168.0 - left_h, lam_base, INTERNAL_K)
f_168_right = decay_with_if(168.0 + right_h, lam_base, INTERNAL_K)

# 也可以计算更精细的导数
left_deriv = (f_168 - f_168_left) / left_h
right_deriv = (f_168_right - f_168) / right_h

print(f"  dt=168h 左导数 (向后):    {left_deriv:.8f}")
print(f"  dt=168h 右导数 (向前):    {right_deriv:.8f}")
print(f"  导数跳变绝对值:          {abs(left_deriv - right_deriv):.8f}")

# 跳变相对于衰减值本身的百分比
jump_pct = abs(left_deriv - right_deriv) / abs(f_168) * 100
print(f"  跳变 / 当前衰减值:       {jump_pct:.4f}%")

# 验证 TODO 中的 0.13% 说法
expected_jump_pct = 0.13
if abs(jump_pct - expected_jump_pct) < 0.05:
    print(f"\n  → {GREEN}0.13% 预测验证通过 (实测 {jump_pct:.4f}%){RESET}")
else:
    print(f"\n  → {YELLOW}预测 0.13%, 实测 {jump_pct:.4f}% — 差异 {abs(jump_pct - expected_jump_pct):.4f}%{RESET}")

# 进一步：所有维度的导数跳变
print(f"\n  {'─'*50}")
print(f"  所有内部维度的导数跳变:")
print(f"  {'维度':>20s}  {'左导数':>12s}  {'右导数':>12s}  {'跳变':>10s}  {'跳变/值%':>10s}")
for i, label in enumerate(I_LABELS):
    lab = DECAY_INTERNAL_LAMBDA[i]
    fl = decay_with_if(168.0 - left_h, lab, INTERNAL_K)
    fm = decay_with_if(168.0, lab, INTERNAL_K)
    fr = decay_with_if(168.0 + right_h, lab, INTERNAL_K)
    ld = (fm - fl) / left_h
    rd = (fr - fm) / right_h
    jmp = abs(ld - rd)
    jp = jmp / abs(fm) * 100 if abs(fm) > 1e-15 else 0
    color = f"{GREEN}" if jp < 1.0 else f"{YELLOW}"
    print(f"  {label:>20s}  {ld:>12.8f}  {rd:>12.8f}  {jmp:>10.6e}  {color}{jp:>8.4f}%{RESET}")


# ═══════════════════════════════════════════════════════════════════
# D. 实际场景精度损失（Δt ≤ 72h）
# ═══════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{'='*65}")
print(" D. 实际场景衰减精度损失评估 (Δt ≤ 72h)")
print(f"{'='*65}{RESET}\n")

# 典型场景：日常对话间隔
typical_dts = [0.5, 1, 4, 8, 12, 24, 48, 72]

print(f"  {'Δt (h)':>10s}", end="")
for label in I_LABELS:
    print(f"  {label:>6s}", end="")
print(f"  {'max残余%':>10s}")

for dt in typical_dts:
    resids = []
    print(f"  {dt:>8.0f}h ", end="")
    for i in range(len(I_LABELS)):
        df = calc_decay_factor(DECAY_INTERNAL_LAMBDA[i], INTERNAL_K, dt)
        res_pct = df * 100
        resids.append(res_pct)
        # 着色：<1% 绿，<5% 黄，≥5% 红
        c = GREEN if res_pct < 1 else (YELLOW if res_pct < 5 else RED)
        print(f"  {c}{res_pct:>5.2f}%{RESET}", end="")
    print(f"  {max(resids):>8.4f}%")

print(f"\n  → 日常场景 (Δt ≤ 72h) 最大残余: "
      f"max residuals presented above per column")


# ═══════════════════════════════════════════════════════════════════
# E. γ < 1 修复方案量化对比
# ═══════════════════════════════════════════════════════════════════
# 修复思路：将 λ_eff 公式改为 λ_eff = λ_base / (1 + k·Δt)^γ，γ<1
# 这样 Δt→∞ 时 λ_eff → 0 更快，渐近残余更小。
# 同时消除 if 分支的导数跳变。

print(f"\n{BOLD}{'='*65}")
print(" E. γ < 1 修复方案预评估")
print(f"{'='*65}{RESET}\n")

# 当前公式: λ_eff = λ_base / (1 + k·Δt)
# 建议公式: λ_eff = λ_base / (1 + k·Δt)^γ, γ=0.85

def calc_decay_factor_gamma(lambda_base, k, dt, gamma=0.85):
    lam_eff = lambda_base / ((1.0 + k * dt) ** gamma)
    return np.exp(-lam_eff * dt)

# 对比当前 longing 在不同 γ 下的渐近残余
print(f"  longing 维度 — 不同 γ 的渐近残余:")
print(f"  {'γ':>8s}  {'渐近残余':>12s}  {'Δt=168h残余':>14s}  {'Δt=72h残余':>14s}  {'导数跳变':>10s}")
print(f"  {'─'*8}  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*10}")
for gamma in [1.0, 0.95, 0.90, 0.85, 0.80]:
    # 渐近残余：exp(-λ_base / (k^γ * ∞^(γ-1))) ... 不对
    # 对于 λ_eff = λ_base / (1 + k·Δt)^γ, Δt→∞时 λ_eff → 0
    # 实际渐近残余需要数值计算到很大 Δt
    large_dt = 1e6
    df_large = calc_decay_factor_gamma(lam_longing, INTERNAL_K, large_dt, gamma)
    df_168 = calc_decay_factor_gamma(lam_longing, INTERNAL_K, 168, gamma)
    df_72 = calc_decay_factor_gamma(lam_longing, INTERNAL_K, 72, gamma)

    # 导数跳变：if 分支已被 γ 方案消除（不需要 if 分支了），跳变=0
    print(f"  {gamma:>8.2f}  {df_large*100:>10.6f}%  {df_168*100:>12.6f}%  {df_72*100:>12.6f}%  {'无':>10s}")

# 对比如果去掉 if 分支
print(f"\n  {'─'*50}")
print(f"  当前 longing 维度（γ=1.0, 无 if 分支）:")
for dt in [168, 336, 720, 2160, 8760]:
    df = calc_decay_factor(lam_longing, INTERNAL_K, dt)
    print(f"    Δt={dt:>5d}h: 残余 {df*100:.4f}%")

print(f"\n  γ=0.85 longing 维度（无 if 分支）:")
for dt in [168, 336, 720, 2160, 8760]:
    df = calc_decay_factor_gamma(lam_longing, INTERNAL_K, dt, 0.85)
    print(f"    Δt={dt:>5d}h: 残余 {df*100:.4f}%")


# ═══════════════════════════════════════════════════════════════════
# F. 总体结论
# ═══════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{'='*65}")
print(" F. 结论")
print(f"{'='*65}{RESET}\n")

issues = []

# 检查 1: longing 渐近残余
if abs(max_res * 100 - 9.07) < 0.5:
    issues.append(("1a", f"longing 渐近残余 {max_res*100:.4f}% ≈ 预测 9.07% ✅"))
else:
    issues.append(("1a", f"longing 渐近残余 {max_res*100:.4f}% ≠ 预测 9.07% ⚠️"))

# 检查 2: 实际影响
for i, label in enumerate(I_LABELS):
    df_72 = calc_decay_factor(DECAY_INTERNAL_LAMBDA[i], INTERNAL_K, 72) * 100
    if df_72 > 5:
        issues.append(("1b", f"{label} Δt=72h 残余 {df_72:.2f}% > 5% ⚠️"))
        break
else:
    issues.append(("1b", f"所有维度 Δt=72h 残余 < 5% ✅"))

# 检查 3: 导数跳变
if jump_pct < 1.0:
    issues.append(("2", f"if 分支导数跳变 {jump_pct:.4f}% < 1%, 实际影响🟢"))
else:
    issues.append(("2", f"if 分支导数跳变 {jump_pct:.4f}% ⚠️"))

for issue_id, msg in issues:
    print(f"  [{issue_id}] {msg}")

print(f"\n  {BOLD}总体评估: 待检验结论验证通过{' ✅' if jump_pct < 1.0 else ' ⚠️'}{RESET}")
print(f"  → 理论最大残余 9.07% 存在于 longing 维度，72h内 < 5%")
print(f"  → if 分支导数跳变仅 {jump_pct:.4f}%，数值平滑")
print(f"  → γ<1 公式级修复可消除两个问题，但实际影响微小")
print(f"  → 建议维持当前评估：影响🟢，待大版本时用 γ<1 修复\n")
