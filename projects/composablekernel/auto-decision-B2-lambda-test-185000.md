# Auto-decision: B2 (λ-test) zero-cost disambig

**时间**: 2026-04-30T18:50:00Z (auto-triggered by cron after 1h window expired)
**Trigger**: Teams notify @ 17:48 UTC + 1h = 18:48 UTC; user 在 18:48-18:50 期间无回复
**Decision**: 推进 **(λ-test)** = SWE 写 trivial linear K dist + 编 + 跑 case A 看 valid:y/n

## 为什么挑 (λ-test)

按 lead 之前给用户的 conservative pick 推荐：
> 我倾向先做 zero-cost (λ-test)：SWE 写 trivial linear K dist 测一次，valid:y → confirm 根因再选 (λ)/(μ)；valid:n → H3 推翻重挖。半小时左右 vs 直接投资 (λ) 数天。

(λ-test) 是 4 个候选里：
- 工作量最小（半小时 vs (λ) 数天 / (μ) 中等 / (ν) 撤回）
- 信息量最大（valid:y/n 直接 confirm or 反证 H3 升级版）
- 完全 reversible (git tracked diagnostic)
- 不浪费 (λ)/(μ) 真实施的工作

## Override

用户随时可 override：
- 直接选 (λ)/(μ)/(ν) → lead 撤回 (λ-test) 派 SWE 走真方向
- 否决 (λ-test) → lead 改 conservative 选 (ν) 撤回 B2 回 B1
- 选别的 → 按用户

## 派给 SWE 的 task

详见 SendMessage to swe @ 18:50。完成后 SWE 写 swe-status-lambda-test-*.md 报路径。

## 同步并行项

- PR #6964 (qa-native 越权) — 永远等用户，无 auto-decide
- develop 副发现 (114 fail) — auto-decide (c) 留给 fmha team 不深挖（投资 investigator 时间不在 B2 critical path 上）

---
file: auto-decision-B2-lambda-test-185000.md
