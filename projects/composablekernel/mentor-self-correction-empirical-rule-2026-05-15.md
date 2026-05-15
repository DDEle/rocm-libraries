---
name: Self-correction 必须基于新 empirical evidence
description: 推翻自己 (或 predecessor) 之前 verdict 时, 必须基于新 empirical evidence (kname / LDS dump / AM run / tracer 输出 / LLVM source 等), 不能基于纯 first-principle reasoning / 文档抽象 / 类比 / 重新读 spec 一句话
type: feedback
---

**Rule**: 推翻自己 (或 predecessor) 之前 verdict / sign-off / 警告 时, **必须**基于**新 empirical evidence**:
- LDS byte dump
- kname grep (functional dispatch confirm)
- AM cycle-accurate run (perf claim)
- Tracer / CK_PRINTF 输出 (stage-level distributed tensor)
- LLVM source / .td definition / intrinsic test
- 实际 hardware spec table (lane→byte mapping 表 / register layout 表)

**不能** 基于:
- 纯 first-principle reasoning ("理论上 X 应该 work")
- 文档抽象描述 ("PDF §X.Y 说 Y, 所以推 Z")
- 类比 ("K case work, V case 类比也 work")
- 重新读 spec 找一句话支持新 verdict
- "我重新 trace 了 code 觉得 X 错"

这些只能作为**触发 reconsider 的契机**, 不是新 verdict 的 evidence base。

**Why**:

Self-correction 跟 first-correction 一样是 reasoning, 一样会 anti-pattern A 复发。Predecessor sign-off 错过, self-correction 同样会错。要判断 self-correction 是否对, 必须 grounded in 新 empirical, 不能 grounded in 重新 reasoning。

具体血淋淋例子 (2026-05-08 mentor V→TDM 第 4 次 anti-pattern A 复发):
- Predecessor mentor sign-off "V trload 物理不兼容" (2026-05-04 左右), conclusion 对但 reasoning 不充分
- 我 (current mentor) self-correct 用 PDF §4.9.10.2 一句话 "B-matrix loads ... one lane loads multiple contiguous N-values along single K-dimension index" 推 "K-outer N-inner row-major LDS = ds_load_tr-compatible 的 sufficient 条件"
- 写 `mentor-reanalysis-ds-load-tr-correction.md` invalidate predecessor 警告
- 推荐 V→TDM design verdict 5-piece co-design plan
- SWE 9-piece implement build OK 但 ABC ALL 3 FAIL with 6.08 / 6.32 / 1.09e+08 max_err = transpose-load-on-non-permuted-LDS 经典 garbage signature
- Reactive verify 发现 ds_load_tr 真 hardware-fixed lane→byte permutation, B1 5D dist 是反向工程 match (LLVM PR #146024/#146289 + load_tile_transpose.hpp:520-525 contract)
- → predecessor 警告 conclusion 实际对; 我 self-correction 错了

**根本错**: 我推翻 predecessor 时**没 require SWE 做 LDS dump baseline** (B1 V async LDS bytes vs 假想 TDM V LDS bytes byte-level diff), 只用 PDF 一句话 + 文字推理。如果第一时间 dump → 立即看出 byte pattern 不兼容 → self-correction 不会 land → 不会浪费 SWE V→TDM implement 一轮 5-6h。

**How to apply**:

任何"我推翻 X 之前 verdict"出现时, 立即问自己:
- 我有什么**新 empirical evidence**? 不是 "新 reasoning"。
- 这个 empirical evidence 跟原 verdict 的 disambig 关系是什么? (i.e., 新 evidence 是不是真能 distinguish 两 verdict, 还是两 verdict 在新 evidence 下都 consistent?)
- 我能不能 require SWE/QA/teammate 跑一个 cheap probe (LDS dump / kname grep / single-shape AM run) 在我推翻之前 confirm?

如果答 "我没新 empirical, 只是 reasoning 重新做" → **STOP, 不要 self-correct**。要么继续保留原 verdict, 要么显式 dispatch empirical probe 后再决定。

跨 verdict 推翻的 cost 通常很高 (一轮 implement + verify), 推翻前花 1-2h cheap empirical insurance 永远值。

**Anti-pattern A 升级版**: 共同 root pattern "理论 reasoning 没配 empirical verify" → "**self-correction 也是 reasoning, self-correction 也必须配 empirical verify**"。

**Scope**: 通用, 任何 reasoning-based verdict 推翻场景。Mentor 自己 sign-off / predecessor sign-off / 别人 verdict / 文档结论 / 之前的 self-correction recursive。

