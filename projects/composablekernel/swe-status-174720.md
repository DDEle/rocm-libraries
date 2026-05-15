# Step B2 H3 升级版 confirm — 升级 lead/用户拍 fix 方向 (λ/μ/ν)

## 现状摘要

B2 trial 多轮 hypothesis 转换后定位 **H3 升级版根因**：TDM 写 LDS box-major 模式 vs ds_load 读 LDS 按 fmha B1 dist (为 async_load 设计) 期望的位置 → 一致地读到错位置 → 一致 garbage。

mentor confirm 90%+ 信心，建议升级 lead 决策不再继续 trace。

## Hypothesis chain (按时间顺序)

| stage | hypothesis | verify result |
|---|---|---|
| α' | group path unmerge typo `H/T/A=0` → `H/T=4` | ✓ confirm 修对 (D4) — 但 batch path 不走 group 这条 |
| X+ | CK core in-place tuple ops 5 处不支持 mixed tuple | ✓ confirm 编通了 — runtime 100% wrong (max err 23) |
| H1 | TDM padding 字段单位错 | ✗ disable padding 无变化 (H4') |
| H4'| LDS write naive vs read xor swizzle 不一致 | ✗ disable xor 无变化 + 数值跟 X+ 几乎一致 |
| H8 (mentor) | cached_global_strides cumulative product != real stride | ✗ (η) bypass 给真 stride，runtime 16-digit 同 cached |
| Hβ | TDM 不读 stride 字段 | ✗ hardcode {99999} 触发 SIGSEGV → TDM 真读 stride[0] |
| Hα | calculate_offset 真 stride 跟 cached 同值 (only [0] 关键) | ✓ — TDM 用 stride[0]=128，cached 跟 (η) 都给 128 |
| Hδ-1 | box_dim 错 | ✗ K box=(8,2) V box=(8,8) 几何对 |
| **H3 升级版** | TDM box-major LDS write vs B1 dist-pattern ds_load read 不兼容 | ⭐ **strong confirm** (mentor 90%+) |

## H3 升级版根因

TDM hardware DMA 写 LDS 是 box-major 顺序（按 box_dim 8×2 contiguous box per thread per call，起点 lds_coord）。

ds_load 读 LDS 按 thread dist pattern 期望 thread-i 在 LDS 位置 P。

两个 thread mapping 规则不同 → 一致地读到错位置 → garbage。

GEMM v1 通过 `BlockGemm::MakeABlockDistributionEncode()` **跟 TDM write pattern 一起设计** dist 解决。fmha B1 抄的 `MakeKDramTileDistribution` 是 async_load 历史脚手架，跟 TDM write pattern **不兼容**。

## fix 方向 (λ/μ/ν) — 求用户决策

### (λ) 重设 fmha K/V dram dist 让 write/read 对齐 TDM box 模式
- **工作量**：大；需要 design 新 dist (mirror GEMM v1 BlockGemm::MakeABlockDistributionEncode 思路) + 完整 ABC verify
- **风险**：B1 教训"动 dist 风险高"；可能引入新 functional bug
- **upside**：真正的 TDM-native fmha 实现

### (μ) 加 LDS 中间 shuffle 层
- **工作量**：中；TDM 写 box-major 后用 async_copy LDS shuffle 成 dist 期望布局，ds_load 再读
- **风险**：额外 sync + 额外 LDS BW；perf 退步可能 (TDM 优势打折)
- **upside**：dist 不动，B1 教训不踩

### (ν) 放弃 TDM B2，回 async_load
- **工作量**：撤回 B2 改动 (restore B1 状态)
- **风险**：B2 (TDM intrinsic 替换) AICK-579 部分目标未达成
- **upside**：B1 已 21-32% 加速，AICK-579 主目标 (gfx1250 fmha fwd 加速) **已达**

## 当前 worktree 状态

8 modified（unstaged）：
- 3 plumbing (lead 加)
- 2 B2 真修复 (pipeline + policy)
- 1 (α') fix (kernel group path unmerge typo) — **独立 latent bug**, task #5 走 PR 流程
- 1 (X+) fix (CK core in-place tuple ops 5 处) — **独立 latent bug**, 类似 (α') 性质
- 1 H4' diagnostic (policy padding short-circuit + pipeline xor=false 4 处) — verify 完成可回退
- 1 (η) helper (tile_window.hpp get_real_global_strides_for_tdm + 3 caller 切换) — bypass cached, 没解决问题但代码逻辑正确，可保留作 robustness
- CK_PRINT diagnostic 已全清

## SWE 提议

**走 (ν) 战略撤退**为最稳：
- AICK-579 主目标 (B1 加速 -21%~-32%) 已达
- B2 (TDM intrinsic 替换) 是 nice-to-have 优化，root cause 在 dist alignment 而非简单 fix
- (λ) 工作量太大需多 SWE-day，时间不可控
- (μ) perf 退步可能让 TDM 收益变 0，得不偿失

但**(α' + X+) 这两个独立 latent bug fix 应该独立保留** (task #5 已在走 (α') PR 流程，类似 X+ 也建议独立 PR)。

如果选 (λ) 我 own design + 实现 + ABC verify，但要预算 1-2 SWE-day。
如果选 (μ) 我 own 设计 LDS shuffle 层 (~半 SWE-day)。
如果选 (ν) 撤回 (~1 hour)。

## 等用户拍

按 lead 工作流，方向决策升级用户。

---
file: swe-status-174720.md
