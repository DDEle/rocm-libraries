# Step B2 Await-User Final Status (worktree cleaned)

Lead 让暂停 trial，等用户拍 (λ/μ/ν)。worktree 已整理，所有 diagnostic 改动全清。

## Worktree 当前 8 modified（git diff stat）

| 文件 | 改动 | 性质 |
|---|---|---|
| `CMakeLists.txt` | +14 -1 | 🟦 lead plumbing (fast 编译) |
| `example/.../01_fmha/CMakeLists.txt` | +20 -5 | 🟦 lead plumbing |
| `example/.../01_fmha/codegen/ops/fmha_fwd.py` | +6 -3 | 🟦 lead plumbing |
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` | +77 -25 | 🟢 B2 真修复 (TDM intrinsic 替换 + sync) |
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` | +123 -0 | 🟢 B2 真修复 (GetLdsPaddingConfig Q/K/V) |
| `include/.../fmha/kernel/fmha_fwd_kernel.hpp` | +18 -7 | 🟡 (α') latent bug fix (group path unmerge typo) |
| `include/.../core/container/container_helper.hpp` | +44 -15 | 🟡 (X+) latent bug fix (inclusive_scan 跟 mixed tuple) |
| `include/.../core/container/tuple.hpp` | +20 -8 | 🟡 (X+) latent bug fix (4 in-place tuple ops) |

🟦 lead 加，不动；🟢 B2 真修复主线；🟡 mentor confirmed latent bugs (跟 B2 主线独立)

## 已清理（diagnostic 改动）

| 改动 | 原文件 | 清理状态 |
|---|---|---|
| D1 ~ Hδ-1 CK_PRINT diagnostic (各阶段 5 处) | pipeline / kernel / tile_window | ✅ 全清 |
| H4' policy short-circuit + _orig backup (3 函数) | tdm_policy.hpp | ✅ 删 short-circuit, _orig 重命名为正式 function |
| H4' pipeline Xor=true → false (4 处 K/V LDS read view) | tdm.hpp | ✅ 全恢复 Xor=true |
| (η) helper get_real_global_strides_for_tdm + 3 caller 切换 | tile_window.hpp | ✅ 函数删, caller 切回 cached → tile_window.hpp 现为 unmodified |

## B2 trial chain 总览

| stage | hypothesis | outcome |
|---|---|---|
| Step A | 验证 WIP 脚手架 | ✓ baseline ABC pass，qr_tdm 编译 fail (distribution bug) |
| **Step B1** | 重写 K/V dist 用 async_load 跑通 | ✓ ABC pass + perf -21~32% (worktree main 已 commit) |
| Step B2 (TDM intrinsic 替换) | 把 async_load_tile → load_tile_tdm | 🔴 ABC fail，trial chain ↓ |
| α' | kernel group path unmerge `H/T/A=0` typo | ✓ confirm 修对 (D4)，独立 latent bug |
| X+ | CK core in-place tuple ops 5 处不支持 mixed | ✓ 编通了，独立 latent bug |
| H1 padding 单位 | disable padding 看是否变好 | ✗ 无变化 |
| H4' LDS swizzle | disable Xor 看是否对 | ✗ 数值跟 X+ 几乎同 |
| H8 (mentor) cached_global_strides | (η) bypass 给真 stride | ✗ runtime 16-digit 同 cached |
| Hβ TDM 不读 stride | hardcode {99999, 99999} | ✓ SIGSEGV → TDM 真读 stride[0] |
| Hα stride 真值同 cached | calculate_offset 算的真 stride 跟 cached cumulative 在 [0] 同 (=128) | ✓ confirm cached helper 在 [0] 巧合对 |
| Hδ-1 box_dim 错 | print K/V box_dim | ✗ K=(8,2)/V=(8,8) 几何对 |
| **H3 升级版** (mentor 90%) | TDM box-major LDS write vs B1 dist (async_load 设计) ds_load read 不兼容 | ⭐ **strong confirm** root cause |

## 升级用户决策的 3 个 fix 方向 (λ/μ/ν)

详见 `swe-status-174720.md`。摘要：
- **(λ)** 重设 fmha K/V dist (1-2 SWE-day, B1 教训"动 dist 风险高")
- **(μ)** 加 LDS shuffle 中间层 (~半 SWE-day, perf 退步可能让 TDM 收益归 0)
- **(ν)** 战略撤退 B2 回 async_load (~1 hour, AICK-579 主目标 B1 -21~32% 加速已达)

SWE 提议 (ν)。

## 等用户拍

- (ν) 撤退：撤回当前 8 modified 中 🟢 B2 真修复 2 处；🟡 (α'+X+) 独立 PR 流程不变
- (λ/μ) 继续做：保留 worktree 现状，开始新设计

按 lead 暂停指示，**不主动派 QA / mentor 任何新 verify task**。等用户决策后由 lead 派下一步。

## 历史 audit (status 文件链)

- `swe-status-155317.md` D 进展 + 对照困境
- `swe-status-155722.md` D1 confirm + 提议 X
- `swe-status-160732.md` D2 反证 mentor，根因转 make_tile_window
- `swe-status-161300.md` D3 100% 锁定 (但 batch path 有第二个引入点)
- `swe-status-163011.md` α' 必要不充分 — 复活 X
- `swe-status-163318.md` Q vs K stacktrace 反转
- `swe-status-163855.md` D5 disambig — case D5-b
- `swe-status-165446.md` X 修对了但需扩 scope
- `swe-status-165650.md` scope check 完 5 处
- `swe-status-174720.md` H3 升级版升级 lead/用户拍 λ/μ/ν
- `swe-mentor-sync-155955.md` 早期 mentor sync K constant<0> 假说
- **本文档**: `swe-status-final-B2-await-user.md`

---
file: swe-status-final-B2-await-user.md
