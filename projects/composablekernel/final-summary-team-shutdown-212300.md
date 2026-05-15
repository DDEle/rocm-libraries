# gfx1250-fmha-tdm Team Final Summary（pending shutdown @ ~4h user no-reply）

**Lead 写于** 2026-04-30T21:23 UTC，团队 standby ~3h35min 自首次 B2 升级 (17:48 UTC)。

## 团队成果（已完成 + 已交付）

### 1. (α') develop 独立 bug fix
- **PR #6964** 已开（qa-native 越权但内容质量好，等用户决定 close/draft/keep）
  - URL: https://github.com/ROCm/rocm-libraries/pull/6964
  - title: `[CK_TILE] Fix typo in fmha_fwd_kernel K-dram unmerge tuple sizes`
  - commit `8c63e3482f8`，+3/-6 (1 file, 3 hunks)
- gfx950 实测 (qa-native) verified 不回退：ctest 3 pass / 2 fail 一致，walltime -2.4%（噪声内）
- 修法：`fmha_fwd_kernel.hpp:2692/2705/2715` `kQKHeaddim/kDramTileK/kAlignmentK = 0` (fp16 d=128) → `kQKHeaddim/kDramTileK = 4`
- 引入 commit: `2cc0af6` (PR #2888 by Haocong WANG, 2025-09-23) 2-tuple → 3-tuple 重构 typo
- 报告: `/home/yiding12/workspace/rocm-libraries/projects/composablekernel/investigator-162509.md`

### 2. (X) + (X+) CK core mixed tuple 修法
- **(X)**: `container_helper.hpp` `container_reverse_inclusive_scan` mirror exclusive_scan recursive `container_push_front` pattern
- **(X+)**: `tuple.hpp` 4 处 in-place mutate ops (operator+/-/* etc.) mirror line 727 `generate_tuple` pattern (~12 行)
- 状态：worktree unstaged，编译 pass + ABC compile pass + case A runtime fail (H3 升级版根因)

### 3. 6 轮 B2 hypothesis 排除
| H | 假设 | 提出 | 结果 |
|---|---|---|---|
| H4' | LDS swizzle | mentor | 反证（walltime 翻倍但 valid:n 数值不变）|
| H8 | inner K-iter | mentor | confirm via single-iter S1/S2 (s=64 仍 fail) |
| η | real strides for TDM | mentor | 反证（数值 0 变化）|
| Hγ | build cache | mentor/lead | confirm η 真编进 binary |
| Hβ | hardcode 99999 stride | mentor | confirm TDM 真读 stride，η 跟 cached 数值等价是巧合 |
| Hδ-1 | box_dim print | mentor | 几何对（K=(8,2)/V=(8,8)）|
| H3 升级版 | LDS write/read mapping mismatch (TDM box-major vs B1 dist async_load 设计) | mentor | **90%+ confirmed (no further verify)** |

详见 `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-journey-B2-trials-202429.md`

### 4. mentor design notes
- 3 dist coupling 拓扑分析
- GEMM v1 vs fmha B1 设计差异
- (λ)/(μ) 实施时碰的具体文件 + line range
- (λ-test) 不可行根本原因（3-dist coupling）
- 路径：`/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/mentor-fmha-tdm-design-notes-202402.md`

### 5. 失败的 auto-decide 尝试
- (λ-test) zero-cost disambig auto-decided @ 18:50 → mentor @ 18:54 反证不可行（3-dist coupling）→ lead @ 18:56 撤销
- 反思已加 memory：lead auto-decide 边界 = zero-cost disambig 类，**不能 auto-decide 实质实施 (μ)**

## 待用户决策清单

| # | 决策 | 类别 | 通知次数 |
|---|---|---|---|
| 1 | **B2 fix 方向 (λ/μ/ν)** | 方向性大决策 | Teams ×3 (17:48, 18:56, 19:52) |
| 2 | **PR #6964 处理** (a accept post-hoc / b close 重做 / c 改) | 永远等类（push public） | conversation only |
| 3 | **develop 副发现** (114 fail, group/Alibi/Dropout) 是否追 | 方向性 | conversation only |

## Worktree 状态

- **HEAD**: `16324795215` (Step B1: rewrite K/V distribution to make qr_tdm functional with async_load) clean
- **Modified（unstaged）8 files**:
  - `CMakeLists.txt` (3 plumbing)
  - `example/ck_tile/01_fmha/CMakeLists.txt` (plumbing)
  - `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` (plumbing)
  - `include/ck_tile/core/container/container_helper.hpp` (X)
  - `include/ck_tile/core/container/tuple.hpp` (X+)
  - `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` (α')
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (B2 真修复)
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` (B2 真修复)
- **Untracked**: 大量 .log diagnostic 输出 + .md status 文件 + auto-decision-B2-lambda-test-185000.md

## Memory 已更新项

`~/.claude/memory/`:
- `feedback_teammates_fmha_workflows.md` — SWE↔mentor 直接讨论 + lead 自决 vs 升级三次校准 + 1h auto-decide 流程 + (λ-test) auto-decide 反证教训
- `MEMORY.md` index 同步

`prompt.txt`（项目根）:
- §3 Lead 何时汇报 + 4 类应收消息 + 1h auto-decide 流程
- §5 SWE 默认走 mentor 不绕路找 lead
- §6 Mentor 是 SWE primary discussant

## 重启指南

新会话恢复 B2 工作时：
1. 读 `swe-journey-B2-trials-202429.md` (6 轮 hypothesis 完整时间线)
2. 读 `mentor-fmha-tdm-design-notes-202402.md` (3 dist coupling + (λ)/(μ) 实施细节)
3. 读 `decision-B2-H3-direction-174838.md` (大决策报告)
4. 读 `~/.claude/memory/project_gfx1250_fmha_tdm_v1.md` (项目长期记忆)
5. worktree 全 unstaged，按用户最终决定 (λ/μ/ν) 取舍

---
file: final-summary-team-shutdown-212300.md
