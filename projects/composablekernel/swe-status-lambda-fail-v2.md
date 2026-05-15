# (λ) Concept Fail — (h) hybrid 也撞墙，方向决策点 #2

**写作时间**: 2026-05-06
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1)

---

## 1. (h) hybrid case A 结果

| | step 1 (K only) | step 4 (K+V) | (h) hybrid (K+V_async B1) |
|---|---|---|---|
| valid | n | n | **n** |
| max abs err | 23.72 (V garbage attractor) | 0.37 | **0.36** |
| pct wrong | 100% | 98.7% | **97.6%** |
| out vs ref | -22.5 全等 garbage | 0.16~0.73 散落 | **0.37~0.81 散落** |
| pattern | 一致 garbage | element permute | element permute |
| K box_dim | (8,2)→(32,16) ✓ | 同 | 同 |

→ K 单独 trivial tile-major 也 element permute。**V revert 没救回**。

## 2. 根因 (mentor confirm + 我推)

`MakeKRegTileDistribution` 用 `WarpGemm::BWarpDstrEncoding` (wmma 16×16×32 B operand)：
- WMMA B operand thread mapping 是 **hardware-fixed lane pattern** (类似 ds_load_tr 8-bank 但 wmma instruction 内部)
- ds_load with BWarpDstrEncoding thread-i 期望读 LDS[bwarp_pattern(thread_i, ij)] — **不是 plain row-major contiguous**

但 K trivial tile-major TDM write → plain row-major LDS。
→ ds_load thread-i 读到 wmma-permuted 位置但 LDS 是 row-major layout → mismatch (跟 V trload 撞墙完全同性质)

**B1 K dist 5D `<N0,N1,N2,K0,K1>` 也是反 engineer 出来匹配 wmma B operand pattern 的**（跟 V 5D 同性质 reverse-engineered）。我们 step 1 改 trivial tile-major 是把这个 reverse-engineered 关系打破了。

## 3. (λ) 路径基础 invalidate

mentor 之前 sign-off (λ) 基于 GEMM v1 trivial tile-major design 作为 reference。但 **GEMM v1 自己 build-gfx1250 也 0 instance** (decision-B2-H3-direction:46-47 mentor 自己写过 "GEMM TDM 在 build-gfx1250 也 0 instance — 这意味着 GEMM v1 的 dist co-design 设计对**也未真跑通过**")。

→ GEMM v1 trivial tile-major 是 **untested design**，没有 known-good runtime reference。fmha mirror 它撞同样 mismatch 墙是合理的 — GEMM 没 enable 没 trigger 而已。

**(λ) 整体 invalidate**：mentor 自己承认是 methodological error — sign-off 基于 untested reference design。

## 4. Sequence design 错误 (反思)

step 1 sequence design 错：
- step 1 实际状态：K trivial tile-major + V 仍 TDM (load_tile_tdm with B1 5D dist)
- V 仍走 TDM 也错位 → garbage attractor max err 23.72 dominate
- → K 单独的 element permute 被 V garbage overshadow，**看不出 K 也错**
- 应该 step 1 就让 V 走 B1 async_load (验证好的) 让 K mismatch 单独暴露 — 这样早 5 hours 发现 (λ) concept fail

## 5. 选项 reassess (mentor + SWE 一起)

**(λ-revised)** 反向 engineer wmma operand thread mapping → 设计 thread-permuted dist 让 box-major write align：
- 5D dist (B1 verified) 是 reverse engineering 结果，但 thread footprint fragmented (每 thread 8 elements 散落)，**TDM box-major 跟 5D 不兼容**
- 需要新 dist family — 同时满足 (a) thread footprint 是 contiguous box (b) 写出 wmma-aligned LDS layout
- 这俩约束可能没解 — wmma B operand thread mapping 是 lane 内 4×4 几何 block，不是 contiguous box
- 若有解，工作量未知 (>1 周 design + verify)
- **HIGH RISK**

**(μ)** K + V 都加 LDS shuffle stage：
- K shuffle: TDM trivial write → LDS reshape → ds_load with reg dist
- V shuffle: TDM trivial write → LDS reshape → ds_load_tr 走原路径
- 工作量 K + V 双 shuffle，比之前估的 (V only shuffle) 大
- perf 退步：每 K 和 V 一次都加 LDS round-trip + sync，**可能比 B1 async_load 还慢**
- TDM 收益 vs shuffle 开销 净结果未知，可能 zero gain or 退步
- **MEDIUM RISK + perf 不确定**

**(ν)** 战略撤退到 B1：
- K + V 都 revert 回 B1 (5D dist + async_load + ds_load_tr)
- AICK-579 partial done — B1 21-32% 加速 stays
- B2 (TDM intrinsic 替换) **未达成**记 future work
- (X+) CK core fix (5 处 in-place tuple ops) 仍价值大，独立 PR
- (α') kernel group path unmerge typo fix 已 PR #6964 (用户管)
- **ZERO RISK**，known-good

## 6. mentor 推 (ν)

理由：
1. (λ) invalidate — methodological error
2. (μ) perf 不确定 — TDM 收益 vs shuffle 开销 净结果未知
3. B1 已 21-32% 加速 verified — AICK-579 主要 deliverable 已达
4. TDM intrinsic 替换 is nice-to-have 不是 must-have；撞硬件 mismatch 墙后强行做不值
5. GEMM v1 自己 untested — fmha 不应该当 GEMM 的 verification ground (work 量逆序)
6. (X+) CK core fix 独立有价值，无论 B2 方向

## 7. Worktree 状态

不动等 lead/用户决策。8 modified:
- `CMakeLists.txt` (lead plumbing)
- `example/ck_tile/01_fmha/CMakeLists.txt` (lead plumbing)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` (lead plumbing)
- `include/ck_tile/core/container/container_helper.hpp` (X+ CK core latent fix — keep 不论方向)
- `include/ck_tile/core/container/tuple.hpp` (X+ CK core latent fix — keep 不论方向)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` (α' latent fix — already PR #6964; +B2 dead code kIsTdmPipeline + has_kIsTdm_trait)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (B2 主线 K/Q TDM + step 1 K trivial tile-major + dead code kIsTdm trait + V revert to B1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` (B2 主线 padding helpers + step 1 K trivial tile-major + V revert to B1 + V padding rename _disabled)

## 8. 决策操作 (按方向不同)

**如 (ν) 撤退**:
- pipeline.hpp: K dram dist revert 回 B1 5D + revert K box_dim print + revert K/Q load 调用从 load_tile_tdm 回 async_load_tile + revert K/Q sync 从 s_wait_tensorcnt 回 block_sync_lds_direct_load
- policy.hpp: revert K/Q/V padding helpers (B2 主线整段) + revert MakeKDramTileDistribution
- kernel.hpp: revert kIsTdmPipeline + has_kIsTdm_trait dead code
- pipeline.hpp: revert kIsTdm trait + tdm_config_q/k/v declarations
- 工作量 1-2 hours
- 跑 ABC 应跟 B1 一致 valid:y + 21-32% 加速

**如 (μ)**:
- mentor + SWE trace ds_load + ds_load_tr_b128 thread mapping pattern
- 设计 K/V 各自 shuffle source/dest dist
- pipeline 加 K + V 各一段 LDS shuffle (TDM write → ds_load + ds_store reshape → ds_load[_tr] consume)
- 工作量 2-3 SWE-day + verify perf
- 跑 ABC verify functional + perf vs B1

**如 (λ-revised)**:
- mentor + SWE 全新 design dist family 同时满足 contiguous box write + wmma-aligned LDS
- 工作量 unknown (>1 周)，可能没解

## 9. 待用户决策

> B2 进展：(h) hybrid 也 valid:n max err 0.36 element permute pattern — 跟 step 4 同性质。**K trivial tile-major 单独打开也撞 hardware mismatch** (TDM box-major write vs ds_load with wmma operand thread mapping 物理上不可同时满足 with simple dist 改造)。mentor 之前 (λ) sign-off 基于 GEMM v1 trivial tile-major design 但 **GEMM v1 自己 build-gfx1250 也 0 instance untested design**，invalidate 整个 (λ) 路径基础。
>
> 三选项：
> - **(λ-revised)** 反向 engineer wmma thread mapping 设计新 dist (HIGH RISK + >1 周 + 可能没解)
> - **(μ)** K + V 双 shuffle stage (MEDIUM RISK + 2-3d + perf 退步可能)
> - **(ν)** 撤退到 B1 (ZERO RISK，AICK-579 半完成 B1 21-32% 加速 stays)
>
> mentor 推 (ν)。等用户拍。
>
> (X+) CK core fix (5 处 in-place tuple ops) 独立 PR 仍 keep 不论方向 — fix 真 bug，价值独立于 B2。

---

file: swe-status-lambda-fail-v2.md
