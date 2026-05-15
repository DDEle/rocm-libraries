# (λ) Pivot — V 撞 hardware-pattern 墙，C/h 选择

**写作时间**: 2026-05-06
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1)

---

## 1. Step 1 (K) + Step 4 (V) 实施结果

| | step 1 (K only) | step 4 (K+V) |
|---|---|---|
| K box_dim | (8,2) → **(32,16)** ✓ | (32, 16) ✓ |
| V box_dim | n/a | (8,8) → **(128, 16)** ✓ |
| valid (case A) | n | n |
| max abs err | 23.72 (V garbage attractor) | **0.3720703** (量级对) |
| pct wrong | 100% (130944) | 98.72% (129272) |
| out sample | 全等 -22.5 | **0.16~0.73 散落** vs ref 0.49~0.51 |
| pattern | 一致 garbage | element permutation |
| walltime | 3.4s | 3.0s |

→ K 走通 TDM-native (box_dim deterministic verify)，V 改后 magnitude + sign 都对，但 element permutation 错位。

## 2. V 撞墙根因 (mentor confirmed)

**`ds_load_tr_b128` 跟 plain row-major LDS 不兼容**。

证据：B1 V async_load 用 5D dist `<KPerThread, NumWarps, KThreadPerWarp>, <NThreads, NPerThread>` 写出 thread-permuted LDS layout 才让 ds_load_tr_b128 读对。**5D dist 是反向 engineer 出来匹配 ds_load_tr hardware fixed pattern 的结果**。

TDM trivial tile-major box-major write 出 plain row-major LDS → 跟 ds_load_tr_b128 期望的 thread-permuted layout mismatch。

## 3. 选项 reassess (mentor + SWE)

| | 描述 | 工作量 | upside | downside | mentor verdict |
|---|---|---|---|---|---|
| **(A)** | 改 V reg dist align trivial tile-major write | — | — | 破 PV GEMM B operand wmma layout | ❌ |
| **(B)** | 改 V dram dist 让 write 出 ds_load_tr 期望 thread-permuted layout | — | — | TDM box-major + thread-permuted **物理上不可同时满足**，5D dist box=(8,2) 类的 fragmented footprint 跟 box-major 假设矛盾 | ❌ |
| **(C)** | μ LDS shuffle stage：TDM 写 plain LDS A → ds_load + ds_store reshape → LDS B (ds_load_tr-friendly) → 走原 ds_load_tr | 1-2 SWE-day 起 | 形式上完成全 TDM | perf 不确定（额外 LDS BW + sync）；trace ds_load_tr pattern 复杂；ffm_lite 看不出 perf | feasible 但贵 |
| **(h)** | hybrid K-TDM + V-async_load (B1 verified path) | 0.5 SWE-day | V 走已知 good path zero-risk；K 拿 TDM 收益；sync 混合不复杂（独立 counter） | AICK-579 部分完成 (V 不是 TDM) | **mentor 推** |

## 4. mentor 推 (h) 不推 (C) 的理由

1. (C) shuffle 工作量比 K (λ) 还大：要 trace ds_load_tr_b128 thread mapping → 设计 shuffle source/dest dist → verify shuffle 正确 + perf 不至于完全抵消
2. (C) perf 不确定：额外 LDS BW + sync 可能让 V 整段比 B1 async_load 还慢；ffm_lite functional sim 看不出来
3. V async_load B1 已 verified 21-32% 加速 (vs qr baseline)，known-good path
4. (h) sync 混合 ≠ 复杂：`s_wait_tensorcnt<N>` (TDM K) 跟 `async_waitcnt<N>` (async V) 是独立硬件 counter
5. AICK-579 spec "FWD load 走 TDM" — K 走 TDM 是 partial 完成，V 留 follow-up (BWD / paged scenarios 扩 TDM coverage 时一起做)

## 5. (h) 实施细节 (如 lead 拍 (h))

- pipeline V 段：revert step 4 改动 (V dram dist + V LDS view conditional + load 调用)
- 保持: K 走 TDM (`load_tile_tdm` + tdm_config_k + s_wait_tensorcnt)
- V 改回: `async_load_tile(v_lds_write_window, v_dram_window)` + `block_sync_lds_direct_load<v_vmem_insts>()` + `load_tile_transpose(v_lds_read_window)`
- V dram dist revert 回 B1 5D `<KPerThread, NumWarps, KThreadPerWarp>, <NThreads, NPerThread>`
- V LDS swizzle 等保持 B1 状态
- kernel.hpp v_dram_window IIFE conditional dispatch revert (V 不再走 TDM dual view)
- pipeline.hpp static_assert revert 回 (kN1, kK1) 顺序
- pipeline class kIsTdm trait 保留 (K 仍走 TDM dual view 不影响)

预估改动 5-8 处 revert。0.5 SWE-day。

## 6. (C) 实施细节 (如 lead 拍 (C))

- pipeline V load 段加 LDS shuffle：
  1. trivial tile-major TDM write to V LDS A (现状 step 4)
  2. `s_wait_tensorcnt_barrier<0>`
  3. ds_load (with 5D-like dist mirror B1) 从 LDS A 拿 register tile
  4. ds_store with 5D-like dist 写到 LDS B
  5. `block_sync_lds()`
  6. `load_tile_transpose(v_lds_read_window from LDS B)` 现状
- LDS smem 可能需要扩 (A + B 两份 buffer，~32 KB) 或 reuse same physical 但要确保 in-place reshape 正确
- mentor + SWE 一起 trace ds_load_tr_b128 LDS pattern 1-2 day

## 7. Worktree 状态

- HEAD: `16324795215` (B1)
- Modified (unstaged) — 加上 step 1 + step 4 改动总计 ~8 文件:
  - `CMakeLists.txt` (lead plumbing)
  - `example/ck_tile/01_fmha/CMakeLists.txt` (lead plumbing)
  - `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` (lead plumbing)
  - `include/ck_tile/core/container/container_helper.hpp` (X+ CK core latent fix)
  - `include/ck_tile/core/container/tuple.hpp` (X+ CK core latent fix)
  - `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` (α' latent fix + λ-2 V dual view dispatch)
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (B2 主线 + step 1 + step 4)
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` (B2 主线 + step 1 + step 4)
- 不 commit，等用户决策

不论 lead 拍 (C) 还是 (h)，K 改动都保留；V 部分按 lead 决策决定 keep / revert / 加 shuffle。

## 8. 待 lead 决策

> B2 (λ) K dist 走通 (box_dim 已 verify 改进 binary, max err 23.72→0.37 量级对)；V trivial tile-major + ds_load_tr 撞 hardware-pattern 不兼容墙（B1 5D dist 是反 engineer 出来匹配 trload pattern 的结果，TDM box-major write 物理上不能同时满足）。
>
> 两条路:
> - **(C)** 加 LDS shuffle stage 完成全 TDM (perf 不确定 + 工作量 1-2d，需 trace ds_load_tr pattern)
> - **(h)** hybrid K-TDM + V-async_load (V 走 B1 verified path, AICK-579 半完成, 工作量 0.5d revert)
>
> **mentor 推 (h)**，等用户拍。

---

file: swe-status-lambda-pivot.md
