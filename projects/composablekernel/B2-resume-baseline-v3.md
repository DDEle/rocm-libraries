# B2 Resume Baseline v3 — Post ds_load_tr Semantics Correction

**写作时间**: 2026-05-06
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1)
**目的**: Fresh team 重启后 lead 读这一份就能完整 catch up，不需要重读 trial 历史 / 全部 status doc。

---

## 1. 当前真实 State

**一句话**: B2 (λ) trial 全失败，但根因被 mentor 三次错 ds_load_tr_b128 semantics 描述误导；user/lead 读 develop production code 修正后真根因 = K reg dist 跟 dram dist 缺 coordination（不是物理不兼容），方向待 LDS dump disambig 后再定。

**详细 timeline 总结**:
1. Step 1 (λ-1) K dram dist trivial tile-major (mirror GEMM v1) → max err 23.72 garbage (mentor sign-off 错认为 K 改生效进 step 4)
2. Step 4 (λ-2) V dual view + RowMajor B trivial tile-major → max err 0.37 量级对 element permute (mentor sign-off "V trload 物理不兼容" 推 (h) hybrid)
3. (h) hybrid auto-decided (V revert B1, K 仍 trivial tile-major) → still valid:n max err 0.36 (mentor 推 (ν) 撤退说 (λ) 物理不可行)
4. **User/lead 读 develop async_trload_policy** 发现 mentor ds_load_tr semantics 三次错 → V 物理不兼容 invalidate，K 真根因 = dist coordination 缺失
5. **三方现共识**: 优先 K LDS dump disambig (~2-3h) 看 LDS layout 是不是 plain row-major 再决定方向

**Worktree status**: HEAD = `16324795215` (B1 commit), 8 modified files unstaged，全保留等用户决策（详见 §6）

**ABC verify status (case A only, B/C 没跑因 A fail)**:
| trial | max err | pct wrong | out sample | 解释 |
|---|---|---|---|---|
| B1 baseline | (valid:y) | 0% | match ref | 21-32% 加速 vs qr baseline |
| step 1 (K only) | 23.72 | 100% | -22.5 全等 | V 仍 TDM garbage attractor dominate K mismatch |
| step 4 (K+V) | 0.37 | 98.7% | 0.16~0.73 散落 | element permute, 量级对 |
| (h) hybrid (K-TDM + V-async B1) | 0.36 | 97.6% | 0.37~0.81 散落 | K mismatch 单独暴露，V async 没救 |

---

## 2. Trial Timeline (精炼 一句话每个)

### B1 (verified baseline)
- **状态**: ABC valid:y, 21-32% 加速 vs qr baseline (sim ms A 2539 / B 374 / C 9113)
- **K dist**: 5D `<N0,N1,N2,K0,K1>` async-mapped (reverse engineered match wmma B operand)
- **V dist**: 5D `<KPerThread, NumWarps, KThreadPerWarp>, <NThreads, NPerThread>` async-mapped
- **Path**: async_load_tile + ds_load (K) + ds_load_tr_b128 (V)
- **Commit**: `16324795215`

### B2 主线 (B1 之后第一手 TDM 替换)
- **改动**: pipeline V/K/Q load 全换 `load_tile_tdm` + TDMConfig + s_wait_tensorcnt_barrier；policy 加 GetLdsPaddingConfigQ/K/V helpers
- **结果**: 编 fail 64 instance，触发 (X+) CK core fix + (α') unmerge typo fix

### B2 hypothesis chain (2026-04-30 trial 时间线; 详见 `swe-journey-B2-trials-202429.md`)
| H | 假设 | 结果 |
|---|---|---|
| H1 | TDMConfig padding 字段单位错 | ❌ disable padding runtime 无变化 |
| H4' | LDS swizzle 严格化 | ❌ disable xor 数值 16-digit 不变 |
| H8 | inner K-iter cumulative | ❌ single K-iter (s=64) 仍 fail |
| η | cached_global_strides cumulative product != real stride | ❌ 用 calculate_offset 真 stride 数值 0 变化 |
| Hγ | build cache stale | ✅ confirm η 真编进 binary (dependent static_assert verify) |
| Hβ | TDM 不读 stride 字段 | ✅ hardcode {99999,99999} SIGSEGV → TDM 真读 stride[0]，η 跟 cached 数值等价是 [0] 数学等价巧合 |
| Hδ-1 | box_dim 错 | ❌ K box (8,2) V box (8,8) 几何对 |
| H3 升级版 | TDM box-major LDS write vs B1 dist (async_load 设计) ds_load read 不兼容 | mentor 90%+ confirm 但**基于错 ds_load_tr semantics** |

### Independent latent fix (keep 独立 PR)
- **(α') fmha_fwd_kernel.hpp:2570-2602**: group path K-dram unmerge typo `kQKHeaddim/kDramTileK/kAlignmentK = 0`，已 PR #6964 (用户管不要追)
- **(X+) container_helper.hpp + tuple.hpp**: 5 处 in-place mutate tuple ops 不支持 mixed tuple `<int, constant<128>>`，独立有价值

### B2 (λ) sub-trials (post-H3 升级版)
- **(λ-test) auto-decided 18:50**: mentor reject (3-dist coupling 不能单改 disambig)
- **(λ) 用户拍方向 (b3bdc63a509)**: mentor design notes propose mirror GEMM v1 trivial tile-major + reg dist 不动
- **Step 1 K only (mirror GEMM v1 ColMajor B)**: K dram dist `tuple<sequence<warpNum=4, kN0/warpNum=16>, sequence<kK0=32>>` + `bool_constant<true>{}`; box_dim (8,2)→(32,16) deterministic verify ✓; case A max err 23.72 (V garbage attractor dominate, K mismatch hidden)
- **Step 4 K+V dual view (V RowMajor B trivial tile-major)**: kernel.hpp dual view dispatch + V dram dist `tuple<sequence<warpNum=4, kK1/warpNum=16>, sequence<kN1=128>>`; box_dim (8,8)→(128,16) ✓; case A max err 0.37 量级对散落
- **(h) hybrid auto-decided 03:06**: V revert B1 async_load + ds_load_tr, K 仍 step 1 trivial tile-major; case A max err 0.36 K mismatch 单独暴露
- **Reanalysis correction**: ds_load_tr semantics 修正后 (h) fail 真根因 = K reg dist coordination 缺失

---

## 3. 修正后真根因 Hypothesis (Clean Version)

### 真实 ds_load_tr_b128 semantics (来源: develop async_trload_policy production code + PDF §4.9.10.2 reference)

- V dram + LDS **全程 K-outer N-inner row-major** layout (V dram view 经 transform_tensor_view 后 view-level (kN1, kK1)，但物理 byte 是 (kK1, kN1) row-major)
- ds_load_tr_b128 hardware feature = **K-outer LDS → K-inner register transpose** (transpose 是 hardware 在 load 时做，不依赖 LDS 是 8-bank/16-col 几何 pattern)
- **TDM 写 plain K-outer N-inner row-major LDS 本来就是 ds_load_tr expected input**

### Mentor 三次错描述 vs 实际 (反 forgetting reference)

| 错描述 | 实际 |
|---|---|
| ds_load_tr 期望 LDS 是 8-bank/16-col 几何 hardware-fixed pattern | LDS 是 plain K-outer N-inner row-major byte 顺序 |
| TDM box-major write + ds_load_tr "物理上不可同时满足" | TDM plain row-major write 就是 ds_load_tr expected input |
| K 用 BWarpDstrEncoding 期望 wmma 几何 LDS layout | K reg dist outer encoding + BWarpDstrEncoding embed 假设 plain row-major LDS, 然后投影 thread-i 到 byte position |

### (h) Fail 真根因

K reg dist (`MakeKRegTileDistribution`, policy:585-616) 结构:
```
outer_encoding (MWarp / NWarp / NIterPerWarp / KIterPerWarp split)
  ↓ make_embed_tile_distribution_encoding
+ BWarpDstrEncoding{}  // wmma 16×16×32 B operand hardware-fixed lane geometry
```

- **outer encoding** (软件层): 4 warps 在 (kN0, kK0) 上 split + per warp wmma tile 数
- **inner (BWarpDstrEncoding)** (硬件层): lane 内 wmma B operand thread mapping (不能改)

**B1 verified path**: B1 5D K dram dist + async_load → 写 LDS 物理 byte 顺序 match reg dist outer encoding 期望的 stride/interleaving

**Trivial tile-major TDM write (step 1)**:
- 4 warps split kN0 by 16 rows，写 plain row-major LDS (byte = (n*kK0 + k) * sizeof)
- ds_load 用 reg dist 算 thread-i bottom_index → byte X
- 但 byte X 实际放的不是 reg dist 期望的 (n_i, k_i) element → element permute (max err 0.36 量级对散落)

**关键 insight**: B1 dram dist (5D) 跟 reg dist outer encoding **隐式 co-designed**；step 1 改 dram dist trivial tile-major 但 reg dist 没动 → coordination broken。

→ **不是 wmma 几何墙物理不兼容，是 dist coordination 缺失**

### V Side (likely 同性质问题)

- V dram dist (B1 5D) + V reg dist (BWarpDstrEncoding embed + transpose helper) 也是 B1 时代 co-designed
- step 4 V trivial tile-major 改 dram dist 没改 reg dist → 同 K 性质 coordination broken
- (h) hybrid V revert B1 时仍走 5D async-mapped path 跟 reg dist 一致 ✓

---

## 4. Next Step Plan — K LDS Dump Disambig (~2-3h)

### 实施 sketch

```cpp
// 在 pipeline operator() (batch path) K TDM write 后 + s_wait_tensorcnt_barrier<0> 后插
if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    auto* k_lds_ptr = static_cast<KDataType*>(smem_ptr);  // K LDS base
    auto* dump_buf = kargs.debug_k_lds_dump_ptr;          // host-visible global buffer
    for (int i = 0; i < kN0 * kK0; ++i) {
        dump_buf[i] = k_lds_ptr[i];
    }
}
```

### Plumbing

1. `FmhaFwdCommonKargs` (kernel.hpp:95) 加 `void* debug_k_lds_dump_ptr`
2. `MakeKargs` 加 `debug_k_lds_dump_ptr` 参数 + setter
3. `example_fmha_fwd.cpp` host alloc `kN0 * kK0 * sizeof(half_t)` device buffer + pass pointer
4. case A 跑完 host memcpy device buffer → host array
5. host computes expected K dram tile (按 case A input + dram K layout) 同 shape array
6. 比对 element by element

### Branch logic (disambig outcome → fix path)

- **Case 1: LDS layout = plain K-outer N-inner row-major (1:1 dram copy)**
  → confirm "TDM trivial tile-major write 出 plain row-major LDS as expected"
  → mismatch 在 K reg dist 不 coordinated
  → trace `make_embed_tile_distribution_encoding` 投影 + 设计 reg dist outer encoding align
  → **fix path = Option α 或 γ; 工作量 1-2d**

- **Case 2: LDS layout ≠ plain row-major (TDM write 行为 surprising)**
  → trivial tile-major TDM 在 fmha context 实施有 bug
  → 不论 reg dist 怎么改都改不对，要先修 LDS write 行为
  → deep TDM trace + design LDS swizzle 或退 (μ)
  → **fix path = (μ) plan B 或 重设计 dram dist; 工作量 3-5d**

### 文件 line range (LDS dump 实施)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp:95-150` (kargs struct)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp:200-300` (MakeKargs)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp:285-390` (batch path K LDS write 后)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp` (host alloc + compare)

---

## 5. Fix Candidates

| | 描述 | 工作量 | risk | 适用 case |
|---|---|---|---|---|
| **(α)** | 改 K reg dist outer encoding 让 thread mapping 匹配 plain row-major LDS layout | 1-2d (K only); +1-2d V | LOW (read-side software 改) | LDS dump = Case 1 (plain row-major) |
| **(γ)** | 改 K dram dist 不是 trivial tile-major 而是 hybrid (warp split kN0 + thread access pattern align B1 reg dist 隐含 stride) | 1-2d (K only); +1-2d V | LOW-MEDIUM | LDS dump = Case 1 if α 不 work |
| **(μ) plan B** | K + V 双 LDS shuffle stage (TDM write plain → ds_load + ds_store reshape → ds_load[_tr] consume) | 2-3d | MEDIUM + perf 不确定 (可能 < B1) | LDS dump = Case 2 或 α/γ 都 fail |
| **(ν) fallback** | 撤退到 B1 (revert all B2 fmha-side 改动) | 1-2h | ZERO | 用户战略选择 (不再 active) |

**No independent Option β**:
- TDM hardware 不支持 write 端 swizzle (per `reference_mi450_kernel_patterns.md`)
- Software LDS view xor transform = read-side dist 调整 (= Option α 等价)
- LDS shuffle stage = (μ) plan B

### V 同性质问题 (likely)

V reg dist 同样从 BlockGemm wmma operand 推 (经 TransposedDstrEncode helper)。step 4 V trivial tile-major 改 dram dist 没改 reg dist → 同 K coordination broken。

→ K disambig + fix 通了之后 V 几乎肯定要做类似 disambig + fix path。总工作量 (λ-revised) = 2-7d。

---

## 6. Worktree 状态 + Commit 策略

### 8 modified files

| 文件 | 类型 | 不论方向 | (ν) 撤退 | (λ-revised) 继续 | (μ) 继续 |
|---|---|---|---|---|---|
| `CMakeLists.txt` | lead plumbing | keep dirty | revert | keep | keep |
| `example/ck_tile/01_fmha/CMakeLists.txt` | lead plumbing | keep dirty | revert | keep | keep |
| `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` | lead plumbing | keep dirty | revert | keep | keep |
| `include/ck_tile/core/container/container_helper.hpp` | (X+) CK core latent fix | **独立 PR** | keep + PR | keep + PR | keep + PR |
| `include/ck_tile/core/container/tuple.hpp` | (X+) CK core latent fix | **独立 PR** | keep + PR | keep + PR | keep + PR |
| `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` | (α' PR #6964 重复 + B2 dead code) | partial | revert dead code, keep (α') | revert dead code if not reused | revert dead code |
| `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` | B2 主线 + step 1 + V revert | continue | revert all | base for further work | base for further work |
| `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` | B2 主线 + step 1 + V revert | continue | revert all | base for further work | base for further work |

### Dead code (kernel.hpp + pipeline.hpp)

- `kIsTdmPipeline` SFINAE detection (kernel.hpp)
- `has_kIsTdm_trait` helper (kernel.hpp)
- `kIsTdm = true` trait (pipeline class)
- `tdm_config_v` declarations (pipeline) — V 不再走 TDM 但 declaration 留着
- `GetLdsPaddingConfigV_original_disabled` (policy) — rename 备用

→ (ν) 撤退时一起 revert; (λ-revised) 重启 V dual view 时复用; (μ) 不需要 revert

### 独立 PR 价值

- **(α') PR #6964**: 已开 + poyenc APPROVED + CI 全绿，user 自管 (lead 不追)
- **(X+) CK core fix**: 5 处 in-place mutate tuple ops mirror generate_tuple recursive pattern; 修真 bug; 独立有价值不论 B2 方向; 推 separate PR

---

## 7. Hardware Semantic 关键修正 (反 Forgetting Reference)

### ds_load_tr_b128 真实行为

- **Input**: K-outer N-inner row-major LDS layout (V LDS naive 2D `(kK1, kN1)` row-major)
- **Output**: K-inner thread distribution register tile (跟 wmma B operand consume 兼容)
- **Hardware**: load 时 transpose, 不依赖 LDS 是几何 pattern; 任何 thread mapping 都可以 read 任 LDS region (shared mem)

### TDM 真实行为

- **Write mode**: box-major hardware DMA, lds_coord 起点 + box_dim contiguous box
- **不支持 write 端 swizzle** (per `reference_mi450_kernel_patterns.md` line 34)
- **Bank pattern**: 不强求 8-bank/16-col 几何 (这是 ds_load_tr **register output** layout 的 attribute，不是 LDS write 约束)
- **跟 IsWarpLevelParallelOnly + NumCoord 互动**: per-thread iteration 决定多少次 TDM issue, IsWarpLevelParallelOnly=true 让 lane 共享 coord (具体语义未深 trace)

### Reference doc 优先级 (写 hardware semantic 前必读)

1. **Production code in develop**: `block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` (B1 verified V dist + LDS layout) — ground truth
2. **Hardware spec PDF**: `p4web.amd.com:1712//gfxip/mi400/...` §4.9.10.2 ds_load_tr semantics
3. **CK ck_tile docs**: `reference_ck_tdm_api.md`, `reference_mi450_kernel_patterns.md`, `reference_mi450_hw_specs.md`
4. **Mentor design notes**: `mentor-fmha-tdm-design-notes-202402.md` — 注意 mentor 之前对 ds_load_tr 描述错三次, 需 cross-check production code

---

## 8. Methodological Lessons (Sign-off Errors + Protocol 改进)

### 三次 mentor sign-off 错 pattern

1. **Sign-off 1 (b3bdc63a509 propose (λ))**: mentor design notes 推 mirror GEMM v1 trivial tile-major; **基础前提错**: GEMM v1 自己 build-gfx1250 也 0 instance untested design, 没 known-good runtime reference (mentor 自己 decision-B2-H3-direction:46-47 写过)
2. **Sign-off 2 (step 1 → step 4 进 V dual view)**: box_dim deterministic verify (8,2)→(32,16) 看做 "K 改生效"; **没 challenge** "K 单独的 element permute 被 V garbage overshadow 看不出来" 可能性 → step 1 sequence 设计错
3. **Sign-off 3 (推 (h) hybrid → 推 (ν))**: 基于错 ds_load_tr semantics 描述说 "V 物理不兼容 / K 同性质撞 wmma 几何墙"; **没 read production async_trload_policy** verify hardware semantic, 直接 first-principle 推 → user/lead read 后修正

### Reviewer Challenge 价值

- User/lead 主动读 develop production code 才发现 mentor 三次错
- 推 K LDS dump empirical verify 比 mentor "无解" 假设更可靠
- → **三方协作中 reviewer (user) 必须独立 challenge 假设, 不能 inherit assumption chain**

### 改进 protocol

1. **写 hardware semantic 描述前先 read production verified code** (e.g. async_trload_policy for ds_load_tr) — 不能 first-principle 推
2. **Sign-off chain 必须独立 verify 基础前提** — 不能 inherit assumption chain
3. **Disambig sequence 设计**: 改 dram dist 之前**先 LDS dump baseline B1 layout** (verified path) + 改后**再 dump 比对**, 不依赖 indirect verify (ABC valid signal 可能被多 mismatch overlay overshadow)
4. **类比推理 (V 撞墙 → K 同性质撞墙) 高风险** — 必须独立 verify 类比 source

### Step 1 Sequence 错误 (具体反思)

- 当时 V 仍走 TDM (B2 主线 load_tile_tdm with B1 5D dist) — V 错 dominate (max err 23.72 garbage attractor)
- → K 单独的 element permute 被 V garbage overshadow 看不出
- **应该 step 1 就让 V 走 B1 async_load** 让 K mismatch 单独暴露 → 早 5+ hours 发现 (λ) 概念问题

---

## 9. Reference Cross-link (其它 Status Doc)

按 priority 读 (after 这份 v3):

| Doc | 何时读 | 内容 |
|---|---|---|
| `mentor-reanalysis-ds-load-tr-correction.md` | 必读 | mentor 自己 reanalysis: hardware semantic 修正 + B1 production code 反证 + acknowledge 三次错 |
| `swe-reanalysis-K-side-correction.md` | 必读 | SWE reanalysis: (h) fail 真根因 + LDS dump implementation plan + 工作量 estimate + reflection |
| `mentor-fmha-tdm-design-notes-202402.md` | 必读 (但要 cross-check production code) | mentor 原 design notes (含错 ds_load_tr 描述, 现 invalidated) + (λ) 实施 plan line range |
| `swe-status-lambda-fail-v2.md` | 历史 reasoning, 现 invalidate | "K 撞 wmma 几何墙" wrong conclusion - 反 example 学习 |
| `swe-status-lambda-pivot.md` | 历史 reasoning, 现 invalidate | step 4 V 撞墙 false conclusion |
| `swe-status-lambda-design.md` | 历史 reference | mentor 第一次 sign-off 设计 + 字段细节 |
| `swe-journey-B2-trials-202429.md` | 历史 trial 时间线 | B2 H1-H8 全 6 轮 hypothesis 的来龙去脉 |
| `B2-lambda-resume-baseline.md` (v1) | 历史 v1 resume | 团队第一次重启 baseline |
| `final-summary-team-shutdown-212300.md` | 历史 v0 shutdown | B2 团队首次 shutdown 全部 state |
| `decision-B2-H3-direction-174838.md` | 历史 decision | (λ/μ/ν) 用户首次拍方向决策报告 |

`~/.claude/memory/` 项目 memory:
- `project_gfx1250_fmha_tdm_v1.md` — B1 完成状态 + B1 trial 教训
- `project_gfx1250_fmha_fwd_dev.md` — 编译/验证流程 + ABC test shape
- `project_gfx1250_fmha_fwd_design.md` — GFX IP team baseline 算法设计
- `reference_ck_tdm_api.md` — TDM descriptor + GEMM v1/v2 设计参考
- `reference_mi450_kernel_patterns.md` — TDM/swizzle/padding 通用模式 + "TDM 不支持 swizzle write" 关键约束
- `reference_mi450_hw_specs.md` — MI450 硬件参数

---

## 10. Fresh Team Lead Action Items (拿到这份 doc 后)

### 启动 sequence

1. `TaskList()` 验证 task 状态 (现 task #1 = "Step B2 (h) hybrid - K-TDM + V-async_load (auto-decided)" in_progress; 用户 final 决定方向后 lead update task subject)
2. 读 这份 v3 doc 全 (此为 source of truth)
3. 按 §9 priority 读 mentor + SWE reanalysis (必读 2 份)
4. Read worktree git diff (`git diff include/ck_tile/ops/fmha/`) 看 8 modified 实际 code
5. (可选) Read mentor 原 design notes + status doc 历史 audit chain (已 invalidated 部分作 anti-pattern reference)
6. 跟 user check 方向 — 等用户拍 LDS dump disambig (推荐) / (μ) / (ν)

### 派 SWE 第一步 task (假设用户拍 LDS dump disambig)

```
Task: K LDS dump disambig 实施
- File 1: kernel.hpp — kargs 加 debug_k_lds_dump_ptr 字段 + MakeKargs 接受
- File 2: pipeline.hpp batch path K TDM write 后 + s_wait_tensorcnt_barrier<0> 后插 dump kernel code
- File 3: example_fmha_fwd.cpp host alloc device buffer + pass + post-run cudaMemcpy + compare with expected K dram tile
- 工作量 ~2-3h dev
- Verify case A 跑完输出 LDS dump
- Branch: LDS = plain row-major → assumption A confirmed → fix path α/γ; LDS ≠ plain row-major → assumption B → 退 (μ) 或 deep trace
- 跟 mentor 内部讨论字段细节 / kargs plumbing pattern
```

### 派 SWE 第二步 (LDS dump 跑完后)

- 如果 Case 1 (plain row-major): 派 SWE trace `make_embed_tile_distribution_encoding` 投影 + design Option α 或 γ + step-by-step implementation (1-2d)
- 如果 Case 2 (≠ row-major): 派 SWE deep TDM trace + design LDS swizzle 或评估 (μ) plan B (3-5d)

### Lead 工作流提醒 (按 feedback memory)

- 技术细节 SWE 直接找 mentor，**不报 lead**
- Lead 只接 4 类: 方向决策 / 跨 repo 协调 / task 派发请求 / Step final
- **不主动催** SWE+mentor 内部循环
- **每个 sign-off 必须独立 verify 基础前提** (不能 inherit assumption chain — 这是 v3 关键 lesson)
- 自决边界: 方向性 → 立即 Teams notify + 1h 等用户 + 自决 (可逆); 不可逆对外 → 永远等

### 团队 layout 提醒

- `~/.claude/teams/gfx1250-fmha-tdm/config.json` 的 `members` list 历史只列 2 个 (mentor + swe)，**qa 不在里面但 functional alive** (tmux pane + inbox 都 alive)
- 派 task 给 qa 用 `SendMessage(to="qa", ...)` 正常工作，**别基于 config 判断 "qa 没 spawn" 然后重 spawn**
- send to lead 一律用 `to: "team-lead"` (不是 `to: "lead"`，会落孤儿 inbox)

### 用户已委托项 (不要再追)

- **PR #6964** ((α') unmerge typo fix): 用户原话 "6964 不用你管，我自己会注意合并"
- **develop 副发现 (114 fp16/bf16 group fail)**: 上轮 auto-decided "留给 fmha team 不深挖"

---

## Appendix — Key Numbers (Quick Reference)

- **fmha shape (case A)**: kBlockSize=256 (4 wave × 64 lane), kM0=64, kN0=64, kK0=32, kN1=128, kK1=128, kQKHeaddim=128, warpNum=4
- **WMMA**: 16×16×32 fp16/bf16
- **B1 baseline sim ms**: A 2539 / B 374 / C 9113
- **K box_dim (B1 vs trivial tile-major)**: (8, 2) → (32, 16)
- **V box_dim (B1 vs RowMajor B trivial tile-major)**: (8, 8) → (128, 16)
- **Case A**: `-prec=fp16 -b=1 -h=1 -s=1023 -mask=0 -mode=0 -d=128`
- **Case B**: `-prec=fp16 -b=1 -h=2 -h_k=1 -s=1023 -s_k=257 -mask=2 -mode=0 -d=128`
- **Case C**: `-prec=bf16 -b=1 -h=1 -s=2047 -mask=0 -mode=0 -d=128`
- **Build**: `cmake -S . -B build-gfx1250 -GNinja --preset dev -DGPU_TARGETS=gfx1250 -DFMHA_FWD_ENABLE_APIS='fwd'`
- **Run**: `/home/yiding12/workspace/rocdtif/run-rocdtif.sh ffm_lite <binary> <args>`

---

file: B2-resume-baseline-v3.md
