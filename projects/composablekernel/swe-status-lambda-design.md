# (λ) Design 定稿 + 实施 Plan — 准备 Implement

**写作时间**: 2026-05-06，B2 (λ) design phase 完成
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1)

---

## 1. Mentor sign-off design 方向

**Mirror GEMM v1 trivial tile-major dram dist**（不是早期 design notes 写的"BlockGemm encode 共驱动 dram dist" — mentor 已纠正）。

核心机理：
- TDM 写 LDS 是 box-major：thread-i 用 lds_coord 起点写一段 box_dim contiguous box
- GEMM v1 把 dram dist 故意做成 trivial tile-major（warp 切 outer 维，K 维单 thread 整段 vector）→ 每 thread per call write footprint 是一段 contiguous LDS row → box-major write 自然落到 LDS row-major 物理位置
- BlockGemm reg dist (B1 已从 `BWarpDstrEncoding` / `AWarpDstrEncoding` 推) 按 LDS row-major 接口约定读 → 跟 dram dist 写自然 align
- → **reg dist 不动，BlockGemm wrapper 不必要**，只需重写 K/V dram dist 让 thread footprint = contiguous LDS row

mentor 早期 design notes 两处错误已澄清：
- "A operand wrapper" 是笔误 → K 实际是 BlockGemm 0 **B operand**
- "BlockGemm encode 共设计 dram dist" 措辞误导 → 真实是 dram dist 独立 trivial tile-major

→ (λ) **从 3 步降为 2 步**: (λ-1) K dram dist + (λ-2) V dram dist。

## 2. 改动清单

### (λ-1) K dram dist

**文件**: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:152-178`

**当前 (B1, 5D async-mapped)**:
```cpp
template <typename Problem, bool LoadOnce = false>
CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
{
    // ... compute N0, N1, N2, K0, K1
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 1>>{});
}
```

**改成 (λ trivial tile-major, GEMM v1 ColMajor B 路径)**:

K 在 BlockGemm 0 是 B operand, K LDS 物理 (kN0, kK0) row-major (policy:400-405) → kN0 是 outer (N major) → ColMajor B 路径。

```cpp
template <typename Problem, bool LoadOnce = false>
CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
{
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
    constexpr index_t kKPerBlock =
        LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;
    constexpr index_t warpNum   = kBlockSize / get_warp_size();

    static_assert(kNPerBlock % warpNum == 0, "kNPerBlock should be divided by warpNum");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,                                                       // R: empty
            tuple<sequence<warpNum, kNPerBlock / warpNum>,                    // X[0]: N-axis warp split
                  sequence<kKPerBlock>>,                                      // X[1]: K-axis full vector
            tuple<sequence<1>>,                                               // PsToRH (warp dim mapping)
            tuple<sequence<0>>,                                               // PsToRH_lid
            sequence<1, 2>,                                                   // YsToD outer
            sequence<1, 0>>{},                                                // YsToD inner
        bool_constant<true>{});                                               // IsWarpLevelParallelOnly
}
```

参数对应：fmha kN0=128, kK0=32, warpNum=4 → `tuple<sequence<4, 32>, sequence<32>>`
per thread per call footprint = (kN0/warpNum, kK0) = (32, 32) = **1024 elements / 2048 bytes**

### (λ-2) V dram dist

**文件**: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:625-648`

**改成 (λ trivial tile-major, GEMM v1 RowMajor B 路径)**:

V 在 BlockGemm 1 是 B operand (PV GEMM, A=P, B=V)，V LDS 物理 (kK1, kN1) row-major (policy:499-504) → kK1 是 outer (K major) → RowMajor B 路径（**注意跟 K 不同！**）。

⚠️ **实施前要先 confirm V dram tile 维度顺序**: grep V `make_tile_window` for V dram window，看 dram tile 是 (kK1, kN1) 还是 (kN1, kK1)。如果 dram 是 (kN1, kK1) 那就要换 ColMajor B 路径 `tuple<sequence<warpNum, kN1/warpNum>, sequence<kK1>>`。

**默认 RowMajor B 草稿（dram tile = (kK1, kN1) 假设）**:
```cpp
template <typename Problem>
CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
{
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
    constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;  // = kK1 in V naming
    constexpr index_t warpNum   = kBlockSize / get_warp_size();

    static_assert(kKPerBlock % warpNum == 0, "kKPerBlock should be divided by warpNum");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<warpNum, kKPerBlock / warpNum>,
                  sequence<kNPerBlock>>,
            tuple<sequence<1>>,
            tuple<sequence<0>>,
            sequence<1, 2>,
            sequence<1, 0>>{},
        bool_constant<true>{});
}
```

参数对应：fmha kK1=128, kN1=128, warpNum=4 → `tuple<sequence<4, 32>, sequence<128>>`
per thread per call footprint = (kK1/warpNum, kN1) = (32, 128) = **4096 elements / 8192 bytes** (~ baseline doc example 量级)

⚠️ **V trload 端 (ds_load_tr_b128) 可能有额外约束** — RowMajor B trivial tile-major 跟 ds_load_tr 8-bank pattern 是否兼容，没静态推；如果 step 5 fail 要 deeper trace V trload 端 thread mapping。

### 不动

| 函数 | line | 不动原因 |
|---|---|---|
| `MakeKRegTileDistribution` | 585-616 | 已从 `WarpGemm::BWarpDstrEncoding` 推，对 LDS row-major naive 假设跟 trivial tile-major write 自然 align |
| `MakeVRegTileDistribution` | 685-720 | 已从 `WarpGemm::AWarpDstrEncoding` 推 (V trload 后给 PV GEMM as B 但 reg dist 用 transpose helper)，同上 |
| `MakeKLdsBlockDescriptor` | 400+ | naive 2D (kN0, kK0) row-major 不变 |
| `MakeVLdsBlockDescriptor` | 499+ | naive 2D (kK1, kN1) row-major 不变 |
| BlockGemm 0/1 wrapper | — | 没东西要 wrap，reg dist 不动接口不变 |
| LDS swizzle (`MakeKLdsBlockDescriptor<..., true>`) | — | B2 H4' 已 verify swizzle 不是根因，保留即可 |

### Padding policy（实施期 disable，跑通后再开）

`GetLdsPaddingConfig<Problem, IsK_or_V>` 在 (λ-1) step 1 / (λ-2) step 4 都先 disable（return false / pad_amount=0），跟 mentor disambig 路径 step 1 一致。

## 3. Disambig 实施 Sequence (mentor 推 option A)

| step | 改动 | verify |
|---|---|---|
| 1 | 仅改 K dram dist + disable K padding (1 处) | QA 编 |
| 2 | 编通 + CK_PRINT K box_dim 期望 (32, 32) | QA 跑 print |
| 3 | 跑 A (s=1023 d=128 fp16 mask=0) | 期望 valid:n 但**数值 ≠ B2 H4' baseline garbage**（confirm K 改有效） |
| 4 | 改 V dram dist + disable V padding (1 处) | QA 编 |
| 5 | 跑 A | 期望 **valid:y** |
| 6 | 跑 B + C | 期望全 valid:y |
| 7 | 重新 enable padding + 收集 perf vs B1 baseline | QA 跑 ABC + perf |

**Branch logic**:
- step 3 数值跟 B2 baseline 16-digit identical → K 改没生效 → Hγ-style dependent static_assert verify 实例化
- step 5 fail → print V box_dim + grep V trload (`ds_load_tr_b128`) thread mapping 重 design V

**对照 baseline (B1 async_load 路径)**:
| Shape | sim ms | walltime |
|---|---|---|
| A (fp16 s=1023) | 2539 | 2.9s |
| B (fp16 GQA causal s=1023×257) | 374 | 0.7s |
| C (bf16 s=2047) | 9113 | 9.5s |

(λ) 期望 perf ≥ baseline（TDM hardware DMA 应该比 async_load 快）。

## 4. 风险

1. **box_dim hardware 上限不确定** — TDM `tile_dim0/1/2` SGPR 字段具体上限没 spec PDF。1024/4096 elements 量级在 baseline doc example (8192 elements) 内，应该 OK；撞上限编时会 error 或 runtime SIGSEGV
2. **V trload 端 ds_load_tr_b128 pattern** — RowMajor B trivial tile-major 跟 8-bank pattern 是否兼容没静态推，可能要 deeper trace
3. **B1 教训"动 dist 风险高"** — (λ) 直接动 K+V dram dist 2 处，但每处只是 trivial tile-major（很简单的 encoding），比 B1 trial 的 5D interleaved 改法风险低
4. **GEMM v1 在 build-gfx1250 也 0 instance** — co-design 路径生产没真跑通过 reference，(λ) 是 new territory

## 5. 工作量估

- step 1-3 (K only): 半天
- step 4-6 (V): 半天
- step 7 (perf): 半天
- mentor pair 讨论 + 状态汇报 buffer: 半天
- **合计**: 1.5-2 SWE-day（mentor 原估 1-2 周是含多轮 design iter 的最坏估，有 sign-off 后实施估快很多）

## 6. 准备 implement

mentor sign-off ✓
设计字段细节 cross-check ✓ (含 `bool_constant<true>{}`、K ColMajor B、V RowMajor B、box_dim 量级)
disambig sequence option A ✓

**只剩 V dram tile 维度顺序 confirm** — 实施时第一步 grep `make_tile_window` for V dram，10 分钟内能定。

请 lead audit 此 plan，pass 后开始 (λ-1) step 1 implement。

---

file: swe-status-lambda-design.md
