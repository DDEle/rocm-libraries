# SWE Reanalysis — K Side Correction (post ds_load_tr semantics 修正)

**写作时间**: 2026-05-06
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1)
**关联**: `mentor-reanalysis-ds-load-tr-correction.md` (mentor reanalysis)

---

## 1. Acknowledge — 之前 K 撞墙 conclusion 基于错前提

**之前 SWE 推论 (status doc `swe-status-lambda-fail-v2.md`)**:
> "K trivial tile-major 也撞墙跟 V 同性质 — wmma B operand thread mapping (BWarpDstrEncoding) 是 hardware-fixed lane pattern (lane 内 4×4 几何 block)，期望非 plain row-major LDS layout"

**这个 conclusion 是 wrong**：
- 推论 base 在 mentor 给的 ds_load_tr_b128 semantics 描述 ("hardware-fixed transpose op，期望 8-bank/16-col 几何 LDS pattern, TDM box-major write 写 plain row-major 物理上做不到")
- User/lead 自己读 develop async_trload_policy 实际 code (`block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp:614-641 V dram dist + LDS layout`) 发现 mentor 描述**第三次错**：
  - V dram + LDS 全程是 **K-outer N-inner row-major** (不是 8-bank/16-col 几何 pattern)
  - ds_load_tr "transpose" feature = K-outer LDS → K-inner register
  - 即 LDS 物理 layout 实际就是 plain row-major K-outer N-inner，hardware 跨这种 layout 做 transpose 进 register
- → "TDM 写 plain row-major LDS 跟 ds_load_tr expected pattern 不兼容" **invalidate**
- → 我推 K 同性质撞 wmma 几何墙 (基于"V 也撞墙因 ds_load_tr expected pattern" 的 analogy 推理) **整个推理链条 broken**

**Methodological reflection**:
- 我推 K 撞 wmma 几何墙时，**没自己 trace BWarpDstrEncoding 的 LDS layout 假设**，直接 借用 mentor V trload mismatch 类比推理
- 类比的 source (V trload mismatch) 本身基于错的 ds_load_tr semantics 描述
- → 错前提 + 类比推理 = double error
- 教训：每个 sign-off 链条 + 类比推理都必须独立 verify 基础前提，不能 inherit 别人的假设

## 2. 重审 (h) revert 后 K side 真根因

(h) revert 之后状态：
- V: 走 B1 verified async_load + B1 5D dram dist + 老 reg dist (BWarpDstrEncoding embed)
- K: trivial tile-major dram dist (step 1 改) + **老 reg dist 没动** (BWarpDstrEncoding embed outer encoding from B1)

**(h) case A 仍 valid:n max err 0.36 element permute** — V 走 B1 verified path 仍错位，说明问题在 K side。

**新 hypothesis (跟 mentor sync align)**:

K reg dist 结构 (`MakeKRegTileDistribution`, policy:585-616):
```cpp
outer_encoding (MWarp / NWarp / NIterPerWarp / KIterPerWarp split)
  ↓ make_embed_tile_distribution_encoding
+ BWarpDstrEncoding{}  // wmma 16×16×32 B operand hardware-fixed lane geometry
```

两层：
- **outer encoding**: 4 warps 在 (kN0, kK0) 维上 split + per warp wmma tile 数
- **inner (BWarpDstrEncoding)**: lane 内 wmma B operand thread mapping (hardware-fixed，**不能改**)

**B1 verified path**: B1 5D K dram dist + async_load → 写 LDS 物理 byte 顺序 match reg dist outer encoding 期望的 wmma-friendly stride/interleaving (B1 已 verified valid:y)。

**Trivial tile-major TDM write (我 step 1 改)**:
- 4 warps split kN0 by 16 rows，写 plain row-major LDS (k0_byte = (n*kK0 + k) * sizeof(KDataType))
- LDS region split by warp-id 不是 mismatch source — shared mem barrier 后 ds_load 任 warp 都能读任 region

**真 mismatch 在 thread-i 投影位置**:
- B1 dram dist 让 LDS 物理 byte **顺序按 reg dist outer encoding 期望的 wmma-friendly stride 摆** (specific bank pattern / element interleaving)
- Trivial tile-major TDM 写出 plain row-major (n*kK0 + k) **没这个 wmma-friendly stride**
- ds_load 用 reg dist 算 thread-i bottom_index → byte X，但 byte X 实际 element 不是 reg dist 期望的 (n_i, k_i)
- → element permute (max err 0.36 量级对 散落)

**关键 insight**: 不是物理不兼容 — 是 dram dist 和 reg dist **没 coordinated**：
- B1 时代 dram dist (5D) 跟 reg dist outer encoding **隐式 co-designed** 让 LDS layout 一致
- 我 step 1 改 dram dist trivial tile-major 但 reg dist 没动 → coordination broken

## 3. (λ) 概念 salvageable

**(λ) 不是 physical incompatibility**：
- ds_load 是 element-by-element 不挑 LDS pattern (mentor confirm)
- 任何 LDS 物理 layout + 任何 dist read pattern 都可以 work，只要 write/read pattern align
- 关键是 **dram dist + reg dist 必须 co-designed**

**Fix candidates** (mentor sync, 待 LDS dump confirm):
- **Option α**: 改 K reg dist outer encoding 让 thread mapping 匹配 plain row-major LDS layout (= TDM trivial tile-major write 出来的 byte 顺序)
- **Option γ**: 改 K dram dist 不是 trivial tile-major 而是某种 hybrid (warp split kN0 + thread-level access pattern align B1 reg dist outer encoding 隐含的 wmma-friendly stride)

**(β 路径 caveat — 不独立)**:
- TDM hardware 不支持 write 端 swizzle (`reference_mi450_kernel_patterns.md`: "Swizzling: MI450 通过 TDM 不支持，只 async load 支持")
- Software 层 LDS view xor transform (`MakeKLdsBlockDescriptor<Problem, false, true>` 已实现) 等价 Option α (view 把 plain row-major byte 重新解释 → ds_load 端 coord 经 xor 算 byte = read-side dist 调整)
- LDS-to-LDS shuffle stage 等价 (μ) plan B
- → 没有独立 Option β

具体哪个 (α 或 γ) work 不确定，要先 LDS dump disambig + trace `make_embed_tile_distribution_encoding` 实际投影。

## 4. K LDS dump disambig plan (mentor sync, 必须做)

### 设计

```cpp
// 在 pipeline operator() (batch path) K TDM write 后 + s_wait_tensorcnt_barrier 后插
if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    auto* k_lds_ptr = static_cast<KDataType*>(smem_ptr);  // K LDS base
    auto* dump_buf = kargs.debug_k_lds_dump_ptr;  // host-visible global buffer
    for (int i = 0; i < kN0 * kK0; ++i) {
        dump_buf[i] = k_lds_ptr[i];
    }
}
```

### 实施 plumbing

1. `FmhaFwdCommonKargs` (kernel.hpp:95) 加 `void* debug_k_lds_dump_ptr` 字段 (debug-only)
2. `MakeKargs` 加 `debug_k_lds_dump_ptr` 参数 + setter
3. `example_fmha_fwd.cpp` host alloc `kN0 * kK0 * sizeof(half_t)` 个 element 的 device buffer，pass pointer 到 kargs
4. case A 跑完 host memcpy device buffer 到 host array
5. 同时 host computes expected K dram tile (按 case A input + dram K layout) 同 shape array
6. 比对：顺位 element 一致 → plain row-major copy ✓；不一致 → TDM write 行为 surprising

### 工作量

~2-3h dev (kargs 字段 + alloc + dump kernel code + host compare script)。

### Branch logic

- **Case 1: LDS layout = plain K-outer N-inner row-major (1:1 dram copy)**
  → confirm "TDM trivial tile-major write 出 plain row-major LDS as expected"
  → mismatch 在 K reg dist 不 coordinated → trace `make_embed_tile_distribution_encoding` + 设计 reg dist outer encoding align (Option α 或 γ)
  → **(λ-revised) 工作量 1-2d**

- **Case 2: LDS layout ≠ plain row-major (TDM write 行为 surprising)**
  → trivial tile-major TDM 在 fmha context 实施有 bug
  → 不论 reg dist 怎么改都改不对，要先修 LDS write 行为
  → 可能 deep TDM trace + design LDS swizzle (Option β) 或退 (μ)
  → **(λ-revised) 工作量 3-5d** + risk 高

## 5. 工作量重估 — (λ-revised)

mentor 之前 quoted (λ-revised) ">1 周" 是基于 "wmma 4×4 几何 block ≠ contiguous box, 无解" 的 (错) 假设。新 hypothesis 下:

| 之前 (基于错前提) | 现在 (based on new hypothesis) |
|---|---|
| HIGH RISK / >1 周 / 可能无解 | K LDS dump disambig 2-3h, then 1-2d 或 3-5d |

详细分支：
- **Step 0**: K LDS dump disambig (~2-3h dev + 1 run)
- **Case 1** (LDS = plain K-outer N-inner row-major 1:1 dram copy): K reg dist outer encoding 调整 (Option α 或 γ) → **1-2d**
- **Case 2** (LDS ≠ row-major, TDM write 行为 surprising): deep TDM trace + design LDS swizzle 或退 (μ) → **3-5d** + risk 高
- **V 同性质问题 (very likely)**: V 也要做类似 disambig + fix → 总 **2-7d**

**新工作量 estimate**: 2-7 SWE-day 取决于 disambig 结果 (vs 之前 quoted ">1 周可能无解")

**关键 sequence design 修正** (吸取 step 1 教训):
- 每改一处 dist 之前先 LDS dump verify 上一步改动产生的 LDS layout 符合预期
- V 改之前先 K 单独跑 + V 走 B1 verified path 让 K mismatch 单独暴露
- 不依赖未 verified reference design (mentor 之前 sign-off 错教训)

## 6. 选项 reassess (post ds_load_tr correction)

| | 描述 | 工作量 (新估) | risk | mentor verdict (旧) | mentor verdict (新) |
|---|---|---|---|---|---|
| **(λ-revised)** | LDS dump disambig + K reg dist coord (+ V 类似 path) | 2-7d | MEDIUM (取决 disambig) | HIGH RISK >1 周可能无解 | feasible，工作量 disambig 后定 |
| **(μ)** | K + V 双 LDS shuffle stage | 2-3d | MEDIUM + perf 不确定 | MEDIUM RISK | 不变 |
| **(ν)** | 撤退到 B1 | 1-2h | ZERO | 推 | reanalysis 后 demote — (λ-revised) 不再 invalidated 应优先 |

mentor + SWE 推 **先做 LDS dump disambig (~2-3h)** 再决定方向，不直接 (ν) 撤退。

## 7. Worktree 状态

不动等用户决策。8 modified:
- `CMakeLists.txt`, `example/.../CMakeLists.txt`, `example/.../fmha_fwd.py` (lead plumbing)
- `include/.../core/container/container_helper.hpp`, `tuple.hpp` (X+ CK core latent fix — keep 不论方向)
- `include/.../fmha/kernel/fmha_fwd_kernel.hpp` (α' latent fix already PR #6964 + B2 dead code kIsTdmPipeline + has_kIsTdm_trait)
- `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (B2 主线 K/Q TDM + step 1 K trivial tile-major + dead code kIsTdm trait + V revert to B1)
- `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` (B2 主线 padding helpers + step 1 K trivial tile-major + V revert to B1 + V padding rename _disabled)

不论方向 (X+) CK core fix + (α') 都 keep 独立 PR。

**Dead code (kIsTdmPipeline + has_kIsTdm_trait in kernel.hpp + kIsTdm trait in pipeline class + tdm_config_v declarations)** 是 step 4 (P1 trait dispatch) 实施后 (h) revert 的 leftover。如果用户拍 (ν) 撤退要一起 revert 这部分；如果拍 (λ-revised) 可能重 enable V dual view 时复用。

## 8. 待用户决策

**推荐**: 优先做 K LDS dump disambig (~2-3h)，用 empirical 数据决定后续方向。

如果用户拍方向：
- **Disambig + (λ-revised)**: 先 LDS dump verify K LDS layout，根据结果推进 K reg dist coord (1-2d) 或 deep trace (3-5d)
- **(μ)**: 跳过 disambig 直接做 K + V 双 shuffle stage (2-3d, perf 不确定)
- **(ν)**: 跳过 disambig 直接撤退到 B1 (1-2h, AICK-579 半完成 stays)

**Reflection (lead 可参考)**:
- 我之前两次 sign-off / status doc 的 conclusion 都基于 mentor 错前提推理
- 教训：每个 sign-off 必须独立 verify 基础前提；类比推理高风险，不能 inherit assumption
- 应该早一步主动 challenge mentor ds_load_tr semantics 描述 (我两次都 just took it)
- 后续 sequence: 改 dram dist 之前先 LDS dump baseline B1 layout，改后再 dump 比对

---

file: swe-reanalysis-K-side-correction.md
