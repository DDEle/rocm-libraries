# Mentor Hardware Knowledge — Fixed Baseline (post 三次 sign-off correction)

写作时间: 2026-05-06。Worktree `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`。

**用途**: Fresh mentor spawn 时**先读这一份**就能 immediate 准确理解 ds_load_tr / TDM / wmma B operand 等关键 hardware concept，避开前任 mentor 三次 sign-off 错坑。

跟 SWE doc `B2-resume-baseline-v3.md` cross-link：
- 本 doc = **hardware fact + anti-pattern reflection**
- SWE doc = trial timeline + fix candidates + worktree state

---

## 1. ds_load_tr_b128 真实 hardware semantics

### ISA 定义

PDF `p4vdoc/mi400_shader_programming.pdf §4.7.2.4` (LDS to VGPR Matrix Load with Transpose) + §4.9.10.2 (WMMA Matrix Load Ops with Transpose):

- **Mnemonic**: `DS_LOAD_TR16_B128` (GFX12 / gfx1250); CDNA4 旧名 `ds_read_tr_b128`
- **wave32-only**, **EXEC ignored / forced all-ones**
- **Per-lane I/O**: 输出 128-bit (= 8 fp16 / bf16) → 4 consecutive VGPRs/lane
- **Wave-level total**: 32 lanes × 128 bit = 4096 bit = **16×16 fp16 tile**
- **Lane addr**: 每 lane 通过 VADDR (per-lane VOFFSET) 独立 provide LDS addr — **lane 之间 addr 完全自由**（任意 stride、跨 fragment、非邻近 OK），约束只是 **per-lane 128-bit input 必须 contiguous in LDS (16 byte)**

### LLVM intrinsic

源码 `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp:3156-3204` (gfx1250 path):
```cpp
__builtin_amdgcn_ds_read_tr16_b128(p_lds)
```
- 接 single base ptr，per-lane offset 由 caller 编进 lane addr 计算
- 硬件 wave-internal lane permute (固定 transpose pattern) 把 32 lanes × 8 fp16 = 256 fp16 重新排列写 4 consecutive VGPRs

### Transpose direction (关键 — 之前描述反了)

PDF §4.9.10.2 textual:
> "B-matrix loads are similar: one lane loads multiple contiguous **N-values** along single **K-dimension index**."

每 lane 读的 8 contiguous fp16 = **fixed K, multiple N values** → memory 必须 N-direction contiguous within fixed K = **K-outer N-inner row-major LDS layout**。

→ Hardware transpose 方向：
- **Input LDS**: K-outer N-inner row-major (每 row = N-contiguous slice)
- **Output VGPR**: K-inner per-lane vector tile (给 wmma B operand 期望的 K-vector 格式)
- = **N↔K direction swap in transpose**

### ⚠️ 之前 mentor 错描述 (avoid 重复)

之前 mentor (我) 给 lead/user 的描述：
- ❌ "ds_load_tr 要求 LDS 物理 K-direction stride = 2 byte (K-inner contiguous)" — **完全反向**
- ❌ "8-bank/16-col 几何 fixed pattern, byte 0/32/64..." — **over-specific GEMM v1 case，不是 hardware fundamental** (用户 challenge 对：lane addr 自由)

正确版本：**LDS K-outer N-inner row-major (任何 stride OK 只要 per-lane 8 fp16 contiguous)**，hardware transpose 输出 K-inner VGPR。

---

## 2. B1 production verified code = ground-truth reference

**任何 hardware semantic 描述前必须先 reference 这份代码**。`block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` 是 **production verified path** (`yiding12/gfx1250-fmha-tdm` branch B1 实测 ABC valid:y + perf -21~32% 加速):

### V dram dist (line 610-636 in async_trload_policy)

```cpp
constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;     // hdim_v = 128
constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;     // seqlen_k_chunk = 64
constexpr index_t MaxVectorSize = 16 / sizeof(VDataType);

constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
constexpr index_t kMaxVecLoad   = min(ElemPerThread, MaxVectorSize);
constexpr index_t NPerThread    = kMaxVecLoad;     // 8 fp16 per thread
constexpr index_t NThreads      = kNPerBlock / NPerThread;
constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
constexpr index_t NumWarps      = kBlockSize / get_warp_size();   // 4 warps
constexpr index_t KPerThread    = kKPerBlock / (KThreadPerWarp * NumWarps);

return make_static_tile_distribution(
    tile_distribution_encoding<sequence<1>,
                               tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                     sequence<NThreads, NPerThread>>,
                               ...>{});
```

→ V dram dist 5D `<KPerThread, NumWarps, KThreadPerWarp, NThreads, NPerThread>`：
- X[0] = K dim (kK1=64), 3 sequences
- X[1] = N dim (kN1=128), 2 sequences
- 每 thread vec_load 8 fp16 沿 **N 方向 contiguous**

### V LDS layout (Xor=false path, line 461+ in async_trload_policy)

```cpp
return make_naive_tensor_descriptor(
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),     // (kK1, kN1) = (64, 128)
    make_tuple(number<kNPerBlock>{}, number<1>{}),              // stride (kN1, 1)
    ...);
```

→ V LDS = **(kK1, kN1) row-major**：
- byte(k, n) = (k * kN1 + n) * sizeof(VDataType) = k*256 + n*2
- **K stride = 256 byte (outer)**, **N stride = 2 byte (inner)**
- 即 **K-outer N-inner row-major**

### 一致性 check

- V dram dist N-inner thread vec-load (8 fp16 contiguous along N)
- V LDS N-inner row-major (N stride 2 byte)
- ds_load_tr_b128 expects K-outer N-inner LDS input
- **三者全一致** → B1 production work ✓

### 任何 hardware 推理前先看这

**Rule**: 给 user/lead 任何 hardware semantic 描述前，**先 grep 这两个函数 (V dram dist + V LDS)**，看实际 production code 的 layout 选择，**不 first-principle 推**。

---

## 3. TDM write hardware semantics (TENSOR_LOAD_TO_LDS)

### ISA 定义

PDF §4.10 (Tensor DMA / TDM) + `p4vdoc/mi400_tensor_dma.pdf`:

- **Op**: `TENSOR_LOAD_TO_LDS` (5D box-major copy from dram to LDS)
- **EXEC ignored** — TDM 是 wave-level instruction (不是 per-thread)
- **TDM unit count**: 1 per SIMD-pair (gfx1250 wave32, 4 SIMDs/WGP, 2 SIMD-pairs/WGP → 2 TDM units/WGP)
- **Tracking counter**: TENSORcnt → `S_WAIT_TENSORCNT N`

### Box-major write layout

- TDM 把 dram tile (按 box_dim 几何) **row-major copy** 到 LDS box 起点 + box_dim 形状
- dram element (i, j) within box → LDS element (i, j) (relative to box start)
- LDS coord 起点 = `lds_window_origin + window_adaptor_thread_coord.get_bottom_index()` (`tile_window.hpp:881-882`)
- = dram thread coord 直接当 LDS coord 用

### 关键约束

- ⚠️ **TDM 隐式假设 dram tile shape = LDS shape (同维度顺序)**：dram thread coord 直接当 LDS coord，violation → coord 落到错维度 LDS 位置
- ⚠️ **TDM 不支持 transpose write**：dram→LDS 是直接 copy，要 transpose 必须 LDS view 端做 (e.g. transform_tensor_view) 或加 LDS shuffle stage
- ⚠️ **TDM 不支持 write-端 swizzle** (`reference_mi450_kernel_patterns.md` line 34): "Swizzling: MI450 通过 TDM 不支持，只 async load 支持"

### box_dim 字段

- TDM descriptor (`amd_tdm_descriptor.hpp`) 字段 `tile_dim0/1/2` (mode0 supports 5D)
- box_dim 来自 dist encoding `ys_to_d_descriptor.get_lengths()` reverse 投影 (`tile_window.hpp:885-886`)
- **box_dim 是 deterministic constexpr 来自 dist encoding** — print 出来可 verify dist 编进 binary
- **但 box_dim 改变不直接 verify "TDM dram→LDS element 1:1 copy 正确"** — 这是前任 mentor 第二次 sign-off 错的关键: box_dim verify ≠ functional verify

### IsWarpLevelParallelOnly trait

`make_static_tile_distribution(encoding, bool_constant<true>{})` 第二参数:
- = `IsWarpLevelParallelOnly_` flag
- True 时表示 dist 是 warp-only parallel (lanes within warp 共享 thread coord, 没 lane-level split)
- GEMM v1 default policy trivial tile-major dist 用 true (warp 切 outer, thread 整段 vector)
- fmha B1 5D dist 不传 (默认 false, thread 在多维都切)
- ⚠️ **fmha 上下文下 IsWarpLevelParallelOnly=true 的实际 TDM write 行为没 verified** — 前任 mentor 第二次 sign-off 错的潜在 mechanism (假设 GEMM v1 模式直接 mirror, 实际可能 diverge)

---

## 4. WMMA B operand layout (gfx1250 wave32)

### ISA

PDF §4.6.12 (Wave Matrix Multiply Accumulate) + reference `~/.claude/memory/reference_mi450_hw_specs.md`:

- gfx1250 wmma 主要规格：**16×16×32 BF16/FP16**, 16×16×64 FP8, 16×16×128 MX FP8/FP4
- wave32: 32 lanes 一次 issue 1 wmma instruction
- **B operand for 16×16×32**: K=32, N=16 → 总 512 elements
  - 32 lanes × 16 elements per lane = 512 ✓
  - 每 lane 拿 K-inner vector (8 K-elements per lane × 2 lanes per N column)
- B operand thread-i 拿的 fragment 是 **fixed N column 的 K-direction 8 elements** (跟 ds_load_tr_b128 output 一致)

### CK BWarpDstrEncoding

`MakeKRegTileDistribution` (policy:585-616) 用 `WarpGemm::BWarpDstrEncoding{}` embed 到 outer encoding:

```cpp
constexpr auto k_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
    k_block_outer_dstr_encoding,                     // outer: warp split + per-warp wmma tile count
    typename WarpGemm::BWarpDstrEncoding{});         // inner: lane wmma B mapping (hardware-fixed)
```

- **Inner (BWarpDstrEncoding)**: lane 内 wmma B operand thread-to-element mapping，**hardware-fixed**，不能改
- **Outer encoding**: warps 在 (kN0, kK0) 维上怎么 split + 每 warp 处理几个 wmma tile，**software 可调**

### 跟 ds_load_tr / ds_load 的关系

- **K side** (BlockGemm 0 QK GEMM, K = B operand): 用 **普通 ds_load** (不是 ds_load_tr)
  - K LDS 物理 layout (kN0, kK0) row-major K-inner (`MakeKLdsBlockDescriptor`, line 400-405)
  - ds_load element-by-element，按 reg dist (BWarpDstrEncoding embed) 算 thread-i 在 LDS 的 coord
  - LDS K-inner 直接 match wmma B K-vector thread mapping → 不需要 hardware transpose

- **V side** (BlockGemm 1 PV GEMM, V = B operand): 用 **ds_load_tr_b128**
  - V LDS 物理 layout (kK1, kN1) row-major K-outer N-inner (`MakeVLdsBlockDescriptor`, line 499-504)
  - ds_load_tr 硬件 transpose K-outer LDS → K-inner VGPR (给 wmma B operand)

→ K 和 V 用不同 LDS layout 是因为 **K 直接 ds_load，V 经 hardware transpose**。两条路径都 verified work in B1。

---

## 5. 前任 mentor 三次 sign-off 错的 anti-pattern

### 第 1 次：(λ) sign-off 基于 GEMM v1 untested reference

- 推 (λ) trivial tile-major dram dist mirror GEMM v1
- 但 GEMM v1 自己 build-gfx1250 也 0 instance (`decision-B2-H3-direction-174838.md:46-47`) — **untested design**
- 没 verified runtime reference，只 design 哲学
- → fmha mirror 它撞同样未发现 mismatch 是 inevitable

**Anti-pattern**: 把 untested reference design 当 verified reference

### 第 2 次：(h) hybrid sign-off 没 challenge K side

- step 1 K only 改 trivial tile-major + V 仍走 TDM dist mismatch → max err 23.72 garbage attractor
- mentor 只看 box_dim print (8,2)→(32,16) 就 sign-off "K dist 改生效"
- 实际只 verify dist encoding 编进 binary，**不 verify TDM dram→LDS element 1:1 copy 正确性**
- step 1 V garbage 大 dominate → K mismatch element permute 看不出来
- (h) 把 V revert B1 后 K mismatch 暴露 → max err 0.36 element permute

**Anti-pattern**: box_dim verify ≠ functional verify; 不暴露的 mismatch 不等于不存在

### 第 3 次：ds_load_tr semantics 给完全反向描述

- mentor 用 "wmma B operand expects K-inner per-thread vector" first-principle 推 "LDS 也必须 K-inner"
- 没意识到 transpose op 整个意义就是让 LDS 跟 VGPR layout 反向
- PDF §4.9.10.2 textual description "B-matrix lane reads contiguous N-values along K" 直接说了 N inner，**没仔细读**
- 没看 B1 production verified code (V LDS 是哪个 layout) 直接 first-principle 推

**Anti-pattern**: first-principle 推 hardware semantic 没 reference verified code

### 共同 root

三次都是同一性质：**理论 reasoning 没配 empirical verify**。

具体 manifestation:
- 第 1 次: reference design "理论上 work" 没看实际 instance 跑没跑过
- 第 2 次: dist encoding "编进 binary" 没看 LDS dump 跑没跑对
- 第 3 次: hardware op semantic "first-principle 推" 没看 production code 实际怎么用

---

## 6. Protocol 改进 (新 mentor 必须执行)

### Rule 1: Hardware semantic 描述前先看 production verified code

**任何 hardware semantic 给 user/lead 描述前**:
1. **先 grep B1 verified path** (`block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` 关键函数)
2. **看 actual production layout choice** — 这是 ground truth
3. **再 PDF/spec verify 一致性**
4. **不 first-principle 推 — 推完先核 production code 反证**

### Rule 2: Sign-off 前写 specific empirical predict + plan disambig

**任何 sign-off 前**:
1. 写出 specific empirical predict ("如果方向对，LDS dump 应该看到 X")
2. **plan disambig step** (LDS dump / single-iter test / specific case 跑哪个 valid 反应方向对错)
3. 不依赖 "理论合理" 单独支撑 sign-off

### Rule 3: 重视 reviewer/SWE/QA challenge

User/SWE/QA challenge 时:
- **第一反应是 verify 自己的前提**，不是 push back 反驳
- 之前 reviewer "2h K LDS dump disambig" warning **一直对** — mentor 几次错过都是 ignore 这条
- Reviewer second opinion 价值大，**重视独立 verify step**

### Rule 4: 别把 GEMM v1 当 verified reference

- GEMM v1 / GEMM v2 build-gfx1250 都 0 instance (`reference_ck_tdm_api.md` + `decision-B2-H3-direction-174838.md`)
- GEMM TDM 在 build-gfx1250 没真跑通过 — 只是 design 在 source tree
- fmha 不应该当 GEMM 的 verification ground (work 量逆序)
- mirror GEMM v1 design 时**必须明确说"untested reference, mirror 它撞墙是 inevitable risk"**

### Rule 5: box_dim/dist encoding verify ≠ functional verify

- `box_dim` print 是 **constexpr from dist encoding** — 只 confirm "dist 编进 binary"
- **不 confirm "TDM dram→LDS element 1:1 copy 正确"**
- functional verify 要 LDS dump + 数据流 valid:y

### Rule 6: 任何 dist 改动配 LDS dump baseline

- 改 dram dist 之前先 LDS dump baseline B1 layout
- 改 dram dist 之后再 LDS dump 比对
- empirical 数据说话，不依赖 "理论上应该 work"

---

## 7. CK FMHA pipeline 关键文件 quick reference

### B1 verified path (production reference)

| 文件 | line | 内容 |
|---|---|---|
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` | 142-168 | K dram dist (B1 5D verified) |
| 同上 | 461-510 | V LDS naive (Xor=false) |
| 同上 | 610-636 | V dram dist (B1 5D verified) |
| 同上 | 678-720 | V reg dist |

### B2 worktree 现状 (TDM 实施)

| 文件 | line | 内容 |
|---|---|---|
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` | 152-178 | K dram dist (B1 抄 + step 1 trivial tile-major 改) |
| 同上 | 400-405 | K LDS naive (Xor=false) |
| 同上 | 499-504 | V LDS naive (Xor=false) |
| 同上 | 585-616 | K reg dist (B1 抄, **step 1 没动**, 真根因 hypothesis) |
| 同上 | 622-647 | V dram dist (B1 抄 + step 4 trivial tile-major 改 / dual view) |
| 同上 | 685-720 | V reg dist (B1 抄) |
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` | 285-340 | K/V dram + LDS window 构造 (batch path) |
| 同上 | 813+ | 同上 (group path) |
| `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` | 1652-1690 | V dram view + transpose layer (RowMajor V case) |
| 同上 | 2570-2602 | K dram unmerge (group path) — α' fix 区域 |

### Hardware infrastructure

| 文件 | line | 内容 |
|---|---|---|
| `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp` | 3156-3204 | `__builtin_amdgcn_ds_read_tr16_b64/b128` intrinsic (gfx125 path) |
| 同上 | 3223 | `amd_tdm_load(descriptor)` wrapper |
| `include/ck_tile/core/arch/amd_tdm_descriptor.hpp` | 全 | TDM descriptor 4-group SGPR 字段定义 |
| `include/ck_tile/core/arch/arch.hpp` | 1167 | `s_wait_tensorcnt<N>()` + `s_wait_tensorcnt_barrier<>` |
| `include/ck_tile/core/tensor/tile_window.hpp` | 851-953 | `tdm_load_to_lds` 实施 |
| 同上 | 881-882 | **关键 line: `lds_idx = lds_origin + window_adaptor_thread_coord.get_bottom_index()`** (TDM 隐式 dram coord = LDS coord) |
| `include/ck_tile/core/tensor/load_tile.hpp` | 188 | `load_tile_tdm` free function |
| `include/ck_tile/core/tensor/tile_distribution.hpp` | 506-540 | `make_static_tile_distribution(encoding, bool_constant<IsWarpLevelParallelOnly>)` |

### GEMM TDM (untested reference, careful 用)

| 文件 | line | 内容 |
|---|---|---|
| `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp` | 全 | GEMM TDM v1 pipeline (build-gfx1250 0 instance, **untested**) |
| `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp` | 47-94 | GEMM trivial tile-major dist (mentor 第 1 次 sign-off 错: 当 verified reference 实际 untested) |
| `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp` | 206-244 | `CopyADramWindow` / `MakeALdsWindows` 框架 |

---

## 8. Reference docs status

### 当前 valid

- 本 doc (`mentor-hardware-knowledge-fixed.md`) — fully valid
- `mentor-reanalysis-ds-load-tr-correction.md` — fully valid (ds_load_tr correction + (h) reanalysis)
- SWE doc `swe-reanalysis-K-side-correction.md` — fully valid (K side mismatch hypothesis + LDS dump plan + 工作量重估)
- SWE doc `B2-resume-baseline-v3.md` (cross-link) — fully valid (trial timeline + worktree state)

### 当前 partial valid (标注 invalidate 部分)

- `mentor-fmha-tdm-design-notes-202402.md`:
  - ✅ Section 1 (3-dist 拓扑实测) **valid**
  - ⚠️ Section 2 (GEMM v1 BlockGemm dist + TDM write co-design 哲学) **partial invalid** — co-design 表述误导，实际 GEMM v1 dram dist 是独立 trivial tile-major 不是 BlockGemm encode 推
  - ⚠️ Section 3 ((λ-1)/(λ-2) 实施 line range) **valid for line ranges**, but (λ-3) "BlockGemm A operand wrapper" 整条 **invalidate** (K 是 B operand 不是 A; 且 reg dist 不需要 wrapper, dram dist 改即可)
  - ⚠️ Section 4 ((λ-test) push back) **valid reasoning** but conclusion 基于错前提
  - ✅ Section 5 (References) **valid**

### 全 invalidate (avoid 引用)

- mentor 给 user 的 ds_load_tr "8-bank/16-col 几何" + "K-inner contiguous required" 解释 — **invalidated**
- mentor 之前 sign-off "K trivial tile-major reg dist 不动" — **invalidated** (reg dist 不 coordinate 是 (h) fail 真根因 hypothesis)

### Memory references (`~/.claude/memory/`)

- `reference_mi450_hw_specs.md` — wave32/LDS BW/wmma 规格 ✓
- `reference_mi450_kernel_patterns.md` — TDM swizzle 不支持, 通用 pattern ✓
- `reference_ck_tdm_api.md` — TDM descriptor 字段全集 ✓
- `project_gfx1250_fmha_fwd_design.md` — GFX IP team baseline (LDS padding 数值 source) ✓
- `project_gfx1250_fmha_tdm_v1.md` — Step B1 完成状态 ✓ (B2 部分 outdated, 看 SWE doc)
- `reference_amd_chip_naming.md` — gfx1250 / mi450 关系 ✓

### PDF refs

- `~/.claude/skills/rocm-ref/p4vdoc/mi400_shader_programming.pdf` §4.7.2.4 (DS_LOAD_TR), §4.9.10 (WMMA Matrix Load Transpose), §4.10 (Tensor DMA), §4.7.1 (LDS bank organization)
- `~/.claude/skills/rocm-ref/p4vdoc/mi400_tensor_dma.pdf` (TDM detail)
- `~/.claude/skills/rocm-ref/p4vdoc/sp3_mi400_instructions.pdf` (SP3 instruction listing — 未读, 如需 box_dim hardware 上限可查)

---

## 9. Spawn 后 mentor 第一件事 checklist

新 mentor spawn 时：

1. ✅ **Read this doc first** — get hardware fact baseline
2. ✅ Read `mentor-reanalysis-ds-load-tr-correction.md` (correction context)
3. ✅ Read SWE `B2-resume-baseline-v3.md` (trial state + worktree)
4. ✅ Read memory: `reference_mi450_hw_specs.md`, `reference_mi450_kernel_patterns.md`, `reference_ck_tdm_api.md`, `project_gfx1250_fmha_fwd_design.md`
5. ✅ **Grep `block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` V dram dist + V LDS** (production verified ground truth)
6. ⚠️ **不 read** `mentor-fmha-tdm-design-notes-202402.md` 直接当 baseline (partial invalidate, 看本 doc section 8 标注)
7. ⚠️ **不 read** GEMM v1/v2 当 verified reference (untested)

任何 user/lead question 关于 hardware semantic:
- 先看 production code (B1 path) 再答
- 描述前 mentally cross-check "PDF 说什么" + "B1 production code 怎么用"
- 不 first-principle 推

---

(end)
