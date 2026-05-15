# Mentor Reanalysis — ds_load_tr_b128 semantics correction + (h) fail 重审

写作时间: 2026-05-06。worktree `~/workspace/rocm-libraries-gfx1250/projects/composablekernel` (branch `yiding12/gfx1250-fmha-tdm`)。

---

## 1. 直接 acknowledge：我描述 ds_load_tr_b128 反了

我之前给 lead 的回信里说 "ds_load_tr 要求 LDS 物理 K-direction stride = 2 byte (K-inner contiguous)"，**完全反向**。

User 读 production code (B1 `qr_async_trload_policy`, verified work，21-32% 加速) 找到事实：
- `MakeVLdsBlockDescriptor` Xor=false (line 461+)：stride `(kNPerBlock, 1)` → V LDS = **K-outer N-inner row-major** (K stride = kN1 = 256 byte, N stride = 2 byte)
- `MakeVDramTileDistribution` (line 610-636)：5 sequence `[KPerThread, NumWarps, KThreadPerWarp, NThreads, NPerThread]`，每 thread vec_load 8 fp16 沿 **N 方向 contiguous**
- ds_load_tr_b128 在这个 K-outer N-inner LDS 上**已 verified work**

→ 跟我说的 "K-inner required" 直接矛盾。

## 2. PDF 再 verify — 修正后 ds_load_tr 真正 semantics

`p4vdoc/mi400_shader_programming.pdf §4.9.10.2`:
> "lane 0 loads 64 bits of contiguous memory and stores it in the matrix: K=0, M=0..7. **B-matrix loads are similar: one lane loads multiple contiguous N-values along single K-dimension index.**"

关键：B-matrix load 时 "one lane loads contiguous **N-values** along single K-dimension index" — 即每 lane 读一段 contiguous-in-memory data，这段 data 是 **fixed K, multiple N values**。

→ Memory 必须 N 是 fast direction (contiguous within fixed K) = **K-outer N-inner row-major**。

Hardware transpose：
- **Input LDS**: K-outer N-inner row-major (每 row 是 N-contiguous slice)
- **Output VGPR**: K-inner per-lane vector tile (给 wmma B operand 期待的 K-vector 格式)

→ ds_load_tr_b128 的 transpose 方向就是 **N-inner LDS → K-inner VGPR** (N↔K swap)。**这正是 V LDS 现状 (kK1, kN1) row-major 的格式**，跟 production 一致。

## 3. V "物理不兼容" conclusion 完全 invalidate

我之前说 "TDM box-major write + ds_load_tr 物理上不可同时满足" 错了。

正确认识：
- V LDS row-major (kK1, kN1) = K-outer N-inner = **正是 ds_load_tr expected input layout**
- TDM box-major write 到这个 LDS layout 应该是 ds_load_tr-compatible 的 — 假设 TDM 真做 dram→LDS plain row-major copy

(g-revised) dual view 路径设计上是**对的**（V dram_naive 物理 (kK1, kN1) 跟 V LDS 同顺序，dual view 跳过 transform_tensor_view 让 TDM 走 (kK1, kN1)）。

那为什么 step 4 (K+V trivial) + (h) hybrid 都 element permute？

## 4. (h) hybrid fail 真根因重审

(h) hybrid 数据：
- K trivial tile-major TDM write + K reg dist (BWarpDstrEncoding 不变)
- V revert 回 B1 (5D dist + async_load + ds_load_tr) — V 完全 verified path

V 不动 → V 不可能是 bug 来源 → **bug 必在 K side**。

K side 链路重审：

**K LDS (kN0, kK0) row-major (line 400-405)**：byte(n, k) = n*kK0 + k = n*64 + k*2
- K is inner (stride 2 byte), N is outer (stride 64 byte)
- 这是 wmma B operand 自然 LDS layout（QK GEMM K 是 B operand，K-inner per-lane vec）

**K 用普通 ds_load** (不是 ds_load_tr) — 调用 `load_tile(k_lds_read_window)` with K reg dist (`MakeKRegTileDistribution` BWarpDstrEncoding)
- ds_load 是 element-by-element，按 thread mapping 算 LDS coord
- 不需要 hardware transpose
- LDS K-inner row-major + wmma B thread mapping → ds_load 拿出来正确给 wmma B operand

理论上：如果 TDM trivial tile-major write 真做 dram→LDS plain row-major copy，K side 完全 OK。

**实际 fail mechanism candidate** (user hint)：

A) **K reg dist 跟 K trivial tile-major dram dist 不 coordinated**
- B1 K reg dist 是基于 "B1 K dram dist 5D + async_load 写出来的 LDS layout" 设计的
- B1 K dram dist 写 LDS 的物理 byte layout 可能**不是 plain row-major** (some thread-permuted pattern reverse-engineered to match wmma B thread access)
- K reg dist 期望那个 "B1-pattern" LDS layout
- 当 K dram dist 改 trivial tile-major + TDM write 出 plain row-major LDS，K reg dist 仍按 "B1-pattern" 读 → 读到错位置 → element permute

B) **TDM trivial tile-major write 实际不是 plain row-major copy**
- IsWarpLevelParallelOnly=true 在 fmha 上下文行为没 verify
- box_dim print (32, 16) 只 confirm dist encoding 编进 binary，没 confirm 物理 byte order
- 可能 TDM box-major write 内部 byte order 不是按 LDS row-major

A 比 B 更可能（B 需要 hardware spec deep trace + LDS dump，A 只需理论分析）。

**Disambig**: 加 K LDS dump after TDM write，对照 dram K tile：
- 如果 LDS 是 plain row-major (kN0, kK0) → confirm A (K reg dist 跟 trivial tile-major 不 coordinate)
- 如果 LDS 不是 plain row-major → confirm B (TDM 行为 surprising)

## 5. 我的 methodological errors（第三次同 pattern reflection）

User explicit point out：
1. (λ) sign-off 基于 GEMM v1 untested reference design
2. (h) sign-off 没 challenge K side 假设
3. ds_load_tr semantics 给完全反向描述

共同 pattern：**理论 reasoning 没配 empirical verify**。

具体反思：
- ds_load_tr semantic 描述错是因为我用 "wmma B operand expects K-inner per-thread vector" 推到 "LDS 也必须 K-inner"，没意识到 transpose op 的整个意义就是让 LDS 跟 VGPR layout 反向
- 应该先看 production code（B1 已 verified V LDS 是哪个 layout）再 reason，不是 first-principle 推
- PDF §4.9.10.2 textual description "B-matrix lane reads contiguous N-values along K" 已经直接答了，我没仔细读这句

承认 reviewer warning 完全对："理论 reasoning 必须配 empirical verify (LDS dump) 才稳"。

## 6. (ν) 推荐 reassess — 不再 "推 ν 撤退"

ds_load_tr semantics 修正后，**(ν) 不再是唯一合理选项**。

修正后选项：

**(λ-revised-downscoped)**: K side coordination
- 范围：调整 K reg dist 让它 align "TDM trivial tile-major 写出的 LDS layout"
- 假设 A 成立：K reg dist 现在期望 B1-pattern LDS，要改成期望 plain row-major LDS
- 工作量：1-2 SWE-day（重设 K reg dist + verify ABC）
- V 完全不动（V 走 B1 verified path）
- 风险：assumption A 可能错（实际是 B），需要 K LDS dump disambig
- **upside**：K-TDM 真 enable，AICK-579 partial 完成度提高（K 走 TDM, V 走 async）

**(λ-revised-full)**: K + V 都改 (λ) trivial tile-major
- K 同上
- V 走 (g-revised) dual view + V dram dist trivial tile-major
- V LDS 物理仍 (kK1, kN1) match B1 verified ds_load_tr input
- V reg dist 也可能要 coordinate
- 工作量：3-4 SWE-day
- **upside**：K + V 全 TDM，完成 AICK-579 B2 full intent

**(ν)** 撤退：仍是 fallback，不再 default 推荐

## 7. 推荐 sequence（修正后）

1. **Step A: K LDS dump disambig** (~2-3h)
   - 加 device-side LDS dump after K TDM write
   - 对照 dram K tile：confirm LDS 是不是 plain row-major (kN0, kK0)
   - 如果 yes → assumption A 成立 → 进 step B (λ-revised-downscoped)
   - 如果 no → assumption B (TDM 行为 surprising) → 深 trace 或退 (μ)

2. **Step B: K reg dist 重设** (1-2 SWE-day)
   - 改 `MakeKRegTileDistribution` 让 thread mapping align plain row-major (kN0, kK0)
   - 不动 BWarpDstrEncoding 框架 (wmma B operand thread layout 是 hardware-fixed)，但需要 outer encoding 部分（warp mapping / Y dim） 跟 trivial tile-major TDM write 出的 LDS layout 一致
   - 跑 ABC 期望 K side valid:y (V 走 B1 path)

3. **Step C: V trivial tile-major** (1-2 SWE-day)
   - (g-revised) dual view + V dram dist trivial tile-major
   - 跑 ABC valid:y → AICK-579 完成

4. **Plan B: 退 (ν)** if step A reveals B 或 step B 实施撞墙

## 8. Standby

按 lead 指示 standby，不开任何 implementation。等用户决策（按修正后选项）。

如果用户选 step A K LDS dump：我可以帮 SWE design dump 实现（device-side memcpy LDS region 到 host visible buffer，~50 lines code）。

如果用户选 (ν) 撤退：仍是 viable，我帮 SWE review revert 范围。

---

## References

- 源码 `block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp:461+, 610-636` (B1 verified V layout reference)
- 源码 `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:400-405` (K LDS naive), `499-504` (V LDS naive)
- PDF `p4vdoc/mi400_shader_programming.pdf §4.9.10.2` (B-matrix lane reads contiguous N-values along K)
- 我之前 message 给 lead (over-specific 8-bank diagram + reframed K-inner conclusion) — **invalidated**

---

(end)
