# Mentor — fmha qr_tdm 设计分析（Step B2 H3 升级版根因）

写作时间: 2026-04-30 ~20:24 local。worktree `~/workspace/rocm-libraries-gfx1250/projects/composablekernel` (branch `yiding12/gfx1250-fmha-tdm`)。

目的：B2 重启时这是 design baseline。不论用户拍 (λ)/(μ)/(ν) 哪个，本文档对应的 dist 拓扑、GEMM v1 对比、line range 都直接可用。

---

## 1. fmha qr_tdm 3-dist 拓扑（实测）

实测来自 `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp:285-303`（batch path；group path 同结构 line 813+）：

```cpp
// (D1) dram window dist —— TDM box_dim 来源
auto k_dram_window =
    make_tile_window(k_dram_block_window_tmp,
                     {physical_seqlen_k_start, 0},
                     Policy::template MakeKDramTileDistribution<Problem>());

// (D2) LDS write window —— 不传 dist (plain offset)
auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
    static_cast<KDataType*>(smem_ptr),
    Policy::template MakeKLdsBlockDescriptor<Problem>());           // Xor=false naive 2D
auto k_lds_write_window =
    make_tile_window(k_lds_write_view,
                     Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
                     {0, 0});                                       // 无第 4 个 dist 参数

// (D3) LDS read window dist —— ds_load 用
auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
    static_cast<KDataType*>(smem_ptr),
    Policy::template MakeKLdsBlockDescriptor<Problem, false, true>()); // Xor=true swizzled (B2 H4' 已 disable)
auto k_lds_read_window =
    make_tile_window(k_lds_read_view,
                     make_tuple(number<kN0>{}, number<kK0>{}),
                     {0, 0},
                     Policy::template MakeKRegTileDistribution<Problem>());
```

V 同结构（line 311-340），用 `MakeVDramTileDistribution` / `MakeVLdsBlockDescriptor` / `MakeVRegTileDistribution`。

### 3 dist 各自来源

| Dist | 来源函数 | Policy 文件 line | 服务对象 |
|---|---|---|---|
| D1 dram | `MakeKDramTileDistribution<Problem>` | `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:152-178`（B1 抄 async_trload_policy:143-168） | TDM box_dim + dram thread coord |
| D2 LDS write | 无 (lengths-only window) | — | TDM 写入 LDS 落点（用 lds_coord 的 plain offset） |
| D3 LDS read | `MakeKRegTileDistribution<Problem>` | `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:581-610`（B1 fork from async_trload_policy:581-619） | ds_load thread-i 读 LDS 哪个位置；下游喂 BlockGemm 0 (QK GEMM) 作 A operand |

V 对应：`MakeVDramTileDistribution` (policy:625+), `MakeVLdsBlockDescriptor`, `MakeVRegTileDistribution` (policy:678+)。

### TDM 写 → ds_load 读 路径

```
load_tile_tdm(tdm_config, k_lds_write_window, k_dram_window)
    ↓
tile_window.tdm_load_to_lds (tile_window.hpp:851)
    ↓
[box_dim 来自 k_dram_window 的 D1 dist; thread coord 来自 D1; 写入 lds_write_window 的 lds_coord 起点]
    ↓
TDM hardware 把 dram 数据按 box-major contiguous 写到 LDS 起点 + box_dim 形状

... [s_wait_tensorcnt_barrier 同步] ...

load_tile(k_lds_read_window)
    ↓
tile_window 用 D3 dist (MakeKRegTileDistribution) 算 thread-i 读 LDS 哪里
    ↓
ds_load 拿出来作为 BlockGemm 0 的 A operand
```

**关键 mismatch 来源**：
- D1 (dram dist) 决定 TDM box-major write 模式
- D3 (LDS read dist) 决定 ds_load thread-i 读位置
- D1 ≠ D3 in general
- async_load 路径下兼容是因为 buffer_load 按 thread-i dist-mapped 写 LDS，跟 D3 自然 align（async_load 实际也是按 D1 写但 thread-i 端，效果上 D1==D3 alignment 在 thread 维上一致）
- TDM write 不是 thread-mapped 而是 box-mapped → 跟 D3 dist-pattern 不一致 → ds_load 读到错位置 → 一致 garbage

---

## 2. GEMM v1 (BlockGemm dist + TDM write co-design) vs fmha B1 (async_load 脚手架)

### GEMM v1 实测：`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp:325-360`

```cpp
// LDS read tile dist 来自 BlockGemm（跟 TDM write pattern 一起设计）
constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
    BlockGemm::MakeABlockDistributionEncode())){};
using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
ALdsTile a_block_tile[2];

// TDM write 用 a_copy_lds_windows[I0/I1] (跟 a_copy_dram_window 一起在 Base::GetAWindows 创建)
Base::GlobalPrefetchTDM(tdm_config_a, a_copy_lds_windows[I0{}], a_copy_dram_window, ...);
```

**GEMM v1 设计哲学**：
- LDS read dist = **BlockGemm 框架统一发**（`BlockGemm::MakeABlockDistributionEncode()`），跟 TDM write 的 dram dist 一起设计 → 保证 thread-i 读到的是 thread-i 期望的数据
- TDM write 走 GlobalPrefetchTDM helper，内部 `load_tile_tdm` 写到 LDS write window

### fmha B1 设计哲学（async_load 脚手架）

- LDS read dist = `MakeKRegTileDistribution<Problem>` —— 这是 **async_load 时代设计**：thread-i 用 buffer_load 写 LDS thread-i 位置，ds_load 用同样 thread-mapped dist 读
- D1 = `MakeKDramTileDistribution<Problem>` —— 也是 async_load 设计：每 thread 决定从 dram 读哪 16 elements
- D1 跟 D3 是 **per-thread dist-mapped 一致**，async_load 链路 OK
- 但 TDM 不按 dist 写 LDS（按 box-major），D1 跟 D3 alignment 假设崩

### 设计对比

| 维度 | GEMM v1 | fmha B1 (qr_tdm) |
|---|---|---|
| LDS read dist 来源 | BlockGemm 统一发 | fmha policy 自己写（fork from async） |
| dram dist 与 LDS read 关系 | 不同但 co-design 保证一致 | 相同设计（async-mapped），假设 thread-i 一致 |
| TDM write 模式 | box-major（通过 BlockGemm dist 适配） | box-major（D1 dist 算 box_dim，但跟 D3 不 align） |
| TDM 真用过 | 写在 production code | gfx1250 build 0 instance 历史 |
| 跑通 verify 状态 | build-gfx1250 也 0 instance（codegen 没 append GEMM TDM）| B2 trial 100% wrong |

**GEMM TDM 在 build-gfx1250 也 0 instance**——这意味着 GEMM v1 的 dist co-design 设计对**也未真跑通过**。如果 GEMM TDM 启用，可能也会撞类似 dist 兼容问题。但 GEMM v1 至少结构上是 co-design，fmha B1 完全不是。

---

## 3. (λ)/(μ) 实施 — 具体文件 + line range

### (λ) 重设全 3 dist 让 TDM write box-major + LDS read 同 box-major + BlockGemm 兼容

**主改文件**：

**(λ-1) 重写 K dram dist + reg dist**
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:152-178` (`MakeKDramTileDistribution`)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:581-610` (`MakeKRegTileDistribution`)
- 两个必须 align ↔ 让 thread-i 写 box 跟 thread-i 读 box 一致

**(λ-2) 同样改 V**
- `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:625-660` (`MakeVDramTileDistribution`)
- `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:678-720` (`MakeVRegTileDistribution`)
- V 走 ds_load_tr_b128，trload 对 box layout 有额外约束（行/列 transpose 模式），box_dim 需要适配 trload pattern

**(λ-3) BlockGemm A operand layout 兼容**
- 改了 D3 (LDS read) 后，`load_tile(k_lds_read_window)` 拿出来的 thread tile shape 变了
- 喂给 `gemm_0` (QK GEMM) 时，BlockGemm 0 的 A operand expected layout 必须 match
- 看 `GetQKBlockGemm<Problem>()` 在 policy 里返的 BlockGemm 类型，看其 A operand 期望 distribution
- 可能需要在 BlockGemm 0 这边给 wrapper 重 dist 适配

**(λ-4) verify**
- 重跑 ABC 对账，至少 A/B/C 三个 case 全 valid:y
- 跟 B1 baseline 对 perf：A 2539 / B 374 / C 9113 ms
- 期望 perf 优于 B1（TDM 真启用应该更快）

**风险**：
- B1 教训"动 dist 风险高"完全适用 —— 4 个 Make* 函数 + BlockGemm wrapper 共 5 处需协调，很容易某处漏 align 导致编过但 valid:n
- BlockGemm 0/1 layout 不熟，可能踩 wmma operand layout 兼容坑
- 工作量估 1-2 周（设计 + 实现 + verify）

### (μ) 加 LDS shuffle 中间层

**思路**：TDM 写 LDS 用 plain box-major naive layout（不依赖 D3）→ 加一步 LDS-to-LDS shuffle 把数据重排成 D3 期望的 layout → ds_load 走原 D3 dist 读

**主改文件**：

**(μ-1) Pipeline 加 shuffle stage**
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (1232 行 pipeline)
- 在 line 359 `load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window)` 之后插入：
  - `s_wait_tensorcnt_barrier<0>()` 等 TDM 完成
  - 一个 LDS-to-LDS shuffle pass（用 ds_load + ds_store 或 async_copy_lds 重排）
  - 输入：TDM 写到的 lds_write_view layout
  - 输出：D3 期望的 lds_read_view layout
- V 同样加（line 461 后）

**(μ-2) Shuffle 实现选择**
- 选 1: `shuffle_tile` API （ck_tile 已有 `core/tensor/shuffle_tile.hpp`）—— 用 distribution 重排 register tile，再 ds_store 回 LDS
- 选 2: 自己写 ds_load box-major + ds_store dist-pattern 两步（直接 LDS 内重排）
- 选 1 更 idiomatic 但要 round-trip register；选 2 更直接但要手写

**(μ-3) Padding/swizzle 重新评估**
- TDM 写 plain naive LDS 不需要 padding 也不能 swizzle
- ds_load 读取 D3 期望的 layout 可能需要 swizzle 避 bank conflict
- 中间 shuffle stage 可能直接做 padding/swizzle 转换

**(μ-4) verify** 同 (λ-4)

**风险**：
- Perf 退步显著：多一次 LDS BW + 多一次 sync
- Shuffle stage 设计正确不容易，需要确保 thread-by-thread 写读一致
- 但**dist 全不动**——B1 教训风险低
- 工作量估 3-5 天

### (ν) 战略撤退保留 B1

**主改文件**：
- 撤回 worktree 当前 8 个 modified（保留 (X+) CK core 修复 push 单独 PR；撤回 fmha-side TDM 改）
- 恢复 B1 状态：`block_fmha_pipeline_qr_ks_vs_tdm.hpp` 用 async_load_tile + ds_load
- 关闭 codegen `qr_tdm` instance 生成
- 文档：B2 状态记 "未达成；保留 B1 21-32% 加速"

**工作量估**：1-2 天（清理 + 文档）

**Side benefit**: (X+) CK core fix（5 处 in-place ops + inclusive_scan）独立 PR 仍价值大——下次任何 mixed tuple TDM 用法不再撞 type system 坑。

---

## 4. (λ-test) 为什么不可行 — 3-dist coupling 是根本原因

之前给 SWE 的 push back（已发）：

**单改 D1 (`MakeKDramTileDistribution`)**：
- TDM box_dim 变 trivial linear (16, 1)
- 但 D3 (`MakeKRegTileDistribution`) 没动 → ds_load 仍按 async_mapped dist 读
- → 仍 mismatch，valid:n 但**不能 disambig**：可能仍是 dist mismatch（confirm H3）也可能是别的（无法 disambig）

**同改 D1 + D3**：
- D3 跟 BlockGemm 0 A operand expected layout 强绑定
- 改了 D3 → A operand layout 错 → QK GEMM 计算错 → P matrix garbage
- → P matrix 错叠加 LDS layout 错 = **双重 garbage**，一致 valid:n 但不能 disambig 哪一层错

**同改 D1 + D3 + BlockGemm A operand wrapper**：
- 这就是 (λ) 本身了，不是 disambig

→ **λ-test 跟 (λ) 工作量相当**，做 disambig 不如直接做 fix。

**正确 disambig 应该绕过 dist 改造**：
- LDS dump verify（device-side memcpy 到 host visible buffer，跟 dram K 对账）—— deterministic
- 单 thread case（blockSize=1，dist 退化到 trivial linear naturally）—— 但 fmha codegen 不一定支持

但已 6+ trial 历史，**继续 disambig 边际收益低**，应直接拍 fix 方向。

---

## 5. References / Sources

### CK 代码（all paths under `~/workspace/rocm-libraries-gfx1250/projects/composablekernel/`）

- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` (1232 行) — qr_tdm pipeline
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` (827 行) — qr_tdm policy
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` — B1 抄的 reference policy
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp` — GEMM TDM v1，3-dist co-design 范例
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp:1131-1192` — GEMM `GetLdsPaddingConfig`（B2 padding policy 抄的 reference）
- `include/ck_tile/core/tensor/tile_window.hpp` — tile_window 实现
  - line 851-953 `tdm_load_to_lds`
  - line 1779-1794 `get_cached_global_strides`（H8 false lead）
- `include/ck_tile/core/tensor/buffer_view.hpp:895-953` — `tdm_get` 调 createTDMDescriptor
- `include/ck_tile/core/tensor/load_tile.hpp:188` — `load_tile_tdm` free function
- `include/ck_tile/core/arch/amd_tdm_descriptor.hpp` — TDM descriptor 字段定义
- `include/ck_tile/core/tensor/tensor_view.hpp:660-703` — pad_tensor_view 实现
- `include/ck_tile/core/container/container_helper.hpp:325-344` — container_reverse_inclusive_scan（X 修过；TODO 注释 line 324）
- `include/ck_tile/core/container/tuple.hpp:695-769` — operator+/-/* in-place vs generate_tuple 版本

### Memory（`~/.claude/memory/`）

- `reference_mi450_hw_specs.md` — MI450 硬件参数速查
- `reference_mi450_kernel_patterns.md` — TDM/swizzle/padding 通用模式
  - 关键引用："Swizzling: MI450 通过 TDM 不支持，只 async load 支持"（→ 解释 H4 swizzle 关掉但 valid 不变 = swizzle 不是根因）
  - "TDM 不交错：一条 TDM 完全展开后才下一条"
- `reference_ck_tdm_api.md` — TDM descriptor 字段全集 + GEMM v1/v2 设计选择
- `project_gfx1250_fmha_fwd_design.md` — GFX IP team baseline (LDS padding K=16B/256B / V=32B/256B 的来源)
- `project_gfx1250_fmha_tdm_v1.md` — Step B1 完成状态 + 6 条教训
- `reference_amd_chip_naming.md` — gfx1250 / mi450 关系

### 其它 doc

- 我没找到的 reference: TDM hardware spec PDF（`p4web.amd.com:1712//gfxip/mi400/...`），SP3 listing
- 也未直接看：BlockGemm 0/1 A/B operand expected layout spec —— (λ) 实施时必读

---

## 6. 给 B2 重启会话的快速 refresh

如果用户拍 (μ)：
- 主修 `block_fmha_pipeline_qr_ks_vs_tdm.hpp` line 359/461 (K/V 各加一次 TDM-后 shuffle stage)
- 不动 policy
- 不动 BlockGemm
- Verify perf 退步幅度（vs B1 baseline ABC 2539/374/9113 ms）

如果用户拍 (λ)：
- 4 个 Make* 函数全改（K/V dram + reg dist）
- 加 BlockGemm A operand wrapper（QK GEMM 0）
- 工作量大；但完全 TDM-native，长期对
- (X+) CK core fix 仍有效保留

如果用户拍 (ν)：
- 撤 fmha-side TDM 改（保留 B1 状态）
- (X+) CK core fix 单独 PR
- B2 状态记未达成

---

(end)
