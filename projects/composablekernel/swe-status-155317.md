# Step B2 (D) 诊断进度 + 对照困境（请 lead 拍 D1/D2/D3）

## 已做

### 1. grep CK_PRINT — 确认用法

位置：`include/ck_tile/core/utility/debug.hpp:14-21`

```cpp
template <auto... val>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT() {}
template <typename... type>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT() {}
```

机制：`[[deprecated]]` attr → 调用时编译器发 deprecated warning，warning message 含 template arg 的 type 拼写。dev preset `-Werror -Wdeprecated-declarations` 下变 error，编译失败但带 type 输出。

### 2. 加 CK_PRINT 在 fmha pipeline batch path

位置：`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` line ~290（K dram window 创建后）

```cpp
// (D) Step B2 diagnostic — probe Q/K dram tensor lengths type to confirm
// the mixed `tuple<int, constant<...>>` shape that breaks
// container_reverse_inclusive_scan in get_cached_global_strides.
// CK_PRINT is [[deprecated]]: with dev -Werror -Wdeprecated-declarations
// this aborts compile with the type spelled out in the warning message.
// Diagnostic only — DO NOT commit.
CK_PRINT<decltype(q_dram_window.get_bottom_tensor_view()
                      .get_tensor_descriptor()
                      .get_lengths())>();
CK_PRINT<decltype(k_dram_window.get_bottom_tensor_view()
                      .get_tensor_descriptor()
                      .get_lengths())>();
```

只加在 batch path（B1 跑 ABC 三个 case 都走 batch），group path 暂未加。

## 对照困境

build-gfx1250 配置下 **没有 working TDM/flat caller instance** 可对照：

- 你的 plumbing 让 build-gfx1250 只编 fmha qr_tdm fast 路径
- `18_flatmm` 和 `wp_pipeline_agmem_bgmem_creg_tdm.hpp`（唯一 `prefetch_for_flat` caller）的 build 子目录只有 `CMakeFiles`，没 cpp instance
- B1 跑通的 fmha baseline 是 async_load + ds_load_tr，**没真用 TDM**
- 整个 build-gfx1250 配置可能从未触发过 `inclusive_scan + mixed tuple` 的 working path

## 三种走法 (D1/D2/D3) 请你拍

### (D1) 单边 confirm

只 print fmha 一边，confirm 是 mixed `tuple<int, constant<128>>`

- **pro**：CK_PRINT 已加，QA 30 秒编一次拿 fmha type，根因 confirm 立即走 (X)
- **con**：mentor 期望的 "对照 working caller all-int" 对照缺失
- **SWE 觉得**：error trace 已 explicit print 过 type，CK_PRINT 是 belt-and-suspenders，单边足够 confirm root cause scope

### (D2) 双边 confirm — 扩 build target

- 改 build config 让 18_flatmm 或 wp 也编（cmake 重 configure + ~5min 编 18_flatmm）
- 让 QA 同样在 18_flatmm 加 CK_PRINT 拿 type
- **pro**：mentor 标准对照
- **con**：~5min 多编 + 改 build config（plumbing 风险）

### (D3) 双边 confirm — fmha 内部硬造 raw dram view 对照

- 在 fmha pipeline 内 own `make_naive_tensor_view<global>(... runtime ints ...)` 一个不经 pad_tensor_view 的 raw view，print 它的 lengths type
- **pro**：单次编译同时拿两边对照
- **con**：硬造的 view 不真实经过 pipeline 路径，对照说服力打折

## SWE 倾向

**(D1)**

- mentor 最主要担心是 SWE 误读 error trace 的 scope（万一 trace 里那段 `tuple<int, constant<128>>` 不是 dram_window lengths 而是别的中间 type）
- CK_PRINT 单边 print 已能 confirm scope（直接 print 的就是 q/k_dram_window 的 lengths）
- (D1) confirm mixed type 后直接走 (X) 修 CK core inclusive_scan（mentor 推荐路径）

## 下一步建议

请你拍 D1/D2/D3 哪条。

- 选 D1：我立即 ping QA 编一次（CK_PRINT 已加）
- 选 D2：我先回退 fmha probe，改 build config 加 18_flatmm，重 cmake configure 再 ping QA
- 选 D3：我在 fmha pipeline 加第二个 CK_PRINT 用 raw view 对照后 ping QA

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- Unstaged dirty:
  - 3 个 plumbing（你加的，不动）
  - 2 个 B2 真修复（pipeline + policy，本身正确）
  - 1 处 CK_PRINT diagnostic（pipeline 加的，diagnostic-only，commit 时丢弃）

**没派 QA，等你拍**。

---
file: swe-status-155317.md
