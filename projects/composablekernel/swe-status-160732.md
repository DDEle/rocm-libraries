# Step B2 (D2) 反证 mentor 假设 — 根因转 make_tile_window

## D2 编译诊断结果（QA `compile-stepB2-D2-160524.log`）

| 位置 | type |
|---|---|
| Kernel:1649 (K **naive**)              | `tuple<int, int>` ✓ |
| Kernel:1660 (K **padded** after pad_tensor_view) | **`tuple<int, int>`** ✓ |
| Pipeline:295 (Q dram_window lengths)   | `tuple<int, constant<128>>` |
| Pipeline:298 (K dram_window lengths)   | **`tuple<int, constant<0>>`** |

## mentor 假设被反证

mentor 16:04 强嫌疑：**pad_tensor_view** 的 mixed 类型推导引入 constant<0>。
**实测**：K naive 和 K padded **都是** `tuple<int, int>`（全 runtime int），pad_tensor_view **没动 lengths type**。

constant<0> 是在 **kernel padded view → pipeline operator() 入口的 k_dram_window**之间的某一步引入的。

## 嫌疑点已锁定（fmha_fwd_kernel.hpp:1715-1718）

```cpp
auto k_dram_window = make_tile_window(
    k_dram,                                                              // (int, int) 全 runtime
    make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),  // (constant<64>, constant<32>) tile
    {0, 0});
```

这是 K `k_dram_block_window_tmp`（传给 pipeline operator() 的实参）。注意是 **2-arg make_tile_window 不带 distribution**。

QA 推测：
- `make_tile_window` 内部把 `(constant<64>, constant<32>)` tile lens 跟 `(int, int)` dram lens 合并/embed
- 某个 transform 把 hdim 维拆成 (residual=0, K0=4, K1=8) → `unmerge<constant<0>, constant<4>, constant<8>>` 即 trace 里看到的
- constant<0> 来自 hdim_q=128 mod kK0=32 = 0 的 residual computation

对比 Q 创建（kernel:1700-1713 类似 pattern）：tile 用 `(kM0=64, kSubQKHeaddim=128)`，hdim_q=128 == kSubQKHeaddim=128，**ratio=1**，可能 special case 让 hdim 维直接成 `constant<128>` 不走 residual unmerge。

## 一致性 confirm

- **K dram window lengths** = pipeline 见到的最终 type = `tuple<int, constant<0>>` ✓
- **kernel padded** = `tuple<int, int>` (D2 实测)
- **`make_tile_window`** = 中间引入 constant<0> 的步骤（强嫌疑，未 print confirm）

## 推荐下一步诊断（5 min 再编一次）

在 fmha_fwd_kernel.hpp:1718 后加第三个 CK_PRINT：
```cpp
auto k_dram_window = make_tile_window(k_dram, ..., {0,0});
CK_PRINT<decltype(k_dram_window.get_bottom_tensor_view()
                      .get_tensor_descriptor().get_lengths())>();
```

如果 print 出 `tuple<int, constant<0>>` → constant<0> 在 `make_tile_window` 引入 (确认 root cause 在 CK core `make_tile_window` 实现的 lengths 推导，跟 fmha 无关 — fmha 调用方式正确)
如果 print 出 `tuple<int, int>` → constant<0> 在 pipeline 内的 `make_tile_window(k_dram_block_window_tmp, {origin}, MakeKDramTileDistribution<>())` 引入 (3-arg 版本带 distribution，distribution 引入)

两者是不同 fix 方向：
- 前者：CK core `make_tile_window` 推导 bug，需修 core 或换 API
- 后者：fmha pipeline 的 K dist 引入，可改 dist 或 wrap

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- Unstaged dirty:
  - 3 plumbing（lead 加，不动）
  - 2 B2 真修复（pipeline + policy，本身正确）
  - 1 D1 CK_PRINT (pipeline 295/298)
  - 2 D2 CK_PRINT (kernel 1649/1660)
- 全部 diagnostic CK_PRINT 待回退（不 commit）

## 等你拍

1. 继续 **D3 诊断**（再加 1 处 CK_PRINT 在 kernel:1718 后，5 min 派 QA 编一次）—— SWE 倾向
2. 直接选方向（α/δ/γ）—— 但根因还没 100% 锁，怕选错

不选方向，等你拍 D3 还是其它。

---
file: swe-status-160732.md
