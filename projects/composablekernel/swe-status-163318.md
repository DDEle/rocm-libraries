# Step B2 — Q vs K disambiguate stacktrace 分析

## Lead 关键观察被 stacktrace 反证

Lead 之前说："Q 也有 `tuple<int, constant<128>>` (D3 实测 pipeline:295)，**Q 没 fail**；K 同样 (α' 后 D4 实测 kernel:2739)，**K 仍 fail**。"

**stacktrace 实测反转**：(α') 之后 **fail 的是 Q，不是 K**。

## stacktrace 关键证据

`compile-stepB2-alpha-verify-162808.log` 4 段重复 fail trace，每段都显示：

```
container_reverse_inclusive_scan<int, ck_tile::constant<128>, multiplies, int>
  ↑ called from
tile_window.hpp:1788   tuple_reverse(container_reverse_inclusive_scan(..., 1))
  ↑ inside
tile_window.hpp:875    auto&& global_strides = get_cached_global_strides();
  ↑ inside
tile_window_with_static_distribution<
    tensor_view<...const _Float16, ...>,
    tuple<constant<64>, constant<128>>,    ← ★ window lengths
    tile_distribution<...
        unmerge<tuple<constant<8>, constant<4>, constant<2>>, false>,    ← 第 1 dim 切分 = 64
        unmerge<tuple<constant<16>, constant<8>>, false>,                  ← 第 2 dim 切分 = 128
        ...
    >, ...>::get_cached_global_strides
  ↑ called from
load_tile.hpp:197      return tile_window.tdm_load_to_lds(tdm_config, lds_tile, ...)
```

**window lengths = `tuple<constant<64>, constant<128>>`**：
- `64 = kM0`（Q 的 M 维 tile）
- `128 = kSubQKHeaddim`（Q 的 hdim tile）
- → 这是 **Q dram_window**，不是 K（K 的 tile = `(kN0=64, kK0=32)`）

dist encoding 也对得上 Q：(8,4,2) product=64=kM0, (16,8) product=128=kSubQKHeaddim。

## 重新解读 (α') 前后

| | (α') 前 (D3 trace) | (α') 后 (verify trace) |
|---|---|---|
| 第一个 fail trace | K dram (constant<0> 来自 unmerge typo) | **Q dram (constant<128> 来自 pad_tensor_view)** |
| K 状态 | fail (lengths=int,const<0>) | **fix 了 (不再 fail)** |
| Q 状态 | 当时被 K 遮蔽（编译器一个 cpp 撞 fail 即停） | **暴露 fail** |

(α') 完整修了 K！下一个 fail 转移到 Q。

## 为什么 K (lengths int,const<128>) 不 fail，Q (lengths int,const<128>) fail？

二者**同样** `tuple<int, constant<128>>` lengths type 进 `get_cached_global_strides`，按 inclusive_scan in-place limit 应该都 fail。但实测 K 不 fail Q fail。

**candidates** (待 verify)：

**(c1) K 跟 Q 走不同 TDM 入口路径**
- Q 在 pipeline:277 `load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window)`
- K 在 pipeline:359/etc `load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window)`
- 都调 `tile_window.tdm_load_to_lds` → `get_cached_global_strides` → inclusive_scan
- 调用 chain 看上去对称

**(c2) K dram_window 经 pipeline 内 distribution 应用后，bottom_tensor_view 的 lengths 在 (α') 后变 all-runtime**
- K 用 3-arg `make_tile_window(k_dram_block_window_tmp, {origin}, MakeKDramTileDistribution<>)` 带 distribution
- Q 用 `make_tile_window(q_dram_block_window_tmp, MakeQDramTileDistribution<>)` 也带 distribution
- 但 K 的 distribution transform 跟 (α')-修过的 unmerge 算式（kernel 层）耦合后可能让 lengths 推导出 all runtime
- Q 的 distribution 不耦合 (α') fix，lengths 保持 (int, constant<128>)

**(c3) inclusive_scan 在 K type 上能编通是因为 multiplies fold**
- multiplies(constant<X>, int) 路径上 X 不是 0 时，可能 fold 成 constant<X>（编译期）
- K 的某个 reduce 路径跟 Q 不同导致 fold 选择不同
- 不太可能但要排除

## 推荐 D5 诊断（5 min zero-risk）

加 1 处 CK_PRINT 在 fmha pipeline batch path **K dram_window 创建后** 看 (α') 之后 K type：
```cpp
auto k_dram_window = make_tile_window(k_dram_block_window_tmp,
                                      {physical_seqlen_k_start, 0},
                                      Policy::template MakeKDramTileDistribution<Problem>());
CK_PRINT<decltype(k_dram_window.get_bottom_tensor_view()
                      .get_tensor_descriptor().get_lengths())>();
```

**预期对照 D3 时的 pipeline:298**:
- D3 时 (α' 前)：K pipeline lengths = `tuple<int, constant<0>>`
- (α') 后：?

3 种可能输出：
- (D5-a) `tuple<int, int>` ← K 经 distribution + (α') 间接变 all runtime → 这就解释为什么 K 不 fail Q fail (Q 没这个 collapse)
- (D5-b) `tuple<int, constant<128>>` ← K 跟 Q 同 type，但走不同 inclusive_scan code path 让 K 不 fail（不太可能但需排除）
- (D5-c) 其它 type ← 重新挖

## 候选方向（取决于 D5 结果）

**如果 D5-a (K 是 all-runtime)**：
- 根因：Q dram_window 是 `(int, constant<128>)`，TDM 拒；K 经过 distribution 推导被 collapse 成 all-runtime 所以 OK
- Fix Q 路径：让 Q dram_window 也变 all-runtime（修 Q dist 或 wrap Q dram view）
- 不需要 (X) 修 CK core；不需要 (β/γ/δ)；只需要让 Q 跟 K 一样 collapse

**如果 D5-b (K 跟 Q 同 type 但 K OK)**：
- 根因：inclusive_scan 在某些 mixed tuple 情况下能编通
- 需要更深挖 multiplies / fold 行为
- Fix 不明朗

## 当前 worktree 状态

- 6 modified（unstaged）：
  - 3 plumbing
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix（kernel group path）
- CK_PRINT 全清
- ABC 没跑（编译 fail）

## 等你拍

按你 disambiguate 之前不选方向规则——D5 加 1 个 CK_PRINT 5 min 拿到关键 type，再决定方向。

实质工作：等你拍 D5 还是别的。

**注**：(α') 改动**保留**——不论 D5 结果，(α') 修的 unmerge typo 是独立 latent bug。

---
file: swe-status-163318.md
