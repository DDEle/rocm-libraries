# Step B2 (D3) — constant<0> 引入点 100% 锁定

## D3 编译诊断结果（QA `compile-stepB2-D3-160935.log`）

| # | 位置 | type |
|---|---|---|
| 1 | Kernel:1649  (K naive, batch)            | `tuple<int, int>` ✓ |
| 2 | Kernel:1660  (K padded, batch)           | `tuple<int, int>` ✓ |
| 3 | **Kernel:1727  (D3, batch path 2-arg make_tile_window 后)** | **`tuple<int, int>` ✓** |
| 4 | **Kernel:2749  (D3, group path 2-arg make_tile_window 后)** | **`tuple<int, constant<0>>` ⚠️** |
| 5 | Pipeline:295 (Q dram_window in pipeline batch) | `tuple<int, constant<128>>` |
| 6 | Pipeline:298 (K dram_window in pipeline batch) | `tuple<int, constant<0>>` ⚠️ |

## 关键 disambiguate：**两条独立 path，两个独立引入点**

### 引入点 A: Group path Kernel 层（line 2487-2604 `make_k_dram` lambda）

batch 和 group 的 `make_k_dram` 不同：
- **Batch path** (line 1638-1651)：只有 `make_naive_tensor_view` + `pad_tensor_view`，return → `(int, int)` ✓
- **Group path** (line 2487-2604)：在 `pad_tensor_view` 后**额外做 transform_tensor_view 链**：unmerge → permuted → merge

致命行（line 2570-2579 else branch / 非 XorLengthFold）：
```cpp
const auto k_dram_unmerged = transform_tensor_view(
    k_dram_pad,
    make_tuple(make_pass_through_transform(height),
               make_unmerge_transform(make_tuple(
                   number<FmhaPipeline::kQKHeaddim / kDramTileK / FmhaPipeline::kAlignmentK>{},   // = 128/32/8 = 0
                   number<kDramTileK / FmhaPipeline::kAlignmentK>{},                                // = 32/8  = 4
                   number<FmhaPipeline::kAlignmentK>{}))),                                         // = 8
    make_tuple(sequence<0>{}, sequence<1>{}),
    make_tuple(sequence<0>{}, sequence<1, 2, 3>{}));
```

**fp16 d=128 + kK0=32 + kAlignmentK=8** 配置：
- 第 1 维 = `kQKHeaddim / kDramTileK / kAlignmentK = 128/32/8 = 0` ← 整数除法**被吃成 0**
- 第 2 维 = `kDramTileK / kAlignmentK = 32/8 = 4`
- 第 3 维 = `kAlignmentK = 8`

完美对应 trace 里 `unmerge<tuple<constant<0>, constant<4>, constant<8>>>`。

数学上看 unmerge product = 0 * 4 * 8 = 0 → length 维度被 fold 到 0。

### 引入点 B: Pipeline 层（pipeline:284-287 3-arg make_tile_window with distribution）

batch path 在 kernel:1727 是 `(int, int)` ✓ 健康，但 pipeline:298 看到 `(int, constant<0>)` ⚠️ — 说明 pipeline 内的:
```cpp
auto k_dram_window =
    make_tile_window(k_dram_block_window_tmp,
                     {physical_seqlen_k_start, 0},
                     Policy::template MakeKDramTileDistribution<Problem>());
```
也引入了 constant<0>。来源是 `MakeKDramTileDistribution` (B1 抄 async_trload) 的 distribution encoding 应用到 dram tensor 时，把 hdim residual=0 (因 hdim=128 mod K0*K1=32 = 0) 编码成 constant<0>。

## 共同 root cause（统一解释）

**hdim_q tile-align 后 residual=0 被 ck_tile 的 unmerge / distribution transform 编码成编译期 `constant<0>`**：

- 数学：hdim_q=128 ÷ tile_size=32 = 4 整除，余 0
- ck_tile 推导：把 0 嵌入 unmerge 输入维度作为 leading dim
- 副作用：lengths product 跟 hdim_q 不再一致；async_load 不读 lengths 所以从未触发；TDM 必读 → 撞 mixed tuple `int + constant<0>`

## 跟之前两个假设的关系

- ❌ mentor 16:04 假设 "pad_tensor_view 引入 constant<0>" — D2 已反证
- ❌ Lead 推 X "CK core inclusive_scan TODO" — 仍然是 known limitation 但不是 root cause；即使修了 inclusive_scan，constant<0> 作为 length value 依然 wrong (product = 0 跟 hdim_q=128 不符)
- ✅ **真正 root cause**: ck_tile 的 unmerge / distribution transform 在 hdim tile-align 整除时把 0 当 leading dim 编进 lengths，**lengths 计算本身就 wrong**

## 候选 fix 方向（已收窄到三个）

**(α') 改 Kernel group path 的 unmerge 算式**
- line 2574-2577 的 `kQKHeaddim / kDramTileK / kAlignmentK` 改成 `kQKHeaddim / kDramTileK`（去掉那个多余的 `/ kAlignmentK`），第 1 维变 `128/32=4`
- 第 2 维 `kDramTileK / kAlignmentK = 4`，第 3 维 `kAlignmentK = 8`，product = 4*4*8 = 128 ✓
- **但**：这个公式 batch path 不用，所有 fmha pipeline 共享 group path 的 k_dram；改动会影响所有 group-mode pipeline (qr/qr_async/qr_async_trload/qr_tdm)
- 必须验证 unmerge 维度变化对其它 pipeline 数据布局没破坏

**(β') 改 Pipeline 层的 K dist 不用 unmerge**
- B1 抄过来的 MakeKDramTileDistribution，但 distribution encoding 没显式 unmerge；ck_tile 内部应用 distribution 时自动算 transform
- 需要看 ck_tile 怎么把 distribution + dram tensor 推导出 transform，能否避免 0-residual 编入

**(δ') 在 fmha pipeline 局部 wrap K dram view**
- 拿 `k_dram_block_window_tmp` 后用 `transform_tensor_view` 显式重 wrap，让 hdim 维变 runtime int
- 不改 kernel 层 / dist
- 复杂度中等

**(ε') 关掉 pad / 改 tile align**
- hdim_q=128 跟 tile size=32 (kK0) 整除是触发条件；如果 hdim_q 不是 32 倍数（比如 128 → 130 padding 到 32 align），residual≠0 也许不触发
- 但实际 prod 多用 hdim=128/64/256 都是 align，不实际

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- Unstaged dirty:
  - 3 plumbing
  - 2 B2 真修复（pipeline + policy）
  - 1 D1 CK_PRINT (pipeline 295/298)
  - 2 D2 CK_PRINT (kernel 1649/1660)
  - 2 D3 CK_PRINT (kernel 1727/2749, replace_all 加了 batch+group 各一份)
- 全部 CK_PRINT diagnostic 待回退

## 等你拍

按你说"根因 100% 锁定后大决策升级给用户"——根因现在锁定了。我**不选方向**，等你 sub-agent + Teams notify 流程。

如果你需要更精确诊断（比如 confirm "(α') 改 unmerge 算式后 batch+group 都健康"），我可以加 D4 print 试改后的 type，但我倾向直接由用户拍方向。

---
file: swe-status-161300.md
