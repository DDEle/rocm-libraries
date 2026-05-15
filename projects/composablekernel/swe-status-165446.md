# Step B2 (X) sanity — 编译仍 fail，但 root cause **同**，需要扩 (X) scope

## (X) 实施后 ABC verify 结果（QA `compile-stepB2-XplusAlpha-165116.log`）

(X) 修对了 `container_reverse_inclusive_scan` —— 之前 inclusive_scan 的 fail 消失。

**但 fail 转移到 `tuple.hpp:722` 的 `operator-`**：

```
tuple.hpp:722:50: error: no viable overloaded '='
 722 |     static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] - y[i]; });
                                                   ~~~~ ^ ~~~~~~~~~~~
```

调用链：
1. `tile_window.hpp:902` `glb_tensor_descriptor.get_lengths() - this->get_window_origin() - ...`
2. `operator-<int, ck_tile::constant<128>, array<int, 2>, false>` (tuple<int,const<128>> - array<int,2>)
3. `tuple.hpp:722` `r[i] = x[i] - y[i]` — r 是 `tuple<int, constant<128>>`，slot 1 是 const，int 不能 assign

## tuple.hpp 共 4 处同 pattern in-place tuple ops

grep 实测：

| line | op | 含义 |
|---|---|---|
| 699 | `operator+(tuple<Xs>, Y)` | tuple + array/multi_index |
| **721** | `operator-(tuple<Xs>, Y)` | tuple - array/multi_index ← QA 撞这个 |
| 743 | `operator*(tuple<Xs>, Y)` | tuple * array/multi_index |
| 756 | `operator*(Scalar, tuple<Xs>)` | scalar * tuple |

每个都用 `tuple<Xs...> r; static_for{... r[i] = ...;}` in-place 模式，跟 inclusive_scan 同样问题。

**对应的 tuple+tuple 版本（line 727 等）已经用 `generate_tuple` 非 in-place 模式**，已经支持 mixed tuple — 是现成的 fix 模板。

## root cause **同** (X)：CK core in-place tuple ops 不支持 mixed tuple

不是新 root cause，是 (X) scope 没覆盖完。所有 4 处都该修。

## 候选方向

**(X+) 扩 (X) scope** ⭐
- 修 tuple.hpp 4 处 in-place ops 改 mirror generate_tuple 模式
- 每处 ~3 行
- 模板就是同文件 line 727 `operator-(tuple, tuple)` 的 generate_tuple lambda
- 影响：所有用 in-place ops 的 caller，但 generate_tuple 跟 in-place 行为等价（都返新 tuple），零语义变化
- 风险：扩了 4 处比 (X) 之前 1 处影响面大，但 pattern 一致

**(X-narrow) 只修 line 721 operator-**
- QA 报错的那一处
- 编一次看是否还撞 operator+/* 之类
- 增量修复，但可能多轮编译

**SWE 倾向 (X+)** — root cause 同，4 处一起 fix 才完整。lead 之前批 (X) 的精神是"修 CK core 让 mixed tuple 在 TDM path 能用"，scope 扩到 4 处更彻底。每处 ~3 行不大。

## 当前 worktree 状态

- 7 modified（unstaged）：
  - 3 plumbing
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix
  - 1 (X) fix in container_helper.hpp inclusive_scan ✓ 已 verify 修对
- D5 CK_PRINT 全清
- ABC 没跑（编译 fail 在 operator-）

## 等你拍

按 "(X) on top of (α')" 大决策升级流程 — 这次不是 root cause 反转，是 (X) 实施细节。请你拍：
1. **(X+) 同时修 4 处** — 一次性彻底 fix CK core mixed tuple pattern
2. **(X-narrow) 只修 line 721** — 增量，编一次看再决定

无论哪个，我**不私自扩**。等你拍。

附信息：
- log: `compile-stepB2-XplusAlpha-165116.log`
- generate_tuple 模板：`tuple.hpp:727` `operator-(tuple, tuple)` 已用 `return generate_tuple([&](auto i) { return x[i] - y[i]; }, number<NSize>{});`

---
file: swe-status-165446.md
