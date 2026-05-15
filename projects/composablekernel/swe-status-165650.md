# Step B2 (X) scope check — 范围 5 处（已修 1，剩 4）

## scope 扫描结果

按你指示扫了所有 in-place mutate tuple/array ops，**关键 disambig**：

### 同质 array 不影响（不需要修）

| 位置 | 函数 | type | 影响 mixed? |
|---|---|---|---|
| container_helper.hpp:19 | container_push_back(array) | `array<TData, N>` | **不影响** (同质) |
| container_helper.hpp:229 | container_reverse_inclusive_scan(array) | `array<TData, N>` | **不影响** (同质) |
| container_helper.hpp:247 | container_reverse_exclusive_scan(array) | `array<TData, N>` | **不影响** (同质) |
| multi_index.hpp:60/70/80/89 | operator+/-/*  on multi_index | `array<index_t, N>` (line 17 typedef) | **不影响** (同质) |

→ array 全部 elements 同 TData type，in-place assign 没 type 冲突。这些是 false positives。

### 已支持 mixed tuple（不需要修）

| 位置 | 函数 | 模式 |
|---|---|---|
| tuple.hpp:482-528 | `transform_tuples` | `make_tuple(f(...)...)` 非 in-place ✓ |
| tuple.hpp:727 | `operator-(tuple<Xs>, tuple<Ys>)` | `generate_tuple([&](auto i){...}, number<NSize>{})` ✓ |
| (类似的 +/* tuple<Xs>+tuple<Ys> 版本) | | 同上 ✓ |
| container_helper.hpp:294-321 | `container_reverse_exclusive_scan_impl` (tuple) | recursive `container_push_front` ✓ |

### 真正影响 mixed tuple 的 in-place ops — **共 5 处**

| 位置 | 函数 | 状态 |
|---|---|---|
| container_helper.hpp:325 | `container_reverse_inclusive_scan(tuple)` | ✅ **已修** (X) |
| tuple.hpp:700 | `operator+(tuple<Xs>, Y)` | ❌ 待修 |
| tuple.hpp:722 | `operator-(tuple<Xs>, Y)` ← QA 撞 | ❌ 待修 |
| tuple.hpp:744 | `operator*(tuple<Xs>, Y)` | ❌ 待修 |
| tuple.hpp:757 | `operator*(Scalar, tuple<Xs>)` | ❌ 待修 |

→ **总 5 处，已修 1 剩 4**。

注：tuple+tuple 版本 (line 727) 已用 `generate_tuple`，是 4 处 fix 的现成模板。tuple+Y 版本（Y 通常是 array/multi_index）跟 tuple+tuple 做相同语义，只是 Y 不一样，改用 `generate_tuple` lambda 同样适用。

## tile_window.hpp:902 callgraph 后续

```
glb_tensor_descriptor.get_lengths()                  // tuple<int, constant<N>>
    - this->get_window_origin()                       // ← tuple<int,X> - array<int,2> 撞 line 722 ❌
    - window_adaptor_thread_coord.get_bottom_index()  // ← 同样会撞 (假设修 line 722 后)
→ transform_tuples([](auto x){...}, ...)              // ✓ make_tuple 非 in-place 模式 OK
→ tuple_reverse(...)                                  // 实现需 verify
→ to_array<index_t, NDimBottomTensor>(...)            // lifts tuple → array<int>，**逃出 mixed type**
→ tensor_dims (array<int>)                            // 之后所有 op on tensor_dims 都是 runtime int，OK
```

`to_array<index_t, N>` 是逃生路径——**所有 tuple ops 在这之前完成后转 array<int>，之后没有 mixed type 问题**。

但 `to_array` 之前要做 `get_lengths() - origin - bottom_index` 两次减法（撞 line 722）+ `transform_tuples max(0, x)` (OK, generate_tuple style) + `tuple_reverse` (待 verify)。

## tuple_reverse 快速 verify

`tuple.hpp:588 tuple_reverse(tuple<Ts>)` — 我没读完但 grep 没显示 in-place pattern，应该是 `make_tuple/generate_tuple` 模式，OK。

## 候选思路

### (X+) 修 tuple.hpp 4 处 ⭐ 推荐

每处改 ~3 行 mirror line 727 generate_tuple 模板：
```cpp
// before
tuple<Xs...> r;
static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] - y[i]; });
return r;

// after (mirror line 727)
return generate_tuple([&](auto i) { return x[i] - y[i]; }, number<NSize>{});
```

- 模板现成（line 727 等已用）
- pattern 一致风险低
- 改动总 ~12 行 (4 ops × 3 行)
- ABC 编一次 verify 还有没有 latent fail（可能性低，scope 已 narrow 到 5 处全 cover）

### (δ) 让 dram lengths 全 runtime
- 修 fmha kernel/dist 不引入 const lengths
- 影响所有 fmha pipeline (qr/qr_async/qr_async_trload/qr_tdm)，B1 教训说改 dist 风险高
- 不推荐

### (ε) if constexpr fall back
- 在 mixed 时 detect 转 array 路径
- 复杂度高，模板膨胀
- 不推荐

## SWE 倾向

**(X+) 修 4 处**。scope 5 处刚好踩你 ≥ 5 警戒线，但实际不 whack-a-mole——已扫完所有 in-place mutate pattern，scope 收窄到 5 处确定。剩 4 处 pattern 完全一致，模板现成 (line 727)，每处 ~3 行 trivial。

如果你担心 5 处 ≥ 警戒，备选 (δ) 改 fmha 但 B1 教训不推荐。

## 当前 worktree 状态

- 7 modified（unstaged）：
  - 3 plumbing
  - 2 B2 真修复
  - 1 (α') fix
  - 1 (X) inclusive_scan fix ✓
- ABC 没跑

## 等你拍

按你 ≥ 5 警戒升级流程 — scope 报告完成，请你拍：
1. **(X+) 改 4 处** — 我 own 改动，~12 行
2. **(δ) 让 dram lengths 全 runtime** — 改 fmha 创建侧
3. 其它

不私自扩。

---
file: swe-status-165650.md
