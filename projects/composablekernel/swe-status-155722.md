# Step B2 (D1) confirm + 提议走 (X)

## D1 编译诊断结果（CK_PRINT 输出）

QA 编一次（`compile-stepB2-D1-155511.log`），CK_PRINT 命中：

| Window | dram tensor lengths type | 备注 |
|---|---|---|
| **Q** | `tuple<int, constant<128>>` | hdim_q=128 提升为编译期常量（符合 mentor 的 pad_tensor_view 优化推断） |
| **K** | `tuple<int, constant<0>>` | **第 2 维变 constant<0>**！异常 — 比 Q 更"病态" |

## K 第 2 维 constant<0> 分析

K 的 dram tensor lengths 不是 `constant<128>` 而是 `constant<0>`——跟 error trace 里 `unmerge<tuple<constant<0>, constant<4>, constant<8>>>` 第一个 constant<0> 同源。

可能含义：
- K 的 transform chain 把 hdim 维"擦除"成 placeholder constant<0>（被另一个 transform 接管 length 计算）
- 或是 fmha policy 的 K dram distribution 引入的伪长度
- 不是 simple "pad_tensor_view 升 128"，K 走了一个比 Q 更复杂的路径

但**核心问题不变**：mixed runtime + compile-time tuple（不论是 constant<128> 还是 constant<0>）触发 `container_reverse_inclusive_scan` 的 `y(i) = r;` in-place assign 失败（int 不能写到 const slot）。

## 根因 100% confirm

- error trace 推断 ✓
- CK_PRINT 实测 ✓
- CK core TODO 注释 ✓
- exclusive_scan 已支持 mixed tuple ✓

→ **(X) 修 CK core inclusive_scan 是正确解**

## (X) 改法草稿（confirm 可行）

`container_helper.hpp` line 294-311 的 `container_reverse_exclusive_scan_impl` 是现成模板：

```cpp
template <typename... Xs, typename Reduce, index_t I, typename YOld, typename ROld>
CK_TILE_HOST_DEVICE constexpr auto container_reverse_exclusive_scan_impl(
    const tuple<Xs...>& x, Reduce reduce, number<I> i, YOld y_old, ROld r_old)
{
    auto r_new = reduce(x[i], r_old);
    auto y_new = container_push_front(y_old, r_new);   // 新 tuple，type heterogeneous OK
    if constexpr(i.value > 1)
        return container_reverse_exclusive_scan_impl(x, reduce, i - number<1>{}, y_new, r_new);
    else
        return y_new;
}
```

inclusive 版 mirror（草稿）：

```cpp
template <typename... Xs, typename Reduce, index_t I, typename YOld, typename ROld>
CK_TILE_HOST_DEVICE constexpr auto container_reverse_inclusive_scan_impl(
    const tuple<Xs...>& x, Reduce reduce, number<I> i, YOld y_old, ROld r_old)
{
    auto r_new = reduce(r_old, x[i]);                   // f(r, x[i]) 跟原 inclusive 一致
    auto y_new = container_push_front(y_old, r_new);
    if constexpr(i.value > 0)                           // inclusive 是 > 0，比 exclusive > 1 多 1 步
        return container_reverse_inclusive_scan_impl(x, reduce, i - number<1>{}, y_new, r_new);
    else
        return y_new;
}

template <typename... Xs, typename Reduce, typename TData>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_inclusive_scan(const tuple<Xs...>& x, Reduce f, TData init)
{
    constexpr index_t NSize = sizeof...(Xs);
    // y[N-1] = f(init, x[N-1]) — inclusive 第一项就含 x[N-1]
    auto r_init = f(init, x[number<NSize - 1>{}]);
    auto y_init = make_tuple(r_init);
    if constexpr (NSize > 1)
        return container_reverse_inclusive_scan_impl(x, f, number<NSize - 2>{}, y_init, r_init);
    else
        return y_init;
}
```

**关键差异跟 exclusive**：
- inclusive[i] = f(init, x[N-1]) op f(x[N-2]) ... op f(x[i]) — 含 x[i]
- exclusive[i] = init op x[N-1] op x[N-2] ... op x[i+1] — 不含 x[i]
- 实现差异：inclusive 的 base case 是 `i.value > 0`（要扫到 i=0），exclusive 是 `i.value > 1`（停在 i=1）
- 起点也不同：inclusive 起点已含 x[N-1]，exclusive 起点是 init

**风险**：
- inclusive_scan 4 caller 全 TDM/flat（gfx1250 新功能），不动 ds_load/async_load 路径，零 baseline 风险
- 需要测试 mixed tuple + all-int tuple 都通过（all-int 路径其实也能用新版，编译器推导出 same type）

## 下一步建议

请你拍 X 走法（这是大决策，你说会异步通知用户）：

1. **(X) 直接走** — SWE 改 CK core inclusive_scan 按上面草稿，commit 时跟 B2 真修复一起
2. **(X) 先走 CK core，B2 后续** — 先单独 CK core change 通过 review，再回头 B2
3. **(Y) 反方向** — 不改 CK core 改 fmha 局部 workaround（mentor 不推荐）

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- Unstaged dirty:
  - 3 plumbing（你加，不动）
  - 2 B2 真修复（pipeline + policy，本身正确）
  - 1 处 CK_PRINT diagnostic（D1 探针，待回退）

**等你拍 (X) 路线**。回退 CK_PRINT diagnostic 我自己做（不commit）。

---
file: swe-status-155722.md
