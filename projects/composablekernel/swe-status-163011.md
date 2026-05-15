# Step B2 (α') 部分有效 — 但仍编不过，重新认知 root cause

## QA verify 结果（`compile-stepB2-alpha-verify-162808.log`）

**Case B (预期)**: 编译仍 fail，64 个 instance 仍 `container_helper.hpp:337: y(i) = r;` 路径。

**关键 error message 变化**：
- (α') 之前：`no known conversion from 'int' to 'const constant<0>'`
- (α') 之后：`no known conversion from 'int' to 'const constant<128>'`

→ **`constant<0>` 确实消失了，(α') fix 工作生效**（D4 type print 也 confirm 过 group path lengths 改成 `tuple<int, constant<128>>`）
→ **但 K dram lengths 仍是 mixed tuple（int + 编译期常量）**，TDM `get_cached_global_strides` 仍拒绝

## 重新认知 root cause（QA 一针见血）

> 问题不是 "0 vs 128"，是 **"compile-time vs runtime"** — `get_cached_global_strides` 要把 runtime int stride 写入 lengths tuple，**任何编译期常量维度都不接受**。

机理：
- `container_reverse_inclusive_scan` 用 `tuple<Xs...> y;` (line 331) in-place 创建跟 input 同 type 的 tuple
- 累加 `r = init * x[N-1] = 1 * 128`，r 是 runtime `int` (TData=int)
- `y(N-1) = r` → 把 int 写到 `constant<128>` slot，**即使值匹配**也不允许（C++ 类型系统硬限制）
- constant<X> 的 operator= 只接受 constant<X> 或 NTTP，不接受 int

→ 任何 compile-time dim 在 lengths tuple 都让 TDM `get_cached_global_strides` 编不过

## 这意味着：(α') 是 **必要但不充分** 的 fix

- (α') 把 group path 的 unmerge typo 修了 — 让 K lengths product 数学一致 (128 vs 0)，**这是真 bug，必须修**
- 但 TDM path 还需要 lengths 全 runtime int，或者 CK core inclusive_scan 支持 mixed tuple

## Lead 之前搁置的 (X) 实际上是必要的

回头看 lead 决策："(X) CK core fix 暂搁——根因不一定在那里"。

**现在数据显示 root cause 实际上包括 (X)**:
- (α') 修 fmha algebra typo（必要） + (X) 修 CK core inclusive_scan limit（充分）
- 缺任一编不过
- async_load 不读 lengths 所以 (X) 限制对它没影响；TDM 强制 read → 撞 (X) 限制

**之前说 "X 修了也没用，因为 lengths value 仍 wrong"** — 那是基于 D2 反证 + 我误以为 root cause 完全在 fmha 一侧。但 (α') verify 证明 lengths value 修对了（128 product 数学正确），TDM 仍拒绝是因为 type 限制。

## 候选方向（更新版）

**(α' + X)** ⭐ 推荐
- (α') keep — 已应用，修 group path typo
- (X) 修 CK core inclusive_scan 用 exclusive_scan 风格 (container_helper.hpp:325-344)
- 同时应用，B2 编通 + lengths value 数学正确

**(α' + δ')** 备选
- (α') keep
- δ': fmha pipeline 局部 wrap K dram window 强 runtime length（去掉 constant<128>）
- 复杂，影响多 pipeline，B1 教训不推荐

**(α' + ε')**
- (α') keep
- 关掉 K dist 的 padding/transform 让 lengths 不再编入 hdim 静态常量
- 改 distribution 风险高（B1 教训）

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- 6 modified（unstaged）：
  - 3 plumbing
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix（fmha_fwd_kernel.hpp group path unmerge 算式）
- CK_PRINT 全部回退完
- ABC 没跑（编译 fail）

## 等你拍

按 lead 之前规则"大决策升级给用户" — 这次决策升级到 (X) 加上 (α')。

我**不动 CK core**，等你 sub-agent + Teams notify 流程。

附信息：
- log: `compile-stepB2-alpha-verify-162808.log`
- (X) fix 模板：`container_helper.hpp:313-321` exclusive_scan 已用 recursive container_push_front 模式可借鉴
- 改动 ~10 行 mirror exclusive
- 影响 4 caller 全 TDM/flat (gfx1250 新功能)，零 baseline 风险

---
file: swe-status-163011.md
