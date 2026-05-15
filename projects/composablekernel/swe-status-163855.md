# Step B2 D5 disambig — case D5-b confirm: Q = K 同 type 都 fail

## D5 编译诊断结果（QA `compile-stepB2-D5-163639.log`）

5 处 CK_PRINT 命中 (pipeline batch + group path) dedup 后**只 1 个 unique type**：

→ **全部都是 `tuple<int, ck_tile::constant<128>>`**

对照 D1：
| | Q lengths | K lengths |
|---|---|---|
| D1 (α' 前) | `tuple<int, constant<128>>` | `tuple<int, constant<0>>` |
| **D5 (α' 后)** | `tuple<int, constant<128>>` | `tuple<int, constant<128>>` ← (α') 把 K 修成同 Q type |

**case D5-b** ✓（你列的三种里"不太可能但需排除"那个）

## 重新解释为什么 D1 只 K fail D5 只 Q fail（实际两者都 fail）

之前我和你的两次互斥假设：
- D1 time: "K fail Q OK" — 你最初判断
- alpha-verify time: "K OK Q fail" — 我后来更正

**都不对**。真相：

每个 fmha instance 是一个独立 .cpp 文件。`load_tile_tdm` 内部走 `tdm_load_to_lds → get_cached_global_strides → container_reverse_inclusive_scan`，对 Q 跟 K 同样的 path。

`y(i) = r;` 的 in-place assign 是 **hard substitution failure**（不是 SFINAE soft fail），编译器**遇到第一个就停**整个 .cpp 编译。

- D1 time: K type 是 `(int, const<0>)`，Q 是 `(int, const<128>)`。编译器 instantiate K dram_window 的 inclusive_scan 时先撞 fail（顺序原因 / typical instantiation order），整个 .cpp fail，Q 的同样 instantiation **没机会触发**
- D5 time: (α') 把 K 修成 `(int, const<128>)` → K Q 同 type，编译器 instantiate Q dram_window 的 inclusive_scan 时先撞 fail（实例化顺序变化），Q fail trace 暴露

**Q K 一直都 fail，只是编译器 substitution failure 一个就停掉，看到的"哪个 fail"是 instantiation order 副作用，不是真的"K fail Q OK" 或反过来**。

QA 第一直觉**完全正确**：TDM `get_cached_global_strides` 不接受任何编译期常量维度。

## root cause 100% 锁定（这次真的）

**TDM path 调 `container_reverse_inclusive_scan(lengths, multiplies<>{}, 1)` 不能处理含编译期常量的 mixed tuple lengths**。

机理（前面已分析）：
- inclusive_scan 用 `tuple<Xs...> y;` (line 331) in-place 创建跟 input 同 type 的 tuple
- `r = init * x[i]`，r 是 runtime int
- `y(i) = r` → 写 int 到 const slot，C++ 类型系统拒
- exclusive_scan (line 313-321) 用 recursive `container_push_front` 已支持 mixed tuple，inclusive 没跟上

不论 hdim 是 constant<128> 还是 constant<0>（或任何其它常量），inclusive_scan 都拒。**(α') 把 constant<0> 改成 constant<128> 让 lengths 数学正确，但 TDM 限制还在**。

## 跟你之前 disambig 框架对应

你给的三种解释：
- (i) Q 不走 inclusive_scan → **错** (D5-b confirm Q 也走)
- (ii) Q 走但 specialization 救它 → **错** (D5-b confirm 同 type 同 fail)
- (iii) Q 内部又被 collapse → **错** (D5 print 显示 Q lengths 仍 `(int, constant<128>)`)

→ **none of the three** — root cause 确实在 CK core inclusive_scan limit，(X) 是 right answer

## (α') keep 的理由不变

(α') 改的 unmerge 算式 `kQKHeaddim/kDramTileK/kAlignmentK = 0` → `kQKHeaddim/kDramTileK = 4` 是独立 latent bug：
- 即使修了 (X) 让 K type `(int, constant<0>)` 编通，K lengths product = 0 ≠ hdim_q=128 仍是数学错误，runtime 行为可能是 silent garbage（async_load 不读 lengths 没暴露，TDM 读了暴露 type fail；如果 (X) 让 type 通了，runtime stride 算出来仍 wrong）
- (α') 在 task #4/#5 已经被 lead 升级独立 PR 流程，对的处理

## 候选方向（更新版）

**(X) + (α')** ⭐ 唯一答案
- (α') 已在 worktree（task #4/#5 PR 流程）— 不变
- (X) 修 CK core inclusive_scan 用 exclusive_scan 风格 (~10 行)
- 影响 4 caller 全 TDM/flat (gfx1250 新功能)，不动 ds_load/async_load 路径，零 baseline 风险
- 修了之后 K Q V 任何带 compile-time const 的 dram lengths 都能进 inclusive_scan，TDM 编通

**其它方向 (β/δ/ε/ζ/η) 都 dead**：
- 让 dram lengths 全 runtime → 要改 fmha kernel 创建侧 + dist 内部，影响所有 fmha pipeline，改坏 baseline 风险高
- 让 Q 走 K 的 collapse → D5-b confirm K 没 collapse，Q 也没需要 mirror

## 当前 worktree 状态

- 6 modified（unstaged）：
  - 3 plumbing
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix（kernel group path unmerge 算式）
- D5 CK_PRINT 待回退（pipeline:262, 301, 801）

## 等你拍

按你 disambig 完成后大决策升级流程 — D5 disambig 完成，**(X) 修 CK core inclusive_scan 是 root cause 唯一解**。

我**不动 CK core**，等你 sub-agent + Teams notify 用户拍 (X) on top of (α')。

附信息：
- (X) fix 模板在 `container_helper.hpp:294-321` exclusive_scan_impl + exclusive_scan，照抄 recursive container_push_front 模式
- 改动 `container_helper.hpp:325-344` inclusive_scan ~10 行
- 完整 stacktrace + log 见 `compile-stepB2-D5-163639.log`

---
file: swe-status-163855.md
