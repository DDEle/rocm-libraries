# 决策报告：是否批准 (X) 修 CK core `container_reverse_inclusive_scan`

时间：2026-04-30 16:41 | 决策者：用户 | 升级人：lead | 当前 worktree：`/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/`

---

## 1. 背景

gfx1250 FMHA 启用 TDM (Tensor DMA) load 路径，处于 **Step B2**：B1 已落地（用 `async_load_tile` 占位编通跑通 ABC 三个 case），B2 把占位换成真正的 `load_tile_tdm`。替换后编译炸在 `include/ck_tile/core/container/container_helper.hpp:353` 的 `y(i) = r;` in-place assign。经过 D1–D5 共 5 轮 CK_PRINT 编译诊断，根因 100% 锁定在 CK core `container_reverse_inclusive_scan` 不支持 mixed `tuple<int, constant<...>>`。

## 2. 诊断时间线

| 阶段 | status 文件 | 关键发现 |
|---|---|---|
| **D1 plan** | `swe-status-155317.md` | CK_PRINT 用 `[[deprecated]]` 在 dev `-Werror` 下 print type；选 D1 单边 confirm |
| **D1 confirm** | `swe-status-155722.md` | Q lengths = `tuple<int, constant<128>>`；K lengths = `tuple<int, constant<0>>`（异常）；提议走 (X) 修 CK core inclusive_scan |
| **lead 反转** | `swe-mentor-sync-155955.md` | K 的 `constant<0>` 像 placeholder，搁置 (X)，转挖 fmha K dram view 创建侧（α/β/γ/δ） |
| **D2 反证** | `swe-status-160732.md` | K naive + K padded 都是 `tuple<int, int>`，pad_tensor_view 没引入 constant<0>；嫌疑转 `make_tile_window` |
| **D3 锁定** | `swe-status-161300.md` | group path kernel:2570-2579 unmerge 算式 `kQKHeaddim/kDramTileK/kAlignmentK = 128/32/8 = 0`（多除一次），第一维被吃成 `constant<0>` —— 这是 fmha 自己的 latent typo |
| **alpha verify** | `swe-status-163011.md` | (α') 把 group path 算式改对（去掉多余 `/kAlignmentK`），K 变 `(int, constant<128>)`，但仍 fail；error 从 `constant<0>` 变 `constant<128>` —— root cause **不是 0 vs 128，是 compile-time vs runtime** |
| **stacktrace 反转** | `swe-status-163318.md` | (α') 后 fail trace 显示是 **Q 在 fail 不是 K**（window lengths = `(constant<64>, constant<128>)`）；提议 D5 print K type 看是不是 collapse 成 all-runtime |
| **D5 confirm** ⭐ | `swe-status-163855.md` | (α') 后 K 跟 Q lengths **都是** `tuple<int, constant<128>>`；编译器遇 hard substitution failure 一次就停，"哪个 fail" 是 instantiation order 副作用；**两者一直都 fail**；(X) 是唯一解 |

## 3. 根因（技术）

`container_reverse_inclusive_scan` 在 `include/ck_tile/core/container/container_helper.hpp:341-360`（含 line 353 的 `y(i) = r;`）用：

```cpp
tuple<Xs...> y;        // line 347 — 跟 input 同 type 创建（含 constant<128> slot）
TData r = init;        // line 349 — runtime int
y(i) = r;              // line 353 — 写 int 到 const slot → C++ 类型系统硬拒（hard substitution failure，不是 SFINAE soft fail）
```

Q + K dram window lengths 在 (α') 后都是 `tuple<int, constant<128>>`（hdim_q=128 被 ck_tile 的 distribution/transform 推成编译期常量，是设计 + 优化）。TDM 路径 `load_tile_tdm → tdm_load_to_lds → get_cached_global_strides → container_reverse_inclusive_scan` 必走；async/ds_load 不调 `get_cached_global_strides` 所以 B1 用 async 占位时不炸。

同文件 `container_reverse_exclusive_scan_impl`（line 310-327，对应的 `container_reverse_exclusive_scan` line 329-337）已用 recursive `container_push_front` 模式天然支持 mixed tuple；line 340 的 TODO 注释 "update to like container_reverse_exclusive_scan to deal with tuple of Numebr<>" 就是说 inclusive 没跟上。

## 4. 已 ruled out 的修法（来自 D5 status）

| 方向 | 为什么不行 |
|---|---|
| **(β)** 改 K distribution 不引入常量 | B1 抄 async_trload，乱改 dist 风险高（B1 教训）；改完 Q 仍是 const 也不解决 |
| **(δ)** fmha pipeline 局部 wrap K dram view 强 runtime length | 复杂，多 pipeline 都要改，影响 baseline；只能修 K 不修 Q |
| **(ε)** 关 pad / 改 tile align 让 hdim 不整除 | prod 用 hdim=128/64/256 都 align，不实际 |
| **(ζ)** 让 Q 跟 K 一样 collapse | D5-b confirm K 没 collapse，根本没这个 collapse 路径 |
| **(η)** 让 dram lengths 全 runtime（改 fmha kernel 创建侧 + dist） | 影响所有 fmha pipeline 包括 baseline，改坏风险高 |

## 5. 推荐方向 (X)

- **改动文件**：`include/ck_tile/core/container/container_helper.hpp` 的 `container_reverse_inclusive_scan`（line 341-360），mirror line 310-327 的 `container_reverse_exclusive_scan_impl` recursive `container_push_front` 模式
- **改动量**：~10 行（base case `i.value > 0` vs exclusive 的 `> 1`，起点 `f(init, x[N-1])` 而非 `init`）
- **影响 caller**：4 个，全在 TDM/flat 路径（gfx1250 独占新功能）
- **不影响**：async/ds_load/baseline pipeline（不调 `get_cached_global_strides → inclusive_scan`）
- **风险**：零 baseline 风险（前提是新实现对 all-int + mixed 两种 tuple 都正确，需 verify）
- **(α') 解耦**：(α') 已应用在 worktree（修 fmha kernel group path 的 `kQKHeaddim/kDramTileK/kAlignmentK=0` typo），是 develop branch latent bug，task #5 走独立 PR 流程，跟 (X) 决策不绑定

## 6. 选项

| 选项 | 描述 | 推荐 |
|---|---|---|
| **(a) 批准 (X)** | SWE 实施 ~10 行 fix + ABC 三 case verify | ⭐ |
| (b) 追加 D6 诊断 | 在 (X) 改完后再加 print 验证 K Q V dram 都进 inclusive_scan 拿对结果 | — |
| (c) 不走 (X) | 用户给替代思路（前提：上面 ruled-out 列表里所有选项都被否过） | — |

## 7. 当前 worktree 状态

- HEAD = B1 commit `16324795215`（clean）
- 6 modified（unstaged）：
  - 3 plumbing（lead 加，不动）
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix（kernel group path unmerge 算式）
- D5 CK_PRINT 待回退（`block_fmha_pipeline_qr_ks_vs_tdm.hpp:262, 301, 801`）
- ABC 没跑（编译 fail）

## 8. 并行 in-flight

- **qa-native** 在 develop 上跑 `ctest test_ck_tile_fmha_fwd` 验证 (α') 不回退（baseline build in flight，task `bwi99jrvl`）—— 跟 (X) 决策独立
- task #5 in_progress：(α') 验证 + 独立 PR 流程

---
file: decision-B2-X-on-alpha-164152.md
