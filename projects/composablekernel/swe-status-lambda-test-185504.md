# Step B2 (λ-test) — push back: 3-dist coupling 让 single-dist 改造不能 disambig

Lead 派 task #7 (λ-test)。SWE 拉 mentor 讨论实施细节，**mentor 实测发现新事实**让 λ-test 设计不可行 → push back 求重决策。

## Mentor 新发现：fmha qr_tdm 有 3 个 dist 互相 coupled

`block_fmha_pipeline_qr_ks_vs_tdm.hpp:285-303`:

```cpp
// 1. dram window dist (TDM box_dim 来源)
auto k_dram_window = make_tile_window(..., MakeKDramTileDistribution<Problem>());

// 2. LDS write window — 没传 dist，plain offset
auto k_lds_write_window = make_tile_window(k_lds_write_view, ...lengths..., {0, 0});

// 3. LDS read window dist (load_tile 读 LDS)
auto k_lds_read_window = make_tile_window(..., MakeKRegTileDistribution<Problem>());
```

`load_tile_tdm(tdm_config, k_lds_write_window, k_dram_window)`:
- TDM box_dim 来自 **dram window 的 dist** (`MakeKDramTileDistribution`)
- 写到 lds_write_window 的位置 (plain offset, 没 dist 编排)

`load_tile(k_lds_read_window)`:
- 用 `MakeKRegTileDistribution` 算 thread-i 读 LDS 哪里
- **跟 BlockGemm 0 强绑定**（async_trload_policy:581 算的，喂 QK BlockGemm 作 A operand）

## λ-test 单改 dram dist 不能 disambig

如果只改 `MakeKDramTileDistribution` 让 TDM box_dim 变 trivial linear (16, 1)：
- TDM 按新 trivial linear box-major 写 LDS
- ds_load 仍按 `MakeKRegTileDistribution` 老 dist 读
- → 仍 mismatch，valid:n 但**不能 disambig** H3（因为单改 dram 没 align write/read）

**同改 dram + LDS read dist** 用同一 trivial linear：
- 改 `MakeKRegTileDistribution` 会破坏 QK GEMM A operand 期望
- 双重 garbage 让结果**更不可解读**

→ **任何"单纯改 dist 一处"的 λ-test 都不能产生有效 disambig 信号**

## 替代 disambig 方案 (mentor 评估)

**(λ-test 简化)**: 把 LDS read 也用 plain offset + 手写 thread-i 16 contiguous K assemble 喂 BlockGemm 0
- 复杂度接近 fix 本身，不划算

**(LDS dump verify)**: 加 thread-0 LDS → debug global mem dump，host 比对 LDS content vs dram K
- Deterministic disambig
- HIP printf 难，但 device-side memcpy 到 host visible global buffer 可行 (kargs 加 debug ptr)
- 工作量中等 (~1h 实施 + 1h verify)

**(single-thread case)**: 跑 blockSize=1 让 dist trivialize
- 但 fmha codegen 不一定支持 blockSize=1 instance
- 风险：本身先 fail 别的 path

## Mentor 推荐次序

1. **首选 push back**: 跳过 λ-test 直接拍 fix 方向
   - H3 升级版 90%+ 信心已经到，mentor 已 6+ trial 历史
   - λ-test 单改不能 disambig；LDS dump 复杂；single-thread 不可靠
   - 收益有限，时间成本接近 fix 本身

2. **次选 LDS dump verify**: 如果用户必须 confirm 才能拍方向
   - Deterministic
   - ~2h 总耗时

## 3 个 fix 方向更新版（mentor 重 articulate）

**(λ)** 重设全 3 dist 让 TDM write box-major + LDS read 同 box-major + BlockGemm 0 兼容
- 工作量比之前评估更大（不只 dram dist，还要重设计 BlockGemm 0 的 A operand layout）
- 风险高
- mentor: ideal 但 work 量大

**(μ)** 加 LDS shuffle 中间层 — TDM 写 plain box-major 后 async_copy_lds shuffle 成 BlockGemm 期望
- dist 全不动
- 额外 sync + 额外 LDS BW
- mentor: 倾向**风险最小**但有 perf 退步

**(ν)** 战略撤退 B2 retain B1 21-32% 加速
- ~1h 撤回 worktree 中 🟢 B2 真修复 2 处
- 🟡 (α'+X+) 独立 PR 流程不变
- AICK-579 主目标已达

## SWE 倾向

跟 mentor 一致：**首选 push back 跳过 λ-test，直接拍 (μ) or (ν)**。

## 等你拍

按你 auto-decided λ-test，但 mentor 实施前发现技术不可行。请你重判：

1. **跳过 λ-test 直接拍 fix 方向 (μ/ν)** ⭐ 我 + mentor 推荐
2. **改做 LDS dump verify** disambig — ~2h 工作量但 deterministic
3. **强制做 dist 改造 λ-test** — mentor 评估不能产生有效信号

无论选哪个，worktree 状态保留 8 modified clean。

---
file: swe-status-lambda-test-185504.md
