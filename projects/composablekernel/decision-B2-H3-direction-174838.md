# B2 H3 升级版根因 confirmed — fix 方向决策报告 (λ vs μ vs ν)

**目的**：用户在不翻聊天历史前提下 refresh 全部上下文，拍 B2 fix 方向。

---

## 1. 背景

gfx1250 FMHA TDM 改造分两步：
- **Step B1**（已完成）：把 `load_tile` 换成 `async_load` 风格，结合 LDS swizzle，已得 21–32% 加速，AICK-579 主目标已达。
- **Step B2**（in progress）：把 B1 的 async_load 进一步换成真正的 TDM intrinsic（`load_tile_tdm`），目标是用硬件 DMA 进一步降 stall。

B2 编译路径上踩了 5 处 CK core in-place mutate tuple ops bug：经历 (α') unmerge typo fix + (X) inclusive_scan fix + (X+) tuple operator+/-/*/* (4 处) fix 后**编译终于 pass**。但 ABC 的 case A runtime fail：max err = 23、100% 元素错、`valid:n`，且数值在 cached/(η)/X+/H4' 多个 trial 之间 **16-digit 完全 identical**（高度 deterministic）。

随后挖了 6 轮 hypothesis 才定位真根因。

## 2. 排除路径（每条 1 句）

- **H4'**（LDS xor swizzle 严格化）：disable xor 后 walltime 翻倍但 valid:n 数值 16-digit 不变 → swizzle 不背锅。
- **H8**（内层 K-iter bug）：单 K-iter (s=64 只 1 K-block) 仍 valid:n → 不是迭代累计问题。
- **η**（用真 stride bypass cached）：valid:n 数值 0 变化 → cached vs real stride 等价，stride 字段不背锅。
- **Hγ**（build cache disambig）：confirm η 真编进 binary 真跑，排除 stale binary。
- **Hβ**（hardcode `{99999,99999}` stride）：SIGSEGV → TDM 真读 stride[0]，η 数值跟 cached 一致是巧合（行 stride=128 数学等价）。
- **Hδ-1**（box_dim print）：K box=(8,2)、V box=(8,8) 几何 OK，单 box 元素数对得上每 thread 搬运量。

## 3. H3 升级版真根因（mentor 90%+ confidence）

来源：`~/.claude/teams/gfx1250-fmha-tdm/inboxes/swe.json` 17:46 mentor 消息（summary `(c) 没现成 API`）。

**核心机理**：
- **TDM 写 LDS 是 box-major 物理布局**：thread-i 从 `lds_coord` 起点写一个 8×2（K）或 8×8（V）的 contiguous box。
- **ds_load 读 LDS 按 fmha B1 dist 期望的 thread-mapped 位置**——B1 dist (`MakeKDramTileDistribution`) 是 async_load 历史脚手架。
- 两端 thread→LDS-position mapping 规则不同 → **一致地读错位置** → **一致 garbage**（这就是为什么 max err 完全 deterministic、数值 16-digit identical 在多轮 trial 之间不变）。

**对照 GEMM v1**（`gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp:331-337`）：
```cpp
constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
    BlockGemm::MakeABlockDistributionEncode())){};
```
GEMM v1 的 `BlockGemm` dist 是**跟 TDM write pattern 一起设计**的，所以 LDS write 跟 read 自然对齐。fmha 没这种共设计——B1 沿用 async_load dist，跟 TDM box-major write 不兼容。

mentor 已确认**没有现成 API 可抄**——必须 design 新方案。

## 4. 三个候选 fix 方向

| | 描述 | 工作量 | upside | downside |
|---|---|---|---|---|
| **(λ)** | 重设 fmha K/V dram dist，让 LDS write/read 对齐 TDM box 模式（mirror GEMM v1 BlockGemm 共设计思路） | 大（1–2 SWE-day，多轮 design iter 可能） | 真正 TDM-native 实现，符合 AICK-579 B2 原意 | B1 教训"动 dist 风险高"；可能引入新 functional bug |
| **(μ)** | 保留 dist，加 LDS 中间 shuffle 层（TDM 写 box-major → async_copy LDS 内部重排成 dist 期望布局 → ds_load） | 中（~半 SWE-day） | dist 不动，B1 教训不踩 | 额外 sync + LDS BW，TDM perf 收益可能被打折甚至变 0 |
| **(ν)** | 放弃 B2，回 B1 async_load | 撤回（~1h） | B1 已 21–32% 加速，AICK-579 主目标已达 | B2（TDM intrinsic 替换）目标未达 |

mentor 推荐：**用户拍**，不替用户做 strategic 选择。

## 5. mentor 信心与 disambig 选项

- H3 升级版 **90%+** confidence；剩 10% 可能还有未发现字段/sync 问题。
- mentor 在 6 轮 trial 中错引方向 2 次（H4' + 早期 H3 时机）→ 自评建议升级而非独自再 trace。
- 如果用户挑战根因，可做 **(λ-test)** zero-cost disambig：临时把 K dist 强制改成"trivial linear（thread-i 读 LDS[i*16..(i+1)*16] contiguous）"测一次。
  - valid:y → confirm dist 是问题，再正式选 (λ)/(μ)
  - valid:n → 推翻 H3 升级版，重启挖
- 但 (λ-test) 实施工作量 ≈ (λ) 真实施一部分，**最好用户先决策**再投入。

## 6. SWE / 我（lead）推荐排序

1. **(a) 先做 (λ-test) zero-cost disambig**：SWE 写 trivial linear K dist 半小时左右测一次，valid:y/n 给最后 confirm；如果 confirm 再选 (λ)/(μ)；如果反证则重启挖根因。**最稳路径**。
2. **(b) 直接 (λ) 投入完整 redesign**：风险高但路径正、TDM-native。
3. **(c) 直接 (μ) 加 shuffle 层**：中庸；perf 不确定，可能让 TDM 收益归零。
4. **(d) (ν) 撤回**：AICK-579 部分算结题（B1 已 21–32%）；B2 留作未来 work。

时间紧 → 跳过 (λ-test) 直接 (μ) 是 pragmatic；要正确解 → (λ-test) → (λ) 是正路。

## 7. 当前 worktree 状态

- HEAD：B1 commit `16324795215`（clean）
- modified（unstaged）共约 8 处：
  - 3 plumbing（lead 加）
  - 2 B2 真修复（pipeline + policy）
  - 1 (α') fix（kernel group path unmerge typo）—— 独立 latent bug
  - 1 (X) fix（`container_helper.hpp` inclusive_scan）—— CK core mixed tuple
  - 4 (X+) fix（`tuple.hpp` operator +/-/*/* in-place）—— CK core mixed tuple
  - mentor confirmed 的其它小 fix
- 所有 diagnostic CK_PRINT（D5、Hδ-1 等）已清。
- ABC 没跑（编译过但 case A valid:n）。

(α') / (X) / (X+) 都是独立 latent bug，无论 B2 方向选什么都建议 keep / 走独立 PR。

## 8. 同步并行项

- **(α') develop bug fix PR #6964**：qa-native 越权开的（流程问题），但内容质量好。等用户拍 accept / close / draft。
- **develop 副发现**：114 个 fp16/bf16 group fail 案例，等用户拍是否追。

---

**等用户决策**：
- B2 方向：(λ-test) → (λ)? 直接 (λ)? (μ)? (ν)?
- (α') PR #6964 处理方式？
- develop 114 fail 是否追？

team-lead 已 pause 所有 trial 推进，QA 不再派新 verify task，等用户回复。
