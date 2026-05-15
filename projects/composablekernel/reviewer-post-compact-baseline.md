# Reviewer Post-Compact Baseline

**写作时间**: 2026-05-06
**Effort 设定**: max (本 session 中后段提升, 必须保留 post-compact)
**状态**: standby — Step B2 ship pending SWE Q padding disable case A 结果 + Step C backlog 待 lead dispatch

---

## 1. Role 定位 — explicit devil's advocate

**核心理由 (anti-pattern reflection — 必读)**:

2026-05-06 mentor 三次 sign-off 错全部基于"理论 reasoning 没配 empirical verify":
1. (λ) 基于 GEMM v1 untested reference design
2. (h) 推 box_dim verify ≠ functional verify, 没 challenge K side 假设
3. ds_load_tr semantics 描述完全反向

SWE+lead 都 inherit mentor 错前提没独立 verify, 直到 user 自己读 production code 发现真根因。

**Reviewer 存在就为防第 4 次同 pattern**:
- Default stance: explicit devil's advocate, 找 SWE+Mentor+Lead 三方共识里的 confirmation bias 盲点
- **找不到 hole 才 endorse** — endorse 时要 explicit 标"reviewer agrees"
- 不 approval-stamping —— 没充分 challenge 过的 sign-off 不 endorse
- 不 perpetual blocker —— challenge 后必须 propose actionable disambig (e.g. "做 LDS dump") 而不 paralyze decision

---

## 2. 通信协议

- **只走 lead 中转** — 不直接 SendMessage SWE/mentor (避免 derail SWE+mentor 内部循环)
- **报告 auto-adapt format**:
  - 同意三方 brief endorse
  - 不同意 full 4-section structured report (root cause / options re-score / recommendation / challenge questions for user)
  - Open question 多但没 strong objection: list critical questions 让 SWE/mentor 再 verify
- **Ad-hoc ping**: lead 用 `SendMessage(to="reviewer", ...)` ping 时附 specific doc paths, 避免漫无边际 read
- **Auto-trigger**: SWE/mentor 写新 status doc / sign-off doc / reanalysis doc 后, reviewer 自动 read 给 challenge

---

## 3. 本 session 5 个关键贡献 pattern (Step B2 disambig 全过程)

### Contribution #1: Q smoking gun trace via 15min file dive

- **Trigger**: lead 报 (β) K-side byte-identical 但 max_err 仍 0.36, 准备走 batch dump 6 stages (~1-2h)
- **Approach**: 15min effort=max trace tdm.hpp + async_trload.hpp 比 B1 vs B2 各 stage call site
- **Finding**: Q 在 B2 走 TDM (line 292) 但 B1 走 async_load (line 272), reader 用同 Xor=true desc — 跟 K (h) hybrid 失败 pattern 完全 isomorphic
- **Lesson**: deep targeted trace 比 sequential disambig 快很多, 跟 K↔V 关系类比着 grep 同名函数 quickly identify divergence
- **Pattern 通用化**: B1 vs B2 类比时, 不能假设 Q/V 跟 K 同步骤; 必须 grep 全 call site 跨文件 verify

### Contribution #2: Option σ 第 4 路径 (K via ds_load_tr)

- **Trigger**: (β) 假设 fail 时 lead 列 ν' / σ / α' 当 fail framework
- **Approach**: 系统列举 fail-scenario 4 paths 时新发现 σ — K LDS layout 改 K-outer 跟 V 同, 用 ds_load_tr 硬件 transpose
- **Finding**: ds_load_tr 是 hardware feature (K-outer LDS → K-inner VGPR), 不依赖 software dist projection 的 byte-position 假设, **比 ds_load 路径 architecturally 更 robust**
- **Lesson**: fail framework 列 candidate options 时, 不要只 inherit mentor/SWE 已提的, systematic 重审 hardware feature space (ds_load vs ds_load_tr / async vs TDM / cluster vs single 等)
- **Pattern 通用化**: 任何 fail framework 的"剩余选项"都要 systematic 重审, 别 stop 在 mentor 之前提的 options 集合

### Contribution #3: Pre-sign-off conditional verify

- **Trigger**: (β')-Q-fix 提议时 lead 问要不要 trace `MakeQLdsBlockDescriptor` 2nd 模板参数确认是 Xor
- **Approach**: 5min trace 验 Q desc signature, confirm pattern 100% match K. 然后给 SWE pre-sign-off (1-line fix change spec), risk LOW, expected B2 Q tile == B1 Q tile
- **Finding (随后被部分推翻)**: pre-sign-off 不能完全替代 empirical verify。**(β')-Q-fix 实测 PARTIAL** — 我假设 Q 跟 K 都只缺 reader desc 一层 fix, 实际 Q 还缺 (λ-Q) dram dist 一层
- **Lesson**: pre-sign-off 价值是减少 sign-off 等待, 但**不能假设两 stage isomorphic single fix work for one means single fix work for another**。完整 co-design chain (writer dist + writer + reader view + reader dist) 必须独立 trace 哪几层 piece 在 stage X 改过, 同样几层 piece 在 X' 都需改
- **Pattern 通用化**: 类比 reasoning 高 risk; pre-sign-off 必须显式说明 architectural assumption + 标记 "需要 empirical confirm"

### Contribution #4: (λ-Q) layer missing piece via code comment self-doc

- **Trigger**: (β')-Q-fix PARTIAL fail (max_err 0.36 → 0.12, tid 1 仍 sentinel)
- **Approach**: 重 trace `MakeKDramTileDistribution` 实施 — 发现 line 142-163 注释**自己写明**了 K (λ-K) 是为 TDM box-major write 兼容做的, B1 5D dram dist + TDM = 100%-wrong garbage
- **Finding**: Q dram dist 仍是 B1 5D async-style, K (λ-K) 的 TDM 适配设计没应用到 Q。Q 缺 (λ-Q) 这一层。
- **Lesson**: **Code comment 是 ground truth, 比 first-principle reasoning 强**。trace 时优先 read author intent (注释), 不要只 read syntax
- **Pattern 通用化**: 任何 multi-piece refactor 的状态判断, 优先 grep author 注释里的 "Why this is needed" / "B2 H3 root cause" 等 self-documenting markers

### Contribution #5: 5 holes audit on "K-side OK byte-identical" conclusion

- **Trigger**: lead 报 K (β) byte-identical → K-side OK → root cause 在 V/softmax/gemm1
- **Approach**: 严肃 audit lead 的 conclusion, 发掘 4 个 hole + 1 个 process improvement
  - Hole #1: dump iter coverage (case A 64 iter, dump 是哪 2 iter?)
  - Hole #2: Q 完全 unverified (gemm0 OTHER input)
  - Hole #3: lead disambig sequence 顺序错 (V 排 S 之前)
  - Hole #4: kIsTdmPipeline / V dual view dispatch leftover (silent diverge 路径)
  - Process: pipeline-level fix verify 要 batch dump 全 stages 不只改的那个
- **Lesson**: lead 给的 conclusion 也要 systematic challenge — 不能因为 lead 说 "K OK" 就跳过 verify 直接 disambig 下游
- **Pattern 通用化**: "X verified → root cause 在 not-X" 这种 conclusion 的 disambig 必须 audit "X 的 verify 是不是 sufficient + 完整" — 数据 coverage / sibling input / disambig 顺序 / silent dispatch leftover 全要 check

---

## 4. 累积的 methodology lessons (3 个 anti-pattern variants)

### Anti-pattern A: First-principle reasoning 没配 empirical verify (mentor 三次错)

- **症状**: 基于 source code 阅读推断 hardware semantic / 解耦 claim, 不查 production verified runtime
- **解法**: Hardware semantic 描述前先 grep B1 verified path actual usage; sign-off 前写 specific empirical predict + plan disambig step

### Anti-pattern B: Indirect signal 当 direct verify (整团队这次错)

- **症状**: max_err 是 end-to-end signal, gemm1 + softmax 干扰让 K-side 改对了也看不出来。整团队 (mentor + SWE + QA + reviewer) 都漏。
- **解法**: 改 stage X 修复 issue Y 时 dump 那个 stage 的 actual output medium (write-side stage → LDS bytes / dram bytes; read-side stage → register distributed tensor via CK_PRINTF; compute stage → register accumulator)
- **Memory**: `~/.claude/memory/feedback_stage_level_verify.md` (lead saved)

### Anti-pattern C: Co-design chain 类比太粗 (reviewer 这次错)

- **症状**: 假设 X (e.g. Q LDS desc) 跟 X' (K LDS desc) isomorphic → 单 fix work for K means same single fix work for Q。错 — K 经历的是 **(λ-K) + (β-K) 双层 fix**, Q 只做 (β'-Q) 缺 (λ-Q)
- **解法**: 改 stage X 时 trace 完整 co-design chain (writer dist + writer + reader view + reader dist) 验证 K 在哪几个 piece 改过, X' 同样几个 piece 都需改
- **Writer coverage 有 5 种 manifestation** (mentor + reviewer + lead 共建): dist scatter / padding skip / sync timing / boundary mask / tile alignment

---

## 5. 当前项目状态 (compact 时刻 snapshot)

### Step B2 状态

- (β-K) verified work: K register tile B1 vs B2 byte-identical
- (β'-Q) PARTIAL: max_err 0.36 → 0.12, tid 0 修, tid 1 仍 sentinel
- (λ-Q) applied: 修 dist scatter coverage
- Q padding disable in-flight (~5min QA case A): 修 padding skip writer coverage
- 期望: case A valid:y + Q dump tid 1 sentinel 消失 → ABC verify → ship audit

### Step C backlog (即将 dispatch — 4 项)

不知具体内容, lead 会 dispatch。预知 Step C 第一件事审 **K/V padding re-enable** 给 bank conflict 减少 (perf optimize, λ-1 disambig 时 temporarily disabled 的注释 intent)。

### worktree 状态 — 多 modified files 不 commit (per ship gate)

最终 single PR 含 (β + β'-Q + λ-Q + Q padding disable + K Xor 参数 cleanup + Q Xor 参数 cleanup + dead code 清掉 kIsTdmPipeline + tdm_config_v + V dual view leftover)。

### Worktree dead code 待清

- `kIsTdmPipeline` SFINAE detection (kernel.hpp:86-104) — 0 处 usage
- `has_kIsTdm_trait` helper — paired with above
- `kIsTdm = true` trait (pipeline class line 77)
- `tdm_config_v` declarations (line 249, 264-267) — V 走 async 这个 setup wasted
- V dual view dispatch (kernel.hpp:1652-1690 区域)
- `GetLdsPaddingConfigV_original_disabled` rename 备用

---

## 6. Resume 时 reading priority

Compact 完 resume, 优先读:

1. **本 doc** (你正读) — role + 5 contributions + lessons + state snapshot
2. `~/.claude/memory/MEMORY.md` — global memory index
3. `~/.claude/memory/feedback_stage_level_verify.md` — lead saved methodology lesson
4. Lead inbox — Step B2 ship 状态 + Step C backlog dispatch
5. (按需) `B2-resume-baseline-v3.md` — 项目 timeline (大部分 superseded by post-(β)/(λ-Q) progress, 但仍 trial 历史 reference)
6. (按需) `mentor-hardware-knowledge-fixed.md` — hardware semantic baseline (ds_load_tr / TDM / WMMA)

---

## 7. Standby protocol post-compact

- **Auto-monitor trigger** (按 prompt §7 file pattern): SWE/mentor 写新 status doc / sign-off doc / reanalysis doc → 主动 read + 写 challenge 给 lead
- **Ad-hoc ping**: lead 用 `SendMessage(to="reviewer", ...)` ping 时按指示 read specific doc + audit
- **Don't trigger**: SWE/mentor 单步诊断 D1/D2 / SWE+mentor 内部 implementation 细节 / QA 跑测试结果 / Lead routine 巡检

如 lead Step C dispatch 后 ping reviewer audit Step C plan, 按本 doc Section 4 三个 anti-pattern 框架 + Section 3 五个 contribution pattern 应用。

---

## 8. 关键 file paths quick ref

### Project
- Worktree: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel`
- Branch: `yiding12/gfx1250-fmha-tdm`
- B1 commit (verified baseline): `16324795215`

### Key code files (relative to worktree)
- B2 pipeline: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp`
- B2 policy: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- B1 reference pipeline: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`
- B1 reference policy: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`
- Kernel: `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`
- WMMA encoding: `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`
- BlockGemm: trace via `BlockGemmARegBRegCRegV2` (Q=A, K=B for QK GEMM; P=A, V=B for PV GEMM)

### Status docs
- `B2-resume-baseline-v3.md` (superseded by progress)
- `mentor-hardware-knowledge-fixed.md` (hardware semantic baseline)
- `swe-status-K-LDS-dump-disambig.md` (K dump experiment)
- `swe-reanalysis-K-side-correction.md` (post ds_load_tr correction)
- `mentor-reanalysis-ds-load-tr-correction.md` (mentor self-correction)

### ABC test cases (case A is current debug case)
- A: fp16 batch s=1023 d=128 mask=0 (sim ms B1 baseline 2539)
- B: fp16 batch GQA+causal s=1023×257 d=128 mask=2 (sim ms 374)
- C: bf16 batch s=2047 d=128 mask=0 (sim ms 9113)

---

(end)
