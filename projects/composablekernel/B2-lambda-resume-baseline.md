# B2 (λ) Resume Baseline (post-compact 用)

写作时间: 2026-05-06，team `gfx1250-fmha-tdm` 重启完成 + 全员 standby 后。
目的：context compact 后 lead 读这一份就能恢复 B2 (λ) 全部 active state。

---

## 当前 active state

- **方向**：用户拍 **(λ)** = 重设 K/V 4 个 dist + BlockGemm A operand wrapper，mirror GEMM v1 co-design
- **Task #1**: pending，B2 (λ) 重设 K/V dist + BlockGemm wrapper，含 ABC checklist
- **Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`，HEAD = `16324795215` (B1 commit)，**8 modified unstaged 全保留**
- **用户最后指令**："重启完先不着急工作，等我指令"——所以 team standby 等用户给"开始"

## 团队 layout (tmux)

- Session: `Claude`
- Window 1 (main): lead `%14`
- Window 2 (teammates): swe `%22`, mentor `%21`, qa `%20`
- 全员 ack ready + standby（不主动工作）

## ⚠️ 团队 config 异常

`~/.claude/teams/gfx1250-fmha-tdm/config.json` 的 `members` list **只有 2 个** (mentor + swe)，**qa 不在里面**。但：
- qa.json inbox 存在
- qa pane %20 alive in tmux
- qa 之前能正常 SendMessage（已 ack ready + standby）

→ qa 实际是 functional 第 3 个 teammate，**别基于 config 判断 "qa 没 spawn" 然后重 spawn**——会冲突。
→ 派 task 给 qa 用 `SendMessage(to="qa", ...)` 正常工作。

## 用户已明确委托项（不要再追）

- **PR #6964** ((α') unmerge typo fix to develop): 用户原话 "**6964不用你管，我自己会注意合并的**"。已 poyenc APPROVED + CI 全绿。lead 不要 status check / 提醒 / 跟踪。
- **develop 副发现 (114 fp16/bf16 group fail)**: 上轮 auto-decided "留给 fmha team 不深挖"。lead 不要派 investigator。

## 关键 context 文档（必读，恢复 (λ) 方向背景）

按 priority 读：

1. `mentor-fmha-tdm-design-notes-202402.md` — **最重要**，含 (λ) 实施 plan 5 处 line range + 3-dist 拓扑分析 + GEMM v1 co-design 对照 + (λ-test) 为何不可行
2. `decision-B2-H3-direction-174838.md` — (λ/μ/ν) 决策利弊对比
3. `swe-journey-B2-trials-202429.md` — 6 轮 hypothesis 完整时间线（参考用，不必精读）
4. `final-summary-team-shutdown-212300.md` — 上轮 shutdown 时全部 state

## (λ) 实施核心要点（mentor design notes 提炼）

5 处 coordinated 改动（任一处 align 漏 = 编过但 valid:n）：

| 文件 | line | 函数 |
|---|---|---|
| `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` | 152-178 | `MakeKDramTileDistribution` |
| 同上 | 581-610 | `MakeKRegTileDistribution` |
| 同上 | 625-660 | `MakeVDramTileDistribution` |
| 同上 | 678-720 | `MakeVRegTileDistribution` (V 还要 trload 模式适配) |
| `block_fmha_pipeline_qr_ks_vs_tdm.hpp` | TBD | `GetQKBlockGemm` 返的 BlockGemm 0 A operand wrapper |

**风险提醒**：
- B1 教训"动 dist 风险高"完全适用
- GEMM v1 在 build-gfx1250 也 0 instance（co-design 设计**也未真跑通过**）—— (λ) 是 new territory，没 GEMM v1 production 案例兜底
- BlockGemm 0/1 wmma operand layout 不熟，可能踩兼容坑
- 预估 1-2 周 SWE 工作量

## 用户给"开始"信号后 lead 应做的

1. `TaskList()` 验证 #1 仍 pending
2. Read 上面 4 个 worktree 文档（特别是 mentor design notes）
3. `TaskUpdate(taskId="1", status="in_progress", owner="swe")`
4. SendMessage SWE：派 (λ-1) K dist 拓扑 design 第一步，**让 SWE 跟 mentor 讨论 design review 过后再 implement**——别让 SWE 直接动代码
5. **不主动催**——按 SWE+mentor 内部协议运转，等 SWE 报"design 收敛要开干"
6. SWE 改完每一处 dist → QA verify → 失败回 SWE+mentor 内部循环

## Lead 工作流提醒（按 feedback memory）

- 技术细节 SWE 直接找 mentor，**不报 lead**
- Lead 只接 4 类: 方向决策 / 跨 repo 协调 / task 派发请求 / Step final
- 反例: 别因为收到 D1/D2/.../D5 单步诊断结果就给用户汇报
- 自决边界: 方向性 → 立即 Teams notify + 1h 等用户 + 自决；可逆方向 = 1h auto-decide；不可逆对外 = 永远等
