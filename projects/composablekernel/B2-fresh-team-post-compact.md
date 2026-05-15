# B2 Fresh Team Post-Compact Recovery Baseline (lead)

写作时间: 2026-05-06，fresh team spawn 完 + 全员 standby + 用户准备 compact lead context 后。
目的: post-compact 后 lead 读这一份就能恢复 active state。

---

## 当前 active state

- **方向**: B2 (λ-revised) 路径 — K reg dist 跟 trivial tile-major TDM write 的 coordination 缺失（不是 wmma 几何墙，mentor 三次 sign-off 错被纠正）
- **Task #2** [pending]: "Step B2 — fresh team spawned, standby for user direction"
- **Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`，HEAD = `16324795215` (B1 commit)，**8 modified unstaged 全保留**
- **用户最后指令**: "compact 完再让 team 开始工作"——所以 team standby 等用户给"开始" signal

## 团队 layout (tmux)

- Session: `Claude`
- Window 1 (main): lead `%14`
- Window 2 (teammates): swe `%30`, reviewer `%32`，剩 `%29` `%31` 是 qa/mentor (config.json members 列表只有 swe/reviewer，qa+mentor 是 functional anomaly 跟之前 session 同 pattern — **不要 重 spawn 它们**)
- 全员 ack ready + standby（不主动工作）

## ⚠️ 团队 config 异常（同上次 session）

`~/.claude/teams/gfx1250-fmha-tdm/config.json` 的 `members` list **只有 2 个** (swe + reviewer)，**qa + mentor 不在里面**。但：
- qa/mentor inbox 存在 + ack ready 消息已收到
- qa/mentor pane alive in tmux window 2
- qa/mentor 之前能正常 SendMessage（已 ack ready + standby）

→ qa/mentor 实际是 functional teammate，**别基于 config 判断 "qa/mentor 没 spawn" 然后重 spawn**——会冲突。
→ 派 task 给 qa/mentor 用 `SendMessage(to="qa", ...)` / `SendMessage(to="mentor", ...)` 正常工作。

## Cron 巡检

- Job ID: `fee9ca71`
- Schedule: `7,37 * * * *` (每小时 7 分和 37 分 = 30min 间隔错峰)
- Session-only，session 死就停，7 天 auto-expire
- Prompt: 巡检 team status，看孤儿 inbox / SWE/mentor 卡 >1h / etc.

## 用户已明确委托项（不要再追）

- **PR #6964** ((α') unmerge typo fix to develop): 用户原话 "**6964不用你管，我自己会注意合并的**"。已 poyenc APPROVED + CI 全绿。lead 不要 status check / 提醒 / 跟踪。
- **develop 副发现 (114 fp16/bf16 group fail)**: 上轮 auto-decided "留给 fmha team 不深挖"。lead 不要派 investigator。

## 关键 context 文档（必读，恢复 (λ-revised) 方向背景）

按 priority 读：

1. **`B2-resume-baseline-v3.md`** — SWE 写的全景 baseline (trial timeline + 4 fix candidates α/γ/μ/ν + worktree state + lead action items)
2. **`mentor-hardware-knowledge-fixed.md`** — mentor 写的 hardware fact baseline + 三次 sign-off 错 anti-pattern + 6 protocol rules
3. **`mentor-reanalysis-ds-load-tr-correction.md`** — ds_load_tr 真实 semantics (K-outer N-inner row-major LDS → K-inner VGPR transpose) + (h) fail 真根因
4. **`swe-reanalysis-K-side-correction.md`** — SWE 修正后真根因 hypothesis + LDS dump disambig plan
5. **`qa-env-notes.md`** — QA env + log inventory

## (λ-revised) 实施核心要点

修正后的真根因：**K reg dist (BWarpDstrEncoding embed) 跟 trivial tile-major dram dist 缺 coordination**，不是 wmma 几何墙物理不兼容。

**推荐 next sequence**:
1. **Step A: K LDS dump disambig** (~2-3h dev) — kargs 加 debug ptr + thread 0 单 block dump LDS region + 跟 dram K tile expected layout 对账
2. **Step B**: 基于 dump 结果选 fix candidate
   - **Option α**: 改 K reg dist outer encoding match plain row-major (1-2d, if dump shows LDS = plain row-major)
   - **Option γ**: 改 K dram dist hybrid match B1 reg dist (3-5d, if dump shows LDS ≠ plain row-major)
3. **Step C**: V 同性质改造 (1-2d，K side 通了之后)
4. **Plan B**: (μ) shuffle stage 或 (ν) 撤退 if Step A reveals deeper issue

## 用户给"开始"信号后 lead 应做的

1. `TaskList()` 验证 #2 仍 pending
2. Read 上面 5 个 worktree 文档（特别是 B2-resume-baseline-v3.md + mentor-hardware-knowledge-fixed.md）
3. `TaskUpdate(taskId="2", status="in_progress", owner="swe")` (或新建 task #3 = LDS dump disambig)
4. SendMessage SWE：派 LDS dump disambig 第一步 task — **让 SWE 跟 mentor 讨论 dump implementation design 过后再 implement**
5. **不主动催**——按 SWE+mentor 内部协议运转
6. SWE 实施 + QA 跑 + dump 结果 → SWE 报方向选择 (α/γ/μ/ν) → reviewer 自动 challenge → lead audit 准入

## Lead 工作流提醒（按 prompt.txt §3）

- 技术细节 SWE 直接找 mentor，**不报 lead**
- Lead 只接 4 类: 方向决策 / 跨 repo 协调 / task 派发请求 / Step final
- 反例: 别因为收到 D1/D2/.../D5 单步诊断结果就给用户汇报
- 自决边界: reversible directional → 立即 Teams notify + 1h 等用户 + 自决；不可逆对外 = 永远等
- **Reviewer 是新加入的 4th role** — devil's advocate，auto-monitor SWE/mentor 新 sign-off doc 给 challenge，只报 lead

## Reviewer 价值 (post-compact 容易忘)

Reviewer 这个角色是这次 session 新加进 team 的（基于 (λ) session 三方 echo chamber 教训）。Lead 应该:
- 看到 SWE/mentor sign-off 时 ad-hoc ping reviewer 做 second opinion
- 用户拍方向前 → ping reviewer 独立审 user 的 challenge 是否成立
- Reviewer 给 brief endorse 时 explicit relay user "reviewer agrees"
- Reviewer 给 full challenge 时 relay user 完整 4-section report
