# B2 Ship Complete — Post-Compact Recovery Baseline (lead)

写作时间: 2026-05-07 04:30，B2 functional milestone 已 ship + X+ rebase chain 完成 + user 准备 compact lead context。
目的: post-compact 后 lead 读这一份就能恢复 active state + 知道下一步候选。

---

## 当前 active state

- **方向**: B2 Step done。等 user 拍下一步 (Step C backlog / AICK 577/578/580 主线 / 休息 / 其他)
- **Active task**: 无 (#11 completed, Task #8 Step C backlog pending 无 owner)
- **Worktree**: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel`，**clean** 无 unstaged
- **用户最后指令**: "我打算先 compact 你的 context 再进行下一步" → standby 不主动 push 方向

---

## ✅ Shipped this session (B2 milestone)

### Branches force-pushed to `internel` (by user 自己执行 push)

```
yiding12/tuple-mixed-inplace-container-ops    079aa02a60e (X+) → b94cefafec5 (gfx1250 ancestor)
yiding12/gfx1250-fmha-tdm                     07b82d03301 (B2)
                                              ↓
                                              db26fd14f99 (B1 rebased)
                                              ↓
                                              153fe1ccfe3 (Init QRKSVS_TDM rebased)
                                              ↓
                                              079aa02a60e (X+, shared parent)
                                              ↓
                                              b94cefafec5 [GFX1250] support GroupedGemm with MX FP8 (#1236)
```

### B2 ABC ship state
| case | valid | max_err | sim_ms | vs B1 |
|---|---|---|---|---|
| A (fp16 dense s=1023) | y ✓ | 0 | 2606 | +0.6% |
| B (fp16 GQA causal s=1023×257) | n | 0.0620 | 369 | -1.2% (KNOWN ISSUE) |
| C (bf16 long s=2047) | y ✓ | 0 | 9362 | +2.7% |

- kname 全 `qr_tdm_vr_npad` ✓
- Perf 全 < +10% B1 远低于 50% threshold
- Q dump 32/32 sanity perfect (case A) — sentinel 完全消除 confirm Q TDM coverage 修好

### Fix stack in B2 commit
- (λ-K) trivial tile-major K dram dist (mirror GEMM v1 ColMajor B layout)
- (β-K) K LDS read view Xor=true → plain row-major
- (λ-Q) trivial tile-major Q dram dist (mirror K λ pattern)
- (β'-Q) Q LDS read view Xor=true → plain row-major
- Q TDM padding disabled (mirror K/V — fix sentinel coverage gap)
- K+Q desc Xor template parameter removed (~80 lines dead code)
- (α') unmerge typo fix in kernel.hpp
- qr_vr emit disabled in codegen (workaround 强制 dispatch qr_tdm)

### Case B known issue (full root cause analysis in B2 commit body)
- **Root #1**: CK core latent bug `tile_window.hpp:1786-1789 get_cached_global_strides()` ignores actual tensor view stride field, uses shape-based packed-default. Triggers in multi-head Q dispatch (case B kargs.stride_q=256 vs cached_strides[0]=128). Affects ALL TDM consumers with non-packed-stride tensors.
- **Root #2** (NEW from γ1 disambig): (λ-Q) trivial tile-major Q dist itself produces wrong thread→element mapping for multi-head GQA — γ1 attempt switching writer (TDM→async) showed both writers produce different garbage with same dist
- → Need cross-repo CK core fix + GFX team Q dist multi-head design review

---

## 📋 Step C backlog (Task #8 pending)

按 user-impact + scope 排序，等 user 决定 priority:

1. **CK core fix `get_cached_global_strides`** — tile_window.hpp:1786-1789 真用 actual stride, 1-2d + cross-repo coordination + regression sweep. 修 case B root #1 + unblock 其它 TDM consumer multi-head.
2. **(λ-Q) Q dist GQA-aware redesign for multi-head** — 1-2d design + impl + verify, 配合 #1 → case B valid:y
3. **Dispatcher prefer qr_tdm over qr_vr** — 替代 qr_vr disable workaround, 1-2d + regression sweep
4. **K/V padding re-enable** per "re-enable after pass" 承诺 — ~1h test + ABC verify, perf 优化

---

## 🛣 AICK-577/578/579/580 主线 (gfx1250 FMHA full scope, per memory `project_gfx1250_fmha_scope.md`)

- AICK-579 K-side TDM 现 partial done (single-head ship, multi-head pending Step C #1+#2)
- AICK-577/578/580 (V-side TDM / BWD / etc.) 没 implement

---

## 团队 layout (tmux)

- Session: `Claude`
- Window 1 (main): lead `%14`
- Window 2 (teammates): swe `%30`, reviewer `%32`，`%29` `%31` 是 qa/mentor (functional anomaly, members config 只列 swe + reviewer 但 qa/mentor 实际 alive)
- 全员 ack ready + standby

⚠️ Team config anomaly (同上次 session): `~/.claude/teams/gfx1250-fmha-tdm/config.json` `members` list 只 2 个 (swe + reviewer)，但 qa/mentor inbox 存在 + alive — 不要重 spawn。

---

## Cron 巡检

- Job ID: `fee9ca71` (session-only, 7,37 分 schedule, 30min 错峰)
- 还在 fire 中 (整 session 一直 active)

---

## 💡 Memory updates this session (universal lessons saved)

`~/.claude/memory/feedback_stage_level_verify.md` (大幅扩充):
1. **Stage-level fix 必须 direct stage dump verify, 不能 only end-to-end signal** (originally added)
2. **Pipeline-level fix verify pattern** — modify single stage but verify full pipeline (batch dump 全 data flow stages 比 sequential 省时间, first divergence stage = bug location)
3. **Signal type matters** — sentinel/uninit / permute/scramble / magnitude-off → 不同 root cause mechanism (sentinel vs permute 整团队踩 K↔Q analogy 误判 1.5h)
4. **Multi-stage co-design verify** — 改 stage X 时 trace 完整 co-design chain, code comment 自身往往 document architectural 约束
5. **Writer coverage 5 manifestations** — dist scatter / padding skip / sync timing / boundary mask / tile alignment

`~/.claude/memory/project_gfx1250_fmha_no_shuffle.md` (existing): (μ) explicit LDS shuffle stage 永久禁

---

## 🔧 Lead pre-flight discipline 教训 (本 session 踩 2 次)

1. **Rebase 前必数 commit count** (`git log A ^B --oneline | wc -l`) — 不只 focused diff verify
2. **本地 branch 跟 remote 同名 branch 可能 lineage 不同** — 不假设 `gfx1250` (本地) = `internel/gfx1250` (remote)，verify before assume
3. **`origin/develop` ≠ `internel/develop`** — 不同 ancestors of CK core fixes (e.g. PR #6964 status differ)
4. **Single-line grep ≠ multi-line pattern match** — multi-line code patterns 用 `python` regex with `[\s\n]+` 或 `rg --multiline`

---

## User 已明确委托项 (不要再追)

- **PR #6964** ((α') unmerge typo fix to develop): 用户原话 "**6964 不用你管，我自己会注意合并的**"。Lead 不 status check / 提醒 / 跟踪
- **Develop 副发现 (114 fp16/bf16 group fail)**: 上轮 auto-decided "留给 fmha team 不深挖"
- **B2 + X+ push**: user 自己 force-push 完成，不创 PR

---

## Lead 工作流提醒 (按 prompt.txt §3)

- 技术细节 SWE 直接找 mentor，**不报 lead**
- Lead 只接 4 类: 方向决策 / 跨 repo 协调 / task 派发请求 / Step final
- 自决边界: reversible directional → 立即 Teams notify + 1h 等用户 + 自决；不可逆对外 = 永远等
- **Reviewer 是 4th role** — devil's advocate，auto-monitor SWE/mentor 新 sign-off doc 给 challenge，只报 lead
- **本 session 用了 3 次 1h auto-decide**:
  1. Option 2 (fix B before ship) at 12:22 (后被 user methodology challenge invalidate, 转 Option 1)
  2. Option γ TDM disable for multi-head Q at 15:22 (后 γ1 fail)
  3. Pivot Option 1 (ship A+C flag B) at 15:22 (最终 ship state)

---

## Reviewer 价值 (post-compact 容易忘)

Reviewer (effort=max 用户中途调高) 这次 session 关键贡献:
1. 找到 Q smoking gun via 15min trace (Q TDM writer + Xor=true reader pattern)
2. 找到 Option σ 真正第 4 路径 (虽然最终没用上)
3. Pre-sign-off (β')-Q-fix conditional verify (avoid premature commit)
4. 找到 (λ-Q) layer missing piece via code comment self-documents evidence
5. 5 holes audit on B1==B2 byte-identical conclusion (catch Q+S+P+O downstream chain)

Lead 应该:
- 看到 SWE/mentor sign-off 时 ad-hoc ping reviewer 做 second opinion
- Reviewer 给 brief endorse 时 explicit relay user "reviewer agrees"
- Reviewer 给 full challenge 时 relay user 完整 4-section report

---

## 用户给"开始"信号后 lead 应做的

1. `TaskList()` 验证 Task #8 Step C backlog 仍 pending
2. Read 这份 baseline doc 全
3. 跟 user check 优先级 — 4 个 Step C item 还是 AICK-577/578/580 V-side TDM
4. 派 task to SWE/mentor/reviewer/qa 按 user 决定方向
5. 不主动 push 方向
