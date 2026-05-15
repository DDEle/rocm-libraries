# Post-Compact Recovery — 2026-05-15

This is the 3rd recovery doc in the chain. Read prior docs first if context fully lost:

1. **`v-tdm-blocked-recovery-2026-05-11.md`** — pre-compact baseline (HEAD `08b61885fc9` shipped, V-TDM blocked, mentor verdict, 4 user options)
2. **`step-c1-attempt-blocked.md`** — older recovery (Step C #1 attempt details)
3. **THIS doc** — what changed since 5/11 compact

---

## TL;DR

- Step C #1 ship state **unchanged** (HEAD `08b61885fc9` on `internel/yiding12/gfx1250-fmha-tdm`)
- V-TDM **still blocked** waiting user pick a/b/c/d (per `v-tdm-blocked-recovery-2026-05-11.md`)
- Worktree **still has uncommitted V-TDM edits** (3 files, broken state) — user has not directed cleanup
- **Team killed**: 4 teammate instances (mentor/swe/qa/reviewer) terminated 2026-05-15 ~03:43 UTC, NOT respawned per user "先不重启 team, 只把它们关了"
- 4 teammate consolidation docs salvaged + 3 new lead memories written this session

## Worktree state (verified 2026-05-15 03:50 UTC)

```
git log: 08b61885fc9 [CK Tile] Step B2 — gfx1250 FMHA FWD TDM Q+K (single-head)
         556abb2b01a Step B1: rewrite K/V distribution ...
         ac982130db8 Init QRKSVS_TDM

git status:
 M include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp
 M include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp
 M include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp
```

3 modified files = V-TDM 9-piece uncommitted (ABC ALL FAIL state per mentor verdict). DO NOT commit, DO NOT revert without user direction.

## Open user decision (carried over from 5/11)

V-TDM path:
- (a) revert all V-TDM edits + done (mentor recommends)
- (b) keep worktree, run LDS dump probe
- (c) fallback (h) hybrid + reclassify Step C #2 defer
- (d) other

## What happened this session (5/14-15)

1. **Many cron patrols** (60+) while user AFK 5/11-5/15. All "Standby, no change". Last team substantive activity was 5/8 15:09 UTC (mentor V-TDM verdict).

2. **5/15 ~03:13 UTC**: User came back, asked lead to dispatch teammate consolidation before restart team.

3. **5/15 03:13-03:27**: Lead dispatched 4 teammates (mentor/swe/qa/reviewer) consolidation prompts. **First version of prompt allowed teammate to choose `~/.claude/memory/` OR project doc** — major mistake.

4. **5/15 03:35-03:37**: 4 teammates all picked memory, all hit `permission_request` stuck on tool-permission gate. Lead `SendMessage` plain text "deny" did NOT unblock (no `permission_response` protocol exists).

5. **5/15 ~03:42**: Lead salvaged 4 teammate attempt contents from inbox `.text` payload via `jq -r '.text | fromjson | .input.content'`, wrote to project doc:
   - `mentor-self-correction-empirical-rule-2026-05-15.md` (anti-pattern A 4th recurrence rule)
   - `swe-am-workflow-gotcha-2026-05-15.md` (rocdtif r5.04 wrapper sed + **fmha AM 必须 `-timer=cpu` 否则 short-circuit** — NEW finding)
   - `qa-verify-heuristics-2026-05-15.md` (ABC case sensitivity table + kname assertion)
   - `reviewer-audit-heuristics-2026-05-15.md` (3 review heuristics)

6. **5/15 ~03:43**: Lead killed 4 teammate instances (`kill <pid>`) per user "只把它们关了". Tmux panes also killed.

7. **5/15 ~03:50**: Lead self-consolidation — 3 new memories written:
   - `~/.claude/memory/feedback_teammate_no_memory_write.md`
   - `~/.claude/memory/reference_teammate_permission_request_handling.md`
   - `~/.claude/memory/feedback_pre_restart_consolidation.md`

8. **MEMORY.md index** updated with 3 new entries.

## Team config state

- `~/.claude/teams/gfx1250-fmha-tdm/config.json` still lists 5 members (lead + 4 dead teammates)
- `inboxes/*.json` retain full message history (NOT cleaned)
- No teammate process alive (verified `ps aux | grep agent-name`)
- Tmux panes %2/%3/%4/%8 killed; only lead pane %1 + code pane %0 remain

If user later wants restart: re-spawn via Agent tool with appropriate prompts. Recovery docs (per-role baselines + this doc + v-tdm-blocked-recovery-2026-05-11.md) cover bootstrap.

## Per-role consolidation doc summary

For future spawn (per-role bootstrap addition to existing `*-post-compact-baseline.md`):

| Role | Doc | Key content |
|---|---|---|
| mentor | `mentor-self-correction-empirical-rule-2026-05-15.md` | Self-correction must use new empirical, not reasoning. Anti-pattern A 4th recurrence (V→TDM verdict reversal without LDS dump → 5-6h SWE waste) |
| swe | `swe-am-workflow-gotcha-2026-05-15.md` | r5.04 wrapper sed; **fmha AM `-timer=cpu` mandatory** (else `-timer=gpu` short-circuits validation via `fmha_fwd_runner.hpp:1648-1652` no_instance early return) |
| qa | `qa-verify-heuristics-2026-05-15.md` | ABC case sensitivity table for V/K/Q-TDM bugs (case A first signal, case C 1e+08 = V/softmax overflow signature); kname grep mandatory pre-numerics |
| reviewer | `reviewer-audit-heuristics-2026-05-15.md` | 3 audit heuristics: narrative-vs-diff inconsistency / by-construction HOLDS-FAILS / use author's own source for self-correction collab |

## New memories this session

| Memory | Purpose |
|---|---|
| `feedback_teammate_no_memory_write.md` | Dispatch teammate doc must explicit project dir, NOT memory; teammate write memory triggers permission gate |
| `reference_teammate_permission_request_handling.md` | Diagnosis (tmux capture-pane fallback) + salvage (jq inbox payload) + unblock (kill+spawn) for stuck teammate; SendMessage has no `permission_response` |
| `feedback_pre_restart_consolidation.md` | Pre-restart consolidation template; prompt must be 100% explicit on path, no choice space |

## Setup state verify checklist (post-compact)

```sh
# 1. Confirm worktree state unchanged
cd /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel
git log --oneline -3
git status --short  # expect 3 M files

# 2. Confirm team killed
ps aux | grep "agent-name" | grep -v grep  # expect empty

# 3. Confirm Docker container still up
docker ps | grep yiding12-gfx1250

# 4. Confirm new memories indexed
grep -c "2026-05-15\|teammate_no_memory\|teammate_permission_request\|pre_restart_consolidation" ~/.claude/memory/MEMORY.md  # expect 3
```

## Standby state

Lead has nothing actionable until user picks V-TDM direction (a/b/c/d) or directs new work. NOT push, NOT PR per user explicit. NOT spawn replacement teammates per user explicit.
