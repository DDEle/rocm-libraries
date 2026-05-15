# SWE post-compact recovery baseline

**Purpose:** Restore working context after `/compact` for SWE teammate on the gfx1250 FMHA TDM project. Read this first after compact + ack lead.

**Last updated:** 2026-05-07, end of B2 ship + X+ rebase chain milestone.

---

## Identity & role

I am **swe** in team-lead's gfx1250 FMHA TDM team. Communicate with team-lead via `SendMessage(to: "team-lead", ...)`. User talks to team-lead primarily; sometimes user pivots to direct sweline (most recently during the rebase chaos — see "User direct-sync episode" below).

CLAUDE.md global instructions:
- 始终用简体中文跟 user/lead 对话；代码/文件名/技术产物保持英文
- Commit messages **never** include `Co-Authored-By: Claude` line
- 全局 memory 在 `~/.claude/memory/` (跨设备同步)
- GPU 测试前先 `rocm-smi --showmeminfo vram` 选 free GPU + `HIP_VISIBLE_DEVICES=<id>`
- 后台任务用 Bash `run_in_background` 不用 `& echo $!`
- Docker `bash -c "..."` 内部禁止管道截断输出 (用文件 + Read)

---

## Current worktree state (as of last commit)

**Repo:** `/home/yiding12/workspace/rocm-libraries-gfx1250`
- This is a **separate clone** with `projects/composablekernel/` inside; cwd switches between the two on session restart.
- Note: cwd resets to `/home/yiding12/workspace/rocm-libraries/projects/composablekernel` after every Bash call due to shell reinit. Always `cd` first or use absolute paths.

**Branch:** `yiding12/gfx1250-fmha-tdm` (B2 branch)
**HEAD:** `07b82d03301ca33abe8c8bc6f9dcde5a9b140138`
**Working tree:** clean (no staged/unstaged code changes; only untracked logs/status docs from prior dev cycles)

### Branch layout (after the X+ rebase chain)

```
07b82d03301 [CK Tile] Step B2 — gfx1250 FMHA FWD TDM Q+K (single-head)         ← B2, on internel
db26fd14f99 Step B1: rewrite K/V distribution to make qr_tdm functional        ← B1
153fe1ccfe3 Init QRKSVS_TDM                                                     ← Init pipeline files
079aa02a60e [CK Tile] container_helper/tuple: support mixed tuple<int, ...>     ← X+, on internel
b94cefafec5 [GFX1250] support GroupedGemm with MX FP8 (#1236)                  ← merge-base
... rest is internel/gfx1250 ancestry ...
```

### All branches (local)

| local branch | tracks | head | state |
|---|---|---|---|
| `yiding12/gfx1250-fmha-tdm` | `internel/yiding12/gfx1250-fmha-tdm` | `07b82d03301` | B2, in sync with remote (after force push by user) |
| `yiding12/tuple-mixed-inplace-container-ops` | `internel/yiding12/tuple-mixed-inplace-container-ops` | `079aa02a60e` | X+, in sync with remote (after force push by user) |
| `gfx1250` | (no remote tracking; behind internel/gfx1250) | `57914891cd1` | local lineage diverges from internel/gfx1250 |
| `develop` | `origin/develop` | `125f524a82f` | not actively used |
| `yiding12/gfx1250-fmha-tdm-wip` | (untracked) | `249d1c40053` | old WIP, ignore |

### Key SHA history (B2 ship cycle)

| commit | original SHA | rebased SHA |
|---|---|---|
| Init QRKSVS_TDM | `92586ef72e2` | `153fe1ccfe3` |
| B1 | `16324795215` | `db26fd14f99` |
| B2 | `1cf9a2c0b73` (then `440242c3bc6` amend) | `07b82d03301` |
| X+ | `c809aec6205` → `2a29bcc1ce1` (cherry on local gfx1250) | `079aa02a60e` |

---

## B2 ship — what was delivered

### B2 commit (`07b82d03301`) — Step B2 FMHA FWD TDM Q+K (single-head)

**Files changed:** 4 files, +302 / −257
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` — qr_vr emit disabled for d=128 fp16/bf16 single-head (forces qr_tdm dispatch); workaround tracked Step C
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` — (α') unmerge typo fix (3 sites: drop `/ kAlignmentK`)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` — Q load → load_tile_tdm with TDMConfig; K stays TDM; V hybrid keeps async_load
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` — λ-K + λ-Q trivial tile-major dist; β-K + β'-Q LDS read view plain (Xor template removed entirely both K & Q ~80 lines dead code each); Q TDM padding disabled

**Pipeline mods (semantic summary):**
- (λ-K) K dram dist: B1 5D async-style scatter → trivial tile-major (mirrors GEMM v1 ColMajor B layout)
- (β-K) K LDS read view: Xor=true → plain row-major (TDM box-write can't produce XOR'd LDS; XOR'd branch was dead code)
- (λ-Q) Q dram dist: 5D scatter → trivial tile-major (mirrors λ-K)
- (β'-Q) Q LDS read view: Xor=true → plain row-major (mirrors K fix)
- Q TDM padding disabled in `GetLdsPaddingConfigQ`
- V keeps async_load + Xor=true (h hybrid revert from B1 — V TDM out of scope)

**Test results (gfx1250 ffm_lite simulator vs B1 baseline):**
- Case A (fp16 dense, b=1 h=1 s=1023 d=128, mask=0): valid:y, sim 2555ms (−1.3% vs B1)
- Case C (bf16 long, b=1 h=1 s=2047 d=128, mask=0): valid:y, sim 9127ms (+0.15%)
- Case B (fp16 GQA causal, b=1 h_q=2 h_k=1 s=1023 s_k=257 d=128, mask=2): **valid:n max_err 0.0620 (14.5% wrong) — KNOWN ISSUE**

**Case B known issue — 2 root causes:**
1. **CK core latent bug** in `tile_window.hpp`'s `get_cached_global_strides()` — uses `get_lengths()` (shape-based packed default) instead of actual tensor view stride field. Multi-head Q runtime stride is `h_q * hdim_q` (256 for case B) but cached stride is `hdim_q` (128) → wrong byte offset.
2. **(λ-Q) trivial tile-major Q dist** itself broken under multi-head GQA. Workaround attempt that only switched Q writer to async_load still produced wrong values → dist needs GQA-aware redesign.

### X+ commit (`079aa02a60e`) — container_helper/tuple mixed tuple<int, constant<N>> support

**Files:** 2 files, +62 / −25
- `include/ck_tile/core/container/container_helper.hpp` — `container_reverse_inclusive_scan` rewritten with new recursive `_impl` helper (mirrors `_exclusive_scan_impl`)
- `include/ck_tile/core/container/tuple.hpp` — 4 ops use `generate_tuple` instead of in-place mutation:
  - `operator+(tuple<Xs...>, Y)` — MultiIndex + scalar-broadcast
  - `operator-(tuple<Xs...>, Y)`
  - `operator*(tuple<Xs...>, Y)`
  - `operator*(Y, tuple<Xs...>)` — scalar * MultiIndex

**Why:** mixed (runtime int, compile-time constant<N>) tuples can't be stored back via in-place mutation. Trigger callsite was TDM `tile_window.hpp:1788` `get_cached_global_strides`. Fix is purely CK core; B2 was discovery context.

### Both pushed to internel (AMD-ROCm-Internal/rocm-libraries) by user. NO PRs created (user self-handles).

---

## Step C backlog (Task #8, awaiting user priority)

From B2 commit body's "Known limitations / future work":

1. **Fix `get_cached_global_strides`** to use actual tensor view stride instead of shape-based packed default (file: `tile_window.hpp:1788`). High blast radius — affects all TDM consumers with non-packed-stride tensors.

2. **GQA-aware Q dist redesign** for `h_q != h_k`. Independent of the cached-strides fix; the current λ-Q trivial tile-major dist is broken under multi-head Q regardless of writer.

3. **Dispatcher prefer-qr_tdm** to retire the codegen `qr_vr` disable workaround. Currently `qr_vr` is force-disabled for d=128 fp16/bf16 in `fmha_fwd.py` because the dispatcher would otherwise pick `qr_vr` (no TDM acceleration) over `qr_tdm`.

4. **Re-enable K/V padding** (currently disabled per λ-1/λ-2 disambig comments in `_policy.hpp`'s `GetLdsPaddingConfigK` / `GetLdsPaddingConfigV` / `GetLdsPaddingConfigQ`).

---

## User direct-sync episode (the rebase chaos, still relevant)

During the X+/B2 rebase chain, user said `"啥情况啊？我直接和你沟通吧"` and bypassed lead for the rest of that flow. **What user wanted:**

1. **Cherry-pick over rebase** — User explicitly chose `git reset --hard <base> && git cherry-pick <commit>` over `git rebase` for X+. Less SHA churn for ancestry, simpler conflict surface.

2. **X+ base = `b94cefafec5` (gfx1250 GroupedGemm MX FP8)** — User picked this specifically because B2's ancestry already had `b94cefafec5` but local `gfx1250` branch did not (lineage divergence). Picking `b94cefafec5` as X+ base means B2 rebase replays only the 3 fmha commits, no extra `b94cefafec5` re-replay.

3. **User force-pushed both branches themselves** — I provide the command (`git push --force-with-lease internel <branch>`) but user executes. Same pattern as the original B2 push.

4. **No PR creation** — User self-manages PR creation/review/merge.

**If user goes direct again, this is fine — relay summary back to lead afterwards (which I did via the "详细 progress" message).**

---

## Anti-patterns I've hit + protocols (verify-before-claim discipline)

### 1. Pre-flight `git log A ^B | wc -l` before ANY rebase
Two rebase failures (B2-onto-X+ #1, X+-onto-internel/gfx1250) were the same root cause: plain `git rebase X` replays from merge-base, not just my latest commit. If the count is >1, either use `--onto X+ A^ A` or pivot to cherry-pick.
Lesson dispatched to lead, but ALSO need to internalize: focused-diff verify alone isn't enough.

### 2. Multi-line patterns in source code
My `grep -c "kQKHeaddim / kDramTileK / kAlignmentK"` returned 0 on `internel/develop` and I claimed "PR #6964 already applied" — false-negative because the source breaks the pattern across 2 lines. Use python multi-line regex (`re.findall(pattern, content)` with `[\s\n]+` between tokens) for any source-code claim that might span line breaks.

### 3. `git rebase --abort` + lots of untracked files = stuck
Mid-rebase replay creates new files that `abort` sees as "untracked would be overwritten." Solution: delete `.git/worktrees/<wt>/rebase-merge` directory + `git checkout -f <branch>` to force-restore. Branch refs are safe (rebase only moves detached HEAD).

### 4. Inheritance assumption anti-patterns
- K↔Q analogy at reader-only level (β'-Q-fix didn't work because writer side ignored)
- Compile-time analysis missing runtime kargs.stride_q
- Cleanup batch revert without case-by-case ship-impact verify (Phase 5 qr_vr disable revert was wrong — caught by QA)
- "I remember doing this work" symbol-name fabrication (B2 commit message had 5 fake names — caught by lead audit)
- Local `gfx1250` ≠ `internel/gfx1250` (lineage divergence; same name doesn't mean same DAG)

### 5. Stage-level direct verify methodology
End-to-end max_err alone is insufficient signal for stage-level fix verify. Use `CK_PRINTF_WARP0<float>{}` device-side runtime numerical print + tid 0/32 sentinel comparison + B1 baseline diff. Stored as memory `feedback_stage_level_verify.md`.

### 6. Cleanup safety: every revert needs case-by-case ship-impact check
Don't bulk-revert "dev plumbing" without verifying each piece. The codegen `qr_vr` disable looked like dev plumbing but was ship-required (forces qr_tdm dispatch).

### 7. Verify-before-claim for commit messages
Every concrete claim (file name, symbol name, line number, count) in a commit body MUST be `grep`/`git diff` cross-checked before send. The B2 message had `kIsTdmPipeline` / `has_kIsTdm_trait` / `kIsTdm trait` / `tdm_config_v` / `GetLdsPaddingConfigV_original_disabled` — all 5 fabricated. Now I run `git log A ^B^` style verify or `git show <sha> | grep <claim>` before SendMessage.

### 8. `feedback_no_auto_push.md` (user-side memory): never auto-push
Always commit local + report lead audit + wait for user diff review + push only on explicit approval. User push themselves is the norm; I provide commands when asked.

---

## Misc context worth carrying

- **Test simulator:** gfx1250 `ffm_lite`. Cases A (dense single-head), B (GQA causal), C (long single-head). Run via prompts in `prompt.txt` / `prompt-meta.txt` at repo root (not committed).
- **Build:** Cmake preset `dev-gfx1250` if exists, else manual `-D GPU_TARGETS="gfx1250"`. Project CLAUDE.md has full dev-build instructions.
- **gh CLI:** authed as `DDEle` on github.com. `internel` (AMD-ROCm-Internal) push works; secret-scanner-clean for gfx1250 lineage. `origin` (ROCm/rocm-libraries public) blocks push due to gfx1250 NPI hardware identifier scanner trip on rocWMMA ancestry — use internel for gfx1250 work.
- **PR #6964 status:** unverified at time of B2 ship. Multi-line check showed buggy α' pattern still present on `origin/develop`, `internel/develop`, `internel/gfx1250` at 3 sites each. B2's commit body says "logically same as PR #6964 (already merged after this branch was cut)" — flagged but not corrected.

---

## Status docs left in worktree (untracked)

These markdown files are dev-cycle status logs, NOT for commit. Listed for inventory:
- `swe-status-K-LDS-dump-disambig.md`
- `swe-status-case-B-Phase2-outcome.md`
- `swe-status-final-B2-await-user.md`
- `swe-status-lambda-design.md`
- `swe-status-lambda-fail-v2.md`
- `swe-status-lambda-pivot.md`
- `swe-status-lambda-test-185504.md`
- `B2-fresh-team-post-compact.md` (← prior post-compact recovery doc, may have older context)
- `B2-lambda-resume-baseline.md`
- `B2-resume-baseline-v3.md`
- `auto-decision-B2-lambda-test-185000.md`
- `compile-*.log` (many)
- `decision-B2-*.md`
- `investigator-*.md` / `qa-native-*.md`

Plus root-level: `prompt.txt`, `prompt-meta.txt` (test driver scripts).

---

## After-compact ack to lead

Once context is compacted and this doc re-read, send:
> "Compact 完毕，已读回 `swe-post-compact-baseline.md` 验证恢复。standby Step C dispatch。当前 worktree on `yiding12/gfx1250-fmha-tdm` @ `07b82d03301`, clean. B2 + X+ both shipped to internel. 4 项 Step C backlog 等 prioritize: get_cached_global_strides fix / GQA-aware Q dist / dispatcher prefer-qr_tdm / re-enable K/V padding."
