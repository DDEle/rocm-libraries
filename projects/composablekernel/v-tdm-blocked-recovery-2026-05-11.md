# V-TDM strategic shift blocked — pre-compact recovery baseline (lead, 2026-05-11)

写作时间: 2026-05-11 01:40 UTC — user 打算 compact context, 写最后 state snapshot 给 resume 用。

**前置 baseline 链** (按顺序 read 才完整):
1. `B2-ship-complete-post-compact.md` (B2 ship 状态, 上上次 compact 时)
2. `step-c1-attempt-blocked.md` (Step C #1 attempt blocked, 上次 compact 时, **AM 部分已过时**)
3. **本份** (V-TDM 9-piece implement fail + mentor verdict V-TDM 不可行 + user AFK 73h+)

---

## 当前 active state (1 句话)

**V-TDM strategic-level blocked**: SWE 9-piece implement build OK 但 ABC ALL FAIL (insane numerics), mentor design re-verify 自校正 (anti-pattern A 第 4 次复发) verdict "V→TDM 在 ds_load_tr reader 下 fundamentally infeasible, 推荐 revert 9-piece + 恢复 (h) hybrid"; user AFK 73h+ 没回, lead 不自决 strategic shift, worktree hold uncommitted。

---

## ✅ Step C #1 已 ship (force-pushed, 历史已确定)

`internel/yiding12/gfx1250-fmha-tdm` HEAD = **`08b61885fc9`** (4 commits cleanly layered):
```
e8b75349836  X+ container_helper/tuple                                        ← X+ base, internel/yiding12/tuple-mixed-inplace-container-ops
dbcea05a8a6  [CK Tile] Fix tile_window get_cached_global_strides              ← #15 framework fix (foundational)
ac982130db8  Init QRKSVS_TDM
556abb2b01a  Step B1: rewrite K/V distribution to make qr_tdm functional...
08b61885fc9  [CK Tile] Step B2 — gfx1250 FMHA FWD TDM Q+K (B2 + path 7 squashed)  ← HEAD
```

ABC ffm_lite 全 valid:y, multi-stride GQA 3/3 valid:y, AM s=128 valid:y, non-TDM 0 regression empirical。

**Backup ref**: `refs/backup/pre-rebase-3a9b17a8537` 保留 (rollback 用)。

---

## 🔴 V-TDM blocked state (worktree dirty, 等 user 拍)

### Worktree state
- HEAD `08b61885fc9` (unchanged)
- **Uncommitted V-TDM edits**: 3 files, +88/-39
  - `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` +14 (path 7 dispatch extend to V)
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` +43-? (writer/reader/tdm_config_v)
  - `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` +70-? (V dram dist, LDS read desc, stale comment)
- B2 baseline binary `build-gfx1250-b2/bin/tile_example_fmha_fwd` preserved (untouched)
- `build-gfx1250/bin/tile_example_fmha_fwd` 是 V-TDM broken binary (build OK, ABC fail)

### Implement empirical (post 9-piece edit + build success)
| case | shape | result |
|---|---|---|
| A (s=1023 fp16 dense) | b=1 h=1 d=128 mask=0 | **valid:n max_err 6.081, 100% wrong** |
| B (s=1023 fp16 GQA causal) | b=1 h=2 h_k=1 s_k=257 mask=2 | valid:n max_err 6.319, 24.99% wrong |
| C (s=2047 bf16 dense) | b=1 h=1 d=128 mask=0 | **valid:n max_err 1.09e+08, 100% wrong** ← V端 garbage 致 post-softmax accumulator overflow |

All 3 dispatch correct kname `qr_tdm_vr_npad...` ✓ — 不是 fallback, broken kernel 真在跑。

### Mentor design re-verify verdict (anti-pattern A 第 4 次复发自校正)

**`mentor-reanalysis-ds-load-tr-correction.md` claim INVALIDATED**:
> 之前: "ds_load_tr_b128 是 byte-position 读, in-shader transpose 是 register-side rearrangement, **independent of how the writer scattered its bytes**"
> 实证: 完全反 — V LDS read view Xor=false + V dram trivial tile-major + V writer load_tile_tdm = 完全 garbage

**真 ds_load_tr_b128 contract** (grep + Read + WebSearch evidence):
- **Hardware-fixed lane→byte permutation** (NOT writer-independent)
- `load_tile_transpose.hpp:520-525` contract: out_tensor dist encoding 必须 = `OutputTileDistributionTraits<input_dist, DataType>::TransposedDstrEncode`
- LLVM PR #146024 + #146289: "lane→address mapping permuted so values land in lanes expected by WMMA B operand"
- B1 5D dist `MakeVDramTileDistribution` 是**反向工程出来 match 这个 hardware permutation** 的, async_load 用此 dist scatter dram bytes 到 LDS lane-期望位置 → ds_load_tr 读到正确 data
- TDM box-major plain row-major copy **做不到 lane-permuted scatter**

→ Predecessor mentor "V trload 物理不兼容" sign-off (后被 self-correct 为 wrong) **实际是对的** (conclusion 对, justification 不充分)

→ (h) hybrid (V async + ds_load_tr B1 path, K+Q TDM) = **current reader primitive 下 principled lower bound**, 不是 "退一步"。真 V-TDM 需要替换 ds_load_tr reader (long-term R&D, 超 Step C scope)。

### Mentor 推荐 path: revert 9-piece + 恢复 (h) hybrid (~2h)

1. SWE revert 9-piece:
   - V dram dist → 5D B1-style (TDM policy 511-537 unchanged from B1 ref)
   - V writer → `async_load_tile` (pipeline 383/896/921 unchanged)
   - V LDS read desc → Xor=true (TDM policy 302-378 unchanged)
   - V reg dist → BWarpDstrEncoding + TransposedDstrEncode (TDM policy 574-608 unchanged)
   - GetLdsPaddingConfigV: keep disabled (no change needed)
   - tdm_config_v: REMOVE
   - kernel.hpp `make_v_dram` path 7 dispatch: REMOVE
   - **Stale comment RESTORE** "V trload 物理不兼容" + reword 加 hardware-permutation rationale + LLVM PR 引用
2. Re-verify ABC post-revert: 应回 HEAD `08b61885fc9` baseline (case A/B/C valid:y)
3. Mentor update `mentor-reanalysis-ds-load-tr-correction.md` mark INVALIDATED + erratum

### Step C #2 reclassify (mentor 推荐)
- 当前 (h) hybrid 是 **lower bound feasibility under current reader primitive**, 不是 "incomplete"
- 真 V-TDM 需 **替换 ds_load_tr reader** (用 plain ds_load + V reg dist 全重设, 接受 wmma B operand thread layout 不能直接 match) — long-term R&D, 不属于 Step C
- 推荐 task #1 状态: "infeasible under current reader primitive, defer to long-term TDM-compatible WMMA B-operand reader R&D"

---

## 🟡 等 user 拍的 4 options (Teams notify URGENT 已发, user AFK 73h+ 没回)

(a) **revert all 回 HEAD `08b61885fc9`**, mentor design re-verify 已出 verdict, ack 后即 close
(b) **保留 worktree**, 让 SWE 做 LDS byte dump + piecewise revert 收集证据 (验 mentor verdict 是否漏 wrinkle)
(c) **直接 fallback (h) hybrid 不再 try V→TDM**, Step C #2 reclassify defer long-term (= mentor 推荐 (a)+(c) 组合)
(d) 别的

Lead 倾向 (a)+(c) 组合 (mentor verdict evidence 充分 + LLVM PR + load_tile_transpose contract + 实证 garbage signature 都 align)。但 strategic-level 不自决, 必须 user 拍。

---

## 📝 New memory written this session

1. `~/.claude/memory/feedback_pre_existing_baseline.md` — pre-existing 判断必须基于上游净 head baseline (5/7)
2. `~/.claude/memory/reference_simulator_ffm_lite_am.md` — 3 simulator 工具区分 ffm_lite/AM/csim, **AM 已通过 r5.04 + Setup 2 docker exec** (4.4min, `-rotating_count=1 -v=0` 必显), npibox 路径已废
3. `~/.claude/memory/feedback_teammate_report_subagent_teams.md` — substantive teammate update → sub-agent 整理 + Teams notify
4. `~/.claude/memory/feedback_no_silent_force_push.md` — force-push 必须 user explicit ack, 不能打包进 rebase/squash dispatch (背景: 5/8 silent force-push 误)
5. `~/.claude/memory/feedback_user_afk_proactive_work.md` — User AFK > **4 小时**才主动 dispatch 独立工作 (8 类 safe + 4 类 NOT safe)
6. `~/.claude/memory/feedback_no_ship_push_proactive.md` — User 是 ship/push/PR 节奏 owner, 不要把 ship 时机当主导框架反复提

**Mentor 推荐写但 user 拍 path 后再写** (避免 user 选 b 后被推翻):
- `feedback_self_correction_needs_empirical.md` — Self-correction 也是 reasoning, 必须配 empirical verify
- `reference_ds_load_tr_lane_mapping.md` — ds_load_tr 是 hardware-fixed lane→byte, B1 5D dist 是 reverse-engineered match

---

## 📋 Tasks state (snapshot)

**Open / pending**:
- #1 pending: Step C / future backlog (含 #3 Dispatcher prefer-qr_tdm + #4 K/V padding re-enable; #2 V-TDM 等 user 决定 reclassify defer)
- #13 in_progress: V-TDM design verdict (Step C #2) — 已出 verdict, 等 user 拍 path
- #20 pending: V-TDM verify pieces 3+5+6 unchanged (skipped, will discard if revert)
- #23 pending: V-TDM LDS dump byte-level verify (skipped, will discard if revert)
- #24 pending: V-TDM AM verify s=128 (skipped, will discard if revert)
- #25 pending: V-TDM D-style FAIL identity (skipped, will discard if revert)
- #26 pending: V-TDM commit (skipped, will discard if revert)

**Completed**:
- #2-#12: Path 7 + Audit 1-4 (Step C #1 ship)
- #14-#19: V-TDM piece 1/2/4/7/8/9 edits (worktree dirty)
- #21-#22: V-TDM build + ABC verify done (但 ABC actually FAIL — task subject 写 "all valid:y" 是预期不是结果)

---

## 🛠 Setup state (resume 后 verify)

| Item | Status | Verify command |
|---|---|---|
| Container `yiding12-gfx1250` | running 应该 (Up 9+ days now) | `docker ps` |
| `~/.local/bin/npibox` v1.3.0 | installed (但**已废弃路径**, 不再用) | `~/.local/bin/npibox --help` |
| Harbor CA cert | installed | `ls /usr/local/share/ca-certificates/registry-sc-harbor.amd.crt` |
| `~/.docker/config.json` sc-harbor entry | present (5/8 00:25) | `python3 -c "import json; print(list(json.load(open('/home/yiding12/.docker/config.json'))['auths'].keys()))"` |
| AM working setup | r5.04 wrapper extracted at `~/workspace/rocdtif/run-rocdtif-r5.04.sh`, recipe in `reference_simulator_ffm_lite_am.md` | 跑 AM 前先 verify wrapper 还在 |
| Docker group active | yes (compact 后看新 shell) | `id | grep docker` |

---

## 👥 Teammates state

Team config `~/.claude/teams/gfx1250-fmha-tdm/`:
- 4 active: team-lead / swe (blue) / qa (green) / mentor (yellow) / reviewer (purple, 重 spawn 后干净 name)
- Reviewer 之前 instance 死了一次 (5/8 09:50), kill pane + 清 config + re-spawn 干净 name=reviewer
- 现在所有 teammate 都 idle standby, 等 user direction

**Recovery doc 链 (compact 后给 teammate read)**:
- 各自 `*-post-compact-baseline.md` (compact 时 baseline)
- `step-c1-attempt-blocked.md` (Step C #1 attempt blocked, **AM 部分 stale**)
- 本份 (V-TDM blocked)

---

## ⏰ Cron state

`3965a1f1` (session-only, 7,37 \* \* \* \*, gfx1250-fmha-tdm 巡检, V-TDM trace+design phase prompt)
- 当前阶段 prompt 已**部分过时** (V-TDM design verdict 已出, ABC fail, mentor verdict 已出, 等 user 拍)
- Resume 后 lead 应该 update cron prompt 反映 current "V-TDM strategic shift blocked, hold worktree, 等 user 拍 (a)/(b)/(c)/(d)"

---

## 🚨 Pre-flight 注意 (resume 后)

1. **NOT push, NOT PR** (user 5/8 explicit, force-push 也不可)
2. **Worktree uncommitted V-TDM edits 是 broken state** — 不要试 build/ABC 期望 work, 那是 ABC fail 的 binary
3. Mentor 自校正 anti-pattern A 第 4 次复发 — future 任何 mentor self-correction 必须 empirical verify (LDS dump baseline + post-fix diff) 才信
4. ds_load_tr_b128 是 hardware-fixed lane permutation (LLVM PR #146024/#146289 evidence), B1 5D dist reverse-engineered match — 不要再被 "PDF 说 K-outer N-inner row-major 就够" 这种 reasoning 误导
5. 强烈建议 user 拍后第一件事: 清 worktree (revert or commit broken state to sandbox branch), 否则后续任何 fmha 改动都被 V-TDM dirty edits 干扰
6. AM working state 已稳定 (r5.04 + Setup 2 docker exec), 不再回 npibox

---

## User 已明确委托项 (不要再追)

- Step C #1 ship 完毕, 不开 PR (user 5/8 决定 "整 scope 后一次发")
- AM probe done, working recipe 在 memory
- 所有 push / PR 由 user 决定, lead/SWE 不主动 push

---

## 📚 Per-role reading priority (compact 后 fresh teammate spawn 用)

### Lead (self)
1. Read `B2-ship-complete-post-compact.md` (远古 baseline)
2. Read `step-c1-attempt-blocked.md` (Step C #1 attempt; AM 部分 stale 见本份 §)
3. Read **本份** (V-TDM blocked, 当前 active state)
4. TaskList() 看 #1 + #13 status
5. 看 user 之前最后 message — 5/8 ~10:30 UTC "我和reviewer 1:1 完了，后面你们主导，我先去干别的了" + 3 答 (urgency 不急 / 自决非不可逆 / NOT push NOT PR)
6. 决定 first action 看 user 是否拍了 (a)/(b)/(c)/(d) — 没拍则 standby

### SWE
1. Read `swe-post-compact-baseline.md`
2. Read `step-c1-attempt-blocked.md`
3. Read **本份** §Worktree state + §Implement empirical + §Mentor 推荐 path
4. **NOT 自动跑 build / ABC / revert** — worktree 是 broken state, 等 user/lead 拍后再动
5. 内化新 memory rule (no silent force-push, no ship/push proactive, AFK 4h, sub-agent + Teams notify, self-correction needs empirical 待 user 拍后写)

### QA
1. Read `qa-post-compact-baseline.md`
2. Read 本份 §Implement empirical (case A/B/C 当前 fail numbers)
3. Standby — 没派 task

### Mentor
1. Read `mentor-post-compact-baseline.md`
2. Read 本份 §Mentor design re-verify verdict (你的 self-correction 错了, anti-pattern A 第 4 次复发, 实证证据 + LLVM source)
3. **内化教训**: self-correction 也是 reasoning, 必须 empirical verify (LDS dump baseline + post-fix diff) 才信。Predecessor mentor "V trload 物理不兼容" sign-off 实际对的
4. Standby for revert dispatch (user 拍 (a)+(c) 后)

### Reviewer
1. Read `reviewer-post-compact-baseline.md`
2. Read 本份 §V-TDM blocked state + §Mentor verdict
3. Standby for user 1:1 (上次 instance 死了, 这次干净)
