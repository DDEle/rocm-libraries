# Step C #1 attempt — blocked, post-session recovery baseline (lead)

写作时间: 2026-05-08 (UTC) — user 关 session + teammates 前最后 state snapshot。
目的: resume 后 read 这一份恢复 active state，picking up Step C #1 attempt + AM probe + 3 个 blocking decision threads。

**前置 baseline**: `B2-ship-complete-post-compact.md` (B2 ship 状态)。这一份是它之后的 Step C #1 attempt 进展。

---

## 当前 active state

- **方向**: Step C #1 (CK core fix `get_cached_global_strides`) 已 attempt 但 fmha A regression QA gate FAIL，commit on local branch 顶部 NOT pushed
- **3 个 thread 全 blocked 等 user 决策**:
  1. fmha A regression path 1/2/3/4
  2. Concern 1 commit body amend (entangled with #1)
  3. AM sample blocked: docker login 没真正写到 user `~/.docker/config.json`
- **Worktree**: clean (untracked 全是 logs/docs)

---

## ✅ Shipped this session

### Branches force-pushed to internel by user
- `yiding12/tuple-mixed-inplace-container-ops` (X+): `e8b75349836` (rebased onto latest internel/gfx1250 `62d40f9730c`)
- `yiding12/gfx1250-fmha-tdm` (B2): `2c46402fba3` (rebased onto new X+)

### Verified post-rebase (Task #12)
| case | valid | max_err | sim_ms | vs pre-rebase baseline |
|---|---|---|---|---|
| A | y | 0 | 2555 | -2.0% |
| B | n | 0.0620 | 366 | -0.8% (known issue 不退化) |
| C | y | 0 | 9226 | -1.5% |

Kname 全 `qr_tdm_vr_npad`，#1360 amd_buffer_addressing_builtin risk 实测 OK。

---

## 🟡 Step C #1 attempt — commit on local NOT pushed

### Current SHA chain
```
37948373fc8  [CK Tile] Fix tile_window get_cached_global_strides...   ← #1 fix attempt
2c46402fba3  [CK Tile] Step B2 — gfx1250 FMHA FWD TDM Q+K (single-head)
27e44f0c9b6  Step B1
1ddb7b44cd8  Init QRKSVS_TDM
e8b75349836  X+
62d40f9730c  internel/gfx1250 ancestor
```

`internel/yiding12/gfx1250-fmha-tdm` 仍是 `2c46402fba3` (B2)。`37948373fc8` ahead by 1 not pushed。

### Fix design (Task #15)
```cpp
// tile_window.hpp +24/-5 (single function get_cached_global_strides)
cached_global_strides_ = generate_array(
    [&](auto i) {
        auto unit_vec = make_zero_multi_index<NDimBottomTensor>();
        unit_vec(i)   = 1;
        const index_t s = glb_tensor_descriptor.calculate_offset(unit_vec);
        return max(s / Traits::PackedSize, index_t{1});
    },
    number<NDimBottomTensor>{});
```

每 top-dim unit-vector query `calculate_offset` → 走 transform chain 拿 actual stride。Clean fix (no fast-path) per user "做对优先". Single PR self-contained。

### 3 hardening 完成情况 (per user 决策 A/B/D 做, C skip)
- **A** (5min mx_gemm pre-existence on internel/gfx1250 净 head): 12 FAIL identity = B2 baseline empty diff → confirmed pre-existing ✓
- **B** (audit list 加 fmha 6th consumer): 完成，标 "actual triggering consumer" stride_q = h_q * hdim_q under GQA non-packed ✓
- **C SKIP** (1-2h transform_tensor_view stride preservation trace): user 拍赌一把，不做 ← **risk 这次踩了**
- **D** (FAIL identity set diff 5 GEMM binaries): 全 empty diff，无 silent regression ✓

---

## 🔴 BLOCKER 1 — fmha A regression QA gate FAIL (待 user 拍 path)

### QA finding (case A/C verify on `37948373fc8`)
| case | valid | max_err | err 数/% | sim_ms | 判定 |
|---|---|---|---|---|---|
| A | **n** ⚠️ | **0.007568** | 45731 / **34.92% wrong** | 2562 | **GATE FAIL** (was y/0) |
| B | n | 0.05737 | 48851 / 18.65% | 382 | pattern shift (err 数 +29% from 37965, perf +4.4%) |
| C | y | 0 | — | 9142 | PASS |

- Kname 全 `qr_tdm_vr_npad` ✓ (不是 dispatch fallback)
- A error magnitude ~2-2.5e-3 (fp16 ULP 数倍)，first errors at out[13/17/23] = early-tile not edge corner
- 误差 pattern = transform-chain 顺序错位 type，不是大爆炸 → **跟 reviewer Concern 2 (α') unmerge→xor→merge_v3 长链风险吻合**
- C bf16 long seq dense 路径无害

**C-skip risk catch mechanism 触发** — fmha (α') 长 transform 链对 #1 fix `calculate_offset(unit_vec)` 改动敏感，A/B 都受影响。Hardening D 全 GEMM 系给了 false confidence — 没 cover fmha 那条最复杂 transform 链。

### 4 候选 path 待 user 拍
1. **激活 Concern C (1-2h)** — SWE/mentor 做 `transform_tensor_view` stride preservation trace, 找 fmha 哪个 transform stage 让 stride field 跟 shape decouple, 再调 #1 fix
2. **调整 #1 fix design** — 不用 `calculate_offset(unit_vec)` (走全 transform chain), 找别的 stride-extraction 方式只拿 base descriptor stride 不 traverse transforms
3. **Revert #1 fix** — back to B2 (`2c46402fba3`), 重新 evaluate Step C #1 approach 整体（可能跟 #2 必须一起做不能单 land）
4. **Hybrid**: keep #1 fix 但加 fast-path detect transform-chain 复杂度, 复杂时 fallback 到 shape-default (违反 user "clean fix no fast-path" 决策但救 fmha)

按 memory `project_gfx1250_fmha_no_shuffle.md` 教训: strategic-level shift 立即报 user 不自决。

---

## 🔴 BLOCKER 2 — Concern 1 commit body amend (entangled with #1)

### Reviewer audit on `37948373fc8` Concern 1 (HIGH)
SWE commit body 写 "+13% sim_ms 是 first-access cost, post-amortize 命中 cache" — reviewer challenge 不 hold 因 sim_ms 是 multi-run steady-state。SWE 后跑 3-run empirical: 508.9 / 511.8 / 512.1 ms vs pre-fix 513 ms = **<1% diff，比 3ms variance 还小**。582ms 是 single-shot noise outlier。

### User pivot (per reviewer relay)
**ffm_lite sim_ms 不作 perf 数** (functional sanity 副产物)，AM 才是 cycle-accurate perf eval 工具。

→ Reviewer 提 4 amend options:
- A: 整段 perf 段删掉 (correctness only)
- B: 加 caveat "ffm_lite 仅 functional"
- C: 用 ffm_lite 3-run median 但加 caveat
- D: 跑 AM 拿真 perf 数据 (depends BLOCKER 3)

User 决策 "**先让 SWE 跑 AM sample 估时间，再决定是否 D**" → BLOCKER 3 在那。

### Entanglement
- 如果 fmha A path 选 1/2/3/4 中 fix 调整 → commit `37948373fc8` 整个废，amend 浪费
- 如果选 push as-is + flag fmha A as known issue → amend 有意义
- 如果 revert → 整个 commit 不存在，不需要 amend
- → **path 拍后才能 amend**

### Reviewer Concern 2 + 3 (audit 完整 5 finding，2 hold)
- Concern 2 (HIGH): 5 D-diff 全 GEMM 系，**没 cover fmha** (α') 长 transform 链 — **已经触发**，QA gate FAIL 就是这条
- Concern 3 (LOW): tdm_epilogue.hpp audit completeness — SWE confirm 是 relay 不是独立 caller, indirect cover via #2-5 OK ✓
- Concern 4 + 5: caller stride field set correctness audit + commit body 完整性 — 都 OK

---

## 🔴 BLOCKER 3 — AM sample blocked (docker login)

### npibox setup 进度 (per SWE Task #15 follow-up)
`npibox` = official AMD-ROCm-Internal CLI tool 跑 FFM/AM/CSIM/JITCU containers (per Confluence `npibox - FFM/AM Model Container System` id 1553380015, v1.3.0 2026-04-23)。

**已 done (持久)**:
- `npibox` v1.3.0 installed at `~/.local/bin/npibox` (via SSH clone repo `AMD-ROCm-Internal/libs-npi-dev-tools` + deploy script，因为 gh CLI scope 看不到 private org)
- Harbor CA cert auto-installed (npibox check + sudo install + update-ca-certificates)
- sudo passwordless ✓
- Docker daemon access ✓

**Stuck blocker**:
- `npibox check_docker_login` (line 597-601) 无条件 `docker login registry-sc-harbor.amd.com` precheck
- Lead verified 2026-05-07 12:22 UTC: user `~/.docker/config.json` 仍**只有 `index.docker.io`** entries (mtime 10:59 UTC)，**没 sc-harbor entry**
- User 之前说 "我login了" 但 config 没体现 (可能 cancel / 别的 path / 别的 host)
- root `/root/.docker/config.json` 也没 sc-harbor (有 `compute-artifactory.amd.com:5000` + docker.io)

### User 需要做
```sh
docker login registry-sc-harbor.amd.com
# Username: yiding12 (NTID)
# Password: NTID password
```

Verify:
```sh
python3 -c "import json; print(list(json.load(open('/home/yiding12/.docker/config.json'))['auths'].keys()))"
# 应该看到 'registry-sc-harbor.amd.com' 在 list 里
```

成功后通知 SWE retry: `sg docker -c "~/.local/bin/npibox docker mi450 am -- <abs-path-binary> -m=128 -n=128 -k=128 -prec=fp16 -warmup=0 -repeat=1"`

### Fallback (如果 npibox AM 仍 fail)
`~/workspace/rocdtif/AM+FFM-LITE/` 含 24 个 rocdtif tarball，5.x series 最新 r5.03 (2026-05-06) / r5.04 (2026-04-30) / r5.02 / r5.01 (current 用) / r4.05 / r4.04。可手 cobble Setup 2 (docker exec) 试别的 version 是否 fix IOMMU/ATPT。

### 之前手 cobble Setup 失败记录 (per SWE 12:12 UTC report)
- **Setup 1** (host source perftools env): AM model boot 13s OK 但 binary HIP fail "no ROCm-capable device" → RPATH/LD_LIBRARY_PATH 不通，放弃
- **Setup 2** (sudo docker exec yiding12-gfx1250 + source env + run wrapper): AM model 真 sim 起来，2 dispatches 启动后 crash at sim time 139220ns: `tb.iommu_mem_access ERROR: ATPT GetHostPageTable code=0x2` + SIGSEGV → IOMMU/ATPT page table 映射不上 binary HIP allocation pattern

---

## 📝 New memory written this session

- `~/.claude/memory/feedback_pre_existing_baseline.md` — Pre-existing 判断必须基于上游净 head baseline (anti-pattern A 复发警示，trigger SWE Task #13 mx_gemm "judgment not necessary")
- `~/.claude/memory/reference_simulator_ffm_lite_am.md` — ffm_lite vs AM 工具区分 + AM setup 待解 (host-only RPATH fail / docker exec IOMMU+ATPT crash + npibox 是 official path)
- MEMORY.md pointer updated 含两条

---

## 📋 Tasks state

- #5/#6/#7/#9/#10/#11/#12/#13: completed
- #8 pending: Step C / future backlog from B2 disambig (Step C #2/#3/#4 待做)
- #15 in_progress: Step C #1 — REOPENED 因 fmha A regression QA gate FAIL，commit `37948373fc8` 在 local branch 顶 NOT pushed，pending user path 决策

---

## 🔧 Worktree state

- Path: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel`
- Branch: `yiding12/gfx1250-fmha-tdm` @ `37948373fc8` (ahead 1 vs internel)
- Status: clean (untracked 全是 session logs + recovery docs，可忽略)

Other branches:
- `yiding12/tuple-mixed-inplace-container-ops` @ `e8b75349836` (X+, in sync with internel)
- `gfx1250` @ `57914891cd1` (behind internel/gfx1250 by 4 — 不是 active)
- `develop` @ `125f524a82f` (rocm-libraries main worktree)
- `yiding12/gfx1250-fmha-tdm-wip` @ `249d1c40053` (old wip, ignore)

---

## 🛠 Setup state (resume 后 verify)

| Item | Status | Resume 后 check |
|---|---|---|
| Container `yiding12-gfx1250` | running (per session) | `docker ps` |
| `~/.local/bin/npibox` v1.3.0 | installed | `~/.local/bin/npibox --help` |
| Harbor CA cert | installed | `ls /usr/local/share/ca-certificates/registry-sc-harbor.amd.crt` |
| `~/.docker/config.json` sc-harbor entry | **missing** (last verified 12:22 UTC) | `python3 -c "import json; print(list(json.load(open('/home/yiding12/.docker/config.json'))['auths'].keys()))"` |
| Docker group active in shell | unclear | `id | grep docker`, fallback `sg docker -c "..."` |

---

## 👥 Teammates resume protocol

Compact 完毕的 teammate recovery doc (上次 compact 时 baseline，**不含本 session Step C #1 attempt + fmha A regression + AM probe**):
- `swe-post-compact-baseline.md`
- `qa-post-compact-baseline.md`
- `mentor-post-compact-baseline.md`
- `reviewer-post-compact-baseline.md`

**Resume 后 lead 应做**:
1. Read 这一份 (`step-c1-attempt-blocked.md`) 全
2. TaskList() verify #15 状态
3. 根据 user 关闭前最后状态决定先动哪个 thread:
   - 如果 user fmha A path 拍了 → 重 spawn SWE/mentor/reviewer/qa (按需) + dispatch
   - 如果 docker login verify 后 → 重 spawn SWE retry npibox AM sample
   - 如果都没拍 → standby
4. 重 spawn teammate 时 prompt 必带 "Read your `*-post-compact-baseline.md` 然后 Read `step-c1-attempt-blocked.md` 接 active state"
5. Re-create cron job `fee9ca71` (7,37 * * * * 巡检) if 还要

---

## ⚠️ Pre-flight 注意 (resume 后)

1. Local commit `37948373fc8` 是 attempt fix，**不是 final** — fmha A regression 没修，不能 push
2. User 决策 "做对优先于做快" 仍生效 — 任何 amend / fix path 都按这个走
3. C-skip 决策已经踩，未来同 user 重审 Concern C 是否激活
4. ffm_lite sim_ms 不作 perf 数 — 任何 commit body / status doc 不能用 ffm_lite 写 perf claim (per memory `reference_simulator_ffm_lite_am.md`)
5. Anti-pattern A (first-principle 没 empirical) + C (co-design chain 类比太粗) 这次都踩，强化 verify 习惯

---

## User 已明确委托项 (不要再追)

- PR #6964 — user 自己合
- B2 + X+ + #15 commit body amend / push — user 自己决定 + push
- Develop 副发现 (114 fp16/bf16 group fail) — defer 给 fmha team

---

## 📚 Per-role reading priority (resume 后新 spawn teammate 用)

每个 role spawn 后的 bootstrap 顺序 (一份 prompt + read 这份 doc 即可，不用让 lead 单独 brief)。

### SWE
1. Read `swe-post-compact-baseline.md` (个人角色 + 8 anti-pattern + protocol)
2. Read 本 doc `step-c1-attempt-blocked.md`：
   - **必读**: Section "Step C #1 attempt — commit on local NOT pushed" (fix code + 3 hardening 完成情况)
   - **必读**: Section "BLOCKER 1 fmha A regression" (你写的 fix 触发了 transform-chain corner case)
   - **必读**: Section "BLOCKER 3 AM sample blocked" (你之前手 cobble Setup 1/2 fail + npibox install 状态 + docker login blocker + fallback rocdtif tarball list)
   - **必读**: Section "Setup state" (container/npibox/cert/docker auth verify checklist)
3. 内化两条新 memory:
   - `~/.claude/memory/feedback_pre_existing_baseline.md` (你 mx_gemm "judgment not necessary" 是反例)
   - `~/.claude/memory/reference_simulator_ffm_lite_am.md` (ffm_lite 不作 perf, AM 才作)

### QA
1. Read `qa-post-compact-baseline.md` (env + ABC 跑法 + kname check 规则)
2. Read 本 doc `step-c1-attempt-blocked.md`：
   - **必读**: Section "BLOCKER 1 fmha A regression" 完整 ABC numbers (case A valid:n max_err 0.007568 34.92% wrong, case B err 数 +29% pattern shift, case C 安然)
   - **必读**: Section "Verified post-rebase (Task #12)" baseline (你之前跑过的 numbers，作为 #15 比对基准)
   - **必读**: Section "Setup state" (Container running 状态 verify)
3. 工作流提醒 (per `qa-post-compact-baseline.md` §6): kname grep 必走，防 cleanup-v1 false-positive 重演

### Mentor
1. Read `mentor-post-compact-baseline.md` (hardware fact baseline + 5 anti-pattern + 6 protocol rules + ds_load_tr semantics + case B 两 root cause)
2. Read 本 doc `step-c1-attempt-blocked.md`：
   - **必读**: Section "Step C #1 attempt" fix design (`calculate_offset(unit_vec)` 走 transform chain 拿 stride)
   - **必读**: Section "BLOCKER 1 fmha A regression" — 你 prep work 那个 GQA-aware Q dist redesign 现在跟 #1 fix 互动出新问题
   - **必读**: Section "Pre-flight 注意" — Anti-pattern C (co-design chain 类比太粗) 这次再次踩，强化
3. Step C #1 path 决定后预期跟 SWE collab debug `transform_tensor_view` chain 哪段让 stride field 跟 shape decouple

### Reviewer
1. Read `reviewer-post-compact-baseline.md` (devil's advocate + 5 contribution pattern + 跟 lead ad-hoc protocol + effort=max)
2. Read 本 doc `step-c1-attempt-blocked.md`：
   - **必读**: Section "BLOCKER 2 Concern 1 amend" (你 audit `37948373fc8` 5 finding 全摘要 + user 跟你 1:1 拍的 4 amend options + ffm_lite/AM 决策)
   - **必读**: Section "BLOCKER 1 fmha A regression" — 你 Concern 2 catch mechanism **触发了** (D-diff 5 GEMM 没 cover fmha 那条)
   - **必读**: Section "New memory written" (`reference_simulator_ffm_lite_am.md` 含你跟 user 拍的工具区分规则 + AM setup 待解状态)
3. **Effort=max 设定要 user 重 set** (resume 不持久) — `/effort max`
4. 内化教训: anti-pattern C (co-design chain 类比太粗 — 5 GEMM analogy isomorphic to fmha) 你这次 audit 漏掉, 后续 catch

### Lead (self, post-compact reload)
1. Read `B2-ship-complete-post-compact.md` (B2 ship baseline 历史)
2. Read 本 doc `step-c1-attempt-blocked.md` 全部
3. TaskList() verify #15 status
4. 决定 first action 看 user 关闭前最后状态:
   - User 在 close 前是否拍了 fmha A path / amend / 其它 → 优先按 user direction
   - 如果都没拍 → standby + cron patrol
