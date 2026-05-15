# QA Post-Compact Baseline — gfx1250 FMHA TDM Step C

写于 2026-05-06 Step B2 cleanup-v2 ship state PASS 之后, Step C 启动前。
Compact 后 QA 读这一份能直接接活, 不需要重新读历史 status doc。

配套 memory:
- `~/.claude/memory/project_ck_dev_setup.md` (Docker / build)
- `~/.claude/memory/project_gfx1250_fmha_fwd_dev.md` (ffm_lite / ABC shape / 验证规则)
- `~/.claude/memory/project_gfx1250_fmha_tdm_v1.md` (B1 baseline + Step B1 教训)
- `~/.claude/memory/feedback_teammates_fmha_workflows.md` (teammate 协议)

也读 worktree 根 prior baseline 系列:
- `qa-env-notes.md` (上一代 QA 写的 env notes, 大部分仍 valid)
- `B2-resume-baseline-v3.md` (B2 disambig 完整 timeline + ds_load_tr semantics 修正)

---

## 1. 当前 Worktree State (Step B2 ship-ready)

**Ship state confirmed at 2026-05-06 cleanup-v2**:
- ABC 三 case 全走 `qr_tdm_vr_npad` (kname verified ✓)
- A: valid:y sim 2555 ms (-1.3% vs B1 2589) — single-head TDM full path
- B: valid:n max err 0.062 (14.50% wrong) sim 369 ms (-1.2% vs B1 374) — **GQA known issue, documented Step C backlog**
- C: valid:y sim 9127 ms (+0.15% vs B1 9113) — bf16 long seq, 几乎一致

5 个 modified files in B2 worktree (TDM ship + (X+) latent fix):
1. `include/ck_tile/core/container/container_helper.hpp` — (X+) CK core latent fix (independent value)
2. `include/ck_tile/core/container/tuple.hpp` — (X+) CK core latent fix
3. `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` — (α') unmerge typo fix (PR #6964 已 merge separately) + B2 dead code removed
4. `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` — λ-K + λ-Q + Q TDM call + (h hybrid V revert)
5. `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` — λ-K + β-K + λ-Q + β'-Q + Q/K/V padding disable + Q/K desc Xor template removed
+ codegen `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` 保留 qr_vr disable 注释 (ship-required, 不是 dev plumbing — Phase 5 cleanup-v1 误 revert 导致 dispatch fallback)

---

## 2. Docker 环境 (与 qa-env-notes.md 一致, 已 verified)

- **长期容器**: `yiding12-gfx1250` (不要 `--rm`, 不要 spawn 新容器)
- **exec 命令**: `sudo docker exec yiding12-gfx1250 bash -c "<cmd>"` — **必须 sudo** (user 不在 docker group)
- **挂载**: `/home/yiding12/workspace` 容器内外路径一致
- **镜像**: `rocm/composable_kernel-private:npi-mi450-latest`
- **B2 worktree** (主 ship branch): `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel`
- **B1 dump worktree** (disambig 用, 还在): `/home/yiding12/workspace/rocm-libraries-gfx1250-b1dump/projects/composablekernel`
- **Build dir**: `build-gfx1250/` (不要碰 `build/`)
- **并行度**: `-j204` (256 核 × 80%)
- **timeout**: build = 600000 (10 min); run = 180000 (3 min)

---

## 3. GPU 选择规则 (gfx1250 pre-silicon — 不需要 GPU)

gfx1250 是 pre-silicon, 用 rocdtif/ffm_lite 仿真跑, **不需要真 GPU**。

→ **跳过 `HIP_VISIBLE_DEVICES`** — 仿真器不 touch GPU。

(global memory 里 GPU 测试规则 `HIP_VISIBLE_DEVICES=<id>` 那条只对真硬件 benchmark 适用, 这里 ffm_lite 不 trigger)

---

## 4. 编译命令 (verified working)

### 全量 (codegen 改 / 第一次 / cmake config 改)
```bash
TS=$(date +%H%M%S); sudo docker exec yiding12-gfx1250 bash -c "
  cd /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel && \
  cmake -S . -B build-gfx1250 -GNinja --preset dev \
    -DGPU_TARGETS=gfx1250 -DFMHA_FWD_ENABLE_APIS='fwd' && \
  ninja -C build-gfx1250 -j204 tile_example_fmha_fwd
" > /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/compile-<phase>-$TS.log 2>&1
```

### Incremental (只改 .hpp / 几个 instance)
```bash
TS=$(date +%H%M%S); sudo docker exec yiding12-gfx1250 bash -c "
  cd /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel && \
  ninja -C build-gfx1250 -j204 tile_example_fmha_fwd
" > /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/compile-<phase>-$TS.log 2>&1
```

要点:
- 用 `--preset dev` (= `clang++` + `BUILD_DEV=ON` -Werror -Weverything)
- `-DFMHA_FWD_ENABLE_APIS='fwd'` 把 5 → 1 API (省时)
- `-GNinja` 显式必须加 (preset 不带 generator)
- log abs path, 带 `$TS` 时间戳防覆盖
- exit=0 = pass; 失败时 `Grep "error:" <log>` 找首条

---

## 5. 跑 ABC 命令模板 (verified ship state)

ffm_lite 仿真器, 不需要 GPU, 不传 env var (除非加 dump instrumentation 时 SWE 显式要求 `CK_DUMP_K_LDS=1` 之类)。

每 case `time { docker exec ... ; } 2>> $LOG` 把 walltime 写进 log。

### Case A (fp16 dense s=1023)
```bash
TS=$(date +%H%M%S); LOG=/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/run-<phase>-A-$TS.log
{ time sudo docker exec yiding12-gfx1250 bash -c "
  /home/yiding12/workspace/rocdtif/run-rocdtif.sh ffm_lite \
    /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/build-gfx1250/bin/tile_example_fmha_fwd \
    -prec=fp16 -b=1 -h=1 -s=1023 -mask=0 -mode=0 -d=128 -iperm=0 -operm=0 \
    -v=1 -warmup=0 -repeat=1 -kname=1
" > "$LOG" 2>&1 ; } 2>> "$LOG"
```

### Case B (fp16 GQA+causal s=1023×257)
```bash
... -prec=fp16 -b=1 -h=2 -h_k=1 -s=1023 -s_k=257 -mask=2 ...
```

### Case C (bf16 long seq s=2047)
```bash
... -prec=bf16 -b=1 -h=1 -s=2047 -mask=0 ...
```

公共: `-mode=0 -d=128 -iperm=0 -operm=0 -v=1 -warmup=0 -repeat=1 -kname=1`

**永不用 `-v=2`** (ffm_lite 下 GPU naive ref 假阳性, project_gfx1250_fmha_fwd_dev.md 验过)

---

## 6. ⚠️ Kname Check 规则 (cleanup phase 抓到 false-positive 教训)

**铁律: 每跑完 case 必须 grep kname 验证走的是 `qr_tdm` 不是 `qr_vr`!**

cleanup-v1 时 ABC 三 case 全 valid:y 但 kname 全是 `qr_vr_psskddv` (Phase 5 误 revert qr_vr disable, dispatcher fallback baseline)。SWE 当时差点 ship 了一个 valid:y 假阳性 (TDM 完全没启用, 实际 perf -23% vs B1)。

**Check 方法**:
```bash
grep -E "qr_tdm|qr_vr" run-<phase>-<case>-$TS.log | head -3
```

**Pass 标志**: kname 含 `qr_tdm_vr_npad` (e.g. `fmha_fwd_d128_fp16_batch_..._qr_tdm_vr_npad_..._ntrload_nsink`)

**Fail 标志**: kname 含 `qr_vr_psskddv` (= dispatcher 走 baseline qr 不走 TDM, 整套 B2 工作没生效)

→ 报告 SWE/lead 时 **必须显式标 kname 含 `qr_tdm` ✓**, 不只是说 valid:y。

---

## 7. 验证判定规则 (铁律)

- **只看 `valid:y/n`**, 不看 diff / ULP / 元素值自己判断
- `valid:n` 一律 fail 报 SWE
- `-v=1` (CPU reference) source of truth
- 失败时给 SWE: max err + pct wrong + first OUT err idx + out 数值 sample (前 5)
- **同时 verify kname 含 `qr_tdm`** (按 §6) — 防 dispatch fallback 假阳性

---

## 8. Log 文件命名 + Inventory

- 命名: `compile-<phase>-$TS.log` / `run-<phase>-<case>-$TS.log`
- abs path 写 worktree root: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/`
- **禁止覆盖**, 给 SWE/lead 报告时**始终带 abs path**

### Step B2 关键 log inventory (compact 后保留作 baseline)

**Ship state v2 (cleanup-v2, current ship)**:
- `compile-cleanup-v2-154202.log` (12KB)
- `run-cleanup-v2-A-154245.log` (kname `qr_tdm` valid:y 2555ms)
- `run-cleanup-v2-B-154248.log` (kname `qr_tdm` valid:n 0.062 369ms)
- `run-cleanup-v2-C-154249.log` (kname `qr_tdm` valid:y 9127ms)

**Ship state v1 (failed, dispatch fallback regression — 留作 anti-pattern)**:
- `compile-cleanup-153318.log` / `run-cleanup-{A,B,C}-*.log` (kname `qr_vr_psskddv` 假阳性)

**B1 baseline reference logs** (B2-resume-baseline-v3.md §1 引用):
- `run-stepA-{A,B,C}.log` (Step A baseline pre-TDM)
- `run-stepB1-trial7-{A,B,C}.log` (B1 verified baseline async_load+ds_load_tr)

**Disambig 历史 (B2 phase, Step C 不需重读 dump 全文, 但 path 留作 reference)**:
- `run-Phase2-{B1,B2}-B-*.log` (case B 9-dump full pipeline disambig)
- `run-Qdump-{B1,B2}-A-*.log` / `run-taskB-{B1,B2}-A-*.log` (Q + K register dumps)
- `run-betaprime-A-*.log` / `run-Qpaddisable-{A,B,C}-*.log` / `run-gamma1-{A,B,C}-*.log` / `run-lambdaQ-A-*.log` (各 trial state)

---

## 9. SWE 协议 (与 prompt.txt §4 + qa-env-notes.md 一致)

- **不改代码** (不调 Edit/Write/NotebookEdit 修源码; 写 doc/notes 例外)
- 收到 SWE "请编+跑 case X" → 编 (incremental 优先) → 跑该 case
- **平时一次只跑一个 case** 节省时间; SWE 明示 "ABC 全跑" 才连跑
- **case A fail → 立即报 SWE 不继续 B/C** (除非 SWE 明示 "fail 也跑全 ABC")
- **kname 必 verify 含 qr_tdm** (cleanup 教训, §6)
- log/全文写文件, 消息里只贴关键行 + abs path
- **每次回应必须真调 SendMessage 工具**, 不要纯文本输出 (踩过死锁坑)
- 收件人名: `to="team-lead"` (不是 `to="lead"` — 落孤儿 inbox); SWE 用 `to="swe"`
- 不主动通知 mentor (除非 SWE 明示 cc)
- **禁所有 git 写**: commit/push/add/merge/rebase/branch 创建; 允许只读 git status/diff/log
- **禁 bash -c 内部 `| tail`/`| grep`/`| head`** (CLAUDE.md 硬规)

---

## 10. Step C Backlog (4 项, lead 启动后会派)

per (β'')-Q-fix + cleanup-v2 ship 之后已知:
1. case B GQA Q dist multi-head edge fix (max err 0.062 → 0)
2. (X+) CK core latent fix 独立 PR (container_helper.hpp + tuple.hpp 5 处 in-place mutate tuple ops)
3. dispatcher prefer-qr_tdm 正确 fix (现在靠 qr_vr disable 注释 hack — 真正修要在 dispatcher 里加 priority)
4. (lead 派时具体讲)

QA 角色: 每个 backlog item 派下来 → 编 + 跑相关 case + verify ship signals (含 kname check) + 报 SWE/lead。standby 等派活, 不主动追 backlog。

---

## 11. Compact 后 Recovery Quick Action (Step C 接活时做)

1. Read 这份 doc (qa-post-compact-baseline.md) 全
2. `sudo docker exec yiding12-gfx1250 echo ok` 探活容器
3. `ls -d /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/build-gfx1250` 验证 build dir
4. `ls -t /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/run-cleanup-v2-*.log` 拿到 ship state baseline log
5. `SendMessage(to="team-lead", message="qa post-compact ready: env reachable=y, build dir=y, ship baseline log 在手")`
6. Standby 等 lead 派 Step C task

---

file: qa-post-compact-baseline.md (Step B2 cleanup-v2 SHIP state confirmed, Step C 启动前)
