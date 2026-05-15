# QA Environment / Build / Test Notes — gfx1250 FMHA TDM

新 QA spawn 时读这一份能立即接活，**不需要再问** docker / build / case 参数 / log 命名。
配套 memory：`~/.claude/memory/project_ck_dev_setup.md` + `project_gfx1250_fmha_fwd_dev.md` + `project_gfx1250_fmha_tdm_v1.md`。

---

## 1. Docker 环境

- **长期容器**: `yiding12-gfx1250`（不要 `--rm`，不要 spawn 新容器）
- **exec 命令**: `sudo docker exec yiding12-gfx1250 bash -c "<cmd>"` — **必须 sudo**（user 不在 docker group）
- **挂载**: `/home/yiding12/workspace` 容器内外路径一致
- **镜像**: `rocm/composable_kernel-private:npi-mi450-latest`
- **worktree root** (容器内/外都用): `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel`

---

## 2. Build 目录隔离

- **用 `build-gfx1250/`**，**不要碰 `build/`**（user 私用调试目录）
- 编译并行度: `-j204`（256 核 × 80%）
- Bash 工具调用编译时加 `timeout=600000` (10 min)；跑测试加 `timeout=180000` (3 min)

---

## 3. 编译命令（已验证 work 的格式）

### 首次配置 + 全量编译
```bash
TS=$(date +%H%M%S)
sudo docker exec yiding12-gfx1250 bash -c "
  cd /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel && \
  cmake -S . -B build-gfx1250 -GNinja --preset dev \
    -DGPU_TARGETS=gfx1250 -DFMHA_FWD_ENABLE_APIS='fwd' && \
  ninja -C build-gfx1250 -j204 tile_example_fmha_fwd
" > /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/compile-<phase>-$TS.log 2>&1
```

### Incremental 重编（SWE 只改了 .hpp，没改 .py / CMakeLists）
```bash
TS=$(date +%H%M%S)
sudo docker exec yiding12-gfx1250 bash -c "
  cd /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel && \
  ninja -C build-gfx1250 -j204 tile_example_fmha_fwd
" > /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/compile-<phase>-$TS.log 2>&1
```

### 关键点
- `-GNinja` 显式必须加（preset 不带 generator）
- `dev` preset 用 `/opt/rocm/llvm/bin/clang++` + `BUILD_DEV=ON` (-Werror)
- `FMHA_FWD_ENABLE_APIS='fwd'` 把 5 → 1 API（编译省时）
- 临时 `--filter` 加速详见 `project_gfx1250_fmha_fwd_dev.md` 手段 2，**不要 commit**

---

## 4. 跑测试命令模板（ffm_lite 仿真）

gfx1250 是 pre-silicon，**不能直接跑**，必须 rocdtif/ffm_lite 仿真。**不需要 GPU**，所以**跳过 `HIP_VISIBLE_DEVICES`**。

```bash
TS=$(date +%H%M%S)
LOG=/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/run-<case>-$TS.log
{ time sudo docker exec yiding12-gfx1250 bash -c "
  /home/yiding12/workspace/rocdtif/run-rocdtif.sh ffm_lite \
    /home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/build-gfx1250/bin/tile_example_fmha_fwd \
    <case args>
" > "$LOG" 2>&1 ; } 2>> "$LOG"
```

### 陷阱
- **binary 必须 absolute path**（wrapper `cd` 到 mktemp，相对路径 exit 127）
- Bash 工具加 `timeout=180000`
- `time { ... } 2>> "$LOG"` 把 walltime 也写进 log
- **绝不用 `v=2`**（ffm_lite 下假阳性，详见 `project_gfx1250_fmha_fwd_dev.md`）

---

## 5. ABC Case 参数

共用: `-mode=0 -d=128 -iperm=0 -operm=0 -v=1 -warmup=0 -repeat=1 -kname=1`

| Case | 特有参数 | 用途 | walltime ref | sim ms ref (B1 baseline) |
|------|---------|------|--------------|--------------------------|
| **A** | `-prec=fp16 -b=1 -h=1 -s=1023 -mask=0` | dense fp16 sanity | ~3s | 2539 |
| **B** | `-prec=fp16 -b=1 -h=2 -h_k=1 -s=1023 -s_k=257 -mask=2` | causal+GQA | ~0.7s | 374 |
| **C** | `-prec=bf16 -b=1 -h=1 -s=2047 -mask=0` | bf16 + 长 seq | ~9.5s | 9113 |

---

## 6. 验证判定规则（铁律）

- **只看 binary 输出的 `valid:y/n`**，不要自己看 diff / ULP / 元素值判断
- `valid:n` 一律视为 fail，立即报 SWE
- `-v=1` (CPU reference) 是 source of truth，**永不换 v=2**
- 同一 case 失败时给 SWE 报：max abs err + pct wrong + out 数值 sample (前几个) — 帮 SWE 区分 garbage vs permutation

---

## 7. Log 文件命名规则

- **必须带 timestamp**（防覆盖）：`compile-<phase>-$TS.log` / `run-<phase>-<case>-$TS.log`
- worktree root 已有大量旧 log，**禁止覆盖**
- 给 SWE / lead 报告时 **始终带完整 abs path**

---

## 8. 现有 Log Inventory（worktree root）

### Useful baseline / reference（**保留**，新 QA 排错可对比）

| 文件 | 含义 |
|------|------|
| `run-stepA-{A,B,C}.log` | Step A baseline，qr pipeline 跑 ABC 全 pass，可证 toolchain 健康 |
| `run-stepB1-trial7-{A,B,C}.log` | **B1 commit `16324795` 验证基准**（async_load + ds_load_tr 路径 ABC 全 pass，sim ms A=2539 B=374 C=9113） |
| `compile-stepB1-trial7-114534.log` | B1 干净编译 reference (32K) |
| `run-lambda1-K-step1-A-034439.log` | (λ-1) K trivial tile-major dist 单独打开 → max err 23.7 garbage |
| `run-lambda2-V-step4-A-040846.log` | (λ-2) V dual view 加上 → max err 0.37 / 98.7% wrong (量级对，permutation 错) |
| `run-h-A-053207.log` | (h) hybrid (K-TDM + V-async revert) → 仍 0.36 / 97.6% wrong (validates K dist 改动是数值错位主因) |

### B2 历史 trial logs（**别 delete，可 archive**）

`compile-stepB2-{D1,D2,D3,D4,D5,D5b,alpha-*,Xplus,XplusAlpha,H4p,eta,hbeta,hgamma,hdelta}-*.log`
+ 对应 `run-stepB2-*.log` —— B2 上一轮 6 hypothesis 的 trial 历史，背景见 `swe-journey-B2-trials-202429.md`。

### 配套 doc（已存在，必读）

`mentor-fmha-tdm-design-notes-202402.md` (mentor 设计) + `swe-journey-B2-trials-202429.md` (B2 trial 历程) + `final-summary-team-shutdown-212300.md` + `B2-lambda-resume-baseline.md` + `mentor-reanalysis-ds-load-tr-correction.md` + `swe-reanalysis-K-side-correction.md`

### Noise（可忽略，但别删）

`build-tdm-stepA.log` / `compile-094651.log` / `configure-stepB1-trial2,3-*.log` —— 早期 plumbing/cmake 探索。

---

## 9. 工作流程

1. SWE 派 "请编 + 跑 case X" → 编（必要时 incremental）→ 跑该 case
2. **平时一次只跑一个 case**（节省时间），SWE 明确 "ABC 全跑" 时才连跑
3. case A fail → 立即报 SWE，**不继续 B/C**
4. 报告格式参考 `project_gfx1250_fmha_tdm_v1.md` 顶部的 QA prompt
5. **每次回应 teammate 必须真正调 SendMessage 工具**，不要纯文本输出

---

## 10. 边界（铁律）

- **绝不改代码**（不调 Edit / Write / NotebookEdit 修源码 — 写 doc/notes 例外）
- **不主动通知 mentor / lead 中间过程** — 只回 SWE；除非 SWE 明确让你 cc
- **不重复跑同一 case "再确认一次"**
- **禁所有 git 写操作**；允许只读 `git status / git diff / git log`
- **不要在 `bash -c "..."` 内部用 `| tail` / `| grep` / `| head`**（CLAUDE.md 硬规定）；输出全量到文件后再用 Read/Grep 工具过滤

---

## 11. (可选) LDS dump disambig 跑测注意

如果 SWE 加 LDS dump 代码（host buffer alloc + kernel 写出 + cpp side dump 到文件）让你 verify：
- dump 输出文件大概会落在容器 cwd（即 `/tmp/rocdtif-run.XXXX/`）— 跑完前用 `tee` 或固定路径输出
- 跟 dram K tile 比对：SWE 应该一起给 reference dump（CPU 算的 expected layout）
- 多半 SWE 会先告知具体 dump 文件名/格式，没说就问，**不要猜**
