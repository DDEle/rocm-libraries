---
name: ffm_lite ABC case sensitivity for gfx1250 fmha TDM verify
description: 在 ffm_lite 上跑 gfx1250 fmha fwd 的 ABC 三 case 各自对 Q/K/V-TDM 类改动的灵敏度差异 + error signature pattern，给后续 QA 选 case 顺序 / 解读 numerics 当 heuristic
type: reference
---

## 关联 memory

- `project_gfx1250_fmha_fwd_dev.md` — ABC shape 定义 + 验证选项陷阱（`-v=1` 强制走 CPU ref）
- `project_gfx1250_fmha_tdm_v1.md` — Step B1 K-distribution fix baseline
- `reference_simulator_ffm_lite_am.md` — ffm_lite 不作 perf 数（functional only）

## 三 case 对 V/K/Q-TDM 类改动的灵敏度

ABC 三 case 在仿真器下表现的 error signature 不一样，**先跑 case A** 是最佳早期信号。

| case | shape | TDM-class bug 表现 |
|---|---|---|
| **A** (s=1023 fp16 dense, b=1 h=1, mask=0) | dense fp16 | **第一暴露**，error 量级 fp16-scale (max_err 6.x)，wrong% 100% — 直接信号 V/K/Q dist 对不上 |
| **B** (s=1023 fp16 GQA causal, b=1 h=2 h_k=1, mask=2) | GQA + causal | wrong% **partial** (20-30% typical) — mask 把后半 token block 掉，误差被局部化；适合验 GQA edge / mask-stride 互动 bug |
| **C** (s=2047 bf16 dense, b=1 h=1, mask=0) | bf16 + 长 seq | **overflow signature**: V-side garbage feed softmax accumulator → max_err 1e+08 (vs A 6.x)；长 seq + bf16 wider range 让爆炸成立 |

## 实证 (V-TDM 9-piece broken state 2026-05-11, kname `qr_tdm_vr_npad...` 三 case 全对)

| case | valid | max_err | wrong% | 解读 |
|---|---|---|---|---|
| A | n | 6.081 | 100% | direct dist mismatch, fp16 ULP-scale |
| B | n | 6.319 | 24.99% | causal mask 收缩 blast radius |
| C | n | **1.09e+08** | 100% | bf16 长 seq + V garbage → softmax accumulator overflow |

**对比 (h) hybrid OK state (HEAD `08b61885fc9` Step C #1 ship)**: ABC 全 valid:y, max_err 0.

## QA 选 case 启发式

1. **新加/改 V/K/Q distribution 或 LDS layout** → 先跑 case A，valid:n 立即停 + 报。A pass 才进 B。
2. **case A pass 但 case B fail** → 大概率 GQA 或 causal mask 互动 bug（V-TDM 9-piece 没踩，但 cleanup-v2 B2 那次 case B max_err 0.062 14% wrong 就是 GQA Q dist multi-head edge）。
3. **case C 出 1e+08 量级 max_err** → V-side / softmax accumulator 强信号，不是普通 dist mismatch。
4. **case A/B/C wrong% 全 100%** → 整 distribution 错位（不是 corner），通常 design-level 问题不是 typo。

## kname 必须先 verify

每 case 跑完 grep kname：
```bash
grep -E "qr_tdm|qr_vr" run-<phase>-<case>-$TS.log | head -3
```
- 含 `qr_tdm_vr_npad` ✓ — 改动真生效
- 含 `qr_vr_psskddv` ✗ — dispatcher fallback baseline，下面所有 numerics 都是假阳性

cleanup-v1 教训：ABC 三 case valid:y 但 kname 全 `qr_vr_psskddv`（Phase 5 误 revert qr_vr disable），差点 ship 一个 TDM 完全没生效的"假绿灯"。

## Why

- A 是最 minimal path（无 mask/GQA/varlen 干扰），任何 dist/layout bug 第一时间暴露
- B 引入 mask + GQA 把误差局部化，适合验互动 bug 而不是 raw dist
- C 用 bf16 + 长 seq 把 V-side 微小污染放大成 catastrophic overflow（softmax exp + accumulator），因此 1e+08 量级是 V-side bug 的"指纹"，不是普通的 dist mismatch

