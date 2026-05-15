---
name: Review audit checklist — commit body / by-construction / collab self-correction
description: Reviewer audit 3 个 reusable heuristic — commit body narrative inconsistency 检测 / "by-construction" reasoning 判断框架 / 跟 author 1:1 触发 anti-pattern A 自校正的 collab pattern
type: feedback
---

# Review Audit Checklist

Reviewer 在 audit shipped commit / sign-off doc / status doc 时常踩到的 3 类 pattern。每条带 detection 信号 + 推理路径 + actionable verdict 模板。

来源: 2026-05-06~05-08 gfx1250-fmha-tdm Step B2 + Step C #1 path 7 ship 期 audit (见 project doc `reviewer-audits-2026-05-08-summary.md`)。

---

## 1. Commit body narrative ↔ staged diff inconsistency

### Symptom

Commit body 顶部 (一段式 summary 或 abstract) 写"做了 X, 走了 Y, fall back 到 Z"; 但底下 detail section 或实际 staged diff 描述的是"做了 X', 路径是 Y' 不是 Y, 没 fall back"。

### Why happens

Trial 期 commit body 多次 amend, 顶部段是较早 trial 的 narrative leftover, 后来 squash / amend 时只更新了 detail section 没同步顶部。

### Detection 信号

- 顶部 summary 段跟 "Resolved issues" / "Pipeline" / "Test" 段说法**不一致** (e.g. summary 说 "GQA falls back to B1", 但 Test 段 multi-stride GQA 表全 valid:y 走 qr_tdm)
- Bullet 数 / order 跟 commit footer "Remaining future work" 列出来的 item 数对不上
- "We did A" 跟 "Note: A is intentionally unchanged" 在同 commit body 里共存

### Audit 推理路径

1. Read 顶部 summary 段, 提取 claim list (e.g. "Q+K TDM, V async, GQA falls back to B1")
2. Read detail section (Pipeline / Kernel / Codegen) 提取 actual change list
3. Read Test section 验 actual behavior (kname grep / valid 表 / multi-stride 表)
4. Cross-check 三方 — 不一致条目就是 stale narrative

### Verdict 模板

```
[MED] B2 commit body narrative inconsistency
- 顶部 summary "GQA falls back to original B1 behavior" stale
- 底下 multi-stride GQA 表 (h=4 h_k=1 stride_q=512 / h=8 h_k=1 stride_q=1024) 全 valid:y 走 qr_tdm_vr_npad
- 推荐 amend wording: "Single-head and multi-head GQA both go through TDM via path 7 dram view dispatch"
- Risk: 不修 future reader 看 commit body 误以为 GQA 是 fallback path, 调 perf 时 spend 时间 trace 不存在的 fallback
```

---

## 2. "By-construction" reasoning HOLDS / FAILS judgment

### Symptom

SWE/mentor commit body / sign-off doc 写 "X by construction can't affect Y" 当 verify 替代 (避免跑全 regression sweep)。

### Why dangerous

By-construction 推理依赖**完整 dependency graph trace**, 漏一条 indirect include / template instantiation / dispatch path 就 false-claim。Anti-pattern A (first-principle 没 empirical) 的 specific case。

### When HOLDS (可信)

- 改动只在 leaf source file (e.g. `fmha_fwd_kernel.hpp`)
- 该 file **not transitively included** by Y 的任何 source/header
- Verify by `grep -r '#include.*fmha_fwd_kernel' <Y dir>` 实际跑空
- (可选) 跑 1 次 spot-check binary 比 byte 跟前 baseline identical 当 sanity

### When FAILS (不可信)

- 改动在 framework 层 (e.g. `tile_window.hpp`) 触及 template 或 type traits — Y 即使没 include 也可能 instantiate path 不同
- 改动 dispatch 逻辑 (e.g. `fmha_fwd.py` codegen / `make_*` factory) — Y 可能 silently 落别 branch
- "By construction" 推理依赖某 macro / `if constexpr` / SFINAE 的 deduction outcome — 这个 outcome 本身需独立 verify

### Audit 推理路径

1. 找 commit 描述 by-construction 的句子, 提取 "X 不影响 Y" claim
2. 实际 grep `#include` chain X→Y 是否真空
3. 如果空, verify HOLDS
4. 如果不空, FAILS — 要求 SWE 跑 actual regression sweep cover Y

### Verdict 模板

```
[LOW] GEMM identity by-construction reasoning HOLDS
- Claim: "(α') Step B2 fix + path 7 dispatch 都 only touch fmha_fwd_kernel.hpp,
  no GEMM example or non-fmha header transitively includes it"
- Verify: grep -r '#include.*fmha_fwd_kernel' example/ck_tile/<gemm_dirs>/ → 空 ✓
- Verify: 5 GEMM FAIL set diff against prior baseline empty ✓
- Verdict: HOLDS, no further regression sweep needed for GEMM binaries
- Caveat (写给 future reviewer): 这条 reasoning 只 cover GEMM examples;
  非 GEMM downstream (e.g. PyTorch Inductor) 不在 scope, 由 caller responsibility
```

---

## 3. Author 1:1 触发 anti-pattern A 自校正的 collab pattern

### Context

Mentor / SWE first-principle reasoning fail 时 (anti-pattern A), reviewer 直接 challenge "你错了" 通常被 author 防御 (尤其 mentor 三次同样 sign-off 错的情境)。下面 pattern 提高自校正概率。

### Pattern: 用 author 自己的 source 反推 (非"我读了别的文献你错了")

1. Reviewer 不直接说"你描述错了"
2. **拉 author 自己之前 endorse 过的 production code** (B1 verified path / 上游已 merge PR / author 自己写的 doc) 当 anchor
3. Frame challenge 为: "你之前说 X verified work, 但 X 的 source code 显示 layout 是 Y, 跟你现在描述的 Z 矛盾, 哪个对?"
4. 让 author 自己 trace X 的 source 重新 reason
5. Author 自己发现矛盾 → self-correction 阻力小

### 例子 (mentor ds_load_tr semantics 三次错的 self-correction)

Reviewer 没说: "你描述 ds_load_tr 反向了, 应该是 K-outer LDS → K-inner VGPR"

Reviewer 说 (via lead relay): "你之前 sign-off B1 是 verified work, B1 V LDS 用 `MakeVLdsBlockDescriptor<Xor=false>` stride `(kNPerBlock, 1)` = K-outer N-inner, 跟你现在 sign-off 'V LDS 必须 K-inner' 矛盾。能 trace B1 实际 LDS layout 是哪个吗?"

→ Mentor 自己 read B1 policy code → 自己 conclude `mentor-reanalysis-ds-load-tr-correction.md` §1: "我描述 ds_load_tr_b128 反了"

### 通用化

适用于任何 author 三次同 sign-off 错且 first-principle 推理强势的情境 (常见于 mentor / 高 seniority team member)。**关键是 anchor 用 author 自己 endorse 过的 source**, 不是 reviewer 拉新 source 当 authority。

### NOT applicable

- Author 第一次 sign-off 错且 evidence 充分 → 直接 challenge OK, 不需绕
- Empirical signal (max_err / kname / dump) 已经 contradict author 推理 → 直接出 evidence, 不需 source-anchor
- Author 主动 ask "我哪错了" → 直接答, 不需 collab framing

---

## 通用 audit workflow

任何 ship audit 套这 3 把尺 + reviewer-baseline §4 三个 anti-pattern (A first-principle 没 empirical / B indirect signal 当 direct verify / C co-design chain 类比太粗):

1. **Narrative consistency** (本 doc §1): 顶部 summary ↔ detail ↔ test result 三方一致?
2. **By-construction claim** (本 doc §2): 任何 "X 不影响 Y" 推理是否 grep 验过?
3. **Anti-pattern A/B/C** (reviewer-baseline §4): empirical / direct signal / co-design 5 piece 全套?
4. **Collab self-correction** (本 doc §3): 发现 anti-pattern A 时, anchor 用 author 自己 source 不直接 challenge

发现 hole → verdict 模板 (severity + finding + recommend amend / actionable next), relay lead, lead 决定是否 forward author。

