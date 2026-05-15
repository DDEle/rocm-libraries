# SWE Status — K LDS Dump Disambig (⚠️ Incomplete - Mentor Challenged)

**写作时间**: 2026-05-06 (initial), 2026-05-06 (revised post-mentor-challenge)
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Branch**: `yiding12/gfx1250-fmha-tdm`，HEAD `16324795215` (B1) + 9 dump-instrumentation edits unstaged
**关联**: `B2-resume-baseline-v3.md` §4 (LDS dump plan), `swe-reanalysis-K-side-correction.md` (root cause hypothesis), `mentor-hardware-knowledge-fixed.md` §1 (ds_load_tr semantics correction)

---

## ⚠️ STATUS UPDATE 1 (post mentor challenge)

**TL;DR (revised)**: B2 K LDS dump 实测 mismatch=0 (TDM 1:1 byte copy plain row-major **confirmed**), 但 mentor catch disambig 漏掉一个关键 fact: **K LDS read view 用 Xor=true** (`MakeKLdsBlockDescriptor<Problem, false, true>`, policy:340-381 XOR transform branch) + **B1 dram dist 跟 B2 trivial tile-major dram dist 都用 lds_write_view (Xor=false plain row-major desc)**。如果 B1 baseline LDS byte 也是 plain row-major (跟 B2 同), 那 read 用 XOR'd byte 就拿不到 reg dist 期望的 element — 但 B1 valid:y empirical confirmed → **必有一个我们没看到的 mechanism (e.g. async_load_tile 内部 auto-derive XOR'd write)**。

→ "Case 1 confirmed → 走 (α) reg dist 调" 这个 inference **跨度太大**，跨过了"B1 LDS byte = plain row-major"这个 implicit assumption。在 verify B1 LDS byte layout 之前，**(α) / (γ) 都不能 sign-off**。

→ **Next action (1-2h)**: 做 B1 baseline K LDS dump 实验拿到 B1 LDS byte ground truth，然后再决定 fix path。

---

## ✅ STATUS UPDATE 2 (post B1 dump experiment — A2 confirmed)

**TL;DR**: B1 K LDS dump 跑完 (after fixing codegen qr_vr disable to force qr_tdm dispatch — first run had kname=qr_vr 漏 dispatch)，**Hypothesis A2 confirmed**: B1 LDS = XOR'd byte layout, B2 LDS = plain row-major (75% mismatch with deterministic chunk-swap pattern)。Mentor + SWE align 推荐 **(β) 1-line read view Xor=true → Xor=false** ~30min 测试 (mentor sign-off ✓ based on production code verify of BWarpDstrEncoding decoupled from LDS storage layout)。

### B1 Dump Result (case A, kname `qr_tdm_vr_npad`, valid:y, sim 2595 ms)

```
[K-LDS-dump-B1] kN0=64 kK0=32 total=2048 mismatch=1536 first_mm=(n=1,k=0) actual=0.835938 expect=0.300049
```

- **n=0**: 32/32 全等 (plain row-major for first row)
- **n=63**: chunks 整体 reverse — actual_C[i] == expect_C[3-i] (8-element chunks, XOR with 3)
- → canonical bank-conflict-avoidance XOR pattern: byte coord = (n * kK0 + xor_k_chunk * kKPack + k_intra) * sizeof, where xor_k_chunk = k_chunk ⊕ (n_outer mod 4)
- **For n=0**: n_outer=0 → XOR with 0 → no reorder ✓
- **For n=63**: n_outer mod 4 = 3 → all chunks reversed ✓

### Paradox 解析

之前 mentor critique 列的 4 个 facts (B1==B2 reg dist + same LDS read/write view + Xor=true read view) 都成立。但 dump 显示 B1 LDS XOR'd, B2 plain。结论：**dist encoding 决定 thread-to-LDS-coord mapping，不是 write view desc 决定 byte 位置**。
- B1 5D dram dist + plain write view → 5D dist projection 让 thread t 写到 XOR'd byte position (跟 read view Xor=true 期望对齐)
- B2 trivial tile-major dram dist + plain write view → trivial dist projection 让 thread t 写到 plain byte position (跟 read view Xor=true 不对齐)
- → **dist 跟 LDS view 之间的 implicit coordination** (B1 5D dist + Xor=true read view = co-designed pair)

### Mentor sign-off — (β) 1-line change

**Production code verify** (per Rule 1):
- `warp_gemm_attribute_mfma.hpp:530-540`: `BWarpDstrEncoding = tile_distribution_encoding<...>` — pure coord encoding, **跟 LDS storage byte layout 解耦**
- `coordinate_transform.hpp:1346-1361`: XOR transform = pure reversible coord transform，不改 element 含义
- → wmma B operand 不 care LDS bytes XOR'd or plain, 只 care reader 给 element 序列正确
- → **(β) 改 read view Xor=true → Xor=false 概念前提成立**

**(β) 实施** (1-line change in tdm.hpp L317):
```cpp
// Before:
auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
    static_cast<KDataType*>(smem_ptr),
    Policy::template MakeKLdsBlockDescriptor<Problem, false, true>());  // Xor=true

// After:
auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
    static_cast<KDataType*>(smem_ptr),
    Policy::template MakeKLdsBlockDescriptor<Problem>());                // Xor=false default
```

(write view L315 已经是 Xor=false 不改 — reader + writer 都用 plain row-major desc, 对齐 ✓)

### (β) Empirical predict

- **(β-success)**: case A valid:y, max_err < 0.001。Latency: 可能 ≈ B1 (TDM offset bank conflict) 或略 > B1 (bank conflict dominate)。B/C 大概率 also pass (XOR fix symmetric)
- **(β-fail unexpectedly)**: reg dist outer encoding 隐含期望 XOR'd LDS (不太可能, 已 verify 解耦) → reveal 更深 mechanism, 走 (γ)

### 为啥先 (β) 不 (γ)

1. **Code change 最小**: 1 模板参数 vs (γ) 设计 dist projection 重写 1-2d
2. **测试最便宜**: 1 build + 1 run ~30min vs (γ) 多轮 trial
3. **Self-disambig 价值**: 结果直接告诉 reg dist 是否真要求 XOR'd input
4. **功能优先**: 即使 perf 略退步, functional fix done = milestone
5. **不阻 (γ)**: 如果 (β) work but perf 退步显著, 再设计 (γ) 优化
6. **Pattern 安全**: read view Xor=true 是 perf 优化 (avoid bank conflict), 不是 correctness requirement → 改成 Xor=false 是 graceful degradation 不是破坏性改动

### (γ) 留给 perf 优化阶段

如果 (β) 功能 pass + perf acceptable (e.g. 还在 -10~-20% vs qr baseline, 比 B1 -21~-32% 弱但仍正向), **可直接 ship (β) 作为 Step B2 done**。Perf 优化留给 Step C。

如果 (β) 功能 pass + perf 太差 (慢于 qr baseline), 再走 (γ) 重设 dram dist 让 TDM 写出 XOR'd byte。

### Mentor sign-off ✓
基于 empirical foundation (B1 dump A2 confirmed) + production code verify (BWarpDstrEncoding 解耦) + 自带 disambig 价值 + 最小 risk。

### Process Lessons (per mentor 提示)

Step B2 这次破解了 mentor 三次 sign-off 错的 anti-pattern:
1. **每个 disambig step 都 empirical verify**, 不 inherit 假设链 (B1 dump catch 了 SWE 第一轮 "Case 1 confirmed → α" 跨度)
2. **Mentor + SWE 双 verify production code** before sign-off (BWarpDstrEncoding 解耦 verify 是 (β) 关键前提)
3. **Reviewer/mentor catch 价值**: SWE 第一轮 disambig 完不完整 mentor 的 4-fact paradox grep catch 到了, 才有 B1 dump 跟进
4. **Process 改进 > fix 本身**: empirical disambig 工艺可复用到后续 V side / B/C cases / Step C perf 优化

---

## 1. Original TL;DR (PRE-CHALLENGE — partially invalidated, retained for audit trail)

K LDS dump disambig 实测 **Case 1 confirmed**：TDM trivial tile-major write 出来的 K LDS 物理 byte 顺序 = plain K-outer N-inner row-major (1:1 dram copy)，跟 expected K dram tile **mismatch=0 / 2048 elements**。

→ ~~Hypothesis A 100% 实证支持：LDS write 端干净，(h) hybrid valid:n element permute 真根因 = **K reg dist (BWarpDstrEncoding embed) 跟 trivial tile-major dram dist coordination 缺失**。~~ **REVISED**: Hypothesis A confirms TDM 1:1 byte copy ✓ but 没 confirm "B1 LDS byte == plain row-major" 隐含前提。Root cause hypothesis 可能错。

→ ~~Recommended fix path = **(α)** 改 K reg dist outer encoding (1-2d, LOW risk)；备选 (γ) 改 dram dist hybrid。具体 α/γ choice 待跟 mentor 内部 sync 后报 lead 决策。~~ **REVISED**: α/γ 都不能现在 sign-off, **deferred pending B1 LDS dump**。

---

## 2. Disambig 实施 (per lead task #3 requirement)

### Implementation diff (9 edits 未 commit, 全在 worktree)

| 文件 | 改动 |
|---|---|
| `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` | (a) `FmhaFwdCommonKargs` L147 加 `void* debug_k_lds_dump_ptr = nullptr` 字段; (b) L2904 单 smem call site 转发 `kargs.debug_k_lds_dump_ptr` |
| `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` | (c) `run()` L160 + `operator()` L1268 加 optional `void* debug_k_lds_dump_ptr = nullptr` 默认参数; (d) L412 后插 dump kernel code (single-thread copy with i_total_loops==0 + i_k0==0 + thread/block 0 guards) |
| `example/ck_tile/01_fmha/fmha_fwd.hpp` | (e) `fmha_fwd_args` L352 加 `debug_k_lds_dump_ptr` 字段; (f) `fmha_fwd_create_kargs_and_grids` L809 后加 `kargs.debug_k_lds_dump_ptr = args.debug_k_lds_dump_ptr;` 一行 propagation |
| `example/ck_tile/01_fmha/fmha_fwd_runner.hpp` | (g) `run_fwd` lambda 加 env var `CK_DUMP_K_LDS=1` 触发 4KB DeviceMem alloc + `Realloc` (避开 implicit copy-assign double-free) + sentinel init + post-run `FromDevice` readback + `std::memcmp` bit-exact 比对 (避开 dev preset `-Wfloat-equal`) + 4 行 `[K-LDS-dump]` 输出 |

### Design protocol (per mentor sign-off)

- **Sync timing**: dump 插在 `if constexpr(i_k0 == 0)` 静态分支内 + `s_wait_tensorcnt_barrier<0>()` 后。Verify (mentor): `s_wait_tensorcnt_barrier` (arch.hpp:1175-1180) = `s_wait_tensorcnt + block_sync_lds`，single-thread copy 不需额外 barrier
- **Runtime guard**: `i_total_loops == 0` 避免 do-loop 跑 16 iter 覆盖 dump_buf (mentor caught 的 critical bug，我之前漏)
- **K LDS layout assumption**: `MakeKLdsBlockDescriptor<Problem>` (policy:412-416) get_lengths = `(kN0, kK0)`, stride `(kK0, 1)` → byte(n,k) = (n*kK0 + k)*sizeof(KDataType) = N-outer K-inner row-major
- **Expected K dram tile (case A)**: BSHD layout (`-iperm=0`)，K_dram[n][k] = `k_host(0, n, 0, k)` for n ∈ [0, kN0=64), k ∈ [0, kK0=32)；hypothesis A 预测 `dump_host[n*kK0 + k] == k_host(0, n, 0, k)`

### Compile/run protocol fix

- 第一次编 fail: `if(a != e)` 触发 `-Wfloat-equal` (dev preset `-Werror -Weverything`)
- 修法: `std::memcmp(&actual_v, &expect_v, sizeof(KDataType))` bit-exact compare，保持 TDM byte-copy 语义 (任何单 bit diff = 真 mismatch，不容忍 fp 抖动)
- 第二次编 pass + run case A pass

---

## 3. Empirical Disambig Result

### Case A run (`-prec=fp16 -b=1 -h=1 -s=1023 -mask=0 -mode=0 -d=128 -iperm=0 -operm=0`)

```
[lambda-K] raw_box_dim: dim0=32 dim1=16 (B1 was (8,2); expect (32,32))
[K-LDS-dump] kN0=64 kK0=32 total=2048 mismatch=0
[K-LDS-dump] row n=0 actual: 0.374268 0.708984 0.46875 0.843262 0.854004 0.0466614 0.377686 0.261963 0.179321 0.128174 0.146851 0.135254 0.556152 0.853516 0.1026 0.595215 0.1604 0.632812 0.175049 0.728516 0.320068 0.909668 0.166992 0.342529 0.567383 0.0859375 0.374268 0.710938 0.769531 0.919434 0.727539 0.0566101
[K-LDS-dump] row n=0 expect: 0.374268 0.708984 0.46875 0.843262 0.854004 0.0466614 0.377686 0.261963 0.179321 0.128174 0.146851 0.135254 0.556152 0.853516 0.1026 0.595215 0.1604 0.632812 0.175049 0.728516 0.320068 0.909668 0.166992 0.342529 0.567383 0.0859375 0.374268 0.710938 0.769531 0.919434 0.727539 0.0566101
[K-LDS-dump] row n=63 actual: 0.00448608 0.586426 0.588867 0.0353088 0.918457 0.00610733 0.745117 0.128662 0.814453 0.671875 0.944336 0.982422 0.956055 0.12915 0.535156 0.420898 0.86084 0.346191 0.136719 0.793945 0.621094 0.882324 0.879395 0.490479 0.495605 0.123169 0.0811157 0.289307 0.295654 0.346191 0.429932 0.85498
[K-LDS-dump] row n=63 expect: 0.00448608 0.586426 0.588867 0.0353088 0.918457 0.00610733 0.745117 0.128662 0.814453 0.671875 0.944336 0.982422 0.956055 0.12915 0.535156 0.420898 0.86084 0.346191 0.136719 0.793945 0.621094 0.882324 0.879395 0.490479 0.495605 0.123169 0.0811157 0.289307 0.295654 0.346191 0.429932 0.85498
```

**ABC verify state (case A only — case B/C 没跑因 disambig only)**：
- valid:n
- max_err = 0.3565674
- pct_wrong = 97.56461%
- out sample (前几个): 0.59 vs 0.51 / 0.81 vs 0.49 / 0.67 vs 0.50 (量级对，散落 element permute)
- 跟之前 (h) hybrid (max_err 0.36 / 97.6%) 数字几乎完美一致 → element permute pattern 一致

**latency**: sim 2577.510 ms / walltime 3.049s vs B1 baseline sim 2539 ms — TDM K path perf **接近 B1 async_load**，~1.5% slowdown (噪声范围)

**kname**: `fmha_fwd_d128_fp16_batch_b64x64x32x128x32x128_r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_tdm_vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_ntrload_nsink` (含 `qr_tdm` ✓)

**Logs (abs path)**:
- compile: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/compile-stepB2-kldsdump-v2-083048.log`
- run: `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/run-stepB2-kldsdump-A-v2-083119.log`

---

## 4. 分析 — Hypothesis A 100% confirm

### 5 种 disambig 情况里只 hit "全等" 这条

按之前 mentor sketch 列的 5 种情况：
| 情况 | 实测吗 |
|---|---|
| 全等 (Case 1: plain row-major 1:1 dram copy) | ✅ |
| 全 0 (sync timing 错或 TDM 没发出) | ❌ |
| per-32-block permute (IsWarpLevelParallelOnly+box_dim 互动) | ❌ |
| 仅前 N element 对其它全错 (box_dim 没覆盖) | ❌ |
| 其它 garbage (Case 2 surprising layout) | ❌ |

### 含义

1. **TDM trivial tile-major write 行为** = 干净 plain K-outer N-inner row-major LDS, byte(n,k) = (n*kK0 + k)*sizeof(KDataType). 跟 K LDS descriptor 设计完全对齐
2. **box_dim=(32, 16)** print 跟 mismatch=0 互证 — dist encoding 编进 binary + functionally write 出预期 byte sequence
3. **(h) hybrid 0.36 / 97.6% wrong 真根因** = K reg dist (BWarpDstrEncoding embed at policy:585-616) 跟 trivial tile-major dram dist 缺 coordination
   - B1 5D dram dist 跟 reg dist outer encoding **隐式 co-designed** 让 LDS layout 一致
   - Step 1 改 dram dist trivial tile-major 但 reg dist 不动 → ds_load 用 reg dist 算 thread-i bottom_index → 落到 byte X，但 byte X 实际 element 不是 reg dist 期望的 (n_i, k_i) → element permute (97.6% wrong, 量级对)
4. **Mentor 三次 ds_load_tr "hardware-fixed pattern" / "wmma 几何墙" / "K trivial tile-major 撞墙" 全 invalidate** — disambig 实证 LDS 物理是 plain row-major 没几何特殊性

### 跟 mentor `mentor-hardware-knowledge-fixed.md` §1 一致

mentor doc §1 已 self-correct: "LDS K-outer N-inner row-major (任何 stride OK 只要 per-lane 8 fp16 contiguous)"。本 disambig 实测 K LDS = plain row-major 完全 align (K side 用普通 ds_load 不是 ds_load_tr，所以 N-outer K-inner，即跟 V trload N-inner 不同方向，但都是 plain row-major 没几何特殊 pattern)。

---

## 5. Fix Path Recommendation

### 推荐 (α) — 改 K reg dist outer encoding

按 baseline doc §5 (α) 描述:
> 改 K reg dist outer encoding 让 thread mapping 匹配 plain row-major LDS layout

**实施 sketch**:
- 改 `MakeKRegTileDistribution` (`block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:585-616`) 的 `k_block_outer_dstr_encoding` (传给 `make_embed_tile_distribution_encoding` 的第一参数)
- 让其 warp split + per-warp wmma tile count + iter pattern 投影到 plain row-major LDS byte 位置 = wmma B operand 期望的 (n_i, k_i)
- BWarpDstrEncoding 内层 (lane wmma B mapping) hardware-fixed **不动**

**理由 prefer (α) over (γ)**:
1. **Disambig 已 verified TDM write 端干净 1:1 copy** (Case 1) — 改 dram dist 等于改一个已 verified 的部分，多一次 sync point 要重 disambig
2. **Reg dist outer encoding 的 software freedom 比 dram dist 大** (BWarpDstrEncoding inner hardware-fixed 但 outer 的 warp split + iter pattern 都 software-controlled)
3. **改 read-side 不影响 perf 路径上 TDM 的 box_dim / cluster size 等** (γ 改 dram dist 可能 perturb perf — 当前 sim 2577 vs B1 2539 ~1.5% 已经接近，γ 风险更高)
4. **概念上更"对症下药"** — mismatch 是 reg dist 算错 thread 在 LDS 的位置，直接修 reg dist 比绕一圈改 dram dist 更直接

**工作量**: 1-2d (mentor sign-off baseline)
**Risk**: LOW
**Verify protocol**: 实施后用现有 K LDS dump plumbing **再跑一次** (dump 不变，但 reg dist 改后从 LDS 读出来的 thread-i 的 element 应该匹配 expected (n_i, k_i)) → 早期 verify reg dist 投影正确再跑 ABC

### 备选 (γ) — 改 K dram dist hybrid

按 baseline doc §5 (γ): warp split kN0 + thread access pattern align B1 reg dist 隐含 wmma-friendly stride。
- **工作量**: 1-2d
- **Risk**: LOW-MEDIUM (write-side 改可能 perturb perf)
- **使用条件**: 如果 (α) 中 reg dist outer encoding software freedom 不够覆盖 BWarpDstrEncoding inner 的约束 → 退到 (γ) 用 dram dist 适配

### 不推荐 (μ) / (ν)

- **(μ)** double LDS shuffle stage: 2-3d + perf 不确定 (可能 < B1)。Disambig 显示 TDM 干净，没必要加 shuffle stage 救场
- **(ν)** 撤退到 B1: ZERO 工作量但放弃 TDM 进展。Disambig 显示有清晰 fix path，没必要撤退

### V 同性质问题

V dram dist (B1 5D) + V reg dist (BWarpDstrEncoding embed via TransposedDstrEncode helper) 跟 K 同性质。step 4 V trivial tile-major 改 dram dist 没改 reg dist 也是 coordination broken。
- (α) 通了之后 V 也 likely 走类似 path
- 总工作量 (α-revised K + V) = 2-4d

---

## 6. Mentor Sync Status

发了 mentor message 同步 disambig 结果 + α vs γ 讨论 (含 implementation sketch + mapping question)，等 mentor confirm 走 α 还是 γ。

**待 mentor confirm 后**:
1. 报 lead 一句话 + 本 doc abs path
2. Lead 走 1h 自决流程或 user 拍方向 (方向性决策走 §3 lead prompt 协议)
3. 用户/lead 拍方向后开 implement

---

## 7. Methodological Reflection (initial — partially valid)

**Empirical disambig 价值实证**:
- Mentor 之前 3 次 sign-off (基于错 ds_load_tr semantics + GEMM v1 untested + box_dim verify ≠ functional verify) 全 invalidate by 一个 ~3h dev 的 K LDS dump
- 本次跟之前 3 次最大不同 = 第一次 sign-off 前**写出 specific empirical predict + plan disambig step** (mentor `mentor-hardware-knowledge-fixed.md` §6 Rule 2)
- 三方 (mentor + SWE + reviewer) 共识 + user 一开始坚持 disambig 的方向 → 准

**避坑 reference**:
- 5 种 disambig 情况列表 (mentor 之前给的) 帮助快速 categorize 实测结果
- box_dim print + mismatch=0 互证 vs box_dim 单独 verify 不够 (第二次 sign-off 错的教训)
- ds_load_tr semantics 走 production code (B1 async_trload_policy) 而不是 first-principle 推 (mentor doc §2 Rule 1)

---

## 8. Mentor Challenge & B1 Dump Plan (post-mentor-review addition)

### Mentor 4 个 critical facts grep 出来 (sub-mentor reanalysis)

1. **B1 K reg dist == B2 K reg dist** (literal 同代码: `async_trload_policy:581-611` 跟 `tdm_policy:597-624`)
2. **B1 K LDS write view == B2 K LDS write view** (literal 同代码 `MakeKLdsBlockDescriptor<Problem>` Xor=false default, 跟 read view `<Problem, false, true>` Xor=true)
3. **K LDS read view Xor=true 进 XOR transform branch** (`tdm_policy.hpp:340-381`):
   - case A: `LDSLayerSize = 256/2 = 128`, `XorLengthFold = 128/32 = 4 > 1` → 进 XOR'd branch
   - reader (ds_load) 算 thread t coord (n', k') → byte 时**不是** `(n'*kK0+k')*2`，是 XOR'd 位置
4. **B1 dram dist 5D 跟 B2 trivial tile-major 都用 LDS write view (Xor=false plain row-major desc)** — async_load 跟 TDM 都走这个 desc 决定 byte 位置

### Paradox

- writer 都写 plain row-major byte 位置 (B2 mismatch=0 实测 confirm)
- reader 用 XOR'd byte 位置访问 (read view Xor=true)
- → reader 拿到的不是它期望的 K[n'][k']，而是 K[XOR'd(n', k')]
- 按这个推理 **B1 也不应该 work**，但 B1 valid:y empirical
- → 必有一个我们没看到的 mechanism

### 我之前 disambig 的 implicit assumption (错)

| 我 verified | 我没 verified |
|---|---|
| ✅ TDM 1:1 byte copy plain row-major | ❌ B1 async_load 写 LDS 是不是也是 plain row-major |
| ✅ box_dim print 编进 binary | ❌ XOR'd read view 真实 access pattern 是什么 byte 位置 |
| | ❌ "B1 LDS layout == B2 LDS layout" implicit assumption |

→ "Case 1 confirmed → 走 (α)" 跨度太大。**(α) 路径成立的核心前提 = "B1 reg dist work → B1 reg dist 适配 B1 LDS layout → 改 reg dist 适配 B2 LDS layout"** — 这要求 **B2 LDS layout = B1 LDS layout (同 plain row-major)** 才成立。如果 B1 LDS XOR'd != B2 plain 那 (α) 等于设计新 outer encoding 投影 plain row-major LDS 同时 BWarpDstrEncoding hardware-fixed 给出正确 wmma B operand thread mapping — 理论可能但实际可行性需要 trace make_embed_tile_distribution_encoding 内部确认。

### B1 K LDS dump 比对实验 (~1-2h SWE work)

**Plan**:
1. `git worktree add` 在 B1 commit (`16324795215`) 开第二 worktree (B2 当前 worktree 不动保 9 edit 状态)
2. Apply LDS dump plumbing only (kargs 字段 + propagation + pipeline.hpp K LDS dump code 插在 B1 K async_load_tile + 等价 wait 后 + runner readback compare)
3. Build + run case A with `CK_DUMP_K_LDS=1`
4. 比对 dump_buf vs plain row-major formula `K_dram[n*d+k]`

**Decision matrix**:
- **mismatch=0**: B1 LDS = plain row-major = B2 LDS → 不是 LDS layout 问题，去找 async_load_tile vs load_tile_tdm 内部 mechanism difference (可能 async 内部 auto-derive XOR'd write，或 LDS dist auto-derive)
- **mismatch != 0 + XOR pattern**: B1 LDS XOR'd != B2 plain → (α) 路径要重设 outer encoding 投影 plain (不简单) **OR** 转 (γ) 让 TDM 写 XOR'd byte 顺序 match B1 layout
- **mismatch != 0 + 其它 pattern**: 深 trace 那 pattern 是什么 mechanism 产生

### Mentor 提供帮 sketch B1 line range (我已 ack 接受 offer)

等 mentor 给 B1 commit (16324795215) 时 K LDS dump 该插的 line range (B1 时 pipeline.hpp K async_load_tile call site / kernel.hpp single-buffer FmhaPipeline{} call site / fmha_fwd.hpp + runner propagation line) 后开 implement。

### Reflection (post-challenge)

我之前犯的错：
- **Sign-off chain inheritance**: 假设 B1 baseline LDS layout = B2 LDS layout 而没 explicit verify (类似 mentor 之前 3 次 sign-off 错的 pattern — 假设 inheritance)
- **跨度太大的 conclusion**: "TDM 1:1 byte copy → fix path is reg dist" 这步 inference 跨过了 "B1 LDS layout == B2 LDS layout" 这个未 verify 的 assumption
- **Predict 不够 specific**: 只 predict "TDM write → plain row-major"，没 explicit predict "B1 baseline → plain row-major" + verify

按 `mentor-hardware-knowledge-fixed.md` §6 标准 这是个真漏洞 — mentor 这次 catch 是 reviewer/mentor protocol 真正起作用的 case。Lucky 跟 mentor 内部讨论 catch 到了，没直接报 lead 走 (α) implement 撞墙。

---

file: swe-status-K-LDS-dump-disambig.md (revised post-mentor-challenge)
