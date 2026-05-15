# Mentor Post-Compact Baseline — gfx1250 FMHA TDM Step B2/C

**写作时间**: 2026-05-06
**目的**: Mentor 自己 compact 后读回这一份立即恢复完整 state, 不需要 re-trace SWE message history
**Cross-link**: 跟 `mentor-hardware-knowledge-fixed.md` (hardware fact + sign-off anti-pattern), `B2-resume-baseline-v3.md` (B2 trial timeline), `mentor-reanalysis-ds-load-tr-correction.md` (ds_load_tr correction history)

---

## 1. 当前真实 state (post-compact 立即恢复用)

**Step B2 接近完成**:
- ABC case A + C 已 valid:y (post (λ-Q) + Q padding disable + (β')-Q-fix XOR=true→false + (β) K read view XOR=true→false)
- Case B (h_q=2/h_k=1 GQA causal s_k=257) 仍 fail max_err 0.062, **smoking gun 已找到** = `tile_window get_cached_global_strides()` shape-based ignores actual stride → multi-head case stride 256 ≠ packed default 128 → TDM 读 dram off-by-one head shift
- 当前在 implement γ1 dispatch (multi-head Q 走 async_load 不走 TDM), with γ1.5/γ2 fallback ladder

**Step C backlog (lead 即将 dispatch)**:
1. **CK core fix `get_cached_global_strides`** (`tile_window.hpp:1779-1794`) — 真改用 actual tensor view stride 不用 shape-based default
2. **(λ-Q) GQA-aware Q dist redesign** — multi-head Q dist 设计支持 stride-aware TDM
3. (其它 2 项 lead 没列具体 — 等 lead dispatch detail)

---

## 2. Hardware fact baseline (rule-grounded, 不要 first-principle 推)

### 2.1 ds_load_tr_b128 真实 semantics

- **Mnemonic**: `DS_LOAD_TR16_B128` (gfx1250) / `ds_read_tr_b128` (CDNA4)
- **wave32 only**, EXEC ignored
- **Per-lane I/O**: 128-bit (8 fp16/bf16) → 4 consecutive VGPR/lane
- **Wave-level total**: 32 lane × 8 fp16 = 16×16 fp16 tile

**Transpose direction (关键)**:
- **Input LDS**: K-outer N-inner row-major (V LDS desc 是 (kK1=32, kN1=128) row-major, K stride = kN1 byte, N stride = sizeof byte)
- **Output VGPR**: K-inner per-lane vector tile (给 wmma B operand 期待的 K-vector)
- = **N↔K swap** in transpose

PDF §4.9.10.2 verbatim: "B-matrix loads are similar: one lane loads multiple contiguous **N-values** along single **K-dimension index**"

### 2.2 K LDS layout (跟 V 不同!)

- **K side** (QK GEMM, K=B operand): **普通 ds_load** (不是 ds_load_tr)
- K LDS naive: `MakeKLdsBlockDescriptor<Problem>` plain (kN0=64, kK0=32) row-major **K-inner**
- K LDS K-inner natural match wmma B operand K-vector → 不需要 hardware transpose

→ **K LDS K-inner row-major**, **V LDS K-outer N-inner row-major** — different by design (V 经 ds_load_tr transpose)

### 2.3 BWarpDstrEncoding 是 pure tile_distribution_encoding

`warp_gemm_attribute_wmma.hpp:116-123` (wmma 16×16×32 fp16/bf16):
```cpp
using BWarpDstrEncoding = tile_distribution_encoding<sequence<kRepeat>, tuple<...>, tuple<...>, ...>;
```
**纯 element-coord encoding** — describes lane-to-element coord mapping for wmma B operand。**完全跟 LDS storage byte layout 解耦**。LDS 是 storage, wmma operand 是 register input — XOR 只 storage 端 bank-conflict 优化。

→ wmma B operand 不 care LDS bytes 是 XOR'd 还是 plain, 只 care reader (ds_load) 给它 element 序列正确。

### 2.4 XOR transform 实施

`coordinate_transform.hpp:1346-1361`:
```cpp
idx_low(0) = idx_up(0);
idx_low(1) = idx_up(1) ^ (idx_up(0) % up_lengths(1));
```
Pure reversible coord transform — 不改 element 含义, 只改 (n, k) → byte mapping。

### 2.5 TDM (Tensor DMA) hardware semantics

- Op: `TENSOR_LOAD_TO_LDS` 5D box-major copy from dram to LDS
- **EXEC ignored** — wave-level instruction, 1 TDM unit per SIMD-pair
- Tracking counter: TENSORcnt → `S_WAIT_TENSORCNT N`
- Box-major write: dram element (i, j) → LDS element (i, j) (relative to box start)
- LDS coord 起点 = `lds_window_origin + window_adaptor_thread_coord.get_bottom_index()` (`tile_window.hpp:881-882`)

**TDM 不支持 swizzle write** (per `~/.claude/memory/reference_mi450_kernel_patterns.md`)

### 2.6 `s_wait_tensorcnt_barrier<N>()` semantics

`arch.hpp:1175-1180`:
```cpp
template <index_t tensorcnt = 0, index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_wait_tensorcnt_barrier()
{
    s_wait_tensorcnt<tensorcnt>();   // wait TDM done
    block_sync_lds<lgkmcnt>();       // = s_waitcnt_barrier (=s_waitcnt + s_barrier)
}
```
**已含 s_barrier** — wait 后 LDS 全 thread visible, dump/load 安全, 不需额外 block_sync_lds。

### 2.7 `get_cached_global_strides()` (THE BUG — 真根因 #2)

`tile_window.hpp:1779-1794`:
```cpp
cached_global_strides_ = to_array<index_t, Base::NDimBottomTensor>(
    transform_tuples([](auto x) { return max(x / Traits::PackedSize, index_t{1}); },
                     tuple_reverse(container_reverse_inclusive_scan(
                         glb_tensor_descriptor.get_lengths(), multiplies<>{}, 1))));
```

**用 `get_lengths()` (= shape) 算 stride, 完全 IGNORES actual tensor view stride field**。

For Q dram view (kernel.hpp:2367-2370):
- shape (1023, 128) — case A 跟 case B 同
- actual stride: case A=128 (packed), case B=256 (multi-head h_q=2)
- cached_global_strides: shape-based packed default → case A 跟 case B 都 = 128
- For case B → cached (128) ≠ actual (256) → TDM 读 dram 每 m row 跳 128 byte 不是 256 byte → 读 head 1 m=0 instead of head 0 m=1 → off-by-one head shift

**B2 trial η/Hβ/Hγ 当年 explore 过这个 hypothesis 但只 case A 测 dismissed** (case A coincidence cached==actual). Case B 暴露真 latent bug。

---

## 3. B2 case B 两 root cause 完整理解

### Root cause #1: Q LDS asymmetric padding

- `GetLdsPaddingConfigQ` returns `(true, pad_amount=3, pad_interval=5)` ENABLED
- `GetLdsPaddingConfigK/V` returns `(false, 0, 0)` DISABLED (per λ-1/λ-2 disambig 历史遗留)
- → Q TDM writes WITH padding awareness into Q LDS desc that's plain row-major (no padding bytes)
- → Symptom: tid 1 全 sentinel value -23.203 (padding byte positions never written)
- **Fix applied**: `GetLdsPaddingConfigQ` → `return make_tuple(number<false>{}, number<0>{}, number<0>{});` mirror K/V

### Root cause #2: cached_global_strides bug (case B specific)

- See §2.7 above
- Symptom: `B2 tid 2 first 5 elem == B1 tid 1 first 5 elem` (off-by-one lane shift)
- **Fix in flight (γ1)**: kernel.hpp 算 `bool use_tdm_for_q = (kargs.nhead_ratio_qk == 1)`, pipeline 接 bool param, multi-head case 走 async_load 绕过 cached_strides bug
- **γ1 risk**: trivial tile-major dist + IsWarpLevelParallelOnly=true + async_load **untested combo** (~30% fail probability)
- **Fallback ladder**: γ1 fail → γ1.5 (revert dist only) → γ2 (full B1 path for multi-head)

### Cross-validation 关键 signal type

- Mentor sign-off error #4 (β')-Q-fix wrong: 推 K↔Q analogy reader-side, 没 verify writer-side coverage
- SWE catch sentinel signal 跟 chunk-swap signal 是不同 root cause — sentinel = writer coverage gap, chunk-swap = XOR mismatch
- → **Signal value type (sentinel/permute/garbage/all-zero) 是 critical disambig info**, 不是 binary "fix work / not work"

---

## 4. 4+ Sign-off anti-pattern (must internalize, list explicit before next sign-off)

### 4.1 Sign-off error 1 (predecessor mentor): (λ) sign-off based on GEMM v1 untested reference
- Recommended trivial tile-major mirror GEMM v1 但 GEMM v1 build-gfx1250 0 instance (untested design)
- **Anti-pattern**: 把 untested reference design 当 verified reference

### 4.2 Sign-off error 2 (predecessor): (h) hybrid box_dim verify ≠ functional verify
- Box_dim print (8,2)→(32,16) deterministic 看做 "K 改生效"
- **Anti-pattern**: dist encoding 编进 binary ≠ functional verify; LDS dump empirical 才算

### 4.3 Sign-off error 3 (predecessor): ds_load_tr semantic 完全反向描述
- 推 "K-inner LDS required" 没意识 transpose 整个意义就是让 LDS 跟 VGPR layout 反向
- **Anti-pattern**: first-principle 推 hardware semantic 没看 production code

### 4.4 Sign-off error 4 (current mentor, 我): (β')-Q-fix 推 reader-only fix
- 推 K↔Q analogy 只 verify reader 层 (Xor=true→false), 没 cross-check writer 层 (K writer async_load XOR'd byte vs Q writer TDM plain byte)
- **Anti-pattern**: K↔Q analogy 跨过 signal type level (writer-side coverage missing)

### 4.5 Sign-off error 5 (current mentor): "Q content head-agnostic compile-time" reasoning
- 推 Q dist 完全 compile-time → block 0 head 0 Q content identical case A 跟 case B
- 跨过 **runtime kargs.stride_q layer** (case A=128, case B=256 multi-head)
- **Anti-pattern**: "compile-time analysis sufficient" 必须列 runtime kargs 影响 layer

### 4.6 共同 root pattern

**All 5 errors 同性质**: 理论 reasoning 没配 empirical verify; inheritance assumption 跨过 verify layer。

**Process improvement after error #5**: Mentor sign-off 必须 explicit list verify items:
- Reader side ✓
- Writer side ✓
- Dist (compile-time) ✓
- Padding setup ✓
- Coverage gap ✓
- Runtime kargs (stride/offset/seqlen/h_q/h_k) ✓
- Co-design assumption (e.g. dist + read view co-evolved historical) ✓

---

## 5. 6 Protocol rules (must follow exactly)

### Rule 1: Hardware semantic 描述前先看 production verified code
- 先 grep B1 verified path (`block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`)
- 看 actual production layout choice — ground truth
- 再 PDF/spec verify 一致性
- **不 first-principle 推**

### Rule 2: Sign-off 前写 specific empirical predict + plan disambig
- Write specific predict (PARTIAL/FULL FAIL/NO-OP/SUCCESS 不同 outcome implication)
- Plan disambig step (LDS dump / single-iter test / specific case 跑哪个 valid 反应方向)
- 不依赖 "理论合理" 单独支撑 sign-off

### Rule 3: 重视 reviewer/SWE/QA challenge
- User/SWE/QA challenge 时第一反应是 verify 自己前提，不是 push back
- Reviewer second opinion 价值大

### Rule 4: 别把 GEMM v1 当 verified reference
- GEMM v1/v2 build-gfx1250 都 0 instance untested
- mirror 时必须明说 "untested reference, mirror 撞墙是 inevitable risk"

### Rule 5: box_dim/dist encoding verify ≠ functional verify
- box_dim print 是 constexpr from dist encoding — only confirms 编进 binary
- functional verify 要 LDS dump + 数据流 valid:y

### Rule 6: 任何 dist 改动配 LDS dump baseline
- 改 dram dist 之前先 LDS dump baseline B1 layout
- 改 dram dist 之后再 LDS dump 比对
- empirical 数据说话，不依赖 "理论上应该 work"

---

## 6. SWE↔Mentor 协议

### 6.1 我是 SWE 的 primary discussant
- SWE default 走 mentor 讨论技术实现细节，不报 lead
- 跟 SWE reach recommendation 后 SWE 自己报 lead 一句话
- **我不单独 cc lead** — 跟 SWE 讨论是 internal pair programming

### 6.2 我只单独 ping lead 这 3 类
1. SWE 跟我讨论后僵局没收敛
2. 我判断方向需用户介入 (大决策, 方案级转向)
3. SWE 失联或方向跑偏

### 6.3 不主动找 SWE
- SWE 主动 ping 我才答, 我不主动 push design suggestion / 不催 lead

### 6.4 回答风格
- 简洁: 重点指出 hardware constraint 或 reference 出处
- 给具体 line number 或 spec 段号, 不含糊
- 不确定的事直说 "不确定, 建议查 X", 不编

---

## 7. Step C backlog (lead 即将 dispatch, 我 prep work)

### 7.1 CK core fix `get_cached_global_strides`

**Scope**: `include/ck_tile/core/tensor/tile_window.hpp:1779-1794`

**Issue**: 用 `get_lengths()` 算 stride, 不用 actual tensor view stride

**Proper fix sketch**: 改 `glb_tensor_descriptor.get_lengths()` → 真用 `glb_tensor_descriptor.calculate_offset(...)` 或类似 API 拿 per-dim stride。需研究 `tensor_descriptor` 接口 to find proper stride accessor。

**Risk**: 影响所有 TDM consumers (GEMM v1/v2 + fmha)。需 careful regression test。GEMM v1/v2 是 2D pure GEMM, packed stride trivial 大概率不影响; fmha case A 也 packed trivial 不影响; fmha case B + 其它 multi-head consumer 是 fix beneficiary。

**My prep**: 
- Study tensor_descriptor API 看怎么 query actual stride (vs from-shape default)
- Identify all callers of `get_cached_global_strides()` (likely tdm_load_to_lds + similar TDM dispatch)
- Plan symmetric LDS dump verify per-stride correctness across both case A (packed) + case B (multi-head)

### 7.2 (λ-Q) GQA-aware Q dist redesign

**Scope**: `tdm_policy.hpp:99-122` (λ-Q) trivial tile-major Q dist

**Issue**: 当前 dist 假设 packed contiguous, 不 aware multi-head stride。即使 cached_global_strides fix 了, dist 自己 element distribution 算法可能也假设 packed default。

**Investigation question**: dist 算 thread-i element coord 时是否跟 actual stride 解耦? 如解耦, dist 不用改; 如 dist 内部 hardcode packed assumption, 也要改。

**My prep**:
- Trace dist projection 算法看 element coord 计算用 shape 还是 stride
- 如 dist 跟 stride 解耦 → 只 fix cached_global_strides 即可
- 如 dist 也用 stride → dist redesign 需 explicit 接收 stride info

### 7.3 工作流约定

- Lead dispatch 后, SWE 是 primary implementer, mentor 是 design discussant
- 我先帮 SWE 深 trace tensor_descriptor API + dist projection 算法 (Read-only)
- SWE 写 fix, mentor review (sign-off 列 6 verify items)
- QA 跑 ABC + 多 stride scenario regression test

---

## 8. Worktree state + key files

**Branch**: `yiding12/gfx1250-fmha-tdm`
**HEAD**: 16324795215 (B1 commit) + B2 modifications unstaged

**Modified files (per 我 last audit)**:
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` — pipeline 主体
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` — policy (Q dist + Q LDS desc + padding configs)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` — Q dram view (else branch L2360+)
- `include/ck_tile/core/container/container_helper.hpp` + `tuple.hpp` — (X+) CK core latent fix (independent PR value)
- 其它 plumbing files (CMakeLists, codegen)

**Key files for Step C work**:
| Path | What |
|---|---|
| `include/ck_tile/core/tensor/tile_window.hpp` L1779-1794 | `get_cached_global_strides` THE BUG |
| `include/ck_tile/core/tensor/tensor_descriptor.hpp` | tensor_descriptor API (need to find stride accessor) |
| `include/ck_tile/core/tensor/tensor_view.hpp` | Where stride is stored at tensor view level |
| `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` L99-122 | (λ-Q) Q dist for multi-head redesign |
| `include/ck_tile/core/arch/amd_tdm_descriptor.hpp` | TDM descriptor stride fields (tensor_dim0/1_stride) |

**B1 verified path reference** (don't modify, ground truth):
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`
- B1 5D scatter Q dist verified work for both single-head + multi-head

---

## 9. Memory references

`~/.claude/memory/`:
- `MEMORY.md` — index
- `reference_mi450_hw_specs.md` — wave32 / LDS / wmma 规格
- `reference_mi450_kernel_patterns.md` — TDM swizzle 不支持 + 通用 pattern  
- `reference_ck_tdm_api.md` — TDM descriptor 4-group SGPR 字段 + builtin 签名
- `project_gfx1250_fmha_fwd_design.md` — GFX IP team baseline
- `project_gfx1250_fmha_tdm_v1.md` — Step B1 完成 state

PDF refs (`~/.claude/skills/rocm-ref/p4vdoc/`):
- `mi400_shader_programming.pdf` §4.7.2.4 (DS_LOAD_TR), §4.9.10 (WMMA Matrix Load Transpose), §4.10 (Tensor DMA), §4.7.1 (LDS bank organization)
- `mi400_tensor_dma.pdf` (TDM detail)

---

## 10. Compact 后第一件事 checklist

1. ✅ **Read this doc first** — restore state
2. ✅ Read `mentor-hardware-knowledge-fixed.md` (full hardware fact + anti-pattern reflection)
3. ✅ Read `B2-resume-baseline-v3.md` (B2 trial timeline)
4. Skim worktree git diff (`git diff include/ck_tile/`) 看当前 modification state
5. SendMessage team-lead ack ready

任何 Step C technical question 来:
- 先 grep B1 verified path
- Skill(skill="rocm-ref") cross-check ISA spec
- 配 specific empirical predict (per Rule 2 + 6 verify items list)
- 不 first-principle 推, 不绕过 disambig

---

(end)
