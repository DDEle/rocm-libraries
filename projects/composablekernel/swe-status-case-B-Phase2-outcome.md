# SWE Status — Case B Phase 2 Disambig Outcome

**写作时间**: 2026-05-06 (Step B2 (λ-Q + Q padding disable) → case A+C pass / case B partial fail → Phase 2 batch dump for case B)
**Worktree**: `~/workspace/rocm-libraries-gfx1250/projects/composablekernel`
**Status**: ⚠️ Phase 2 unexpected outcome — Q broken case-B-specific, mask boundary hypothesis invalidated, **awaiting mentor mechanism trace** before fix path

---

## 1. TL;DR

Phase 2 batch dump (S+P+O at first + boundary n_block iter) reveals **first divergence at Q itself**, not at downstream mask/softmax stages as predicted。

| dump | case A (last verify) | case B Phase 2 |
|---|---|---|
| Q | 32/32 ✓ | **2/32** ✗ |
| K iter0 / K last | 32/32 ✓ | 32/32 ✓ |
| S first / P first / O first | n/a | 2/32 each (mirror Q error) |
| S boundary | n/a | 0/32 (GEMM 累积放大) |
| P boundary | n/a | 32/32 (mask saturation 巧合 cancel Q error) |
| O boundary | n/a | 2/32 (mirror) |

→ Downstream stages all mirror Q's 2/32 pattern (pure propagation, not new bug)
→ All mask/softmax/GEMM-downstream hypotheses **invalidated**

## 2. Killer signal: off-by-one lane shift

QA hint: `B2 tid 2 first 5 elements == B1 tid 1 first 5 elements`

→ (λ-Q) Q dist makes B2 lane N read B1 lane (N-1)'s data — **lane index shifted by 1** specifically in case B's multi-head config (h_q=2 / h_k=1)
→ Case A (h=1) doesn't trigger shift, passed 32/32
→ Case B (h=2 GQA) triggers shift, fails 2/32

## 3. Reasoning gap acknowledged (anti-pattern recurrence)

Mentor + SWE 推理:
> "Q dist 完全 by template params (BlockFmhaShape compile-time), head info 是 runtime kargs"
> "block 0 head 0 Q tile content **identical** between case A and case B"

**WRONG** — Missed:
- kernel.hpp Q dram window construction may use h_q/h_k for offset/stride
- GridSize for multi-head case dispatches blockIdx differently
- (λ-Q) `IsWarpLevelParallelOnly=true` may interact wrong with GQA dispatch
- Compile-time dist + runtime dispatch chain are separable concerns

**Recurrence of anti-pattern**: 第二次 inheritance assumption variant (前一次是 K↔Q analogy 单 reader-side analogy 跨过 writer + dist + padding 多 layer chain; 这次是 single-layer compile-time dist 推理跨过 multi-layer dispatch chain)。

## 4. ✅ Mechanism FULLY PROVEN by triangulation (mentor trace + 3 evidence)

### 🔥 Smoking gun: `get_cached_global_strides()` ignores actual stride

**tile_window.hpp:1786-1789**:
```cpp
cached_global_strides_ = to_array<...>(
    transform_tuples([](auto x) { return max(x / Traits::PackedSize, index_t{1}); },
                     tuple_reverse(container_reverse_inclusive_scan(
                         glb_tensor_descriptor.get_lengths(), multiplies<>{}, 1))));
```

**Critical**: 用 `get_lengths()` (= shape) 做 `container_reverse_inclusive_scan` 算 cached stride — **completely IGNORES actual tensor view stride field** (kargs.stride_q)。

### Triangulation evidence

| Evidence | Source | Case A | Case B |
|---|---|---|---|
| **kargs.stride_q formula** | fmha_fwd_runner.hpp `stride_q = (i_perm ? hdim_q : nhead_q * hdim_q)` | 1*128=128 | 2*128=**256** |
| **cached_global_strides[0]** | tile_window.hpp:1786 = `tuple_reverse(scan(get_lengths(), *, 1))` = shape[1] = hdim_q | 128 | 128 |
| **cached vs actual match?** | derived | ✓ match | ✗ **mismatch** (cached=128, actual=256) |
| **Q dump empirical** | QA Phase 2 case B | 32/32 (case A 上轮) | **2/32 with off-by-one lane shift** |

→ For case A: cached==actual → TDM byte offset correct → Q correct
→ For case B: cached(128) != actual(256) → TDM 用 cached 当 stride 读 dram → 每 m row 跳 128 byte 不是 256 byte → 读到 head 1 element instead of head 0 next m → off-by-one head shift

### B2 tid 2 == B1 tid 1 perfect mechanism match

- B2 lane 2 期望读 m=2 head=0 (offset 2*256 = 512 byte)
- 但 cached stride=128 → actual offset 2*128 = 256 byte = m=1 head=0 location (BSHD)
- = exactly what B1 lane 1 reads correctly → **shifted by 1 m row = 1 lane 错位** ✓

### Historical context (B2 trial η repeat)

B2 trial η/Hβ/Hγ 当年探 `cached_global_strides cumulative product != real stride` 但**只在 case A 测**, dismissed because case A cached==actual。**没在 case B 测** → 这个 cached vs actual mismatch 是真 latent bug 这次 case B 暴露。

→ Same anti-pattern recurrence: "single-case verify pass = mechanism not present" assumption

## 5. Fix path candidates (待 lead 拍方向)

This is **CK core latent bug** affecting any TDM use with non-packed-stride tensor。fmha 是 first TDM user with multi-head → first to hit。

### Option α (CK core proper fix)
- 改 `get_cached_global_strides()` 真用 actual tensor view stride, not shape-based default
- Clean architectural fix
- **Risk**: affects ALL TDM consumers (GEMM v1/v2 + future), 需 careful regression test
- **Workload**: 1-2d (core fix + regression sweep)
- **Best for**: long-term proper solution

### Option β (fmha-local view transform workaround)  
- 在 Q dram view 构造时做 `transform_tensor_view` 把 multi-head stride bake 成 shape — single-head 视图的 stride 变 packed default 跟 cached 一致
- Doesn't touch CK core
- **Workload**: ~1d (need view transform design + verify)
- **Risk**: medium (transform 算法 case-by-case, e.g. how does i_perm interact)

### Option γ (TDM disable for multi-head Q, keep K/V TDM)
- Q 走 async_load (B1 path) 当 h_q != 1; K/V 仍 TDM
- K/V 跟 multi-head 无关 (K/V indexed by h_k not h_q)
- **Workload**: ~30min (conditional dispatch in pipeline.hpp + remove (λ-Q) for h_q != 1 path)
- **Risk**: LOW (B1 path verified work for multi-head Q)
- **Trade-off**: lose TDM perf benefit on Q for multi-head case (Q is one-time load, perf impact small)
- **Best for**: fast functional ship, defer proper fix

## 6. Recommended next step

**Option γ (workaround)** — fast practical functional ship:
- ~30min implement + verify ABC (case A keeps current TDM Q, case B falls back to async Q for multi-head)
- Ships ABC全 valid:y today
- (λ-Q) + Q padding disable + (β')-Q-fix all keep for h_q=1 path (case A/C)
- Option α/β leave to follow-up Step C (perf phase) when have time for proper CK core fix

Mentor leans Option γ for milestone fast-path + Option α for long-term proper fix。我同意。

## 7. Process improvements (additional lessons from this Phase 2)

7. **Compile-time analysis ≠ runtime kargs coverage** — Q dist 是 compile-time but kargs.stride_q 是 runtime, 决定 dram view stride。"head-agnostic compile-time" inference 跨过 runtime stride layer。**Mentor sign-off "compile-time analysis sufficient" 必须列 runtime kargs 影响 (stride/offset/seqlen/h_q/h_k)** before claim case-agnostic。
8. **Latent CK core bug 通过 fmha 第一次暴露** — fmha 是 first TDM user with multi-head, 暴露 ck_tile/core latent assumption (cached strides = packed default)。GEMM TDM 没多 head 不 trigger。
9. **Single-case verify pass ≠ mechanism not present** — η trial 当年只 case A 测 cached vs actual, dismissed because match。应该用 case differing in stride_q (即 multi-head case B) 测 — 同 pattern 这次 又 catch 到。

## 5. Worktree state (preserved for fix dispatch)

- (λ-K) trivial tile-major K dram dist + (β-K) K LDS read view Xor=false + K LDS desc Xor template removed (Task A cleanup)
- (λ-Q) trivial tile-major Q dram dist + (β'-Q) Q LDS read view Xor=false + Q LDS desc Xor template removed
- Q padding disabled (mirror K/V GetLdsPaddingConfig pattern)
- K register tile dump + Q register tile dump + Phase 2 (S/P/O) dumps (CK_PRINTF_WARP0 instrumentation)
- B1 worktree (`rocm-libraries-gfx1250-b1dump`) has matching dump instrumentation for diff baseline

## 6. Open questions (待 mentor trace 后 updated)

1. Where does kernel.hpp construct Q dram window for case B? Same code path as case A or h_q-conditional branch?
2. blockIdx.{x,y,z} encoding for h_q=2: which dim encodes head_id vs m_block_id?
3. Why does (λ-Q) trivial tile-major dist produce off-by-one lane shift specifically when h_q != h_k?
4. Is reviewer's NWarp R 维 caveat now relevant in revised framing? (Earlier demoted because NWarp compile-time same across ABC — but maybe interacts with multi-head dispatch in subtle way)

## 7. Fix path candidates (待 mechanism reveal 后 evaluate)

待 mentor trace 出来根据 mechanism 选:
- **(λ-Q-multihead-fix)** if dispatch层 small offset issue: 1-line kernel.hpp tweak or (λ-Q) dist sequence reorder
- **(λ-Q-redesign)** if dist fundamentally wrong for GQA: rewrite (λ-Q) with GQA-aware encoding
- **(ν' partial)** if too complex: revert (λ-Q) for case B path only, keep for case A (different dispatch dispatching) — likely not feasible structurally
- **Pivot ship A+C** if fix is hard: per Option 1 framework, ship A+C clean + B as known issue follow-up

---

## 8. Process improvements (lessons learned, updated post-Phase-2)

1. **Stage-level fix verify must direct dump** (not max_err signal) — already in `feedback_stage_level_verify.md` memory
2. **Signal value type matters** (sentinel vs permute vs shift) — already noted post-Q-dump
3. **Multi-stage co-design lesson** — apply fix to all symmetric stages (Q padding asymmetric was missed cleanup)
4. **Compile-time analysis ≠ runtime dispatch coverage** — NEW lesson from Phase 2 outcome
5. **Cross-case verify is not optional** — case A pass doesn't prove case B will pass for same kernel template; runtime args (h_q/h_k/seqlen/mask_type) trigger different `if constexpr` and dispatch paths that compile-time analysis may not cover
6. **Mentor predict 措辞要分 PARTIAL/FULL FAIL/NO-OP** (mentor self-reflection) — already noted

---

file: swe-status-case-B-Phase2-outcome.md (in flight, updated post-mentor-trace)
