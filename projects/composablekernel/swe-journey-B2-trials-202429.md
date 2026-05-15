# Step B2 trial journey — async_load → load_tile_tdm 替换的完整时间线

**用途**：B2 真重启时（不论用户拍 (λ)/(μ)/(ν)）让新会话快速 refresh，不必翻 inbox 历史。

**前置上下文**：
- Step A：fmha qr_tdm 脚手架编不过（distribution bug）—— 见 `~/.claude/memory/project_gfx1250_fmha_tdm_v1.md`
- Step B1：抄 async_trload K/V dist override 让 qr_tdm 用 async_load 跑通 ABC + perf -21~32%（已 commit `16324795215`）
- Step B2 任务：把所有 `async_load_tile(lds, dram)` 替换成 `load_tile_tdm(tdm_config, lds, dram)` —— 真用 TDM intrinsic

## Hypothesis chain 总表

| # | hypothesis | 来源 | verify 方法 | 结果 | 关键 status |
|---|---|---|---|---|---|
| 0 | B2 实施后能编通 | SWE | 跑 QA 编 1 次 | ❌ 64 instance fail，inclusive_scan in-place fail | `compile-stepB2-152615.log` |
| 1 | (α') kernel group path unmerge typo `H/T/A=0` | SWE D3 grep | 改算式 + D4 print 后 lengths | ✅ confirm 修对 (group path lengths constant<0> → constant<128>) | `swe-status-161300.md` |
| 2 | (X) CK core inclusive_scan in-place 不支持 mixed tuple | SWE D5 + mentor 16:04 confirm | 改用 recursive container_push_front | ✅ inclusive_scan substitution failure 消失 | `swe-status-163855.md` |
| 3 | (X+) CK core 4 处 in-place tuple ops 同 pattern | SWE scope check + mentor sign-off | 4 处改 generate_tuple mirror | ✅ 编通了 | `swe-status-165650.md` |
| 4 | H1 TDMConfig padding 字段单位错 | mentor (B2 风险预判 #3) | disable padding | ❌ runtime 无变化 | (H4' verify 中) |
| 5 | H4 / H4' LDS write naive vs read xor swizzle 不一致 | mentor 16:04 → 自我反证 | disable padding + disable xor 4 处 | ❌ runtime 几乎不变 (out -22.78 → -21.94, max err 23.31 → 23.72) | `compile-stepB2-alpha-verify-162808.log` |
| 6 | H8 cached_global_strides cumulative product != real stride | mentor 17:00 | (η) helper bypass + calculate_offset 算真 stride | ❌ runtime 16-digit 同 cached | `compile-stepB2-eta-A` |
| 7 | Hγ build cache stale 让 (η) 没真编进 binary | SWE + mentor 17:30 | dependent static_assert lazy trigger | ✅ confirm (η) 真实例化进 binary | `compile-stepB2-hgamma-173533.log` |
| 8 | Hβ TDM 不读 stride 字段 | SWE | hardcode `{99999, 99999}` | ❌ SIGSEGV → TDM **真读 stride[0]** (但 [1] 不依赖关键值) | `run-stepB2-hbeta-A-173852.log` |
| 9 | Hα stride 真值同 cached (cached [0]=128 巧合对) | SWE 推导 | 数学验证 (M*N, N) vs (N, 1) 在 [0] 同 | ✅ confirm (η) 跟 cached 16-digit 一致是因 [0] 同 | (Hβ verify 中) |
| 10 | Hδ-1 box_dim 错 | mentor 18:00 | print K/V box_dim | ❌ K=(8,2)/V=(8,8) 几何对 | `compile-stepB2-hdelta-174326.log` |
| 11 | **H3 升级版** TDM box-major LDS write vs B1 dist (async_load) ds_load read 不兼容 | mentor 18:30 (90%+ confirm) | GEMM v1 设计对照分析 + box_dim 几何 confirm + 排除其它 | ⭐ **strong confirm** root cause | `swe-status-174720.md` |
| 12 | (λ-test) 单改 `MakeKDramTileDistribution` trivial linear 看 valid | lead 18:50 auto-decide | mentor 实施前发现 3-dist coupling | ❌ task 取消 — 单改不能 disambig | `swe-status-lambda-test-185504.md` |

## 详细时间线（每轮 hypothesis 详情）

### Round 0: B2 启动 + 编译 fail
- **改动**: pipeline 11 处 async_load_tile → load_tile_tdm + sync 改 s_wait_tensorcnt_barrier<0> + policy 加 GetLdsPaddingConfig × 3 + TDMConfig 构造
- **结果**: 编译 fail `container_helper.hpp:337: y(i) = r;` 错误 — int 写到 const constant<128> slot
- **触发**: 60+ instance 全军覆没

### Round 1: (α') kernel group path unmerge typo 
- **来源**: SWE D3 grep 发现 `fmha_fwd_kernel.hpp:2570-2602` group path `make_k_dram` lambda 内 unmerge 算式
- **bug**: `kQKHeaddim/kDramTileK/kAlignmentK = 128/32/8 = 0` （多除一次）
- **fix**: 改 `kQKHeaddim/kDramTileK = 4`, layout 一致 4*4*8=128 ✓
- **D4 verify**: group path K dram lengths 从 `tuple<int, constant<0>>` → `tuple<int, constant<128>>` ✓
- **性质**: independent latent bug。fix 在 worktree，task #5 走独立 PR (qa-native 验证)
- **status**: `swe-status-161300.md`

### Round 2-3: (X) → (X+) CK core in-place tuple ops 不支持 mixed tuple
- **来源**: SWE D5 disambig confirm `tuple<int, constant<128>>` 触发 inclusive_scan in-place fail；mentor confirm scope = 5 处 (1 inclusive_scan + 4 tuple ops)
- **bug**: `tuple<Xs...> y; y[i] = r;` in-place pattern 不能写 runtime int 到 compile-time constant slot
- **fix**: 5 处全改 generate_tuple / recursive container_push_front mirror 已支持 mixed tuple 的 sister overload (line 705/727/772)
- **verify**: 编通了
- **性质**: independent latent bug
- **status**: `swe-status-165446.md`, `swe-status-165650.md`, `swe-status-163855.md`

### Round 4: H1 TDMConfig padding 字段单位
- **来源**: mentor B2 风险预判 #3 ("padding 单位 dword 还是 byte 不明")
- **verify (H4' 顺带)**: disable padding (3 GetLdsPaddingConfig 全 return false)
- **结果**: runtime 无变化 → padding 不是根因
- **status**: 包在 H4' verify 中

### Round 5: H4 / H4' LDS swizzle write/read mismatch
- **来源**: mentor 16:04 ("TDM 不支持 swizzle write，但 fmha LDS read 用 Xor=true → 一致 garbage")
- **verify**: 同时 disable padding (A) + disable xor 4 处 (B)
- **结果**: runtime 几乎无变化 (out -22.78 → -21.94, max err 23.31 → 23.72) — sim 5941 vs X+ 2536 perf 2.3x 退步证明改动生效但 functional 没变
- **结论**: ❌ swizzle 不是根因
- **status**: `compile-stepB2-alpha-verify-162808.log`

### Round 6: H8 cached_global_strides cumulative product != real stride
- **来源**: mentor 17:00 (深读 tile_window.hpp:1779 cached_global_strides 实现)
- **(η) 实施**: 加 `get_real_global_strides_for_tdm()` 用 `calculate_offset(unit_idx)` 算真 stride，3 处 TDM caller (line 875/1036/1286) 切换到新函数；prefetch_for_flat (line 1192) 保留 cached 版本
- **verify**: ABC 测 — runtime 16-digit 同 cached
- **结论**: ❌ stride helper 改动 0 runtime 影响

### Round 7: Hγ build cache 让 (η) 没真编进 binary
- **来源**: SWE 怀疑 + mentor 17:30 推荐 verify
- **verify**: 在 `get_real_global_strides_for_tdm` 加 dependent static_assert (`sizeof(typename Base::DataType) == 0`)
- **结果**: ✅ 编 fail 含 "INSTANTIATED" — `(η)` 真实例化进 binary
- **结论**: build cache 不是问题，(η) 真生效但 runtime 0 影响
- **status**: `compile-stepB2-hgamma-173533.log`

### Round 8: Hβ TDM 不读 stride 字段
- **来源**: SWE
- **verify**: hardcode `{99999, 99999}` 在 (η) 函数体
- **结果**: A 跑 SIGSEGV (exit 139)
- **结论**: TDM **真读 stride[0]** (= 128 in fmha case)，stride[1] 不依赖关键值
- **status**: `run-stepB2-hbeta-A-173852.log`

### Round 9: Hα stride 真值同 cached
- **来源**: SWE 推导（基于 Hβ 结果）
- **verify**: 数学验证
  - cached strides for K dram (s, 128) = (M*N, N)→reverse→(N, M*N) → array{128, 128*s}
  - (η) calculate_offset(unit_idx)→array{128, 1}
  - [0] 同 (=128) → TDM fetch dram offset 一致
  - [1] 不同但 hardware 不依赖关键值 → runtime 同
- **结论**: ✅ confirm (η) 跟 cached 16-digit 一致是 [0] 同
- **stride 字段已不是嫌疑**

### Round 10: Hδ-1 box_dim 错
- **来源**: mentor 18:00
- **verify**: 加 CK_PRINT 在 tile_window.hpp:885 print ys_to_d_descriptor.get_lengths() + raw_box_dim + box_dim
- **结果**: K box (8, 2) (16 elements/call) ✓ 跟 mentor 预期一致；V box (8, 8) (64 elements/call) ✓ V dist 几何也合理
- **结论**: ❌ box_dim 几何对
- **status**: `compile-stepB2-hdelta-174326.log`

### Round 11: **H3 升级版** — TDM box-major write vs B1 dist (async_load 设计) ds_load read 不兼容
- **来源**: mentor 18:30 (重新分析 + 排除剩余 hypothesis)
- **mechanism**:
  - GEMM v1: `BlockGemm::MakeABlockDistributionEncode()` **跟 TDM write pattern 一起设计** dist 解决
  - fmha B1 抄的 `MakeKDramTileDistribution` 是 **async_load 历史脚手架**，跟 TDM write pattern **不兼容**
  - TDM hardware DMA 写 LDS 是 box-major 顺序 (按 box_dim 8×2 contiguous box per thread per call，起点 lds_coord)
  - ds_load 按 thread dist pattern 期望 thread-i 在 LDS 位置 P
  - 两个 thread mapping 规则不同 → 一致地读到错位置 → garbage
- **confirm 程度**: 90%+ (mentor 不敢 100%，trial 历史让她保守)
- **status**: `swe-status-174720.md`

### Round 12: (λ-test) 取消
- **来源**: lead 18:50 auto-decide (用户 1h 窗口未回 → zero-cost disambig)
- **目标**: 单改 `MakeKDramTileDistribution` trivial linear 跑 A 看 valid
- **mentor 实施前发现**: fmha qr_tdm 实际 3 个 dist 互相 coupled (dram + LDS write + LDS read)，且 LDS read dist 跟 BlockGemm 0 A operand 强绑定
- **结论**: 单改不能 disambig；同改会破坏 BlockGemm 0 → 双重 garbage
- **task #7 已取消**
- **status**: `swe-status-lambda-test-185504.md`

## 6 条 fork-miss 教训 (B1+B2 累计)

1. **fork policy 检查所有 `Make*` / `Get*` override 函数列表 + diff 现有 override** (B1 教训)
2. **trial 编完跑完没变化时 verify binary 真重编了** (B1 教训 → B2 Hγ verify) — 用 dependent static_assert lazy trigger
3. **早点动态二分** (缩 d / s 看 single-iter vs multi-iter) — B2 Round 12 single-iter test 5 min disambig 比静态分析快
4. **未验证的代码注释 = 反指** (B1 教训) — `cached_global_strides` naming 误导是同性质
5. **同一编译错误改一次还在 → 拉 mentor**
6. **同一 case 被 QA 报 fail 第 2 次 → 拉 mentor**

## 当前 worktree 状态 (8 modified clean)

| 文件 | 改动 | 性质 |
|---|---|---|
| `CMakeLists.txt` | +14 -1 | 🟦 lead plumbing |
| `example/.../01_fmha/CMakeLists.txt` | +20 -5 | 🟦 lead plumbing |
| `example/.../01_fmha/codegen/ops/fmha_fwd.py` | +6 -3 | 🟦 lead plumbing |
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp` | +77 -25 | 🟢 B2 主线 (TDM intrinsic 替换 + sync) |
| `include/.../fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp` | +123 -0 | 🟢 B2 主线 (GetLdsPaddingConfig × 3) |
| `include/.../fmha/kernel/fmha_fwd_kernel.hpp` | +18 -7 | 🟡 (α') latent bug fix |
| `include/.../core/container/container_helper.hpp` | +44 -15 | 🟡 (X+) latent bug fix |
| `include/.../core/container/tuple.hpp` | +20 -8 | 🟡 (X+) latent bug fix |

🟦 lead 加；🟢 B2 主线（如选 (ν) 撤退要回退）；🟡 mentor confirmed latent bugs（任何方向都保留）

## 关键 status 文件 audit chain (按时间顺序)

1. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-mentor-sync-155955.md` — 早期 mentor sync K constant<0> 假说
2. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-155317.md` — D 进展 + 对照困境
3. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-155722.md` — D1 confirm + 提议 X
4. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-160732.md` — D2 反证 mentor pad_tensor_view，根因转 make_tile_window
5. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-161300.md` — D3 锁 (α') kernel group path unmerge typo + pipeline distribution 引入 constant<0>
6. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-163011.md` — α' 必要不充分 — 复活 X
7. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-163318.md` — Q vs K stacktrace 反转
8. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-163855.md` — D5 disambig case D5-b (Q=K 同 type 都 fail)
9. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-165446.md` — X 修对了但需扩 scope (operator-)
10. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-165650.md` — scope check 完 5 处
11. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-174720.md` — H3 升级版升级 lead/用户拍 λ/μ/ν
12. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-final-B2-await-user.md` — worktree 整理 await user
13. `/home/yiding12/workspace/rocm-libraries-gfx1250/projects/composablekernel/swe-status-lambda-test-185504.md` — λ-test push back 3-dist coupling

## 编译 / run log 时间序列

| time | log | content |
|---|---|---|
| 152615 | `compile-stepB2-152615.log` | B2 编 fail 64 instance |
| 155511 | `compile-stepB2-D1-155511.log` | D1 K dram lengths CK_PRINT |
| 160524 | `compile-stepB2-D2-160524.log` | D2 K naive vs padded type |
| 160935 | `compile-stepB2-D3-160935.log` | D3 batch vs group path 分裂 |
| 162510 | `compile-stepB2-alpha-D4-162510.log` | (α') D4 verify K group path 修对 |
| 162808 | `compile-stepB2-alpha-verify-162808.log` | (α' + X+) ABC 编通 + A valid:n |
| 163639 | `compile-stepB2-D5-163639.log` | D5 Q vs K post-α' 同 type |
| 165116 | `compile-stepB2-XplusAlpha-165116.log` | (X+) ABC 编通 + A valid:n |
| 170100 | `compile-stepB2-Xplus-170100.log` | X+ ABC verify (重复) |
| 170752 | `compile-stepB2-H4p-170752.log` | H4' (disable padding+xor) 编 |
| 170944 | `run-stepB2-H4p-A-170944.log` | H4' A valid:n |
| 171958 | `run-stepB2-singleiter-S1-171958.log` | single-iter s=64 d=128 仍 fail |
| 172006 | `run-stepB2-singleiter-S2-172006.log` | single-iter s=64 d=64 (实际 d=128+pad) 仍 fail |
| 172912 | `compile-stepB2-eta-172912.log` | (η) 编通 |
| 173009 | `run-stepB2-eta-A-173009.log` | (η) A valid:n 16-digit 同 H4' |
| 173533 | `compile-stepB2-hgamma-173533.log` | Hγ static_assert 触发 confirm 实例化 |
| 173757 | `compile-stepB2-hbeta-173757.log` | Hβ hardcode 99999 编通 |
| 173852 | `run-stepB2-hbeta-A-173852.log` | Hβ A SIGSEGV |
| 174326 | `compile-stepB2-hdelta-174326.log` | Hδ-1 box_dim print |

---
file: swe-journey-B2-trials-202429.md
