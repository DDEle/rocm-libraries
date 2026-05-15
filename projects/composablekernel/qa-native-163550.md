# qa-native — gfx950 native (α') 验证报告

**时间**：2026-04-30 16:35–17:25
**worktree**：`/tmp/ck-qa-native-163550/projects/composablekernel`
**branch**：detached at `b3bdc63a5095fe4d5e9b0b2dfc1231af4514e3d1` (develop HEAD)
**arch**：gfx950 (本机原生，8 卡，全空闲)
**HIP_VISIBLE_DEVICES**：0
**docker image**：`rocm/atom-dev:latest` (ROCm 7.2.2 / clang 22 / cmake 3.28)
**target**：`test_ck_tile_fmha_fwd` (umbrella，依赖 fp16/bf16/fp8bf16/mxfp8/mxfp4 五个 gtest 二进制)

> 注：team-lead 提到的 `test_fmha_fwd` 实际 target 名是 `test_ck_tile_fmha_fwd`。

## 结论

✅ **(α') patch on gfx950 native = 不回退，无 perf 偏移**

- 编译通过（无新 warning 阻塞 / 无 link 错）
- ctest pass/fail 数完全一致
- 失败 case 集合**完全相同**（同 114 gtest fail + 同 2 GPU memory access fault）
- 选定代表 case 延迟变化 ≤ 8%（噪声范围内，且方向混杂）
- 编译时长 baseline 950s vs α' 937s（-1.4%，无回归）

⚠️ **重要副发现**（与 (α') 验证目标无关，但需让 lead 知晓）：
develop b3bdc63a 的 gfx950 baseline **本身就有 fmha fwd group path 的 GPU memory access fault + 大量 verify failure**，集中在 `Alibi (group)` 与 `Dropout (group)` 两个 suite 的 fp16/bf16 变体；fp8bf16/mxfp8/mxfp4 全过。**(α') patch 不修复也不引入这些 failure**——它们是另一个 pre-existing bug。

## ctest target 级别对比

| Test target | Baseline | (α') | Δ |
|---|---|---|---|
| test_ck_tile_fmha_fwd_fp16 | SIGPIPE in 149.24s | SIGPIPE in 147.41s | -1.2% |
| test_ck_tile_fmha_fwd_bf16 | SIGPIPE in 143.91s | SIGPIPE in 138.86s | -3.5% |
| test_ck_tile_fmha_fwd_fp8bf16 | Passed in 36.53s | Passed in 36.32s | -0.6% |
| test_ck_tile_fmha_fwd_mxfp8 | Passed in 30.45s | Passed in 30.20s | -0.8% |
| test_ck_tile_fmha_fwd_mxfp4 | Passed in 43.03s | Passed in 40.59s | -5.7% |
| **总时间** | **403.19s** | **393.42s** | **-2.4%** |
| **pass/fail** | **3 pass / 2 fail** | **3 pass / 2 fail** | **same** |

`SIGPIPE` 是 ctest 报的，根因是 GPU memory access fault 直接 abort 了 gtest 进程（gtest 还没把所有结果写完就被杀）。两个 fail 的 target 在 baseline 与 α' 中**死在完全相同的位置**：`Dropout.FmhaFwdBf16/62, [bf16|group|bshd] b:4, h:3/1, s:100/768, d:96/128, p_drop:0.123, mask:b(-1:0)`。

## gtest 内层（在死掉前完成的 case 中）

| 类别 | Baseline | (α') |
|---|---|---|
| `[       OK ]` 总数 | 1520 | 1520 |
| `[  FAILED  ]` 总数 | 114 | 114 |
| GPU memory access fault | 2 | 2 |
| Alibi.FmhaFwdBf16 fail | 48 | 48 |
| Alibi.FmhaFwdFp16 fail | 48 | 48 |
| Dropout.FmhaFwdBf16 fail | 9 | 9 |
| Dropout.FmhaFwdFp16 fail | 9 | 9 |

所有失败 case 的 GetParam 集合 1:1 一致（用 `diff` 对比 sorted FAILED 行确认）。

## 选定代表 case 延迟对比（passing cases，越小越快）

| Case | Baseline | (α') | Δ |
|---|---|---|---|
| HDimPadding.FmhaFwdFp16/0 | 458 ms | 423 ms | -7.6% |
| HDimPadding.FmhaFwdBf16/0 | 257 ms | 266 ms | +3.5% |
| ElementwiseBias.FmhaFwdFp16/7 | 209 ms | 201 ms | -3.8% |
| ElementwiseBias.FmhaFwdFp16/40 | 219 ms | 208 ms | -5.0% |
| Dropout.FmhaFwdFp16/0 | 200 ms | 205 ms | +2.5% |

方向混杂、幅度 ≤ 8%——属于 ms 级 kernel 计时噪声，**无系统性 perf 偏移**。

## 编译时长对比

| | Baseline | (α') |
|---|---|---|
| `ninja test_ck_tile_fmha_fwd` 总时长 | ~950s | 937s |
| 编译单元数 | 8124 | 8119 |

(α') 是 incremental rebuild（仅触发受 `fmha_fwd_kernel.hpp` 影响的 TU 重编），但因为这个 header 是 instance generation 的根，受影响的 TU 几乎是全集，所以 wall time 接近全量。无回归。

## Patch diff

`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`，三处（line 2692/2705/2715 → 2692/2704/2713）：

```diff
                         make_unmerge_transform(
-                            make_tuple(number<FmhaPipeline::kQKHeaddim / kDramTileK /
-                                              FmhaPipeline::kAlignmentK>{},
+                            make_tuple(number<FmhaPipeline::kQKHeaddim / kDramTileK>{},
                                        number<kDramTileK / FmhaPipeline::kAlignmentK>{},
                                        number<FmhaPipeline::kAlignmentK>{}))),
```

```diff
                         make_pass_through_transform(
-                            number<FmhaPipeline::kQKHeaddim / kDramTileK /
-                                   FmhaPipeline::kAlignmentK>{}),
+                            number<FmhaPipeline::kQKHeaddim / kDramTileK>{}),
```

```diff
                         make_merge_transform_v3_division_mod(
-                            make_tuple(number<FmhaPipeline::kQKHeaddim / kDramTileK /
-                                              FmhaPipeline::kAlignmentK>{},
+                            make_tuple(number<FmhaPipeline::kQKHeaddim / kDramTileK>{},
                                        number<kDramTileK / FmhaPipeline::kAlignmentK>{},
                                        number<FmhaPipeline::kAlignmentK>{}))),
```

## 运行环境备注

- 走 sudo docker 进 `rocm/atom-dev:latest`（主机无 cmake/clang）
- 容器加 `--device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host --shm-size=16G`
- `--group-add render` 在主机用 name 失败（image 内无该 group），改用数字 GID `993` 即可

## 清理

worktree 已删除（`sudo rm -rf /tmp/ck-qa-native-163550` + `git worktree prune`）。所有原始日志文件随之清理；本报告内联了所有关键数据。如需复现，按 ## 运行环境备注 + ## Patch diff 重新跑即可。
