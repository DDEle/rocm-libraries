# investigator 报告 — fmha_fwd_kernel.hpp K-dram unmerge 公式 typo 调查

工作目录: `/home/yiding12/workspace/rocm-libraries/projects/composablekernel`
HEAD: `b3bdc63a5095fe4d5e9b0b2dfc1231af4514e3d1` (== origin/develop)

---

## 结论 (TL;DR)

**确认 develop 上有同一个 typo bug**。原公式
`number<FmhaPipeline::kQKHeaddim / kDramTileK / FmhaPipeline::kAlignmentK>{}`
对所有 fp16/bf16 hdim=128/64 case 都得到 0 或不等于 hdim 的 product，
**多除了一次 `/ kAlignmentK`**。

应改为：
`number<FmhaPipeline::kQKHeaddim / kDramTileK>{}`

修改面：lambda `make_k_dram` 在 batch 和 group 模式下都会进，**不是 group-only 路径**，而是被 `kIsGroupMode` 之外的公共代码段直接调用（line 2725 无 mode 判断）。

之所以只在 TDM 路径暴露：async_load 不实际 traverse 这个 outer-tile 维（length=0 被默默忽略，stride 已正确），TDM 必须按描述符遍历整段 hdim → 撞 `tuple<int, constant<0>>`。

---

## 1. develop 上的代码 (Q1)

文件: `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`
（grep 仅此一文件命中 `kQKHeaddim / kDramTileK`）

### lambda `make_k_dram` 定义位置

- **2605–2722** —— `const auto make_k_dram = [&](const KDataType* data, index_t height) { ... };`
- **2725** —— 调用 `return make_k_dram(k_ptr, kargs.seqlen_k);`（无 batch/group 分支，**两种模式共用**）
- 上面 2618–2619 处定义：
  ```cpp
  constexpr auto kDramTileK =
      FmhaPipeline::kKLoadOnce ? FmhaPipeline::kQKHeaddim : FmhaPipeline::kK0;
  ```

### 三处带 typo 的 `kQKHeaddim / kDramTileK / kAlignmentK`

```
2692:  number<FmhaPipeline::kQKHeaddim / kDramTileK / FmhaPipeline::kAlignmentK>{},  // unmerge dim 0
2705:  number<FmhaPipeline::kQKHeaddim / kDramTileK / FmhaPipeline::kAlignmentK>{}   // pass_through 配套
2715:  number<FmhaPipeline::kQKHeaddim / kDramTileK / FmhaPipeline::kAlignmentK>{},  // merge dim 0 (反向)
```

它们组成一个 unmerge → xor_permute → merge 的对称三段：

```cpp
// Step 1: unmerge hdim → (a, b, c)
make_unmerge_transform(make_tuple(
    number<kQKHeaddim / kDramTileK / kAlignmentK>{},   // a  ← typo
    number<kDramTileK / kAlignmentK>{},                // b
    number<kAlignmentK>{}))                            // c

// Step 2: XOR permute (height, b)
make_xor_transform(make_tuple(height, number<kDramTileK / kAlignmentK>{}))

// Step 3: merge (a, b, c) → hdim   ← 必须与 Step 1 严格一致
```

`kIsGroupMode` 的 dispatch 点（仅供参考）：line 1376, 1536, 2359；它们都早于 line 2605 的 lambda 定义，且 lambda 不依赖 `kIsGroupMode`。

---

## 2. git blame / 引入历史 (Q2)

`git blame -L 2680,2720 include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`：

- **bac01261813d** (Haocong WANG, 2025-08-13)：原始 2-tuple 版本 `(kQKHeaddim/kAlignmentK, kAlignmentK)`，product = kQKHeaddim ✓
- **2cc0af6a815a** (Haocong WANG, 2025-09-23, PR **#2888** "[CK_TILE] FMHA FWD bug fix")：把 2-tuple 改成 3-tuple；**这就是 typo 引入点**

### commit 2cc0af6a815a 干的事

1. 在 `block_fmha_pipeline_qr_ks_vs_async_trload.hpp` 加上 `static constexpr bool kKLoadOnce = BlockFmhaShape::kM0 >= 64;`（之前没有 kKLoadOnce 概念）
2. 在 `fmha_fwd_kernel.hpp` 加上 `kDramTileK = kKLoadOnce ? kQKHeaddim : kK0;`
3. 把 `else (XorLengthFold == 1)` 分支的 K dram unmerge/xor/merge 三步全部从 2-tuple 改成 3-tuple，引入 typo
4. 删掉 `example/ck_tile/01_fmha/script/fmha_fwd_known_fails_gfx950.txt` 里 4 条 hdim=128 的 batch-mode async_trload known-fail 测试

### commit message 解读

- 标题: "[CK_TILE] FMHA FWD bug fix (#2888)"
- 子提交 message: "tempsave debug" / "fix the bug in fmha fwd_kernel" / "Remove unnecessary changes" / "Fix the buggy part" / "remove fmha fwd known failure cases"
- **没有解释为什么三除**——纯口语 message，没说明 unmerge 公式的几何意义

### 后续相关 commit

- **ea56057c19cc** (Qianfeng, 2025-11-24, PR #3271 "Fix a bug for qr_ks_vs_async_trload pipeline")
  把 `kKLoadOnce = kM0 >= 64` 改成 `kM0 > 64`；同时改了 `PrefillCase` 阈值。**没改 typo**。

---

## 3. 数学验证 (Q3)

### 几何意图（推导）

3-tuple unmerge 想把 hdim 拆成 `(#tiles, #aligned-blocks-per-tile, alignment)`：

| 维 | 想要的语义 | 想要的长度 |
| -- | -------- | -------- |
| a  | 一个 hdim 里有几个 dram tile | `kQKHeaddim / kDramTileK` |
| b  | 一个 dram tile 里有几个 alignment 块 | `kDramTileK / kAlignmentK` |
| c  | 一个 alignment 块的元素数 | `kAlignmentK` |

product 应当 = `(kQKHeaddim/kDramTileK) * (kDramTileK/kAlignmentK) * kAlignmentK = kQKHeaddim` ✓

XOR permute 取 `(height, b)`，意图是在每个 dram tile 内部沿 width 方向做 bank-conflict 反交错（b 维大小 = `kDramTileK/kAlignmentK`）。这也跟意图一致。

### 当前 (typo) 公式 product 实算

公式 `(kQKHeaddim / kDramTileK / kAlignmentK, kDramTileK / kAlignmentK, kAlignmentK)`
注意 C++ 的整数除法：`a/b/c` ≡ `(a/b)/c`。

| Pipeline 配置 | hdim | kDramTileK | kAlignK | a | b | c | product | vs hdim |
| ------------ | ---- | ---------- | ------- | - | - | - | ------- | ------- |
| fp16 hdim128 kKLoadOnce=true  | 128 | 128 | 8 | 0 | 16 | 8 | **0**   | ❌ |
| fp16 hdim128 kKLoadOnce=false (kK0=32) | 128 | 32  | 8 | 0 | 4  | 8 | **0**   | ❌ |
| fp16 hdim128 kKLoadOnce=false (kK0=16) | 128 | 16  | 8 | 1 | 2  | 8 | **16**  | ❌ |
| fp16 hdim64  kKLoadOnce=true  | 64  | 64  | 8 | 0 | 8  | 8 | **0**   | ❌ |
| fp16 hdim256 kKLoadOnce=false (kK0=32) | 256 | 32  | 8 | 1 | 4  | 8 | **32**  | ❌ |

**所有列出的真实 fp16/bf16 配置都 wrong**。

### 修复后 product

改成 `(kQKHeaddim / kDramTileK, kDramTileK / kAlignmentK, kAlignmentK)`：

| 配置 | a | b | c | product |
| ---- | - | - | - | ------- |
| fp16 hdim128 kDramTileK=128 kAlignK=8 | 1 | 16 | 8 | **128** ✓ |
| fp16 hdim128 kDramTileK=32  kAlignK=8 | 4 | 4  | 8 | **128** ✓ |
| fp16 hdim64  kDramTileK=64  kAlignK=8 | 1 | 8  | 8 | **64**  ✓ |

### 是不是有别的 transform 把缺失维度补回来？

**否**。看 transform 链上下文（line 2687–2721）：

```
unmerge: hdim → (a, b, c) at sequence<1, 2, 3>
xor    : (height @0, b @1) → (height', b'); pass_through a, c
merge  : (a, b, c) → hdim
```

三步对称，merge 用同样的 (a, b, c) 把它合回单一维。**没有任何 transform 在外面补一个额外的 ×kAlignmentK 因子**。如果 unmerge 的 a 维 length=0，merge 的 a 维 length 也是 0，整段长度 product=0，与原始 hdim 不一致。这是一个纯 typo，不是有意压缩。

XOR_LENGTH_FOLD 分支（2628–2683）保持 2-tuple 版本不变，与新加的 kDramTileK 概念无关，那条路径没受影响。

---

## 4. 谁在跑这条代码（gfx950 / group mode 实际触达） (Q4)

### `make_k_dram` lambda 的可达性

- 定义在 line 2605，调用在 line 2725，无 mode 守卫 → **batch 和 group 都跑**。
- 进入 typo 三段的条件：`!(CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD && XorLengthFold > 1)`
  - `XorLengthFold = LDSLayerSize / kQKHeaddim`，`LDSLayerSize = 256 * PackedSize / sizeof(KDataType)`
  - fp16/bf16 (sizeof=2, PackedSize=1): LDSLayerSize=128, hdim=128 → XorLengthFold=1 → 走 typo 分支
  - fp16/bf16 hdim=64: LDSLayerSize=128, hdim=64 → XorLengthFold=2 → 走 XOR_LENGTH_FOLD 分支（不踩雷）
  - 所以 **fp16 hdim=128 的 K dram 必走 typo 分支**

### codegen 实际生成哪些 instance

`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` 1121–1142（gfx9 fp16/bf16）：

```python
if (hdim, hdim_v) in [(64, 64), (128, 128)] and ...:
    pipelines.append(FmhaFwdPipeline("qr_async_trload", "row", "f", "f", "f", "f", ...))
    pipelines.append(FmhaFwdPipeline("qr_async_trload", "row", "f", "f", "t", "t", ...))
    pipelines.append(FmhaFwdPipeline("qr_async_trload", "row", "t", "t", "f", "f", ...))  # group mode spad
```

第 3 行注释 "group mode spad" 直接确认：**fp16/bf16 + (128,128) + group_mode + qr_async_trload pipeline 是当前生成出来并发布的 instance**。

`fmha_fwd.py:1141` 的 `qr_async_trload` 群模式 hdim=128 fp16 instance 在 gfx9 全部架构 (gfx942/gfx950) 都会编出来。它们之所以**通过测试**：async load 路径（即 `BlockFmhaPipelineQRKSVSAsyncTrload`）只用 stride、不实际 traverse 这个 outer-tile 维 → length=0 被默默吃掉。

### group path 真实跑过的 pipeline tag 列表

只看 fp16/bf16 + (hdim,hdim_v)=(128,128) + group：实测只有 `qr_async_trload`。其它 tag (`qr`, `qr_async`) 在 gfx9 codegen 路径下不会用 (128,128)+group 触发 K-dram lambda 的 typo 分支（要么走 XOR_LENGTH_FOLD，要么 hdim 不到 128）。

### batch vs group 的差异

batch 和 group 在 `make_k_dram` 内部行为**完全一样**，唯一区别是上面 2359 处 group 模式重新算 `kargs.seqlen_k` 为单个 batch 里 cu_seqlen 的差。所以：

- **typo 不是 group-only**——batch fp16 hdim=128 也会编 typo IR
- 之所以以前 batch async_trload + hdim=128 在 gfx950 known_fails 列表上 4 行（commit 2cc0af6a815a 删掉），是因为 typo 引入前的 2-tuple 版本就有别的 bug；2cc0af6a815a 把 2-tuple 改成 typo 3-tuple 后，async path 因为不读 length 反而通过测试，known-fail 被删
- TDM 路径**会**读 length，所以 typo 直接撞 `tuple<int, constant<0>>`

---

## 5. 修复 safety + 测试覆盖 (Q5)

### 修法

3 处 `kQKHeaddim / kDramTileK / kAlignmentK` → `kQKHeaddim / kDramTileK`
（line 2692, 2705, 2715，全在 lambda `make_k_dram` 的 `else` 分支）

### gfx942 / gfx950 上是否 safe

**理论上 safe**：

- 修正后 outer-tile dim 从 length=0 变成 length=4 (hdim128/kDramTileK32) 或 1 (hdim128/kDramTileK128)；
- stride 计算（rightmost-first）：c stride 1 → b stride kAlignmentK → a stride kDramTileK，**修复前后 stride 完全一致**；
- async load 路径以前用 outer_idx=0 单点 access，没问题；修复后 a 维 length 变正，async 不会去 traverse 但 stride 不变，**等价性维持**；
- 唯一风险：如果 async pipeline 内部某处用 length 算 thread mapping / boundary check，从 0 改成正数可能触发以前没走过的代码。

### 推荐验证步骤（实修前）

1. `tile_example_fmha_fwd -prec=fp16 -mode=0 -d=128 -s=1024 -h=2 -b=2`（batch + async + hdim128）— 应当 pass，证明 stride 等价
2. `tile_example_fmha_fwd -prec=fp16 -mode=1 -d=128`（group + async + hdim128）— 应当 pass
3. 跑 `test/ck_tile/fmha/test_fmha_fwd.cpp` 全集（覆盖 batch+group, hdim 64/96/128/192）

### 测试覆盖现状

`test/ck_tile/fmha/test_fmha_fwd.cpp` 第 27–46 行 `TestConfigsGfx9`:

```cpp
static constexpr auto HDimValues = std::array{
    /*64,128,*/ std::tuple{96, 128}, std::tuple{128, -1}, std::tuple{192, 128}, ...};
static constexpr auto ModeValues = std::array{mode_enum::batch, mode_enum::group};
```

→ **gfx9 已经覆盖 hdim=128 + (batch & group)**。这条 group+hdim=128 fp16 测试在 develop 上跑过（应该是绿的，因为 async path 吞 length=0），所以修复后必须重新跑一遍验证仍然绿。

### 兜底建议

修复时同步加一个 `static_assert` 或注释，防止再次写错：

```cpp
static_assert(FmhaPipeline::kQKHeaddim % kDramTileK == 0,
              "kQKHeaddim must be a multiple of kDramTileK");
static_assert(kDramTileK % FmhaPipeline::kAlignmentK == 0,
              "kDramTileK must be a multiple of kAlignmentK");
```

并在 lambda 顶部加注释解释 (#tiles, #aligned-blocks-per-tile, alignment) 三层语义。

---

## 附 — 命令复现路径

```bash
git rev-parse HEAD                                      # b3bdc63a5095...
grep -n "kQKHeaddim / kDramTileK" \
  include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp   # 2692, 2705, 2715
git blame -L 2680,2720 \
  include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp   # bac01261813d, 2cc0af6a815a
git show 2cc0af6a815a                                   # 引入 typo 的 PR #2888
git show ea56057c19cc                                   # 后续 PR #3271，未改 typo
```
