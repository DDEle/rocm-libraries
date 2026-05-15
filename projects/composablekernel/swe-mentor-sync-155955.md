# Mentor Sync — D1 反转，根因不在 CK core，转挖 fmha K dram view

Lead 反转方向：D1 CK_PRINT 显示 K dram lengths 第 2 维是 **`constant<0>`** 不是 `constant<128>`。这不是 pad_tensor_view 升常量这么简单——`constant<0>` 像 placeholder，根因可能在 fmha 自己的 K dram view / dist 创建路径。

请你帮一起判断"修法是不是在 K dram view 创建侧而不是 CK core inclusive_scan"。

## D1 CK_PRINT 实测结果（QA 编 `compile-stepB2-D1-155511.log`）

| Window | dram tensor lengths type |
|---|---|
| Q | `tuple<int, constant<128>>` ← pad_tensor_view 升 hdim_q=128 为常量（正常） |
| **K** | **`tuple<int, constant<0>>`** ← 第 2 维 constant<0>！异常 |

## Lead 提的 4 个问题

### 1. K dram window 创建链路

3 层：

a) **fmha kernel 层** (`fmha_fwd_kernel.hpp:1638-1650`)：
```cpp
const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
    k_ptr,
    make_tuple(kargs.seqlen_k, kargs.hdim_q),  // 全 runtime int
    make_tuple(kargs.stride_k, 1),
    number<FmhaPipeline::kAlignmentK>{},
    number<1>{});

constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
return pad_tensor_view(
    k_dram_naive,
    make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),  // pad align to (64, 32)
    sequence<kPadSeqLenK_, kPadHeadDimQ>{});  // pad on which dim
```

b) **fmha pipeline make_tile_window** (`block_fmha_pipeline_qr_ks_vs_tdm.hpp:284-287`)：
```cpp
auto k_dram_window =
    make_tile_window(k_dram_block_window_tmp,
                     {physical_seqlen_k_start, 0},
                     Policy::template MakeKDramTileDistribution<Problem>());
```

c) **fmha policy MakeKDramTileDistribution** (`block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp:152-178`)（B1 抄 async_trload）：
```cpp
constexpr index_t kBlockSize = Problem::kBlockSize;       // wave32 → 128
constexpr index_t kNPerBlock = kN0 = 64;
constexpr index_t kKPerBlock = kK0 = 32;                  // not hdim, just one K-iter
constexpr index_t MaxVectorSize = 16/2 = 8;
constexpr index_t ElemPerThread = 64*32/128 = 16;
constexpr index_t K1 = min(8, 16) = 8;
constexpr index_t K0 = 32/8 = 4;
constexpr index_t N2 = 32/4 = 8;
constexpr index_t N1 = 128/32 = 4;
constexpr index_t N0 = 64/(8*4) = 2;

return make_static_tile_distribution(
    tile_distribution_encoding<sequence<1>,
                               tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,  // ((2,4,8), (4,8))
                               tuple<sequence<1>, sequence<1, 2>>,
                               tuple<sequence<1>, sequence<2, 0>>,
                               sequence<1, 2>,
                               sequence<0, 1>>{});
```

### 2. `unmerge<constant<0>, constant<4>, constant<8>>` 来源？

从 dist encoding，第二维 sequence<K0=4, K1=8> 算的是 (4, 8) — 但 trace 是 **3 维 (0, 4, 8)** 多出来一个 constant<0>。

候选解释：
- **(a)** pad_tensor_view 引入 right_pad，pad amount = `constant<0>`（hdim_q=128 已 align kK0=32 整除，pad=0），unmerge 把 (original_runtime, pad=0) 跟 dist 的 (4, 8) 合并成 (constant<0>, constant<4>, constant<8>)？
- **(b)** distribution 的 transform chain 自己引入的 placeholder？
- **(c)** pipeline 里 LoadOnce branch 的副作用？我们走的是 LoadOnce=false（K-iter loop），但 group path 用 LoadOnce=true

不确定，需要你看 ck_tile transform 推导规则。

### 3. 为什么 B1 async_load 没炸

async_load 路径不调 `get_cached_global_strides()`：
- async load → `tile_window.async_load_raw` → buffer instr 直接用 base + offset，不需要 dim strides
- TDM load → `tdm_load_to_lds` → `get_cached_global_strides` → `container_reverse_inclusive_scan` → 撞 mixed tuple in-place 限制

K dist 的 constant<0> 一直存在，只是 async path 不读 lengths 所以从未触发。

### 4. 修法在哪？— 待你判断

候选方向：

**(α) 修 K dram view 创建侧** — 让 K lengths 第 2 维变 runtime int 或 constant<128>
- 改 fmha_fwd_kernel.hpp:1639 K naive 创建（影响所有 fmha pipeline）
- 或改 pad_tensor_view 调用方式

**(β) 修 K distribution** — 让 transform chain 不引入 constant<0>
- 重写 MakeKDramTileDistribution（B1 抄 async_trload，B1 教训说不要乱改）
- 或局部 override 让 hdim 维 length 推导不引入 placeholder

**(γ) 仍走 (X) 修 CK core** — 不论 constant<0> 还是 constant<128>，都是 mixed tuple 触发 in-place scan TODO，CK core fix 也能解
- 但根因可能不在 CK core，"修核心库绕过自身 bug" 是错路径
- Lead 已搁置 (X)

**(δ) 在 fmha pipeline 局部 rebuild K dram view** — 拿 raw dram pointer 重建一个 lengths 全 runtime 的 view 给 TDM 用
- 不改 dist / kernel 层
- 风险：双 view 同步 + distribution 关联

## SWE 倾向

不知道 (α)/(β)/(γ)/(δ) 哪个对。我已陷入 transform 推导细节（trace 里 unmerge 含 constant<0> 的来源没看明白）。

**两个具体问题请你解答**：

1. K dram window 第 2 维 `constant<0>` 是 fmha **K dist 设计意图**（placeholder 表示"runtime 接管"）还是 **bug**（不该出现的伪长度）？
2. 同样 K dist 配 async_trload 实际跑过吗？如果跑过且 work（hdim 推导走另一路径），那 constant<0> 是设计 OK 但只跟 TDM path 不兼容；如果 async_trload 也是死代码（gfx1250 上从未实例化），那 constant<0> 根本没被压力测过——可能就是 bug。

## 当前 worktree 状态

- HEAD: B1 commit `16324795215` (clean)
- Unstaged dirty:
  - 3 plumbing（lead 加，不动）
  - 2 B2 真修复（pipeline + policy，本身正确）
  - 1 处 CK_PRINT diagnostic（D1 探针，待回退）

(X) CK core fix 暂搁。等你判断方向。

---
file: swe-mentor-sync-155955.md
