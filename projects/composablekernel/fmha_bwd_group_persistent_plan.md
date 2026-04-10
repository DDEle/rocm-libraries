# Plan: FMHA BWD Group Mode Deterministic Persistent Kernel

## Context

当前 FMHA backward pass 的 persistent kernel（将 GPU blocks 数固定为 CU 数，让每个 CU 自行遍历其分配的 tiles）只支持 batch mode + deterministic 组合。Group mode（变长序列，packed layout）的 deterministic 版本已经正常工作，但未使用 persistent 调度——每个 block 对应一个固定的 `(tile_n, head, batch)` tile，导致当 total tiles < num_CUs 时出现空转 CU，性能不佳。

目标：按照 [gist](https://gist.github.com/DDEle/9dd3ca37d3229a1617d87a438801a23b) 中的 `dispatch_algo` 算法，为 group mode deterministic 实现 persistent kernel 调度。

---

## 算法（来自 gist）

**核心思想**：用前缀和使每个 CU 能独立计算自己的工作区间，不需要跨 CU 通信。

**Workload 定义**：
- 每个 batch `b` 有 `num_chunks[b] = ceil(seqlen_k[b] / kN0)` 个 K-tiles
- 每个 chunk 的工作量 = `seqlen_q[b]`（Q 方向 tiles 数）
- 每个 (batch, head) pair 的 workload = `head_workload[b] = num_chunks[b] * seqlen_q[b]`
- 全局前缀和：`prefix_batch[b] = sum_{i<b}(nhead * head_workload[i])`

**CPU 侧预计算**（在 `PrepareWorkspaceHost` 中完成）：
- `target_w = ceil(prefix_batch[nbatch] / num_cus)` — 每个 CU 的目标工作量
- `cu_start_ibatch[cu_id]`：O(N_BATCH + C) 双指针扫描，记录每个 CU 从哪个 batch 开始

**GPU 侧（每个 CU 的 operator() 循环）**：
1. 计算 `w_lo = cu_id * target_w`, `w_hi = (cu_id+1) * target_w`
2. 从 `cu_start_ibatch[cu_id]` 开始遍历 batch
3. 对每个 batch 内计算 `head_start = (w_lo - prefix_batch[b]) / head_workload[b]`
4. 对每个 head，计算 `c_start`, `c_end`（chunk 范围），`isplit = ceil(wc_start / target_w)`
5. 对范围内每个 chunk_idx，调用 `run_(kargs, dim3(chunk_idx, head_idx, ibatch), isplit, nsplits[ibatch])`

**`isplit` 含义**：同一个 (batch, head) 的不同 chunks 被分配到不同 CU 处理；`isplit` 是该 CU 在 convert_dq 的 dq_acc 中写入的分片索引，保证不同 CU 写不同 slot，实现无冲突 deterministic 累加。

**`nsplits[b]` 的新含义**：`per_batch_max_cus[b]`——该 batch 中最多有多少个 CU 共享同一个 head（等于 convert_dq 中需要 reduce 的 split 数）：
```
nsplits[b] = 1 + ceil((num_chunks[b] - 1) * seqlen_q[b] / target_w)
```
来自 gist 第 98-101 行。

---

## 需要修改的文件

**唯一文件**：`include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`

其他文件（codegen、CMakeLists、example host code）不需要改动，因为 persistent vs 非 persistent 是 kernel 内部逻辑，对外接口不变。

---

## 详细改动

### 1. `kUsePersistent` 定义（2处）

**FmhaBwdDQDKDVKernel**（约第 231 行）：
```cpp
// 现在：
static constexpr bool kUsePersistent = kIsDeterministic && !kIsGroupMode && !kUseQrQtrDorPipeline;
// 改为：
static constexpr bool kUsePersistent = kIsDeterministic && !kUseQrQtrDorPipeline;
```

**FmhaBwdConvertQGradKernel**（约第 1882 行）：
```cpp
// 现在：
static constexpr bool kUsePersistent = kIsDeterministic && !kIsGroupMode;
// 改为：
static constexpr bool kUsePersistent = kIsDeterministic;
```

### 2. `FmhaBwdWorkspaceManager`：新增 CPU 侧调度表的 size 函数

在 group mode persistent 时，CPU workspace 中额外存两个数组：
- `index_t prefix_batch[nbatch+1]`
- `index_t cu_start_ibatch[num_cus]`

新增 size/offset 函数：
```cpp
static size_t GetPrefixBatchSize(int batch)
{
    if constexpr(kIsGroupMode && kIsDeterministic)
        return integer_least_multiple(sizeof(index_t) * (batch + 1), ALIGNMENT);
    return 0;
}
static size_t GetCuStartIbatchSize(int num_cus)
{
    if constexpr(kIsGroupMode && kIsDeterministic)
        return integer_least_multiple(sizeof(index_t) * num_cus, ALIGNMENT);
    return 0;
}
static size_t GetPrefixBatchOffset(int batch)
{
    return GetDqAccSplitsSize<false>(batch) + GetDqAccOffsetsSize(batch);
}
static size_t GetCuStartIbatchOffset(int batch)
{
    return GetPrefixBatchOffset(batch) + GetPrefixBatchSize(batch);
}
```

### 3. Kargs 结构体扩展

新增 group mode persistent 专用字段（条件包含于 `FmhaBwdGroupModeKargs`，仅当 `kIsGroupMode && kIsDeterministic`）：

```cpp
struct FmhaBwdGroupModePersistentKargs
{
    const index_t* prefix_batch_ptr;    // [nbatch+1], CPU workspace
    const index_t* cu_start_ibatch_ptr; // [num_cus], CPU workspace
};
```

`FmhaBwdGroupModeKargs` 继承该结构（条件性，类似其他 Kargs 的 conditional_t 模式）。

### 4. `PrepareWorkspaceHost` 扩展

在现有 group mode deterministic 分支中，计算完 `nsplits[i]` 和 `offsets[i]` 之后，追加：

```cpp
// 计算 prefix_batch 和 cu_start_ibatch（group mode persistent）
auto* prefix_batch = reinterpret_cast<index_t*>(
    reinterpret_cast<char*>(cpu_ws) + GetPrefixBatchOffset(batch_size));
auto* cu_start_ibatch = reinterpret_cast<index_t*>(
    reinterpret_cast<char*>(cpu_ws) + GetCuStartIbatchOffset(batch_size));

// 计算新的 nsplits[b] = per_batch_max_cus[b]
const index_t num_cus = get_num_cus();
prefix_batch[0] = 0;
for(index_t b = 0; b < batch_size; ++b)
{
    const index_t sq = seqstart_qs[b+1] - seqstart_qs[b];
    const index_t nc = integer_divide_ceil(seqstart_ks[b+1] - seqstart_ks[b], kN0);
    const index_t hw = nc * sq;
    prefix_batch[b+1] = prefix_batch[b] + nhead_q * hw;
}
const index_t target_w = integer_divide_ceil(prefix_batch[batch_size], num_cus);
// 重新计算 nsplits[b] 为 per_batch_max_cus[b]
for(index_t b = 0; b < batch_size; ++b)
{
    const index_t sq = seqstart_qs[b+1] - seqstart_qs[b];
    const index_t nc = integer_divide_ceil(seqstart_ks[b+1] - seqstart_ks[b], kN0);
    const index_t num_workload_rest = (nc > 0) ? (nc - 1) * sq : 0;
    nsplits[b] = 1 + (num_workload_rest > 0 ? integer_divide_ceil(num_workload_rest, target_w) : 0);
}
// 重新计算 offsets（nsplits 改变后 workspace 大小也变）
offsets[0] = 0;
for(index_t b = 0; b < batch_size - 1; ++b)
{
    offsets[b+1] = offsets[b] + static_cast<long_index_t>(nhead_q) * nsplits[b]
                   * (seqstart_qs[b+1] - seqstart_qs[b]) * hdim_q;
}
// 填写 cu_start_ibatch（双指针扫描）
index_t cu_lo = 0;
for(index_t b = 0; b < batch_size; ++b)
{
    const index_t cu_hi = min(num_cus,
        integer_divide_ceil(prefix_batch[b+1], target_w));
    for(index_t c = cu_lo; c < cu_hi; ++c)
        cu_start_ibatch[c] = b;
    cu_lo = cu_hi;
}
for(index_t c = cu_lo; c < num_cus; ++c)
    cu_start_ibatch[c] = batch_size; // empty CU sentinel
```

### 5. Group Mode `MakeKargsImpl` 扩展

在设置 `dq_acc_batch_offset_ptr` 之后追加：
```cpp
if constexpr(kUsePersistent)
{
    kargs.batch = batch;
    kargs.nsplits_ptr = reinterpret_cast<const index_t*>(
        ws + WorkspaceManager::GetDqAccSplitsOffset(batch));
    kargs.prefix_batch_ptr = reinterpret_cast<const index_t*>(
        ws + WorkspaceManager::GetPrefixBatchOffset(batch));
    kargs.cu_start_ibatch_ptr = reinterpret_cast<const index_t*>(
        ws + WorkspaceManager::GetCuStartIbatchOffset(batch));
}
```

### 6. `operator()` 扩展：Group Mode Persistent 循环

在现有 `kUsePersistent` 分支内，增加 `kIsGroupMode` 分支：

```cpp
else // kUsePersistent
{
    if constexpr(!kIsGroupMode)
    {
        // 现有 batch mode persistent 逻辑（不变）
        ...
    }
    else
    {
        // 新增：group mode persistent 逻辑（实现 gist dispatch_algo GPU 侧）
        const index_t cu_id  = blockIdx.x;
        const index_t num_cu = gridDim.x;
        const index_t nbatch = kargs.batch;

        const index_t total_w  = kargs.prefix_batch_ptr[nbatch];
        if(total_w == 0) return;
        const index_t target_w = integer_divide_ceil(total_w, num_cu);

        const index_t w_lo = cu_id * target_w;
        const index_t w_hi = min((cu_id + 1) * target_w, total_w);
        if(w_lo >= total_w) return; // 空 CU

        for(index_t ibatch = kargs.cu_start_ibatch_ptr[cu_id]; ibatch < nbatch; ++ibatch)
        {
            const index_t pb = kargs.prefix_batch_ptr[ibatch];
            if(pb >= w_hi) break;

            const index_t sq = kargs.seqlen_q_ptr
                ? kargs.seqlen_q_ptr[ibatch]
                : (kargs.seqstart_q_ptr[ibatch+1] - kargs.seqstart_q_ptr[ibatch]);
            const index_t sk = kargs.seqlen_k_ptr
                ? kargs.seqlen_k_ptr[ibatch]
                : (kargs.seqstart_k_ptr[ibatch+1] - kargs.seqstart_k_ptr[ibatch]);
            const index_t nc = integer_divide_ceil(sk, FmhaPipeline::kN0);
            const index_t hw = nc * sq; // head_workload[ibatch]
            const index_t nsplits_b = kargs.nsplits_ptr[ibatch];

            const index_t head_start = max(index_t(0), (w_lo - pb) / hw);
            for(index_t head_idx = head_start; head_idx < kargs.nhead_q; ++head_idx)
            {
                const index_t w_head = pb + head_idx * hw;
                if(w_head >= w_hi) break;

                const index_t wc_start = max(index_t(0), w_lo - w_head);
                const index_t isplit   = integer_divide_ceil(wc_start, target_w);
                const index_t c_start  = integer_divide_ceil(wc_start, sq);
                const index_t c_end    = integer_divide_ceil(min(hw, w_hi - w_head), sq);

                for(index_t chunk_idx = c_start; chunk_idx < c_end; ++chunk_idx)
                    run_(kargs, dim3(chunk_idx, head_idx, ibatch), isplit, nsplits_b);
            }
        }
    }
}
```

### 7. `PrepareWorkspaceDevice`：同步修改清零条件

```cpp
constexpr bool kUsePersistent =
    !kUseQrQtrDorPipeline && kIsDeterministic; // 移除 !kIsGroupMode
```

### 8. `run_()` 内部：dQ 写入策略

`DstInMemOp` 选择逻辑（不变）：
```cpp
constexpr auto DstInMemOp = conditional_expr<(kUseKSplit && !kUsePersistent)>(
    memory_operation_enum::set, memory_operation_enum::atomic_add);
```
- **非 persistent deterministic**：`set`——每个 block 独占一个 split slot（tile_n → isplit 一一对应）
- **persistent（batch 和 group）**：`atomic_add`——一个 CU 可能顺序处理同一个 `(ibatch, head, isplit)` 的多个 chunk，必须累加
- **非 deterministic**：`atomic_add`（单一共享 accumulator）

因此 group mode persistent 与 batch mode persistent 一样使用 `atomic_add`，dq_acc buffer 在启动前需清零。

---

## 关键约束与注意事项

1. **isplit 唯一性**：gist 算法保证同一 `(ibatch, head_idx, isplit)` 只被一个 CU 使用（test 1b）。
2. **nsplits 含义变化**：group mode 的 `nsplits[b]` 从 `ceil(sk/kN0)` 改为 `per_batch_max_cus[b]`，dq_acc workspace 变小。
3. **convert_dq 不变**：`convert_dq` 已经正确读取 `nsplits_ptr[i_batch]`，只要 `nsplits` 值正确即可工作。
4. **`prefix_batch` 单位**：是 nheads * head_workload，全局累积值，GPU 端 `pb = prefix_batch[ibatch]`。

---

## 验证方式

### Recipe 5 的限制条件（CMakeLists.txt `--receipt 5`）

默认编译只生成满足以下所有条件的 instances（`codegen/ops/fmha_bwd.py` 第 1116–1124 行）：
- `dtype == "fp16"`
- `bias == "no"`
- `dropout == "no"`
- `mode == "group"`
- `deterministic == "t"`

验证参数**必须**与此匹配，否则找不到对应 instance 会 fallback 或报错。

```bash
# 1. 编译（CMakeLists 默认使用 --receipt 5，生成 group+deterministic+fp16+no_bias+no_dropout）
cmake -S . -B build-gfx950 -GNinja --preset dev -DGPU_TARGETS='gfx950'
cmake --build build-gfx950 -j205 --target tile_example_fmha_bwd

# 2. CPU 验证——必须加 -deterministic=1，且 -h 不能太大（避免 OOM）
cd build-gfx950
./bin/tile_example_fmha_bwd -v=1 -mode=1 -b=4 -h=2 -s=512  -d=64  -prec=fp16 -deterministic=1
./bin/tile_example_fmha_bwd -v=1 -mode=1 -b=4 -h=2 -s=512  -d=128 -prec=fp16 -deterministic=1
./bin/tile_example_fmha_bwd -v=1 -mode=1 -b=8 -h=2 -s=256  -d=64  -prec=fp16 -deterministic=1 -mask=1

# 实测结果（gfx950, valid:y）:
# [fp16|group] b:4,h:2,s:512,d:64  no-mask  workspace:21MiB  valid:y
# [fp16|group] b:4,h:2,s:512,d:128 no-mask  workspace:47MiB  valid:y
# [fp16|group] b:8,h:2,s:256,d:64  causal   workspace:6MiB   valid:y

# 3. 性能测试（-v=0 禁用 CPU 验证）
./bin/tile_example_fmha_bwd -v=0 -mode=1 -b=4 -h=32 -s=4096 -d=64  -prec=fp16 -deterministic=1 -warmup=5 -repeat=20

# 4. 如需测试 mask / bf16 等其他变体，需要先修改 CMakeLists.txt 的 --receipt 或加 --filter，
#    或者直接用 FMHA_BWD_FILTER 过滤单个 instance 编译
```
