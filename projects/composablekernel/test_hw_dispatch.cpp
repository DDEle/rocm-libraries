// test_hw_dispatch.cpp
//
// 验证 HIP grid block ID 与硬件实际 dispatch 位置的对应关系。
// 每个 block 的第一个 thread 读取硬件寄存器，记录：
//   - blockIdx.x（逻辑 block ID）
//   - XCC_ID（哪个 XCD die）
//   - SE_ID（XCD 内的 Shader Engine）
//   - SH_ID（SE 内的 Shader Array）
//   - CU_ID（Shader Array 内的 CU）
//   - SIMD_ID（CU 内的 SIMD）
//   - WAVE_ID（SIMD 内的 wave slot）
//
// HW_ID 寄存器位域（GFX9 / CDNA3 / CDNA4，gfx9_shader_programming.pdf Sec 2.12）：
//   bits  3:0  = WAVE_ID   wave buffer slot
//   bits  5:4  = SIMD_ID   SIMD within CU
//   bits  7:6  = PIPE_ID   dispatch pipeline
//   bits 11:8  = CU_ID     Compute Unit within Shader Array
//   bit  12    = SH_ID     Shader Array within SE (0 or 1)
//   bits 14:13 = SE_ID     Shader Engine within XCD
//   bits 19:16 = TG_ID     Thread Group ID
//   bits 23:20 = VM_ID     Virtual Memory ID
//
// XCC_ID 寄存器（MI300+ 新增，gfx9_shader_programming.pdf Sec 14.2.4）：
//   S_GETREG_B32 Sn, hwreg(20)   → 当前 wave 所在的 XCD 编号
//
// 编译：
//   hipcc -O2 --offload-arch=gfx942 test_hw_dispatch.cpp -o test_hw_dispatch
//   hipcc -O2 --offload-arch=gfx950 test_hw_dispatch.cpp -o test_hw_dispatch
//
// 运行：
//   ./test_hw_dispatch [num_blocks] [threads_per_block]
//   ./test_hw_dispatch 1024 64

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <tuple>


struct BlockInfo
{
    int block_id;
    int xcc_id;
    int se_id;
    int sh_id;   // Shader Array within SE
    int cu_id;
    int simd_id;
    int wave_id;
    int pipe_id;
    int tg_id;   // Thread Group ID（即 blockIdx.x mod hardware TG slots）
    uint64_t clock;  // 开始执行时的硬件时钟
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int WAVES_PER_SIMD>
__attribute__((amdgpu_waves_per_eu(WAVES_PER_SIMD, WAVES_PER_SIMD)))
__global__ void query_hw_placement(BlockInfo* out, int num_blocks, int delay_iters)
{
    int block_id = (int)blockIdx.x;

    // 所有 thread 都跑 delay，保持 wave slot 占用直到 kernel 结束
    uint64_t t = wall_clock64();
    for(int i = 0; i < delay_iters; i++)
        asm volatile("s_nop 15" ::);

    // 只有 thread 0 读硬件寄存器并记录结果
    if(threadIdx.x == 0) {
        uint32_t hw_id = 0;
        uint32_t xcc_id = 0;

        asm volatile("s_getreg_b32 %0, hwreg(4)"  : "=s"(hw_id));
        asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(xcc_id));

        BlockInfo info;
        info.block_id = block_id;
        info.xcc_id   = (int)(xcc_id & 0xFF);
        info.wave_id  = (int)((hw_id >> 0) & 0xF);
        info.simd_id  = (int)((hw_id >> 4) & 0x3);
        info.pipe_id  = (int)((hw_id >> 6) & 0x3);
        info.cu_id    = (int)((hw_id >> 8) & 0xF);
        info.sh_id    = (int)((hw_id >> 12) & 0x1);
        info.se_id    = (int)((hw_id >> 13) & 0x3);
        info.tg_id    = (int)((hw_id >> 16) & 0xF);
        info.clock    = t;

        out[block_id] = info;
    }
    __builtin_amdgcn_s_barrier();
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

#define HIP_CHECK(expr)                                                    \
    do                                                                     \
    {                                                                      \
        hipError_t err = (expr);                                           \
        if(err != hipSuccess)                                              \
        {                                                                  \
            fprintf(stderr,                                                \
                    "HIP error %s at %s:%d\n",                            \
                    hipGetErrorString(err),                                \
                    __FILE__,                                              \
                    __LINE__);                                             \
            exit(1);                                                       \
        }                                                                  \
    } while(0)

int main(int argc, char* argv[])
{
    int num_blocks        = (argc > 1) ? atoi(argv[1]) : 512;
    int threads_per_block = (argc > 2) ? atoi(argv[2]) : 64;
    int waves_per_simd    = (argc > 3) ? atoi(argv[3]) : 8;
    int delay_iters       = (argc > 4) ? atoi(argv[4]) : 0;

    // ---- device info ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("gcnArchName: %s\n", prop.gcnArchName);
    printf("multiProcessorCount (CU count): %d\n", prop.multiProcessorCount);
    printf("\n");
    printf("num_blocks=%d  threads_per_block=%d  waves_per_simd=%d  delay_iters=%d\n\n",
           num_blocks, threads_per_block, waves_per_simd, delay_iters);

    // ---- alloc ----
    BlockInfo* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(BlockInfo)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(BlockInfo)));

    // ---- launch ----
    auto launch = [&](auto kernel) {
        hipLaunchKernelGGL(kernel,
                           dim3(num_blocks),
                           dim3(threads_per_block),
                           0,
                           nullptr,
                           d_out,
                           num_blocks,
                           delay_iters);
    };
    switch(waves_per_simd)
    {
    case 1: launch(query_hw_placement<1>); break;
    case 2: launch(query_hw_placement<2>); break;
    case 3: launch(query_hw_placement<3>); break;
    case 4: launch(query_hw_placement<4>); break;
    case 5: launch(query_hw_placement<5>); break;
    case 6: launch(query_hw_placement<6>); break;
    case 7: launch(query_hw_placement<7>); break;
    case 8: launch(query_hw_placement<8>); break;
    default:
        fprintf(stderr, "waves_per_simd must be 1-8\n");
        exit(1);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- copy back ----
    BlockInfo* h_out = (BlockInfo*)malloc(num_blocks * sizeof(BlockInfo));
    HIP_CHECK(hipMemcpy(h_out, d_out, num_blocks * sizeof(BlockInfo), hipMemcpyDeviceToHost));

    // ---- 计算相对时钟（以全局最小值为基准）----
    uint64_t t0 = h_out[0].clock;
    for(int i = 1; i < num_blocks; i++)
        if(h_out[i].clock < t0) t0 = h_out[i].clock;

    // ---- 打印每个 block 的信息 ----
    printf("%-10s %-6s %-5s %-5s %-5s %-7s %-7s %-7s %-6s %-12s\n",
           "block_id", "xcc", "se", "sh", "cu", "simd", "wave", "pipe", "tg_id", "clock(cyc)");
    printf("%s\n", std::string(82, '-').c_str());

    int print_limit = num_blocks;
    for(int i = 0; i < print_limit; i++)
    {
        const BlockInfo& b = h_out[i];
        printf("%-10d %-6d %-5d %-5d %-5d %-7d %-7d %-7d %-6d %-12llu\n",
               b.block_id, b.xcc_id, b.se_id, b.sh_id, b.cu_id,
               b.simd_id, b.wave_id, b.pipe_id, b.tg_id,
               (unsigned long long)(b.clock - t0));
    }

    // ---- XCD 0 上最早的 N 个 block（按 clock 排序）----
    printf("\n=== XCD 0 上最早开始的 block（按时钟排序）===\n");
    std::vector<const BlockInfo*> xcd0;
    for(int i = 0; i < num_blocks; i++)
        if(h_out[i].xcc_id == 0) xcd0.push_back(&h_out[i]);
    std::sort(xcd0.begin(), xcd0.end(),
              [](const BlockInfo* a, const BlockInfo* b){ return a->clock < b->clock; });

    uint64_t first_clock = xcd0[0]->clock;
    // 估算第一批：clock 在最早值 + 一定窗口内的都算（用前128个观察窗口）
    int show = std::min((int)xcd0.size(), 160);
    printf("%-10s %-5s %-5s %-5s %-7s %-7s %-12s\n",
           "block_id", "se", "cu", "simd", "wave", "tg_id", "clock(cyc)");
    printf("%s\n", std::string(60, '-').c_str());
    for(int i = 0; i < show; i++)
    {
        const BlockInfo* b = xcd0[i];
        printf("%-10d %-5d %-5d %-5d %-7d %-7d %-12llu\n",
               b->block_id, b->se_id, b->cu_id, b->simd_id, b->wave_id, b->tg_id,
               (unsigned long long)(b->clock - first_clock));
    }

    // ---- 统计：每个 XCD 分到多少 blocks ----
    printf("\n=== Blocks per XCD ===\n");
    std::map<int, int> xcd_count;
    for(int i = 0; i < num_blocks; i++)
        xcd_count[h_out[i].xcc_id]++;
    for(auto& [xcd, cnt] : xcd_count)
        printf("  XCD %d : %d blocks\n", xcd, cnt);

    // ---- 统计：round-robin 验证 ----
    // 若硬件按 round-robin 分配，block i 应落在 XCD (i % NUM_XCDS)
    // 统计有多少 block 符合预期
    printf("\n=== Round-robin 验证 ===\n");
    int num_xcds = (int)xcd_count.size();
    if(num_xcds > 1)
    {
        int match = 0;
        for(int i = 0; i < num_blocks; i++)
        {
            if(h_out[i].xcc_id == i % num_xcds)
                match++;
        }
        printf("  检测到 %d 个 XCD\n", num_xcds);
        printf("  符合 block_id %% num_xcds == xcc_id 的 block 数：%d / %d (%.1f%%)\n",
               match, num_blocks, 100.0 * match / num_blocks);
    }
    else
    {
        printf("  只检测到 1 个 XCD（可能是单 die 设备，或 XCC_ID 不可用）\n");
    }

    // ---- 统计：(xcd, se, sh, cu) 组合出现次数 ----
    printf("\n=== CU 分布（每个物理 CU 被分配到的 block 数）===\n");
    // key: (xcd, se, sh, cu_id)
    std::map<std::tuple<int,int,int,int>, int> cu_count;
    for(int i = 0; i < num_blocks; i++)
    {
        const BlockInfo& b = h_out[i];
        cu_count[{b.xcc_id, b.se_id, b.sh_id, b.cu_id}]++;
    }
    printf("  共观察到 %d 个不同 (xcd,se,sh,cu) 组合\n", (int)cu_count.size());

    // 统计每个 xcd 内有多少个不同 (se,sh,cu) 组合
    std::map<int, std::set<std::tuple<int,int,int>>> cu_per_xcd;
    for(auto& [key, cnt] : cu_count)
    {
        auto [xcd, se, sh, cu] = key;
        cu_per_xcd[xcd].insert({se, sh, cu});
    }
    printf("  每个 XCD 内不同 CU 数：\n");
    for(auto& [xcd, cus] : cu_per_xcd)
        printf("    XCD %d : %d 个 CU\n", xcd, (int)cus.size());

    // 打印所有 (xcd,se,sh,cu) 及其 block 数，以 XCD=0 为例
    printf("\n  XCD=0 的详细 CU 分布：\n");
    printf("  %-5s %-5s %-5s %-8s %s\n", "se", "sh", "cu", "blocks", "");
    for(auto& [key, cnt] : cu_count)
    {
        auto [xcd, se, sh, cu] = key;
        if(xcd != 0) continue;
        printf("  %-5d %-5d %-5d %-8d\n", se, sh, cu, cnt);
    }

    // ---- clock 直方图：按批次统计 block 数 ----
    if(delay_iters > 0)
    {
        printf("\n=== Clock 直方图（XCD 0，每批次 block 数）===\n");
        // 找 XCD 0 的 clock 范围
        uint64_t cmin = UINT64_MAX, cmax = 0;
        for(int i = 0; i < num_blocks; i++)
        {
            if(h_out[i].xcc_id != 0) continue;
            uint64_t c = h_out[i].clock - t0;
            if(c < cmin) cmin = c;
            if(c > cmax) cmax = c;
        }
        // 用 32 个 bucket 做直方图
        const int NBUCKETS = 32;
        uint64_t span   = cmax - cmin + 1;
        uint64_t bwidth = (span + NBUCKETS - 1) / NBUCKETS;
        int hist[NBUCKETS] = {};
        for(int i = 0; i < num_blocks; i++)
        {
            if(h_out[i].xcc_id != 0) continue;
            int b = (int)((h_out[i].clock - t0 - cmin) / bwidth);
            if(b >= NBUCKETS) b = NBUCKETS - 1;
            hist[b]++;
        }
        printf("  clock range: [%llu, %llu] cyc  bucket_width=%llu cyc\n",
               (unsigned long long)cmin, (unsigned long long)cmax,
               (unsigned long long)bwidth);
        printf("  %-12s %-12s %s\n", "clock_start", "blocks", "bar");
        for(int b = 0; b < NBUCKETS; b++)
        {
            if(hist[b] == 0) continue;
            printf("  %-12llu %-12d %s\n",
                   (unsigned long long)(cmin + b * bwidth),
                   hist[b],
                   std::string(hist[b] / 2, '#').c_str());
        }
    }

    // ---- 统计：相邻 block 落在同一 XCD 的比例 ----
    printf("\n=== 相邻 block 同 XCD 比例（RemapXCD 效果验证）===\n");
    int same_xcd_adjacent = 0;
    for(int i = 0; i + 1 < num_blocks; i++)
    {
        if(h_out[i].xcc_id == h_out[i + 1].xcc_id)
            same_xcd_adjacent++;
    }
    printf("  相邻 block 在同一 XCD 的比例：%d / %d (%.1f%%)\n",
           same_xcd_adjacent, num_blocks - 1,
           100.0 * same_xcd_adjacent / (num_blocks - 1));
    printf("  （round-robin 下此值接近 0%%；RemapXCD 后此值接近 100%%）\n");

    free(h_out);
    HIP_CHECK(hipFree(d_out));
    return 0;
}
