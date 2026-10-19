// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Boundary barrier for the fused GEMM.A2A multi-GPU arm: one single-thread
// kernel per rank, add / wait / subtract over an arrival counter that each
// rank owns and its peers increment.

#include <Tensile/FusedA2AKernArg.hpp> // FUSED_A2A_MAX_RANKS

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    namespace Client
    {
        // A whole line for the one live u32, and the region is memset once at
        // allocation and never again.
        constexpr size_t FUSED_A2A_ARRIVAL_BYTES = 64;

        // Entry j is rank j's arrival counter, reachable from every rank once
        // peer access is on.
        struct A2AArrivalPeers
        {
            uint32_t* p[FUSED_A2A_MAX_RANKS] = {};
        };

        // world == 1 leaves need == 0: nothing to add, nothing to wait for.
        static __global__ void a2aBoundaryBarrierKernel(A2AArrivalPeers peers,
                                                        int             myRank,
                                                        int             world)
        {
            const uint32_t need = (uint32_t)(world - 1);

            for(int j = 0; j < world; j++)
            {
                if(j != myRank)
                    __hip_atomic_fetch_add(
                        peers.p[j], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }

            while(__hip_atomic_load(peers.p[myRank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)
                  < need)
                ;

            __hip_atomic_fetch_sub(
                peers.p[myRank], need, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        struct A2AQueueWarm
        {
            uint64_t const* rptr[FUSED_A2A_MAX_RANKS] = {};
            uint64_t const* wptr[FUSED_A2A_MAX_RANKS] = {};
            uint32_t const* ring[FUSED_A2A_MAX_RANKS] = {};
            int             n                         = 0;
            uint32_t        ringPages                 = 0;
            // [0] sum of the first rptr read's s_memrealtime ticks, [1] same for the first ring read, [2] launches
            uint64_t*       sink                      = nullptr;
        };

        static __global__ void a2aQueueWarmKernel(A2AQueueWarm q)
        {
            uint64_t acc = 0;
            if(threadIdx.x == 0 && q.n > 0)
            {
                const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
                const uint64_t r  = __hip_atomic_load(q.rptr[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                __asm__ volatile("" ::"r"(r));
                const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
                const uint32_t g  = __hip_atomic_load(q.ring[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                __asm__ volatile("" ::"v"(g));
                const uint64_t t2 = __builtin_amdgcn_s_memrealtime();
                atomicAdd((unsigned long long*)&q.sink[0], (unsigned long long)(t1 - t0));
                atomicAdd((unsigned long long*)&q.sink[1], (unsigned long long)(t2 - t1));
                atomicAdd((unsigned long long*)&q.sink[2], 1ull);
                acc += r + g;
            }
            for(int i = 0; i < q.n; i++)
            {
                if(threadIdx.x == 0)
                    acc += __hip_atomic_load(q.rptr[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                           + __hip_atomic_load(q.wptr[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                for(uint32_t p = threadIdx.x; p < q.ringPages; p += blockDim.x)
                    acc += __hip_atomic_load(
                        q.ring[i] + p * 1024, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            if(acc == 0x0123456789abcdefull)
                q.sink[3] = 1;
        }
    } // namespace Client
} // namespace TensileLite
