// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "FusedA2ACounterSentinel.hpp"

#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Utils.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        constexpr int A2A_GEMM_POISON_BYTE = 0xC7;

        struct A2AGemmCounterLayout
        {
            size_t featureBlocks = 0;
            size_t tokenBlocks   = 0;
            size_t batches       = 0;
            size_t payloadBytes  = 0;
            size_t allocBytes    = 0;
        };

        // batches mirrors a2aBatchSpan: iB = (b * F) / numCu over b in [0, tokenBlocks).
        inline A2AGemmCounterLayout a2aGemmCounterLayout(size_t   M,
                                                         size_t   N,
                                                         size_t   mt0,
                                                         size_t   mt1,
                                                         size_t   numCu,
                                                         uint32_t world,
                                                         uint32_t subRounds)
        {
            A2AGemmCounterLayout l;
            l.featureBlocks = (M + mt0 - 1) / mt0;
            l.tokenBlocks   = (N + mt1 - 1) / mt1;
            l.batches       = (l.tokenBlocks - 1) * l.featureBlocks / numCu + 1;
            l.payloadBytes  = fusedA2AMode1PayloadBytes(
                world, (uint32_t)l.tokenBlocks, (uint32_t)l.batches, subRounds);
            l.allocBytes = l.payloadBytes + FUSED_A2A_COUNTER_SENTINEL_BYTES;
            return l;
        }

        // dst: rank's logical B, [N, K] with ldb = K. xs[r]: rank r's x, {K/W, N, W}.
        inline void a2aGemmInterleaveB(void*                           dst,
                                       std::vector<void const*> const& xs,
                                       size_t                          rank,
                                       size_t                          N,
                                       size_t                          K,
                                       size_t                          elementBytes)
        {
            const size_t world  = xs.size();
            const size_t kLocal = K / world;
            for(size_t n = 0; n < N; n++)
                for(size_t r = 0; r < world; r++)
                    std::memcpy(static_cast<char*>(dst) + (n * K + r * kLocal) * elementBytes,
                                static_cast<char const*>(xs[r])
                                    + (rank * N + n) * kLocal * elementBytes,
                                kLocal * elementBytes);
        }

        // words: rank's mode-1 flag block. Empty when seq == launches, every peer's done
        // equals seq, and every other word of the block (including done[rank] and done[s >= world])
        // is 0. At world 1 every word of the block is 0.
        inline std::string
            a2aGemmCommResidue(uint32_t const* words, size_t rank, size_t world, size_t launches)
        {
            constexpr size_t doneWord = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
            constexpr size_t seqWord  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
            const uint32_t   seq      = words[seqWord];
            if(world > 1 && seq != launches)
                return concatenate("seq=", seq, ", launches=", launches);
            for(size_t w = 0; w < FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t); w++)
            {
                if(world > 1 && w == seqWord)
                    continue;
                const bool   isDone = w >= doneWord && w < doneWord + FUSED_A2A_MAX_RANKS;
                const size_t s      = w - doneWord;
                if(isDone && s < world && s != rank)
                {
                    if(words[w] != seq)
                        return concatenate("done[", s, "]=", words[w], ", seq=", seq);
                }
                else if(words[w] != 0)
                    return concatenate(isDone ? "done[" : "word[",
                                       isDone ? s : w,
                                       "]=",
                                       words[w],
                                       ", expected 0");
            }
            return "";
        }
    } // namespace Client
} // namespace TensileLite
