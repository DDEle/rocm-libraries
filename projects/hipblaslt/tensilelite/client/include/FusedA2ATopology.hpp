// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/FusedA2AKernArg.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // The k-th engine set in `mask`, wrapping; 0 for an empty mask.
        inline uint32_t sdmaMaskEngine(uint32_t mask, size_t k)
        {
            if(mask == 0)
                return 0;
            size_t engines = 0;
            for(uint32_t b = 0; b < 32; ++b)
                engines += (mask >> b) & 1u;
            k %= engines;
            for(uint32_t b = 0;; ++b)
                if((mask & (1u << b)) && k-- == 0)
                    return b;
        }

        // spread: engine sdmaMaskEngine(link mask, k); otherwise sdmaQueueEngine with remote = k.
        struct SdmaQueuePlan
        {
            size_t peer;
            bool   spread;
            size_t k;
        };

        // The queues to create over the ranks of `peers`, in creation order. directMask 0: every
        // rank, unspread, k counting the earlier ranks that are not `self`; otherwise the ranks
        // outside directMask, spread, k counting them. Both in rank order.
        inline std::vector<SdmaQueuePlan>
            sdmaQueuePlan(std::vector<int> const& peers, int self, uint32_t directMask)
        {
            std::vector<SdmaQueuePlan> plan;
            size_t                     k = 0;
            for(size_t j = 0; j < peers.size(); j++)
                if(directMask == 0)
                    plan.push_back({j, false, peers[j] == self ? 0 : k++});
                else if(((directMask >> j) & 1u) == 0)
                    plan.push_back({j, true, k++});
            return plan;
        }
    } // namespace Client
} // namespace TensileLite
