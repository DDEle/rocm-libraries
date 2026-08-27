// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Builds the per-peer kernarg groups for a fused GEMM.A2A launch. Depends on the
// Tensile ABI header alone.

#include <Tensile/FusedA2AKernArg.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <vector>

namespace rocblaslt
{
    // Slot indices within FUSED_A2A_PEER_FIELDS; the order itself is the contract with
    // SdmaRingEmitter.py.
    constexpr size_t kFusedA2AFlagSlot     = 0;
    constexpr size_t kFusedA2ARecvSlot     = 1;
    constexpr size_t kFusedA2AQueueBufSlot = 2;
    constexpr size_t kFusedA2ARptrSlot     = 3;
    constexpr size_t kFusedA2AWptrSlot     = 4;
    constexpr size_t kFusedA2ADoorbellSlot = 5;

    // The drain bits this layer asks for. IN_KERNEL maps to DRAIN_RECV; DRAIN_SEND has no
    // attribute at this layer.
    constexpr uint32_t fusedA2ADrainFor(hipblasLtA2ACompletionMode_t mode)
    {
        return mode == HIPBLASLT_A2A_COMPLETION_IN_KERNEL ? TensileLite::FUSED_A2A_DRAIN_RECV : 0u;
    }

    // One group per rank in FUSED_A2A_PEER_FIELDS order. Fills the flag, recv and SDMA
    // queue slots. Any of the three sources may be null, leaving its slots null. Returns
    // an empty list when world does not fit the segment.
    inline std::vector<TensileLite::FusedA2APeerFields>
        buildFusedA2APeerFields(void* const*                peerFlag,
                                void* const*                recvPtrs,
                                uint32_t                    world,
                                const hipblasLtSdmaQueue_t* queues = nullptr)
    {
        const bool fits = world <= uint32_t(TensileLite::FUSED_A2A_MAX_RANKS)
                          && TensileLite::fusedA2AWorldSizeValid(static_cast<int>(world));
        if(!fits)
            return {};

        std::vector<TensileLite::FusedA2APeerFields> peers(world);
        for(uint32_t j = 0; j < world; ++j)
        {
            peers[j][kFusedA2AFlagSlot] = peerFlag != nullptr ? peerFlag[j] : nullptr;
            peers[j][kFusedA2ARecvSlot] = recvPtrs != nullptr ? recvPtrs[j] : nullptr;
            if(queues != nullptr)
            {
                peers[j][kFusedA2AQueueBufSlot] = queues[j].queueBuf;
                peers[j][kFusedA2ARptrSlot]     = queues[j].rptr;
                peers[j][kFusedA2AWptrSlot]     = queues[j].wptr;
                peers[j][kFusedA2ADoorbellSlot] = queues[j].doorbell;
            }
        }
        return peers;
    }
}
