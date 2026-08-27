// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Builds the per-peer kernarg groups for a fused GEMM.A2A launch. Depends on the
// Tensile ABI header alone.

#include <Tensile/FusedA2AKernArg.hpp>

#include <cstdint>
#include <vector>

namespace rocblaslt
{
    // Slot indices within FUSED_A2A_PEER_FIELDS; the order itself is the contract with
    // SdmaRingEmitter.py.
    constexpr size_t kFusedA2AFlagSlot = 0;
    constexpr size_t kFusedA2ARecvSlot = 1;

    // One group per rank in FUSED_A2A_PEER_FIELDS order. Fills the flag and recv slots
    // and leaves the four SDMA queue slots null. Either array may be null. Returns an
    // empty list when world does not fit the segment.
    inline std::vector<TensileLite::FusedA2APeerFields>
        buildFusedA2APeerFields(void* const* peerFlag, void* const* recvPtrs, uint32_t world)
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
        }
        return peers;
    }
}
