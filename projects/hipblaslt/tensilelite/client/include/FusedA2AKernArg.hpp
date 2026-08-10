// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the fused GEMM.A2A kernarg segment ABI.
//
// The layout is defined on two sides that must stay byte-identical: the kernel
// side (Tensile/Components/Signature.py fusedA2AKernArgLayout + addArg) and the
// host side here. These declarations live in a header rather than inside
// FusedA2AClient.cpp so that tests/FusedA2AKernArg_test.cpp can drive the real
// append sequence; while it was a TU-private helper the test could only
// reproduce it, and a reproduction cannot catch the drift it exists to detect.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <Tensile/KernelArguments.hpp>

namespace TensileLite
{
    namespace Client
    {
        // Compile-time fixed slot count for the fused-A2A kernarg segment. MUST
        // match FUSED_A2A_MAX_RANKS in Tensile/Components/Signature.py: the
        // kernel metadata always reserves 8 peer_ptr slots regardless of the
        // runtime world size, so the host must append exactly 8 (unused slots
        // j>=W filled with nullptr).
        constexpr int FUSED_A2A_MAX_RANKS = 8;

        // Byte offset of recv inside a peer block; flag occupies [0, MAX_RANKS*4).
        // Mirrored in Tensile/Components/Signature.py.
        constexpr size_t FUSED_A2A_PEER_RECV_OFFSET = 4096;
        static_assert(FUSED_A2A_MAX_RANKS * sizeof(uint32_t) <= FUSED_A2A_PEER_RECV_OFFSET,
                      "flag array must fit below the recv offset inside a peer block");

        // ...but it is not a free constant. The DRAIN barrier's EXEC mask
        // (Tensile/Components/GlobalWriteBatch.py _emitFusedA2AHandshake) is one
        // S_BFM whose width operand is the runtime W, and that operand is a
        // TRUNCATED bit field -- 6 bits on the wave64 arm (S_BFM_B64:
        // ((1 << src0[5:0]) - 1) << src1[5:0]) and 5 on the wave32 arm
        // (S_BFM_B32, the same with [4:0]). A width at the field's modulus wraps
        // to zero instead of saturating, so W=64 (resp. 32) yields an EMPTY EXEC.
        //
        // The bound below is 31, the wave32 arm's, not the wave64 arm's 63:
        // neither this header nor Signature.py knows the wave width of the kernel
        // that will consume the segment -- both constants are fixed long before a
        // solution's WavefrontSize is in hand -- so the only sound bound is the
        // one that holds for both arms. Every fused config today is wave64, so 31
        // costs nothing real; raising past it means first proving no fused config
        // is wave32. The twin check lives at Signature.py's FUSED_A2A_MAX_RANKS.
        //
        // 31 is the mask's CEILING, not a recommendation, and the bound is
        // necessary rather than sufficient. The shipped value is 8 because no node
        // is known to carry more than 8 GPUs -- it is the world size this ABI is
        // built for, not a placeholder awaiting a raise. Moving toward 31 would
        // satisfy the assertion below while growing this segment from 108 B to
        // 292 B, widening the kernarg slot count, and deepening the two unrolled
        // per-rank scans in GlobalWriteBatch.py to ~30 iterations each.
        static_assert(FUSED_A2A_MAX_RANKS <= 31,
                      "FUSED_A2A_MAX_RANKS exceeds the 31 the DRAIN EXEC mask can encode: "
                      "the S_BFM width operand is 5 bits on the wave32 arm and 6 on the "
                      "wave64 arm, so a world size of 32 (resp. 64) wraps the width to 0, "
                      "EXEC becomes empty, the DRAIN poll never issues, and the barrier is "
                      "silently skipped -- the epilogue then reads peer tiles that have not "
                      "arrived. That is a wrong answer, not a hang. Re-derive the mask "
                      "before raising this.");

        // Expected byte growth of args after appending the fused segment:
        //   (MAX_RANKS peer + 1 counter + 1 FusedSdmaQueues) pointers * 8B
        //   + 7 scalars * 4B = 108B.
        constexpr size_t FUSED_A2A_SEGMENT_BYTES = (FUSED_A2A_MAX_RANKS + 2) * 8 + 7 * 4;

        // Whether a requested world size can be expressed in the segment above.
        //
        // The bound is an ABI property, not a machine property: ranks past
        // FUSED_A2A_MAX_RANKS have no peer_ptr slot at all, so a PUSH
        // workgroup targeting them consumes whatever the kernel metadata default
        // is -- silent corruption or a DRAIN hang. A device-count check cannot
        // stand in for it, and cannot stand in for the lower bound either: for
        // W <= 0 a `deviceCount < W` comparison reads false and falls through,
        // leaving a zero or negative W to be used as a divisor.
        constexpr bool fusedA2AWorldSizeValid(int worldSize)
        {
            return worldSize >= 1 && worldSize <= FUSED_A2A_MAX_RANKS;
        }

        // Append the fixed-size fused-A2A kernarg segment to `args` in the exact
        // emission order of Signature.py fusedA2AKernArgLayout().
        //
        // Alignment: peer_ptr_0 is appendAligned<void*> so it lands on an 8-byte
        // boundary, mirroring how the kernel metadata 8-aligns the first
        // SIG_GLOBALBUFFER arg of the segment. The remaining pointers (8B) and
        // scalars (4B) are appended contiguously with no interior padding,
        // matching the Python layout (off += 8 / off += 4).
        //
        // peerPtrs holds this device's view and may be shorter than
        // FUSED_A2A_MAX_RANKS; the remaining slots are filled with nullptr.
        inline void appendFusedSegment(KernelArguments&          args,
                                       std::vector<void*> const& peerPtrs, // size W (device d's per-peer block bases)
                                       void*                     counterPtr,
                                       void*                     sdmaQueues, // W-element SdmaQueueDeviceHandle array
                                       uint32_t                  myRank,
                                       uint32_t                  worldSize,
                                       uint32_t                  nShard,
                                       uint32_t                  drain,
                                       uint32_t                  an,
                                       uint32_t                  tilesPerRank,
                                       uint32_t                  tokenTiles)
        {
            size_t before = args.size();

            for(int j = 0; j < FUSED_A2A_MAX_RANKS; j++)
            {
                void* p = (j < (int)peerPtrs.size()) ? peerPtrs[j] : nullptr;
                if(j == 0)
                    args.appendAligned<void*>("peer_ptr_0", p);
                else
                    args.append<void*>("peer_ptr_" + std::to_string(j), p);
            }
            args.append<void*>("counter_ptr", counterPtr);
            args.append<void*>("FusedSdmaQueues", sdmaQueues);
            args.append<uint32_t>("FusedMyRank", myRank);
            args.append<uint32_t>("FusedW", worldSize);
            args.append<uint32_t>("FusedNShard", nShard);
            args.append<uint32_t>("FusedDrain", drain);
            // Kernarg "FusedAM" (renamed from FusedAN in Task 6 alongside
            // Signature.py); the value `an` carries AM (A2A width along FEATURE)
            // from the swapped client.
            args.append<uint32_t>("FusedAM", an);
            args.append<uint32_t>("FusedTilesPerRank", tilesPerRank);
            args.append<uint32_t>("FusedTokenTiles", tokenTiles);

            size_t grew = args.size() - before;
            if(grew != FUSED_A2A_SEGMENT_BYTES)
            {
                std::cerr << "[fused-a2a] WARNING: fused segment grew args by " << grew
                          << " bytes, expected " << FUSED_A2A_SEGMENT_BYTES
                          << " (alignment/padding mismatch — epilogue will read wrong offsets)"
                          << std::endl;
            }
        }
    } // namespace Client
} // namespace TensileLite
