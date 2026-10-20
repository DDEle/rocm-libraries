// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for TensileLite::appendFusedSegment (no GPU, no file I/O).

#include <gtest/gtest.h>

#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/KernelArguments.hpp>

using namespace TensileLite;

namespace
{
    std::vector<FusedA2APeerFields> makePeers(size_t n)
    {
        std::vector<FusedA2APeerFields> peers(n);
        for(size_t j = 0; j < n; j++)
            for(size_t k = 0; k < FUSED_A2A_SLOT_COUNT; k++)
                peers[j][k] = reinterpret_cast<void*>(0x1000 + j * 0x100 + k * 8);
        return peers;
    }

    void appendWorld(KernelArgumentsCounter& args, uint32_t world, size_t peerCount)
    {
        void* counter = reinterpret_cast<void*>(0xC0FFEE00);
        appendFusedSegment(args, makePeers(peerCount), counter, 0, world, 0, 0, 304);
    }
}

TEST(FusedA2AKernArg, RejectsWorldSizeOutOfRange)
{
    struct Case
    {
        uint32_t world;
        size_t   peers;
    };
    for(auto const& c : std::vector<Case>{{FUSED_A2A_MAX_RANKS + 1, FUSED_A2A_MAX_RANKS + 1},
                                          {16, 16},
                                          {0, 0},
                                          {16, FUSED_A2A_MAX_RANKS}})
    {
        KernelArgumentsCounter args;
        EXPECT_THROW(appendWorld(args, c.world, c.peers), std::runtime_error)
            << "world " << c.world << " peers " << c.peers;
    }
}
