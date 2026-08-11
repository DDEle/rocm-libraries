// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Golden offset-table regression for the fused GEMM.A2A kernarg segment.
//
// The segment layout is defined on TWO sides that must stay byte-identical: the
// kernel side (Tensile/Components/Signature.py fusedA2AKernArgLayout + addArg
// sequence) and the host side (appendFusedSegment). There is no cross-language
// test harness, so both sides are pinned against the SAME golden table -- here
// (C++) and in Tensile/Tests/unit/test_fusedA2AKernArgLayout.py (Python). A
// one-sided change to either reddens its own golden test.
//
// This test drives the REAL appendFusedSegment out of client/include/
// FusedA2AKernArg.hpp.
//
// Every field is written with a distinct sentinel and read back at its golden
// offset, so a reorder (values land transposed), a type change (following
// fields shift) and a rename (the caller stops compiling) all redden here.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <FusedA2AKernArg.hpp>
#include <Tensile/KernelArguments.hpp>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // Sentinels. Distinct per field, so a transposition cannot go unnoticed.
    void* peerPtr(int j)
    {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + j));
    }
    void* const kCounterPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000));
    void* const kSdmaQueues = reinterpret_cast<void*>(static_cast<uintptr_t>(0x4000));

    constexpr uint32_t kMyRank       = 0xA1;
    constexpr uint32_t kWorldSize    = 0xA3;
    constexpr uint32_t kNShard       = 0xA4;
    constexpr uint32_t kDrain        = 0xA5;
    constexpr uint32_t kAM           = 0xA6;
    constexpr uint32_t kTilesPerRank = 0xB1;
    constexpr uint32_t kTokenTiles   = 0xB2;

    // Independently written oracle: {arg name, intra-segment byte offset, byte
    // size, expected value}. Byte-identical in name and offset to GOLDEN_LAYOUT
    // in the Python test.
    struct GoldenArg
    {
        const char* name;
        size_t      offset;
        size_t      size;
        uint64_t    value;
    };

    std::vector<GoldenArg> goldenTable()
    {
        std::vector<GoldenArg> t;
        size_t                 off = 0;
        for(int j = 0; j < FUSED_A2A_MAX_RANKS; ++j) // peer_ptr_0..7 : 0..56
        {
            t.push_back({"peer_ptr", off, 8, static_cast<uint64_t>(0x1000 + j)});
            off += 8;
        }
        t.push_back({"counter_ptr", off, 8, 0x3000}); // 64
        off += 8;
        t.push_back({"FusedSdmaQueues", off, 8, 0x4000}); // 72 (8-aligned, no padding)
        off += 8;
        const std::pair<const char*, uint32_t> scalars[] = {{"FusedMyRank", kMyRank},
                                                            {"FusedW", kWorldSize},
                                                            {"FusedNShard", kNShard},
                                                            {"FusedDrain", kDrain},
                                                            {"FusedAM", kAM},
                                                            {"FusedTilesPerRank", kTilesPerRank},
                                                            {"FusedTokenTiles", kTokenTiles}};
        for(auto const& s : scalars)
        {
            t.push_back({s.first, off, 4, s.second}); // 80,84,88,92,96,100,104
            off += 4;
        }
        return t;
    }

    // Drive the real appendFusedSegment with the sentinels above. `slots` is how
    // many peer pointers the caller supplies; the function pads the rest of the
    // fixed FUSED_A2A_MAX_RANKS slots with nullptr.
    void appendWithSentinels(KernelArguments& args, int slots = FUSED_A2A_MAX_RANKS)
    {
        std::vector<void*> peers;
        for(int j = 0; j < slots; ++j)
            peers.push_back(peerPtr(j));
        appendFusedSegment(args,
                           peers,
                           kCounterPtr,
                           kSdmaQueues,
                           kMyRank,
                           kWorldSize,
                           kNShard,
                           kDrain,
                           kAM,
                           kTilesPerRank,
                           kTokenTiles);
    }

    uint64_t readAt(KernelArguments const& args, size_t base, size_t off, size_t sz)
    {
        uint64_t v = 0;
        std::memcpy(&v, static_cast<uint8_t const*>(args.data()) + base + off, sz);
        return v;
    }
}

// Total byte growth of the segment == 108, matching the Python side.
TEST(FusedA2AKernArg, SegmentBytesIs108)
{
    EXPECT_EQ(FUSED_A2A_SEGMENT_BYTES, 108u);

    KernelArguments args(/*log=*/false);
    size_t          before = args.size();
    appendWithSentinels(args);
    EXPECT_EQ(args.size() - before, FUSED_A2A_SEGMENT_BYTES);
}

// Each arg lands at its golden intra-segment offset carrying its own value.
// The segment is appended onto an empty KernelArguments, so the segment base is
// 0 and intra-offsets can be compared directly -- the same convention the
// Python layout uses.
TEST(FusedA2AKernArg, FieldOffsetsAndValuesMatchGolden)
{
    KernelArguments args(/*log=*/false);
    size_t          base = args.size();
    appendWithSentinels(args);

    ASSERT_EQ(args.size() - base, FUSED_A2A_SEGMENT_BYTES);

    for(auto const& a : goldenTable())
    {
        EXPECT_EQ(readAt(args, base, a.offset, a.size), a.value)
            << a.name << " is not at intra-segment offset " << a.offset;
    }

    // Tight packing: the last arg ends exactly at the segment size.
    const auto table = goldenTable();
    EXPECT_EQ(table.back().offset + table.back().size, FUSED_A2A_SEGMENT_BYTES);
}

// The kernel metadata always reserves FUSED_A2A_MAX_RANKS slots regardless of
// the runtime world size, so a caller supplying fewer must get nullptr in the
// remainder rather than a short segment.
TEST(FusedA2AKernArg, UnusedRankSlotsAreNullAndSegmentStaysFixed)
{
    KernelArguments args(/*log=*/false);
    size_t          base = args.size();
    appendWithSentinels(args, /*slots=*/2);

    EXPECT_EQ(args.size() - base, FUSED_A2A_SEGMENT_BYTES);

    for(int j = 0; j < FUSED_A2A_MAX_RANKS; ++j)
    {
        const uint64_t expectPeer = (j < 2) ? static_cast<uint64_t>(0x1000 + j) : 0u;
        EXPECT_EQ(readAt(args, base, 8 * j, 8), expectPeer) << "peer_ptr_" << j;
    }
}

// In the client the segment is appended after the common args, not onto an
// empty buffer. peer_ptr_0 uses appendAligned<void*>, so an 8-aligned base must
// still produce exactly FUSED_A2A_SEGMENT_BYTES of growth with no padding.
TEST(FusedA2AKernArg, GrowthIsUnchangedAtAnAlignedNonZeroBase)
{
    KernelArguments args(/*log=*/false);
    for(int i = 0; i < 13; ++i)
        args.append<uint64_t>("prefix_" + std::to_string(i), i);

    size_t base = args.size();
    ASSERT_EQ(base % 8, 0u);
    appendWithSentinels(args);

    EXPECT_EQ(args.size() - base, FUSED_A2A_SEGMENT_BYTES);
    for(auto const& a : goldenTable())
    {
        EXPECT_EQ(readAt(args, base, a.offset, a.size), a.value)
            << a.name << " shifted when the segment was appended at base " << base;
    }
}

// FusedSdmaQueues sits directly after counter_ptr so the pointer group stays
// 8-aligned and contiguous; FusedTilesPerRank/FusedTokenTiles are still the
// last two scalars, appended after everything else.
TEST(FusedA2AKernArg, SdmaQueuesFollowCounterTilesAndTokensAreLast)
{
    const auto table         = goldenTable();
    size_t     counterOffset = 0, sdmaQueuesOffset = 0, scalarMax = 0;
    for(auto const& a : table)
    {
        std::string n = a.name;
        if(n == "counter_ptr")
            counterOffset = a.offset;
        if(n == "FusedSdmaQueues")
            sdmaQueuesOffset = a.offset;
        if(n != "FusedTilesPerRank" && n != "FusedTokenTiles")
            scalarMax = std::max(scalarMax, a.offset);
    }
    EXPECT_EQ(sdmaQueuesOffset, counterOffset + 8)
        << "FusedSdmaQueues must directly follow counter_ptr";
    for(auto const& a : table)
    {
        std::string n = a.name;
        if(n == "FusedTilesPerRank" || n == "FusedTokenTiles")
            EXPECT_GT(a.offset, scalarMax) << n << " must be appended after everything else";
    }
}

// The world size the client accepts is bounded by the ABI, not by the machine:
// the segment reserves exactly FUSED_A2A_MAX_RANKS peer_ptr slots, so a larger
// W has no pointer at all for the ranks past the end. The lower bound rejects
// 0 and negatives.
TEST(FusedA2AWorldSize, AcceptsExactlyTheRepresentableRange)
{
    for(int w = 1; w <= FUSED_A2A_MAX_RANKS; ++w)
        EXPECT_TRUE(fusedA2AWorldSizeValid(w)) << "W=" << w << " fits the kernarg segment";

    EXPECT_FALSE(fusedA2AWorldSizeValid(0));
    EXPECT_FALSE(fusedA2AWorldSizeValid(-1));
    EXPECT_FALSE(fusedA2AWorldSizeValid(FUSED_A2A_MAX_RANKS + 1));
}
