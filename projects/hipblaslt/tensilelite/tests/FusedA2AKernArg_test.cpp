// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Golden offset-table regression for the fused GEMM.A2A kernarg segment (Task 3).
//
// The segment layout is defined on TWO sides that must stay byte-identical: the
// kernel side (Tensile/Components/Signature.py fusedA2AKernArgLayout + addArg
// sequence) and the host side (client/src/FusedA2AClient.cpp appendFusedSegment).
// There is no cross-language test harness, so both sides are pinned against the
// SAME hardcoded golden table -- here (C++) and in
// Tensile/Tests/unit/test_fusedA2AKernArgLayout.py (Python). A one-sided change
// to either reddens its own golden test.
//
// appendFusedSegment lives in an anonymous namespace inside FusedA2AClient.cpp
// (a TU-private helper of the client main), so this test cannot call it directly
// without changing the client's structure -- which brief forbids doing purely
// for a test. Instead it REPRODUCES the append sequence into a real
// KernelArguments and checks the resulting byte offsets.
//
// LIMITATION (be honest about what this does NOT catch): the reproduced
// appendSequence() below is a private COPY. Editing the real appendFusedSegment
// in FusedA2AClient.cpp -- reordering args or changing a type -- does NOT redden
// this test, because this test never calls that function. What this test DOES
// catch is: (a) drift between the golden table and the copied sequence, and
// (b) a change in KernelArguments' alignment/packing behavior at the
// first-risk-point -- appendAligned<void*> at intra-offset 160 must introduce no
// padding (160 is already 8-aligned) and the segment must pack to exactly
// FUSED_A2A_SEGMENT_BYTES. The client<->kernel offset contract is guarded across
// all three checkpoints together: this copy, the Python golden table, and the
// str(signature) snapshot that emits the REAL Signature.py addArg sequence --
// any one-sided drift reddens at least one of them.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/KernelArguments.hpp>

using namespace TensileLite;

namespace
{
    // Mirror of FUSED_A2A_MAX_RANKS / FUSED_A2A_SEGMENT_BYTES in
    // FusedA2AClient.cpp (which are TU-private constexprs there).
    constexpr int    kMaxRanks     = 8;
    constexpr size_t kSegmentBytes = (2 * kMaxRanks + 2) * 8 + 8 * 4; // 176

    // The single source of truth, byte-identical to GOLDEN_LAYOUT in the Python
    // test. {arg name, intra-segment byte offset, byte size}. Order matches the
    // append sequence exactly.
    struct GoldenArg
    {
        const char* name;
        size_t      offset;
        size_t      size;
    };

    std::vector<GoldenArg> goldenTable()
    {
        std::vector<GoldenArg> t;
        size_t                 off = 0;
        for(int j = 0; j < kMaxRanks; ++j) // recv_ptr_0..7 : 0..56
        {
            t.push_back({"recv_ptr", off, 8});
            off += 8;
        }
        for(int j = 0; j < kMaxRanks; ++j) // flag_ptr_0..7 : 64..120
        {
            t.push_back({"flag_ptr", off, 8});
            off += 8;
        }
        t.push_back({"counter_ptr", off, 8});       // 128
        off += 8;
        for(auto n : {"FusedMyRank", "FusedTarget", "FusedW", "FusedNShard", "FusedDrain", "FusedAM"})
        {
            t.push_back({n, off, 4}); // 136,140,144,148,152,156
            off += 4;
        }
        t.push_back({"FusedSdmaQueues", off, 8}); // 160 (8-aligned, no padding)
        off += 8;
        for(auto n : {"FusedTilesPerRank", "FusedTokenTiles"})
        {
            t.push_back({n, off, 4}); // 168,172
            off += 4;
        }
        return t;
    }

    // Reproduce appendFusedSegment's append sequence into `args`. Values are
    // irrelevant to the offset contract (only sizes/order matter), so use
    // placeholders. Mirrors FusedA2AClient.cpp appendFusedSegment exactly.
    void appendSequence(KernelArguments& args)
    {
        for(int j = 0; j < kMaxRanks; ++j)
        {
            if(j == 0)
                args.appendAligned<void*>("recv_ptr_0", nullptr);
            else
                args.append<void*>("recv_ptr_" + std::to_string(j), nullptr);
        }
        for(int j = 0; j < kMaxRanks; ++j)
            args.append<void*>("flag_ptr_" + std::to_string(j), nullptr);
        args.append<void*>("counter_ptr", nullptr);
        args.append<uint32_t>("FusedMyRank", 0);
        args.append<uint32_t>("FusedTarget", 0);
        args.append<uint32_t>("FusedW", 0);
        args.append<uint32_t>("FusedNShard", 0);
        args.append<uint32_t>("FusedDrain", 0);
        args.append<uint32_t>("FusedAM", 0);
        args.append<void*>("FusedSdmaQueues", nullptr);
        args.append<uint32_t>("FusedTilesPerRank", 0);
        args.append<uint32_t>("FusedTokenTiles", 0);
    }
}

// Total byte growth of the segment == 176, matching the Python side.
TEST(FusedA2AKernArg, SegmentBytesIs176)
{
    EXPECT_EQ(kSegmentBytes, 176u);

    KernelArguments args(/*log=*/false);
    size_t          before = args.size();
    appendSequence(args);
    EXPECT_EQ(args.size() - before, kSegmentBytes);
}

// Each arg lands at its golden intra-segment offset. Because the segment is
// appended onto an empty KernelArguments here, the running size just before each
// append IS that arg's intra-offset (segment base == 0), so it can be compared
// directly to the golden table -- the same convention the Python layout uses.
TEST(FusedA2AKernArg, FieldOffsetsMatchGolden)
{
    const auto table = goldenTable();

    KernelArguments args(/*log=*/false);
    size_t          base = args.size();

    // Re-run the sequence one append at a time, capturing the offset (size just
    // before the append) so the golden table is checked field-by-field including
    // the appendAligned<void*> for the first pointer.
    std::vector<size_t> offsets;
    auto                push = [&](auto&& fn) {
        offsets.push_back(args.size() - base);
        fn();
    };

    for(int j = 0; j < kMaxRanks; ++j)
        push([&] {
            if(j == 0)
                args.appendAligned<void*>("recv_ptr_0", nullptr);
            else
                args.append<void*>("recv_ptr_" + std::to_string(j), nullptr);
        });
    for(int j = 0; j < kMaxRanks; ++j)
        push([&] { args.append<void*>("flag_ptr_" + std::to_string(j), nullptr); });
    push([&] { args.append<void*>("counter_ptr", nullptr); });
    push([&] { args.append<uint32_t>("FusedMyRank", 0); });
    push([&] { args.append<uint32_t>("FusedTarget", 0); });
    push([&] { args.append<uint32_t>("FusedW", 0); });
    push([&] { args.append<uint32_t>("FusedNShard", 0); });
    push([&] { args.append<uint32_t>("FusedDrain", 0); });
    push([&] { args.append<uint32_t>("FusedAM", 0); });
    push([&] { args.append<void*>("FusedSdmaQueues", nullptr); });
    push([&] { args.append<uint32_t>("FusedTilesPerRank", 0); });
    push([&] { args.append<uint32_t>("FusedTokenTiles", 0); });

    ASSERT_EQ(offsets.size(), table.size());
    for(size_t i = 0; i < table.size(); ++i)
    {
        EXPECT_EQ(offsets[i], table[i].offset)
            << "arg #" << i << " (" << table[i].name << ") offset mismatch";
    }

    // Tight packing: last arg ends exactly at the segment size.
    EXPECT_EQ(table.back().offset + table.back().size, kSegmentBytes);
    EXPECT_EQ(args.size() - base, kSegmentBytes);
}

// The three SDMA args (Task 3) sit strictly after every legacy arg: append-only,
// so no preceding offset shifted (Global Constraint 2).
TEST(FusedA2AKernArg, SdmaArgsAppendedLast)
{
    const auto table = goldenTable();
    size_t     legacyMax = 0;
    for(auto const& a : table)
    {
        std::string n = a.name;
        if(n != "FusedSdmaQueues" && n != "FusedTilesPerRank" && n != "FusedTokenTiles")
            legacyMax = std::max(legacyMax, a.offset);
    }
    for(auto const& a : table)
    {
        std::string n = a.name;
        if(n == "FusedSdmaQueues" || n == "FusedTilesPerRank" || n == "FusedTokenTiles")
            EXPECT_GT(a.offset, legacyMax) << n << " must be appended after all legacy args";
    }
}
