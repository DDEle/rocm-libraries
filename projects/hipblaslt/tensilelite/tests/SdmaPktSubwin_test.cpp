// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Golden-vector regression test for the SDMA rectangular sub-window copy packet
// (SdmaPktSubwin.hpp). The packet layout + encoding conventions were validated
// byte-for-byte on MI355X; this test freezes that encoding so later codegen
// changes cannot silently corrupt it.
//
// Two vectors are pinned:
//   * HARNESS -- the exact parameters the on-hardware validation harness used
//     (~/sdma_rect_test.cpp), including the intentionally PADDED dst pitch
//     (kShard + kDstPad = 2624). This is the only byte sequence with real HW
//     backing and is the primary anchor.
//   * PRODUCTION -- the shape we will actually emit: an unpadded recv buffer
//     ([src, token, feature_shard], rowStride == nShard == 2560). It has no HW
//     bytes to copy, so its expected dwords are derived by hand from the same
//     field rules and annotated per dword. It is also the rect_x == dst_pitch
//     boundary case (the X extent exactly fills one row), a degenerate-looking
//     shape that is easy to break later, so it is worth its own anchor.

#include <gtest/gtest.h>

#include "SdmaPktSubwin.hpp"

using namespace TensileLite;

namespace
{
    // Shared problem shape (elements). Matches ~/sdma_rect_test.cpp.
    constexpr unsigned kN        = 18432;  // full output width == src row pitch
    constexpr unsigned kShard    = 2560;   // columns per peer == rect X extent
    constexpr unsigned kBand     = 256;    // rows per packet == rect Y extent
    constexpr unsigned kM        = 2048;
    constexpr unsigned kDstPad   = 64;
    constexpr unsigned kDstPitch = kShard + kDstPad;  // 2624, padded (harness)
    constexpr unsigned kDstRows  = kM + kBand;        // 2304

    constexpr unsigned kSrcSlice = kM * kN;             // src_slice_pitch (harness)
    constexpr unsigned kDstSlice = kDstRows * kDstPitch; // dst_slice_pitch (harness)

    // View the packet as 13 raw dwords for comparison.
    const unsigned int* asDwords(const SDMA_PKT_COPY_LINEAR_SUBWIN& p)
    {
        return reinterpret_cast<const unsigned int*>(&p);
    }
}

// -- HARNESS vector: dstPitch = 2624 (padded). Byte-for-byte the packet the
//    MI355X validation emitted (with zero addresses, which the harness fills at
//    runtime; the address dwords are shape-independent and checked as 0 here). --
TEST(SdmaPktSubwin, HarnessGoldenVector)
{
    auto p = makeCopyRectPacket(/*srcBase=*/0,
                                /*srcX=*/kShard, /*srcY=*/0,
                                /*srcPitch=*/kN, /*srcSlicePitch=*/kSrcSlice,
                                /*dstBase=*/0,
                                /*dstX=*/0, /*dstY=*/0,
                                /*dstPitch=*/kDstPitch, /*dstSlicePitch=*/kDstSlice,
                                /*rectX=*/kShard, /*rectY=*/kBand,
                                /*elementSizeLog2=*/1);

    // Field-by-field derivation of each expected dword:
    //   DW0  op=1 | sub_op=4<<8 | elementsize=1<<29            = 0x20000401
    //   DW1  src_addr_lo (0 here)                              = 0x00000000
    //   DW2  src_addr_hi (0 here)                              = 0x00000000
    //   DW3  src_x=2560 | src_y=0<<16                          = 0x00000A00
    //   DW4  src_z=0 | (src_pitch-1=18431)<<13                 = 0x08FFE000
    //   DW5  src_slice_pitch-1 = kM*kN-1 = 37748735            = 0x023FFFFF
    //   DW6  dst_addr_lo (0 here)                              = 0x00000000
    //   DW7  dst_addr_hi (0 here)                              = 0x00000000
    //   DW8  dst_x=0 | dst_y=0<<16                             = 0x00000000
    //   DW9  dst_z=0 | (dst_pitch-1=2623)<<13                  = 0x0147E000
    //   DW10 dst_slice_pitch-1 = 2304*2624-1 = 6045695         = 0x005C3FFF
    //   DW11 rect_x-1=2559 | (rect_y-1=255)<<16                = 0x00FF09FF
    //   DW12 rect_z=0, cache/swizzle all default               = 0x00000000
    const unsigned int expected[13] = {
        0x20000401u, 0x00000000u, 0x00000000u, 0x00000A00u, 0x08FFE000u,
        0x023FFFFFu, 0x00000000u, 0x00000000u, 0x00000000u, 0x0147E000u,
        0x005C3FFFu, 0x00FF09FFu, 0x00000000u};

    const unsigned int* got = asDwords(p);
    for(int i = 0; i < 13; ++i)
        EXPECT_EQ(got[i], expected[i]) << "harness DW_" << i << " mismatch";
}

// -- PRODUCTION vector: dstPitch = 2560 (unpadded recv buffer). No HW bytes;
//    expected dwords derived by hand from the same rules. Only the dst pitch
//    and dst slice pitch differ from the harness vector. --
TEST(SdmaPktSubwin, ProductionGoldenVector)
{
    constexpr unsigned kProdDstPitch = 2560;              // == kShard, no pad
    constexpr unsigned kProdDstSlice = kBand * kProdDstPitch;  // one band's plane

    auto p = makeCopyRectPacket(/*srcBase=*/0,
                                /*srcX=*/kShard, /*srcY=*/0,
                                /*srcPitch=*/kN, /*srcSlicePitch=*/kSrcSlice,
                                /*dstBase=*/0,
                                /*dstX=*/0, /*dstY=*/0,
                                /*dstPitch=*/kProdDstPitch, /*dstSlicePitch=*/kProdDstSlice,
                                /*rectX=*/kShard, /*rectY=*/kBand,
                                /*elementSizeLog2=*/1);

    // Derivation (same field rules; only DW9/DW10 change vs harness):
    //   DW0  op=1 | sub_op=4<<8 | elementsize=1<<29            = 0x20000401
    //   DW3  src_x=2560                                        = 0x00000A00
    //   DW4  (src_pitch-1=18431)<<13                           = 0x08FFE000
    //   DW5  src_slice_pitch-1 = kM*kN-1                       = 0x023FFFFF
    //   DW9  dst_z=0 | (dst_pitch-1=2559)<<13                  = 0x013FE000
    //          note: dst_pitch field (2559) == rect_x field (2559) here --
    //          the rect exactly fills one destination row (boundary case).
    //   DW10 dst_slice_pitch-1 = 256*2560-1 = 655359           = 0x0009FFFF
    //   DW11 rect_x-1=2559 | (rect_y-1=255)<<16                = 0x00FF09FF
    const unsigned int expected[13] = {
        0x20000401u, 0x00000000u, 0x00000000u, 0x00000A00u, 0x08FFE000u,
        0x023FFFFFu, 0x00000000u, 0x00000000u, 0x00000000u, 0x013FE000u,
        0x0009FFFFu, 0x00FF09FFu, 0x00000000u};

    const unsigned int* got = asDwords(p);
    for(int i = 0; i < 13; ++i)
        EXPECT_EQ(got[i], expected[i]) << "production DW_" << i << " mismatch";

    // Explicitly pin the "looks degenerate" boundary: X extent fills the row.
    EXPECT_EQ(p.DW_11_UNION.rect_x, p.DW_9_UNION.dst_pitch)
        << "production shape must be the rect_x == dst_pitch boundary case";
}

// Structural sanity: the packet really is 13 dwords and the sub-op is the
// rect variant. (The layout offsets are locked by static_assert in the header;
// this just documents the runtime-visible expectation.)
TEST(SdmaPktSubwin, PacketSizeAndSubOp)
{
    EXPECT_EQ(sizeof(SDMA_PKT_COPY_LINEAR_SUBWIN), 13u * sizeof(unsigned int));
    auto p = makeCopyRectPacket(0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1);
    EXPECT_EQ(p.HEADER_UNION.op, SDMA_OP_COPY_SUBWIN);
    EXPECT_EQ(p.HEADER_UNION.sub_op, SDMA_SUBOP_COPY_LINEAR_RECT);
}

// -- ATOMIC ADD64 golden vector. The route raises a destination flag with a
//    fetch-add-1 packet (MORI CreateAtomicIncPacket form). This freezes the
//    8-dword encoding. NOTE: unlike the SUBWIN vectors above, this packet has
//    NO on-hardware backing yet (see the header's RISK note); the expected
//    dwords are derived by hand from the MORI field rules. --
TEST(SdmaPktAtomic, Add64GoldenVector)
{
    // A flag slot address with distinct lo/hi bytes so a lo/hi swap would show.
    constexpr unsigned long long kFlagAddr = 0x0000ABCD12345678ull;
    auto p = makeAtomicAdd64Packet(kFlagAddr, /*addend=*/1);

    // Field-by-field derivation of each expected dword:
    //   DW0  op=10 | operation=47<<25                           = 0x5E00000A
    //   DW1  addr_lo                                            = 0x12345678
    //   DW2  addr_hi                                            = 0x0000ABCD
    //   DW3  src_data_lo = addend lo = 1                        = 0x00000001
    //   DW4  src_data_hi = addend hi = 0                        = 0x00000000
    //   DW5  cmp_data_lo (unused for fetch-add)                 = 0x00000000
    //   DW6  cmp_data_hi (unused for fetch-add)                 = 0x00000000
    //   DW7  loop_interval = 0                                  = 0x00000000
    const unsigned int expected[8] = {
        0x5E00000Au, 0x12345678u, 0x0000ABCDu, 0x00000001u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

    const unsigned int* got = reinterpret_cast<const unsigned int*>(&p);
    for(int i = 0; i < 8; ++i)
        EXPECT_EQ(got[i], expected[i]) << "atomic DW_" << i << " mismatch";
}

// Structural sanity for the ATOMIC packet: 8 dwords, op/operation fields, and
// the fetch-add (l==0) form. Layout offsets are locked by static_assert.
TEST(SdmaPktAtomic, PacketSizeAndOp)
{
    EXPECT_EQ(sizeof(SDMA_PKT_ATOMIC), 8u * sizeof(unsigned int));
    auto p = makeAtomicAdd64Packet(0, 1);
    EXPECT_EQ(p.HEADER_UNION.op, SDMA_OP_ATOMIC);
    EXPECT_EQ(p.HEADER_UNION.operation, SDMA_ATOMIC_ADD64);
    EXPECT_EQ(p.HEADER_UNION.l, 0u);
}
