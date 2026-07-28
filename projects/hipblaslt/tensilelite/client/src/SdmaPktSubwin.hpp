// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// SDMA_PKT_COPY_LINEAR_SUBWIN (sub_op 4) packet definition for the fused
// GEMM+AllToAll SDMA offload route (ROCM-27524).
//
// This is the canonical, header-only home for the rectangular-copy SDMA packet
// that later codegen tasks fill in GPU assembly. The bit-field layout, the
// minus-one extent/pitch convention and the ELEMENTSIZE scaling below are the
// exact form that was validated byte-for-byte on MI355X (3 peers x 8 bands = 24
// packets, every dword bit-accurate, sentinel margin untouched). MORI only
// defines the flat COPY_LINEAR packet, so this struct is transcribed from the
// OSS 4.4 sdma.pkt field positions (cross-checked against ROCR's
// sdma_registers.h and the kernel's vega10_sdma_pkt_open.h -- all three agree).
//
// DO NOT reorder fields, change types, or "clean up" the reserved gaps: the
// static_asserts and the golden-vector unit test (SdmaPktSubwin_test.cpp) exist
// precisely to catch such drift. GFX12+ uses a DIFFERENT layout of the same
// size; this one is gfx9xx / gfx95x only.

#pragma once

#include <cstddef>

namespace TensileLite
{
    // SDMA COPY opcode (header op field) and the LINEAR_RECT sub-opcode.
    constexpr unsigned int SDMA_OP_COPY_SUBWIN          = 1;
    constexpr unsigned int SDMA_SUBOP_COPY_LINEAR_RECT  = 4;

    // -----------------------------------------------------------------------
    // 13-dword rectangular sub-window copy packet, pre-GFX12 layout.
    // -----------------------------------------------------------------------
    typedef struct SDMA_PKT_COPY_LINEAR_SUBWIN_TAG
    {
        union
        {
            struct
            {
                unsigned int op : 8;           // [7:0]
                unsigned int sub_op : 8;       // [15:8]
                unsigned int reserved_0 : 2;   // [17:16]
                unsigned int tmz : 1;          // [18]
                unsigned int reserved_1 : 10;  // [28:19]
                unsigned int elementsize : 3;  // [31:29]
            };
            unsigned int DW_0_DATA;
        } HEADER_UNION;

        union { unsigned int src_addr_31_0;  unsigned int DW_1_DATA; } SRC_ADDR_LO_UNION;
        union { unsigned int src_addr_63_32; unsigned int DW_2_DATA; } SRC_ADDR_HI_UNION;

        union
        {
            struct
            {
                unsigned int src_x : 14;       // [13:0]
                unsigned int reserved_0 : 2;
                unsigned int src_y : 14;       // [29:16]
                unsigned int reserved_1 : 2;
            };
            unsigned int DW_3_DATA;
        } DW_3_UNION;

        union
        {
            struct
            {
                unsigned int src_z : 11;       // [10:0]
                unsigned int reserved_0 : 2;
                unsigned int src_pitch : 19;   // [31:13]
            };
            unsigned int DW_4_DATA;
        } DW_4_UNION;

        union
        {
            struct
            {
                unsigned int src_slice_pitch : 28;  // [27:0]
                unsigned int reserved_0 : 4;
            };
            unsigned int DW_5_DATA;
        } DW_5_UNION;

        union { unsigned int dst_addr_31_0;  unsigned int DW_6_DATA; } DST_ADDR_LO_UNION;
        union { unsigned int dst_addr_63_32; unsigned int DW_7_DATA; } DST_ADDR_HI_UNION;

        union
        {
            struct
            {
                unsigned int dst_x : 14;       // [13:0]
                unsigned int reserved_0 : 2;
                unsigned int dst_y : 14;       // [29:16]
                unsigned int reserved_1 : 2;
            };
            unsigned int DW_8_DATA;
        } DW_8_UNION;

        union
        {
            struct
            {
                unsigned int dst_z : 11;       // [10:0]
                unsigned int reserved_0 : 2;
                unsigned int dst_pitch : 19;   // [31:13]
            };
            unsigned int DW_9_DATA;
        } DW_9_UNION;

        union
        {
            struct
            {
                unsigned int dst_slice_pitch : 28;  // [27:0]
                unsigned int reserved_0 : 4;
            };
            unsigned int DW_10_DATA;
        } DW_10_UNION;

        union
        {
            struct
            {
                unsigned int rect_x : 14;      // [13:0]
                unsigned int reserved_0 : 2;
                unsigned int rect_y : 14;      // [29:16]
                unsigned int reserved_1 : 2;
            };
            unsigned int DW_11_DATA;
        } DW_11_UNION;

        union
        {
            struct
            {
                unsigned int rect_z : 11;             // [10:0]
                unsigned int reserved_0 : 5;
                unsigned int dst_sw : 2;              // [17:16]
                unsigned int dst_cache_policy : 3;    // [20:18]
                unsigned int reserved_1 : 3;
                unsigned int src_sw : 2;              // [25:24]
                unsigned int src_cache_policy : 3;    // [28:26]
                unsigned int reserved_2 : 3;
            };
            unsigned int DW_12_DATA;
        } DW_12_UNION;
    } SDMA_PKT_COPY_LINEAR_SUBWIN;

    // -- Guardrails: total size and the offset of every dword. If a field is
    //    reordered or a reserved gap is resized, one of these will fire. --
    static_assert(sizeof(SDMA_PKT_COPY_LINEAR_SUBWIN) == 13 * sizeof(unsigned int),
                  "SUBWIN packet must be exactly 13 dwords");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, HEADER_UNION)      ==  0 * sizeof(unsigned int), "DW0 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, SRC_ADDR_LO_UNION) ==  1 * sizeof(unsigned int), "DW1 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, SRC_ADDR_HI_UNION) ==  2 * sizeof(unsigned int), "DW2 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_3_UNION)        ==  3 * sizeof(unsigned int), "DW3 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_4_UNION)        ==  4 * sizeof(unsigned int), "DW4 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_5_UNION)        ==  5 * sizeof(unsigned int), "DW5 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DST_ADDR_LO_UNION) ==  6 * sizeof(unsigned int), "DW6 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DST_ADDR_HI_UNION) ==  7 * sizeof(unsigned int), "DW7 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_8_UNION)        ==  8 * sizeof(unsigned int), "DW8 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_9_UNION)        ==  9 * sizeof(unsigned int), "DW9 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_10_UNION)       == 10 * sizeof(unsigned int), "DW10 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_11_UNION)       == 11 * sizeof(unsigned int), "DW11 offset");
    static_assert(offsetof(SDMA_PKT_COPY_LINEAR_SUBWIN, DW_12_UNION)       == 12 * sizeof(unsigned int), "DW12 offset");

    // -----------------------------------------------------------------------
    // Fill a rect-copy packet. This mirrors the validated harness builder
    // exactly and encodes the two conventions this whole route depends on:
    //   * every extent/pitch is stored MINUS ONE (hardware adds one back);
    //   * all coordinates/extents/pitches are in ELEMENTS, with the element
    //     size carried out-of-band in the ELEMENTSIZE header field
    //     (elementSizeLog2: bf16 => 1, i.e. 2-byte elements).
    // Addresses are the raw 64-bit pointers, split lo/hi. Slice pitch is a
    // don't-care for a single-plane copy (rect_z == 0) but is still encoded.
    // -----------------------------------------------------------------------
    inline SDMA_PKT_COPY_LINEAR_SUBWIN makeCopyRectPacket(unsigned long long srcBase,
                                                          unsigned          srcX,
                                                          unsigned          srcY,
                                                          unsigned          srcPitch,
                                                          unsigned          srcSlicePitch,
                                                          unsigned long long dstBase,
                                                          unsigned          dstX,
                                                          unsigned          dstY,
                                                          unsigned          dstPitch,
                                                          unsigned          dstSlicePitch,
                                                          unsigned          rectX,
                                                          unsigned          rectY,
                                                          unsigned          elementSizeLog2)
    {
        SDMA_PKT_COPY_LINEAR_SUBWIN p = {};

        p.HEADER_UNION.op          = SDMA_OP_COPY_SUBWIN;
        p.HEADER_UNION.sub_op      = SDMA_SUBOP_COPY_LINEAR_RECT;
        p.HEADER_UNION.elementsize = elementSizeLog2;

        p.SRC_ADDR_LO_UNION.src_addr_31_0  = (unsigned)(srcBase & 0xffffffffull);
        p.SRC_ADDR_HI_UNION.src_addr_63_32 = (unsigned)(srcBase >> 32);
        p.DW_3_UNION.src_x                 = srcX;
        p.DW_3_UNION.src_y                 = srcY;
        p.DW_4_UNION.src_z                 = 0;
        p.DW_4_UNION.src_pitch             = srcPitch - 1;
        p.DW_5_UNION.src_slice_pitch       = srcSlicePitch - 1;

        p.DST_ADDR_LO_UNION.dst_addr_31_0  = (unsigned)(dstBase & 0xffffffffull);
        p.DST_ADDR_HI_UNION.dst_addr_63_32 = (unsigned)(dstBase >> 32);
        p.DW_8_UNION.dst_x                 = dstX;
        p.DW_8_UNION.dst_y                 = dstY;
        p.DW_9_UNION.dst_z                 = 0;
        p.DW_9_UNION.dst_pitch             = dstPitch - 1;
        p.DW_10_UNION.dst_slice_pitch      = dstSlicePitch - 1;

        p.DW_11_UNION.rect_x = rectX - 1;
        p.DW_11_UNION.rect_y = rectY - 1;
        p.DW_12_UNION.rect_z = 0;  // one plane

        return p;
    }

} // namespace TensileLite
