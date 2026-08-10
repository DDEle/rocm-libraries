// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// SDMA packet definitions for the fused GEMM+AllToAll SDMA offload route
// (ROCM-27524). Despite the file name, this is the home for BOTH packets the
// route emits: the rectangular sub-window copy (SDMA_PKT_COPY_LINEAR_SUBWIN,
// below) and the 64-bit atomic add (SDMA_PKT_ATOMIC, at the bottom) that raises
// the destination flag once a copy lands. The name is kept as-is on purpose --
// renaming would churn the T1 gtest target and CMake for no functional gain.
//
// This is the canonical, header-only home for the packets that later codegen
// tasks fill in GPU assembly. The COPY_SUBWIN bit-field layout, the
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
//
// NOT DEAD CODE -- do not delete. No client-runtime translation unit includes
// this header: its only includer is its own gtest, so a reachability scan reads
// it as a self-justifying orphan, and it has been proposed for deletion on that
// basis before. Keeping it is a deliberate call. The packets that actually ship
// are built in GPU assembly by the Python mirror,
// Tensile/Components/SdmaPacketEmitter.py; what this header contributes is
// PROVENANCE, because its gtest drives the real encoder with the dword vectors
// validated byte-for-byte on MI355X -- that is where the Python side's goldens
// get their authority, and without it they are self-referential. The two files
// are held together mechanically by
// Tensile/Tests/unit/test_sdma_header_mirror.py, which compares opcodes, packet
// lengths, field widths and field offsets across the language boundary and
// FAILS (does not skip) if this header goes missing.

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

    // -----------------------------------------------------------------------
    // SDMA ATOMIC packet (op 10) -- 8 dwords, pre-GFX12 layout.
    //
    // Unlike the SUBWIN struct above (hand-transcribed because MORI lacks it),
    // this is copied VERBATIM from MORI's production
    // mori/include/mori/core/transport/sdma/sdma_pkt_struct.h
    // (SDMA_PKT_ATOMIC_TAG, 8 unions == 8 dwords). The route uses it in the
    // fetch-add form of MORI's CreateAtomicIncPacket (anvil_device.hpp:72):
    // op=ATOMIC, ADDR=flag slot, SRC_DATA=1 (the increment); the CMP_DATA / LOOP
    // dwords stay zero for a plain fetch-add. Because this is a fetch-add the
    // "l" (loop/return-old) header bit is left 0.
    //
    // `operation` is a 7-bit index into the TC atomic op table (ADD_RTN_32 = 15,
    // ADD_RTN_64 = 47). RTN means the op returns the pre-op value; this packet
    // has no field to receive one, so SDMA drops it.
    // -----------------------------------------------------------------------
    constexpr unsigned int SDMA_OP_ATOMIC         = 10;   // header op field
    constexpr unsigned int SDMA_ATOMIC_ADD_RTN_32 = 15;   // header operation field (fetch-add, 32-bit)

    typedef struct SDMA_PKT_ATOMIC_TAG
    {
        union
        {
            struct
            {
                unsigned int op : 8;           // [7:0]
                unsigned int sub_op : 8;       // [15:8]
                unsigned int l : 1;            // [16]  loop / return-old (0 for fetch-add)
                unsigned int reserved_0 : 8;   // [24:17]
                unsigned int operation : 7;    // [31:25]
            };
            unsigned int DW_0_DATA;
        } HEADER_UNION;

        union { unsigned int addr_31_0;      unsigned int DW_1_DATA; } ADDR_LO_UNION;
        union { unsigned int addr_63_32;     unsigned int DW_2_DATA; } ADDR_HI_UNION;
        union { unsigned int src_data_31_0;  unsigned int DW_3_DATA; } SRC_DATA_LO_UNION;
        union { unsigned int src_data_63_32; unsigned int DW_4_DATA; } SRC_DATA_HI_UNION;
        union { unsigned int cmp_data_31_0;  unsigned int DW_5_DATA; } CMP_DATA_LO_UNION;
        union { unsigned int cmp_data_63_32; unsigned int DW_6_DATA; } CMP_DATA_HI_UNION;

        union
        {
            struct
            {
                unsigned int loop_interval : 13;  // [12:0]
                unsigned int reserved_0 : 19;
            };
            unsigned int DW_7_DATA;
        } LOOP_UNION;
    } SDMA_PKT_ATOMIC;

    static_assert(sizeof(SDMA_PKT_ATOMIC) == 8 * sizeof(unsigned int),
                  "ATOMIC packet must be exactly 8 dwords");
    static_assert(offsetof(SDMA_PKT_ATOMIC, HEADER_UNION)       == 0 * sizeof(unsigned int), "ATOMIC DW0 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, ADDR_LO_UNION)      == 1 * sizeof(unsigned int), "ATOMIC DW1 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, ADDR_HI_UNION)      == 2 * sizeof(unsigned int), "ATOMIC DW2 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, SRC_DATA_LO_UNION)  == 3 * sizeof(unsigned int), "ATOMIC DW3 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, SRC_DATA_HI_UNION)  == 4 * sizeof(unsigned int), "ATOMIC DW4 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, CMP_DATA_LO_UNION)  == 5 * sizeof(unsigned int), "ATOMIC DW5 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, CMP_DATA_HI_UNION)  == 6 * sizeof(unsigned int), "ATOMIC DW6 offset");
    static_assert(offsetof(SDMA_PKT_ATOMIC, LOOP_UNION)         == 7 * sizeof(unsigned int), "ATOMIC DW7 offset");

    // -----------------------------------------------------------------------
    // Fill an ADD_RTN_32 fetch-add packet targeting `dstAddr` with `addend` (the
    // route passes addend == 1 to raise a flag). Mirrors MORI CreateAtomicIncPacket
    // but takes the addend explicitly. Address is the raw 64-bit pointer split
    // lo/hi; compare + loop dwords stay zero (unused for a plain fetch-add).
    // SRC_DATA_HI is read by the 64-bit ops only and stays zero.
    // -----------------------------------------------------------------------
    inline SDMA_PKT_ATOMIC makeAtomicAdd32Packet(unsigned long long dstAddr,
                                                 unsigned int      addend)
    {
        SDMA_PKT_ATOMIC p = {};

        p.HEADER_UNION.op        = SDMA_OP_ATOMIC;
        p.HEADER_UNION.operation = SDMA_ATOMIC_ADD_RTN_32;

        p.ADDR_LO_UNION.addr_31_0      = (unsigned)(dstAddr & 0xffffffffull);
        p.ADDR_HI_UNION.addr_63_32     = (unsigned)(dstAddr >> 32);
        p.SRC_DATA_LO_UNION.src_data_31_0  = addend;

        return p;
    }

} // namespace TensileLite
