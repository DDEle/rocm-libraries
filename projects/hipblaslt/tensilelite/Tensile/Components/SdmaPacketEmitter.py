# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter.
#
# Packet-DEPENDENT counterpart to SdmaRingEmitter (which is packet-INDEPENDENT
# ring plumbing): this module turns the all-to-all geometry (kernarg values +
# WG ids) into the 13-dword COPY_SUBWIN and 8-dword ATOMIC ADD_RTN_32 packet
# dword arrays, laid out in SGPRs; SdmaRingEmitter.emitPlacePacket then writes
# those dwords into the ring with s_store.
#
# The dwords live in SGPRs because every input is wave-uniform: the geometry
# comes from kernarg values and WG ids, so nothing here is per-lane. The ring
# stores are scalar for the same reason, which is what lets the whole submit
# path avoid v_readfirstlane and VCC (see SdmaRingEmitter).
#
# The live caller is GlobalWriteBatch._emitFusedA2ASdmaIssue, via
# emitBuildCopyPacket / emitBuildAtomicPacket.
#
# PROVENANCE of the dword layout. Bit positions are transcribed from AMD OSS 4.4
# sdma.pkt field positions, cross-checked against ROCR's sdma_registers.h and
# the kernel's vega10_sdma_pkt_open.h -- all three agree. The layout was then
# validated byte-for-byte on MI355X (24 packets, every dword bit-accurate) and
# end-to-end at MED/W=4 (recv byte-exact).
#
# Two conventions the route depends on, neither derivable from the field names:
# every extent and pitch is stored MINUS ONE (the hardware adds it back), and
# coords/extents/pitches are in ELEMENTS of the size named in the header, not in
# bytes. Do not "clean up" the reserved gaps or the <<13 pitch placement -- they
# look arbitrary because they are hardware-mandated. GFX12+ uses a DIFFERENT
# layout of the same size: this encoding is gfx9xx / gfx95x ONLY.
#
# Packet geometry, per (peer p, token-tile j), with this card == myRank. The
# geometry below is in bf16 elements; the packet's own addressing granularity is
# PACKET_ELEMENT_SIZE_LOG2 (see the two-element-sizes note further down):
#   COPY_SUBWIN:
#     src  = D + (j*MT1)*ldd + p*nShard
#     dst  = peer_ptr[p] + recvOffset + (myRank*N + j*MT1)*nShard
#     src pitch = ldd  ;  dst pitch = nShard
#     rect X = nShard (feature, contiguous) ; rect Y = min(MT1, N - j*MT1) (token)
#   ATOMIC ADD_RTN_32 -> peer_ptr[p] + myRank*4, addend 1 (raise the dest flag).
#
# COORDINATES ARE FOLDED INTO THE BASE ADDRESSES: src_x/src_y/dst_x/dst_y are
# emitted as a literal 0 and the offset added into the 64-bit base instead
# (addr(x, y) = base + y*pitch*elem + x*elem). What stays field-encoded is the
# pitches, the slice pitches and the rect extents; the emitter packs them
# unmasked, and their bounds are enforced at launch time by the guards in
# client/src/FusedA2AClient.cpp::runFusedA2A.
################################################################################

from rocisa.container import sgpr
from rocisa.code import Module
from rocisa.instruction import (
    SMovB32,
    SMulI32, SMulHIU32, SAddU32, SAddCU32, SAddU64, SSubU32, SMinU32,
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SOrB32,
)


# ---- COPY_SUBWIN header op/sub_op -------------------------------------------
SDMA_OP_COPY_SUBWIN         = 1
SDMA_SUBOP_COPY_LINEAR_RECT = 4
COPY_PACKET_DWORDS          = 13

# ---- ATOMIC header op/operation ---------------------------------------------
# operation is a 7-bit index into the TC atomic op table (ADD_RTN_32 = 15,
# ADD_RTN_64 = 47); RTN means the op returns the pre-op value, which SDMA drops.
SDMA_OP_ATOMIC         = 10
SDMA_ATOMIC_ADD_RTN_32 = 15
ATOMIC_PACKET_DWORDS   = 8

# TWO DIFFERENT "ELEMENT SIZES". Conflating them is a silent 8x address error:
#
#   D_DATA_ELEMENT_LOG2 = log2(sizeof(bf16)), the width of one D element in
#     BYTES. Converts element-unit geometry into the byte offset folded into
#     the base address, so it NEVER changes with the one below.
#
#   PACKET_ELEMENT_SIZE_LOG2 = the packet's ADDRESSING GRANULARITY (header field
#     [31:29]): the unit for x, the pitches, the slice pitches and rect_x. NOT
#     for y or rect_y, which are row indices the hardware does not scale.
#
# 16-byte elements are validated on MI355X (MED, W=4, recv byte-exact). The
# field is 3 bits wide, but that is not evidence the hardware accepts every
# encoding.
D_DATA_ELEMENT_LOG2      = 1
PACKET_ELEMENT_SIZE_LOG2 = 4

# How far to shift a bf16-element count down into packet-element units. Derived,
# never hardcoded: if either constant moves, every scaled field follows.
ELEMENT_SHIFT = PACKET_ELEMENT_SIZE_LOG2 - D_DATA_ELEMENT_LOG2   # 3

# Field widths (bits) of the dwords the _pack* helpers build. Stated here for
# the reader only -- the helpers do NOT mask, so these are the bounds the
# launch-time guards in FusedA2AClient.cpp must keep the geometry inside:
#   rect_x / rect_y      14 bits, at [13:0] and [29:16]
#   src_pitch/dst_pitch  19 bits, at [31:13] (above the 11-bit z field)
#   src/dst slice pitch  28 bits, at [27:0]


# Compile-time header dwords: op/sub_op/elementsize are all immediates, so DW0
# of each packet is folded here rather than built at runtime.
#
# COPY_HEADER_DW0 is 0x80000401, i.e. ABOVE INT32_MAX, so it must reach SMovB32
# as a hex STRING. rocisa's InstructionInput has no 64-bit integer variant: an
# int that does not fit a C++ 32-bit int falls through to the double branch and
# renders as "2147484673.0". Every immediate below therefore goes through hex().
# (s_mov_b32 itself takes the value fine -- a 32-bit literal constant; only the
# Python-side rendering is the hazard.)
COPY_HEADER_DW0 = ((SDMA_OP_COPY_SUBWIN & 0xFF)
                   | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8)
                   | ((PACKET_ELEMENT_SIZE_LOG2 & 0x7) << 29))
# The l bit (16) is left 0, which selects a plain fetch-add.
ATOMIC_HEADER_DW0 = ((SDMA_OP_ATOMIC & 0xFF) | ((SDMA_ATOMIC_ADD_RTN_32 & 0x7F) << 25))


class SdmaPacketEmitter:
    """Builds the COPY_SUBWIN + ATOMIC packet dword arrays in SGPRs from runtime
    inputs, in the layout documented at the top of this file.

    Every method takes the registers it uses and allocates none: unlike
    SdmaRingEmitter, which checks scratch in and out of `w`'s pools, this class
    never touches a pool and so takes no `w` at all. The dword SGPR block it
    fills is what SdmaRingEmitter.emitPlacePacket writes to the ring, so the two
    emitters compose without either knowing the other's internals. The three encoding conventions (minus-one extents/pitches,
    element units, field bit positions) are isolated in the `_pack*` helpers.

    The `_pack*` helpers build each field IN PLACE in its packet slot rather
    than through a scratch register: the destination is a plain SGPR like their
    inputs, so the "compute, then relay into the packet" step the VGPR layout
    forced has no reason to exist. Only the rect dword still needs one scratch,
    to hold the second extent while it is shifted.

    ALIASING CONTRACT: the packet block must be disjoint from every input SGPR
    and from `tmpS`. In-place packing writes the slot before reading some of its
    inputs, so an aliasing caller loses the input rather than merely the output.
    The block is also written in field order and never re-read here, so an
    overlapping second packet (see emitPlacePacket's reuse note) must not be
    built until the first one's stores have been emitted.
    """

    def __init__(self, macroTile1: int):
        # MT1 (token extent / rect_y) is a compile-time solution constant; the
        # geometric fields (p, j, myRank, M, N, nShard) are runtime SGPRs. The
        # element size is NOT per-instance: PACKET_ELEMENT_SIZE_LOG2 feeds the
        # module-level COPY_HEADER_DW0 and ELEMENT_SHIFT, and only the 16-byte
        # encoding is hardware-validated.
        self.mt1 = macroTile1

    # ---- field-packing helpers (isolate the encoding conventions) -----------

    def _toPacketElements(self, module, dstS, srcS, comment):
        """Convert a bf16-element count into packet-element units (>> 3).

        Applies to X-DIRECTION quantities ONLY: the pitches, the slice pitches
        and rect_x. It must NOT be applied to y or rect_y, which are row indices
        the hardware does not scale by ELEMENTSIZE, nor to the folded base
        addresses, which are byte offsets.

        Callers skip this helper entirely when the shift is 0. Divisibility is a
        launch-time precondition (FusedA2AClient.cpp); a non-multiple would
        truncate here."""
        module.add(SLShiftRightB32(dst=sgpr(dstS), src=sgpr(srcS),
                                   shiftHex=ELEMENT_SHIFT,
                                   comment=comment + " (bf16 elems -> packet elems)"))
        return module

    def _packPitchMinus1(self, module, dstS, pitchS, comment):
        """dword = (pitch - 1) << 13 -- the 19-bit pitch field at [31:13]; the z
        field [10:0] is left 0. Minus-one is the hardware pitch convention (it
        adds one back). The pitch arrives in bf16 elements and is scaled to
        packet elements first.

        Built in place in the packet slot; needs no scratch. dstS must not alias
        pitchS -- the scaling step writes dstS before the subtract reads it.

        NOT masked to 19 bits: an over-range pitch ORs into the neighbouring
        field. The bound is a launch-time precondition (FusedA2AClient.cpp)."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, dstS, pitchS, comment)
            src = dstS
        else:
            src = pitchS
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(src), src1=1,
                           comment=comment + " (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(dstS), src=sgpr(dstS), shiftHex=13,
                                  comment=comment + " (<< 13)"))
        return module

    def _packSliceMinus1(self, module, dstS, sliceS, comment):
        """dword = slice_pitch - 1 -- the 28-bit slice field at [27:0]. Scaled to
        packet elements like the pitches. Built in place; needs no scratch.

        NOT masked to 28 bits, and unlike the pitch and rect fields it is not
        bounds-checked at launch either -- but it is NOT a free field. The
        reference implementation asserts RECT_X * RECT_Y <= SRC_SLICE_PITCH (and
        the same for DST), which holds here by construction rather than by a
        guard; see emitComputeCopyFields for the two values and why each
        satisfies it."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, dstS, sliceS, comment)
            src = dstS
        else:
            src = sliceS
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(src), src1=1,
                           comment=comment + " (slice - 1)"))
        return module

    def _packRectMinus1(self, module, dstS, rectXS, rectYS, tmpS, comment):
        """dword = (rectX - 1) | ((rectY - 1) << 16) -- two 14-bit extents at
        [13:0] and [29:16], NEITHER masked: an over-range rect_x ORs straight
        into rect_y. Both bounds are launch-time preconditions
        (FusedA2AClient.cpp).

        BOTH extents are runtime SGPRs: rectY cannot be the compile-time MT1
        because the last token-tile is partial when N % MT1 != 0, and an
        unclamped MT1 would read past the end of D (emitComputeCopyFields clamps
        it). tmpS is ONE scratch SGPR -- the only `_pack*` helper still needing
        any, because the two extents have to exist at once to be OR-ed.

        rect_x IS scaled to packet elements; rect_y is NOT -- it counts ROWS,
        and ELEMENTSIZE scales only the X direction."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, dstS, rectXS, comment + " (rectX)")
            rectXsrc = dstS
        else:
            rectXsrc = rectXS
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(rectXsrc), src1=1,
                           comment=comment + " (rectX - 1)"))
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(rectYS), src1=1,
                           comment=comment + " (rectY - 1, rows: NOT scaled)"))
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(tmpS), shiftHex=16,
                                  comment=comment + " ((rectY-1) << 16)"))
        module.add(SOrB32(dst=sgpr(dstS), src0=sgpr(dstS), src1=sgpr(tmpS),
                          comment=comment + " | (rectY-1) << 16"))
        return module

    # ---- COPY_SUBWIN builder -----------------------------------------------

    def emitBuildCopyPacket(self, module, pktS,
                            srcBaseS, srcPitchS, srcSliceS,
                            dstBaseS, dstPitchS, dstSliceS,
                            rectXS, rectYS, tmpS):
        """Build the 13 COPY_SUBWIN dwords into pktS[0:13] from runtime SGPR
        inputs (pitches and extents in element units; caller does the
        arithmetic that produces them -- see emitComputeCopyFields). tmpS is ONE
        scratch SGPR, used only by the rect dword.

        All four coordinates are ZERO: emitComputeCopyFields folded them into
        srcBaseS / dstBaseS, so DW3 and DW8 are literal-0 moves rather than
        field packing. They are still written (the ring copies a fixed 13-dword
        block, and a stale register would be read as a coordinate).

        The two 64-bit bases are the one place a relay survives: DW1/DW2 and
        DW6/DW7 land on odd packet slots, and the 64-bit ops that produce the
        bases need a 2-aligned SReg_64 pair, so they cannot be computed in
        place. Two s_mov each is the whole cost.

        Field -> dword map:
          DW0 header (immediate), DW1/2 srcBase, DW3 0, DW4 srcPitch-1,
          DW5 srcSlice-1, DW6/7 dstBase, DW8 0, DW9 dstPitch-1,
          DW10 dstSlice-1, DW11 (rectX-1|rectY-1), DW12 0.
        """
        module.add(SMovB32(dst=sgpr(pktS + 0), src=hex(COPY_HEADER_DW0),
                           comment="SUBWIN DW0: op=COPY sub_op=RECT elementsize=log2(%dB)"
                                   % (1 << PACKET_ELEMENT_SIZE_LOG2)))
        module.add(SMovB32(dst=sgpr(pktS + 1), src=sgpr(srcBaseS + 0),
                           comment="SUBWIN DW1: srcBase lo"))
        module.add(SMovB32(dst=sgpr(pktS + 2), src=sgpr(srcBaseS + 1),
                           comment="SUBWIN DW2: srcBase hi"))
        module.add(SMovB32(dst=sgpr(pktS + 3), src=hex(0),
                           comment="SUBWIN DW3: src_x=0|src_y=0 (folded into srcBase)"))
        self._packPitchMinus1(module, pktS + 4, srcPitchS, "SUBWIN DW4: src_pitch-1")
        self._packSliceMinus1(module, pktS + 5, srcSliceS, "SUBWIN DW5: src_slice-1")
        module.add(SMovB32(dst=sgpr(pktS + 6), src=sgpr(dstBaseS + 0),
                           comment="SUBWIN DW6: dstBase lo"))
        module.add(SMovB32(dst=sgpr(pktS + 7), src=sgpr(dstBaseS + 1),
                           comment="SUBWIN DW7: dstBase hi"))
        module.add(SMovB32(dst=sgpr(pktS + 8), src=hex(0),
                           comment="SUBWIN DW8: dst_x=0|dst_y=0 (folded into dstBase)"))
        self._packPitchMinus1(module, pktS + 9, dstPitchS, "SUBWIN DW9: dst_pitch-1")
        self._packSliceMinus1(module, pktS + 10, dstSliceS, "SUBWIN DW10: dst_slice-1")
        self._packRectMinus1(module, pktS + 11, rectXS, rectYS, tmpS,
                             "SUBWIN DW11: rect_x-1|rect_y-1")
        module.add(SMovB32(dst=sgpr(pktS + 12), src=hex(0),
                           comment="SUBWIN DW12: rect_z=0, default cache/swizzle"))
        return module

    # ---- ATOMIC ADD_RTN_32 builder ------------------------------------------

    def emitBuildAtomicPacket(self, module, pktS, dstAddrS, addend=1):
        """Build the 8 ATOMIC ADD_RTN_32 dwords into pktS[0:8]: raise peer_ptr[p]
        [myRank] by `addend` (== 1). dstAddrS is a 2-SGPR pointer to the flag
        slot (caller computes peer_ptr[p] + myRank*4 -- see emitComputeFlagAddr;
        the stride is 4 because this ADD_RTN_32 writes 4 bytes). addend is a
        compile-time immediate (1).

        The caller may hand this the SAME block it used for the COPY packet --
        it is 8 dwords against the COPY's 13. See emitPlacePacket's reuse note
        for why that is safe and what ordering it requires."""
        module.add(SMovB32(dst=sgpr(pktS + 0), src=hex(ATOMIC_HEADER_DW0),
                           comment="ATOMIC DW0: op=ATOMIC operation=ADD_RTN_32"))
        module.add(SMovB32(dst=sgpr(pktS + 1), src=sgpr(dstAddrS + 0),
                           comment="ATOMIC DW1: addr lo"))
        module.add(SMovB32(dst=sgpr(pktS + 2), src=sgpr(dstAddrS + 1),
                           comment="ATOMIC DW2: addr hi"))
        module.add(SMovB32(dst=sgpr(pktS + 3), src=hex(addend & 0xFFFFFFFF),
                           comment="ATOMIC DW3: src_data lo (addend)"))
        module.add(SMovB32(dst=sgpr(pktS + 4), src=hex(0),
                           comment="ATOMIC DW4: src_data hi (unused by ADD_RTN_32)"))
        module.add(SMovB32(dst=sgpr(pktS + 5), src=hex(0),
                           comment="ATOMIC DW5: cmp_data lo (unused)"))
        module.add(SMovB32(dst=sgpr(pktS + 6), src=hex(0),
                           comment="ATOMIC DW6: cmp_data hi (unused)"))
        module.add(SMovB32(dst=sgpr(pktS + 7), src=hex(0),
                           comment="ATOMIC DW7: loop_interval=0"))
        return module

    # ---- field arithmetic (runtime geometry -> the SGPR inputs above) --

    def emitComputeCopyFields(self, module,
                              pS, jS, myRankS, mS, nS, nShardS,
                              addressDS, srcPitchS, recvBaseS,
                              outSrcBaseS, outSrcSliceS, outDstSliceS,
                              outRectYS, tmpS, tokenRowS, tmp64S):
        """Compute the runtime COPY inputs from (p, j, myRank, M, N, nShard),
        FOLDING the four coordinates into the two 64-bit base addresses:

          outSrcBase = AddressD + (j*MT1*ldd    + p*nShard) * sizeof(bf16)
          recvBase  += (myRank*N + j*MT1) * nShard          * sizeof(bf16)
          src_slice  = M * N                 (whole D plane)
          dst_slice  = MT1 * nShard          (one band's plane)
          rect_y     = min(MT1, N - j*MT1)   (clamped: last token-tile is partial)

        src_pitch = ldd and dst_pitch = nShard are passed straight through by the
        caller, as is rect_x = nShard. MT1 is the compile-time token extent.

        NEITHER SLICE PITCH IS A DON'T-CARE, despite the copy being a single
        plane (rect_z = 0). The reference implementation asserts
        RECT_X * RECT_Y <= SLICE_PITCH for both sides, and nothing at launch
        checks it, so the values above are what make it hold:
          dst is EXACTLY TIGHT -- (nShard>>3)*rect_y <= (MT1*nShard)>>3 reduces
            to rect_y <= MT1, which the clamp below guarantees with no slack.
          src has margin -- (nShard>>3)*rect_y <= (M*N)>>3 reduces to
            nShard*rect_y <= M*N, and nShard = AM/W <= M with rect_y <= N.
        Shrinking either one to a constant would break the assertion.

        BOTH folds are 64-BIT and must stay that way: neither product is bounded
        by anything now that the coordinates are folded in. Each widening
        multiply writes its high half first, so tmp64S+1 must not alias either
        source.

        The elements->bytes shift uses D_DATA_ELEMENT_LOG2, NOT
        PACKET_ELEMENT_SIZE_LOG2: a byte offset does not scale with the packet's
        addressing granularity.

        recvBaseS is updated IN PLACE; addressDS is read-only. All three
        scratch registers are dead on return and none is read by the caller:
        tmpS is one SGPR reused between uses, tokenRowS is one SGPR that must
        NOT be reused because j*MT1 stays live across the whole body (src fold,
        dst row, rect_y clamp), and tmp64S is a 2-ALIGNED pair (the 64-bit ops
        need SReg_64 alignment).
        """
        module.add(SMulI32(dst=sgpr(tokenRowS), src0=sgpr(jS), src1=self.mt1,
                           comment="token row of tile j = j * MT1 (folded into the bases)"))
        module.add(SMulI32(dst=sgpr(outSrcSliceS), src0=sgpr(mS), src1=sgpr(nS),
                           comment="src_slice = M * N (whole D plane; bounds RECT_X*RECT_Y)"))

        # --- src fold: AddressD + (j*MT1*ldd + p*nShard) * sizeof(bf16) ---
        module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr(pS), src1=sgpr(nShardS),
                           comment="src_x = p * nShard (folded into the base, not a field)"))
        module.add(SMulHIU32(dst=sgpr(tmp64S + 1), src0=sgpr(tokenRowS), src1=sgpr(srcPitchS),
                             comment="src row offset = j*MT1 * ldd (64-bit: unbounded in N and ldd) (hi)"))
        module.add(SMulI32(dst=sgpr(tmp64S + 0), src0=sgpr(tokenRowS), src1=sgpr(srcPitchS),
                           comment="src row offset = j*MT1 * ldd (64-bit: unbounded in N and ldd) (lo)"))
        module.add(SAddU32(dst=sgpr(tmp64S + 0), src0=sgpr(tmp64S + 0), src1=sgpr(tmpS),
                           comment="+ p*nShard (feature offset)"))
        module.add(SAddCU32(dst=sgpr(tmp64S + 1), src0=sgpr(tmp64S + 1), src1=0,
                            comment="propagate carry into the high word"))
        module.add(SLShiftLeftB64(dst=sgpr(tmp64S, 2), src=sgpr(tmp64S, 2),
                                  shiftHex=D_DATA_ELEMENT_LOG2,
                                  comment="src offset: elements -> bytes (sizeof(bf16))"))
        module.add(SAddU64(dst=sgpr(outSrcBaseS, 2), src0=sgpr(addressDS, 2),
                           src1=sgpr(tmp64S, 2),
                           comment="srcBase = D + src offset (src_x/src_y now 0)"))

        # --- dst fold: recvBase += (myRank*N + j*MT1) * nShard * sizeof(bf16) ---
        module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr(myRankS), src1=sgpr(nS),
                           comment="myRank * N"))
        module.add(SAddU32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(tokenRowS),
                           comment="dst row = myRank*N + j*MT1 (folded, not a field)"))
        module.add(SMulHIU32(dst=sgpr(tmp64S + 1), src0=sgpr(tmpS), src1=sgpr(nShardS),
                             comment="dst row offset = dst row * nShard (64-bit: unbounded in W and N) (hi)"))
        module.add(SMulI32(dst=sgpr(tmp64S + 0), src0=sgpr(tmpS), src1=sgpr(nShardS),
                           comment="dst row offset = dst row * nShard (64-bit: unbounded in W and N) (lo)"))
        module.add(SLShiftLeftB64(dst=sgpr(tmp64S, 2), src=sgpr(tmp64S, 2),
                                  shiftHex=D_DATA_ELEMENT_LOG2,
                                  comment="dst offset: elements -> bytes (sizeof(bf16))"))
        module.add(SAddU64(dst=sgpr(recvBaseS, 2), src0=sgpr(recvBaseS, 2),
                           src1=sgpr(tmp64S, 2),
                           comment="dstBase = recv slot + dst offset (dst_x/dst_y now 0)"))

        module.add(SMulI32(dst=sgpr(outDstSliceS), src0=sgpr(nShardS), src1=self.mt1,
                           comment="dst_slice = MT1 * nShard (one band's plane)"))
        # rect_y = min(MT1, N - j*MT1): the tail token-tile is partial when
        # N % MT1 != 0; an unclamped MT1 would read past the end of D.
        module.add(SSubU32(dst=sgpr(outRectYS), src0=sgpr(nS), src1=sgpr(tokenRowS),
                           comment="N - j*MT1 (tokens left in this tile)"))
        module.add(SMinU32(dst=sgpr(outRectYS), src0=sgpr(outRectYS), src1=self.mt1,
                           comment="rect_y = min(MT1, N - j*MT1) (clamp tail tile)"))
        return module

    def emitComputeFlagAddr(self, module, flagBaseS, myRankS, outAddrS, tmpS):
        """Compute the ATOMIC target peer_ptr[p] + myRank*4 into outAddrS (2
        SGPRs), a 64-bit add. flagBaseS is peer_ptr[p] (already selected by the
        caller via _fusedA2ALoadFlagBaseByRank). tmpS is one scratch SGPR.

        The flag is indexed by SOURCE rank only -- source j's tokenTiles ATOMICs
        accumulate into one slot, matching the "== tokenTiles" drain predicate.
        """
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(myRankS), shiftHex=2,
                                  comment="myRank * 4 (u32 flag-slot byte offset: the ATOMIC is an ADD_RTN_32)"))
        module.add(SAddU32(dst=sgpr(outAddrS + 0), src0=sgpr(flagBaseS + 0), src1=sgpr(tmpS),
                           comment="flag addr lo = peer_ptr[p] + myRank*4"))
        module.add(SAddCU32(dst=sgpr(outAddrS + 1), src0=sgpr(flagBaseS + 1), src1=0,
                            comment="flag addr hi (carry)"))
        return module
