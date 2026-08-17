# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter: turns the all-to-all geometry (kernarg
# values + WG ids) into the 13-dword COPY_SUBWIN and 8-dword ATOMIC ADD_RTN_32
# dword arrays.  SdmaRingEmitter -- the packet-INDEPENDENT ring plumbing --
# then writes them into the ring with s_store.
#
# The dwords live in SGPRs because every input is wave-uniform, which is what
# lets the whole submit path avoid v_readfirstlane and VCC.
#
# Packet geometry, per (peer p, token-tile j), with this card == myRank, in
# bf16 elements:
#   COPY_SUBWIN:
#     src = D + (j*MT1)*ldd + p*nShard                     src pitch = ldd
#     dst = peer_ptr[p] + recvOffset + (myRank*N + j*MT1)*nShard
#                                                          dst pitch = nShard
#     rect X = nShard (feature, contiguous) ; rect Y = min(MT1, N - j*MT1)
#   ATOMIC ADD_RTN_32 -> peer_ptr[p] + myRank*4, addend 1 (raise the dest flag).
#
# Encoding conventions, none of them derivable from the field names: every
# extent and pitch is stored MINUS ONE (the hardware adds it back); coords,
# extents and pitches are in ELEMENTS of the size named in the header, not in
# bytes; and the four coordinates are FOLDED INTO THE BASE ADDRESSES
# (addr(x, y) = base + y*pitch*elem + x*elem), so src_x/src_y/dst_x/dst_y are
# emitted as a literal 0.  The reserved gaps and the <<13 pitch placement look
# arbitrary because they are hardware-mandated.  Every field is packed
# UNMASKED, so an over-range value ORs into its neighbour; the bounds are
# enforced at launch by client/src/FusedA2AClient.cpp::runFusedA2A.
#
# This encoding is gfx9xx / gfx95x ONLY -- GFX12+ uses a different layout of
# the same size.  Bit positions are transcribed from AMD OSS 4.4 sdma.pkt,
# cross-checked against ROCR's sdma_registers.h and the kernel's
# vega10_sdma_pkt_open.h (all three agree), then validated byte-for-byte on
# MI355X and end-to-end -- recv byte-exact -- at MED / N2000 / N2047 / N2048 /
# N4096, W=4 and W=8.
################################################################################

from rocisa.container import sgpr
from rocisa.instruction import (
    SMovB32,
    SMulI32, SMulHIU32, SAddU32, SAddCU32, SAddU64, SSubU32, SMinU32,
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SOrB32,
)


SDMA_OP_COPY_SUBWIN         = 1
SDMA_SUBOP_COPY_LINEAR_RECT = 4
COPY_PACKET_DWORDS          = 13

# operation is a 7-bit index into the TC atomic op table (ADD_RTN_32 = 15,
# ADD_RTN_64 = 47); RTN means the op returns the pre-op value, which SDMA drops.
SDMA_OP_ATOMIC         = 10
SDMA_ATOMIC_ADD_RTN_32 = 15
ATOMIC_PACKET_DWORDS   = 8

# Two different element sizes; conflating them is a silent 8x address error.
D_DATA_ELEMENT_LOG2      = 1   # sizeof(bf16) in BYTES: element geometry -> byte offset
PACKET_ELEMENT_SIZE_LOG2 = 4   # packet addressing granularity (header [31:29]); only 16B is validated
ELEMENT_SHIFT = PACKET_ELEMENT_SIZE_LOG2 - D_DATA_ELEMENT_LOG2   # 3

COPY_HEADER_DW0 = ((SDMA_OP_COPY_SUBWIN & 0xFF)
                   | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8)
                   | ((PACKET_ELEMENT_SIZE_LOG2 & 0x7) << 29))
ATOMIC_HEADER_DW0 = ((SDMA_OP_ATOMIC & 0xFF) | ((SDMA_ATOMIC_ADD_RTN_32 & 0x7F) << 25))


class SdmaPacketEmitter:
    """Builds the COPY_SUBWIN + ATOMIC packet dword arrays in SGPRs, in the
    layout documented at the top of this file.

    Allocates nothing -- every method takes the registers it uses, so unlike
    SdmaRingEmitter this class never touches a pool and takes no `w`.

    ALIASING CONTRACT: the packet block must be disjoint from every input SGPR
    and from `tmpS`.  The `_pack*` helpers build each field IN PLACE in its
    packet slot, writing the slot before reading some of its inputs, so an
    aliasing caller loses the input rather than merely the output.  The block
    is written in field order and never re-read here, so an overlapping second
    packet must not be built until the first one's stores have been emitted
    (see emitPlacePacket's reuse note).
    """

    def __init__(self, macroTile1: int):
        # MT1 (token extent / rect_y) is a compile-time solution constant; the
        # geometric fields (p, j, myRank, M, N, nShard) are runtime SGPRs.
        self.mt1 = macroTile1

    # ---- field-packing helpers (isolate the encoding conventions) -----------

    def _toPacketElements(self, module, dstS, srcS, comment):
        """Convert a bf16-element count into packet-element units.  Applies to
        X-DIRECTION quantities ONLY.  Divisibility is a launch-time
        precondition (FusedA2AClient.cpp); a non-multiple would truncate."""
        module.add(SLShiftRightB32(dst=sgpr(dstS), src=sgpr(srcS),
                                   shiftHex=ELEMENT_SHIFT,
                                   comment=comment + " (bf16 elems -> packet elems)"))

    def _packPitchMinus1(self, module, dstS, pitchS, comment):
        """dword = (pitch - 1) << 13 -- the 19-bit pitch field at [31:13]; the
        z field [10:0] is left 0.  Built in place, so dstS must not alias
        pitchS: the scaling step writes dstS before the subtract reads it."""
        self._toPacketElements(module, dstS, pitchS, comment)
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(dstS), src1=1,
                           comment=comment + " (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(dstS), src=sgpr(dstS), shiftHex=13,
                                  comment=comment + " (<< 13)"))

    def _packSliceMinus1(self, module, dstS, sliceS, comment):
        """dword = slice_pitch - 1 -- the 28-bit slice field at [27:0].  NOT a
        free field despite rect_z being 0; see emitComputeCopyFields for the
        assertion it has to satisfy."""
        self._toPacketElements(module, dstS, sliceS, comment)
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(dstS), src1=1,
                           comment=comment + " (slice - 1)"))

    def _packRectMinus1(self, module, dstS, rectXS, rectYS, tmpS, comment):
        """dword = (rectX - 1) | ((rectY - 1) << 16) -- two 14-bit extents at
        [13:0] and [29:16].  rect_x IS scaled to packet elements; rect_y is
        NOT, it counts ROWS.  rectY is a runtime SGPR rather than the
        compile-time MT1 because the last token-tile is partial when
        N % MT1 != 0, and an unclamped MT1 would read past the end of D.  tmpS
        is one scratch SGPR -- the only `_pack*` helper needing any, because
        both extents have to exist at once to be OR-ed."""
        self._toPacketElements(module, dstS, rectXS, comment + " (rectX)")
        module.add(SSubU32(dst=sgpr(dstS), src0=sgpr(dstS), src1=1,
                           comment=comment + " (rectX - 1)"))
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(rectYS), src1=1,
                           comment=comment + " (rectY - 1, rows: NOT scaled)"))
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(tmpS), shiftHex=16,
                                  comment=comment + " ((rectY-1) << 16)"))
        module.add(SOrB32(dst=sgpr(dstS), src0=sgpr(dstS), src1=sgpr(tmpS),
                          comment=comment + " | (rectY-1) << 16"))

    # ---- COPY_SUBWIN builder -----------------------------------------------

    def emitBuildCopyPacket(self, module, pktS,
                            srcBaseS, srcPitchS, srcSliceS,
                            dstBaseS, dstPitchS, dstSliceS,
                            rectXS, rectYS, tmpS):
        """Build the 13 COPY_SUBWIN dwords into pktS[0:13] from runtime SGPR
        inputs in element units (see emitComputeCopyFields).  tmpS is one
        scratch SGPR, used only by the rect dword.

        The literal-0 coordinates are still written: the ring copies a fixed
        13-dword block, and a stale register would be read as a coordinate.
        The two 64-bit bases are the one place a relay survives -- DW1/DW2 and
        DW6/DW7 land on odd packet slots while the 64-bit ops that produce the
        bases need a 2-aligned SReg_64 pair.
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

    # ---- ATOMIC ADD_RTN_32 builder ------------------------------------------

    def emitBuildAtomicPacket(self, module, pktS, dstAddrS):
        """Build the 8 ATOMIC ADD_RTN_32 dwords into pktS[0:8]: raise
        peer_ptr[p][myRank] by 1.  dstAddrS is a 2-SGPR pointer to the flag
        slot (see emitComputeFlagAddr; the stride is 4 because this
        ADD_RTN_32 writes 4 bytes).

        The caller may hand this the SAME block it used for the COPY packet --
        8 dwords against the COPY's 13.  See emitPlacePacket's reuse note for
        why that is safe and what ordering it requires."""
        module.add(SMovB32(dst=sgpr(pktS + 0), src=hex(ATOMIC_HEADER_DW0),
                           comment="ATOMIC DW0: op=ATOMIC operation=ADD_RTN_32"))
        module.add(SMovB32(dst=sgpr(pktS + 1), src=sgpr(dstAddrS + 0),
                           comment="ATOMIC DW1: addr lo"))
        module.add(SMovB32(dst=sgpr(pktS + 2), src=sgpr(dstAddrS + 1),
                           comment="ATOMIC DW2: addr hi"))
        module.add(SMovB32(dst=sgpr(pktS + 3), src=hex(1),
                           comment="ATOMIC DW3: src_data lo (addend)"))
        module.add(SMovB32(dst=sgpr(pktS + 4), src=hex(0),
                           comment="ATOMIC DW4: src_data hi (unused by ADD_RTN_32)"))
        module.add(SMovB32(dst=sgpr(pktS + 5), src=hex(0),
                           comment="ATOMIC DW5: cmp_data lo (unused)"))
        module.add(SMovB32(dst=sgpr(pktS + 6), src=hex(0),
                           comment="ATOMIC DW6: cmp_data hi (unused)"))
        module.add(SMovB32(dst=sgpr(pktS + 7), src=hex(0),
                           comment="ATOMIC DW7: loop_interval=0"))

    # ---- field arithmetic (runtime geometry -> the SGPR inputs above) --

    def emitComputeCopyFields(self, module,
                              pS, jS, myRankS, mS, nS, nShardS,
                              addressDS, srcPitchS, recvBaseS,
                              outSrcBaseS, outSrcSliceS, outDstSliceS,
                              outRectYS, tmpS, tokenRowS, tmp64S):
        """Compute the runtime COPY inputs from (p, j, myRank, M, N, nShard),
        folding the four coordinates into the two 64-bit base addresses.
        src_pitch = ldd, dst_pitch = nShard and rect_x = nShard are passed
        straight through by the caller.

        NEITHER SLICE PITCH IS A DON'T-CARE, despite the copy being a single
        plane.  The values below are what make the reference implementation's
        RECT_X * RECT_Y <= SLICE_PITCH assertion hold:
          dst is EXACTLY TIGHT -- (nShard>>3)*rect_y <= (MT1*nShard)>>3 reduces
            to rect_y <= MT1, which the clamp below guarantees with no slack.
          src has margin -- (nShard>>3)*rect_y <= (M*N)>>3 reduces to
            nShard*rect_y <= M*N, and nShard = AM/W <= M with rect_y <= N.
        Shrinking either one to a constant would break the assertion.

        BOTH folds are 64-BIT and must stay that way: neither product is
        bounded now that the coordinates are folded in.  Each widening multiply
        writes its high half first, so tmp64S+1 must not alias either source.
        The elements->bytes shift uses D_DATA_ELEMENT_LOG2, NOT
        PACKET_ELEMENT_SIZE_LOG2: a byte offset does not scale with the
        packet's addressing granularity.

        recvBaseS is updated IN PLACE; addressDS is read-only.  All three
        scratch registers are dead on return: tmpS is reused between uses,
        tokenRowS must NOT be (j*MT1 stays live across the whole body), and
        tmp64S is a 2-ALIGNED pair.
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
        module.add(SSubU32(dst=sgpr(outRectYS), src0=sgpr(nS), src1=sgpr(tokenRowS),
                           comment="N - j*MT1 (tokens left in this tile)"))
        module.add(SMinU32(dst=sgpr(outRectYS), src0=sgpr(outRectYS), src1=self.mt1,
                           comment="rect_y = min(MT1, N - j*MT1) (clamp tail tile)"))

    def emitComputeFlagAddr(self, module, flagBaseS, myRankS, outAddrS, tmpS):
        """Compute the ATOMIC target peer_ptr[p] + myRank*4 into outAddrS (2
        SGPRs), a 64-bit add.  flagBaseS is peer_ptr[p], already selected by
        the caller.  tmpS is one scratch SGPR.

        The flag is indexed by SOURCE rank only -- source j's tokenTiles
        ATOMICs accumulate into one slot, matching the "== tokenTiles" drain
        predicate.
        """
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(myRankS), shiftHex=2,
                                  comment="myRank * 4 (u32 flag-slot byte offset: the ATOMIC is an ADD_RTN_32)"))
        module.add(SAddU32(dst=sgpr(outAddrS + 0), src0=sgpr(flagBaseS + 0), src1=sgpr(tmpS),
                           comment="flag addr lo = peer_ptr[p] + myRank*4"))
        module.add(SAddCU32(dst=sgpr(outAddrS + 1), src0=sgpr(flagBaseS + 1), src1=0,
                            comment="flag addr hi (carry)"))
