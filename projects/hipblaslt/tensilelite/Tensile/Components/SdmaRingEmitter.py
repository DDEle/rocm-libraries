# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA ring-buffer producer emitter.
#
# Packet-INDEPENDENT rocisa emitter for the ring skeleton: the assembly
# a GPU producer runs to reserve space in a host-created SDMA ring, place
# already-built packet dwords, and ring the doorbell -- WITHOUT knowing what the
# packet is. SdmaPacketEmitter builds the packet dwords and calls placePacket.
# The caller is GlobalWriteBatch._emitFusedA2ASdmaIssue, which invokes
# emitReserveQueueSpace, then emitPlacePacket twice (COPY then ATOMIC), then
# emitSubmitPacket.
#
# The device handle it consumes is the W-element SdmaQueueDeviceHandle array
# (client/include/SdmaQueue.hpp) passed in via the FusedSdmaQueues kernarg
# (intra-segment offset 160). The 7x8-byte field layout below is the byte
# contract locked by the static_asserts in that header; do not reorder.
#
# THIS WHOLE PATH IS SCALAR -- every store, every load, and the CAS. Nothing
# here touches a VGPR, EXEC or VCC. That was arrived at one piece at a time,
# each measured on gfx950/MI355X first, because the ISA cannot answer any of it:
#
#   stores  an SDMA engine executed a ring packet written by s_store...glc and
#           a doorbell rung by s_store_dwordx2...glc, with a "doorbell not
#           rung" control proving the engine does not merely poll wptr.
#   CAS     512 workgroups spanning all 8 XCDs incremented one 64-bit counter
#           through s_atomic_cmpswap_x2...glc with zero lost updates, against a
#           non-atomic control of the same shape that lost 99.5% of them.
#   loads   a spin on s_load...glc observed a value published by a peer
#           workgroup, and separately one published by the host, in tens of
#           thousands of iterations each -- i.e. it really re-reads. The same
#           spin WITHOUT glc never observed either one in 20 million iterations.
#           A load in a spin is a different question from a one-shot load, and
#           the failure mode is a hang rather than a wrong value.
#
# SMEM has NO scope field, just the legacy GLC bit (CDNA4 ISA Table 75);
# sc0/sc1/nt/slc are rejected outright by the assembler on SMEM.
# ⚠ AND GLC MEANS THREE DIFFERENT THINGS depending on the opcode:
#     STORE   forces the write past BOTH the K$ and L2 -- one bit covering
#             every scope the vector side needed three combinations for.
#     LOAD    forces the read past both, which is what turns a repeated load
#             into an actual poll instead of one value read many times.
#     ATOMIC  selects return-of-pre-op (ISA 8.2.2) and says nothing about
#             caches -- there is no bit left to ask for a scope, which is why
#             the CAS had to be measured rather than reasoned about.
#
# ⚠ A scalar store WITHOUT glc is never visible to anyone -- not to the SDMA
# engine, not to the host, not even after the kernel ends, because the
# end-of-kernel release flushes L2 and the vector caches but not the K$. No
# fault, no diagnostic, just a line sitting dirty forever. That is why glc is
# hard-coded in the emitters below and is NOT a parameter: there is no caller
# for whom omitting it would be correct.
################################################################################

from rocisa.container import sgpr, SMEMModifiers
from rocisa.code import Module, Label
from rocisa.instruction import (
    SMovB32, SLoadB64,
    SAddU32, SAddCU32, SSubU32, SSubBU32,
    SAndB32, SLShiftRightB32,
    SCmpEQU32, SCmpLtU32,
    SCBranchSCC0, SCBranchSCC1, SBranch,
    SWaitCnt, SSleep,
    SStoreB32, SStoreB64, SStoreB128, SAtomicCmpswapX2,
)


def _smemStore(glc: bool = True):
    """SMEMModifiers for a ring store. glc is not optional in practice -- see the
    hazard note in the file header -- so this exists to make every store site
    read the same and to keep `isStore` from being forgotten."""
    assert glc, "a scalar ring store without glc is never visible to anyone"
    return SMEMModifiers(glc=True, isStore=True)


# 256 KB SDMA ring, matching client/include/SdmaQueue.hpp SDMA_QUEUE_SIZE. Power of
# two => WrapIntoRing is an AND mask, never a divide.
SDMA_QUEUE_SIZE = 256 * 1024
assert (SDMA_QUEUE_SIZE & (SDMA_QUEUE_SIZE - 1)) == 0, "ring size must be a power of two"

# Byte offsets of every SdmaQueueDeviceHandle field (contract: the static_asserts
# in client/include/SdmaQueue.hpp lock these). Pointers are 8 bytes; the last field
# is a VALUE seed, not a pointer.
OFF_queueBuf         = 0    # ring base (uint32_t*, dword-addressed)
OFF_rptr             = 8    # hardware read pointer  (SYSTEM-scope read)
OFF_wptr             = 16   # hardware write pointer (AGENT-scope write)
OFF_doorbell         = 24   # doorbell               (SYSTEM-scope write)
OFF_cachedWptr       = 32   # producer reservation cursor (AGENT-scope CAS)
OFF_committedWptr    = 40   # commit-serialization cursor (AGENT-scope)
OFF_cachedHwReadIndex = 48  # per-producer private cache SEED (value, never stored back)


class SdmaRingEmitter:
    """Packet-independent SDMA ring producer, emitted as rocisa Modules.

    One instance is stateless; every method takes the registers it operates on
    (the caller -- KernelWriterAssembly -- owns the pools) plus a
    `w` context exposing `.sgprPool` and `.labels`, mirroring the
    GL2PrefetchLoad component. Persistent per-producer state lives in caller-
    owned SGPRs:
      * handleBase (2 SGPRs): pointer to this peer's SdmaQueueDeviceHandle.
      * cachedHwReadIdx (2 SGPRs): the private CanWriteUpto cache. The caller
        seeds it ONCE from handle+48 at setup; this emitter reads and refreshes
        it in-register and NEVER stores it back to memory (see the note in
        client/include/SdmaQueue.hpp).

    Field pointers (queueBuf/rptr/wptr/doorbell/cachedWptr/committedWptr) are
    loaded on demand with s_load_dwordx2 from handleBase+offset, matching the
    on-demand kernarg loads in _fusedA2ALoadRecvBase.

    NO EXEC OR VCC CONTRACT. Two debts used to live here and both are now gone
    with their causes rather than merely documented:

      * EXEC != 0 was required because every memory-to-SGPR step went through
        v_readfirstlane_b32, which on CDNA4 "overrides the EXEC mask for the
        VGPR read" -- at EXEC == 0 it is not skipped but forced to lane 0,
        so an all-inactive entry silently compared against a never-loaded
        register instead of faulting. There is no readfirstlane left.
      * VCC was clobbered by the v_add_co_u32 pair that formed the ring
        address. The scalar address math (s_add_u32 / s_addc_u32) has no carry
        register.

    Both were "ACCEPTED DEBT, fails silently when violated" for as long as any
    part of this path was vector. Do not reintroduce a readfirstlane or a VOP2
    carry form to save an instruction; the value here is that a caller cannot
    get it wrong, not the instruction count.

    The emitter still takes `w` for `.sgprPool` and `.labels`. It no longer
    touches `.vgprPool`.
    """

    def __init__(self, queueSize: int = SDMA_QUEUE_SIZE):
        assert (queueSize & (queueSize - 1)) == 0, "ring size must be a power of two"
        self.queueSize = queueSize
        self.ringMask  = queueSize - 1

    # ---- small helpers -----------------------------------------------------

    def _loadFieldPtr(self, module, w, dstPairS, handleBaseS, byteOff):
        """Load an 8-byte handle field (pointer) at handleBase+byteOff into an
        aligned SGPR pair. Used for queueBuf/rptr/wptr/doorbell/cachedWptr/
        committedWptr. handleBase is a raw SGPR pointer pair and byteOff is a
        compile-time immediate, so this needs no argLoader and the emitter stays
        usable outside a full KernelWriter. Caller waits (kmcnt) before use."""
        module.add(SLoadB64(dst=sgpr(dstPairS, 2), base=sgpr(handleBaseS, 2),
                            soffset=hex(byteOff),
                            comment="load handle field at +0x%x" % byteOff))
        return module

    def _wrapIntoRing(self, module, dstS, srcS, comment=""):
        """dst = src & (queueSize-1). WrapIntoRing without a divide (ring size is
        a power of two). src/dst are the low dword of a byte index; the ring is
        <4 GiB so the high dword of a wrapped index is always 0."""
        module.add(SAndB32(dst=sgpr(dstS), src0=sgpr(srcS), src1=self.ringMask,
                           comment=comment or "WrapIntoRing: idx & (SDMA_QUEUE_SIZE-1)"))
        return module

    # ---- CanWriteUpto ------------------------------------------------------

    def emitCanWriteUpto(self, module, w, handleBaseS, cachedHwReadIdxS,
                         uptoIdxS, resultS, tmpPairS):
        """Two-level full check.

        Fast path uses the private cache only (no memory traffic):
            if (upto - cachedHwReadIndex) < queueSize: return true
        Slow path (cache says full) reads the hardware rptr with s_load...glc,
        refreshes the cache, and re-tests. `resultS` is set to 1 (can write) or
        0 (full); the caller branches on it. All index math is 64-bit
        (idx pair = S:S+1).

        CALLER CONTRACT: `resultS` MUST be disjoint from `uptoIdxS`,
        `cachedHwReadIdxS` and `tmpPairS`. It is defaulted to 0 as the very first
        emitted instruction -- before the first READ of those inputs -- so an
        aliasing caller would have its input clobbered, not merely its output
        overwritten late.
        """
        canLabel  = Label(w.labels.getNameInc("sdma_canwrite_ok"),  "CanWriteUpto: room in ring")
        fullLabel = Label(w.labels.getNameInc("sdma_canwrite_full"), "CanWriteUpto: cache says full -> read rptr")
        doneLabel = Label(w.labels.getNameInc("sdma_canwrite_done"), "CanWriteUpto: done")

        # resultS defaults to 0 (full): the refresh-retest tail branch below
        # ("hi != 0 -> full") jumps straight to doneLabel without writing it,
        # and the caller's register is live across CAS retries -- so a stale 1
        # from a previous iteration would claim space on a full ring.
        module.add(SMovB32(dst=sgpr(resultS), src=0, comment="CanWriteUpto = false (default)"))

        # tmp = upto - cachedHwReadIndex (64-bit), then compare tmp < queueSize.
        # queueSize < 2^32 so if the high dword of the difference is nonzero the
        # gap is huge (>= 2^32) => definitely not < queueSize => full.
        self._emitU64Sub(module, tmpPairS, uptoIdxS, cachedHwReadIdxS,
                         "CanWriteUpto: upto - cachedHwReadIndex")
        module.add(SCmpEQU32(src0=sgpr(tmpPairS + 1), src1=0, comment="diff hi == 0? (gap < 2^32)"))
        module.add(SCBranchSCC0(labelName=fullLabel.getLabelName(), comment="hi != 0 -> gap huge -> full path"))
        module.add(SCmpLtU32(src0=sgpr(tmpPairS + 0), src1=self.queueSize,
                             comment="diff < queueSize? (fast-path room check)"))
        module.add(SCBranchSCC1(labelName=canLabel.getLabelName(), comment="room via cached index"))

        # Slow path: read the hardware rptr, refresh the cache, retest. The read
        # lands STRAIGHT IN the caller's private pair -- it is the refresh, so
        # there is nothing to relay. cachedHwReadIdxS must be 2-ALIGNED, which
        # the caller already guarantees for its own 64-bit arithmetic.
        #
        # ⚠ THIS PATH IS ALMOST NEVER EXERCISED end to end: it runs only when
        # the room check fails, i.e. when the SDMA engine has fallen a whole
        # ring behind. A passing validation run is therefore NOT evidence about
        # these instructions. What they rest on is the direct measurement of
        # s_load...glc under a spin (see the file header).
        module.add(fullLabel)
        rptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_cw_rptrPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, rptrPtrS, handleBaseS, OFF_rptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait rptr pointer load"))
        module.add(SLoadB64(dst=sgpr(cachedHwReadIdxS, 2), base=sgpr(rptrPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="cachedHwReadIndex = hardware rptr"))
        module.add(SWaitCnt(kmcnt=0, comment="wait rptr load"))
        w.sgprPool.checkIn(rptrPtrS)
        # retest with refreshed cache.
        self._emitU64Sub(module, tmpPairS, uptoIdxS, cachedHwReadIdxS,
                         "CanWriteUpto: upto - refreshed rptr")
        module.add(SCmpEQU32(src0=sgpr(tmpPairS + 1), src1=0, comment="diff hi == 0?"))
        module.add(SCBranchSCC0(labelName=doneLabel.getLabelName(),
                                comment="hi != 0 -> full (result already defaulted to 0)"))
        module.add(SCmpLtU32(src0=sgpr(tmpPairS + 0), src1=self.queueSize, comment="diff < queueSize?"))
        module.add(SCBranchSCC1(labelName=canLabel.getLabelName(), comment="room after refresh"))
        # Still full: resultS is already 0 from the default above and nothing on
        # the way here writes it, so this path only has to skip canLabel's store.
        module.add(SBranch(labelName=doneLabel.getLabelName(), comment="full -> done (result still 0)"))
        module.add(canLabel)
        module.add(SMovB32(dst=sgpr(resultS), src=1, comment="CanWriteUpto = true (room)"))
        module.add(doneLabel)
        return module

    # ---- ReserveQueueSpace (CAS, NOT fetch_add) ----------------------------

    def emitReserveQueueSpace(self, module, w, handleBaseS, cachedHwReadIdxS,
                              sizeInBytes, outCurS, outOffsetS):
        """Reserve `sizeInBytes` in the ring via a compare-exchange loop and
        compute the wrap-padding.

        MUST be CAS, not fetch_add: on wrap the reservation also pads the ring
        tail (offset = queueSize - WrapIntoRing(cur)), and that padding depends
        on the CURRENT cur_index -- so "compute new index" and "claim the slot"
        must be one atomic step. A fetch_add would let two producers compute
        different padding yet both believe they claimed the slot.

        Seeded once, then:
          off   = (WrapIntoRing(cur) + size > queueSize) ? queueSize-WrapIntoRing(cur) : 0
          new   = cur + size + off
          if CanWriteUpto(new) and CAS(cachedWptr, cur -> new) succeeds: break
          else cur = the CAS's pre-op return, and retry
        Outputs: outCurS (2 SGPRs) = reserved base index; outOffsetS (1 SGPR) =
        pad bytes. sizeInBytes is a compile-time packet size (immediate).

        cur IS NOT RE-READ PER ITERATION. A failing CAS already returns the
        current memory value, so re-loading would be asking for something the
        previous instruction just handed over. Only the seed is a load.

        THE SEED IS A HINT, not a correctness input, which is worth knowing
        because it is the one read here whose freshness nothing depends on.
        The CAS self-corrects -- a wrong seed just loses the first race
        and comes back with the truth. The one failure it could cause is a
        livelock, if a bogus seed made CanWriteUpto say "full" forever and the
        CAS that would fix it were never reached. That cannot happen here:
        cachedWptr only ever increases, so a stale seed is SMALLER than the
        truth, which makes `new` smaller and the gap to rptr smaller, i.e.
        strictly more likely to pass the room check. Staleness can cost an
        extra CAS round; it cannot wedge the loop.
        """
        loopLabel  = Label(w.labels.getNameInc("sdma_reserve_loop"), "ReserveQueueSpace: CAS retry loop")
        noPadLabel = Label(w.labels.getNameInc("sdma_reserve_nopad"), "ReserveQueueSpace: no wrap padding")
        retryLabel = Label(w.labels.getNameInc("sdma_reserve_retry"), "ReserveQueueSpace: lost the race")
        doneLabel  = Label(w.labels.getNameInc("sdma_reserve_done"),  "ReserveQueueSpace: reserved")

        cachedWptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_cwPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, cachedWptrPtrS, handleBaseS, OFF_cachedWptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait cachedWptr pointer load"))

        newIdxS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_new", preventOverflow=False)
        wrapS   = w.sgprPool.checkOut(1, tag="sdma_rsv_wrap", preventOverflow=False)
        canS    = w.sgprPool.checkOut(1, tag="sdma_rsv_can", preventOverflow=False)
        tmpPair = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_tmp", preventOverflow=False)

        # CAS data block: [0:1]=swap(new) and the pre-op return, [2:3]=compare(cur).
        # 4-ALIGNED -- SMEM requires a multiple of four for a 4-dword SDATA and
        # the assembler rejects anything else outright.
        casDataS = w.sgprPool.checkOutAligned(4, 4, tag="sdma_rsv_casData", preventOverflow=False)

        # Seed the compare slot with the current value (see the docstring on why
        # a possibly-stale seed is safe here).
        module.add(SLoadB64(dst=sgpr(casDataS + 2, 2), base=sgpr(cachedWptrPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="seed cur = cachedWptr (hint; the CAS self-corrects)"))
        module.add(SWaitCnt(kmcnt=0, comment="wait cachedWptr seed"))

        module.add(loopLabel)
        module.add(SMovB32(dst=sgpr(outCurS + 0), src=sgpr(casDataS + 2), comment="cur lo = compare slot"))
        module.add(SMovB32(dst=sgpr(outCurS + 1), src=sgpr(casDataS + 3), comment="cur hi = compare slot"))

        # off = 0 by default; if WrapIntoRing(cur)+size > queueSize -> pad tail.
        module.add(SMovB32(dst=sgpr(outOffsetS), src=0, comment="offset = 0 (no pad)"))
        self._wrapIntoRing(module, wrapS, outCurS + 0, "WrapIntoRing(cur)")
        module.add(SAddU32(dst=sgpr(tmpPair), src0=sgpr(wrapS), src1=sizeInBytes,
                           comment="WrapIntoRing(cur) + size"))
        module.add(SCmpLtU32(src0=sgpr(tmpPair), src1=self.queueSize + 1,
                             comment="wrap+size <= queueSize? (fits without wrap)"))
        module.add(SCBranchSCC1(labelName=noPadLabel.getLabelName(), comment="fits -> no padding"))
        module.add(SSubU32(dst=sgpr(outOffsetS), src0=self.queueSize, src1=sgpr(wrapS),
                           comment="offset = queueSize - WrapIntoRing(cur) (pad ring tail)"))
        module.add(noPadLabel)

        # new = cur + size + offset (64-bit).
        module.add(SAddU32(dst=sgpr(newIdxS + 0), src0=sgpr(outCurS + 0), src1=sizeInBytes,
                           comment="new lo = cur + size"))
        module.add(SAddCU32(dst=sgpr(newIdxS + 1), src0=sgpr(outCurS + 1), src1=0, comment="new hi (carry)"))
        module.add(SAddU32(dst=sgpr(newIdxS + 0), src0=sgpr(newIdxS + 0), src1=sgpr(outOffsetS),
                           comment="new lo += offset"))
        module.add(SAddCU32(dst=sgpr(newIdxS + 1), src0=sgpr(newIdxS + 1), src1=0, comment="new hi (carry)"))

        # CanWriteUpto(new)? if not, retry (a concurrent consumer may free space).
        self.emitCanWriteUpto(module, w, handleBaseS, cachedHwReadIdxS, newIdxS, canS, tmpPair)
        module.add(SCmpEQU32(src0=sgpr(canS), src1=0, comment="CanWriteUpto == false?"))
        module.add(SCBranchSCC1(labelName=loopLabel.getLabelName(), comment="full -> retry"))

        # CAS(cachedWptr, cur -> new): data[0:1]=new(swap), data[2:3]=cur(compare).
        module.add(SMovB32(dst=sgpr(casDataS + 0), src=sgpr(newIdxS + 0), comment="swap lo = new"))
        module.add(SMovB32(dst=sgpr(casDataS + 1), src=sgpr(newIdxS + 1), comment="swap hi = new"))
        # (casDataS+2/3 already hold cur = the compare value.)
        self._emitReserveCas(module, w, cachedWptrPtrS, casDataS)
        module.add(SWaitCnt(kmcnt=0, comment="wait CAS return"))
        # The pre-op memory value comes back in casDataS[0:1]; we won iff it == cur.
        module.add(SCmpEQU32(src0=sgpr(casDataS + 0), src1=sgpr(outCurS + 0),
                             comment="CAS pre-op lo == cur lo? (won the slot)"))
        module.add(SCBranchSCC0(labelName=retryLabel.getLabelName(), comment="lost race -> retry"))
        module.add(SCmpEQU32(src0=sgpr(casDataS + 1), src1=sgpr(outCurS + 1), comment="pre-op hi == cur hi?"))
        module.add(SCBranchSCC0(labelName=retryLabel.getLabelName(), comment="lost race -> retry"))
        module.add(SBranch(labelName=doneLabel.getLabelName(), comment="won -> reserved"))

        # Lost the race: the pre-op value IS the current cur, so move it into the
        # compare slot and go round again. This is the only path that updates
        # cur, which is why the loop body itself never reloads it.
        module.add(retryLabel)
        module.add(SMovB32(dst=sgpr(casDataS + 2), src=sgpr(casDataS + 0),
                           comment="cur lo = CAS pre-op (refresh from the failed swap)"))
        module.add(SMovB32(dst=sgpr(casDataS + 3), src=sgpr(casDataS + 1), comment="cur hi = CAS pre-op"))
        module.add(SBranch(labelName=loopLabel.getLabelName(), comment="retry with the refreshed cur"))
        module.add(doneLabel)

        w.sgprPool.checkIn(casDataS)
        w.sgprPool.checkIn(cachedWptrPtrS)
        w.sgprPool.checkIn(newIdxS)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(canS)
        w.sgprPool.checkIn(tmpPair)
        return module

    def _emitReserveCas(self, module, w, cachedWptrPtrS, casDataS):
        """64-bit compare-exchange of cachedWptr, returning the pre-op value.

        Isolated so the CAS primitive can be swapped without touching the
        reserve logic. casDataS is a 4-ALIGNED run of 4 SGPRs:
        [0:1]=swap(new), [2:3]=compare(cur), and the pre-op value comes back
        over [0:1] -- the operand layout is the same one the vector form used
        (CDNA4 ISA, S_ATOMIC_CMPSWAP_X2).

        glc HAS A DIFFERENT JOB HERE than on the stores in this file. On a
        scalar store it forces the write past the K$ and L2; on a scalar atomic
        it selects return-of-pre-op (ISA 8.2.2). That leaves SMEM with no bit at
        all to request a coherence scope, and the ISA never says what scope you
        get -- which matters because this CAS is the multi-producer mutual
        exclusion for the whole ring. Measured on gfx950 before adopting it: 512
        workgroups across all 8 XCDs incrementing one 64-bit counter through
        this instruction lost zero updates, against a non-atomic control on the
        same shape that lost 99.5% of them.

        TWO CLAUSE RULES APPLY, both currently satisfied by the surrounding code
        rather than by anything that would complain if they stopped being:
          * "Atomics ... must be in a single-instruction clause" (ISA 8.2). The
            s_mov before and the s_waitcnt after keep this one alone; do not let
            another SMEM op become adjacent to it.
          * The retry path overwrites the compare half from the pre-op half,
            i.e. it rewrites this instruction's own source registers. That is
            explicitly blessed: "an atomic that returns the pre-op value
            overwrites its data source, which is acceptable" (same section).
        """
        module.add(SAtomicCmpswapX2(
            dst=sgpr(casDataS, 4), base=sgpr(cachedWptrPtrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True),
            comment="CAS cachedWptr cur->new (glc = return pre-op)"))
        return module

    # ---- placePacket -------------------------------------------------------

    def emitPlacePacket(self, module, w, handleBaseS, packetDwordsS, numDwords,
                        pendingWptrS, offsetS):
        """Write `offsetS` bytes of zero-padding (NOPs) then `numDwords` packet
        dwords into the ring, all with s_store...glc. Advances pendingWptrS
        (2 SGPRs) by offset then by the packet size. `packetDwordsS` is the base
        SGPR of the already-built packet (SdmaPacketEmitter fills it);
        `numDwords` is compile-time.

        Ring addressing is per-dword: base_dword = WrapIntoRing(pending)/4, and
        each store targets queueBuf[base_dword + i]. queueBuf is a uint32_t*, so
        the byte address is queueBuf + WrapIntoRing(pending) (already a dword-
        aligned byte offset). That fold happens once per placement, in
        _emitRingByteAddr; the individual dwords then ride immediate offsets.
        Wrap padding is a small runtime loop, since the offset is the general
        reserve result rather than a compile-time constant.

        Pure scalar: no EXEC requirement, no VCC (see the class docstring).

        BLOCK REUSE. The caller may pass the SAME packetDwordsS block for a
        later packet -- the ATOMIC is 8 dwords where the COPY is 13 -- but only
        AFTER this call has emitted the COPY's stores. Rebuilding the block
        first would overwrite dwords this placement has not stored yet, and
        nothing here would notice.

        What makes the reuse safe once the stores ARE emitted is that a scalar
        store's source registers only have to survive its CLAUSE, not its
        completion. The ISA ties the restriction to ATC XNACK replay and scopes
        it accordingly: "instructions in scalar memory clauses must not
        overwrite the sources of any of the instructions in the clause... A
        clause is broken by any non-memory instruction" (CDNA4 ISA 8.2). The
        rebuild starts with s_mov, which breaks the clause, so the replay window
        of these stores is closed before their registers change. NO s_waitcnt is
        needed for this and adding one would only serialize the submit -- the
        ISA never asks for completion here, only for the clause boundary.
        """
        queueBufPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_pp_qbuf", preventOverflow=False)
        self._loadFieldPtr(module, w, queueBufPtrS, handleBaseS, OFF_queueBuf)
        module.add(SWaitCnt(kmcnt=0, comment="wait queueBuf pointer load"))

        wrapS   = w.sgprPool.checkOut(1, tag="sdma_pp_wrap", preventOverflow=False)
        cntS    = w.sgprPool.checkOut(1, tag="sdma_pp_cnt", preventOverflow=False)
        # 2-ALIGNED: this pair is the SBASE of every store below, and SMEM
        # rejects an odd SBASE.
        addrS   = w.sgprPool.checkOutAligned(2, 2, tag="sdma_pp_addr", preventOverflow=False)
        zeroS   = w.sgprPool.checkOut(1, tag="sdma_pp_zero", preventOverflow=False)
        module.add(SMovB32(dst=sgpr(zeroS), src=0, comment="padding NOP value = 0"))

        # ---- padding: store `offset` bytes of zero at WrapIntoRing(pending). ----
        padLoop = Label(w.labels.getNameInc("sdma_pp_padloop"), "placePacket: zero-pad ring tail")
        padDone = Label(w.labels.getNameInc("sdma_pp_paddone"), "placePacket: padding done")
        # numOffsetDwords = offset / 4; if 0, skip the pad loop entirely.
        module.add(SLShiftRightB32(dst=sgpr(cntS), src=sgpr(offsetS), shiftHex=2,
                                   comment="numOffsetDwords = offset / 4"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="no padding?"))
        module.add(SCBranchSCC1(labelName=padDone.getLabelName(), comment="offset==0 -> skip pad"))
        module.add(padLoop)
        self._wrapIntoRing(module, wrapS, pendingWptrS + 0, "WrapIntoRing(pending) (pad)")
        self._emitRingByteAddr(module, addrS, queueBufPtrS, wrapS)
        module.add(SStoreB32(
            src=sgpr(zeroS), base=sgpr(addrS, 2), soffset=hex(0), smem=_smemStore(),
            comment="ring[wrap] = 0 padding NOP"))
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=4,
                           comment="pending += 4 (one padded dword)"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))
        module.add(SSubU32(dst=sgpr(cntS), src0=sgpr(cntS), src1=1, comment="numOffsetDwords -= 1"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="pad done?"))
        module.add(SCBranchSCC0(labelName=padLoop.getLabelName(), comment="more padding"))
        module.add(padDone)

        # ---- packet: store numDwords packet dwords at WrapIntoRing(pending). ----
        # Recompute base after padding advanced pending. numDwords is compile-time,
        # so unroll (one warp writes <=64).
        self._wrapIntoRing(module, wrapS, pendingWptrS + 0, "WrapIntoRing(pending) (packet base)")
        self._emitRingByteAddr(module, addrS, queueBufPtrS, wrapS)
        for i, width in self._packetStoreWidths(packetDwordsS, numDwords):
            op = {1: SStoreB32, 2: SStoreB64, 4: SStoreB128}[width]
            src = sgpr(packetDwordsS + i) if width == 1 else sgpr(packetDwordsS + i, width)
            module.add(op(
                src=src, base=sgpr(addrS, 2), soffset=hex(i * 4), smem=_smemStore(),
                comment="ring[base + %d] = packet dword%s"
                        % (i, "" if width == 1 else "s %d..%d" % (i, i + width - 1))))
        # pending += numDwords*4 (packet size).
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=numDwords * 4,
                           comment="pending += packet size"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))

        w.sgprPool.checkIn(addrS)
        w.sgprPool.checkIn(zeroS)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(cntS)
        w.sgprPool.checkIn(queueBufPtrS)
        return module

    @staticmethod
    def _packetStoreWidths(baseS, numDwords):
        """Split a run of `numDwords` consecutive SGPRs starting at `baseS` into
        the widest scalar stores gfx950 will take. Returns [(dwordIndex, width)].

        SDATA ALIGNMENT IS STRICTER THAN THE VECTOR RULE THIS REPLACES. A global
        store only needed an EVEN first register, so any odd base cost one
        leading b32 and everything after it widened. SMEM wants the real thing:
        "SDST must be even for two Dwords, or a multiple of four for larger"
        (CDNA4 ISA 8.4), confirmed at the assembler -- s_store_dwordx4 s[9:12]
        and s_store_dwordx2 s[9:10] are both rejected for register alignment.
        So the width is re-decided at every step from the CURRENT register's
        alignment; a run cannot simply widen once and stay wide.

        x4 IS THE CEILING: SMEM stores write 1-4 Dwords (loads read up to 16),
        and gfx950 has no s_store_dwordx8 -- the assembler answers "did you mean
        s_store_dword, s_store_dwordx2, s_store_dwordx4?". rocisa DOES expose
        SStoreB256 / SStoreB512 bindings; they are load-shaped leftovers and
        must not be reached for here.

        Ring ADDRESS alignment is not a constraint: SMEM ignores the two LSBs of
        the byte address (CDNA4 ISA 8.2, "the two LSBs are ignored and treated as
        if they were zero") and 8.4 states OFFSET has no alignment restriction,
        so a Dword-aligned packet start is enough. Note this is the ISA answering
        by omission -- there is no positive statement that an x4 store may cross
        a 16-byte boundary -- and the 84-byte reservation stride guarantees
        packets do land at every 4-byte phase, so the end-to-end run is what
        actually confirms it."""
        out, i = [], 0
        while i < numDwords:
            reg, left = baseS + i, numDwords - i
            if reg % 4 == 0 and left >= 4:
                width = 4
            elif reg % 2 == 0 and left >= 2:
                width = 2
            else:
                width = 1
            out.append((i, width)); i += width
        return out

    def _emitRingByteAddr(self, module, dstPairS, queueBufPtrS, wrapS):
        """Compute the 64-bit byte address queueBuf + WrapIntoRing(pending) into
        the 2-ALIGNED SGPR pair dstPairS. queueBuf is a byte-addressable base;
        the wrapped index is already a byte offset (<4 GiB, so it adds only into
        the low dword with carry).

        dstPairS becomes the SBASE of every store in the placement, and SMEM
        requires an even SBASE (measured at the assembler: s_store with s[3:4]
        is rejected for register alignment), hence 2-aligned rather than merely
        consecutive.

        Folding the wrap into the base here, once per placement, is what lets
        the stores address their dwords with plain IMMEDIATE offsets. The
        alternative -- leaving the base at queueBuf and passing the wrap as
        SOFFSET -- reads better but is a trap: rocisa's SMEMModifiers omits the
        `offset:` text entirely when the offset is 0, so the FIRST store of each
        packet would silently drop from the IMM=1/SOE=1 encoding to IMM=0/SOE=0
        and put the runtime SGPR in the OFFSET field, which Table 39 documents
        as immediate-or-M0 only for stores. One store per packet encoded
        differently from its neighbours is not a bug anyone finds by reading."""
        module.add(SAddU32(dst=sgpr(dstPairS + 0), src0=sgpr(queueBufPtrS + 0), src1=sgpr(wrapS),
                           comment="addr lo = queueBuf + WrapIntoRing(pending)"))
        module.add(SAddCU32(dst=sgpr(dstPairS + 1), src0=sgpr(queueBufPtrS + 1), src1=0,
                            comment="addr hi (carry)"))
        return module

    # ---- submitPacket ------------------------------------------------------

    def emitSubmitPacket(self, module, w, handleBaseS, baseS, pendingWptrS):
        """Serialize this producer's commit behind earlier reservations, then
        publish the packet.

        (1) spin until committedWptr == base (this producer's turn; earlier
            reservations commit in order), polling with s_load...glc.
        (2) Publish sequence (any bit wrong => timing hang):
              store wptr = pending        s_store_dwordx2 glc
              s_waitcnt lgkmcnt(0)
              store doorbell = pending    s_store_dwordx2 glc   <-- rings the engine
              store committedWptr = pend  s_store_dwordx2 glc   <-- unblocks next producer
            The value written to wptr/doorbell/committedWptr is the new absolute
            byte wptr (pending), NOT an increment. A wait precedes the doorbell
            so the wptr store is globally ordered before the engine is told to
            read up to it.

            All three are scalar, so the ordering waits are lgkmcnt rather than
            vmcnt -- and pending is written straight from its SGPR pair, which
            is why the two v_mov relays that used to stage it are gone. The
            doorbell is an MMIO/BAR write and needs no wider modifier than the
            others: SMEM has one cache bit, and glc already carries it past the
            K$ and L2. That a scalar store can reach the doorbell aperture at
            all is measured, not assumed.

        baseS / pendingWptrS are 2-SGPR byte indices from the reserve+place pair.
        Emitted by a single elected lane, so NO s_barrier here: in single-lane
        assembly the s_waitcnt already orders memory and an s_barrier would
        deadlock.

        "A single elected lane" means exactly one, never zero -- an election
        leaving EXEC == 0 spins on lane 0's never-loaded register rather than
        skipping the loop (see the class docstring).
        """
        # --- (1) spin: committedWptr == base ---
        commPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_commPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, commPtrS, handleBaseS, OFF_committedWptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr pointer load"))

        # 2-ALIGNED: this pair is the x2 destination of the poll below.
        pollS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_poll", preventOverflow=False)

        spinLabel = Label(w.labels.getNameInc("sdma_submit_spin"), "submitPacket: wait committedWptr == base")
        spinDone  = Label(w.labels.getNameInc("sdma_submit_ready"), "submitPacket: our turn")
        module.add(spinLabel)
        module.add(SSleep(simm16=1, comment="submitPacket: backoff between polls (must stay INSIDE the spin body)"))
        # glc is what makes this a POLL rather than one read repeated: it forces
        # the load past the K$ and L2 every pass. Measured, because a one-shot
        # load and a load in a spin are different questions -- a scalar load
        # without glc was still returning its first value after 20 million
        # iterations, which here would be a hang rather than a wrong answer.
        module.add(SLoadB64(dst=sgpr(pollS, 2), base=sgpr(commPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="poll committedWptr"))
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr load"))
        module.add(SCmpEQU32(src0=sgpr(pollS + 0), src1=sgpr(baseS + 0), comment="committedWptr lo == base lo?"))
        module.add(SCBranchSCC0(labelName=spinLabel.getLabelName(), comment="not our turn -> spin"))
        module.add(SCmpEQU32(src0=sgpr(pollS + 1), src1=sgpr(baseS + 1), comment="committedWptr hi == base hi?"))
        module.add(SCBranchSCC0(labelName=spinLabel.getLabelName(), comment="not our turn -> spin"))
        module.add(spinDone)
        # The packet stores are scalar, so what has to drain here is lgkmcnt,
        # not vmcnt. Getting this counter wrong is a timing-only failure: the
        # doorbell would reach the engine ahead of the packet it announces.
        module.add(SWaitCnt(kmcnt=0, comment="ensure our packet stores are globally visible before wptr"))

        # The published value is pending (absolute byte wptr, NOT an increment).
        # A scalar store takes it straight from the SGPR pair -- no staging.
        # pendingWptrS must be 2-ALIGNED for the x2 SDATA; the caller allocates
        # it that way.

        # --- (2a) store wptr = pending ---
        wptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_wptrPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, wptrPtrS, handleBaseS, OFF_wptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait wptr pointer load"))
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(wptrPtrS, 2), soffset=hex(0), smem=_smemStore(),
            comment="store wptr = pending"))
        w.sgprPool.checkIn(wptrPtrS)

        # --- order the wptr store before the doorbell ---
        module.add(SWaitCnt(kmcnt=0, comment="s_waitcnt lgkmcnt(0): wptr store visible before doorbell"))

        # --- (2b) store doorbell = pending -> rings the engine ---
        dbPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_dbPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, dbPtrS, handleBaseS, OFF_doorbell)
        module.add(SWaitCnt(kmcnt=0, comment="wait doorbell pointer load"))
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(dbPtrS, 2), soffset=hex(0), smem=_smemStore(),
            comment="ring doorbell = pending"))
        w.sgprPool.checkIn(dbPtrS)
        module.add(SWaitCnt(kmcnt=0, comment="wait doorbell store issued"))

        # --- (2c) store committedWptr = pending -> unblocks next producer ---
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(commPtrS, 2), soffset=hex(0), smem=_smemStore(),
            comment="store committedWptr = pending"))
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr store issued"))

        w.sgprPool.checkIn(pollS)
        w.sgprPool.checkIn(commPtrS)
        return module

    # ---- utility: 64-bit sub -------------------------------------------------

    def _emitU64Sub(self, module, dstPairS, aPairS, bPairS, comment):
        """dst = a - b (64-bit) via s_sub_u32 / s_subb_u32.

        Not rocisa's SSubU64: that is a plain CommonInstruction with no
        capability fallback, so it emits s_sub_u64 unconditionally and gfx950
        rejects the mnemonic. SAddU64 is NOT the same shape -- it is a
        CompositeInstruction and does lower to s_add_u32/s_addc_u32 here, which
        is why the sibling addition elsewhere can use it and this cannot."""
        module.add(SSubU32(dst=sgpr(dstPairS + 0), src0=sgpr(aPairS + 0), src1=sgpr(bPairS + 0),
                           comment=comment + " (lo)"))
        module.add(SSubBU32(dst=sgpr(dstPairS + 1), src0=sgpr(aPairS + 1), src1=sgpr(bPairS + 1),
                            comment=comment + " (hi, borrow)"))
        return module
