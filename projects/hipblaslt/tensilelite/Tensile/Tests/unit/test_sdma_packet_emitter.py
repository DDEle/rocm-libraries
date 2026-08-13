#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter tests (NOGPU).
#
# The emitter (Tensile/Components/SdmaPacketEmitter.py) turns the all-to-all
# geometry into COPY_SUBWIN + ATOMIC ADD_RTN_32 packet dword arrays. The
# pure-Python encoders are pinned against the golden dword vectors below (see
# the provenance note above them, and test_sdma_header_mirror.py for the C++
# cross-check); the rocisa emitters are asserted on semantic field-packing
# features -- not a whole-text snapshot -- and assembled on gfx950. Part 2b
# covers the coordinate fold: src_x/src_y/dst_x/dst_y move out of the packet
# fields and into the 64-bit base addresses.
################################################################################

import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

import rocisa                                                     # noqa: E402

from rocisa import rocIsa                                          # noqa: E402
from rocisa.register import RegisterPool                           # noqa: E402
from rocisa.enum import RegisterType                               # noqa: E402
from rocisa.code import Module                                     # noqa: E402
from Tensile.Common.Architectures import gfxToIsa                  # noqa: E402
from Tensile.Components.SdmaPacketEmitter import (                 # noqa: E402
    SdmaPacketEmitter,
    encodeCopyDwords, encodeAtomicDwords,
    COPY_HEADER_DW0, ATOMIC_HEADER_DW0,
    COPY_PACKET_DWORDS, ATOMIC_PACKET_DWORDS,
    BF16_ELEMENT_SIZE_LOG2, D_DATA_ELEMENT_LOG2,
    PACKET_ELEMENT_SIZE_LOG2, ELEMENT_SHIFT, ELEMENT_MULTIPLE,
    checkA2AFieldsFit, XY_FIELD_LIMIT, PITCH_FIELD_LIMIT,
)

# ---- shared full-shape constants, element units -------------------------------
M       = 18432   # feature (src row pitch)
N       = 2048    # token
NSHARD  = 2560    # feature shard per rank (rect X extent, prod dst pitch)
MT1     = 256     # token tile (rect Y extent)
_GFX    = "gfx950"


# ---------------------------------------------------------------------------
# Part 1: pure-Python encoders vs the golden dwords
# ---------------------------------------------------------------------------
# PROVENANCE OF THESE VECTORS -- hand-written constants, independent of the code
# under test, so the check is not circular. Authority differs per vector:
#   _HARNESS_COPY_GOLDEN -- ran on MI355X (3 peers x 8 bands = 24 packets) and
#     independently hand-verified against the field spec.
#   _PROD_COPY_GOLDEN -- no hardware backing; only DW9/DW10 differ from the
#     harness vector (dst pitch 2560 vs 2624), checkable by inspection:
#     (2560-1)<<13 == 0x013FE000, 256*2560-1 == 0x0009FFFF.
#   _ATOMIC_GOLDEN -- derived from the SDMA_PKT_ATOMIC layout with the TC
#     atomic op table's ADD_RTN_32 selector: DW0 == 10 | (15<<25) == 0x1E00000A.
#
# The bit positions (minus-one extents/pitches, ELEMENTSIZE scaling, the <<13
# pitch placement) come from AMD OSS 4.4 sdma.pkt, cross-checked against ROCR
# sdma_registers.h and vega10_sdma_pkt_open.h; GFX12+ uses a different layout,
# so these vectors are gfx9xx / gfx95x only.
#
# Do not "update the golden" to make a red test pass -- a mismatch means either
# the packing code drifted or the rule itself changed (then cite the new source).

# HARNESS: dst pitch padded to 2624 (kShard+kDstPad), the only byte sequence
# with real MI355X backing.
_HARNESS_COPY_GOLDEN = [
    0x20000401, 0x00000000, 0x00000000, 0x00000A00, 0x08FFE000,
    0x023FFFFF, 0x00000000, 0x00000000, 0x00000000, 0x0147E000,
    0x005C3FFF, 0x00FF09FF, 0x00000000]

# PRODUCTION: unpadded recv buffer (dst pitch == nShard == 2560); the shape we
# actually emit. Only DW9/DW10 differ from harness.
_PROD_COPY_GOLDEN = [
    0x20000401, 0x00000000, 0x00000000, 0x00000A00, 0x08FFE000,
    0x023FFFFF, 0x00000000, 0x00000000, 0x00000000, 0x013FE000,
    0x0009FFFF, 0x00FF09FF, 0x00000000]

# EDGE token-tile: N is NOT a multiple of MT1, so the last tile (j = 7) covers
# only N_EDGE - 7*MT1 = 2000 - 1792 = 208 tokens and rect_y must be CLAMPED to
# 208. NO hardware backing; hand-derived from the same rules, and every dword
# that differs from _PROD_COPY_GOLDEN is checkable by inspection:
#   DW3  src_x|src_y   = 2560 | 1792<<16          == 0x07000A00
#   DW5  src_slice-1   = 18432*2000 - 1           == 0x02327FFF
#   DW8  dst_x|dst_y   = 0 | (0*2000 + 1792)<<16  == 0x07000000
#   DW11 rect_x-1|rect_y-1 = 2559 | (208-1)<<16   == 0x00CF09FF
# An UNCLAMPED rect_y would put 0x00FF in the high half and make the engine read
# 48 token rows past the end of D.
_EDGE_N       = 2000
_EDGE_J       = 7
_EDGE_RECT_Y  = _EDGE_N - _EDGE_J * MT1   # 208
_EDGE_COPY_GOLDEN = [
    0x20000401, 0x00000000, 0x00000000, 0x07000A00, 0x08FFE000,
    0x02327FFF, 0x00000000, 0x00000000, 0x07000000, 0x013FE000,
    0x0009FFFF, 0x00CF09FF, 0x00000000]

# ATOMIC ADD_RTN_32 to a flag slot with distinct lo/hi bytes (matches the C++ test).
_ATOMIC_ADDR = 0x0000ABCD12345678
_ATOMIC_GOLDEN = [
    0x1E00000A, 0x12345678, 0x0000ABCD, 0x00000001,
    0x00000000, 0x00000000, 0x00000000, 0x00000000]


def test_encoder_matches_harness_golden():
    # Harness: srcX=nShard (p==1 shard offset folded into src_x=nShard via the
    # golden's p=1 origin), srcY=0, dst all-zero coords, padded dst pitch.
    got = encodeCopyDwords(
        srcBase=0, srcX=NSHARD, srcY=0, srcPitch=M, srcSlicePitch=N * M,
        dstBase=0, dstX=0, dstY=0, dstPitch=2624, dstSlicePitch=2304 * 2624,
        rectX=NSHARD, rectY=MT1, elementSizeLog2=1)
    assert got == _HARNESS_COPY_GOLDEN, \
        "\n".join("DW%d got 0x%08X exp 0x%08X" % (i, g, e)
                  for i, (g, e) in enumerate(zip(got, _HARNESS_COPY_GOLDEN)) if g != e)


def test_encoder_matches_production_golden():
    got = encodeCopyDwords(
        srcBase=0, srcX=NSHARD, srcY=0, srcPitch=M, srcSlicePitch=N * M,
        dstBase=0, dstX=0, dstY=0, dstPitch=NSHARD, dstSlicePitch=MT1 * NSHARD,
        rectX=NSHARD, rectY=MT1, elementSizeLog2=1)
    assert got == _PROD_COPY_GOLDEN, \
        "\n".join("DW%d got 0x%08X exp 0x%08X" % (i, g, e)
                  for i, (g, e) in enumerate(zip(got, _PROD_COPY_GOLDEN)) if g != e)


def test_encoder_matches_atomic_golden():
    got = encodeAtomicDwords(_ATOMIC_ADDR, addend=1)
    assert got == _ATOMIC_GOLDEN, \
        "\n".join("DW%d got 0x%08X exp 0x%08X" % (i, g, e)
                  for i, (g, e) in enumerate(zip(got, _ATOMIC_GOLDEN)) if g != e)


def test_production_boundary_rect_x_equals_dst_pitch():
    # The production shape is the rect_x == dst_pitch boundary case (the X extent
    # exactly fills one destination row): both fields encode to 2559. This mirrors
    # the C++ test's explicit boundary pin -- a degenerate-looking shape that is
    # easy to break later.
    got = encodeCopyDwords(
        srcBase=0, srcX=NSHARD, srcY=0, srcPitch=M, srcSlicePitch=N * M,
        dstBase=0, dstX=0, dstY=0, dstPitch=NSHARD, dstSlicePitch=MT1 * NSHARD,
        rectX=NSHARD, rectY=MT1, elementSizeLog2=1)
    dst_pitch_field = (got[9] >> 13) & ((1 << 19) - 1)
    rect_x_field = got[11] & ((1 << 14) - 1)
    assert dst_pitch_field == rect_x_field == (NSHARD - 1)


# ---------------------------------------------------------------------------
# Part 2: field arithmetic reproduces the golden coordinates
# ---------------------------------------------------------------------------
# The pure-Python encoder above takes coordinates as arguments; here we verify
# that the formulas (as the emitter computes them) produce those exact
# coordinates for a representative (p, j, myRank), so the whole packet built from
# scratch equals the production golden.

def _fields_from_geometry(p, j, myRank, n=N):
    """Reproduce emitComputeCopyFields in Python (element units)."""
    return dict(
        srcX=p * NSHARD,
        srcY=j * MT1,
        srcSlice=M * n,
        dstY=myRank * n + j * MT1,
        dstSlice=MT1 * NSHARD,
        rectY=min(MT1, n - j * MT1))


def test_geometry_p1_j0_myrank0_equals_production_golden():
    f = _fields_from_geometry(p=1, j=0, myRank=0)
    got = encodeCopyDwords(
        srcBase=0, srcX=f["srcX"], srcY=f["srcY"], srcPitch=M, srcSlicePitch=f["srcSlice"],
        dstBase=0, dstX=0, dstY=f["dstY"], dstPitch=NSHARD, dstSlicePitch=f["dstSlice"],
        rectX=NSHARD, rectY=f["rectY"], elementSizeLog2=1)
    assert got == _PROD_COPY_GOLDEN


def test_geometry_edge_token_tile_clamps_rect_y():
    # Tail token-tile with N % MT1 != 0: rect_y must clamp to N - j*MT1 (208),
    # not stay at MT1 (256). Pinned to the hand-derived edge golden above.
    f = _fields_from_geometry(p=1, j=_EDGE_J, myRank=0, n=_EDGE_N)
    assert f["rectY"] == _EDGE_RECT_Y
    got = encodeCopyDwords(
        srcBase=0, srcX=f["srcX"], srcY=f["srcY"], srcPitch=M, srcSlicePitch=f["srcSlice"],
        dstBase=0, dstX=0, dstY=f["dstY"], dstPitch=NSHARD, dstSlicePitch=f["dstSlice"],
        rectX=NSHARD, rectY=f["rectY"], elementSizeLog2=1)
    assert got == _EDGE_COPY_GOLDEN, \
        "\n".join("DW%d got 0x%08X exp 0x%08X" % (i, g, e)
                  for i, (g, e) in enumerate(zip(got, _EDGE_COPY_GOLDEN)) if g != e)
    # The unclamped form differs exactly in DW11's high half -- the read-overrun.
    unclamped = encodeCopyDwords(
        srcBase=0, srcX=f["srcX"], srcY=f["srcY"], srcPitch=M, srcSlicePitch=f["srcSlice"],
        dstBase=0, dstX=0, dstY=f["dstY"], dstPitch=NSHARD, dstSlicePitch=f["dstSlice"],
        rectX=NSHARD, rectY=MT1, elementSizeLog2=1)
    assert unclamped[11] != got[11] and unclamped[:11] == got[:11]


def test_self_rank_packet_well_formed():
    # p == myRank (self-rank copy: still goes through SDMA).
    for myRank in range(4):
        f = _fields_from_geometry(p=myRank, j=0, myRank=myRank)
        got = encodeCopyDwords(
            srcBase=0, srcX=f["srcX"], srcY=f["srcY"], srcPitch=M, srcSlicePitch=f["srcSlice"],
            dstBase=0, dstX=0, dstY=f["dstY"], dstPitch=NSHARD, dstSlicePitch=f["dstSlice"],
            rectX=NSHARD, rectY=f["rectY"], elementSizeLog2=1)
        # header + rect are rank-independent; only src_x / dst_y move. This is the
        # bf16-granular encoder form, so its header is NOT the shipped
        # COPY_HEADER_DW0 (which now carries elementsize=log2(16)).
        assert (got[0] >> 29) == BF16_ELEMENT_SIZE_LOG2
        assert (got[3] & 0x3FFF) == (myRank * NSHARD) & 0x3FFF


# ---------------------------------------------------------------------------
# Part 2b: the coordinate fold (what the shipped emitter actually does)
# ---------------------------------------------------------------------------
# emitComputeCopyFields no longer emits src_x/src_y/dst_x/dst_y as packet fields:
# it adds them into the 64-bit base addresses and leaves all four at 0. The
# hardware rule is addr(x,y) = base + y*pitch*elem + x*elem, so the two forms must
# name the same byte.
#
# WHAT THIS LAYER DOES AND DOES NOT PROVE. It decodes the address back out of the
# ENCODED dwords (unpacking the bitfields, undoing the minus-one pitch) and
# compares the two forms. That catches a wrong term, a wrong shift, or a dropped
# element->byte conversion in the fold arithmetic. It does NOT prove the emitted
# ASSEMBLY computes this arithmetic -- that is the structural tests (operand
# registers), the handshake golden, and ultimately the on-card recv comparison.

def _decode_first_element_addr(dw, side, elementSizeLog2=BF16_ELEMENT_SIZE_LOG2):
    """Byte address of the sub-window's first element, reconstructed from the
    encoded dwords via the hardware rule addr = base + y*pitch*elem + x*elem.

    Deliberately reads the packed BITS rather than the inputs, so it re-derives
    the minus-one pitch convention instead of trusting it."""
    b, c, p = (1, 3, 4) if side == "src" else (6, 8, 9)
    base  = dw[b] | (dw[b + 1] << 32)
    x     =  dw[c]        & _XY_MASK
    y     = (dw[c] >> 16) & _XY_MASK
    pitch = ((dw[p] >> 13) & _PITCH_MASK) + 1     # stored minus one
    elem  = 1 << elementSizeLog2
    return base + y * pitch * elem + x * elem


_XY_MASK    = (1 << 14) - 1
_PITCH_MASK = (1 << 19) - 1


def _encode_pair(p, j, myRank, w, n, nShard, ldd, dBase, recvBase):
    """Encode one (peer, token-tile) copy BOTH ways and return (coordForm, folded).

    coordForm is the pre-fold packet the MI355X golden validated; folded is what
    the emitter ships. The fold arithmetic here mirrors emitComputeCopyFields."""
    srcY   = j * MT1
    srcX   = p * nShard
    dstRow = myRank * n + srcY
    rectY  = min(MT1, n - srcY)
    common = dict(srcPitch=ldd, srcSlicePitch=M * n, dstPitch=nShard,
                  dstSlicePitch=MT1 * nShard, rectX=nShard, rectY=rectY,
                  elementSizeLog2=BF16_ELEMENT_SIZE_LOG2)
    coord = encodeCopyDwords(srcBase=dBase, srcX=srcX, srcY=srcY,
                             dstBase=recvBase, dstX=0, dstY=dstRow, **common)
    elemBytes = 1 << D_DATA_ELEMENT_LOG2
    folded = encodeCopyDwords(
        srcBase=dBase + (srcY * ldd + srcX) * elemBytes, srcX=0, srcY=0,
        dstBase=recvBase + (dstRow * nShard) * elemBytes, dstX=0, dstY=0, **common)
    return coord, folded


# Geometries where BOTH forms are representable (the coordinate form needs
# src_x and dst_y under 2^14), so the two can be compared at all.
_FOLD_CASES = [
    (w, n, nShard, p, j, myRank)
    for w, n, nShard in ((1, 2048, 2560), (2, 2048, 1280), (4, 2048, 640), (4, 512, 256))
    for p in range(w)
    for myRank in range(w)
    for j in (0, 1, (n // MT1) - 1)
]


@pytest.mark.parametrize("w,n,nShard,p,j,myRank", _FOLD_CASES)
def test_fold_addresses_identical_to_coordinate_form(w, n, nShard, p, j, myRank):
    dBase, recvBase = 0x7F0000100000, 0x7F0000900000
    coord, folded = _encode_pair(p, j, myRank, w, n, nShard, M, dBase, recvBase)
    for side in ("src", "dst"):
        assert _decode_first_element_addr(folded, side) \
            == _decode_first_element_addr(coord, side), \
            "%s address differs after folding (W=%d N=%d nShard=%d p=%d j=%d rank=%d)" \
            % (side, w, n, nShard, p, j, myRank)
    # Everything that is NOT an address must be untouched by the fold.
    assert folded[0] == coord[0]                    # header
    assert folded[4:6] == coord[4:6]                # src pitch / slice
    assert folded[9:13] == coord[9:13]              # dst pitch / slice, rect, DW12
    # ...and the four coordinate fields really are gone.
    assert folded[3] == 0 and folded[8] == 0


def test_fold_equivalence_check_is_sensitive():
    """Negative control for the test above.

    The comparison is between two encodings of the same geometry, so it would
    stay green if BOTH sides were wrong in the same way -- or if the decoder
    ignored the term under test. Perturb the folded base by ONE element and
    confirm the check goes red, once per side."""
    dBase, recvBase = 0x7F0000100000, 0x7F0000900000
    coord, _ = _encode_pair(1, 1, 1, 4, 2048, 640, M, dBase, recvBase)
    elemBytes = 1 << D_DATA_ELEMENT_LOG2
    for side, badSrc, badDst in (("src", elemBytes, 0), ("dst", 0, elemBytes)):
        _, wrong = _encode_pair(1, 1, 1, 4, 2048, 640, M,
                                dBase + badSrc, recvBase + badDst)
        assert _decode_first_element_addr(wrong, side) \
            != _decode_first_element_addr(coord, side), \
            "%s: a one-element error slipped past the equivalence check" % side


def test_fold_encodes_geometry_the_coordinate_form_cannot():
    """W=8 with N=4096 needs dst_y = (W-1)*N + (tokenTiles-1)*MT1 = 32512, past
    the 14-bit field the coordinate form uses. Folded, the same copy encodes
    exactly, and the address is the one the formula asks for -- computed here
    independently of the encoder."""
    w, n, nShard, ldd = 8, 4096, 1280, 18432
    p, j, myRank = 7, (n // MT1) - 1, 7
    dstRow = myRank * n + j * MT1
    # Premises: dst_y is the ONLY out-of-range term (32512); src_x fits at 8960.
    assert dstRow == 32512 and dstRow >= (1 << 14), \
        "premise: dst_y must overflow the old 14-bit field"
    assert p * nShard == 8960 < (1 << 14), \
        "premise: src_x fit -- dst_y alone is what rejected this geometry"

    dBase, recvBase = 0x7F0000100000, 0x7F0000900000
    _, folded = _encode_pair(p, j, myRank, w, n, nShard, ldd, dBase, recvBase)
    assert folded[3] == 0 and folded[8] == 0
    elemBytes = 1 << D_DATA_ELEMENT_LOG2
    assert _decode_first_element_addr(folded, "src") \
        == dBase + (j * MT1 * ldd + p * nShard) * elemBytes
    assert _decode_first_element_addr(folded, "dst") \
        == recvBase + dstRow * nShard * elemBytes


def test_field_fit_guard_rejects_each_overflowing_field():
    # Every X-direction bound below is on the SCALED value (>> ELEMENT_SHIFT),
    # because that is what the packet field holds. Written in terms of the
    # constants, not literals, so the arithmetic follows if the element widens
    # again.
    RECT_X_MAX = XY_FIELD_LIMIT << ELEMENT_SHIFT       # nShard ceiling: 131072
    PITCH_MAX  = PITCH_FIELD_LIMIT << ELEMENT_SHIFT    # ldd ceiling: 4194304

    # Positive control first: the shipping shape must NOT raise, or the negative
    # cases below prove nothing.
    checkA2AFieldsFit(numRanks=4, nShard=NSHARD, macroTile1=MT1, srcPitch=M)

    # (a) rect_x at its ceiling. This is the ONE coordinate-space term the fold
    #     could not remove -- it is the copy's X extent, not an offset.
    with pytest.raises(ValueError, match="rect"):
        checkA2AFieldsFit(numRanks=1, nShard=RECT_X_MAX, macroTile1=MT1, srcPitch=M)

    # (b) src_pitch = ldd at the 19-bit limit; every other term in range.
    with pytest.raises(ValueError, match="pitch"):
        checkA2AFieldsFit(numRanks=4, nShard=NSHARD, macroTile1=MT1, srcPitch=PITCH_MAX)

    # (c) W > FUSED_A2A_MAX_RANKS(8): rect_x and the pitch are both in range, so
    #     ONLY the rank bound can trip -- the kernarg segment has no slot for 8+.
    with pytest.raises(ValueError, match="FUSED_A2A_MAX_RANKS"):
        checkA2AFieldsFit(numRanks=16, nShard=256, macroTile1=MT1, srcPitch=M)

    # (d) NOT divisible by the packet element. Both terms, separately, and both
    #     comfortably inside every range bound -- so only divisibility can trip.
    #     A right shift would truncate these and copy a short band.
    with pytest.raises(ValueError, match="multiple"):
        checkA2AFieldsFit(numRanks=4, nShard=NSHARD + 1, macroTile1=MT1, srcPitch=M)
    with pytest.raises(ValueError, match="multiple"):
        checkA2AFieldsFit(numRanks=4, nShard=NSHARD, macroTile1=MT1, srcPitch=M + 1)

    # Boundaries: at the ceiling rejects, one element below passes -- both terms.
    checkA2AFieldsFit(numRanks=1, nShard=RECT_X_MAX - ELEMENT_MULTIPLE,
                      macroTile1=MT1, srcPitch=M)
    checkA2AFieldsFit(numRanks=4, nShard=NSHARD, macroTile1=MT1,
                      srcPitch=PITCH_MAX - ELEMENT_MULTIPLE)

    # This geometry (W=8, N=4096 -> dst_y=32512 in the old coordinate form) must
    # be ACCEPTED. N is not even an argument any more -- if someone reintroduces
    # a coordinate term here, this is what catches it.
    checkA2AFieldsFit(numRanks=8, nShard=1280, macroTile1=MT1, srcPitch=18432)

    # The production shape at W=8 needs nShard = 131072/8 = 16384, which would
    # overflow rect_x under a narrower element; the wider element makes it
    # exactly encodable.
    checkA2AFieldsFit(numRanks=8, nShard=16384, macroTile1=MT1, srcPitch=18432)


# ---------------------------------------------------------------------------
# Part 3: rocisa emitter structural asserts (semantic field-packing features)
# ---------------------------------------------------------------------------

def _init_gfx950():
    ri = rocIsa.getInstance()
    isa = gfxToIsa(_GFX)
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)
    return ri


def _mock_writer():
    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.vgprPool.checkOut(1)   # v0 reserved (Serial)
    w.sgprPool.checkOut(8)   # reserve s0..s7
    return w


def _render_copy():
    return _render_copy_ns().text


def _render_copy_ns():
    """Render emitBuildCopyPacket AND return the registers it used (see _has_alu
    for why tests assert on operand registers instead of comment text)."""
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    pkt = w.vgprPool.checkOut(COPY_PACKET_DWORDS, "pkt")
    s = [w.sgprPool.checkOutAligned(2, 2, "b%d" % i, preventOverflow=False) for i in range(2)]
    srcBase, dstBase = s[0], s[1]
    fld = [w.sgprPool.checkOut(1, "f%d" % i, preventOverflow=False) for i in range(7)]
    (srcPitch, srcSlice, dstPitch, dstSlice, rectX, rectY, _spare) = fld
    tmp = w.sgprPool.checkOut(2, "tmp", preventOverflow=False)  # rect dword needs 2
    m = Module("copy")
    em.emitBuildCopyPacket(m, w, pkt, srcBase, srcPitch, srcSlice,
                           dstBase, dstPitch, dstSlice, rectX, rectY, tmp)
    return SimpleNamespace(text=str(m), pkt=pkt, srcBase=srcBase, dstBase=dstBase,
                           srcPitch=srcPitch, srcSlice=srcSlice,
                           dstPitch=dstPitch, dstSlice=dstSlice,
                           rectX=rectX, rectY=rectY, tmp=tmp)


def _render_atomic():
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    pkt = w.vgprPool.checkOut(ATOMIC_PACKET_DWORDS, "pkt")
    addr = w.sgprPool.checkOutAligned(2, 2, "addr", preventOverflow=False)
    m = Module("atomic")
    em.emitBuildAtomicPacket(m, w, pkt, addr, addend=1)
    return str(m)


def _render_fields():
    return _render_fields_ns().text


def _render_fields_ns():
    """Render emitComputeCopyFields AND return the SGPR indices it was given (see
    _has_alu for why tests assert on operand registers instead of comment text);
    for the folded base addresses, a swapped multiply here is a silent
    cross-rank data-placement bug."""
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    ins = [w.sgprPool.checkOut(1, "in%d" % i, preventOverflow=False) for i in range(7)]
    (p, j, myRank, mS, nS, nShardS, srcPitchS) = ins
    # The 64-bit operands must be 2-aligned, exactly as the production caller
    # allocates them (GlobalWriteBatch._emitFusedA2ASdmaIssue).
    addressD  = w.sgprPool.checkOutAligned(2, 2, "addressD", preventOverflow=False)
    recvBase  = w.sgprPool.checkOutAligned(2, 2, "recvBase", preventOverflow=False)
    outSrcBase = w.sgprPool.checkOutAligned(2, 2, "srcBase", preventOverflow=False)
    tmp64     = w.sgprPool.checkOutAligned(2, 2, "tmp64", preventOverflow=False)
    outs = [w.sgprPool.checkOut(1, "o%d" % i, preventOverflow=False) for i in range(5)]
    (srcY, srcSlice, dstSlice, rectY, tmp) = outs
    m = Module("fields")
    em.emitComputeCopyFields(m, w, p, j, myRank, mS, nS, nShardS,
                             addressD, srcPitchS, recvBase,
                             outSrcBase, srcY, srcSlice, dstSlice, rectY,
                             tmp, tmp64)
    return SimpleNamespace(text=str(m), p=p, j=j, myRank=myRank, mS=mS, nS=nS,
                           nShardS=nShardS, srcPitchS=srcPitchS, addressD=addressD,
                           recvBase=recvBase, outSrcBase=outSrcBase, tmp64=tmp64,
                           srcY=srcY, srcSlice=srcSlice,
                           dstSlice=dstSlice, rectY=rectY, tmp=tmp)


def _render_flag_addr():
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    flagBase = w.sgprPool.checkOutAligned(2, 2, "flagBase", preventOverflow=False)
    myRank = w.sgprPool.checkOut(1, "myRank", preventOverflow=False)
    outAddr = w.sgprPool.checkOutAligned(2, 2, "outAddr", preventOverflow=False)
    tmp = w.sgprPool.checkOut(1, "tmp", preventOverflow=False)
    m = Module("flagaddr")
    em.emitComputeFlagAddr(m, w, flagBase, myRank, outAddr, tmp)
    return str(m)


def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _reg64(n):
    """How rocisa renders an aligned SGPR pair: `s[n:n+1]`. Pass this to _has_alu
    for 64-bit operands, which do not render as a bare `sN`."""
    return "s[%d:%d]" % (n, n + 1)


def _code(ln):
    return ln.split("//")[0]


def _has_alu(lines, mnemonic, dst, operands):
    """True iff some line is `<mnemonic> s<dst>, <ops...>` (code side only, i.e.
    ignoring the comment) with the given dst SGPR and the exact set of source
    operands, order-independent. `operands` are the source tokens to match:
    ints are SGPR indices rendered as `sN`, strings are matched literally (e.g.
    an immediate "256"). Asserting on the operand registers -- not the comment --
    is what makes these tests catch a swapped multiply."""
    want_srcs = sorted("s%d" % o if isinstance(o, int) else str(o) for o in operands)
    dst_tok = dst if isinstance(dst, str) else "s%d" % dst
    for ln in lines:
        code = _code(ln).strip()
        if not code.startswith(mnemonic + " "):
            continue
        toks = [t.strip() for t in code[len(mnemonic):].split(",")]
        if not toks or toks[0] != dst_tok:
            continue
        got_srcs = sorted(toks[1:])
        if got_srcs == want_srcs:
            return True
    return False


class TestCopyStructural:

    def test_header_immediate_is_copy_rect_with_packet_element(self):
        # DW0 = op=COPY(1) | sub_op=RECT(4)<<8 | elementsize<<29. The elementsize
        # is log2(16)=4 (the packet addresses in 16-byte elements), NOT log2(2):
        # 4<<29 = 0x80000000, so the immediate sits ABOVE INT32_MAX -- which is
        # exactly the value class rocisa renders as a float unless it is passed as
        # hex. test_bitfield_immediates_never_render_as_float covers the mechanism;
        # this pins that the shipped header really is in that class.
        assert COPY_HEADER_DW0 == 0x80000401
        assert COPY_HEADER_DW0 > 0x7FFFFFFF
        assert (COPY_HEADER_DW0 >> 29) == PACKET_ELEMENT_SIZE_LOG2
        lines = _lines(_render_copy())
        hdr = [ln for ln in lines if "DW0" in ln and "v_mov_b32" in ln]
        assert hdr and hex(COPY_HEADER_DW0) in _code(hdr[0]), \
            f"DW0 must move the COPY_HEADER immediate {hex(COPY_HEADER_DW0)}: {hdr}"

    def test_pitch_minus_one_then_shift13(self):
        # Pitch fields are (pitch-1) << 13. Assert both a minus-one and a <<13.
        lines = _lines(_render_copy())
        assert any("s_sub_u32" in _code(ln) and ", 1" in _code(ln) and "pitch - 1" in ln
                   for ln in lines), "expected a pitch-1 subtract"
        assert any("s_lshl_b32" in _code(ln) and ", 13" in _code(ln) for ln in lines), \
            "expected a << 13 for the pitch field"

    def test_rect_y_comes_from_a_register_not_an_immediate(self):
        # rect_y must be RUNTIME (min(MT1, N - j*MT1)): a folded (MT1-1)<<16
        # immediate would copy a full tile out of a partial tail tile and read
        # past the end of D. Assert the real operands: (rectY-1) into tmp+1,
        # shifted left 16, then OR'd with (rectX-1).
        ns = _render_copy_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_sub_u32", ns.tmp + 1, [ns.rectY, "1"]), \
            "rect_y-1 must be computed from the rectY SGPR"
        assert _has_alu(lines, "s_lshl_b32", ns.tmp + 1, [ns.tmp + 1, "16"]), \
            "(rect_y-1) must be shifted into bits [29:16]"
        assert _has_alu(lines, "s_or_b32", ns.tmp, [ns.tmp, ns.tmp + 1]), \
            "DW11 must OR (rect_x-1) with (rect_y-1)<<16"
        assert str((MT1 - 1) << 16) not in ns.text and hex((MT1 - 1) << 16) not in ns.text, \
            "rect_y must not be folded as the compile-time (MT1-1)<<16 immediate"

    def test_all_13_dwords_written(self):
        # Every packet dword v_mov must be present (DW0..DW12).
        text = _render_copy()
        for i in range(COPY_PACKET_DWORDS):
            assert ("DW%d" % i) in text, f"missing packet dword DW{i}"

    def test_only_x_direction_fields_are_element_scaled(self):
        # The packet addresses in 16-byte elements, so pitches / slice pitches /
        # rect_x are shifted right by ELEMENT_SHIFT. rect_y must NOT be: it counts
        # ROWS, and ELEMENTSIZE scales only the X direction (rocm-ref
        # sdma-engines.md). Scaling rect_y would copy one eighth of every band --
        # and the recv buffer would still be the right SIZE, so nothing but a
        # numeric comparison would notice.
        ns = _render_copy_ns()
        lines = _lines(ns.text)
        shifted = [ln for ln in lines
                   if "s_lshr_b32" in _code(ln) and (", %d" % ELEMENT_SHIFT) in _code(ln)]
        # src_pitch, src_slice, dst_pitch, dst_slice, rect_x -- five, no more.
        assert len(shifted) == 5, \
            "expected exactly 5 element-scaled fields, got %d: %s" % (len(shifted), shifted)
        assert not any(str(ns.rectY) == _code(ln).split(",")[1].strip().lstrip("s")
                       for ln in shifted), "rect_y must NOT be element-scaled"
        # And the scaled operand for rect_x really is rectX.
        assert _has_alu(lines, "s_lshr_b32", ns.tmp, [ns.rectX, str(ELEMENT_SHIFT)]), \
            "rect_x must be the register that gets scaled"
        # rect_y-1 comes straight off the rectY register, unscaled.
        assert _has_alu(lines, "s_sub_u32", ns.tmp + 1, [ns.rectY, "1"]), \
            "rect_y-1 must be computed directly from rectY, with no scaling"

    def test_all_four_coordinates_are_literal_zero(self):
        # The coordinates are folded into the base addresses, so DW3 and DW8 must
        # be plain `v_mov_b32 vN, 0` -- no register, no shift, no OR. If any
        # coordinate leaked back into a field it would ALSO still be in the base,
        # double-counting the offset and scattering into the wrong recv slot.
        ns = _render_copy_ns()
        lines = _lines(ns.text)
        for dw in (3, 8):
            movs = [ln for ln in lines
                    if ("DW%d:" % dw) in ln and "v_mov_b32" in _code(ln)]
            assert len(movs) == 1, "expected exactly one v_mov for DW%d: %s" % (dw, movs)
            assert re.search(r"v_mov_b32 v\d+, 0x0\b", _code(movs[0])), \
                "DW%d must be an immediate 0 (coordinates are folded into the " \
                "base): %s" % (dw, _code(movs[0]))
        # And nothing may shift a value into the coordinate y position (bit 16)
        # except the rect dword, which is a different field entirely.
        shifts16 = [ln for ln in lines
                    if "s_lshl_b32" in _code(ln) and ", 16" in _code(ln)]
        assert all("rect" in ln.lower() for ln in shifts16), \
            "only rect_y may be shifted to bit 16 now: %s" % shifts16


class TestAtomicStructural:

    def test_header_immediate_is_atomic_add_rtn_32(self):
        assert ATOMIC_HEADER_DW0 == 0x1E00000A
        lines = _lines(_render_atomic())
        hdr = [ln for ln in lines if "DW0" in ln and "v_mov_b32" in ln]
        assert hdr and hex(ATOMIC_HEADER_DW0) in _code(hdr[0]), \
            f"ATOMIC DW0 must move {hex(ATOMIC_HEADER_DW0)}: {hdr}"

    def test_addend_is_one(self):
        text = _render_atomic()
        assert "src_data lo (addend)" in text
        lines = _lines(text)
        addlo = [ln for ln in lines if "src_data lo" in ln]
        assert addlo and re.search(r"v_mov_b32 v\d+, 0x1\b", _code(addlo[0])), \
            f"ATOMIC addend lo must be immediate 1: {addlo}"

    def test_bitfield_immediates_never_render_as_float(self):
        # 0x9E00000A is the value that actually breaks (rocisa renders an int
        # above INT32_MAX as a float); every immediate the emitter currently
        # emits is below the boundary and would pass without the hex form.
        _init_gfx950()
        w = _mock_writer()
        em = SdmaPacketEmitter(macroTile1=MT1)
        pkt = w.vgprPool.checkOut(1, "pkt")
        m = Module("hi")
        em._movImm(m, pkt, 10 | (0x4F << 25), "bit-31 immediate")  # 0x9E00000A
        code = _code(_lines(str(m))[0])
        assert "0x9e00000a" in code.lower(), \
            f"bit-31 immediate must render as hex, got: {code}"
        assert ".0" not in code, \
            f"immediate rendered as a FLOAT -- the assembler will encode its " \
            f"IEEE-754 bits, not the value: {code}"

    def test_all_8_dwords_written(self):
        text = _render_atomic()
        for i in range(ATOMIC_PACKET_DWORDS):
            assert ("DW%d" % i) in text, f"missing atomic dword DW{i}"


class TestFieldArithmetic:

    def test_src_base_folds_row_offset_plus_feature_offset(self):
        # srcBase = AddressD + (j*MT1*ldd + p*nShard) * 2. Assert the real operand
        # registers at every step -- a swapped multiply here silently misplaces the
        # source band across ranks, and no field-level check can see it any more
        # now that the coordinates are gone from the packet.
        ns = _render_fields_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_mul_i32", ns.srcY, [ns.j, str(MT1)]), \
            "expected s_mul_i32 srcY, j, MT1"
        assert _has_alu(lines, "s_mul_i32", ns.tmp, [ns.p, ns.nShardS]), \
            "expected s_mul_i32 tmp, p, nShard (the feature offset term)"
        # 32x32 -> 64 widening multiply of the row index by the pitch.
        assert _has_alu(lines, "s_mul_hi_u32", ns.tmp64 + 1, [ns.srcY, ns.srcPitchS]), \
            "expected s_mul_hi_u32 tmp64+1, srcY, ldd (high word of j*MT1*ldd)"
        assert _has_alu(lines, "s_mul_i32", ns.tmp64 + 0, [ns.srcY, ns.srcPitchS]), \
            "expected s_mul_i32 tmp64+0, srcY, ldd (low word)"
        # ...then + p*nShard WITH carry, scaled to bytes, added onto AddressD.
        assert _has_alu(lines, "s_add_u32", ns.tmp64 + 0, [ns.tmp64 + 0, ns.tmp]), \
            "expected the feature offset added into the low word"
        assert _has_alu(lines, "s_addc_u32", ns.tmp64 + 1, [ns.tmp64 + 1, "0"]), \
            "the low-word add MUST propagate its carry -- otherwise a large " \
            "j*MT1*ldd truncates the address"
        assert _has_alu(lines, "s_lshl_b64", _reg64(ns.tmp64),
                        [_reg64(ns.tmp64), str(D_DATA_ELEMENT_LOG2)]), \
            "expected s_lshl_b64 tmp64, tmp64, log2(sizeof(bf16))"
        assert _has_alu(lines, "s_add_u32", ns.outSrcBase + 0,
                        [ns.addressD + 0, ns.tmp64 + 0]) and \
               _has_alu(lines, "s_addc_u32", ns.outSrcBase + 1,
                        [ns.addressD + 1, ns.tmp64 + 1]), \
            "srcBase must be a 64-bit add of AddressD + offset (lo add + hi addc)"
        # AddressD is persistent: it must never be a destination.
        for reg in (ns.addressD, ns.addressD + 1):
            assert not any(_code(ln).strip().split(",")[0].endswith(" s%d" % reg)
                           for ln in lines), \
                "AddressD s%d was written -- it is a persistent register" % reg

    def test_dst_base_folds_myrank_n_plus_srcy_times_nshard(self):
        # recvBase += (myRank*N + j*MT1) * nShard * 2, updated IN PLACE.
        ns = _render_fields_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_mul_i32", ns.tmp, [ns.myRank, ns.nS]), \
            "expected s_mul_i32 tmp, myRank, N"
        assert _has_alu(lines, "s_add_u32", ns.tmp, [ns.tmp, ns.srcY]), \
            "expected s_add_u32 tmp, tmp(myRank*N), srcY(j*MT1)"
        assert _has_alu(lines, "s_mul_hi_u32", ns.tmp64 + 1, [ns.tmp, ns.nShardS]), \
            "expected s_mul_hi_u32 tmp64+1, dstRow, nShard (high word)"
        assert _has_alu(lines, "s_mul_i32", ns.tmp64 + 0, [ns.tmp, ns.nShardS]), \
            "expected s_mul_i32 tmp64+0, dstRow, nShard (low word)"
        assert _has_alu(lines, "s_add_u32", ns.recvBase + 0,
                        [ns.recvBase + 0, ns.tmp64 + 0]) and \
               _has_alu(lines, "s_addc_u32", ns.recvBase + 1,
                        [ns.recvBase + 1, ns.tmp64 + 1]), \
            "recvBase must be updated in place by a 64-bit add"

    def test_both_folds_are_64_bit(self):
        # The single most dangerous way to get this wrong is a 32-bit multiply:
        # it is correct for every shape small enough to test cheaply and wrong for
        # the large ones the fold exists to enable. Pin the count of widening
        # multiplies at exactly two -- one per side.
        lines = _lines(_render_fields_ns().text)
        hi = [ln for ln in lines if _code(ln).strip().startswith("s_mul_hi_u32 ")]
        assert len(hi) == 2, \
            "expected exactly 2 s_mul_hi_u32 (one per folded base), got %d: %s" \
            % (len(hi), hi)
        # Signed high-multiply would misread a stride with bit 31 set as negative.
        assert not any("s_mul_hi_i32" in _code(ln) for ln in lines), \
            "the widening multiply must be UNSIGNED"

    def test_rect_y_is_clamped_to_tokens_left(self):
        # rect_y = min(MT1, N - j*MT1): a subtract of src_y (== j*MT1) from N
        # followed by an s_min_u32 against the compile-time MT1. Asserted on the
        # real operands so a missing clamp (or a subtract from the wrong term)
        # fails -- an unclamped rect_y reads past the end of D on the tail tile.
        ns = _render_fields_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_sub_u32", ns.rectY, [ns.nS, ns.srcY]), \
            "expected s_sub_u32 rectY, N, srcY (tokens left in this tile)"
        assert _has_alu(lines, "s_min_u32", ns.rectY, [ns.rectY, str(MT1)]), \
            "expected s_min_u32 rectY, rectY, MT1 (clamp the tail tile)"

    def test_flag_addr_stride_is_myrank_times_4_64bit(self):
        # flag addr = peer_ptr[p] + myRank*4, a 64-bit add (lo add + hi carry).
        # Stride is 4 (u32 flag slots): the ATOMIC is an ADD_RTN_32, a 4-byte
        # write. Must agree with the host's flagBytes and the poll's j*4.
        lines = _lines(_render_flag_addr())
        assert any("s_lshl_b32" in _code(ln) and ", 2" in _code(ln) for ln in lines), \
            "expected myRank*4 (shift left 2, u32 flag-slot stride)"
        # An *8 (shift 3) stride must NOT reappear: with a W*4-byte allocation it
        # would put the top rank's atomic past the end of the flag buffer.
        assert not any("s_lshl_b32" in _code(ln) and ", 3" in _code(ln) for ln in lines), \
            "flag stride must be *4 (u32 slots), not the old *8"
        assert any("s_add_u32" in _code(ln) for ln in lines) and \
               any("s_addc_u32" in _code(ln) for ln in lines), \
            "flag addr must be a 64-bit add (add lo + addc hi)"


# ---------------------------------------------------------------------------
# Part 4: assemble every emitter's rendered text (MUST, per plan)
# ---------------------------------------------------------------------------
def _assembler():
    return shutil.which("amdclang++") or (
        "/opt/rocm/bin/amdclang++" if os.path.exists("/opt/rocm/bin/amdclang++") else None)


@pytest.mark.skipif(_assembler() is None, reason="no amdclang++ to assemble with")
class TestAssembles:

    @pytest.mark.parametrize("render", [_render_copy, _render_atomic, _render_fields, _render_flag_addr],
                             ids=["copy", "atomic", "fields", "flagaddr"])
    def test_body_assembles(self, render, tmp_path):
        body = render()
        asm = _MINIMAL_KERNEL % {"gfx": _GFX, "body": body}
        s_path = tmp_path / "sdma_pkt.s"
        o_path = tmp_path / "sdma_pkt.o"
        s_path.write_text(asm)
        r = subprocess.run(
            [_assembler(), "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=%s" % _GFX, "-mcode-object-version=5", "-c", str(s_path), "-o", str(o_path)],
            capture_output=True, text=True)
        assert r.returncode == 0, f"assembly failed:\n{r.stderr}\n---\n{asm}"


_MINIMAL_KERNEL = """\
.amdgcn_target "amdgcn-amd-amdhsa--%(gfx)s"
.text
.globl test_kernel
.p2align 8
.type test_kernel,@function
test_kernel:
%(body)s
  s_endpgm
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr 100
  .amdhsa_accum_offset 64
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
.end_amdhsa_kernel
"""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
