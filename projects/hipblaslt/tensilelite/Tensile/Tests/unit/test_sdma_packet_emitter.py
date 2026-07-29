#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter tests (Task 5, NOGPU).
#
# The emitter (Tensile/Components/SdmaPacketEmitter.py) turns the §1.3 all-to-all
# geometry into the COPY_SUBWIN + ATOMIC ADD64 packet dword arrays. Two surfaces
# are tested and cross-checked:
#   * pure-Python encoders (encodeCopyDwords / encodeAtomicDwords) are pinned to
#     the golden dword vectors below, in BOTH the harness form (padded dst pitch
#     2624) and the production form (unpadded nShard=2560). This is the plan's
#     named verification: the emitter's immediates must equal the
#     byte-for-byte-on-MI355X golden. See the provenance note above the vectors
#     -- this file is self-contained on that point and does not depend on the
#     C++ packet header, which is slated for removal (nothing in the client
#     runtime ever consumed it).
#   * the rocisa emitters (emitBuildCopyPacket / emitBuildAtomicPacket /
#     emitComputeCopyFields / emitComputeFlagAddr) are asserted on their SEMANTIC
#     field-packing features (header immediates, minus-one encoding, shift
#     positions) -- NOT a whole-text snapshot -- and every one is run through the
#     gfx950 assembler (a MUST, not a bonus: Task 4 caught an illegal opcode this
#     way).
#
# Boundary cases pinned: self-rank (p == myRank) still produces a well-formed
# packet, and j at both ends of [0, tokenTiles) keeps dst_y within the 14-bit
# field.
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

# rocisa is imported transitively; skip cleanly if the C++ module is not built.
rocisa = pytest.importorskip("rocisa")

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
    BF16_ELEMENT_SIZE_LOG2,
)

# ---- shared full-shape constants (§1.2), element units -----------------------
M       = 18432   # feature (src row pitch)
N       = 2048    # token
NSHARD  = 2560    # feature shard per rank (rect X extent, prod dst pitch)
MT1     = 256     # token tile (rect Y extent)
_GFX    = "gfx950"


# ---------------------------------------------------------------------------
# Part 1: pure-Python encoders vs the golden dwords (the plan's named check)
# ---------------------------------------------------------------------------
# PROVENANCE OF THESE VECTORS -- they are hand-written constants on purpose. A
# golden must be an EXTERNAL reference; regenerating it from the code under test
# would make the check circular. Their authority differs per vector, and the
# difference matters:
#
#   _HARNESS_COPY_GOLDEN -- the only sequence with real hardware backing. These
#     exact bytes ran on MI355X (3 peers x 8 bands = 24 packets; every dword
#     bit-accurate, sentinel margin untouched). Independently, the Task 1
#     reviewer hand-recomputed all 13 dwords from the field spec rather than
#     accepting program output, which is what rules out "the golden is just
#     whatever the encoder printed".
#   _PROD_COPY_GOLDEN -- NO hardware backing. Derived by hand from the same
#     rules; only DW9/DW10 differ from the harness vector (dst pitch 2560 vs
#     2624), and both are checkable by inspection: (2560-1)<<13 == 0x013FE000,
#     256*2560-1 == 0x0009FFFF.
#   _ATOMIC_GOLDEN -- NO hardware backing and no second source. Derived from
#     MORI's SDMA_PKT_ATOMIC only: DW0 == 10 | (47<<25) == 0x5E00000A. First
#     real execution is Task 7/8.
#
# The BIT POSITIONS these encode (minus-one extents/pitches, ELEMENTSIZE
# scaling, the <<13 pitch placement) come from AMD OSS 4.4 sdma.pkt, cross-
# checked against ROCR sdma_registers.h and the kernel's vega10_sdma_pkt_open.h
# -- all three agree. GFX12+ uses a DIFFERENT layout of the same size; these
# vectors are gfx9xx / gfx95x only.
#
# DO NOT "update the golden" to make a red test pass. A mismatch means either
# the packing code drifted (fix the code) or the rule itself changed (then you
# need a new source, and this note must be updated to cite it).

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

# ATOMIC ADD64 to a flag slot with distinct lo/hi bytes (matches the C++ test).
_ATOMIC_ADDR = 0x0000ABCD12345678
_ATOMIC_GOLDEN = [
    0x5E00000A, 0x12345678, 0x0000ABCD, 0x00000001,
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
# Part 2: §1.3 field arithmetic reproduces the golden coordinates
# ---------------------------------------------------------------------------
# The pure-Python encoder above takes coordinates as arguments; here we verify
# that the §1.3 formulas (as the emitter computes them) produce those exact
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
    # p == myRank (self-rank copy, §1.5: still goes through SDMA). dst_y folds in
    # myRank*N; assert it stays within the 14-bit dst_y field for the largest rank.
    for myRank in range(4):
        f = _fields_from_geometry(p=myRank, j=0, myRank=myRank)
        assert f["dstY"] < (1 << 14), f"dst_y {f['dstY']} overflows 14-bit field"
        got = encodeCopyDwords(
            srcBase=0, srcX=f["srcX"], srcY=f["srcY"], srcPitch=M, srcSlicePitch=f["srcSlice"],
            dstBase=0, dstX=0, dstY=f["dstY"], dstPitch=NSHARD, dstSlicePitch=f["dstSlice"],
            rectX=NSHARD, rectY=f["rectY"], elementSizeLog2=1)
        # header + rect are rank-independent; only src_x / dst_y move.
        assert got[0] == COPY_HEADER_DW0
        assert (got[3] & 0x3FFF) == (myRank * NSHARD) & 0x3FFF


def test_token_tile_boundaries_fit_fields():
    # j at both ends of [0, tokenTiles); tokenTiles = N/MT1 = 8. dst_y = myRank*N
    # + j*MT1 must fit the 14-bit field even at the top rank + top tile.
    tokenTiles = N // MT1
    for j in (0, tokenTiles - 1):
        f = _fields_from_geometry(p=0, j=j, myRank=3)
        assert f["dstY"] < (1 << 14), f"dst_y {f['dstY']} (j={j}) overflows 14-bit field"
        # src_y = j*MT1 must also fit its 14-bit field.
        assert f["srcY"] < (1 << 14)


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
    """Render emitBuildCopyPacket AND return the registers it used, so tests can
    assert on real operands (e.g. DW8 == dst_y<<16 with the actual dstY SGPR)
    instead of comment text."""
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    pkt = w.vgprPool.checkOut(COPY_PACKET_DWORDS, "pkt")
    s = [w.sgprPool.checkOutAligned(2, 2, "b%d" % i, preventOverflow=False) for i in range(2)]
    srcBase, dstBase = s[0], s[1]
    fld = [w.sgprPool.checkOut(1, "f%d" % i, preventOverflow=False) for i in range(9)]
    (srcX, srcY, srcPitch, srcSlice, dstY, dstPitch, dstSlice, rectX, rectY) = fld
    tmp = w.sgprPool.checkOut(2, "tmp", preventOverflow=False)  # rect dword needs 2
    m = Module("copy")
    em.emitBuildCopyPacket(m, w, pkt, srcBase, srcX, srcY, srcPitch, srcSlice,
                           dstBase, dstY, dstPitch, dstSlice, rectX, rectY, tmp)
    return SimpleNamespace(text=str(m), pkt=pkt, srcBase=srcBase, dstBase=dstBase,
                           srcX=srcX, srcY=srcY, srcPitch=srcPitch, srcSlice=srcSlice,
                           dstY=dstY, dstPitch=dstPitch, dstSlice=dstSlice,
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
    """Render emitComputeCopyFields AND return the SGPR indices it was given, so
    tests can assert on actual operand registers (s_mul_i32 s<out>, s<a>, s<b>)
    rather than on comment text -- a comment-only assert stays green even if the
    multiply operands are swapped, which for src/dst coordinates is a silent
    cross-rank data-placement bug."""
    _init_gfx950()
    w = _mock_writer()
    em = SdmaPacketEmitter(macroTile1=MT1)
    ins = [w.sgprPool.checkOut(1, "in%d" % i, preventOverflow=False) for i in range(6)]
    (p, j, myRank, mS, nS, nShardS) = ins
    outs = [w.sgprPool.checkOut(1, "o%d" % i, preventOverflow=False) for i in range(7)]
    (srcX, srcY, srcSlice, dstY, dstSlice, rectY, tmp) = outs
    m = Module("fields")
    em.emitComputeCopyFields(m, w, p, j, myRank, mS, nS, nShardS,
                             srcX, srcY, srcSlice, dstY, dstSlice, rectY, tmp)
    return SimpleNamespace(text=str(m), p=p, j=j, myRank=myRank, mS=mS, nS=nS,
                           nShardS=nShardS, srcX=srcX, srcY=srcY, srcSlice=srcSlice,
                           dstY=dstY, dstSlice=dstSlice, rectY=rectY, tmp=tmp)


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
    dst_tok = "s%d" % dst
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

    def test_header_immediate_is_copy_rect_bf16(self):
        # DW0 must be the exact op=COPY(1) | sub_op=RECT(4)<<8 | elementsize(1)<<29
        # immediate -- the same constant the C++ golden pins.
        assert COPY_HEADER_DW0 == 0x20000401
        lines = _lines(_render_copy())
        hdr = [ln for ln in lines if "DW0" in ln and "v_mov_b32" in ln]
        assert hdr and str(COPY_HEADER_DW0) in _code(hdr[0]), \
            f"DW0 must move the COPY_HEADER immediate {COPY_HEADER_DW0}: {hdr}"

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

    def test_dst_x_is_zero(self):
        # dst_x is always 0 (recv slot base points at the shard start), so DW8 is
        # just dst_y<<16. Assert the real shift operand (s_lshl_b32 tmp, dstY, 16)
        # and that no OR mixes another register into it -- a comment-only check
        # would stay green if dst_x were accidentally added back in.
        ns = _render_copy_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_lshl_b32", ns.tmp, [ns.dstY, "16"]), \
            "DW8 must be s_lshl_b32 tmp, dstY, 16 (dst_x==0, dst_y<<16 only)"
        # No s_or into tmp that would fold a dst_x register into DW8.
        assert not any(_code(ln).strip().startswith("s_or_b32 s%d," % ns.tmp)
                       and "dst" in ln.lower() and "<< 16" not in ln
                       for ln in lines), "DW8 must not OR a dst_x term"


class TestAtomicStructural:

    def test_header_immediate_is_atomic_add64(self):
        assert ATOMIC_HEADER_DW0 == 0x5E00000A
        lines = _lines(_render_atomic())
        hdr = [ln for ln in lines if "DW0" in ln and "v_mov_b32" in ln]
        assert hdr and str(ATOMIC_HEADER_DW0) in _code(hdr[0]), \
            f"ATOMIC DW0 must move {ATOMIC_HEADER_DW0}: {hdr}"

    def test_addend_is_one(self):
        text = _render_atomic()
        assert "src_data lo (addend)" in text
        lines = _lines(text)
        addlo = [ln for ln in lines if "src_data lo" in ln]
        assert addlo and re.search(r"v_mov_b32 v\d+, 1\b", _code(addlo[0])), \
            f"ATOMIC addend lo must be immediate 1: {addlo}"

    def test_all_8_dwords_written(self):
        text = _render_atomic()
        for i in range(ATOMIC_PACKET_DWORDS):
            assert ("DW%d" % i) in text, f"missing atomic dword DW{i}"


class TestFieldArithmetic:

    def test_src_x_is_p_times_nshard(self):
        # src_x = p * nShard: assert the actual multiply operands (srcX = p, nShard),
        # not the comment -- a swapped operand here silently misplaces the source
        # feature offset across ranks.
        ns = _render_fields_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_mul_i32", ns.srcX, [ns.p, ns.nShardS]), \
            "src_x must be s_mul_i32 srcX, p, nShard (operands, not comment)"

    def test_dst_y_folds_myrank_n_plus_srcy(self):
        # dst_y = myRank*N + j*MT1: a multiply (myRank*N into tmp) then an add
        # (tmp + src_y). Assert the real operands so a wrong factor/addend fails.
        ns = _render_fields_ns()
        lines = _lines(ns.text)
        assert _has_alu(lines, "s_mul_i32", ns.tmp, [ns.myRank, ns.nS]), \
            "expected s_mul_i32 tmp, myRank, N"
        assert _has_alu(lines, "s_add_u32", ns.dstY, [ns.tmp, ns.srcY]), \
            "expected s_add_u32 dstY, tmp(myRank*N), srcY(j*MT1)"

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

    def test_flag_addr_stride_is_myrank_times_8_64bit(self):
        # flag addr = flag_ptr[p] + myRank*8, a 64-bit add (lo add + hi carry).
        # Stride is 8 (u64 flag slots): the ATOMIC is an ADD64 (8-byte write), so
        # a *4 u32 stride would heap-overrun myRank=3's write. See the corrected
        # plan §1.3 and the T7 dependency note on emitComputeFlagAddr.
        lines = _lines(_render_flag_addr())
        assert any("s_lshl_b32" in _code(ln) and ", 3" in _code(ln) for ln in lines), \
            "expected myRank*8 (shift left 3, u64 flag-slot stride)"
        # A *4 (shift 2) stride must NOT reappear -- that was the overrun defect.
        assert not any("s_lshl_b32" in _code(ln) and ", 2" in _code(ln) for ln in lines), \
            "flag stride must be *8, not the defective *4"
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
