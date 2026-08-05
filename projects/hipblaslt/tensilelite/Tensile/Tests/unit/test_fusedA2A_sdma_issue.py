#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import shutil
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
from rocisa.label import LabelManager                              # noqa: E402
from rocisa.code import Module, TextBlock                          # noqa: E402
from Tensile.Common.Architectures import gfxToIsa                  # noqa: E402
import Tensile.Component                                           # noqa: E402,F401  MUST precede the next import (circular-import guard)
from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter  # noqa: E402

_GFX = "gfx950"


def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _code(ln):
    return ln.split("//")[0]


def _render():
    ri = rocIsa.getInstance()
    isa = gfxToIsa(_GFX)
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, 64)

    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.labels = LabelManager()
    w.vgprPool.checkOut(1)   # v0 reserved (Serial)
    w.sgprPool.checkOut(8)   # reserve s0..s7
    w.sgprs = {"AddressD": w.sgprPool.checkOutAligned(2, 2, "AddressD", preventOverflow=False)}
    indexChars = [str(i) for i in range(8)]
    indexChars[1] = "1J"                      # matches KernelWriter's "1"+INDEX_CHARS[1]
    w.states = SimpleNamespace(fusedA2AKernArgBase=0, indexChars=indexChars)
    w.argLoader = SimpleNamespace(
        loadKernArg=lambda *a, **k: TextBlock("// loadKernArg stub\n"))

    gwb = object.__new__(GlobalWriteBatchWriter)
    gwb.kernel = {"MacroTile0": 256, "MacroTile1": 256, "PackedC1IndicesX": [1]}
    gwb.parentWriter = w

    dstRank  = w.sgprPool.checkOut(1, "dstRank",  preventOverflow=False)
    myRank   = w.sgprPool.checkOut(1, "myRank",   preventOverflow=False)
    nShard   = w.sgprPool.checkOut(1, "nShard",   preventOverflow=False)
    flagBase = w.sgprPool.checkOutAligned(2, 2, "flagBase", preventOverflow=False)
    tmp      = w.sgprPool.checkOut(2, "tmp", preventOverflow=False)   # MUST be 2 wide

    m = Module("sdmaIssue")
    gwb._emitFusedA2ASdmaIssue(m, dstRank, myRank, nShard, flagBase, tmp)
    return _lines(str(m))


def test_src_pitch_is_the_D_token_stride_not_the_M_extent():
    lines = _render()
    # NEGATIVE -- this is the assertion that pins the bug (RED on HEAD).
    assert not any("SUBWIN DW4" in ln and "s[sgprSizesFree+0]" in _code(ln)
                   for ln in lines), \
        "src_pitch must not be the M extent (SizesFree+0); it must be D's token-axis stride"
    # POSITIVE -- DW4 is sourced from the stride sgpr.
    assert any("s_sub_u32" in _code(ln) and "s[sgprStrideD1J]" in _code(ln)
               and "SUBWIN DW4" in ln for ln in lines), \
        "expected DW4 src_pitch-1 to subtract from s[sgprStrideD1J]"


def test_src_slice_still_uses_M_times_N():
    # Anti-regression: emitComputeCopyFields' mS must NOT be rewritten.
    lines = _render()
    assert any("s_mul_i32" in _code(ln) and "s[sgprSizesFree+0]" in _code(ln)
               and "s[sgprSizesFree+1]" in _code(ln) for ln in lines), \
        "src_slice = M * N must still multiply SizesFree+0 by SizesFree+1"
