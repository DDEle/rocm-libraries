#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused-A2A segment-first workgroup remap: lifts the PUSH segment (wg0 < AM_tiles)
# out of its 16 per-token-tile bands into one contiguous run at the front of the
# grid.
################################################################################

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Champion shape: M=18432 N=4096, MT=256x256, AM=10240 => 72 x 16 grid, A=40.
_N0, _N1, _A = 72, 16, 40


def _remap():
    from Tensile.Components.WorkGroupMappingAlgos import fusedA2AWgRemapIndex
    return fusedA2AWgRemapIndex


def test_champion_shape_matches_the_design_table():
    """The eight points the design pins by hand."""
    f = _remap()
    want = {0: (0, 0), 39: (39, 0), 40: (0, 1), 639: (39, 15),
            640: (40, 0), 671: (71, 0), 672: (40, 1), 1151: (71, 15)}
    for t, expect in want.items():
        wg0, wg1 = t % _N0, t // _N0
        assert f(wg0, wg1, _N0, _N1, _A) == expect, f"t={t}"


def test_is_a_bijection_on_the_champion_grid():
    """The remap is a bijection over the whole grid."""
    f = _remap()
    seen = {f(t % _N0, t // _N0, _N0, _N1, _A) for t in range(_N0 * _N1)}
    assert len(seen) == _N0 * _N1
    assert seen == {(m, j) for m in range(_N0) for j in range(_N1)}


@pytest.mark.parametrize("n0,n1,a", [
    (72, 16, 40), (72, 16, 1), (72, 16, 71), (5, 3, 2), (1, 7, 0), (9, 1, 4),
])
def test_range_is_closed_inside_the_precondition(n0, n1, a):
    """m < n0 and j < n1 for every input, given the precondition a <= n0."""
    f = _remap()
    for t in range(n0 * n1):
        m, j = f(t % n0, t // n0, n0, n1, a)
        assert 0 <= m < n0 and 0 <= j < n1, f"t={t} -> ({m},{j})"


@pytest.mark.parametrize("a", [0, _N0])
def test_degenerate_am_tiles_fall_back_to_identity(a):
    """A=0 (no PUSH region) and A=N0 (no local region) are identities."""
    f = _remap()
    for t in range(_N0 * _N1):
        wg0, wg1 = t % _N0, t // _N0
        assert f(wg0, wg1, _N0, _N1, a) == (wg0, wg1), f"t={t} A={a}"


def test_single_token_tile_is_an_identity_for_every_am_tiles():
    """N1 == 1 leaves nothing to reorder, and the formula knows it."""
    f = _remap()
    for a in range(_N0 + 1):
        for wg0 in range(_N0):
            assert f(wg0, 0, _N0, 1, a) == (wg0, 0), f"wg0={wg0} A={a}"


def test_inverse_round_trips():
    """The inverse formula round-trips (m, j) back to the dispatch index t."""
    f = _remap()
    S = _A * _N1
    L = _N0 - _A
    for t in range(_N0 * _N1):
        m, j = f(t % _N0, t // _N0, _N0, _N1, _A)
        back = j * _A + m if m < _A else S + j * L + (m - _A)
        assert back == t, f"t={t} -> ({m},{j}) -> {back}"


def test_the_last_push_workgroup_moves_to_the_front():
    """The last PUSH work-group moves from near the end of the grid to just past its midpoint."""
    f = _remap()
    before = max(t for t in range(_N0 * _N1) if (t % _N0) < _A)
    after = max(t for t in range(_N0 * _N1) if f(t % _N0, t // _N0, _N0, _N1, _A)[0] < _A)
    assert before == 1119
    assert after == 639


def _renderRemap(wavefrontSize: int = 64, fused: int = 1) -> str:
    """Render FusedA2AWgRemap standalone.

    The argLoader echoes its arguments rather than returning a fixed string.
    """
    import shutil
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from rocisa.code import TextBlock
    from Tensile.Common.Architectures import gfxToIsa
    import Tensile.Component  # noqa: F401  MUST precede the next import
    from Tensile.Components.WorkGroupMappingAlgos import FusedA2AWgRemap

    def loadKernArgEcho(*a, **k):
        fields = [str(x) for x in a] + ["%s=%s" % (n, k[n]) for n in sorted(k)]
        return TextBlock("// loadKernArg %s\n" % " ".join(fields))

    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, wavefrontSize)

    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.vgprPool.checkOut(1)
    w.sgprPool.checkOut(8)
    w.states = SimpleNamespace(fusedA2AKernArgBase=0)
    w.argLoader = SimpleNamespace(loadKernArg=loadKernArgEcho)

    kernel = {"MacroTile0": 256, "WavefrontSize": wavefrontSize, "FusedGemmA2A": fused}
    return str(FusedA2AWgRemap(w, kernel))


def _remapCode(text):
    return [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]


def test_emits_nothing_when_the_kernel_is_not_fused():
    """A non-fused kernel emits nothing."""
    assert _remapCode(_renderRemap(fused=0)) == []


def test_emitted_block_has_no_branch():
    """The emitted code contains no branch instruction."""
    code = _remapCode(_renderRemap())
    assert not [ln for ln in code if ln.startswith("s_cbranch")], code


def test_divides_exactly_once():
    """One divide, shared by both cases via the cselects."""
    code = _remapCode(_renderRemap())
    assert len([ln for ln in code if ln.startswith("v_rcp_f64")]) == 1, code


def test_scc_is_not_clobbered_between_the_compare_and_the_selects():
    """The three s_cselect_b32 selects must immediately follow the s_cmp_lt_u32
    compare, with no SCC writer in between."""
    code = _remapCode(_renderRemap())
    cmp_i = next(i for i, ln in enumerate(code) if ln.startswith("s_cmp_lt_u32"))
    sel = [i for i, ln in enumerate(code) if ln.startswith("s_cselect_b32")]
    assert len(sel) == 3, f"expected three selects (dividend, divisor, base), got {sel}"
    assert sel == [cmp_i + 1, cmp_i + 2, cmp_i + 3], \
        f"selects must immediately follow the compare: cmp={cmp_i} sel={sel}"


def test_reads_fused_am_and_shifts_by_log2_macrotile0():
    """A = FusedAM >> log2(MT0), the same expression the epilogue's PUSH gate uses."""
    from Tensile.Components.Signature import fusedA2AKernArgLayout
    text = _renderRemap()
    want = hex(fusedA2AKernArgLayout()["FusedAM"])
    assert want in text, f"FusedAM ({want}) is not the kernarg this reads:\n{text}"
    code = _remapCode(text)
    shifts = [ln for ln in code if ln.startswith("s_lshr_b32")]
    assert len(shifts) == 1, shifts
    assert shifts[0].rstrip().endswith("8"), f"MT0=256 => shift by 8, got {shifts[0]}"


def test_writes_both_workgroup_registers():
    """m -> WorkGroup0 and j -> WorkGroup1."""
    code = _remapCode(_renderRemap())
    assert any("sgprWorkGroup0" in ln for ln in code), code
    assert any("sgprWorkGroup1" in ln for ln in code), code
