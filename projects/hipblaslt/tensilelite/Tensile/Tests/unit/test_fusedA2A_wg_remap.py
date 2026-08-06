#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused-A2A segment-first workgroup remap: the PUSH segment (wg0 < AM_tiles) is
# lifted out of its 16 per-token-tile bands and laid down as one run at the front
# of the grid, so the last PUSH work-group's dispatch index falls from 97.1% to
# 55.5% and the SDMA tail gets ~355 us of head start (ROCM-27524, D15 Step 2).
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
    """The eight points the design pins by hand (design section 2.5)."""
    f = _remap()
    want = {0: (0, 0), 39: (39, 0), 40: (0, 1), 639: (39, 15),
            640: (40, 0), 671: (71, 0), 672: (40, 1), 1151: (71, 15)}
    for t, expect in want.items():
        wg0, wg1 = t % _N0, t // _N0
        assert f(wg0, wg1, _N0, _N1, _A) == expect, f"t={t}"


def test_is_a_bijection_on_the_champion_grid():
    """Bijectivity is what leaves the counter/DRAIN election untouched.

    Every (dst_rank, j) bucket must still receive exactly tilesPerRank increments
    and counter3's FusedTotalWGs is order-independent -- both follow from this and
    from nothing else.
    """
    f = _remap()
    seen = {f(t % _N0, t // _N0, _N0, _N1, _A) for t in range(_N0 * _N1)}
    assert len(seen) == _N0 * _N1
    assert seen == {(m, j) for m in range(_N0) for j in range(_N1)}


@pytest.mark.parametrize("n0,n1,a", [
    (72, 16, 40), (72, 16, 1), (72, 16, 71), (5, 3, 2), (1, 7, 0), (9, 1, 4),
])
def test_range_is_closed_inside_the_precondition(n0, n1, a):
    """m < n0 and j < n1 for every input, given the precondition a <= n0.

    Bijectivity alone does NOT catch an out-of-range image: a map whose values
    escape [0,n0) can still be injective on its own image. This is the assertion
    that would fire if the formula ever started producing an M-tile index past the
    end of D, so it cannot be folded into the bijection test.
    """
    f = _remap()
    for t in range(n0 * n1):
        m, j = f(t % n0, t // n0, n0, n1, a)
        assert 0 <= m < n0 and 0 <= j < n1, f"t={t} -> ({m},{j})"


@pytest.mark.parametrize("a", [0, _N0])
def test_degenerate_am_tiles_fall_back_to_identity(a):
    """A=0 (no PUSH region) and A=N0 (no local region) are identities.

    Not defensive padding -- they drop out of the formula. A=0 forces the local
    branch with L=N0; A=N0 forces the PUSH branch with divisor N0. Both are the
    reason no guard is emitted (design section 3, properties 2 and 3).
    """
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
    """The inverse in design section 2.3 is what the tests and any future host-side
    reasoning use to go from (m, j) back to a dispatch index."""
    f = _remap()
    S = _A * _N1
    L = _N0 - _A
    for t in range(_N0 * _N1):
        m, j = f(t % _N0, t // _N0, _N0, _N1, _A)
        back = j * _A + m if m < _A else S + j * L + (m - _A)
        assert back == t, f"t={t} -> ({m},{j}) -> {back}"


def test_the_last_push_workgroup_moves_to_the_front():
    """The whole point, stated as an assertion.

    Before: the PUSH segment is split across 16 token-tile bands, so the last PUSH
    work-group sits at (N1-1)*N0 + A - 1 = 1119 of 1152 (97.1%).  After: it sits at
    A*N1 - 1 = 639 (55.5%).  That 42-point shift is the ~355 us of head start the
    SDMA tail gets, and every downstream number in the design rests on it.
    """
    f = _remap()
    before = max(t for t in range(_N0 * _N1) if (t % _N0) < _A)
    after = max(t for t in range(_N0 * _N1) if f(t % _N0, t // _N0, _N0, _N1, _A)[0] < _A)
    assert before == 1119
    assert after == 639
