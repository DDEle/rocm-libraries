#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused GEMM.A2A kernarg segment offset contract.
#
# The layout must stay byte-identical between the kernel side
# (Tensile/Components/Signature.py fusedA2AKernArgLayout) and the host side
# (client/src/FusedA2AClient.cpp appendFusedSegment); both are pinned against
# the same golden table here and in tests/FusedA2AKernArg_test.cpp (C++).
#
# Also guards: every pointer arg (peer_ptr_0..7, counter_ptr,
# FusedSdmaQueues) sits contiguously at an 8-aligned offset ahead of every
# scalar arg.
################################################################################

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

import Tensile.Components.Signature as sig  # noqa: E402


# Byte offsets are relative to the segment base (peer_ptr_0 == 0).
GOLDEN_LAYOUT = {
    # 8 peer block pointers (8B each): 0..56. Slot j is peer j's block base;
    # flag sits at offset 0 of that block, recv at FUSED_A2A_PEER_RECV_OFFSET.
    **{"peer_ptr_%u" % j: 8 * j for j in range(8)},
    "counter_ptr":       64,    # 8B
    "FusedSdmaQueues":   72,    # 8B pointer
    "FusedMyRank":       80,    # 4B
    "FusedW":            84,    # 4B
    "FusedNShard":       88,    # 4B
    "FusedDrain":        92,    # 4B
    "FusedAM":           96,    # 4B
    "FusedTilesPerRank": 100,   # 4B
    "FusedTokenTiles":   104,   # 4B
}
GOLDEN_SEGMENT_BYTES = 108

# Per-arg byte sizes: pointers 8B, u32 scalars 4B.
_POINTER_ARGS = (
    ["peer_ptr_%u" % j for j in range(8)]
    + ["counter_ptr", "FusedSdmaQueues"]
)


def _arg_size(name):
    return 8 if name in _POINTER_ARGS else 4


def test_layout_matches_golden():
    """Every key/value of fusedA2AKernArgLayout() equals the golden table."""
    layout = sig.fusedA2AKernArgLayout()
    assert layout == GOLDEN_LAYOUT


def test_segment_bytes_is_108():
    assert sig.FUSED_A2A_SEGMENT_BYTES == GOLDEN_SEGMENT_BYTES


def test_segment_is_tightly_packed():
    """The last arg ends exactly at SEGMENT_BYTES (no trailing padding, no gap)."""
    layout = sig.fusedA2AKernArgLayout()
    end = max(off + _arg_size(name) for name, off in layout.items())
    assert end == sig.FUSED_A2A_SEGMENT_BYTES


def test_pointers_are_contiguous_and_8_aligned():
    """Every pointer arg sits at an 8-aligned offset, before every scalar."""
    layout = sig.fusedA2AKernArgLayout()
    ptrOffs = sorted(layout[n] for n in _POINTER_ARGS)
    assert all(o % 8 == 0 for o in ptrOffs), ptrOffs
    assert ptrOffs == list(range(0, 8 * len(_POINTER_ARGS), 8))
    firstScalar = min(o for n, o in layout.items() if n not in _POINTER_ARGS)
    assert max(ptrOffs) + 8 == firstScalar


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
