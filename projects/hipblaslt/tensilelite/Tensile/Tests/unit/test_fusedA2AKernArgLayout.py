#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused GEMM.A2A kernarg segment offset contract (Task 3).
#
# The fused-A2A kernarg segment layout is defined in TWO places that MUST stay
# byte-identical: the kernel side (Tensile/Components/Signature.py
# fusedA2AKernArgLayout + the addArg sequence) and the host side
# (client/src/FusedA2AClient.cpp appendFusedSegment). There is no cross-language
# test harness, so both sides are pinned against the SAME hardcoded golden table
# here (Python) and in tests/FusedA2AKernArg_test.cpp (C++): a one-sided change
# to either side reddens its own golden test.
#
# This test also guards the append-only invariant (Global Constraint 2 of the
# SDMA codegen plan): the three SDMA args (FusedSdmaQueues / FusedTilesPerRank /
# FusedTokenTiles) are appended at the very END, so every preceding offset is
# unchanged from earlier tasks.
################################################################################

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Signature.py imports rocisa at module load; skip cleanly if the C++ module is
# not built in this environment rather than erroring at collection time.
sig = pytest.importorskip("Tensile.Components.Signature")


# The single source of truth for this test. Any offset/size change on either the
# Python or the C++ side must be reflected here, which is exactly what makes a
# one-sided drift fail. Byte offsets are relative to the segment base
# (recv_ptr_0 == 0).
GOLDEN_LAYOUT = {
    # 8 recv pointers (8B each): 0..56
    **{"recv_ptr_%u" % j: 8 * j for j in range(8)},
    # 8 flag pointers (8B each): 64..120
    **{"flag_ptr_%u" % j: 64 + 8 * j for j in range(8)},
    "counter_ptr":       128,   # 8B
    "FusedMyRank":       136,   # 4B
    "FusedTarget":       140,   # 4B (deprecated slot, retained)
    "FusedW":            144,   # 4B
    "FusedNShard":       148,   # 4B
    "FusedDrain":        152,   # 4B
    "FusedAM":           156,   # 4B
    # SDMA offload args (Task 3), appended at the end.
    "FusedSdmaQueues":   160,   # 8B pointer (segment base 8-aligned => no padding)
    "FusedTilesPerRank": 168,   # 4B
    "FusedTokenTiles":   172,   # 4B
}
GOLDEN_SEGMENT_BYTES = 176

# Per-arg byte sizes, needed to check the tight-packing invariant
# (max(offset)+size == SEGMENT_BYTES). Pointers 8B, u32 scalars 4B.
_POINTER_ARGS = (
    ["recv_ptr_%u" % j for j in range(8)]
    + ["flag_ptr_%u" % j for j in range(8)]
    + ["counter_ptr", "FusedSdmaQueues"]
)


def _arg_size(name):
    return 8 if name in _POINTER_ARGS else 4


def test_layout_matches_golden():
    """Every key/value of fusedA2AKernArgLayout() equals the golden table."""
    layout = sig.fusedA2AKernArgLayout()
    assert layout == GOLDEN_LAYOUT


def test_segment_bytes_is_176():
    assert sig.FUSED_A2A_SEGMENT_BYTES == GOLDEN_SEGMENT_BYTES


def test_segment_is_tightly_packed():
    """The last arg ends exactly at SEGMENT_BYTES (no trailing padding, no gap)."""
    layout = sig.fusedA2AKernArgLayout()
    end = max(off + _arg_size(name) for name, off in layout.items())
    assert end == sig.FUSED_A2A_SEGMENT_BYTES


def test_sdma_args_appended_last():
    """The three SDMA args sit strictly after every legacy arg (append-only)."""
    layout = sig.fusedA2AKernArgLayout()
    sdma = ("FusedSdmaQueues", "FusedTilesPerRank", "FusedTokenTiles")
    legacy_max = max(off for name, off in layout.items() if name not in sdma)
    assert all(layout[name] > legacy_max for name in sdma)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
