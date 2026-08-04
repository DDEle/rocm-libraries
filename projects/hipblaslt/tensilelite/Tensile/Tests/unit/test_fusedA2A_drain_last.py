#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# D15 Step 1 -- DRAIN ownership moves from W per-peer spinners to the single
# globally-last workgroup, and the poll becomes one vector load reduced with
# VCCZ. See notes/ROCM-27524/d15-drain-last-and-wg-remap-design.md.
################################################################################

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)


def test_drain_owner_is_a_validated_solution_parameter():
    from Tensile.Common.ValidParameters import validParameters

    assert validParameters["FusedA2ADrainOwner"] == [0, 1]


def test_drain_owner_defaults_to_the_current_per_peer_behaviour():
    from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters

    defaults = {k: v for d in defaultBenchmarkCommonParameters for k, v in d.items()}
    assert defaults["FusedA2ADrainOwner"] == [0], \
        "default must reproduce today's behaviour so it is a clean regression baseline"


def test_drain_owner_is_distinct_from_the_vestigial_FusedA2ADrain():
    # FusedA2ADrain exists but is never read by codegen (the runtime switch is
    # the FusedDrain kernarg). Guard against someone collapsing the two.
    from Tensile.Common.ValidParameters import validParameters

    assert "FusedA2ADrain" in validParameters
    assert "FusedA2ADrainOwner" in validParameters
