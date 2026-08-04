#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# D15 Step 1 -- DRAIN ownership moves from W per-peer spinners to the single
# globally-last workgroup, and the poll becomes one vector load reduced with
# VCCZ (ROCM-27524, deferred item D15 Step 1).
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
    # the FusedDrain kernarg). Guard against someone collapsing the two: their
    # defaults disagree, so a merged knob cannot preserve both.
    from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters

    defaults = {k: v for d in defaultBenchmarkCommonParameters for k, v in d.items()}
    assert defaults["FusedA2ADrain"] == [1], "vestigial knob keeps its historical default"
    assert defaults["FusedA2ADrainOwner"] == [0]
    assert defaults["FusedA2ADrain"] != defaults["FusedA2ADrainOwner"], \
        "two distinct knobs: WHETHER (vestigial) vs WHO (drain owner)"


def test_total_wgs_latch_multiplies_the_two_grid_dims():
    pytest.importorskip("rocisa")
    from rocisa.code import Module
    from Tensile.Components.GlobalWriteBatch import emitFusedA2ATotalWGsLatch

    m = Module("latch")
    emitFusedA2ATotalWGsLatch(m, "FusedTotalWGs")
    text = str(m)

    assert "s_mul_i32" in text
    assert "NumWorkGroups0" in text and "NumWorkGroups1" in text
    assert "FusedTotalWGs" in text
    # exactly one instruction -- this is a hot-path prologue, not a place to grow
    code = [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]
    assert len(code) == 1, code


def test_total_wgs_latch_is_gated_on_the_owner_knob_not_the_vestigial_one():
    # Every site of this feature must branch on FusedA2ADrainOwner (WHO drains).
    # Swapping in the similarly named but vestigial FusedA2ADrain -- which defaults
    # to 1 and is never read by codegen -- would allocate the SGPR, emit the latch
    # and reject batched problems for every fused kernel, decoupled from the
    # election Task 4 gates on the same knob.
    sites = (("Tensile/KernelWriter.py", 'defineSgpr("FusedTotalWGs"'),
             ("Tensile/KernelWriterAssembly.py", "emitFusedA2ATotalWGsLatch(module"),
             ("Tensile/SolutionStructs/Solution.py",
              '"FusedA2ADrainOwner=1 requires no batch dim'))
    for relpath, marker in sites:
        with open(os.path.join(TENSILE_ROOT, relpath)) as f:
            lines = f.read().splitlines()
        hits = [i for i, ln in enumerate(lines) if marker in ln]
        assert hits, f"{relpath}: no call site containing {marker!r}"
        for i in hits:
            guard = next(ln for ln in reversed(lines[:i]) if ln.lstrip().startswith("if "))
            assert '"FusedA2ADrainOwner"]' in guard, (relpath, i + 1, guard)
            assert '"FusedA2ADrain"]' not in guard, (relpath, i + 1, guard)
