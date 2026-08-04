#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# D15 Step 1 -- DRAIN ownership moves from W per-peer spinners to the single
# globally-last workgroup, and the poll becomes one vector load reduced with
# VCCZ (ROCM-27524, deferred item D15 Step 1).
################################################################################

import ast
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

    # Assert on the comment-stripped instruction, never on `text`: the
    # instruction's own comment reads "FusedTotalWGs = NumWorkGroups0 *
    # NumWorkGroups1", so every name below is satisfied by the comment alone and
    # a `in text` check stays green even with the operands wired wrong.
    code = [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]
    # exactly one instruction -- this is a hot-path prologue, not a place to grow
    assert len(code) == 1, code
    assert code[0].startswith("s_mul_i32 "), code[0]
    # by position, so a dst/src swap is caught too
    dst, src0, src1 = (op.strip() for op in code[0].split(None, 1)[1].split(","))
    assert "sgprFusedTotalWGs" in dst, code[0]
    assert "sgprNumWorkGroups0" in src0, code[0]
    assert "sgprNumWorkGroups1" in src1, code[0]


def test_drain_owner_also_locks_out_the_runtime_GSU_override():
    # Rejecting GlobalSplitU != 1 only pins the compile-time value. SupportUserGSU
    # defaults to True, and ContractionSolution.cpp honours problem.getParams().gsu()
    # when it is set -- so a runtime caller could re-inflate the grid under an
    # election target that was latched as a compile-time constant. Structural pin
    # (no toolchain needed): the lockout must live in the FusedGemmA2A block, be
    # guarded on FusedA2ADrainOwner, and sit on the ACCEPTED path -- after the
    # rejects, never nested inside one, or it never runs for the config it protects.
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())

    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    body = blocks[0].body

    TARGET = "state['InternalSupportParams']['SupportUserGSU'] = False"

    def index_of(pred):
        return next((i for i, stmt in enumerate(body) if pred(stmt)), None)

    lockout = index_of(lambda s: isinstance(s, ast.If)
                       and "FusedA2ADrainOwner" in ast.unparse(s.test)
                       and any(ast.unparse(b) == TARGET for b in s.body))
    assert lockout is not None, \
        "FusedGemmA2A block has no `if ...FusedA2ADrainOwner...:` setting " + TARGET

    gsu_reject = index_of(lambda s: isinstance(s, ast.If)
                          and "FusedA2ADrainOwner" in ast.unparse(s.test)
                          and "GlobalSplitU" in ast.unparse(s.test))
    assert gsu_reject is not None, "the compile-time GlobalSplitU rejection went missing"
    # Nesting the lockout inside the reject guard collapses the two indices: that
    # branch returns, so the flag would only ever be set on a dead solution.
    assert lockout > gsu_reject, \
        "lockout must be a sibling AFTER the GSU rejection, not inside/before it"

    # A top-level return before it would make it unreachable outright.
    assert not any(isinstance(s, (ast.Return, ast.Raise)) for s in body[:lockout]), \
        "a statement before the lockout leaves the block: the lockout is dead code"


def test_total_wgs_latch_is_gated_on_the_owner_knob_not_the_vestigial_one():
    # Every site of this feature must branch on FusedA2ADrainOwner (WHO drains).
    # Swapping in the similarly named but vestigial FusedA2ADrain -- which defaults
    # to 1 and is never read by codegen -- would allocate the SGPR, emit the latch
    # and reject batched problems for every fused kernel, decoupled from the
    # election Task 4 gates on the same knob.
    sites = (("Tensile/KernelWriter.py", 'defineSgpr("FusedTotalWGs"'),
             ("Tensile/KernelWriterAssembly.py", "emitFusedA2ATotalWGsLatch(module"),
             ("Tensile/SolutionStructs/Solution.py",
              '"FusedA2ADrainOwner=1 requires no batch dim'),
             ("Tensile/SolutionStructs/Solution.py",
              '"FusedA2ADrainOwner=1 requires GlobalSplitU=1'))
    for relpath, marker in sites:
        with open(os.path.join(TENSILE_ROOT, relpath)) as f:
            lines = f.read().splitlines()
        hits = [i for i, ln in enumerate(lines) if marker in ln]
        assert hits, f"{relpath}: no call site containing {marker!r}"
        for i in hits:
            guard = next((ln for ln in reversed(lines[:i])
                          if ln.lstrip().startswith("if ")), None)
            assert guard is not None, \
                f"{relpath}:{i + 1}: no preceding `if` for site {marker!r}"
            assert '"FusedA2ADrainOwner"]' in guard, (relpath, i + 1, guard)
            assert '"FusedA2ADrain"]' not in guard, (relpath, i + 1, guard)
