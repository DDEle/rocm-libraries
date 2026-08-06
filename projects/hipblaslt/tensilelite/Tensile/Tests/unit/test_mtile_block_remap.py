#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# MTileBlockWidth: the segment-first remap with A pinned at compile time, for
# kernels with no A2A to derive it from. The point of the parameter is that the
# permutation is the SAME one the fused path uses, so the reorder benefit measured
# on the fused line can be attributed to the dispatch order alone -- which only
# holds if the two emitters really do lower to the same arithmetic. That is what
# most of this file checks (ROCM-27524, D15 Step 3).
################################################################################

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

_N0, _N1 = 72, 16


def _render(emitter, **kernelKw):
    """Render one remap emitter against a stub writer; -> assembly text."""
    import shutil
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from rocisa.code import TextBlock
    from Tensile.Common.Architectures import gfxToIsa
    import Tensile.Component  # noqa: F401  MUST precede the next import
    import Tensile.Components.WorkGroupMappingAlgos as W

    def loadKernArgEcho(*a, **k):
        fields = [str(x) for x in a] + ["%s=%s" % (n, k[n]) for n in sorted(k)]
        return TextBlock("// loadKernArg %s\n" % " ".join(fields))

    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, 64)

    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.vgprPool.checkOut(1)
    w.sgprPool.checkOut(8)
    w.states = SimpleNamespace(fusedA2AKernArgBase=0)
    w.argLoader = SimpleNamespace(loadKernArg=loadKernArgEcho)

    kernel = {"MacroTile0": 256, "WavefrontSize": 64, "FusedGemmA2A": 0,
              "MTileBlockWidth": 0}
    kernel.update(kernelKw)
    return str(getattr(W, emitter)(w, kernel))


def _code(text):
    """Strip comments and blanks; -> list of instruction lines.

    Drops both trailing `//` comments and whole-line `/* */` banners (addComment1),
    so the comparisons below are about emitted instructions and not about prose.
    """
    out = []
    for line in text.splitlines():
        line = line.split("//")[0].strip()
        if not line or line.startswith("/*"):
            continue
        out.append(line)
    return out


def _opcodes(text):
    return [ln.split()[0] for ln in _code(text)]


def test_emits_nothing_when_the_parameter_is_off():
    """Zero regression surface: every kernel that does not ask for it is untouched."""
    assert _code(_render("MTileBlockRemap", MTileBlockWidth=0)) == []


def test_declines_when_the_kernel_is_fused():
    """The fused path owns the remap: A there must stay tied to FusedAM, or the grid
    splits at one boundary while the epilogue classifies PUSH/local at another."""
    assert _code(_render("MTileBlockRemap", MTileBlockWidth=36, FusedGemmA2A=1)) == []


def test_arithmetic_is_identical_to_the_fused_emitter():
    """The whole claim of this parameter in one assertion.

    Everything after A is in place must match the fused lowering instruction for
    instruction. If the two ever drift, the pure-GEMM measurement stops being
    evidence about the fused line's dispatch order and nothing else would say so --
    both emitters would still produce correct, bijective, plausible-looking code.
    """
    mine = _opcodes(_render("MTileBlockRemap", MTileBlockWidth=36))
    fused = _opcodes(_render("FusedA2AWgRemap", FusedGemmA2A=1))

    # Prologues differ by construction: kernarg load + shift vs mov + clamp.
    assert mine[:2] == ["s_mov_b32", "s_min_u32"]
    assert fused[:2] == ["s_waitcnt", "s_lshr_b32"]
    assert mine[2:] == fused[2:], "lowering drifted from the fused emitter"


def test_clamps_the_width_against_the_runtime_grid():
    """R comes from the yaml and nothing has compared it to N0, which is a runtime
    value. Unclamped, R > N0 underflows L = N0-A and the map stops being a bijection
    -- work-groups collide and tiles go uncomputed, with correct-looking output for
    every tile that does get computed. One instruction closes it."""
    code = _code(_render("MTileBlockRemap", MTileBlockWidth=36))
    mins = [ln for ln in code if ln.startswith("s_min_u32")]
    assert len(mins) == 1, code
    assert "NumWorkGroups0" in mins[0]
    # ...and it must precede the subtraction it protects.
    assert code.index(mins[0]) < min(i for i, ln in enumerate(code)
                                     if ln.startswith("s_sub_u32"))


def test_no_branch():
    """Straight-line, same as the fused path: the two cases are selected with
    s_cselect_b32, and the divide corrects itself by writing EXEC, not by branching."""
    code = _code(_render("MTileBlockRemap", MTileBlockWidth=36))
    assert not [ln for ln in code if ln.startswith("s_cbranch") or ln.startswith("s_branch")]


@pytest.mark.parametrize("R", [4, 28, 36, 40, 71])
def test_width_is_baked_as_an_immediate(R):
    """A is compile-time here; a kernarg read would defeat the point (and there is
    no A2A kernarg in a non-fused kernel to read)."""
    text = _render("MTileBlockRemap", MTileBlockWidth=R)
    code = _code(text)
    assert any(ln.startswith("s_mov_b32") and hex(R) in ln for ln in code), code
    assert "loadKernArg" not in text


@pytest.mark.parametrize("R", [4, 28, 36, 40])
def test_matches_the_reference_model_the_fused_path_is_tested_on(R):
    """The permutation this parameter selects is fusedA2AWgRemapIndex(A=R) -- the
    same function whose bijectivity is property-tested for the fused line. Pinned
    here so the pure-GEMM numbers can be laid next to the fused ones by A."""
    from Tensile.Components.WorkGroupMappingAlgos import fusedA2AWgRemapIndex as f
    seen = [f(t % _N0, t // _N0, _N0, _N1, R) for t in range(_N0 * _N1)]
    assert len(set(seen)) == _N0 * _N1
    assert set(seen) == {(m, j) for m in range(_N0) for j in range(_N1)}
    # m-inner with run R at the front -- the property WorkGroupMapping cannot express.
    assert seen[:R] == [(m, 0) for m in range(R)]
    assert seen[R] == (0, 1)


def test_parameter_is_plumbed_end_to_end():
    """Three separate registries must all know the name, and only one of them fails
    loudly. Missing from validParameters, a yaml using it is rejected (fine).
    Missing from the defaults, the value is dropped in silence and codegen raises
    KeyError. Missing from the Min name set, every value produces the SAME kernel
    filename, so a sweep reports one kernel's timing several times over."""
    from Tensile.Common.ValidParameters import validParameters
    from Tensile.Common.GlobalParameters import defaultSolution
    from Tensile.Common.RequiredParameters import getRequiredParametersMin

    assert 36 in validParameters["MTileBlockWidth"]
    assert 0 in validParameters["MTileBlockWidth"]
    assert defaultSolution["MTileBlockWidth"] == 0, "default must be off"
    assert "MTileBlockWidth" in getRequiredParametersMin()
