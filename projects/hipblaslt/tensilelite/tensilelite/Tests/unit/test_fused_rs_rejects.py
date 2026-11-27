# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Problem, solution and predicate guards for ``ProblemType["FusedA2A"] = "GemmRS"``."""

import pytest

import tensilelite.Contractions as C
from tensilelite.Components.WorkGroupMappingAlgos import ClusterAIDRemap
from tensilelite.SolutionStructs.Naming import getKernelNameMin
from tensilelite.SolutionStructs.Problem import ProblemSizes, ProblemType, isFusedPush
from test_fused_a2a_rejects import _GFX1250, _derive, _gp_gfx942  # noqa: F401

pytestmark = pytest.mark.unit

_RS_PT = {"UseBeta": False}


def _derive_rs(request, assembler, capsys, arch="gfx1250", mode="GemmRS", **overrides):
    iim = request.getfixturevalue(f"{arch}_iim")
    params = dict(_GFX1250) if arch == "gfx1250" else {"arch": arch, "PrefetchLocalRead": 1}
    if mode == "GemmRS":
        params["FusedRSWait"] = 0
    params.update(overrides)
    params["arch"] = arch
    params["ProblemType"] = {**_RS_PT, **overrides.get("ProblemType", {})}
    return _derive(iim, assembler, capsys, mode=mode, **params)


@pytest.mark.parametrize("mode, expected", [
    ("GemmA2A", True), ("GemmRS", True), ("A2AGemm", False), ("Disabled", False)])
def test_is_fused_push(mode, expected):
    assert isFusedPush({"FusedA2A": mode}) is expected
    assert isFusedPush(_rsType(FusedA2A=mode)) is expected


def test_is_fused_push_without_problem_type():
    assert isFusedPush(None) is False
    assert isFusedPush({}) is False


@pytest.mark.parametrize("wait", [0, 1, 2])
@pytest.mark.parametrize("cluster", [[1, 1], [2, 2], [4, 2], [2, 4]])
def test_gemm_rs_gfx1250_is_accepted(_gp_gfx942, request, assembler, capsys, cluster, wait):
    sol, out = _derive_rs(request, assembler, capsys, ClusterDim=cluster, FusedRSWait=wait)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False
    assert sol["FusedRSWait"] == wait


@pytest.mark.parametrize("wait", [-1, 3])
def test_gemm_rs_rejects_unknown_fused_rs_wait(_gp_gfx942, request, assembler, capsys, wait):
    sol, out = _derive_rs(request, assembler, capsys, FusedRSWait=wait)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmRS requires FusedRSWait 0, 1 or 2" in out


@pytest.mark.parametrize("overrides, msg", [
    ({"InterleaveAlpha": 2}, "InterleaveAlpha != 2"),
    ({"CompactLoopStore": True}, "CompactLoopStore=False"),
    ({"StoreRemapVectorWidth": -1}, "StoreRemapVectorWidth=0"),
    ({"StoreRemapVectorWidth": 8}, "StoreRemapVectorWidth=0"),
])
@pytest.mark.parametrize("wait", [1, 2])
def test_gemm_rs_wait_rejects(_gp_gfx942, request, assembler, capsys, overrides, msg, wait):
    sol, out = _derive_rs(request, assembler, capsys, FusedRSWait=wait, **overrides)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmRS with FusedRSWait=%d requires %s" % (wait, msg) in out


def test_gemm_rs_wait_0_keeps_interleave_alpha_2(_gp_gfx942, request, assembler, capsys):
    sol, out = _derive_rs(request, assembler, capsys, InterleaveAlpha=2)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_gemm_rs_rejects_non_gfx1250(_gp_gfx942, request, assembler, capsys, arch):
    sol, out = _derive_rs(request, assembler, capsys, arch=arch)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmRS requires a gfx1250 ISA" in out


@pytest.mark.parametrize("overrides, msg", [
    ({"ClusterAIDMapping": -2, "ClusterDim": [4, 2]}, "ClusterAIDMapping=-1"),
    ({"ClusterAIDMapping": -1, "ClusterDim": [4, 2], "WorkGroupMapping": 2}, "WorkGroupMapping=1"),
    ({"WorkGroupMapping": 0}, "WorkGroupMapping=1"),
    ({"WorkGroupMapping": -1}, "WorkGroupMapping=1"),
    ({"WorkGroupMappingXCC": 2}, "WorkGroupMappingXCC=1"),
    ({"SpaceFillingAlgo": [0]}, "SpaceFillingAlgo=[]"),
    ({"UseSubtileImpl": True}, "UseSubtileImpl=0"),
    ({"ClusterDim": [3, 1]}, "a power-of-two ClusterDim"),
    ({"ClusterDim": [1, 3]}, "a power-of-two ClusterDim"),
    ({"ClusterDim": [3, 2]}, "a power-of-two ClusterDim"),
    ({"ProblemType": {"HighPrecisionAccumulate": False}}, "HighPrecisionAccumulate"),
    ({"ProblemType": {"DestDataType": "s"}}, "a bf16 D"),
])
def test_gemm_rs_rejects(_gp_gfx942, request, assembler, capsys, overrides, msg):
    sol, out = _derive_rs(request, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmRS requires " + msg in out


@pytest.mark.parametrize("mode", ["GemmA2A", "Disabled"])
def test_fused_rs_wait_rejected_without_gemm_rs(_gp_gfx942, request, assembler, capsys, mode):
    sol, out = _derive_rs(request, assembler, capsys, mode=mode, FusedRSWait=0)
    assert sol.get("Valid") is False
    assert "FusedRSWait != 1 requires FusedA2A=GemmRS" in out


def test_gemm_rs_rejects_beta():
    with pytest.raises(Exception, match="FusedA2A=GemmRS does not support .*UseBeta"):
        _rsType(UseBeta=True)


def test_gemm_rs_name_token():
    assert "_GemmRS" in str(_rsType())
    assert "_GemmRS" not in str(_rsType(FusedA2A="GemmA2A", UseBeta=True))


def test_gemm_rs_problem_type_predicates():
    cpt = C.ProblemType.FromOriginalState(_rsType().state)
    preds = {p.tag: p.value for p in cpt.predicates(True, True, True)}
    assert preds["FusedA2A"] == "GemmRS"
    assert "BatchSizeEqual" in preds


def test_fused_rs_wait_default_keeps_kernel_names(_gp_gfx942, request, assembler, capsys):
    a2a, out = _derive_rs(request, assembler, capsys, mode="GemmA2A")
    assert a2a.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    name = getKernelNameMin(a2a._state, False)
    del a2a._state["FusedRSWait"]
    assert getKernelNameMin(a2a._state, False) == name
    rs0, out = _derive_rs(request, assembler, capsys)
    assert rs0.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert "_FRSW0" in getKernelNameMin(rs0._state, False)
    assert "FRSW" not in getKernelNameMin(dict(rs0._state, FusedRSWait=1), False)
    assert "_FRSW2" in getKernelNameMin(dict(rs0._state, FusedRSWait=2), False)


# ---------------------------------------------------------------------------
# Exact problem sizes.
# ---------------------------------------------------------------------------
_RS_CONFIG = {
    "OperationType": "GEMM", "DataType": "b", "DestDataType": "b", "ComputeDataType": "s",
    "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
    "UseBeta": False, "Batched": True, "FusedA2A": "GemmRS",
}


def _rsType(**overrides):
    return ProblemType({**_RS_CONFIG, **overrides}, False)


def _sizes(*exacts):
    return ProblemSizes(_rsType(), [{"Exact": e} for e in exacts])


def test_extent_is_m():
    assert _sizes({"sizes": [8192, 16384, 1, 2048], "world": 4}).exacts[0].a2aExtent == 8192


def test_extent_equal_to_m_is_accepted():
    assert _sizes({"sizes": [8192, 16384, 1, 2048], "world": 4, "a2aExtent": 8192}).exacts[0].a2aExtent == 8192


@pytest.mark.parametrize("extent", [2048, 1])
def test_extent_other_than_m_is_an_error(extent):
    with pytest.raises(RuntimeError, match="a2aExtent=%d must equal M=8192" % extent):
        _sizes({"sizes": [8192, 16384, 1, 2048], "world": 4, "a2aExtent": extent})


def test_n_must_divide_by_world():
    with pytest.raises(RuntimeError, match="N=16386 is not a multiple of world=4"):
        _sizes({"sizes": [8192, 16386, 1, 2048], "world": 4})


def test_gemm_rs_requires_world():
    with pytest.raises(RuntimeError, match="must set 'world'"):
        _sizes({"sizes": [8192, 16384, 1, 2048]})


# ---------------------------------------------------------------------------
# GemmRSShape and the size mapping.
# ---------------------------------------------------------------------------
def _compound(sol):
    return C.ProblemPredicate.CompoundPredicates(
        dict(sol._state), C.ProblemType.FromOriginalState(sol["ProblemType"]))


@pytest.mark.parametrize("cluster", [[1, 1], [4, 2], [2, 4]])
def test_gemm_rs_shape_predicate(_gp_gfx942, request, assembler, capsys, cluster):
    sol, out = _derive_rs(request, assembler, capsys, ClusterDim=cluster,
                          mi=[16, 16, 32, 1, 1, 8, 4, 2, 2])
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert (sol["MacroTile0"], sol["MacroTile1"]) == (256, 128)
    preds = _compound(sol)
    assert [p.value for p in preds if p.tag == "GemmRSShape"] == [[256, 128, *cluster, 16, 16]]
    assert "GemmA2AShape" not in [p.tag for p in preds]


def test_gemm_rs_shape_takes_the_gemm_a2a_shape_position(_gp_gfx942, request, assembler, capsys):
    rs, out = _derive_rs(request, assembler, capsys)
    assert rs.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    a2a = dict(rs._state)
    a2a["ProblemType"] = dict(rs["ProblemType"], FusedA2A="GemmA2A")
    a2aTags = [p.tag for p in C.ProblemPredicate.CompoundPredicates(
        a2a, C.ProblemType.FromOriginalState(a2a["ProblemType"]))]
    rsTags = [p.tag for p in _compound(rs)]
    assert rsTags == ["GemmRSShape" if t == "GemmA2AShape" else t for t in a2aTags]


@pytest.mark.parametrize("wait", [0, 1, 2])
def test_size_mapping_carries_fused_rs_wait(_gp_gfx942, request, assembler, capsys, wait):
    sol, out = _derive_rs(request, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    mapping = C.SizeMapping.FromOriginalState(dict(sol._state, FusedRSWait=wait))
    assert "fusedRSWait" in C.SizeMapping.StateKeys
    assert mapping.fusedRSWait == wait


# ---------------------------------------------------------------------------
# Codegen guards reached before any GemmRS mapping exists.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode, empty", [("GemmRS", True), ("GemmA2A", True), ("Disabled", False)])
def test_cluster_aid_remap_skips_fused_push(mode, empty):
    class _Labels:
        def getNameInc(self, name):
            return name

    class _Writer:
        labels = _Labels()

    kernel = {"ClusterAIDMapping": -2, "ClusterDim": [4, 2], "TileProcessingStrategy": "None",
              "ProblemType": {"FusedA2A": mode}}
    try:
        module = ClusterAIDRemap(_Writer(), kernel, None)
    except Exception:
        assert not empty, "the remap ran past the fused-push early return"
        return
    assert (len(module.items()) == 0) is empty
