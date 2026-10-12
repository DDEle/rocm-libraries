# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

import Tensile.SolutionLibrary as SL
from Tensile.SolutionStructs.Problem import ProblemSizes, ProblemType
from Tensile.TensileMergeLibrary import fixSizeInconsistencies

pytestmark = pytest.mark.unit

_FUSED_CONFIG = {
    "OperationType": "GEMM",
    "DataType": "b",
    "DestDataType": "b",
    "ComputeDataType": "s",
    "HighPrecisionAccumulate": True,
    "TransposeA": True,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
    "FusedGemmA2A": True,
}


def _problemType(**overrides):
    return ProblemType({**_FUSED_CONFIG, **overrides}, False)


def _sizes(problemType, *exacts):
    return ProblemSizes(problemType, [{"Exact": e} for e in exacts])


class TestExactDictWorld:
    def test_world_and_extent_are_kept(self):
        ps = _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 4, "a2aExtent": 2048})
        assert (ps.exacts[0].world, ps.exacts[0].a2aExtent) == (4, 2048)

    def test_extent_defaults_to_m(self):
        ps = _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 2})
        assert ps.exacts[0].a2aExtent == 3072

    def test_fused_requires_world(self):
        with pytest.raises(RuntimeError, match="must set 'world'"):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048]})

    def test_world_must_be_positive(self):
        with pytest.raises(RuntimeError, match="world=0"):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 0})

    def test_non_fused_rejects_world(self):
        with pytest.raises(RuntimeError, match="only to FusedGemmA2A"):
            _sizes(_problemType(FusedGemmA2A=False), {"sizes": [3072, 512, 1, 2048], "world": 2})

    def test_non_fused_problem_has_no_world(self):
        ps = _sizes(_problemType(FusedGemmA2A=False), {"sizes": [3072, 512, 1, 2048]})
        assert (ps.exacts[0].world, ps.exacts[0].a2aExtent) == (None, None)

    def test_fused_rejects_exact_list(self):
        with pytest.raises(RuntimeError, match="Exact dict"):
            _sizes(_problemType(), [3072, 512, 1, 2048])

    def test_fused_rejects_range(self):
        with pytest.raises(RuntimeError, match="Range"):
            ProblemSizes(_problemType(), [{"Range": [[3072], [512], [1], [2048]]}])


class TestFusedProblemTypeRejects:
    @pytest.mark.parametrize("overrides, name", [
        ({"UseBias": 1}, "UseBias"),
        ({"UseScaleAB": "Scalar"}, "UseScaleAB"),
        ({"UseScaleCD": True}, "UseScaleCD"),
        ({"UseScaleAlphaVec": 1}, "UseScaleAlphaVec"),
        ({"OutputAmaxD": True}, "OutputAmaxD"),
        ({"GroupedGemm": True}, "GroupedGemm"),
        ({"Activation": True, "ActivationType": "hipblaslt_all"}, "Activation"),
    ])
    def test_rejects(self, overrides, name):
        with pytest.raises(Exception, match="FusedGemmA2A does not support .*%s" % name):
            _problemType(**overrides)

    def test_plain_fused_is_accepted(self):
        assert _problemType()["FusedGemmA2A"]


class _FakeSol:
    def __init__(self, index):
        self.index = index


class TestFusedMatchingKey:
    _ROW = [[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 4, 2048], [0, 1000.0]]

    def test_world_and_extent_join_the_key(self):
        d = {"indexOrder": [2, 3, 0, 1], "distance": "Equality", "table": [self._ROW]}
        lib = SL.MatchingLibrary.FromOriginalState(d, [_FakeSol(0)], fusedGemmA2A=True)
        assert [p.tag for p in lib.properties][-2:] == ["FusedA2AWorld", "FusedA2AExtent"]
        assert lib.table[0]["key"] == [3072, 512, 1, 2048, 4, 2048]

    def test_non_fused_key_unchanged(self):
        row = [self._ROW[0][:8], self._ROW[1]]
        d = {"indexOrder": [2, 3, 0, 1], "distance": "Equality", "table": [row]}
        lib = SL.MatchingLibrary.FromOriginalState(d, [_FakeSol(0)])
        assert lib.table[0]["key"] == [3072, 512, 1, 2048]
        assert len(lib.properties) == 4


class TestMergeKeepsWorld:
    def test_strips_lds_only(self):
        rows, n = fixSizeInconsistencies(
            [[[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 2, 2048], [0, 1.0]],
             [[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 4, 2048], [0, 2.0]]], "base")
        assert n == 2
        assert sorted(r[0] for r in rows) == [[3072, 512, 1, 2048, 2, 2048],
                                              [3072, 512, 1, 2048, 4, 2048]]

    def test_non_fused_unchanged(self):
        rows, _ = fixSizeInconsistencies([[[128, 128, 1, 64, 128, 128, 64, 64], [0, 1.0]]], "base")
        assert rows[0][0] == [128, 128, 1, 64]
