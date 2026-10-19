# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit

_BASE = {
    "OperationType": "GEMM",
    "DataType": "b",
    "DestDataType": "b",
    "ComputeDataType": "s",
    "HighPrecisionAccumulate": True,
    "TransposeA": True,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
}


def _pt(**overrides):
    return ProblemType({**_BASE, **overrides}, False)


class TestModeValues:
    def test_default_is_none(self):
        assert _pt()["FusedA2A"] == "None"

    @pytest.mark.parametrize("mode", ["GemmA2A", "A2AGemm"])
    def test_modes_round_trip(self, mode):
        assert _pt(FusedA2A=mode)["FusedA2A"] == mode

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(Exception, match="FusedA2A"):
            _pt(FusedA2A="Gemm")


class TestRetiredKeys:
    def test_fused_gemm_a2a_true_is_rejected(self):
        with pytest.raises(Exception, match="FusedA2A: GemmA2A"):
            _pt(FusedGemmA2A=True)

    def test_fused_a2a_mode_one_is_rejected(self):
        with pytest.raises(Exception, match="FusedA2A: A2AGemm"):
            _pt(FusedA2AMode=1)

    def test_false_and_zero_are_ignored(self):
        assert _pt(FusedGemmA2A=False, FusedA2AMode=0)["FusedA2A"] == "None"


class TestA2AGemmProblemTypeRejects:
    @pytest.mark.parametrize("overrides, name", [
        ({"UseBias": 1}, "UseBias"),
        ({"TransposeB": True}, "TransposeB"),
    ])
    def test_rejects(self, overrides, name):
        with pytest.raises(Exception, match="FusedA2A=A2AGemm does not support .*%s" % name):
            _pt(FusedA2A="A2AGemm", **overrides)


class TestName:
    def test_gemm_a2a_keeps_its_tag(self):
        assert "_FusedA2A" in str(_pt(FusedA2A="GemmA2A"))

    def test_a2a_gemm_has_its_own_tag(self):
        assert "_A2AGemm" in str(_pt(FusedA2A="A2AGemm"))
        assert "_A2AGemm" not in str(_pt())


class TestPredicate:
    @pytest.mark.parametrize("mode", ["None", "GemmA2A", "A2AGemm"])
    def test_every_solution_carries_the_mode(self, mode):
        cpt = ContractionsProblemType.FromOriginalState(_pt(FusedA2A=mode).state)
        preds = {p.tag: p.value for p in cpt.predicates(True, True, True)}
        assert preds["FusedA2A"] == mode
        assert "FusedGemmA2A" not in preds

    @pytest.mark.parametrize("mode, expected", [("None", False), ("GemmA2A", True), ("A2AGemm", True)])
    def test_batch_one_follows_either_mode(self, mode, expected):
        cpt = ContractionsProblemType.FromOriginalState(_pt(FusedA2A=mode).state)
        tags = [p.tag for p in cpt.predicates(True, True, True)]
        assert ("BatchSizeEqual" in tags) is expected
