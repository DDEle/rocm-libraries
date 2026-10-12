# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.SolutionStructs.Problem import ProblemSizes, ProblemType

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
