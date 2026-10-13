# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
from unittest.mock import patch

import pytest

import Tensile.ClientWriter as CW
import Tensile.SolutionLibrary as SL
from Tensile.Contractions import ProblemType as ContractionsProblemType
from Tensile.SolutionStructs.Problem import Problem, ProblemSizes, ProblemSizesMock, ProblemType
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

    @pytest.mark.parametrize("extent", [0, -2048, 4096])
    def test_extent_must_be_within_m(self, extent):
        with pytest.raises(RuntimeError, match="a2aExtent=%d" % extent):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 2, "a2aExtent": extent})

    def test_fused_rejects_batch(self):
        with pytest.raises(RuntimeError, match="batch=2"):
            _sizes(_problemType(), {"sizes": [3072, 512, 2, 2048], "world": 2})

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


class TestClientWriterWorld:
    def _params(self, fused):
        cpt = ContractionsProblemType.FromOriginalState(_problemType(FusedGemmA2A=fused).state)
        problem = Problem(sizes=[3072, 512, 1, 2048, 3072, 3072, 2048, 2048],
                          world=4 if fused else None, a2aExtent=2048 if fused else None)
        return CW.problemSizeParams(cpt, problem, [0])

    def test_fused_writes_world_and_extent(self):
        params = self._params(True)
        assert ("world", 4) in params and ("fused-a2a-extent", 2048) in params

    def test_non_fused_writes_neither(self):
        keys = [k for k, _ in self._params(False)]
        assert "world" not in keys and "fused-a2a-extent" not in keys


class TestProblemSizesMockWorld:
    def test_fused_rows_split_world(self):
        mock = ProblemSizesMock([[[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 8, 2048], [0, 1.0]]], True)
        p = mock.problems[0]
        assert (list(p.sizes), p.world, p.a2aExtent) == ([3072, 512, 1, 2048, 3072, 3072, 2048, 2048], 8, 2048)

    def test_non_fused_rows_untouched(self):
        mock = ProblemSizesMock([[[128, 128, 1, 64], [0, 1.0]]])
        assert (list(mock.problems[0].sizes), mock.problems[0].world) == ([128, 128, 1, 64], None)


class _MockDataType:
    def flopsPerMac(self):
        return 2


class _FusedMockProblemType(dict):
    def __init__(self):
        super().__init__({
            "Index0": 0, "Index1": 1, "IndexUnroll": 3, "TotalIndices": 4, "NumIndicesLD": 0,
            "NumIndicesC": 3, "IndexAssignmentsA": [3, 0, 2], "IndexAssignmentsB": [3, 1, 2],
            "IndexAssignmentsLD": [], "DataType": _MockDataType(), "OperationType": "GEMM",
            "TileAwareSelection": False, "GroupedGemm": False, "FusedGemmA2A": True,
        })

    def __hash__(self):
        return id(self)


class _MockSolutionDict(dict):
    def __init__(self, idx):
        super().__init__({"MacroTile0": 256, "MacroTile1": 256, "SolutionIndex": idx})

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _FusedSizes:
    def __init__(self, keys):
        self.problems = []
        self.ranges = []
        self.exacts = [Problem(sizes=k[:4], world=k[4], a2aExtent=k[5]) for k in keys]


class TestLibraryLogicWorldKey:
    def test_winner_per_world(self, tmp_path):
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters
        from Tensile.LibraryLogic import LogicAnalyzer
        keys = [(3072, 512, 1, 2048, 2, 2048), (3072, 512, 1, 2048, 4, 2048)]
        path = tmp_path / "bench.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["GFlops", "SizeI", "SizeJ", "SizeK", "SizeL", "World", "A2AExtent", "TotalFlops", "S0", "S1"])
            w.writerow(["GFlops", *keys[0], 1, 20.0, 10.0])
            w.writerow(["GFlops", *keys[1], 1, 10.0, 30.0])
        name = lambda sol, *a, **k: "Sol%d" % sol["SolutionIndex"]
        with patch("Tensile.LibraryLogic.getSolutionNameMin", name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", name):
            la = LogicAnalyzer(_FusedMockProblemType(), [_FusedSizes(keys)],
                               [[_MockSolutionDict(0), _MockSolutionDict(1)]], [str(path)],
                               dict(defaultAnalysisParameters), splitGSU=False)
        assert la.exactWinners[keys[0]][0] == 0
        assert la.exactWinners[keys[1]][0] == 1

    def test_extra_column_before_world(self, tmp_path):
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters
        from Tensile.LibraryLogic import LogicAnalyzer
        keys = [(3072, 512, 1, 2048, 2, 2048), (3072, 512, 1, 2048, 4, 2048)]
        path = tmp_path / "bench.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["GFlops", "SizeI", "SizeJ", "SizeK", "SizeL", "TotalFlops", "World", "A2AExtent", "S0", "S1"])
            w.writerow(["GFlops", *keys[0][:4], 1, keys[0][4], keys[0][5], 20.0, 10.0])
            w.writerow(["GFlops", *keys[1][:4], 1, keys[1][4], keys[1][5], 10.0, 30.0])
        name = lambda sol, *a, **k: "Sol%d" % sol["SolutionIndex"]
        with patch("Tensile.LibraryLogic.getSolutionNameMin", name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", name):
            la = LogicAnalyzer(_FusedMockProblemType(), [_FusedSizes(keys)],
                               [[_MockSolutionDict(0), _MockSolutionDict(1)]], [str(path)],
                               dict(defaultAnalysisParameters), splitGSU=False)
        assert la.exactWinners[keys[0]][0] == 0
        assert la.exactWinners[keys[1]][0] == 1

    def test_missing_world_header_exits(self, tmp_path):
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters
        from Tensile.LibraryLogic import LogicAnalyzer
        keys = [(3072, 512, 1, 2048, 2, 2048)]
        path = tmp_path / "bench.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["GFlops", "SizeI", "SizeJ", "SizeK", "SizeL", "A2AExtent", "TotalFlops", "S0"])
            w.writerow(["GFlops", 3072, 512, 1, 2048, 2048, 1, 20.0])
        name = lambda sol, *a, **k: "Sol%d" % sol["SolutionIndex"]
        with patch("Tensile.LibraryLogic.getSolutionNameMin", name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", name), \
             pytest.raises(SystemExit):
            LogicAnalyzer(_FusedMockProblemType(), [_FusedSizes(keys)],
                          [[_MockSolutionDict(0)]], [str(path)],
                          dict(defaultAnalysisParameters), splitGSU=False)

    def test_missing_world_row_is_skipped(self, tmp_path):
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters
        from Tensile.LibraryLogic import LogicAnalyzer
        keys = [(3072, 512, 1, 2048, 2, 2048), (3072, 512, 1, 2048, 8, 2048)]
        path = tmp_path / "bench.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["GFlops", "SizeI", "SizeJ", "SizeK", "SizeL", "World", "A2AExtent", "TotalFlops", "S0"])
            w.writerow(["GFlops", *keys[0], 1, 20.0])
        name = lambda sol, *a, **k: "Sol%d" % sol["SolutionIndex"]
        with patch("Tensile.LibraryLogic.getSolutionNameMin", name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", name):
            la = LogicAnalyzer(_FusedMockProblemType(), [_FusedSizes(keys)],
                               [[_MockSolutionDict(0)]], [str(path)],
                               dict(defaultAnalysisParameters), splitGSU=False)
        assert list(la.exactWinners) == [keys[0]]


class TestSyntheticCsvWorld:
    def test_columns(self, tmp_path):
        from Tensile.BenchmarkProblems import _writeSyntheticResultsCSV

        class _PS:
            problems = [Problem(sizes=[3072, 512, 1, 2048, 3072, 3072, 2048, 2048], world=4, a2aExtent=2048)]

        path = tmp_path / "s.csv"
        _writeSyntheticResultsCSV(str(path), _PS(), "gfx950", 1)
        rows = list(csv.reader(open(path)))
        assert rows[1][1:11] == ["3072", "512", "1", "2048", "3072", "3072", "2048", "2048", "4", "2048"]


class TestBenchmarkDataRoundTrip:
    def test_exact_line_reloads_with_world(self):
        import yaml
        pt = _problemType()
        written = _sizes(pt, {"sizes": [3072, 512, 1, 2048], "world": 4, "a2aExtent": 2048}).exacts[0]
        loaded = yaml.load("- Exact: {}\n".format(written), yaml.CSafeLoader)
        ps = ProblemSizes(pt, loaded)
        assert (ps.exacts[0].world, ps.exacts[0].a2aExtent) == (4, 2048)
