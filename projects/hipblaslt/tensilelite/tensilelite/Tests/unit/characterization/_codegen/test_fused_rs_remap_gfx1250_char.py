# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) dispatch-order remap characterization.

Runs the emitted FusedRSWgRemap instructions on every work-group of a set of
grids, for every rank of several world sizes, with and without a workgroup
cluster, and checks the tile, FusedRSDst and the latched segment sizes.
"""

import os
import pathlib
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Components.Signature import fusedA2AKernArgLayout
from tensilelite.Components.WorkGroupMappingAlgos import FusedRSWgRemap

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_PLAIN = os.path.join(_COMM, "fused_a2a.yaml")
_CLUSTER = os.path.join(_COMM, "fused_a2a_cluster.yaml")

# Comments on the remap's own instructions.
_REMAP_COMMENTS = {
    "FusedMyRank, FusedW", "counter_ptr", "wait FusedMyRank, FusedW and counter_ptr",
    "R = N0 / Cs (dispatch units per band)", "bands = N1 / Ck", "cy", "cx", "u = cy*R ...",
    "... + cx (cluster dispatch index)", "peer x", "peer y", "u = wg1*N0 ...", "... + wg0 (dispatch index)",
    "FusedRSDst = dst if PUSH else -1", "WorkGroup0 = m * Cs + peer x", "WorkGroup1 = band * Ck + peer y",
}
_TOTAL_WGS = "FusedTotalWGs = NumWorkGroups0 * NumWorkGroups1 (counter3 election target)"
_RANK_W = "FusedMyRank, FusedW"
_SKIPPED = ("s_wait_", "s_set_vgpr_msb", "s_nop", "s_delay_alu")
_BRANCH = re.compile(r"^(s_(c?branch|setpc|swappc|endpgm)|\S+:$)")
_COUNTER_PTR = "s[sgprFusedCounterPtr:sgprFusedCounterPtr+1]"
_NEG1 = 0xFFFFFFFF

_WORLDS = [1, 2, 3, 4, 8]
# (dispatch units per band R, bands per segment)
_GRIDS = [(1, 1), (3, 2), (4, 3), (5, 1)]


def _rs_config(tmp_path, cluster_dim, wait=0):
    with open(_CLUSTER if cluster_dim else _PLAIN) as f:
        config = yaml.safe_load(f)
    problem_type, group = config["BenchmarkProblems"][0]
    problem_type.update(FusedA2A="GemmRS", UseBeta=False)
    forks = [p for p in group["ForkParameters"]
             if not {"ClusterDim", "ClusterAIDMapping"} & set(p)] + [{"FusedRSWait": [wait]}]
    if cluster_dim:
        forks += [{"ClusterDim": [list(cluster_dim)]}, {"ClusterAIDMapping": [-1]}]
    group["ForkParameters"] = forks
    group["BenchmarkFinalParameters"] = [{"ProblemSizes": [
        {"Exact": {"sizes": [4096, 2048, 1, 2048], "world": 2}}]}]
    path = pathlib.Path(tmp_path) / "fused_rs.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _remap_window(src):
    """(instruction, comment) from the first remap instruction to the FusedTotalWGs latch after it."""
    lines = []
    for ln in src.splitlines():
        code, _, comment = ln.partition("//")
        lines.append((code.strip(), comment.strip()))
    idx = [i for i, (code, comment) in enumerate(lines) if code and comment in _REMAP_COMMENTS | {_TOTAL_WGS}]
    assert idx, "no fused-RS remap in the kernel"
    return [(code, comment) for code, comment in lines[idx[0]:idx[-1] + 1] if code]


def _remap_body(src):
    return [(code, comment) for code, comment in _remap_window(src) if not code.startswith(_SKIPPED)]


def _pair(operand):
    lo, hi = re.fullmatch(r"s\[(\d+):(\d+)\]", operand).groups()
    return f"s{lo}", f"s{hi}"


def _run_remap(body, n0, n1, rank, world, wg0, wg1):
    """One lane of the remap: the registers after `body`."""
    reg = {"s[sgprNumWorkGroups0]": n0, "s[sgprNumWorkGroups1]": n1,
           "s[sgprWorkGroup0]": wg0, "s[sgprWorkGroup1]": wg1, "exec_lo": 1}
    u32 = lambda v: v & 0xFFFFFFFF
    val = lambda x: reg[x] if x in reg else u32(int(x, 0))
    # op -> (result, SCC): None sets SCC = result != 0, "keep" leaves SCC.
    salu = {
        "s_lshr_b32": lambda a, b: (a >> (b & 31), None),
        "s_lshl_b32": lambda a, b: (u32(a << (b & 31)), None),
        "s_and_b32": lambda a, b: (a & b, None),
        "s_or_b32": lambda a, b: (a | b, None),
        "s_mul_i32": lambda a, b: (u32(a * b), "keep"),
        "s_add_u32": lambda a, b: (u32(a + b), int(a + b > 0xFFFFFFFF)),
        "s_sub_u32": lambda a, b: (u32(a - b), int(b > a)),
        "s_max_u32": lambda a, b: (max(a, b), int(a > b)),
    }
    valu = {
        "v_mul_lo_u32": lambda a, b: a * b, "v_sub_nc_u32": lambda a, b: a - b,
        "v_cmp_ge_u32": lambda a, b: int(a >= b),
    }
    scc, loads = None, []
    for ln, comment in body:
        op, rest = ln.split(None, 1)
        a = [x.strip() for x in rest.split(",")]
        if op.startswith("v_") and op != "v_readfirstlane_b32" and not reg["exec_lo"] & 1:
            continue
        if op in salu:
            res, flag = salu[op](val(a[1]), val(a[2]))
            reg[a[0]] = res
            if flag is None:
                scc = int(res != 0)
            elif flag != "keep":
                scc = flag
        elif op in valu:
            reg[a[0]] = u32(valu[op](val(a[1]), val(a[2])))
        elif op == "s_load_b64":
            offset = int(a[2], 0)
            loads.append((a[0], comment, offset))
            if comment == _RANK_W:
                lo, hi = _pair(a[0])
                reg[lo], reg[hi] = rank, world
        elif op == "s_cmp_lt_u32":
            scc = int(val(a[0]) < val(a[1]))
        elif op == "s_cmp_ge_u32":
            scc = int(val(a[0]) >= val(a[1]))
        elif op == "s_cselect_b32":
            assert scc is not None, f"{ln!r} reads SCC before any write"
            reg[a[0]] = val(a[1]) if scc else val(a[2])
        elif op in ("s_mov_b32", "v_readfirstlane_b32"):
            reg[a[0]] = val(a[1])
        elif op == "v_cvt_f64_u32":
            reg[a[0]] = float(val(a[1]))
        elif op == "v_rcp_f64":
            reg[a[0]] = 1.0 / reg[a[1]]
        elif op == "v_mul_f64":
            reg[a[0]] = reg[a[1]] * reg[a[2]]
        elif op == "v_cvt_u32_f64":
            reg[a[0]] = int(reg[a[1]])
        elif op == "v_add_nc_u32":
            reg[a[0]] = u32(val(a[1]) + val(a[2]))
        else:
            pytest.fail(f"remap instruction not modelled: {ln!r}")
    return reg, loads


def _check_kernel_args(loads):
    layout = fusedA2AKernArgLayout()
    assert layout["FusedW"] == layout["FusedMyRank"] + 4
    counter = [o for d, c, o in loads if d == _COUNTER_PTR and c == "counter_ptr"]
    rank_w = [o for d, c, o in loads if c == _RANK_W]
    assert len(counter) == 1 and rank_w == [counter[0] + layout["FusedMyRank"] - layout["counter_ptr"]], loads


def _check_remap(base, src, cs, ck):
    body = _remap_body(src)
    mt1 = int(re.search(r"segN = segBands \* \(MT1=(\d+)\)", src).group(1))
    for line, _comment in _remap_window(src):
        assert not _BRANCH.match(line), f"Kernel {base!r}: branch or label inside the remap: {line!r}"
    for world in _WORLDS:
        for r_units, seg_bands in _GRIDS:
            n0, n1 = r_units * cs, world * seg_bands * ck
            push_units = r_units * (world - 1) * seg_bands
            for rank in range(world) if world <= 4 else (0, 3, world - 1):
                where = f"Kernel {base!r} W={world} rank={rank} grid {(n0, n1)}"
                others = sorted(set(range(world)) - {rank})
                out, rounds = {}, {}
                for wg1 in range(n1):
                    for wg0 in range(n0):
                        reg, loads = _run_remap(body, n0, n1, rank, world, wg0, wg1)
                        _check_kernel_args(loads)
                        assert reg["s[sgprFusedTokenTiles]"] == seg_bands, f"{where}: segBands latch"
                        assert reg["s[sgprFusedNShard]"] == n1 * mt1 // world, f"{where}: segN latch"
                        assert (reg["s[sgprNumWorkGroups0]"], reg["s[sgprNumWorkGroups1]"]) == (n0, n1), (
                            f"{where}: the remap changed NumWorkGroups0/1")
                        assert reg["s[sgprFusedTotalWGs]"] == n0 * n1, f"{where}: FusedTotalWGs"
                        tile = (reg["s[sgprWorkGroup0]"], reg["s[sgprWorkGroup1]"])
                        dst = reg["s[sgprFusedRSDst]"]
                        out[(wg0, wg1)] = tile
                        u = wg0 // cs + (n0 // cs) * (wg1 // ck)
                        q, m = divmod(u, r_units)
                        band, peer = divmod(tile[1], ck)
                        owner, local = divmod(band, seg_bands)
                        assert (tile[0] // cs, tile[0] % cs, peer) == (m, wg0 % cs, wg1 % ck), (
                            f"{where}: work-group {(wg0, wg1)} -> {tile} is not M block {m} of its cluster")
                        if u < push_units:
                            assert owner == (rank + 1 + q % (world - 1)) % world and dst == owner != rank, (
                                f"{where}: PUSH unit {u} -> owner {owner}, FusedRSDst {dst}")
                            assert local == q // (world - 1), f"{where}: PUSH unit {u} -> band {local}"
                            rounds.setdefault(q // (world - 1), []).append(owner)
                        else:
                            assert owner == rank and dst == _NEG1, (
                                f"{where}: RECV unit {u} -> owner {owner}, FusedRSDst {dst}")
                            assert local == q - (world - 1) * seg_bands, f"{where}: RECV unit {u} -> band {local}"
                assert sorted(out.values()) == sorted(out), f"{where}: not a bijection"
                assert sorted(rounds) == list(range(seg_bands if world > 1 else 0)), f"{where}: rounds"
                for b, owners in rounds.items():
                    assert sorted(set(owners)) == others, f"{where}: round {b} owners {owners}"
                    assert len(owners) == len(others) * r_units * cs * ck, f"{where}: round {b} size"


def _emit(tmp_path, cluster_dim):
    results = emit_kernels_from_config(_rs_config(tmp_path, cluster_dim), limit=8, arch=_ARCH,
                                       cluster_dim=cluster_dim)
    assert results
    for base, src, err in results:
        assert err == 0, base
    return results


def test_fused_rs_gfx1250_wg_remap(tmp_path):
    results = _emit(tmp_path, None)
    assert len(results) == 2, [b for (b, _s, _e) in results]
    for base, src, _err in results:
        assert_assembles(src, base)
        _check_remap(base, src, 1, 1)


@pytest.mark.parametrize("cluster_dim", [[2, 2], [4, 2], [2, 4]])
def test_fused_rs_gfx1250_cluster_wg_remap(tmp_path, cluster_dim):
    results = _emit(tmp_path, cluster_dim)
    assert len(results) == 1, [b for (b, _s, _e) in results]
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "s_sendmsg_rtn_b32" not in src
        _check_remap(base, src, *cluster_dim)


@pytest.mark.parametrize("cluster_dim", [[3, 2], [2, 3], [6, 1]])
def test_fused_rs_remap_refuses_non_power_of_two_clusters(cluster_dim):
    kernel = {"ClusterDim": cluster_dim, "ProblemType": {"FusedA2A": "GemmRS"}}
    with pytest.raises(RuntimeError, match="power-of-two ClusterDim"):
        FusedRSWgRemap(None, kernel)
