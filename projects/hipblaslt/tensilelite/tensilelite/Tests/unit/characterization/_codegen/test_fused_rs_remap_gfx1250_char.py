# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) dispatch-order remap characterization.

Runs the emitted FusedRSWgRemap instructions on every work-group of a set of
grids, for every rank of several world sizes, with and without a workgroup
cluster, and checks the tile, FusedRSDst and the latched segment sizes.
"""

import re

import pytest

import fused_rs_launch as L
from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Components.Signature import fusedA2AKernArgLayout
from tensilelite.Components.WorkGroupMappingAlgos import FusedRSWgRemap

pytestmark = pytest.mark.unit

_BRANCH = re.compile(r"^(s_(c?branch|setpc|swappc|endpgm)|\S+:$)")
_COUNTER_PTR = "s[sgprFusedCounterPtr:sgprFusedCounterPtr+1]"
_NEG1 = 0xFFFFFFFF

_WORLDS = [1, 2, 3, 4, 8]
# (dispatch units per band R, bands per segment)
_GRIDS = [(1, 1), (3, 2), (4, 3), (5, 1)]


def _check_kernel_args(window):
    layout = fusedA2AKernArgLayout()
    assert layout["FusedW"] == layout["FusedMyRank"] + 4
    loads = [(a[0], c, int(a[2], 0)) for code, c in window if code.startswith("s_load_b64")
             for a in [[x.strip() for x in code.split(None, 1)[1].split(",")]]]
    counter = [o for d, c, o in loads if d == _COUNTER_PTR and c == "counter_ptr"]
    rank_w = [o for d, c, o in loads if c == L.RANK_W]
    assert len(counter) == 1 and rank_w == [counter[0] + layout["FusedMyRank"] - layout["counter_ptr"]], loads


def _check_remap(base, src, cs, ck):
    kernel = L.Kernel(src, (cs, ck))
    mt1 = int(re.search(r"segN = segBands \* \(MT1=(\d+)\)", src).group(1))
    first, last = kernel.remap
    window = [(code, comment) for code, comment in kernel.lines[first:last + 1] if code]
    for line, _comment in window:
        assert not _BRANCH.match(line), f"Kernel {base!r}: branch or label inside the remap: {line!r}"
    _check_kernel_args(window)
    for world in _WORLDS:
        for r_units, seg_bands in _GRIDS:
            machine = L.World(kernel, world, r_units, seg_bands, 3)
            n0, n1 = r_units * cs, world * seg_bands * ck
            push_units = r_units * (world - 1) * seg_bands
            for rank in range(world) if world <= 4 else (0, 3, world - 1):
                where = f"Kernel {base!r} W={world} rank={rank} grid {(n0, n1)}"
                others = sorted(set(range(world)) - {rank})
                out, rounds = {}, {}
                for wg1 in range(n1):
                    for wg0 in range(n0):
                        reg = machine.remap(rank, wg0, wg1)
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
    results = emit_kernels_from_config(L.rs_config(tmp_path, cluster_dim), limit=8, arch=L.ARCH,
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
