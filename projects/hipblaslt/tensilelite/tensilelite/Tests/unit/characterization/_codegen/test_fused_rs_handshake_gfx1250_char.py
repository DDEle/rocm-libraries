# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) handshake characterization.

Runs the emitted store-pass gate and handshake of every work-group of every rank (fused_rs_launch),
twice on the same memory, with the last work-group to tally on each rank a RECV or a PUSH unit, and
checks the counter3 election, the DRAIN waits and that every flag, counter and cursor is back at
zero for the next launch.
"""

import pytest

import fused_rs_launch as L
from tensilelite.Components.Signature import FUSED_A2A_CURSOR_BYTES, FUSED_A2A_DRAIN_SEND

pytestmark = pytest.mark.unit

_CLUSTERS = [None, (2, 2), (2, 4)]
_WORLDS = [1, 2, 3, 4]
_GRIDS = [(1, 1), (2, 2)]


def _kernels(cluster_dim):
    results = L.emit(cluster_dim)
    assert results and all(err == 0 for _b, _s, err in results)
    return [(base, L.Kernel(src, cluster_dim)) for base, src, _err in results]


def _check_launch(where, world, launch, n):
    w = world.w
    assert sorted(launch.drains) == list(range(w)), f"{where}: DRAIN ran on ranks {launch.drains}"
    for rank, comment, exec_mask in launch.polls:
        assert w > 1, f"{where}: W=1 executed a wait"
        if comment == L.POLL:
            assert exec_mask == ((1 << w) - 1) & ~(1 << rank), f"{where}: rank {rank} polls lanes {exec_mask:#x}"
        else:
            assert exec_mask == 1 and world.drain & FUSED_A2A_DRAIN_SEND, f"{where}: outbound poll"
    inbound = {r for r, c, _e in launch.polls if c == L.POLL}
    assert inbound == (set(range(w)) if w > 1 else set()), f"{where}: inbound waits on ranks {inbound}"
    for r in range(w):
        flags = [world.mem.get(world.flags(r) + 4 * j, 0) for j in range(w)]
        assert flags == [0] * w, f"{where}: rank {r} flags {flags}"
        sent = 0 if world.drain & FUSED_A2A_DRAIN_SEND else (n + 1) * (w - 1)
        assert world.mem.get(world.outbound(r), 0) == sent, f"{where}: rank {r} outbound"
        cursors = [world.mem.get(world.counters(r) + a, 0) for a in range(0, FUSED_A2A_CURSOR_BYTES, 4)]
        assert not any(cursors), f"{where}: rank {r} cursors {cursors}"
        for addr in world.counter_addrs(r)[1:]:
            assert world.mem.get(addr, 0) == 0, f"{where}: rank {r} counter {addr:#x}"


@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_handshake(cluster_dim):
    for base, k in _kernels(cluster_dim):
        for world_size in _WORLDS:
            for r_units, seg_bands in _GRIDS:
                for last in ("recv", "push") if world_size > 1 else ("recv",):
                    for drain in (3, 1, 0):
                        world = L.World(k, world_size, r_units, seg_bands, drain)
                        for n in range(2):
                            where = (f"Kernel {base!r} W={world_size} grid {(r_units, seg_bands)} last {last} "
                                     f"FusedDrain={drain} launch {n}")
                            launch = world.launch(last)
                            _check_launch(where, world, launch, n)
                            if last == "push":
                                assert launch.passes[-1][2] == "PUSH", where


def test_fused_rs_gfx1250_handshake_every_store_variant():
    for base, k in _kernels((2, 2)) + _kernels(None):
        for gate in k.gates:
            world = L.World(k, 3, 2, 2, 3)
            for n, last in enumerate(("push", "recv")):
                _check_launch(f"Kernel {base!r} gate line {gate} last {last}", world, world.launch(last, gate), n)
