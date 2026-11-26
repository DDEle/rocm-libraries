# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) handshake characterization.

Runs the emitted store-pass gate and handshake of every work-group of every rank (fused_rs_launch),
twice on the same memory, with the last work-group to tally on each rank a RECV or a PUSH unit, and
checks the counter3 election, the DRAIN waits and that every flag, counter and cursor is back at
zero for the next launch.  For FusedRSWait 1 and 2, also runs every work-group of every rank at once from the
RECV wait, in several orders, and checks that each rank's DRAIN follows every work-group's wait.  Also runs
launches back to back on two rotating slots and flag channels with the ranks running ahead of each other,
and checks that no packet lands in a slot or channel whose rank has not ended the launch before.
"""

import random

import pytest

import fused_rs_launch as L
from tensilelite.Components.Signature import (FUSED_A2A_CURSOR_BYTES, FUSED_A2A_DRAIN_SEND,
                                              fusedA2AKernArgLayout)

pytestmark = pytest.mark.unit

_CLUSTERS = [None, (2, 2), (2, 4)]
_WORLDS = [1, 2, 3, 4]
_GRIDS = [(1, 1), (2, 2)]


def _kernels(cluster_dim, wait=0):
    results = L.emit(cluster_dim, wait)
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


@pytest.mark.parametrize("wait", [0, 1, 2])
@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_handshake(cluster_dim, wait):
    for base, k in _kernels(cluster_dim, wait):
        for world_size in _WORLDS:
            for r_units, seg_bands in _GRIDS:
                for last in ("recv", "push") if world_size > 1 else ("recv",):
                    for drain in (3, 2, 1, 0):
                        world = L.World(k, world_size, r_units, seg_bands, drain)
                        for n in range(2):
                            where = (f"Kernel {base!r} W={world_size} grid {(r_units, seg_bands)} last {last} "
                                     f"FusedDrain={drain} launch {n}")
                            launch = world.launch(last)
                            _check_launch(where, world, launch, n)
                            if last == "push":
                                assert launch.passes[-1][2] == "PUSH", where


@pytest.mark.parametrize("wait", [0, 1, 2])
def test_fused_rs_gfx1250_handshake_every_store_variant(wait):
    for base, k in _kernels((2, 2), wait) + _kernels(None, wait):
        for gate in k.gates:
            world = L.World(k, 3, 2, 2, 3)
            for n, last in enumerate(("push", "recv")):
                _check_launch(f"Kernel {base!r} gate line {gate} last {last}", world, world.launch(last, gate), n)


@pytest.mark.parametrize("wait", [1, 2])
@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_handshake_with_recv_wait(cluster_dim, wait):
    for base, k in _kernels(cluster_dim, wait):
        for gate in (k.gates[0], k.gates[-1]):
            for world_size in _WORLDS[1:]:
                world = L.World(k, world_size, 2, 2, 3)
                units = [(r, i) for r in range(world_size) for i in range(len(world.work_groups(r)))]
                for n, seed in enumerate((None, "reversed", 1, 2)):
                    order = units if seed is None else units[::-1] if seed == "reversed" else \
                        random.Random(seed).sample(units, len(units))
                    where = f"Kernel {base!r} gate line {gate} W={world_size} order {seed} launch {n}"
                    launch = world.launch_all(order, gate)
                    _check_launch(where, world, launch, n)
                    for r in range(world_size):
                        recv = {(t[3]["s[sgprWorkGroup0]"], t[3]["s[sgprWorkGroup1]"])
                                for t in world.work_groups(r) if t[3]["s[sgprFusedRSDst]"] == L.M32}
                        log = [(i, kind, wg) for i, (kind, rank, wg) in enumerate(launch.log) if rank == r]
                        ready = [wg for _i, kind, wg in log if kind == "ready" and wg in recv]
                        assert sorted(ready) == sorted(recv), f"{where}: rank {r} RECV units through the wait"
                        drain = [i for i, kind, _wg in log if kind == "drain"]
                        assert len(drain) == 1 and all(i < drain[0] for i, kind, _wg in log if kind == "ready"), (
                            f"{where}: rank {r} DRAIN before a work-group left the RECV wait")


def _target(world, addr):
    """(rank, slot, flag slot?) of a receive-buffer, flag-slot or outbound-counter address."""
    span = world.w * world.seg_n * world.m * 2
    for b in range(world.w):
        for slot in (0, 1):
            if 0 <= addr - world.recv(b, slot) < span:
                return b, slot, False
            if 0 <= addr - world.flags(b, slot) < 4 * world.w:
                return b, slot, True
            if addr == world.outbound(b, slot):
                return b, slot, False
    raise AssertionError(f"packet target {addr:#x} is in no receive buffer or flag block")


def _check_streams(where, world, launch, launches, slots=2):
    w = world.w
    pos = {(kind, r, n): i for i, (kind, r, n) in enumerate(launch.log) if kind in ("start", "end")}
    assert len(pos) == 2 * w * launches, f"{where}: launches started / ended {sorted(pos)}"
    for i, (kind, src, info) in enumerate(launch.log):
        if kind not in ("copy", "atomic"):
            continue
        n, addr = info
        b, slot, inbound = _target(world, addr)
        if kind == "copy" or inbound:
            assert i < pos[("end", b, n)], f"{where}: rank {src} launch {n} {kind} lands after rank {b} ended it"
        if slots == 2:
            assert slot == n % 2, f"{where}: rank {src} launch {n} {kind} into slot {slot}"
        if n >= slots:
            assert i > pos[("end", b, n - slots)], (
                f"{where}: rank {src} launch {n} {kind} lands in rank {b}'s slot {slot} before rank {b} ended "
                f"launch {n - slots}")
    for r in range(w):
        for slot in range(slots):
            flags = [world.mem.get(world.flags(r, slot) + 4 * j, 0) for j in range(w)]
            assert flags == [0] * w, f"{where}: rank {r} channel {slot} flags {flags}"
            cursors = [world.mem.get(world.counters(r, slot) + a, 0) for a in range(0, FUSED_A2A_CURSOR_BYTES, 4)]
            assert not any(cursors), f"{where}: rank {r} slot {slot} cursors {cursors}"
            for addr in world.counter_addrs(r, slot)[1:]:
                assert world.mem.get(addr, 0) == 0, f"{where}: rank {r} slot {slot} counter {addr:#x}"


def _overlaps(launch, w, launches):
    """True if some rank started launch n + 1 before another rank ended launch n."""
    pos = {(kind, r, n): i for i, (kind, r, n) in enumerate(launch.log) if kind in ("start", "end")}
    return any(pos[("start", a, n + 1)] < pos[("end", b, n)]
               for n in range(launches - 1) for a in range(w) for b in range(w))


@pytest.mark.parametrize("wait", [0, 1, 2])
@pytest.mark.parametrize("cluster_dim", [None, (2, 2)])
def test_fused_rs_gfx1250_back_to_back_launches(cluster_dim, wait):
    launches = 4
    for base, k in _kernels(cluster_dim, wait):
        for world_size in _WORLDS[1:]:
            for drain in (3, 0):
                schedules = {"even": [1] * world_size, "rank 0 ahead": [1] + [5] * (world_size - 1),
                             "last rank ahead": [5] * (world_size - 1) + [1]}
                for name, periods in schedules.items():
                    world = L.World(k, world_size, 2, 2, drain)
                    where = f"Kernel {base!r} W={world_size} FusedDrain={drain} schedule {name}"
                    launch = world.launch_streams(launches, periods)
                    _check_streams(where, world, launch, launches)
                    if name != "even":
                        assert _overlaps(launch, world_size, launches), f"{where}: no rank ran ahead"


def _alias_slot1_to_slot0(world):
    layout = fusedA2AKernArgLayout()
    for r in range(world.w):
        ka = world.kernarg(r, 1) + world.k.fused_base
        for j in range(world.w):
            world.store64(ka + layout["peer_%d_flagPtr" % j], world.flags(j))
            world.store64(ka + layout["peer_%d_bufPtr" % j], world.recv(j))


@pytest.mark.parametrize("wait", [0, 1, 2])
def test_fused_rs_gfx1250_back_to_back_launches_on_one_slot_fail(wait):
    for base, k in _kernels(None, wait):
        world = L.World(k, 2, 2, 2, 3)
        _alias_slot1_to_slot0(world)
        with pytest.raises(AssertionError, match=r"lands in rank \d+'s slot 0 before rank \d+ ended launch"):
            _check_streams(f"Kernel {base!r}", world, world.launch_streams(4, [1, 5]), 4, slots=1)
