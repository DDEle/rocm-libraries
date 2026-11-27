# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) per-band arrival (FusedRSWait=2) characterization.

Runs the emitted handshake of every work-group of every rank (fused_rs_launch) for several world sizes,
cluster shapes and bands per segment, and checks the band ATOMIC behind each band COPY, the flag slots DRAIN
finds and clears, the wait of a RECV work-group on its band's bit and the rows it then reads, and the band
ATOMIC and the RECV wait in the final .s.
"""

import re
from collections import Counter

import pytest

import fused_rs_launch as L
from config_harness import assert_assembles
from test_fused_rs_push_gfx1250_char import _check_copies, _check_counters, _check_queues
from tensilelite.Components.SdmaRingEmitter import SDMA_QUEUE_SIZE
from tensilelite.Components.Signature import FUSED_A2A_CURSOR_BYTES

pytestmark = pytest.mark.unit

_CLUSTERS = [None, (2, 2), (2, 4)]
_WORLDS = [1, 2, 3, 4]
_SEG_BANDS = [1, 2, 16, 32]
_OWN = 0x5A5A0000
_ROUNDS = 10 ** 6
_BAND_LANE = "fused-RS: the band ATOMIC lane"
_RESERVE = "WrapIntoRing(cur) + size"


def _kernels(cluster_dim, wait=2):
    results = L.emit(cluster_dim, wait)
    assert results and all(err == 0 for _b, _s, err in results)
    return [(base, L.Kernel(src, cluster_dim)) for base, src, _err in results]


def _world(k, world_size, seg_bands, start=0):
    """Every rank's own flag slot holds _OWN; every SDMA queue starts at ring byte `start`."""
    world = L.World(k, world_size, 1, seg_bands, drain=3)
    for r in range(world_size):
        world.mem[world.flags(r) + 4 * r] = _OWN
        for j in range(world_size):
            world.read[(r, j)] = start
            world.store64(world.rptr(r, j), start)
            world.store64(world.rptr(r, j) + 8, start)
    return world


def _check_launch(where, world, launch):
    w, full = world.w, (1 << world.seg_bands) - 1
    _check_copies(where, world, launch)
    _check_queues(where, world, launch, 2)
    _check_counters(where, world, launch)
    for r in range(w):
        found = launch.inbound.get(r)
        assert found == ([_OWN if j == r else full for j in range(w)] if w > 1 else None), (
            f"{where}: rank {r} DRAIN found {found}")
        flags = [world.mem.get(world.flags(r) + 4 * j, 0) for j in range(w)]
        assert flags == [_OWN if j == r else 0 for j in range(w)], f"{where}: rank {r} flags {flags}"
        assert world.mem.get(world.outbound(r), 0) == 0, f"{where}: rank {r} outbound"
        cursors = [world.mem.get(world.counters(r) + a, 0) for a in range(0, FUSED_A2A_CURSOR_BYTES, 4)]
        assert not any(cursors), f"{where}: rank {r} cursors {cursors}"


@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_band_launch(cluster_dim):
    for base, k in _kernels(cluster_dim):
        for world_size in _WORLDS:
            for seg_bands in _SEG_BANDS:
                world = _world(k, world_size, seg_bands)
                for n in range(2):
                    _check_launch(f"Kernel {base!r} W={world_size} segBands={seg_bands} launch {n}", world,
                                  world.launch())


@pytest.mark.parametrize("start", [SDMA_QUEUE_SIZE - 84, SDMA_QUEUE_SIZE - 80, SDMA_QUEUE_SIZE - 4])
def test_fused_rs_gfx1250_band_reservation_at_the_ring_end(start):
    for base, k in _kernels((2, 2)):
        for world_size in (2, 3):
            world = _world(k, world_size, 2, start)
            for n in range(2):
                _check_launch(f"Kernel {base!r} W={world_size} start {start} launch {n}", world, world.launch())


def _spin(g, rounds, where):
    for _ in range(rounds):
        try:
            next(g)
        except StopIteration:
            pytest.fail(f"{where}: finished")


@pytest.mark.parametrize("cluster_dim", [None, (2, 4)])
def test_fused_rs_gfx1250_band_drain_expectation(cluster_dim):
    for base, k in _kernels(cluster_dim):
        for seg_bands in _SEG_BANDS:
            full = (1 << seg_bands) - 1
            for world_size in (2, 3):
                rank = world_size - 1
                others = [j for j in range(world_size) if j != rank]
                world = _world(k, world_size, seg_bands)
                reg = world.work_groups(rank)[-1][3]
                assert reg["s[sgprFusedRSDst]"] == L.M32
                for missing in sorted({0, seg_bands - 1}):
                    where = f"Kernel {base!r} segBands={seg_bands} W={world_size} band {missing} missing"
                    world.mem[world.counter_addrs(rank)[2]] = world.n0 * world.n1 - 1
                    world.mem[world.outbound(rank)] = world_size - 1
                    slots = [_OWN if j == rank else full for j in range(world_size)]
                    slots[others[0]] &= ~(1 << missing)
                    for j in range(world_size):
                        world.mem[world.flags(rank) + 4 * j] = slots[j]
                    launch = L.Launch()
                    g = world.run_wg(launch, rank, reg, k.gates[0])
                    _spin(g, 100, where)
                    assert rank in launch.drains and rank not in launch.inbound, where
                    world.mem[world.flags(rank) + 4 * others[0]] = full
                    for _ in g:
                        pass
                    assert launch.inbound[rank] == [_OWN if j == rank else full for j in range(world_size)], where
                    flags = [world.mem[world.flags(rank) + 4 * j] for j in range(world_size)]
                    assert flags == [_OWN if j == rank else 0 for j in range(world_size)], f"{where}: {flags}"


@pytest.mark.parametrize("seg_bands", [3, 32])
@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_band_recv_wait(cluster_dim, seg_bands):
    full = (1 << seg_bands) - 1
    for base, k in _kernels(cluster_dim):
        band_rows = k.mt1 * k.ck
        for tree in k.trees:
            for world_size in _WORLDS:
                for rank in range(world_size):
                    band = seg_bands - 1 - (rank + world_size) % 3
                    where = f"Kernel {base!r} {tree} W={world_size} rank {rank} band {band}"
                    world = L.World(k, world_size, 1, seg_bands, drain=3)
                    reg = [r for _u, _w0, _w1, r in world.work_groups(rank) if r["s[sgprFusedRSDst]"] == L.M32
                           and r["s[sgprWorkGroup1]"] // k.ck == rank * seg_bands + band][-1]
                    others = [j for j in range(world_size) if j != rank]
                    flag = lambda j: world.flags(rank) + 4 * j
                    for j in others:
                        world.mem[flag(j)] = full & ~(1 << band)
                    tile = world.tile(rank, reg, k.trees[tree], lambda w, r, i: 1.0, 1.0, lambda s, m, n: 0.5)
                    for j in others:
                        assert not tile.run(200), f"{where}: passed the wait without band {band} of source {j}"
                        assert not tile.recv_reads and not tile.stores, f"{where}: read or stored before arrival"
                        assert tile.at[0] == "spin", f"{where}: wave 0 at {tile.at[0]}"
                        world.mem[flag(j)] = 1 << band if j != others[-1] else full
                    assert tile.run(_ROUNDS), f"{where}: did not finish"
                    assert set(tile.passes.values()) == {"RECV"} and not tile.oob and not tile.unacquired, where
                    assert ("poll" in {e for _w, e in tile.events}) == (world_size > 1), f"{where}: polls"
                    m0, n0 = reg["s[sgprWorkGroup0]"] * k.mt0, reg["s[sgprWorkGroup1]"] * k.mt1
                    first = rank * world.seg_n + band * band_rows
                    assert first <= n0 and n0 + k.mt1 <= first + band_rows, f"{where}: tile rows from {n0}"
                    want = sorted((src, m, n) for src in others
                                  for m in range(m0, m0 + k.mt0) for n in range(n0, n0 + k.mt1))
                    assert sorted(tile.recv_reads) == want, f"{where}: {len(tile.recv_reads)} receive reads"


def _block(k, start, end):
    return [(i, code, c) for i, (code, c) in enumerate(k.lines[start:end], start) if code]


@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_band_code(cluster_dim):
    for wait in (0, 1):
        for base, k in _kernels(cluster_dim, wait):
            sizes = Counter(int(code.split(",")[-1]) for code, c in k.lines if c == _RESERVE)
            assert sizes == {52: len(k.gates), 64: len(k.gates)}, f"Kernel {base!r}: reservations {sizes}"
            assert not [c for _code, c in k.lines if c == _BAND_LANE], f"Kernel {base!r}: band ATOMIC"
    for base, k in _kernels(cluster_dim):
        assert_assembles(k.src, base)
        sizes = Counter(int(code.split(",")[-1]) for code, c in k.lines if c == _RESERVE)
        assert sizes == {84: len(k.gates), 32: len(k.gates)}, f"Kernel {base!r}: reservations {sizes}"
        lanes = [i for i, (_code, c) in enumerate(k.lines) if c == _BAND_LANE]
        assert len(lanes) == len(k.gates), f"Kernel {base!r}: {len(lanes)} band ATOMICs"
        execs = [(j, code) for j, code, _c in _block(k, 0, len(k.lines)) if code.startswith("s_mov_b32 exec_lo")]
        for i in lanes:
            where = f"Kernel {base!r} line {i}"
            at = next(n for n, (j, _code) in enumerate(execs) if j == i)
            assert [code for _j, code in execs[at - 1:at + 3]] == [
                "s_mov_b32 exec_lo, 1", "s_mov_b32 exec_lo, 2", "s_mov_b32 exec_lo, -1", "s_mov_b32 exec_lo, 1"], (
                f"{where}: EXEC {execs[at - 1:at + 3]}")
            end = execs[at + 2][0]
            body = _block(k, i, end + 1)
            assert not [code for _j, code, _c in body if code.endswith(":") or "branch" in code], f"{where}: branch"
            stores = [c for _j, code, c in body if code.startswith("global_store")]
            assert stores == ["ring[addr + 0] = packet dwords 0..3", "ring[addr + 4] = packet dwords 4..7"], (
                f"{where}: {stores}")
            bit = next(code for _j, code, c in _block(k, i - 60, i) if c == "fused-RS: band bit").split()[1].rstrip(",")
            assert any(code.endswith(", " + bit) and c == "ATOMIC DW3: src_data lo (addend)"
                       for _j, code, c in body), f"{where}: the ATOMIC does not add {bit}"
        for wait in k.waits:
            block = _block(k, wait, k.wait_end(wait))
            bit = [code.split()[1].rstrip(",") for _j, code, c in block if c == "fused-RS: band bit"]
            assert len(bit) == 1, f"Kernel {base!r} wait {wait}: band bits {bit}"
            for poll in (L.RECV_POLL, L.RECV_RELOAD):
                at = next(n for n, (_j, _code, c) in enumerate(block) if c == poll)
                dst = re.match(r"global_load_b32 (\S+),", block[at][1]).group(1)
                ops = [code for _j, code, _c in block[at + 1:] if not code.startswith(("s_wait", "s_set", "s_nop"))]
                assert ops[:2] == ["v_and_b32 %s, %s, %s" % (dst, bit[0], dst),
                                   "v_cmp_ne_u32 vcc_lo, %s, %s" % (bit[0], dst)], f"Kernel {base!r} {poll}: {ops[:2]}"
