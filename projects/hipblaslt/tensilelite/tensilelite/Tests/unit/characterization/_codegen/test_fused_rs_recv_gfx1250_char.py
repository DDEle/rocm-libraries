# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) RECV wait and reduce (FusedRSWait=1) characterization.

Runs every wave of one work-group (fused_rs_launch.Tile) from the start of each store code tree through
the RECV wait, the receive-slot reduce and the D stores, against modelled flags and receive buffers, for
every rank of several world sizes, and checks the wait, the load waits, the receive addresses and the
stored values.
"""

import functools

import pytest

import fused_rs_launch as L
from config_harness import assert_assembles

pytestmark = pytest.mark.unit

_CLUSTERS = [None, (2, 2), (2, 4)]
_WORLDS = [1, 2, 3, 4]
_OWN_FLAG = 7
_ROUNDS = 10 ** 6
_BIG = 2.0 ** 20


def _kernels(cluster_dim, wait):
    results = L.emit(cluster_dim, wait)
    assert results and all(err == 0 for _b, _s, err in results)
    return [(base, L.Kernel(src, cluster_dim)) for base, src, _err in results]


def _alpha(tree):
    return 1.0 if "OptNLL" in tree else 2.0


def _acc(fraction):
    """Accumulator values; with `fraction`, an odd multiple of 2**-7 is added (not exact in BF16)."""
    return lambda wave, r, lane: float((r * 7 + lane * 13 + wave * 29) % 61 - 30) + (
        ((r * 5 + lane * 3 + wave) % 64 * 2 + 1) * 2.0 ** -7 if fraction else 0.0)


def _recv_value(src, m, n):
    """BF16-exact; every fourth element is +-2**20 by the parity of src."""
    if (m + n) % 4 == 0:
        return _BIG if src % 2 else -_BIG
    return ((src * 37 + m * 3 + n * 5) % 29 - 14) * 0.25


def _fsum(values):
    x = values[0]
    for y in values[1:]:
        x = L.f32(L.f32_bits(x + y))
    return x


def _stored(world, rank, tile):
    """{(m, n): BF16 bits} of the D stores of the tile."""
    out = {}
    for _wave, addr in tile.stores:
        rel = addr - world.d(rank)
        n, byte = divmod(rel, world.m * 2)
        word = world.mem[addr]
        out[(byte // 2, n)] = word & 0xFFFF
        out[(byte // 2 + 1, n)] = word >> 16
    return out


def _origin(k, reg):
    return reg["s[sgprWorkGroup0]"] * k.mt0, reg["s[sgprWorkGroup1]"] * k.mt1


@functools.lru_cache(maxsize=None)
def _position(cluster_dim, base, tree):
    """{(dm, dn): (wave, register, lane)} of the accumulator a tile element is stored from, from W=1 launches
    (no wait, no source to add) whose accumulators hold wave, register % 256, register // 256 and lane."""
    k = dict(_kernels(cluster_dim, 1))[base]
    parts = []
    for part in (lambda w, r, i: w, lambda w, r, i: r % 256, lambda w, r, i: r // 256, lambda w, r, i: i):
        world = L.World(k, 1, 1, 1, drain=3)
        _u, _w0, _w1, reg = world.work_groups(0)[0]
        tile = world.tile(0, reg, k.trees[tree], lambda w, r, i, part=part: float(part(w, r, i)), 1.0, _recv_value)
        assert tile.run(_ROUNDS) and set(tile.passes.values()) == {"RECV"}
        assert not tile.oob and not tile.recv_reads and not [e for e in tile.events if e[1] != "barrier"]
        m0, n0 = _origin(k, reg)
        parts.append({(m - m0, n - n0): int(L.f32(v << 16)) for (m, n), v in _stored(world, 0, tile).items()})
    assert set(parts[0]) == {(m, n) for m in range(k.mt0) for n in range(k.mt1)}, "D stores do not tile the tile"
    pos = {e: (parts[0][e], parts[1][e] + 256 * parts[2][e], parts[3][e]) for e in parts[0]}
    assert len(set(pos.values())) == len(pos), "two tile elements stored from one accumulator"
    return pos


def _check_order(where, k, tile, world_size):
    for wave in range(k.waves):
        ev = [kind for w, kind in tile.events if w == wave]
        if world_size == 1:
            assert ev == [], f"{where}: wave {wave} events {ev}"
        elif wave == 0:
            assert ev[-2:] == ["inv", "barrier"] and "poll" in ev, f"{where}: wave 0 events end {ev[-3:]}"
            assert ev.count("inv") == 1 and ev.count("barrier") == 1, f"{where}: wave 0 events {set(ev)}"
        else:
            assert ev == ["barrier"], f"{where}: wave {wave} events {ev}"
    assert not tile.unacquired, f"{where}: {len(tile.unacquired)} receive reads before global_inv retired"


@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_recv_wait_and_reduce(cluster_dim):
    rounding = {"own rounded first": 0, "sources added first": 0}
    for base, k in _kernels(cluster_dim, 1):
        for tree in k.trees:
            alpha = _alpha(tree)
            pos = _position(cluster_dim, base, tree)
            for world_size in _WORLDS:
                for rank in range(world_size):
                    where = f"Kernel {base!r} {tree} W={world_size} rank {rank}"
                    world = L.World(k, world_size, 1, 1, drain=3)
                    wgs = world.work_groups(rank)
                    recv = [reg for _u, _w0, _w1, reg in wgs if reg["s[sgprFusedRSDst]"] == L.M32]
                    others = [j for j in range(world_size) if j != rank]
                    flag = lambda j: world.flags(rank) + 4 * j
                    world.mem[flag(rank)] = _OWN_FLAG
                    for j in others[:-1]:
                        world.mem[flag(j)] = 1
                    reg = recv[-1]
                    tile = world.tile(rank, reg, k.trees[tree], _acc(True), alpha, _recv_value)
                    if others:
                        assert not tile.run(200), f"{where}: passed the wait without source {others[-1]}"
                        assert not tile.recv_reads and not tile.stores, f"{where}: read or stored before arrival"
                        assert tile.at[0] == "spin" and all(tile.at[w] == "barrier" for w in range(1, k.waves)), (
                            f"{where}: waves at {tile.at}")
                        world.mem[flag(others[-1])] = 1
                    assert tile.run(_ROUNDS), f"{where}: did not finish"
                    assert set(tile.passes.values()) == {"RECV"}, f"{where}: {tile.passes}"
                    assert not tile.oob, f"{where}: out-of-range accesses {tile.oob[:4]}"
                    _check_order(where, k, tile, world_size)

                    m0, n0 = _origin(k, reg)
                    assert world.seg_n * rank <= n0 and n0 + k.mt1 <= world.seg_n * (rank + 1), where
                    want = sorted((src, m, n) for src in others
                                  for m in range(m0, m0 + k.mt0) for n in range(n0, n0 + k.mt1))
                    assert sorted(tile.recv_reads) == want, (
                        f"{where}: {len(tile.recv_reads)} receive reads, {len(want)} expected; own slot read: "
                        f"{any(src == rank for src, _m, _n in tile.recv_reads)}")

                    stored = _stored(world, rank, tile)
                    assert set(stored) == {(m, n) for m in range(m0, m0 + k.mt0)
                                           for n in range(n0, n0 + k.mt1)}, f"{where}: D stores"
                    for (m, n), got in stored.items():
                        x0 = L.f32(L.f32_bits(alpha * _acc(True)(*pos[(m - m0, n - n0)])))
                        recv = [_recv_value(src, m, n) for src in others]
                        want = L.bf16(L.f32_bits(_fsum([x0] + recv)))
                        assert got == want, f"{where}: D[{m}, {n}] = {got:#06x}, want {want:#06x}"
                        if others:
                            rounded = L.f32(L.bf16(L.f32_bits(x0)) << 16)
                            rounding["own rounded first"] += want != L.bf16(L.f32_bits(_fsum([rounded] + recv)))
                            rounding["sources added first"] += want != L.bf16(L.f32_bits(_fsum(recv + [x0])))

                    if world_size > 1:
                        push = next(r for _u, _w0, _w1, r in wgs if r["s[sgprFusedRSDst]"] != L.M32)
                        for j in range(world_size):
                            world.mem[flag(j)] = 0
                        tile = world.tile(rank, push, k.trees[tree], _acc(False), alpha, _recv_value)
                        assert tile.run(_ROUNDS) and set(tile.passes.values()) == {"PUSH"}, f"{where}: PUSH unit"
                        assert not tile.events and not tile.recv_reads, f"{where}: PUSH unit waited or read"
    assert all(rounding.values()), f"the reference does not tell the summation apart: {rounding}"


def _wait_block(k, start):
    """(code, comment) from the RECV wait at `start` to its end label."""
    return [(code, c) for code, c in k.lines[start:k.wait_end(start)] if code]


@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_recv_wait_code(cluster_dim):
    for base, k in _kernels(cluster_dim, 0):
        assert not [c for _code, c in k.lines if c in (L.RECV_WAIT, L.RECV_POLL, L.RECV_LOAD)], (
            f"Kernel {base!r}: FusedRSWait=0 waits or reduces")
    for base, k in _kernels(cluster_dim, 1):
        assert_assembles(k.src, base)
        waits = [i for i, (_code, c) in enumerate(k.lines) if c == L.RECV_WAIT]
        trees = sorted(k.trees.values())
        assert len(waits) == len(trees), f"Kernel {base!r}: {len(waits)} RECV waits, {len(trees)} store trees"
        for tree, wait in zip(trees, waits):
            first_gate = min(g for g in k.gates if g > tree)
            assert tree < wait < first_gate, f"Kernel {base!r}: wait at {wait} not before the gates of {tree}"
            codes = [code for code, _c in _wait_block(k, wait)]
            inv = codes.index("global_inv scope:SCOPE_SYS")
            first = lambda p, at=0: next(i for i, c in enumerate(codes[at:], at) if c.startswith(p))
            order = [first("global_load_b32"), first("s_sleep"), inv, first("s_wait_loadcnt 0", inv),
                     first("s_barrier_signal -1", inv), first("s_barrier_wait -1", inv)]
            assert order == sorted(order), f"Kernel {base!r}: wait order {order}"
            assert "scope:SCOPE_SYS" in codes[order[0]], codes[order[0]]
            assert sum(c.startswith("global_inv") for c in codes) == 1
            assert sum(c.startswith("s_barrier_wait") for c in codes) == 1
        for gate in k.gates:
            push, recv = k.passes(gate)
            assert not [c for _code, c in push if c == L.RECV_LOAD], f"Kernel {base!r}: reduce in the PUSH pass"
            stores = [code for code, c in recv if c == "store D"]
            assert stores and not any("scope:SCOPE_DEV" in code for code in stores), f"Kernel {base!r}: RECV store"
