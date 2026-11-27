# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+RS (FusedA2A: GemmRS) PUSH pass characterization.

Runs the emitted store-pass gate and handshake of every work-group of every rank (fused_rs_launch)
and checks the SDMA COPY packets, the queue tails and the counter wrap limits; checks the two store
passes the gate selects between.
"""

import re
from collections import defaultdict

import pytest

import fused_rs_launch as L
from config_harness import assert_assembles

pytestmark = pytest.mark.unit

_CLUSTERS = [None, (2, 2), (4, 2), (2, 4)]
_WORLDS = [1, 2, 3, 4]
# (dispatch units per band, bands per segment)
_GRIDS = [(1, 1), (2, 2), (1, 3)]
_D_STORE = "store D"
_ELECT = "wave 0 elects the WG's single writer"
_TALLY = "old3 = atomic_inc(counter3), wrap at FusedTotalWGs-1, return pre-op"
_NO_SDST = ("s_cmp", "s_cbranch", "s_branch", "s_wait", "s_barrier", "s_nop", "s_set", "s_delay", "s_prefetch",
            "s_sleep", "s_endpgm", "s_store", "s_buffer_store", "s_atomic", "s_dcache")


def _kernels(cluster_dim, wait=0):
    results = L.emit(cluster_dim, wait)
    assert results
    for base, _src, err in results:
        assert err == 0, base
    return [(base, L.Kernel(src, cluster_dim)) for base, src, _err in results]


def _check_copies(where, world, launch):
    k, w, m, seg_n = world.k, world.w, world.m, world.seg_n
    band_rows = k.mt1 * k.ck
    assert len(launch.copies) == w * (w - 1) * world.seg_bands, f"{where}: {len(launch.copies)} COPYs"
    rows = defaultdict(list)
    for r, j, f in launch.copies:
        at = f"{where}: COPY rank {r} -> {j} {f}"
        assert j != r, at
        assert f["rect_x"] * L.PACKET_BYTES == 2 * m and f["rect_y"] == band_rows, at
        assert f["src_pitch"] * L.PACKET_BYTES == 2 * m and f["dst_pitch"] * L.PACKET_BYTES == 2 * m, at
        assert f["src_slice"] == f["dst_slice"] == f["rect_x"] * f["rect_y"], at
        token, rem = divmod(f["src"] - world.d(r), 2 * m)
        assert rem == 0 and token % band_rows == 0, at
        assert j * seg_n <= token and token + band_rows <= (j + 1) * seg_n, f"{at}: token {token} outside segment {j}"
        slot_row, rem = divmod(f["dst"] - world.recv(j), 2 * m)
        assert rem == 0 and slot_row == r * seg_n + token - j * seg_n, f"{at}: receive row {slot_row}"
        rows[(j, r)].append(token)
    for j in range(w):
        for r in range(w):
            want = [] if r == j else list(range(j * seg_n, (j + 1) * seg_n, band_rows))
            assert sorted(rows.get((j, r), [])) == want, f"{where}: rank {r} sends {rows.get((j, r))} to {j}"


def _check_queues(where, world, launch, wait):
    """FusedRSWait 0, 1: the band COPYs, then +1 to the owner's flag slot and +1 to the outbound counter.
    FusedRSWait 2: each band COPY followed by +(1 << band in the segment) to the owner's flag slot, then +1 to
    the outbound counter."""
    band_rows = world.k.mt1 * world.k.ck
    for r in range(world.w):
        for j in range(world.w):
            q = launch.queues.get((r, j), [])
            if j == r:
                assert q == [], f"{where}: rank {r} uses its own queue"
                continue
            at = f"{where}: queue {(r, j)}"
            assert q[-1] == ("ATOMIC", world.outbound(r), 1), f"{at} ends {q[-1]}"
            if wait != 2:
                assert [p[0] for p in q] == ["COPY"] * world.seg_bands + ["ATOMIC"] * 2, at
                assert q[-2] == ("ATOMIC", world.flags(j) + 4 * r, 1), f"{at} tail {q[-2]}"
                continue
            assert [p[0] for p in q] == ["COPY", "ATOMIC"] * world.seg_bands + ["ATOMIC"], at
            bits = []
            for (_c, f), atomic in zip(q[:-1:2], q[1:-1:2]):
                band = ((f["src"] - world.d(r)) // (2 * world.m) - j * world.seg_n) // band_rows
                assert atomic == ("ATOMIC", world.flags(j) + 4 * r, 1 << band), f"{at}: band {band} {atomic}"
                bits.append(band)
            assert sorted(bits) == list(range(world.seg_bands)), f"{at}: bands {bits}"


def _check_counters(where, world, launch):
    k = world.k
    expect = {}
    for r in range(world.w):
        c1, c2, c3 = world.counter_addrs(r)
        expect[c3] = world.n0 * world.n1
        for j in set(range(world.w)) - {r}:
            expect[c2 + 4 * j] = world.seg_bands
            for band in range(j * world.seg_bands, (j + 1) * world.seg_bands):
                expect[c1 + 4 * band] = world.n0 * k.ck
    limits = defaultdict(list)
    for _r, addr, limit in launch.incs:
        limits[addr].append(limit)
    assert {a: len(v) for a, v in limits.items()} == expect, f"{where}: counter increments"
    for addr, v in limits.items():
        assert set(v) == {len(v) - 1}, f"{where}: counter {addr:#x}: {len(v)} increments, limits {set(v)}"
        assert world.mem.get(addr, 0) == 0, f"{where}: counter {addr:#x} left at {world.mem[addr]}"


@pytest.mark.parametrize("wait", [0, 1, 2])
@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_push(cluster_dim, wait):
    for base, k in _kernels(cluster_dim, wait):
        for world_size in _WORLDS:
            for r_units, seg_bands in _GRIDS:
                world = L.World(k, world_size, r_units, seg_bands, drain=3)
                for n in range(2):
                    where = f"Kernel {base!r} W={world_size} grid {(r_units, seg_bands)} launch {n}"
                    launch = world.launch()
                    _check_copies(where, world, launch)
                    _check_queues(where, world, launch, wait)
                    _check_counters(where, world, launch)


@pytest.mark.parametrize("wait", [0, 1, 2])
@pytest.mark.parametrize("cluster_dim", _CLUSTERS)
def test_fused_rs_gfx1250_store_passes(cluster_dim, wait):
    for base, k in _kernels(cluster_dim, wait):
        assert_assembles(k.src, base)
        for gate in k.gates:
            push, recv = k.passes(gate)
            comments = lambda lines: {c for _code, c in lines}
            codes = lambda lines: " ".join(code for code, _c in lines)
            assert {"fused-RS: old1 = atomic_inc(counter1[band])", "fused-RS COPY: token row = band * %d"
                    % (k.mt1 * k.ck)} <= comments(push), f"Kernel {base!r}: PUSH pass without election or COPY"
            assert not {L.DRAIN_START, L.POLL, L.SEND_POLL} & comments(push), f"Kernel {base!r}: DRAIN in PUSH"
            assert "fusedRS_drain" not in codes(push), f"Kernel {base!r}: DRAIN label in the PUSH pass"
            assert {L.DRAIN_START, L.POLL, _TALLY} <= comments(recv)
            assert not [c for c in comments(recv) if "counter1" in c or "counter2" in c or "COPY" in c], (
                f"Kernel {base!r}: election or COPY in the RECV pass")
            assert "label_sdma" not in codes(recv), f"Kernel {base!r}: SDMA ring code in the RECV pass"
            push_stores = [code for code, c in push if c == _D_STORE]
            recv_stores = [code for code, c in recv if c == _D_STORE]
            assert push_stores and len(push_stores) == len(recv_stores), f"Kernel {base!r}: D stores"
            assert all("scope:SCOPE_DEV" in code for code in push_stores), f"Kernel {base!r}: PUSH store scope"
            assert not any("scope:SCOPE_DEV" in code for code in recv_stores), f"Kernel {base!r}: RECV store scope"
            _check_push_fence(f"Kernel {base!r} gate line {gate}", push)
            for lines in (push, recv):
                _check_handshake_inputs(f"Kernel {base!r} gate line {gate}", k, lines)


def _check_push_fence(where, push):
    codes = [code for code, _c in push if code]
    start = next(i for i, (_code, c) in enumerate(push) if c == L.HANDSHAKE_START)
    elect = next(i for i, (_code, c) in enumerate(push) if i > start and c == _ELECT)
    fence = [code for code, _c in push[start:elect] if code.startswith(("s_wait_storecnt", "s_barrier"))]
    assert fence == ["s_wait_storecnt 0", "s_barrier_signal -1", "s_barrier_wait -1"], f"{where}: {fence}"
    assert codes.count("s_barrier_wait -1") == 1, f"{where}: PUSH pass barriers"


def _check_handshake_inputs(where, k, lines):
    """No instruction from the gate to the handshake writes an SGPR the handshake reads on entry."""
    inputs = set()
    for name, width in L.HANDSHAKE_SGPRS:
        inputs |= set(range(k.sreg("sgpr" + name), k.sreg("sgpr" + name) + width))
    end = next(i for i, (_code, c) in enumerate(lines) if c == L.HANDSHAKE_START)
    for code, _c in lines[:end]:
        op, _, rest = code.partition(" ")
        if not op.startswith(("s_", "v_readfirstlane", "v_cmp")) or op.startswith(_NO_SDST) or not rest:
            continue
        m = re.fullmatch(r"s(\d+)|s\[([^\]:]+)(?::([^\]]+))?\]", rest.split(",")[0].strip())
        if not m:
            continue
        lo = int(m.group(1)) if m.group(1) else k.eval(m.group(2))
        hi = k.eval(m.group(3)) if m.group(3) else lo
        assert not inputs & set(range(lo, hi + 1)), f"{where}: {code!r} writes a handshake input"

