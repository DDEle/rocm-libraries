# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) token-tile band characterization.

Runs the emitted band arithmetic of the epilogue handshake (the FusedTokenTiles
latch, the election wrap limit, the counter index and the COPY row geometry)
for every peer and token-tile of a set of problems, and checks that each
(peer, band) elects one work-group and the COPY packets tile the N rows.
"""

import os
import re
from collections import defaultdict

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_PLAIN = os.path.join(_COMM, "fused_a2a.yaml")
_CLUSTER = os.path.join(_COMM, "fused_a2a_cluster.yaml")

_N = (1, 255, 256, 1023, 1024, 1025, 1792, 1800, 2048, 3000)
_WORLD = (1, 2, 4, 8)
_TILES_PER_RANK = (1, 3)

_LATCH = "= ceil(N / "
_WRAP_FIRST = "tilesPerRank = FusedNShard >> log2"
_WRAP_LAST = "wrap limit = "
_WRAP = ("tilesPerRank = ", "band = WorkGroup1", "band row = ", "N - band row", "rows in this band",
         "token-tiles in this band", ">> log2(MT1=", "tilesPerRank * token-tiles", _WRAP_LAST)
_INDEX_FIRST = "dst_rank * tokenTiles"
_INDEX_LAST = "counter index = "
_COPY = ("token row of tile j", "dst_slice = ", "N - j*MT1", "rect_y = min(")

_OPS = {
    "s_add_u32": lambda a, b: a + b, "s_sub_u32": lambda a, b: a - b,
    "s_mul_i32": lambda a, b: a * b, "s_min_u32": min,
    "s_lshr_b32": lambda a, b: a >> (b & 31), "s_lshl_b32": lambda a, b: a << (b & 31),
}


def _parse(ln):
    op, rest = ln.split("//")[0].strip().split(None, 1)
    return op, [x.strip() for x in rest.split(",")]


def _segments(src):
    """The band instructions of the first handshake, as (latch, wrap, index, copy) lists."""
    raw = src.splitlines()
    at = lambda what, start=0: next(i for i in range(start, len(raw)) if what in raw[i])
    latch = at(_LATCH)
    wrap0 = at(_WRAP_FIRST)
    wrap1 = at(_WRAP_LAST, wrap0)
    index = at(_INDEX_FIRST, wrap1)
    copy = [at(what, index) for what in _COPY]
    wrap = [ln for ln in raw[wrap0:wrap1 + 1] if any(what in ln.split("//", 1)[-1] for what in _WRAP)]
    return ([raw[latch], raw[latch + 1]], wrap,
            [raw[index], raw[at(_INDEX_LAST, index)]], [raw[i] for i in copy])


def _run(lines, reg):
    u32 = lambda v: v & 0xFFFFFFFF
    val = lambda x: reg[x] if x in reg else int(x, 0)
    for ln in lines:
        op, a = _parse(ln)
        assert op in _OPS, f"band instruction not modelled: {ln!r}"
        reg[a[0]] = u32(_OPS[op](val(a[1]), val(a[2])))
    return reg


def _check_bands(base, src, ck):
    mt0, mt1 = (int(x) for x in re.search(r"_MT(\d+)x(\d+)x", src).groups())
    latch, wrap, index, copy = _segments(src)
    dst_rank = _parse(index[0])[1][1]
    limit_reg, index_reg = _parse(wrap[-1])[1][0], _parse(index[-1])[1][0]
    row_reg, slice_reg, _rows_left, rect_y_reg = (_parse(ln)[1][0] for ln in copy)
    band_rows = mt1 * ck
    for world in _WORLD:
        for tiles_per_rank in _TILES_PER_RANK:
            n_shard = tiles_per_rank * mt0
            for n in _N:
                where = f"Kernel {base!r} W={world} nShard={n_shard} N={n}"
                inputs = {"s[sgprSizesFree+1]": n, "s[sgprFusedNShard]": n_shard}
                bands = _run(latch, dict(inputs))["s[sgprFusedTokenTiles]"]
                assert bands == -(-n // band_rows), f"{where}: {bands} bands"
                wgs, limits, packets = defaultdict(int), defaultdict(set), defaultdict(set)
                for peer in range(world):
                    for tile in range(-(-n // mt1)):
                        reg = _run(wrap, dict(inputs, **{"s[sgprWorkGroup1]": tile,
                                                         "s[sgprFusedTokenTiles]": bands}))
                        reg[dst_rank] = peer
                        reg = _run(index + copy, reg)
                        at = reg[index_reg]
                        wgs[at] += tiles_per_rank
                        limits[at].add(reg[limit_reg])
                        packets[at].add((peer, reg[row_reg], reg[rect_y_reg], reg[slice_reg]))
                assert sorted(wgs) == list(range(world * bands)), f"{where}: counter indices {sorted(wgs)}"
                for at, count in wgs.items():
                    assert limits[at] == {count - 1}, (
                        f"{where}: counter {at} takes {count} work-groups, wrap limit {limits[at]}")
                    assert len(packets[at]) == 1, f"{where}: counter {at} builds {packets[at]}"
                for peer in range(world):
                    rows = sorted((r, y) for (p, r, y, _s) in (next(iter(v)) for v in packets.values())
                                  if p == peer)
                    assert [r for r, _y in rows] == list(range(0, n, band_rows)), f"{where}: peer {peer} rows {rows}"
                    assert sum(y for _r, y in rows) == n, f"{where}: peer {peer} rect_y {rows}"
                slices = {s for v in packets.values() for (_p, _r, _y, s) in v}
                assert slices == {band_rows * n_shard}, f"{where}: dst_slice {slices}"


def test_fused_a2a_gfx1250_bands_without_cluster():
    results = emit_kernels_from_config(_PLAIN, limit=8, arch=_ARCH)
    assert results
    for base, src, err in results:
        assert err == 0
        _check_bands(base, src, 1)


@pytest.mark.parametrize("cluster_dim", [[4, 2], [2, 4]])
def test_fused_a2a_gfx1250_bands(cluster_dim):
    results = emit_kernels_from_config(_CLUSTER, limit=8, arch=_ARCH, cluster_dim=cluster_dim)
    assert len(results) == 2, [b for (b, _s, _e) in results]
    for base, src, err in results:
        assert err == 0
        assert_assembles(src, base)
        _check_bands(base, src, cluster_dim[1])
