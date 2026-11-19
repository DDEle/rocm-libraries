# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) work-group remap characterization.

Runs the emitted FusedA2AWgRemap instructions on every work-group of a set of
grids, with and without a workgroup cluster, and checks the resulting tile
assignment.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_PLAIN = os.path.join(_COMM, "fused_a2a.yaml")
_CLUSTER = os.path.join(_COMM, "fused_a2a_cluster.yaml")

_REMAP_WAIT = "wait FusedAM for the WG remap"
_REMAP_END = "label_FusedA2AWgRemapEnd:"
_REMAP_LAST = "m += base"
_PUSH_GATE = "AM_tiles <= WorkGroup0?"
_SKIPPED = ("s_wait_", "s_set_vgpr_msb", "s_nop", "s_delay_alu", "s_load_b64")

# (NumWorkGroups0, NumWorkGroups1, AM_tiles)
_GRIDS = [(72, 8, 40), (16, 8, 8), (12, 2, 8), (12, 4, 8), (12, 8, 4), (16, 4, 0), (16, 4, 16), (24, 12, 12),
          (13, 4, 8), (12, 3, 8), (12, 4, 6), (12, 4, 7), (8, 8, 4), (4, 4, 4), (36, 20, 20)]


def _remap_body(src):
    raw = src.splitlines()
    wait = next(i for i, ln in enumerate(raw) if _REMAP_WAIT in ln)
    start = max(i for i in range(wait) if raw[i].startswith("s_load_b32"))
    last = _REMAP_END if any(ln.startswith(_REMAP_END) for ln in raw) else _REMAP_LAST
    end = next(i for i in range(wait, len(raw)) if last in raw[i])
    body = [ln.split("//")[0].strip() for ln in raw[start:end + 1]]
    return [ln for ln in body if ln and not ln.startswith(_SKIPPED)]


def _run_remap(body, n0, n1, am, wg0, wg1, xcc=0):
    """One lane of the remap: (WorkGroup0, WorkGroup1) after `body`."""
    reg = {"s[sgprNumWorkGroups0]": n0, "s[sgprNumWorkGroups1]": n1,
           "s[sgprWorkGroup0]": wg0, "s[sgprWorkGroup1]": wg1, "exec_lo": 1}
    labels = {ln[:-1]: i for i, ln in enumerate(body) if ln.endswith(":")}
    u32 = lambda v: v & 0xFFFFFFFF
    val = lambda x: reg[x] if x in reg else u32(int(x, 0))
    scalar = {
        "s_lshr_b32": lambda a, b: a >> (b & 31), "s_lshl_b32": lambda a, b: a << (b & 31),
        "s_or_b32": lambda a, b: a | b, "s_and_b32": lambda a, b: a & b,
        "s_mul_i32": lambda a, b: a * b, "s_add_u32": lambda a, b: a + b, "s_sub_u32": lambda a, b: a - b,
        "v_mul_lo_u32": lambda a, b: a * b, "v_sub_nc_u32": lambda a, b: a - b,
        "v_cmp_ge_u32": lambda a, b: int(a >= b),
    }
    scc, pc = 0, 0
    while pc < len(body):
        ln = body[pc]
        pc += 1
        if ln.endswith(":"):
            continue
        op, rest = ln.split(None, 1)
        a = [x.strip() for x in rest.split(",")]
        if op in scalar:
            reg[a[0]] = u32(scalar[op](val(a[1]), val(a[2])))
        elif op == "s_load_b32":
            reg[a[0]] = am
        elif op == "s_sendmsg_rtn_b32":
            reg[a[0]] = xcc << 16 | 0xA5
        elif op == "s_cmp_eq_u32":
            scc = int(val(a[0]) == val(a[1]))
        elif op == "s_cmp_lt_u32":
            scc = int(val(a[0]) < val(a[1]))
        elif op == "s_cmp_ge_i32":
            scc = int((val(a[0]) ^ 0x80000000) >= (val(a[1]) ^ 0x80000000))
        elif op == "s_branch":
            pc = labels[a[0]]
        elif op == "s_cselect_b32":
            reg[a[0]] = val(a[1]) if scc else val(a[2])
        elif op == "s_cbranch_scc0":
            pc = pc if scc else labels[a[0]]
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
            if reg["exec_lo"] & 1:
                reg[a[0]] = u32(val(a[1]) + val(a[2]))
        else:
            pytest.fail(f"remap instruction not modelled: {ln!r}")
    return reg["s[sgprWorkGroup0]"], reg["s[sgprWorkGroup1]"]


def _check_remap(base, src, cs, ck, aid=False):
    """`aid`: clusters are dispatched round-robin over 8 XCCs, 4 per AID."""
    body = _remap_body(src)
    mt0 = int(re.search(r"FusedAM >> log2\(MT0=(\d+)\)", src).group(1))
    remapped = halved = 0
    for n0, n1, a in _GRIDS:
        for first_xcc in (0, 5) if aid else (0,):
            cluster = lambda wg0, wg1: wg0 // cs + (wg1 // ck) * (n0 // cs)
            xcc = lambda wg0, wg1: (cluster(wg0, wg1) + first_xcc) % 8
            out = {(wg0, wg1): _run_remap(body, n0, n1, a * mt0, wg0, wg1, xcc(wg0, wg1))
                   for wg1 in range(n1) for wg0 in range(n0)}
            where = f"Kernel {base!r} grid {(n0, n1, a)} first XCC {first_xcc}"
            assert sorted(out.values()) == sorted(out), f"{where}: not a bijection"
            if (n0 | a) % cs or n1 % ck:
                assert all(k == v for k, v in out.items()), f"{where}: partial clusters must keep their tiles"
                continue
            remapped += any(k != v for k, v in out.items())
            for (wg0, wg1), tile in out.items():
                m, j = out[(wg0 - wg0 % cs, wg1 - wg1 % ck)]
                assert (m % cs, j % ck) == (0, 0) and tile == (m + wg0 % cs, j + wg1 % ck), (
                    f"{where}: work-group {(wg0, wg1)} left its cluster's tile block: {tile}")
            split = aid and (n0 // cs) * (n1 // ck) % 8 == 0 and (a // cs | n0 // cs) % 2 == 0
            halved += split
            streams = {}
            for wg1 in range(0, n1, ck):
                for wg0 in range(0, n0, cs):
                    key = (cluster(wg0, wg1) // 8, xcc(wg0, wg1) % 4) if split else cluster(wg0, wg1)
                    streams.setdefault(xcc(wg0, wg1) // 4 if split else 0, []).append((key, out[(wg0, wg1)][0]))
            for tiles in streams.values():
                push = [m < a for _key, m in sorted(tiles)]
                assert push == sorted(push, reverse=True), f"{where}: PUSH clusters are not dispatched first"
            if split:
                cols = [{m for _key, m in tiles} for tiles in streams.values()]
                assert len(cols) == 2 and not cols[0] & cols[1], f"{where}: the two AIDs share M tiles"
    assert remapped, f"Kernel {base!r}: no grid was remapped"
    assert halved or not aid, f"Kernel {base!r}: no grid was split between the AIDs"


def test_fused_a2a_gfx1250_wg_remap():
    results = emit_kernels_from_config(_PLAIN, limit=8, arch=_ARCH)
    assert results
    for base, src, err in results:
        assert err == 0
        _check_remap(base, src, 1, 1)


@pytest.mark.parametrize("cluster_dim", [[4, 2], [2, 4]])
def test_fused_a2a_gfx1250_cluster_wg_remap(cluster_dim, snapshot):
    results = emit_kernels_from_config(_CLUSTER, limit=8, arch=_ARCH, cluster_dim=cluster_dim)
    assert len(results) == 2, [b for (b, _s, _e) in results]
    aids = []
    for base, src, err in results:
        assert err == 0
        assert_assembles(src, base)
        epilogue = src[src.index(_PUSH_GATE):]
        assert not re.search(r"^s_barrier_\w+ -3", epilogue, re.M), (
            f"Kernel {base!r}: cluster barrier past the PUSH/local gate")
        aids.append(len(re.findall(r"^s_sendmsg_rtn_b32", src, re.M)))
        _check_remap(base, src, *cluster_dim, aid=bool(aids[-1]))
    assert sorted(aids) == [0, 1]
    assert sorted(b for (b, _s, _e) in results) == snapshot
