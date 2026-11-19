# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 A2A+GEMM (FusedA2A: A2AGemm) codegen characterization under FusedA2ATransport=View.

Emits the a2a_gemm_loop.yaml solutions with FusedA2ATransport=View added to every
fork and checks that each kernel assembles, carries _FAATV_ in its name, stays within
the gfx950 SGPR budget, emits no SDMA packet, ring, election, tally or done-block code,
that the setup loads FusedNumCu and sets A2AViewK = S * (iB * G * Qb + (b - iB * nb) / G) and
A2AViewStep = S * Qb - (S-1), G = max(W-1, 1), B = NumWorkGroups1, F = NumWorkGroups0,
b = WorkGroup1, nb = clamp(FusedNumCu / F, 1, B), iB = b / nb, Qb = ceil(nb / G), and that every
segment switch polls flag[A2AShardIdx - 1]
with sc0 sc1, compares it unsigned against A2AViewK + 1 with no scalar load of the view block and
no subtraction in between, and then advances A2AViewK by A2AViewStep after a segment's last
sub-round, else by 1, outside the main loop body.
"""

import functools
import math
import os
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Components.Signature import fusedA2AKernArgLayout

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_MAX_SGPRS = 102

_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx950", "a2a_gemm_loop.yaml",
    )
)
_NAME = re.compile(r"\.amdhsa_kernel (\S+)")
_NEXT_FREE_SGPR = re.compile(r"^\s*\.amdhsa_next_free_sgpr (\d+)\b", re.M)
_GROUPS = [0, 1, 2, 3, 4, 5]
_LIMIT = 24

_FORBIDDEN = (
    "0x80000401", str(0x80000401),
    "0x1e00000a", "0x1E00000A", str(0x1E00000A),
    "s_atomic",
    "global_store", "s_store", "flat_store",
    "A2AFlagRow", "A2ABlockLo", "A2ABlockCount", "A2ADrain",
    "label_A2ASkipEnqueue", "label_a2a_exit_tally_end", "label_a2a_drain_send_end",
)

_SERIAL_LANE0 = re.compile(
    r"^(?:v_readfirstlane_b32 s\d+, v\[vgprSerial\]"
    r"|v_mov_b32 v\[vgprSerial\], s\d+"
    r"|global_load_dword v\[vgprSerial\], v\[vgprSerial\], s\[sgprFusedCounterPtr:sgprFusedCounterPtr\+1\] sc0 sc1 )")


@pytest.fixture(scope="module")
def view_config(tmp_path_factory):
    with open(_CONFIG) as fh:
        config = yaml.safe_load(fh)
    for problem in config["BenchmarkProblems"]:
        for group in problem[1:]:
            group["ForkParameters"].append({"FusedA2ATransport": ["View"]})
    path = tmp_path_factory.mktemp("a2a_gemm_view") / "a2a_gemm_view.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return str(path)


@functools.lru_cache(maxsize=None)
def _emit(config, group):
    return emit_kernels_from_config(config, limit=_LIMIT, arch=_ARCH, size_group_index=group)


def _param(name, key):
    return int(re.search(rf"_{key}(\d+)_", name).group(1))


def _switches(src):
    """Bodies of the View segment switches, from the flag index to the EXEC restore."""
    return re.findall(r"^s_sub_u32 s\d+, s\[sgprA2AShardIdx\], 1 +// i = segment - 1$.*?"
                      r"^label_A2AFlagReady\w*:\n[^\n]*\n^s_mov_b64 exec, -1\b[^\n]*",
                      src, re.M | re.S)


def _divide(dividend, divisor, quotient, remainder=None):
    """scalarUInt24DivideAndRemainder of dividend by divisor, its quotient read into `quotient`
    (None: not read) and its remainder into `remainder` (None: not read)."""
    tail = (rf"^v_readfirstlane_b32 {quotient}, v\d+ +// quotient\n" if quotient else r"^s_nop 0\n") \
         + (rf"^v_readfirstlane_b32 {remainder}, v\d+ +// remainder\n" if remainder else "")
    return (rf"^v_cvt_f64_u32 v\[\d+:\d+\], {divisor} +//.*\n"
            r"^v_rcp_f64 v\[\d+:\d+\], v\[\d+:\d+\] +//.*\n"
            rf"^v_cvt_f64_u32 v\[\d+:\d+\], {dividend} +//.*\n"
            r"(?:^v_.*\n|^s_mov_b64 exec, -1 .*\n)*?"
            + tail)


def _check_setup(base, src, subRounds):
    """nb, iB, b - bLo and Qb from FusedNumCu, NumWorkGroups0/1 and WorkGroup1, then
    A2AViewK = S * (iB * G * Qb + (b - bLo) / G) and A2AViewStep = S * Qb - (S-1), after b % G."""
    layout = fusedA2AKernArgLayout()
    loads = dict(re.findall(r"^s_load_dword s\[sgpr(A2AFeatIdx|A2AViewStep)\], "
                            r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)$", src, re.M))
    assert set(loads) == {"A2AFeatIdx", "A2AViewStep"} and \
        int(loads["A2AViewStep"], 16) - int(loads["A2AFeatIdx"], 16) == \
        layout["FusedNumCu"] - layout["FusedMyRank"], f"Kernel {base!r}: A2AViewStep is not loaded from FusedNumCu"
    log2S = int(math.log2(subRounds))
    scale = (rf"^s_lshl_b32 s\[sgprA2AViewK\], s\[sgprA2AViewK\], {log2S} +// K = S \* .*\n"
             rf"^s_lshl_b32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], {log2S} +// S \* Qb\n"
             rf"^s_sub_u32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], {subRounds - 1} +// step = S \* Qb - \(S-1\)\n"
             ) if subRounds > 1 else ""
    setup = re.findall(
        r"^s_sub_u32 (?P<g>s\d+), s\[sgprA2AW\], 1 +// W - 1\n"
        r"^s_max_u32 (?P=g), (?P=g), 1 +// peer queue count, at least 1\n"
        + _divide(r"s\[sgprWorkGroup1\]", r"(?P=g)", None, r"s\[sgprA2AShardIdx\]")
        + _divide(r"s\[sgprA2AViewStep\]", r"s\[sgprNumWorkGroups0\]", r"s\[sgprA2AViewStep\]")
        + r"^s_min_u32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], s\[sgprNumWorkGroups1\] +//.*\n"
        r"^s_max_u32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], 1 +//.*\n"
        + _divide(r"s\[sgprWorkGroup1\]", r"s\[sgprA2AViewStep\]", r"s\[sgprA2AViewK\]", r"(?P<off>s\d+)")
        + _divide(r"(?P=off)", r"(?P=g)", r"(?P=off)")
        + r"^s_add_u32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], (?P=g) +// nb \+ \(W-1\)\n"
        r"^s_sub_u32 s\[sgprA2AViewStep\], s\[sgprA2AViewStep\], 1 +// nb \+ \(W-1\) - 1\n"
        + _divide(r"s\[sgprA2AViewStep\]", r"(?P=g)", r"s\[sgprA2AViewStep\]")
        + r"^s_mul_i32 s\[sgprA2AViewK\], s\[sgprA2AViewK\], (?P=g) +//.*\n"
        r"^s_mul_i32 s\[sgprA2AViewK\], s\[sgprA2AViewK\], s\[sgprA2AViewStep\] +//.*\n"
        r"^s_add_u32 s\[sgprA2AViewK\], s\[sgprA2AViewK\], (?P=off) +//.*\n"
        + scale
        + r"^s_add_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardIdx\], 1 +// gathered segment = peer queue \+ 1$",
        src, re.M)
    assert len(setup) == 1, \
        f"Kernel {base!r}: setup does not compute A2AViewK = S * (iB * G * Qb + (b - bLo) / G) " \
        "and A2AViewStep = S * Qb - (S-1)"


def _check_switch(base, switch, subRounds):
    slot = re.match(r"s_sub_u32 (s\d+), s\[sgprA2AShardIdx\], 1 ", switch).group(1)
    assert re.search(rf"^s_lshl_b32 {slot}, {slot}, 2 +// flag\[i\] byte offset$", switch, re.M), \
        f"Kernel {base!r}: flag index is not (A2AShardIdx - 1) * 4"
    polls = re.findall(rf"^v_mov_b32 v\[vgprSerial\], {slot} +// poll offset\n"
                       r"^global_load_dword v\[vgprSerial\], v\[vgprSerial\], "
                       r"s\[sgprFusedCounterPtr:sgprFusedCounterPtr\+1\] sc0 sc1 +//", switch, re.M)
    assert len(polls) == 2, f"Kernel {base!r}: flag[i] is not polled at offset 0 with sc0 sc1, once plus one retry"

    scalarLoads = re.findall(r"^s_load\w* .*$", switch, re.M)
    assert not [line for line in scalarLoads if "FusedCounterPtr" in line or " glc" in line], \
        f"Kernel {base!r}: segment switch reads the view block or bypasses the cache with a scalar load: {scalarLoads}"

    advance = r"^s_sub_u32 s\[sgprA2AShardCounter\], s\[sgprA2AShardCounter\], 1 +// segments left\n" \
              r"^s_add_u32 (?P<need>s\d+), s\[sgprA2AViewK\], 1 +// k \+ 1\n" \
              r"^s_add_u32 s\[sgprA2AViewK\], s\[sgprA2AViewK\], {step} +// K of the next unit$"
    if subRounds > 1:
        h = re.search(rf"^s_and_b32 (s\d+), \1, {subRounds - 1} +// sub-round = ", switch, re.M).group(1)
        need = re.search(rf"^s_cmp_eq_u32 {h}, {subRounds - 1} +// last sub-round\?\n"
                         r"^s_cselect_b32 s\[sgprA2AShardIdx\], s\d+, s\[sgprA2AShardIdx\] +//.*\n"
                         r"^s_cselect_b32 (?P<step>s\d+), s\[sgprA2AViewStep\], 1 +//.*\n"
                         + advance.format(step=r"(?P=step)"), switch, re.M)
    else:
        need = re.search(advance.format(step=r"s\[sgprA2AViewStep\]"), switch, re.M)
    assert need, f"Kernel {base!r}: k + 1 is not A2AViewK + 1, or A2AViewK does not step by A2AViewStep / 1"
    need = need.group("need")
    assert "WorkGroup1" not in switch, f"Kernel {base!r}: segment switch reads WorkGroup1"

    compares = re.findall(r"^v_readfirstlane_b32 (s\d+), v\[vgprSerial\] +// flag\[i\]\n"
                          rf"^(s_cmp_\w+) \1, {need} +//", switch, re.M)
    assert [cmp for _reg, cmp in compares] == ["s_cmp_ge_u32", "s_cmp_lt_u32"], \
        f"Kernel {base!r}: flag[i] is not compared unsigned against k + 1: {compares}"
    assert re.search(r"^s_sleep 1\b", switch, re.M), f"Kernel {base!r}: the retry does not back off"

    touched = [line for line in switch.splitlines()
               if re.match(r"(v_|ds_|buffer_|global_|s_barrier)", line) and not _SERIAL_LANE0.match(line)]
    assert not touched, f"Kernel {base!r}: segment switch emits {touched}"
    save = re.search(r"^s_mov_b64 exec, 1\b.*\n^v_readfirstlane_b32 (s\d+), v\[vgprSerial\]", switch, re.M)
    assert save and re.search(rf"^label_A2AFlagReady\w*:\n^v_mov_b32 v\[vgprSerial\], {save.group(1)}\b.*\n"
                              r"^s_mov_b64 exec, -1\b", switch, re.M), \
        f"Kernel {base!r}: segment switch does not restore Serial lane 0 and EXEC"


@pytest.mark.parametrize("group", _GROUPS)
def test_a2a_gemm_view_gfx950(view_config, group):
    results = _emit(view_config, group)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [(b, e) for (b, _s, e) in results if e != 0]
    for base, src, _err in results:
        assert_assembles(src, base)
        name = _NAME.search(src).group(1)
        assert "_FAATV_" in name, f"Kernel {base!r}: no _FAATV_ in the kernel name"
        sgprs = _NEXT_FREE_SGPR.findall(src)
        assert len(sgprs) == 1 and int(sgprs[0]) <= _MAX_SGPRS, \
            f"Kernel {base!r}: .amdhsa_next_free_sgpr {sgprs}, limit {_MAX_SGPRS}"
        found = [f for f in _FORBIDDEN if f in src]
        assert not found, f"Kernel {base!r}: View kernel emits {found}"
        assert re.search(r"^\.set sgprA2AViewK, \d+$", src, re.M) and \
            re.search(r"^\.set sgprA2AViewStep, \d+$", src, re.M), f"Kernel {base!r}: A2AViewK/A2AViewStep not defined"

        subRounds = _param(name, "AASR")
        _check_setup(base, src, subRounds)
        pgr = _param(name, "PGR")
        subtile = _param(name, "USI")
        eps = 0 if subtile else _param(name, "EPS")
        ulsgro = 0 if subtile else _param(name, "ULSGRO")
        switches = _switches(src)
        assert len(switches) == max(pgr - 1, 0) + 1 + eps + ulsgro, \
            f"Kernel {base!r}: {len(switches)} segment switches, PGR={pgr} EPS={eps} ULSGRO={ulsgro}"
        assert src.count("label_A2AWaitFlag") == 2 * len(switches), \
            f"Kernel {base!r}: a flag wait outside the segment switches"
        for switch in switches:
            _check_switch(base, switch, subRounds)
        writesK = re.findall(r"^\S+ s\[sgprA2AViewK\],", src, re.M)
        assert len(writesK) == (5 if subRounds > 1 else 4) + len(switches), \
            f"Kernel {base!r}: A2AViewK written outside the setup and the segment switches"

        # Non-subtile EPS/ULSGRO kernels switch at the first loop copy's exit, between label_LoopBeginL
        # and label_LoopEndL.
        loopEnd = r"^s_cbranch_scc0 label_LoopBeginL\b" if subtile else \
                  (r"^label_LoopEndL:$" if eps == 0 and ulsgro == 0 else None)
        if loopEnd is not None:
            loop = re.search(rf"^label_LoopBeginL:$.*?{loopEnd}", src, re.M | re.S)
            assert loop and "A2A" not in loop.group(0) and "FusedCounterPtr" not in loop.group(0), \
                f"Kernel {base!r}: A2A code inside the loop body"


def test_a2a_gemm_view_gfx950_coverage(view_config):
    """The groups above cover MT256 subtile and non-subtile, each with A2ASubRounds 1, 2 and 4."""
    seen = set()
    for group in _GROUPS:
        for _base, src, _err in _emit(view_config, group):
            name = _NAME.search(src).group(1)
            seen.add((re.search(r"_MT(\d+)x", name).group(1), _param(name, "USI"), _param(name, "AASR")))
    want = {(mt, usi, s) for mt in ("256",) for usi in (0, 1) for s in (1, 2, 4)}
    assert want <= seen, f"missing {sorted(want - seen)}"
