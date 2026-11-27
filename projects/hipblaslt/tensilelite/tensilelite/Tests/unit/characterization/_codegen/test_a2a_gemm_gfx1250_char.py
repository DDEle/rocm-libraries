# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused A2A+GEMM (FusedA2A: A2AGemm) codegen characterization.

Covers the gfx1250 arms of the A2AGemm paths in KernelWriterAssembly.py: the
election, tallies and ring go through VMEM atomics on one lane, the flag and
done[] accesses are system scope, EXEC is wave32 throughout, and a segment
switch rebuilds the TDM descriptor's global address without waiting on the
tensor loads, leaving the main loop as the plain GEMM's.
"""

import os
import re
from collections import Counter

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "common", "comm", "gfx1250", "a2a_gemm.yaml",
    )
)

_A2A_MARKERS = (
    "global_atomic_inc_u32",
    "global_atomic_cmpswap_b64",
    "global_store_b128",
)
# ATOMIC DW0 and COPY_LINEAR_SUBWIN DW12 with their scope fields at SCOPE_SYS.
_GFX12_PACKET_CONSTANTS = ("0x1e30000a", "0xc0c0000")

_SCHEDULER_FILL = ("s_wait_alu", "s_set_vgpr_msb", "s_nop", "s_wait_dscnt")
_EXACT_IN_LOOP = ("s_wait_tensorcnt", "s_wait_loadcnt", "s_wait_kmcnt", "s_barrier")

# The segment switch: from the flag-slot arithmetic to the branch back into its poll.
_SWITCH_START = "flag slot = flag row + segment"
_SWITCH_END = "segment short -> poll again"
# Barriers of the A2AGemm protocol, matched on their comments.
_A2A_BARRIERS = ("ticket published", "ticket read", "every wave passed its last flag wait")


def _instructions(src):
    return [ln.strip() for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith(("/", ";", "."))
            and not ln.strip().split("//")[0].rstrip().endswith(":")]


def _switches(src):
    out, cur = [], None
    for ln in _instructions(src):
        if _SWITCH_START in ln:
            cur = []
        if cur is not None:
            cur.append(ln)
            if _SWITCH_END in ln:
                out.append(cur)
                cur = None
    return out


def _last_exec_write(insts, i):
    for ln in reversed(insts[:i]):
        m = re.match(r"s_mov_b32 exec_lo, (\S+)", ln)
        if m:
            return m.group(1)
    return None


def _loop_lines(src):
    lines = src.splitlines()
    begin = next(i for i, ln in enumerate(lines) if ln.startswith("label_LoopBeginL:"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("label_LoopEndL:"))
    return _instructions("\n".join(lines[begin + 1:end]))


def _loop_body(src):
    return [ln.split()[0] for ln in _loop_lines(src) if not ln.startswith(_SCHEDULER_FILL)]


def _loop_syncs(src):
    return [ln.split("//")[0].strip() for ln in _loop_lines(src) if ln.startswith(_EXACT_IN_LOOP)]


@pytest.fixture(scope="module")
def results():
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    return results


@pytest.fixture(scope="module")
def plain_results(tmp_path_factory):
    """The same forks as plain GEMMs, with the knobs A2AGemm pins set to the pinned values."""
    config = yaml.safe_load(open(_CONFIG))
    group = config["BenchmarkProblems"][0]
    group[0]["FusedA2A"] = "Disabled"
    group[1]["ForkParameters"] += [{"OptNoLoadLoop": [0]},
                                   {"AssertSummationElementMultiple": [64]}]
    for size in group[1]["BenchmarkFinalParameters"][0]["ProblemSizes"]:
        size["Exact"].pop("world", None)
    path = tmp_path_factory.mktemp("a2a_gemm_plain") / "plain.yaml"
    path.write_text(yaml.safe_dump(config))
    return emit_kernels_from_config(str(path), limit=8, arch=_ARCH)


def test_a2a_gemm_gfx1250_emits_assembly(results):
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        missing = [m for m in _A2A_MARKERS + _GFX12_PACKET_CONSTANTS if m not in src]
        assert not missing, f"Kernel {base!r}: missing {missing}"


def test_a2a_gemm_gfx1250_no_scalar_memory_writes(results):
    for base, src, _err in results:
        bad = [ln for ln in _instructions(src) if ln.startswith(("s_atomic", "s_store"))]
        assert not bad, f"Kernel {base!r}: scalar atomics or stores: {bad[:3]}"


def test_a2a_gemm_gfx1250_exec_is_wave32(results):
    for base, src, _err in results:
        bad = [ln for ln in _instructions(src)
               if "exec_hi" in ln or re.match(r"s_\w+_b64 exec", ln)]
        assert not bad, f"Kernel {base!r}: wave64 EXEC writes: {bad[:3]}"


def test_a2a_gemm_gfx1250_flags_are_system_scope(results):
    for base, src, _err in results:
        for what in ("poll flag[b][i]", "poll done[s]", "flag = 0", "done[s] = 0",
                     "done[myRank] = 1"):
            lines = [ln for ln in _instructions(src) if what in ln]
            assert lines, f"Kernel {base!r}: no {what!r}"
            wrong = [ln for ln in lines if "scope:SCOPE_SYS" not in ln]
            assert not wrong, f"Kernel {base!r}: {what!r} without SCOPE_SYS: {wrong[:2]}"


def test_a2a_gemm_gfx1250_packet_stores_alone_under_narrow_exec(results):
    """In the enqueue's copy loop, each narrowed EXEC span holds only the ring stores and
    ends in the full-mask reset."""
    for base, src, _err in results:
        insts = _instructions(src)
        begin = next(i for i, ln in enumerate(insts) if "blocks placed this pass" in ln)
        end = next(i for i in range(begin, len(insts)) if "pending += signal bytes" in insts[i])
        loop, spans, cur = insts[begin:end], [], None
        for ln in loop:
            if re.match(r"s_\w+ exec_lo", ln):
                assert ln.startswith("s_mov_b32 exec_lo"), f"Kernel {base!r}: {ln}"
                if ln.startswith("s_mov_b32 exec_lo, -1"):
                    spans.append(cur)
                    cur = None
                else:
                    cur = []
            elif cur is not None and not ln.startswith(_SCHEDULER_FILL):
                cur.append(ln)
        assert len(spans) == 2 and cur is None, f"Kernel {base!r}: {len(spans)} store spans"
        bad = [ln for sp in spans for ln in sp if not ln.startswith("global_store_")]
        assert not bad, f"Kernel {base!r}: non-store work under a narrow EXEC: {bad[:3]}"


def test_a2a_gemm_gfx1250_atomics_on_one_lane(results):
    for base, src, _err in results:
        insts = _instructions(src)
        idx = [i for i, ln in enumerate(insts)
               if ln.startswith(("global_atomic_cmpswap_b64", "global_atomic_max_u64",
                                 "global_atomic_inc_u32"))]
        assert idx, f"Kernel {base!r}: no ring or counter atomics"
        wrong = [insts[i] for i in idx if _last_exec_write(insts, i) != "1"]
        assert not wrong, f"Kernel {base!r}: atomics not on lane 0 alone: {wrong[:2]}"


def test_a2a_gemm_gfx1250_keeps_its_barriers(results):
    for base, src, _err in results:
        insts = _instructions(src)
        missing = [b for b in _A2A_BARRIERS
                   if not any(ln.startswith("s_barrier_wait") and b in ln for ln in insts)]
        assert not missing, f"Kernel {base!r}: barriers dropped: {missing}"


def _after_loop_tensor_waits(src):
    lines = src.splitlines()
    begin = next(i for i, ln in enumerate(lines) if ln.startswith("label_LoopEndL:"))
    end = next(i for i in range(begin, len(lines))
               if lines[i].startswith("label_PrefetchGlobalLastIterEnd"))
    return sum(ln.strip().startswith("s_wait_tensorcnt") for ln in lines[begin:end])


def test_a2a_gemm_gfx1250_no_load_loop_drains_like_plain_gemm(results, plain_results):
    plain = {frozenset(_name_tokens(psrc)): psrc for (_pbase, psrc, _e) in plain_results}
    for base, src, _err in results:
        want = _after_loop_tensor_waits(plain[frozenset(_name_tokens(src))])
        got = _after_loop_tensor_waits(src)
        assert got >= want, (
            f"Kernel {base!r}: {got} tensor-count waits after the main loop, plain GEMM has {want}")


def test_a2a_gemm_gfx1250_switch_has_no_tdm_drain(results):
    for base, src, _err in results:
        switches = _switches(src)
        assert switches, f"Kernel {base!r}: no segment switch"
        for sw in switches:
            bad = [ln for ln in sw if ln.startswith(("s_wait_tensorcnt", "s_barrier"))]
            assert not bad, f"Kernel {base!r}: segment switch waits on TDM or barriers: {bad}"


def test_a2a_gemm_gfx1250_switch_rebuilds_the_descriptor(results):
    for base, src, _err in results:
        for sw in _switches(src):
            lo = [ln for ln in sw if re.match(r"s_add_u32 s\[sgprtdm\w+Group0\+2\]", ln)]
            hi = [ln for ln in sw if re.match(r"s_addc_u32 s\[sgprtdm\w+Group0\+3\]", ln)]
            ty = [ln for ln in sw
                  if re.match(r"s_or_b32 s\[sgprtdm\w+Group0\+3\], .*0x80000000", ln)]
            sel = [ln for ln in sw if ln.startswith("s_cselect_b32") and "tile base" in ln]
            assert lo and hi and ty and len(sel) == 2, (
                f"Kernel {base!r}: switch does not rebuild the descriptor address: "
                f"{len(lo)} lo, {len(hi)} hi, {len(ty)} type, {len(sel)} base selects")


def _name_tokens(src):
    name = re.search(r"^\.globl (\S+)", src, re.M).group(1)
    return set(name.split("_")) - {"A2AGemm", "AASR1"}


def test_a2a_gemm_gfx1250_switch_selects_like_the_loop_increment(results):
    """The tile base a switch picks for a wave is of the tensor its loop increment is."""
    for base, src, _err in results:
        incs = [ln for ln in _instructions(src)
                if re.match(r"s_cselect_b32 s\[sgprtdmABIncs\], s\[sgprGlobalReadIncs([AB])", ln)]
        assert incs, f"Kernel {base!r}: no tdmABIncs select"
        first = re.match(r"s_cselect_b32 s\[sgprtdmABIncs\], s\[sgprGlobalReadIncs([AB])", incs[0]).group(1)
        for sw in _switches(src):
            sel = [ln for ln in sw if ln.startswith("s_cselect_b32") and "tile base" in ln]
            wrong = [ln for ln in sel if not re.search(r"s_cselect_b32 \S+, s\[sgprA2ATileBase%s" % first, ln)]
            assert not wrong, f"Kernel {base!r}: tile-base select disagrees with tdmABIncs: {wrong}"


def test_a2a_gemm_gfx1250_step_back_only_after_pgr2_loop_exit(results):
    for base, src, _err in results:
        pgr = int(re.search(r"_PGR(\d)_", re.search(r"^\.globl (\S+)", src, re.M).group(1)).group(1))
        backs = sum("one tdmABIncs back" in ln for ln in _instructions(src))
        assert backs == (1 if pgr >= 2 else 0), f"Kernel {base!r} (PGR{pgr}): {backs} step-backs"


def test_a2a_gemm_gfx1250_loop_body_matches_plain_gemm(results, plain_results):
    """Same instructions in the main loop as the plain GEMM of the same forks, in any order."""
    plain = {frozenset(_name_tokens(psrc)): (pbase, psrc) for (pbase, psrc, _e) in plain_results}
    for base, src, _err in results:
        pbase, psrc = plain[frozenset(_name_tokens(src))]
        fused, ref = Counter(_loop_body(src)), Counter(_loop_body(psrc))
        assert fused == ref, (
            f"Kernel {base!r}: main loop differs from {pbase!r}: "
            f"+{dict(fused - ref)} -{dict(ref - fused)}")
        fused, ref = Counter(_loop_syncs(src)), Counter(_loop_syncs(psrc))
        assert fused == ref, (
            f"Kernel {base!r}: main-loop waits and barriers differ from {pbase!r}: "
            f"+{dict(fused - ref)} -{dict(ref - fused)}")


def test_a2a_gemm_gfx1250_golden(results, snapshot):
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
