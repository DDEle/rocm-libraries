#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# D15 Step 1 -- DRAIN ownership moves from W per-peer spinners to the single
# globally-last workgroup, and the poll becomes one vector load reduced with
# VCCZ (ROCM-27524, deferred item D15 Step 1).
################################################################################

import ast
import os
import re
import shutil
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

_GFX = "gfx950"
_GOLDEN = {
    0: os.path.join(SCRIPT_DIR, "test_data", "fusedA2A_handshake_per_peer.golden.s"),
    1: os.path.join(SCRIPT_DIR, "test_data", "fusedA2A_handshake_last_wg.golden.s"),
}


def _renderHandshake(drainOwner: int) -> str:
    """Render _emitFusedA2AHandshake standalone for one FusedA2ADrainOwner value.

    Same shape as test_fusedA2A_sdma_issue.py's _render, but driving the whole
    handshake, which reads more off kernel/parentWriter (WavefrontSize via the
    `wavelen` property, FusedGemmA2A, FusedA2ADrainOwner).

    The argLoader stub echoes its arguments instead of a fixed comment. The real
    loadKernArg is unavailable here, and a constant stub would render all eleven
    kernarg loads as the same line -- which would make both any "the counter_ptr
    kernarg is loaded" assertion and the golden below blind to a wrong offset.
    """
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from rocisa.label import LabelManager
    from rocisa.code import Module, TextBlock
    from Tensile.Common.Architectures import gfxToIsa
    import Tensile.Component  # noqa: F401  MUST precede the next import (circular-import guard)
    from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

    def loadKernArgEcho(*a, **k):
        fields = [str(x) for x in a] + ["%s=%s" % (n, k[n]) for n in sorted(k)]
        return TextBlock("// loadKernArg %s\n" % " ".join(fields))

    ri = rocIsa.getInstance()
    isa = gfxToIsa(_GFX)
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, 64)

    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.labels = LabelManager()
    w.vgprPool.checkOut(1)   # v0 reserved (Serial)
    w.sgprPool.checkOut(8)   # reserve s0..s7
    w.sgprs = {"AddressD": w.sgprPool.checkOutAligned(2, 2, "AddressD", preventOverflow=False)}
    indexChars = [str(i) for i in range(8)]
    indexChars[1] = "1J"                      # matches KernelWriter's "1"+INDEX_CHARS[1]
    w.states = SimpleNamespace(fusedA2AKernArgBase=0, indexChars=indexChars)
    w.argLoader = SimpleNamespace(loadKernArg=loadKernArgEcho)

    gwb = object.__new__(GlobalWriteBatchWriter)
    gwb.kernel = {"MacroTile0": 256, "MacroTile1": 256, "PackedC1IndicesX": [1],
                  "WavefrontSize": 64, "FusedGemmA2A": 1,
                  "FusedA2ADrainOwner": drainOwner}
    gwb.parentWriter = w

    m = Module("handshake")
    gwb._emitFusedA2AHandshake(m)
    return str(m)


@pytest.fixture
def renderHandshake():
    pytest.importorskip("rocisa")
    return _renderHandshake


def _codeLines(text):
    """Comment-stripped, blank-stripped instruction stream.

    Assertions about *what the kernel does* must run against this, never the raw
    text: every operand name below also appears in the instruction's own comment,
    so an `in text` check stays green with the operands wired wrong.
    """
    return [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]


# The echoing argLoader stub renders every kernarg load as a comment line carrying
# its destination register, width and offset -- which is what makes the check below
# possible at all.
_LOAD_RE = re.compile(r"//\s*loadKernArg (\d+) KernArgAddress .*?dword=(\d+).*?sgprOffset=(\S+)")


def _deadKernargLoads(text):
    """Kernarg loads whose register is overwritten by another load before any read.

    The register pool recycles a slot the moment it is checked in, so two kernarg
    values can share one physical SGPR. That is fine while their live ranges do not
    overlap, and silently wrong when they do -- and it is invisible in review,
    because the *comment* on each later instruction still names the value the author
    intended ("myRank * N" reading a register that now holds AM_tiles).

    Reports (reg, offset, clobberOffset) when a load is destroyed before its first
    read. Deliberately conservative: legitimate reuse always happens after the last
    read of the previous value, so it cannot trip this, and a value clobbered after
    one read but before a later one is not modelled (no test here needs that).
    Reads are counted on the comment-stripped instruction only, so a register named
    solely in a comment does not mask a clobber.
    """
    raw = [ln for ln in text.splitlines() if ln.strip()]
    code = [ln.split("//")[0] for ln in raw]
    loads = []
    for i, ln in enumerate(raw):
        m = _LOAD_RE.search(ln)
        if m and m.group(2) == "1":          # single-dword loads; pairs render as s[n:n+1]
            loads.append((i, int(m.group(1)), m.group(3)))

    dead = []
    for n, (i, reg, off) in enumerate(loads):
        pat = re.compile(r"\bs%d\b" % reg)
        firstRead = next((j for j in range(i + 1, len(code)) if pat.search(code[j])), None)
        if firstRead is None:
            continue
        for (j, reg2, off2) in loads[n + 1:]:
            if reg2 == reg and off2 != off and j < firstRead:
                dead.append((reg, off, off2))
    return dead


def test_drain_owner_is_a_validated_solution_parameter():
    from Tensile.Common.ValidParameters import validParameters

    assert validParameters["FusedA2ADrainOwner"] == [0, 1]


def test_drain_owner_defaults_to_the_current_per_peer_behaviour():
    from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters

    defaults = {k: v for d in defaultBenchmarkCommonParameters for k, v in d.items()}
    assert defaults["FusedA2ADrainOwner"] == [0], \
        "default must reproduce today's behaviour so it is a clean regression baseline"


def test_drain_owner_is_distinct_from_the_vestigial_FusedA2ADrain():
    # FusedA2ADrain exists but is never read by codegen (the runtime switch is
    # the FusedDrain kernarg). Guard against someone collapsing the two: their
    # defaults disagree, so a merged knob cannot preserve both.
    from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters

    defaults = {k: v for d in defaultBenchmarkCommonParameters for k, v in d.items()}
    assert defaults["FusedA2ADrain"] == [1], "vestigial knob keeps its historical default"
    assert defaults["FusedA2ADrainOwner"] == [0]
    assert defaults["FusedA2ADrain"] != defaults["FusedA2ADrainOwner"], \
        "two distinct knobs: WHETHER (vestigial) vs WHO (drain owner)"


def test_total_wgs_latch_multiplies_the_two_grid_dims():
    pytest.importorskip("rocisa")
    from rocisa.code import Module
    from Tensile.Components.GlobalWriteBatch import emitFusedA2ATotalWGsLatch

    m = Module("latch")
    emitFusedA2ATotalWGsLatch(m, "FusedTotalWGs")
    text = str(m)

    # Assert on the comment-stripped instruction, never on `text`: the
    # instruction's own comment reads "FusedTotalWGs = NumWorkGroups0 *
    # NumWorkGroups1", so every name below is satisfied by the comment alone and
    # a `in text` check stays green even with the operands wired wrong.
    code = [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]
    # exactly one instruction -- this is a hot-path prologue, not a place to grow
    assert len(code) == 1, code
    assert code[0].startswith("s_mul_i32 "), code[0]
    # by position, so a dst/src swap is caught too. Count first: a bare 3-tuple
    # unpack would die with an opaque ValueError if the operand count drifted.
    ops = [op.strip() for op in code[0].split(None, 1)[1].split(",")]
    assert len(ops) == 3, f"expected a 3-operand s_mul_i32, got {len(ops)}: {code[0]}"
    dst, src0, src1 = ops
    assert "sgprFusedTotalWGs" in dst, code[0]
    assert "sgprNumWorkGroups0" in src0, code[0]
    assert "sgprNumWorkGroups1" in src1, code[0]


def test_drain_owner_also_locks_out_the_runtime_GSU_override():
    # Rejecting GlobalSplitU != 1 only pins the compile-time value. SupportUserGSU
    # defaults to True, and ContractionSolution.cpp honours problem.getParams().gsu()
    # when it is set -- so a runtime caller could re-inflate the grid under an
    # election target that was latched as a compile-time constant. Structural pin
    # (no toolchain needed): the lockout must live in the FusedGemmA2A block, be
    # guarded on FusedA2ADrainOwner, and sit on the ACCEPTED path -- after the
    # rejects, never nested inside one, or it never runs for the config it protects.
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())

    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    body = blocks[0].body

    TARGET = "state['InternalSupportParams']['SupportUserGSU'] = False"

    def index_of(pred):
        return next((i for i, stmt in enumerate(body) if pred(stmt)), None)

    lockout = index_of(lambda s: isinstance(s, ast.If)
                       and "FusedA2ADrainOwner" in ast.unparse(s.test)
                       and any(ast.unparse(b) == TARGET for b in s.body))
    assert lockout is not None, \
        "FusedGemmA2A block has no `if ...FusedA2ADrainOwner...:` setting " + TARGET

    gsu_reject = index_of(lambda s: isinstance(s, ast.If)
                          and "FusedA2ADrainOwner" in ast.unparse(s.test)
                          and "GlobalSplitU" in ast.unparse(s.test))
    assert gsu_reject is not None, "the compile-time GlobalSplitU rejection went missing"
    # Nesting the lockout inside the reject guard collapses the two indices: that
    # branch returns, so the flag would only ever be set on a dead solution.
    assert lockout > gsu_reject, \
        "lockout must be a sibling AFTER the GSU rejection, not inside/before it"

    # A top-level return before it would make it unreachable outright.
    assert not any(isinstance(s, (ast.Return, ast.Raise)) for s in body[:lockout]), \
        "a statement before the lockout leaves the block: the lockout is dead code"


def test_total_wgs_latch_is_gated_on_the_owner_knob_not_the_vestigial_one():
    # Every site of this feature must branch on FusedA2ADrainOwner (WHO drains).
    # Swapping in the similarly named but vestigial FusedA2ADrain -- which defaults
    # to 1 and is never read by codegen -- would allocate the SGPR, emit the latch
    # and reject batched problems for every fused kernel, decoupled from the
    # election Task 4 gates on the same knob.
    sites = (("Tensile/KernelWriter.py", 'defineSgpr("FusedTotalWGs"'),
             ("Tensile/KernelWriterAssembly.py", "emitFusedA2ATotalWGsLatch(module"),
             ("Tensile/SolutionStructs/Solution.py",
              '"FusedA2ADrainOwner=1 requires no batch dim'),
             ("Tensile/SolutionStructs/Solution.py",
              '"FusedA2ADrainOwner=1 requires GlobalSplitU=1'))
    for relpath, marker in sites:
        with open(os.path.join(TENSILE_ROOT, relpath)) as f:
            lines = f.read().splitlines()
        hits = [i for i, ln in enumerate(lines) if marker in ln]
        assert hits, f"{relpath}: no call site containing {marker!r}"
        for i in hits:
            guard = next((ln for ln in reversed(lines[:i])
                          if ln.lstrip().startswith("if ")), None)
            assert guard is not None, \
                f"{relpath}:{i + 1}: no preceding `if` for site {marker!r}"
            assert '"FusedA2ADrainOwner"]' in guard, (relpath, i + 1, guard)
            assert '"FusedA2ADrain"]' not in guard, (relpath, i + 1, guard)


################################################################################
# Task 4 -- the handshake preamble is hoisted above the PUSH gate and every WG
# tallies itself at counter3.
################################################################################


def _pushGateIndex(text):
    """Index, in the comment-stripped stream, of the PUSH gate's branch.

    Located by its comment (the only way to tell it apart from the six other
    s_cbranch_scc0 in the handshake); every assertion about it then runs against
    the comment-stripped instruction.
    """
    raw = [l for l in text.splitlines() if l.strip()]
    hits = [i for i, ln in enumerate(raw) if "not a PUSH WG" in ln]
    assert len(hits) == 1, f"expected exactly one PUSH gate branch, got {hits}"
    # _codeLines drops nothing but blanks, so raw and code indices coincide only
    # if no line is comment-only. Re-find by identity instead.
    code = _codeLines(text)
    gate = raw[hits[0]].split("//")[0].strip()
    assert gate, f"PUSH gate line is comment-only: {raw[hits[0]]!r}"
    return code.index(gate), gate


def test_push_gate_falls_through_to_counter3_not_the_exit(renderHandshake):
    """FusedA2ADrainOwner=1: local WGs must still reach the counter3 tally.

    The gate's not-taken edge used to jump the whole handshake. It must now land
    on the counter3 block, which is what makes the tally cover local WGs -- i.e.
    what makes FusedTotalWGs (= every surviving WG) the right election target.
    """
    text = renderHandshake(1)
    assert "fusedA2A_counter3" in text, "no counter3 label emitted"
    _, gate = _pushGateIndex(text)
    assert gate.startswith("s_cbranch_scc0 "), gate
    assert "fusedA2A_counter3" in gate, \
        f"PUSH gate must branch to the counter3 tally, not the exit: {gate}"


def test_preamble_is_hoisted_above_the_push_gate(renderHandshake):
    """The barrier and the wave-0 election must precede the PUSH gate.

    This is the reordering itself: local WGs only reach counter3 if they walk the
    preamble, and the single-lane EXEC the tally needs is set before the gate.
    """
    text = renderHandshake(1)
    code = _codeLines(text)
    gateIdx, _ = _pushGateIndex(text)

    barrier = [i for i, ln in enumerate(code) if ln.startswith("s_barrier")]
    assert len(barrier) == 1, f"expected one s_barrier, got {barrier}"
    assert barrier[0] < gateIdx, "s_barrier must be hoisted above the PUSH gate"

    # wave-0 election: readfirstlane of Serial, then the non-wave-0 branch out.
    election = [i for i, ln in enumerate(code)
                if ln.startswith("v_readfirstlane_b32") and "vgprSerial" in ln]
    assert len(election) == 1, f"expected one Serial readfirstlane, got {election}"
    assert election[0] < gateIdx, "the wave-0 election must precede the PUSH gate"

    # single-lane EXEC (exec, 1) is what makes the tally fire once per WG.
    lane0 = [i for i, ln in enumerate(code) if ln.replace(" ", "").endswith("exec,1")]
    assert len(lane0) == 1, f"expected one single-lane EXEC write, got {lane0}"
    assert lane0[0] < gateIdx, "EXEC must be narrowed to lane 0 before the PUSH gate"


def test_store_wait_precedes_the_barrier(renderHandshake):
    """Reordering these breaks the 'all waves' stores landed' guarantee.

    Each wave retires its own stores, then the barrier joins them; the other
    order lets the SDMA engine read a band that is not yet in HBM. Note the
    emitted mnemonic is vmcnt, not vscnt -- SWaitCnt(vscnt=0) lowers to
    `s_waitcnt vmcnt(0)` on gfx950, so keying on "vscnt" would never match.
    """
    for owner in (0, 1):
        code = _codeLines(renderHandshake(owner))
        waits = [i for i, ln in enumerate(code)
                 if ln.startswith("s_waitcnt") and ("vmcnt" in ln or "vscnt" in ln)]
        barrier = [i for i, ln in enumerate(code) if ln.startswith("s_barrier")]
        assert waits, f"owner={owner}: no vector-memory wait emitted at all"
        assert len(barrier) == 1, f"owner={owner}: expected one s_barrier, got {barrier}"
        assert waits[0] < barrier[0], \
            f"owner={owner}: a store wait must precede s_barrier (wait {waits[0]}, barrier {barrier[0]})"


def test_counter3_is_incremented_exactly_once(renderHandshake):
    text = renderHandshake(1)
    code = _codeLines(text)
    label = [i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3")]
    assert len(label) == 1, f"expected one counter3 label definition, got {label}"
    tail = code[label[0]:]
    atomics = [ln for ln in tail if ln.startswith("global_atomic")]
    assert len(atomics) == 1, f"counter3 must add exactly once, got: {atomics}"
    assert atomics[0].startswith("global_atomic_add "), atomics[0]


def test_counter3_election_compares_against_the_latched_total(renderHandshake):
    """The tally must be compared against FusedTotalWGs (Task 3's prologue latch).

    Comparing against anything else -- tokenTiles, TilesPerRank -- would re-elect
    a per-peer owner under a knob that promises a single global one.
    """
    code = _codeLines(renderHandshake(1))
    label = next(i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3"))
    # Anchor on the tally itself: the atomic, then the first compare after it. The
    # relocated DRAIN also sits after the label and carries compares of its own.
    atomic = next(i for i, ln in enumerate(code[label:], label)
                  if ln.startswith("global_atomic_add "))
    cmp_i = next(i for i, ln in enumerate(code[atomic:], atomic)
                 if ln.startswith("s_cmp_eq_u32"))
    assert "sgprFusedTotalWGs" in code[cmp_i], code[cmp_i]
    # and it must be what decides the DRAIN, i.e. immediately consumed by the branch.
    assert code[cmp_i + 1].startswith("s_cbranch_scc0 "), code[cmp_i:cmp_i + 2]


def test_no_kernarg_value_is_clobbered_before_it_is_read(renderHandshake):
    """Reordering the blocks must not make two kernarg values share a live range.

    `gateSgpr` is checked in before `myRankSgpr` is checked out, so the pool hands
    back the same physical SGPR and the PUSH gate's FusedAM shares a register with
    FusedMyRank. Under DrainOwner=0 the gate is emitted first and its value is dead
    by then, so the reuse is correct. Emitting the gate LAST turns it into a
    clobber: my_rank is destroyed after argModule set it, and the SDMA block goes on
    to compute dst_y = AM_tiles*N and flag_ptr[p] + AM_tiles*8 -- wrong band, and an
    ATOMIC past the W-slot flag allocation.

    Structural because it cannot be seen any other way: the emitted comments still
    read "myRank * N" and "myRank * 8" over the clobbered register.
    """
    for owner in (0, 1):
        dead = _deadKernargLoads(renderHandshake(owner))
        assert not dead, (
            f"DrainOwner={owner}: kernarg value overwritten before first read "
            f"(reg, loaded, clobbered-by) = {dead}")


def test_drain_owner_mode_emits_no_drain_yet(renderHandshake):
    """Task 5 owns attaching the DRAIN; Task 4 must not relocate the per-peer one.

    The per-peer poll waits on flag[dst_rank], and _fusedA2ALoadFlagBaseAndRank
    derives dst_rank by scanning j in range(1, FUSED_A2A_MAX_RANKS) for the largest
    j with j*n_shard <= WorkGroup0*MT0. The counter3 winner is frequently a LOCAL
    WG, whose WorkGroup0*MT0 is >= AM = W*n_shard (host: nShard = AM/W), so the scan
    runs past this card's peers: for W <= 7 it yields dst_rank >= W, an
    out-of-bounds read past the host's flagBytes = W*sizeof(uint64_t) allocation;
    at W = 8 it saturates at 7 and merely polls the wrong peer. Either way it is the
    wrong predicate, so the counter3 block ends at the election branch and the
    all-W-slot poll is attached later.
    """
    text = renderHandshake(1)
    assert "fusedA2A_drain" not in text, \
        "the per-peer DRAIN must not be emitted under DrainOwner=1 (reads flag[dst_rank>=W])"

    code = _codeLines(text)
    # Positive form: the election branch is the last instruction of the handshake,
    # i.e. nothing at all sits between it and the exit label. A bare string-absence
    # check would stay green if a DRAIN were re-added under different label names.
    cmp_i = next(i for i, ln in enumerate(code)
                 if ln.startswith("s_cmp_eq_u32") and "sgprFusedTotalWGs" in ln)
    assert code[cmp_i + 1].startswith("s_cbranch_scc0 "), code[cmp_i:cmp_i + 3]
    assert code[cmp_i + 2].startswith("label_fusedA2A_handshake_after"), \
        f"counter3 block must end at the election branch, found: {code[cmp_i + 2:cmp_i + 5]}"

    # ...and the per-peer path keeps its DRAIN: this removal must not leak into the
    # regression baseline (the golden pins the rest).
    assert "fusedA2A_drain_poll" in renderHandshake(0), \
        "DrainOwner=0 must still emit the per-peer DRAIN"


def test_every_surviving_wg_runs_the_handshake_once():
    """The counter3 target is FusedTotalWGs, so arrivals must equal survivors.

    test_push_gate_falls_through_to_counter3_not_the_exit covers the half of that
    inside the handshake (both edges of the PUSH gate reach the tally). This covers
    the other half, which lives in the caller: under FusedGemmA2A the whole store
    body -- handshake included -- is emitted TWICE around one hoisted dispatch gate,
    a PUSH pass and a LOCAL pass, and each WG runs exactly one of them.

    Two ways that could break, both silent-hang rather than wrong-answer:
      - gating the handshake call on states.fusedA2ADispatchMode, so only the PUSH
        pass carries it and local WGs never tally;
      - dropping one of the two passes.
    Either leaves the tally short of FusedTotalWGs forever and the DRAIN never fires.
    """
    with open(os.path.join(TENSILE_ROOT, "Tensile/Components/GlobalWriteBatch.py")) as f:
        gwb = ast.parse(f.read())

    def callsHandshake(node):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "_emitFusedA2AHandshake"
                   for c in ast.walk(node))

    sites = [n for n in ast.walk(gwb) if isinstance(n, ast.If) and callsHandshake(n)]
    assert len(sites) == 1, f"expected one guarded _emitFusedA2AHandshake call, got {len(sites)}"
    guard = ast.unparse(sites[0].test)
    assert "fusedA2ADispatchMode" not in guard, \
        f"handshake must run in BOTH dispatch passes, not just PUSH: {guard}"

    with open(os.path.join(TENSILE_ROOT, "Tensile/KernelWriterAssembly.py")) as f:
        kwa = ast.parse(f.read())
    hoist = [n for n in ast.walk(kwa) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "kernel['FusedGemmA2A']"]
    assert len(hoist) == 1, f"expected one fused-A2A hoist block, got {len(hoist)}"
    passes = [c for stmt in hoist[0].body for c in ast.walk(stmt)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
              and c.func.id == "_emit_batch_loop"]
    assert len(passes) == 2, \
        f"expected a PUSH pass and a LOCAL pass, got {len(passes)} batch-loop emissions"


def test_per_peer_mode_emits_no_counter3_machinery(renderHandshake):
    """DrainOwner=0 is the regression baseline: the new path must be absent."""
    text = renderHandshake(0)
    assert "counter3" not in text
    assert "FusedTotalWGs" not in text


@pytest.mark.parametrize("owner", [0, 1])
def test_rendering_is_byte_identical_to_the_golden(renderHandshake, owner):
    """Characterization pin on both paths.

    The structural tests each assert one fact, so between them they leave gaps --
    a register clobber introduced by reordering blocks sat in exactly such a gap and
    reached review. These compare the whole rendering, so any drift shows up
    whether or not someone thought to assert on it.

    owner=0 is the regression baseline, captured from the pre-refactor emitter: if
    it goes red, the per-peer path moved. owner=1 was captured after the clobber fix
    and pins the new path, which nothing else golds.

    Intentionally changing either path? Regenerate deliberately:
        python Tensile/Tests/unit/test_fusedA2A_drain_last.py --update-golden
    and justify the diff in review -- do not regenerate to silence a red.
    """
    with open(_GOLDEN[owner]) as f:
        golden = f.read()
    assert renderHandshake(owner) == golden, \
        f"DrainOwner={owner} rendering drifted from the golden; see the docstring"


if __name__ == "__main__":
    if "--update-golden" in sys.argv:
        for _owner, _path in sorted(_GOLDEN.items()):
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "w") as f:
                f.write(_renderHandshake(_owner))
            print("wrote %s" % _path)
    else:
        print("pass --update-golden to regenerate the handshake goldens")
