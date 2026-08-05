#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused-A2A DRAIN ownership: the single globally-last workgroup drains, elected
# by a grid-wide counter3, and polls all W flag slots with one vector load
# reduced by VCCZ -- replacing W per-peer spinners, each of which idled a whole
# CU at the champion kernel's 1-WG/CU occupancy (ROCM-27524, D15 Step 1).
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
_GOLDEN = os.path.join(SCRIPT_DIR, "test_data", "fusedA2A_handshake.golden.s")


def _renderHandshake(wavefrontSize: int = 64) -> str:
    """Render _emitFusedA2AHandshake standalone.

    Same shape as test_fusedA2A_sdma_issue.py's _render, but driving the whole
    handshake, which reads more off kernel/parentWriter (WavefrontSize via the
    `wavelen` property, FusedGemmA2A).

    `wavefrontSize` picks the wave width. It defaults to 64 -- what every fused
    config runs and what the golden below captures -- but the handshake has a wave32
    arm (EXEC is one dword, `exec_lo`, and the mask instruction is the B32 one), and
    a fixture that could only render wave64 left that arm covered by nothing at all.

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
    ri.setKernel(isa, wavefrontSize)

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
                  "WavefrontSize": wavefrontSize, "FusedGemmA2A": 1}
    gwb.parentWriter = w

    m = Module("handshake")
    gwb._emitFusedA2AHandshake(m)
    return str(m)


def _requireRocisa():
    """Import rocisa, skipping ONLY when it is genuinely absent.

    rocisa's staleness gate raises a plain `ImportError` when the C++ sources are
    newer than the built `_rocisa.so`. `pytest.importorskip` swallowed `ImportError`
    outright before pytest 8.2, and `pyproject.toml` floors pytest at 5.4.1 -- so on a
    resolved-old pytest a stale checkout would turn every test here into a SKIP and
    still report the run green, with this file (the only coverage of the fused-A2A
    DRAIN barrier) contributing nothing. Presenting as a pass is the whole hazard, so
    the two cases are separated by hand rather than left to importorskip's default:
    absent -> skip (rocisa is an optional build artifact), anything else -> fail loud.
    """
    try:
        import rocisa  # noqa: F401
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "rocisa":
            pytest.skip("rocisa is not installed")
        # Installed but unimportable -- stale bindings being the expected cause.
        # `invoke rocisa` is the only rebuild this environment picks up: a bare
        # `cmake --build` leaves a .so that the loaded package never resolves to.
        pytest.fail(
            "rocisa is installed but failed to import, so this file's coverage of the "
            "fused-A2A DRAIN barrier did NOT run.\n"
            "  Rebuild with: invoke rocisa\n"
            f"  {type(exc).__name__}: {exc}",
            pytrace=False,
        )


@pytest.fixture
def renderHandshake():
    _requireRocisa()
    return _renderHandshake


def _codeLines(text):
    """Comment-stripped, blank-stripped instruction stream.

    Assertions about *what the kernel does* must run against this, never the raw
    text: every operand name below also appears in the instruction's own comment,
    so an `in text` check stays green with the operands wired wrong.
    """
    return [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]


def _ops(line):
    """Operands of a comment-stripped instruction, in source order.

    Operands must be compared BY POSITION -- `"s28" in line` cannot tell a dst from a
    src, and would keep an assertion green through an operand swap.
    """
    parts = line.split(None, 1)
    return [] if len(parts) < 2 else [op.strip() for op in parts[1].split(",")]


def _baseReg(operand):
    """Base SGPR of an operand written either as `s28` or as the pair `s[28:29]`.

    The wave64 lowering writes EXEC from an aligned pair, so the register the EXEC
    write names is not spelled the same as the one the mask arithmetic writes; both
    have to reduce to the same name before they can be tied together.
    """
    m = re.fullmatch(r"s\[(\d+):\d+\]", operand) or re.fullmatch(r"s(\d+)", operand)
    return "s" + m.group(1) if m else None


def _braceBlock(src, start):
    """Text of the brace-balanced { ... } block opening at or after offset `start`."""
    openIdx = src.index("{", start)
    depth = 0
    for i in range(openIdx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[openIdx:i + 1]
    raise AssertionError("unbalanced braces after offset %d" % start)


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


def _maskWidthProvenance(text):
    """Where the s_bfm width operand's value came from.

    Every other assertion about the mask can only see the operand's SHAPE -- that it
    is a register. Nothing in the emitted text distinguishes an s_bfm reading W from
    the identical line reading a register the pool has since handed to something
    else; the comment says "(1 << W) - 1" either way. That is the exact failure mode
    _deadKernargLoads exists for, one step further along.

    The echoing argLoader stub is the only place a register is tied to the kernarg it
    was loaded from, so resolve the width register back through it. Returns
    (sgprOffset of the last single-dword kernarg load into that register before the
    s_bfm, list of instructions writing it in between) -- the second must be empty or
    the offset says nothing about the value at the s_bfm.
    """
    raw = [ln for ln in text.splitlines() if ln.strip()]
    code = [ln.split("//")[0].strip() for ln in raw]
    masks = [i for i, ln in enumerate(code) if ln.startswith("s_bfm_b")]
    assert len(masks) == 1, f"expected exactly one s_bfm, got {[code[i] for i in masks]}"
    i = masks[0]
    width = _ops(code[i])[1]

    fed = [(j, m.group(3)) for j in range(i)
           for m in [_LOAD_RE.search(raw[j])] if m and m.group(2) == "1"
           and "s" + m.group(1) == width]
    assert fed, f"the s_bfm width operand {width} is never loaded from a kernarg: {code[i]}"
    j, offset = fed[-1]
    writes = re.compile(r"^\S+\s+%s\b" % re.escape(width))
    return offset, [code[k] for k in range(j + 1, i) if writes.match(code[k])]


def test_total_wgs_latch_multiplies_the_two_grid_dims():
    _requireRocisa()
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


def test_fused_a2a_also_locks_out_the_runtime_GSU_override():
    # Rejecting GlobalSplitU != 1 only pins the compile-time value. SupportUserGSU
    # defaults to True, and ContractionSolution.cpp honours problem.getParams().gsu()
    # when it is set -- so a runtime caller could re-inflate the grid under an
    # election target that was latched as a compile-time constant. Structural pin
    # (no toolchain needed): the lockout must live in the FusedGemmA2A block and sit
    # on the ACCEPTED path -- after the rejects, never nested inside one, or it
    # never runs for the config it protects.
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())

    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    body = blocks[0].body

    TARGET = "state['InternalSupportParams']['SupportUserGSU'] = False"

    def index_of(pred):
        return next((i for i, stmt in enumerate(body) if pred(stmt)), None)

    # Unconditional now: a bare assignment in the block body, not behind an `if`.
    lockout = index_of(lambda s: ast.unparse(s) == TARGET)
    assert lockout is not None, f"FusedGemmA2A block no longer sets {TARGET}"

    gsu_reject = index_of(lambda s: isinstance(s, ast.If)
                          and "GlobalSplitU" in ast.unparse(s.test)
                          and any(isinstance(b, ast.Return) for b in s.body))
    assert gsu_reject is not None, "the compile-time GlobalSplitU rejection went missing"
    # Placing the lockout before the reject would set it on solutions that are then
    # thrown away, and leave the accepted ones untouched.
    assert lockout > gsu_reject, \
        "lockout must come AFTER the GSU rejection, not inside/before it"

    # A top-level return before it would make it unreachable outright.
    assert not any(isinstance(s, (ast.Return, ast.Raise)) for s in body[:lockout]), \
        "a statement before the lockout leaves the block: the lockout is dead code"


def test_batch_guard_is_host_side_not_compile_time():
    """The batch guard must test the EXTENT, which only the host can see.

    Compile time knows only that a batch index is DECLARED. Every fused config sets
    `Batched: True` -> NumIndicesBatch == 1 while running extent 1, so a
    compile-time rejection on NumIndicesBatch matched every solution and generated
    ZERO kernels -- a failure that looks like "no solutions found", not like a bad
    number. It survived review because it sat behind a default-off flag that nothing
    ever enabled.
    """
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())
    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    # Scoped to the block: NumIndicesBatch is a legitimate name elsewhere in the file.
    rejects = [ast.unparse(s) for s in blocks[0].body
               if isinstance(s, ast.If) and "NumIndicesBatch" in ast.unparse(s.test)]
    assert not rejects, \
        f"fused block still rejects on a DECLARED batch index (zero kernels): {rejects}"

    # Host side. Grepping for `batchIndices()` / `batchSize(` proves only that the
    # API is CALLED -- it stays green with the predicate inverted (`!= 2` accepts the
    # extent 2 that breaks the election, and rejects the extent 1 every fused config
    # runs) or with the bail-out deleted, i.e. with a guard that guards nothing. So
    # pin the predicate and the early return, scoped to the loop's own block.
    host = os.path.join(TENSILE_ROOT, "client/src/FusedA2AClient.cpp")
    with open(host) as f:
        src = f.read()
    loop = re.search(r"for\s*\(.*?problem->batchIndices\(\)\.size\(\).*?\)\s*\{", src, re.S)
    assert loop, \
        "FusedA2AClient.cpp does not loop over batchIndices(); the guard was deleted, not moved"
    body = _braceBlock(src, loop.end() - 1)
    preds = re.findall(
        r"if\s*\(\s*problem->batchSize\(\s*\w+\s*\)\s*(!=|==|<|>|<=|>=)\s*(\d+)\s*\)", body)
    assert preds == [("!=", "1")], \
        f"the batch guard must reject every extent other than 1; found {preds} in:\n{body}"
    assert re.search(r"\breturn\s+1\s*;", body), \
        f"the batch guard detects a bad extent but never bails out (no `return 1;`):\n{body}"


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
    """Local WGs must reach the counter3 tally, not skip the handshake.

    The gate's not-taken edge used to jump the whole handshake. It must now land
    on the counter3 block, which is what makes the tally cover local WGs -- i.e.
    what makes FusedTotalWGs (= every surviving WG) the right election target.
    """
    text = renderHandshake()
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
    text = renderHandshake()
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

    # single-lane EXEC (exec, 1) is what makes the tally fire once per WG. There are
    # exactly two: this one, and the DRAIN restoring the width after its wide poll.
    # Pinned exactly -- a third would mean some region silently lost its width.
    lane0 = [i for i, ln in enumerate(code) if ln.replace(" ", "").endswith("exec,1")]
    assert len(lane0) == 2, \
        f"expected two single-lane EXEC writes (tally + post-DRAIN restore), got {lane0}"
    assert lane0[0] < gateIdx, "EXEC must be narrowed to lane 0 before the PUSH gate"


def test_store_wait_precedes_the_barrier(renderHandshake):
    """Reordering these breaks the 'all waves' stores landed' guarantee.

    Each wave retires its own stores, then the barrier joins them; the other
    order lets the SDMA engine read a band that is not yet in HBM. Note the
    emitted mnemonic is vmcnt, not vscnt -- SWaitCnt(vscnt=0) lowers to
    `s_waitcnt vmcnt(0)` on gfx950, so keying on "vscnt" would never match.
    """
    code = _codeLines(renderHandshake())
    waits = [i for i, ln in enumerate(code)
             if ln.startswith("s_waitcnt") and ("vmcnt" in ln or "vscnt" in ln)]
    barrier = [i for i, ln in enumerate(code) if ln.startswith("s_barrier")]
    assert waits, "no vector-memory wait emitted at all"
    assert len(barrier) == 1, f"expected one s_barrier, got {barrier}"
    assert waits[0] < barrier[0], \
        f"a store wait must precede s_barrier (wait {waits[0]}, barrier {barrier[0]})"


def test_counter3_is_incremented_exactly_once(renderHandshake):
    text = renderHandshake()
    code = _codeLines(text)
    label = [i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3")]
    assert len(label) == 1, f"expected one counter3 label definition, got {label}"
    tail = code[label[0]:]
    atomics = [ln for ln in tail if ln.startswith("global_atomic")]
    assert len(atomics) == 1, f"counter3 must add exactly once, got: {atomics}"
    assert atomics[0].startswith("global_atomic_add "), atomics[0]


def test_counter3_election_compares_against_the_latched_total(renderHandshake):
    """The tally must be compared against FusedTotalWGs (Task 3's prologue latch).

    Comparing against anything else -- tokenTiles, TilesPerRank -- would elect a
    per-peer owner again, which is the occupancy problem this replaced.
    """
    code = _codeLines(renderHandshake())
    label = next(i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3"))
    # Anchor on the tally itself: the atomic, then the first compare after it. The
    # DRAIN also sits after the label and carries compares of its own.
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
    back the same physical SGPR and the PUSH gate's FusedAM shares one with
    FusedMyRank. That reuse is only correct while the gate is emitted AHEAD of the
    arg reads, where its value is already dead. Emitting the gate after them turns
    it into a clobber: my_rank is destroyed after argModule set it, and the SDMA
    block goes on to compute dst_y = AM_tiles*N and flag_ptr[p] + AM_tiles*8 --
    wrong band, and an ATOMIC past the W-slot flag allocation.

    Structural because it cannot be seen any other way: the emitted comments still
    read "myRank * N" and "myRank * 8" over the clobbered register.
    """
    dead = _deadKernargLoads(renderHandshake())
    assert not dead, \
        f"kernarg value overwritten before first read (reg, loaded, clobbered-by) = {dead}"


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
    def batchLoopCalls(node):
        return [c for c in ast.walk(node) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == "_emit_batch_loop"]

    # Several `if kernel['FusedGemmA2A']:` blocks exist (the prologue latch is
    # another); the hoist is the one that emits the store body.
    hoist = [n for n in ast.walk(kwa) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "kernel['FusedGemmA2A']"
             and batchLoopCalls(n)]
    assert len(hoist) == 1, f"expected one fused-A2A store-body hoist block, got {len(hoist)}"
    passes = [c for stmt in hoist[0].body for c in batchLoopCalls(stmt)]
    assert len(passes) == 2, \
        f"expected a PUSH pass and a LOCAL pass, got {len(passes)} batch-loop emissions"


@pytest.mark.parametrize("wavefrontSize", [64, 32])
def test_drain_poll_runs_under_an_exec_wider_than_one_lane(renderHandshake, wavefrontSize):
    """The DRAIN load must not be issued at EXEC=1.

    Everything before the poll runs single-lane, so the counter atomics fire once
    per WG rather than once per lane. A vector load issued at that width loads in
    lane 0 ONLY -- the other W-1 lanes are masked and never load -- so the v_cmp
    sets VCC from lane 0 alone and VCCZ becomes a one-slot predicate wearing the
    shape of a W-slot one. The DRAIN would then return as soon as the FIRST peer's
    slot filled: a barrier that looks correct under light load and corrupts data
    under real traffic.

    Asserting the load and the vccnz branch merely EXIST cannot see that, which is
    the whole point of this test. So: between the last EXEC write before the poll
    and the poll itself, EXEC must have been set from a computed mask rather than
    to the literal 1.
    """
    code = _codeLines(renderHandshake(wavefrontSize))
    poll = next(i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_drain_poll"))
    load = next(i for i, ln in enumerate(code[poll:], poll) if ln.startswith("global_load"))

    execWrites = [i for i, ln in enumerate(code[:load])
                  if ln.startswith("s_mov_b32 exec") or ln.startswith("s_mov_b64 exec")]
    assert execWrites, "no EXEC write before the DRAIN poll at all"
    last = code[execWrites[-1]]
    # `s_mov_b64 exec, 1` is the single-lane width the tally needs; reaching the
    # poll with that still in effect is the bug.
    assert not last.rstrip().endswith(", 1"), \
        f"DRAIN poll is issued at EXEC=1 (lane 0 only): {last}"
    # and the width must come from a register, i.e. be derived from the runtime W --
    # a wider literal would be wrong for every W but one.
    src = last.split(",", 1)[1].strip()
    assert src.startswith("s"), \
        f"EXEC for the poll must be a computed W-lane mask, not a literal: {last}"

    # The mask itself: one S_BFM, which computes ((1 << src0[5:0]) - 1) << src1[5:0]
    # -- i.e. width W at offset 0 -- anchored to the register the EXEC write actually
    # READS. It replaced a four-instruction arithmetic build (mov 1; shl W; sub 1;
    # zero the hi dword) whose middle two were each a silent hang if dropped: without
    # the `- 1` EXEC is 1 << W, a single lane at slot index W, one past the last real
    # peer and on a slot nobody ever writes; without the hi-dword zeroing lanes 32..63
    # poll whatever the pool left in the odd register. S_BFM has no separable halves,
    # so what remains to pin is that it is the mask instruction, that it is the right
    # WIDTH for the wave (a B32 feeding an EXEC pair leaves the hi dword untouched --
    # exactly the second hang above), that its width operand is a register rather than
    # a literal, and that its offset is 0.
    base = _baseReg(src)
    assert base, f"cannot resolve the EXEC mask's base SGPR from {src!r}: {last}"
    region = code[:execWrites[-1]]
    masks = [i for i, ln in enumerate(region)
             if ln.startswith("s_bfm_b") and len(_ops(ln)) == 3
             and _baseReg(_ops(ln)[0]) == base]
    assert len(masks) == 1, \
        f"expected exactly one s_bfm writing {base} ((1 << W) - 1), got: {[region[i] for i in masks]}"
    maskLine = region[masks[0]]
    # Widths are pinned against the kernel's declared wave size, not against each
    # other: deriving the expected mnemonic from the emitted EXEC write would stay
    # green if BOTH narrowed to b32 under wave64.
    wantExec = "s_mov_b32" if wavefrontSize == 32 else "s_mov_b64"
    wantMask = "s_bfm_b32" if wavefrontSize == 32 else "s_bfm_b64"
    assert last.startswith(wantExec + " "), f"wave{wavefrontSize} EXEC write must be {wantExec}: {last}"
    assert maskLine.startswith(wantMask + " "), \
        f"wave{wavefrontSize} mask must be {wantMask}: {maskLine}"
    dst, width, offset = _ops(maskLine)
    assert dst == src, \
        f"the s_bfm destination is not the operand EXEC reads ({src}): {maskLine}"
    # src0 is the WIDTH and src1 the offset. Swapped, this is ((1 << 0) - 1) << W == 0:
    # EXEC = 0, every lane masked off, the poll load never issues and the flags are
    # never read -- and the comment still says "(1 << W) - 1".
    assert _baseReg(width), \
        f"the s_bfm width must be a register (the runtime W), not a literal: {maskLine}"
    assert offset == "0", f"the s_bfm offset must be 0, got {offset!r}: {maskLine}"
    # Nothing may write the mask pair between building it and reading it into EXEC.
    # This is the window where a remnant of the arithmetic build would still be live:
    # one left AHEAD of the s_bfm is overwritten by it (S_BFM writes its destination
    # whole) and changes nothing THAT REACHES EXEC -- s_lshl_b32/s_sub_u32 carry
    # ImplicitWriteSCC and S_BFM does not, so such a leftover does still change SCC --
    # while one left behind it corrupts the mask.
    hiReg = "s%d" % (int(base[1:]) + 1)
    clobber = [ln for ln in region[masks[0] + 1:]
               if _ops(ln) and _baseReg(_ops(ln)[0]) in (base, hiReg)]
    assert not clobber, \
        f"{clobber} writes the mask register between `{maskLine}` and `{last}`"

    # And the arithmetic build must be GONE, not merely joined by the s_bfm. All four
    # of its members are fatal on their own if they are ever the operative value: a
    # seed of 2 gives (2 << W) - 1, i.e. W+1 lanes with the extra one on slot W; a
    # missing `- 1` gives 1 << W, one lane on that same never-filled slot; a missing
    # hi-dword zero leaves lanes 32..63 polling whatever the pool left there. All
    # three are silent hangs. A hybrid keeping any of them is dead code today and one
    # reordering away from live, so pin the absence rather than trusting the position.
    #
    # Matched by exact shape, not by "writes base": `base` is a recycled pool register
    # carrying fourteen unrelated values earlier in this handshake, among them
    # `s_lshl_b32 s28, s22, 3` (a flag-pointer scale, not a mask shift) and
    # `s_mov_b32 s28, s26` -- a looser predicate reports those and is a false alarm.
    def isOldBuild(ln):
        ops = _ops(ln)
        return ((ln.startswith("s_mov_b32 ")  and ops == [base, "1"])          # seed 1
             or (ln.startswith("s_lshl_b32 ") and ops[:2] == [base, base]      # 1 << W
                 and len(ops) == 3 and _baseReg(ops[2]))
             or (ln.startswith("s_sub_u32 ")  and ops == [base, base, "1"])    # - 1
             or (ln.startswith("s_mov_b32 ")  and ops == [hiReg, "0"]))        # hi = 0

    survivors = [ln for ln in region if isOldBuild(ln)]
    assert not survivors, \
        (f"the arithmetic mask build survived alongside `{maskLine}`: {survivors} -- "
         f"the mask must come from the s_bfm and nothing else")


def test_drain_exec_mask_width_is_the_FusedW_kernarg(renderHandshake):
    """The mask width must be W, not merely *a* register.

    "the width operand is a register" is a shape check, and shape checks are how a
    register-content bug hid on this branch before: the assembly read
    `s_mul_i32 s19, s10, ...  // myRank * N` while s10 held AM_tiles, because the
    comment is frozen at construction time and the register is decided at emission
    time by the pool. A width operand pointing at the wrong SGPR gives EXEC an
    arbitrary lane count -- too few and the barrier releases early, too many and the
    extra lanes poll slots past the W-slot flag allocation -- and reads correctly.
    """
    text = renderHandshake()
    # after renderHandshake(), which imports Tensile.Component first (circular-import guard)
    from Tensile.Components.Signature import fusedA2AKernArgLayout

    offset, clobbers = _maskWidthProvenance(text)
    # The fixture puts the fused segment at base 0, so the segment-relative offset
    # the layout reports is the absolute one the emitter passes.
    want = hex(fusedA2AKernArgLayout()["FusedW"])
    assert offset == want, \
        f"the s_bfm width register was last loaded from {offset}, not FusedW ({want})"
    assert not clobbers, \
        f"the width register is overwritten between its FusedW load and the s_bfm: {clobbers}"


def test_rendering_is_byte_identical_to_the_golden(renderHandshake):
    """Characterization pin on the whole handshake.

    The structural tests each assert one fact, so between them they leave gaps -- a
    register clobber introduced by reordering blocks sat in exactly such a gap and
    reached review. This compares the whole rendering, so drift shows up whether or
    not someone thought to assert on it.

    Intentionally changing the handshake? Regenerate deliberately:
        python Tensile/Tests/unit/test_fusedA2A_drain_last.py --update-golden
    and justify the diff in review -- do not regenerate to silence a red.
    """
    with open(_GOLDEN) as f:
        golden = f.read()
    assert renderHandshake() == golden, \
        "handshake rendering drifted from the golden; see the docstring"


def test_max_ranks_guard_fires_when_the_constant_outgrows_the_mask():
    """The import-time bound on FUSED_A2A_MAX_RANKS must have teeth.

    The guard in Signature.py exists for an event that has never happened -- someone
    raising the constant past what the S_BFM width operand can encode -- so no
    ordinary run exercises it, and a guard nobody has watched fail is not evidence.
    This is that mutation, made permanent: the module-level `if` is located, lifted
    out, and re-executed against a constant of 32 (the wave32 arm's first wrapping
    width). Locating it also pins that it still EXISTS -- deleting the guard reddens
    here rather than passing quietly, which a test that merely re-checked
    `FUSED_A2A_MAX_RANKS <= 31` would not do.
    """
    path = os.path.join(TENSILE_ROOT, "Tensile/Components/Signature.py")
    with open(path) as f:
        mod = ast.parse(f.read(), filename=path)

    guards = [n for n in mod.body if isinstance(n, ast.If)
              and "FUSED_A2A_MAX_RANKS" in ast.unparse(n.test)
              and any(isinstance(s, ast.Raise) for s in ast.walk(n))]
    assert len(guards) == 1, \
        (f"expected exactly one module-level FUSED_A2A_MAX_RANKS bound guard in "
         f"{path}, found {len(guards)} -- the DRAIN EXEC mask bound is unenforced")

    # 8 must pass and 32 must not; a guard that raises unconditionally, or one whose
    # comparison drifted the wrong way, fails one of these two.
    def run(value):
        ns = {"FUSED_A2A_MAX_RANKS": value}
        exec(compile(ast.Module(body=[guards[0]], type_ignores=[]), path, "exec"), ns)

    run(8)  # the shipped value: must not raise
    with pytest.raises(ValueError, match="EXEC becomes empty"):
        run(32)


if __name__ == "__main__":
    if "--update-golden" in sys.argv:
        os.makedirs(os.path.dirname(_GOLDEN), exist_ok=True)
        with open(_GOLDEN, "w") as f:
            f.write(_renderHandshake())
        print("wrote %s" % _GOLDEN)
    else:
        print("pass --update-golden to regenerate the handshake golden")
