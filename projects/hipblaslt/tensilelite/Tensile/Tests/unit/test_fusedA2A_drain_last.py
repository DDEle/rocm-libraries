#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Fused-A2A DRAIN ownership: the single globally-last workgroup drains, elected
# by a grid-wide counter3, and polls all W flag slots with one vector load
# reduced by VCCZ.
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

    `wavefrontSize` picks the wave width and defaults to 64. The argLoader stub
    echoes its arguments instead of rendering a fixed comment.
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
    w.sgprPool.checkOut(8)
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


@pytest.fixture
def renderHandshake():
    return _renderHandshake


def _codeLines(text):
    """Comment-stripped, blank-stripped instruction stream."""
    return [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]


def _ops(line):
    """Operands of a comment-stripped instruction, in source order."""
    parts = line.split(None, 1)
    return [] if len(parts) < 2 else [op.strip() for op in parts[1].split(",")]


def _baseReg(operand):
    """Base SGPR of an operand written either as `s28` or as the pair `s[28:29]`."""
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


# Matches the echoing argLoader stub's rendering of one kernarg load.
_LOAD_RE = re.compile(r"//\s*loadKernArg (\d+) KernArgAddress .*?dword=(\d+).*?sgprOffset=(\S+)")


def _deadKernargLoads(text):
    """Kernarg loads whose register is overwritten by another load before any read.

    Reports (reg, offset, clobberOffset). A value clobbered after one read but
    before a later one is not modelled. Reads are counted on the comment-stripped
    instruction only.
    """
    raw = [ln for ln in text.splitlines() if ln.strip()]
    code = [ln.split("//")[0] for ln in raw]
    loads = []
    for i, ln in enumerate(raw):
        m = _LOAD_RE.search(ln)
        if m and m.group(2) == "1":          # single-dword loads; pairs render as s[n:n+1]
            loads.append((i, int(m.group(1)), m.group(3)))

    assert loads, \
        "_LOAD_RE matched no kernarg loads; the emitted comment format has likely " \
        "drifted, so this check is inspecting nothing: %s" % _LOAD_RE.pattern

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

    Returns (sgprOffset of the last single-dword kernarg load into that register
    before the s_bfm, list of instructions writing it in between).
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
    from rocisa.code import Module
    from Tensile.Components.GlobalWriteBatch import emitFusedA2ATotalWGsLatch

    m = Module("latch")
    emitFusedA2ATotalWGsLatch(m, "FusedTotalWGs")
    text = str(m)

    code = [ln for ln in (l.split("//")[0].strip() for l in text.splitlines()) if ln]
    assert len(code) == 1, code
    assert code[0].startswith("s_mul_i32 "), code[0]
    ops = [op.strip() for op in code[0].split(None, 1)[1].split(",")]
    assert len(ops) == 3, f"expected a 3-operand s_mul_i32, got {len(ops)}: {code[0]}"
    dst, src0, src1 = ops
    assert "sgprFusedTotalWGs" in dst, code[0]
    assert "sgprNumWorkGroups0" in src0, code[0]
    assert "sgprNumWorkGroups1" in src1, code[0]


def test_fused_a2a_also_locks_out_the_runtime_GSU_override():
    """The SupportUserGSU lockout sits in the FusedGemmA2A block, on the accepted path."""
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())

    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    body = blocks[0].body

    TARGET = "state['InternalSupportParams']['SupportUserGSU'] = False"

    def index_of(pred):
        return next((i for i, stmt in enumerate(body) if pred(stmt)), None)

    lockout = index_of(lambda s: ast.unparse(s) == TARGET)
    assert lockout is not None, f"FusedGemmA2A block no longer sets {TARGET}"

    gsu_reject = index_of(lambda s: isinstance(s, ast.If)
                          and "GlobalSplitU" in ast.unparse(s.test)
                          and any(isinstance(b, ast.Return) for b in s.body))
    assert gsu_reject is not None, "the compile-time GlobalSplitU rejection went missing"
    assert lockout > gsu_reject, \
        "lockout must come AFTER the GSU rejection, not inside/before it"

    assert not any(isinstance(s, (ast.Return, ast.Raise)) for s in body[:lockout]), \
        "a statement before the lockout leaves the block: the lockout is dead code"


def test_batch_guard_is_host_side_not_compile_time():
    """The batch guard must test the EXTENT, which only the host can see."""
    with open(os.path.join(TENSILE_ROOT, "Tensile/SolutionStructs/Solution.py")) as f:
        tree = ast.parse(f.read())
    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "state['FusedGemmA2A']"]
    assert len(blocks) == 1, "expected exactly one `if state['FusedGemmA2A']:` block"
    rejects = [ast.unparse(s) for s in blocks[0].body
               if isinstance(s, ast.If) and "NumIndicesBatch" in ast.unparse(s.test)]
    assert not rejects, \
        f"fused block still rejects on a DECLARED batch index (zero kernels): {rejects}"

    host = os.path.join(TENSILE_ROOT, "client/src/FusedA2AClient.cpp")
    with open(host) as f:
        src = f.read()
    # Strip comments first: every check below is a text match.
    src = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.S))

    # The count may reach the predicate through a local.
    countNames = {"problem->batchIndices().size()"}
    countNames.update(re.findall(r"\b(\w+)\s*=\s*problem->batchIndices\(\)\.size\(\)", src))

    guards = [m for m in re.finditer(r"if\s*\(([^{]*?)\)\s*\{", src, re.S)
              if "batchSize(" in m.group(1)]
    assert len(guards) == 1, \
        f"expected exactly one host guard testing a batch EXTENT, found {len(guards)}"
    cond = guards[0].group(1)
    body = _braceBlock(src, guards[0].end() - 1)

    extent = re.findall(r"problem->batchSize\(\s*\w+\s*\)\s*(!=|==|<|>|<=|>=)\s*(\d+)", cond)
    assert extent == [("!=", "1")], \
        f"the batch guard must reject every extent other than 1; found {extent} in: {cond}"

    counts = [(op, v) for n in countNames
              for op, v in re.findall(re.escape(n) + r"\s*(!=|==|<|>|<=|>=)\s*(\d+)", cond)]
    assert (">", "1") in counts, \
        f"the batch guard must refuse a second DECLARED batch index; found {counts} in: {cond}"

    assert re.search(r"\breturn\s+1\s*;", body), \
        f"the batch guard detects a bad batch shape but never bails out (no `return 1;`):\n{body}"


################################################################################
# The handshake preamble is hoisted above the PUSH gate and every WG tallies
# itself at counter3.
################################################################################


def _pushGateIndex(text):
    """Index, in the comment-stripped stream, of the PUSH gate's branch.

    Located by its comment.
    """
    raw = [l for l in text.splitlines() if l.strip()]
    hits = [i for i, ln in enumerate(raw) if "not a PUSH WG" in ln]
    assert len(hits) == 1, f"expected exactly one PUSH gate branch, got {hits}"
    code = _codeLines(text)
    gate = raw[hits[0]].split("//")[0].strip()
    assert gate, f"PUSH gate line is comment-only: {raw[hits[0]]!r}"
    return code.index(gate), gate


def test_push_gate_falls_through_to_the_local_tally_then_counter3(renderHandshake):
    """The gate's not-taken edge reaches counter3 via the local-tally block."""
    text = renderHandshake()
    code = _codeLines(text)
    _, gate = _pushGateIndex(text)
    assert gate == "s_cbranch_scc0 label_fusedA2A_local_tally", gate

    localIdx = next(i for i, ln in enumerate(code)
                    if ln.startswith("label_fusedA2A_local_tally"))
    c3Idx = next(i for i, ln in enumerate(code)
                 if ln.startswith("label_fusedA2A_counter3"))
    assert localIdx < c3Idx, "the local tally must fall through into counter3"
    # Fall-through: no unconditional branch between them.
    between = [ln for ln in code[localIdx + 1:c3Idx] if ln.startswith("s_branch ")]
    assert not between, f"the local tally does not fall through: {between}"


def test_local_path_skips_the_barrier_but_keeps_the_election(renderHandshake):
    """The store wait and the barrier are PUSH-only; the election is not."""
    text = renderHandshake()
    code = _codeLines(text)
    gateIdx, _ = _pushGateIndex(text)

    # (a) the gate's predicate must be work-group-uniform
    assert code[gateIdx - 1].startswith("s_cmp_"), code[gateIdx - 1]
    assert "sgprWorkGroup0" in code[gateIdx - 1], code[gateIdx - 1]

    # (b) exactly one barrier, and it is BELOW the gate (PUSH-only)
    barrier = [i for i, ln in enumerate(code) if ln.startswith("s_barrier")]
    assert len(barrier) == 1, f"expected one s_barrier, got {barrier}"
    assert barrier[0] > gateIdx, "s_barrier must be PUSH-only, i.e. below the gate"

    # (c) the store wait stays immediately ahead of it
    assert code[barrier[0] - 1].startswith("s_waitcnt"), code[barrier[0] - 1]

    # (d) two Serial elections: PUSH (below the gate) and local (after its label)
    election = [i for i, ln in enumerate(code)
                if ln.startswith("v_readfirstlane_b32") and "vgprSerial" in ln]
    assert len(election) == 2, f"expected two Serial readfirstlanes, got {election}"
    localIdx = next(i for i, ln in enumerate(code)
                    if ln.startswith("label_fusedA2A_local_tally"))
    assert gateIdx < election[0] < localIdx < election[1], \
        f"gate={gateIdx} elections={election} localTally={localIdx}"

    # (e) three single-lane EXEC writes, pinned EXACTLY -- do not relax to >=
    lane0 = [i for i, ln in enumerate(code) if ln.replace(" ", "").endswith("exec,1")]
    assert len(lane0) == 3, (
        f"expected three single-lane EXEC writes (PUSH election, local tally, "
        f"post-DRAIN restore), got {lane0}")


def test_store_wait_precedes_the_barrier(renderHandshake):
    """A vector-memory wait must precede s_barrier.

    SWaitCnt(vscnt=0) lowers to `s_waitcnt vmcnt(0)` on gfx950, so both spellings
    are accepted.
    """
    code = _codeLines(renderHandshake())
    waits = [i for i, ln in enumerate(code)
             if ln.startswith("s_waitcnt") and ("vmcnt" in ln or "vscnt" in ln)]
    barrier = [i for i, ln in enumerate(code) if ln.startswith("s_barrier")]
    assert waits, "no vector-memory wait emitted at all"
    assert len(barrier) == 1, f"expected one s_barrier, got {barrier}"
    assert waits[0] < barrier[0], \
        f"a store wait must precede s_barrier (wait {waits[0]}, barrier {barrier[0]})"


def test_counter3_is_incremented_exactly_once_by_a_scalar_atomic(renderHandshake):
    """The tally fires exactly once per work-group, as an SMEM atomic."""
    code = _codeLines(renderHandshake())
    label = [i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3")]
    assert len(label) == 1, f"expected one counter3 label definition, got {label}"
    tail = code[label[0]:]
    assert not [ln for ln in tail if ln.startswith("global_atomic")], \
        "the tally must not use a vector atomic"
    atomics = [ln for ln in tail if ln.startswith("s_atomic")]
    assert len(atomics) == 1, f"counter3 must increment exactly once, got: {atomics}"
    assert atomics[0].startswith("s_atomic_inc "), atomics[0]
    # GLC is what makes an SMEM atomic return its pre-op value (CDNA4 ISA Table 75).
    assert atomics[0].rstrip().endswith("glc"), atomics[0]


def test_counter3_election_compares_against_the_latched_total(renderHandshake):
    """The tally is compared against FusedTotalWGs-1, the prologue latch minus one."""
    code = _codeLines(renderHandshake())
    label = next(i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_counter3"))

    # the limit register is derived from the latch, and from nothing else
    subs = [(i, ln) for i, ln in enumerate(code[label:], label)
            if ln.startswith("s_sub_u32") and "sgprFusedTotalWGs" in ln]
    assert len(subs) == 1, f"expected one FusedTotalWGs-derived limit, got {subs}"
    subOps = _ops(subs[0][1])
    limitReg = subOps[0]
    assert subOps[1] == "s[sgprFusedTotalWGs]" and subOps[2] == "1", subs[0][1]

    # Anchor on the tally: the atomic, then the first compare after it.
    atomic = next(i for i, ln in enumerate(code[label:], label)
                  if ln.startswith("s_atomic_inc "))
    sdataReg = _ops(code[atomic])[0]
    cmp_i = next(i for i, ln in enumerate(code[atomic:], atomic)
                 if ln.startswith("s_cmp_eq_u32"))
    assert _ops(code[cmp_i]) == [sdataReg, limitReg], (
        f"the election must compare the atomic's returned SDATA ({sdataReg}) "
        f"against FusedTotalWGs-1 ({limitReg}), got {code[cmp_i]}")
    # and it must be what decides the DRAIN, i.e. immediately consumed by the branch
    assert code[cmp_i + 1].startswith("s_cbranch_scc0 "), code[cmp_i:cmp_i + 2]


def test_no_kernarg_value_is_clobbered_before_it_is_read(renderHandshake):
    """No two kernarg values share a live range in the emitted handshake."""
    dead = _deadKernargLoads(renderHandshake())
    assert not dead, \
        f"kernarg value overwritten before first read (reg, loaded, clobbered-by) = {dead}"


def test_every_surviving_wg_runs_the_handshake_once():
    """The store body is emitted twice around one hoisted dispatch gate.

    A PUSH pass and a LOCAL pass, each work-group running exactly one of them, and
    the handshake call inside is not gated on the dispatch mode.
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
    """The DRAIN load must not be issued at EXEC=1."""
    code = _codeLines(renderHandshake(wavefrontSize))
    poll = next(i for i, ln in enumerate(code) if ln.startswith("label_fusedA2A_drain_poll"))
    load = next(i for i, ln in enumerate(code[poll:], poll) if ln.startswith("global_load"))

    execWrites = [i for i, ln in enumerate(code[:load])
                  if ln.startswith("s_mov_b32 exec") or ln.startswith("s_mov_b64 exec")]
    assert execWrites, "no EXEC write before the DRAIN poll at all"
    last = code[execWrites[-1]]
    assert not last.rstrip().endswith(", 1"), \
        f"DRAIN poll is issued at EXEC=1 (lane 0 only): {last}"
    # The width must come from a register, i.e. be derived from the runtime W.
    src = last.split(",", 1)[1].strip()
    assert src.startswith("s"), \
        f"EXEC for the poll must be a computed W-lane mask, not a literal: {last}"

    # One S_BFM computing ((1 << src0[5:0]) - 1) << src1[5:0], i.e. width W at
    # offset 0, anchored to the register the EXEC write reads.
    base = _baseReg(src)
    assert base, f"cannot resolve the EXEC mask's base SGPR from {src!r}: {last}"
    region = code[:execWrites[-1]]
    masks = [i for i, ln in enumerate(region)
             if ln.startswith("s_bfm_b") and len(_ops(ln)) == 3
             and _baseReg(_ops(ln)[0]) == base]
    assert len(masks) == 1, \
        f"expected exactly one s_bfm writing {base} ((1 << W) - 1), got: {[region[i] for i in masks]}"
    maskLine = region[masks[0]]
    # Widths are pinned against the kernel's declared wave size, not against each other.
    wantExec = "s_mov_b32" if wavefrontSize == 32 else "s_mov_b64"
    wantMask = "s_bfm_b32" if wavefrontSize == 32 else "s_bfm_b64"
    assert last.startswith(wantExec + " "), f"wave{wavefrontSize} EXEC write must be {wantExec}: {last}"
    assert maskLine.startswith(wantMask + " "), \
        f"wave{wavefrontSize} mask must be {wantMask}: {maskLine}"
    dst, width, offset = _ops(maskLine)
    assert dst == src, \
        f"the s_bfm destination is not the operand EXEC reads ({src}): {maskLine}"
    # src0 is the WIDTH, src1 the offset.
    assert _baseReg(width), \
        f"the s_bfm width must be a register (the runtime W), not a literal: {maskLine}"
    assert offset == "0", f"the s_bfm offset must be 0, got {offset!r}: {maskLine}"
    # Nothing may write the mask pair between the s_bfm and the EXEC read. The scan
    # starts after masks[0] because S_BFM overwrites its destination whole.
    hiReg = "s%d" % (int(base[1:]) + 1)
    clobber = [ln for ln in region[masks[0] + 1:]
               if _ops(ln) and _baseReg(_ops(ln)[0]) in (base, hiReg)]
    assert not clobber, \
        f"{clobber} writes the mask register between `{maskLine}` and `{last}`"

    # The arithmetic build must be GONE, not merely joined by the s_bfm. Matched by
    # exact shape rather than "writes base": `base` is a recycled pool register that
    # also carries unrelated values earlier in the handshake.
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
    """The mask width register must be the one loaded from the FusedW kernarg."""
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

    Intentionally changing the handshake? Regenerate deliberately:
        python Tensile/Tests/unit/test_fusedA2A_drain_last.py --update-golden
    and justify the diff in review -- do not regenerate to silence a red.
    """
    with open(_GOLDEN) as f:
        golden = f.read()
    assert renderHandshake() == golden, \
        "handshake rendering drifted from the golden; see the docstring"


def test_max_ranks_guard_fires_when_the_constant_outgrows_the_mask():
    """The import-time bound on FUSED_A2A_MAX_RANKS must exist and have teeth.

    The EXEC-mask guard's `if` in Signature.py is located, lifted out, and
    re-executed against constants either side of the bound.
    """
    path = os.path.join(TENSILE_ROOT, "Tensile/Components/Signature.py")
    with open(path) as f:
        mod = ast.parse(f.read(), filename=path)

    # "EXEC becomes empty" also names the string this test's own pytest.raises matches below.
    guards = [n for n in mod.body if isinstance(n, ast.If)
              and "FUSED_A2A_MAX_RANKS" in ast.unparse(n.test)
              and any(isinstance(s, ast.Raise) for s in ast.walk(n))
              and "EXEC becomes empty" in ast.unparse(n)]
    assert len(guards) == 1, \
        (f"expected exactly one module-level FUSED_A2A_MAX_RANKS guard raising on "
         f"'EXEC becomes empty' in {path}, found {len(guards)} -- the DRAIN EXEC mask "
         f"bound is unenforced")

    def run(value):
        ns = {"FUSED_A2A_MAX_RANKS": value}
        exec(compile(ast.Module(body=[guards[0]], type_ignores=[]), path, "exec"), ns)

    run(8)   # the shipped value: must not raise
    run(31)  # the boundary, the widest value S_BFM_B32 can encode: must not raise
    with pytest.raises(ValueError, match="EXEC becomes empty"):
        run(32)


def test_max_ranks_twins_hold_the_same_value():
    """The Python and C++ declarations of FUSED_A2A_MAX_RANKS must agree.

    The constant is declared twice: Signature.py sizes the kernarg segment the
    kernel reads, FusedA2AKernArg.hpp sizes what the host appends.
    """
    py_path = os.path.join(TENSILE_ROOT, "Tensile/Components/Signature.py")
    with open(py_path) as f:
        py_tree = ast.parse(f.read(), filename=py_path)
    py_vals = [n.value.value for n in py_tree.body
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "FUSED_A2A_MAX_RANKS" for t in n.targets)
               and isinstance(n.value, ast.Constant)]
    assert len(py_vals) == 1, \
        f"expected exactly one module-level FUSED_A2A_MAX_RANKS in {py_path}, got {py_vals}"

    hpp_path = os.path.join(TENSILE_ROOT, "client/include/FusedA2AKernArg.hpp")
    with open(hpp_path) as f:
        cpp_vals = re.findall(
            r"constexpr\s+int\s+FUSED_A2A_MAX_RANKS\s*=\s*(\d+)\s*;", f.read())
    assert len(cpp_vals) == 1, \
        f"expected exactly one constexpr FUSED_A2A_MAX_RANKS in {hpp_path}, got {cpp_vals}"

    assert py_vals[0] == int(cpp_vals[0]), (
        f"FUSED_A2A_MAX_RANKS disagrees across the ABI: Signature.py={py_vals[0]} "
        f"vs FusedA2AKernArg.hpp={cpp_vals[0]}. The host would append a differently "
        f"sized fused segment than the kernel metadata reserves.")


def test_peer_recv_offset_twins_hold_the_same_value():
    """The Python and C++ declarations of FUSED_A2A_PEER_RECV_OFFSET must agree."""
    py_path = os.path.join(TENSILE_ROOT, "Tensile/Components/Signature.py")
    with open(py_path) as f:
        py_tree = ast.parse(f.read(), filename=py_path)
    py_vals = [n.value.value for n in py_tree.body
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "FUSED_A2A_PEER_RECV_OFFSET" for t in n.targets)
               and isinstance(n.value, ast.Constant)]
    assert len(py_vals) == 1, \
        f"expected exactly one module-level FUSED_A2A_PEER_RECV_OFFSET in {py_path}, got {py_vals}"

    hpp_path = os.path.join(TENSILE_ROOT, "client/include/FusedA2AKernArg.hpp")
    with open(hpp_path) as f:
        cpp_vals = re.findall(
            r"constexpr\s+size_t\s+FUSED_A2A_PEER_RECV_OFFSET\s*=\s*(\d+)\s*;", f.read())
    assert len(cpp_vals) == 1, \
        f"expected exactly one constexpr FUSED_A2A_PEER_RECV_OFFSET in {hpp_path}, got {cpp_vals}"

    assert py_vals[0] == int(cpp_vals[0]), (
        f"FUSED_A2A_PEER_RECV_OFFSET disagrees across the ABI: Signature.py={py_vals[0]} "
        f"vs FusedA2AKernArg.hpp={cpp_vals[0]}. The host would write recv into a "
        f"different offset than the one the kernel epilogue reads from.")


if __name__ == "__main__":
    if "--update-golden" in sys.argv:
        os.makedirs(os.path.dirname(_GOLDEN), exist_ok=True)
        with open(_GOLDEN, "w") as f:
            f.write(_renderHandshake())
        print("wrote %s" % _GOLDEN)
    else:
        print("pass --update-golden to regenerate the handshake golden")
