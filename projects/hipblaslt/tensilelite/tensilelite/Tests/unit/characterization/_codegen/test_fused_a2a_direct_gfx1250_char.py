# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) direct-peer handshake characterization.

Per handshake of every emitted kernel: the direct-peer mask is FusedDrain[15:8]; its
dst_rank bit gates the SDMA COPY submit and the queue-tail ATOMICs; on the direct path
no SDMA queue field is touched and one system-scope atomic raises
peer_flagPtr[dst_rank][my_rank]; the DRAIN_SEND poll compares against
W - FusedDrain[23:16]. Matched on instructions and operands only.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config
from tensilelite.Components.SdmaRingEmitter import OFF_flagPtr, PEER_GROUP_BYTES
from tensilelite.Components.Signature import (
    FUSED_A2A_COUNTER1_OFFSET, FUSED_A2A_COUNTER2_OFFSET, FUSED_A2A_COUNTER3_OFFSET,
    FUSED_A2A_DIRECT_COUNT_SHIFT, FUSED_A2A_DIRECT_MASK_SHIFT, FUSED_A2A_MAX_RANKS,
    FUSED_A2A_OUTBOUND_OFFSET, fusedA2AKernArgLayout)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_CONFIGS = {
    "plain": (os.path.join(_COMM, "fused_a2a.yaml"), None),
    "cluster": (os.path.join(_COMM, "fused_a2a_cluster.yaml"), [2, 4]),
}

_KERNARG = "s[sgprKernArgAddress:sgprKernArgAddress+1]"
_PEER_GROUP = "s[sgprFusedPeerGroupPtr]"
_MASK_BFE = (FUSED_A2A_MAX_RANKS << 16) | FUSED_A2A_DIRECT_MASK_SHIFT
_COUNT_BFE = (FUSED_A2A_MAX_RANKS << 16) | FUSED_A2A_DIRECT_COUNT_SHIFT
_HINTS = ("s_wait_alu", "s_set_vgpr_msb", "s_nop", "s_delay_alu")
_NO_DST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_wait", "s_nop", "s_set_vgpr_msb",
           "s_barrier", "s_delay_alu", "s_prefetch", "global_store", "buffer_store")


def _lines(src):
    """(label, op, operands, modifiers) per code line; label is set on label lines only."""
    out = []
    for ln in src.splitlines():
        s = ln.split("//")[0].strip()
        if not s or s.startswith(("/", ";", ".")):
            continue
        if s.endswith(":"):
            out.append((s[:-1], None, [], {}))
            continue
        op, _, rest = s.partition(" ")
        toks = [t for t in re.split(r",\s*|\s+", rest) if t]
        mods = dict(t.split(":", 1) for t in toks if re.match(r"[a-z_]+:", t))
        out.append((None, op, [t for t in toks if not re.match(r"[a-z_]+:", t)], mods))
    return out


def _regs(tok):
    m = re.fullmatch(r"([sv])\[(\d+):(\d+)\]", tok)
    if m:
        return {f"{m[1]}{i}" for i in range(int(m[2]), int(m[3]) + 1)}
    m = re.fullmatch(r"s\[(\w+?)(?:\+(\d+))?(?::\1\+(\d+))?\]", tok)
    if m:
        lo = int(m[2] or 0)
        return {f"{m[1]}+{i}" for i in range(lo, int(m[3] or lo) + 1)}
    return {tok}


def _writes(line):
    _l, op, ops, _m = line
    if op is None or op.startswith(_NO_DST) or not ops:
        return set()
    return _regs(ops[0])


def _last_def(lines, order, pos, tok):
    """Index of the last instruction in order[:pos] that writes any register of tok."""
    want = _regs(tok)
    for i in reversed(order[:pos]):
        if _writes(lines[i]) & want:
            return i
    raise AssertionError(f"no definition of {tok}")


def _is_kernarg_load(line, op, off):
    return line[1] == op and line[2][1:] == [_KERNARG, hex(off)]


def _touches_queue_or_exec(line, fusedBase):
    _l, op, ops, mods = line
    if op is None:
        return False
    if op.startswith("s_load") and _PEER_GROUP in ops:
        return int(mods.get("offset", "0"), 0) != fusedBase + OFF_flagPtr
    return (op.startswith("global_store") or op in ("global_atomic_cmpswap_b64", "global_atomic_max_u64")
            or (op.startswith("s_mov") and ops[0] == "exec_lo"))


def _prev_non_hint(lines, i):
    i -= 1
    while lines[i][1] is not None and lines[i][1].startswith(_HINTS):
        i -= 1
    return i


def _gate(lines, i, maskReg):
    """dst_rank register if lines[i] is a conditional branch on the mask bit, else None."""
    if lines[i][1] != "s_cbranch_scc1":
        return None
    _l, op, ops, _m = lines[_prev_non_hint(lines, i)]
    return ops[1] if op == "s_bitcmp1_b32" and ops[0] == maskReg else None


def _atomic_offset(line):
    return int(line[3].get("offset", "0"), 0) if line[1] == "global_atomic_inc_u32" else None


def _walk_direct(lines, labels, start, maskReg):
    """Indices executed from start, taking exactly the mask-gated branches, up to the counter3 atomic."""
    trace, i = [], start
    for _ in range(4000):
        label, op, ops, _m = lines[i]
        if label is not None:
            i += 1
            continue
        trace.append(i)
        if _atomic_offset(lines[i]) == FUSED_A2A_COUNTER3_OFFSET:
            return trace
        if op == "s_branch" or (op.startswith("s_cbranch") and _gate(lines, i, maskReg)):
            j = labels[ops[0]]
            assert j > i, f"direct path loops back at {op} {ops[0]}"
            i = j
        else:
            i += 1
    raise AssertionError("direct path does not reach the counter3 tally")


def _check_handshake(lines, labels, fusedBase, maskLoad, maskBfe, end):
    layout = fusedA2AKernArgLayout()
    linear = list(range(len(lines)))
    maskReg = lines[maskBfe][2][0]
    assert any(lines[i][1] == "s_wait_kmcnt" and int(lines[i][2][0], 0) == 0
               for i in range(maskLoad, maskBfe)), "mask extracted before its FusedDrain load is waited on"

    # Entered at the election's lane-0 EXEC.
    starts = [i for i in range(maskLoad) if lines[i][1] == "s_mov_b32" and lines[i][2][0] == "exec_lo"]
    assert starts and lines[starts[-1]][2][1] == "1", "handshake not entered at EXEC = lane 0"

    # Non-direct path: every SDMA queue touch sits inside one of the two mask-gated spans.
    region = range(maskLoad, end)
    gates = [(i, labels[lines[i][2][0]]) for i in region if _gate(lines, i, maskReg)]
    assert len(gates) == 2, f"expected 2 mask-gated branches, found {len(gates)}"
    (b1, l1), (b2, l2) = gates
    c1 = [i for i in region if _atomic_offset(lines[i]) == FUSED_A2A_COUNTER1_OFFSET]
    c2 = [i for i in region if _atomic_offset(lines[i]) == FUSED_A2A_COUNTER2_OFFSET]
    assert len(c1) == 1 and len(c2) == 1, "counter1 / counter2 atomic missing or repeated"
    assert c1[0] < b1 < l1 <= c2[0] < b2 < l2, "mask tests out of order with the counter elections"
    for b in (b1, b2):
        assert _last_def(lines, linear, _prev_non_hint(lines, b), maskReg) == maskBfe, (
            "mask register redefined before a mask test")
    tailExit = _prev_non_hint(lines, l2)
    assert lines[tailExit][1] == "s_branch" and labels[lines[tailExit][2][0]] > l2, (
        "the queue-tail path falls into the direct-peer block")
    pushEnd = labels[lines[tailExit][2][0]]
    sdma = [i for i in range(maskLoad, pushEnd) if _touches_queue_or_exec(lines[i], fusedBase)]
    span1 = [i for i in sdma if b1 < i < l1]
    span2 = [i for i in sdma if b2 < i < l2]
    assert span1 and span2, "a mask-gated span holds no SDMA submit"
    assert len(span1) + len(span2) == len(sdma), (
        f"{len(sdma) - len(span1) - len(span2)} SDMA queue touches outside the mask-gated spans")

    # Direct path: no SDMA queue field, counter1/2 unchanged, one system-scope flag atomic.
    trace = _walk_direct(lines, labels, starts[-1], maskReg)
    assert not [i for i in trace[1:] if _touches_queue_or_exec(lines[i], fusedBase)], (
        "direct path touches an SDMA queue or EXEC")
    assert sum(1 for i in trace if _gate(lines, i, maskReg)) == 2, "direct path skips a mask test"
    atomics = [i for i in trace if lines[i][1].startswith("global_atomic")]
    assert [_atomic_offset(lines[i]) for i in atomics] == [
        FUSED_A2A_COUNTER1_OFFSET, FUSED_A2A_COUNTER2_OFFSET, 0, FUSED_A2A_COUNTER3_OFFSET], (
        "direct path atomics are not counter1, counter2, flag, counter3")
    scopes = [lines[i][3].get("scope") for i in atomics]
    assert scopes == ["SCOPE_DEV", "SCOPE_DEV", "SCOPE_SYS", "SCOPE_DEV"], scopes

    def defOf(user, tok):
        return _last_def(lines, trace, trace.index(user), tok)

    _l, _op, (_ret, vOff, vData, saddr), _m = lines[atomics[2]]
    assert lines[defOf(atomics[2], vOff)][2][1] == "0", "direct flag atomic has a lane offset"
    dataAt = defOf(atomics[2], vData)
    assert lines[dataAt][1] == "v_mov_b32", "direct flag atomic data is not a v_mov of the wrap limit"
    limit = lines[defOf(dataAt, lines[dataAt][2][1])]
    assert limit[1] == "s_mov_b32" and int(limit[2][1], 0) & 0xFFFFFFFF == 0xFFFFFFFF, (
        f"direct flag atomic wrap limit is {limit[2][1]}, not 0xFFFFFFFF")
    lo, hi = sorted(_regs(saddr), key=lambda r: int(r[1:]))
    loAt, hiAt = defOf(atomics[2], lo), defOf(atomics[2], hi)
    addLo, addHi = lines[loAt], lines[hiAt]
    assert addLo[1] == "s_add_u32" and addHi[1] == "s_addc_u32" and addHi[2][2] == "0", (
        "direct flag address is not a 64-bit add")
    flagLo, rankOff = addLo[2][1], addLo[2][2]
    base = defOf(loAt, flagLo)
    assert lines[base][1] == "s_load_b128" and lines[base][2][2] == _PEER_GROUP, (
        "direct flag base is not a peer-group load")
    assert int(lines[base][3]["offset"], 0) == fusedBase + OFF_flagPtr, "direct flag base is not peer_flagPtr"
    assert min(_regs(lines[base][2][0]), key=lambda r: int(r[1:])) == flagLo, (
        "flag addr lo is not peer_flagPtr lo")
    assert defOf(hiAt, addHi[2][1]) == base, "flag addr hi is not peer_flagPtr hi"
    shlAt = defOf(loAt, rankOff)
    shl = lines[shlAt]
    assert shl[1] == "s_lshl_b32" and shl[2][2] == "2", "flag slot offset is not rank * 4"
    rank = defOf(shlAt, shl[2][1])
    assert _is_kernarg_load(lines[rank], "s_load_b32", fusedBase + layout["FusedMyRank"]), (
        "flag slot is not indexed by my_rank")

    # The tested bit is dst_rank: the register that selects the peer group.
    dstRank = _gate(lines, b1, maskReg)
    assert dstRank == _gate(lines, b2, maskReg), "the two mask tests use different bit registers"
    assert _last_def(lines, linear, b2, dstRank) == _last_def(lines, linear, b1, dstRank), (
        "dst_rank redefined between the mask tests")
    sel = _last_def(lines, trace, trace.index(base), _PEER_GROUP)
    assert lines[sel][1] == "s_mul_i32" and lines[sel][2][1:] == [dstRank, str(PEER_GROUP_BYTES)], (
        "mask test does not use dst_rank")
    assert _last_def(lines, trace, trace.index(b1), dstRank) < sel, "dst_rank redefined after the peer-group select"


def _check_drain_send(lines, fusedBase, start, end):
    layout = fusedA2AKernArgLayout()
    order = list(range(len(lines)))
    polls = [i for i in range(start, end) if lines[i][1] == "global_load_b32"
             and int(lines[i][3].get("offset", "0"), 0) == FUSED_A2A_OUTBOUND_OFFSET]
    assert len(polls) == 1, f"expected one outbound poll, found {len(polls)}"
    cmp = next(i for i in range(polls[0], end) if lines[i][1] == "v_cmp_ne_u32")
    assert lines[cmp][2][1] == lines[polls[0]][2][0], "outbound compare does not read the polled value"
    subAt = _last_def(lines, order, cmp, lines[cmp][2][2])
    sub = lines[subAt]
    assert sub[1] == "s_sub_u32", f"DRAIN_SEND target is {sub[1]}, not W - count"
    w = lines[_last_def(lines, order, subAt, sub[2][1])]
    assert _is_kernarg_load(w, "s_load_b32", fusedBase + layout["FusedW"]), "minuend is not FusedW"
    bfeAt = _last_def(lines, order, subAt, sub[2][2])
    bfe = lines[bfeAt]
    assert bfe[1] == "s_bfe_u32" and int(bfe[2][2], 0) == _COUNT_BFE, "subtrahend is not FusedDrain[23:16]"
    drain = lines[_last_def(lines, order, bfeAt, bfe[2][1])]
    assert _is_kernarg_load(drain, "s_load_b32", fusedBase + layout["FusedDrain"]), (
        "direct-peer count is not from FusedDrain")


def _check_kernel(src):
    layout = fusedA2AKernArgLayout()
    lines = _lines(src)
    labels = {ln[0]: i for i, ln in enumerate(lines) if ln[0] is not None}
    order = list(range(len(lines)))
    fusedBase = min(int(ln[3]["offset"], 0) for ln in lines
                    if ln[1] == "s_load_b128" and ln[2][1:3] == [_KERNARG, _PEER_GROUP]) - OFF_flagPtr
    drainOff = fusedBase + layout["FusedDrain"]
    hs = []
    for i, ln in enumerate(lines):
        if ln[1] == "s_bfe_u32" and ln[2][0] == ln[2][1] and int(ln[2][2], 0) == _MASK_BFE:
            load = _last_def(lines, order, i, ln[2][1])
            if _is_kernarg_load(lines[load], "s_load_b32", drainOff):
                hs.append((load, i))
    c3 = [i for i, ln in enumerate(lines) if _atomic_offset(ln) == FUSED_A2A_COUNTER3_OFFSET]
    assert hs and len(hs) == len(c3), f"{len(hs)} direct-peer mask loads for {len(c3)} handshakes"
    bounds = [h[0] for h in hs[1:]] + [len(lines)]
    for n, ((load, bfe), c3i, nxt) in enumerate(zip(hs, c3, bounds)):
        try:
            assert load < c3i < nxt, "counter3 tally outside its handshake"
            _check_handshake(lines, labels, fusedBase, load, bfe, c3i)
            _check_drain_send(lines, fusedBase, c3i, nxt)
        except AssertionError as e:
            raise AssertionError(f"handshake {n}: {e}") from e
    return len(hs)


@pytest.fixture(scope="module", params=sorted(_CONFIGS))
def results(request):
    config, clusterDim = _CONFIGS[request.param]
    out = emit_kernels_from_config(config, limit=8, arch=_ARCH, cluster_dim=clusterDim)
    assert out and all(err == 0 for (_b, _s, err) in out), [(b, e) for (b, _s, e) in out]
    return out


def test_fused_a2a_gfx1250_direct_peer_handshake(results):
    for base, src, _err in results:
        assert _check_kernel(src) > 0, base


def _drop_gate(src, target):
    """src without the s_bitcmp1_b32 + s_cbranch_scc1 pair that branches to target-labelled code."""
    lines = src.splitlines()
    out = []
    for i, ln in enumerate(lines):
        branch = rf"s_cbranch_scc1 label_fusedA2A_{target}(_\d+)?(\s|$)"
        if ln.startswith("s_bitcmp1_b32") and i + 1 < len(lines) and re.match(branch, lines[i + 1]):
            continue
        if re.match(branch, ln):
            continue
        out.append(ln)
    return "\n".join(out)


def _drop_count(src):
    """src with every DRAIN_SEND target W - count replaced by W."""
    return re.sub(r"^s_sub_u32 (s\d+), (s\d+), \1\b", r"s_mov_b32 \1, \2", src, flags=re.M)


def _clobber_mask(src):
    return re.sub(rf"^(s_bfe_u32 (s\d+), \2, {_MASK_BFE}\b.*)$", r"\1\ns_mov_b32 \2, 0", src, flags=re.M)


def _late_mask_load(src):
    """src with each mask load moved to just before its s_bfe_u32, past the kernarg wait."""
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(rf"s_bfe_u32 (s\d+), \1, {_MASK_BFE}\b", ln)
        if m:
            j = max(k for k in range(i) if lines[k].startswith(f"s_load_b32 {m[1]},"))
            lines.insert(i - 1, lines.pop(j))
    return "\n".join(lines)


def _device_scope_flag(src):
    return re.sub(r"^(global_atomic_inc_u32 [^/]*?\bs\[\d+:\d+\]) scope:SCOPE_SYS", r"\1 scope:SCOPE_DEV",
                  src, flags=re.M)


@pytest.mark.parametrize("mutate", [
    lambda s: _drop_gate(s, "direct_nocopy"),
    lambda s: _drop_gate(s, "direct_flag"),
    _drop_count,
    _device_scope_flag,
    _clobber_mask,
    _late_mask_load,
], ids=["no-copy-gate", "no-tail-gate", "send-target-W", "flag-device-scope", "clobber-mask",
        "late-mask-load"])
def test_fused_a2a_gfx1250_direct_peer_checker_rejects(results, mutate):
    _base, src, _err = results[0]
    mutated = mutate(src)
    assert mutated.splitlines() != src.splitlines()
    with pytest.raises(AssertionError):
        _check_kernel(mutated)
