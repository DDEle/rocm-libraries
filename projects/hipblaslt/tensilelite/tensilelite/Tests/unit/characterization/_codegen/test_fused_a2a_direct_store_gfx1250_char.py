# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) direct stores into the peer receive buffer: structure.

Per PUSH replay of every emitted kernel. The GSU=1 NonEdge, full-width Then and OptNLL replays
end every store batch with s_cmp_lg_u32 DwStep, 0 / s_cbranch_scc1 into a cold block that
repeats the batch's D stores in order (same data registers and width) as
scope:SCOPE_SYS th:TH_STORE_RT stores into the SRD loaded by the replay's prologue, then
branches back. MB, VW1 and LOCAL replays carry none of it. Matched on instructions and
operands only.
"""

import collections
import os
import re

import pytest

from config_harness import emit_kernels_from_config
from tensilelite.Components.GlobalWriteBatch import GlobalWriteBatchWriter
from tensilelite.Components.Signature import fusedA2AKernArgLayout

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_CONFIGS = {
    "plain": (os.path.join(_COMM, "fused_a2a.yaml"), None),
    "cluster": (os.path.join(_COMM, "fused_a2a_cluster.yaml"), [2, 4]),
}

_SRD_D = "s[sgprSrdD:sgprSrdD+3]"
_HINTS = ("s_wait_alu", "s_set_vgpr_msb", "s_nop", "s_delay_alu", "s_prefetch", "s_wait_xcnt")
_NO_DST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_wait", "s_nop", "s_set_vgpr_msb",
           "s_barrier", "s_delay_alu", "s_prefetch", "global_store", "buffer_store")
_SCC_READ = ("s_cbranch_scc", "s_cselect", "s_addc", "s_subb", "s_cmov")
_SCC_WRITE = ("s_cmp", "s_bitcmp", "s_add_", "s_sub_", "s_and", "s_or", "s_xor", "s_nand", "s_nor",
              "s_xnor", "s_not", "s_lshl", "s_lshr", "s_ashr", "s_bfe", "s_min", "s_max", "s_abs",
              "s_bcnt")


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
    tok = re.sub(r"\.[hl]$", "", tok)
    m = re.fullmatch(r"v\[(?:vgprValuC\+)?(\d+)(?:-\d+)?(?::(\d+)(?:-\d+)?)?\]", tok)
    if m:
        return {f"v{i}" for i in range(int(m[1]), int(m[2] or m[1]) + 1)}
    m = re.fullmatch(r"([sv])\[(\d+):(\d+)\]", tok)
    if m:
        return {f"{m[1]}{i}" for i in range(int(m[2]), int(m[3]) + 1)}
    m = re.fullmatch(r"s\[(\w+?)(?:\+(\d+))?(?::\1\+(\d+))?\]", tok)
    if m:
        lo = int(m[2] or 0)
        return {f"{m[1]}+{i}" for i in range(lo, int(m[3] or lo) + 1)}
    if tok.startswith("exec"):
        return {"exec"}
    return {tok}


def _writes(line):
    _l, op, ops, _m = line
    if op is None or op.startswith(_NO_DST) or not ops:
        return set()
    return _regs(ops[0])


def _prev_non_hint(lines, i):
    i -= 1
    while lines[i][1] is not None and lines[i][1].startswith(_HINTS):
        i -= 1
    return i


def _is_d_store(line):
    return line[1] is not None and line[1].startswith("buffer_store") and line[2][2] == _SRD_D \
        and line[3].get("scope") == "SCOPE_DEV"


def _is_sys_store(line):
    return line[1] is not None and line[1].startswith("buffer_store") and line[3].get("scope") == "SCOPE_SYS"


def _scc_written_first(lines, labels, i):
    """True if, walking from lines[i], an instruction writes SCC before one reads it."""
    for _ in range(4000):
        _l, op, ops, _m = lines[i]
        if op is not None and op.startswith(_SCC_READ):
            return False
        if op is not None and op.startswith(_SCC_WRITE):
            return True
        i = labels[ops[0]] if op == "s_branch" else i + 1
    raise AssertionError("no SCC access after the return label")


def _replays(lines, labels):
    """(replay label, gate branch, PUSH end, LOCAL label, LOCAL end) per hoisted PUSH/LOCAL pair."""
    out = []
    for g, (_l, op, ops, _m) in enumerate(lines):
        if op != "s_cbranch_scc1" or not ops[0].startswith("label_fusedA2A_hoist_local"):
            continue
        name = next(lines[i][0] for i in range(g, -1, -1)
                    if lines[i][0] is not None and lines[i][0].startswith("label_GW_"))
        end = next(i for i in range(g, len(lines)) if lines[i][1] == "s_branch"
                   and lines[i][2][0].startswith("label_fusedA2A_hoist_after"))
        out.append((name, g, end, labels[ops[0]], labels[lines[end][2][0]]))
    return out


def _check_direct(lines, labels, name, gate, end, local, numBatches):
    drainOff = 0x78 + fusedA2AKernArgLayout()["FusedDrain"]
    load = next(i for i in range(gate, -1, -1) if lines[i][1] == "s_load_b64")
    assert int(lines[load][2][2], 0) == drainOff, "gate does not load FusedDrain, FusedAM as one pair"
    drainLo = sorted(_regs(lines[load][2][0]), key=lambda r: int(r[1:]))[0]
    proEnd = next(i for i in range(gate, end) if lines[i][0] is not None and lines[i][0].endswith("_pro_end"))
    base = lines[proEnd][0][len("label_"):-len("_pro_end")]
    pro = [i for i in range(gate + 1, proEnd) if lines[i][1] is not None]
    first = lines[next(i for i in pro if not lines[i][1].startswith(_HINTS))]
    assert first[1] == "s_mov_b32" and first[2][1] == "0", "prologue does not start with DwStep = 0"
    step = first[2][0]
    assert any(lines[i][1] == "s_and_b32" and lines[i][2][1:] == [drainLo, "0xff00"] for i in pro), (
        "prologue does not test FusedDrain[15:8]")
    srdLoad = [i for i in pro if lines[i][1] == "s_load_b64"]
    assert len(srdLoad) == 1, "prologue does not load one peer_bufPtr"
    srdLo = int(re.fullmatch(r"s\[(\d+):\d+\]", lines[srdLoad[0]][2][0])[1])
    srd = f"s[{srdLo}:{srdLo + 3}]"
    lane = lines[next(i for i in pro if lines[i][1] == "v_sub_nc_u32")][2][0]
    for i in pro:
        assert not (_writes(lines[i]) & {"exec"}) and not lines[i][1].startswith(("s_wait_storecnt", "s_barrier")), (
            f"prologue has {lines[i][1]}")
        assert not {r for r in _writes(lines[i]) if re.fullmatch(r"v\d+", r)} - _regs(lane), (
            f"prologue writes {lines[i][2][0]}")

    branches = [i for i in range(proEnd, end) if lines[i][1] == "s_cbranch_scc1" and "_blk_" in lines[i][2][0]]
    assert len(branches) == numBatches, f"{len(branches)} direct-store branches for {numBatches} batches"
    exec1 = next(i for i in range(proEnd, end) if lines[i][1] == "s_mov_b32" and lines[i][2] == ["exec_lo", "-1"])
    storecnt = next(i for i in range(exec1, end) if lines[i][1] == "s_wait_storecnt" and lines[i][2] == ["0"])
    assert branches[-1] < exec1 < storecnt, "last direct-store branch is not before the handshake"
    assert not any(_is_d_store(lines[i]) for i in range(branches[-1], end)), "D store after the last batch tail"
    assert not any(_writes(lines[i]) & (_regs(srd) | _regs(step) | _regs(lane)) for i in range(proEnd, branches[-1])), (
        "SrdRecv, DwStep or DwLane redefined in the store batches")

    start = proEnd
    for k, b in enumerate(branches):
        blk = lines[b][2][0]
        assert blk == f"label_{base}_blk_{k}", f"batch {k} branches to {blk}"
        cmp = lines[_prev_non_hint(lines, b)]
        assert cmp[1] == "s_cmp_lg_u32" and cmp[2] == [step, "0"], f"batch {k} tail does not test DwStep"
        ret = f"label_{base}_ret_{k}"
        assert lines[b + 1][0] == ret, f"batch {k} branch is not followed by its return label"
        assert _scc_written_first(lines, labels, b + 1), f"SCC read before written after {ret}"

        assert blk in labels, f"no cold block {blk}"
        c0 = labels[blk]
        assert end < c0 < local, f"cold block {blk} outside the PUSH replay's cold region"
        c1 = next(i for i in range(c0 + 1, len(lines)) if lines[i][0] is not None)
        tail = lines[_prev_non_hint(lines, c1)]
        assert tail[1] == "s_branch" and tail[2] == [ret], f"cold block {blk} does not return to {ret}"
        cold = range(c0 + 1, c1)

        dStores = [i for i in range(start, b) if _is_d_store(lines[i])]
        rStores = [i for i in cold if _is_sys_store(lines[i])]
        assert dStores and len(dStores) == len(rStores), f"batch {k}: {len(dStores)} D stores, {len(rStores)} recv"
        data = set()
        for d, r in zip(dStores, rStores):
            dl, rl = lines[d], lines[r]
            assert rl[1] == dl[1] and _regs(rl[2][0]) == _regs(dl[2][0]), (
                f"batch {k}: recv {rl[1]} {rl[2][0]} does not repeat D {dl[1]} {dl[2][0]}")
            assert rl[2][2] == srd and rl[3].get("th") == "TH_STORE_RT", f"batch {k}: recv store not into SrdRecv"
            if name.endswith("_NonEdge"):
                assert rl[3].get("offset") == dl[3].get("offset"), f"batch {k}: recv offset differs from D"
            regs = _regs(dl[2][0])
            assert not any(_writes(lines[i]) & regs for i in range(d + 1, b)), (
                f"batch {k}: data of D store {dl[2][0]} rewritten before the batch tail")
            data |= regs
        for i in cold:
            assert not (_writes(lines[i]) & (data | {"exec"} | _regs(srd) | _regs(step) | _regs(lane))), (
                f"cold block {blk} writes {lines[i][2][0]}")
            assert lines[i][1] is None or not lines[i][1].startswith(("s_wait_storecnt", "s_barrier")), (
                f"cold block {blk} has {lines[i][1]}")
            assert lines[i][0] is None, f"label inside cold block {blk}"
        start = b
    return base


def _check_kernel(src, batches):
    lines = _lines(src)
    labels = {ln[0]: i for i, ln in enumerate(lines) if ln[0] is not None}
    replays = _replays(lines, labels)
    vws = [int(re.search(r"_VW(\d+)_", r[0])[1]) for r in replays]
    seen = []
    for (name, gate, end, local, localEnd), vw in zip(replays, vws):
        eligible = vw == max(vws) and ("_GSU1_" in name or "_OptNLL_" in name)
        direct = any(lines[i][0] is not None and lines[i][0].endswith("_pro_end") for i in range(gate, end))
        assert direct == eligible, f"{name}: direct stores {'present' if direct else 'missing'}"
        if direct:
            proEnd = next(lines[i][0] for i in range(gate, end)
                          if lines[i][0] is not None and lines[i][0].endswith("_pro_end"))
            base = proEnd[len("label_"):-len("_pro_end")]
            seen.append(_check_direct(lines, labels, name, gate, end, local, batches.get(base, 0)))
        quiet = [range(local, localEnd)] + ([] if direct else [range(gate, end)])
        for span in quiet:
            assert not any(_is_sys_store(lines[i]) or (lines[i][1] == "s_cbranch_scc1" and "_blk_" in lines[i][2][0])
                           for i in span), f"{name}: direct stores in a non-direct replay"
    assert seen and sorted(seen) == sorted(batches), (sorted(seen), sorted(batches))
    blocks = sum(1 for ln in lines if ln[0] is not None and re.search(r"_dw(_\d+)?_blk_\d+$", ln[0]))
    assert blocks == sum(batches.values()), f"{blocks} cold blocks for {sum(batches.values())} batches"
    return len(seen)


@pytest.fixture(scope="module", params=sorted(_CONFIGS))
def results(request):
    config, clusterDim = _CONFIGS[request.param]
    counts = collections.Counter()
    emit = GlobalWriteBatchWriter.emit

    def countingEmit(self):
        direct = self.parentWriter.states.fusedA2ADirect
        if direct is not None:
            counts[(self.kernel["BaseName"], direct.labelBase)] += 1
        return emit(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GlobalWriteBatchWriter, "emit", countingEmit)
        out = emit_kernels_from_config(config, limit=8, arch=_ARCH, cluster_dim=clusterDim)
    assert out and all(err == 0 for (_b, _s, err) in out), [(b, e) for (b, _s, e) in out]
    return [(b, s, {lb: n for (bb, lb), n in counts.items() if bb == b}) for (b, s, _e) in out]


def test_fused_a2a_gfx1250_direct_store_structure(results):
    for base, src, batches in results:
        assert _check_kernel(src, batches) > 0, base


def _first_cold_store(src):
    lines = src.splitlines()
    blk = next(i for i, ln in enumerate(lines) if re.match(r"label_fusedA2A_dw\w*_blk_\d+:", ln))
    return lines, next(i for i in range(blk, len(lines)) if "scope:SCOPE_SYS" in lines[i]
                       and lines[i].startswith("buffer_store"))


def _drop_recv_store(src):
    lines, i = _first_cold_store(src)
    return "\n".join(lines[:i] + lines[i + 1:])


def _recv_device_scope(src):
    lines, i = _first_cold_store(src)
    lines[i] = lines[i].replace("scope:SCOPE_SYS", "scope:SCOPE_DEV")
    return "\n".join(lines)


def _recv_other_data(src):
    lines, i = _first_cold_store(src)
    lines[i] = re.sub(r"^(buffer_store_b\d+ v\[)(\d+)((?:-\d+)?:)(\d+)",
                      lambda m: f"{m[1]}{int(m[2]) + 4}{m[3]}{int(m[4]) + 4}", lines[i])
    return "\n".join(lines)


def _exec_in_cold_block(src):
    lines, i = _first_cold_store(src)
    lines.insert(i, "s_mov_b32 exec_lo, -1")
    return "\n".join(lines)


def _scc_read_after_return(src):
    return re.sub(r"^(label_fusedA2A_dw\w*_ret_\d+:.*)$", r"\1\ns_cselect_b32 s6, 1, 0", src, count=1, flags=re.M)


def _clobber_data_before_tail(src):
    lines = src.splitlines()
    b = next(i for i, ln in enumerate(lines) if re.match(r"s_cbranch_scc1 label_fusedA2A_dw\w*_blk_\d+", ln))
    d = max(i for i in range(b) if lines[i].startswith("buffer_store") and "scope:SCOPE_DEV" in lines[i])
    reg = re.match(r"buffer_store_b\d+ (v\[[^:\]]+)", lines[d])[1] + "]"
    lines.insert(b - 1, f"v_mov_b32 {reg}, 0")
    return "\n".join(lines)


def _drop_cold_block(src):
    lines = src.splitlines()
    blk = next(i for i, ln in enumerate(lines) if re.match(r"label_fusedA2A_dw\w*_blk_\d+:", ln))
    end = next(i for i in range(blk, len(lines)) if lines[i].startswith("s_branch label_fusedA2A_dw"))
    return "\n".join(lines[:blk] + lines[end + 1:])


def _vgpr_write_in_prologue(src):
    return re.sub(r"^(label_fusedA2A_dw\w*_pro_end:)", r"v_mov_b32 v0, 0\n\1", src, count=1, flags=re.M)


def _drop_last_tail(src):
    """src without the s_cmp_lg_u32, branch and return label of the first replay's last batch."""
    lines = src.splitlines()
    tails = [(i, re.match(r"s_cbranch_scc1 label_(fusedA2A_dw\w*?)_blk_\d+", ln)) for i, ln in enumerate(lines)]
    tails = [(i, m[1]) for i, m in tails if m]
    b = max(i for i, base in tails if base == tails[0][1])
    return "\n".join(lines[:b - 1] + lines[b + 2:])


@pytest.mark.parametrize("mutate", [
    _drop_recv_store, _recv_device_scope, _recv_other_data, _exec_in_cold_block, _scc_read_after_return,
    _clobber_data_before_tail, _drop_last_tail, _drop_cold_block, _vgpr_write_in_prologue,
], ids=["drop-recv-store", "recv-device-scope", "recv-other-data", "exec-in-cold-block",
        "scc-read-after-return", "clobber-data", "drop-last-tail", "drop-cold-block", "vgpr-write-in-prologue"])
def test_fused_a2a_gfx1250_direct_store_checker_rejects(results, mutate):
    _base, src, batches = results[0]
    mutated = mutate(src)
    assert mutated.splitlines() != src.splitlines()
    with pytest.raises(AssertionError):
        _check_kernel(mutated, batches)
