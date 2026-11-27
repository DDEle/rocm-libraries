# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Executes the store-pass gate and epilogue handshake of an emitted gfx1250 FusedA2A=GemmRS kernel.

Every work-group of every rank runs, as wave 0 of the work-group, from the store-pass gate through its
handshake against one shared memory, after the emitted FusedRSWgRemap has given it its tile.  An SDMA
model executes the packets the work-groups put on their queues.  Not a test module.
"""

import functools
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import pytest

from config_harness import emit_kernels_from_config
from test_fused_rs_remap_gfx1250_char import _remap_body, _rs_config as rs_config, _run_remap
from tensilelite.Components.SdmaPacketEmitter import (
    ATOMIC_PACKET_DWORDS, COPY_DW_DST, COPY_DW_RECT, COPY_DW_SRC, COPY_PACKET_DWORDS,
    SDMA_OP_ATOMIC, SDMA_OP_COPY_SUBWIN, SIGNAL_DW_ADDR)
from tensilelite.Components.SdmaRingEmitter import SDMA_QUEUE_SIZE
from tensilelite.Components.Signature import (
    FUSED_A2A_COUNTER1_OFFSET, FUSED_A2A_COUNTER2_OFFSET, FUSED_A2A_COUNTER3_OFFSET,
    FUSED_A2A_OUTBOUND_OFFSET, fusedA2AKernArgLayout)

ARCH = "gfx1250"

GATE = "FusedRSDst < 0? (own segment)"
HANDSHAKE_START = "fused-RS: full exec before wave-0 election"
HANDSHAKE_END = "fused-RS: restore full exec after the handshake"
POLL = "fused-RS: poll flag[lane]"
SEND_POLL = "fused-RS: poll the outbound counter"
DRAIN_START = "fused-RS: W == 1?"

M32 = 0xFFFFFFFF
LANES = 32
PACKET_BYTES = 16
# (name, width) of the SGPRs the handshake reads on entry.
HANDSHAKE_SGPRS = (("WorkGroup0", 1), ("WorkGroup1", 1), ("FusedRSDst", 1), ("FusedTokenTiles", 1),
                   ("FusedNShard", 1), ("FusedTotalWGs", 1), ("KernArgAddress", 2), ("FusedCounterPtr", 2),
                   ("AddressD", 2), ("SizesFree", 2), ("StrideD1J", 1), ("FusedPeerGroupPtr", 1))
_SKIPPED = ("s_wait", "s_set_vgpr_msb", "s_nop", "s_sleep", "s_prefetch_inst", "s_barrier", "s_delay_alu")
_SPIN_LIMIT = 10000


@functools.lru_cache(maxsize=None)
def emit(cluster_dim=None):
    """[(base, src, err)] of the GemmRS kernels; cluster_dim is a tuple or None."""
    with tempfile.TemporaryDirectory() as tmp:
        return emit_kernels_from_config(rs_config(tmp, cluster_dim), limit=8, arch=ARCH,
                                        cluster_dim=list(cluster_dim) if cluster_dim else None)


def _split(line):
    code, _, comment = line.partition("//")
    return code.strip(), comment.strip()


class Kernel:
    """Parsed .s of one GemmRS kernel."""

    def __init__(self, src, cluster_dim):
        self.src = src
        self.cs, self.ck = cluster_dim or (1, 1)
        self.mt0, self.mt1 = (int(x) for x in re.search(r"_MT(\d+)x(\d+)x", src).groups())
        self.lines = [_split(ln) for ln in src.splitlines()]
        self.sets = {}
        for code, _c in self.lines:
            m = re.fullmatch(r"\.set (\w+), (.+)", code)
            if m:
                self.sets[m.group(1)] = m.group(2).strip()
        self.labels = {}
        for i, (code, _c) in enumerate(self.lines):
            m = re.fullmatch(r"([\w.]+):", code)
            if m:
                self.labels[m.group(1)] = i
        self.gates = [i for i, (code, c) in enumerate(self.lines) if code and c == GATE]
        assert self.gates, "no GemmRS store-pass gate"
        self.remap = _remap_body(src)
        layout = fusedA2AKernArgLayout()
        rank_w = next(code for code, c in self.lines if c == "FusedMyRank, FusedW")
        self.fused_base = int(rank_w.split(",")[-1], 0) - layout["FusedMyRank"]
        self._decoded = {}

    def eval(self, expr):
        total = 0
        for sign, term in re.findall(r"([+-]?)\s*([\w]+)", expr):
            value = int(term) if term.isdigit() else self.eval(self.sets[term])
            total += -value if sign == "-" else value
        return total

    def sreg(self, name):
        return self.eval(name)

    def passes(self, gate):
        """(PUSH lines, RECV lines) of the two store passes behind the gate at line `gate`."""
        recv_label = self.lines[gate + 1][0].split()[1]
        assert self.lines[gate + 1][0].startswith("s_cbranch_scc1") and "fusedRS_hoist_recv" in recv_label
        recv = self.labels[recv_label]
        branch = self.lines[recv - 1][0].split()
        assert branch[0] == "s_branch" and "fusedRS_hoist_after" in branch[1], self.lines[recv - 1]
        after = self.labels[branch[1]]
        return self.lines[gate + 2:recv], self.lines[recv + 1:after]

    def decode(self, i):
        if i not in self._decoded:
            self._decoded[i] = self._decode(self.lines[i][0])
        return self._decoded[i]

    def _decode(self, code):
        if not code or code.startswith(".") or code.endswith(":"):
            return None
        op, _, rest = code.partition(" ")
        if op.startswith(_SKIPPED):
            return None
        parts = [p.strip() for p in rest.split(",")] if rest else []
        mods = {}
        if parts:
            last = parts[-1].split()
            parts[-1] = last[0]
            for m in last[1:]:
                k, _, v = m.partition(":")
                mods[k] = v
        return op, [self._operand(p) for p in parts], mods

    def _operand(self, tok):
        if tok in ("exec_lo", "vcc_lo", "off", "null"):
            return (tok,)
        m = re.fullmatch(r"([sv])(\d+)", tok)
        if m:
            return (m.group(1), int(m.group(2)), 1)
        m = re.fullmatch(r"([sv])\[([^\]:]+)(?::([^\]]+))?\]", tok)
        if m:
            # A VGPR above 255 is written as its number minus the s_set_vgpr_msb offset.
            unmsb = lambda e: re.sub(r"-(256|512|768)$", "", e) if m.group(1) == "v" else e
            lo = self.eval(unmsb(m.group(2)))
            hi = self.eval(unmsb(m.group(3))) if m.group(3) else lo
            return (m.group(1), lo, hi - lo + 1)
        if re.fullmatch(r"label_\S+", tok):
            return ("label", tok)
        return ("imm", int(tok, 0) & M32)


class Hang(Exception):
    pass


@dataclass
class Launch:
    copies: list = field(default_factory=list)          # (rank, queue, fields)
    queues: dict = field(default_factory=lambda: defaultdict(list))   # (rank, queue) -> [("COPY"|"ATOMIC", ...)]
    incs: list = field(default_factory=list)            # (rank, addr, limit)
    polls: list = field(default_factory=list)           # (rank, comment, exec)
    drains: list = field(default_factory=list)          # rank
    passes: list = field(default_factory=list)          # (rank, wg, "PUSH"|"RECV")


class World:
    """The memory of W ranks and their SDMA queues."""

    def __init__(self, kernel, world, r_units, seg_bands, drain):
        self.k, self.w, self.drain = kernel, world, drain
        self.n0, self.n1 = r_units * kernel.cs, world * seg_bands * kernel.ck
        self.m, self.n = self.n0 * kernel.mt0, self.n1 * kernel.mt1
        self.seg_bands, self.seg_n = seg_bands, self.n // world
        self.mem = {}
        self.read = {}
        for r in range(world):
            ka = self.kernarg(r) + kernel.fused_base
            layout = fusedA2AKernArgLayout()
            for j in range(world):
                for name, addr in (("flagPtr", self.flags(j)), ("bufPtr", self.recv(j)),
                                   ("queueBuf", self.queue(r, j)), ("rptr", self.rptr(r, j)),
                                   ("wptr", self.rptr(r, j) + 8), ("doorbell", self.rptr(r, j) + 16)):
                    self.store64(ka + layout["peer_%d_%s" % (j, name)], addr)
                self.read[(r, j)] = 0
            self.store64(ka + layout["counter_ptr"], self.counters(r))
            for name, value in (("FusedMyRank", r), ("FusedW", world), ("FusedDrain", drain),
                                ("FusedAM", self.m), ("FusedNumCu", 256)):
                self.mem[ka + layout[name]] = value

    kernarg = staticmethod(lambda r: 0x10_0000_0000 + r * 0x1_0000_0000)
    d = staticmethod(lambda r: 0x20_0000_0000 + r * 0x1_0000_0000)
    recv = staticmethod(lambda r: 0x30_0000_0000 + r * 0x1_0000_0000)
    flags = staticmethod(lambda r: 0x40_0000_0000 + r * 0x1_0000)
    counters = staticmethod(lambda r: 0x50_0000_0000 + r * 0x10_0000)
    queue = staticmethod(lambda r, j: 0x60_0000_0000 + (r * 8 + j) * 0x100_0000)
    rptr = staticmethod(lambda r, j: 0x70_0000_0000 + (r * 8 + j) * 0x100)

    def load64(self, a):
        return self.mem.get(a, 0) | self.mem.get(a + 4, 0) << 32

    def store64(self, a, v):
        self.mem[a], self.mem[a + 4] = v & M32, v >> 32

    def pump(self, launch):
        """The SDMA engines execute every packet up to their queue's wptr."""
        for (r, j), done in list(self.read.items()):
            wptr = self.load64(self.rptr(r, j) + 8)
            while done < wptr:
                at = self.queue(r, j) + done % SDMA_QUEUE_SIZE
                dw = [self.mem.get(at + 4 * i, 0) for i in range(COPY_PACKET_DWORDS)]
                op = dw[0] & 0xFF
                if op == 0:
                    done += 4
                elif op == SDMA_OP_COPY_SUBWIN:
                    fields = dict(
                        src=dw[COPY_DW_SRC] | dw[COPY_DW_SRC + 1] << 32,
                        dst=dw[COPY_DW_DST] | dw[COPY_DW_DST + 1] << 32,
                        src_pitch=(dw[4] >> 16) + 1, src_slice=dw[5] + 1,
                        dst_pitch=(dw[9] >> 16) + 1, dst_slice=dw[10] + 1,
                        rect_x=(dw[COPY_DW_RECT] & 0xFFFF) + 1, rect_y=(dw[COPY_DW_RECT] >> 16) + 1)
                    launch.copies.append((r, j, fields))
                    launch.queues[(r, j)].append(("COPY", fields))
                    done += 4 * COPY_PACKET_DWORDS
                elif op == SDMA_OP_ATOMIC:
                    addr = dw[SIGNAL_DW_ADDR] | dw[SIGNAL_DW_ADDR + 1] << 32
                    self.mem[addr] = (self.mem.get(addr, 0) + dw[3]) & M32
                    launch.queues[(r, j)].append(("ATOMIC", addr))
                    done += 4 * ATOMIC_PACKET_DWORDS
                else:
                    pytest.fail(f"queue {(r, j)}: unknown SDMA packet {dw[0]:#x}")
            self.read[(r, j)] = done
            self.store64(self.rptr(r, j), done)

    def work_groups(self, rank):
        """[(u, wg0, wg1, registers after the remap)] in dispatch order."""
        k = self.k
        out = []
        for wg1 in range(self.n1):
            for wg0 in range(self.n0):
                reg, _ = _run_remap(k.remap, self.n0, self.n1, rank, self.w, wg0, wg1)
                u = wg0 // k.cs + (self.n0 // k.cs) * (wg1 // k.ck)
                out.append((u, wg0, wg1, reg))
        return sorted(out, key=lambda t: t[:3])

    def run_wg(self, launch, rank, reg, gate):
        """Generator: wave 0 of one work-group from the gate to the end of its handshake; yields on
        every taken backward branch."""
        k = self.k
        s = {}
        for name in ("WorkGroup0", "WorkGroup1", "FusedRSDst", "FusedTokenTiles", "FusedNShard",
                     "FusedTotalWGs"):
            s[k.sreg("sgpr" + name)] = reg["s[sgpr%s]" % name]
        for name, value in (("KernArgAddress", self.kernarg(rank)), ("FusedCounterPtr", self.counters(rank)),
                            ("AddressD", self.d(rank))):
            s[k.sreg("sgpr" + name)], s[k.sreg("sgpr" + name) + 1] = value & M32, value >> 32
        s[k.sreg("sgprSizesFree+0")] = self.m
        s[k.sreg("sgprSizesFree+1")] = self.n
        s[k.sreg("sgprStrideD1J")] = self.m
        s[k.sreg("sgprFusedPeerGroupPtr")] = 0
        v = {k.sreg("vgprSerial"): list(range(LANES))}
        st = dict(s=s, v=v, scc=None, exec=M32, vcc=0)

        pc = gate
        recv_label = k.lines[gate + 1][0].split()[1]
        steps = 0
        started = False
        while True:
            code, comment = k.lines[pc]
            if not started and comment == HANDSHAKE_START:
                started = True
                launch.passes.append((rank, (reg["s[sgprWorkGroup0]"], reg["s[sgprWorkGroup1]"]),
                                      "RECV" if pc > k.labels[recv_label] else "PUSH"))
            if not started and pc > gate + 1:
                pc += 1
                continue
            steps += 1
            if steps > _SPIN_LIMIT * 50:
                raise Hang(f"rank {rank}: {steps} instructions")
            if comment in (POLL, SEND_POLL):
                launch.polls.append((rank, comment, st["exec"]))
            if comment == DRAIN_START:
                launch.drains.append(rank)
            nxt = self._step(launch, rank, st, pc)
            if comment == HANDSHAKE_END:
                return
            if nxt <= pc:
                yield rank, pc, k.lines[pc][0]
            pc = nxt

    def _step(self, launch, rank, st, pc):
        k = self.k
        ins = k.decode(pc)
        if ins is None:
            return pc + 1
        op, a, mods = ins
        s, v = st["s"], st["v"]

        def rd(o, lane=0, i=0):
            kind = o[0]
            if kind == "imm":
                return o[1]
            if kind == "s":
                assert o[1] + i in s, f"line {pc} {k.lines[pc][0]!r} reads unset s{o[1] + i}"
                return s[o[1] + i]
            if kind == "v":
                val = v.get(o[1] + i, [None] * LANES)[lane]
                assert val is not None, f"line {pc} {k.lines[pc][0]!r} reads unset v{o[1] + i} lane {lane}"
                return val
            if kind == "exec_lo":
                return st["exec"]
            if kind == "vcc_lo":
                return st["vcc"]
            pytest.fail(f"line {pc}: cannot read {o}")

        def rd64(o):
            return rd(o) | rd(o, i=1) << 32

        def ws(o, val, i=0):
            if o[0] == "exec_lo":
                st["exec"] = val & M32
            else:
                assert o[0] == "s", (pc, o)
                s[o[1] + i] = val & M32

        def ws64(o, val):
            ws(o, val), ws(o, val >> 32, 1)

        def wv(o, lane, val, i=0):
            assert o[1] + i != k.sreg("vgprSerial"), f"line {pc} writes Serial"
            v.setdefault(o[1] + i, [None] * LANES)[lane] = val & M32

        lanes = [i for i in range(LANES) if st["exec"] >> i & 1]
        label = lambda o: k.labels[o[1]]

        if op == "s_branch":
            return label(a[0])
        if op in ("s_cbranch_scc0", "s_cbranch_scc1"):
            assert st["scc"] is not None, f"line {pc} reads SCC before any write"
            return label(a[0]) if st["scc"] == (op[-1] == "1") else pc + 1
        if op in ("s_cbranch_vccnz", "s_cbranch_vccz"):
            return label(a[0]) if (st["vcc"] != 0) == (op == "s_cbranch_vccnz") else pc + 1
        if op == "s_mov_b32":
            ws(a[0], rd(a[1]))
        elif op == "s_mov_b64":
            ws64(a[0], rd64(a[1]))
        elif op in ("s_add_u32", "s_addc_u32"):
            r = rd(a[1]) + rd(a[2]) + (st["scc"] if op == "s_addc_u32" else 0)
            ws(a[0], r)
            st["scc"] = int(r > M32)
        elif op in ("s_sub_u32", "s_subb_u32"):
            x, b = rd(a[1]), rd(a[2]) + (st["scc"] if op == "s_subb_u32" else 0)
            ws(a[0], x - b)
            st["scc"] = int(b > x)
        elif op == "s_mul_i32":
            ws(a[0], rd(a[1]) * rd(a[2]))
        elif op == "s_mul_hi_u32":
            ws(a[0], rd(a[1]) * rd(a[2]) >> 32)
        elif op in ("s_lshl_b32", "s_lshr_b32", "s_and_b32", "s_andn2_b32", "s_or_b32"):
            x, y = rd(a[1]), rd(a[2])
            r = {"s_lshl_b32": lambda: x << (y & 31), "s_lshr_b32": lambda: x >> (y & 31),
                 "s_and_b32": lambda: x & y, "s_andn2_b32": lambda: x & ~y & M32,
                 "s_or_b32": lambda: x | y}[op]() & M32
            ws(a[0], r)
            st["scc"] = int(r != 0)
        elif op == "s_lshl_b64":
            r = rd64(a[1]) << (rd(a[2]) & 63) & (2 ** 64 - 1)
            ws64(a[0], r)
            st["scc"] = int(r != 0)
        elif op == "s_add_u64":
            ws64(a[0], rd64(a[1]) + rd64(a[2]) & (2 ** 64 - 1))
        elif op == "s_bfm_b32":
            ws(a[0], ((1 << (rd(a[1]) & 31)) - 1) << (rd(a[2]) & 31))
        elif op in ("s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_lt_u32", "s_cmp_le_u32", "s_cmp_ge_u32",
                    "s_cmp_gt_u32"):
            x, y = rd(a[0]), rd(a[1])
            st["scc"] = int({"eq": x == y, "lg": x != y, "lt": x < y, "le": x <= y, "ge": x >= y,
                             "gt": x > y}[op[6:8]])
        elif op == "s_cmp_lt_i32":
            sx = lambda x: x - (1 << 32) if x >> 31 else x
            st["scc"] = int(sx(rd(a[0])) < sx(rd(a[1])))
        elif op == "s_cmp_eq_u64":
            st["scc"] = int(rd64(a[0]) == rd64(a[1]))
        elif op in ("s_load_b32", "s_load_b64", "s_load_b128"):
            addr = rd64(a[1]) + rd(a[2]) + int(mods.get("offset", "0"), 0)
            for i in range(a[0][2]):
                ws(a[0], self.mem.get(addr + 4 * i, 0), i)
        elif op == "v_readfirstlane_b32":
            ws(a[0], rd(a[1], lanes[0] if lanes else 0))
        elif op == "v_cmp_ne_u32":
            st["vcc"] = sum(1 << i for i in lanes if rd(a[1], i) != rd(a[2], i))
        elif op in ("v_mov_b32", "v_add_nc_u32", "v_lshlrev_b32", "v_lshl_or_b32", "v_add_co_u32",
                    "v_add_co_ci_u32"):
            vcc = 0
            for i in lanes:
                if op == "v_mov_b32":
                    r = rd(a[1], i)
                elif op == "v_add_nc_u32":
                    r = rd(a[1], i) + rd(a[2], i)
                elif op == "v_lshlrev_b32":
                    r = rd(a[2], i) << (rd(a[1], i) & 31)
                elif op == "v_lshl_or_b32":
                    r = (rd(a[1], i) << (rd(a[2], i) & 31)) | rd(a[3], i)
                else:
                    r = rd(a[2], i) + rd(a[3], i) + (st["vcc"] >> i & 1 if op == "v_add_co_ci_u32" else 0)
                    vcc |= int(r > M32) << i
                wv(a[0], i, r)
            if op in ("v_add_co_u32", "v_add_co_ci_u32"):
                st["vcc"] = vcc
        elif op.startswith("global_"):
            self._global(launch, rank, st, op, a, mods, lanes, rd, wv)
        else:
            pytest.fail(f"line {pc}: instruction not modelled: {k.lines[pc][0]!r}")
        return pc + 1

    def _global(self, launch, rank, st, op, a, mods, lanes, rd, wv):
        store = op.startswith("global_store")
        vaddr, saddr = (a[0], a[2]) if store else (a[1], a[3] if op.startswith("global_atomic") else a[2])
        off = int(mods.get("offset", "0"), 0)
        for i in lanes:
            if saddr[0] == "off":
                addr = rd(vaddr, i) | rd(vaddr, i, 1) << 32
            else:
                addr = (rd(saddr) | rd(saddr, 0, 1) << 32) + rd(vaddr, i)
            addr += off
            if store:
                for d in range(a[1][2]):
                    self.mem[addr + 4 * d] = rd(a[1], i, d)
            elif op == "global_load_b32":
                wv(a[0], i, self.mem.get(addr, 0))
            elif op == "global_atomic_inc_u32":
                old, limit = self.mem.get(addr, 0), rd(a[2], i)
                self.mem[addr] = 0 if old >= limit else old + 1
                launch.incs.append((rank, addr, limit))
                wv(a[0], i, old)
            elif op in ("global_atomic_max_u64", "global_atomic_cmpswap_b64"):
                old = self.load64(addr)
                data = rd(a[2], i) | rd(a[2], i, 1) << 32
                if op == "global_atomic_max_u64":
                    self.store64(addr, max(old, data))
                elif old == (rd(a[2], i, 2) | rd(a[2], i, 3) << 32):
                    self.store64(addr, data)
                wv(a[0], i, old & M32)
                wv(a[0], i, old >> 32, 1)
            else:
                pytest.fail(f"global instruction not modelled: {op}")

    def launch(self, last="recv", gate=None):
        """One launch on every rank; `last` picks the final work-group of each rank: the last RECV
        unit, or the last PUSH unit."""
        k = self.k
        gate = k.gates[0] if gate is None else gate
        launch = Launch()
        finals = []
        for rank in range(self.w):
            wgs = self.work_groups(rank)
            if last == "push":
                push = [t for t in wgs if t[3]["s[sgprFusedRSDst]"] != M32]
                wgs.remove(push[-1])
                wgs.append(push[-1])
            for _u, _w0, _w1, reg in wgs[:-1]:
                self._finish([self.run_wg(launch, rank, reg, gate)], launch)
            finals.append(self.run_wg(launch, rank, wgs[-1][3], gate))
        self._finish(finals, launch)
        self.pump(launch)
        return launch

    def _finish(self, gens, launch):
        live, rounds, at = list(gens), 0, {}
        while live:
            rounds += 1
            if rounds > _SPIN_LIMIT:
                raise Hang(f"work-groups still spinning at {list(at.values())}")
            for g in list(live):
                try:
                    at[id(g)] = next(g)
                except StopIteration:
                    live.remove(g)
                    at.pop(id(g), None)
            self.pump(launch)

    def counter_addrs(self, rank):
        base = self.counters(rank)
        return (base + FUSED_A2A_COUNTER1_OFFSET, base + FUSED_A2A_COUNTER2_OFFSET,
                base + FUSED_A2A_COUNTER3_OFFSET)

    def outbound(self, rank):
        return self.flags(rank) + FUSED_A2A_OUTBOUND_OFFSET
