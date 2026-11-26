# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) direct stores into the peer receive buffer: addresses.

Interprets the emitted integer instructions of every kernel for all lanes of one work-group at a
time: the D SRD set-up, then the epilogue from the store-coordinate set-up through the dispatch,
the store gate, the direct-store prologue, the store batches and their cold blocks, up to the
handshake's first EXEC write. Each receive store is paired with the D store of the same order and
checked against recv[dst][(src*N + t)*nShard + f] = D_src[(dst*nShard + f) + t*ldd], with (f, t)
taken from the D store's byte address. A value written by an instruction the interpreter does not
model is poisoned; a poisoned address, SRD, offset or branch condition fails the test.
"""

import os
import re
import struct

import numpy as np
import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_COMM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250"))
_CONFIGS = (os.path.join(_COMM, "fused_a2a.yaml"), os.path.join(_COMM, "fused_a2a_cluster.yaml"))

_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1
_BPE = 2
_ARG_SHIFT = 0x10
_ADDR_D = 0x0000_1234_5600_0000
_ADDR_C = 0x0000_2345_6700_0000
_BUF = tuple(((0x00AB_0000 + 0x100 * j) << 32) | 0xFFF0_0000 for j in range(8))
_ONE = 0x3F800000
_TWO = 0x40000000
_K = 4096

_NO_DST = ("s_wait", "s_nop", "s_prefetch", "s_delay_alu", "s_sleep", "s_barrier", "s_setprio", "s_clause",
           "s_set_vgpr_msb", "s_cmp", "s_bitcmp", "s_branch", "s_cbranch", "s_setpc", "s_endpgm", "tensor_",
           "v_nop")
_SCC_READ = ("s_cselect", "s_addc", "s_subb", "s_cmov", "s_cbranch_scc")


class _Bad(AssertionError):
    pass


class _Poison:
    __slots__ = ("why",)

    def __init__(self, why):
        self.why = why

    def __repr__(self):
        return f"<poison: {self.why}>"


class _Lbl:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class _Pc:
    __slots__ = ()


class _KernArg:
    __slots__ = ()


_PC, _KA = _Pc(), _KernArg()


def _eval(expr, sym):
    """(value, VGPR bank) of a register index or .set expression; the bank is a -256/-512/-768 term."""
    total = bank = 0
    for sign, term in re.findall(r"([+-]?)([^+-]+)", expr.replace(" ", "")):
        if re.fullmatch(r"\d+|0x[0-9a-fA-F]+", term):
            v = int(term, 0)
            if sign == "-" and v in (256, 512, 768):
                bank = v
        else:
            v = sym[term]
        total += -v if sign == "-" else v
    return total, bank


class _Ins:
    __slots__ = ("op", "rest", "sym", "text", "ops", "mods")

    def __init__(self, op, rest, sym, text):
        self.op, self.rest, self.sym, self.text = op, rest, sym, text
        self.ops = None

    def decode(self):
        toks = [t.strip() for t in self.rest.split(",")] if self.rest else []
        mods = {}
        if toks:
            last = toks[-1].split()
            toks[-1] = last[0] if last else ""
            for m in last[1:]:
                k, _, v = m.partition(":")
                mods[k] = v or True
        self.ops = [_operand(t, self.sym) for t in toks if t]
        self.mods = mods
        return self


_SPECIAL = {"exec_lo": ("exec",), "exec": ("exec",), "vcc_lo": ("vcc",), "vcc": ("vcc",), "null": ("null",),
            "m0": ("m0",), "scc": ("scc",)}


def _operand(tok, sym):
    if tok in _SPECIAL:
        return _SPECIAL[tok]
    m = re.fullmatch(r"([sv])(\d+)", tok)
    if m:
        return (m[1], int(m[2]), 1, 0)
    m = re.fullmatch(r"([sv])\[([^:\]]+)(?::([^\]]+))?\]", tok)
    if m:
        lo, bank = _eval(m[2], sym)
        hi, hbank = _eval(m[3], sym) if m[3] else (lo, bank)
        if m[1] == "s":
            return ("s", lo, hi - lo + 1, 0)
        return ("v", lo + bank, hi + hbank - lo - bank + 1, bank)
    if tok.startswith("label_"):
        return ("lbl", tok)
    if re.fullmatch(r"-?(\d+|0x[0-9a-fA-F]+)", tok):
        return ("imm", int(tok, 0))
    if re.fullmatch(r"-?\d+\.\d*", tok):
        return ("imm", struct.unpack("<I", struct.pack("<f", float(tok)))[0])
    if tok in sym:
        return ("imm", sym[tok])
    return ("raw", tok)


def _parse(src):
    entries, sym, snap, meta = [], {}, {}, False
    for ln in src.splitlines():
        s = re.sub(r"/\*.*?\*/", "", ln.split("//", 1)[0]).strip()
        if s.startswith(".amdgpu_metadata"):
            meta = True
        if meta:
            meta = not s.startswith(".end_amdgpu_metadata")
            continue
        if not s:
            continue
        if s.startswith(".set "):
            name, _, val = s[5:].partition(",")
            try:
                sym[name.strip()] = _eval(val.strip(), sym)[0]
            except KeyError:
                sym.pop(name.strip(), None)
            snap = None
            continue
        if s.startswith((".", ";")):
            continue
        if s.endswith(":") and " " not in s:
            entries.append(s[:-1])
            continue
        if snap is None:
            snap = dict(sym)
        op, _, rest = s.partition(" ")
        entries.append(_Ins(op, rest.strip(), snap, s))
    return entries


class _Kernel:
    def __init__(self, base, src):
        self.base, self.src = base, src
        self.entries = e = _parse(src)
        self.labels = {x: i for i, x in enumerate(e) if type(x) is str}
        self.mt0, self.mt1 = (int(x) for x in re.search(r"_MT(\d+)x(\d+)x", src).groups())
        self.lanes = int(re.search(r"\.max_flat_workgroup_size:\s+(\d+)", src)[1])
        assert "s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10" in src, base
        names = re.findall(r"\.name:\s+(\S.*?)\s*\n\s+\.size:\s+(\d+)\s*\n\s+\.offset:\s+(\d+)", src)
        self.args = {n: int(off) - _ARG_SHIFT for n, _sz, off in names}
        self.local = {i for x, i in self.labels.items() if x.startswith("label_fusedA2A_hoist_local")}
        gwb0 = self.labels["label_GW_B0_MB"]
        self.entry = max(i for i in self.labels.values() if i < gwb0)
        self.symD = self.ins(next(i for i in range(gwb0, len(e)) if self.isins(i))).sym
        self.srdD = self.symD["sgprSrdD"]
        optEnd = [i for i in range(len(e)) if self.isins(i) and self.op(i).startswith("s_cbranch")
                  and self.ins(i).rest.split()[0] == "label_OptNLL_End"]
        self.optEntry = max(i for i in self.labels.values() if i < optEnd[0]) if optEnd else None
        first = self.entry if self.optEntry is None else self.optEntry
        writes = [i for i in range(first) if self.isins(i) and _dst(self.ins(i))
                  and self.entries[i].rest.split(",")[0].startswith("s[sgprSrdD")]
        self.srdStart = max(i for i in self.labels.values() if i < writes[0])
        self.srdStop = min(i for i in self.labels.values() if i > writes[-1])
        self.optEnd = self.labels.get("label_OptNLL_End")
        self.replays = []
        for g in range(len(e)):
            if not (self.isins(g) and self.op(g) == "s_cbranch_scc1"
                    and self.ins(g).rest.startswith("label_fusedA2A_hoist_local")):
                continue
            leaf = max(i for i, x in enumerate(e[:g]) if type(x) is str and x.startswith("label_GW_"))
            end = next(i for i in range(g, len(e)) if self.isins(i) and self.op(i) == "s_branch"
                       and self.ins(i).rest.startswith("label_fusedA2A_hoist_after"))
            local = self.labels[self.ins(g).rest.split()[0]]
            tails = [i for i in range(g, end) if self.isins(i) and self.op(i) == "s_cbranch_scc1"
                     and end < self.labels.get(self.ins(i).rest.split()[0], -1) < local]
            self.replays.append(dict(name=e[leaf], leaf=leaf, gate=g, end=end, local=local, tails=tails))

    def isins(self, i):
        return type(self.entries[i]) is _Ins

    def op(self, i):
        return self.entries[i].op

    def ins(self, i):
        x = self.entries[i]
        return x if x.ops is not None else x.decode()

    def replayAt(self, i):
        return next(r for r in self.replays if r["leaf"] <= i <= r["local"])


def _dst(ins):
    if ins.op.startswith(_NO_DST) or not ins.ops or ins.op.startswith(("buffer_store", "global_store")):
        return None
    return ins.ops[0]


class _State:
    def __init__(self, lanes):
        self.lanes = lanes
        self.s, self.v = {}, {}
        self.scc = self.vcc = self.m0 = _Poison("not injected")
        self.exec = np.ones(lanes, dtype=bool)
        self.msb = None
        self.mem = {}
        self.stores = []
        self.maxpc = -1
        self.lane = np.arange(lanes, dtype=np.uint64)
        self.bit = np.arange(lanes) % 32

    def sget(self, i):
        x = self.s.get(i)
        if x is None:
            x = self.s[i] = _Poison(f"s{i} not injected")
        return x

    def vget(self, i):
        x = self.v.get(i)
        if x is None:
            x = self.v[i] = _Poison(f"v{i} not injected")
        return x

    def toMask(self, x):
        if type(x) is np.ndarray:
            return x
        return ((x >> self.bit) & 1).astype(bool)

    def waveUniform(self, mask, why):
        w = mask.reshape(-1, 32).any(axis=1)
        return int(w[0]) if (w == w[0]).all() else _Poison(why)


def _isp(*xs):
    return next((x for x in xs if type(x) is _Poison), None)


def _sread(st, o):
    k = o[0]
    if k == "s":
        return st.sget(o[1])
    if k == "imm":
        return o[1] & _M32
    if k == "null":
        return 0
    if k == "exec":
        return st.exec
    if k == "vcc":
        return st.vcc
    if k == "m0":
        return st.m0
    if k == "scc":
        return st.scc
    if k == "lbl":
        return _Lbl(o[1])
    raise _Bad(f"scalar operand {o} not modelled")


def _sread64(st, o):
    if o[0] == "s":
        return st.sget(o[1]), st.sget(o[1] + 1)
    if o[0] == "imm":
        return o[1] & _M32, (o[1] & _M64) >> 32
    raise _Bad(f"64-bit operand {o} not modelled")


def _swrite(st, o, x):
    if o[0] == "s" and o[2] == 1:
        st.s[o[1]] = x
    elif o[0] == "vcc":
        st.vcc = x
    elif o[0] == "m0":
        st.m0 = x
    elif o[0] == "null":
        pass
    else:
        raise _Bad(f"scalar destination {o} not modelled")


def _vread(st, o):
    k = o[0]
    if k == "v":
        return st.vget(o[1])
    x = _sread(st, o)
    if type(x) is np.ndarray:
        raise _Bad(f"lane mask {o} as a VALU source")
    return x


def _signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def _vsigned(x):
    if type(x) is np.ndarray:
        return x.astype(np.int64) - ((x & 0x80000000).astype(np.int64) << 1)
    return _signed(x)


_SALU = {
    "s_add_u32": (lambda a, b: a + b, lambda a, b, r: r >> 32),
    "s_sub_u32": (lambda a, b: a - b, lambda a, b, r: int(a < b)),
    "s_add_i32": (lambda a, b: a + b, lambda a, b, r: int(not -2**31 <= _signed(a) + _signed(b) < 2**31)),
    "s_sub_i32": (lambda a, b: a - b, lambda a, b, r: int(not -2**31 <= _signed(a) - _signed(b) < 2**31)),
    "s_mul_i32": (lambda a, b: a * b, None),
    "s_mul_hi_u32": (lambda a, b: (a * b) >> 32, None),
    "s_lshl_b32": (lambda a, b: a << (b & 31), lambda a, b, r: int(r & _M32 != 0)),
    "s_lshr_b32": (lambda a, b: a >> (b & 31), lambda a, b, r: int(r != 0)),
    "s_ashr_i32": (lambda a, b: _signed(a) >> (b & 31), lambda a, b, r: int(r & _M32 != 0)),
    "s_and_b32": (lambda a, b: a & b, lambda a, b, r: int(r != 0)),
    "s_or_b32": (lambda a, b: a | b, lambda a, b, r: int(r != 0)),
    "s_xor_b32": (lambda a, b: a ^ b, lambda a, b, r: int(r != 0)),
    "s_andn2_b32": (lambda a, b: a & ~b & _M32, lambda a, b, r: int(r != 0)),
    "s_and_not1_b32": (lambda a, b: a & ~b & _M32, lambda a, b, r: int(r != 0)),
    "s_min_u32": (min, lambda a, b, r: int(a < b)),
    "s_max_u32": (max, lambda a, b, r: int(a > b)),
    "s_min_i32": (lambda a, b: min(_signed(a), _signed(b)), lambda a, b, r: int(_signed(a) < _signed(b))),
    "s_max_i32": (lambda a, b: max(_signed(a), _signed(b)), lambda a, b, r: int(_signed(a) > _signed(b))),
    "s_bfe_u32": (lambda a, b: (a >> (b & 31)) & ((1 << ((b >> 16) & 0x7F)) - 1), lambda a, b, r: int(r != 0)),
    "s_bfm_b32": (lambda a, b: ((1 << (a & 31)) - 1) << (b & 31), None),
    "s_lshl1_add_u32": (lambda a, b: (a << 1) + b, lambda a, b, r: r >> 32),
    "s_lshl2_add_u32": (lambda a, b: (a << 2) + b, lambda a, b, r: r >> 32),
}
_MASK_OPS = {"s_and_b32": np.logical_and, "s_or_b32": np.logical_or, "s_xor_b32": np.logical_xor,
             "s_andn2_b32": lambda a, b: a & ~b, "s_and_not1_b32": lambda a, b: a & ~b}
_CMP = {"eq": lambda a, b: a == b, "lg": lambda a, b: a != b, "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}
_VALU = {
    "v_mov_b32": (1, lambda a: a),
    "v_add_nc_u32": (2, lambda a, b: a + b),
    "v_add_nc_i32": (2, lambda a, b: a + b),
    "v_add_u32": (2, lambda a, b: a + b),
    "v_sub_nc_u32": (2, lambda a, b: a - b + (1 << 32)),
    "v_subrev_nc_u32": (2, lambda a, b: b - a + (1 << 32)),
    "v_mul_lo_u32": (2, lambda a, b: a * b),
    "v_mul_hi_u32": (2, lambda a, b: (a * b) >> 32),
    "v_lshlrev_b32": (2, lambda a, b: b << (a & 31)),
    "v_lshrrev_b32": (2, lambda a, b: b >> (a & 31)),
    "v_and_b32": (2, lambda a, b: a & b),
    "v_or_b32": (2, lambda a, b: a | b),
    "v_xor_b32": (2, lambda a, b: a ^ b),
    "v_add_lshl_u32": (3, lambda a, b, c: ((a + b) & _M32) << (c & 31)),
    "v_lshl_add_u32": (3, lambda a, b, c: ((a << (b & 31)) & _M32) + c),
    "v_lshl_or_b32": (3, lambda a, b, c: ((a << (b & 31)) & _M32) | c),
    "v_add3_u32": (3, lambda a, b, c: a + b + c),
    "v_and_or_b32": (3, lambda a, b, c: (a & b) | c),
    "v_or3_b32": (3, lambda a, b, c: a | b | c),
}


def _asU64(x):
    return x.astype(np.uint64) if type(x) is np.ndarray else x


def _checkMsb(st, ins, srcs):
    if st.msb is None:
        return
    d = ins.ops[0]
    if d[0] == "v" and d[3] != 256 * ((st.msb >> 6) & 3):
        raise _Bad(f"{ins.text!r}: destination bank {d[3]} under s_set_vgpr_msb {st.msb}")
    for k, o in enumerate(srcs[:3]):
        if o[0] == "v" and o[3] != 256 * ((st.msb >> (2 * k)) & 3):
            raise _Bad(f"{ins.text!r}: source {k} bank {o[3]} under s_set_vgpr_msb {st.msb}")


def _poisonDst(st, ins, why):
    for o in ins.ops[:1] + [o for o in ins.ops[1:2] if "_co_" in ins.op or ins.op.startswith("v_cmp")]:
        if o[0] == "v":
            for i in range(o[2]):
                st.v[o[1] + i] = _Poison(why)
        elif o[0] == "s":
            for i in range(o[2]):
                st.s[o[1] + i] = _Poison(why)
        elif o[0] in ("vcc", "m0"):
            _swrite(st, o, _Poison(why))
        elif o[0] == "exec":
            raise _Bad(f"{ins.text!r}: EXEC write not modelled")
    if ins.op.startswith("s_"):
        st.scc = _Poison(why)


def _width(op):
    m = re.search(r"_b(\d+)$", op)
    if not m:
        raise _Bad(f"store width of {op} not modelled")
    return int(m[1]) // 8


def _step(K, st, pc, ins):
    """Next pc, or a (reason,) tuple to stop."""
    op, o = ins.op, ins.ops
    if op in _SALU:
        a, b = _sread(st, o[1]), _sread(st, o[2])
        if type(a) is np.ndarray or type(b) is np.ndarray:
            if op not in _MASK_OPS:
                raise _Bad(f"{ins.text!r}: lane mask operand")
            r = _MASK_OPS[op](st.toMask(a), st.toMask(b))
            _swrite(st, o[0], r)
            st.scc = st.waveUniform(r, f"{ins.text!r}: per-wave SCC")
            return pc + 1
        if type(a) in (_Pc, _Lbl) or type(b) in (_Pc, _Lbl):
            if op not in ("s_add_u32", "s_add_i32") or _Lbl not in (type(a), type(b)):
                raise _Bad(f"{ins.text!r}: branch target arithmetic not modelled")
            _swrite(st, o[0], a if type(a) is _Lbl else b)
            return pc + 1
        p = _isp(a, b)
        if p is not None:
            _swrite(st, o[0], p)
            if _SALU[op][1] is not None:
                st.scc = p
            return pc + 1
        fn, sccFn = _SALU[op]
        r = fn(a, b)
        _swrite(st, o[0], r & _M32)
        if sccFn is not None:
            st.scc = sccFn(a, b, r)
        return pc + 1
    if op in ("s_addc_u32", "s_subb_u32"):
        a, b, c = _sread(st, o[1]), _sread(st, o[2]), st.scc
        if type(a) is _Pc:
            _swrite(st, o[0], _PC)
            return pc + 1
        p = _isp(a, b, c)
        if p is not None:
            _swrite(st, o[0], p)
            st.scc = p
            return pc + 1
        r = a + b + c if op == "s_addc_u32" else a - b - c
        _swrite(st, o[0], r & _M32)
        st.scc = r >> 32 if op == "s_addc_u32" else int(a < b + c)
        return pc + 1
    if op.startswith("s_cmp_"):
        m = re.fullmatch(r"s_cmp_(\w\w)_([ui])(32|64)", op)
        if not m:
            raise _Bad(f"{ins.text!r} not modelled")
        if m[3] == "64":
            a, b = _sread64(st, o[0]), _sread64(st, o[1])
            p = _isp(*a, *b)
            st.scc = p if p is not None else int(_CMP[m[1]](a[0] | a[1] << 32, b[0] | b[1] << 32))
            return pc + 1
        a, b = _sread(st, o[0]), _sread(st, o[1])
        p = _isp(a, b)
        if p is None and (type(a) is not int or type(b) is not int):
            raise _Bad(f"{ins.text!r}: non-integer compare")
        if p is not None:
            st.scc = p
        else:
            if m[2] == "i":
                a, b = _signed(a), _signed(b)
            st.scc = int(_CMP[m[1]](a, b))
        return pc + 1
    if op in ("s_bitcmp1_b32", "s_bitcmp0_b32"):
        a, b = _sread(st, o[0]), _sread(st, o[1])
        p = _isp(a, b)
        st.scc = p if p is not None else int(((a >> (b & 31)) & 1) == (op == "s_bitcmp1_b32"))
        return pc + 1
    if op == "s_mov_b32":
        _swrite(st, o[0], _sread(st, o[1]))
        return pc + 1
    if op in ("s_mov_b64", "s_cselect_b64"):
        src = o[1] if op == "s_mov_b64" or st.scc == 1 else o[2]
        if op == "s_cselect_b64" and type(st.scc) is _Poison:
            lo = hi = st.scc
        else:
            lo, hi = _sread64(st, src)
        st.s[o[0][1]], st.s[o[0][1] + 1] = lo, hi
        return pc + 1
    if op == "s_cselect_b32":
        a, b = _sread(st, o[1]), _sread(st, o[2])
        _swrite(st, o[0], st.scc if type(st.scc) is _Poison else (a if st.scc else b))
        return pc + 1
    if op == "s_cmov_b32":
        if type(st.scc) is _Poison:
            _swrite(st, o[0], st.scc)
        elif st.scc:
            _swrite(st, o[0], _sread(st, o[1]))
        return pc + 1
    if op == "s_abs_i32":
        a = _sread(st, o[1])
        r = a if type(a) is _Poison else abs(_signed(a)) & _M32
        _swrite(st, o[0], r)
        st.scc = r if type(r) is _Poison else int(r != 0)
        return pc + 1
    if op in ("s_lshl_b64", "s_add_u64", "s_sub_u64"):
        a = _sread64(st, o[1])
        b = _sread(st, o[2]) if op == "s_lshl_b64" else _sread64(st, o[2])
        bs = b if type(b) is tuple else (b,)
        p = _isp(*a, *bs)
        if p is not None:
            st.s[o[0][1]] = st.s[o[0][1] + 1] = p
            if op == "s_lshl_b64":
                st.scc = p
            return pc + 1
        av = a[0] | a[1] << 32
        if op == "s_lshl_b64":
            r = (av << (b & 63)) & _M64
            st.scc = int(r != 0)
        else:
            bv = b[0] | b[1] << 32
            r = (av + bv if op == "s_add_u64" else av - bv) & _M64
        st.s[o[0][1]], st.s[o[0][1] + 1] = r & _M32, r >> 32
        return pc + 1
    if op == "s_getpc_b64":
        st.s[o[0][1]], st.s[o[0][1] + 1] = _PC, _PC
        return pc + 1
    if op == "s_setpc_b64":
        t = st.sget(o[0][1])
        if type(t) is not _Lbl:
            raise _Bad(f"{ins.text!r}: target {t!r}")
        return K.labels[t.name]
    if op == "s_branch":
        return K.labels[o[0][1]]
    if op.startswith("s_cbranch_"):
        cond = op[len("s_cbranch_"):]
        if cond in ("scc0", "scc1"):
            c = st.scc
        elif cond in ("vccz", "vccnz"):
            c = st.vcc if type(st.vcc) is _Poison else st.waveUniform(st.toMask(st.vcc), "per-wave VCC")
        elif cond in ("execz", "execnz"):
            c = st.waveUniform(st.exec, "per-wave EXEC")
        else:
            raise _Bad(f"{ins.text!r} not modelled")
        if type(c) is _Poison:
            raise _Bad(f"{ins.text!r}: branch on {c.why}")
        take = bool(c) == (cond in ("scc1", "vccnz", "execnz"))
        return K.labels[o[0][1]] if take else pc + 1
    if op.startswith("s_load_b"):
        n = int(op[len("s_load_b"):]) // 32
        base = st.sget(o[1][1])
        dst = o[0]
        if base is not _KA:
            for i in range(n):
                st.s[dst[1] + i] = _Poison(f"{ins.text!r}: load from {base!r}")
            return pc + 1
        off = _sread(st, o[2]) if len(o) > 2 else 0
        if type(off) is _Poison:
            raise _Bad(f"{ins.text!r}: offset {off.why}")
        addr = off + int(ins.mods.get("offset", "0"), 0)
        for i in range(n):
            st.s[dst[1] + i] = st.mem.get(addr + 4 * i, _Poison(f"kernarg 0x{addr + 4 * i:x} not injected"))
        return pc + 1
    if op == "s_set_vgpr_msb":
        st.msb = o[0][1] & 0xFF
        return pc + 1
    if op == "s_endpgm":
        return ("end",)
    if op.startswith(_NO_DST):
        return pc + 1

    if op in _VALU:
        n, fn = _VALU[op]
        srcs = o[1:1 + n]
        _checkMsb(st, ins, srcs)
        args = [_asU64(_vread(st, x)) for x in srcs]
        p = _isp(*args)
        if p is not None:
            st.v[o[0][1]] = p
        elif op in ("v_mul_lo_u32", "v_mul_hi_u32") and any(type(a) is np.ndarray for a in args):
            a, b = (np.asarray(x, dtype=np.uint64) for x in args)
            prod = [int(x) * int(y) for x, y in zip(np.broadcast_to(a, st.lane.shape), np.broadcast_to(b, st.lane.shape))]
            prod = [(x >> 32) if op == "v_mul_hi_u32" else x for x in prod]
            st.v[o[0][1]] = np.array([x & _M32 for x in prod], dtype=np.uint64)
        else:
            r = fn(*args)
            st.v[o[0][1]] = (r & _M32) if type(r) is int else (r & np.uint64(_M32))
        return pc + 1
    if op.startswith("v_cmp_") and not op.startswith("v_cmpx"):
        m = re.fullmatch(r"v_cmp_(\w\w)_([ui])32", op)
        if not m:
            _poisonDst(st, ins, f"{ins.text!r} not modelled")
            return pc + 1
        _checkMsb(st, ins, o[1:3])
        a, b = _vread(st, o[1]), _vread(st, o[2])
        p = _isp(a, b)
        if p is not None:
            _swrite(st, o[0], p)
            return pc + 1
        if m[2] == "i":
            a, b = _vsigned(a), _vsigned(b)
        r = np.broadcast_to(np.asarray(_CMP[m[1]](a, b)), (st.lanes,)).copy()
        _swrite(st, o[0], r)
        return pc + 1
    if op == "v_cndmask_b32":
        _checkMsb(st, ins, o[1:3])
        a, b, c = _vread(st, o[1]), _vread(st, o[2]), _sread(st, o[3])
        p = _isp(c, a, b)
        st.v[o[0][1]] = p if p is not None else np.where(st.toMask(c), np.uint64(0) + b, np.uint64(0) + a)
        return pc + 1
    if op in ("v_add_co_u32", "v_add_co_ci_u32"):
        _checkMsb(st, ins, o[2:4])
        a, b = _asU64(_vread(st, o[2])), _asU64(_vread(st, o[3]))
        c = st.toMask(_sread(st, o[4])).astype(np.uint64) if op == "v_add_co_ci_u32" else 0
        p = _isp(a, b, c)
        if p is not None:
            st.v[o[0][1]] = p
            _swrite(st, o[1], p)
            return pc + 1
        r = np.broadcast_to(np.uint64(0) + a + b + c, (st.lanes,))
        st.v[o[0][1]] = r & np.uint64(_M32)
        _swrite(st, o[1], (r >> np.uint64(32)) != 0)
        return pc + 1
    if op == "v_readfirstlane_b32":
        a = _vread(st, o[1])
        if type(a) is np.ndarray:
            w = a[::32]
            a = int(w[0]) if (w == w[0]).all() else _Poison(f"{ins.text!r}: per-wave value")
        _swrite(st, o[0], a)
        return pc + 1
    if op.startswith("buffer_store_"):
        vdata, vaddr, srsrc, soff = o[:4]
        if not ins.mods.get("offen") or srsrc[0] != "s":
            raise _Bad(f"{ins.text!r}: addressing mode not modelled")
        srd = [st.sget(srsrc[1] + i) for i in range(4)]
        va, so = _vread(st, vaddr), _sread(st, soff)
        p = _isp(*srd, va, so)
        if p is not None:
            raise _Bad(f"{ins.text!r}: address from {p.why}")
        if any(type(x) is not int for x in srd + [so]):
            raise _Bad(f"{ins.text!r}: SRD or soffset is not an integer")
        off = np.broadcast_to(np.uint64(0) + _asU64(va) + so + int(ins.mods.get("offset", "0"), 0), (st.lanes,))
        base = srd[0] | (srd[1] & 0x1FFFFFF) << 32
        nr = (srd[2] << 7) | (srd[1] >> 25)
        data = tuple(st.vget(vdata[1] + i) for i in range(vdata[2]))
        kind = "D" if srsrc[1] == K.srdD else "R"
        st.stores.append((kind, pc, _width(op), base, nr, srd[3], off.copy(), data, st.exec.copy()))
        return pc + 1
    if op.startswith(("buffer_load", "global_load", "ds_load", "ds_read", "v_wmma", "scratch_load")):
        _poisonDst(st, ins, f"{ins.text!r}")
        return pc + 1
    if op.startswith(("global_", "buffer_", "ds_", "flat_", "scratch_")):
        raise _Bad(f"{ins.text!r}: memory write before the handshake")
    if op.startswith(("v_", "s_")) and ins.ops and ins.ops[0][0] in ("v", "s", "vcc", "m0", "exec"):
        _poisonDst(st, ins, f"{ins.text!r} not modelled")
        return pc + 1
    raise _Bad(f"{ins.text!r} not modelled")


def _run(K, st, pc, stops, limit=300000):
    """(reason, pc, last label_GW_ label) after running from pc until a stop."""
    entries = K.entries
    leaf = None
    for _ in range(limit):
        x = entries[pc]
        if type(x) is str:
            if pc in stops:
                return stops[pc], pc, leaf
            if pc in K.local:
                return "local", pc, leaf
            if pc == K.optEnd:
                return "noOptNLL", pc, leaf
            if x.startswith("label_GW_"):
                leaf = x
            pc += 1
            continue
        ins = x if x.ops is not None else x.decode()
        d = _dst(ins)
        if (d is not None and d[0] == "exec") or "saveexec" in ins.op or ins.op.startswith("v_cmpx"):
            return "exec", pc, leaf
        if pc > st.maxpc:
            st.maxpc = pc
        try:
            nxt = _step(K, st, pc, ins)
        except _Bad:
            raise
        except Exception as e:
            raise _Bad(f"{ins.text!r}: {type(e).__name__}: {e}") from e
        if type(nxt) is tuple:
            return nxt[0], pc, leaf
        pc = nxt
    raise _Bad(f"no stop within {limit} instructions")


class _Case:
    def __init__(self, K, *, W, ns, N, r, w0, w1, mask, beta=0, alpha=_ONE, ldd=None, extraCols=1):
        self.W, self.ns, self.N, self.r, self.w0, self.w1 = W, ns, N, r, w0, w1
        self.mask, self.beta, self.alpha = mask, beta, alpha
        self.am = W * ns
        self.M = self.am + extraCols * K.mt0
        self.ldd = ldd or self.M
        self.drain = 3 | (mask << 8) | (bin(mask).count("1") << 16)

    def __repr__(self):
        return (f"W={self.W} nShard={self.ns} N={self.N} r={self.r} wg=({self.w0},{self.w1}) mask=0x{self.mask:x} "
                f"beta=0x{self.beta:x} alpha=0x{self.alpha:x} M={self.M} ldd={self.ldd}")


def _inject(K, st, c):
    sym = K.symD
    mem = {}

    def arg(name, value, size=4):
        mem[K.args[name]] = value & _M32
        if size == 8:
            mem[K.args[name] + 4] = value >> 32

    arg("FusedMyRank", c.r)
    arg("FusedW", c.W)
    arg("FusedDrain", c.drain)
    arg("FusedAM", c.am)
    for j in range(c.W):
        arg(f"peer_{j}_bufPtr", _BUF[j], 8)
    st.mem = mem
    nwg0, nwg1 = -(-c.M // K.mt0), -(-c.N // K.mt1)
    vals = {
        ("sgprKernArgAddress", 0): _KA, ("sgprKernArgAddress", 1): _KA,
        ("sgprWorkGroup0", 0): c.w0, ("sgprWorkGroup1", 0): c.w1, ("sgprWorkGroup2", 0): 0,
        ("sgprNumWorkGroups0", 0): nwg0, ("sgprNumWorkGroups1", 0): nwg1,
        ("sgprSizesFree", 0): c.M, ("sgprSizesFree", 1): c.N, ("sgprSizesFree", 2): 1, ("sgprSizesSum", 0): _K,
        ("sgprStridesD", 0): c.ldd, ("sgprStridesD", 1): c.ldd * c.N,
        ("sgprStridesC", 0): c.ldd, ("sgprStridesC", 1): c.ldd * c.N,
        ("sgprAddressD", 0): _ADDR_D & _M32, ("sgprAddressD", 1): _ADDR_D >> 32,
        ("sgprAddressC", 0): _ADDR_C & _M32, ("sgprAddressC", 1): _ADDR_C >> 32,
        ("sgprAlpha", 0): c.alpha, ("sgprBeta", 0): c.beta, ("sgprGSU", 0): 1, ("sgprArgType", 0): 0,
        ("sgprGSULog2BpeD", 0): 1, ("sgprGSULog2BpeC", 0): 1, ("sgprGSUSumIdx", 0): 0,
        ("sgprFusedNShard", 0): c.ns,
    }
    srdSym = K.ins(next(i for i in range(K.srdStart, len(K.entries)) if K.isins(i))).sym
    for (name, k), v in vals.items():
        at = srdSym.get(name, sym.get(name))
        if at is not None:
            st.s[at + k] = v
    st.v[sym["vgprSerial"]] = st.lane.copy()


def _simulate(K, c):
    """Interpreted run of one work-group: (stop reason, stop pc, replay label, state)."""
    st = _State(K.lanes)
    _inject(K, st, c)
    reason, _pc, _l = _run(K, st, K.srdStart, {K.srdStop: "srd"})
    assert reason == "srd", f"{K.base}: D SRD set-up stopped at {reason}"
    st.maxpc = -1
    reason = "noOptNLL"
    if K.optEntry is not None:
        reason, pc, leaf = _run(K, st, K.optEntry, {})
    if reason == "noOptNLL":
        reason, pc, leaf = _run(K, st, K.entry, {})
    return reason, pc, leaf, st


def _check(K, c, sim=None, expectDirect=None):
    """Asserts the interpreted stores of case c; returns (replay label, direct, D stores, recv stores)."""
    reason, pc, leaf, st = sim or _simulate(K, c)
    where = f"{K.base} {leaf} {c!r}"
    push = c.w0 < c.am // K.mt0
    if not push:
        assert reason == "local" and not st.stores, f"{where}: local work-group ended at {reason}"
        return leaf, False, [], []
    assert reason == "exec", f"{where}: stopped at {reason} ({K.entries[pc]!r})"
    rp = K.replayAt(pc)
    assert rp["gate"] < pc < rp["end"], f"{where}: EXEC write outside the PUSH store path"
    assert not any(K.isins(i) and K.op(i).startswith("buffer_store") and K.ins(i).ops[2][1] == K.srdD
                   for i in range(pc, rp["end"])), f"{where}: D store after the handshake entry"
    D = [x for x in st.stores if x[0] == "D"]
    R = [x for x in st.stores if x[0] == "R"]
    assert D, f"{where}: no D store"
    for x in st.stores:
        assert x[8].all(), f"{where}: store with a partial EXEC"
    cold = st.maxpc > rp["end"]
    rows = min(K.mt1, c.N - c.w1 * K.mt1)
    p = c.w0 * K.mt0 // c.ns
    direct = bool(c.mask >> p & 1)
    if expectDirect is not None:
        assert direct == expectDirect, where
    slabLo = _BUF[p] + _BPE * c.r * c.N * c.ns
    slabHi = slabLo + _BPE * c.N * c.ns

    allE = []
    for (_k, dpc, w, base, nr, d3, off, data, _e) in D:
        inb = off + np.uint64(w) <= np.uint64(nr)
        assert (inb | (off >= np.uint64(nr))).all(), f"{where}: D store at line {dpc} partially in bounds"
        a = np.uint64(base) + off[inb] - np.uint64(_ADDR_D)
        assert base >= _ADDR_D and (a % np.uint64(_BPE) == 0).all(), f"{where}: D address not on an element"
        allE.append(a // np.uint64(_BPE))
    e = np.concatenate(allE).astype(np.int64)
    m, t = e % c.ldd, e // c.ldd
    vw = D[0][2] // _BPE
    assert all(x[2] == D[0][2] for x in D), f"{where}: D store widths differ"
    assert ((m >= c.w0 * K.mt0) & (m + vw <= (c.w0 + 1) * K.mt0)).all(), f"{where}: D column outside the tile"
    assert ((t >= c.w1 * K.mt1) & (t < c.w1 * K.mt1 + rows)).all(), f"{where}: D row outside the tile or >= N"
    assert len(np.unique(e)) == len(e) and len(e) * vw == K.mt0 * rows, (
        f"{where}: D stores {len(e)} units of {vw} for a {K.mt0}x{rows} tile")

    if not direct:
        assert not R, f"{where}: {len(R)} receive stores for a non-direct dst_rank {p}"
        assert not cold, f"{where}: cold block executed for a non-direct dst_rank {p}"
        return leaf, False, D, R
    assert cold and len(R) == len(D), f"{where}: {len(R)} receive stores for {len(D)} D stores"
    for k, (dx, rx) in enumerate(zip(D, R)):
        _k, dpc, w, dbase, dnr, dd3, doff, ddata, _e = dx
        _k, rpc, rw, rbase, rnr, rd3, roff, rdata, _e = rx
        at = f"{where}: D store {k} (line {dpc}) / receive store (line {rpc})"
        assert rw == w and all(a is b for a, b in zip(rdata, ddata)) and len(rdata) == len(ddata), (
            f"{at}: data or width differs")
        assert rnr != 0 and slabLo <= rbase and rbase + rnr <= slabHi, (
            f"{at}: window [0x{rbase:x}, +{rnr}) outside slab [0x{slabLo:x}, 0x{slabHi:x})")
        assert rd3 == dd3, f"{at}: dword3 0x{rd3:x} != D's 0x{dd3:x}"
        dIn = doff + np.uint64(w) <= np.uint64(dnr)
        rIn = roff + np.uint64(w) <= np.uint64(rnr)
        rOut = roff >= np.uint64(rnr)
        bad = np.nonzero(dIn != rIn)[0]
        assert not len(bad), (
            f"{at}: lane {bad[0]} D {'in' if dIn[bad[0]] else 'out of'} bounds, receive offset {int(roff[bad[0]])} "
            f"NR {rnr}")
        assert (rIn | rOut).all(), f"{at}: receive store partially in bounds"
        ee = (np.uint64(dbase) + doff[dIn] - np.uint64(_ADDR_D)) // np.uint64(_BPE)
        mm, tt = (ee % np.uint64(c.ldd)).astype(np.int64), (ee // np.uint64(c.ldd)).astype(np.int64)
        want = _BUF[p] + _BPE * ((c.r * c.N + tt) * c.ns + mm - p * c.ns)
        got = (np.uint64(rbase) + roff[dIn]).astype(np.int64)
        bad = np.nonzero(got != want)[0]
        if len(bad):
            lane = np.nonzero(dIn)[0][bad[0]]
            raise AssertionError(
                f"{at}: lane {lane} (f={mm[bad[0]] - p * c.ns}, t={tt[bad[0]]}) writes 0x{int(got[bad[0]]):x} "
                f"= bufPtr[{p}]+{int(got[bad[0]]) - _BUF[p]}, expected bufPtr[{p}]+{int(want[bad[0]]) - _BUF[p]} "
                f"(window base bufPtr[{p}]+{rbase - _BUF[p]}, NR {rnr}, offset {int(roff[lane])})")
    return leaf, True, D, R


def _sameD(a, b):
    return len(a) == len(b) and all(x[1] == y[1] and x[3] == y[3] and x[4] == y[4] and (x[6] == y[6]).all()
                                    for x, y in zip(a, b))


def _sweepCases(K, Ws=(1, 2, 4, 8), shards=(1, 2, 5), Ns=(100, 1023, 1024, 1792, 1800, 2048)):
    """Per (W, nShard, N): PUSH tiles (first, first), (first, last), (middle, middle), (last, first) and
    (last, last), each with one direct and one non-direct mask (bit p / 0 or all / all but p), and one
    LOCAL tile; r, beta, alpha and ldd rotate over the cases."""
    out = []
    n = 0
    for W in Ws:
        for k in shards:
            ns = k * K.mt0
            for N in Ns:
                nwg1 = -(-N // K.mt1)
                last = W * ns // K.mt0 - 1
                full = (1 << W) - 1
                for w0, w1 in ((0, 0), (0, nwg1 - 1), (last // 2, nwg1 // 2), (last, 0), (last, nwg1 - 1)):
                    p = w0 * K.mt0 // ns
                    pair = [(1 << p, 0), (full, full & ~(1 << p))][n % 2]
                    common = dict(W=W, ns=ns, N=N, r=(n * 3) % W, w0=w0, w1=w1, beta=(0, _ONE)[(n // 2) % 2],
                                  alpha=(_ONE, _TWO)[(n // 4) % 2], ldd=W * ns + K.mt0 + 64 * (n % 3 == 0))
                    out.append([dict(common, mask=m) for m in pair])
                    n += 1
                out.append([dict(W=W, ns=ns, N=N, r=W - 1, w0=last + 1, w1=nwg1 - 1, mask=full)])
    return out


@pytest.mark.parametrize("which", range(6))
def test_fused_a2a_gfx1250_direct_store_addresses(kernels, which):
    K = kernels[which]
    seen = {}
    for group in _sweepCases(K):
        stores = []
        for kw in group:
            leaf, direct, D, _R = _check(K, _Case(K, **kw))
            stores.append(D)
            if D:
                seen.setdefault(leaf, set()).add(direct)
        assert all(_sameD(stores[0], x) for x in stores[1:]), f"{K.base} {group}: D stores depend on the mask"
    assert set(seen) == {r["name"] for r in K.replays if r["tails"]}, f"{K.base}: replays reached {sorted(seen)}"
    assert all(v == {True, False} for v in seen.values()), f"{K.base}: {seen}"


def _elements(d, lane):
    e = int((np.uint64(d[3]) + d[6][lane] - np.uint64(_ADDR_D)) // np.uint64(_BPE))
    return e % 10240, e // 10240


def test_fused_a2a_gfx1250_direct_store_worked_examples(kernels):
    big = [K for K in kernels if K.mt0 == 256]
    assert len(big) == 5
    laneChecked = 0
    for K in big:
        a = _Case(K, W=8, ns=1280, N=2048, r=3, w0=10, w1=5, mask=0x0C, extraCols=0, ldd=10240)
        assert a.drain == 0x00020C03 and a.M == 10240
        leaf, direct, D, R = _check(K, a, expectDirect=True)
        assert {(x[3], x[4]) for x in R} == {(_BUF[2] + 19_005_440, 655_360)}, f"{K.base} {leaf}"
        hit = [(lane, rr[3] + int(rr[6][lane])) for d, rr in zip(D, R) for lane in range(K.lanes)
               if _elements(d, lane) == (2728, 1442)]
        assert [h[1] for h in hit] == [_BUF[2] + 19_420_496], f"{K.base}: element (2728, 1442) -> {hit}"
        if _elements(D[0], 53) == (2728, 1312):
            laneChecked += 1
            assert int(R[0][6][53]) == 82_256 and hit[0][0] == 53, f"{K.base}: DwLane {int(R[0][6][53])}"

        for r, end in ((3, None), (7, 36_864_000)):
            b = _Case(K, W=8, ns=1280, N=1800, r=r, w0=10, w1=7, mask=0x0C, extraCols=0, ldd=10240)
            leaf, direct, D, R = _check(K, b, expectDirect=True)
            base = _BUF[2] + 2560 * (r * 1800 + 1792)
            assert {(x[3], x[4]) for x in R} == {(base, 20_480)}, f"{K.base} {leaf} r={r}"
            assert r != 3 or base == _BUF[2] + 18_411_520
            assert end is None or base + 20_480 == _BUF[2] + end
            written = sum(int((x[6] + np.uint64(x[2]) <= np.uint64(x[4])).sum()) * x[2] for x in R)
            assert written == _BPE * 256 * 8, f"{K.base}: {written} bytes written for an 8-row tile"
            if _elements(D[0], 0) == (2560, 1792) and _elements(D[0], 16)[1] == 1824:
                lane0 = sorted(int(x[6][0]) for x in R)
                assert 17_920 in lane0 and 20_480 in lane0 and int(R[0][6][16]) == 81_920, f"{K.base}: {lane0}"
    assert laneChecked, "no kernel holds example A's lane 53 element"


def test_fused_a2a_gfx1250_direct_store_exhaustive_small(kernels):
    """W=2, nShard=MT0, N=300, both ranks, every work-group (PUSH and LOCAL) and lane, mask all."""
    for K in kernels:
        W, ns, N = 2, K.mt0, 300
        size = _BPE * W * N * ns
        recv = [np.zeros(size, dtype=np.int32) for _ in range(W)]
        dcount = np.zeros(_BPE * (W * ns + K.mt0) * N, dtype=np.int32)
        stray = []
        for r in range(W):
            for w1 in range(-(-N // K.mt1)):
                for w0 in range(W * ns // K.mt0 + 1):
                    c = _Case(K, W=W, ns=ns, N=N, r=r, w0=w0, w1=w1, mask=3, beta=(0, _ONE)[(w0 + w1) % 2])
                    _leaf, _direct, D, R = _check(K, c)
                    for x in R:
                        inb = x[6] + np.uint64(x[2]) <= np.uint64(x[4])
                        for a in (np.uint64(x[3]) + x[6][inb]).astype(np.int64).tolist():
                            q = next((j for j in range(W) if _BUF[j] <= a < _BUF[j] + size), None)
                            if q is None:
                                stray.append(a)
                            else:
                                recv[q][a - _BUF[q]:a - _BUF[q] + x[2]] += 1
                    for x in D if r == 0 else ():
                        inb = x[6] + np.uint64(x[2]) <= np.uint64(x[4])
                        for a in (np.uint64(x[3]) + x[6][inb] - np.uint64(_ADDR_D)).astype(np.int64).tolist():
                            dcount[a:a + x[2]] += 1
        assert not stray, f"{K.base}: receive stores outside every buffer: {stray[:4]}"
        for q in range(W):
            assert (recv[q] == 1).all(), f"{K.base}: buffer {q} byte write counts {np.bincount(recv[q]).tolist()}"
        assert (dcount.reshape(N, -1)[:, :_BPE * W * ns] == 1).all(), f"{K.base}: D tiles do not cover AM x N"


def _hint(op):
    return op.startswith(("s_wait", "s_nop", "s_set_vgpr_msb", "s_delay_alu", "s_prefetch"))


_SCC_W = ("s_cmp", "s_bitcmp", "s_add_", "s_sub_", "s_addc", "s_subb", "s_and_", "s_or_", "s_xor_", "s_lshl",
          "s_lshr", "s_ashr", "s_bfe", "s_min", "s_max", "s_abs")


def _regsOf(o):
    if o[0] in ("s", "v"):
        return {(o[0], o[1] + i) for i in range(o[2])}
    return {(o[0],)} if o[0] in ("vcc", "exec", "m0") else set()


def _slice(K, lo, hi, need):
    """Indices in (lo, hi), in program order, whose results reach the registers in need at hi."""
    out = []
    for i in range(hi - 1, lo, -1):
        if not K.isins(i):
            continue
        ins = K.ins(i)
        d = _dst(ins)
        w = _regsOf(d) if d else set()
        if ins.op.startswith(_SCC_W):
            w = w | {("scc",)}
        if w & need:
            out.append(i)
            need = (need - w).union(*(_regsOf(o) for o in (ins.ops[1:] if d else ins.ops)))
            if ins.op.startswith(_SCC_READ):
                need.add(("scc",))
    return out[::-1]


def _decoders(K, rp):
    """(prologue end, decision register, handshake slice, handshake bit test) of a direct replay."""
    proEnd = min(i for i in K.labels.values() if i > rp["gate"])
    t0 = rp["tails"][0]
    ins = K.ins(max(i for i in range(proEnd, t0) if K.isins(i) and not _hint(K.op(i))))
    assert ins.op == "s_cmp_lg_u32" and ins.ops[1] == ("imm", 0), f"{K.base} {rp['name']}: tail test {ins.text!r}"
    reg = ins.ops[0]
    assert not any(K.isins(i) and _dst(K.ins(i)) and _regsOf(_dst(K.ins(i))) & _regsOf(reg)
                   for i in range(proEnd, t0)), f"{K.base} {rp['name']}: decision register redefined"
    entry = next(i for i in range(rp["tails"][-1], rp["end"]) if K.isins(i) and _dst(K.ins(i))
                 and _dst(K.ins(i))[0] == "exec")
    bit = next(i for i in range(entry, rp["end"]) if K.isins(i) and K.op(i) == "s_bitcmp1_b32"
               and K.op(min(j for j in range(i + 1, rp["end"]) if K.isins(j) and not _hint(K.op(j))))
               == "s_cbranch_scc1")
    return proEnd, reg, _slice(K, entry, bit, set().union(*(_regsOf(o) for o in K.ins(bit).ops))), bit


def _decide(K, rp, dec, c):
    proEnd, reg, sl, bit = dec
    st = _State(32)
    _inject(K, st, c)
    reason, _pc, _l = _run(K, st, rp["leaf"], {proEnd: "pro"})
    assert reason == "pro", f"{K.base} {rp['name']} {c!r}: prologue ended at {reason}"
    store = st.sget(reg[1])
    hs = _State(32)
    _inject(K, hs, c)
    for i in sl + [bit]:
        _step(K, hs, i, K.ins(i))
    assert type(store) is int and type(hs.scc) is int, f"{K.base} {rp['name']} {c!r}: {store!r} {hs.scc!r}"
    return store != 0, bool(hs.scc)


def test_fused_a2a_gfx1250_direct_decode_agrees(kernels):
    """The store prologue and the handshake decide 'direct' alike, for every (W, mask, WG0) on the
    first direct replay of each kernel and for masks 0, bit p, all and all but p on the other replays."""
    for K in kernels:
        reps = [r for r in K.replays if r["tails"]]
        assert reps
        for n, rp in enumerate(reps):
            dec = _decoders(K, rp)
            for W in (1, 2, 4, 8):
                full = (1 << W) - 1
                for k in (1, 2, 5):
                    ns = k * K.mt0
                    for w0 in range(W * ns // K.mt0):
                        p = w0 * K.mt0 // ns
                        for mask in range(1 << W) if n == 0 else sorted({0, 1 << p, full, full & ~(1 << p)}):
                            c = _Case(K, W=W, ns=ns, N=1024, r=W - 1, w0=w0, w1=0, mask=mask)
                            want = bool(mask >> p & 1)
                            got = _decide(K, rp, dec, c)
                            assert got == (want, want), (
                                f"{K.base} {rp['name']} {c!r}: prologue {got[0]}, handshake {got[1]}, "
                                f"FusedDrain bit {8 + p} = {want}")


def _quick(K):
    for W, k in ((2, 1), (8, 5)):
        ns = k * K.mt0
        for N in (1800, 2048):
            nwg1 = -(-N // K.mt1)
            last = W * ns // K.mt0 - 1
            for w0, w1 in ((0, 0), (last // 2, nwg1 // 2), (last, nwg1 - 1)):
                for mask in ((1 << W) - 1, 0):
                    _check(K, _Case(K, W=W, ns=ns, N=N, r=W - 1, w0=w0, w1=w1, mask=mask))


def _at(lines, start, pattern):
    return next(i for i in range(start, len(lines)) if re.match(pattern, lines[i]))


def _cold(lines):
    i = _at(lines, 0, r"buffer_store_b\d+ .*scope:SCOPE_SYS")
    return max(k for k in range(i) if re.match(r"\w+:", lines[k]))


def _rowMultiplier(lines):
    i = _at(lines, _cold(lines), r"s_mul_i32 s\d+, s\d+, \d+\b")
    lines[i] = re.sub(r"(s_mul_i32 s\d+, s\d+, )(\d+)\b", lambda m: f"{m[1]}{int(m[2]) + 1}", lines[i])


def _windowSize(lines):
    i = _at(lines, 0, r"s_lshr_b32 s\d+, s\[sgprFusedNShard\], 6\b")
    lines[i] = lines[i].replace("], 6", "], 5", 1)


def _immediateOffset(lines):
    i = _at(lines, _cold(lines), r"buffer_store_b\d+ .*scope:SCOPE_SYS")
    lines[i] = re.sub(r"offset:(\d+)", lambda m: f"offset:{int(m[1]) + 16}", lines[i])


def _rankTimesM(lines):
    i = _at(lines, 0, r"s_mul_i32 (s\d+), \1, s\[sgprSizesFree\+1\]")
    lines[i] = lines[i].replace("s[sgprSizesFree+1]", "s[sgprSizesFree+0]")


def _noStepReset(lines):
    i = _at(lines, 0, r"s_and_b32 s\d+, s\d+, 0xff00\b")
    del lines[max(k for k in range(i) if re.match(r"s_mov_b32 s\d+, 0\b", lines[k]))]


def _unmodelledLaneOp(lines):
    i = _at(lines, _at(lines, 0, r"s_and_b32 s\d+, s\d+, 0xff00\b"), r"v_lshlrev_b32 (v\[[^]]+\]|v\d+), 1, \1(?=\s|$)")
    reg = lines[i].split()[1].rstrip(",")
    lines.insert(i + 1, f"v_bfe_u32 {reg}, {reg}, 0, 31")


@pytest.mark.parametrize("which", [0, 2])
@pytest.mark.parametrize("mutate", [_rowMultiplier, _windowSize, _immediateOffset, _rankTimesM, _noStepReset,
                                    _unmodelledLaneOp],
                         ids=["row-multiplier", "window-size", "immediate-offset", "rank-times-M",
                              "no-DwStep-reset", "unmodelled-lane-op"])
def test_fused_a2a_gfx1250_direct_store_addresses_reject(kernels, mutate, which):
    K = kernels[which]
    _quick(K)
    lines = K.src.splitlines()
    mutate(lines)
    assert lines != K.src.splitlines()
    with pytest.raises(AssertionError):
        _quick(_Kernel(K.base, "\n".join(lines)))


@pytest.fixture(scope="module")
def kernels():
    out = []
    for config in _CONFIGS:
        res = emit_kernels_from_config(config, limit=8, arch=_ARCH)
        assert res and all(err == 0 for (_b, _s, err) in res), [(b, e) for (b, _s, e) in res]
        out += [_Kernel(b, s) for (b, s, _e) in res]
    assert len(out) == 6, [k.base for k in out]
    return out
