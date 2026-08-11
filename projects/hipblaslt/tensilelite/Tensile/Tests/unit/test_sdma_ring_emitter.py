#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA ring-buffer producer emitter structural tests (NOGPU).
#
# The emitter (Tensile/Components/SdmaRingEmitter.py) is a packet-independent
# rocisa translation of MORI's anvil device ring skeleton. It IS wired into a
# live kernel (GlobalWriteBatch._emitFusedA2ASdmaIssue calls
# emitReserveQueueSpace -- grep the function name, not a line number, which
# rots), but these tests stay
# deliberately out-of-kernel: "verify" means render each method's Module to
# assembly text and assert on the SEMANTIC features that a wrong emit would
# corrupt -- instruction mnemonic + scope bits (sc0/sc1) +
# operand offsets -- NOT a whole-text snapshot (which reddens on any benign
# edit and tells you nothing when it does).
#
# The five load-bearing invariants (a wrong one is a timing-dependent hang that
# is nearly impossible to debug on hardware, so they are pinned here where a
# regression is a one-line diff):
#   1. scope bits per field: rptr read = SYSTEM (sc0 sc1); queueBuf / wptr /
#      cachedWptr / committedWptr access = AGENT (sc1); doorbell write = SYSTEM.
#   2. submit publish order: wptr store -> s_waitcnt vmcnt(0) -> doorbell store
#      -> committedWptr store, with BOTH vmcnt(0) barriers present, asserted by
#      relative text position (not mere presence).
#   3. reserve uses CAS (compare-swap), never an atomic_add fetch-and-increment.
#   4. the wrap branch emits a zero-store (ring-tail NOP padding).
#   5. cachedHwReadIndex (handle+48) is NEVER stored back (private cache, not
#      shared state).
#
# Bonus: the rendered .s is wrapped in a minimal kernel and run through the
# assembler to catch "reads right but is not legal opcode/operand" errors.
################################################################################

import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# rocisa is imported transitively; skip cleanly if the C++ module is not built.
import rocisa                                                   # noqa: E402

from rocisa import rocIsa                                        # noqa: E402
from rocisa.register import RegisterPool                         # noqa: E402
from rocisa.enum import RegisterType                             # noqa: E402
from rocisa.label import LabelManager                            # noqa: E402
from rocisa.code import Module                                   # noqa: E402
from Tensile.Common.Architectures import gfxToIsa                # noqa: E402
from Tensile.Components.SdmaRingEmitter import (                 # noqa: E402
    SdmaRingEmitter, SDMA_QUEUE_SIZE, OFF_cachedHwReadIndex,
)

# This emitter targets gfx950 (CDNA4); the scope-bit assertions below are the
# SC[1:0] encoding (sc0/sc1), not gfx1250 scope:/th: syntax. Init the singleton
# for gfx950 regardless of the host GPU so the asmCaps (which pick sc0/sc1 vs
# glc/slc) are the ones this code was written against.
_GFX = "gfx950"
_PKT_DWORDS = 10  # arbitrary packet size for placePacket exercise (the real emitter sets the actual size)


def _init_gfx950():
    ri = rocIsa.getInstance()
    isa = gfxToIsa(_GFX)
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)
    return ri


def _mock_writer():
    """Minimal writer context the emitter needs: register pools + a label
    manager. Mirrors the GL2Prefetch mock writer -- the emitter owns no state,
    the caller owns the pools."""
    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.labels = LabelManager()
    w.vgprPool.checkOut(1)   # v0 reserved (Serial)
    w.sgprPool.checkOut(8)   # reserve s0..s7 (kernarg ptr etc.)
    return w


def _persistent_sgprs(w):
    """Allocate the caller-owned persistent SGPRs the emitter reads: handleBase
    (2), cachedHwReadIdx (2), plus scratch outputs. Returns a namespace."""
    ns = SimpleNamespace()
    ns.handleBase = w.sgprPool.checkOutAligned(2, 2, "handleBase", preventOverflow=False)
    ns.cachedHwIdx = w.sgprPool.checkOutAligned(2, 2, "cachedHwIdx", preventOverflow=False)
    ns.cur = w.sgprPool.checkOutAligned(2, 2, "cur", preventOverflow=False)
    ns.off = w.sgprPool.checkOut(1, "padoff", preventOverflow=False)
    ns.base = w.sgprPool.checkOutAligned(2, 2, "base", preventOverflow=False)
    ns.pend = w.sgprPool.checkOutAligned(2, 2, "pend", preventOverflow=False)
    return ns


def _render(method):
    """Render one emitter method to assembly text.  `method` is a callable
    (module, writer, emitter, persistent-sgprs) -> None."""
    _init_gfx950()
    w = _mock_writer()
    em = SdmaRingEmitter()
    ns = _persistent_sgprs(w)
    m = Module("test")
    method(m, w, em, ns)
    return str(m)


# ---- rendered text of each method (module-scoped so it renders once) --------

def _render_reserve():
    return _render(lambda m, w, em, ns:
                   em.emitReserveQueueSpace(m, w, ns.handleBase, ns.cachedHwIdx, 40, ns.cur, ns.off))


def _render_canwrite():
    return _render(lambda m, w, em, ns:
                   em.emitCanWriteUpto(m, w, ns.handleBase, ns.cachedHwIdx, ns.cur, ns.off,
                                       w.sgprPool.checkOutAligned(2, 2, "t", preventOverflow=False)))


def _render_place():
    def go(m, w, em, ns):
        pkt = w.vgprPool.checkOut(_PKT_DWORDS, "pkt")
        em.emitPlacePacket(m, w, ns.handleBase, pkt, _PKT_DWORDS, ns.pend, ns.off)
    return _render(go)


def _render_submit():
    return _render(lambda m, w, em, ns:
                   em.emitSubmitPacket(m, w, ns.handleBase, ns.base, ns.pend))


# A line-level helper: return the list of (idx, line) for lines whose text
# matches a mnemonic substring, so tests assert on order by index.
def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _first_idx(lines, pred):
    for i, ln in enumerate(lines):
        if pred(ln):
            return i
    return -1


# ---------------------------------------------------------------------------
# Invariant 1: scope bits per field
# ---------------------------------------------------------------------------
class TestScopeBits:

    def test_rptr_read_is_system_scope(self):
        # CanWriteUpto's slow path loads the hardware rptr at SYSTEM scope.
        lines = _lines(_render_canwrite())
        rptr = [ln for ln in lines if "load hardware rptr" in ln]
        assert rptr, "expected a rptr load in CanWriteUpto"
        assert all("global_load" in ln and " sc0 sc1" in ln for ln in rptr), \
            f"rptr load must be SYSTEM scope (sc0 sc1): {rptr}"

    def test_cachedwptr_access_is_agent_scope(self):
        # reserve reads cachedWptr (AGENT, sc1) and CAS-writes it (device, sc0).
        lines = _lines(_render_reserve())
        load = [ln for ln in lines if "load cachedWptr" in ln]
        assert load and all("global_load" in ln and _has_sc1_only(ln) for ln in load), \
            f"cachedWptr load must be AGENT scope (sc1, not sc0): {load}"

    def test_committedwptr_access_is_agent_scope(self):
        lines = _lines(_render_submit())
        acc = [ln for ln in lines if "committedWptr" in ln and ("global_load" in ln or "global_store" in ln)]
        assert acc, "expected committedWptr load+store in submitPacket"
        assert all(_has_sc1_only(ln) for ln in acc), \
            f"committedWptr access must be AGENT scope (sc1 only): {acc}"

    def test_wptr_store_is_agent_scope(self):
        lines = _lines(_render_submit())
        st = [ln for ln in lines if "store wptr" in ln]
        assert st and all("global_store" in ln and _has_sc1_only(ln) for ln in st), \
            f"wptr store must be AGENT scope (sc1 only): {st}"

    def test_doorbell_store_is_system_scope(self):
        lines = _lines(_render_submit())
        db = [ln for ln in lines if "doorbell" in ln and "global_store" in ln]
        assert db and all(" sc0 sc1" in ln for ln in db), \
            f"doorbell store must be SYSTEM scope (sc0 sc1): {db}"

    def test_ring_stores_are_agent_scope(self):
        # placePacket writes padding + packet dwords to the ring at AGENT scope.
        lines = _lines(_render_place())
        st = [ln for ln in lines if "global_store" in ln and "ring[" in ln]
        assert st, "expected ring stores in placePacket"
        assert all(_has_sc1_only(ln) for ln in st), \
            f"ring stores must be AGENT scope (sc1 only): {st}"


def _has_sc1_only(line):
    """True iff the instruction carries sc1 but NOT sc0 (AGENT/device scope).
    Matches the trailing modifier group so a 'sc1' inside a comment is ignored."""
    code = line.split("//")[0]
    return re.search(r"\bsc1\b", code) is not None and re.search(r"\bsc0\b", code) is None


# ---------------------------------------------------------------------------
# Invariant 2: submit publish order
# ---------------------------------------------------------------------------
class TestSubmitOrder:

    def test_publish_sequence_order(self):
        lines = _lines(_render_submit())
        i_wptr = _first_idx(lines, lambda l: "store wptr = pending" in l and "global_store" in l)
        i_db = _first_idx(lines, lambda l: "doorbell" in l and "global_store" in l)
        i_comm = _first_idx(lines, lambda l: "store committedWptr = pending" in l and "global_store" in l)
        assert -1 not in (i_wptr, i_db, i_comm), \
            f"missing a publish store: wptr={i_wptr} db={i_db} comm={i_comm}"
        assert i_wptr < i_db < i_comm, \
            f"publish order must be wptr({i_wptr}) < doorbell({i_db}) < committed({i_comm})"

    def test_vmcnt_barrier_between_wptr_and_doorbell(self):
        # An s_waitcnt vmcnt(0) sits between the wptr store and the doorbell
        # store (orders the wptr write ahead of the ring).
        lines = _lines(_render_submit())
        i_wptr = _first_idx(lines, lambda l: "store wptr = pending" in l and "global_store" in l)
        i_db = _first_idx(lines, lambda l: "doorbell" in l and "global_store" in l)
        between = [ln for ln in lines[i_wptr + 1:i_db] if "s_waitcnt vmcnt(0)" in ln]
        assert between, "expected s_waitcnt vmcnt(0) between wptr store and doorbell store"

    def test_vmcnt_barrier_between_doorbell_and_committed(self):
        # MORI anvil_device.hpp:226: an s_waitcnt vmcnt(0)
        # sits between the doorbell store and the committedWptr store, so the
        # doorbell (engine kick) is ordered before committedWptr unblocks the next
        # producer. This is on the plan's timing-hang critical path, so it is
        # pinned here -- a refactor that drops it must redden a test.
        lines = _lines(_render_submit())
        i_db = _first_idx(lines, lambda l: "doorbell" in l and "global_store" in l)
        i_comm = _first_idx(lines, lambda l: "store committedWptr = pending" in l and "global_store" in l)
        assert -1 not in (i_db, i_comm), f"missing a publish store: db={i_db} comm={i_comm}"
        between = [ln for ln in lines[i_db + 1:i_comm] if "s_waitcnt vmcnt(0)" in ln]
        assert between, "expected s_waitcnt vmcnt(0) between doorbell store and committedWptr store"

    def test_two_vmcnt_barriers_present(self):
        # One vmcnt(0) orders the packet stores before wptr; one orders wptr
        # before the doorbell. Both must be present.
        lines = _lines(_render_submit())
        i_wptr = _first_idx(lines, lambda l: "store wptr = pending" in l and "global_store" in l)
        i_db = _first_idx(lines, lambda l: "doorbell" in l and "global_store" in l)
        pre_wptr = [ln for ln in lines[:i_wptr] if "s_waitcnt vmcnt(0)" in ln]
        mid = [ln for ln in lines[i_wptr + 1:i_db] if "s_waitcnt vmcnt(0)" in ln]
        assert pre_wptr, "expected a vmcnt(0) before the wptr store (packet stores visible)"
        assert mid, "expected a vmcnt(0) between wptr and doorbell"

    def test_spins_on_committed_equals_base(self):
        # (1) submitPacket serializes behind earlier reservations by spinning
        # until committedWptr == base (a loop with a back-branch).
        text = _render_submit()
        assert "committedWptr == base" in text, "expected the committed==base spin comment"
        lines = _lines(text)
        assert any("s_cbranch_scc0" in ln and "spin" in ln for ln in lines), \
            "expected a back-branch that re-polls committedWptr"

    def test_spin_backoff_is_inside_the_loop_body(self):
        # Regression guard: the s_sleep backoff must sit INSIDE the spin cycle,
        # i.e. between the spin label and the FIRST back-branch. If it drifts
        # after the back-branches it runs exactly once, on loop exit, and the
        # committedWptr poll becomes a tight zero-backoff spin.
        lines = _lines(_render_submit())
        def _code(l): return l.split("//")[0]
        i_label = _first_idx(lines, lambda l: _code(l).strip().startswith("label_sdma_submit_spin:"))
        i_sleep = _first_idx(lines, lambda l: _code(l).strip().startswith("s_sleep"))
        backs = [i for i, l in enumerate(lines)
                 if "s_cbranch_scc0" in _code(l) and "label_sdma_submit_spin" in _code(l)]
        n_sleep = sum(1 for l in lines if _code(l).strip().startswith("s_sleep"))
        assert i_label != -1, "spin label not emitted"
        assert i_sleep != -1, "no s_sleep backoff emitted in submitPacket"
        assert len(backs) >= 1, "no back-branch to the spin label"
        assert n_sleep == 1, (
            "expected exactly one s_sleep in submitPacket, found %d "
            "(a duplicate left on the exit path would slip past the positional check)" % n_sleep)
        assert i_label < i_sleep < min(backs), (
            "s_sleep must precede EVERY back-branch: label=%d sleep=%d first_back_branch=%d"
            % (i_label, i_sleep, min(backs)))


# ---------------------------------------------------------------------------
# Invariant 3: reserve uses CAS, not fetch_add
# ---------------------------------------------------------------------------
class TestReserveIsCas:

    def test_uses_compare_swap(self):
        lines = _lines(_render_reserve())
        cas = [ln for ln in lines if "atomic_cmpswap" in ln.split("//")[0]]
        assert cas, "ReserveQueueSpace must use a compare-swap atomic"

    def test_cas_is_x2_with_sc0_on_gfx950(self):
        # The reserve CAS must be the 64-bit global compare-swap with the gfx9
        # "_x2" mnemonic and sc0 (return-of-pre-op; the assembler REQUIRES sc0 on
        # this op). This is the load-bearing detail: a wrong mnemonic/suffix or a
        # missing sc0 is a silent illegal-instruction or a dropped-return bug.
        code_lines = [ln for ln in _lines(_render_reserve())
                      if "atomic_cmpswap" in ln.split("//")[0]]
        assert code_lines, "expected a global CAS in ReserveQueueSpace"
        for ln in code_lines:
            code = ln.split("//")[0]
            assert "global_atomic_cmpswap_x2" in code, \
                f"CAS must be global_atomic_cmpswap_x2 on gfx950 (not _b64/_dwordx2/buffer_): {ln}"
            assert re.search(r"\bsc0\b", code), \
                f"CAS must carry sc0 (return-of-pre-op; assembler requires it): {ln}"

    def test_no_atomic_add_reservation(self):
        # A fetch_add reservation (the WRONG primitive) would show an atomic_add
        # writing the reservation cursor; assert none exists.
        code = "\n".join(ln.split("//")[0] for ln in _lines(_render_reserve()))
        assert "atomic_add" not in code, \
            "ReserveQueueSpace must NOT use atomic_add (fetch_add breaks wrap padding)"

    def test_cas_retry_backbranch(self):
        # A CAS reserve retries on lost race: a back-branch to the loop head.
        lines = _lines(_render_reserve())
        assert any("s_cbranch_scc0" in ln and "retry" in ln for ln in lines), \
            "expected a CAS lost-race retry branch"


# ---------------------------------------------------------------------------
# Invariant 4: wrap branch emits zero-padding stores
# ---------------------------------------------------------------------------
class TestWrapPadding:

    def test_padding_zero_store(self):
        lines = _lines(_render_place())
        pad = [ln for ln in lines if "global_store" in ln and "padding NOP" in ln]
        assert pad, "placePacket must emit a zero-store for ring-tail padding"

    def test_padding_value_is_zero(self):
        # The padded value comes from a register set to 0.
        text = _render_place()
        assert "padding NOP value = 0" in text, "expected a v_mov ...,0 seeding the padding value"

    def test_reserve_computes_wrap_padding(self):
        # ReserveQueueSpace computes offset = queueSize - WrapIntoRing(cur) on wrap.
        text = _render_reserve()
        assert "pad ring tail" in text, "reserve must compute the wrap-padding offset"


# ---------------------------------------------------------------------------
# Invariant 5: cachedHwReadIndex (handle+48) is never stored back
# ---------------------------------------------------------------------------
class TestCachedHwReadIndexNeverStored:

    def test_no_store_to_handle_plus_48(self):
        # The private cache seed lives at handle+48. No emitter method may store
        # to it (it is per-producer register state, not shared memory).
        assert OFF_cachedHwReadIndex == 48
        for text in (_render_reserve(), _render_canwrite(), _render_place(), _render_submit()):
            code = "\n".join(ln.split("//")[0] for ln in _lines(text))
            # A store to handle+48 would appear as an s_load/global access at
            # offset 0x30 off handleBase; assert no field pointer at 0x30 is
            # loaded (the emitter only touches offsets 0/8/16/24/32/40).
            assert "0x30" not in code, \
                "no access to handle+48 (cachedHwReadIndex) may be emitted"

    def test_only_expected_field_offsets_accessed(self):
        # Whitelist: queueBuf(0) rptr(8) wptr(16) doorbell(24) cachedWptr(32)
        # committedWptr(40). Any other handleBase offset is a layout mistake.
        allowed = {"0x0", "0x8", "0x10", "0x18", "0x20", "0x28"}
        for text in (_render_reserve(), _render_canwrite(), _render_place(), _render_submit()):
            for ln in _lines(text):
                m = re.search(r"s_load_dwordx2 .*, s\[\d+:\d+\], (0x[0-9a-fA-F]+)", ln)
                if m:
                    assert m.group(1) in allowed, f"unexpected handle field offset {m.group(1)}: {ln}"


# ---------------------------------------------------------------------------
# Invariant 6: CanWriteUpto leaves resultS defined on EVERY path
# ---------------------------------------------------------------------------
class TestCanWriteUptoResultAlwaysDefined:

    @staticmethod
    def _result_reg(lines):
        i = _first_idx(lines, lambda l: "CanWriteUpto = true (room)" in l)
        assert i != -1, "expected the 'CanWriteUpto = true (room)' write"
        m = re.match(r"s_mov_b32 (s\d+), 1$", lines[i].split("//")[0].strip())
        assert m, f"could not identify resultS from: {lines[i]}"
        return m.group(1)

    def test_result_defaulted_to_zero_before_first_branch(self):
        # The refresh-retest tail branch ("hi != 0 -> full") jumps straight to
        # the done label WITHOUT writing resultS. So resultS must be defaulted
        # to 0 before the first branch, or the caller reads a stale value.
        lines = _lines(_render_canwrite())
        reg = self._result_reg(lines)
        i_branch = _first_idx(lines, lambda l: "s_cbranch" in l.split("//")[0])
        assert i_branch != -1, "expected a branch in CanWriteUpto"
        init = [ln for ln in lines[:i_branch]
                if re.match(r"s_mov_b32 %s, 0$" % reg, ln.split("//")[0].strip())]
        assert init, (
            "CanWriteUpto must emit 's_mov_b32 %s, 0' before its first branch; "
            "the 'hi != 0 -> full' branch skips both resultS writes.\nprologue:\n%s"
            % (reg, "\n".join(lines[:i_branch + 1])))


# ---------------------------------------------------------------------------
# Bonus: assemble the rendered text (catches illegal opcodes/operands)
# ---------------------------------------------------------------------------
def _assembler():
    return shutil.which("amdclang++") or (
        "/opt/rocm/bin/amdclang++" if os.path.exists("/opt/rocm/bin/amdclang++") else None)


@pytest.mark.skipif(_assembler() is None, reason="no amdclang++ to assemble with")
class TestAssembles:

    @pytest.mark.parametrize("render", [_render_reserve, _render_canwrite, _render_place, _render_submit],
                             ids=["reserve", "canwrite", "place", "submit"])
    def test_body_assembles(self, render, tmp_path):
        body = render()
        # Wrap the body in a minimal gfx950 kernel. The body references v0..vN
        # and s0..sN directly (numeric), so no .set directives are needed; the
        # assembler only checks opcode/operand legality, which is the point.
        asm = _MINIMAL_KERNEL % {"gfx": _GFX, "body": body}
        s_path = tmp_path / "sdma_ring.s"
        o_path = tmp_path / "sdma_ring.o"
        s_path.write_text(asm)
        r = subprocess.run(
            [_assembler(), "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=%s" % _GFX, "-mcode-object-version=5", "-c", str(s_path), "-o", str(o_path)],
            capture_output=True, text=True)
        assert r.returncode == 0, f"assembly failed:\n{r.stderr}\n---\n{asm}"


# Minimal assemblable kernel: just the body between a label and s_endpgm. The
# body's registers are self-contained (numeric v/s indices); we bump the
# declared vgpr/sgpr counts high enough to cover them.
_MINIMAL_KERNEL = """\
.amdgcn_target "amdgcn-amd-amdhsa--%(gfx)s"
.text
.globl test_kernel
.p2align 8
.type test_kernel,@function
test_kernel:
%(body)s
  s_endpgm
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr 100
  .amdhsa_accum_offset 64
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
.end_amdhsa_kernel
"""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
