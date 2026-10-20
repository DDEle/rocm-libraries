# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A2A-GEMM (FusedA2A: A2AGemm) outer shard-loop codegen characterization.

The index structure is the stock GEMM one: K is a single summation index
holding the whole W*k_local reduction. The shard loop wraps the unroll loop,
which covers K/W per round; the shard count is a runtime scalar, not a problem
dimension.
"""

import copy
import functools
import re

import pytest

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = "tensilelite/Tests/common/comm/gfx950/a2a_gemm_loop.yaml"


@functools.lru_cache(maxsize=None)
def _src():
    _base, src, err = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0]
    assert err == 0
    return src


@functools.lru_cache(maxsize=None)
def _derived_state():
    return derive_states(_CONFIG, arch="gfx950", limit_solutions=1)[0]


def _state():
    return copy.deepcopy(_derived_state())


def _placed_floor(body):
    """The SGPR holding the first block of the batch an enqueuer places."""
    m = re.search(r"^s_mov_b32 s(\d+), s\[sgprA2ABlockLo\][^\n]*// this batch$", body, re.M)
    assert m, "the placed batch is not seeded from this batch"
    return m.group(1)


def _packing_pass():
    """One enqueuer's pass: the election atomic through the skip label."""
    src = _src()
    head = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
    tail = re.search(r"^label_A2ASkipEnqueue:", src, re.M)
    assert head and tail, "no a2a packing pass in the emitted kernel"
    return src[head.end() : tail.start()]


def _group_loop():
    body = _packing_pass()
    head = re.search(r"^label_a2a_group_loop\w*:", body, re.M)
    tail = re.search(r"^s_cbranch_scc1 label_a2a_group_loop\w*", body, re.M)
    assert head and tail, "no group loop in the packing pass"
    return body[head.end() : tail.start()]


def _prologue():
    return _src()[: _src().index("label_A2AShardLoopBegin:")]


def _transition():
    src = _src()
    return src[src.index("A2A_TRANSITION begin") : src.index("A2A_TRANSITION end")]


class TestA2AGemmBatchNumbering:
    """The prologue's batch span: the CU count it divides by."""

    def _fused_w_offset(self, src):
        m = re.search(
            r"s_load_dword s\[sgprA2AShardCounter\], "
            r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
            src,
        )
        assert m, "no FusedW kernarg load in the emitted kernel"
        return int(m.group(1), 16)

    def test_cu_count_is_read_from_the_kernarg(self):
        from Tensile.Components.Signature import fusedA2AKernArgLayout

        src = _src()
        layout = fusedA2AKernArgLayout()
        want = self._fused_w_offset(src) + layout["FusedNumCu"] - layout["FusedW"]
        loaded = [
            int(h, 16)
            for h in re.findall(
                r"s_load_dword s(?:\[sgpr\w+\]|\d+), "
                r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
                src,
            )
        ]
        assert want in loaded, "FusedNumCu at %#x is never loaded" % want


class TestA2AGemmElection:
    """The batch's enqueuer work-group: the gates, the SMEM atomic, its target, the LDS broadcast."""

    # log2(WavefrontSize) of the config.
    _WAVE_SHIFT = 6

    def _atomic(self, src):
        m = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
        assert m, "no s_atomic_inc in the emitted kernel"
        return m

    def _ticket(self, src):
        """The SGPR the election atomic returns its pre-op value in."""
        return self._atomic(src).group(0).split(",")[0].split()[-1]

    def _wave(self, src):
        """The SGPR holding the wave index."""
        m = re.search(
            r"v_readfirstlane_b32 s(\d+), v\[vgprSerial\][^\n]*\n"
            r"s_lshr_b32 s\1, s\1, %d[^\n]*// wave index" % self._WAVE_SHIFT,
            src[: self._atomic(src).start()],
        )
        assert m, "no wave index before the election atomic"
        return "s" + m.group(1)

    def test_only_feature_tile_0_draws(self):
        src = _src()
        assert re.search(
            r"s_cmp_lg_u32 s\[sgprWorkGroup0\], 0[^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue",
            src[: self._atomic(src).start()],
        ), "work-groups other than feature tile 0 still draw a ticket"

    def test_only_wave_0_draws(self):
        src = _src()
        assert re.search(
            r"s_cmp_lg_u32 %s, 0[^\n]*\n"
            r"s_cbranch_scc1 label_a2a_ticket_drawn\w*[^\n]*\n"
            r"s_atomic_inc " % re.escape(self._wave(src)),
            src,
        ), "a wave other than wave 0 reaches the election atomic"

    def test_election_target_is_the_batch_block_count(self):
        src = _src()
        m = re.search(r"s_sub_u32 s(\d+), s\[sgprA2ABlockCount\], 1[^\n]*// DATA", src)
        assert m, "the election DATA is not count - 1"
        assert self._atomic(src).group(0).startswith("s_atomic_inc s%s," % m.group(1))

    def test_ticket_0_wins_through_an_lds_broadcast(self):
        src = _src()
        tick = re.escape(self._ticket(src))
        assert re.search(
            r"v_mov_b32 v(\d+), %s[^\n]*\n"
            r"ds_write_b32 v\d+, v\1[^\n]*\n"
            r"s_waitcnt lgkmcnt\(0\)[^\n]*\n"
            r"label_a2a_ticket_drawn\w*:[^\n]*\n"
            r"s_barrier[^\n]*\n"
            r"v_mov_b32 v(\d+), 0[^\n]*\n"
            r"ds_read_b32 v(\d+), v\2[^\n]*\n"
            r"s_waitcnt lgkmcnt\(0\)[^\n]*\n"
            r"v_readfirstlane_b32 %s, v\3[^\n]*\n"
            r"s_barrier[^\n]*\n"
            r"s_cmp_lg_u32 %s, 0[^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue" % (tick, tick, tick),
            src[self._atomic(src).end() :],
        ), "the work-group does not take wave 0's ticket from LDS and enqueue only on ticket 0"

    def test_the_election_slot_reads_the_batch_index_the_divide_produced(self):
        # These read their first operand rather than writing it.
        readers = ("s_cmp", "v_cmp", "s_cbranch", "s_branch", "s_waitcnt", "s_nop", "s_setpc")

        lines = _src().splitlines()
        last_div = max(i for i, l in enumerate(lines) if "// iB = b * F / numCu" in l)
        quotient = next(
            (i, m)
            for i, m in (
                (i, re.match(r"\s*v_readfirstlane_b32 (s\d+),", lines[i]))
                for i in range(last_div + 1, len(lines))
            )
            if m
        )
        iB = quotient[1].group(1)

        slot = next(i for i, l in enumerate(lines) if "// iB * 4" in l)
        assert re.match(r"\s*s_lshl_b32 s\d+, %s, 2\b" % iB, lines[slot]), (
            "the election slot does not shift the divide's quotient register %s:\n%s"
            % (iB, lines[slot])
        )

        clobbers = [
            l.strip()
            for l in lines[quotient[0] + 1 : slot]
            if not l.strip().startswith(readers)
            and re.match(r"\s*\w+ %s\s*," % iB, l)
        ]
        assert not clobbers, (
            "%s carries the batch index into the election, but these write it first:\n%s"
            % (iB, "\n".join(clobbers))
        )


class TestA2AGemmEnqueueLoops:
    """One enqueuer's packing pass: one reservation and one submit per batch placed."""

    def test_room_check_cache_is_seeded_before_the_reserve_loop(self):
        body = _packing_pass()
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the packing pass"
        assert "cachedHwReadIndex = hardware rptr" in body[: rsv.start()]

    def test_hardware_wptr_and_rptr_reads_share_one_wait(self):
        body = _packing_pass()
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the packing pass"
        assert re.search(
            r"^s_load_dwordx2 [^\n]*glc[^\n]*\n"
            r"(?:(?!s_waitcnt)[^\n]*\n)*?"
            r"s_load_dwordx2 [^\n]*glc",
            body[: rsv.start()],
            re.M,
        ), "a s_waitcnt separates the hardware wptr and rptr reads"

    def test_reservation_size_is_computed_per_queue(self):
        body = _packing_pass()
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the packing pass"
        assert "// reservation = count" in body[: rsv.start()], (
            "the reservation size is computed outside the packing pass"
        )

    def test_reservation_counts_one_atomic_per_group(self):
        from Tensile.Components.SdmaPacketEmitter import (ATOMIC_PACKET_DWORDS,
                                                          COPY_PACKET_DWORDS)

        body = _packing_pass()
        m = re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_min_u32 s\1, s\1, s\[sgprA2ABlockCount\][^\n]*\n"
            r"s_mul_i32 s\1, s\1, %d[^\n]*\n"
            r"s_mul_i32 s(\d+), s\[sgprA2ABlockCount\], %d[^\n]*\n"
            r"s_add_u32 s\1, s\1, s\2"
            % (ATOMIC_PACKET_DWORDS * 4, COPY_PACKET_DWORDS * 4),
            body,
        )
        assert m, "the reservation is not count COPYs + min(W-1, count) ATOMICs"

    def test_the_group_loop_never_waits(self):
        body = _packing_pass()
        head = re.search(r"^label_a2a_group_loop\w*:", body, re.M)
        tail = re.search(r"^s_cbranch_scc1 label_a2a_group_loop\w*", body, re.M)
        assert head and tail, "no group loop in the packing pass"
        loop = body[head.end() : tail.start()]
        gate = re.search(r"^s_cbranch_scc0 (label_a2a_early_doorbell_skip\w*)", loop, re.M)
        assert gate, "no first-group gate in the group loop"
        loop = loop[: gate.start()] + loop[loop.index(gate.group(1) + ":") :]
        assert "s_waitcnt" not in loop, "the packing loop waits between ring stores"

    def test_the_submit_waits_for_the_ring_stores(self):
        body = _packing_pass()
        tail = re.search(r"^s_cbranch_scc1 label_a2a_group_loop\w*[^\n]*\n", body, re.M)
        assert tail, "no group loop in the packing pass"
        assert body[tail.end() :].startswith("s_waitcnt vmcnt(0)")


class TestA2AGemmRoundGroupOrder:
    """Each queue emits its blocks in stride-(W-1) groups, one group per round.

    Round r reads block b from queue (b + r - 1) mod (W-1).
    """

    def test_lanes_below_the_batch_floor_are_masked(self):
        body = _packing_pass()
        assert re.search(
            r"v_cmp_le_i32 vcc, s%s, v(\d+)[^\n]*\n"
            r"s_mov_b64 exec, vcc" % _placed_floor(body),
            body,
        ), "lanes past the batch's first block still store a COPY"

    def test_the_atomic_rides_the_groups_last_pass(self):
        body = _packing_pass()
        head = re.search(r"^label_a2a_copy_loop\w*:", body, re.M)
        tail = re.search(r"^s_cbranch_scc1 label_a2a_copy_loop\w*", body, re.M)
        nxt = re.search(r"^label_a2a_group_next\w*:", body, re.M)
        assert head and tail and nxt, "no copy loop / group-next in the packing pass"
        loop = body[head.end() : tail.start()]
        m = re.search(
            r"s_cmp_eq_u32 s\d+, 0[^\n]*// the group's last pass\?\n"
            r"s_cselect_b32 s(\d+), 0x80000000, 0[^\n]*\n"
            r"s_or_b32 exec_hi, exec_hi, s\1[^\n]*\n"
            r"(?:(?!global_store)[^\n]*\n)*"
            r"global_store_dwordx4 [^\n]*\n"
            r"global_store_dwordx4 [^\n]*\n"
            r"s_andn2_b32 exec_hi, exec_hi, s\1",
            loop,
        )
        assert m, "the ATOMIC's eight dwords are not stored by the last lane on the last pass"
        assert len(re.findall(r"^global_store_dword\w* [^\n]*$", loop, re.M)) == 4, (
            "the copy loop stores more than one packet per lane"
        )
        assert "global_store" not in body[tail.end() : nxt.start()], (
            "the group places packets outside its copy loop"
        )

    def test_ring_stores_bypass_the_caches(self):
        stores = re.findall(r"^global_store_dword\w* [^\n]*$", _packing_pass(), re.M)
        assert stores, "no vector ring stores in the packing pass"
        assert all(" sc0 sc1" in s for s in stores), "a ring store is cached"

    def test_packet_registers_are_not_rewritten_under_a_store(self):
        body = _packing_pass()
        for m in re.finditer(r"^global_store_dwordx4 [^\n]*$", body, re.M):
            nxt = re.search(r"^(v_\w+|s_nop 1)", body[m.end() :], re.M)
            assert nxt and nxt.group(1) == "s_nop 1", "no wait states after the ring stores"


class TestA2AGemmEarlyDoorbell:
    """The queue's first group is published before the rest are placed."""

    def _submit(self, body):
        return body[body.rindex("load wptr / doorbell pointers") :]

    def test_the_group_loop_rings_the_doorbell(self):
        loop = _group_loop()
        assert loop.count("ring doorbell = pending") == 1

    def test_the_early_doorbell_follows_the_groups_stores(self):
        loop = _group_loop()
        ring = loop.index("ring doorbell = pending")
        wait = loop.rindex("s_waitcnt vmcnt(0)", 0, ring)
        assert loop.rindex("global_store_dwordx4", 0, ring) < wait

    def test_the_early_doorbell_is_taken_on_the_first_group_only(self):
        body = _packing_pass()
        m = re.search(r"s_mov_b32 s(\d+), s\[sgprA2ABlockCount\][^\n]*// blocks left", body)
        assert m, "no block counter seeded from A2ABlockCount"
        loop = _group_loop()
        g = re.search(
            r"s_cmp_eq_u32 s%s, s\[sgprA2ABlockCount\][^\n]*\n"
            r"s_cbranch_scc0 (label_\w+)" % m.group(1),
            loop,
        )
        assert g, "the early doorbell is not gated on the queue's first group"
        assert loop.index(g.group(0)) < loop.index("ring doorbell = pending")
        assert loop.index("ring doorbell = pending") < loop.index(g.group(1) + ":")

    def test_the_early_doorbell_waits_for_its_turn(self):
        loop = _group_loop()
        early = loop[: loop.index("ring doorbell = pending")]
        m = re.search(
            r"s_load_dwordx2 s\[(\d+):(\d+)\], s\[sgprA2ACounterPtr:sgprA2ACounterPtr\+1\], "
            r"s\d+ offset:8 glc[^\n]*\n"
            r"(?:[^\n]*\n)*?"
            r"s_sub_u32 s\1, s\1, s\d+[^\n]*\n"
            r"s_subb_u32 s\2, s\2, s\d+[^\n]*\n"
            r"s_cbranch_scc1 (label_\w+)",
            early,
        )
        assert m, "the early doorbell does not check committedWptr >= base"
        assert loop.index(m.group(3) + ":") > loop.index("ring doorbell = pending")

    def test_the_early_doorbell_stores_wptr_first(self):
        loop = _group_loop()
        assert re.search(
            r"s_store_dwordx2 [^\n]*// store wptr = pending\n"
            r"s_waitcnt lgkmcnt\(0\)[^\n]*\n"
            r"s_store_dwordx2 [^\n]*// ring doorbell = pending",
            loop,
        ), "the early doorbell is not preceded by an ordered wptr store"

    def test_the_early_doorbell_leaves_the_commit_cursor(self):
        loop = _group_loop()
        assert "store committedWptr" not in loop

    def test_the_submit_takes_its_turn_at_or_past_the_base(self):
        submit = self._submit(_packing_pass())
        assert re.search(
            r"s_load_dwordx2 s\[(\d+):(\d+)\][^\n]*// poll committedWptr\n"
            r"s_waitcnt lgkmcnt\(0\)[^\n]*\n"
            r"s_sub_u32 s\1, s\1, s\d+[^\n]*\n"
            r"s_subb_u32 s\2, s\2, s\d+[^\n]*\n"
            r"s_cbranch_scc1 label_sdma_submit_spin",
            submit,
        ), "the submit spin waits for committedWptr == base exactly"


class TestA2AGemmPacketBody:
    """The COPY_SUBWIN per token block and the ATOMIC per group the enqueuer places."""

    # MacroTile1 of the config: one token block's row count, and rect_y once the
    # tail block is behind us.
    _MT_TOKEN = 256

    def test_lane_invariant_dwords_are_set_before_the_group_loop(self):
        body = _packing_pass()
        loop = _group_loop()
        pre = body[: body.index(loop)]
        for header in ("SUBWIN DW0", "ATOMIC DW0"):
            assert header in pre, "%s is not set before the group loop" % header
            assert header not in loop, "%s is rebuilt per group" % header

    def test_each_lane_clamps_its_row_count(self):
        assert re.search(
            r"v_sub_u32 v(\d+), s\[sgprSizeJ\], v(\d+)[^\n]*\n"
            r"v_min_i32 v\1, %d, v\1" % self._MT_TOKEN,
            _group_loop(),
        ), "rect_y is not min(MT_token, tokens left) per lane"


class TestA2AGemmSubRoundsName:
    def _names(self, subRounds):
        from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin

        state = _state()
        state["A2ASubRounds"] = subRounds
        return getKernelNameMin(state, False), getSolutionNameMin(state, False)

    def test_sub_rounds_split_the_names(self):
        names = [self._names(s) for s in (1, 2, 4)]
        for which in (0, 1):
            assert len({n[which] for n in names}) == 3

    def test_single_sub_round_names_carry_sr1(self):
        assert all("_SR1_" in n for n in self._names(1))


class TestA2AGemmPredicates:
    """The runtime shard guard rides on the solution's compound predicate list."""

    @pytest.mark.parametrize("subRounds, shape", [(1, 64), (2, 128)])
    def test_shape_carries_depth_u_times_sub_rounds(self, subRounds, shape):
        from Tensile.Contractions import ProblemPredicate, ProblemType

        state = _state()
        state["A2ASubRounds"] = subRounds
        pt = ProblemType.FromOriginalState(state["ProblemType"])
        predicates = {p.tag: p.value for p in ProblemPredicate.CompoundPredicates(state, pt)}
        assert predicates["A2AGemmShape"] == shape


class TestA2AGemmPerBlockMisphase:
    """Token block b walks the peer queues from b % (W-1), rotating one per round.

    A2AShardIdx holds the gathered segment of the round about to be entered;
    the queue it names is that segment minus one.
    """

    def test_round_zero_seeds_the_segment_from_the_token_block(self):
        assert re.search(
            r"v_sub_u32 v(\d+), s\[sgprWorkGroup1\], v\d+[^\n]*\n"
            r"(?:s_nop[^\n]*\n)?"
            r"v_readfirstlane_b32 s\[sgprA2AShardIdx\], v\1[^\n]*\n"
            r"s_add_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardIdx\], 1",
            _prologue(),
        ), "every token block opens on the same peer queue"


class TestA2AGemmFlagSpin:
    """The consumer's wait on the local arrival flag, one per shard boundary."""

    def _poll(self, body):
        m = re.search(r"^global_load_dword [^\n]*$", body, re.M)
        assert m, "no flag poll in the transition phase"
        return m.group(0)

    def test_poll_bypasses_the_cache(self):
        line = self._poll(_transition())
        assert " sc0" in line and " sc1" in line, (
            "the poll leaves the flag cacheable in L2"
        )

    def test_poll_spins_on_the_flag(self):
        assert re.search(
            r"^(label_A2AWaitFlag\w*):\n"
            r"global_load_dword[^\n]*\n"
            r"s_waitcnt[^\n]*\n"
            r"v_cmp_lt_u32 vcc, v\d+, 1[^\n]*\n"
            r"s_cbranch_vccnz \1",
            _transition(),
            re.M,
        ), "the flag is read once instead of polled until the segment lands"

    def test_poll_precedes_the_gathered_rebind(self):
        body = _transition()
        assert body.index("label_A2AWaitFlag") < body.index("sgprSrdB"), (
            "srdB is rebound before the flag poll"
        )


class TestA2AGemmEnqueueAhead:
    """Batch k's enqueuers place batch k+1; batch 0's place batches 0 and 1."""

    def _election(self, src):
        """Up to the election atomic."""
        m = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
        assert m, "no s_atomic_inc in the emitted kernel"
        return src[: m.start()]

    def test_the_last_batch_but_batch_0_skips_the_election(self):
        assert re.search(
            r"s_add_u32 s(\d+), s\[sgprA2ABlockLo\], s\[sgprA2ABlockCount\][^\n]*\n"
            r"s_cmp_eq_u32 s\1, s\[sgprNumWorkGroups1\][^\n]*\n"
            r"s_cselect_b32 s\1, s\[sgprA2ABlockLo\], 0[^\n]*\n"
            r"s_cmp_lg_u32 s\1, 0[^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue",
            self._election(_src()),
        ), "the last batch still draws a ticket"

    def test_one_batch_ahead_stops_after_the_next_batch(self):
        tail = _packing_pass()
        lo = _placed_floor(tail)
        assert re.search(
            r"s_cmp_lg_u32 s%s, s\[sgprA2ABlockLo\][^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue[^\n]*\n"
            r"s_add_u32 s(\d+), s%s, s\[sgprA2ABlockCount\][^\n]*\n"
            r"s_cmp_lt_u32 s\1, s\[sgprNumWorkGroups1\][^\n]*\n"
            r"s_cbranch_scc1 label_a2a_next_batch\w*[^\n]*\n$" % (lo, lo),
            tail,
        ), "batch 0 does not go on to batch 1 only, or batch k > 0 does not stop"


class TestA2AGemmExitTally:
    """The tail after the shard loop: block and grid tallies, the grid winner's flag
    clear and done publish, and the KernelEnd drain-send wait."""

    def _tail(self, src):
        """The shard loop's back-edge through the tally's end label."""
        head = list(re.finditer(r"^s_cbranch_scc0 label_A2AShardLoopBegin", src, re.M))
        end = re.search(r"^label_a2a_exit_tally_end\w*:", src, re.M)
        assert head and end, "no exit tally after the shard loop"
        assert head[-1].end() < end.start()
        return src[head[-1].end() : end.start()]

    def _wait(self, src):
        """KernelEnd through s_endpgm."""
        m = re.search(r"^label_KernelEnd:\n(.*?)^s_endpgm", src, re.M | re.S)
        assert m, "no KernelEnd in the emitted kernel"
        return m.group(1)

    def _atomics(self, tail):
        return list(re.finditer(r"^s_atomic_inc (s\d+), (s\[\d+:\d+\]), (s\d+) glc", tail, re.M))

    def test_every_wave_passes_a_barrier_before_wave_0_tallies(self):
        tail = self._tail(_src())
        before = tail[: self._atomics(tail)[0].start()]
        barrier = before.find("s_barrier")
        wave = re.search(r"s_cmp_ge_u32 s\d+, 64[^\n]*\ns_cbranch_scc1 label_a2a_exit_tally_end", before)
        assert barrier >= 0 and wave and barrier < wave.start()

    def test_grid_winner_clears_the_flag_block_at_system_scope(self):
        from Tensile.Components.Signature import FUSED_A2A_MODE1_FLAG_OFFSET

        tail = self._tail(_src())
        clear = tail[self._atomics(tail)[1].end() :]
        assert re.search(
            r"global_store_dword v\d+, v\d+, s\[sgprA2ACounterPtr:sgprA2ACounterPtr\+1\] "
            r"offset:%d sc0 sc1" % FUSED_A2A_MODE1_FLAG_OFFSET,
            clear,
        ), "the flag clear does not reach memory"

    def test_drain_send_bit_gates_the_publish(self):
        from Tensile.Components.Signature import FUSED_A2A_DRAIN_SEND

        tail = self._tail(_src())
        gate = re.search(
            r"s_and_b32 (s\d+), \1, %d[^\n]*\n"
            r"s_cbranch_scc0 label_a2a_exit_tally_end" % FUSED_A2A_DRAIN_SEND,
            tail,
        )
        assert gate, "the publish is not gated on FUSED_A2A_DRAIN_SEND"
        assert tail.find("label_a2a_publish_done") > gate.end()

    def test_kernel_end_waits_for_every_other_peer_done(self):
        wait = self._wait(_src())
        assert re.search(r"s_cmp_eq_u32 s\[sgprA2AShardIdx\], 0[^\n]*\ns_cbranch_scc1 label_a2a_drain_send_end", wait)
        assert re.search(
            r"s_cmp_eq_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardCounter\][^\n]*\n"
            r"s_cbranch_scc1 label_a2a_drain_send_next",
            wait,
        ), "the wait does not skip this rank"
        assert re.search(
            r"global_load_dword (v\d+), v\d+, s\[sgprA2ACounterPtr:sgprA2ACounterPtr\+1\] sc0 sc1[^\n]*\n"
            r"s_waitcnt vmcnt\(0\)[^\n]*\n"
            r"v_cmp_gt_u32 vcc, s\[sgprA2AFeatIdx\], \1[^\n]*\n"
            r"s_cbranch_vccnz label_a2a_drain_send_spin",
            wait,
        ), "the wait does not spin on done[s] < seq at system scope"
