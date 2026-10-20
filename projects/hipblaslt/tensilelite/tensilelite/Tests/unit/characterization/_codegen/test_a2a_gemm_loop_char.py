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
import pathlib
import re

import pytest

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = "tensilelite/Tests/common/comm/gfx950/a2a_gemm_loop.yaml"


@functools.lru_cache(maxsize=None)
def _src():
    return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]


@functools.lru_cache(maxsize=None)
def _derived_state(config):
    return derive_states(config, arch="gfx950", limit_solutions=1)[0]


def _state(config=_CONFIG):
    return copy.deepcopy(_derived_state(config))


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


class TestA2AGemmSegmentAbi:
    """Arg order across the layout list, the addArg sequence and the C++ twin."""

    _CPP_KERNARG = "include/Tensile/FusedA2AKernArg.hpp"

    def _emitted_segment(self):
        names = re.findall(r"-\s+\.name:\s+(\S+)", _src())
        return names[names.index("peer_0_flagPtr"):]

    def _layout_args(self):
        from Tensile.Components.Signature import _FUSED_A2A_SEGMENT_ARGS

        return list(_FUSED_A2A_SEGMENT_ARGS)

    def _cpp_source(self):
        hdr = pathlib.Path(self._CPP_KERNARG)
        if not hdr.is_file():
            pytest.skip("run from the tensilelite root to reach the C++ headers")
        return hdr.read_text()

    def test_segment_ends_with_the_cu_count(self):
        assert self._emitted_segment()[-2:] == ["FusedAM", "FusedNumCu"]

    def test_addarg_order_matches_the_layout_list(self):
        assert self._emitted_segment() == [n for n, _ in self._layout_args()]

    def test_cpp_twin_appends_the_scalars_in_the_emitted_order(self):
        appended = re.findall(r'append<[^>]+>\(\s*"([A-Za-z_]\w*)"', self._cpp_source())
        assert appended == [n for n in self._emitted_segment() if not n.startswith("peer_")]

    def test_cpp_twin_counts_every_four_byte_scalar(self):
        m = re.search(r"FUSED_A2A_SLOT_COUNT \+ 1\) \* 8 \+ (\d+) \* 4", self._cpp_source())
        assert m, "FUSED_A2A_SEGMENT_BYTES no longer has the scalar term"
        assert int(m.group(1)) == sum(1 for _, size in self._layout_args() if size == 4)


class TestA2AGemmBatchNumbering:
    """The prologue's batch span: the first block and the block count."""

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

    def test_block_span_is_clamped_to_the_token_block_count(self):
        assert re.search(r"s_min_u32[^\n]*sgprNumWorkGroups1", _src())

    def test_span_lands_in_two_long_lived_scalars(self):
        src = _src()
        assert ".set sgprA2ABlockLo," in src
        assert ".set sgprA2ABlockCount," in src

    def test_block_count_is_the_span_width(self):
        assert re.search(
            r"s_sub_u32 s\[sgprA2ABlockCount\][^\n]*sgprA2ABlockLo", _src()
        )

    def test_flag_row_aliases_the_block_count_past_the_enqueue(self):
        src = _src()
        count = re.search(r"^\.set sgprA2ABlockCount, (\d+)$", src, re.M)
        row = re.search(r"^\.set sgprA2AFlagRow, (\d+)$", src, re.M)
        assert count and row, "no A2AFlagRow alias in the emitted kernel"
        assert row.group(1) == count.group(1), "A2AFlagRow takes a register of its own"
        assert src.index("label_A2ASkipEnqueue:") < row.start(), (
            "A2AFlagRow is written while the enqueuer still reads A2ABlockCount"
        )

    def test_flag_row_is_the_last_block_of_the_group(self):
        src = _src()
        assert re.search(r"s_sub_u32 s\d+, s\[sgprWorkGroup1\], s\[sgprA2ABlockLo\]", src), (
            "the group offset is not taken from the batch's first block"
        )
        assert re.search(
            r"v_readfirstlane_b32 s\[sgprA2AFlagRow\], v\d+[^\n]*\n"
            r"s_add_u32 s\[sgprA2AFlagRow\], s\[sgprA2AFlagRow\], s\[sgprA2ABlockLo\][^\n]*\n"
            r"s_mul_i32 s\[sgprA2AFlagRow\], s\[sgprA2AFlagRow\], s\d+",
            src,
        ), "the flag row is not (bLo + (b - bLo) % (W-1)) * (W-1)"


class TestA2AGemmElection:
    """The batch's enqueuer work-group: the gates, the SMEM atomic, its target, the LDS broadcast."""

    # NumThreads / WavefrontSize of the config, and log2(WavefrontSize).
    _NUM_WAVES = 4
    _WAVE_SHIFT = 6

    def _atomic(self, src):
        m = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
        assert m, "no s_atomic_inc in the emitted kernel"
        return m

    def _ticket(self, src):
        """The SGPR the election atomic returns its pre-op value in."""
        return self._atomic(src).group(0).split(",")[0].split()[-1]

    def test_election_atomic_carries_glc(self):
        assert " glc" in self._atomic(_src()).group(0)

    def test_single_rank_skips_before_the_election_atomic(self):
        src = _src()
        assert re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_cmp_eq_u32 s\1, 0[^\n]*// W == 1\?\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue",
            src[: self._atomic(src).start()],
        ), "W == 1 still draws a ticket"

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

    def test_wave_w_serves_queues_w_up_by_num_waves(self):
        src = _src()
        tick = re.escape(self._ticket(src))
        wave = re.escape(self._wave(src))
        m = re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_cmp_ge_u32 %s, s\1\b[^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue[^\n]*\n"
            r"s_mov_b32 %s, %s\b" % (wave, tick, wave),
            src[self._atomic(src).end() :],
        )
        assert m, "the first queue is not the wave index, or waves at W-1 and up still enqueue"
        assert re.search(
            r"s_add_u32 %s, %s, %d[^\n]*\n"
            r"s_cmp_lt_u32 %s, s%s\b[^\n]*\n"
            r"s_cbranch_scc1 label_a2a_place_\w+[^\n]*\n"
            r"s_mov_b32 %s, %s\b"
            % (tick, tick, self._NUM_WAVES, tick, m.group(1), tick, wave),
            src[m.end() :],
        ), "a wave does not step its queue by numWaves below W-1"

    def test_the_served_rank_carries_the_ticket(self):
        src = _src()
        assert re.search(
            r"s_add_u32 s(\d+), s\1, 1[^\n]*// myRank \+ 1\n"
            r"s_add_u32 s\1, s\1, %s\b[^\n]*\n"
            r"s_sub_u32 s(\d+), s\1, s\[sgprA2AShardCounter\][^\n]*\n"
            r"s_cmp_ge_u32 s\1, s\[sgprA2AShardCounter\][^\n]*\n"
            r"s_cselect_b32 s\1, s\2, s\1\b" % re.escape(self._ticket(src)),
            src,
        ), "the served rank is not myRank + 1 + ticket wrapped once at W"

    def test_the_queue_index_is_the_ticket(self):
        src = _src()
        tick = re.escape(self._ticket(src))
        assert re.search(
            r"s_add_u32 s\d+, s\d+, %s[^\n]*// flag slot" % tick, src
        ), "the flag slot is not indexed by the election ticket"
        assert re.search(
            r"s_add_u32 s\d+, %s, 1[^\n]*// gathered segment" % tick, src
        ), "the gathered segment is not indexed by the election ticket"

    def test_the_group_start_carries_the_ticket(self):
        src = _src()
        assert re.search(
            r"s_sub_u32 s(\d+), s(\d+), s\1[^\n]*// queue 0's group start[^\n]*\n"
            r"s_add_u32 s\1, s\1, %s\b[^\n]*\n"
            r"s_sub_u32 s(\d+), s\1, s\3[^\n]*\n"
            r"s_cmp_gt_i32 s\1, s\2[^\n]*\n"
            r"s_cselect_b32 s\1, s\3, s\1\b" % re.escape(self._ticket(src)),
            src,
        ), "the group start is not queue 0's + ticket wrapped down once by W-1"

    def test_election_counters_start_past_the_line_aligned_flag_block(self):
        from Tensile.Components.Signature import (
            FUSED_A2A_LINE_BYTES,
            FUSED_A2A_MODE1_FLAG_OFFSET,
        )

        src = _src()
        mask = hex(0xFFFFFFFF & -FUSED_A2A_LINE_BYTES)
        assert re.search(
            r"s_and_b32 s\d+, s\d+, %s\b" % mask, src
        ), "the flag block width is not rounded up to a line"
        assert re.search(
            r"^s_atomic_inc s\d+, s\[\d+:\d+\], %s\b" % hex(FUSED_A2A_MODE1_FLAG_OFFSET),
            self._atomic(src).group(0),
        ), "the election atomic does not sit at the flag block offset"

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

    def test_each_queue_reserves_once_and_submits_once(self):
        body = _packing_pass()
        assert body.count("s_atomic_cmpswap_x2") == 1
        assert body.count("store committedWptr = pending") == 1

    def test_cursors_are_raised_before_the_reserve_loop(self):
        body = _packing_pass()
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the packing pass"
        assert body[: rsv.start()].count("s_atomic_umax_x2") == 2

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

    def test_the_block_counter_drops_by_a_group_at_a_time(self):
        body = _packing_pass()
        m = re.search(
            r"s_mov_b32 s(\d+), s\[sgprA2ABlockCount\][^\n]*\n"
            r"v_readfirstlane_b32 s\d+, v\d+[^\n]*\n"
            r"label_a2a_group_loop\w*:",
            body,
        )
        assert m, "the block counter is not seeded from A2ABlockCount before the group loop"
        n = re.search(
            r"s_add_u32 s(\d+), s\1, 1[^\n]*// blocks in this group", body
        )
        assert n, "the group's block count is not quotient + 1"
        assert re.search(
            r"s_sub_u32 s%s, s%s, s%s[^\n]*// group placed" % (m.group(1), m.group(1), n.group(1)),
            body,
        ), "the block counter does not count one group at a time"

    def test_the_group_size_is_the_blocks_down_to_the_batch_floor(self):
        body = _packing_pass()
        assert re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_sub_u32 s(\d+), s\d+, s%s[^\n]*// group start - bLo\n"
            r"v_cvt_f64_u32 v\[\d+:\d+\], s\1" % _placed_floor(body),
            body,
        ), "the group size is not (group start - bLo) / (W-1)"

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

    def test_the_submit_ends_the_packing_pass(self):
        body = _packing_pass()
        tail = body[body.rindex("store committedWptr = pending") :].splitlines()[1:]
        assert [l.split()[0] for l in tail if l.strip()] == [
            "s_waitcnt", "s_add_u32", "s_cmp_lt_u32", "s_cbranch_scc1", "s_mov_b32",
            "s_cmp_lg_u32", "s_cbranch_scc1", "s_add_u32", "s_cmp_lt_u32", "s_cbranch_scc1"
        ], "the enqueuer does more than pick the next queue and batch after a submit:\n%s" % "\n".join(tail)


class TestA2AGemmRoundGroupOrder:
    """Each queue emits its blocks in stride-(W-1) groups, one group per round.

    Round r reads block b from queue (b + r - 1) mod (W-1).
    """

    def test_lanes_step_a_whole_group(self):
        body = _packing_pass()
        m = re.search(
            r"v_and_b32 v(\d+), 63, v\[vgprSerial\][^\n]*\n"
            r"v_mul_u32_u24 v(\d+), s\[sgprA2AShardCounter\], v\1[^\n]*\n"
            r"v_sub_u32 v\2, v\2, v\1",
            body,
        )
        assert m, "the lane stride is not W-1 blocks"
        assert re.search(r"v_sub_u32 v\d+, s\d+, v%s[^\n]*// the lane's block" % m.group(2), body), (
            "the lanes walk consecutive blocks instead of one round's group"
        )

    def test_the_signal_targets_the_groups_last_block(self):
        body = _packing_pass()
        last = re.search(
            r"v_readfirstlane_b32 s\d+, v\d+[^\n]*// quotient\n"
            r"v_readfirstlane_b32 s(\d+), v\d+[^\n]*// remainder\n"
            r"(?:[^\n]*\n)?"
            r"s_add_u32 s\1, s\1, s%s[^\n]*// the group's last block" % _placed_floor(body),
            body,
        )
        assert last, "the group's last block is not bLo + (group start - bLo) % (W-1)"
        assert re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_mul_i32 s\1, s%s, s\1[^\n]*// bLast \* \(W-1\)" % last.group(1),
            body,
        ), "the flag slot is not taken at the group's last block"

    def test_lanes_below_the_batch_floor_are_masked(self):
        body = _packing_pass()
        assert re.search(
            r"v_cmp_le_i32 vcc, s%s, v(\d+)[^\n]*\n"
            r"s_mov_b64 exec, vcc" % _placed_floor(body),
            body,
        ), "lanes past the batch's first block still store a COPY"

    def test_empty_groups_are_branched_over(self):
        body = _packing_pass()
        assert re.search(
            r"label_a2a_group_loop\w*:[^\n]*\n"
            r"s_cmp_lt_i32 s\d+, s%s[^\n]*\n"
            r"s_cbranch_scc1 label_a2a_group_next" % _placed_floor(body),
            body,
        ), "a group holding no block still places packets"

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

    def test_the_signal_lane_holds_no_block(self):
        assert re.search(
            r"s_mov_b32 exec_lo, 0[^\n]*\n"
            r"s_mov_b32 exec_hi, 0x80000000[^\n]*\n"
            r"v_mov_b32 v\d+, 0x40000000",
            _packing_pass(),
        ), "the last lane's block stride does not keep it below the batch floor"

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

    # srdShiftLeft["B"] (GlobalReadVectorWidthB, 8 here) * bpeGR (bf16).
    _PRE_PAD = 16

    def _segment_base(self, src):
        from Tensile.Components.Signature import fusedA2AKernArgLayout

        m = re.search(
            r"s_load_dword s\[sgprA2AShardCounter\], "
            r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
            src,
        )
        assert m, "no FusedW kernarg load in the emitted kernel"
        return int(m.group(1), 16) - fusedA2AKernArgLayout()["FusedW"]

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

    def test_copy_source_is_the_peer_input_pointer(self):
        from Tensile.Components.SdmaRingEmitter import OFF_recvPtr

        src = _src()
        want = self._segment_base(src) + OFF_recvPtr
        assert (
            "offset:%d " % want in _packing_pass()
            or "offset:%d\n" % want in _packing_pass()
        ), "the peer input pointer at offset %d is never loaded" % want

    def test_atomic_target_is_the_local_flag_block(self):
        from Tensile.Components.Signature import FUSED_A2A_MODE1_FLAG_OFFSET

        body = _packing_pass()
        assert re.search(
            r"s_add_u32 s\d+, s\[sgprA2ACounterPtr\], s\d+[^\n]*\n"
            r"s_addc_u32 s\d+, s\[sgprA2ACounterPtr\+1\], 0",
            body,
        ), "the ATOMIC target is not derived from this device's counter block"
        assert re.search(
            r"s_add(?:c)?_u32 s\d+, s\d+, %d\b" % FUSED_A2A_MODE1_FLAG_OFFSET, body
        ), "the flag slot does not sit at the mode-1 flag block offset"

    def test_destination_undoes_the_address_pre_pad(self):
        assert re.search(
            r"s_add_u32 s\d+, s\[sgprAddressB\], %d\b" % self._PRE_PAD,
            _packing_pass(),
        ), "the destination base never re-adds the AddressB pre-pad"


class TestA2AGemmRegisterBudget:
    """The emitted kernel's SGPR high-water mark."""

    def test_kernel_stays_within_the_sgpr_ceiling(self):
        found = re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", _src())
        assert found, "no .amdhsa_next_free_sgpr in the emitted kernel"
        assert int(found[0]) <= 102


class TestA2AGemmSolutionProblemType:
    """The mode reaches the serialized solution, where the host kernarg gate reads it."""

    def test_state_keys_carry_the_mode(self):
        from Tensile.Contractions import ProblemType

        assert "fusedA2A" in ProblemType.StateKeys


class TestA2AGemmSubRoundsReachTheLibrary:
    def _size_mapping(self, subRounds):
        from Tensile.Contractions import SizeMapping

        state = _state()
        state["A2ASubRounds"] = subRounds
        return SizeMapping.FromOriginalState(state)

    def test_state_keys_carry_sub_rounds(self):
        from Tensile.Contractions import SizeMapping

        assert "a2aSubRounds" in SizeMapping.StateKeys

    @pytest.mark.parametrize("subRounds", [1, 2, 4])
    def test_sub_rounds_round_trip(self, subRounds):
        assert self._size_mapping(subRounds).a2aSubRounds == subRounds

    def test_cpp_maps_sub_rounds(self):
        hdr = pathlib.Path("include/Tensile/Serialization/ContractionSolution.hpp")
        if not hdr.is_file():
            pytest.skip("run from the tensilelite root to reach the C++ headers")
        assert '"a2aSubRounds", s.a2aSubRounds' in hdr.read_text()


class TestA2AGemmSubRoundsName:
    def _names(self, config, subRounds, mode=None):
        from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin

        state = _state(config)
        state["A2ASubRounds"] = subRounds
        if mode is not None:
            state["ProblemType"]["FusedA2A"] = mode
        return getKernelNameMin(state, False), getSolutionNameMin(state, False)

    def test_sub_rounds_split_the_names(self):
        names = [self._names(_CONFIG, s) for s in (1, 2, 4)]
        for which in (0, 1):
            assert len({n[which] for n in names}) == 3

    def test_single_sub_round_names_carry_sr1(self):
        assert all("_SR1_" in n for n in self._names(_CONFIG, 1))

    @pytest.mark.parametrize(
        "config, mode",
        [("tensilelite/Tests/common/comm/gfx950/fused_a2a.yaml", None), (_CONFIG, "Disabled")],
        ids=["GemmA2A", "stock"],
    )
    def test_other_modes_ignore_sub_rounds(self, config, mode):
        assert self._names(config, 2, mode) == self._names(config, 1, mode)


class TestA2AGemmPredicates:
    """The runtime shard guard rides on the solution's compound predicate list."""

    _CPP_PREDICATES = "include/Tensile/Serialization/ContractionPredicates.hpp"

    def _predicates(self, mode=None, subRounds=None):
        from Tensile.Contractions import ProblemPredicate, ProblemType

        state = _state()
        if mode is not None:
            state["ProblemType"]["FusedA2A"] = mode
        if subRounds is not None:
            state["A2ASubRounds"] = subRounds
        pt = ProblemType.FromOriginalState(state["ProblemType"])
        return {p.tag: p.value for p in ProblemPredicate.CompoundPredicates(state, pt)}

    def test_shape_carries_depth_u_times_sub_rounds(self):
        assert self._predicates()["A2AGemmShape"] == 64

    def test_sub_rounds_scale_the_shape_divisor(self):
        assert self._predicates(subRounds=2)["A2AGemmShape"] == 128

    def test_mode_zero_omits_it(self):
        assert "A2AGemmShape" not in self._predicates(mode="Disabled")

    def test_cpp_registers_it(self):
        hdr = pathlib.Path(self._CPP_PREDICATES)
        if not hdr.is_file():
            pytest.skip("run from the tensilelite root to reach the C++ headers")
        src = hdr.read_text()
        assert "Pair<Predicates::Contraction::A2AGemmShape>" in src
        assert "MappingTraits<Predicates::Contraction::A2AGemmShape," in src


class TestA2AGemmShardLoop:
    """The shard loop wraps the unroll loop and closes before the store."""

    def test_initc_precedes_the_shard_loop(self):
        src = _src()
        assert src.index("initC") < src.index("A2AShardLoopBegin"), (
            "initC is emitted inside the shard loop"
        )

    def test_shard_loop_closes_before_the_summation_end(self):
        src = _src()
        assert src.rindex("A2AShardLoopBegin") < src.index("Summation_End"), (
            "the back edge lands after endSummation"
        )

    @pytest.mark.parametrize("mat", ["C", "D"])
    def test_store_srd_accumulation_is_outside_the_shard_loop(self, mat):
        src = _src()
        accum = "s[sgprSrd%s+0], s[sgprSrd%s+0]" % (mat, mat)
        assert accum in src, "computeStoreSrdStart no longer accumulates into Srd%s" % mat
        body = src[src.index("label_A2AShardLoopBegin:"):src.rindex("A2AShardLoopBegin")]
        assert accum not in body, "Srd%s accumulation is inside the shard loop" % mat


class TestA2AGemmTransitionPhase:
    """The transition phase sits between the tail loop and the back edge."""

    def test_transition_phase_sits_inside_the_shard_loop(self):
        src = _src()
        assert src.index("A2AShardLoopBegin") < src.index("A2A_TRANSITION begin"), (
            "the transition phase is emitted before the shard loop opens"
        )
        assert src.index("A2A_TRANSITION end") < src.rindex("A2AShardLoopBegin"), (
            "the transition phase is emitted after the back edge"
        )

    def test_transition_restores_both_shadow_limits(self):
        """ShadowLimit is reset each round, not carried."""
        body = _transition()
        assert "ShadowLimitA" in body, "srdA's limit stays drained across rounds"
        assert "ShadowLimitB" in body, "srdB's limit stays drained across rounds"

    def test_transition_is_branched_over_on_the_final_round(self):
        body = _transition()
        skip = "s_cbranch_scc1 label_A2ATransitionEnd"
        assert "s_cmp_eq_i32 s[sgprA2AShardCounter], 1" in body, (
            "the transition phase runs on the final round too"
        )
        assert body.index(skip) < body.index("A2AShardIdx"), (
            "the skip branch sits after part of the next round's setup"
        )

    def test_the_skip_target_clears_the_whole_transition(self):
        body = _transition()
        assert body.index("label_A2ATransitionEnd:") > body.rindex("base += shard offset"), (
            "the skip label precedes the last rebind"
        )


class TestA2AGemmFeatureShardIndex:
    """A is indexed by feature shard (my_rank + segment) mod W, B by the segment."""

    _SHIFT = (
        r"s_mul_i32 s(\d+), s\d+, s\[sgpr%s\][^\n]*\n"
        r"s_lshl_b64 s\[\d+:\d+\][^\n]*\n"
        r"s_add_u32 s\[sgprSrd%s\+0\], s\[sgprSrd%s\+0\], s\1"
    )

    def test_round_zero_seeds_the_index_from_my_rank(self):
        from Tensile.Components.Signature import fusedA2AKernArgLayout

        prologue = _prologue()
        loads = dict(
            re.findall(
                r"s_load_dword s\[sgpr(A2AShardCounter|A2AFeatIdx)\], "
                r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
                prologue,
            )
        )
        assert "A2AFeatIdx" in loads, "round 0 never reads FusedMyRank"
        layout = fusedA2AKernArgLayout()
        delta = layout["FusedW"] - layout["FusedMyRank"]
        assert int(loads["A2AShardCounter"], 16) - int(loads["A2AFeatIdx"], 16) == delta, (
            "the feature shard is seeded from some kernarg other than FusedMyRank"
        )

    def test_round_zero_shifts_srda_by_the_feature_shard(self):
        prologue = _prologue()
        assert re.search(self._SHIFT % ("A2AFeatIdx", "A", "A"), prologue), (
            "round 0 reads feature shard 0 instead of this rank's own"
        )

    def test_the_index_wraps_at_w_before_the_rebind(self):
        body = _transition()
        m = re.search(
            r"s_add_u32 s\[sgprA2AFeatIdx\], s(\d+), s\[sgprA2AShardIdx\][^\n]*\n"
            r"s_sub_u32 s\1, s\[sgprA2AFeatIdx\], s\d+[^\n]*\n"
            r"s_min_u32 s\[sgprA2AFeatIdx\], s\[sgprA2AFeatIdx\], s\1",
            body,
        )
        assert m, (
            "the feature shard is not myRank + this block's segment, wrapped at W"
        )
        assert m.end() < body.index("s[sgprSrdA+0], s[sgprSrdA+0]"), (
            "srdA is rebound before the feature shard wraps"
        )

    def test_transition_splits_the_two_indices(self):
        body = _transition()
        assert re.search(self._SHIFT % ("A2AFeatIdx", "A", "A"), body), (
            "srdA steps by the gathered segment instead of the feature shard"
        )
        assert re.search(self._SHIFT % ("A2AShardIdx", "B", "B"), body), (
            "srdB no longer steps by the gathered segment"
        )


class TestA2AGemmLocalRoundReadsX:
    """Round 0 reads rows [myRank*nToken, +nToken) of this rank's x, not gathered segment 0."""

    def _rebase(self, src):
        """The prologue between the B SRD setup and the shard-loop label."""
        head = src.index("label_StridedBatchedGemmLoadB_End:")
        return src[head : src.index("label_A2AShardLoopBegin:")]

    def _x_load(self, body):
        from Tensile.Components.SdmaRingEmitter import OFF_recvPtr, PEER_GROUP_BYTES

        m = re.search(
            r"s_mul_i32 s(\d+), s\[sgprA2AFeatIdx\], %d[^\n]*\n"
            r"s_load_dwordx2 s\[(\d+):(\d+)\], s\[sgprKernArgAddress:sgprKernArgAddress\+1\], "
            r"s\1 offset:(\d+)" % PEER_GROUP_BYTES,
            body,
        )
        assert m, "round 0 never loads the x pointer of peer group myRank"
        return m, OFF_recvPtr

    def test_round_zero_loads_this_ranks_x_pointer(self):
        src = _src()
        m, offRecv = self._x_load(self._rebase(src))
        segBase = TestA2AGemmPacketBody()._segment_base(src)
        assert int(m.group(4)) == segBase + offRecv, "the load misses the peer x slot"

    def test_round_zero_steps_to_this_ranks_rows(self):
        body = self._rebase(_src())
        assert re.search(
            r"s_mul_i32 s(\d+), s\[sgprA2AFeatIdx\], s\[sgprSizeJ\][^\n]*\n"
            r"s_mul_hi_u32 s(\d+), s\1, s\[sgprStrideB1J\][^\n]*\n"
            r"s_mul_i32 s(\d+), s\1, s\[sgprStrideB1J\]",
            body,
        ), "round 0 does not offset x by myRank * nToken rows"

    def test_round_zero_rebases_srdb_from_addressb_onto_x(self):
        body = self._rebase(_src())
        m, _ = self._x_load(body)
        lo, hi = m.group(2), m.group(3)
        tail = body[m.end() :]
        sub = re.search(
            r"s_sub_u32 s%s, s%s, s\[sgprAddressB\+0\][^\n]*\n"
            r"s_subb_u32 s%s, s%s, s\[sgprAddressB\+1\]" % (lo, lo, hi, hi),
            tail,
        )
        assert sub, "the x base is not taken relative to AddressB"
        add = re.search(
            r"s_add_u32 s\[sgprSrdB\+0\], s\[sgprSrdB\+0\], s%s[^\n]*\n"
            r"s_addc_u32 s\[sgprSrdB\+1\], s\[sgprSrdB\+1\], s%s" % (lo, hi),
            tail,
        )
        assert add and add.start() > sub.end(), "srdB is not moved onto x"

    def test_the_rebase_keeps_the_address_pre_pad(self):
        body = self._rebase(_src())
        m, _ = self._x_load(body)
        assert re.search(
            r"s_sub_u32 s%s, s%s, %d[^\n]*\n"
            r"s_subb_u32 s%s, s%s, 0" % (m.group(2), m.group(2), TestA2AGemmPacketBody._PRE_PAD,
                                         m.group(3), m.group(3)),
            body[m.end() :],
        ), "the rebased srdB drops the AddressB pre-pad"


class TestA2AGemmPerBlockMisphase:
    """Token block b walks the peer queues from b % (W-1), rotating one per round.

    A2AShardIdx holds the gathered segment of the round about to be entered;
    the queue it names is that segment minus one.
    """

    def test_the_divisor_is_the_peer_queue_count_clamped_to_one(self):
        assert re.search(
            r"s_sub_u32 s(\d+), s\[sgprA2AShardCounter\], 1[^\n]*\n"
            r"s_max_u32 s\1, s\1, 1",
            _prologue(),
        ), "the divisor is not max(W-1, 1)"

    def test_round_zero_seeds_the_segment_from_the_token_block(self):
        assert re.search(
            r"v_sub_u32 v(\d+), s\[sgprWorkGroup1\], v\d+[^\n]*\n"
            r"(?:s_nop[^\n]*\n)?"
            r"v_readfirstlane_b32 s\[sgprA2AShardIdx\], v\1[^\n]*\n"
            r"s_add_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardIdx\], 1",
            _prologue(),
        ), "every token block opens on the same peer queue"

    def test_the_segment_rotates_back_to_one_at_w(self):
        assert re.search(
            r"s_add_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardIdx\], 1[^\n]*\n"
            r"s_waitcnt[^\n]*\n"
            r"s_cmp_lt_u32 s\[sgprA2AShardIdx\], s(\d+)[^\n]*\n"
            r"s_cselect_b32 s\[sgprA2AShardIdx\], s\[sgprA2AShardIdx\], 1",
            _transition(),
        ), "the segment runs past W-1 instead of wrapping to the first peer queue"

    def test_the_segment_advances_after_both_rebinds(self):
        body = _transition()
        assert body.index("s_add_u32 s[sgprA2AShardIdx], s[sgprA2AShardIdx], 1") \
            > body.rindex("base += shard offset"), (
            "the segment advances before the rebinds"
        )


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

    def test_poll_reads_this_device_flag_block(self):
        from Tensile.Components.Signature import FUSED_A2A_MODE1_FLAG_OFFSET

        line = self._poll(_transition())
        assert "s[sgprA2ACounterPtr:sgprA2ACounterPtr+1]" in line, (
            "the poll does not read this device's counter block"
        )
        assert "offset:%d" % FUSED_A2A_MODE1_FLAG_OFFSET in line, (
            "the poll misses the mode-1 flag block offset"
        )

    def test_flag_slot_is_indexed_by_the_queue_not_the_rank(self):
        body = _transition()
        m = re.search(
            r"s_add_u32 s(\d+), s\[sgprA2AFlagRow\], s\[sgprA2AShardIdx\][^\n]*\n"
            r"s_sub_u32 s\1, s\1, 1[^\n]*\n"
            r"s_lshl_b32 s\1, s\1, 2",
            body,
        )
        assert m, "the polled slot is not the group's flag row + queue index"
        assert m.start() < body.index("s_add_u32 s[sgprA2AShardIdx]"), (
            "the slot is taken after the shard index advances"
        )

    def test_poll_precedes_the_gathered_rebind(self):
        body = _transition()
        assert body.index("label_A2AWaitFlag") < body.index("sgprSrdB"), (
            "srdB is rebound before the flag poll"
        )

    def test_poll_is_branched_over_on_the_final_round(self):
        body = _transition()
        assert body.index("s_cbranch_scc1 label_A2ATransitionEnd") \
            < body.index("label_A2AWaitFlag"), (
            "the flag poll precedes the final-round skip branch"
        )


class TestA2AGemmPerRoundState:
    """State the shard loop body re-establishes on every round."""

    def _body(self):
        src = _src()
        return src[src.index("label_A2AShardLoopBegin:"):src.rindex("A2AShardLoopBegin")]

    def _before_the_tail_loop_branch(self):
        body = self._body()
        return body[:body.index("s_cbranch_scc1 label_SkipTailLoopL")]

    def test_lds_write_address_is_reloaded_into_m0(self):
        assert "s_mov_b32 m0, s[sgprLocalWriteAddrA]" in self._body(), (
            "the DirectToLds m0 update is hoisted out of the shard loop"
        )

    def test_inner_trip_count_is_recomputed_from_k_local(self):
        assert "s[sgprLoopCounterL], s[sgprA2AKLocal]" in self._body(), (
            "the unroll trip count is hoisted out of the shard loop, or divides "
            "the whole W*k_local reduction instead of one shard"
        )

    def test_shard_offsets_step_per_tensor_layout(self):
        lines = _transition().splitlines()
        a = next(l for l in lines if "elements per A shard" in l)
        b = next(l for l in lines if "elements per B shard" in l)
        assert "sgprA2AKLocal" in a, "A steps a whole tensor instead of k_local along K"
        assert "sgprSizeJ" in b and "sgprStrideB1J" in b, (
            "B steps something other than one [nToken, k_local] segment"
        )

    def test_k_local_divide_sits_outside_the_shard_loop(self):
        assert "k_local = K / W" in _src()
        assert "k_local = K / W" not in self._body()

    def test_no_load_loop_is_inside_the_body(self):
        assert "NoLoadLoop" in self._body(), (
            "the drain phase is hoisted out of the shard loop"
        )

    def test_local_write_address_is_reset_on_every_path(self):
        reachable = self._before_the_tail_loop_branch()
        assert reachable.count("Set LWA to first buffer offset") == 2, (
            "srdA/srdB local-write addresses keep the round's buffer parity"
        )

    def test_local_read_address_reset_sits_behind_the_tail_loop_branch(self):
        assert "Set LRA to first buffer offset" not in self._before_the_tail_loop_branch()

    def test_transition_phase_resets_both_local_read_addresses(self):
        assert _transition().count("Set LRA to first buffer offset") == 2, (
            "local-read addresses keep the round's buffer parity while local-write "
            "addresses are reset"
        )

    def test_transition_phase_writes_gsu_sum_idx(self):
        assert "GSUSumIdx" in _transition()


class TestA2AGemmF64PairAlignment:
    """f64 operands are register pairs the assembler requires to start even."""

    def test_every_f64_vgpr_pair_starts_on_an_even_register(self):
        emitted = [
            line.strip()
            for line in _src().splitlines()
            if "f64" in line.split(" ", 1)[0]
        ]
        assert emitted, "no f64 instruction in the kernel"
        odd = [
            line
            for line in emitted
            if any(int(lo) % 2 for lo in re.findall(r"\bv\[(\d+):\d+\]", line))
        ]
        assert not odd, "odd-aligned f64 vgpr tuples:\n" + "\n".join(odd)


class TestA2AGemmEnqueueAhead:
    """Batch k's enqueuers place batch k+1; batch 0's place batches 0 and 1."""

    def _election(self, src):
        """Up to the election atomic."""
        m = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
        assert m, "no s_atomic_inc in the emitted kernel"
        return src[: m.start()]

    def _pass(self, tail):
        """The packing pass alone: the place label through the submit."""
        head = re.search(r"^label_a2a_place_batch\w*:", tail, re.M)
        end = re.search(r"^s_cmp_\w+ s\d+, s\[sgpr(A2ABlockLo|NumWorkGroups1)\]", tail[head.end() :], re.M)
        assert head and end, "no packing pass between the place label and the loop test"
        return tail[head.end() : head.end() + end.start()]

    def test_the_last_batch_but_batch_0_skips_the_election(self):
        assert re.search(
            r"s_add_u32 s(\d+), s\[sgprA2ABlockLo\], s\[sgprA2ABlockCount\][^\n]*\n"
            r"s_cmp_eq_u32 s\1, s\[sgprNumWorkGroups1\][^\n]*\n"
            r"s_cselect_b32 s\1, s\[sgprA2ABlockLo\], 0[^\n]*\n"
            r"s_cmp_lg_u32 s\1, 0[^\n]*\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue",
            self._election(_src()),
        ), "the last batch still draws a ticket"

    def test_the_pass_reads_the_placed_batch(self):
        tail = _packing_pass()
        lo, body = _placed_floor(tail), self._pass(tail)
        assert "sgprA2ABlockLo" not in body, "the pass reads this batch's floor"
        assert "v_cmp_le_i32 vcc, s%s, v" % lo in body, "lanes are not masked at the placed floor"

    def test_the_pass_keeps_the_placed_floor_and_the_cu_count(self):
        tail = _packing_pass()
        lo, body = _placed_floor(tail), self._pass(tail)
        cu = re.search(r"^s_mul_i32 s\d+, s\d+, s(\d+)[^\n]*// iB \* numCu$", tail, re.M)
        assert cu, "the next batch does not read the CU count"
        for reg in (lo, cu.group(1)):
            assert not re.search(r"^(?!s_cmp)\w+ s%s\b" % reg, body, re.M), "the pass writes s%s" % reg

    def test_the_next_batch_starts_at_the_previous_top(self):
        tail = _packing_pass()
        lo = _placed_floor(tail)
        assert re.search(
            r"^label_a2a_next_batch\w*:[^\n]*\n"
            r"s_add_u32 s%s, s%s, s\[sgprA2ABlockCount\][^\n]*\n" % (lo, lo),
            tail,
            re.M,
        ), "the next batch does not start at bLo + count"
        assert re.search(
            r"s_min_u32 s(\d+), s\1, s\[sgprNumWorkGroups1\][^\n]*\n"
            r"s_sub_u32 s\[sgprA2ABlockCount\], s\1, s%s[^\n]*\n"
            r"label_a2a_place_batch" % lo,
            tail,
        ), "the next batch's count is not its clamped top minus its floor"

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

    def test_the_flag_row_floor_is_written_once(self):
        assert len(re.findall(r"^(?!s_cmp)\w+ s\[sgprA2ABlockLo\],", _src(), re.M)) == 1


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

    def test_single_rank_skips_the_tally(self):
        tail = self._tail(_src())
        assert re.search(
            r"s_cmp_eq_u32 s\[sgprA2AKLocal\], 1[^\n]*\n"
            r"s_cbranch_scc1 label_a2a_exit_tally_end",
            tail[: self._atomics(tail)[0].start()],
        ), "W == 1 still tallies"

    def test_every_wave_passes_a_barrier_before_wave_0_tallies(self):
        tail = self._tail(_src())
        before = tail[: self._atomics(tail)[0].start()]
        barrier = before.find("s_barrier")
        wave = re.search(r"s_cmp_ge_u32 s\d+, 64[^\n]*\ns_cbranch_scc1 label_a2a_exit_tally_end", before)
        assert barrier >= 0 and wave and barrier < wave.start()

    def test_block_then_grid_tally_wrap_at_their_counts(self):
        tail = self._tail(_src())
        atomics = self._atomics(tail)
        assert len(atomics) == 2
        for m, slot, wrap in zip(atomics, ("sgprWorkGroup1", "sgprNumWorkGroups1"),
                                 ("sgprNumWorkGroups0", "sgprNumWorkGroups1")):
            seg = tail[: m.start()]
            assert re.search(r"s_lshl_b32 %s, s\[%s\], 2[^\n]*\n" % (m.group(3), slot), seg)
            assert re.search(r"s_sub_u32 %s, s\[%s\], 1[^\n]*\n$" % (m.group(1), wrap), seg)
            after = tail[m.end() :]
            assert re.search(
                r"s_sub_u32 (s\d+), s\[%s\], 1[^\n]*\n"
                r"s_cmp_lg_u32 %s, \1[^\n]*\n"
                r"s_cbranch_scc1 label_a2a_exit_tally_end" % (wrap, m.group(1)),
                after,
            ), "the %s tally does not gate on its last arrival" % slot

    def test_tallies_sit_past_the_flag_block(self):
        from Tensile.Components.Signature import FUSED_A2A_MODE1_FLAG_OFFSET

        tail = self._tail(_src())
        base = self._atomics(tail)[0].group(2)
        lo = base[2:].split(":")[0]
        m = re.search(r"s_add_u32 s%s, s\[sgprA2ACounterPtr\], (s\d+)" % lo, tail)
        assert m, "the tally base is not counter_ptr + the flag block"
        assert re.search(r"s_and_b32 %s, %s, 0xffffffc0" % (m.group(1), m.group(1)), tail)
        assert tail.count("s_add_u32 %s, %s, %d" % ((self._atomics(tail)[0].group(3),) * 2
                                                     + (FUSED_A2A_MODE1_FLAG_OFFSET,))) == 2

    def test_election_counters_sit_past_the_tallies(self):
        src = _src()
        m = re.search(
            r"s_lshl_b32 (s\d+), s\[sgprNumWorkGroups1\], 2[^\n]*\n"
            r"s_add_u32 \1, \1, 67[^\n]*\n"
            r"s_and_b32 \1, \1, 0xffffffc0[^\n]*\n"
            r"s_add_u32 (s\d+), \2, \1[^\n]*\+ tally region",
            src[: re.search(r"^s_atomic_inc ", src, re.M).start()],
        )
        assert m, "the election counter offset skips no tally region"

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

    def test_publish_stores_seq_to_every_other_peer_at_system_scope(self):
        from Tensile.Components.Signature import FUSED_A2A_MODE1_SEQ_OFFSET

        tail = self._tail(_src())
        assert re.search(
            r"v_add_u32 (v\d+), 1, \1[^\n]*\n"
            r"global_store_dword v\d+, \1, s\[sgprA2ACounterPtr:sgprA2ACounterPtr\+1\] "
            r"offset:%d sc0 sc1" % FUSED_A2A_MODE1_SEQ_OFFSET,
            tail,
        ), "seq is not bumped in this rank's flagPtr block"
        loop = tail[tail.find("label_a2a_publish_done:") :]
        assert re.search(
            r"s_cmp_eq_u32 s\[sgprA2AShardIdx\], s\[sgprA2AShardCounter\][^\n]*\n"
            r"s_cbranch_scc1 label_a2a_publish_next",
            loop,
        ), "the publish does not skip this rank"
        assert re.search(r"global_store_dword v\d+, v\d+, s\[\d+:\d+\] sc0 sc1", loop)
        assert tail.rstrip().endswith("s_mov_b32 s[sgprA2AShardIdx], 1                    // arm the drain-send wait")

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
