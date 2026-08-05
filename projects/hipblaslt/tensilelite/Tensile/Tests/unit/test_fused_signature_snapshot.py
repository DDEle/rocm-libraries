#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Emitted-metadata check for the fused-A2A kernarg segment.
#
# WHAT THIS PINS. fusedA2AKernArgLayout() states the segment's offsets as
# arithmetic. That arithmetic is only a claim about what rocisa will emit; the
# offsets the kernel actually gets come out of SignatureBase.addArg(), which
# packs and aligns on its own rules. This file drives a REAL SignatureBase,
# reads the offsets back out of the emitted metadata, and compares them to the
# table. A change to either side that the other does not match reddens here.
#
# WHAT IT DOES NOT PIN, and why the name is "snapshot" rather than "contract".
# The addArg sequence below is a hand-transcribed mirror of the fused block in
# SignatureDefault.__call__, not a call into it -- driving the real thing needs
# a full kernel dict and a Solution, which is a much larger fixture. So this
# closes the layout-vs-rocisa gap but NOT the mirror-vs-real-emitter gap: edit
# SignatureDefault.__call__'s fused block without editing the mirror below and
# nothing here goes red. Closing that too means building the real signature;
# until someone does, treat a green run here as "the arithmetic agrees with
# rocisa", not "the kernel gets these offsets".
#
# This began as an uncollected __main__ script, which meant the check existed,
# was correct, and never ran. That is why the filename now starts with test_.
################################################################################

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from rocisa.code import SignatureBase
from rocisa.enum import SignatureValueKind as SVK

# Import the Component package first so the Tensile.Components sub-package is
# fully initialised before we reach Signature; importing Signature directly as
# the very first Tensile module re-enters the Components package mid-init and
# trips its circular-import guard (pytest avoids this by loading Tensile first).
import Tensile.Component  # noqa: F401
import Tensile.Components.Signature as sigmod

FUSED_A2A_MAX_RANKS = sigmod.FUSED_A2A_MAX_RANKS
FUSED_A2A_SEGMENT_BYTES = sigmod.FUSED_A2A_SEGMENT_BYTES
fusedA2AKernArgLayout = sigmod.fusedA2AKernArgLayout

# The last argument of the fused block. Used to derive the segment's real extent
# from emitted metadata instead of restating 176 as a literal.
LAST_ARG = "FusedTokenTiles"

SDMA_TAIL = ["FusedSdmaQueues", "FusedTilesPerRank", "FusedTokenTiles"]


def build_fused_signature():
    """Emit just the fused-A2A addArg tail onto a fresh SignatureBase.

    Hand-mirrors SignatureDefault.__call__'s fused block -- see the header note
    on what that does and does not buy.
    """
    sig = SignatureBase(
        kernelName="fused_a2a_snapshot",
        kernArgsVersion=1,
        codeObjectVersion="default",
        groupSegmentSize=0,
        sgprWorkGroup=[1, 0, 0],
        vgprWorkItem=0,
        flatWorkGroupSize=256,
    )
    for j in range(FUSED_A2A_MAX_RANKS):
        sig.addArg("recv_ptr_%u" % j, SVK.SIG_GLOBALBUFFER, "void", "generic")
    for j in range(FUSED_A2A_MAX_RANKS):
        sig.addArg("flag_ptr_%u" % j, SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("counter_ptr", SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("FusedMyRank", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedTarget", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedW", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedNShard", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedDrain", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedAM", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedSdmaQueues", SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("FusedTilesPerRank", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedTokenTiles", SVK.SIG_VALUE, "u32")
    return sig


def parse_args(text):
    """Pull {argName: (offset, size)} out of emitted .args metadata.

    Order-independent: the emitter writes .name/.size/.offset, but nothing in the
    format guarantees that order, so each arg is committed once both numbers have
    been seen rather than at a fixed line. The kernel's own leading `.name:` entry
    carries neither and is dropped.
    """
    args = {}
    name, pending = None, {}
    for line in text.splitlines():
        s = line.strip()
        if ".name:" in s:
            name, pending = s.split(".name:", 1)[1].split(",")[0].strip(), {}
        elif name is not None and (s.startswith(".offset:") or s.startswith(".size:")):
            key = "offset" if s.startswith(".offset:") else "size"
            pending[key] = int(s.split(":", 1)[1].split(",")[0].strip())
            if len(pending) == 2:
                args[name] = (pending["offset"], pending["size"])
                name, pending = None, {}
    return args


def _emitted():
    """Emitted metadata and the layout table, with the scrape itself guarded.

    A metadata format change would make parse_args return {}, and a comparison
    driven off an empty set passes vacuously. Same for an empty layout table.
    Both are asserted here so that failure mode is loud rather than green.
    """
    args = parse_args(str(build_fused_signature()))
    assert args, "parsed no args out of the emitted metadata; the .name/.offset/" \
                 ".size format has likely changed, so nothing below checks anything"
    layout = fusedA2AKernArgLayout()
    assert layout, "fusedA2AKernArgLayout() returned an empty table; the comparisons " \
                   "below would pass vacuously"
    return args, layout


def test_emitted_offsets_match_the_layout_table():
    """Every offset fusedA2AKernArgLayout() predicts is the one rocisa emits."""
    args, layout = _emitted()
    mismatched = {n: (args.get(n, (None, None))[0], off)
                  for n, off in layout.items() if args.get(n, (None, None))[0] != off}
    assert not mismatched, \
        "emitted offset != fusedA2AKernArgLayout(), as {arg: (emitted, expected)}: %r" \
        % mismatched


def test_sdma_args_are_appended_last():
    """The three SDMA args sit at the end, so no earlier offset can move.

    Append-only is what lets a host built against an older layout keep working
    for every argument it already knew about.
    """
    args, _ = _emitted()
    byOffset = sorted(args, key=lambda n: args[n][0])
    assert byOffset[-3:] == SDMA_TAIL, \
        "SDMA args are not the last three; emitted order is %r" % byOffset


def test_segment_bytes_matches_the_emitted_extent():
    """FUSED_A2A_SEGMENT_BYTES equals where the emitted segment actually ends.

    Derived from metadata rather than compared against a transcribed 176, so it
    fails if either the constant or the argument list moves without the other.
    """
    args, _ = _emitted()
    offset, size = args[LAST_ARG]
    assert offset + size == FUSED_A2A_SEGMENT_BYTES, \
        "FUSED_A2A_SEGMENT_BYTES=%d but the segment ends at %d (%s @ %d + %d)" \
        % (FUSED_A2A_SEGMENT_BYTES, offset + size, LAST_ARG, offset, size)
