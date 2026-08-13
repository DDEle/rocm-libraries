#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Emitted-metadata check for the fused-A2A kernarg segment.
#
# Drives a real SignatureBase and compares fusedA2AKernArgLayout()'s offsets
# against what rocisa actually emits.
#
# The addArg sequence below is a hand-transcribed mirror of the fused block in
# SignatureDefault.__call__, not a call into it, so it does not catch the two
# drifting out of sync with each other.
################################################################################

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from rocisa.code import SignatureBase
from rocisa.enum import SignatureValueKind as SVK

# Must import Component before Signature, or the circular-import guard trips.
import Tensile.Component  # noqa: F401
import Tensile.Components.Signature as sigmod

FUSED_A2A_MAX_RANKS = sigmod.FUSED_A2A_MAX_RANKS
FUSED_A2A_SEGMENT_BYTES = sigmod.FUSED_A2A_SEGMENT_BYTES
fusedA2AKernArgLayout = sigmod.fusedA2AKernArgLayout

# Last argument of the fused block.
LAST_ARG = "FusedTokenTiles"

# All pointer args, in emitted order (contiguous at the front).
POINTER_ARGS = ["peer_ptr_%u" % j for j in range(FUSED_A2A_MAX_RANKS)] + \
    ["counter_ptr", "FusedSdmaQueues"]


def build_fused_signature():
    """Emit just the fused-A2A addArg tail onto a fresh SignatureBase, hand-mirroring
    SignatureDefault.__call__'s fused block (see the header note)."""
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
        sig.addArg("peer_ptr_%u" % j, SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("counter_ptr",     SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("FusedSdmaQueues", SVK.SIG_GLOBALBUFFER, "void", "generic")
    sig.addArg("FusedMyRank",       SVK.SIG_VALUE, "u32")
    sig.addArg("FusedW",            SVK.SIG_VALUE, "u32")
    sig.addArg("FusedNShard",       SVK.SIG_VALUE, "u32")
    sig.addArg("FusedDrain",        SVK.SIG_VALUE, "u32")
    sig.addArg("FusedAM",           SVK.SIG_VALUE, "u32")
    sig.addArg("FusedTilesPerRank", SVK.SIG_VALUE, "u32")
    sig.addArg("FusedTokenTiles",   SVK.SIG_VALUE, "u32")
    return sig


def parse_args(text):
    """Pull {argName: (offset, size)} out of emitted .args metadata.

    Order-independent: commits an arg once both .offset and .size have been seen,
    rather than at a fixed line. The kernel's own leading `.name:` entry carries
    neither and is dropped.
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
    """Emitted metadata and the layout table, both guarded against being empty."""
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


def test_pointers_are_appended_first():
    """Every pointer arg sits contiguously at the front, ahead of any scalar."""
    args, _ = _emitted()
    byOffset = sorted(args, key=lambda n: args[n][0])
    assert byOffset[:len(POINTER_ARGS)] == POINTER_ARGS, \
        "pointer args are not the leading block; emitted order is %r" % byOffset


def test_segment_bytes_matches_the_emitted_extent():
    """FUSED_A2A_SEGMENT_BYTES equals where the emitted segment actually ends."""
    args, _ = _emitted()
    offset, size = args[LAST_ARG]
    assert offset + size == FUSED_A2A_SEGMENT_BYTES, \
        "FUSED_A2A_SEGMENT_BYTES=%d but the segment ends at %d (%s @ %d + %d)" \
        % (FUSED_A2A_SEGMENT_BYTES, offset + size, LAST_ARG, offset, size)
