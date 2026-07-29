#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# str(signature) snapshot for the fused-A2A kernarg segment (Task 3 verification).
#
# Builds a real rocisa SignatureBase, emits ONLY the fused-A2A addArg sequence
# (the exact tail SignatureDefault.__call__ appends when kernel["FusedGemmA2A"]
# is set), and prints the resulting .name/.size/.offset metadata. This confirms
# the three new SDMA args (FusedSdmaQueues / FusedTilesPerRank / FusedTokenTiles)
# appear at the END with offsets matching fusedA2AKernArgLayout(). Run in the
# container (needs rocisa built):
#   python Tensile/Tests/unit/snapshot_fused_signature.py

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


def build_fused_signature():
    """Emit just the fused-A2A addArg tail onto a fresh SignatureBase."""
    sig = SignatureBase(
        kernelName="fused_a2a_snapshot",
        kernArgsVersion=1,
        codeObjectVersion="default",
        groupSegmentSize=0,
        sgprWorkGroup=[1, 0, 0],
        vgprWorkItem=0,
        flatWorkGroupSize=256,
    )
    # Exact mirror of SignatureDefault.__call__ fused block.
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


def parse_offsets(text):
    """Pull {argName: offset} out of emitted .args metadata."""
    offsets = {}
    name = None
    for line in text.splitlines():
        s = line.strip()
        if ".name:" in s:
            name = s.split(".name:", 1)[1].split(",")[0].strip()
        elif s.startswith(".offset:") and name is not None:
            offsets[name] = int(s.split(":", 1)[1].split(",")[0].strip())
    return offsets


def main():
    sig = build_fused_signature()
    text = str(sig)
    got = parse_offsets(text)
    layout = fusedA2AKernArgLayout()

    print("=== fused-A2A addArg metadata (name @ offset) ===")
    for name in layout:
        emitted = got.get(name, "<missing>")
        expected = layout[name]
        mark = "OK" if emitted == expected else "MISMATCH"
        print("  %-18s emitted=%-6s golden=%-6d %s" % (name, str(emitted), expected, mark))

    sdma = ["FusedSdmaQueues", "FusedTilesPerRank", "FusedTokenTiles"]
    print("\n=== SDMA tail (must be last three, offsets 160/168/172) ===")
    for name in sdma:
        print("  %-18s @ %s" % (name, got.get(name)))

    print("\nFUSED_A2A_SEGMENT_BYTES =", FUSED_A2A_SEGMENT_BYTES)

    ok = all(got.get(n) == layout[n] for n in layout) and FUSED_A2A_SEGMENT_BYTES == 176
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
