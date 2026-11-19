# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The Python FusedA2ATransport ids and view-block offsets match FusedA2AKernArg.hpp.

Reads the TensileLite headers from the source tree, which the installed test
artifact does not stage.
"""

import re
from pathlib import Path

import pytest

from tensilelite import Contractions as C
from tensilelite.Components import Signature as S

pytestmark = pytest.mark.unit

_HEADER = Path(__file__).resolve().parents[4] / "include" / "Tensile" / "FusedA2AKernArg.hpp"


def _cpp_constants(ctype, prefix):
    text = _HEADER.read_text(encoding="utf-8")
    return {
        name: int(value)
        for name, value in re.findall(
            r"constexpr\s+%s\s+(%s\w+)\s*=\s*(\d+);" % (ctype, prefix), text
        )
    }


def test_view_offsets_match_cpp():
    py = {name: getattr(S, name) for name in dir(S) if name.startswith("FUSED_A2A_VIEW_")}
    assert py == _cpp_constants("size_t", "FUSED_A2A_VIEW_")
    assert py == {
        "FUSED_A2A_VIEW_FLAG_OFFSET": 0,
        "FUSED_A2A_VIEW_BASE_OFFSET": 64,
        "FUSED_A2A_VIEW_EXPECTED_OFFSET": 128,
        "FUSED_A2A_VIEW_OUTBOUND_OFFSET": 192,
        "FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET": 196,
        "FUSED_A2A_VIEW_EPOCH_OFFSET": 200,
        "FUSED_A2A_VIEW_ERROR_OFFSET": 204,
        "FUSED_A2A_VIEW_SIG_OFFSET": 256,
    }
    assert S.FUSED_A2A_VIEW_BASE_OFFSET >= S.FUSED_A2A_MAX_RANKS * 4
    assert S.FUSED_A2A_VIEW_EXPECTED_OFFSET - S.FUSED_A2A_VIEW_BASE_OFFSET >= S.FUSED_A2A_MAX_RANKS * 4
    assert S.FUSED_A2A_VIEW_OUTBOUND_OFFSET - S.FUSED_A2A_VIEW_EXPECTED_OFFSET >= S.FUSED_A2A_MAX_RANKS * 4


def test_transport_ids_match_cpp():
    assert _cpp_constants("int", "FUSED_A2A_TRANSPORT_") == {
        "FUSED_A2A_TRANSPORT_" + re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper(): value
        for name, value in C.FUSED_A2A_TRANSPORT_IDS.items()
    }
