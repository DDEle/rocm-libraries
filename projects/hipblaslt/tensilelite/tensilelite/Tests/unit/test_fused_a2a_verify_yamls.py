# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The --fused-a2a-verify yamls are their comm yamls with one warmup and 22 enqueues."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_COMM = Path(__file__).resolve().parent.parent / "common" / "comm" / "gfx950"


def _load(name):
    return yaml.safe_load((_COMM / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("base", ["a2a_gemm_view", "fused_a2a_view"])
def test_verify_yaml_matches_its_comm_yaml(base):
    plain = _load(base + ".yaml")
    verify = _load(base + "_verify.yaml")
    assert verify["GlobalParameters"]["NumWarmups"] == 1
    assert verify["GlobalParameters"]["EnqueuesPerSync"] == 22
    assert "skip" in verify["TestParameters"]["marks"]
    assert "skip" not in plain["TestParameters"]["marks"]
    for doc in (plain, verify):
        doc.pop("TestParameters")
        for key in ("NumWarmups", "EnqueuesPerSync"):
            doc["GlobalParameters"].pop(key, None)
    assert verify == plain
