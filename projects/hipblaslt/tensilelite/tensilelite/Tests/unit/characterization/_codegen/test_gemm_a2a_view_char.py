# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 FusedA2A: GemmA2A with FusedA2ATransport: View (SignalView) codegen.

Emits the kernels of Tests/common/comm/gfx950/fused_a2a.yaml with an extra
``FusedA2ATransport: [View]`` fork (UseSubtileImpl True and False) and checks:

- no SDMA ring or packet code, no counter2/counter3 tally, no DRAIN, no
  cursor or flag clears;
- every PUSH-pass store body ends with the SignalView sequence: EXEC restore,
  ``s_waitcnt vmcnt(0)``, ``s_barrier``, the wave-0 election, the dst_rank scan,
  the ``(dst_rank*tokenTiles + j)*4`` address off FusedCounterPtr, and one
  ``s_atomic_inc ... 0x100 glc`` with SDATA = -1;
- LOCAL-pass store bodies emit no signal;
- every D store of a PUSH pass carries ``sc1``, none of a LOCAL pass does;
- FusedTotalWGs and FusedPeerGroupPtr are not defined and
  ``.amdhsa_next_free_sgpr`` is at most 102;
- the kernels assemble for gfx950.
"""

import os
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_SGPR_LIMIT = 102

_BASE_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx950", "fused_a2a.yaml",
    )
)

_USI = re.compile(r"\.amdhsa_kernel \S*_USI(\d+)_")
_NEXT_FREE_SGPR = re.compile(r"^\s*\.amdhsa_next_free_sgpr (\d+)", re.M)
_D_STORE = re.compile(r"^buffer_store\S*\s.*s\[sgprSrdD:sgprSrdD\+3\]")
_MEMORY_OP = re.compile(r"^(buffer_|global_|flat_|ds_|s_load|s_store|s_buffer|s_atomic)")
_LABEL = re.compile(r"^\S+:")
_PUSH_GATE = re.compile(r"^s_cbranch_scc1 (label_fusedA2A_hoist_local(_\d+)?)\s")

# Absent from every View kernel; each is emitted only by the InKernel handshake.
_FORBIDDEN = (
    "0x80000401",            # COPY_SUBWIN header
    "0x1e00000a",            # ATOMIC ADD_RTN_32 header
    "s_atomic_umax_x2",
    "s_atomic_cmpswap_x2",
    "s_bfm_b64",
    "s_store_dword",
    "s_sleep",
    "// clear",
    "clear SDMA cursors",
    "counter2",
    "counter3",
    "sgprFusedTotalWGs",
    "sgprFusedPeerGroupPtr",
)

# Mnemonics from the SignalView EXEC restore through the PUSH pass's branch over
# the LOCAL pass. FUSED_A2A_MAX_RANKS = 8 gives 7 scan steps.
_SIGNAL_VIEW = (
    ["s_mov_b64", "s_waitcnt", "s_barrier",
     "v_readfirstlane_b32", "s_cmp_eq_u32", "s_cbranch_scc0", "s_mov_b64",
     "s_lshl_b32", "s_mov_b32"]
    + ["s_mul_i32", "s_cmp_le_u32", "s_cselect_b32"] * 7
    + ["s_mul_i32", "s_add_u32", "s_lshl_b32", "s_add_u32", "s_addc_u32",
       "s_mov_b32", "s_atomic_inc", "s_waitcnt",
       "s_mov_b64", "s_branch"]
)


@pytest.fixture(scope="module")
def view_kernels(tmp_path_factory):
    with open(_BASE_CONFIG) as fh:
        config = yaml.safe_load(fh)
    config["BenchmarkProblems"][0][1]["ForkParameters"].append({"FusedA2ATransport": ["View"]})
    path = tmp_path_factory.mktemp("gemm_a2a_view") / "fused_a2a_view.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    results = emit_kernels_from_config(str(path), limit=8, arch=_ARCH, expected_fork_count=2)
    assert len(results) == 2, [b for (b, _s, _e) in results]
    assert all(err == 0 for (_b, _s, err) in results), [(b, e) for (b, _s, e) in results]
    return [(b, s) for (b, s, _e) in results]


def _instructions(src):
    return [l for l in src.splitlines() if l and not l.startswith(("/", " ", "\t", "."))]


def _passes(ins):
    """[(push, local)] instruction slices of every hoisted PUSH/LOCAL store body."""
    out = []
    i = 0
    while i < len(ins):
        m = _PUSH_GATE.match(ins[i])
        if not m:
            i += 1
            continue
        local, suffix = m.group(1), m.group(2) or ""
        j = next(k for k in range(i, len(ins))
                 if ins[k].startswith(f"s_branch label_fusedA2A_hoist_after{suffix} "))
        k0 = next(k for k in range(j, len(ins)) if ins[k].startswith(f"{local}:"))
        k1 = next(k for k in range(k0, len(ins))
                  if ins[k].startswith(f"label_fusedA2A_hoist_after{suffix}:"))
        out.append((ins[i + 1:j + 1], ins[k0 + 1:k1]))
        i = k1
    return out


def test_gemm_a2a_view_covers_both_subtile_paths(view_kernels):
    usi = sorted(_USI.search(src).group(1) for (_b, src) in view_kernels)
    assert usi == ["0", "1"]
    for base, src in view_kernels:
        assert "_FAATV_" in src, f"{base}: View kernel name lacks _FAATV_"


def test_gemm_a2a_view_drops_in_kernel_transport(view_kernels):
    for base, src in view_kernels:
        low = src.lower()
        present = [m for m in _FORBIDDEN if m.lower() in low]
        assert not present, f"{base}: View kernel still emits {present}"


def test_gemm_a2a_view_signal_per_push_pass(view_kernels):
    """One s_atomic_inc per PUSH-pass store body (one per edge/beta/OptNLL store
    variant), none in the LOCAL pass and none elsewhere."""
    for base, src in view_kernels:
        ins = _instructions(src)
        passes = _passes(ins)
        assert passes, f"{base}: no hoisted PUSH/LOCAL store bodies"
        atomics = [l for l in ins if l.startswith("s_atomic")]
        assert len(atomics) == len(passes), (
            f"{base}: {len(atomics)} atomics for {len(passes)} PUSH passes")
        for push, local in passes:
            assert not any(l.startswith("s_atomic") for l in local), f"{base}: LOCAL pass signals"
            assert sum(l.startswith("s_atomic") for l in push) == 1, f"{base}: PUSH pass atomics"

            start = next(k for k in range(len(push) - 1, -1, -1)
                         if push[k].startswith("s_mov_b64 exec, -1")
                         and push[k + 1].startswith("s_waitcnt vmcnt(0)"))
            tail = [l for l in push[start:] if not _LABEL.match(l)]
            assert [l.split()[0] for l in tail] == _SIGNAL_VIEW, (
                f"{base}: SignalView sequence differs:\n" + "\n".join(tail))
            assert tail[1].startswith("s_waitcnt vmcnt(0)")
            assert tail[-4].startswith("s_atomic_inc")
            assert tail[-3].startswith("s_waitcnt lgkmcnt(0)")
            assert tail[-2].startswith("s_mov_b64 exec, -1")

            stores = [k for k, l in enumerate(push) if _D_STORE.match(l)]
            if stores:
                between = [l for l in push[stores[-1] + 1:start] if _MEMORY_OP.match(l)]
                assert not between, f"{base}: {between} between last D store and signal"

            data = re.match(r"s_mov_b32 (s\d+), -1\s", tail[-5])
            assert data, f"{base}: SDATA is not -1: {tail[-5]}"
            atomic = re.match(r"s_atomic_inc (s\d+), s\[(\d+):(\d+)\], 0x100 glc(\s|$)", tail[-4])
            assert atomic and atomic.group(1) == data.group(1), f"{base}: {tail[-4]}"
            lo, hi = int(atomic.group(2)), int(atomic.group(3))
            assert re.match(rf"s_add_u32 s{lo}, s\[sgprFusedCounterPtr\], s\d+\s", tail[-7]), tail[-7]
            assert re.match(rf"s_addc_u32 s{hi}, s\[sgprFusedCounterPtr\+1\], 0\s", tail[-6]), tail[-6]
            assert re.match(r"s_mul_i32 (s\d+), s\d+, s\[sgprFusedTokenTiles\]\s", tail[-10]), tail[-10]
            assert re.match(r"s_add_u32 (s\d+), \1, s\[sgprWorkGroup1\]\s", tail[-9]), tail[-9]
            assert re.match(r"s_lshl_b32 (s\d+), \1, 2\s", tail[-8]), tail[-8]


def test_gemm_a2a_view_push_stores_sc1(view_kernels):
    for base, src in view_kernels:
        for push, local in _passes(_instructions(src)):
            pushStores = [l for l in push if _D_STORE.match(l)]
            localStores = [l for l in local if _D_STORE.match(l)]
            assert pushStores and len(pushStores) == len(localStores), base
            assert all(re.search(r"\ssc1(\s|$)", l) for l in pushStores), base
            assert not any(re.search(r"\ssc1(\s|$)", l) for l in localStores), base


def test_gemm_a2a_view_sgpr_budget(view_kernels):
    for base, src in view_kernels:
        found = _NEXT_FREE_SGPR.findall(src)
        assert len(found) == 1, f"{base}: .amdhsa_next_free_sgpr missing"
        assert int(found[0]) <= _SGPR_LIMIT, f"{base}: next_free_sgpr {found[0]}"


def test_gemm_a2a_view_assembles(view_kernels):
    for base, src in view_kernels:
        assert_assembles(src, base)
