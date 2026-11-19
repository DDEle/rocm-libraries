# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Order of the FusedA2ATransport=View host calls in the tensilelite client.

Reads the client sources from the source tree, which the installed test
artifact does not stage.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CLIENT = Path(__file__).resolve().parents[4] / "client"


def _read(rel):
    return (_CLIENT / rel).read_text(encoding="utf-8")


def _block(text, opener):
    start = text.index(opener)
    at = text.index("{", start)
    depth = 0
    for i in range(at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    raise AssertionError("unbalanced block after " + opener)


def _in_order(text, *needles):
    at = -1
    for needle in map(_squash, needles):
        nxt = text.find(needle, at + 1)
        assert nxt > at, "%r not found after offset %d" % (needle, at)
        at = nxt


def _squash(text):
    return re.sub(r"\s+", "", text)


def test_warmup_events_bracket_arm_gemm_and_wait():
    launch_all = _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))
    view = _block(launch_all, "if(viewTransport)")
    _in_order(
        view,
        "hipEventRecord(startEvents->card(d)[invocation].front(),ctx.stream)",
        "viewArm(view,ctx.stream,viewArmMode)",
        "launchKernels(launch[d],ctx.stream,nullptr,nullptr)",
        "viewWait(view,ctx.stream)",
        "hipEventRecord(stopEvents->card(d)[invocation].back(),ctx.stream)",
    )


def test_failed_gemm_launch_releases_the_armed_queues():
    launch_all = _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))
    view = _block(launch_all, "if(viewTransport)")
    _in_order(
        view, "try{viewArm(", "launchKernels(", "}catch(...){viewAbort(view,ctx.stream);throw;}"
    )


def test_queues_drain_before_view_blocks_are_reset_or_freed():
    main = _squash(_read("main.cpp"))
    _in_order(
        main,
        "catch(std::runtime_error const& err)",
        "viewDrainQueues(*ctx)",
        "listeners.postSolution()",
    )
    context = _read("src/DeviceContext.cpp")
    reset = _squash(_block(context, "void resetCommBuffers("))
    _in_order(reset, "viewDrainQueues(*ctx)", "hipMemset(")
    release = _squash(_block(context, "void DeviceContext::release()"))
    _in_order(release, "viewDrainQueues(*this)", "hipFree(viewBlock)")


def test_wait_releases_to_the_epoch_threshold_and_drains():
    wait = _squash(_block(_read("src/ViewTransport.cpp"), "__global__ void viewWaitKernel("))
    assert "0xFFFFFFFF" not in wait
    _in_order(
        wait,
        "if(viewLoad32(error)!=0)return;",
        "viewReleaseSig(",
        "viewQueuesDrained(a)",
        "VIEW_ERROR_STUCK",
    )


def test_arm_kernel_books_halted_and_skipped_queues():
    arm = _squash(_block(_read("src/ViewTransport.cpp"), "__global__ void viewArmKernel("))
    _in_order(
        arm,
        "viewLoad32(error)",
        "__syncthreads()",
        "viewArmOrBook(a,q,epoch,halt,viewAdd32,viewStore32)",
    )
    host = _squash(_block(_read("src/ViewTransport.cpp"), "void viewArmHost("))
    _in_order(
        host,
        "viewArmHalted(a,error)",
        "viewResetFlag(a,q,storeOnDevice)",
        "viewBookSkipped(a,q,addOnDevice)",
        "viewArmQueue(a,q,epoch)",
    )
