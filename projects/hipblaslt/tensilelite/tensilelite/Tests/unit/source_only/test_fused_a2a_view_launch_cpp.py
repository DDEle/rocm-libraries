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
    view = _block(
        launch_all[launch_all.index("try{ScopedDevicedevice(ctx.deviceId);if(viewTransport)") :],
        "if(viewTransport)",
    )
    _in_order(
        view,
        "hipEventRecord(startEvents->card(d)[invocation].front(),ctx.stream)",
        "viewArm(view,ctx.stream,viewArmMode)",
        "launchKernels(launch[d],ctx.stream,nullptr,nullptr)",
        "viewWait(view,ctx.stream)",
        "hipEventRecord(stopEvents->card(d)[invocation].back(),ctx.stream)",
    )


def test_one_try_covers_the_whole_device_body_and_aborts_every_later_device():
    launch_all = _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))
    loop = launch_all[launch_all.index("for(size_td=0;d<active.size();d++){auto&ctx=*active[d];") :]
    _in_order(
        loop,
        "ViewArmArgsview{};boolarmed=false;if(viewTransport)view=viewArgsOf(d);"
        "try{ScopedDevicedevice(ctx.deviceId);if(viewTransport){",
        "throwAt(ViewThrowSite::Start,d);",
        "hipEventRecord(startEvents->card(d)[invocation].front(),ctx.stream)",
        "throwAt(ViewThrowSite::Arm,d);",
        "viewArm(view,ctx.stream,viewArmMode);armed=true;",
        "throwAt(ViewThrowSite::Launch,d);",
        "launchKernels(launch[d],ctx.stream,nullptr,nullptr)",
        "throwAt(ViewThrowSite::Wait,d);",
        "viewWait(view,ctx.stream);",
        "throwAt(ViewThrowSite::Stop,d);",
        "hipEventRecord(stopEvents->card(d)[invocation].back(),ctx.stream)",
        "throwAt(ViewThrowSite::Check,d);",
        "verifyCheck(ctx,verifyLayoutOf,dataSlot,verifyLaunch);if(!joinSplit)publish(d);"
        "}catch(...){if(viewTransport){static_cast<void>(hipSetDevice(ctx.deviceId));"
        "viewAbort(view,ctx.stream,armed);for(size_te=d+1;e<active.size();e++){"
        "static_cast<void>(hipSetDevice(active[e]->deviceId));"
        "viewAbort(viewArgsOf(e),active[e]->stream,false);}}throw;}}",
    )
    assert launch_all.count("viewAbort(") == 2
    assert launch_all.count("publish(d)") == 1


def test_launches_stop_below_the_launch_limit():
    launch_all = _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))
    _in_order(
        launch_all,
        "constuint64_tviewLaunch=viewLaunches;if(viewTransport){",
        "constuint64_tlimit=viewLaunchLimit(viewArgs[launchSlot][0]);if(viewLaunch>=limit)throw",
        "viewLaunches++;}",
        "for(size_td=0;d<active.size();d++)",
    )
    main = _squash(_read("main.cpp"))
    _in_order(main, "uint64_tviewLaunches=0;", "while(listeners.needMoreRunsInSolution())")


def test_error_path_releases_every_tail_before_a_bounded_sync():
    main = _squash(_read("main.cpp"))
    catch = main[main.index("catch(std::runtime_errorconst&err){reporters->report(ResultKey::Validation") :]
    _in_order(
        catch,
        "for(autoconst&ctx:devices)if(!viewReleaseTailHost(*ctx,viewRelease))",
        "for(autoconst&ctx:devices){hipError_tsyncErr=hipSetDevice(ctx->deviceId);",
        "if(syncErr==hipSuccess)syncErr=ctx->viewQueuesArmed?viewSyncBounded(*ctx):hipDeviceSynchronize();",
        "viewDrainQueues(*ctx)",
    )
    first_sync = catch.index("hipDeviceSynchronize")
    assert catch.index("viewReleaseTailHost") < first_sync
    vt = _read("src/ViewTransport.cpp")
    release = _squash(_block(vt, "bool viewReleaseTailHost("))
    _in_order(
        release,
        "returnviewReleaseTail(tail,path,viewRelease32,[&](uint32_t*to,size_tbytes){"
        "returnhipMemcpyAsync(to,ctx.viewReleaseHost,bytes,hipMemcpyHostToDevice,"
        "ctx.viewReleaseStream)==hipSuccess;},",
        "[&](uint32_t*to,size_tcount,uint32_tvalue){static_cast<void>(hipGetLastError());"
        "hipLaunchKernelGGL(viewFillKernel,::dim3(1),::dim3(64),0,ctx.viewReleaseStream,to,count,"
        "value);returnhipGetLastError()==hipSuccess;});",
    )
    assert "ctx.stream" not in release
    sync = _squash(_block(vt, "hipError_t viewSyncBounded("))
    assert sync == _squash(
        """hipError_t viewSyncBounded(DeviceContext& ctx) noexcept
        {
            return viewSyncWithin(
                [&] {
                    const hipError_t e = hipStreamQuery(ctx.stream);
                    return e != hipSuccess || ctx.viewReleaseStream == nullptr
                               ? e
                               : hipStreamQuery(ctx.viewReleaseStream);
                },
                [] { return hipDeviceSynchronize(); },
                viewNow,
                viewSleep,
                std::chrono::milliseconds(kViewSyncBoundMs));
        }"""
    )
    assert "std::this_thread::sleep_for(std::chrono::milliseconds(1));" in _squash(
        _block(vt, "void viewSleep(")
    )
    drain = _squash(_block(vt, "bool viewDrainQueues("))
    assert "hipStreamSynchronize" not in drain
    _in_order(
        drain,
        "viewFillKernel",
        "if(viewPollIdle([&]{returnhipStreamQuery(ctx.stream);},viewNow,viewSleep,deadline)"
        "!=hipSuccess)returnfalse;",
    )
    context = _squash(_read("src/DeviceContext.cpp"))
    _in_order(
        context,
        "hipHostMalloc(&ctx.viewReleaseHost,VIEW_TAIL_RELEASE_BYTES)",
        "viewFillReleaseSource(ctx.viewReleaseHost);",
    )


def test_a_busy_device_returns_before_the_listeners_sync():
    main = _squash(_read("main.cpp"))
    catch = main[main.index("catch(std::runtime_errorconst&err){reporters->report(ResultKey::Validation") :]
    _in_order(
        catch,
        "deviceLost=true;deviceBusy=deviceBusy||syncErr==hipErrorNotReady;",
        "if(!viewDrainQueues(*ctx)){deviceLost=true;deviceBusy=true;",
        "if(deviceBusy){flushTimingBuffer();std::cout.flush();std::cerr.flush();std::clog.flush();"
        "std::_Exit(std::clamp(listeners.error(),1,255));}",
        "listeners.postSolution();",
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
    assert (
        "if(viewDrainQueues(*this)){for(void*p:{viewBlock,viewStats,viewDelayScratch})"
        "if(p!=nullptr)static_cast<void>(hipFree(p));if(viewReleaseHost!=nullptr)"
        "static_cast<void>(hipHostFree(viewReleaseHost));if(viewReleaseStream!=nullptr)"
        "static_cast<void>(hipStreamDestroy(viewReleaseStream));}" in release
    )
    assert release.count("viewDrainQueues(") == 1


def test_wait_releases_to_the_epoch_threshold_and_drains():
    wait = _squash(_block(_read("src/ViewTransport.cpp"), "__global__ void viewWaitKernel("))
    assert "0xFFFFFFFF" not in wait
    _in_order(
        wait,
        "if(viewLoad32(error)!=0)return;",
        "while(!viewWaitDone(a,epoch))",
        "if(a.mode!=VIEW_MODE_GEMM_A2A)return;",
        "viewReleaseSig(a,epoch);",
        "viewReleaseEntry(a,epoch,viewMax32);",
        "viewQueuesDrained(a)",
        "VIEW_ERROR_STUCK",
    )
    vt = _read("src/ViewTransport.cpp")
    host = _squash(_block(vt, "void viewWait("))
    assert "if(args.mode!=VIEW_MODE_GEMM_A2A&&!viewExitWaits(args)&&args.stats==nullptr)return;" in host
    assert "viewStampKernel" not in vt
    abort = _squash(_block(vt, "void viewAbort("))
    _in_order(
        abort,
        "if(!armed)hipLaunchKernelGGL(viewArmKernel<0>,::dim3(1),::dim3(64),0,stream,args,true);",
        "elsehipLaunchKernelGGL(viewWaitKernel,::dim3(1),::dim3(64),0,stream,args,true);",
    )
    arm = _squash(_block(vt, "void viewArm("))
    _in_order(arm, "else{static_cast<void>(hipGetLastError());if(", "HIP_CHECK_EXC(hipGetLastError());")


def test_arm_kernel_books_halted_and_skipped_queues():
    arm = _squash(_block(_read("src/ViewTransport.cpp"), "__global__ void viewArmKernel("))
    _in_order(
        arm,
        "viewLoad32(error)",
        "__syncthreads()",
        "viewArmThread<Faults>(a,q,epoch,halt,bookOnly,viewAdd32,viewStore32,viewRelease32)",
        "__syncthreads()",
        "if(bookOnly)__hip_atomic_fetch_or(error,VIEW_ERROR_ABORTED,",
        "viewAdvance(a,epoch,view);",
    )
    thread = _squash(_block(_read("include/ViewTransport.hpp"), "inline uint32_t viewArmThread("))
    _in_order(thread, "viewPublishEntry(a,q,epoch,release);", "returnviewArmOrBook<Faults>(")
    host = _squash(_block(_read("src/ViewTransport.cpp"), "void viewArmHost("))
    _in_order(
        host,
        "HIP_CHECK_EXC(hipStreamSynchronize(stream));",
        "uint32_t&error=header[FUSED_A2A_VIEW_ERROR_OFFSET/sizeof(uint32_t)];"
        "error|=viewArmHostQueues(a,epoch,error,addOnDevice,storeOnDevice);"
        "viewAdvance(a,epoch,header);",
    )


def test_a2a_gemm_arm_reads_sub_rounds_and_peer_groups_from_the_size_mapping():
    make = _squash(_block(_read("src/ViewTransport.cpp"), "ViewArmArgs makeViewArmArgs("))
    a2aGemm = _block(make, "if(gemm.fusedA2A()==FusedA2A::A2AGemm)")
    _in_order(
        a2aGemm,
        "constuint64_tsubRounds=solution.sizeMapping.a2aSubRounds;",
        "a.subRounds=(uint32_t)subRounds;",
        "a.peerGroups=(uint32_t)solution.sizeMapping.a2aPeerGroups;",
    )
    assert make.count("a.peerGroups=") == 1


def test_arm_args_point_every_queue_at_the_tails():
    make = _squash(_block(_read("src/ViewTransport.cpp"), "ViewArmArgs makeViewArmArgs("))
    _in_order(
        make,
        "constsize_ttailOffset=fusedA2AViewTailOffset(problem,solution);",
        "std::vector<uint32_t*>tails(world);for(size_tj=0;j<world;j++){",
        "devices[j]->viewBlockBytes<tailOffset+VIEW_TAIL_BYTES)throw",
        "tails[j]=viewTailOf(devices[j]->viewBlock,tailOffset);}",
        "if((debug&VIEW_DEBUG_DELAY_CHAIN)!=0&&delayRank>=world)throwstd::runtime_error(",
        "if((debug&VIEW_DEBUG_DELAY_CHAIN)!=0&&rank==delayRank)",
        "a.delayScratch=viewAddress(own.viewDelayScratch);",
        "viewWireTails(a,tails.data());returna;",
    )
    for word in ("a.tail=", ".ready=", ".entryOut=", ".pulled="):
        assert word not in make, word
    reset = _squash(_block(_read("src/DeviceContext.cpp"), "void resetCommBuffers("))
    _in_order(
        reset,
        "constsize_tviewBytes=viewTail!=0?viewTail+VIEW_TAIL_BYTES:0;",
        "hipStreamCreateWithFlags(&ctx.viewReleaseStream,hipStreamNonBlocking)",
        "hipMemset(ctx.viewBlock,0,ctx.viewBlockBytes)",
        "ctx.viewTailOffset=viewTail;",
        "if(viewBytes!=0)forEachDevice(devices,[](DeviceContext&){HIP_CHECK_EXC(hipDeviceSynchronize());});",
    )


def test_the_tail_is_armed_only_once_the_view_block_holds_it():
    reset = _squash(_block(_read("src/DeviceContext.cpp"), "void resetCommBuffers("))
    view = reset[reset.index("if(viewBytes!=0){") :]
    sized = "HIP_CHECK_EXC(hipMemset(ctx.viewBlock,0,ctx.viewBlockBytes));"
    before = view[: view.index(sized)]
    for step in (
        "hipStreamCreateWithFlags(&ctx.viewReleaseStream",
        "hipHostMalloc(&ctx.viewReleaseHost",
        "hipMalloc(&ctx.viewDelayScratch",
        "hipExtMallocWithFlags(&ctx.viewBlock,viewBytes,hipDeviceMallocFinegrained)",
    ):
        assert step in before, step
    assert "ctx.viewQueuesArmed" not in before
    assert "ctx.viewTailOffset" not in before
    _in_order(view, sized, "ctx.viewQueuesArmed=true;ctx.viewTailOffset=viewTail;")
    assert reset.count("ctx.viewQueuesArmed=true;") == 1
    assert reset.count("ctx.viewTailOffset=") == 1


def test_run_iterations_check_the_readiness_counters():
    sv = _read("src/SynchronizerValidator.cpp")
    pre = _squash(_block(sv, "void SynchronizerValidator::preSolution("))
    _in_order(
        pre,
        "m_viewTailIterations=m_viewTailChecked=m_viewTailMismatches=0;",
        "m_viewTransport=solution!=nullptr&&fusedA2AViewTransport(*solution);"
        "m_viewTailOffset=m_viewTransport&&m_problem!=nullptr"
        "?fusedA2AViewTailOffset(*m_problem,*solution):0;"
        "m_viewMode=m_viewTransport&&solution->problemType.fusedA2A==FusedA2A::GemmA2A"
        "?VIEW_MODE_GEMM_A2A:VIEW_MODE_A2A_GEMM;",
    )
    run = _squash(_block(sv, "void SynchronizerValidator::checkRunIteration()"))
    _in_order(run, 'checkVerify("runiteration");', 'checkViewTails("runiteration");')
    tails = _squash(_block(sv, "void SynchronizerValidator::checkViewTails("))
    _in_order(
        tails,
        "if(!m_viewTransport)return;m_viewTailIterations++;"
        "constViewTailCheckschecks=viewTailChecks((uint32_t)m_viewDebug);"
        "if(m_devices==nullptr||m_viewTailOffset==0||(!checks.entry&&!checks.pulled))return;"
        "syncAll(*m_devices);",
        "readBuffer(device->deviceId,device->viewBlock,kViewHeaderWords*sizeof(uint32_t))",
        "readBuffer(device->deviceId,viewTailOf(device->viewBlock,m_viewTailOffset),VIEW_TAIL_BYTES)",
        "if(!viewCheckTails(m_viewMode,(uint32_t)world,headers.data(),tails.data(),checks,bad))return;"
        "m_viewTailChecked++;m_viewTailMismatches+=(uint32_t)bad.size();for(uint32_td:bad){",
        "m_failedInSolution=true;",
    )
    post = _squash(_block(sv, "void SynchronizerValidator::postSolution()"))
    assert (
        'if(m_viewTransport)std::cout<<"VIEW_TAILSiterations="<<m_viewTailIterations'
        '<<"checked="<<m_viewTailChecked<<"mismatched="<<m_viewTailMismatches'
        '<<"name="<<m_solutionName<<std::endl;' in post
    )
    main = _squash(_read("main.cpp"))
    _in_order(main, "synchronizerValidator->checkRunIteration();", "viewDrainQueues(*ctx)")


def test_throw_at_fires_once_and_only_in_view_launches():
    main = _squash(_read("main.cpp"))
    assert "ViewThrowAtviewThrowAt=fusedA2A?viewParseThrowAt(" in main
    launch_all = _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))
    assert (
        "autothrowAt=[&](ViewThrowSitesite,size_td){"
        "if(viewThrowNow(viewThrowAt,site,(uint32_t)d,viewLaunch))throw" in launch_all
    )
    assert launch_all.count("viewThrowAt") == 1
    body = _block(
        launch_all[launch_all.index("try{ScopedDevicedevice(ctx.deviceId);if(viewTransport)") :],
        "if(viewTransport)",
    )
    assert launch_all.count("throwAt(ViewThrowSite::") == body.count("throwAt(ViewThrowSite::") == 6
