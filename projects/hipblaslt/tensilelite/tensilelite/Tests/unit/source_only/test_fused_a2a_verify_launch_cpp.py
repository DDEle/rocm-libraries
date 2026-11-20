# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host wiring of --fused-a2a-verify in the tensilelite client.

Reads the client sources from the source tree, which the installed test
artifact does not stage. The logic the wiring calls is covered by
tests/ViewTransport_test.cpp.
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


def _squash(text):
    return re.sub(r"\s+", "", text)


def _in_order(text, *needles):
    at = -1
    for needle in map(_squash, needles):
        nxt = text.find(needle, at + 1)
        assert nxt > at, "%r not found after offset %d" % (needle, at)
        at = nxt


def _once(text, needle):
    assert text.count(_squash(needle)) == 1, needle


def _main():
    return _squash(_read("main.cpp"))


def _launch_all():
    return _squash(_block(_read("main.cpp"), "auto   launchAll = [&]"))


def test_launch_runs_producers_between_barriers_then_arm_gemm_and_check():
    _in_order(
        _launch_all(),
        "constsize_tdataSlot=verify?launchSlot%device0->inputArr.size():0;",
        "if(verify){",
        "verifyLaunch=verifyState.beginLaunch(dataSlot);",
        "constboolbarrier=verifyBarrier(verifyOptions,verifyLaunch);",
        "if(barrier)syncAll(active);",
        "if(!verifyInputsCaptured){",
        "verifyCaptureInputs(ctx,verifyLayoutOf);",
        "verifyInputsCaptured=true;",
        "verifyProduced=true;",
        "verifyState.addSkew(verifyProduce(*active[d],verifyLayoutOf,dataSlot,verifyLaunch,verifyOptions,"
        "(uint32_t)d));",
        "if(barrier)syncAll(active);",
        "ViewArmArgsview=viewArgs[launchSlot][d];",
        "if(verify)view.debug=verifyViewDebug(view.debug,verifyLaunch);",
        "viewArm(view,ctx.stream,viewArmMode)",
        "launchKernels(launch[d],ctx.stream,nullptr,nullptr)",
        "viewWait(view,ctx.stream)",
        "if(verify&&verifyState.checks())verifyCheck(ctx,verifyLayoutOf,dataSlot,verifyLaunch);",
    )
    main = _main()
    _once(main, "verifyCheck(")
    _once(main, "verifyProduce(")
    _once(main, "verifyState.reset(device0->inputArr.size());")


def test_layout_sizes_the_goldens_and_every_run_iteration_is_counted():
    main = _main()
    _in_order(
        main,
        "constboolverify=fusedA2A&&verifyOptions.mode!=VERIFY_OFF;",
        "if(verify){verifyLayoutOf=verifyLayout(*problem,active.size());"
        "verifyState.reset(device0->inputArr.size());}",
        "resetCommBuffers(*problem,*solution,active,viewDebug,verify?&verifyLayoutOf:nullptr);",
        "while(listeners.needMoreRunsInSolution()){",
        "verifyState.iterations++;",
        "if(resetInput){",
        '"VERIFY_SOLUTIONworld="<<active.size()<<"launches="<<verifyState.launches<<"golden="',
        '"slots="<<verifyState.last.size()<<"iterations="<<verifyState.iterations'
        '<<"skewed="<<verifyState.skewed<<"skew_ticks="<<verifyState.skewTicks<<"name="',
    )
    _once(main, "resetCommBuffers(")
    _once(main, "verifyState.iterations++;")
    reset = _squash(_block(_read("src/DeviceContext.cpp"), "void resetCommBuffers("))
    _in_order(
        reset,
        "if(verify!=nullptr){verifyGrowGoldens(ctx.golden,*verify,",
        "[](void*p){HIP_CHECK_EXC(hipFree(p));},",
        "[](size_tbytes){void*p=nullptr;HIP_CHECK_EXC(hipMalloc(&p,bytes));returnp;});",
        "verifyResetBlock(ctx.verifyBlock);",
    )
    release = _squash(_block(_read("src/DeviceContext.cpp"), "void DeviceContext::release()"))
    assert "for(void*p:{verifyBlock,golden.a.p,golden.b.p,golden.d.p,golden.recv.p})" in release


def test_the_check_follows_every_transport_branch():
    launch_all = _launch_all()
    loop = launch_all[launch_all.index("for(size_td=0;d<active.size();d++){auto&ctx=*active[d];") :]
    check = loop.index("verifyCheck(")
    for branch in ("if(viewTransport)", "elseif(inKernelNoops)", "elseif(startEvents)"):
        assert -1 < loop.index(branch) < check, branch


def test_verify_args_map_every_option():
    check = _squash(_block(_read("main.cpp"), "void checkFusedA2AVerifyArgs("))
    for line in (
        'v.verify=args["fused-a2a-verify"].as<int>();',
        'v.barrier=args["fused-a2a-verify-barrier"].as<int>();',
        'v.fence=args["fused-a2a-verify-fence"].as<int>();',
        'v.skewUs=args["fused-a2a-verify-skew-us"].as<int>();',
        'v.numElementsToValidate=args["num-elements-to-validate"].as<int>();',
        'v.numWarmups=args["num-warmups"].as<int>();',
        'v.cEqualD=args["c-equal-d"].as<bool>();',
        'v.sparse=args["sparse"].as<int>()!=0;',
        'v.boundsCheckNaN=args["bounds-check"].as<BoundsCheckMode>()==BoundsCheckMode::NaN;',
        'v.scales=!args["use-scaleAB"].as<std::string>().empty()||args["use-scaleCD"].as<bool>()'
        '||args["use-scaleAlphaVec"].as<int>()!=0||args["mx-a-block"].as<int>()!=0'
        '||args["mx-b-block"].as<int>()!=0;',
        'v.swizzled=args["swizzle-tensor-a"].as<bool>()||args["swizzle-tensor-b"].as<bool>();',
        'v.gemmA2A=args["fused-a2a"].as<FusedA2A>()==FusedA2A::GemmA2A;',
        'v.betaZero=args["init-beta"].as<InitMode>()==InitMode::Zero;',
        'v.activation=args["activation-type"].as<ActivationType>()!=ActivationType::None;',
        'v.bias=args["use-bias"].as<int>()!=0;',
        'v.fusedA2A=args["fused-a2a"].as<FusedA2A>()!=FusedA2A::Disabled;',
    ):
        assert _squash(line) in check, line
    _in_order(check, "conststd::stringerror=verifyArgsError(v);", "if(!error.empty())throw")


def test_verify_args_are_checked_unconditionally():
    main = _main()
    _once(main, "checkFusedA2AVerifyArgs(args);")
    assert "if(fusedA2A)checkFusedA2AArgs(args);checkFusedA2AVerifyArgs(args);" in main
    assert "checkFusedA2AVerifyArgs" not in _squash(_block(_read("main.cpp"), "void checkFusedA2AArgs("))


def test_golden_is_captured_once_after_the_first_validation_and_only_when_it_passed():
    main = _main()
    _in_order(
        main,
        "listeners.validateWarmups({0},warmupStartEvents,warmupStopEvents);",
        "if(verify&&verifyState.golden==VerifyGolden::Pending)",
        "verifyGoldenAfterValidation(verifyState.golden,"
        "referenceValidator->validatedSolution(),referenceValidator->errorInSolution());",
        "if(verifyState.golden==VerifyGolden::Captured)forEachDevice(active,[&](DeviceContext&ctx){"
        "verifyCaptureGolden(ctx,verifyLayoutOf);",
    )
    _once(main, "verifyCaptureGolden(")


def test_gemm_a2a_is_normalized_before_the_cpu_validation_of_each_sync():
    _in_order(
        _main(),
        "listeners.postEnqueues(startEvents,stopEvents);",
        "syncAll(active);",
        "for(autoconst&[slot,e]:verifyState.takeLaunched())",
        "verifyNormalize(ctx,verifyLayoutOf,slot,e);",
        "syncAll(active);",
        "listeners.validateEnqueues(",
    )


def test_inputs_are_restored_after_every_solution_that_produced():
    main = _main()
    _in_order(
        main,
        "boolverifyProduced=false;",
        "try{checkFusedA2ASolution(",
        "catch(std::runtime_errorconst&err){reporters->report(ResultKey::Validation,\"INVALID\");",
        "if(verifyProduced&&!deviceLost){try{syncAll(active);",
        "verifyRestore(ctx,verifyLayoutOf);",
        "listeners.postSolution();",
    )
    _in_order(main, "boolverifyInputsCaptured=false;", "while(solutionIterator->moreSolutionsInProblem())")


def test_input_reset_waits_for_every_device():
    _in_order(
        _main(),
        "if(resetInput){if(verify)syncAll(active);",
        "ctx.inputArr[0]=ctx.dataInit->prepareGPUInputs(problem);",
    )


def test_run_iteration_reads_every_check_block_after_a_barrier():
    validator = _read("src/SynchronizerValidator.cpp")
    run = _squash(_block(validator, "void SynchronizerValidator::checkRunIteration()"))
    _in_order(run, 'checkVerify("runiteration");', "checkLaunches(")
    verify = _squash(_block(validator, "void SynchronizerValidator::checkVerify("))
    _in_order(
        verify,
        "syncAll(*m_devices);",
        "for(autoconst&device:*m_devices)",
        "constVerifyCountsc=verifyReadBlock(*device);",
        "constVerifyBlockVerdictverdict=verifyTakeBlock(m_verifyTotal,c);",
        "if(!verdict.report)continue;",
        '"VERIFYdev="',
        "if(!verdict.failed)continue;",
        "m_reporter->log(LogLevel::Error,",
        "m_failedInSolution=true;",
    )
    loop = verify[verify.index("for(autoconst&device:*m_devices)") :]
    assert "return" not in loop
    assert "verifyAccumulate" not in verify
    assert verify.count("continue;") == 3
    post = _squash(_block(validator, "void SynchronizerValidator::postSolution()"))
    _in_order(
        post,
        'checkVerify("interruptedruniteration");',
        "if(m_verify!=0&&m_verifyIterations!=0)",
        '"VERIFY_TOTALchecked="',
    )
    assert "m_verifyIterations=0;" in _squash(_block(validator, "void SynchronizerValidator::preSolution("))
    begin = _squash(_block(validator, "void SynchronizerValidator::beginRunIteration()"))
    assert begin.startswith("voidSynchronizerValidator::beginRunIteration(){m_verifyIterations++;")


def test_one_op_kernel_loads_system_coherent_and_fences_on_request():
    source = _read("src/ViewVerify.cpp")
    kernel = _squash(_block(source, "__global__ void verifyOpKernel("))
    _in_order(
        kernel,
        "__atomic_thread_fence(__ATOMIC_ACQUIRE);",
        "constVerifySpant=verifySpan(blockIdx.x,threadIdx.x,gridDim.x,blockDim.x);",
        "verifyRunOp(op,t.first,t.step,block,t.lead);",
    )
    assert kernel.endswith("if(op.fence)__threadfence_system();}")
    run = _squash(_block(source, "void run(DeviceContext& ctx"))
    assert "hipLaunchKernelGGL(verifyOpKernel,::dim3(verifyBlocks(op.count)),::dim3(VERIFY_THREADS),0,ctx.stream,op," in run
    _in_order(
        _squash(_block(source, "uint64_t verifyProduce(")),
        "uint64_tticks=0;",
        "if(verifySkews(options,launch,rank,layout.world)){",
        "hipDeviceGetAttribute(&wallClockKHz,hipDeviceAttributeWallClockRate,ctx.deviceId)",
        "ticks=verifySkewTicks(options.skewUs,(uint64_t)wallClockKHz);",
        "hipLaunchKernelGGL(verifySkewKernel,::dim3(1),::dim3(1),0,ctx.stream,ticks);",
        "verifyProduceOps(layout,s,g,launch,options.fence)",
        "returnticks;",
    )
    tensor = _squash(_block(source, "VerifyTensor tensorOf("))
    assert "std::tie(v.stride,v.period)=verifyStridePeriod(t.strides(),dim);" in tensor
    goldens = _squash(_block(source, "VerifyGoldens goldensOf("))
    assert "returnverifyGoldensOf(ctx.golden,layout);" in goldens
    header = _squash(_block(_read("include/ViewVerify.hpp"), "verifyRunOp(VerifyOp const& op,"))
    _in_order(
        header,
        "caseVerifyOpKind::Copy:op.dst[i]=viewLoad32(op.src+i);",
        "caseVerifyOpKind::Check:bad+=!verifyCheckWord(viewLoad32(op.dst+i),",
    )
    _in_order(
        _squash(_block(source, "void verifyRestore(")),
        "for(size_tslot=0;slot<ctx.inputArr.size();slot++)",
        "verifyRestoreOps(layout,slotOf(ctx,slot),g)",
    )
    layout = _squash(_block(source, "VerifyLayout verifyLayout("))
    for token in (
        "s.a=tensorOf(gemm->a(),s.gemmA2A?gemm->freeIndicesA()[0].i:gemm->boundIndices()[0].a);",
        "s.d=tensorOf(tensors[ContractionProblemGemm::TENSOR::D],s.gemmA2A?gemm->freeIndices()[0].d:none);",
        "s.extent=(uint64_t)gemm->fusedA2AExtent();",
        "s.tokens=gemm->freeSizeB(0);",
        "returnverifyLayoutOf(s);",
    ):
        assert token in layout, token


def test_arm_kernel_without_copy_faults_is_the_fault_free_instantiation():
    arm = _squash(_block(_read("src/ViewTransport.cpp"), "void viewArm(ViewArmArgs const& args"))
    _in_order(
        arm,
        "if((args.debug&VIEW_DEBUG_COPY_FAULTS)!=0)hipLaunchKernelGGL(viewArmKernel<VIEW_DEBUG_COPY_FAULTS>,",
        "elsehipLaunchKernelGGL(viewArmKernel<0>,",
    )
