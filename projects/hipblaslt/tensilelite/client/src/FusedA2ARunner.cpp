// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FusedA2ARunner.hpp"

#include "DataInitialization.hpp"
#include "FusedA2ACounterSentinel.hpp"

#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

// SdmaQueue.hpp is header-only and pulls in hsakmt, which is only on the
// include path (and linked) under this option.
#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // log2(16-byte SDMA packet element / 2-byte bf16).
            constexpr size_t kElemShift    = 3;
            constexpr size_t kElemMultiple = size_t(1) << kElemShift;

            // Bytes reserved ahead of recv for the flag array. The ABI passes flag
            // and recv as independent pointers; carving both from one allocation is
            // this client's choice, not a layout the kernel knows about.
            constexpr size_t kFlagBytes = 4096;
            static_assert(FUSED_A2A_FLAG_BLOCK_BYTES <= kFlagBytes,
                          "flag block must fit ahead of recv in the shared allocation");

            constexpr int kOutboundPolls = 1000;
        }

        int FusedA2ARunner::worldSize(po::variables_map const& args)
        {
#ifndef TENSILELITE_ENABLE_SDMA
            throw std::runtime_error(
                "[fused-a2a] this client was built without TENSILELITE_ENABLE_SDMA, so no SDMA "
                "rings exist, but the fused epilogue unconditionally submits SDMA packets and "
                "would dereference a null queue handle. Reconfigure with "
                "-DTENSILELITE_ENABLE_SDMA=ON.");
#endif
            const int toValidate = args["num-elements-to-validate"].as<int>();
            if(toValidate > 0)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] --num-elements-to-validate=",
                    toValidate,
                    " is not supported; this path compares every element or none. Use -1 for "
                    "all, 0 for none."));
            if(toValidate != 0 && args["print-max"].as<int>() <= 0)
                throw std::runtime_error(
                    "[fused-a2a] validation needs --print-max > 0; with print-max <= 0 a "
                    "mismatch in D is reported as PASSED.");
            for(auto name : {"init-a", "init-b"})
                if(args[name].as<InitMode>() != InitMode::Random)
                    throw std::runtime_error(concatenate("[fused-a2a] --",
                                                         name,
                                                         " must be ",
                                                         InitMode::Random,
                                                         "; got ",
                                                         args[name].as<InitMode>()));
            if(args["device-idx"].as<int>() != 0)
                throw std::runtime_error(
                    "[fused-a2a] only --device-idx=0 is supported; rank r runs on device r. Use "
                    "HIP_VISIBLE_DEVICES to pick the set.");
            auto const& flush = args["icache-flush-args"].as<std::vector<bool>>();
            if(args["rotating-buffer-size"].as<int32_t>() != 0
               || args["icache-rotate-copies"].as<int>() != 0
               || std::find(flush.begin(), flush.end(), true) != flush.end())
                throw std::runtime_error(
                    "[fused-a2a] --rotating-buffer-size, --icache-rotate-copies and "
                    "--icache-flush-args are not supported; leave them at 0, 0 and false.");

            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            const int worldArg = args["fused-a2a-world"].as<int>();
            const int world    = worldArg > 0 ? worldArg : deviceCount;
            if(!fusedA2AWorldSizeValid(world))
                throw std::runtime_error(
                    concatenate("[fused-a2a] world size W=",
                                world,
                                " is out of range; the kernarg segment reserves exactly ",
                                FUSED_A2A_MAX_RANKS,
                                " peer groups. Require 1 <= W <= ",
                                FUSED_A2A_MAX_RANKS,
                                "."));
            if(deviceCount < world)
                throw std::runtime_error(
                    concatenate("[fused-a2a] need ", world, " devices, found ", deviceCount));
            return world;
        }

        void FusedA2ARunner::configureProblem(po::variables_map const& args,
                                              ContractionProblemGemm&  problem,
                                              int                      world,
                                              int                      runIdx)
        {
            const size_t numBatch = problem.batchIndices().size();
            if(numBatch > 1 || (numBatch == 1 && problem.batchSize(0) != 1))
                throw std::runtime_error("[fused-a2a] batched GEMM is not supported");
            if(problem.a().dataType() != rocisa::DataType::BFloat16
               || problem.b().dataType() != rocisa::DataType::BFloat16)
                throw std::runtime_error(concatenate("[fused-a2a] only bf16 A and B are supported, "
                                                     "but A=",
                                                     problem.a().dataType(),
                                                     " B=",
                                                     problem.b().dataType()));

            // The first `AM` FEATURE columns go all-to-all; [AM, M) stay local.
            auto const& amArgs = args["fused-a2a-am"].as<std::vector<int>>();
            if(amArgs.size() > 1 && (size_t)runIdx >= amArgs.size())
                throw std::runtime_error(concatenate(
                    "[fused-a2a] --fused-a2a-am has ",
                    amArgs.size(),
                    " values, fewer than the problems selected by "
                    "--problem-start-idx/--num-problems; pass one value per selected problem, or "
                    "one value for all."));
            const uint32_t M = (uint32_t)problem.freeSizeA(0);
            const uint32_t am
                = amArgs.empty() ? M : (uint32_t)amArgs[amArgs.size() == 1 ? 0 : runIdx];

            // freeIndices()[j].d is the D dim for free index j: 0 = A's M(feature),
            // 1 = B's N(token).
            const auto&  dDesc    = problem.d();
            const size_t dMStride = dDesc.strides()[problem.freeIndices()[0].d];
            const size_t dNStride = dDesc.strides()[problem.freeIndices()[1].d];
            if(dMStride != 1)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] D's feature axis is not contiguous: D mStride=",
                    dMStride,
                    " must be 1. The packet copies n_shard 16-byte elements as one contiguous "
                    "run, so a strided feature axis would ship unrelated data to every peer."));
            // dNStride is the packet's src_pitch (StrideD1J), a 19-bit field.
            if(dNStride % kElemMultiple != 0)
                throw std::runtime_error(
                    concatenate("[fused-a2a] ldd(D nStride)=",
                                dNStride,
                                " must be a multiple of ",
                                kElemMultiple,
                                "; the emitter's >>",
                                kElemShift,
                                " would truncate the pitch and skew every token row."));
            if((dNStride >> kElemShift) >= (1u << 19))
                throw std::runtime_error(concatenate(
                    "[fused-a2a] ldd(D nStride)=",
                    dNStride,
                    " overflows the SDMA packet's 19-bit src_pitch field (ldd must be < ",
                    (size_t(1u << 19) << kElemShift),
                    "). Reduce M or the D padding."));

            problem.setFusedA2AExtent(am);
            problem.setFusedA2AWorld((uint32_t)world);
        }

        FusedA2ARunner::FusedA2ARunner(po::variables_map const&                           args,
                                       Hardware const&                                    hardware,
                                       ContractionProblemGemm const&                      problem,
                                       ContractionSolution const&                         solution,
                                       std::vector<std::shared_ptr<DeviceContext>> const& devices)
            : m_devices(devices)
            , m_checkWptr(args["num-elements-to-validate"].as<int>() != 0)
            , m_drainSend(args["fused-a2a-drain-send"].as<int>() ? 1u : 0u)
            , m_am((uint32_t)problem.fusedA2AExtent())
        {
            const int      W = (int)devices.size();
            const uint32_t drain
                = (args["fused-a2a-drain-recv"].as<int>() ? FUSED_A2A_DRAIN_RECV : 0u)
                  | (m_drainSend ? FUSED_A2A_DRAIN_SEND : 0u);

            if(!solution.problemType.fusedGemmA2A)
                throw std::runtime_error(
                    "[fused-a2a] solution was not built with FusedGemmA2A; it has no fused "
                    "epilogue to drive.");

            {
                auto const&                        pt            = solution.problemType;
                const std::pair<bool, char const*> unsupported[] = {
                    {pt.useBias != 0, "bias"},
                    {pt.useE, "E"},
                    {pt.useGateResidual, "gate residual"},
                    {!pt.useScaleAB.empty(), "scaleA/scaleB"},
                    {pt.useScaleCD, "scaleC/scaleD"},
                    {pt.useScaleAlphaVec != 0, "scaleAlphaVec"},
                    {pt.outputAmaxD, "amaxD"},
                    {pt.sparse != 0, "sparse metadata"},
                    {pt.groupedGemm, "grouped GEMM"},
                    {pt.activationType != ActivationType::None, "activation"},
                    {solution.requiredWorkspaceSize(problem, hardware) != 0, "workspace"},
                };
                std::string missing;
                for(auto const& [needed, name] : unsupported)
                    if(needed)
                        missing += (missing.empty() ? "" : ", ") + std::string(name);
                if(!missing.empty())
                    throw std::runtime_error(
                        "[fused-a2a] solution needs inputs this client does not set: " + missing);
            }

            // The kernel epilogue derives dst_rank and the counter index from MT0/MT1.
            const uint32_t macroTileM = (uint32_t)solution.sizeMapping.macroTile.x;
            const uint32_t macroTileN = (uint32_t)solution.sizeMapping.macroTile.y;
            if(macroTileM == 0 || macroTileN == 0)
                throw std::runtime_error(
                    concatenate("[fused-a2a] solution macro-tile is zero (MT0=",
                                macroTileM,
                                " MT1=",
                                macroTileN,
                                ")"));

            // M/N-swap (col-major first-class): A=w[feature,K], B=x[token,K].
            const uint32_t M          = (uint32_t)problem.freeSizeA(0);
            const uint32_t N          = (uint32_t)problem.freeSizeB(0);
            const uint32_t K          = (uint32_t)problem.boundSize(0);
            const uint32_t AM         = m_am;
            const uint32_t nShard     = AM / (uint32_t)W;
            const uint32_t tokenTiles = (N + macroTileN - 1) / macroTileN;

            if(AM % (uint32_t)W != 0 || (nShard % macroTileM) != 0 || (M % macroTileM) != 0
               || (AM % macroTileM) != 0 || AM > M)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] problem shape violates fused-A2A constraints: M(feature)=",
                    M,
                    " N(token)=",
                    N,
                    " AM=",
                    AM,
                    " W=",
                    W,
                    " MacroTile0(feature)=",
                    macroTileM,
                    ". Require AM % W == 0, (AM/W) % ",
                    macroTileM,
                    " == 0, M % ",
                    macroTileM,
                    " == 0, AM % ",
                    macroTileM,
                    " == 0, AM <= M; otherwise the DRAIN barrier deadlocks."));

            if(!(*solution.problemPredicate)(problem))
            {
                std::ostringstream msg;
                msg << "[fused-a2a] solution predicate does not match the problem:\n";
                solution.problemPredicate->debugEval(problem, msg);
                throw std::runtime_error(msg.str());
            }

            // SDMA COPY_SUBWIN rect_x/rect_y are 14-bit; rect_x = n_shard scaled into
            // 16-byte packet elements, rect_y <= MT1. `>=` is one tighter than the
            // hardware: the extents are minus-one encoded.
            if(nShard % kElemMultiple != 0)
                throw std::runtime_error(concatenate("[fused-a2a] n_shard=AM/W=",
                                                     nShard,
                                                     " must be a multiple of ",
                                                     kElemMultiple,
                                                     "; the emitter's >>",
                                                     kElemShift,
                                                     " would truncate it and copy a short band."));
            const size_t maxRectX = (size_t)nShard >> kElemShift;
            const size_t maxRectY = (size_t)macroTileN;
            if(maxRectX >= (1u << 14) || maxRectY >= (1u << 14))
                throw std::runtime_error(concatenate(
                    "[fused-a2a] geometry overflows the SDMA packet's 14-bit rect fields: rect_x="
                    "n_shard>>",
                    kElemShift,
                    "=",
                    maxRectX,
                    " rect_y=MT1=",
                    maxRectY,
                    "; each must be < ",
                    (1u << 14),
                    ". Reduce AM or raise W (the bound is AM < ",
                    (size_t(1u << 14) << kElemShift),
                    "*W)."));

            // recv is feature-contiguous [W, token, feature_shard]. Token is padded to
            // a whole MacroTile1 tile: the PUSH store writes the full macro-tile edge
            // with no edge clamp.
            const size_t nTokenPad = (size_t)tokenTiles * macroTileN;
            m_recvBytes            = (size_t)W * nTokenPad * nShard * sizeof(uint16_t);
            m_counterBytes         = fusedA2ACounterPayloadBytes((uint32_t)W, tokenTiles);
            m_counterAllocBytes    = fusedA2ACounterAllocBytes((uint32_t)W, tokenTiles);

            std::cout << "[fused-a2a] W=" << W << " nFeature(M)=" << M << " nToken(N)=" << N
                      << " K=" << K << " AM=" << AM << " nShard=" << nShard
                      << " tokenTiles=" << tokenTiles << " drain=" << drain << "\n";

            std::vector<uint32_t> guard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
            fusedA2ACounterSentinelFill(guard.data());

            m_peer.assign(W, nullptr);
            m_counter.assign(W, nullptr);
            m_flag.assign(W, nullptr);
            m_recv.assign(W, nullptr);
            m_start.assign(W, nullptr);
            m_stop.assign(W, nullptr);
            for(int d = 0; d < W; d++)
            {
                ScopedDevice device(devices[d]->deviceId);
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(hipExtMallocWithFlags(
                    &m_peer[d], kFlagBytes + m_recvBytes, hipDeviceMallocFinegrained));
                m_flag[d] = m_peer[d];
                m_recv[d] = (char*)m_peer[d] + kFlagBytes;
                HIP_CHECK_EXC(hipMemset(m_peer[d], 0, kFlagBytes));
                HIP_CHECK_EXC(hipMalloc(&m_counter[d], m_counterAllocBytes));
                // The guard tail sits past m_counterBytes, so the per-launch memset
                // leaves it untouched.
                HIP_CHECK_EXC(hipMemcpy((char*)m_counter[d] + m_counterBytes,
                                        guard.data(),
                                        FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                        hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipEventCreate(&m_start[d]));
                HIP_CHECK_EXC(hipEventCreate(&m_stop[d]));
            }

            for(int s = 0; s < W; s++)
            {
                ScopedDevice device(devices[s]->deviceId);
                for(int t = 0; t < W; t++)
                {
                    if(t == s)
                        continue;
                    int canAccess = 0;
                    HIP_CHECK_EXC(hipDeviceCanAccessPeer(
                        &canAccess, devices[s]->deviceId, devices[t]->deviceId));
                    if(!canAccess)
                        throw std::runtime_error(
                            concatenate("[fused-a2a] device ", s, " cannot P2P device ", t));
                    hipError_t pe = hipDeviceEnablePeerAccess(devices[t]->deviceId, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
            }

            // One ring per (device, peer), created after P2P enable so peer pages are
            // already mapped. The self entry (j == d) is a loopback queue, which gives
            // this card's own flag slot a real producer.
            m_queues.resize(W);
#ifdef TENSILELITE_ENABLE_SDMA
            {
                std::vector<uint32_t> nodes(W);
                for(int j = 0; j < W; j++)
                    nodes[j] = sdmaNodeIdForDevice(devices[j]->deviceId);
                for(int d = 0; d < W; d++)
                {
                    ScopedDevice device(devices[d]->deviceId);
                    for(int j = 0; j < W; j++)
                        m_queues[d].push_back(std::make_shared<SdmaQueue>(
                            nodes[d], sdmaSelectEngine(nodes[d], nodes[j])));
                }
            }
#endif

            // Per-queue engine write pointer, carried across iterations.
            if(m_checkWptr)
            {
                m_prevWptr.resize(W);
#ifdef TENSILELITE_ENABLE_SDMA
                for(int d = 0; d < W; d++)
                    for(auto const& q : m_queues[d])
                        m_prevWptr[d].push_back(*q->queueResource().Queue_write_ptr_aql);
#endif
            }

            m_kernels.resize(W);
            for(int d = 0; d < W; d++)
            {
                ScopedDevice      device(devices[d]->deviceId);
                ContractionInputs inputs
                    = dynamic_cast<ContractionInputs const&>(*devices[d]->inputs);
#ifdef TENSILELITE_ENABLE_SDMA
                // One group per peer, flattened here so the packer needs no hsakmt type.
                for(int j = 0; j < W; j++)
                {
                    const HsaQueueResource& r = m_queues[d][j]->queueResource();
                    inputs.fusedA2APeers.push_back({m_flag[j],
                                                    m_recv[j],
                                                    m_queues[d][j]->ringBase(),
                                                    (void*)r.Queue_read_ptr_aql,
                                                    (void*)r.Queue_write_ptr_aql,
                                                    (void*)r.Queue_DoorBell_aql});
                }
#endif
                inputs.fusedA2ACounter = m_counter[d];
                inputs.fusedA2AMyRank  = (uint32_t)d;
                inputs.fusedA2ADrain   = drain;

                m_kernels[d]
                    = solution.solve(problem, inputs, hardware, nullptr, 0, devices[d]->stream);
                if(m_kernels[d].size() != 1)
                    throw std::runtime_error(concatenate("[fused-a2a] expected exactly one "
                                                         "kernel on device ",
                                                         d,
                                                         ", got ",
                                                         m_kernels[d].size()));
            }
        }

        FusedA2ARunner::~FusedA2ARunner()
        {
            m_queues.clear();
            for(size_t d = 0; d < m_devices.size(); d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                static_cast<void>(hipEventDestroy(m_start[d]));
                static_cast<void>(hipEventDestroy(m_stop[d]));
                static_cast<void>(hipFree(m_peer[d]));
                static_cast<void>(hipFree(m_counter[d]));
            }
        }

        FusedA2AIteration FusedA2ARunner::launchIteration(int iteration)
        {
            const int         W = (int)m_devices.size();
            FusedA2AIteration result;
            result.cardUs.assign(W, -1.0);

            // Re-arm before the start event: otherwise the DRAIN barrier releases
            // trivially and a stale-correct recv masks a broken scatter.
            for(int d = 0; d < W; d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                HIP_CHECK_EXC(hipMemset(m_counter[d], 0, m_counterBytes));
                HIP_CHECK_EXC(hipMemset(m_flag[d], 0, FUSED_A2A_FLAG_BLOCK_BYTES));
                HIP_CHECK_EXC(hipMemset(m_recv[d], 0, m_recvBytes));
            }
            for(int d = 0; d < W; d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                HIP_CHECK_EXC(hipDeviceSynchronize());
            }

            {
                ScopedDevice restore(m_devices[0]->deviceId);
                for(int d = 0; d < W; d++)
                {
                    hipStream_t stream = m_devices[d]->stream;
                    HIP_CHECK_EXC(hipSetDevice(m_devices[d]->deviceId));
                    HIP_CHECK_EXC(hipEventRecord(m_start[d], stream));
                    HIP_CHECK_EXC(m_devices[d]->adapter->launchKernels(
                        m_kernels[d], stream, nullptr, nullptr));
                    HIP_CHECK_EXC(hipEventRecord(m_stop[d], stream));
                }
            }

            for(int d = 0; d < W; d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                hipError_t   se = hipStreamSynchronize(m_devices[d]->stream);
                if(se != hipSuccess)
                {
                    std::cerr << "[fused-a2a] device " << d << " kernel FAILED (iter " << iteration
                              << "): " << hipGetErrorString(se) << std::endl;
                    result.ok = false;
                    continue;
                }
                float ms = 0.0f;
                HIP_CHECK_EXC(hipEventElapsedTime(&ms, m_start[d], m_stop[d]));
                result.cardUs[d] = (double)ms * 1000.0;
            }

            // Non-throwing reads: a device already wedged by a failed launch
            // degrades to a warning instead of masking the kernel error above.
            for(int d = 0; d < W; d++)
            {
                ScopedDevice          device(m_devices[d]->deviceId);
                std::vector<uint32_t> guard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                hipError_t            ge = hipMemcpy(guard.data(),
                                          (char const*)m_counter[d] + m_counterBytes,
                                          FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                          hipMemcpyDeviceToHost);
                if(ge != hipSuccess)
                {
                    std::cerr << "[fused-a2a] WARNING: could not read counter guard on device " << d
                              << " (iter " << iteration << "): " << hipGetErrorString(ge)
                              << std::endl;
                    continue;
                }
                int bad = fusedA2ACounterSentinelFirstBad(guard.data());
                if(bad >= 0)
                {
                    std::cerr << "[fused-a2a] COUNTER OVERRUN iter=" << iteration << " device=" << d
                              << ": guard word " << bad << " (byte "
                              << m_counterBytes + (size_t)bad * sizeof(uint32_t) << " of a "
                              << m_counterAllocBytes << "-byte allocation) holds 0x" << std::hex
                              << guard[bad] << ", expected 0x"
                              << fusedA2ACounterSentinelWord((size_t)bad) << std::dec
                              << " -- a counter index ran past the " << m_counterBytes
                              << "-byte payload" << std::endl;
                    result.ok = false;
                }
            }

            const uint32_t expected = m_drainSend ? 0u : (uint32_t)W;
            for(int d = 0; m_am != 0 && d < W; d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                uint32_t     outbound = ~0u;
                hipError_t   oe       = hipSuccess;
                for(int spin = 0; spin < kOutboundPolls && outbound != expected; spin++)
                {
                    oe = hipMemcpy(&outbound,
                                   (char const*)m_flag[d] + FUSED_A2A_OUTBOUND_OFFSET,
                                   sizeof(outbound),
                                   hipMemcpyDeviceToHost);
                    if(oe != hipSuccess)
                        break;
                }
                if(oe != hipSuccess)
                {
                    std::cerr << "[fused-a2a] WARNING: could not read the outbound counter on "
                                 "device "
                              << d << " (iter " << iteration << "): " << hipGetErrorString(oe)
                              << std::endl;
                }
                else if(outbound != expected)
                {
                    std::cerr << "[fused-a2a] OUTBOUND SIGNAL SHORT iter=" << iteration
                              << " device=" << d << ": counter at byte "
                              << FUSED_A2A_OUTBOUND_OFFSET << " of the flag block reads "
                              << outbound << " after " << kOutboundPolls << " polls, expected "
                              << expected
                              << (m_drainSend ? " -- the kernel polls this counter to W and then"
                                                " clears it, so a non-zero value means its"
                                                " drainSend segment never ran"
                                              : " -- some queue never carried its completion"
                                                " ATOMIC, so drainSend would hang on it")
                              << std::endl;
                    result.ok = false;
                }
            }

            // The engine would read a backwards write pointer as ~4 GB of packets
            // that were never written.
#ifdef TENSILELITE_ENABLE_SDMA
            for(int d = 0; m_checkWptr && d < W; d++)
            {
                for(size_t q = 0; q < m_queues[d].size(); q++)
                {
                    const uint64_t now = *m_queues[d][q]->queueResource().Queue_write_ptr_aql;
                    const uint64_t was = m_prevWptr[d][q];
                    if(now < was)
                    {
                        std::cerr << "[fused-a2a] WPTR WENT BACKWARDS iter=" << iteration
                                  << " device=" << d << " queue=" << q << ": " << was << " -> "
                                  << now
                                  << " -- a producer reserved from a cursor behind the "
                                     "hardware write pointer; the engine now sees ~"
                                  << ((was - now) >> 20)
                                  << " MB of packets that were never "
                                     "written"
                                  << std::endl;
                        result.ok = false;
                    }
                    else if((now - was) % 4 != 0)
                    {
                        std::cerr << "[fused-a2a] WPTR NOT DWORD-ALIGNED iter=" << iteration
                                  << " device=" << d << " queue=" << q << ": " << was << " -> "
                                  << now << " (delta " << (now - was)
                                  << ") -- packets and wrap padding are whole dwords, so this "
                                     "is a torn or garbage publish"
                                  << std::endl;
                        result.ok = false;
                    }
                    m_prevWptr[d][q] = now;
                }
            }
#endif
            return result;
        }
    } // namespace Client
} // namespace TensileLite
