// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ViewTransport.hpp"
#include "DeviceContext.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        __device__ inline void viewStamp(uint64_t* slot)
        {
            __hip_atomic_store(
                slot, (uint64_t)wall_clock64(), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        // Stamps `slot`, when set, on scope exit.
        struct ViewStampOnExit
        {
            uint64_t* slot;
            __device__ ~ViewStampOnExit()
            {
                if(slot != nullptr)
                    viewStamp(slot);
            }
        };

        // One thread per queue (viewArmThread), then thread 0 advances the header, after
        // setting VIEW_ERROR_ABORTED when bookOnly. With a.stats: thread 0 stamps
        // VIEW_STATS_ARM_IN, and each queue's thread raises VIEW_STATS_ARM_OUT to the clock
        // after its doorbell. Faults: as viewBuildUnits.
        template <uint32_t Faults>
        __global__ void viewArmKernel(ViewArmArgs a, bool bookOnly)
        {
            const uint64_t  armIn = a.stats != nullptr ? (uint64_t)wall_clock64() : 0;
            uint32_t* const view  = a.view;
            uint32_t* const error = view + FUSED_A2A_VIEW_ERROR_OFFSET / sizeof(uint32_t);
            const uint32_t  q     = threadIdx.x;
            const uint32_t  epoch
                = viewLoad32(view + FUSED_A2A_VIEW_EPOCH_OFFSET / sizeof(uint32_t));
            const uint32_t halt = viewLoad32(error);
            if(a.stats != nullptr && q == 0)
                __hip_atomic_store(a.stats + viewStatsIndex(epoch, VIEW_STATS_ARM_IN),
                                   armIn,
                                   __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            __syncthreads();
            if(q < a.numQueues)
            {
                const uint32_t bits = viewArmThread<Faults>(
                    a, q, epoch, halt, bookOnly, viewAdd32, viewStore32, viewRelease32);
                if(bits != 0)
                    __hip_atomic_fetch_or(error, bits, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                if(a.stats != nullptr)
                    __hip_atomic_fetch_max(a.stats + viewStatsIndex(epoch, VIEW_STATS_ARM_OUT),
                                           (uint64_t)wall_clock64(),
                                           __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_SYSTEM);
            }
            __syncthreads();
            if(q == 0)
            {
                if(bookOnly)
                    __hip_atomic_fetch_or(
                        error, VIEW_ERROR_ABORTED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                viewAdvance(a, epoch, view);
            }
        }

        // Single thread. Returns at once when the error word is set, unless `abort`; else
        // spins, bounded, until viewWaitDone. A2AGemm: on timeout sets VIEW_ERROR_TIMEOUT, on
        // `abort` VIEW_ERROR_ABORTED. GemmA2A: on timeout or `abort` sets the error bit, releases
        // sig[] with viewReleaseSig and own entry[] with viewReleaseEntry, then waits for the
        // own queues to drain, setting VIEW_ERROR_STUCK if they do not.
        // With a.stats and not `abort`: stamps VIEW_STATS_GEMM_END on entry and
        // VIEW_STATS_WAIT_OUT on exit, in the record of epoch - 1.
        __global__ void viewWaitKernel(ViewArmArgs a, bool abort)
        {
            if(threadIdx.x != 0)
                return;
            uint32_t* const view  = a.view;
            uint32_t* const error = view + FUSED_A2A_VIEW_ERROR_OFFSET / sizeof(uint32_t);
            const uint32_t  epoch = viewLoad32(view + FUSED_A2A_VIEW_EPOCH_OFFSET / sizeof(uint32_t));
            ViewStampOnExit waitOut{nullptr};
            if(a.stats != nullptr && !abort)
            {
                viewStamp(a.stats + viewStatsIndex(epoch - 1u, VIEW_STATS_GEMM_END));
                waitOut.slot = a.stats + viewStatsIndex(epoch - 1u, VIEW_STATS_WAIT_OUT);
            }
            uint64_t start = (uint64_t)wall_clock64();
            if(!abort)
            {
                if(viewLoad32(error) != 0)
                    return;
                while(!viewWaitDone(a, epoch))
                {
                    if((uint64_t)wall_clock64() - start > a.waitTimeoutTicks)
                        break;
                    __builtin_amdgcn_s_sleep(1);
                }
                if(viewWaitDone(a, epoch))
                    return;
            }
            __hip_atomic_fetch_or(error,
                                  abort ? VIEW_ERROR_ABORTED : VIEW_ERROR_TIMEOUT,
                                  __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
            if(a.mode != VIEW_MODE_GEMM_A2A)
                return;
            viewReleaseSig(a, epoch);
            viewReleaseEntry(a, epoch, viewMax32);
            start = (uint64_t)wall_clock64();
            while(!viewQueuesDrained(a))
            {
                if((uint64_t)wall_clock64() - start > a.waitTimeoutTicks)
                {
                    __hip_atomic_fetch_or(
                        error, VIEW_ERROR_STUCK, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                    return;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }

        __global__ void viewNoopKernel() {}

        __global__ void viewFillKernel(uint32_t* words, size_t count, uint32_t value)
        {
            for(size_t i = threadIdx.x; i < count; i += blockDim.x)
                viewStore32(words + i, value);
        }

        namespace
        {
            void copyToDevice(void* device, void const* host, size_t offset, size_t bytes)
            {
                HIP_CHECK_EXC(hipMemcpy(static_cast<char*>(device) + offset,
                                        static_cast<char const*>(host) + offset,
                                        bytes,
                                        hipMemcpyHostToDevice));
            }

            void addOnDevice(uint32_t* p, uint32_t addend)
            {
                uint32_t v = 0;
                HIP_CHECK_EXC(hipMemcpy(&v, p, sizeof(v), hipMemcpyDeviceToHost));
                v += addend;
                HIP_CHECK_EXC(hipMemcpy(p, &v, sizeof(v), hipMemcpyHostToDevice));
            }

            void storeOnDevice(uint32_t* p, uint32_t value)
            {
                HIP_CHECK_EXC(hipMemcpy(p, &value, sizeof(value), hipMemcpyHostToDevice));
            }

            // viewArmHostQueues and viewAdvance on the header, after synchronizing `stream`.
            void viewArmHost(ViewArmArgs const& a, hipStream_t stream)
            {
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
                uint32_t header[kViewHeaderWords];
                HIP_CHECK_EXC(hipMemcpy(header, a.view, sizeof(header), hipMemcpyDeviceToHost));
                const uint32_t epoch = header[FUSED_A2A_VIEW_EPOCH_OFFSET / sizeof(uint32_t)];
                uint32_t&      error = header[FUSED_A2A_VIEW_ERROR_OFFSET / sizeof(uint32_t)];
                error |= viewArmHostQueues(a, epoch, error, addOnDevice, storeOnDevice);
                viewAdvance(a, epoch, header);
                copyToDevice(a.view,
                             header,
                             FUSED_A2A_VIEW_BASE_OFFSET,
                             FUSED_A2A_VIEW_EXPECTED_OFFSET + FUSED_A2A_MAX_RANKS * sizeof(uint32_t)
                                 - FUSED_A2A_VIEW_BASE_OFFSET);
                copyToDevice(a.view,
                             header,
                             FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET,
                             FUSED_A2A_VIEW_ERROR_OFFSET + sizeof(uint32_t)
                                 - FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET);
            }
        }

        ViewArmArgs makeViewArmArgs(ContractionProblem const&                          problem,
                                    ContractionSolution const&                         solution,
                                    std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                    size_t                                             rank,
                                    size_t                                             launch,
                                    uint32_t                                           coalesce,
                                    Hardware const&                                    hardware,
                                    int                                                debug,
                                    uint32_t                                           delayRank,
                                    uint32_t                                           delayCopies)
        {
#ifndef TENSILELITE_ENABLE_SDMA
            static_cast<void>(delayRank);
            static_cast<void>(delayCopies);
            throw std::runtime_error("[fused-a2a] FusedA2ATransport=View needs a client built with "
                                     "TENSILELITE_ENABLE_SDMA.");
#else
            auto const&  gemm     = dynamic_cast<ContractionProblemGemm const&>(problem);
            auto const&  own      = *devices[rank];
            const size_t world    = devices.size();
            const size_t slot     = launch % own.inputArr.size();
            auto         inputsOf = [&](size_t j) -> ContractionInputs const& {
                return dynamic_cast<ContractionInputs const&>(*devices[j]->inputArr[slot]);
            };
            const size_t           tailOffset = fusedA2AViewTailOffset(problem, solution);
            std::vector<uint32_t*> tails(world);
            for(size_t j = 0; j < world; j++)
            {
                if(devices[j]->viewBlock == nullptr
                   || devices[j]->viewBlockBytes < tailOffset + VIEW_TAIL_BYTES)
                    throw std::runtime_error("[fused-a2a] no view block on device "
                                             + std::to_string(devices[j]->deviceId) + ".");
                tails[j] = viewTailOf(devices[j]->viewBlock, tailOffset);
            }

            int wallClockKHz = 0;
            HIP_CHECK_EXC(hipDeviceGetAttribute(
                &wallClockKHz, hipDeviceAttributeWallClockRate, own.deviceId));

            ViewArmArgs a{};
            a.rank             = (uint32_t)rank;
            a.world            = (uint32_t)world;
            a.mt1              = (uint32_t)solution.sizeMapping.macroTile.y;
            a.nTokens          = (uint32_t)gemm.freeSizeB(0);
            a.tokenBlocks      = (a.nTokens + a.mt1 - 1) / a.mt1;
            a.coalesce         = coalesce;
            a.view             = static_cast<uint32_t*>(own.viewBlock);
            a.delayRank        = delayRank;
            a.delayCopies      = delayCopies;
            a.waitTimeoutTicks = (uint64_t)wallClockKHz * kViewWaitTimeoutMs;
            a.debug            = (uint32_t)debug;
            if((debug & VIEW_DEBUG_TIMELINE) != 0)
            {
                if(own.viewStats == nullptr)
                    throw std::runtime_error("[fused-a2a] no view stats ring on device "
                                             + std::to_string(own.deviceId) + ".");
                a.stats = static_cast<uint64_t*>(own.viewStats);
            }

            auto setQueue
                = [&](uint32_t q, size_t peer, uint64_t src, uint64_t dst, uint32_t* flag) {
                      SdmaQueue const&        queue = *own.sdmaQueues[peer];
                      HsaQueueResource const& r     = queue.queueResource();
                      a.queues[q] = {static_cast<uint32_t*>(queue.ringBase()),
                                     SDMA_QUEUE_SIZE,
                                     reinterpret_cast<uint64_t*>(r.Queue_read_ptr_aql),
                                     reinterpret_cast<uint64_t*>(r.Queue_write_ptr_aql),
                                     reinterpret_cast<uint64_t*>(r.Queue_DoorBell_aql),
                                     src,
                                     dst,
                                     flag,
                                     nullptr,
                                     nullptr,
                                     nullptr};
                  };

            if(gemm.fusedA2A() == FusedA2A::A2AGemm)
            {
                auto const* amd = dynamic_cast<AMDGPU const*>(&hardware);
                if(amd == nullptr || amd->computeUnitCount == 0)
                    throw std::runtime_error("[fused-a2a] no CU count for FusedNumCu.");
                const uint64_t bpe       = gemm.b().elementBytes();
                const uint64_t subRounds = solution.sizeMapping.a2aSubRounds;
                const uint64_t mt0       = solution.sizeMapping.macroTile.x;
                a.featureTiles           = (uint32_t)((gemm.freeSizeA(0) + mt0 - 1) / mt0);
                a.numCu                  = (uint32_t)amd->computeUnitCount;
                const uint64_t ldb       = (uint64_t)gemm.fusedA2AExtent() / world;
                const uint64_t rowBytes  = ldb * bpe;
                const uint64_t segBytes  = a.nTokens * rowBytes;
                a.mode                   = VIEW_MODE_A2A_GEMM;
                a.numQueues              = a.world - 1;
                a.subRounds              = (uint32_t)subRounds;
                a.peerGroups             = (uint32_t)solution.sizeMapping.a2aPeerGroups;
                a.srcPitch               = (uint32_t)(rowBytes / 16);
                a.dstPitch               = a.srcPitch;
                a.slice                  = a.srcPitch * a.mt1;
                a.rectX                  = (uint32_t)(a.srcPitch / subRounds);
                a.srcBlockStride         = a.mt1 * rowBytes;
                a.dstBlockStride         = a.srcBlockStride;
                a.subRoundStride         = rowBytes / subRounds;
                uint32_t* const flag     = a.view + FUSED_A2A_VIEW_FLAG_OFFSET / sizeof(uint32_t);
                for(uint32_t i = 0; i < a.numQueues; i++)
                {
                    const size_t s = (rank + 1 + i) % world;
                    setQueue(i,
                             s,
                             viewAddress(inputsOf(s).b) + rank * segBytes,
                             viewAddress(inputsOf(rank).a2aBuf) + (i + 1) * segBytes,
                             flag + i);
                }
                if((debug & VIEW_DEBUG_DELAY_CHAIN) != 0 && delayRank >= world)
                    throw std::runtime_error("[fused-a2a] --fused-a2a-view-delay-rank "
                                             + std::to_string(delayRank) + " is not below "
                                             + std::to_string(world) + ".");
                if((debug & VIEW_DEBUG_DELAY_CHAIN) != 0 && rank == delayRank)
                {
                    if(own.viewDelayScratch == nullptr
                       || 2 * a.srcBlockStride > VIEW_DELAY_SCRATCH_BYTES)
                        throw std::runtime_error(
                            "[fused-a2a] --fused-a2a-view-debug 128 needs a delay scratch of "
                            + std::to_string(2 * a.srcBlockStride) + " bytes on device "
                            + std::to_string(own.deviceId) + ".");
                    a.delayScratch = viewAddress(own.viewDelayScratch);
                }
            }
            else
            {
                const uint64_t bpe    = gemm.d().elementBytes();
                const uint64_t nShard = (uint64_t)gemm.fusedA2AExtent() / world;
                const uint64_t ldd    = gemm.d().strides()[1];
                a.mode                = VIEW_MODE_GEMM_A2A;
                a.numQueues           = a.world;
                a.subRounds           = 1;
                a.srcPitch            = (uint32_t)(ldd * bpe / 16);
                a.dstPitch            = (uint32_t)(nShard * bpe / 16);
                a.slice               = (uint32_t)(a.mt1 * nShard * bpe / 16);
                a.rectX               = a.dstPitch;
                a.srcBlockStride      = a.mt1 * ldd * bpe;
                a.dstBlockStride      = a.mt1 * nShard * bpe;
                a.subRoundStride      = 0;
                a.tilesPerRank        = (uint32_t)(nShard / solution.sizeMapping.macroTile.x);
                for(uint32_t p = 0; p < a.numQueues; p++)
                    setQueue(p,
                             p,
                             viewAddress(inputsOf(rank).d) + p * nShard * bpe,
                             viewAddress(inputsOf(p).a2aBuf) + rank * a.nTokens * nShard * bpe,
                             static_cast<uint32_t*>(devices[p]->viewBlock)
                                 + FUSED_A2A_VIEW_FLAG_OFFSET / sizeof(uint32_t) + rank);
            }
            viewWireTails(a, tails.data());
            return a;
#endif
        }

        void viewArm(ViewArmArgs const& args, hipStream_t stream, int arm)
        {
            if(arm == VIEW_ARM_HOST)
                viewArmHost(args, stream);
            else
            {
                static_cast<void>(hipGetLastError());
                if((args.debug & VIEW_DEBUG_COPY_FAULTS) != 0)
                    hipLaunchKernelGGL(viewArmKernel<VIEW_DEBUG_COPY_FAULTS>,
                                       ::dim3(1),
                                       ::dim3(64),
                                       0,
                                       stream,
                                       args,
                                       false);
                else
                    hipLaunchKernelGGL(
                        viewArmKernel<0>, ::dim3(1), ::dim3(64), 0, stream, args, false);
                HIP_CHECK_EXC(hipGetLastError());
            }
        }

        void viewWait(ViewArmArgs const& args, hipStream_t stream)
        {
            if(args.mode != VIEW_MODE_GEMM_A2A && !viewExitWaits(args) && args.stats == nullptr)
                return;
            hipLaunchKernelGGL(viewWaitKernel, ::dim3(1), ::dim3(64), 0, stream, args, false);
            HIP_CHECK_EXC(hipGetLastError());
        }

        void viewAbort(ViewArmArgs const& args, hipStream_t stream, bool armed) noexcept
        {
            if(!armed)
                hipLaunchKernelGGL(viewArmKernel<0>, ::dim3(1), ::dim3(64), 0, stream, args, true);
            else
                hipLaunchKernelGGL(viewWaitKernel, ::dim3(1), ::dim3(64), 0, stream, args, true);
            static_cast<void>(hipGetLastError());
        }

        namespace
        {
            auto viewNow()
            {
                return std::chrono::steady_clock::now();
            }

            void viewSleep()
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        bool viewReleaseTailHost(DeviceContext& ctx, int path) noexcept
        {
            if(!ctx.viewQueuesArmed || ctx.viewBlock == nullptr)
                return true;
            uint32_t* const tail = reinterpret_cast<uint32_t*>(static_cast<char*>(ctx.viewBlock)
                                                               + ctx.viewTailOffset);
            if(path != VIEW_RELEASE_HOST_STORE
               && (ctx.viewReleaseStream == nullptr || ctx.viewReleaseHost == nullptr
                   || hipSetDevice(ctx.deviceId) != hipSuccess))
                return false;
            return viewReleaseTail(
                tail,
                path,
                viewRelease32,
                [&](uint32_t* to, size_t bytes) {
                    return hipMemcpyAsync(to,
                                          ctx.viewReleaseHost,
                                          bytes,
                                          hipMemcpyHostToDevice,
                                          ctx.viewReleaseStream)
                           == hipSuccess;
                },
                [&](uint32_t* to, size_t count, uint32_t value) {
                    static_cast<void>(hipGetLastError());
                    hipLaunchKernelGGL(viewFillKernel,
                                       ::dim3(1),
                                       ::dim3(64),
                                       0,
                                       ctx.viewReleaseStream,
                                       to,
                                       count,
                                       value);
                    return hipGetLastError() == hipSuccess;
                });
        }

        hipError_t viewSyncBounded(DeviceContext& ctx) noexcept
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
        }

        bool viewDrainQueues(DeviceContext& ctx) noexcept
        {
            if(!ctx.viewQueuesArmed)
                return true;
#ifdef TENSILELITE_ENABLE_SDMA
            try
            {
                auto drained = [&] {
                    for(auto const& queue : ctx.sdmaQueues)
                    {
                        HsaQueueResource const& r = queue->queueResource();
                        if(viewLoad64(reinterpret_cast<uint64_t*>(r.Queue_read_ptr_aql))
                           != viewLoad64(reinterpret_cast<uint64_t*>(r.Queue_write_ptr_aql)))
                            return false;
                    }
                    return true;
                };
                if(!drained())
                {
                    if(ctx.viewBlock != nullptr)
                    {
                        ScopedDevice device(ctx.deviceId);
                        hipLaunchKernelGGL(viewFillKernel,
                                           ::dim3(1),
                                           ::dim3(256),
                                           0,
                                           ctx.stream,
                                           static_cast<uint32_t*>(ctx.viewBlock) + kViewHeaderWords,
                                           (ctx.viewBlockBytes - FUSED_A2A_VIEW_SIG_OFFSET)
                                               / sizeof(uint32_t),
                                           0xFFFFFFFFu);
                        HIP_CHECK_EXC(hipGetLastError());
                    }
                    const auto deadline = std::chrono::steady_clock::now()
                                          + std::chrono::milliseconds(kViewWaitTimeoutMs);
                    if(viewPollIdle([&] { return hipStreamQuery(ctx.stream); },
                                    viewNow,
                                    viewSleep,
                                    deadline)
                       != hipSuccess)
                        return false;
                    while(!drained())
                    {
                        if(std::chrono::steady_clock::now() > deadline)
                            return false;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            }
            catch(...)
            {
                return false;
            }
#endif
            ctx.viewQueuesArmed = false;
            return true;
        }

        void viewNoop(hipStream_t stream)
        {
            hipLaunchKernelGGL(viewNoopKernel, ::dim3(1), ::dim3(1), 0, stream);
            HIP_CHECK_EXC(hipGetLastError());
        }

        std::string viewStatsLine(int device, uint64_t const* ring)
        {
            using Samples = std::vector<double>;
            auto field    = [&](uint32_t r, uint32_t f) { return ring[viewStatsIndex(r, f)]; };
            auto us       = [](uint64_t to, uint64_t from) {
                return (double)(int64_t)(to - from) / VIEW_STATS_TICKS_PER_US;
            };
            auto median = [](Samples v) {
                if(v.empty())
                    return std::string("na");
                std::sort(v.begin(), v.end());
                const size_t n = v.size();
                const double m = n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
                char         text[32];
                std::snprintf(text, sizeof(text), "%.1f", m);
                return std::string(text);
            };
            auto list = [&](std::vector<Samples> const& perQueue, std::vector<bool> const& used) {
                std::string text;
                for(uint32_t q = 0; q < FUSED_A2A_MAX_RANKS; q++)
                    if(used[q])
                        text += (text.empty() ? "" : ",") + median(perQueue[q]);
                return text.empty() ? std::string("na") : text;
            };

            uint32_t             n = 0;
            Samples              arm, gemm, wait, total, firstStart, lastEnd;
            std::vector<Samples> qDur(FUSED_A2A_MAX_RANKS), qEnd(FUSED_A2A_MAX_RANKS),
                qReady(FUSED_A2A_MAX_RANKS);
            std::vector<bool>    used(FUSED_A2A_MAX_RANKS, false);
            for(uint32_t r = 0; r < VIEW_STATS_RECORDS; r++)
            {
                const uint64_t gemmEnd = field(r, VIEW_STATS_GEMM_END);
                if(gemmEnd == 0)
                    continue;
                n++;
                const uint64_t armIn   = field(r, VIEW_STATS_ARM_IN);
                const uint64_t armOut  = field(r, VIEW_STATS_ARM_OUT);
                const uint64_t waitOut = field(r, VIEW_STATS_WAIT_OUT);
                arm.push_back(us(armOut, armIn));
                gemm.push_back(us(gemmEnd, armOut));
                wait.push_back(us(waitOut, gemmEnd));
                total.push_back(us(waitOut, armIn));
                Samples starts, ends;
                for(uint32_t q = 0; q < FUSED_A2A_MAX_RANKS; q++)
                {
                    const uint64_t start = field(r, VIEW_STATS_TS_START + 2 * q);
                    const uint64_t end   = field(r, VIEW_STATS_TS_END + 2 * q);
                    if(start != 0)
                        starts.push_back(us(start, armOut));
                    if(end != 0)
                    {
                        used[q] = true;
                        ends.push_back(us(end, armOut));
                        qEnd[q].push_back(ends.back());
                    }
                    if(start != 0 && end != 0)
                        qDur[q].push_back(us(end, start));
                    const uint64_t ready = field(r, VIEW_STATS_TS_READY + q);
                    if(start != 0 && ready != 0)
                        qReady[q].push_back(us(ready, start));
                }
                if(!starts.empty())
                    firstStart.push_back(*std::min_element(starts.begin(), starts.end()));
                if(!ends.empty())
                    lastEnd.push_back(*std::max_element(ends.begin(), ends.end()));
            }
            return "VIEWSTAT dev=" + std::to_string(device) + " n=" + std::to_string(n)
                   + " arm_us=" + median(arm) + " gemm_us=" + median(gemm)
                   + " wait_us=" + median(wait) + " total_us=" + median(total)
                   + " xfer_first_start_us=" + median(firstStart)
                   + " xfer_last_end_us=" + median(lastEnd) + " q_dur_us=" + list(qDur, used)
                   + " q_end_us=" + list(qEnd, used) + " q_ready_us=" + list(qReady, used);
        }
    } // namespace Client
} // namespace TensileLite
