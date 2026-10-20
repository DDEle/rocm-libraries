// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        class DataInitialization;
        class SdmaQueue;

        class ScopedDevice
        {
        public:
            explicit ScopedDevice(int deviceId)
            {
                HIP_CHECK_EXC(hipGetDevice(&m_prevDeviceId));
                HIP_CHECK_EXC(hipSetDevice(deviceId));
            }

            ~ScopedDevice()
            {
                static_cast<void>(hipSetDevice(m_prevDeviceId));
            }

            ScopedDevice(ScopedDevice const&)            = delete;
            ScopedDevice& operator=(ScopedDevice const&) = delete;

        private:
            int m_prevDeviceId = 0;
        };

        struct DeviceContext
        {
            // sdmaPeers: device ids to enable P2P to and open one SDMA queue each for,
            // indexed by rank; the entry equal to deviceId is a loopback queue.
            DeviceContext(int                     deviceId,
                          bool                    useDefaultStream,
                          size_t                  commFlagsBytes = 0,
                          std::vector<int> const& sdmaPeers      = {});
            ~DeviceContext();

            DeviceContext(DeviceContext const&)            = delete;
            DeviceContext& operator=(DeviceContext const&) = delete;

            int                                         deviceId;
            hipStream_t                                 stream = nullptr;
            std::shared_ptr<hip::SolutionAdapter>       adapter;
            std::shared_ptr<DataInitialization>         dataInit;
            std::shared_ptr<ProblemInputs>              inputs;
            std::vector<std::shared_ptr<ProblemInputs>> inputArr;
            size_t                                      commFlagsBytes;
            void*                                       commFlags = nullptr;
            std::vector<std::shared_ptr<SdmaQueue>>     sdmaQueues;
            void*                                       a2aCounter      = nullptr;
            size_t                                      a2aCounterBytes = 0;
            size_t                                      a2aPayloadBytes = 0;
            size_t                                      launches        = 0;

            // Grows the A2A+GEMM counter, zeroes its payload and arms the guard tail;
            // zeroes commFlags and poisons every slot's B.
            void armA2AGemm(size_t counterAllocBytes, size_t counterPayloadBytes, size_t bBytes);

        private:
            void release();
        };

        constexpr size_t fusedA2ACommChannels(FusedA2A mode)
        {
            return mode == FusedA2A::GemmA2A ? 2 : 1;
        }

        constexpr size_t fusedA2ACommFlagsBytes(FusedA2A mode)
        {
            return mode == FusedA2A::Disabled
                       ? 0
                       : fusedA2ACommChannels(mode) * FUSED_A2A_FLAG_BLOCK_BYTES;
        }

        template <typename F>
        void forEachDevice(std::vector<std::shared_ptr<DeviceContext>> const& devices, F&& f)
        {
            for(auto const& ctx : devices)
            {
                ScopedDevice guard(ctx->deviceId);
                f(*ctx);
            }
        }

        // devices: the problem's ranks, rank r on devices[r]. A no-op unless the problem is
        // fused. GemmA2A zeroes every slot's A2A_BUF; A2AGemm arms every rank.
        void resetA2ABuffers(ContractionProblem const&                          problem,
                             ContractionSolution const&                         solution,
                             Hardware const&                                    hardware,
                             std::vector<std::shared_ptr<DeviceContext>> const& devices);

        // Copies devices[rank]'s inputs into scratch and adds the peers of the launch's slot
        // and channel.
        ProblemInputs const* rankInputs(ContractionInputs&                                 scratch,
                                        ContractionProblem const&                          problem,
                                        ContractionSolution const&                         solution,
                                        Hardware const&                                    hardware,
                                        std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                        size_t                                             rank,
                                        size_t                                             launch);
    } // namespace Client
} // namespace TensileLite
