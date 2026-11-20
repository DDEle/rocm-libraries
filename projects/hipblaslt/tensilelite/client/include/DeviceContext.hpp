// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ViewVerify.hpp"

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
            // sdmaRotate: the `rotate` of sdmaQueueEngine.
            DeviceContext(int                     deviceId,
                          bool                    useDefaultStream,
                          size_t                  commFlagsBytes = 0,
                          std::vector<int> const& sdmaPeers      = {},
                          int                     sdmaRotate     = 0);
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
            // FusedA2ATransport=View block, fine-grained, sized by resetCommBuffers.
            void*  viewBlock      = nullptr;
            size_t viewBlockBytes = 0;
            // VIEW_DEBUG_TIMELINE ring, VIEW_STATS_BYTES, fine-grained.
            void* viewStats = nullptr;
            // Set by resetCommBuffers for a View solution, cleared by viewDrainQueues.
            bool viewQueuesArmed = false;
            // --fused-a2a-verify, sized by resetCommBuffers: the check block (VERIFY_BLOCK_BYTES,
            // fine-grained) and the goldens.
            void*               verifyBlock = nullptr;
            VerifyGoldenBuffers golden;

        private:
            void release();
        };

        constexpr size_t kCommChannels   = 2;
        constexpr size_t kCommFlagsBytes = kCommChannels * FUSED_A2A_FLAG_BLOCK_BYTES;

        constexpr int kFusedA2ASdmaRotateDefault = -1;

        template <typename F>
        void forEachDevice(std::vector<std::shared_ptr<DeviceContext>> const& devices, F&& f)
        {
            for(auto const& ctx : devices)
            {
                ScopedDevice guard(ctx->deviceId);
                f(*ctx);
            }
        }

        // devices: the problem's ranks, rank r on devices[r]. These are no-ops unless the
        // problem is fused.
        void checkFusedA2ASolution(ContractionProblem const&  problem,
                                   ContractionSolution const& solution,
                                   Hardware const&            hardware,
                                   size_t                     world);

        // First runs viewDrainQueues on every device and throws if one fails. Also sizes and
        // zeroes every device's view block for a FusedA2ATransport=View solution, and with
        // VIEW_DEBUG_TIMELINE in viewDebug its viewStats ring. With a verify layout, grows the
        // goldens to it and resets the check block.
        void resetCommBuffers(ContractionProblem const&                          problem,
                              ContractionSolution const&                         solution,
                              std::vector<std::shared_ptr<DeviceContext>> const& devices,
                              int                                                viewDebug = 0,
                              VerifyLayout const*                                verify = nullptr);

        bool fusedA2AViewTransport(ContractionSolution const& solution);

        // Copies devices[rank]'s inputs into scratch and adds the peers of the launch's slot
        // and channel. viewTransport: counter_ptr is devices[rank]'s view block instead of the
        // Synchronizer.
        ProblemInputs const* rankInputs(ContractionInputs&                                 scratch,
                                        ContractionProblem const&                          problem,
                                        std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                        size_t                                             rank,
                                        size_t                                             launch,
                                        bool viewTransport = false);
    } // namespace Client
} // namespace TensileLite
