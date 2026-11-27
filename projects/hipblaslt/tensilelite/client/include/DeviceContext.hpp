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
#include <utility>
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
            // sdmaPeers: device ids to enable P2P to and open one SDMA queue each for outside
            // directMask, indexed by rank; the entry equal to deviceId is a loopback queue.
            // sdmaRotate: the `rotate` of sdmaQueueEngine.
            // fusedA2ADirect: directMask = fusedA2ADirectMask.
            DeviceContext(int                     deviceId,
                          bool                    useDefaultStream,
                          size_t                  commFlagsBytes = 0,
                          std::vector<int> const& sdmaPeers      = {},
                          int                     sdmaRotate     = 0,
                          bool                    fusedA2ADirect = false);
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
            // Bit j: sdmaPeers[j] is direct (fusedA2ADirectMask); its sdmaQueues entry is null.
            uint32_t directMask = 0;

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

        void resetCommBuffers(ContractionProblem const&                          problem,
                              std::vector<std::shared_ptr<DeviceContext>> const& devices);

        // Rotating slots to request for `warmups` warmup launches, `profilerRuns` profiler
        // launches and `syncs` groups of `enqueues` launches: at least 2 for GemmRS at world > 1
        // when two launches can be queued with no host synchronization between them.
        size_t fusedRotatingSlots(ContractionProblem const& problem,
                                  size_t                    warmups,
                                  size_t                    profilerRuns,
                                  size_t                    syncs,
                                  size_t                    enqueues);

        // Throws when fusedRotatingSlots would request 2 for GemmRS and there are fewer.
        void checkFusedRotatingSlots(ContractionProblem const& problem,
                                     size_t                    slots,
                                     size_t                    warmups,
                                     size_t                    profilerRuns,
                                     size_t                    syncs,
                                     size_t                    enqueues);

        int a2aBufClearByte(FusedA2A mode);

        // The buffer rank `rank` writes to on peer `peer`: the peer's a2aBuf, or its B for
        // A2AGemm on another rank.
        void const* peerBuffer(FusedA2A                 mode,
                               size_t                   peer,
                               size_t                   rank,
                               ContractionInputs const& peerInputs);

        // Copies devices[rank]'s inputs into scratch and adds the peers of the launch's slot
        // and channel.
        ProblemInputs const* rankInputs(ContractionInputs&                                 scratch,
                                        ContractionProblem const&                          problem,
                                        std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                        size_t                                             rank,
                                        size_t                                             launch);

        // Same as forEachDevice, also passing the device's index in `devices`.
        template <typename F>
        void forEachDeviceIndexed(std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                  F&&                                                f)
        {
            for(size_t d = 0; d < devices.size(); d++)
            {
                ScopedDevice guard(devices[d]->deviceId);
                f(*devices[d], d);
            }
        }

        // {offset, bytes} of every other source's slot in rank `rank`'s GemmRS receive buffer.
        std::vector<std::pair<size_t, size_t>>
            gemmRSPeerSlotBytes(ContractionProblemGemm const& problem, size_t rank);

        // Whether launch `launch` (rotating slot launch % slots) is followed by another launch of
        // its slot before launch `end`; false with fewer than 2 slots.
        bool gemmRSSlotReusedBefore(size_t launch, size_t slots, size_t end);

        // Fills the gemmRSPeerSlotBytes ranges of every rank with 0xFF (BF16 NaN). Every call is a
        // no-op unless `validating` and the problem is GemmRS with world > 1.
        class GemmRSRecvPoison
        {
        public:
            GemmRSRecvPoison(ContractionProblem const&                          problem,
                             std::vector<std::shared_ptr<DeviceContext>> const& devices,
                             bool                                               validating);

            bool enabled() const
            {
                return m_gemm != nullptr;
            }

            // Synchronizes every stream, fills every rotating slot, synchronizes again.
            void fillIdle();

            // The next validation follows launch end - 1.
            void validateBefore(size_t end)
            {
                m_end = end;
            }

            // Fills, on rank d's stream, the slot of launch `launch` when gemmRSSlotReusedBefore.
            void afterLaunch(size_t d, size_t launch);

            size_t idleFills() const
            {
                return m_idleFills;
            }

            size_t launchFills() const
            {
                return m_launchFills;
            }

        private:
            void fill(size_t d, size_t slot);

            ContractionProblemGemm const*                      m_gemm = nullptr;
            std::vector<std::shared_ptr<DeviceContext>> const& m_devices;
            size_t                                             m_end         = 0;
            size_t                                             m_idleFills   = 0;
            size_t                                             m_launchFills = 0;
        };
    } // namespace Client
} // namespace TensileLite
