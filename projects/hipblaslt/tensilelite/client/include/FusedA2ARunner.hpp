// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DeviceContext.hpp"
#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        class SdmaQueue;

        struct FusedA2AIteration
        {
            // -1 marks a card whose kernel returned a HIP error.
            std::vector<double> cardUs;
            bool                ok = true;
        };

        // Rank r runs on device r.
        class FusedA2ARunner
        {
        public:
            static int  worldSize(po::variables_map const& args);
            static void configureProblem(po::variables_map const& args,
                                         ContractionProblemGemm&  problem,
                                         int                      world,
                                         int                      runIdx);

            FusedA2ARunner(po::variables_map const&                           args,
                           Hardware const&                                    hardware,
                           ContractionProblemGemm const&                      problem,
                           ContractionSolution const&                         solution,
                           std::vector<std::shared_ptr<DeviceContext>> const& devices);
            ~FusedA2ARunner();

            FusedA2ARunner(FusedA2ARunner const&)            = delete;
            FusedA2ARunner& operator=(FusedA2ARunner const&) = delete;

            // All W cards are enqueued before any is synchronized.
            FusedA2AIteration launchIteration(int iteration);

            std::vector<void*> const& recv() const
            {
                return m_recv;
            }

        private:
            std::vector<std::shared_ptr<DeviceContext>> m_devices;

            bool     m_checkWptr;
            uint32_t m_drainSend;
            uint32_t m_am;
            size_t   m_recvBytes;
            size_t   m_counterBytes;
            size_t   m_counterAllocBytes;

            std::vector<void*>                                   m_peer;
            std::vector<void*>                                   m_counter;
            std::vector<void*>                                   m_flag;
            std::vector<void*>                                   m_recv;
            std::vector<std::vector<std::shared_ptr<SdmaQueue>>> m_queues;
            std::vector<std::vector<uint64_t>>                   m_prevWptr;
            std::vector<std::vector<KernelInvocation>>           m_kernels;
            std::vector<hipEvent_t>                              m_start;
            std::vector<hipEvent_t>                              m_stop;
        };
    } // namespace Client
} // namespace TensileLite
