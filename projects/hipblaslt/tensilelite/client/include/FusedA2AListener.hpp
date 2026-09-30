// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DeviceContext.hpp"
#include "FusedA2ARunner.hpp"
#include "ProgramOptions.hpp"
#include "ReferenceValidator.hpp"
#include "RunListener.hpp"
#include "TimingEvents.hpp"

#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Validates each card's whole D through its own ReferenceValidator and
        // each card's recv against the slices of every sender's golden. The
        // per-card validators are not on the chain; this listener reports one
        // Validation value per solution.
        class FusedA2AListener : public RunListener
        {
        public:
            FusedA2AListener(po::variables_map const&                    args,
                             std::vector<std::shared_ptr<DeviceContext>> devices);

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual void preProblem(ContractionProblem* const problem) override;
            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }

            virtual size_t numWarmupRuns() override
            {
                return m_enabled ? 1 : 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override {}
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues(hipStream_t const& stream) override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return m_errors;
            }

            // recv[d] is card d's device-side recv buffer.
            void postIteration(int                       iteration,
                               bool                      measured,
                               FusedA2AIteration const&  result,
                               std::vector<void*> const& recv);

            // Per measured iteration with every card reporting: the slowest card.
            std::vector<double> const& latenciesUs() const
            {
                return m_latenciesUs;
            }

        private:
            bool recvMatches(int iteration, std::vector<void*> const& recv);

            std::vector<std::shared_ptr<DeviceContext>>      m_devices;
            std::vector<std::shared_ptr<ReferenceValidator>> m_validators;
            TimingEvents                                     m_noEvents{0, 0};

            bool                    m_enabled;
            ContractionProblemGemm* m_problem = nullptr;

            bool                             m_executed     = false;
            int                              m_firstFailure = -1;
            int                              m_skipped      = 0;
            std::vector<double>              m_latenciesUs;
            std::vector<std::vector<double>> m_cardUs;
            std::vector<int>                 m_slowest;
            int                              m_errors = 0;
        };
    } // namespace Client
} // namespace TensileLite
