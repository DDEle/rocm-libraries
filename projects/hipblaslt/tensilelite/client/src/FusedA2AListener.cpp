// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FusedA2AListener.hpp"

#include "Reference.hpp"
#include "ResultReporter.hpp"

#include "MetaResultReporter.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

namespace TensileLite
{
    namespace Client
    {
        FusedA2AListener::FusedA2AListener(po::variables_map const&                    args,
                                           std::vector<std::shared_ptr<DeviceContext>> devices)
            : m_devices(std::move(devices))
            , m_enabled(args["num-elements-to-validate"].as<int>() != 0)
        {
            auto silent = std::make_shared<MetaResultReporter>();
            for(auto const& device : m_devices)
            {
                m_validators.push_back(
                    std::make_shared<ReferenceValidator>(args, device->dataInit, true));
                m_validators.back()->setReporter(silent);
            }
        }

        void FusedA2AListener::preProblem(ContractionProblem* const problem)
        {
            m_problem = dynamic_cast<ContractionProblemGemm*>(problem);
            for(size_t d = 0; d < m_devices.size(); d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                m_validators[d]->preProblem(problem);
            }
        }

        void FusedA2AListener::preSolution(ContractionSolution* const solution)
        {
            for(size_t d = 0; d < m_devices.size(); d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                m_validators[d]->preSolution(solution);
            }
            m_executed     = false;
            m_firstFailure = -1;
            m_skipped      = 0;
            m_latenciesUs.clear();
            m_cardUs.assign(m_devices.size(), {});
            m_slowest.assign(m_devices.size(), 0);
        }

        void FusedA2AListener::postIteration(int                       iteration,
                                             bool                      measured,
                                             FusedA2AIteration const&  result,
                                             std::vector<void*> const& recv)
        {
            m_executed = true;

            bool pass = result.ok;
            if(pass && m_enabled)
            {
                for(size_t d = 0; d < m_devices.size(); d++)
                {
                    ScopedDevice device(m_devices[d]->deviceId);
                    m_validators[d]->validateEnqueues(m_devices[d]->inputs, m_noEvents, m_noEvents);
                }
                pass = recvMatches(iteration, recv);
            }
            if(!pass && m_firstFailure < 0)
                m_firstFailure = iteration;

            if(!measured)
                return;
            auto const& cardUs = result.cardUs;
            if(std::any_of(cardUs.begin(), cardUs.end(), [](double us) { return us < 0.0; }))
            {
                m_skipped++;
                return;
            }
            size_t slowest = std::max_element(cardUs.begin(), cardUs.end()) - cardUs.begin();
            m_latenciesUs.push_back(cardUs[slowest]);
            m_slowest[slowest]++;
            for(size_t d = 0; d < cardUs.size(); d++)
                m_cardUs[d].push_back(cardUs[d]);
        }

        bool FusedA2AListener::recvMatches(int iteration, std::vector<void*> const& recv)
        {
            const size_t W        = m_devices.size();
            const size_t N        = m_problem->freeSizeB(0);
            const size_t nShard   = (size_t)m_problem->fusedA2AExtent() / W;
            auto const&  dDesc    = m_problem->d();
            const size_t dMStride = dDesc.strides()[m_problem->freeIndices()[0].d];
            const size_t dNStride = dDesc.strides()[m_problem->freeIndices()[1].d];

            // Slot src on card dst holds features [dst*nShard, (dst+1)*nShard) of
            // card src's golden, for every token.
            std::vector<BFloat16> host(W * N * nShard);
            bool                  pass = true;
            for(size_t dst = 0; dst < W; dst++)
            {
                {
                    ScopedDevice device(m_devices[dst]->deviceId);
                    HIP_CHECK_EXC(hipMemcpy(host.data(),
                                            recv[dst],
                                            host.size() * sizeof(BFloat16),
                                            hipMemcpyDeviceToHost));
                }
                size_t mismatches = 0;
                for(size_t src = 0; src < W; src++)
                {
                    auto const& golden = dynamic_cast<ContractionInputs const&>(
                        *m_validators[src]->referenceInputs());
                    auto const* goldenD = static_cast<BFloat16 const*>(golden.d);
                    for(size_t t = 0; t < N; t++)
                    {
                        for(size_t f = 0; f < nShard; f++)
                        {
                            BFloat16 got  = host[(src * N + t) * nShard + f];
                            BFloat16 want = goldenD[(dst * nShard + f) * dMStride + t * dNStride];
                            if(AlmostEqual(want, got))
                                continue;
                            if(mismatches < 5)
                                std::cerr << "[fused-a2a] RECV MISMATCH iter=" << iteration
                                          << " card=" << dst << " src=" << src << " t=" << t
                                          << " f=" << f << " got=" << (float)got
                                          << " want=" << (float)want << "\n";
                            mismatches++;
                        }
                    }
                }
                if(mismatches)
                {
                    std::cerr << "[fused-a2a] RECV card " << dst << ": " << mismatches
                              << " mismatches (iter " << iteration << ")\n";
                    pass = false;
                }
            }
            return pass;
        }

        void FusedA2AListener::postSolution()
        {
            bool failed = m_firstFailure >= 0;
            for(auto const& validator : m_validators)
            {
                int before = validator->error();
                validator->postSolution();
                if(validator->error() > before)
                    failed = true;
            }
            if(!m_executed)
                return;

            if(m_firstFailure >= 0)
                std::cout << "[fused-a2a] first failing iteration = " << m_firstFailure << "\n";
            if(m_skipped)
                std::cout << "[fused-a2a] timing: " << m_skipped
                          << " measured iteration(s) excluded (not all " << m_devices.size()
                          << " cards reported)\n";

            // The slowest-card histogram separates the three sources of the
            // max-vs-mean gap: concentrated on the first-enqueued id means
            // enqueue-order skew, concentrated elsewhere means one slow card,
            // spread evenly means order statistics over W roughly-iid cards.
            if(!m_latenciesUs.empty())
            {
                const size_t n     = m_latenciesUs.size();
                auto         pctOf = [](std::vector<double> v, double p) {
                    std::sort(v.begin(), v.end());
                    return v[std::min(v.size() - 1, (size_t)(p * (double)v.size()))];
                };

                std::cout << std::fixed << std::setprecision(1);
                for(size_t d = 0; d < m_devices.size(); d++)
                {
                    double sum = 0.0;
                    for(double v : m_cardUs[d])
                        sum += v;
                    std::cout << "[fused-a2a] per-card dev " << d << ": mean=" << (sum / (double)n)
                              << " us p50=" << pctOf(m_cardUs[d], 0.5)
                              << " us p90=" << pctOf(m_cardUs[d], 0.9) << " us  slowest in "
                              << m_slowest[d] << "/" << n << " iters\n";
                }

                std::vector<double> spread(n);
                for(size_t i = 0; i < n; i++)
                {
                    double lo = m_cardUs[0][i], hi = m_cardUs[0][i];
                    for(size_t d = 1; d < m_devices.size(); d++)
                    {
                        lo = std::min(lo, m_cardUs[d][i]);
                        hi = std::max(hi, m_cardUs[d][i]);
                    }
                    spread[i] = hi - lo;
                }
                std::cout << "[fused-a2a] per-card spread (max-min across " << m_devices.size()
                          << " cards): p50=" << pctOf(spread, 0.5)
                          << " us p90=" << pctOf(spread, 0.9) << " us max=" << pctOf(spread, 1.0)
                          << " us" << std::defaultfloat << std::endl;
            }

            m_reporter->report(ResultKey::Validation,
                               failed ? "FAILED" : (m_enabled ? "PASSED" : "NO_CHECK"));
            if(failed)
                m_errors++;
        }
    } // namespace Client
} // namespace TensileLite
