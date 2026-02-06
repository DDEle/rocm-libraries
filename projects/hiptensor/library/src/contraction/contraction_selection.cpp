/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <numeric>

#ifndef CHECK_HIP_ALLOC
#define CHECK_HIP_ALLOC(status)               \
    if(status != hipSuccess)                  \
    {                                         \
        return HIPTENSOR_STATUS_ALLOC_FAILED; \
    }
#endif

#include <ck/stream_config.hpp>

#include "contraction_selection.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "performance.hpp"
#include "util.hpp"

namespace hiptensor
{
    hiptensorStatus_t bruteForceModel(ContractionSolution**              winner,
                                      std::vector<ContractionSolution*>& candidates,
                                      hiptensorDataType_t                typeA,
                                      std::vector<std::size_t> const&    a_ms_ks_lengths,
                                      std::vector<std::size_t> const&    a_ms_ks_strides,
                                      std::vector<int32_t> const&        a_ms_ks_modes,
                                      hiptensorDataType_t                typeB,
                                      std::vector<std::size_t> const&    b_ns_ks_lengths,
                                      std::vector<std::size_t> const&    b_ns_ks_strides,
                                      std::vector<int32_t> const&        b_ns_ks_modes,
                                      hiptensorDataType_t                typeD,
                                      std::vector<std::size_t> const&    d_ms_ns_lengths,
                                      std::vector<std::size_t> const&    d_ms_ns_strides,
                                      std::vector<int32_t> const&        d_ms_ns_modes,
                                      hiptensorDataType_t                typeE,
                                      std::vector<std::size_t> const&    e_ms_ns_lengths,
                                      std::vector<std::size_t> const&    e_ms_ns_strides,
                                      std::vector<int32_t> const&        e_ms_ns_modes,
                                      hiptensorComputeDescriptor_t       computeType,
                                      const uint64_t                     workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementsFromLengths(a_ms_ks_lengths) * hiptensorDataTypeSize(typeA);
        auto sizeB = elementsFromLengths(b_ns_ks_lengths) * hiptensorDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementsFromLengths(d_ms_ns_lengths) * hiptensorDataTypeSize(typeD);
        }
        auto sizeE = elementsFromLengths(e_ms_ns_lengths) * hiptensorDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *wspace;

        /*
         * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
         * ```
         * alphaF = hiptensor::readVal<float>(
         *      alpha, convertToComputeType(HipTensorDataType_v<typename Traits::ComputeDataT>));
         * ```
         * Hence, the `alpha` and `bete` need to point to a ComputeData value
         */
        ScalarData alpha;
        ScalarData beta;
        if(computeType == HIPTENSOR_COMPUTE_DESC_C32F || computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            writeVal(&alpha, computeType, {computeType, 1.02, 1.03});
            writeVal(&beta, computeType, {computeType, 1.04, 1.05});
        }
        else
        {
            writeVal(&alpha, computeType, ScalarData(computeType, 1.02));
            writeVal(&beta, computeType, ScalarData(computeType, 1.03));
        }

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

        std::string          best_op_name;
        ContractionSolution* bestSolution = nullptr;
        PerfMetrics          bestMetrics  = {
            0,
            "",
            0,
            0,
            0,
        };

        std::vector<float> sol_times(candidates.size(), std::numeric_limits<float>::max());
        std::vector<int>   indices(candidates.size());
        std::iota(indices.begin(), indices.end(), 0);
        int idx = 0;
        for(auto* solution : candidates)
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();

            auto [errorCode, time] = (*solution)(&alpha,
                                                 A_d,
                                                 B_d,
                                                 &beta,
                                                 D_d,
                                                 E_d,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 e_ms_ns_modes,
                                                 wspace,
                                                 workspaceSize,
                                                 StreamConfig{
                                                     nullptr, // stream id
                                                     true, // time_kernel
                                                     0, // log_level
                                                     options->coldRuns(), // cold_niters
                                                     options->hotRuns(), // nrepeat
                                                 });
            if(errorCode == HIPTENSOR_STATUS_SUCCESS && time > 0)
            {
                // Make sure to time the kernels
                int32_t m, n, k;
                std::tie(m, n, k) = solution->problemDims();
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = solution->problemBytes();

                PerfMetrics metrics = {
                    solution->uid(), // id
                    solution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                using hiptensor::Logger;
                auto& logger = Logger::instance();

                // Log brute force timings for actor critic training
                if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE)
                {
                    // Log Kernel performances access
                    char msg[256];
                    snprintf(msg,
                             sizeof(msg),
                             "KernelId: %lu, KernelName: %s, AvgTime: %0.3f ms",
                             solution->uid(),
                             solution->kernelName().c_str(),
                             time);

                    logger->logHeuristics("BRUTE_FORCE_KERNEL_PERF", msg);
                }

                if(metrics > bestMetrics)
                {
                    bestSolution = solution;
                    bestMetrics  = metrics;
                }

                sol_times[idx] = time;
            }

            idx++;
        }

        CHECK_HIP_ALLOC(hipFree(A_d));
        CHECK_HIP_ALLOC(hipFree(B_d));
        CHECK_HIP_ALLOC(hipFree(D_d));
        CHECK_HIP_ALLOC(hipFree(E_d));
        CHECK_HIP_ALLOC(hipFree(wspace));

        *winner = bestSolution;

        //Sort candidates based on performance (from fastest to slowest)
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return sol_times[i] < sol_times[j];
        });
        std::vector<ContractionSolution*> tmpCandidates = candidates;
        candidates.clear();
        for(auto idx : indices)
            candidates.push_back(tmpCandidates[idx]);

        if(bestSolution == nullptr)
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
        else
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 7592803058165012586ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 12963881581420137103ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12963881581420137103ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 7592803058165012586ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14572544347617725147ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14572544347617725147ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 13173011218142769592ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 5799361496203926984ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 11068597454742748374ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 4539585187510392307ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 8054964659815705137ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 4539585187510392307ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 2317674114976786230ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2317674114976786230ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 2317674114976786230ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 12241437837959333440ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 11152060091307708334ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 13479669740975670080ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 5329463245387935369ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 5329463245387935369ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 5329463245387935369ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13479669740975670080ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 13479669740975670080ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 15321674118078740235ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13009807876834380836ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 305893002468456073ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15321674118078740235ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 6882237411832340455ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15321674118078740235ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 872672380373754190ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 872672380373754190ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 16476891743625221381ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 16476891743625221381ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 16476891743625221381ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 16476891743625221381ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 58303249112943560ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    // if (rank == 1 || (rank == 1 && (a_ms_ks_lengths[3] == 1 || b_ns_ks_lengths[3] == 1)))
                    {
                        unique_id = 58303249112943560ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2303552229010777601ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 58303249112943560ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 58303249112943560ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 58303249112943560ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 2303552229010777601ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 13661944739771767642ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 3369570672026525371ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 3369570672026525371ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 13661944739771767642ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13661944739771767642ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 13661944739771767642ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 8806035383379090954ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13745027619546548194ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 10975998290795516145ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1018134096730844503ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 5123488503279960080ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 1018134096730844503ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 9967477699864925937ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 14071475272156866885ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14071475272156866885ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 15452087623356707112ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8307633941691601884ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 17045489070375640837ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 9340331219857506915ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9340331219857506915ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 17045489070375640837ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 17045489070375640837ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8058086607921061755ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 15841120447249960783ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 9365615260785767349ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 6421129557103868873ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15841120447249960783ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 6472255990607251790ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15841120447249960783ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 9344798352708026060ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 16299024124514902126ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 378062791888302715ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 76527422265261696ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 378062791888302715ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 378062791888302715ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 378062791888302715ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 378062791888302715ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 211015953214891608ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 15273669775026901782ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 15273669775026901782ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1852617601289447141ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 1852617601289447141ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 5750553339656692298ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 16667434208488764313ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 3304158588024671429ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14607853679262083734ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15108266721455137072ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 15108266721455137072ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15108266721455137072ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 13825918879176996502ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13825918879176996502ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 17141562253969597117ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 6384780398804323250ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 14527221648179966882ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 17436875102140281580ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 17436875102140281580ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14527221648179966882ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14527221648179966882ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14527221648179966882ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 9676818724448448763ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13024694112602607256ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12669183620564506227ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 9676818724448448763ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 9676818724448448763ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 9676818724448448763ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 11208787066124811014ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 11208787066124811014ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14522095938220523368ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14522095938220523368ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14522095938220523368ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14522095938220523368ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 8251132190088736039ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 2897979232477761524ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 11575020378192019953ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2046007293163647751ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 2046007293163647751ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 11575020378192019953ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 9408086384082452854ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 9408086384082452854ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 9408086384082452854ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 6588505822864401664ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 3073499294115377607ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 539439838650790588ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 539439838650790588ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 539439838650790588ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 13613206280884761703ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13613206280884761703ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 13613206280884761703ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 2008216990064456310ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 4373449368168185126ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13613206280884761703ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15116758930810193332ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float,
                                float,
                                float,
                                float,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 11232444990449383730ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13489983279354414642ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 13489983279354414642ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 11232444990449383730ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 11232444990449383730ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 11933153923365960148ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 17357177205022860370ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 5926125609917490146ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 6098505431290410348ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 17357177205022860370ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 17357177205022860370ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 17357177205022860370ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 15864809842584901464ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8067958629699904967ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 15864809842584901464ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 6775599605174985174ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 6775599605174985174ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 5326563676026437938ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 8067958629699904967ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8116863550692548667ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 8100982996427419773ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 14796705998938117206ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14796705998938117206ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8100982996427419773ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 8100982996427419773ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8100982996427419773ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 9606383751520935483ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 9566739697123109032ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 17158178843835067496ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15786846799311798473ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 15786846799311798473ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15786846799311798473ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 17939389824758640014ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 10640128726648594287ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 5794367356792942822ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 13933081369664111675ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 14929556035596302079ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 5619866926768181192ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 5619866926768181192ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14929556035596302079ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14929556035596302079ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14929556035596302079ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 14411691597838362439ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 6720065723374347618ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 844285384873827484ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14411691597838362439ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 16256098050562613535ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14411691597838362439ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2224053047801499357ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 3431382583157381293ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 5422513160360085353ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 3431382583157381293ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 3431382583157381293ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14915761978535949477ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 18207091374964962208ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9629623372085111383ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13775481144453754924ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 18207091374964962208ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9629623372085111383ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13775481144453754924ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8731312090917039005ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 16870758234615651290ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 14901158961446820896ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 16870758234615651290ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8188562791036959263ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 16870758234615651290ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 16870758234615651290ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 18207091374964962208ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 16948282955506101335ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 16870758234615651290ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15355329505248522280ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14642257549075851915ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14642257549075851915ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 11397325986760035504ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9857262295729909659ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 16017423316310263739ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 11397325986760035504ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 9857262295729909659ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 4879133739527452570ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 1546833748864254436ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 12057130050439892271ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 13038089902448627981ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12057130050439892271ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 11269655469469274301ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 11269655469469274301ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 12057130050439892271ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 11269655469469274301ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 2143493311543532856ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 6406117030749216765ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8021137963958390646ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 3248584345341330494ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 3879892272436099392ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 7950787545240972863ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 4041813994497895944ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 4041813994497895944ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 4041813994497895944ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 4041813994497895944ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 4041813994497895944ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 7591632339673577634ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 2054609181761357786ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 14145390177844245465ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 3745477831989292138ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 15089541083742110906ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 15089541083742110906ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 3745477831989292138ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 3745477831989292138ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 11298912133205666592ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 2244461952804365798ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8740757002353521848ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 16832013176668170275ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 2244461952804365798ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 4658749468762430362ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 2244461952804365798ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 4348837698146370003ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 1688099565795560288ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 4363356859752806590ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 8820054978816893265ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 12104612309054914898ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 13363982225602645541ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8820054978816893265ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 13247354055351735256ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 13247354055351735256ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 2872521460614732031ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8820054978816893265ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 13363982225602645541ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 5665483458853925283ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 7734338110657628671ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 6174015219279904210ull;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 15330878641001915472ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 11537900932066889768ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8338926107119209426ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 11537900932066889768ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 11537900932066889768ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 11537900932066889768ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 11537900932066889768ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 10254320286859648634ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 15705829219230515535ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 12959721676360111684ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 10254320286859648634ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 10254320286859648634ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 10254320286859648634ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(handle->getDevice().getGcnArch() == HipDevice::hipGcnArch_t::GFX1250)
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
                else
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 0;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 0;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 0;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 0;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 0;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 0;
                    }
                }
            }
            else
            {
                if(options->isColMajorStrides())
                {
                    // m1n1k1
                    if(rank == 1)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 1322366267556764247ull;
                    }
                }
                else
                {
                    bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                                || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                    // rank2 dim1 case
                    if(rank == 2 && dim1)
                    {
                        unique_id = 14051358583041094215ull;
                    }
                    // m1n1k1
                    else if(rank == 1)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                    // m2n2k2
                    else if(rank == 2)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                    // m3n3k3
                    else if(rank == 3)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                    // m4n4k4
                    else if(rank == 4)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                    // m5n5k5
                    else if(rank == 5)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                    // m6n6k6
                    else if(rank == 6)
                    {
                        unique_id = 8503926755447648324ull;
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    hiptensorStatus_t
        actorCriticModel(const hiptensorHandle_t                                 handle,
                         ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         hiptensorComputeDescriptor_t                            computeType,
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
           && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(handle,
                                                                winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(handle,
                                                                winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(handle,
                                                                    winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(handle,
                                                                    winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(handle,
                                                             winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>::selectWinner(handle,
                                                              winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>::selectWinner(handle,
                                                              winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipFloatComplex>::selectWinner(handle,
                                                                       winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == HIPTENSOR_C_32F
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipFloatComplex>::selectWinner(handle,
                                                                       winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipDoubleComplex>::selectWinner(handle,
                                                                        winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == HIPTENSOR_C_64F
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipDoubleComplex>::selectWinner(handle,
                                                                        winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
