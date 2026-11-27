/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "DeviceContext.hpp"
#include "RunListener.hpp"

#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "DataInitialization.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace Client
    {

        class ClientProblemFactory;

        // GemmRS references from golden[r], rank r's reference D. For every rank d, recvs[d] is
        // d's receive buffer (A2A_BUF): slot s holds golden[s] over d's token segment and slot d
        // is zero. reduced[d] is golden[d] with d's token segment replaced by
        // bf16(sum over s of float(golden[s])).
        void gemmRSReferences(ContractionProblemGemm const&      problem,
                              std::vector<void const*> const&    golden,
                              std::vector<std::vector<uint8_t>>& recvs,
                              std::vector<std::vector<uint8_t>>& reduced);

        struct GemmRSOwnRounding
        {
            size_t elements       = 0;
            size_t differing      = 0;
            size_t accepted       = 0;
            double maxDiff        = 0;
            double boundAtMaxDiff = 0;
            double maxRatio       = 0;
        };

        std::ostream& operator<<(std::ostream& stream, GemmRSOwnRounding const& stats);

        // D elements [first, last) spanning rank's token segment.
        std::pair<size_t, size_t>
            gemmRSOwnSpan(ContractionProblemGemm const& problem, size_t rank, size_t world);

        // Over rank's token segment, with Y = sum over s of float(golden[s][i]) and e = half a BF16
        // ulp of golden[rank][i] (the ulp of zero and subnormals is 2^-133), sets reduced[i] to the
        // device value where both are finite and it lies in [bf16(Y - e), bf16(Y + e)]. The bound
        // at i is the distance from reduced[i] to the end of that interval on the device value's
        // side. Assumes every partial and partial sum is exact in FP32. device holds D from
        // gemmRSOwnSpan's first element. Returns the comparison with reduced before the change.
        GemmRSOwnRounding gemmRSAcceptOwnRounding(ContractionProblemGemm const&   problem,
                                                  size_t                          rank,
                                                  std::vector<void const*> const& golden,
                                                  void const*                     device,
                                                  std::vector<uint8_t>&           reduced);

        // Reads result.d over gemmRSOwnSpan (from the device if result.gpu), applies
        // gemmRSAcceptOwnRounding to reduced and restores reduced on destruction.
        class GemmRSOwnRoundingScope
        {
        public:
            GemmRSOwnRoundingScope(ContractionProblemGemm const&   problem,
                                   size_t                          rank,
                                   std::vector<void const*> const& golden,
                                   ContractionInputs const&        result,
                                   std::vector<uint8_t>&           reduced);
            ~GemmRSOwnRoundingScope();
            GemmRSOwnRoundingScope(GemmRSOwnRoundingScope const&)            = delete;
            GemmRSOwnRoundingScope& operator=(GemmRSOwnRoundingScope const&) = delete;

            GemmRSOwnRounding const& stats() const
            {
                return m_stats;
            }

        private:
            BFloat16*             m_first = nullptr;
            std::vector<BFloat16> m_saved;
            GemmRSOwnRounding     m_stats;
        };

        bool validatesRecvBuffer(FusedA2A mode);

        // D is compared against gemmRSReferences' reduced D rather than the golden D.
        bool comparesReducedD(FusedA2A mode, int fusedRSWait);

        class ReferenceValidator : public RunListener
        {
        public:
            ReferenceValidator(po::variables_map const& args);

            virtual bool needMoreBenchmarkRuns() const override;
            virtual void preBenchmarkRun() override;
            virtual void postBenchmarkRun() override;

            virtual void preProblem(ContractionProblem* const problem) override;
            virtual void postProblem() override;

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override;

            virtual size_t numWarmupRuns() override;
            virtual void   setNumWarmupRuns(size_t count) override;
            virtual void   preWarmup() override;
            virtual void   postWarmup(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents) override;
            virtual void   validateWarmups(std::vector<size_t> const& slots,
                                           TimingEvents const&        startEvents,
                                           TimingEvents const&        stopEvents) override;

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
            virtual void preEnqueues() override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents) override
            {
            }
            virtual void validateEnqueues(std::vector<size_t> const& slots,
                                          TimingEvents const&        startEvents,
                                          TimingEvents const&        stopEvents) override;

            bool validate(ContractionProblemGemm const& problem,
                          ContractionInputs const&      reference,
                          ContractionInputs const&      result);

            bool checkResults(TensorDescriptor const& tensor,
                              void const*             refPtr,
                              void const*             resPtr,
                              size_t                  maxElements,
                              bool                    isgpu,
                              size_t                  validationStride,
                              double                  threshold);

            template <typename ValidType>
            bool checkResultsTyped(TensorDescriptor const& tensor,
                                   ValidType const*        reference,
                                   ValidType const*        result,
                                   size_t                  maxElement,
                                   bool                    isgpu,
                                   size_t                  validationStride,
                                   double                  threshold);

            void printTensors(ContractionProblemGemm const& problem,
                              ContractionInputs const&      reference,
                              ContractionInputs const&      result);

            virtual void finalizeReport() override;

            virtual int error() const override;

            std::shared_ptr<ProblemInputs> referenceInputs(size_t card) const
            {
                return m_referenceInputs[card];
            }

            void setSplit(ClientProblemFactory const* split)
            {
                m_split = split;
            }

        protected:
            // Allow lifecycle tests to observe validation without device transfers.
            virtual void validateSlots(std::vector<size_t> const& slots);

        private:
            void allocateResultBuffer(size_t bytes);

            // The problem device `card` runs: its slice under --split-devices, else m_problem.
            ContractionProblem* deviceProblem(size_t card) const;

            // --split-devices: point every non-owner device's reference A or B (the shared
            // operand) at the owner's host data, the data the kernels read.
            void redirectSharedInputs();

            bool shouldSkipNullTensor(const std::string& tensorName,
                                      bool hasNullPointer,
                                      bool hasZeroElements) const;

            std::vector<std::shared_ptr<ProblemInputs>> m_referenceInputs;
            std::vector<std::vector<uint8_t>>           m_referenceRecvs;
            std::vector<std::vector<uint8_t>>           m_gemmRSReduced;
            std::vector<std::vector<uint8_t>>           m_a2aGemmB;

            size_t                   m_cpuResultBufferSize = 0;
            std::shared_ptr<uint8_t> m_cpuResultBuffer;

            ContractionProblem* m_problem = nullptr;

            ClientProblemFactory const* m_split = nullptr;

            // The unsplit operand is shared: every device validates against the owner's data.
            bool m_sharedA = false;
            bool m_sharedB = false;

            bool m_enabled;
            FusedA2A m_fusedA2A              = FusedA2A::Disabled;
            int      m_fusedRSWait           = 0;
            bool     m_validateAfterEnqueues = false;

            int  m_elementsToValidate;
            bool m_printValids;
            int  m_printMax;

            bool m_printTensorA;
            bool m_printTensorB;
            bool m_printTensorC;
            bool m_printTensorD;
            bool m_printTensorRef;
            bool m_printTensorBias;
            bool m_printTensorGate;
            bool m_printTensorScaleAlphaVec;
            bool m_printTensorAmaxD;
            bool m_printAny;

            int m_numBenchmarkRuns = 0;

            bool   m_validatedSolution = false;
            bool   m_errorInSolution   = false;
            bool   m_error             = false;
            bool   m_executedSolution  = false;
            size_t m_errorsReported    = 0;

            bool validateSolution(size_t card, std::shared_ptr<ProblemInputs> inputs);
            void referenceRecvs(ContractionProblemGemm const& problem);
            std::vector<void const*> goldenDs() const;
            void interleaveA2AGemmReferences(ContractionProblemGemm const& problem);
        };
    } // namespace Client
} // namespace TensileLite
