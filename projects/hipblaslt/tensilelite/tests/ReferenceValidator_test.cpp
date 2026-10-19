// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DeviceContext.hpp"
#include "ProgramOptions.hpp"
#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include "TimingEvents.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    template <typename T>
    void setOption(po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    po::variables_map makeValidatorArgs()
    {
        po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Ailk_Bljk_Cijk_Dijk"));
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{{64, 64, 1, 64}});
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::Float);
        setOption(args, "a-type", rocisa::DataType::Float);
        setOption(args, "b-type", rocisa::DataType::Float);
        setOption(args, "c-type", rocisa::DataType::Float);
        setOption(args, "d-type", rocisa::DataType::Float);
        setOption(args, "alpha-type", rocisa::DataType::Float);
        setOption(args, "beta-type", rocisa::DataType::Float);
        setOption(args, "compute-input-type-A", rocisa::DataType::Float);
        setOption(args, "compute-input-type-B", rocisa::DataType::Float);
        setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
        setOption(args, "activation-compute-type", rocisa::DataType::Float);
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "mx-scale-format", 0);
        setOption(args, "fused-a2a", std::string("None"));
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensorOps{});
        setOption(args, "b-ops", TensorOps{});
        setOption(args, "c-ops", TensorOps{});
        setOption(args, "d-ops", TensorOps{});
        setOption(args, "use-e", false);
        setOption(args, "use-gradient", false);
        setOption(args, "output-amaxD", false);
        setOption(args, "use-scaleAB", std::string());
        setOption(args, "use-scaleCD", false);
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        setOption(args, "device-idx", 0);
        setOption(args, "pristine-on-gpu", true);
        setOption(args, "prune-mode", PruneSparseMode::PruneRandom);
        setOption(args, "rotating-buffer-size", 0);
        setOption(args, "rotating-buffer-mode", 0);
        setOption(args, "bounds-check", BoundsCheckMode::Disable);
        setOption(args, "init-a", InitMode::Random);
        setOption(args, "init-b", InitMode::Random);
        setOption(args, "init-c", InitMode::Zero);
        setOption(args, "init-d", InitMode::Zero);
        setOption(args, "init-alpha", InitMode::One);
        setOption(args, "init-beta", InitMode::Zero);
        setOption(args, "num-elements-to-validate", -1);
        setOption(args, "print-valids", false);
        setOption(args, "print-max", 4);
        setOption(args, "print-tensor-a", false);
        setOption(args, "print-tensor-b", false);
        setOption(args, "print-tensor-c", false);
        setOption(args, "print-tensor-d", false);
        setOption(args, "print-tensor-ref", false);
        setOption(args, "print-tensor-bias", false);
        setOption(args, "print-tensor-gate", false);
        setOption(args, "print-tensor-scale-alpha-vec", false);
        setOption(args, "print-tensor-amaxd", false);
        return args;
    }

    class CapturingReporter : public ResultReporter
    {
    public:
        void reportValue_string(std::string const& key, std::string const& value) override
        {
            strings[key] = value;
        }
        void reportValue_uint(std::string const& key, uint64_t value) override {}
        void reportValue_int(std::string const& key, int64_t value) override {}
        void reportValue_double(std::string const& key, double value) override {}
        void reportValue_sizes(std::string const& key, std::vector<size_t> const& value) override {}
        void reportValue_vecOfSizes(std::string const&                      key,
                                    std::vector<std::vector<size_t>> const& value) override
        {
        }
        void finalizeReport() override {}

        std::map<std::string, std::string> strings;
    };

    class ReferenceValidatorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int n = 0;
            if(hipGetDeviceCount(&n) != hipSuccess || n < 1)
                GTEST_SKIP() << "needs a device";
            factory          = std::make_unique<ClientProblemFactory>(args);
            dataInit         = std::make_shared<DataInitialization>(args, *factory);
            problem          = factory->problems()[0].get();
            device           = std::make_shared<DeviceContext>(0, true);
            device->dataInit = dataInit;
        }

        std::unique_ptr<ReferenceValidator> start()
        {
            devices        = {device};
            auto validator = std::make_unique<ReferenceValidator>(args);
            validator->setDevices(&devices);
            validator->setReporter(reporter);
            validator->preProblem(problem);
            validator->preSolution(&solution);
            gpu = std::dynamic_pointer_cast<ContractionInputs>(dataInit->prepareGPUInputs(problem));
            device->inputArr = {gpu};
            ref = std::dynamic_pointer_cast<ContractionInputs>(validator->referenceInputs(0));
            auto const& d = dynamic_cast<ContractionProblemGemm*>(problem)->d();
            HIP_CHECK_EXC(
                hipMemcpy(gpu->d, ref->d, d.totalAllocatedBytes(), hipMemcpyHostToDevice));
            return validator;
        }

        void corruptFirstElement()
        {
            float bad = static_cast<float const*>(ref->d)[0] + 100.0f;
            HIP_CHECK_EXC(hipMemcpy(gpu->d, &bad, sizeof(float), hipMemcpyHostToDevice));
        }

        std::string validation()
        {
            return reporter->strings.at(ResultKey::Validation);
        }

        po::variables_map                     args = makeValidatorArgs();
        std::unique_ptr<ClientProblemFactory> factory;
        std::shared_ptr<DataInitialization>   dataInit;
        std::shared_ptr<DeviceContext>        device;
        std::vector<std::shared_ptr<DeviceContext>> devices;
        ContractionProblem*                   problem = nullptr;
        ContractionSolution                   solution;
        std::shared_ptr<CapturingReporter>    reporter = std::make_shared<CapturingReporter>();
        std::shared_ptr<ContractionInputs>    gpu;
        std::shared_ptr<ContractionInputs>    ref;
    };
}

TEST_F(ReferenceValidatorTest, OneShotValidatesOnlyTheFirstRun)
{
    auto validator = start();
    EXPECT_TRUE(validator->needMoreRunsInSolution());
    EXPECT_EQ(validator->numWarmupRuns(), 1u);
    TimingEvents events({}, 1, 1);
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    EXPECT_FALSE(validator->needMoreRunsInSolution());
    corruptFirstElement();
    validator->validateEnqueues({0}, events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "PASSED");
    EXPECT_EQ(validator->error(), 0);
}

namespace
{
    class A2AGemmReferenceTest : public ReferenceValidatorTest
    {
    protected:
        static constexpr size_t world = 2;

        A2AGemmReferenceTest()
        {
            setOption(args, "fused-a2a", std::string("A2AGemm"));
            setOption(args, "world", std::vector<int>{static_cast<int>(world)});
            setOption(args, "fused-a2a-extent", std::vector<int>{0});
        }

        void startRanks()
        {
            solution.problemType.fusedA2A = FusedA2A::A2AGemm;
            device1                       = std::make_shared<DeviceContext>(0, true);
            device1->dataInit             = std::make_shared<DataInitialization>(args, *factory);
            devices                       = {device, device1};
            validator                     = std::make_unique<ReferenceValidator>(args);
            validator->setDevices(&devices);
            validator->setReporter(reporter);
            validator->preProblem(problem);
            validator->preSolution(&solution);
            for(size_t d = 0; d < world; d++)
            {
                devices[d]->inputArr = {std::dynamic_pointer_cast<ContractionInputs>(
                    devices[d]->dataInit->prepareGPUInputs(problem))};
                ref[d]               = std::dynamic_pointer_cast<ContractionInputs>(
                    validator->referenceInputs(d));
            }
        }

        ContractionProblemGemm const& gemm() const
        {
            return dynamic_cast<ContractionProblemGemm const&>(*problem);
        }

        // Rank d's device B: segment j holds the x block rank d of rank (d + j) % world.
        void uploadGathered(size_t d)
        {
            const size_t segBytes = gemm().freeSizeB(0) * (gemm().boundSize(0) / world)
                                    * gemm().b().elementBytes();
            auto const& in = dynamic_cast<ContractionInputs const&>(*devices[d]->inputArr[0]);
            for(size_t j = 0; j < world; j++)
                HIP_CHECK_EXC(hipMemcpy(static_cast<uint8_t*>(const_cast<void*>(in.b)) + j * segBytes,
                                        static_cast<uint8_t const*>(ref[(d + j) % world]->a2aBuf)
                                            + d * segBytes,
                                        segBytes,
                                        hipMemcpyHostToDevice));
            HIP_CHECK_EXC(hipMemcpy(in.d,
                                    ref[d]->d,
                                    gemm().d().totalAllocatedBytes(),
                                    hipMemcpyHostToDevice));
        }

        std::string validate()
        {
            TimingEvents events({}, 1, 1);
            validator->validateWarmups({0}, events, events);
            validator->postWarmup(events, events);
            validator->postSolution();
            return validation();
        }

        std::shared_ptr<DeviceContext>      device1;
        std::unique_ptr<ReferenceValidator> validator;
        std::shared_ptr<ContractionInputs>  ref[world];
    };
}

TEST_F(A2AGemmReferenceTest, ReferenceBInterleavesEveryRanksX)
{
    startRanks();
    const size_t N      = gemm().freeSizeB(0);
    const size_t K      = gemm().boundSize(0);
    const size_t kLocal = K / world;
    const size_t elem   = gemm().b().elementBytes();
    for(size_t d = 0; d < world; d++)
        for(size_t n = 0; n < N; n++)
            for(size_t r = 0; r < world; r++)
                EXPECT_EQ(0,
                          std::memcmp(static_cast<uint8_t const*>(ref[d]->b) + (n * K + r * kLocal) * elem,
                                      static_cast<uint8_t const*>(ref[r]->a2aBuf)
                                          + (d * N + n) * kLocal * elem,
                                      kLocal * elem))
                    << "rank " << d << " n " << n << " from " << r;
}

TEST_F(A2AGemmReferenceTest, GatheredSegmentsMatchTheSourceX)
{
    startRanks();
    const size_t xBytes = gemm().tensors()[ContractionProblemGemm::TENSOR::A2A_BUF]
                              .totalAllocatedBytes();
    ASSERT_NE(0, std::memcmp(ref[0]->a2aBuf, ref[1]->a2aBuf, xBytes));
    for(size_t d = 0; d < world; d++)
        uploadGathered(d);
    EXPECT_EQ(validate(), "PASSED");
    EXPECT_EQ(validator->error(), 0);
}

TEST_F(A2AGemmReferenceTest, CorruptGatheredSegmentFailsTheSolution)
{
    startRanks();
    for(size_t d = 0; d < world; d++)
        uploadGathered(d);
    const size_t segBytes = gemm().freeSizeB(0) * (gemm().boundSize(0) / world)
                            * gemm().b().elementBytes();
    uint8_t bad = ~static_cast<uint8_t const*>(ref[1]->a2aBuf)[0];
    HIP_CHECK_EXC(hipMemcpy(static_cast<uint8_t*>(const_cast<void*>(
                                dynamic_cast<ContractionInputs const&>(*devices[0]->inputArr[0]).b))
                                + segBytes,
                            &bad,
                            1,
                            hipMemcpyHostToDevice));
    EXPECT_EQ(validate(), "FAILED");
    EXPECT_EQ(validator->error(), 1);
}
