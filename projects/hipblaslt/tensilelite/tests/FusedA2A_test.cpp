// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DeviceContext.hpp"
#include "FusedA2ARunner.hpp"
#include "ProgramOptions.hpp"
#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include "TimingEvents.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    constexpr size_t kM     = 256;
    constexpr size_t kN     = 16;
    constexpr size_t kK     = 32;
    constexpr int    kWorld = 2;

    template <typename T>
    void setOption(po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    po::variables_map makeFusedArgs(int elementsToValidate)
    {
        po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Alik_Bljk_Cijk_Dijk"));
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{{kM, kN, 1, kK}});
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", true);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::BFloat16);
        setOption(args, "a-type", rocisa::DataType::BFloat16);
        setOption(args, "b-type", rocisa::DataType::BFloat16);
        setOption(args, "c-type", rocisa::DataType::BFloat16);
        setOption(args, "d-type", rocisa::DataType::BFloat16);
        setOption(args, "alpha-type", rocisa::DataType::Float);
        setOption(args, "beta-type", rocisa::DataType::Float);
        setOption(args, "compute-input-type-A", rocisa::DataType::BFloat16);
        setOption(args, "compute-input-type-B", rocisa::DataType::BFloat16);
        setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
        setOption(args, "activation-compute-type", rocisa::DataType::Float);
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "mx-scale-format", 0);
        setOption(args, "fused-gemm-a2a", true);
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensorOps{});
        setOption(args, "b-ops", TensorOps{});
        setOption(args, "c-ops", TensorOps{});
        setOption(args, "d-ops", TensorOps{});
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
        setOption(args, "icache-rotate-copies", 0);
        setOption(args, "icache-flush-args", std::vector<bool>(1, false));
        setOption(args, "fused-a2a-drain-recv", 1);
        setOption(args, "fused-a2a-drain-send", 1);
        setOption(args, "use-user-args", false);
        setOption(args, "bounds-check", BoundsCheckMode::Disable);
        setOption(args, "init-a", InitMode::Random);
        setOption(args, "init-b", InitMode::Random);
        setOption(args, "init-c", InitMode::Zero);
        setOption(args, "init-d", InitMode::Zero);
        setOption(args, "init-alpha", InitMode::One);
        setOption(args, "init-beta", InitMode::Zero);
        setOption(args, "num-elements-to-validate", elementsToValidate);
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

    class FusedReferenceValidatorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int n = 0;
            if(hipGetDeviceCount(&n) != hipSuccess || n < kWorld)
                GTEST_SKIP() << "needs " << kWorld << " devices";
        }

        void TearDown() override
        {
            for(size_t d = 0; d < recv.size(); d++)
            {
                ScopedDevice device(devices[d]->deviceId);
                static_cast<void>(hipFree(recv[d]));
            }
            for(size_t i = 0; i < otherD.size(); i++)
            {
                ScopedDevice device(devices[i % kWorld]->deviceId);
                static_cast<void>(hipFree(otherD[i]));
            }
        }

        // Slot 0 of every card holds the correct D and recv after this returns.
        void start()
        {
            args    = makeFusedArgs(-1);
            factory = std::make_unique<ClientProblemFactory>(args);
            problem = dynamic_cast<ContractionProblemGemm*>(factory->problems()[0].get());
            problem->setFusedA2AExtent(kM);
            problem->setFusedA2AWorld(kWorld);
            for(int d = 0; d < kWorld; d++)
            {
                devices.push_back(std::make_shared<DeviceContext>(d, true));
                ScopedDevice device(d);
                devices[d]->dataInit = std::make_shared<DataInitialization>(args, *factory);
            }

            validator = std::make_shared<ReferenceValidator>(args, devices);
            validator->setReporter(reporter);
            validator->preProblem(problem);
            solution.problemType.fusedGemmA2A = true;
            validator->preSolution(&solution);
            for(int d = 0; d < kWorld; d++)
            {
                ScopedDevice device(d);
                devices[d]->inputArr = {devices[d]->dataInit->prepareGPUInputs(problem)};
                golden.push_back(goldenD(d));
                writeD(d, golden[d]);
            }

            const size_t nShard = kM / kWorld;
            for(int dst = 0; dst < kWorld; dst++)
            {
                std::vector<BFloat16> slots(kWorld * kN * nShard);
                for(int src = 0; src < kWorld; src++)
                    for(size_t t = 0; t < kN; t++)
                        for(size_t f = 0; f < nShard; f++)
                            slots[(src * kN + t) * nShard + f]
                                = golden[src][dIndex(dst * nShard + f, t)];
                ScopedDevice device(dst);
                recv.push_back(nullptr);
                HIP_CHECK_EXC(hipMalloc(&recv[dst], slots.size() * sizeof(BFloat16)));
                HIP_CHECK_EXC(hipMemcpy(recv[dst],
                                        slots.data(),
                                        slots.size() * sizeof(BFloat16),
                                        hipMemcpyHostToDevice));
            }
            for(int d = 0; d < kWorld; d++)
                for(int j = 0; j < kWorld; j++)
                {
                    FusedA2APeerFields peer{};
                    peer[FUSED_A2A_SLOT_RECV_PTR] = recv[j];
                    inputs(d, 0).fusedA2APeers.push_back(peer);
                }
        }

        ContractionInputs& inputs(int d, size_t slot)
        {
            return dynamic_cast<ContractionInputs&>(*devices[d]->inputArr[slot]);
        }

        size_t dIndex(size_t m, size_t n) const
        {
            auto const& d = problem->d();
            return m * d.strides()[problem->freeIndices()[0].d]
                   + n * d.strides()[problem->freeIndices()[1].d];
        }

        std::vector<BFloat16> goldenD(int d)
        {
            auto const&           gpu = inputs(d, 0);
            std::vector<BFloat16> a(problem->a().totalAllocatedElements());
            std::vector<BFloat16> b(problem->b().totalAllocatedElements());
            HIP_CHECK_EXC(
                hipMemcpy(a.data(), gpu.a, a.size() * sizeof(BFloat16), hipMemcpyDeviceToHost));
            HIP_CHECK_EXC(
                hipMemcpy(b.data(), gpu.b, b.size() * sizeof(BFloat16), hipMemcpyDeviceToHost));
            const size_t aFree  = problem->a().strides()[problem->freeIndicesA()[0].i];
            const size_t aBound = problem->a().strides()[problem->boundIndices()[0].a];
            const size_t bFree  = problem->b().strides()[problem->freeIndicesB()[0].i];
            const size_t bBound = problem->b().strides()[problem->boundIndices()[0].b];

            std::vector<BFloat16> out(problem->d().totalAllocatedElements(), BFloat16(0.0f));
            for(size_t m = 0; m < kM; m++)
                for(size_t n = 0; n < kN; n++)
                {
                    float acc = 0.0f;
                    for(size_t k = 0; k < kK; k++)
                        acc += (float)a[m * aFree + k * aBound] * (float)b[n * bFree + k * bBound];
                    out[dIndex(m, n)] = BFloat16(acc);
                }
            return out;
        }

        void writeD(int d, std::vector<BFloat16> const& values)
        {
            ScopedDevice device(d);
            HIP_CHECK_EXC(hipMemcpy(inputs(d, 0).d,
                                    values.data(),
                                    values.size() * sizeof(BFloat16),
                                    hipMemcpyHostToDevice));
        }

        // Appends a slot whose D holds values[d] on card d and shares slot 0's recv.
        void addSlot(std::vector<std::vector<BFloat16>> const& values)
        {
            for(int d = 0; d < kWorld; d++)
            {
                ScopedDevice device(d);
                otherD.push_back(nullptr);
                HIP_CHECK_EXC(hipMalloc(&otherD.back(), values[d].size() * sizeof(BFloat16)));
                HIP_CHECK_EXC(hipMemcpy(otherD.back(),
                                        values[d].data(),
                                        values[d].size() * sizeof(BFloat16),
                                        hipMemcpyHostToDevice));
                auto slot = std::make_shared<ContractionInputs>(inputs(d, 0));
                slot->d   = otherD.back();
                devices[d]->inputArr.push_back(slot);
            }
        }

        void corruptRecv(int dst, size_t element)
        {
            BFloat16     bad(100.0f);
            ScopedDevice device(dst);
            HIP_CHECK_EXC(hipMemcpy(static_cast<BFloat16*>(recv[dst]) + element,
                                    &bad,
                                    sizeof(bad),
                                    hipMemcpyHostToDevice));
        }

        std::string validation()
        {
            return reporter->strings.at(ResultKey::Validation);
        }

        po::variables_map                           args;
        std::unique_ptr<ClientProblemFactory>       factory;
        ContractionProblemGemm*                     problem = nullptr;
        ContractionSolution                         solution;
        std::vector<std::shared_ptr<DeviceContext>> devices;
        std::shared_ptr<ReferenceValidator>         validator;
        std::shared_ptr<CapturingReporter> reporter = std::make_shared<CapturingReporter>();
        TimingEvents                                events{{}, 0, 0};
        std::vector<std::vector<BFloat16>> golden;
        std::vector<void*>                 recv;
        std::vector<void*>                 otherD;
    };

    std::string worldSizeError(po::variables_map const& args)
    {
        try
        {
            FusedA2ARunner::worldSize(args);
        }
        catch(std::runtime_error const& e)
        {
            return e.what();
        }
        return "";
    }
}

TEST_F(FusedReferenceValidatorTest, PassesWhenEveryCardMatches)
{
    start();
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "PASSED");
    EXPECT_EQ(validator->error(), 0);
}

TEST_F(FusedReferenceValidatorTest, FailsOnARecvMismatch)
{
    start();
    corruptRecv(1, 3);
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "FAILED");
    EXPECT_EQ(validator->error(), 1);
}

TEST_F(FusedReferenceValidatorTest, FailsOnADMismatchOnOneCard)
{
    start();
    auto bad               = golden[1];
    bad[dIndex(kM - 1, 0)] = BFloat16(100.0f);
    writeD(1, bad);
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "FAILED");
    EXPECT_EQ(validator->error(), 1);
}

TEST_F(FusedReferenceValidatorTest, PassesWhenEverySlotMatches)
{
    start();
    addSlot(golden);
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    validator->validateEnqueues({0, 1}, events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "PASSED");
}

TEST_F(FusedReferenceValidatorTest, ValidatesEverySlotAfterASync)
{
    start();
    auto bad                  = golden;
    bad[1][dIndex(kM - 1, 0)] = BFloat16(100.0f);
    addSlot(bad);
    validator->validateWarmups({0}, events, events);
    validator->postWarmup(events, events);
    validator->validateEnqueues({0, 1}, events, events);
    validator->postSolution();
    EXPECT_EQ(validation(), "FAILED");
    EXPECT_EQ(validator->error(), 1);
}

TEST(FusedA2ARunnerConfigTest, RejectsSettingsThatWouldHideAMismatch)
{
    auto args = makeFusedArgs(-1);
    setOption(args, "print-max", 0);
    EXPECT_THROW(FusedA2ARunner::worldSize(args), std::runtime_error);

    args = makeFusedArgs(7);
    EXPECT_THROW(FusedA2ARunner::worldSize(args), std::runtime_error);
}

TEST(FusedA2ARunnerConfigTest, RejectsICacheFlush)
{
    auto args = makeFusedArgs(-1);
    setOption(args, "icache-flush-args", std::vector<bool>{false, true});
    EXPECT_THROW(FusedA2ARunner::worldSize(args), std::runtime_error);
}

TEST(FusedA2ARunnerConfigTest, AcceptsAValidatingSetup)
{
#ifndef TENSILELITE_ENABLE_SDMA
    GTEST_SKIP() << "needs TENSILELITE_ENABLE_SDMA";
#endif
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n < 1)
        GTEST_SKIP() << "needs a device";
    auto args = makeFusedArgs(-1);
    setOption(args, "fused-a2a-world", 1);
    EXPECT_EQ(FusedA2ARunner::worldSize(args), 1);

    setOption(args, "icache-rotate-copies", -1);
    setOption(args, "rotating-buffer-size", 512);
    EXPECT_EQ(FusedA2ARunner::worldSize(args), 1);
}

TEST(FusedA2ARunnerConfigTest, RejectsDrainOff)
{
    for(auto name : {"fused-a2a-drain-recv", "fused-a2a-drain-send"})
    {
        auto args = makeFusedArgs(-1);
        EXPECT_EQ(worldSizeError(args).find("drain"), std::string::npos) << name;
        setOption(args, name, 0);
        EXPECT_NE(worldSizeError(args).find(name), std::string::npos) << name;
    }
}

TEST(FusedA2ARunnerConfigTest, RejectsUserArgs)
{
    auto args = makeFusedArgs(-1);
    EXPECT_EQ(worldSizeError(args).find("use-user-args"), std::string::npos);
    setOption(args, "use-user-args", true);
    EXPECT_NE(worldSizeError(args).find("use-user-args"), std::string::npos);
}
