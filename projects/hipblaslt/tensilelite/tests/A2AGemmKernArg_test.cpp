// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Tensile/ContractionSolution.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;

namespace
{
    constexpr size_t M = 3072, N = 2048, K = 4096;

    ContractionProblemGemm problemWithB(uint32_t world, size_t ldb, size_t strideB)
    {
        auto p = ContractionProblemGemm::GEMM_Strides(true,
                                                      false,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      M,
                                                      N,
                                                      K,
                                                      1,
                                                      K,
                                                      K * M,
                                                      ldb,
                                                      strideB,
                                                      M,
                                                      M * N,
                                                      M,
                                                      M * N,
                                                      0.0);
        p.setFusedA2A(FusedA2A::A2AGemm);
        p.setWorld(world);
        return p;
    }

    void configure(ContractionSolution& s)
    {
        s.kernelName                      = "a2a_gemm_symbol";
        s.problemType.fusedA2A            = FusedA2A::A2AGemm;
        s.sizeMapping.macroTile           = TensileLite::dim3(256, 256, 1);
        s.sizeMapping.depthU              = 64;
        s.sizeMapping.workGroupSize       = TensileLite::dim3(256, 1, 1);
        s.sizeMapping.matrixInstruction   = {16, 16, 32, 1};
        s.sizeMapping.globalSplitU        = 1;
        s.sizeMapping.globalAccumulation  = 0;
        s.sizeMapping.workGroupMapping    = 1;
        s.sizeMapping.workGroupMappingXCC = 1;
        s.sizeMapping.CUOccupancy         = 1;
    }

    std::vector<uint32_t> packedStridesB(ContractionProblemGemm const& problem)
    {
        ContractionSolution solution;
        configure(solution);
        auto device = TensileLite::testing::makeDevice(
            TensileLite::testing::_MI350_CHIP_ID, TensileLite::testing::_CPX_CU, "mi350cpx");
        ContractionInputs inputs;
        inputs.alpha = 1.0f;
        inputs.beta  = 0.0f;
        auto launch  = solution.resolvePersistentSettings(problem, device);
        auto call
            = solution.generateSingleCall<true>(problem, inputs, device, launch, GSUSettings{});
        std::vector<uint32_t> strides;
        for(auto name : {"strideB1", "strideB2"})
        {
            auto arg = KernelArguments::const_iterator(call.args, name);
            if(arg == call.args.end())
                throw std::runtime_error(std::string("Missing packed argument ") + name);
            uint32_t value = 0;
            std::memcpy(&value, (*arg).first, sizeof(value));
            strides.push_back(value);
        }
        return strides;
    }
}

TEST(A2AGemmKernArg, DenseBDescriptorPacksTheShardStrides)
{
    for(uint32_t w : {1u, 2u, 4u, 8u})
    {
        auto strides = packedStridesB(problemWithB(w, K, N * K));
        EXPECT_EQ(strides, (std::vector<uint32_t>{uint32_t(K / w), uint32_t(N * K / w)}))
            << "world " << w;
    }
}

TEST(A2AGemmKernArg, DeviceUserArgsRejectA2AGemm)
{
    ContractionInputs inputs;
    inputs.alpha          = 1.0f;
    inputs.beta           = 0.0f;
    inputs.activationArgs = {0.0f, 0.0f};
    ContractionGroupedInputs grouped;
    grouped.grouped = {inputs};
    DeviceUserArguments<float> args{};
    auto userArgs = [&](ContractionProblemGemm p) {
        p.setAlphaType(rocisa::DataType::Float);
        p.setBetaType(rocisa::DataType::Float);
        setDeviceUserArgs<float>({p}, grouped, &args);
    };

    auto stock = problemWithB(1, K, N * K);
    stock.setFusedA2A(FusedA2A::Disabled);
    EXPECT_NO_THROW(userArgs(stock));
    try
    {
        userArgs(problemWithB(2, K, N * K));
        ADD_FAILURE() << "setDeviceUserArgs accepted an A2AGemm problem";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_STREQ(e.what(), "A2AGemm does not support device user arguments");
    }
}
