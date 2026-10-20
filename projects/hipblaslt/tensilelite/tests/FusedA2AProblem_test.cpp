// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "FallbackTestUtils.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/FusedA2AKernArg.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm fusedProblem(
        size_t M, size_t N, size_t ldd, uint32_t world, int64_t extent, size_t batch = 1)
    {
        constexpr size_t K = 2048;
        auto             p = ContractionProblemGemm::GEMM_Strides(true,
                                                      false,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      M,
                                                      N,
                                                      K,
                                                      batch,
                                                      K,
                                                      K * M,
                                                      K,
                                                      K * N,
                                                      ldd,
                                                      ldd * N,
                                                      ldd,
                                                      ldd * N,
                                                      0.0);
        p.setFusedA2A(FusedA2A::GemmA2A);
        p.setWorld(world);
        p.setFusedA2AExtent(extent);
        return p;
    }
}

TEST(FusedA2AProblem, ShapePredicate)
{
    Predicates::Contraction::FusedA2AShape mt256(256);
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 4, 2048)));
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 8, 2048)));
    // AM % (MT0 * W) != 0
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 8, 3072)));
    // AM > M
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 4096)));
    // M % MT0 != 0
    EXPECT_FALSE(mt256(fusedProblem(3000, 512, 3000, 2, 2048)));
    // ldd % 8 != 0
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3076, 2, 2048)));
    // (ldd >> 3) == 2^19
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, size_t(1) << 22, 2, 2048)));
    // (nShard >> 3) == 16352 and 16384
    EXPECT_TRUE(mt256(fusedProblem(261632, 16, 261632, 2, 261632)));
    EXPECT_FALSE(mt256(fusedProblem(262144, 16, 262144, 2, 262144)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 0)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, -2048)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 2048, 2)));

    auto plain
        = ContractionProblemGemm::GEMM(false, true, 3000, 500, 100, 3000, 500, 3000, 0.0, false, 1);
    EXPECT_TRUE(mt256(plain));
}

namespace
{
    template <typename T>
    void setOption(Client::po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    Client::po::variables_map factoryArgs(bool fused, size_t problemCount)
    {
        Client::po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Alik_Bljk_Cijk_Dijk"));
        setOption(args,
                  "problem-size",
                  std::vector<std::vector<size_t>>(problemCount, {3072, 512, 1, 2048}));
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::BFloat16);
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "fused-a2a", std::string(fused ? "GemmA2A" : "Disabled"));
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensorOps{});
        setOption(args, "b-ops", TensorOps{});
        setOption(args, "c-ops", TensorOps{});
        setOption(args, "d-ops", TensorOps{});
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        return args;
    }

    ContractionProblemGemm const& gemmAt(Client::ClientProblemFactory const& factory, size_t i)
    {
        return *std::dynamic_pointer_cast<ContractionProblemGemm>(factory.problems().at(i));
    }
}

TEST(FusedA2AClientProblemFactory, PerProblemWorldAndExtent)
{
    auto args = factoryArgs(true, 2);
    setOption(args, "world", std::vector<int>{2, 4});
    setOption(args, "fused-a2a-extent", std::vector<int>{2048, 2048});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 2u);
    uint32_t const worlds[] = {2, 4};
    for(size_t i = 0; i < 2; i++)
    {
        auto const& p = gemmAt(factory, i);
        EXPECT_EQ(p.fusedA2A(), FusedA2A::GemmA2A);
        EXPECT_EQ(p.world(), worlds[i]);
        EXPECT_EQ(p.fusedA2AExtent(), 2048);
        size_t const nShard = 2048 / worlds[i];
        auto const&  recv   = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
        EXPECT_EQ(recv.sizes(), (std::vector<size_t>{nShard, 512, worlds[i]}));
        EXPECT_EQ(recv.strides(), (std::vector<size_t>{1, nShard, nShard * 512}));
    }
}

TEST(FusedA2AClientProblemFactory, InvalidWorldOrExtentThrows)
{
    struct Case
    {
        bool             fused;
        std::vector<int> worlds;
        std::vector<int> extents;
    };
    for(auto const& c : std::vector<Case>{{true, {2, 4}, {2048}},
                                          {true, {0}, {2048}},
                                          {true, {FUSED_A2A_MAX_RANKS + 1}, {2048}},
                                          {true, {2}, {0}},
                                          {false, {2}, {}}})
    {
        auto args = factoryArgs(c.fused, c.worlds.size());
        setOption(args, "world", c.worlds);
        setOption(args, "fused-a2a-extent", c.extents);
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument)
            << c.fused << " " << c.worlds.at(0) << " " << c.extents.size();
    }
}

TEST(FusedA2AClientProblemFactory, NonFusedKeepsDefaultWorld)
{
    Client::ClientProblemFactory factory(factoryArgs(false, 1));
    ASSERT_EQ(factory.problems().size(), 1u);
    EXPECT_EQ(gemmAt(factory, 0).world(), 1u);
}

TEST(FusedA2AProblem, ModeNamesRoundTrip)
{
    for(int i = 0; i < static_cast<int>(FusedA2A::Count); i++)
    {
        auto mode = static_cast<FusedA2A>(i);
        EXPECT_EQ(fusedA2AFromString(ToString(mode)), mode);
    }
    EXPECT_EQ(ToString(FusedA2A::A2AGemm), "A2AGemm");
    EXPECT_THROW(fusedA2AFromString("true"), std::invalid_argument);
}

TEST(FusedA2AProblem, ModePredicateIsolatesTheThreeKinds)
{
    const FusedA2A modes[] = {FusedA2A::Disabled, FusedA2A::GemmA2A, FusedA2A::A2AGemm};
    for(auto problemMode : modes)
        for(auto solutionMode : modes)
        {
            auto p = fusedProblem(3072, 512, 3072, 4, 2048);
            p.setFusedA2A(problemMode);
            Predicates::Contraction::FusedA2AEqual pred(solutionMode);
            EXPECT_EQ(pred(p), problemMode == solutionMode)
                << ToString(problemMode) << " vs " << ToString(solutionMode);
        }
}

namespace
{
    ContractionProblemGemm
        a2aGemmProblem(size_t K, uint32_t world, size_t ldb = 0, size_t batch = 1)
    {
        constexpr size_t M = 3072, N = 2048;
        if(ldb == 0)
            ldb = K;
        auto p = ContractionProblemGemm::GEMM_Strides(true,
                                                      false,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      M,
                                                      N,
                                                      K,
                                                      batch,
                                                      K,
                                                      K * M,
                                                      ldb,
                                                      ldb * N,
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

TEST(FusedA2AProblem, A2AGemmShapeAcceptsTheCommShapes)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    for(uint32_t w : {1u, 2u, 4u, 8u})
        EXPECT_TRUE(du64(a2aGemmProblem(4096, w))) << w;
}

TEST(FusedA2AProblem, A2AGemmShapeRejectsAShardThatIsNotWholeDepthU)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    EXPECT_FALSE(du64(a2aGemmProblem(4352, 8)));
}

TEST(FusedA2AProblem, A2AGemmShapeSubRoundsJoinTheDivisor)
{
    EXPECT_TRUE(Predicates::Contraction::A2AGemmShape(64)(a2aGemmProblem(4352, 4)));
    EXPECT_FALSE(Predicates::Contraction::A2AGemmShape(128)(a2aGemmProblem(4352, 4)));
}

TEST(FusedA2AProblem, A2AGemmShapeRejectsPaddedB)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 4, 4096 + 64)));
}

TEST(FusedA2AProblem, A2AGemmShapeRejectsBatch)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 4, 0, 2)));
}

TEST(FusedA2AProblem, A2AGemmShapeRejectsWorldZero)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 0)));
}

TEST(FusedA2AProblem, A2AGemmShapeRejectsGsuCoalesced)
{
    auto p = a2aGemmProblem(4096, 4);
    p.setParams().setGSUC(true);
    EXPECT_FALSE(Predicates::Contraction::A2AGemmShape(64)(p));
}

TEST(FusedA2AProblem, A2AGemmDenseBDescriptorPacksTheShardStrides)
{
    for(uint32_t w : {1u, 2u, 4u, 8u})
    {
        auto strides = packedStridesB(a2aGemmProblem(4096, w));
        EXPECT_EQ(strides, (std::vector<uint32_t>{uint32_t(4096 / w), uint32_t(2048 * 4096 / w)}))
            << "world " << w;
    }
}

TEST(FusedA2AProblem, DeviceUserArgsRejectA2AGemm)
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

    auto stock = a2aGemmProblem(4096, 1);
    stock.setFusedA2A(FusedA2A::Disabled);
    EXPECT_NO_THROW(userArgs(stock));
    try
    {
        userArgs(a2aGemmProblem(4096, 2));
        ADD_FAILURE() << "setDeviceUserArgs accepted an A2AGemm problem";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_STREQ(e.what(), "A2AGemm does not support device user arguments");
    }
}

TEST(FusedA2AProblem, A2ABufHoldsTheRecvSlotsForGemmA2A)
{
    auto p = fusedProblem(3072, 512, 3072, 4, 2048);
    p.setA2ABuf();
    auto const& buf = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
    EXPECT_EQ(buf.sizes(), (std::vector<size_t>{512, 512, 4}));
    EXPECT_EQ(buf.strides(), (std::vector<size_t>{1, 512, 512 * 512}));
    EXPECT_EQ(buf.dataType(), p.d().dataType());
}

TEST(FusedA2AProblem, A2ABufHoldsXForA2AGemm)
{
    auto p = a2aGemmProblem(4096, 4);
    p.setA2ABuf();
    auto const& buf = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
    EXPECT_EQ(buf.sizes(), (std::vector<size_t>{1024, 2048, 4}));
    EXPECT_EQ(buf.strides(), (std::vector<size_t>{1, 1024, 1024 * 2048}));
    EXPECT_EQ(buf.dataType(), p.b().dataType());
    EXPECT_EQ(buf.totalAllocatedElements(), p.b().totalAllocatedElements());
}

TEST(FusedA2AClientProblemFactory, A2AGemmProblemsCarryWorldAndBuf)
{
    auto args = factoryArgs(true, 2);
    setOption(args, "fused-a2a", std::string("A2AGemm"));
    setOption(args, "world", std::vector<int>{2, 4});
    setOption(args, "fused-a2a-extent", std::vector<int>{0, 0});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 2u);
    uint32_t const worlds[] = {2, 4};
    for(size_t i = 0; i < 2; i++)
    {
        auto const& p = gemmAt(factory, i);
        EXPECT_EQ(p.fusedA2A(), FusedA2A::A2AGemm);
        EXPECT_EQ(p.world(), worlds[i]);
        auto const& buf = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
        EXPECT_EQ(buf.sizes(), (std::vector<size_t>{2048 / worlds[i], 512, worlds[i]}));
        EXPECT_EQ(buf.dataType(), p.b().dataType());
    }
}

TEST(FusedA2AClientProblemFactory, A2AGemmRejectsExtentAndIndivisibleK)
{
    for(auto const& c : std::vector<std::pair<int, int>>{{2, 2048}, {3, 0}})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", std::string("A2AGemm"));
        setOption(args, "world", std::vector<int>{c.first});
        setOption(args, "fused-a2a-extent", std::vector<int>{c.second});
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument)
            << c.first << " " << c.second;
    }
}
