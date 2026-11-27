// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "DeviceContext.hpp"
#include "FallbackTestUtils.hpp"
#include "ReferenceValidator.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/KernelArguments.hpp>
#include <Tensile/SolutionLibrary.hpp>

#ifdef TENSILE_YAML
#include <Tensile/Serialization.hpp>
#include <Tensile/llvm/YAML.hpp>
#endif

#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm fusedProblem(size_t           M,
                                        size_t           N,
                                        size_t           ldd,
                                        uint32_t         world,
                                        int64_t          extent,
                                        size_t           batch = 1,
                                        rocisa::DataType dType = rocisa::DataType::BFloat16)
    {
        constexpr size_t K = 2048;
        auto             p = ContractionProblemGemm::GEMM_Strides(true,
                                                      false,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      dType,
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
    Predicates::Contraction::GemmA2AShape mt256({256, 19, 14});
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 4, 2048)));
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 8, 2048)));
    // D is not 2 bytes
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 4, 2048, 1, rocisa::DataType::Float)));
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

TEST(FusedA2AProblem, ShapePredicateGfx12FieldWidths)
{
    Predicates::Contraction::GemmA2AShape gfx9({256, 19, 14});
    Predicates::Contraction::GemmA2AShape gfx12({256, 16, 16});
    // (ldd >> 3) == 2^16 - 1 and 2^16
    EXPECT_TRUE(gfx12(fusedProblem(3072, 512, 524280, 2, 2048)));
    EXPECT_FALSE(gfx12(fusedProblem(3072, 512, size_t(1) << 19, 2, 2048)));
    EXPECT_TRUE(gfx9(fusedProblem(3072, 512, size_t(1) << 19, 2, 2048)));
    // (nShard >> 3) == 16384
    EXPECT_TRUE(gfx12(fusedProblem(262144, 16, 262144, 2, 262144)));
}

TEST(FusedA2AProblem, SegmentNeedsCuCount)
{
    KernelArguments args(false);
    EXPECT_THROW(appendFusedSegment(args, FusedA2APeerTable{}, nullptr, 0, 4, 0, 2048, 0),
                 std::runtime_error);
}

TEST(FusedA2AProblem, GemmTypeOnlyMatchesTheFusedMode)
{
    auto fused = fusedProblem(3072, 512, 3072, 2, 2048);
    auto plain = fused;
    plain.setFusedA2A(FusedA2A::Disabled);

    ContractionSolution solution;
    solution.problemType.transA      = fused.transA();
    solution.problemType.transB      = fused.transB();
    solution.problemType.aType       = fused.a().dataType();
    solution.problemType.bType       = fused.b().dataType();
    solution.problemType.cType       = fused.c().dataType();
    solution.problemType.dType       = fused.d().dataType();
    solution.problemType.computeType = fused.computeType();

    solution.problemType.fusedA2A = FusedA2A::GemmA2A;
    EXPECT_TRUE(isGemmTypeSame(solution, fused));
    EXPECT_FALSE(isGemmTypeSame(solution, plain));

    solution.problemType.fusedA2A = FusedA2A::Disabled;
    EXPECT_FALSE(isGemmTypeSame(solution, fused));
    EXPECT_TRUE(isGemmTypeSame(solution, plain));
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
        setOption(args, "fused-a2a", fused ? FusedA2A::GemmA2A : FusedA2A::Disabled);
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
                                          {true, {2}, {3073}},
                                          {true, {2}, {4096}},
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

namespace
{
    ContractionProblemGemm
        a2aGemmProblem(size_t K, uint32_t world, size_t ldb = 0, size_t batch = 1, size_t N = 2048)
    {
        constexpr size_t M = 3072;
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
        p.setFusedA2AExtent(K);
        return p;
    }
}

TEST(FusedA2AProblem, A2AGemmShapePredicate)
{
    Predicates::Contraction::A2AGemmShape du64(64);
    EXPECT_TRUE(du64(a2aGemmProblem(4096, 4)));
    // K / W % DU != 0
    EXPECT_FALSE(du64(a2aGemmProblem(4352, 8)));
    // K / W % (DU * SubRounds) != 0
    EXPECT_TRUE(du64(a2aGemmProblem(4352, 4)));
    EXPECT_FALSE(Predicates::Contraction::A2AGemmShape(128)(a2aGemmProblem(4352, 4)));
    // (K / W) * bpeB / 16 > 2^14
    EXPECT_TRUE(du64(a2aGemmProblem(524288, 4)));
    EXPECT_FALSE(du64(a2aGemmProblem(524544, 4)));
    // N * (K / W) * bpeB % 128 != 0
    Predicates::Contraction::A2AGemmShape du8(8);
    EXPECT_TRUE(du8(a2aGemmProblem(16, 2, 0, 1, 8)));
    EXPECT_FALSE(du8(a2aGemmProblem(16, 2, 0, 1, 1)));
    // ldb != K
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 4, 4096 + 64)));
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 4, 0, 2)));
    EXPECT_FALSE(du64(a2aGemmProblem(4096, 0)));
    EXPECT_FALSE(du64(a2aGemmProblem(0, 4)));

    auto gsuc = a2aGemmProblem(4096, 4);
    gsuc.setParams().setGSUC(true);
    EXPECT_FALSE(du64(gsuc));

    auto pointerArray = a2aGemmProblem(4096, 4);
    pointerArray.setBatchMode(ContractionProblemGemm::BATCHMODE::POINTER_ARRAY);
    EXPECT_FALSE(du64(pointerArray));

    auto partialK = a2aGemmProblem(4096, 4);
    partialK.setFusedA2AExtent(2048);
    EXPECT_FALSE(du64(partialK));
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
    auto                       p = a2aGemmProblem(4096, 2);
    p.setAlphaType(rocisa::DataType::Float);
    p.setBetaType(rocisa::DataType::Float);
    EXPECT_THROW(setDeviceUserArgs<float>({p}, grouped, &args), std::runtime_error);
}

// factoryArgs uses problem-size {3072, 512, 1, 2048}, so K is 2048.
TEST(FusedA2AClientProblemFactory, A2AGemmRejectsPartialExtentAndIndivisibleK)
{
    for(auto const& c : std::vector<std::pair<int, int>>{{2, 1024}, {2, 0}, {3, 2048}})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", FusedA2A::A2AGemm);
        setOption(args, "world", std::vector<int>{c.first});
        setOption(args, "fused-a2a-extent", std::vector<int>{c.second});
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument)
            << c.first << " " << c.second;
    }
}

TEST(FusedA2AClientProblemFactory, A2AGemmExtentDefaultsToK)
{
    auto args = factoryArgs(true, 1);
    setOption(args, "fused-a2a", FusedA2A::A2AGemm);
    setOption(args, "world", std::vector<int>{2});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 1u);
    auto const&  p      = gemmAt(factory, 0);
    size_t const kShard = 2048 / 2;
    EXPECT_EQ(p.fusedA2AExtent(), 2048);
    auto const& gathered = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
    EXPECT_EQ(gathered.sizes(), (std::vector<size_t>{kShard, 512, 2}));
    EXPECT_EQ(gathered.strides(), (std::vector<size_t>{1, kShard, kShard * 512}));
}

TEST(FusedA2AClientProblemFactory, A2AGemmAcceptsKBeyondM)
{
    auto args = factoryArgs(true, 1);
    setOption(args, "problem-size", std::vector<std::vector<size_t>>{{1024, 512, 1, 2048}});
    setOption(args, "fused-a2a", FusedA2A::A2AGemm);
    setOption(args, "world", std::vector<int>{2});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 1u);
    EXPECT_EQ(gemmAt(factory, 0).fusedA2AExtent(), 2048);
}

// factoryArgs uses problem-size {3072, 512, 1, 2048}, so M is 3072 and N is 512.
TEST(FusedA2AClientProblemFactory, GemmRSRejectsExtentOtherThanMAndIndivisibleN)
{
    for(auto const& c : std::vector<std::pair<int, int>>{{2, 2048}, {2, 0}, {2, 4096}, {3, 3072}})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", FusedA2A::GemmRS);
        setOption(args, "world", std::vector<int>{c.first});
        setOption(args, "fused-a2a-extent", std::vector<int>{c.second});
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument)
            << c.first << " " << c.second;
    }
}

TEST(FusedA2AClientProblemFactory, GemmRSReceiveBufferHoldsOneTokenSegmentPerSource)
{
    for(int world : {1, 2, 4})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", FusedA2A::GemmRS);
        setOption(args, "world", std::vector<int>{world});
        setOption(args, "fused-a2a-extent", std::vector<int>{3072});
        Client::ClientProblemFactory factory(args);
        ASSERT_EQ(factory.problems().size(), 1u);
        auto const&  p    = gemmAt(factory, 0);
        size_t const segN = 512 / world;
        auto const&  recv = p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF];
        EXPECT_EQ(recv.dataType(), p.d().dataType());
        EXPECT_EQ(recv.sizes(), (std::vector<size_t>{3072, segN, size_t(world)})) << world;
        EXPECT_EQ(recv.strides(), (std::vector<size_t>{1, 3072, 3072 * segN})) << world;
    }
}

TEST(FusedA2AClientProblemFactory, GemmRSExtentDefaultsToM)
{
    auto args = factoryArgs(true, 1);
    setOption(args, "fused-a2a", FusedA2A::GemmRS);
    setOption(args, "world", std::vector<int>{2});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 1u);
    EXPECT_EQ(gemmAt(factory, 0).fusedA2AExtent(), 3072);
}

namespace
{
    using Predicates::Contraction::GemmRSShape;

    // GemmRSShape value: {MT0, MT1, Cs, Ck, pitch bits, rect_x bits}.
    GemmRSShape const rs256({256, 256, 2, 2, 16, 16});

    ContractionProblemGemm rsProblem(size_t           M,
                                     size_t           N,
                                     uint32_t         world,
                                     size_t           ldd   = 0,
                                     size_t           batch = 1,
                                     rocisa::DataType dType = rocisa::DataType::BFloat16)
    {
        auto p = fusedProblem(M, N, ldd ? ldd : M, world, M, batch, dType);
        p.setFusedA2A(FusedA2A::GemmRS);
        return p;
    }
}

TEST(FusedA2AProblem, GemmRSModeRoundTrips)
{
    EXPECT_EQ(static_cast<int>(FusedA2A::GemmRS), static_cast<int>(FusedA2A::A2AGemm) + 1);
    EXPECT_EQ(static_cast<int>(FusedA2A::Count), static_cast<int>(FusedA2A::GemmRS) + 1);
    EXPECT_EQ(ToString(FusedA2A::GemmRS), "GemmRS");
    for(int i = 0; i < static_cast<int>(FusedA2A::Count); i++)
    {
        auto const         mode = static_cast<FusedA2A>(i);
        FusedA2A           parsed{FusedA2A::Count};
        std::istringstream in(ToString(mode));
        in >> parsed;
        EXPECT_EQ(parsed, mode) << ToString(mode);
    }
    FusedA2A           parsed{};
    std::istringstream in("GemmRs");
    EXPECT_THROW(in >> parsed, std::invalid_argument);
}

TEST(GemmRSShape, OnlyConstrainsGemmRS)
{
    auto indivisible = rsProblem(8192, 3 * 1024, 4);
    EXPECT_FALSE(rs256(indivisible));
    indivisible.setFusedA2A(FusedA2A::GemmA2A);
    EXPECT_TRUE(rs256(indivisible));
    indivisible.setFusedA2A(FusedA2A::Disabled);
    EXPECT_TRUE(rs256(indivisible));
}

TEST(GemmRSShape, TP4OutputProjection)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 4)));
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 2)));
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 1)));
}

TEST(GemmRSShape, World)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 2048, 4)));
    EXPECT_FALSE(rs256(rsProblem(8192, 2048, 0)));
}

TEST(GemmRSShape, Batch)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 4, 0, 1)));
    EXPECT_FALSE(rs256(rsProblem(8192, 16384, 4, 0, 2)));
}

TEST(GemmRSShape, DIsTwoBytes)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 4, 0, 1, rocisa::DataType::Half)));
    EXPECT_FALSE(rs256(rsProblem(8192, 16384, 4, 0, 1, rocisa::DataType::Float)));
}

TEST(GemmRSShape, NDividesIntoWorldBands)
{
    // W * MT1 * Ck = 4 * 256 * 2 = 2048.
    EXPECT_TRUE(rs256(rsProblem(8192, 3 * 2048, 4)));
    EXPECT_FALSE(rs256(rsProblem(8192, 3 * 2048 + 1024, 4)));
    // W = 3: 3 * 256 * 2 = 1536.
    EXPECT_TRUE(rs256(rsProblem(8192, 2 * 1536, 3)));
    EXPECT_FALSE(rs256(rsProblem(8192, 2 * 1536, 4)));
    GemmRSShape const ck1({256, 256, 2, 1, 16, 16});
    EXPECT_TRUE(ck1(rsProblem(8192, 3 * 2048 + 1024, 4)));
}

TEST(GemmRSShape, MDividesIntoClusterTiles)
{
    // MT0 * Cs = 512.
    EXPECT_TRUE(rs256(rsProblem(8192 + 512, 16384, 4)));
    EXPECT_FALSE(rs256(rsProblem(8192 + 256, 16384, 4)));
    GemmRSShape const cs1({256, 256, 1, 2, 16, 16});
    EXPECT_TRUE(cs1(rsProblem(8192 + 256, 16384, 4)));
}

TEST(GemmRSShape, MStrideIsOne)
{
    auto p = rsProblem(8192, 16384, 4);
    EXPECT_TRUE(rs256(p));
    p.resetTensor(ContractionProblemGemm::TENSOR::D,
                  rocisa::DataType::BFloat16,
                  {8192, 16384, 1},
                  {2, 8192, 8192 * 16384});
    EXPECT_FALSE(rs256(p));
}

TEST(GemmRSShape, LddIsM)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 16384, 4, 8192)));
    EXPECT_FALSE(rs256(rsProblem(8192, 16384, 4, 8192 + 8)));
}

TEST(GemmRSShape, MIsAMultipleOf8)
{
    GemmRSShape const mt1({1, 256, 1, 2, 16, 16});
    EXPECT_TRUE(mt1(rsProblem(8200, 16384, 4)));
    EXPECT_FALSE(mt1(rsProblem(8196, 16384, 4)));
}

TEST(GemmRSShape, MFitsThePitchField)
{
    GemmRSShape const pitch10({8, 256, 1, 2, 10, 16});
    EXPECT_TRUE(pitch10(rsProblem(8184, 16384, 4)));
    EXPECT_FALSE(pitch10(rsProblem(8192, 16384, 4)));
}

TEST(GemmRSShape, MFitsTheRectXField)
{
    GemmRSShape const rectX10({8, 256, 1, 2, 16, 10});
    EXPECT_TRUE(rectX10(rsProblem(8184, 16384, 4)));
    EXPECT_FALSE(rectX10(rsProblem(8192, 16384, 4)));
}

TEST(GemmRSShape, Gfx1250FieldWidths)
{
    // (M >> 3) == 2^16 - 64 and 2^16.
    EXPECT_TRUE(rs256(rsProblem(524288 - 512, 2048, 4)));
    EXPECT_FALSE(rs256(rsProblem(524288, 2048, 4)));
}

TEST(GemmRSShape, AtLeastOneBandPerSegment)
{
    EXPECT_TRUE(rs256(rsProblem(8192, 2048, 4)));
    EXPECT_FALSE(rs256(rsProblem(8192, 0, 4)));
}

TEST(GemmRSShape, AtMost32BandsPerSegment)
{
    // N / (W * MT1 * Ck) == 32 and 33.
    EXPECT_TRUE(rs256(rsProblem(8192, 32 * 2048, 4)));
    EXPECT_FALSE(rs256(rsProblem(8192, 33 * 2048, 4)));
}

#ifdef TENSILE_YAML
TEST(GemmRSShape, YamlRoundTrip)
{
    using Pointer = std::shared_ptr<Predicates::Predicate<ContractionProblemGemm>>;
    auto read     = [](std::string const& text) {
        Pointer           p;
        llvm::yaml::Input in(text);
        in >> p;
        EXPECT_FALSE(in.error()) << text;
        return p;
    };
    auto const  first = read("{type: GemmRSShape, value: [256, 128, 2, 4, 16, 16]}");
    auto const* shape = dynamic_cast<GemmRSShape const*>(first.get());
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->value, (std::array<int64_t, 6>{256, 128, 2, 4, 16, 16}));

    std::string text;
    {
        llvm::raw_string_ostream stream(text);
        llvm::yaml::Output       out(stream);
        Pointer                  copy = first;
        out << copy;
    }
    auto const  second = read(text);
    auto const* again  = dynamic_cast<GemmRSShape const*>(second.get());
    ASSERT_NE(again, nullptr) << text;
    EXPECT_EQ(again->value, shape->value);
}

TEST(GemmRSShape, FusedRSWaitYamlRoundTrip)
{
    SizeMapping written;
    written.fusedRSWait = 2;
    std::string text;
    {
        llvm::raw_string_ostream stream(text);
        llvm::yaml::Output       out(stream);
        out << written;
    }
    EXPECT_NE(text.find("fusedRSWait"), std::string::npos) << text;
    SizeMapping       read;
    llvm::yaml::Input in(text);
    in >> read;
    ASSERT_FALSE(in.error()) << text;
    EXPECT_EQ(read.fusedRSWait, 2);
    EXPECT_EQ(SizeMapping{}.fusedRSWait, 1);
}
#endif

TEST(GemmRSShape, RuntimeWgmOverrideThrows)
{
    ContractionSolution solution;
    solution.internalArgsSupport.version  = 3;
    solution.internalArgsSupport.wgm      = false;
    solution.internalArgsSupport.staggerU = false;
    auto problem = ContractionProblemGemm::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);
    auto pack    = [&](int32_t autoWGM, size_t autoWGMXCC) {
        uint32_t arg0 = 0, arg1 = 0;
        solution.calculateInternalArgs<false>(
            arg0, arg1, nullptr, problem.getParams(), autoWGM, autoWGMXCC, 0, 0, 0, 0, 0, 1, {});
    };

    for(auto mode : {FusedA2A::GemmA2A, FusedA2A::GemmRS})
    {
        SCOPED_TRACE(ToString(mode));
        bool const rs                 = mode == FusedA2A::GemmRS;
        solution.problemType.fusedA2A = mode;
        auto expectThrowIfRS          = [&](int32_t autoWGM, size_t autoWGMXCC) {
            if(rs)
                EXPECT_THROW(pack(autoWGM, autoWGMXCC), std::runtime_error)
                    << autoWGM << " " << autoWGMXCC;
            else
                EXPECT_NO_THROW(pack(autoWGM, autoWGMXCC)) << autoWGM << " " << autoWGMXCC;
        };

        problem.setParams().setWgm(0);
        problem.setParams().setWGMXCC(0);
        EXPECT_NO_THROW(pack(1, 1));
        EXPECT_NO_THROW(pack(1, 0));
        for(int32_t wgm : {0, 6, -8})
            expectThrowIfRS(wgm, 1);
        expectThrowIfRS(1, 8);

        problem.setParams().setWgm(4);
        expectThrowIfRS(1, 1);
        problem.setParams().setWgm(0);
        problem.setParams().setWGMXCC(8);
        expectThrowIfRS(1, 1);
    }
}

TEST(GemmRSShape, KernargFusedAMIsM)
{
    ContractionSolution solution;
    solution.kernelName                                    = "gemm_rs";
    solution.sizeMapping.tileProcessingStrategy            = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.workAssignment                    = WorkAssignment::StaticGrid;
    solution.sizeMapping.macroTile                         = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU                            = 64;
    solution.sizeMapping.workGroupSize                     = TensileLite::dim3(256, 1, 1);
    solution.sizeMapping.waveNum                           = 4;
    solution.sizeMapping.matrixInstruction                 = {16, 16, 4, 1};
    solution.sizeMapping.globalSplitU                      = 1;
    solution.sizeMapping.globalAccumulation                = 0;
    solution.sizeMapping.workGroupMapping                  = 1;
    solution.sizeMapping.workGroupMappingXCC               = 1;
    solution.sizeMapping.CUOccupancy                       = 1;
    solution.internalArgsSupport.version                   = 3;
    solution.internalArgsSupport.persistentLoopArgsVersion = 1;
    solution.internalArgsSupport.useUniversalArgs          = true;

    auto device = TensileLite::testing::makeDevice(
        TensileLite::testing::_MI350_CHIP_ID, TensileLite::testing::_CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    device.persistentFixedGrid   = 7;
    ContractionInputs inputs;
    inputs.alpha = 1.0f;
    inputs.beta  = 0.0f;

    for(auto mode : {FusedA2A::GemmA2A, FusedA2A::GemmRS})
    {
        SCOPED_TRACE(ToString(mode));
        auto problem
            = ContractionProblemGemm::GEMM(false, false, 3072, 512, 256, 3072, 256, 3072, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setFusedA2A(mode);
        problem.setWorld(2);
        problem.setFusedA2AExtent(1024);
        solution.problemType.fusedA2A = mode;
        auto call = solution.generateSingleCall<true>(
            problem, inputs, device, solution.resolvePersistentSettings(problem, device), GSUSettings{});
        KernelArguments::const_iterator it(call.args, "FusedAM");
        ASSERT_TRUE(it != call.args.end());
        uint32_t am = 0;
        ASSERT_EQ((*it).second, sizeof(am));
        std::memcpy(&am, (*it).first, sizeof(am));
        EXPECT_EQ(am, mode == FusedA2A::GemmRS ? 3072u : 1024u);
    }
}

TEST(GemmRSReference, SumsEveryRankOverItsOwnTokensAndRoundsOnce)
{
    // M = 2, N = 3, W = 3: rank d owns token column d.
    constexpr size_t M = 2, N = 3, W = 3;
    auto const       problem = rsProblem(M, N, W);
    auto const       value   = [](size_t r, size_t t, size_t f) {
        return f == 0 ? (r == 0 ? 1.0f : 0x1p-8f) : float(1 + r + 4 * t);
    };
    std::vector<std::vector<BFloat16>> golden(W, std::vector<BFloat16>(M * N));
    std::vector<void const*>           goldenPtrs;
    for(size_t r = 0; r < W; r++)
    {
        for(size_t t = 0; t < N; t++)
            for(size_t f = 0; f < M; f++)
                golden[r][t * M + f] = BFloat16(value(r, t, f));
        goldenPtrs.push_back(golden[r].data());
    }

    std::vector<std::vector<uint8_t>> recvs, reduced;
    Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced);
    ASSERT_EQ(recvs.size(), W);
    ASSERT_EQ(reduced.size(), W);
    auto at = [](std::vector<uint8_t> const& bytes, size_t i) {
        BFloat16 v;
        std::memcpy(&v, bytes.data() + i * sizeof(BFloat16), sizeof(BFloat16));
        return v;
    };
    for(size_t d = 0; d < W; d++)
    {
        SCOPED_TRACE(d);
        ASSERT_EQ(recvs[d].size(), M * (N / W) * W * sizeof(BFloat16));
        ASSERT_EQ(reduced[d].size(), M * N * sizeof(BFloat16));
        for(size_t src = 0; src < W; src++)
            for(size_t f = 0; f < M; f++)
                EXPECT_EQ(at(recvs[d], src * M + f).data,
                          src == d ? 0 : golden[src][d * M + f].data)
                    << src << " " << f;
        for(size_t t = 0; t < N; t++)
            for(size_t f = 0; f < M; f++)
                if(t != d)
                    EXPECT_EQ(at(reduced[d], t * M + f).data, golden[d][t * M + f].data);
        EXPECT_EQ(float(at(reduced[d], d * M)), 1.0f + 0x1p-7f);
        EXPECT_EQ(float(at(reduced[d], d * M + 1)), float(6 + 12 * d));
        EXPECT_NE(at(reduced[d], d * M).data, golden[d][d * M].data);
        EXPECT_NE(at(reduced[d], d * M + 1).data, golden[d][d * M + 1].data);
    }
}

TEST(GemmRSReference, ReceiveSlotsAreSourceMajor)
{
    // M = 3, N = 6, W = 2: rank d owns tokens [3d, 3d + 3).
    constexpr size_t M = 3, N = 6, W = 2, S = N / W;
    auto const       problem = rsProblem(M, N, W);
    std::vector<std::vector<BFloat16>> golden(W, std::vector<BFloat16>(M * N));
    std::vector<void const*>           goldenPtrs;
    for(size_t r = 0; r < W; r++)
    {
        for(size_t t = 0; t < N; t++)
            for(size_t f = 0; f < M; f++)
                golden[r][t * M + f] = BFloat16(float(1 + 100 * r + 10 * t + f));
        goldenPtrs.push_back(golden[r].data());
    }

    std::vector<std::vector<uint8_t>> recvs, reduced;
    Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced);
    ASSERT_EQ(recvs.size(), W);
    for(size_t d = 0; d < W; d++)
    {
        ASSERT_EQ(recvs[d].size(), M * S * W * sizeof(BFloat16));
        auto const* recv = reinterpret_cast<BFloat16 const*>(recvs[d].data());
        for(size_t src = 0; src < W; src++)
            for(size_t t = 0; t < S; t++)
                for(size_t f = 0; f < M; f++)
                    EXPECT_EQ(recv[f + M * t + M * S * src].data,
                              src == d ? 0 : golden[src][(d * S + t) * M + f].data)
                        << d << " " << src << " " << t << " " << f;
    }
}

TEST(GemmRSReference, DIsComparedToTheReducedDOnlyWhenTheKernelWaits)
{
    for(auto mode : {FusedA2A::Disabled, FusedA2A::GemmA2A, FusedA2A::A2AGemm})
        for(int wait : {0, 1, 2})
            EXPECT_FALSE(Client::comparesReducedD(mode, wait)) << ToString(mode) << " " << wait;
    EXPECT_FALSE(Client::comparesReducedD(FusedA2A::GemmRS, 0));
    EXPECT_TRUE(Client::comparesReducedD(FusedA2A::GemmRS, 1));
    EXPECT_TRUE(Client::comparesReducedD(FusedA2A::GemmRS, 2));
}

TEST(GemmRSReference, ReceiveBufferIsValidatedForThePushModes)
{
    EXPECT_FALSE(Client::validatesRecvBuffer(FusedA2A::Disabled));
    EXPECT_FALSE(Client::validatesRecvBuffer(FusedA2A::A2AGemm));
    EXPECT_TRUE(Client::validatesRecvBuffer(FusedA2A::GemmA2A));
    EXPECT_TRUE(Client::validatesRecvBuffer(FusedA2A::GemmRS));
}

TEST(GemmRSCommBuffers, PeersPointAtTheirReceiveBuffer)
{
    int               buf = 0, b = 0;
    ContractionInputs peer;
    peer.a2aBuf = &buf;
    peer.b      = &b;
    for(auto mode : {FusedA2A::GemmA2A, FusedA2A::GemmRS, FusedA2A::A2AGemm})
        for(size_t j : {0, 1})
        {
            SCOPED_TRACE(concatenate(ToString(mode), " ", j));
            void const* expected = mode == FusedA2A::A2AGemm && j != 0
                                       ? static_cast<void const*>(&b)
                                       : static_cast<void const*>(&buf);
            EXPECT_EQ(Client::peerBuffer(mode, j, 0, peer), expected);
        }
}

TEST(GemmRSCommBuffers, ReceiveBufferIsClearedToZero)
{
    EXPECT_EQ(Client::a2aBufClearByte(FusedA2A::GemmRS), 0);
    EXPECT_EQ(Client::a2aBufClearByte(FusedA2A::GemmA2A), 0);
    EXPECT_EQ(Client::a2aBufClearByte(FusedA2A::A2AGemm), 0xC7);
}

TEST(GemmRSCommBuffers, PoisonCoversEveryOtherSourcesSlot)
{
    for(int world : {2, 4, 8})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", FusedA2A::GemmRS);
        setOption(args, "world", std::vector<int>{world});
        Client::ClientProblemFactory factory(args);
        ASSERT_EQ(factory.problems().size(), 1u);
        auto const&  p         = gemmAt(factory, 0);
        size_t const slotBytes = 3072 * (512 / world) * 2;
        ASSERT_EQ(p.tensors()[ContractionProblemGemm::TENSOR::A2A_BUF].totalAllocatedBytes(),
                  slotBytes * world);
        for(size_t rank = 0; rank < size_t(world); rank++)
        {
            std::vector<std::pair<size_t, size_t>> expected;
            for(size_t s = 0; s < size_t(world); s++)
                if(s != rank)
                    expected.emplace_back(s * slotBytes, slotBytes);
            EXPECT_EQ(Client::gemmRSPeerSlotBytes(p, rank), expected) << world << " " << rank;
        }
    }
}

TEST(GemmRSCommBuffers, PoisonAfterALaunchOnlyWhenItsSlotIsReusedBeforeTheValidation)
{
    // Warmup 0 alone, then warmup 1 and 3 enqueues before the next validation, 2 slots.
    EXPECT_FALSE(Client::gemmRSSlotReusedBefore(0, 2, 1));
    EXPECT_TRUE(Client::gemmRSSlotReusedBefore(1, 2, 5));
    EXPECT_TRUE(Client::gemmRSSlotReusedBefore(2, 2, 5));
    EXPECT_FALSE(Client::gemmRSSlotReusedBefore(3, 2, 5));
    EXPECT_FALSE(Client::gemmRSSlotReusedBefore(4, 2, 5));
    // 4 enqueues on 3 slots: only launch 0's slot comes back.
    EXPECT_TRUE(Client::gemmRSSlotReusedBefore(0, 3, 4));
    EXPECT_FALSE(Client::gemmRSSlotReusedBefore(1, 3, 4));
    // One slot: never.
    EXPECT_FALSE(Client::gemmRSSlotReusedBefore(0, 1, 4));
}

TEST(GemmRSCommBuffers, PoisonIsOffUnlessValidatingGemmRSAcrossRanks)
{
    std::vector<std::shared_ptr<Client::DeviceContext>> none;
    auto problemOf = [](FusedA2A mode, int world) {
        auto args = factoryArgs(true, 1);
        setOption(args, "fused-a2a", mode);
        setOption(args, "world", std::vector<int>{world});
        setOption(args, "fused-a2a-extent", std::vector<int>{3072});
        return Client::ClientProblemFactory(args).problems().at(0);
    };
    EXPECT_TRUE(Client::GemmRSRecvPoison(*problemOf(FusedA2A::GemmRS, 2), none, true).enabled());
    EXPECT_FALSE(Client::GemmRSRecvPoison(*problemOf(FusedA2A::GemmRS, 2), none, false).enabled());
    EXPECT_FALSE(Client::GemmRSRecvPoison(*problemOf(FusedA2A::GemmRS, 1), none, true).enabled());
    EXPECT_FALSE(Client::GemmRSRecvPoison(*problemOf(FusedA2A::GemmA2A, 2), none, true).enabled());

    Client::GemmRSRecvPoison off(*problemOf(FusedA2A::GemmRS, 2), none, false);
    off.fillIdle();
    off.validateBefore(8);
    off.afterLaunch(0, 0);
    EXPECT_EQ(off.idleFills(), 0u);
    EXPECT_EQ(off.launchFills(), 0u);
}

TEST(GemmRSReference, RejectsTokensNotDivisibleByWorld)
{
    auto const               problem = rsProblem(2, 4, 3);
    std::vector<BFloat16>    golden(2 * 4);
    std::vector<void const*> goldenPtrs(3, golden.data());
    std::vector<std::vector<uint8_t>> recvs, reduced;
    EXPECT_THROW(Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced),
                 std::runtime_error);
}

TEST(GemmRSReference, RejectsBatch)
{
    auto const               problem = rsProblem(1, 2, 2, 0, 2);
    std::vector<BFloat16>    golden(2 * 2 * 2);
    std::vector<void const*> goldenPtrs(2, golden.data());
    std::vector<std::vector<uint8_t>> recvs, reduced;
    EXPECT_THROW(Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced),
                 std::runtime_error);
}

namespace
{
    // M = 1, N = 2, W = 2: rank 0's segment is element 0, where rank 0's partial is own and rank
    // 1's is peer. Returns the reduced reference there afterwards.
    float acceptOwnRounding(float                      own,
                            float                      peer,
                            float                      device,
                            Client::GemmRSOwnRounding* stats = nullptr)
    {
        auto const               problem = rsProblem(1, 2, 2);
        std::vector<BFloat16>    g0(2, BFloat16(own)), g1(2, BFloat16(peer));
        std::vector<void const*> golden{g0.data(), g1.data()};
        std::vector<std::vector<uint8_t>> recvs, reduced;
        Client::gemmRSReferences(problem, golden, recvs, reduced);
        BFloat16 const dev(device);
        auto const     s = Client::gemmRSAcceptOwnRounding(problem, 0, golden, &dev, reduced[0]);
        if(stats != nullptr)
            *stats = s;
        auto const* r = reinterpret_cast<BFloat16 const*>(reduced[0].data());
        EXPECT_EQ(r[1].data, g0[1].data);
        return float(r[0]);
    }
}

TEST(GemmRSOwnRounding, AcceptsTheSumMovedByHalfAnUlpOfTheOwnPartial)
{
    // own 1000: half ulp 2.
    EXPECT_EQ(acceptOwnRounding(1000, -1000, 2), 2.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -1000, 2.015625f), 0.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -1000, -2), -2.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -1000, -2.015625f), 0.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -936, 66), 66.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -936, 66.5f), 64.0f);
    // own 1024: half ulp 4.
    EXPECT_EQ(acceptOwnRounding(1024, -1024, 4), 4.0f);
    EXPECT_EQ(acceptOwnRounding(1024, -1024, 4.03125f), 0.0f);
}

TEST(GemmRSOwnRounding, NegativeValues)
{
    EXPECT_EQ(acceptOwnRounding(-1000, 936, -66), -66.0f);
    EXPECT_EQ(acceptOwnRounding(-1000, 936, -66.5f), -64.0f);
    EXPECT_EQ(acceptOwnRounding(-1000, 936, -62), -62.0f);
    EXPECT_EQ(acceptOwnRounding(-1000, 936, -61.5f), -64.0f);
}

TEST(GemmRSOwnRounding, NoResultUlpBeyondTheOwnTerm)
{
    // own 100: half ulp 0.25; the sum 120 has spacing 0.5.
    EXPECT_EQ(acceptOwnRounding(100, 20, 120.5f), 120.0f);
    EXPECT_EQ(acceptOwnRounding(100, 20, 119.5f), 120.0f);
    EXPECT_EQ(acceptOwnRounding(0, 3, 3.015625f), 3.0f);
}

TEST(GemmRSOwnRounding, ZeroAndSubnormalOwnPartialsUseTheSubnormalSpacing)
{
    float const minSub = 0x1p-133f;
    // Half ulp 2^-134: the ends of the interval are BF16 ties.
    EXPECT_EQ(acceptOwnRounding(0, minSub, 2 * minSub), 2 * minSub);
    EXPECT_EQ(acceptOwnRounding(0, minSub, 3 * minSub), minSub);
    EXPECT_EQ(acceptOwnRounding(0, 0, minSub), 0.0f);
    EXPECT_EQ(acceptOwnRounding(0x1p-130f, minSub, 10 * minSub), 10 * minSub);
    EXPECT_EQ(acceptOwnRounding(0x1p-130f, minSub, 8 * minSub), 8 * minSub);
    EXPECT_EQ(acceptOwnRounding(0x1p-130f, minSub, 11 * minSub), 9 * minSub);
}

TEST(GemmRSOwnRounding, NonFiniteIsLeftToTheGenericComparison)
{
    float const inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(acceptOwnRounding(inf, 3, 3), inf);
    EXPECT_TRUE(std::isnan(acceptOwnRounding(inf, -inf, 3)));
    EXPECT_EQ(acceptOwnRounding(1000, -1000, inf), 0.0f);
    EXPECT_EQ(acceptOwnRounding(1000, -1000, std::nanf("")), 0.0f);
}

TEST(GemmRSOwnRounding, OwnSpanCoversTheRanksTokens)
{
    // M = 3, ldd = 4, N = 6, W = 3.
    auto const problem = rsProblem(3, 6, 3, 4);
    EXPECT_EQ(Client::gemmRSOwnSpan(problem, 0, 3), std::make_pair(size_t(0), size_t(7)));
    EXPECT_EQ(Client::gemmRSOwnSpan(problem, 1, 3), std::make_pair(size_t(8), size_t(15)));
    EXPECT_EQ(Client::gemmRSOwnSpan(problem, 2, 3), std::make_pair(size_t(16), size_t(23)));
}

namespace
{
    // M = 3, ldd = 4, N = 6, W = 3: rank r owns tokens [2r, 2r + 2); f = 3 is padding. Every
    // segment sums to 0 with an own half ulp of at least 2.
    constexpr size_t rsM = 3, rsLdd = 4, rsN = 6, rsW = 3, rsS = rsN / rsW;

    std::vector<std::vector<BFloat16>> rsGolden()
    {
        std::vector<std::vector<BFloat16>> golden;
        for(float v : {1000.0f, 1000.0f, -2000.0f})
            golden.emplace_back(rsLdd * rsN, BFloat16(v));
        return golden;
    }
}

TEST(GemmRSOwnRounding, TouchesOnlyTheRanksTokenSegment)
{
    auto const               problem = rsProblem(rsM, rsN, rsW, rsLdd);
    auto const               golden  = rsGolden();
    std::vector<void const*> goldenPtrs{golden[0].data(), golden[1].data(), golden[2].data()};
    std::vector<std::vector<uint8_t>> recvs, reduced;
    Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced);
    std::vector<BFloat16> device(rsLdd * rsN, BFloat16(1.0f));
    device[1 + 3 * rsLdd] = BFloat16(3.0f);
    for(size_t rank = 0; rank < rsW; rank++)
    {
        SCOPED_TRACE(rank);
        auto const first = Client::gemmRSOwnSpan(problem, rank, rsW).first;
        Client::gemmRSAcceptOwnRounding(
            problem, rank, goldenPtrs, device.data() + first, reduced[rank]);
        auto const* r = reinterpret_cast<BFloat16 const*>(reduced[rank].data());
        ASSERT_EQ(reduced[rank].size(), (rsLdd * rsN - 1) * sizeof(BFloat16));
        for(size_t t = 0; t < rsN; t++)
            for(size_t f = 0; f < rsLdd && f + t * rsLdd < rsLdd * rsN - 1; f++)
            {
                size_t const i      = f + t * rsLdd;
                bool const   inside = f < rsM && t / rsS == rank;
                float const  want   = !inside                ? float(golden[rank][i])
                                      : i == 1 + 3 * rsLdd ? 0.0f
                                                           : 1.0f;
                EXPECT_EQ(float(r[i]), want) << t << " " << f;
            }
    }
}

TEST(GemmRSOwnRounding, AcceptsTheUnroundedOwnPartialAfterCancellation)
{
    // M = 1, N = 2, W = 2. Rank 0's partial 999 rounds to golden 1000; rank 1's is -1000. The
    // device adds the unrounded 999 and stores -1, the reference bf16(1000 - 1000) = 0.
    ASSERT_EQ(float(BFloat16(999.0f)), 1000.0f);
    for(float device : {-1.0f, 1.0f, -2.0f, -3.0f})
    {
        SCOPED_TRACE(device);
        EXPECT_EQ(acceptOwnRounding(999, -1000, device), device == -3.0f ? 0.0f : device);
    }
}

TEST(GemmRSOwnRounding, CountsAndMaximaAreTakenBeforeTheAcceptance)
{
    // M = 4, N = 2, W = 2: rank 0's segment is elements 0..3.
    auto const            problem = rsProblem(4, 2, 2);
    std::vector<BFloat16> g0(8, BFloat16(5.0f)), g1(8, BFloat16(6.0f)), device(4);
    float const           own[] = {1000, 100, 1000, 7}, peer[] = {-1000, 20, -1000, 0};
    float const           dev[] = {-1, 120, 3, 7};
    for(size_t i = 0; i < 4; i++)
    {
        g0[i]     = BFloat16(own[i]);
        g1[i]     = BFloat16(peer[i]);
        device[i] = BFloat16(dev[i]);
    }
    std::vector<void const*>          golden{g0.data(), g1.data()};
    std::vector<std::vector<uint8_t>> recvs, reduced;
    Client::gemmRSReferences(problem, golden, recvs, reduced);
    auto const stats
        = Client::gemmRSAcceptOwnRounding(problem, 0, golden, device.data(), reduced[0]);
    EXPECT_EQ(stats.elements, 4u);
    EXPECT_EQ(stats.differing, 2u);
    EXPECT_EQ(stats.accepted, 1u);
    EXPECT_EQ(stats.maxDiff, 3.0);
    EXPECT_EQ(stats.boundAtMaxDiff, 2.0);
    EXPECT_EQ(stats.maxRatio, 1.5);
    auto const* r = reinterpret_cast<BFloat16 const*>(reduced[0].data());
    EXPECT_EQ(float(r[0]), -1.0f);
    EXPECT_EQ(float(r[1]), 120.0f);
    EXPECT_EQ(float(r[2]), 0.0f);
    EXPECT_EQ(float(r[3]), 7.0f);
    std::ostringstream line;
    line << std::fixed << std::setprecision(2) << stats << " " << 0.5;
    EXPECT_EQ(line.str(),
              "own-segment elements 4, differing from reference 2, accepted within rounding bound "
              "1, max |device - reference| 3 (bound there 2), max |device - reference| / bound "
              "1.5 0.50");
}

TEST(GemmRSOwnRounding, StatisticsOfEqualAndZeroBoundElements)
{
    Client::GemmRSOwnRounding stats;
    EXPECT_EQ(acceptOwnRounding(100, 20, 120, &stats), 120.0f);
    EXPECT_EQ(stats.elements, 1u);
    EXPECT_EQ(stats.differing, 0u);
    EXPECT_EQ(stats.maxDiff, 0.0);
    EXPECT_EQ(stats.maxRatio, 0.0);

    EXPECT_EQ(acceptOwnRounding(100, 20, 120.5f, &stats), 120.0f);
    EXPECT_EQ(stats.differing, 1u);
    EXPECT_EQ(stats.accepted, 0u);
    EXPECT_EQ(stats.maxDiff, 0.5);
    EXPECT_EQ(stats.boundAtMaxDiff, 0.0);
    EXPECT_EQ(stats.maxRatio, std::numeric_limits<double>::infinity());

    std::ostringstream line;
    line << stats;
    EXPECT_EQ(line.str(),
              "own-segment elements 1, differing from reference 1, accepted within rounding bound "
              "0, max |device - reference| 0.5 (bound there 0), max |device - reference| / bound "
              "inf");

    acceptOwnRounding(0, 0x1p-130f, 0x1p-130f + 0x1p-133f, &stats);
    line.str("");
    line << stats;
    EXPECT_NE(line.str().find("max |device - reference| 9.1835496157991212e-41 "),
              std::string::npos)
        << line.str();
}

TEST(GemmRSOwnRounding, ScopeReadsTheOwnSpanOfDAndRestoresTheReference)
{
    auto const               problem = rsProblem(rsM, rsN, rsW, rsLdd);
    auto const               golden  = rsGolden();
    std::vector<void const*> goldenPtrs{golden[0].data(), golden[1].data(), golden[2].data()};
    std::vector<std::vector<uint8_t>> recvs, reduced;
    Client::gemmRSReferences(problem, goldenPtrs, recvs, reduced);
    auto const            before = reduced[1];
    std::vector<BFloat16> d(rsLdd * rsN, BFloat16(1.0f));
    ContractionInputs     result;
    result.d   = d.data();
    result.gpu = false;
    {
        Client::GemmRSOwnRoundingScope scope(problem, 1, goldenPtrs, result, reduced[1]);
        EXPECT_EQ(scope.stats().elements, rsM * rsS);
        EXPECT_EQ(scope.stats().accepted, rsM * rsS);
        auto const* r = reinterpret_cast<BFloat16 const*>(reduced[1].data());
        auto const* b = reinterpret_cast<BFloat16 const*>(before.data());
        for(size_t i = 0; i + 1 < rsLdd * rsN; i++)
        {
            bool const inside = i % rsLdd < rsM && i / rsLdd / rsS == 1;
            EXPECT_EQ(float(r[i]), inside ? 1.0f : float(b[i])) << i;
        }
    }
    EXPECT_EQ(reduced[1], before);

    EXPECT_THROW(
        {
            Client::GemmRSOwnRoundingScope scope(problem, 1, goldenPtrs, result, reduced[1]);
            throw std::runtime_error("validate");
        },
        std::runtime_error);
    EXPECT_EQ(reduced[1], before);
}

TEST(GemmRSRotatingSlots, BackToBackLaunchesNeedTwoSlots)
{
    auto const rs  = rsProblem(8192, 16384, 4);
    auto       a2a = rs;
    a2a.setFusedA2A(FusedA2A::GemmA2A);
    auto const rs1 = rsProblem(8192, 16384, 1);
    struct Case
    {
        size_t warmups, profilerRuns, syncs, enqueues;
        bool   backToBack;
    };
    for(auto const& c : std::vector<Case>{{1, 0, 0, 3, false},
                                          {0, 0, 3, 1, false},
                                          {1, 0, 0, 0, false},
                                          {0, 0, 0, 0, false},
                                          {0, 1, 0, 0, false},
                                          {2, 0, 0, 1, true},
                                          {0, 0, 1, 2, true},
                                          {1, 0, 1, 1, true},
                                          {3, 0, 2, 4, true},
                                          {1, 1, 0, 0, true},
                                          {0, 1, 1, 1, true}})
    {
        SCOPED_TRACE(
            concatenate(c.warmups, " ", c.profilerRuns, " ", c.syncs, " ", c.enqueues));
        size_t const plain = std::max(c.warmups, c.syncs * c.enqueues);
        auto slots = [&](ContractionProblem const& p) {
            return Client::fusedRotatingSlots(p, c.warmups, c.profilerRuns, c.syncs, c.enqueues);
        };
        auto check = [&](ContractionProblem const& p, size_t n) {
            Client::checkFusedRotatingSlots(p, n, c.warmups, c.profilerRuns, c.syncs, c.enqueues);
        };
        EXPECT_EQ(slots(a2a), plain);
        EXPECT_EQ(slots(rs1), plain);
        EXPECT_EQ(slots(rs), c.backToBack ? std::max<size_t>(plain, 2) : plain);
        EXPECT_NO_THROW(check(a2a, 1));
        EXPECT_NO_THROW(check(rs1, 1));
        EXPECT_NO_THROW(check(rs, 2));
        if(c.backToBack)
            EXPECT_THROW(check(rs, 1), std::runtime_error);
        else
            EXPECT_NO_THROW(check(rs, 1));
    }
}

TEST(GemmRSCounter, UsesTheGemmA2ACounterLayout)
{
    ContractionSolution solution;
    solution.sizeMapping.macroTile = TensileLite::dim3(256, 256, 1);
    AMDGPU     noCuCount;
    auto       problem = rsProblem(8192, 16384, 4);
    auto const tiles   = 16384 / 256;
    // The GemmA2A layout: counter1 at +256, W * tokenTiles u32.
    for(auto [bytes, fits] : std::vector<std::pair<size_t, bool>>{
            {256 + 4 * tiles * 4, true}, {256 + 4 * tiles * 4 - 4, false}})
    {
        problem.setSynchronizer(rocisa::DataType::Int32, bytes / 4);
        for(auto mode : {FusedA2A::GemmA2A, FusedA2A::GemmRS})
        {
            SCOPED_TRACE(concatenate(ToString(mode), " ", bytes));
            problem.setFusedA2A(mode);
            solution.problemType.fusedA2A = mode;
            if(fits)
                EXPECT_NO_THROW(Client::checkFusedA2ASolution(problem, solution, noCuCount, 4));
            else
                EXPECT_THROW(Client::checkFusedA2ASolution(problem, solution, noCuCount, 4),
                             std::runtime_error);
        }
    }
}
