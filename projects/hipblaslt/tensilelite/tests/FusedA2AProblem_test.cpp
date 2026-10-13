// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/FusedA2AKernArg.hpp>

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
        p.setFusedGemmA2A(true);
        p.setFusedA2AWorld(world);
        p.setFusedA2AExtent(extent);
        return p;
    }
}

TEST(FusedA2AProblem, PropertiesReadWorldAndExtent)
{
    auto p = fusedProblem(3072, 512, 3072, 4, 2048);
    EXPECT_EQ(Contraction::FusedA2AWorld()(p), 4u);
    EXPECT_EQ(Contraction::FusedA2AExtent()(p), 2048u);
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

TEST(FusedA2AProblem, RecvIsWorldByTokensByShard)
{
    auto p = fusedProblem(3072, 511, 3072, 4, 2048);
    EXPECT_EQ(p.tensors()[ContractionProblemGemm::TENSOR::A2A_RECV].totalAllocatedElements(), 0u);
    p.setA2ARecv();
    auto const& recv = p.tensors()[ContractionProblemGemm::TENSOR::A2A_RECV];
    EXPECT_EQ(recv.getName(), "a2aRecv");
    EXPECT_EQ(recv.dataType(), rocisa::DataType::BFloat16);
    EXPECT_EQ(recv.totalAllocatedElements(), size_t(4) * 511 * 512);
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
        setOption(args, "fused-gemm-a2a", fused);
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
        EXPECT_TRUE(p.fusedGemmA2A());
        EXPECT_EQ(p.fusedA2AWorld(), worlds[i]);
        EXPECT_EQ(p.fusedA2AExtent(), 2048);
        EXPECT_EQ(p.tensors()[ContractionProblemGemm::TENSOR::A2A_RECV].totalAllocatedElements(),
                  size_t(worlds[i]) * 512 * (2048 / worlds[i]));
    }
}

TEST(FusedA2AClientProblemFactory, CountMismatchThrows)
{
    auto args = factoryArgs(true, 2);
    setOption(args, "world", std::vector<int>{2, 4});
    setOption(args, "fused-a2a-extent", std::vector<int>{2048});
    EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument);
}

TEST(FusedA2AClientProblemFactory, WorldOutOfRangeThrows)
{
    for(int world : {0, FUSED_A2A_MAX_RANKS + 1})
    {
        auto args = factoryArgs(true, 1);
        setOption(args, "world", std::vector<int>{world});
        setOption(args, "fused-a2a-extent", std::vector<int>{2048});
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument) << world;
    }
}

TEST(FusedA2AClientProblemFactory, WorldWithoutFusedThrows)
{
    auto args = factoryArgs(false, 1);
    setOption(args, "world", std::vector<int>{2});
    EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument);
}

TEST(FusedA2AClientProblemFactory, NonFusedKeepsDefaultWorld)
{
    Client::ClientProblemFactory factory(factoryArgs(false, 1));
    ASSERT_EQ(factory.problems().size(), 1u);
    auto const& p = gemmAt(factory, 0);
    EXPECT_FALSE(p.fusedGemmA2A());
    EXPECT_EQ(p.fusedA2AWorld(), ContractionProblemGemm().fusedA2AWorld());
    EXPECT_EQ(p.tensors()[ContractionProblemGemm::TENSOR::A2A_RECV].totalAllocatedElements(), 0u);
}
