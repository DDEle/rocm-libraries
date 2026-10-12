// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm
        fusedProblem(size_t M, size_t N, size_t ldd, uint32_t world, int64_t extent)
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
                                                      1,
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
