// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/tdm.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {

struct TDMTestParams
{
    index_t m         = 16;
    index_t n         = 16;
    index_t x_stride  = -1;
    index_t y_stride  = -1;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    void normalize()
    {
        if(x_stride < 0)
            x_stride = n;
        if(y_stride < 0)
            y_stride = n;
    }
};

// Test fixture class
class TDMBasicTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}

    template <typename DataType, typename Layout = tensor_layout::gemm::RowMajor>
    bool run_tdm_test(const TDMTestParams& params)
    {
        using ComputeDataType = remove_cvref_t<DataType>;
        using ComputeLayout   = Layout;

        HostTensor<ComputeDataType> x_host({params.m, params.n}, {params.x_stride, 1});
        HostTensor<ComputeDataType> y_host({params.m, params.n}, {params.y_stride, 1});
        FillUniformDistribution<ComputeDataType>{-.5f, .5f}(x_host);

        DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
        DeviceMem y_buf(y_host.get_element_space_size_in_bytes());

        x_buf.ToDevice(x_host.data());
        y_buf.SetZero();

        constexpr index_t tensor_rank = 2;
        constexpr index_t tile_m      = 16;
        constexpr index_t tile_n      = 16;
        constexpr index_t warp_m      = 1;
        constexpr index_t warp_n      = 1;
        constexpr index_t warp_tile_m = 16;
        constexpr index_t warp_tile_n = 16;

        using TDMShape = TDMTileShape<tensor_rank,
                                      sequence<tile_m, tile_n>,
                                      sequence<warp_m, warp_n>,
                                      sequence<warp_tile_m, warp_tile_n>>;

        using TDMTraits  = TDMPipelineTraits<ComputeDataType, ComputeLayout>;
        using TDMProblem = TDMPipelineProblem<TDMShape, TDMTraits>;

        dim3 grid((params.m + tile_m - 1) / tile_m, (params.n + tile_n - 1) / tile_n);
        assert(is_wave32());
        const index_t block_size = warp_m * warp_n * 32; // 32 is warp size
        dim3 block(block_size);

        TDMCopyKernel<TDMProblem> tdm_kernel;

        stream_config s{nullptr, false, 0, params.warmup, params.repeat};

        TDMCopyDeviceKernArgs args{x_buf.GetDeviceBuffer(),
                                   y_buf.GetDeviceBuffer(),
                                   params.m,
                                   params.n,
                                   params.x_stride,
                                   params.y_stride};

        launch_kernel(s, make_kernel(tdm_kernel, grid, block, 0, args));
        y_buf.FromDevice(y_host.data());

        if(params.do_validation)
        {
            bool passed = check_err(y_host, x_host, "Error: Incorrect tdm copy results!");
            return passed;
        }

        return true;
    }
};

TEST_F(TDMBasicTest, FP16SanityTest)
{
    TDMTestParams params;
    params.m             = 16;
    params.n             = 16;
    params.x_stride      = -1;
    params.y_stride      = -1;
    params.do_validation = 1;
    params.warmup        = 0;
    params.repeat        = 1;
    params.normalize();

    EXPECT_TRUE(run_tdm_test<fp16_t>(params));
}

TEST_F(TDMBasicTest, FP8SanityTest)
{
    TDMTestParams params;
    params.m             = 16;
    params.n             = 16;
    params.x_stride      = -1;
    params.y_stride      = -1;
    params.do_validation = 1;
    params.warmup        = 0;
    params.repeat        = 1;
    params.normalize();

    EXPECT_TRUE(run_tdm_test<fp8_t>(params));
}

TEST_F(TDMBasicTest, FP16LargeDimTest)
{
    TDMTestParams params;
    params.m             = 256;
    params.n             = 256;
    params.x_stride      = -1;
    params.y_stride      = -1;
    params.do_validation = 1;
    params.warmup        = 0;
    params.repeat        = 1;
    params.normalize();

    EXPECT_TRUE(run_tdm_test<fp16_t>(params));
}

} // namespace test
} // namespace ck_tile

// Main function for running tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
