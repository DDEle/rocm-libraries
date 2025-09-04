// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/tdm.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {

// Test fixture class
class TDMBasicTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Setup code if needed
    }

    void TearDown() override
    {
        // Cleanup code if needed
    }

    template <typename DataType>
    bool run_tdm_test(index_t m,
                      index_t n,
                      index_t x_stride,
                      index_t y_stride,
                      int do_validation,
                      int warmup,
                      int repeat)
    {
        using ComputeDataType = remove_cvref_t<DataType>;

        HostTensor<ComputeDataType> x_host({m, n}, {x_stride, 1});
        HostTensor<ComputeDataType> y_host({m, n}, {y_stride, 1});
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

        using TDMTraits  = TDMPipelineTraits<ComputeDataType>;
        using TDMProblem = TDMPipelineProblem<TDMShape, TDMTraits>;

        dim3 grid((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);
        const index_t block_size =
            warp_m * warp_n * (is_wave32() ? get_warp_size() / 2 : get_warp_size());
        dim3 block(block_size);

        TDMCopyKernel<TDMProblem> tdm_kernel;

        stream_config s{nullptr, false, 0, warmup, repeat};

        launch_kernel(s,
                      make_kernel(tdm_kernel,
                                  grid,
                                  block,
                                  0,
                                  make_tuple(m, n),
                                  make_tuple(x_stride, 1),
                                  make_tuple(y_stride, 1),
                                  x_buf.GetDeviceBuffer(),
                                  y_buf.GetDeviceBuffer()));
        y_buf.FromDevice(y_host.data());

        if(do_validation)
        {
            bool passed = check_err(y_host, x_host, "Error: Incorrect tdm copy results!");
            return passed;
        }

        return true;
    }
};

TEST_F(TDMBasicTest, FP16SanityTest)
{
    index_t m        = 16;
    index_t n        = 16;
    index_t x_stride = -1;
    index_t y_stride = -1;
    if(x_stride < 0)
        x_stride = n;
    if(y_stride < 0)
        y_stride = n;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    EXPECT_TRUE(run_tdm_test<fp16_t>(m, n, x_stride, y_stride, do_validation, warmup, repeat));
}

TEST_F(TDMBasicTest, FP8SanityTest)
{
    index_t m        = 16;
    index_t n        = 16;
    index_t x_stride = -1;
    index_t y_stride = -1;
    if(x_stride < 0)
        x_stride = n;
    if(y_stride < 0)
        y_stride = n;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    EXPECT_TRUE(run_tdm_test<fp8_t>(m, n, x_stride, y_stride, do_validation, warmup, repeat));
}

TEST_F(TDMBasicTest, FP16LargeDimTest)
{
    index_t m        = 256;
    index_t n        = 256;
    index_t x_stride = -1;
    index_t y_stride = -1;
    if(x_stride < 0)
        x_stride = n;
    if(y_stride < 0)
        y_stride = n;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    EXPECT_TRUE(run_tdm_test<fp16_t>(m, n, x_stride, y_stride, do_validation, warmup, repeat));
}

} // namespace test
} // namespace ck_tile

// Main function for running tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
