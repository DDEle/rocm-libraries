// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/tdm.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {

using F16 = half_t;
using F8  = fp8_t;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

struct TDMTestParams
{
    index_t m         = 16;
    index_t n         = 16;
    index_t x_stride  = -1;
    index_t y_stride  = -1;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    template <typename Layout>
    void normalize()
    {
        if constexpr(std::is_same_v<Layout, tensor_layout::gemm::RowMajor>)
        {
            if(x_stride < 0)
                x_stride = n;
            if(y_stride < 0)
                y_stride = n;
        }
        else
        {
            if(x_stride < 0)
                x_stride = m;
            if(y_stride < 0)
                y_stride = m;
        }
    }
};

using TestTypes = ::testing::
    Types<std::tuple<F16, Row>, std::tuple<F16, Col>, std::tuple<F8, Row>, std::tuple<F8, Col>>;

template <typename TypeParam>
class TDMBasicTypedTest : public ::testing::Test
{
    protected:
    using DataType = std::tuple_element_t<0, TypeParam>;
    using Layout   = std::tuple_element_t<1, TypeParam>;

    bool run_tdm_test(const TDMTestParams& params)
    {
        using ComputeDataType = remove_cvref_t<DataType>;
        using ComputeLayout   = Layout;

        const std::vector<index_t> dims =
            std::is_same_v<ComputeLayout, tensor_layout::gemm::ColumnMajor>
                ? std::vector<index_t>{params.n, params.m}
                : std::vector<index_t>{params.m, params.n};

        HostTensor<ComputeDataType> x_host({dims[0], dims[1]}, {params.x_stride, 1});
        HostTensor<ComputeDataType> y_host({dims[0], dims[1]}, {params.y_stride, 1});
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

TYPED_TEST_SUITE(TDMBasicTypedTest, TestTypes);

TYPED_TEST(TDMBasicTypedTest, SanityTest)
{
    TDMTestParams params;
    params.m = 16;
    params.n = 16;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

TYPED_TEST(TDMBasicTypedTest, RectangleTest)
{
    TDMTestParams params;
    params.m = 64;
    params.n = 32;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

TYPED_TEST(TDMBasicTypedTest, LargeDimTest)
{
    TDMTestParams params;
    params.m = 256;
    params.n = 256;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

} // namespace test
} // namespace ck_tile

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
