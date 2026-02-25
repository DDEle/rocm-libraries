/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_REDUCTION_FUNCTIONS_HPP
#define GUARD_REDUCTION_FUNCTIONS_HPP

#include "configuration.hpp"
#include "static_unroll.hpp"

// NOTE: This header should be independent from batchnorm_functions.hpp
// Even is in OpenCL implementation, these functions are only enabled under
// certain condition. But now, these template will not be compiled before
// calling them.
// #include "batchnorm_functions.hpp"
namespace miopen {
namespace reduction {

namespace detail {
template <int N>
struct log2_floor
{
    constexpr static int value = log2_floor<(N >> 1)>::value + 1;
};
template <>
struct log2_floor<1>
{
    constexpr static int value = 0;
};
template <int N>
constexpr static int log2_floor_v = log2_floor<N>::value;

template <int N>
struct log2_ceil
{
    constexpr static int value = log2_floor_v<N> + ((1 << log2_floor_v<N>) == N ? 0 : 1);
};
template <int N>
constexpr static int log2_ceil_v = log2_ceil<N>::value;

} // namespace detail

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    lcl_data_x[lid] = x;
    lcl_data_y[lid] = y;
    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        __syncthreads();
    }

    x = lcl_data_x[0] * scale;
    y = lcl_data_y[0] * scale;
}

// Caller must ensure: SizeLclData >= (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize
// @warning Undefined behavior if SizeLclData is too small
// Caller must ensure: All lanes must be active
// @warning Undefined behavior if lanes are masked
template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    const unsigned int ldsidx = lid / warpSize;
    constexpr unsigned long long mask = 0xFFFFFFFFFFFFFFFFull;
    x = __reduce_add_sync(mask, x);
    y = __reduce_add_sync(mask, y);
    // Last thread
    if((lid % warpSize) == warpSize-1)
    {
        lcl_data_x[ldsidx] = x;
        lcl_data_y[ldsidx] = y;
    }

    __syncthreads();

    x = y = 0;

    // This could be changed to clang loop unroll(full), because the size is small
    static_unroll_count<unsigned int, 0, SizeLclData, 1, 2>{[&](unsigned int i) {
        x += lcl_data_x[i];
        y += lcl_data_y[i];
    }};

    x *= scale;
    y *= scale;
}

} // namespace reduction
} // namespace miopen

#endif
