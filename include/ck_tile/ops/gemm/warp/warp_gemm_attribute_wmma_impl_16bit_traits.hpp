// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
namespace ck_tile {
// fp16 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, fp16_t, fp16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, fp16_t, fp16_t, float, 16>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// bf16 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, bf16_t, bf16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, bf16_t, bf16_t, float, 16>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// fp16 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, fp16_t, fp16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp16_t, fp16_t, float, 16>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// bf16 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, bf16_t, bf16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf16_t, bf16_t, float, 16>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// fp16 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, fp16_t, fp16_t, float, 16, 16, 32>
    : WmmaTraitsBase<gfx12_t, fp16_t, fp16_t, float, 32>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        return __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a_vec, 0, b_vec, 0, c_vec, false, false);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// bf16 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, bf16_t, bf16_t, float, 16, 16, 32>
    : WmmaTraitsBase<gfx12_t, bf16_t, bf16_t, float, 32>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        return __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a_vec, 0, b_vec, 0, c_vec, false, false);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};
} // namespace ck_tile
