// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {

#ifdef __gfx1250__
template <typename T>
CK_TILE_DEVICE __attribute__((address_space(1))) T* to_global(const T* ptr)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-qual"
    return (__attribute__((address_space(1))) T*)(ptr);
#pragma clang diagnostic pop
}
#endif // __gfx1250__

// Struct specializations for CLUSTER_LOAD_B32/B64/B128.
// Primary template intentionally undefined — compile error for unsupported sizes.
template <index_t bytes>
struct cluster_load;

template <>
struct cluster_load<4>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 4, "cluster_load<4> requires a 4-byte type");
#ifdef __gfx1250__
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b32(
            to_global<int>(reinterpret_cast<const int*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0,
                      "cluster_load is only supported on cluster-load capable targets");
        return T{};
#endif
    }
};

template <>
struct cluster_load<8>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 8, "cluster_load<8> requires an 8-byte type");
#ifdef __gfx1250__
        // Builtin requires LLVM native vector, not HIP int2.
        using vec2i_t = __attribute__((vector_size(8))) int;
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b64(
            to_global<vec2i_t>(reinterpret_cast<const vec2i_t*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0,
                      "cluster_load is only supported on cluster-load capable targets");
        return T{};
#endif
    }
};

template <>
struct cluster_load<16>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 16, "cluster_load<16> requires a 16-byte type");
#ifdef __gfx1250__
        // Builtin requires LLVM native vector, not HIP int4.
        using vec4i_t = __attribute__((vector_size(16))) int;
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b128(
            to_global<vec4i_t>(reinterpret_cast<const vec4i_t*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0,
                      "cluster_load is only supported on cluster-load capable targets");
        return T{};
#endif
    }
};

template <typename T>
CK_TILE_DEVICE T cluster_multicast_load(const T* addr, int mask)
{
    return cluster_load<sizeof(T)>{}(addr, mask);
}
} // namespace ck_tile
