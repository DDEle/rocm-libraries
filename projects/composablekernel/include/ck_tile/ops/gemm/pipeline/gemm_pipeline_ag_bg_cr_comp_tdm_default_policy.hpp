// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompTDM
struct GemmPipelineAgBgCrCompTDMDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy>
{
    // currently implement basic situation: the tile is divided into same parts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t warpNum   = BlockSize / get_warp_size();

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ALayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::AsLayoutTuple>>>;

        // Tile : MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            static_assert(MPerBlock % warpNum == 0, "MPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, MPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : KPerBlock * MPerBlock
        else
        {
            static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<MPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t warpNum   = BlockSize / get_warp_size();

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using BLayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::BsLayoutTuple>>>;

        // Tile : KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<NPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : NPerBlock * KPerBlock
        else
        {
            static_assert(NPerBlock % warpNum == 0, "NPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, NPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        return make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                            make_tuple(number<KPerBlock>{}, number<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        return make_naive_tensor_descriptor(make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                            make_tuple(number<KPerBlock>{}, number<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
