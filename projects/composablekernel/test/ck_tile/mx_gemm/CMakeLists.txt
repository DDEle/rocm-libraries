# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# MX GEMM tests using TDM pipeline with microscaling support
# Currently built only for gfx125 (gfx1250) which supports WMMA MX instructions

set(EXAMPLE_MX_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_MX_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx12")
    if(GPU_TARGETS MATCHES "gfx12")
        list(APPEND EXAMPLE_MX_GEMM_COMPILE_OPTIONS -DARCH_GFX12)
    endif()

    if(GPU_TARGETS MATCHES "gfx125")
        add_gtest_executable(test_ck_tile_mx_gemm_pipeline_tdm_wmma test_mx_gemm_pipeline_comp_tdm_wmma.cpp)
        target_compile_options(test_ck_tile_mx_gemm_pipeline_tdm_wmma PRIVATE ${EXAMPLE_MX_GEMM_COMPILE_OPTIONS})
    endif()
else()
    message(DEBUG "Skipping ck_tile mx_gemm tests for current target")
endif()
