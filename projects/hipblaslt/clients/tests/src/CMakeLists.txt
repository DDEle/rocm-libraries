# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caching_library_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/secure_env_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/arch_revision_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_stats_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk5_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/row_uniformity_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ulp_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_epilogue_chain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_a2a_comm_multirank_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_a2a_peer_fields_gtest.cpp
)

# Per-source, not target-wide (see the NOTE below): handle.h resolves "rocblaslt.h"
# out of rocblaslt/include.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_a2a_comm_multirank_gtest.cpp
    TARGET_DIRECTORY hipblaslt-test
    PROPERTIES
        COMPILE_OPTIONS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../library/src/amd_detail/rocblaslt/include"
)

# NOTE: secure_env_gtest.cpp (ROCM-26729) intentionally includes the header-only
# rocblaslt_secure_env.hpp by relative path rather than via a target include
# directory. Adding the internal rocblaslt/src/include dir to hipblaslt-test
# shadows the clients' own "utility.hpp" with the library's internal one and
# breaks the build of unrelated test sources, so it must not be added here.
