# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_a2a_multiprocess
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_multiprocess.cpp")
target_link_libraries(sample_a2a_multiprocess PRIVATE sample-helper)

# Per-source: handle.h resolves "rocblaslt.h" out of rocblaslt/include and
# <Tensile/FusedA2AKernArg.hpp> out of tensilelite/include.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_multiprocess.cpp
    PROPERTIES
        COMPILE_OPTIONS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../library/src/amd_detail/rocblaslt/include;-I${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/include"
)
