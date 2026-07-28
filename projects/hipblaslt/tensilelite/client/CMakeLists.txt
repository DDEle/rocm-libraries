# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared core for tensilelite-client and cpu-gemm-driver.
#
# Both executables consume nearly the same sources, link line, include
# directories, compile definitions, C++ standard, and sanitizer flags. Owning
# them on a single STATIC helper target removes a `foreach(target ...)` loop
# that applied identical properties to two targets, a `COMMON_CLIENT_SOURCES`
# variable that hid the source list from `target_sources` (and from IDE
# navigation), and three duplicated `target_include_directories` calls.
#
# Each executable is now a thin wrapper that adds only its `main` translation
# unit and links to this helper.
add_library(tensilelite-client-common STATIC)
add_library(tensilelite::client-common ALIAS tensilelite-client-common)

if(NOT WIN32)
    find_package(amd_smi REQUIRED)
else()
    find_package(amd_smi)
endif()

# hsakmt (KFD thunk) + hsa-runtime64 are needed by SdmaQueue.cpp for the
# fused-A2A SDMA queue. ROCm 7.x's hsakmt cmake config declares a link
# dependency on the imported target numa::numa but never provides it, so
# find_package(hsakmt) fails at generate time ("target numa::numa not found")
# unless we define it first from the system libnuma. Mirror MORI's workaround
# (mori/src/application/CMakeLists.txt).
if(NOT WIN32)
    if(NOT TARGET numa::numa)
        find_library(NUMA_LIBRARY NAMES numa)
        find_path(NUMA_INCLUDE_DIR NAMES numa.h)
        if(NUMA_LIBRARY)
            add_library(numa::numa UNKNOWN IMPORTED)
            set_target_properties(numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
            if(NUMA_INCLUDE_DIR)
                set_target_properties(numa::numa PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
            endif()
        else()
            message(FATAL_ERROR "libnuma not found but required by hsakmt::hsakmt. "
                                "Install it (e.g. 'apt install libnuma-dev').")
        endif()
    endif()
    find_package(hsa-runtime64 REQUIRED)
    find_package(hsakmt REQUIRED)
endif()

target_link_libraries(tensilelite-client-common
    PUBLIC
        hip::device
        hip::host
        roc::tensilelite-host
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
)
if(amd_smi_FOUND)
    target_link_libraries(tensilelite-client-common PUBLIC amd_smi)
endif()
if(NOT WIN32)
    target_link_libraries(tensilelite-client-common
        PUBLIC hsakmt::hsakmt hsa-runtime64::hsa-runtime64)
endif()

# When mxDataGenerator is enabled, hipblaslt::mxdatagen propagates the
# HIPBLASLT_ENABLE_MXDATAGENERATOR macro, the C++20 floor, the include path
# for `<mxDataGen.hpp>`, hipblaslt::headers (so `<hipblaslt/...>` API headers
# resolve), and the link to roc::mxDataGenerator. PUBLIC so the executables
# below pick the macro up too.
if(HIPBLASLT_ENABLE_MXDATAGENERATOR)
    target_link_libraries(tensilelite-client-common PUBLIC hipblaslt::mxdatagen)
endif()

target_include_directories(tensilelite-client-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(tensilelite-client-common
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-client-common PUBLIC)
endif()

# Enable coverage instrumentation for client code when TENSILELITE_ENABLE_COVERAGE is ON
if(TENSILELITE_ENABLE_COVERAGE)
    message(STATUS "Enabling C++ code coverage instrumentation for tensilelite-client-common")
    target_compile_options(tensilelite-client-common PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(tensilelite-client-common PUBLIC -fprofile-instr-generate -fcoverage-mapping)
endif()

# f16c gives the validator hot loop hardware Half<->float conversion, which is
# what lets the auto-vectorizer turn the per-element comparison into AVX2 code.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mf16c HIPBLASLT_CXX_HAS_MF16C)
if(HIPBLASLT_CXX_HAS_MF16C)
    target_compile_options(tensilelite-client-common PRIVATE -mf16c)
endif()

option(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK "Enable in-process ROCprofiler-SDK counters" OFF)
target_compile_definitions(tensilelite-client-common
    PUBLIC
        TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=$<BOOL:${TENSILELITE_CLIENT_ENABLE_ROCPROFSDK}>
)
if(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    find_package(rocprofiler-sdk)
    if(NOT rocprofiler-sdk_FOUND)
        message(FATAL_ERROR "ROCprofiler-SDK not found.")
    endif()
endif()

add_executable(tensilelite-client "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tensilelite-client PRIVATE tensilelite-client-common)

add_executable(cpu-gemm-driver "${CMAKE_CURRENT_SOURCE_DIR}/cpu_gemm_driver.cpp")
target_link_libraries(cpu-gemm-driver PRIVATE tensilelite-client-common)

# rocprofiler-sdk is only used by tensilelite-client (in-process counters).
if(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    target_link_libraries(tensilelite-client PRIVATE rocprofiler-sdk)
endif()

add_subdirectory(include)
add_subdirectory(src)
