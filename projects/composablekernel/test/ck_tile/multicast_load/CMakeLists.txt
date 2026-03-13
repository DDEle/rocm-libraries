# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_cluster_load_multicast test_cluster_load_multicast.cpp)
endif()
