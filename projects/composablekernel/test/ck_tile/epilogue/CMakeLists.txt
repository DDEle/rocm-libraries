# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_cshuffle_epilogue test_cshuffle_epilogue.cpp)
else()
    message(DEBUG "Skipping ck_tile_cshuffle tests for current target ")
endif()
