if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx120")
    add_gtest_executable(test_ck_tile_cshuffle_epilogue test_cshuffle_epilogue.cpp)
else()
    message(DEBUG "Skipping ck_tile_cshuffle tests for current target ")
endif()

