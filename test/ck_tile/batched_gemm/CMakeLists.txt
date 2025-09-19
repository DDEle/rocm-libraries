if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx120")
    add_gtest_executable(test_ck_tile_batched_gemm test_batched_gemm.cpp)
endif()
