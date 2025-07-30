add_custom_target(test_synchronization)

add_gtest_executable(test_monitor_mwait monitor_mwait.cpp)
if(result EQUAL 0)
    target_link_libraries(test_monitor_mwait PRIVATE utility)
    add_dependencies(test_synchronization test_monitor_mwait)
endif()


