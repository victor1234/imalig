enable_testing()

include(CTest)
include(Catch)

add_executable(unit_tests imalig_tests.cpp)

target_link_libraries(unit_tests PRIVATE ${IA_TARGET_NAME})
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain)

# Discover tests
catch_discover_tests(unit_tests)
