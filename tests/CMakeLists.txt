enable_testing()

include(CTest)
include(Catch)

add_executable(unit_tests imalig_tests.cpp)

target_link_libraries(unit_tests PRIVATE ${IA_TARGET_NAME})
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain)

# Copy fixtures
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/image.jpg
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/image.jpg
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/camera-parameters.yml
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/camera-parameters.yml
    COPYONLY
)

# Discover tests
catch_discover_tests(unit_tests)
