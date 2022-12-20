include(Catch)

add_executable(benchmarks benchmarks.cpp)

target_link_libraries(benchmarks PRIVATE ${IA_TARGET_NAME})
target_link_libraries(benchmarks PRIVATE Catch2::Catch2WithMain)

# Copy fixtures
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/image.jpg
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/image.jpg
    COPYONLY
)
