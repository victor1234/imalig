cmake_minimum_required(VERSION 3.16)

project(imalig VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)

# Fetch argumentum
set(FETCHCONTENT_QUIET OFF)
set(ARGUMENTUM_BUILD_STATIC_LIBS ON)
include(FetchContent)
fetchcontent_declare(
    Argumentum
    GIT_REPOSITORY https://github.com/mmahnic/argumentum.git
    GIT_TAG v0.3.2
)
fetchcontent_makeavailable(Argumentum)

# Create target
set(IA_TARGET_NAME ${PROJECT_NAME})
add_library(${IA_TARGET_NAME} sources/imalig.cpp sources/BarcodeDetector.cpp)

# Add alias
add_library(${IA_TARGET_NAME}::${IA_TARGET_NAME} ALIAS ${IA_TARGET_NAME})

# Include paths
target_include_directories(
    ${IA_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link libraries
target_link_libraries(${IA_TARGET_NAME} opencv::opencv)
target_link_libraries(${IA_TARGET_NAME} Ceres::ceres)
target_link_libraries(${IA_TARGET_NAME} spdlog::spdlog)
target_link_libraries(${IA_TARGET_NAME} Argumentum::argumentum)

# Set compiler features
target_compile_features(${IA_TARGET_NAME} PUBLIC cxx_std_17)
set_target_properties(${IA_TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)

# Install rules
include(GNUInstallDirs)

# Create package targets file
install(TARGETS ${IA_TARGET_NAME} EXPORT ${IA_TARGET_NAME}-target)
install(
    EXPORT ${IA_TARGET_NAME}-target
    FILE ${IA_TARGET_NAME}-targets.cmake
    NAMESPACE ${IA_TARGET_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

include(CMakePackageConfigHelpers)

# Create package config file
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${IA_TARGET_NAME}-config.cmake.in"
    "${IA_TARGET_NAME}-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# Create package version file
write_basic_package_version_file(
    ${IA_TARGET_NAME}-config-version.cmake
    COMPATIBILITY ExactVersion
)

# Install cmake files
install(
    FILES
        "${PROJECT_BINARY_DIR}/${IA_TARGET_NAME}-config.cmake"
        "${PROJECT_BINARY_DIR}/${IA_TARGET_NAME}-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# Install headers
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/imalig
    TYPE INCLUDE
    PATTERN "*.md"
    EXCLUDE
)
