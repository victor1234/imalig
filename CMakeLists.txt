cmake_minimum_required(VERSION 3.16)

project(imalig)

find_package(OpenCV REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Ceres REQUIRED)

set(IA_TARGET_NAME ${PROJECT_NAME})
add_library(${IA_TARGET_NAME} sources/imalig.cpp sources/BarcodeDetector.cpp)

# Set compiler features
target_compile_features(${IA_TARGET_NAME} PUBLIC cxx_std_11)
set_target_properties(${IA_TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

# Include paths
target_include_directories(
    ${IA_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${IA_TARGET_NAME} opencv::opencv)
target_link_libraries(${IA_TARGET_NAME} Ceres::ceres)

# Set compiler features
target_compile_features(${IA_TARGET_NAME} PUBLIC cxx_std_20)
set_target_properties(${IA_TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
