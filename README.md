# Imalig
Precise image alignment and camera pose estimation library. Useful for AR, drone landing, SLAM.

_Under active development. Please check later._

## Features
- High precision subpixel image alignment
- Crossplatform, `arm64` support
- Easy to deploy, `conan` support

## Install
### With conan (recommended way)
#### Dependencies
- conan >= 1.54.0
```
pipx install conan
```
#### Build
```
mkdir build && cd build
conan install .. --build=missing
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

### Without conan
#### Dependencies
- OpenCV
- ceres-solver
- Eigen
#### Build
```
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## Example
```
./imalig_example 23 image.jpg
```

## CMake integration
```cmake
find_package(imalig REQUIRED)

add_executable(program program.cpp)
target_link_libraries(program imalig::imalig)
```
