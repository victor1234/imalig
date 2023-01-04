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
- conan >= 1.53.0
```
pipx install conan
```
#### Build
```
mkdir build && cd build
conan install .. --build=missing
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
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
```

## Example
```
./imalig_example 23 image.jpg
```
