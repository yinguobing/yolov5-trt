cmake_minimum_required(VERSION 3.16.0)
project(yolov5 VERSION 0.1.0)

include(CTest)
enable_testing()

set(DEPRECATION_FLAG "-Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEPRECATION_FLAG}")

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Found CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT-8.2.5.1)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")

# TensorRT infer
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT-nvinfer libs: ${TENSORRT_LIBRARY_INFER}")

# TensorRT onnxparser
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT-onnxparser libs: ${TENSORRT_LIBRARY_ONNXPARSER}")
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(yolov5 yolov5.cpp)
target_include_directories(
    yolov5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS} 
    ${TENSORRT_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS} 
)

target_link_libraries(
    yolov5
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIBRARY} 
    ${OpenCV_LIBS}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
