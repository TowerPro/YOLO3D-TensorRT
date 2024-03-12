cmake_minimum_required(VERSION 2.8)
project(test)

set(CMAKE_C_COMPILER "/home/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/home/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

add_compile_options(-w -Wdeprecated-declarations)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/include/opencv4)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/logc/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cuda/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ffmpeg/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cuda/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ffmpeg/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)

set(ALL_SRC)
file(GLOB ALL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/logc/src/*.c
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(demo ${ALL_SRC})
target_link_libraries(demo opencv_calib3d opencv_core opencv_dnn opencv_features2d 
                        opencv_flann opencv_gapi opencv_highgui opencv_imgcodecs 
                        opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching 
                        opencv_video opencv_videoio pthread nvcaffe_parser nvinfer nvinfer_plugin
                        nvonnxparser nvparsers nvsample_cudaprocess nvvpi EGL nvdla_compiler nvmedia
                        cublas cublasLt cudart cufft cufftw cuinj64 cupti curand cusolver cudnn nvbuf_utils
                        cusparse nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc
                        npps nvToolsExt nvblas nvgraph nvperf_host nvperf_target nvrtc-builtins nvrtc nvrm
                        nvrm_graphics nvtvmr nvos EGL_nvidia GLdispatch nvdc nvparser nvdla_runtime nvddk_vic
                        nvbuf_fdmap nvrm_gpu nvidia-fatbinaryloader nvidia-glsi nvimp nvidia-rmapi-tegra stdc++ m
                        avcodec avdevice avfilter avformat avutil swresample swscale)

target_link_libraries(demo PRIVATE libc_nonshared.a)