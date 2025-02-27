cmake_minimum_required(VERSION 3.7)

include(CheckLanguage)
check_language(CXX)
check_language(CUDA)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage.")
set(CUDA_ARCHS "86" CACHE STRING "semi-colon seperated Compute Capability codes without dots, like 62;86")

# Enforce C++17 for nvcc
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(PoT_CUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# Only for debugging. Use can use CLion with cuda-gdb to debug the device code.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -O0 -Xptxas=\"-v\"")  # -G enables device debugging (its a nvcc only flag)
endif()

# For benchmarking
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3 -Xptxas=\"-v\" --fmad=true")
endif()

# For profiling, lineinfo allows to see the line number of the kernel in the profiler
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo -O3 -Xptxas=\"-v\" --fmad=true")
endif()

set(target_name "9")
set(kernel_version ${${KERNEL_EXAMPLE}})
add_executable(${target_name}
        kernels.cu
        cnpy.cpp
        main.cpp
)
target_include_directories(${target_name}
        PRIVATE
        .
        ../../common
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(${target_name} CUDA::cudart z)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
target_compile_definitions(
        ${target_name}
        PRIVATE
        TARGETNAME=\"${target_name}\"
        TARGETKERNEL=\"${kernel_version}\"

        vBK=8
        vTM=16
        vTN=4

        vUF0=1
        vUF1=1
        vUF2=1
        vUF3=1
        vUF4=1
        vUF5=1
        vUF6=1
        vUF7=1
        vUF8=1
        vUF9=1
)
target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)