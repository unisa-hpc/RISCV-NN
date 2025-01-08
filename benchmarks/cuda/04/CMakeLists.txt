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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -O0 -Xptxas=\"-v\"")  # -G enables device debugging (its a nvcc only flag)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3 -Xptxas=\"-v\" --fmad=false")
endif()

set(target_name "output_file")
get_filename_component(DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(KERNEL_EXAMPLE kernel_${DIR_NAME})
set(${KERNEL_EXAMPLE} 1 CACHE STRING "Integer specifying the kernel version to be used in the compilation of example ${target_name}.")

set(kernel_version ${${KERNEL_EXAMPLE}})
add_executable(${target_name}
        kernels/${kernel_version}/kernel.cu
        main.cpp
)
target_include_directories(${target_name}
        PRIVATE
        kernels/${kernel_version}/
        ../../common
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(${target_name} CUDA::cublas CUDA::cudart)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
target_compile_definitions(${target_name} PRIVATE TARGETNAME=\"${target_name}\" TARGETKERNEL=\"${kernel_version}\")
#target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -G -lineinfo>)
target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# enable host compiler vectorization
target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)