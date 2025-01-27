//
// Created by saleh on 12/30/24.
//

#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#include <iostream>
#include <functional>
#include <cuda_runtime.h>

constexpr float MAX_ERR_FLOAT = 0.000001f;
#define CHECK(E) if(E!=cudaError_t::cudaSuccess) std::cerr<<"CUDA API FAILED, File: "<<__FILE__<<", Line: "<< __LINE__ << ", Error: "<< cudaGetErrorString(E) << std::endl;


#endif //COMMON_CUDA_H
