/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh.
//

#pragma once

#include <memory>

extern void LaunchKernelMatmulBase(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernelMatmulPotUint8Packed2(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);


extern void LaunchKernelMatmulPotUint8Packed4(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);