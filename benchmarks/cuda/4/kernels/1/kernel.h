#pragma once

#include <memory>

constexpr unsigned fixedMaskWidth = 3;

extern void LaunchKernel01(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel01_PoT(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel01_PoT16(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint16_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel01_PoT4bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel04(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel04_PoT(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel04_PoT4bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel04_PoT2bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC);

extern void LaunchKernel08(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC);
