//
// Created by saleh on 11/19/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <functional>
#include <string>
#include <cmath>
#include "common01.h"

constexpr size_t RUNS = 32;
constexpr int padding = 0;

// fallback logics
#ifndef SKIP_SCALAR_AND_VERIFICATION
#define SKIP_SCALAR_AND_VERIFICATION 0
#endif
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif
#ifndef UNROLL_FACTOR1
#define UNROLL_FACTOR1 1
#endif
#ifndef UNROLL_FACTOR2
#define UNROLL_FACTOR2 3
#endif
#ifndef UNROLL_FACTOR3
#define UNROLL_FACTOR3 3
#endif
#ifndef I_H
#define I_H 256
#endif
#ifndef I_W
#define I_W 256
#endif
#ifndef K_H
#define K_H 3
#endif
#ifndef K_W
#define K_W 3
#endif
#ifndef C_I
#define C_I 3
#endif
#ifndef C_O
#define C_O 16
#endif
#ifndef S_X
#define S_X 1
#endif
#ifndef S_Y
#define S_Y 1
#endif

// do not change these values
constexpr int input_height = I_H;
constexpr int input_width = I_W;
constexpr int kernel_height = K_H;
constexpr int kernel_width = K_W;

constexpr int stride_y = S_X;
constexpr int stride_x = S_Y;
constexpr int channel_in = C_I;
constexpr int channel_out = C_O;

constexpr size_t GetOutHeight(size_t in_height, size_t kernel_height, size_t stride) {
    return (in_height - (kernel_height - 1)) / stride;
}

constexpr size_t GetOutWidth(size_t in_width, size_t kernel_width, size_t stride) {
    return (in_width - (kernel_width - 1)) / stride;
}
