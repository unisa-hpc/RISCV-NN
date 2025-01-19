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

constexpr int BENCH_ID = 3;
constexpr size_t RUNS = 32;
constexpr int padding = 0;

constexpr int UNROLL_FACTOR0_DEFAULT = 1;
constexpr int UNROLL_FACTOR1_DEFAULT = 1;
constexpr int UNROLL_FACTOR2_DEFAULT = 1;
constexpr int UNROLL_FACTOR3_DEFAULT = 1;
constexpr int I_H_DEFAULT = 256;
constexpr int I_W_DEFAULT = 256;
constexpr int K_H_DEFAULT = 3;
constexpr int K_W_DEFAULT = 3;
constexpr int C_I_DEFAULT = 3;
constexpr int C_O_DEFAULT = 16;
constexpr int S_X_DEFAULT = 1;
constexpr int S_Y_DEFAULT = 1;


#ifndef SKIP_SCALAR_AND_VERIFICATION
#define SKIP_SCALAR_AND_VERIFICATION 0
#endif

// fallback to the defaults if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 UNROLL_FACTOR0_DEFAULT
#endif
#ifndef UNROLL_FACTOR1
#define UNROLL_FACTOR1 UNROLL_FACTOR1_DEFAULT
#endif
#ifndef UNROLL_FACTOR2
#define UNROLL_FACTOR2 UNROLL_FACTOR2_DEFAULT
#endif
#ifndef UNROLL_FACTOR3
#define UNROLL_FACTOR3 UNROLL_FACTOR3_DEFAULT
#endif
#ifndef I_H
#define I_H I_H_DEFAULT
#endif
#ifndef I_W
#define I_W I_W_DEFAULT
#endif
#ifndef K_H
#define K_H K_H_DEFAULT
#endif
#ifndef K_W
#define K_W K_W_DEFAULT
#endif
#ifndef C_I
#define C_I C_I_DEFAULT
#endif
#ifndef C_O
#define C_O C_O_DEFAULT
#endif
#ifndef S_X
#define S_X S_X_DEFAULT
#endif
#ifndef S_Y
#define S_Y S_Y_DEFAULT
#endif

// preprocessor to check if the factors are all equal to their defaults
// if `ALWAYS_REPORT` is defined, it will always report the stats for non-participants in auto-tuning
#ifndef ALWAYS_REPORT
#define ARE_ALL_DEFAULT (UNROLL_FACTOR0 == UNROLL_FACTOR0_DEFAULT && UNROLL_FACTOR1 == UNROLL_FACTOR1_DEFAULT && UNROLL_FACTOR2 == UNROLL_FACTOR2_DEFAULT && UNROLL_FACTOR3 == UNROLL_FACTOR3_DEFAULT && I_H == I_H_DEFAULT && I_W == I_W_DEFAULT && K_H == K_H_DEFAULT && K_W == K_W_DEFAULT && C_I == C_I_DEFAULT && C_O == C_O_DEFAULT && S_X == S_X_DEFAULT && S_Y == S_Y_DEFAULT)
#define ALWAYS_REPORT_STR "false"
#else
#define ARE_ALL_DEFAULT false
#define ALWAYS_REPORT_STR "true"
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

