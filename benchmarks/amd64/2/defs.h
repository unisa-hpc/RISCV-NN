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

constexpr int BENCH_ID = 2;
constexpr size_t RUNS = 64;
constexpr int VECTOR_SIZE = 256;
constexpr size_t VECTOR_ELEMENTS = VECTOR_SIZE / (8 * sizeof(int32_t));

constexpr int UNROLL_FACTOR0_DEFAULT = 1;
constexpr int UNROLL_FACTOR1_DEFAULT = 1;
constexpr int UNROLL_FACTOR2_DEFAULT = 1;

// fallback to the default if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 UNROLL_FACTOR0_DEFAULT
#endif

// fallback to the default if not defined
#ifndef UNROLL_FACTOR1
#define UNROLL_FACTOR1 UNROLL_FACTOR1_DEFAULT
#endif

// fallback to the default if not defined
#ifndef UNROLL_FACTOR2
#define UNROLL_FACTOR2 UNROLL_FACTOR2_DEFAULT
#endif

// fallback to 256 if not defined
#ifndef N
#define N 256
#endif

// AUTOTUNE_BASELINE_KERNELS controlls wheter to have the baseline kernels in the autotuning or not
#ifdef AUTOTUNE_BASELINE_KERNELS
#define UNROLL_FACTOR0_BASELINE UNROLL_FACTOR0
#define UNROLL_FACTOR1_BASELINE UNROLL_FACTOR1
#define UNROLL_FACTOR2_BASELINE UNROLL_FACTOR2
#else
#define UNROLL_FACTOR0_BASELINE UNROLL_FACTOR0_DEFAULT
#define UNROLL_FACTOR1_BASELINE UNROLL_FACTOR1_DEFAULT
#define UNROLL_FACTOR2_BASELINE UNROLL_FACTOR2_DEFAULT
#endif