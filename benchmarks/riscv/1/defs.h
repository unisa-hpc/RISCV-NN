//
// Created by saleh on 11/19/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <cmath>
#include <riscv_vector.h>
#include "common01.h"

constexpr int BENCH_ID = 1;
constexpr size_t RUNS = 75;

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

// This is just a flag, so we can separate the best config runs from best config runs with the default unrolling factors
// to compute the gain we get from autotuning, in the pyhton plotting scripts. It does not affect any of the unrolling factors values.
#ifdef AUTOTUNE_IS_DISABLED
#define FLAG_AUTOTUNE_DISABLED 1
#else
#define FLAG_AUTOTUNE_DISABLED 0
#endif

#ifndef ONLY_RUN_OURS
#define RUN_BASELINES 1
#else
#define RUN_BASELINES 0
#endif