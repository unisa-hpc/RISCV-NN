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
constexpr size_t RUNS = 64;

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

// preprocessor to check if the factors are all equal to their defaults
// if `ALWAYS_REPORT` is defined, it will always report the stats for non-participants in auto-tuning
#ifndef ALWAYS_REPORT
#define ARE_ALL_DEFAULT (UNROLL_FACTOR0 == UNROLL_FACTOR0_DEFAULT && UNROLL_FACTOR1 == UNROLL_FACTOR1_DEFAULT && UNROLL_FACTOR2 == UNROLL_FACTOR2_DEFAULT)
#define ALWAYS_REPORT_STR "false"
#else
#define ARE_ALL_DEFAULT false
#define ALWAYS_REPORT_STR "true"
#endif
