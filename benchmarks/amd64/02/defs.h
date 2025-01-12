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

constexpr size_t RUNS = 256;
constexpr int VECTOR_SIZE = 256;
constexpr size_t VECTOR_ELEMENTS = VECTOR_SIZE / (8 * sizeof(int32_t));

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR1
#define UNROLL_FACTOR1 1
#endif

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR2
#define UNROLL_FACTOR2 1
#endif

// fallback to 256 if not defined
#ifndef N
#define N 256
#endif
