/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

#include "defs.h"
#include "common01.h"

// append SUFFIX to the function name
void FUNCTION_NAME(vector_matmul_scalar) (
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c ) {
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            c[j * N + i] = 0;
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; ++k) {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}
