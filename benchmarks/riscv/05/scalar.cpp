//
// Created by saleh on 11/19/24.
//

#include "common01.h"
#include "defs.h"

void FUNCTION_NAME(vector_matmul_scalar)(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c) {
    constexpr int FACTOR = UNROLL_FACTOR0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k) {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}
