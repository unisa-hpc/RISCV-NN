//
// Created by saleh on 11/19/24.
//

#include "common01.h"
#include "defs.h"

void FUNCTION_NAME(vector_matmul_scalar)(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k) {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}
