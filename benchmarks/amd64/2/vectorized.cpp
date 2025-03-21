/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

#include "defs.h"

constexpr int VECTOR_ELEMENTS = 16;

// Use template params for pragmas. Using defined variables in pragmas does not work.
void vector_matmul_avx(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            __m512i vec_s = _mm512_setzero_si512();
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                auto *ptr_a = a + j * N + k; // `a` is row major
                auto *ptr_b = b + i * N + k; // `b` is col major
                __m512i vec_a = _mm512_load_si512((__m512i *)ptr_a);
                __m512i vec_b = _mm512_load_si512((__m512i *)ptr_b);
                __m512i vec_mul = _mm512_mullo_epi32(vec_a, vec_b);
                vec_s = _mm512_add_epi32(vec_s, vec_mul);
            }
            const int32_t sum = _mm512_reduce_add_epi32(vec_s);
            c[j * N + i] = sum;
        }
    }
}

// Use template params for pragmas. Using defined variables in pragmas does not work.
void vector_matmul_shift(
    const int32_t *__restrict__ a,
    const uint32_t *__restrict__ b,
    int32_t *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            __m512i vec_s = _mm512_setzero_si512();
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                auto *ptr_a = a + j * N + k; // `a` is row major
                auto *ptr_b = b + i * N + k; // `b` is col major
                __m512i vec_a = _mm512_load_si512((__m512i *)ptr_a);
                __m512i vec_b = _mm512_load_si512((__m512i *)ptr_b);
                __m512i vec_mul = _mm512_sllv_epi32(vec_a, vec_b);
                vec_s = _mm512_add_epi32(vec_s, vec_mul);
            }
            const int32_t sum = _mm512_reduce_add_epi32(vec_s);
            c[j * N + i] = sum;
        }
    }
}