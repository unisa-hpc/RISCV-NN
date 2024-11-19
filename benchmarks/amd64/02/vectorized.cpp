#include "defs.h"

int32_t reduce_avx2(const __m256i& vec) {
    // Horizontal addition
    // Step 1: Add adjacent pairs
    __m256i v1 = _mm256_hadd_epi32(vec, vec); // 0, 2, 4, 6, 1, 3, 5, 7
    __m256i v2 = _mm256_hadd_epi32(v1, v1); // 0, 4, 1, 5, 2, 6, 3, 7

    // Step 2: Extract the final result
    int result = _mm256_extract_epi32(v2, 0) + _mm256_extract_epi32(v2, 4);

    return result;
}

// Use template params for pragmas. Using defined variables in pragmas does not work.
void vector_matmul_avx(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
) {
    constexpr int FACTOR = UNROLL_FACTOR0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            __m256i vec_s = _mm256_setzero_si256();
#pragma GCC unroll FACTOR
            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                auto* ptr_a = a + j * N + k; // `a` is row major
                auto* ptr_b = b + i * N + k; // `b` is col major
                __m256i vec_a = _mm256_load_si256((__m256i*)ptr_a);
                __m256i vec_b = _mm256_load_si256((__m256i*)ptr_b);
                __m256i vec_mul = _mm256_mullo_epi32(vec_a, vec_b);
                vec_s = _mm256_add_epi32(vec_s, vec_mul);
            }
            const int32_t sum = reduce_avx2(vec_s);
            c[j * N + i] = sum;
        }
    }
}

// Use template params for pragmas. Using defined variables in pragmas does not work.
void vector_matmul_shift(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
) {
    constexpr int FACTOR = UNROLL_FACTOR0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            __m256i vec_s = _mm256_setzero_si256();
#pragma GCC unroll FACTOR
            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                auto* ptr_a = a + j * N + k; // `a` is row major
                auto* ptr_b = b + i * N + k; // `b` is col major
                __m256i vec_a = _mm256_load_si256((__m256i*)ptr_a);
                __m256i vec_b = _mm256_load_si256((__m256i*)ptr_b);
                __m256i vec_mul = _mm256_sllv_epi32(vec_a, vec_b);
                vec_s = _mm256_add_epi32(vec_s, vec_mul);
            }
            const int32_t sum = reduce_avx2(vec_s);
            c[j * N + i] = sum;
        }
    }
}