#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <functional>
#include <string>
#include <cmath>
#include "common01.h"

constexpr size_t RUNS = 32;
constexpr size_t N = 256;
constexpr int VECTOR_SIZE = 256;
constexpr size_t VECTOR_ELEMENTS = VECTOR_SIZE / (8 * sizeof(int32_t));

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif

int32_t reduce_avx2(const __m256i& vec) {
    // Horizontal addition
    // Step 1: Add adjacent pairs
    __m256i v1 = _mm256_hadd_epi32(vec, vec); // 0, 2, 4, 6, 1, 3, 5, 7
    __m256i v2 = _mm256_hadd_epi32(v1, v1); // 0, 4, 1, 5, 2, 6, 3, 7

    // Step 2: Extract the final result
    int result = _mm256_extract_epi32(v2, 0) + _mm256_extract_epi32(v2, 4);

    return result;
}

void vector_matmul_scalar(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k) {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}

// Use template params for pragmas. Using defined variables in pragmas does not work.
template <int FACTOR>
void vector_matmul_avx(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
) {
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
template <int FACTOR>
void vector_matmul_shift(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
) {
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

// Function to verify the results of scalar and RVV methods
void verify_results(const int32_t* c1, const int32_t* c2) {
    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            if (c1[j * N + i] != c2[j * N + i]) {
                std::cerr << "Results mismatch at index " << i << std::endl;
                std::cerr << "c1[" << j << ", " << i << "] = " << c1[j * N + i] << std::endl;
                std::cerr << "c2[" << j << ", " << i << "] = " << c2[j * N + i] << std::endl;
                return;
            }
        }
    }
    std::cout << "Results match!" << std::endl;
}

void wipe(int32_t* p, size_t len) {
    for (size_t i = 0; i < len; i++) {
        p[i] = 0;
    }
}

int main(int argc, char** argv) {
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment

    std::cout << "UNROLLING FACTOR: " << UNROLL_FACTOR0 << std::endl;

    auto* a_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);;
    auto* b_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);;
    auto* c_scalar_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto* c_avx_mul_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto* c_avx_shift_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);

    wipe(c_scalar_ptr, N * N);
    wipe(c_avx_mul_ptr, N * N);
    wipe(c_avx_shift_ptr, N * N);

    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            a_ptr[j * N + i] = static_cast<int32_t>(19); // `a` is row major
            b_ptr[i * N + j] = static_cast<int32_t>(std::pow(2, 14)); // `b` is col major
        }
    }

    {
        timer_stats tp("Scalar Matmul With Mul", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_scalar(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    {
        timer_stats tp("AVX Matmul With Mul", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_avx<UNROLL_FACTOR0>(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
    verify_results(c_scalar_ptr, c_avx_mul_ptr);

    // parse the B array to make it contain logs over actual powers of 2
    for (size_t i = 0; i < N * N; i++) {
        const auto v = static_cast<int32_t>(std::log2(b_ptr[i]));
        if (std::pow(2, v) != b_ptr[i]) {
            std::cerr << "Error: " << b_ptr[i] << " is not a power of 2" << std::endl;
            return 1;
        }
        b_ptr[i] = v;
    }
    {
        timer_stats tp("AVX Matmul With Shift", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_shift<UNROLL_FACTOR0>(a_ptr, b_ptr, c_avx_shift_ptr);
        }
    }
    verify_results(c_scalar_ptr, c_avx_shift_ptr);

    return 0;
}
