//
// Created by saleh on 11/4/24.
//

#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include "common01.h"

constexpr size_t RUNS = 1000;
constexpr size_t N = 1024 * 1024; // 16M elements
constexpr size_t VECTOR_ELEMENTS = 8; // 8 elements in a vector (int32)

void vector_mul_scalar(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
                       int32_t* __restrict__ ptr_c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int32_t value = ptr_a[i];
        int32_t multiplier = ptr_b[i];
        int32_t result;

        asm volatile (
            "movl %1, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            //"movl %1, %%eax;"      // Reset eax to the original value
            "imull %2, %%eax;"
            "movl %%eax, %0;"
            : "=r" (result)
            : "r" (value), "r" (multiplier)
            : "eax"
        );

        ptr_c[i] = result;
    }
}

// Function to mul vectors using RVV intrinsics
void vector_mul_avx(const int32_t* __restrict__ volatile a, const int32_t* __restrict__ b, int32_t* __restrict__ c,
                    size_t n) {
    for (size_t i = 0; i < n; i += VECTOR_ELEMENTS) {
        // load 8 int32 from aligned mem into vec_tmp
        __m256i vec_a = _mm256_load_si256((__m256i*)&a[i]);
        __m256i vec_b = _mm256_load_si256((__m256i*)&b[i]);

        __m256i vec_c;
        // Perform vector multiplication
        asm volatile (
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"
            "vpmulld %[vec_a], %[vec_b], %[vec_c];"

            : [vec_c] "=x" (vec_c)
            : [vec_a] "x" (vec_a), [vec_b] "x" (vec_b)
        );

        // store vec_c into aligned mem
        _mm256_store_si256((__m256i*)&c[i], vec_c);
    }
}

// Function to add vectors using scalar operations
void vector_shift_scalar(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
                         int32_t* __restrict__ ptr_c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int32_t value = ptr_a[i];
        int32_t shift_amount = ptr_b[i];
        int32_t result;

        // Inline assembly for shift operation with true dependency chain
        asm volatile (
            "movl %1, %%eax;"         // Load initial value into eax
            "movl %2, %%ecx;"         // Load shift amount into ecx

            // Repeat the shifts for latency measurement without modifying the final result
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"

            // Now perform the final shift left once more to produce the correct result
            "shll %%cl, %%eax;"       // Final left shift

            "movl %%eax, %0;"         // Store final result in output
            : "=r" (result)           // Output operand
            : "r" (value), "r" (shift_amount)  // Input operands
            : "eax", "ecx"            // Clobbered registers
        );

        ptr_c[i] = result;
    }
}

// Function to add vectors using RVV intrinsics
void vector_shift_avx(const int32_t* __restrict__ volatile ptr_a, const int32_t* __restrict__ ptr_b, int32_t* __restrict__ ptr_c,
                      size_t n) {
    for (size_t i = 0; i < n; i += VECTOR_ELEMENTS) {
        __m256i values = _mm256_loadu_si256((__m256i*)&ptr_a[i]);
        __m256i shift_amounts = _mm256_loadu_si256((__m256i*)&ptr_b[i]);
        __m256i result;

        // Inline assembly with AVX2 instructions for chained shifts
        asm volatile (
            "vmovdqa %1, %%ymm0;"       // Load initial values into ymm0
            "vmovdqa %2, %%ymm1;"       // Load shift amounts into ymm1

            // Repeat left and right shifts for dependency chain
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"
            "vpsrlvd %%ymm1, %%ymm0, %%ymm0;"


            "vmovdqa %1, %%ymm0;"       // Reset initial values
            "vpsllvd %%ymm1, %%ymm0, %%ymm0;"   // Final left shift (10th time)

            "vmovdqa %%ymm0, %0;"       // Store final result
            : "=m" (result)             // Output operand
            : "m" (values), "m" (shift_amounts) // Input operands
            : "ymm0", "ymm1"            // Clobbered registers
        );

        // Store the result back to memory
        _mm256_storeu_si256((__m256i*)&ptr_c[i], result);
    }
}

// Function to verify the results of scalar and RVV methods
void verify_results(const int32_t* scalar_result, const int32_t* rvv_result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        auto t1 = scalar_result[i];
        auto t2 = rvv_result[i];
        if (t1 != t2) {
            std::cerr << "Results mismatch at index " << i << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    size_t ALIGNMENT = 32; // 32-byte alignment
    if (N % VECTOR_ELEMENTS != 0) {
        std::cerr << "Size of the vectors should be a multiple of 256 bytes" << std::endl;
        std::exit(1);
    }
    if (N % ALIGNMENT != 0) {
        std::cerr << "Size of the vectors should be a multiple of 32 bytes" << std::endl;
        std::exit(1);
    }

    auto CheckAlloc = [](int32_t* p) {
        if (p == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            std::exit(1);
        }
    };

    int32_t *a_ptr, *b_ptr, *c_scalar_ptr, *c_avx_ptr;
    a_ptr = nullptr; b_ptr = nullptr; c_scalar_ptr = nullptr; c_avx_ptr = nullptr;
    a_ptr = aligned_alloc_array<int32_t>(N, ALIGNMENT);
    b_ptr = aligned_alloc_array<int32_t>(N, ALIGNMENT);
    c_scalar_ptr = aligned_alloc_array<int32_t>(N, ALIGNMENT);
    c_avx_ptr = aligned_alloc_array<int32_t>(N, ALIGNMENT);



    CheckAlloc(a_ptr);
    CheckAlloc(b_ptr);
    CheckAlloc(c_scalar_ptr);
    CheckAlloc(c_avx_ptr);

    for (size_t i = 0; i < N; i++) { c_scalar_ptr[i] = c_avx_ptr[i] = 0; }
    for (size_t i = 0; i < N; i++) {
        a_ptr[i] = 1;
        b_ptr[i] = 1;
    }

    // Measure time for scalar vector addition
    {
        TimerStats tp("Scalar Vector Multiplication");
        for (volatile size_t i = 0; i < RUNS; i++) {
            TimerScope ts(tp);
            vector_mul_scalar(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for RVV vector multiplication
    {
        TimerStats tp("AVX Vector Multiplication");
        for (volatile size_t i = 0; i < RUNS; i++) {
            TimerScope ts(tp);
            vector_mul_avx(a_ptr, b_ptr, c_avx_ptr, N);
        }
    }

    // Verify results
    verify_results(c_scalar_ptr, c_avx_ptr, N);

    // Measure time for scalar vector addition
    {
        TimerStats tp("Scalar Vector Shift");
        for (volatile size_t i = 0; i < RUNS; i++) {
            TimerScope ts(tp);
            vector_shift_scalar(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for scalar vector addition
    {
        TimerStats tp("AVX Vector Shift");
        for (volatile size_t i = 0; i < RUNS; i++) {
            TimerScope ts(tp);
            vector_shift_avx(a_ptr, b_ptr, c_avx_ptr, N);
        }
    }

    // Verify results
    verify_results(c_scalar_ptr, c_avx_ptr, N);

    free(a_ptr);
    free(b_ptr);
    free(c_scalar_ptr);
    free(c_avx_ptr);

    return 0;
}
