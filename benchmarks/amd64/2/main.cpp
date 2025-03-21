/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh on 11/19/24.
//

#include "defs.h"
#include "codebook.h"

extern void vector_matmul_scalar_autovec (
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
);

extern void vector_matmul_scalar_noautovec (
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
);

extern void vector_matmul_shift(
    const int32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    int32_t* __restrict__ c
);

extern void vector_matmul_avx(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c
);

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
    constexpr size_t ALIGNMENT = 64; // 64-byte alignment
    const int RUNS_SCALAR = RUNS<=7 ? RUNS : 7;

    std::cout << "N: " << N << std::endl;
    std::cout << "FLAG_AUTOTUNE_DISABLED: " << FLAG_AUTOTUNE_DISABLED << std::endl;
    std::cout << "UNROLL_FACTOR0: " << UNROLL_FACTOR0 << std::endl;
    std::cout << "UNROLL_FACTOR1: " << UNROLL_FACTOR1 << std::endl;
    std::cout << "UNROLL_FACTOR2: " << UNROLL_FACTOR2 << std::endl;
    std::cout << "UNROLL_FACTOR0_BASELINE: " << UNROLL_FACTOR0_BASELINE << std::endl;
    std::cout << "UNROLL_FACTOR1_BASELINE: " << UNROLL_FACTOR1_BASELINE << std::endl;
    std::cout << "UNROLL_FACTOR2_BASELINE: " << UNROLL_FACTOR2_BASELINE << std::endl;

    std::cout << "RUNS: " << RUNS << std::endl;
    std::cout << "RUNS_SCALAR: " << RUNS_SCALAR << std::endl;
    #ifdef AUTOTUNE_BASELINE_KERNELS
    std::cout << "AUTOTUNE_BASELINE_KERNELS: YES" << std::endl;
    #else
    std::cout << "AUTOTUNE_BASELINE_KERNELS: NO" << std::endl;
    #endif
    std::cout << "RUN_BASELINES: " << RUN_BASELINES << std::endl;
    std::cout << "RUN_SCALARS: " << RUN_SCALARS << std::endl;

    auto* a_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);;
    auto* b_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);;
    auto* bp_ptr = aligned_alloc_array<uint32_t>(N*N, ALIGNMENT);;
    auto* c_scalar_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto* c_avx_mul_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto* c_avx_shift_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);

    wipe(c_scalar_ptr, N * N);
    wipe(c_avx_mul_ptr, N * N);
    wipe(c_avx_shift_ptr, N * N);

    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            a_ptr[j * N + i] = static_cast<int32_t>(19); // `a` is row major
            // random >positive< integer between 0 and 10
            int r = rand();
            r = r % 10;
            b_ptr[i * N + j] = static_cast<int32_t>(std::pow(2, r)); // `b` is col major
            bp_ptr[i * N + j] = static_cast<uint32_t>(r); // `b` is col major
        }
    }

    if (RUN_BASELINES && RUN_SCALARS) {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::ScalarNoAutoVec, true, 0), //"Scalar Matmul With Mul NoAutovec",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0_BASELINE},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1_BASELINE},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2_BASELINE},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            !true
        );
        for (volatile size_t i = 0; i < RUNS_SCALAR; i++) {
            timer_scope ts(tp);
            vector_matmul_scalar_noautovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    if (RUN_BASELINES && RUN_SCALARS) {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::ScalarAutoVec, true, 0), //"Scalar Matmul With Mul Autovec",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0_BASELINE},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1_BASELINE},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2_BASELINE},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            !true
        );
        for (volatile size_t i = 0; i < RUNS_SCALAR; i++) {
            timer_scope ts(tp);
            vector_matmul_scalar_autovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    if (RUN_BASELINES) {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::AVX512, true, 0), //"AVX Matmul With Mul",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0_BASELINE},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1_BASELINE},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2_BASELINE},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            !true
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_avx(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
    if (RUN_BASELINES && RUN_SCALARS) verify_results(c_scalar_ptr, c_avx_mul_ptr);

    {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::AVX512, false, 0), //"AVX Matmul With Shift",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            false // always report
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_shift(a_ptr, bp_ptr, c_avx_shift_ptr);
        }
    }
    if (RUN_BASELINES && RUN_SCALARS) verify_results(c_scalar_ptr, c_avx_shift_ptr);

    return 0;
}