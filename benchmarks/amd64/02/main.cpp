//
// Created by saleh on 11/19/24.
//

#include "defs.h"

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
    const int32_t* __restrict__ b,
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
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment

    std::cout << "UNROLL_FACTOR0: " << UNROLL_FACTOR0 << std::endl;

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
        timer_stats tp("Scalar Matmul With Mul NoAutovec", {{"UNROLL_FACTOR0", UNROLL_FACTOR0}, {"N", N}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_scalar_noautovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    {
        timer_stats tp("Scalar Matmul With Mul Autovec", {{"UNROLL_FACTOR0", UNROLL_FACTOR0}, {"N", N}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_scalar_autovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    {
        timer_stats tp("AVX Matmul With Mul", {{"UNROLL_FACTOR0", UNROLL_FACTOR0}, {"N", N}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_avx(a_ptr, b_ptr, c_avx_mul_ptr);
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
        timer_stats tp("AVX Matmul With Shift", {{"UNROLL_FACTOR0", UNROLL_FACTOR0}, {"N", N}});
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_matmul_shift(a_ptr, b_ptr, c_avx_shift_ptr);
        }
    }
    verify_results(c_scalar_ptr, c_avx_shift_ptr);

    return 0;
}