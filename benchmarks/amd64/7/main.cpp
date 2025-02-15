//
// Created by saleh on 11/19/24.
//

#include "defs.h"
#include "codebook.h"

extern void vector_matmul_scalar_autovec (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c
);

extern void vector_matmul_scalar_noautovec (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c
);

extern void avx2_matmul_mul_nopack_float(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ c
);

extern void avx512_matmul_mul_nopack_float(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c
);

extern void avx512_matmul_floatbitmanipu_nopack_float_uint8(
    const float *__restrict__ a,
    const uint8_t *__restrict__ b,
    float *__restrict__ c);

int main(int argc, char** argv) {
    constexpr int ALIGNMENT = 64;
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

    aligned_tensor<float> a_tensor({N, N}, ALIGNMENT);
    aligned_tensor<float> b_tensor({N, N}, ALIGNMENT);
    aligned_tensor<float> c_tensor_scalar({N, N}, ALIGNMENT);
    aligned_tensor<float> c_tensor_avx5_mul({N, N}, ALIGNMENT);
    
    c_tensor_scalar.wipe();
    c_tensor_avx5_mul.wipe();
    a_tensor.initialize(aligned_tensor<float>::init_type::random, -10000.f, 10000.f);
    
    auto *a_ptr = a_tensor.data_t();
    auto *b_ptr = b_tensor.data_t();
    auto *c_scalar_ptr = c_tensor_scalar.data_t();
    auto *c_avx_mul_ptr = c_tensor_avx5_mul.data_t();
    
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t idx = j * N + i;
            b_ptr[i * N + j] = static_cast<float>(std::pow(2, rand() % 7)); ///TODO: Check the range, this bench uses the RVV approach for neg. numbers
            if (rand() % 2) {
                b_ptr[i * N + j] *= -1;
            }
        }
    }

    if (RUN_BASELINES) {
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
        for (volatile size_t i = 0; i < RUNS_SCALAR; ++i) {
            timer_scope ts(tp);
            vector_matmul_scalar_noautovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }

    if (RUN_BASELINES) {
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
        for (volatile size_t i = 0; i < RUNS_SCALAR; ++i) {
            timer_scope ts(tp);
            vector_matmul_scalar_autovec(a_ptr, b_ptr, c_scalar_ptr);
        }
    }

    if (RUN_BASELINES) {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::AVX2, true, 0), //"AVX2 Matmul With Mul Float",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0_BASELINE},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1_BASELINE},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2_BASELINE},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            !true
        );
        for (volatile size_t i = 0; i < RUNS; ++i) {
            timer_scope ts(tp);
            avx2_matmul_mul_nopack_float(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
    if (RUN_BASELINES) c_tensor_scalar.compare(c_tensor_avx5_mul);
    c_tensor_avx5_mul.wipe();

    if (RUN_BASELINES) {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::AVX512, true, 0), //"AVX512 Matmul With Mul Float",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0_BASELINE},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1_BASELINE},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2_BASELINE},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            !true
        );
        for (volatile size_t i = 0; i < RUNS; ++i) {
            timer_scope ts(tp);
            avx512_matmul_mul_nopack_float(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
    if (RUN_BASELINES) c_tensor_scalar.compare(c_tensor_avx5_mul);
    c_tensor_avx5_mul.wipe();

    // Convert to PoT
    aligned_tensor<uint8_t> b_pot_tensor({N, N}, ALIGNMENT);
    auto *bp_ptr = b_pot_tensor.data_t();

    // Parse the B array to make it contain logs over actual powers of 2
    for (size_t i = 0; i < N * N; i++) {
        uint8_t v = std::log2(std::abs(b_ptr[i]));
        if (std::pow(2, v) != std::abs(b_ptr[i])) {
            std::cerr << "Error: " << b_ptr[i] << " is not a power of 2" << std::endl;
            return 1;
        }
        if (b_ptr[i] < 0) { // Add the magic number for the negative numbers
            v |= 0b100000;
        }
        bp_ptr[i] = v;
    }

    {
        timer_stats tp(
            get_code_name(BENCH_ID, kernel_kind::AVX512, false, 0), //"AVX512 Matmul BitManipu float:uint8 nopack",
            {
                {"UNROLL_FACTOR0", UNROLL_FACTOR0},
                {"UNROLL_FACTOR1", UNROLL_FACTOR1},
                {"UNROLL_FACTOR2", UNROLL_FACTOR2},
                {"N", N}, {"FLAG_AUTOTUNE_DISABLED", FLAG_AUTOTUNE_DISABLED}
            },
            false
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            avx512_matmul_floatbitmanipu_nopack_float_uint8(a_ptr, bp_ptr, c_avx_mul_ptr);
        }
    }
    if (RUN_BASELINES) c_tensor_scalar.compare(c_tensor_avx5_mul);

    return 0;
}