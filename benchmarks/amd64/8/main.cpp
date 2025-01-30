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

extern void avx512_matmul_floatbitmanipu_nopack_float_uint8_no_magic(
    const float *__restrict__ a,
    const uint16_t *__restrict__ b,
    float *__restrict__ c);

int main(int argc, char** argv) {
    constexpr int ALIGNMENT = 32;
    const int RUNS_SCALAR = RUNS<=3 ? RUNS : 3;

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
    aligned_tensor<uint16_t> b_pot_tensor({N, N}, ALIGNMENT);
    aligned_tensor<float> c_tensor_scalar({N, N}, ALIGNMENT);
    aligned_tensor<float> c_tensor_avx5_mul({N, N}, ALIGNMENT);
    
    c_tensor_scalar.wipe();
    c_tensor_avx5_mul.wipe();
    a_tensor.initialize(aligned_tensor<float>::init_type::random, -10000.f, 10000.f);
    
    auto *a_ptr = a_tensor.data_t();
    auto *b_ptr = b_tensor.data_t();
    auto *bp_ptr = b_pot_tensor.data_t();
    auto *c_scalar_ptr = c_tensor_scalar.data_t();
    auto *c_avx_mul_ptr = c_tensor_avx5_mul.data_t();

    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            size_t idx = j * N + i;
            float val = static_cast<float>(pow(2, rand() % 7));
            if (rand() % 2 == 0) {
                val = -val;
            }
            b_ptr[i * N + j] = val; // `b` is col major
            uint8_t pot_e = std::log2(std::abs(val));
            uint8_t pot_s = (val < 0) ? 1 : 0;

            // This is suitable for float32
            // Be careful that even though the weight is 16-bit, the exponent is 8-bit.
            bp_ptr[i * N + j] = (pot_s << 15) | ((pot_e & 0xFF) << 7);
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
            avx512_matmul_floatbitmanipu_nopack_float_uint8_no_magic(a_ptr, bp_ptr, c_avx_mul_ptr);
        }
    }
    if (RUN_BASELINES) c_tensor_scalar.compare(c_tensor_avx5_mul);

    return 0;
}