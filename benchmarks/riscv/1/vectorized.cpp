//
// Created by saleh on 11/19/24.
// The RVV kernels are written by G. Pagano
//

#include "defs.h"
#include "common01.h"


void rvv_matmul_mul_nopack_int32(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            size_t vl = 0;
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += vl) {
                vl = __riscv_vsetvl_e32m1(N - k);
                auto *ptr_a = a + j * N + k; // `a` is row major
                auto *ptr_b = b + i * N + k; // `b` is col major
                vint32m1_t vec_a = __riscv_vle32_v_i32m1(ptr_a, vl);
                vint32m1_t vec_b = __riscv_vle32_v_i32m1(ptr_b, vl);
                vec_s = __riscv_vmacc_vv_i32m1(vec_s, vec_a, vec_b, vl);
            }
            const int32_t sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
        }
    }
}

void rvv_matmul_shift_nopack_int32(
    const int32_t *__restrict__ a,
    const uint32_t *__restrict__ b,
    int32_t *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    #pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
        #pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            size_t vl = 0;
            #pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += vl) {
                vl = __riscv_vsetvl_e32m1(N - k);
                auto *ptr_a = a + j * N + k; // `a` is row major
                auto *ptr_b = b + i * N + k; // `b` is col major
                vint32m1_t vec_a = __riscv_vle32_v_i32m1(ptr_a, vl);
                vuint32m1_t vec_b = __riscv_vle32_v_u32m1(ptr_b, vl);
                vint32m1_t vec_mul = __riscv_vsll_vv_i32m1(vec_a, vec_b, vl);
                vec_s = __riscv_vadd_vv_i32m1(vec_s, vec_mul, vl);
            }
            const int32_t sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
        }
    }
}