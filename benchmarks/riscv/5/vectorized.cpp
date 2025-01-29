//
// Created by saleh on 11/19/24.
// The RVV kernels are written by G. Pagano
//

#include "defs.h"
#include "common01.h"


void rvv_matmul_mul_nopack_float(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;

    size_t vlmax = __riscv_vsetvlmax_e32m1();
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
            vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += __riscv_vsetvl_e32m1(N - k)) {
                auto vl = __riscv_vsetvl_e32m1(N - k);
                auto *ptr_a = a + j * N + k;
                auto *ptr_b = b + i * N + k;
                vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
                vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
            }
            float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
        }
    }
}


/**
* @brief Matmul with float bits manipulation of the exponent field.
* Packed: no, Data type: float:uint8_t (1 word per uint8_t)
*/
void rvv_matmul_floatbitmanipu_nopack_float_uint8(
    const float *__restrict__ a,
    const uint8_t *__restrict__ b,
    float *__restrict__ c) {
    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    #pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
      	#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);
            vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
            // iterate over vec_a
            #pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += __riscv_vsetvl_e32m4(N - k)) {
                auto vl = __riscv_vsetvl_e32m4(N - k);
                auto *ptr_a = a + j * N + k;
                vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
                auto *ptr_b = b + i * N + k;
                vuint8m1_t vec_b = __riscv_vle8_v_u8m1(ptr_b, vl);
                // create a mask to execute the flowchart
                auto negative_mask = __riscv_vmsgtu_vx_u8m1_b8(vec_b, 0b11111, vl);
                // remove the 32 magic value from vec b
                vec_b = __riscv_vsub_vx_u8m1_tumu(negative_mask, vec_b, vec_b, 0b100000, vl);
                // expand to u16
                auto ryti = __riscv_vwcvtu_x_x_v_u32m4(__riscv_vwcvtu_x_x_v_u16m2(vec_b, vl), vl);
                // shift to line up the exponents
                ryti = __riscv_vsll_vx_u32m4(ryti, 23, vl);
                // reinterpret vec_as as uint
                auto intryti = __riscv_vreinterpret_v_f32m4_u32m4(vec_a);
                intryti = __riscv_vadd_vv_u32m4(ryti, intryti, vl);
                // handle sign, 0x80000000 -> 0b10000..00
                intryti = __riscv_vxor_vx_u32m4_m(negative_mask, intryti, 0x80000000, vl);
                //  reinterpret to f32
                auto final_thing = __riscv_vreinterpret_v_u32m4_f32m4(intryti);
                vec_s = __riscv_vfadd_vv_f32m4(vec_s, final_thing, vl);
            }
            float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
        }
    }
}