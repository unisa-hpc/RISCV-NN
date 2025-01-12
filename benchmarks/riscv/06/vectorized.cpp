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
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
            vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
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
* Packed: yes, Data type: float:uint8_t (2 words per uint8_t)
*/
void rvv_matmul_floatbitmanipu_packed2_float_uint8(
    const float *__restrict__ a,
    const uint8_t *__restrict__ b,
    float *__restrict__ c) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    auto elements = __riscv_vsetvlmax_e8m2();
    uint8_t indexes[elements];
    for (int i = 0; i < elements; i++) {
        if (i % 2)
            indexes[i] = i / 2;
        else
            indexes[i] = i / 2 + elements / 2;
    }

    vuint8m2_t index = __riscv_vle8_v_u8m2(indexes, elements);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            vfloat32m8_t vec_s = __riscv_vfmv_v_f_f32m8(0, vlmax);
            vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
            // iterate over vec_a
            for (int k = 0; k < N; k += __riscv_vsetvl_e32m8(N - k)) {
                // Load vec_a
                auto *ptr_a = a + j * N + k;
                // to make it match vec_b it has to be m8
                auto maxvl = __riscv_vsetvl_e32m8(N - k);
                vfloat32m8_t vec_a = __riscv_vle32_v_f32m8(ptr_a, maxvl);

                // load vec_b
                // e8m1 -> e32m4, since we are loading packed values it equates to e32m8
                auto vl = __riscv_vsetvl_e8m1(N - k);
                auto *ptr_b = b + i * N / 2 + k / 2;
                auto vec_b = __riscv_vle8_v_u8m1(ptr_b, vl);

                // extract the 2 halves
                // extract 4 bit values
                auto lower_vec_b = __riscv_vand_vx_u8m1(vec_b, 0b1111, vl);

                // shift the entire thing right by 4 and mask again
                auto upper_vec_b = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vec_b, 4, vl), 0b1111, vl);

                vuint8m2_t conj_b;
                conj_b = __riscv_vset_v_u8m1_u8m2(conj_b, 0, upper_vec_b);
                conj_b = __riscv_vset_v_u8m1_u8m2(conj_b, 1, lower_vec_b);
                conj_b = __riscv_vrgather_vv_u8m2(conj_b, index, elements);

                // create a negative mask
                auto negative_mask = __riscv_vmsgtu_vx_u8m2_b4(conj_b, 0b0111, maxvl);
                // remove the  magic value from vec b
                conj_b = __riscv_vsub_vx_u8m2_tumu(negative_mask, conj_b, conj_b, 0b1000, maxvl);

                auto expanded_conj_b = __riscv_vwcvtu_x_x_v_u32m8(__riscv_vwcvtu_x_x_v_u16m4(conj_b, maxvl), maxvl);
                // shift to line up the exponents
                expanded_conj_b = __riscv_vsll_vx_u32m8(expanded_conj_b, 23, maxvl);
                auto unsigned_vec_a = __riscv_vreinterpret_v_f32m8_u32m8(vec_a);
                unsigned_vec_a = __riscv_vadd_vv_u32m8(unsigned_vec_a, expanded_conj_b, maxvl);
                vec_a = __riscv_vreinterpret_v_u32m8_f32m8(unsigned_vec_a);

                vec_s = __riscv_vfadd_vv_f32m8(vec_s, vec_a, maxvl);
            }
            float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m8_f32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
        }
    }
}