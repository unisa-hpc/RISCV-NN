#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <cmath>
#include <riscv_vector.h>
#include "common01.h"

constexpr size_t RUNS = 16;
constexpr size_t N = 512;

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif

void vector_matmul_scalar(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k)
            {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}

template <int FACTOR>
void vector_matmul_rvv(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            size_t vl = 0;
            #pragma GCC unroll FACTOR
            for (int k = 0; k < N; k += vl)
            {
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

template <int FACTOR>
void vector_matmul_shift(
    const int32_t *__restrict__ a,
    const uint32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            size_t vl = 0;
            #pragma GCC unroll FACTOR
            for (int k = 0; k < N; k += vl)
            {
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

// Function to verify the results of scalar and RVV methods
void verify_results(const int32_t *c1, const int32_t *c2)
{
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            if (c1[j * N + i] != c2[j * N + i])
            {
                std::cerr << "Results mismatch at index " << i << std::endl;
                std::cerr << "c1[" << j << ", " << i << "] = " << c1[j * N + i] << std::endl;
                std::cerr << "c2[" << j << ", " << i << "] = " << c2[j * N + i] << std::endl;
                return;
            }
        }
    }
    std::cout << "Results match!" << std::endl;
}

void wipe(int32_t *p, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        p[i] = 0;
    }
}

void init(int32_t *p, size_t len, bool PoT)
{
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            if (PoT)
                p[i * N + j] = static_cast<int32_t>(std::pow(2, rand() % 16)); // `a` is row major
            else
                p[i * N + j] = static_cast<int32_t>(rand() % 32678); // `a` is row major
        }
    }
}

int main(int argc, char **argv)
{
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment

    auto *a_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto *b_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto *c_scalar_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto *c_rvv_mul_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);
    auto *c_avx_shift_ptr = aligned_alloc_array<int32_t>(N*N, ALIGNMENT);

    wipe(c_scalar_ptr, N * N);
    wipe(c_rvv_mul_ptr, N * N);
    wipe(c_avx_shift_ptr, N * N);

    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            a_ptr[j * N + i] = static_cast<int32_t>(rand() % 32678); // `a` is row major
        }
    }

    init(b_ptr, N * N, true);

    {
        timer_stats tp("Scalar Matmul With Mul", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            timer_scope ts(tp);
            vector_matmul_scalar(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
    {
        timer_stats tp("RVV Matmul With Mul", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            timer_scope ts(tp);
            vector_matmul_rvv<UNROLL_FACTOR0>(a_ptr, b_ptr, c_rvv_mul_ptr);
        }
    }
    verify_results(c_scalar_ptr, c_rvv_mul_ptr);

    // parse the B array to make it contain logs over actual powers of 2
    for (size_t i = 0; i < N * N; i++)
    {
        const auto v = static_cast<int32_t>(std::log2(b_ptr[i]));
        if (std::pow(2, v) != b_ptr[i])
        {
            std::cerr << "Error: " << b_ptr[i] << " is not a power of 2" << std::endl;
            return 1;
        }
        b_ptr[i] = v;
    }

    // shift needs an unsigned vector
    auto new_b_ptr = reinterpret_cast<uint32_t *>(b_ptr);

    {
        timer_stats tp("RVV Matmul With Shift", {{"unroll_factor", UNROLL_FACTOR0}});
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            timer_scope ts(tp);
            vector_matmul_shift<UNROLL_FACTOR0>(a_ptr, new_b_ptr, c_avx_shift_ptr);
        }
    }
    verify_results(c_scalar_ptr, c_avx_shift_ptr);

    return 0;
}
