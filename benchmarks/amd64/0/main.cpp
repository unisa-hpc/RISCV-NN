#include "defs.h"
#include "codebook.h"

extern void vector_mul_scalar_autovec(
    const int32_t* __restrict__ ptr_a,
    const int32_t* __restrict__ ptr_b,
    int32_t* __restrict__ ptr_c,
    size_t n
);

extern void vector_mul_scalar_noautovec(
    const int32_t* __restrict__ ptr_a,
    const int32_t* __restrict__ ptr_b,
    int32_t* __restrict__ ptr_c,
    size_t n
);

extern void vector_shift_scalar_autovec(
    const int32_t* __restrict__ ptr_a,
    const int32_t* __restrict__ ptr_b,
    int32_t* __restrict__ ptr_c,
    size_t n
);

extern void vector_shift_scalar_noautovec(
    const int32_t* __restrict__ ptr_a,
    const int32_t* __restrict__ ptr_b,
    int32_t* __restrict__ ptr_c,
    size_t n
);

extern void vector_mul_avx(
    const int32_t* __restrict__ volatile a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c,
    size_t n
);

extern void vector_shift_avx(
    const int32_t* __restrict__ volatile ptr_a,
    const int32_t* __restrict__ ptr_b,
    int32_t* __restrict__ ptr_c,
    size_t n
);

void verify_results(const int32_t* scalar_result, const int32_t* avx_result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        auto t1 = scalar_result[i];
        auto t2 = avx_result[i];
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
        b_ptr[i] = 10;
    }

    // Measure time for scalar vector addition
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::ScalarAutoVec, true, 0)); // "Scalar Vector Multiplication (AutoVec)"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_mul_scalar_autovec(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for scalar vector addition
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::ScalarNoAutoVec, true, 0)); //"Scalar Vector Multiplication (NoAutoVec)"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_mul_scalar_noautovec(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for avx2 vector multiplication
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::AVX2, true, 0)); //"AVX Vector Multiplication"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_mul_avx(a_ptr, b_ptr, c_avx_ptr, N);
        }
    }

    // Verify results
    verify_results(c_scalar_ptr, c_avx_ptr, N);

    // Measure time for scalar vector addition
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::ScalarAutoVec, true, 1)); //"Scalar Vector Shift (AutoVec)"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_shift_scalar_autovec(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for scalar vector addition
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::ScalarNoAutoVec, true, 1)); //"Scalar Vector Shift (NoAutoVec)"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            vector_shift_scalar_noautovec(a_ptr, b_ptr, c_scalar_ptr, N);
        }
    }

    // Measure time for scalar vector addition
    {
        timer_stats tp(get_code_name(BENCH_ID, kernel_kind::AVX2, false, 0)); //"AVX Vector Shift"
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
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