#include "defs.h"

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
