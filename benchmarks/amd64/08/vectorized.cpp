#include "defs.h"


// Helper function: Horizontal reduction of __m256 vector to a single float
float reduce_avx2_float32(__m256 vec) {
    // Perform horizontal addition
    __m128 lo = _mm256_castps256_ps128(vec);   // Lower 128 bits
    __m128 hi = _mm256_extractf128_ps(vec, 1); // Upper 128 bits
    __m128 sum128 = _mm_add_ps(lo, hi);        // Add lower and upper parts
    sum128 = _mm_hadd_ps(sum128, sum128);      // Horizontal add
    sum128 = _mm_hadd_ps(sum128, sum128);      // Final horizontal add
    return _mm_cvtss_f32(sum128);              // Extract the lowest float
}

// Vectorized matrix multiplication for float32 using AVX
void avx2_matmul_mul_nopack_float(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ c) {
    constexpr int VECTOR_ELEMENTS = 8; // AVX processes 8 float32 elements at once
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            __m256 vec_s = _mm256_setzero_ps(); // Initialize accumulator to zero

            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                const float *ptr_a = a + j * N + k; // `a` is row major
                const float *ptr_b = b + i * N + k; // `b` is column major

                __m256 vec_a = _mm256_load_ps(ptr_a); // Load 8 float32 elements from `a`
                __m256 vec_b = _mm256_load_ps(ptr_b); // Load 8 float32 elements from `b`

                __m256 vec_mul = _mm256_mul_ps(vec_a, vec_b); // Multiply 8 elements
                vec_s = _mm256_add_ps(vec_s, vec_mul);        // Accumulate results
            }

            // Horizontal reduction of vec_s to get the final sum for this position
            float sum = reduce_avx2_float32(vec_s);
            c[j * N + i] = sum;
        }
    }
}

void avx512_matmul_mul_nopack_float(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ c) {

    constexpr int VECTOR_ELEMENTS = 16; // AVX512

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            __m512 vec_s = _mm512_setzero_ps(); // Initialize accumulator to zero

            for (int k = 0; k < N; k += VECTOR_ELEMENTS) {
                const float *ptr_a = a + j * N + k; // `a` is row major
                const float *ptr_b = b + i * N + k; // `b` is column major

                __m512 vec_a = _mm512_load_ps(ptr_a); // Load 8 float32 elements from `a`
                __m512 vec_b = _mm512_load_ps(ptr_b); // Load 8 float32 elements from `b`

                __m512 vec_mul = _mm512_mul_ps(vec_a, vec_b); // Multiply 8 elements
                vec_s = _mm512_add_ps(vec_s, vec_mul);        // Accumulate results
            }

            // Horizontal reduction of vec_s to get the final sum for this position
            // float sum = reduce_avx512(c_accum);
            float sum = _mm512_reduce_add_ps(vec_s);

            c[j * N + i] = sum;
        }
    }
}

// Handling negative numbers WITHOUT using the magic number, a bit slower than the magic number approach
void avx512_matmul_floatbitmanipu_nopack_float_uint8_no_magic(
    const float *__restrict__ a,
    const uint16_t *__restrict__ b,
    float *__restrict__ c) {

    // This kernel is for float32 which has:
    // 1 sign bit, 8 exponent bits, 23 mantissa bits
    // The weight words are encoded as 16-bit uint16_ts with:
    //          1 sign bit, 8 exponent bits, 7 zeros
    // Note that the exponent bits are not two's complement encoded. The sign bit is just a flag.

    // We take float numbers from `a` and weight words from `b`
    // Arrays A and C are represented as uint32_t but encoded in float32.

    // We are replacing float MUL against PoT weights with `addition` of float vec_a's (sign, exponent) with (PoT sign, PoT exponent)

    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;

    alignas(64) const uint16_t indexes[32] = {
        0xFF, 0, 0xFF, 1, 0xFF, 2, 0xFF, 3,
        0xFF, 4, 0xFF, 5, 0xFF, 6, 0xFF, 7,
        0xFF, 8, 0xFF, 9, 0xFF, 10, 0xFF, 11,
        0xFF, 12, 0xFF, 13, 0xFF, 14, 0xFF, 15};

    // Load permutation indexes
    __m512i idx = _mm512_load_si512(indexes);

    #pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j) {
        #pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i) {
            __m512 vec_s = _mm512_setzero_ps(); // Initialize accumulator to zero
            #pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += 16) {
                const uint16_t *ptr_b = b + i * N + k; // `b` is column major
                const float *ptr_a = a + j * N + k;    // `a` is row major

                // Load 32 words of uint16_t from `a` (dont forget that amd64 is little-endian)
                // w0_lo, w0_hi, w1_lo, w1_hi, ..., w15_lo, w15_hi
                __m512 vec_a = _mm512_load_ps(reinterpret_cast<const __m512i *>(ptr_a));
                auto vec_a_int = _mm512_castps_si512(vec_a);
                //__m512 vec_a_f = _mm512_castsi512_ps(vec_a);
                __m256i vec_b = _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr_b));
                __m512i vec_b512 = _mm512_castsi256_si512(vec_b);
                __m512i vec_b512_expanded = _mm512_permutexvar_epi16(idx, vec_b512);

                __m512i vec_sum = _mm512_add_epi16(vec_a_int, vec_b512_expanded);
                __m512 vec_f = _mm512_castsi512_ps(vec_sum);
                vec_s = _mm512_add_ps(vec_s, vec_f);
            }

            float sum = _mm512_reduce_add_ps(vec_s);
            c[j * N + i] = sum;
        }
    }
}
