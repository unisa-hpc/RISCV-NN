#include "defs.h"

// Helper function: Horizontal reduction of __m256 vector to a single float
float reduce_avx2_float32(__m256 vec)
{
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
    float *__restrict__ c)
{
    constexpr int VECTOR_ELEMENTS = 8; // AVX processes 8 float32 elements at once
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j)
    {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i)
        {
            __m256 vec_s = _mm256_setzero_ps(); // Initialize accumulator to zero

#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += VECTOR_ELEMENTS)
            {
                const float *ptr_a = a + j * N + k; // `a` is row major
                const float *ptr_b = b + i * N + k; // `b` is column major

                __m256 vec_a = _mm256_load_ps(ptr_a); // Load 8 float32 elements from `a`
                __m256 vec_b = _mm256_load_ps(ptr_b); // Load 8 float32 elements from `b`

                //__m256 vec_mul = _mm256_mul_ps(vec_a, vec_b); // Multiply 8 elements
                // vec_s = _mm256_add_ps(vec_s, vec_mul);        // Accumulate results

                // using FMA
                vec_s = _mm256_fmadd_ps(vec_a, vec_b, vec_s); // a*b+c  --> args=[a,b,c]
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
    float *__restrict__ c)
{

    constexpr int VECTOR_ELEMENTS = 16; // AVX512
    constexpr int FACTOR0 = UNROLL_FACTOR0_BASELINE;
    constexpr int FACTOR1 = UNROLL_FACTOR1_BASELINE;
    constexpr int FACTOR2 = UNROLL_FACTOR2_BASELINE;
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j)
    {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i)
        {
            __m512 vec_s = _mm512_setzero_ps(); // Initialize accumulator to zero

#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += VECTOR_ELEMENTS)
            {
                const float *ptr_a = a + j * N + k; // `a` is row major
                const float *ptr_b = b + i * N + k; // `b` is column major

                __m512 vec_a = _mm512_load_ps(ptr_a); // Load 8 float32 elements from `a`
                __m512 vec_b = _mm512_load_ps(ptr_b); // Load 8 float32 elements from `b`

                //__m512 vec_mul = _mm512_mul_ps(vec_a, vec_b); // Multiply 8 elements
                // vec_s = _mm512_add_ps(vec_s, vec_mul);        // Accumulate results

                // using FMA
                vec_s = _mm512_fmadd_ps(vec_a, vec_b, vec_s); // a*b+c  --> args=[a,b,c]
            }

            // Horizontal reduction of vec_s to get the final sum for this position
            // float sum = reduce_avx512(c_accum);
            float sum = _mm512_reduce_add_ps(vec_s);

            c[j * N + i] = sum;
        }
    }
}

const __m512 SIGN_MASK = _mm512_set1_ps(-0.0);
const __m512 INF = _mm512_set1_ps(std::numeric_limits<float>::infinity());

__mmask16 is_not_infinity(__m512 x)
{
    x = _mm512_andnot_ps(SIGN_MASK, x);
    const auto mask = _mm512_cmp_ps_mask(x, INF, _CMP_NEQ_UQ);
    return mask;
}

// Handling negative numbers using the magic number
void avx512_matmul_floatbitmanipu_nopack_float_uint8(
    const float *__restrict__ a,
    const uint8_t *__restrict__ b,
    float *__restrict__ c)
{
    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;

    auto negative_detection = _mm512_set1_epi32(0b11111);
    auto sign_removal = _mm512_set1_epi32(0b100000);
    auto sign_vec = _mm512_set1_epi32(0b10000000000000000000000000000000);
#pragma GCC unroll FACTOR0
    for (int j = 0; j < N; ++j)
    {
#pragma GCC unroll FACTOR1
        for (int i = 0; i < N; ++i)
        {
            __m512 vec_s = _mm512_setzero_ps(); // Initialize accumulator to zero
#pragma GCC unroll FACTOR2
            for (int k = 0; k < N; k += 64)
            {
                // Load the main vector B
                const uint8_t *ptr_b = b + i * N + k; // `b` is column major
                __m512i vec_b = _mm512_load_si512(reinterpret_cast<const __m512i *>(ptr_b));
                for (int t = 0; t < 4; t++)
                {
                    // start the splitting process. 4 vectors of 128 bits each, converted to 4 vectors of 512bits each
                    auto b1 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(vec_b, t));
                    // 1st Pass
                    const float *ptr_a = a + j * N + k + 16 * t; // `a` is row major
                    __m512 vec_a = _mm512_load_ps(reinterpret_cast<const __m512i *>(ptr_a));
                    const auto infinity_mask = is_not_infinity(vec_a);
                    auto vec_a_int = _mm512_castps_si512(vec_a);
                    auto negative_mask = _mm512_cmpgt_epi32_mask(b1, negative_detection);
                    b1 = _mm512_mask_sub_epi32(b1, negative_mask, b1, sign_removal);
                    // shift vec_b to line up the exponents
                    auto shifted_vec = _mm512_slli_epi32(b1, 23);
                    // add exponents
                    __m512i vec_sum = _mm512_mask_add_epi32(vec_a_int, infinity_mask, vec_a_int, shifted_vec);
                    vec_sum = _mm512_mask_xor_epi32(vec_sum, negative_mask, vec_sum, sign_vec);
                    // accumulate
                    __m512 vec_f = _mm512_castsi512_ps(vec_sum);
                    vec_s = _mm512_add_ps(vec_s, vec_f);
                }
            }

            float sum = _mm512_reduce_add_ps(vec_s);
            c[j * N + i] = sum;
        }
    }
}
