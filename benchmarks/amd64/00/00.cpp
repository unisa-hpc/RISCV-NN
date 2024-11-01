//
// Created by saleh on 11/1/24.
//

#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <functional>
#include <string>

constexpr size_t N = 1024 * 1024 * 16; // 16M elements
constexpr size_t VECTOR_ELEMENTS = 8; // 8 elements in a vector (int32)


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class TimerScope {
private:
    std::chrono::system_clock::time_point m_oTimerLast;
    const std::string m_strName;

public:
    TimerScope(const std::string& name) : m_strName(name) {
        m_oTimerLast = high_resolution_clock::now();
    }

    ~TimerScope() {
        ReportFromLast(m_strName);
    }

    template <class StdTimeResolution = std::milli>
    float FromLast() {
        auto now = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = now - m_oTimerLast;
        m_oTimerLast = now;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    float ReportFromLast(const std::string& msg = "") {
        auto t = FromLast<StdTimeResolution>();
        std::cout << "Elapsed " << msg << ": " << t << " ." << std::endl;
        return t;
    }

    template <class StdTimeResolution = std::milli>
    static inline float ForLambda(const std::function<void()>& operation) {
        auto t1 = high_resolution_clock::now();
        operation();
        auto t2 = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = t2 - t1;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    static inline float ReportForLambda(const std::function<void()>& operation) {
        auto t = ForLambda<StdTimeResolution>(operation);
        std::cout << "Elapsed: " << t << " ." << std::endl;
        return t;
    }
};

void vector_mul_scalar(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
                       int32_t* __restrict__ ptr_c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int32_t value = ptr_a[i];
        int32_t multiplier = ptr_b[i];
        int32_t result;

        asm volatile (
            "movl %1, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            "imull %2, %%eax;"
            //"movl %1, %%eax;"      // Reset eax to the original value
            "imull %2, %%eax;"
            "movl %%eax, %0;"
            : "=r" (result)
            : "r" (value), "r" (multiplier)
            : "eax"
        );

        ptr_c[i] = result;
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

// Function to add vectors using scalar operations
void vector_shift_scalar(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
                         int32_t* __restrict__ ptr_c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int32_t value = ptr_a[i];
        int32_t shift_amount = ptr_b[i];
        int32_t result;

        // Inline assembly for shift operation with true dependency chain
        asm volatile (
            "movl %1, %%eax;"         // Load initial value into eax
            "movl %2, %%ecx;"         // Load shift amount into ecx

            // Repeat the shifts for latency measurement without modifying the final result
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"
            "shll %%cl, %%eax;"
            "shrl %%cl, %%eax;"

            // Now perform the final shift left once more to produce the correct result
            "shll %%cl, %%eax;"       // Final left shift

            "movl %%eax, %0;"         // Store final result in output
            : "=r" (result)           // Output operand
            : "r" (value), "r" (shift_amount)  // Input operands
            : "eax", "ecx"            // Clobbered registers
        );

        ptr_c[i] = result;
    }
}

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

// Function to verify the results of scalar and RVV methods
void verify_results(const int32_t* scalar_result, const int32_t* rvv_result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        auto t1 = scalar_result[i];
        auto t2 = rvv_result[i];
        if (t1 != t2) {
            std::cerr << "Results mismatch at index " << i << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment
    if (N % VECTOR_ELEMENTS != 0) {
        std::cerr << "Size of the vectors should be a multiple of 256 bytes" << std::endl;
        std::exit(1);
    }

    auto CheckAlloc = [](int32_t* p) {
        if (p == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            std::exit(1);
        }
    };

    auto* a_ptr = static_cast<int32_t*>(aligned_alloc(ALIGNMENT, N * sizeof(int32_t)));
    auto* b_ptr = static_cast<int32_t*>(aligned_alloc(ALIGNMENT, N * sizeof(int32_t)));
    auto* c_scalar_ptr = static_cast<int32_t*>(aligned_alloc(ALIGNMENT, N * sizeof(int32_t)));
    auto* c_avx_ptr = static_cast<int32_t*>(aligned_alloc(ALIGNMENT, N * sizeof(int32_t)));
    CheckAlloc(a_ptr);
    CheckAlloc(b_ptr);
    CheckAlloc(c_scalar_ptr);
    CheckAlloc(c_avx_ptr);

    for (size_t i = 0; i < N; i++) { c_scalar_ptr[i] = c_avx_ptr[i] = 0; }
    for (size_t i = 0; i < N; i++) {
        a_ptr[i] = 1;
        b_ptr[i] = 5;
    }

    // Measure time for scalar vector addition
    {
        TimerScope ts("Scalar Vector Multiplication");
        vector_mul_scalar(a_ptr, b_ptr, c_scalar_ptr, N);
    }

    // Measure time for RVV vector multiplication
    {
        TimerScope ts("AVX Vector Multiplication");
        vector_mul_avx(a_ptr, b_ptr, c_avx_ptr, N);
    }

    // Verify results
    verify_results(c_scalar_ptr, c_avx_ptr, N);

    // Measure time for scalar vector addition

    {
        TimerScope ts("Scalar Vector Shift");
        vector_shift_scalar(a_ptr, b_ptr, c_scalar_ptr, N);
    }

    // Measure time for scalar vector addition

    {
        TimerScope ts("AVX Vector Shift");
        vector_shift_avx(a_ptr, b_ptr, c_avx_ptr, N);
    }

    // Verify results
    verify_results(c_scalar_ptr, c_avx_ptr, N);

    free(a_ptr);
    free(b_ptr);
    free(c_scalar_ptr);
    free(c_avx_ptr);

    return 0;
}
