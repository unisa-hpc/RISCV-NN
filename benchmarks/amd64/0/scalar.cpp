#include "defs.h"
#include "common01.h"

// Function to add vectors using scalar operations
void FUNCTION_NAME(vector_shift_scalar)(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
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

void FUNCTION_NAME(vector_mul_scalar)(const int32_t* __restrict__ ptr_a, const int32_t* __restrict__ ptr_b,
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
