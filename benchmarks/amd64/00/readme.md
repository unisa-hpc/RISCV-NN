# Benchmark

**ARCH**: `AMD64`

Multiplication vs. Shift with scalar and AVX2 instructions using inline assembly.

## Sub Benchmarks

`N=50` which is how many times important instructions are repeated to make it easier to measure the effects of their
instruction-latency and throughput.
The idea is to create true chain of dependencies so that the CPU pipeline would actually wait for each result.

### `vector_mul_scalar`

Multiply the elements of two vectors N times, with scalar instructions. So the result of 2 * 2 becomes `4^N`.

### `vector_mul_avx`

Multiply the elements of two vectors N times, with AVX2 instructions.

### `vector_shift_scalar`

Shift the element in the first vector to the left X bits, then to the right X bits, `N/2` times, with scalar instructions.
Here, X is the value of the element in the second vector.

### `vector_shift_avx`

Shift the element in the first vector to the left X bits, then to the right X bits, `N/2` times, with AVX2 instructions.
Here, X is the value of the element in the second vector.

## Notes

### Note 1

TODO note.
