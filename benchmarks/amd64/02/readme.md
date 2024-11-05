# Benchmark
**ARCH**: `AMD64`

Matrix multiplication using scalar and vector MUL instructions and vector SHIFT instructions. **No** inline assembly is used.

## Sub Benchmarks
### `vector_matmul_scalar`
Scalar matrix multiplication with MUL instructions.
### `vector_matmul_avx`
AVX2 matrix multiplication with MUL instructions.
### `vector_matmul_shift`
AVX2 matrix multiplication with SHIFT instructions.

## Notes
### The Content of Matrix B
Before running `vector_matmul_shift`, the content of the second matrix is converted to `matB = log2(matB)`.
This is done to ensure that the SHIFT instructions will still generate a valid result.
