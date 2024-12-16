# Benchmark
**ARCH**: `RISCV`  

Matrix multiplication using scalar and vector MUL instructions and vector SHIFT instructions. **No** inline assembly is used.
This benchmark uses int32_t for all tensors. No virtual packing.
All the kernels in this benchmark **cannot** handle negative values.

## Sub Benchmarks
### `vector_matmul_scalar`
Scalar matrix multiplication with MUL instructions.
### `vector_matmul_rvv`
RVV matrix multiplication with MUL instructions.
### `vector_matmul_shift`
RVV matrix multiplication with SHIFT instructions.

## Notes
### The Content of Matrix B
Before running `vector_matmul_shift`, the content of the second matrix is converted to `matB = log2(matB)`. 
This is done to ensure that the SHIFT instructions will still generate a valid result.

### Unrolling Factor
When nothing set, the unrolling factor of 1 is used. To change the unrolling factor, compile like this:
```bash
bash build.riscv.00.sh 02.cpp "-DUNROLL_FACTOR=4"
```
The json files will be created in the dump directory containing the profiled stats.