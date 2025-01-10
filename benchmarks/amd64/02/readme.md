# Benchmark

**ARCH**: `AMD64`

Matrix multiplication using scalar and vector MUL instructions and vector SHIFT instructions. **No** inline assembly is
used.
This benchmark uses int32_t for all tensors. No virtual packing.
All the kernels in this benchmark **cannot** handle negative values.

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

### Unrolling Factor

When nothing set, the unrolling factor of 1 is used. To change the unrolling factor, compile like this:

```bash
bash build.amd64.00.sh 02.cpp "-DUNROLL_FACTOR0=4 -DN=1024"
```

The json files will be created in the dump directory containing the profiled stats.

### Using `runme.sh`

This bash script is used for two purposes:

1. To build and run the benchmark for specific ranges of tunable parameters (N, unrolling factors, etc) and to extract
   the best combination of the tunable parameters and save them in a json file. Note that this mode will concatenate the
   new data that is to be written to the best comb json file, if the file already exists.
2. To run the benchmark with the best combination of the tunable parameters using the json file in the dumps dir.

#### Usage

```bash
bash runme.sh --machine=foo -d # to delete the entries in the dumps dir for this machine.
bash runme.sh --machine=foo --auto-tune # to auto-tune the benchmark for the machine foo and get the best comb.
bash runme.sh --machine=foo # to run the benchmark with the best comb for the machine foo and plot the results.
```