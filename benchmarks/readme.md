# Benchmarks

| BenchID | ISA/Platform | Matmul? | TypeA                      | TypeB                 | WordsPackedInB | CanHdlNeg? | NegMethod | Notes                          |
|---------|--------------|---------|----------------------------|-----------------------|----------------|------------|-----------|--------------------------------|
| 00      | AVX2         | N       | -                          | -                     | -              | -          | -         | Inline Assembly, MUL vs. Shift |
| 01      | RVV-1.0      | Y       | Int32                      | Int32                 | 1              | N          | -         |                                |
| 02      | AVX512      | Y       | Int32                      | Int32                 | 1              | N          | -         |                                |
| 03      | AVX2         | N       | -                          | -                     | -              | N          | -         | Conv2D, Int32, NoPack          |
| 04      | CUDA         | Y       | Float32, Float32 as Uint32 | Float32, Uint16 Uint8 | Mixed          | Y          | Directly  | -                              |
| 05      | RVV-1.0      | Y       | Float32                    | Uint8                 | 1              | Y          | MagicNum  | -                              |
| 06      | RVV-1.0      | Y       | Float32                    | Uint8                 | 2              | Y          | MagicNum  | -                              |
| 07      | AVX512       | Y       | Float32                    | Uint8                 | 1              | Y          | MagicNum  | -                              |
| 08      | AVX512       | Y       | Float32                    | Uint8                 | 1              | Y          | Directly  | -                              |

## Naming Convention

> **Example**: avx512_matmul_floatbitmanipu_nopack_float_uint8_no_magic
- **ISA**: avx512
- **Workload**: matmul
- **Approach**: Float Bit Manipulation
- **Is B Packed**: No
- **Type A**: Float
- **Type B**: Uint8
- **Can Handle Negatives**: Yes
- **Negatives Handling Method**: Directly (no magic number)
---
> **Example**: PoT4
- **Exponent Bits**: 3
- **Sign Bit**: 1
---
> **Example**: Uint16, NoPack, PoT9
This means that we have enough bits to have full 8 bit exponent.
- **Exponent Bits**: 8
- **Sign Bit**: 1
---
> **Example**: Uint8, NoPack, PoT8
Since we have 1 sign bit, we can only have a maximum of 7 exponent bits.
- **Exponent Bits**: 7
- **Sign Bit**: 1
---
> **Example**: Uint8, Packed2, PoT4
Two words of 4 bits (3 exponent bits and 1 sign bit) that are packed into a single 8 bit word.
- **Exponent Bits**: 3 * 2
- **Sign Bit**: 1 * 2
---
> **Example**: Uint8, Packed4, PoT2
Four words of 2 bits (1 exponent bit and 1 sign bit) that are packed into a single 8 bit word.
- **Exponent Bits**: 1 * 4
- **Sign Bit**: 1 * 4
---
# Rules

1. Each benchmark should:
    - Be a separate file in the `benchmarks` directory.
    - not depend on any other benchmark.
    - repeat the same operation multiple times.
    - have a readme explaining the benchmark.
    - have a unique benchmark ID.
    - have `main.cpp`, `scalar.cpp`, `vectorized.cpp`, and `defs.h`.
    - The function in `scalar.cpp` should be defined as `T FUNCTION_NAME(foo_bar) {}`.
2. No cmake. Just custom commands calling the compiler directly.
3. The compiler and its version should be printed in the benchmark output, by the bash script or by the program itself.
4. The compiler flags should be printed in the benchmark output, by the bash script or by the program itself.
5. The build scripts should be located in the script directory. Any benchmark should have a relative symlink to a build
   script.
6. The build scripts should also run the compiled program.

## Dumps Directory
The build scripts will create a text file at the dumps directory, named `bechId*.txt`. All the sub dump directories of
all the runs of the current bench ID will be appended into this text file. 

# How to replicate the results
1. For each benchmark, run the run-me bash script.
2. TODO