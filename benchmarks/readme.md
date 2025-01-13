# Benchmarks

| BenchID | ISA/Platform | Matmul? | TypeA                      | TypeB                 | WordsPackedInB | CanHdlNeg? | NegMethod | Notes                          |
|---------|--------------|---------|----------------------------|-----------------------|----------------|------------|-----------|--------------------------------|
| 00      | AVX2         | N       | -                          | -                     | -              | -          | -         | Inline Assembly, MUL vs. Shift |
| 01      | RVV-1.0      | Y       | Int32                      | Int32                 | 1              | N          | -         |                                |
| 02      | AVX2         | Y       | Int32                      | Int32                 | 1              | N          | -         |                                |
| 03      | AVX2         | N       | -                          | -                     | -              | N          | -         | Conv2D, Int32, NoPack          |
| 04      | CUDA         | Y       | Float32, Float32 as Uint32 | Float32, Uint16 Uint8 | Mixed          | Y          | Directly  | -                              |
| 05      | RVV-1.0      | Y       | Float32                    | Uint8                 | 1              | Y          | MagicNum  | -                              |
| 06      | RVV-1.0      | Y       | Float32                    | Uint8                 | 2              | Y          | MagicNum  | -                              |
| 07      | AVX512       | Y       | Float32                    | Uint8                 | 1              | Y          | MagicNum  | -                              |
| 08      | AVX512       | Y       | Float32                    | Uint8                 | 1              | Y          | Directly  | -                              |


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