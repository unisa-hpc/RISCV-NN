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