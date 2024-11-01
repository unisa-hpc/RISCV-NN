# Rules
1. Each benchmark should:
    - Be a separate file in the `benchmarks` directory.
    - not depend on any other benchmark.
    - not depend on any other files or libraries.
    - repeat the same operation multiple times.
    - have a readme explaining the benchmark.
    - have a unique benchmark ID.
    - be named as its benchmark ID.
2. No cmake. Just custom commands calling the compiler directly.
3. The compiler and its version should be printed in the benchmark output, by the bash script or by the program itself.
4. The compiler flags should be printed in the benchmark output, by the bash script or by the program itself.
5. The build scripts should be located in the script directory. Any benchmark should have a relative symlink to a build script.
6. The build scripts should also run the compiled program.
