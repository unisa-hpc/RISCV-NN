# Benchmark

**ARCH**: `(CUDA)`

## Autotuning
### 1. Autotune
Run the autotuner script on a machine with a CUDA GPU. You can run it multiple times for various autotuning ranges, iteration and time limits, and optimization algorithms.
```bash
python autotune.py --help
python autotune.py --maxiter=10000 --reps=7 --time=7200
```
At the moment, you have to modify the script to change the algorithm and the ranges.

### 2. Plot the results
After running the autotuner, you will have multiple GPU sub directories within `benchmarks/dumps/` containing json files.
You need to feed the path of all these sub directories to the plotter script.
```bash
python autotuner_convergence.py --subdump="PATH1" --subdump="PATH2" --output="/tmp"
```
The results will be saved in the output directory as multiple SVG files.

## CMake-based Project
You can also build the kernels as a standalone project using CMake.
All you have to do is to open this directory as a CMake project in your favorite IDE or build it in a separate directory.
```bash
mkdir build
cd build
cmake ..
make
```
