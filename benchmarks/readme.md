# Benchmarks

| BenchID | ISA/Platform | Type                           | Negative<br/>Handling | Notes |
|---------|--------------|--------------------------------|-----------------------|-------|
| 01      | RVV-1.0      | I32:U32:E31:P1                 | Y                     | FXPoT |
| 02      | AVX512       | I32:U32:E31:P1                 | N                     | FXPoT |
| 05      | RVV-1.0      | F32:U8:E5:P1                   | Y                     | FPoT  |
| 06      | RVV-1.0      | F32:U8:E3:P2                   | Y                     | FPoT  |
| 07      | AVX512       | F32:U8:E5:P1                   | Y                     | FPoT  |
| 08      | AVX512       | F32:U16:E8:P1                  | Y                     | FPoT  |
| 09      | AVX512       | F32:U8:E3:P2<br/> F32:U8:E1:P4 | Y                     | FPoT  |
| 10      | AVX512       | F32:U8:E5:P1 + InfNan          | Y                     | FPoT  |

## How to use

### Dependencies

To automatically install the required dependencies on any AMD64 linux machine, run:

```bash
cd scripts
bash deps.amd64.sh
```

Make sure you have enough disk space at `~`. Be patient, it will take a while.
If for any reason the script fails, you need to manually delete `~/riscvnn_rootdir` and try again.
The script has some options to skip Spack or Conda steps, you can configure them in the code.

### Setting the CPU autotuning ranges

Modify the file at:

```bash
benchmarks/common/ranges.matmul.sh
```

Don't run it. Just save your changes. This only affects the CPU (AMD64, RISCV64) benchmarks.

### Running the AMD64 benchmarks

Before running the benchmarks, you need to activate the environment:

```bash
source ~/riscvnn_rootdir/activate_env.sh
```

Then, you can start the script to run all the benchmarks in the same terminal:

```bash
bash benchmarks/amd64/run_all.sh <MachineName>
```

Use `-h` to see the available options.

### Running the RISCV64 benchmarks

Since Conda is not available for RISCV64, you only need to make sure you have the required packages for
`GCC 14.2, LLVM 17, and LLVM 18`.
Then, you can start the script to run all the benchmarks with:

```bash
bash benchmarks/riscv/run_all.sh <MachineName>
```

### Running the NVIDIA benchmarks

You can either use the CMake project to evaluate the code manually or use the provided Python script for autotuning with
`kernel_tuner`.
To initiate the autotuning process, run:

```bash
cd gpu_cuda/9
python autotuner.py --maxiter=50000 --time=7200 --reps 7 <CUDA CAPABILITY>
```

The script will use `nvrtc` to compile the kernels. Make sure to use the correct CUDA capability for your GPU.
Like the AMD64 benchmarks, the raw data is stored in `benchmarks/dumps` directory.

### Plotting the GPU results

Use the provided Python script to generate the plots for the GPU benchmarks:

```bash
python benchmarks/gpu_cuda/9/autotuner_convergence.py --subdump=benchmarks/dumps/<THE CUDA RUN SUBDUMP DIR>
```

The generated plots are stored in `/tmp`.

### Gathering Raw Data

All the profiled data is stored in `benchmarks/dumps` directory. You can back them up manually if needed.

### How to reproduce the plots

The Python scripts to automatically parse the JSON files from multiple `dumps` directories and generate the plots are
provided in `benchmarks/common/python/plot.speedups.py` directory.
This script accepts multiple instances of `--dumps=<path>` arguments and is able to concatenate bench-runs from multiple
compilers and machines autonomously.
Furthermore, you can request caching the parse data to save time on subsequent runs with `--s-from=<path>` and
`--s-to=<path>` arguments. If the provided `--s-from` path does not exist, the script will fall back to parsing the data
from scratch. If `--s-to` is provided, the script will cache the parsed data to the specified path before terminating.
All the generated plots are stored at `/tmp`.

### SLURM?

Yes, you can run the benchmarks on a SLURM cluster as well.
You can find the SLURM sbatch file at `benchmarks/amd64/amd64/run_all_amd64.sbatch`.
Depending on you permissions, you might have a 24 hours limit for SLURM jobs. In that case, you have to modify
`benchamrks/amd64/run_all.sh` to split the benchmarks into smaller chunks manually.
The sbatch file will request a single CPU node reserving all the CPU cores for a socket to prevent other processes using
the cache. Therefore, depending on how many cores your cluster has per CPU, you need to modify the sbatch file. The
script is configured for Xeon8260. Don't forget to modify job name, partition, and account fields.

# Something broke?
Feel free to open an issue on GitHub, we would be happy to help you out.



