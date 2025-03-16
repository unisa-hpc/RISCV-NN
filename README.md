[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Funisa-hpc%2FRISCV-NN&count_bg=%2379C83D&title_bg=%23555555&icon=postwoman.svg&icon_color=%23E7E7E7&title=Visits&edge_flat=false)](https://hits.seeyoufarm.com) 

# Demystifying Power-of-Two Quantization: Evaluating Inference on AVX, RVV, and CUDA

This repository contains the code for the paper "Demystifying Power-of-Two Quantization: Evaluating Inference on AVX,
RVV, and CUDA".

## Abstract

We present a comprehensive evaluation of power-of-two quantization for MatMul inference on modern hardware.
Some key findings include:

- Any PoT quantization that relies solely on SHIFT being faster than MUL is going to struggle gaining speedups.
- Floating-point PoT is effective and practical for inference applications on AVX512 and RVV-1.0.
- Getting speedups from floating-point PoT on CUDA is way more challenging.
- Floating-point PoT needs extra logic to handle the edge cases.

## What we provide

This repository contains benchmarks for MatMul inference on AVX512, RVV-1.0, and CUDA.
We provide PoT kernels, scalar, scalar autovectorized, and handwritten vectorized baseline kernels.
Moreover, the scripts for reproducing the results on AMD64, RISCV64, and NVIDIA GPUs are provided. The required SLURM
sbatch scripts are also included.

Finally, the Python recipes for training actual PoT quantized models with different methods and quantization
configurations are provided.

## Used HW/SW

### Hardware

| Kind          | #1            | #2               |
|---------------|---------------|------------------|
| AMD64 - Intel | Xeon5218      | Xeon8260         |
| AMD64 - AMD   | Ryzen 9 7950X | -                |
| RISCV64       | SpacemiT-K1   | -                |
| GPU - Nvidia  | V100S         | Jetson Orin Nano |

### Software

- GCC 13.3 and 14.2, LLVM 17 and 18
  - Spack for AMD64
  - Supplied by packages by Bianbu OS for BPi-F3
- Miniforge3 for misc. libraries and utilities
- CUDA


## Citation

Please use the following BibTeX entry to cite our work:

```
TBD.
```

