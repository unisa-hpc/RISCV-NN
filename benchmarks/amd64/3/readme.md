# Benchmark
**ARCH**: `AMD64`

AVX2 direct Conv2D with row-major OCHW layout with **VALID** padding **only**. **NO** inline assembly is used.

## Sub Benchmarks
### `conv2d_direct_padding_ochw_scalar_noautovec`
Scalar kernel with MUL instructions, not auto-vectorized.
### `conv2d_direct_padding_ochw_scalar_autovec`
Scalar kernel with MUL instructions, auto-vectorized.
### `conv2d_direct_padding_ochw_avx_try18`
AVX2 kernel with MUL instructions, with intrinsics.
### shift
TODO.

## Notes
### Effective Kernels
It seems that iterating over the output tensor's domain yields the better performance compared to iterating over the input tensor's domain.

### Can we do better?
Yes. 
- Prefetching the input tensor's data can improve the performance.
- Tiling in both dimensions. Right now only one AVX vector gets computed at a time in LS axis.

### Unrolling Factors
- `UNROLL_FACTOR0`: The loop for channels out.
- `UNROLL_FACTOR1`: The loop for channels in.
- `UNROLL_FACTOR2`: The loop for kernel height.
- `UNROLL_FACTOR3`: The loop for kernel width.

### Parameters
- `I_H`: Input height.
- `I_W`: Input width.
- `K_H`: Kernel height.
- `K_W`: Kernel width.
- `C_I`: Channels in.
- `C_O`: Channels out.
- `S_X`: Stride.
- `S_Y`: Stride.