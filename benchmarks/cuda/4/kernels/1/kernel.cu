#include "../../common_cuda.h"
#include "kernel.h"
#include <cassert>
#include <cuda/std/detail/libcxx/include/cstdint>

// MxK * KxN = MxN
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// device function to print uint32_t as binary
__device__
void print_uint32_as_binary(const char *msg, uint32_t x) {
    printf("%s: ", msg);
    for (int i = 31; i >= 0; i--) {
        printf("%d", (x >> i) & 1);
    }
    printf("\n");
}

__global__ void KernelMatMul01(
    size_t matrix_size,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t M = matrix_size;
    size_t K = matrix_size;

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = tmp;
    }
}

__global__ void KernelMatMul01_PoT(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t M = matrix_size;
    size_t K = matrix_size;

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            uint32_t bits_b = B[k * N + y];

            // float32: 1 bit sign, 8 bit exponent, 23 bit mantissa
            // high half of float32: 1 bit sign, 8 bit exponent, 7 bit mantissa
            bits_b = (bits_b&0x7F /*make the misplaced sign bit zero*/) | (bits_b>>7) << 8 /*move the sign bit to bit index 8*/;
            bits_b = bits_b << 23;
            uint32_t pot_mul = A[x * K + k] + bits_b;
            tmp += __int_as_float(pot_mul);
        }
        // C =A@B
        C[x * N + y] = tmp;
    }
}

__global__ void KernelMatMul01_PoT16(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t M = matrix_size;
    size_t K = matrix_size;

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            uint32_t bits_b = B[k * N + y];
            bits_b = bits_b << 16;
            uint32_t pot_mul = A[x * K + k] + bits_b;

            // only for the first thread of the first block
            //if (x == 0 && y == 0 && k == 0) {
            //    print_uint32_as_binary("A[x * K + k]", A[x * K + k]);
            //    print_uint32_as_binary("bits_b", bits_b);
            //    printf("pot_mul: %f\n", __int_as_float(pot_mul));
            //}

            tmp += __int_as_float(pot_mul);
        }
        // C =A@B
        C[x * N + y] = tmp;
    }
}

__global__ void KernelMatMul01_PoT4bit(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t M = matrix_size;
    size_t K = matrix_size;

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; // should be mul'ed by 2, each thread processes 2 out elements

    auto unpack_low_4bits_to_uint32 = [](uint8_t x/*, bool dbg*/) -> uint32_t {
        uint32_t res;
        res = (x & 0x07) | ((x>>3 & 1) << 8); // float32: 1 bit sign, 8 bit exponent, 23 bit mantissa
        //if (dbg) {
        //    printf("Input: %d\n", x);
        //    print_uint32_as_binary("res_not_shifted_l23", res);
        //    print_uint32_as_binary("res", res<<23);
        //}
        return res << 23;
    };

    // if statement is necessary to make things work under tile quantization
    if (x < M && 2*y < N) {
        float tmp0 = 0.0;
        float tmp1 = 0.0;
        for (int k = 0; k < K; ++k) {
            uint8_t bits_packed = B[k * N/2 + y];
            tmp0 += __uint_as_float(A[x * K + k] + unpack_low_4bits_to_uint32(bits_packed/*, x==0 && y==0 & k==0*/));
            tmp1 += __uint_as_float(A[x * K + k] + unpack_low_4bits_to_uint32(bits_packed>>4/*, x==0 && y==0 & k==0*/));
        }
        // C =A@B
        C[x * N + 2*y + 0] = tmp0;
        C[x * N + 2*y + 1] = tmp1;
    }
}

// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh

/**
 * A(MxK) * B(KxN) = C(MxN)
 * @tparam BM Block size in A's axis 0
 * @tparam BN Block size in B's axis 1
 * @tparam BK Block size in A and B's common axis
 * @tparam TM Tile size in A's axis 0 within a block. See https://siboehm.com/assets/img/CUDA-MMM/kernel_4_1D_blocktiling.png
 * @param matrix_size
 * @param A
 * @param B
 * @param C
 */
template <const int BM, const int BN, const int BK, const int TM>
__global__ void KernelMatMul04(
    size_t matrix_size,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t K = matrix_size;

    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh
template <const int BM, const int BN, const int BK, const int TM>
__global__ void KernelMatMul04_PoT(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t K = matrix_size;

    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ uint32_t As[BM * BK];
    __shared__ uint8_t Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            uint32_t bits_b = Bs[dotIdx * BN + threadCol];
            bits_b = (bits_b&0x7F /*make the misplaced sign bit zero*/) | ((bits_b>>7) << 8) /*move the sign bit to bit index 8*/;
            bits_b = bits_b << 23;

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {

                uint32_t pot_mul = As[(threadRow * TM + resIdx) * BK + dotIdx] + bits_b;

                //print_uint32_as_binary("bits_b", bits_b);
                //print_uint32_as_binary("As[(threadRow * TM + resIdx) * BK + dotIdx]", As[(threadRow * TM + resIdx) * BK + dotIdx]);
                //printf("pot_mul: %f\n", __int_as_float(pot_mul));

                threadResults[resIdx] += __uint_as_float(pot_mul);
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh
// This version stores B in the shared memory in a packed format, and unpacks it when used.
template <const int BM, const int BN, const int BK, const int TM>
__global__ void KernelMatMul04_PoT4bit(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t K = matrix_size;

    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Each thread now handles 2 consecutive columns
    const int threadCol = threadIdx.x % (BN/2);  // Adjusted for packed values
    const int threadRow = threadIdx.x / (BN/2);

    __shared__ uint32_t As[BM * BK];
    __shared__ uint8_t Bs[BK * BN/2];  // packed vals

    // Adjust pointers - note B and C are now indexed differently
    A += cRow * BM * K;
    B += cCol * (BN/2);  // Each thread block handles BN/2 packed values
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN/2 * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % (BN/2);  // For packed values
    const uint innerRowB = threadIdx.x / (BN/2);

    // Double the thread results since each thread handles 2 columns
    float threadResults[TM][2] = {{0.0f}};  // [TM rows][2 columns]

    auto unpack_low_4bits_to_uint32 = [](uint8_t x/*, bool dbg*/) -> uint32_t {
        uint32_t res;
        res = (x & 0x07) | ((x>>3 & 1) << 8); // float32: 1 bit sign, 8 bit exponent, 23 bit mantissa
        //if (dbg) {
        //    printf("Input: %d\n", x);
        //    print_uint32_as_binary("res_not_shifted_l23", res);
        //    print_uint32_as_binary("res", res<<23);
        //}
        return res << 23;
    };

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * (BN/2) + innerColB] = B[innerRowB * (N/2) + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * (N/2);

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.

            // Each word from Bs (smem) is two 4 bit elements packed into a byte.
            uint32_t b_bits_lo = unpack_low_4bits_to_uint32(Bs[dotIdx * BN/2 + threadCol]);
            uint32_t b_bits_hi = unpack_low_4bits_to_uint32(Bs[dotIdx * BN/2 + threadCol]>>4);

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx][0] += __uint_as_float(As[(threadRow / 2 * TM + resIdx) * BK + dotIdx] + b_bits_lo);
                threadResults[resIdx][1] += __uint_as_float(As[(threadRow / 2 * TM + resIdx) * BK + dotIdx] + b_bits_hi);
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow/2 * TM + resIdx) * N + threadCol*2 + 0] = threadResults[resIdx][0];
        C[(threadRow/2 * TM + resIdx) * N + threadCol*2 + 1] = threadResults[resIdx][1];
    }
}


// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh
// This version stores B in the shared memory in a packed format, and unpacks it when used.
template <const int BM, const int BN, const int BK, const int TM>
__global__ void KernelMatMul04_PoT2bit(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {
    size_t N = matrix_size;
    size_t K = matrix_size;

    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Each thread now handles 2 consecutive columns
    const int threadCol = threadIdx.x % (BN/4);  // Adjusted for packed values
    const int threadRow = threadIdx.x / (BN/4);

    __shared__ uint32_t As[BM * BK];
    __shared__ uint8_t Bs[BK * BN/4];  // packed vals

    // Adjust pointers - note B and C are now indexed differently
    A += cRow * BM * K;
    B += cCol * (BN/4);  // Each thread block handles BN/4 packed values
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN/4 * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % (BN/4);  // For packed values
    const uint innerRowB = threadIdx.x / (BN/4);

    float threadResults[TM][4] = {{0.0f}};

    auto unpack_low_2bits_to_uint32 = [](uint8_t x/*, bool dbg*/) -> uint32_t {
        // word 0: bits 0-1, word 1: bits 2-3, word 2: bits 4-5, word 3: bits 6-7
        uint32_t res;
        res = (x & 0x01) | ((x>>1 & 1) << 8); // float32: 1 bit sign, 8 bit exponent, 23 bit mantissa
        //if (dbg) {
        //    printf("Input: %d\n", x);
        //    print_uint32_as_binary("res_not_shifted_l23", res);
        //    print_uint32_as_binary("res", res<<23);
        //}
        return res << 23;
    };

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * (BN/4) + innerColB] = B[innerRowB * (N/4) + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * (N/4);

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.

            // Each word from Bs (smem) is two 4 bit elements packed into a byte.
            uint32_t b_bits_w0 = unpack_low_2bits_to_uint32(Bs[dotIdx * BN/4 + threadCol]);
            uint32_t b_bits_w1 = unpack_low_2bits_to_uint32(Bs[dotIdx * BN/4 + threadCol]>>2);
            uint32_t b_bits_w2 = unpack_low_2bits_to_uint32(Bs[dotIdx * BN/4 + threadCol]>>4);
            uint32_t b_bits_w3 = unpack_low_2bits_to_uint32(Bs[dotIdx * BN/4 + threadCol]>>6);

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx][0] += __uint_as_float(As[(threadRow / 4 * TM + resIdx) * BK + dotIdx] + b_bits_w0);
                threadResults[resIdx][1] += __uint_as_float(As[(threadRow / 4 * TM + resIdx) * BK + dotIdx] + b_bits_w1);
                threadResults[resIdx][2] += __uint_as_float(As[(threadRow / 4 * TM + resIdx) * BK + dotIdx] + b_bits_w2);
                threadResults[resIdx][3] += __uint_as_float(As[(threadRow / 4 * TM + resIdx) * BK + dotIdx] + b_bits_w3);
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow/4 * TM + resIdx) * N + threadCol*4 + 0] = threadResults[resIdx][0];
        C[(threadRow/4 * TM + resIdx) * N + threadCol*4 + 1] = threadResults[resIdx][1];
        C[(threadRow/4 * TM + resIdx) * N + threadCol*4 + 2] = threadResults[resIdx][2];
        C[(threadRow/4 * TM + resIdx) * N + threadCol*4 + 3] = threadResults[resIdx][3];
    }
}


// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/8_kernel_bank_extra_col.cuh
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void KernelMatMul08(
    size_t matrix_size,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    size_t N = matrix_size;
    size_t K = matrix_size;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    const int extraCols = 5;
    __shared__ float Bs[BK * (BN + extraCols)];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        tmp = reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
        __syncthreads();

        // advance blocktile
        A += BK; // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // load C vector into registers
            float4 tmp;
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            // write
            reinterpret_cast<float4*>(
                    &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}

// https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/1_naive.cuh
void LaunchKernel01(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC) {
    dim3 gridDim(CEIL_DIV(matrix_size, 32), CEIL_DIV(matrix_size, 32));
    dim3 blockDim(32, 32);

    KernelMatMul01<<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel01_PoT(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    dim3 gridDim(CEIL_DIV(matrix_size, 32), CEIL_DIV(matrix_size, 32));
    dim3 blockDim(32, 32);

    KernelMatMul01_PoT<<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel01_PoT16(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint16_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    dim3 gridDim(CEIL_DIV(matrix_size, 32), CEIL_DIV(matrix_size, 32));
    dim3 blockDim(32, 32);

    KernelMatMul01_PoT16<<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel01_PoT4bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    dim3 gridDim(CEIL_DIV(matrix_size, 32), CEIL_DIV(matrix_size/2, 32)); // each thread processes 2 out elements
    dim3 blockDim(32, 32);

    KernelMatMul01_PoT4bit<<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel04(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(matrix_size, BN), CEIL_DIV(matrix_size, BM));
    dim3 blockDim((BM * BN) / TM);

    KernelMatMul04<BM, BN, BK, TM> <<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel04_PoT(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(matrix_size, BN), CEIL_DIV(matrix_size, BM));
    dim3 blockDim((BM * BN) / TM);

    KernelMatMul04_PoT<BM, BN, BK, TM> <<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );
}

void LaunchKernel04_PoT4bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(matrix_size, BN), CEIL_DIV(matrix_size, BM));
    dim3 blockDim((BM * BN) / TM/2);

    KernelMatMul04_PoT4bit<BM, BN, BK, TM> <<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );

}

void LaunchKernel04_PoT2bits(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(matrix_size, BN), CEIL_DIV(matrix_size, BM));
    dim3 blockDim((BM * BN) / TM/4);

    KernelMatMul04_PoT2bit<BM, BN, BK, TM> <<< gridDim, blockDim, 0, stream>>>(
        matrix_size,
        tnA,
        tnB,
        tnC
    );

}

void LaunchKernel08(
    cudaStream_t& stream,
    size_t matrix_size,
    const float* __restrict__ tnA,
    const float* __restrict__ tnB,
    float* __restrict__ tnC) {

    const size_t M = matrix_size;
    const size_t N = matrix_size;
    const size_t K = matrix_size;

    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    if (M >= 128 and N >= 128) {
        const uint BM = 128;
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        KernelMatMul08<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    } else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        KernelMatMul08<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
}
