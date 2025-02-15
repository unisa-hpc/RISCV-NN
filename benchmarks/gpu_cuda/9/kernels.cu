#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void KernelMatmulBase(
    size_t matrix_size,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    constexpr uint UF0 = vUF0;
    constexpr uint UF1 = vUF1;
    constexpr uint UF2 = vUF2;
    constexpr uint UF3 = vUF3;
    constexpr uint UF4 = vUF4;
    constexpr uint UF5 = vUF5;
    constexpr uint UF6 = vUF6;
    constexpr uint UF7 = vUF7;
    constexpr uint UF8 = vUF8;
    constexpr uint UF9 = vUF9;

    const size_t N = matrix_size;
    const size_t K = matrix_size;

    const uint tid_x = threadIdx.x % ((BN)/TN);
    const uint tid_y = threadIdx.x / ((BN)/TN);

    // Allocate space for the current blocktile in smem
    __shared__ float As[BK * BM];  // Note: dimensions swapped for col-major
    constexpr int extraCols = 5;
    __shared__ float Bs[BK * (BN + extraCols)];

    float threadResults[TM * (TN)] = {0.0f};
    float regM[TM] = {0};
    float regN[TN] = {0};

#pragma unroll UF0
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load row-major A into col-major As
#pragma unroll UF1
        for (uint tid = threadIdx.x; tid < BM * (BK/4); tid += blockDim.x) {
            const uint col = tid / BM;        // Which column group (each group is 4 elements)
            const uint row = tid % BM;        // Which row in the block

            // Load 4 consecutive elements in the row
            float4 tmp = reinterpret_cast<const float4*>(&A[(blockIdx.y*BM + row) * K + bkIdx + col*4])[0];

            // Store in col-major order, 4 elements at a time
            As[(col*4) * BM + row] = tmp.x;
            As[(col*4 + 1) * BM + row] = tmp.y;
            As[(col*4 + 2) * BM + row] = tmp.z;
            As[(col*4 + 3) * BM + row] = tmp.w;
        }

#pragma unroll UF2
        for (uint tid=threadIdx.x; tid < BK * (BN); tid += blockDim.x) {
            const uint _tid_x = tid % (BN);
            const uint _tid_y = tid / (BN);
            const float tmp = B[(bkIdx+_tid_y)*(N) + blockIdx.x*(BN) + _tid_x];
            Bs[(_tid_y)*(BN + extraCols) + _tid_x] = tmp;
        }
        __syncthreads();
#pragma unroll UF3
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll UF4
            for (uint i = 0; i < TM; ++i) {
                // Modified to access As in col-major layout
                regM[i] = As[dotIdx * BM + tid_y*TM + i];
            }
#pragma unroll UF5
            for (uint i = 0; i < TN; ++i) {
                auto val = Bs[(dotIdx)*(BN+extraCols) + (tid_x*TN+i)];
                regN[i] = val;
            }
#pragma unroll UF6
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll UF7
                for (uint resIdxN = 0; resIdxN < (TN); ++resIdxN) {
                    threadResults[resIdxM * (TN) + resIdxN] += regM[resIdxM] * regN[resIdxN];
                    //printf("regN[%d]: %d\n", resIdxN, regN[resIdxN]);
                    //printf("threadResults[%d]: %f\n", resIdxM * (TN*2) + resIdxN, threadResults[resIdxM * (TN*2) + resIdxN]);
                }
            }
        }
        __syncthreads();
    }

    /*
    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (uint resIdxN = 0; resIdxN < (TN*2); resIdxN++) {
            const auto val = threadResults[resIdxM * (TN*2) + resIdxN];
            const size_t idx = (blockIdx.y * BM + tid_y*TM+resIdxM)*N + (blockIdx.x * BN + tid_x*(TN*2)+resIdxN);
            C[idx] = val;
            if (idx < 512) {
                //printf("threadResults offset: %u, blockIdx.y: %d, blockIdx.x: %d, tid_y: %d, tid_x: %d, resIdxM: %d, resIdxN: %d, baseIdx: %lu\n",resIdxM * TN + resIdxN, blockIdx.y, blockIdx.x, tid_y, tid_x, resIdxM, resIdxN, idx);
            }
            // only for the first thread
            //if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && resIdxM==0) {
            //    printf("threadResults offset: %u, idx:%lu\n", resIdxM * TN*2 + resIdxN, idx);
            //}

        }
    }
    */
#pragma unroll UF8
    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
#pragma unroll UF9
        for (uint resIdxN = 0; resIdxN < (TN/4); resIdxN++) {
            float4 val;
            // Pack 4 consecutive results into float4
            val.x = threadResults[resIdxM * (TN) + resIdxN*4];
            val.y = threadResults[resIdxM * (TN) + resIdxN*4 + 1];
            val.z = threadResults[resIdxM * (TN) + resIdxN*4 + 2];
            val.w = threadResults[resIdxM * (TN) + resIdxN*4 + 3];

            // Calculate base index for the group of 4 elements
            const size_t baseIdx = (blockIdx.y * BM + tid_y*TM + resIdxM)*N +
                                 (blockIdx.x * BN + tid_x*(TN) + resIdxN*4);
            //if (baseIdx < 128) {
            //    printf("threadResults offset: %u, blockIdx.y: %d, blockIdx.x: %d, tid_y: %d, tid_x: %d, resIdxM: %d, resIdxN: %d, baseIdx: %lu\n",resIdxM * TN + resIdxN*4, blockIdx.y, blockIdx.x, tid_y, tid_x, resIdxM, resIdxN, baseIdx);
            //}

            // Store four elements at once
            reinterpret_cast<float4*>(&C[baseIdx])[0] = val;

            // Verify results
            //if (val.x != Gold[baseIdx] ||
            //    val.y != Gold[baseIdx + 1] ||
            //    val.z != Gold[baseIdx + 2] ||
            //    val.w != Gold[baseIdx + 3]) {
            //    printf("Mismatch at (%lu): (%f,%f,%f,%f) != (%f,%f,%f,%f)\n",
            //           baseIdx,
            //           val.x, val.y, val.z, val.w,
            //           Gold[baseIdx], Gold[baseIdx + 1],
            //           Gold[baseIdx + 2], Gold[baseIdx + 3]);
            //}

        }
    }

}

#ifndef ONLY_KERNELS
void LaunchKernelMatmulBase(
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
        KernelMatmulBase<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
    else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        KernelMatmulBase<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
}
#endif
// =================================================================================================
// =================================================================================================

__constant__ uint32_t lut_pot_uint8_packed2[16] = {
    (0 << 23), (1 << 23), (2 << 23), (3 << 23),
    (4 << 23), (5 << 23), (6 << 23), (7 << 23),
    (256 << 23), ((256 + 1) << 23), ((256 + 2) << 23), ((256 + 3) << 23),
    ((256 + 4) << 23), ((256 + 5) << 23), ((256 + 6) << 23), ((256 + 7) << 23)
};

template <int BM, int BN, int BK, int TM, int TN>
__global__ void KernelMatmulPotUint8Packed2(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {

    constexpr uint UF0 = vUF0;
    constexpr uint UF1 = vUF1;
    constexpr uint UF2 = vUF2;
    constexpr uint UF3 = vUF3;
    constexpr uint UF4 = vUF4;
    constexpr uint UF5 = vUF5;
    constexpr uint UF6 = vUF6;
    constexpr uint UF7 = vUF7;
    constexpr uint UF8 = vUF8;
    constexpr uint UF9 = vUF9;

    const size_t N = matrix_size;
    const size_t K = matrix_size;

    // Map threadIdx.x into a 2D (tid_y, tid_x) layout
    const uint tid_x = threadIdx.x % ((BN/2) / TN);
    const uint tid_y = threadIdx.x / ((BN/2) / TN);

    // Shared memory tiles for A (stored in col–major order) and B.
    __shared__ uint32_t As[BK * BM];
    constexpr int extraCols = 5;
    __shared__ uint8_t Bs[BK * (BN/2 + extraCols)];

    // Per–thread accumulation registers.
    float threadResults[TM * (TN*2)] = {0.0f};
    uint32_t regM[TM];
    uint8_t  regN[TN*2];

    // Loop over tiles along the K dimension.
#pragma unroll UF0
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // *** Load A into shared memory ***
        // Each thread loads one (or more) 4–element group from A.
#pragma unroll UF1
        for (uint tid = threadIdx.x; tid < BM * (BK/4); tid += blockDim.x) {
            const uint col = tid / BM;  // group index (each group = 4 elements)
            const uint row = tid % BM;  // row index within the block

            // Load 4 consecutive elements as a vector.
            uint4 tmp = reinterpret_cast<const uint4*>(
                &A[(blockIdx.y * BM + row) * K + bkIdx + col * 4]
            )[0];

            // Store in shared memory in col–major order.
            As[(col * 4 + 0) * BM + row] = tmp.x;
            As[(col * 4 + 1) * BM + row] = tmp.y;
            As[(col * 4 + 2) * BM + row] = tmp.z;
            As[(col * 4 + 3) * BM + row] = tmp.w;
        }

        // *** Load B into shared memory ***
#pragma unroll UF2
        for (uint tid = threadIdx.x; tid < BK * (BN/2); tid += blockDim.x) {
            const uint _tid_x = tid % (BN/2);
            const uint _tid_y = tid / (BN/2);
            Bs[_tid_y * (BN/2 + extraCols) + _tid_x] =
                B[(bkIdx + _tid_y) * (N/2) + blockIdx.x * (BN/2) + _tid_x];
        }
        __syncthreads();

        // Process the BK–length inner dimension.
#pragma unroll UF3
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Load a block of A from shared memory into registers.
#pragma unroll UF4
            for (int i = 0; i < TM; i++) {
                regM[i] = As[dotIdx * BM + tid_y * TM + i];
            }

            // Load a block of B from shared memory, splitting each byte into two 4–bit values.
#pragma unroll UF5
            for (int i = 0; i < TN; i++) {
                uint8_t packedVal = Bs[dotIdx * (BN/2 + extraCols) + (tid_x * TN + i)];
                regN[2 * i + 0] = packedVal & 0x0F;
                regN[2 * i + 1] = packedVal >> 4;
            }

            // Accumulate the “dot–product” result.
#pragma unroll UF6
            for (int m = 0; m < TM; m++) {
#pragma unroll UF7
                for (int n = 0; n < (TN * 2); n++) {
                    // Instead of computing the bit–tweaked value,
                    // use the precomputed constant table.
                    uint32_t offset = lut_pot_uint8_packed2[regN[n]];
                    threadResults[m * (TN * 2) + n] +=
                        __uint_as_float(regM[m] + offset);
                }
            }
        }
        __syncthreads();
    }

    // *** Write the accumulated results back to global memory ***
#pragma unroll UF8
    for (int m = 0; m < TM; m++) {
#pragma unroll UF9
        for (int n = 0; n < 2 * (TN / 4); n++) {
            float4 val;
            int base = m * (2 * TN) + n * 4;
            val.x = threadResults[base + 0];
            val.y = threadResults[base + 1];
            val.z = threadResults[base + 2];
            val.w = threadResults[base + 3];

            const size_t idx = (blockIdx.y * BM + tid_y * TM + m) * N +
                               (blockIdx.x * BN + tid_x * (2 * TN) + n * 4);
            reinterpret_cast<float4*>(&C[idx])[0] = val;
        }
    }
}

#ifndef ONLY_KERNELS
void LaunchKernelMatmulPotUint8Packed2(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
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
        dim3 blockDim((BM * BN) / (TM * TN) / 2);
        std::cout << "Launching KernelMatMul08_PoT4bits with BM=128, BN=128" << std::endl;
        std::cout << "gridDim: " << gridDim.x << " " << gridDim.y << std::endl;
        std::cout << "blockDim: " << blockDim.x << std::endl;
        //KernelMatMul08_PoT4bits_As_Colmajor_vecloadstore_fixing_zero_vecstore<BM, BN, BK, TM, TN>
        KernelMatmulPotUint8Packed2<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
    else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN) / 2);
        std::cout << "Launching KernelMatMul08_PoT4bits with BM=128, BN=128" << std::endl;
        std::cout << "gridDim: " << gridDim.x << " " << gridDim.y << std::endl;
        std::cout << "blockDim: " << blockDim.x << std::endl;
        //KernelMatMul08_PoT4bits_As_Colmajor_vecloadstore_fixing_zero_vecstore<BM, BN, BK, TM, TN>
        KernelMatmulPotUint8Packed2<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
}
#endif
// =================================================================================================
// =================================================================================================

// Lookup table for 2-bit values (4 possible values)
__constant__ uint32_t lut_pot_uint8_packed4[4] = {
    (0 << 23),         // 00
    (1 << 23),         // 01
    (256 << 23),       // 10
    ((256 + 1) << 23)  // 11
};

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void KernelMatmulPotUint8Packed4(
    size_t matrix_size,
    const uint32_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C) {

    constexpr uint UF0 = vUF0;
    constexpr uint UF1 = vUF1;
    constexpr uint UF2 = vUF2;
    constexpr uint UF3 = vUF3;
    constexpr uint UF4 = vUF4;
    constexpr uint UF5 = vUF5;
    constexpr uint UF6 = vUF6;
    constexpr uint UF7 = vUF7;
    constexpr uint UF8 = vUF8;
    constexpr uint UF9 = vUF9;

    const size_t N = matrix_size;
    const size_t K = matrix_size;

    const uint tid_x = threadIdx.x % ((BN/4)/TN);
    const uint tid_y = threadIdx.x / ((BN/4)/TN);

    __shared__ uint32_t As[BK * BM];  
    constexpr int extraCols = 5;
    __shared__ uint8_t Bs[BK * (BN/4 + extraCols)];

    float threadResults[TM * (TN*4)] = {0.0f};
    uint32_t regM[TM] = {0};
    uint8_t regN[TN*4] = {0};

#pragma unroll UF0
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load row-major A into col-major As
#pragma unroll UF1
        for (uint tid = threadIdx.x; tid < BM * (BK/4); tid += blockDim.x) {
            const uint col = tid / BM;        // Which column group (each group is 4 elements)
            const uint row = tid % BM;        // Which row in the block

            // Load 4 consecutive elements in the row
            uint4 tmp = reinterpret_cast<const uint4*>(&A[(blockIdx.y*BM + row) * K + bkIdx + col*4])[0];

            // Store in col-major order, 4 elements at a time
            As[(col*4) * BM + row] = tmp.x;
            As[(col*4 + 1) * BM + row] = tmp.y;
            As[(col*4 + 2) * BM + row] = tmp.z;
            As[(col*4 + 3) * BM + row] = tmp.w;
        }

#pragma unroll UF2
        for (uint tid=threadIdx.x; tid < BK * (BN/4); tid += blockDim.x) {
            const uint _tid_x = tid % (BN/4);
            const uint _tid_y = tid / (BN/4);
            const uint8_t tmp = B[(bkIdx+_tid_y)*(N/4) + blockIdx.x*(BN/4) + _tid_x];
            Bs[(_tid_y)*(BN/4 + extraCols) + _tid_x] = tmp;
        }
        __syncthreads();

#pragma unroll UF3
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll UF4
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + tid_y*TM + i];
            }

#pragma unroll UF5
            for (uint i = 0; i < TN; ++i) {
                auto val = Bs[(dotIdx)*(BN/4+extraCols) + (tid_x*TN+i)];
                // Extract 2-bit values and store them separately
                regN[4*i+0] = val & 0x03;
                regN[4*i+1] = (val >> 2) & 0x03;
                regN[4*i+2] = (val >> 4) & 0x03;
                regN[4*i+3] = (val >> 6) & 0x03;
            }

#pragma unroll UF6
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll UF7
                for (uint resIdxN = 0; resIdxN < (TN*4); ++resIdxN) {
                    // Use the lookup table instead of bit manipulation
                    uint32_t offset = lut_pot_uint8_packed4[regN[resIdxN]];
                    threadResults[resIdxM * (TN*4) + resIdxN] +=
                        __uint_as_float(regM[resIdxM] + offset);
                }
            }
        }
        __syncthreads();
    }

    // Write results using float4 for coalesced memory access
#pragma unroll UF8
    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
#pragma unroll UF9
        for (uint resIdxN = 0; resIdxN < 4*(TN/4); resIdxN++) {
            float4 val;
            // Pack 4 consecutive results into float4
            val.x = threadResults[resIdxM * (4*TN) + resIdxN*4];
            val.y = threadResults[resIdxM * (4*TN) + resIdxN*4 + 1];
            val.z = threadResults[resIdxM * (4*TN) + resIdxN*4 + 2];
            val.w = threadResults[resIdxM * (4*TN) + resIdxN*4 + 3];

            // Calculate base index for the group of 4 elements
            const size_t baseIdx = (blockIdx.y * BM + tid_y*TM + resIdxM)*N +
                                 (blockIdx.x * BN + tid_x*(4*TN) + resIdxN*4);

            // Store four elements at once
            reinterpret_cast<float4*>(&C[baseIdx])[0] = val;
        }
    }
}

#ifndef ONLY_KERNELS
void LaunchKernelMatmulPotUint8Packed4(
    cudaStream_t& stream,
    size_t matrix_size,
    const uint32_t* __restrict__ tnA,
    const uint8_t* __restrict__ tnB,
    float* __restrict__ tnC) {
    const size_t M = matrix_size;
    const size_t N = matrix_size;
    const size_t K = matrix_size;

    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 4; // when this is 8, we end up with a compiled kernel with too many registers because of regN.
    if (M >= 128 and N >= 128) {
        const uint BM = 128;
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN) / 4);
        std::cout << "Launching KernelMatMul08_PoT2bits with BM=128, BN=128" << std::endl;
        std::cout << "gridDim: " << gridDim.x << " " << gridDim.y << std::endl;
        std::cout << "blockDim: " << blockDim.x << std::endl;
        //KernelMatMul08_PoT2bits_fixed<BM, BN, BK, TM, TN>
        KernelMatmulPotUint8Packed4<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
    else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN) / 4);
        std::cout << "Launching KernelMatMul08_PoT42its with BM=128, BN=128" << std::endl;
        std::cout << "gridDim: " << gridDim.x << " " << gridDim.y << std::endl;
        std::cout << "blockDim: " << blockDim.x << std::endl;
        //KernelMatMul08_PoT2bits_fixed<BM, BN, BK, TM, TN>
        KernelMatmulPotUint8Packed4<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(matrix_size, tnA, tnB, tnC);
    }
}
#endif
// =================================================================================================
// =================================================================================================




