#include <cassert>
#include "common01.h"
#include "CTensor.h"
#include "CRandFiller.h"
#include "kernel.h"
#include <iostream>
#include "common02.h"
#include "defs.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INCLUDE_2BIT_POT

void checkCublas(cublasStatus_t result, const char* const func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in function " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void matmul_cpu_gold(
    const CTensor<float> &tnA,
    const CTensor<float> &tnB,
    CTensor<float> &outTnC) {
    assert(tnA.GetShape()[1] == tnB.GetShape()[0]);
    outTnC = CTensor<float>({tnA.GetShape()[0], tnB.GetShape()[1]});

    auto pA = tnA.GetPtrHost();
    auto pB = tnB.GetPtrHost();
    auto pC = outTnC.GetPtrHost();
    for (size_t j = 0; j < tnA.GetShape()[0]; j++) {
        for (size_t i = 0; i < tnB.GetShape()[1]; i++) {
            float sum = 0;
            for (size_t k = 0; k < tnA.GetShape()[1]; k++) {
                sum += pA[j * tnA.GetShape()[1] + k] * pB[k * tnB.GetShape()[1] + i];
            }
            pC[j * tnB.GetShape()[1] + i] = sum;
        }
    }
}

void matmul_gpu_cublas_gold(
    cublasHandle_t handle,
    const CTensor<float> &tnA,
    const CTensor<float> &tnB,
    CTensor<float> &outTnC) {

    assert(tnA.GetShape()[1] == tnB.GetShape()[0]);
    assert(tnA.GetShape()[0] == tnB.GetShape()[1]);
    const size_t size_N = tnA.GetShape()[0];

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /// TODO: Fix this. It's broken.
    // Still broken. I guess cublas uses col-major for output tensor and that leads to mismatches.
    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_N, size_N, size_N,
            &alpha,
            tnA.GetPtrDevice(), size_N,
            tnB.GetPtrDevice(), size_N,
            &beta,
            outTnC.GetPtrDevice(), size_N
        ),
        "cublasSgemm"
    );

}

int main(int argc, char *argv[]) {
    std::cout << "RUNS: " << RUNS << std::endl;
    std::cout << "N: " << N << std::endl;

    CTensor<float> tnA({N, N});
    CTensor<float> tnB({N, N});
    CTensor<uint8_t> tnB_pot({N, N});
    CTensor<uint16_t> tnB_pot16({N, N});
    CTensor<uint8_t> tnB_pot_packed4bit({N, N/2});
    CTensor<uint8_t> tnB_pot_packed2bit({N, N/4});
    CTensor<float> tnC({N, N});
    CTensor<float> goldTnC({N, N});
    CTensor<float> goldCublasTnC({N, N});

    CRandFiller<float> rand_float(-5.0, 5.0);
#ifndef INCLUDE_2BIT_POT
    CRandFiller<int> rand_int(-5, 5); // good for 4bits (1s + 3e) and higher
#else
    CRandFiller<int> rand_int(-2, 2); // good for 2bits (1s + 1e)
#endif

    tnA.Fill(&rand_float, FillTypes::kRandom);
    for (size_t i=0; i<tnB.GetSize(); i++) {
        int r = rand_int.GetRand();
        int r_positive = std::abs(r);
        unsigned char is_neg = r < 0;
        unsigned char exponent = 1 << (int) std::log2(r_positive);

        // Sign (1bit) | Exponent (7bits)
        // For float32 we have to shift the sign bit to the left 1 bit so
        // we can add 1(sign) + 8(exponent) with float32's data sign+exponent
        tnB_pot.GetPtrHost()[i] = is_neg << 7 | (exponent & 0x7F);
        tnB_pot16.GetPtrHost()[i] = is_neg << 15 | ((exponent & 0xFF)<<7);

        if (i < 4 )std::cout << "tbB_pot16[" << i << "]: " << (int)tnB_pot16.GetPtrHost()[i] << std::endl;

        tnB.GetPtrHost()[i] = static_cast<float>(std::pow(2, exponent));
        tnB.GetPtrHost()[i] *= is_neg ? -1.f : 1.f;
    }

    // Make tnB_pot_packed4bit for row-major tnB
    for (int j=0; j<N; j++) {
        for (int i=0; i<N; i+=2) {
            unsigned char e0 = tnB_pot.GetPtrHost()[j*N+i+0] & 0x7F;
            unsigned char s0 = tnB_pot.GetPtrHost()[j*N+i+0] >> 7;
            unsigned char e1 = tnB_pot.GetPtrHost()[j*N+i+1] & 0x7F;
            unsigned char s1 = tnB_pot.GetPtrHost()[j*N+i+1] >> 7;
            unsigned char w0_4bit = (e0 & 0x7) | (s0 << 3);
            unsigned char w1_4bit = (e1 & 0x7) | (s1 << 3);
            tnB_pot_packed4bit.GetPtrHost()[j*N/2+i/2] = w0_4bit | (w1_4bit << 4);
        }
    }

    // Make tnB_pot_packed2bit for row-major tnB
    for (int j=0; j<N; j++) {
        for (int i=0; i<N; i+=4) {
            unsigned char e0 = tnB_pot.GetPtrHost()[j*N+i+0] & 0x7F;
            unsigned char s0 = tnB_pot.GetPtrHost()[j*N+i+0] >> 7;

            unsigned char e1 = tnB_pot.GetPtrHost()[j*N+i+1] & 0x7F;
            unsigned char s1 = tnB_pot.GetPtrHost()[j*N+i+1] >> 7;

            unsigned char e2 = tnB_pot.GetPtrHost()[j*N+i+2] & 0x7F;
            unsigned char s2 = tnB_pot.GetPtrHost()[j*N+i+2] >> 7;

            unsigned char e3 = tnB_pot.GetPtrHost()[j*N+i+3] & 0x7F;
            unsigned char s3 = tnB_pot.GetPtrHost()[j*N+i+3] >> 7;

            unsigned char w0_2bit = (e0 & 0x1) | (s0 << 1);
            unsigned char w1_2bit = (e1 & 0x1) | (s1 << 1);
            unsigned char w2_2bit = (e2 & 0x1) | (s2 << 1);
            unsigned char w3_2bit = (e3 & 0x1) | (s3 << 1);

            tnB_pot_packed2bit.GetPtrHost()[j*N/4+i/4] = w0_2bit | (w1_2bit << 2)| (w2_2bit << 4)| (w3_2bit << 6);
        }
    }

    std::cout << "Computing goldTnC..." << std::endl;
    matmul_cpu_gold(tnA, tnB, goldTnC);

    std::cout << "Preparing to launch on GPU..." << std::endl;

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create CUDA events
    cudaEvent_t start_events[N], stop_events[N];
    for (int i = 0; i < N; ++i) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&stop_events[i]);
    }

    tnA.H2D();
    tnB.H2D();
    tnB_pot.H2D();
    tnB_pot16.H2D();
    tnB_pot_packed4bit.H2D();
    tnB_pot_packed2bit.H2D();

    cudaDeviceSynchronize();

    {
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "cublasCreate");
        matmul_gpu_cublas_gold(handle, tnA, tnB, goldCublasTnC);
        cudaDeviceSynchronize();
        goldCublasTnC.D2H();
        checkCublas(cublasDestroy(handle), "cublasDestroy");
        if (!goldCublasTnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: goldCublasTnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: goldCublasTnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel01");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel04");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel04(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel08");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel08(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel01_PoT");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel01_PoT(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel01_PoT16");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel01_PoT16(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot16.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel04_PoT");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel04_PoT(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel01_PoT4bit");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel01_PoT4bits(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot_packed4bit.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

    {
        timer_stats stats("LaunchKernel04_PoT4bit");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel04_PoT4bits(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot_packed4bit.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }

#ifdef INCLUDE_2BIT_POT
    {
        timer_stats stats("LaunchKernel04_PoT2bit");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            //LaunchKernel01(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), tnC.GetPtrDevice());
            LaunchKernel04_PoT2bits(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot_packed2bit.GetPtrDevice(), tnC.GetPtrDevice());
        }
        tnC.D2H(); cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 1e-5)) {
            std::cerr << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
    }
#endif

}

