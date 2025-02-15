#include <cassert>
#include "common01.h"
#include "CTensor.h"
#include "CRandFiller.h"
#include "kernels.h"
#include <iostream>
#include "common02.h"

#include <cuda_runtime.h>

constexpr size_t N = 1024;
constexpr unsigned RUNS = 5;
#define INCLUDE_2BIT_POT

int main(int argc, char *argv[]) {
    CTensor<float> tnA({N, N});
    CTensor<float> tnB({N, N});
    CTensor<uint8_t> tnB_pot({N, N});
    CTensor<uint8_t> tnB_pot_packed4bit({N, N/2});
    CTensor<uint8_t> tnB_pot_packed2bit({N, N/4});
    CTensor<float> tnC({N, N});
    CTensor<float> goldTnC({N, N});

    CRandFiller<float> rand_float(-0.5, 0.5);
#ifndef INCLUDE_2BIT_POT
    CRandFiller<int> rand_int(-5, 5); // good for 4bits (1s + 3e) and higher
#else
    CRandFiller<int> rand_int(-2, 2); // good for 2bits (1s + 1e)
#endif

    tnA.Fill(&rand_float, FillTypes::kRandom);
    for (size_t i=0; i<tnB.GetSize(); i++) {
        int r = rand_int.GetRand();
        //int r = 2;
        int r_positive = std::abs(r);
        unsigned char is_neg = r < 0;
        unsigned char exponent = 1 << (int) std::log2(r_positive);

        // Sign (1bit) | Exponent (7bits)
        // For float32 we have to shift the sign bit to the left 1 bit so
        // we can add 1(sign) + 8(exponent) with float32's data sign+exponent
        tnB_pot.GetPtrHost()[i] = is_neg << 7 | (exponent & 0x7F);

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
    tnB_pot_packed4bit.H2D();
    tnB_pot_packed2bit.H2D();

    cudaDeviceSynchronize();

    // This is our golden reference. We won't be using CPU kernel since that will be super slow for large N values.
    {
        timer_stats stats("LaunchKernelMatmulBase");
        for (volatile int i = 0; i < RUNS; ++i) {
            timer_scope_cuda timer(stats, stream);
            LaunchKernelMatmulBase(stream, N, tnA.GetPtrDevice(), tnB.GetPtrDevice(), goldTnC.GetPtrDevice());
        }
        goldTnC.D2H();
        cudaDeviceSynchronize();
        // No wiping for goldTnC since we will be using it for comparison.
    }

    cudaDeviceSynchronize();
    {
        timer_stats stats("LaunchKernelMatmulPotUint8Packed2");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            LaunchKernelMatmulPotUint8Packed2(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot_packed4bit.GetPtrDevice(), tnC.GetPtrDevice(), goldTnC.GetPtrDevice());
        }
        tnC.D2H();
        cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 0.001f)) {
            std::cout << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
        tnC.Wipe(); // wiping to avoid any false-match in the next comparison.
    }

    {
        timer_stats stats("LaunchKernelMatmulPotUint8Packed4");
        for (volatile int i = 0; i < RUNS; ++i) {
            // This is a blocking measurement.
            // Each iteration will be blocked until the kernel finishes.
            // Even though we are using streams and kernels are launches asynchronously.
            timer_scope_cuda timer(stats, stream);
            LaunchKernelMatmulPotUint8Packed4(stream, N, reinterpret_cast<const uint32_t*>(tnA.GetPtrDevice()), tnB_pot_packed2bit.GetPtrDevice(), tnC.GetPtrDevice(), goldTnC.GetPtrDevice());
        }
        tnC.D2H();
        cudaDeviceSynchronize();
        if (!tnC.CompareHostData(goldTnC, 0.001f)) {
            std::cout << "Error: tnC and goldTnC are not equal." << std::endl;
        } else {
            std::cout << "Success: tnC and goldTnC are equal." << std::endl;
        }
        tnC.Wipe(); // wiping to avoid any false-match in the next comparison.
    }
    cudaDeviceSynchronize();

}

