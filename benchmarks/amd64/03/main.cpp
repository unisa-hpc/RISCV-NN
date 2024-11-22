//
// Created by saleh on 11/19/24.
//

#include "defs.h"
#include "common01.h"

extern void conv2d_direct_padding_ochw_scalar_noautovec(
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ kernel,
    int32_t* __restrict__ output,
    int input_height,
    int input_width,
    int channels_in,
    int channels_out,
    int kernel_size_x,
    int kernel_size_y,
    int stride_x,
    int stride_y,
    bool padding_valid,
    bool padding_same);

extern void conv2d_direct_padding_ochw_scalar_autovec(
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ kernel,
    int32_t* __restrict__ output,
    int input_height,
    int input_width,
    int channels_in,
    int channels_out,
    int kernel_size_x,
    int kernel_size_y,
    int stride_x,
    int stride_y,
    bool padding_valid,
    bool padding_same);

// Since this is a template function, we cannot define it in the static library.
template <typename T, int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, bool padding_valid, bool padding_same>
void conv2d_direct_padding_ochw_avx_try18(
    const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* __restrict__ output,
    int input_height,
    int input_width,
    int channels_in,
    int channels_out) {

    constexpr int FACTOR0 = UNROLL_FACTOR0;
    constexpr int FACTOR1 = UNROLL_FACTOR1;
    constexpr int FACTOR2 = UNROLL_FACTOR2;
    constexpr int FACTOR3 = UNROLL_FACTOR3;

    if ((padding_valid && !padding_same)) {
        // valid padding
        constexpr int VEC_SIZE = 256 / (8 * sizeof(T));
        const int heightOut = GetOutHeight(input_height, kernel_size_y, stride_y);
        const int widthOut = GetOutWidth(input_width, kernel_size_x, stride_x);
        const int widthOutSafe = widthOut - (widthOut % VEC_SIZE);

        for (int h = 0; h < heightOut; h++) {
            const int h_start = h * stride_y;

            for (int w = 0; w < widthOutSafe; w += VEC_SIZE) {
                const int w_start = w * stride_x;
#pragma GCC unroll (FACTOR0)
                for (int o = 0; o < channels_out; o++) {
                    __m256i sum = _mm256_setzero_si256();
#pragma GCC unroll (FACTOR1)
                    for (int p = 0; p < channels_in; p++) {
#pragma GCC unroll (FACTOR2)
                        for (int kh = 0; kh < kernel_size_y; kh++) {
                            const int ih = h_start + kh;
                            if (ih < 0 || ih >= input_height) continue;

#pragma GCC unroll (FACTOR3)
                            for (int kw = 0; kw < kernel_size_x; kw++) {
                                const int iw = w_start + kw;
                                if (iw < 0 || iw >= input_width) continue;
                                __m256i in_val;

                                if (stride_x == 1 && stride_y == 1) {
                                    size_t input_idx =
                                        (p) * input_height * input_width +
                                        (ih) * input_width +
                                        (iw);

                                    // if channels_in is not a multiple of 8, we have to use unaligned load.
                                    in_val = _mm256_loadu_si256((__m256i*)&input[input_idx]);
                                } else {
                                    int32_t strided_inputs[8];
                                    for (int v = 0; v < VEC_SIZE; v++) {
                                        const int iw = w_start + (v * stride_x) + kw;
                                        if (iw < 0 || iw >= input_width) continue;

                                        size_t input_idx =
                                            (p) * input_height * input_width +
                                            (ih) * input_width +
                                            iw;
                                        strided_inputs[v] = input[input_idx];
                                    }
                                    // if channels_in is not a multiple of 8, we have to use unaligned load.
                                    in_val = _mm256_loadu_si256((__m256i*)strided_inputs);
                                }

                                size_t kernel_idx =
                                    (o) * channels_in * kernel_size_y * kernel_size_x +
                                    (p) * kernel_size_y * kernel_size_x +
                                    (kh) * kernel_size_x +
                                    (kw);
                                __m256i k_val = _mm256_set1_epi32(kernel[kernel_idx]);
                                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(in_val, k_val));
                            }
                        }
                    }

                    size_t output_idx =
                        (o) * heightOut * widthOut +
                        (h) * widthOut +
                        (w);
                    _mm256_storeu_si256((__m256i*)&output[output_idx], sum);

                    // if our input is 1024*1024, then our output will be 1022*1022. That is not a multiple of 8.
                    // Leading to overlapping writes. If you set it to 1026*1026, then it will work.
                }
            }
        }


        // Handle the remaining columns with scalar code
        for (int h = 0; h < heightOut; h++) {
            const int h_start = h * stride_y;

            for (int w = widthOutSafe; w < widthOut; w++) {
                const int w_start = w * stride_x;
#pragma GCC unroll (FACTOR0)
                for (int o = 0; o < channels_out; o++) {
                    int32_t sum = 0;
#pragma GCC unroll (FACTOR1)
                    for (int p = 0; p < channels_in; p++) {
#pragma GCC unroll (FACTOR2)
                        for (int kh = 0; kh < kernel_size_y; kh++) {
                            const int ih = h_start + kh;
                            if (ih < 0 || ih >= input_height) continue;
#pragma GCC unroll (FACTOR3)
                            for (int kw = 0; kw < kernel_size_x; kw++) {
                                const int iw = w_start + kw;
                                if (iw < 0 || iw >= input_width) continue;

                                size_t input_idx =
                                    (p) * input_height * input_width +
                                    (ih) * input_width +
                                    (iw);

                                size_t kernel_idx =
                                    (o) * channels_in * kernel_size_y * kernel_size_x +
                                    (p) * kernel_size_y * kernel_size_x +
                                    (kh) * kernel_size_x +
                                    (kw);
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }

                    size_t output_idx =
                        (o) * heightOut * widthOut +
                        (h) * widthOut +
                        (w);
                    output[output_idx] = sum;
                }
            }
        }
    }
    else {
        throw std::invalid_argument("The requested SAME padding scheme is not supported.");
    }
}

void verify_results_ohw(
    const int32_t* c1, const int32_t* c2,
    size_t output_height, size_t output_width, size_t channels) {
    unsigned missmatch = 0; bool break_flag = false;
    for (size_t o = 0; o < channels && !break_flag; o++) { // OHW layout
        for (size_t j = 0; j < output_height && !break_flag; j++) {
            for (size_t i = 0; i < output_width && !break_flag; i++) {
                const size_t idx = o * output_height * output_width + j * output_width + i;
                if (c1[idx] != c2[idx] ) {
                    std::cout << "Results mismatch at index (c,j,i) " << o << " , " << j << " , " << i << std::endl;
                    std::cout << c1[idx] << " != " << c2[idx] << std::endl;
                    if (missmatch++ > 50) {
                        std::cout << "MISMATCH - more than 50 elements do not match." << std::endl;
                        break_flag = true;
                    }
                }
            }
        }
    }
    if (missmatch == 0)
        std::cout << "Results match!" << std::endl;
    else
        std::cout << "Results DO NOT match!" << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
}

void wipe(int32_t* p, size_t len) {
    for (size_t i = 0; i < len; i++) {
        p[i] = 0;
    }
}

void init(int32_t* p, size_t len, bool PoT) {
    constexpr int32_t half_range = 1024;
    for (size_t j = 0; j < len; j++) {
        if (PoT) {
            // fill with values in range -half_range to +half_range, but only power of two values
            const int32_t r = rand() % (half_range*2) - half_range;
            // convert r to the nearest power of two
            const int32_t v =  (1 << static_cast<int32_t>(log2(r))) * (r > 0 ? 1 : -1);
            p[j] = v;
        } else {
            // fill with values in range -half_range to +half_range
            p[j] = static_cast<int32_t>(rand() % (half_range*2) - half_range);
        }

    }
}

int main(int argc, char** argv) {
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment
    std::cout << "UNROLLING FACTOR 0: " << UNROLL_FACTOR0 << std::endl;
    std::cout << "UNROLLING FACTOR 1: " << UNROLL_FACTOR1 << std::endl;
    std::cout << "UNROLLING FACTOR 2: " << UNROLL_FACTOR2 << std::endl;
    std::cout << "UNROLLING FACTOR 3: " << UNROLL_FACTOR3 << std::endl;
    std::cout << "INPUT HEIGHT: " << I_H << std::endl;
    std::cout << "INPUT WIDTH: " << I_W << std::endl;
    std::cout << "KERNEL HEIGHT: " << K_H << std::endl;
    std::cout << "KERNEL WIDTH: " << K_W << std::endl;
    std::cout << "CHANNELS IN: " << C_I << std::endl;
    std::cout << "CHANNELS OUT: " << C_O << std::endl;
    std::cout << "STRIDE X: " << S_X << std::endl;
    std::cout << "STRIDE Y: " << S_Y << std::endl;
    std::cout << "PADDING: " << padding << std::endl;

    const size_t _out_height = GetOutHeight(input_height, kernel_height, stride_y);
    const size_t _out_width = GetOutWidth(input_width, kernel_width, stride_x);

    int32_t* input_ptr = aligned_alloc_array<int32_t>(
        channel_in * input_height * input_width,
        ALIGNMENT
    );
    int32_t* kernel_ptr = aligned_alloc_array<int32_t>(
        channel_out * channel_in * kernel_height * kernel_width,
        ALIGNMENT
    );
    int32_t* c_scalar_noautovec_ptr = aligned_alloc_array<int32_t>(
        channel_out * _out_height * _out_width,
        ALIGNMENT
    );
    int32_t* c_scalar_autovec_ptr = aligned_alloc_array<int32_t>(
        channel_out * _out_height * _out_width,
        ALIGNMENT
    );
    int32_t* c_avx_ptr = aligned_alloc_array<int32_t>(
        channel_out * _out_height * _out_width,
        ALIGNMENT
    );

    for (size_t j = 0; j < input_height; j++) {
        for (size_t i = 0; i < input_width; i++) {
            input_ptr[j * input_height + i] = static_cast<int32_t>(rand() % 32678);
        }
    }
    init(input_ptr, channel_in * input_height * input_width, true);
    init(kernel_ptr, channel_out * channel_in * kernel_height * kernel_width, true);


    wipe(c_scalar_noautovec_ptr, channel_out * _out_height * _out_width);
    {
        timer_stats tp(
            "Scalar Direct OCHW Conv2D With Mul NoAutovec",
            {
              {"UNROLL_FACTOR0", UNROLL_FACTOR0},
              {"UNROLL_FACTOR1", UNROLL_FACTOR1},
              {"UNROLL_FACTOR2", UNROLL_FACTOR2},
              {"UNROLL_FACTOR3", UNROLL_FACTOR3},
              {"I_H", I_H},
              {"I_W", I_W},
              {"K_H", K_H},
              {"K_W", K_W},
              {"C_I", C_I},
              {"C_O", C_O},
              {"S_X", S_X},
              {"S_Y", S_Y}
            }
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            conv2d_direct_padding_ochw_scalar_noautovec(
                input_ptr, kernel_ptr, c_scalar_noautovec_ptr,
                input_height, input_width, channel_in, channel_out,
                kernel_height, kernel_width, stride_x, stride_y,
                true, false
            );
        }
    }
    // no need to verify, this is our gold.

    std::cout << "Preparing to launch..." << std::endl;
    wipe(c_scalar_autovec_ptr, channel_out * _out_height * _out_width);
    {
        timer_stats tp(
            "Scalar Direct OCHW Conv2D With Mul Autovec",
            {
              {"UNROLL_FACTOR0", UNROLL_FACTOR0},
              {"UNROLL_FACTOR1", UNROLL_FACTOR1},
              {"UNROLL_FACTOR2", UNROLL_FACTOR2},
              {"UNROLL_FACTOR3", UNROLL_FACTOR3},
              {"I_H", I_H},
              {"I_W", I_W},
              {"K_H", K_H},
              {"K_W", K_W},
              {"C_I", C_I},
              {"C_O", C_O},
              {"S_X", S_X},
              {"S_Y", S_Y}
            }
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            conv2d_direct_padding_ochw_scalar_autovec(
                input_ptr, kernel_ptr, c_scalar_autovec_ptr,
                input_height, input_width, channel_in, channel_out,
                kernel_height, kernel_width, stride_x, stride_y,
                true, false
            );
        }
    }
    // no need to verify, this is our gold.


    std::cout << "Preparing to launch..." << std::endl;
    wipe(c_scalar_autovec_ptr, channel_out * _out_height * _out_width);
    {
        timer_stats tp(
            "Vectorized Direct OCHW Conv2D With Mul AVX2",
            {
              {"UNROLL_FACTOR0", UNROLL_FACTOR0},
              {"UNROLL_FACTOR1", UNROLL_FACTOR1},
              {"UNROLL_FACTOR2", UNROLL_FACTOR2},
              {"UNROLL_FACTOR3", UNROLL_FACTOR3},
              {"I_H", I_H},
              {"I_W", I_W},
              {"K_H", K_H},
              {"K_W", K_W},
              {"C_I", C_I},
              {"C_O", C_O},
              {"S_X", S_X},
              {"S_Y", S_Y}
            }
        );
        for (volatile size_t i = 0; i < RUNS; i++) {
            timer_scope ts(tp);
            conv2d_direct_padding_ochw_avx_try18<int32_t, kernel_width, kernel_height, stride_x, stride_y, true, false>(
                input_ptr, kernel_ptr, c_avx_ptr,
                input_height, input_width, channel_in, channel_out
            );
        }
    }
    verify_results_ohw(c_scalar_noautovec_ptr, c_avx_ptr, _out_height, _out_width, channel_out);

    free(input_ptr);
    free(kernel_ptr);
    free(c_scalar_noautovec_ptr);
    free(c_scalar_autovec_ptr);
    free(c_avx_ptr);

    return 0;
}
