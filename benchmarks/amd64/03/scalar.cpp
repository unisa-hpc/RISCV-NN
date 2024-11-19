//
// Created by saleh on 11/19/24.
//

#include "defs.h"
#include "common01.h"

void FUNCTION_NAME(conv2d_direct_padding_ochw_scalar)(
    const int32_t* __restrict__ input, // Input tensor [C_in][H][W]
    const int32_t* __restrict__ kernel, // Kernel tensor [C_out][C_in][KH][KW]
    int32_t* __restrict__ output, // Output tensor [C_out][H_out][W_out]
    int input_height, // Input height
    int input_width, // Input width
    int channels_in, // Input channels
    int channels_out, // Output channels
    int kernel_size_x,
    int kernel_size_y,
    int stride_x,
    int stride_y,
    bool padding_valid,
    bool padding_same) {
    if (padding_valid && !padding_same) {
        const size_t heightOut = GetOutHeight(input_height, kernel_size_y, stride_y);
        const size_t widthOut = GetOutWidth(input_width, kernel_size_x, stride_x);
        // -------------------

        for (int h = 0; h < input_height; h += stride_y) {
            for (int w = 0; w < input_width; w += stride_x) {
                if (h + kernel_size_y - 1 < input_height && w + kernel_size_x - 1 < input_width) {
                    for (int o = 0; o < channels_out; o++) {
                        int32_t _sum = 0;
                        for (int c = 0; c < channels_in; c++) {
                            // h and w are zero-based, so it's correct.
                            for (int j = 0; j < kernel_size_y; j++) {
                                for (int i = 0; i < kernel_size_x; i++) {
                                    size_t idxI =
                                        /*b * height * width * chIn +*/
                                        c * input_height * input_width +
                                        (h + j) * input_width +
                                        (w + i);
                                    size_t idxW =
                                        o * channels_in * kernel_size_y * kernel_size_x +
                                        c * kernel_size_y * kernel_size_x +
                                        j * kernel_size_x +
                                        i;
                                    _sum = _sum + input[idxI] * kernel[idxW];
                                }
                            }
                        }

                        auto idxO =
                            /*b * heightOut * widthOut * chOut + */
                            o * heightOut * widthOut +
                            (h / stride_y) * widthOut +
                            (w / stride_x);

                        output[idxO] = _sum;
                    }
                }
            }
        }
    }
    else {
        throw std::invalid_argument("The requested padding scheme is not supported.");
    }
}