/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <stdio.h>
#include <algorithm>

#include "models/my_cycles.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

static void print_shape(const tflite::RuntimeShape& shape) {
    if (shape.DimensionsCount() == 0) {
        printf("*, *, *, *, ");
        return;
    } else if (shape.DimensionsCount() != 4) {
        printf("*ERR* shape dims should be 4, but are %ld\n",
               shape.DimensionsCount());
    }
    printf("%ld, %ld, %ld, %ld, ", shape.Dims(0), shape.Dims(1), shape.Dims(2),
           shape.Dims(3));
}

void my_print_conv_params(const tflite::ConvParams& params,
                          const tflite::RuntimeShape& input_shape,
                          const tflite::RuntimeShape& filter_shape,
                          const tflite::RuntimeShape& output_shape) {
    auto& padding = params.padding_values;
    printf("\npadding_width: %d, padding_height: %d, \npadding_width_offset: %d, padding_height_offset: %d, \nstride_width: %d, stride_height: %d, \ndilation_width_factor: %d, dilation_height_factor: %d, \ninput_offset: %ld, weights_offset: %ld, \noutput_offset: %ld, output_multiplier: %ld, output_shift: %d, \nquantized_activation_min: %ld, quantized_activation_max: %ld\n",
           padding.width, padding.height,
           padding.width_offset, padding.height_offset,
           params.stride_width, params.stride_height,
           params.dilation_width_factor, params.dilation_height_factor,
           params.input_offset, params.weights_offset,
           params.output_offset, params.output_multiplier, params.output_shift,
           params.quantized_activation_min, params.quantized_activation_max);
    // print input_shape
    printf("input shape: ");
    print_shape(input_shape);
    printf("\n");
    // print filter_shape
    printf("filter shape: ");
    print_shape(filter_shape);
    printf("\n");
    // print output_shape
    printf("output shape: ");
    print_shape(output_shape);
    printf("\n");
}

namespace tflite {
namespace reference_integer_ops {

void print_3d_matrix(const int8_t* matrix, int channels, int height, int width) {
    printf("Input matrix 5ch.: (channels, height, width) = (%d, %d, %d)\n", channels, height, width);
    for (int c = 0; c < 5; ++c) {
        printf("Channel %d:\n", c);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                printf("%4d ", matrix[c * height * width + h * width + w]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_2d_col_matrix(const int8_t* matrix, int rows, int cols) {
    printf("Output matrix: (rows, cols) = (%d, %d)\n", rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%4d ", matrix[r * cols + c]);
        }
        printf("\n");
    }
}

#define MAX_COL_SIZE 10000000

int8_t im2col_get_pixel(const int8_t* im, int height, int width, int channels, int row, int col, int channel, int pad_h, int pad_w) {
    row -= pad_h;
    col -= pad_w;
    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;

    return im[col + width * (row + height * channel)];
}

void im2col(const int8_t* data_im,
            int channels,
            int height,
            int width,
            int kernel_h,
            int kernel_w,
            int stride_h,
            int stride_w,
            int pad_h,
            int pad_w,
            int dilate_rate_h,
            int dilate_rate_w,
            int8_t input_offset,
            int8_t* data_col) {
    int c, h, w;
    int dilate_ksize_h = (dilate_rate_h - 1) * (kernel_h + 1) + kernel_h;
    int dilate_ksize_w = (dilate_rate_w - 1) * (kernel_w + 1) + kernel_w;
    int height_col = (height + 2 * pad_h - dilate_ksize_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - dilate_ksize_w) / stride_w + 1;

    int channels_col = channels * kernel_h * kernel_w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_w * kernel_h);
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilate_rate_h + h * stride_h;
                int im_col = w_offset * dilate_rate_w + w * stride_w;
                int col_index = c * (height_col * width_col) + h * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad_h, pad_w) +
                                      input_offset;
                // printf("im_row = %d, im_col = %d, pixel = %f\n", im_row, im_col, data_col[col_index]);
            }
        }
    }
}

void reshape_filter_to_2d(const int8_t* filter_data, int8_t* filter_im2col, int filter_channel, int filter_height, int filter_width, int filter_input_depth) {
    for (int out_channel = 0; out_channel < filter_channel; ++out_channel) {
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                    int col_index = out_channel * (filter_height * filter_width * filter_input_depth) +
                                    filter_y * (filter_width * filter_input_depth) +
                                    filter_x * filter_input_depth + in_channel;
                    filter_im2col[col_index] = filter_data[out_channel * (filter_height * filter_width * filter_input_depth) +
                                                           filter_y * (filter_width * filter_input_depth) +
                                                           filter_x * filter_input_depth + in_channel];
                }
            }
        }
    }
}

// Simple matrix multiplication function
void matrix_multiply(const int8_t* a, const int8_t* b, int32_t* c, int a_rows, int a_cols, int b_cols) {
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            c[i * b_cols + j] = 0;
            for (int k = 0; k < a_cols; ++k) {
                c[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }
}

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data) {
    // print parameters.
    my_print_conv_params(params, input_shape, filter_shape, output_shape);

    // Get parameters.
    const int32_t input_offset = params.input_offset;  // r = s(q - Z)
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int32_t output_offset = params.output_offset;

    // Set min and max value of the output.
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    int8_t input_im2col[MAX_COL_SIZE];
    int8_t weight_im2col[MAX_COL_SIZE];
    int32_t result_im2col[MAX_COL_SIZE];
    // Consistency check.
    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
        TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    // Check dimensions of the tensors.
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    int8_t input_im2col[MAX_COL_SIZE];
    int8_t weight_im2col[MAX_COL_SIZE];
    int32_t result_im2col[MAX_COL_SIZE];

    /*
        TODO for every batch:
        1. convert input and filter to im2col format
        2. run matrix multiplication
        3. convert result to output format (col2im), add bias, quantize, activation
    */
    int groups = input_depth / filter_input_depth;
    for (int batch = 0; batch < batches; ++batch) {
        // im2col for input
        im2col(&input_data[Offset(input_shape, batch, 0, 0, 0)],
               input_depth, input_height, input_width,
               filter_height, filter_width,
               stride_height, stride_width,
               pad_height, pad_width,
               dilation_height_factor, dilation_width_factor,
               input_offset,
               input_im2col);

        // reshape filter to 2D
        reshape_filter_to_2d(filter_data, weight_im2col, output_depth, filter_height, filter_width, filter_input_depth);

        // Perform matrix multiplication
        matrix_multiply(weight_im2col, input_im2col, result_im2col,
                        output_depth, input_depth * filter_height * filter_width,
                        output_height * output_width);

        // col2im for output, perform quantization, bias addition, and activation
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    int col_index = out_channel * (output_height * output_width) +
                                    out_y * output_width + out_x;
                    int32_t acc = result_im2col[col_index];

                    // Add bias
                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }

                    // Apply per-channel quantization
                    acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);

                    // Store the result
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int8_t>(acc);
                }
            }
        }
    }
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_input,
    int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data) {
    TFLITE_DCHECK(unpacked_filter_data != nullptr);
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        filter_input, filter_shape.FlatSize(), unpacked_filter_data);
    ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                   input_data, filter_shape, unpacked_filter_data, bias_shape,
                   bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int16_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const AccumScalar* bias_data,
    const RuntimeShape& output_shape,
    int16_t* output_data) {
    // Get parameters.
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;

    // Set min and max value of the output.
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // Consistency check.
    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
        TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    // Check dimensions of the tensors.
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    const int groups = input_depth / filter_input_depth;
    TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int filters_per_group = output_depth / groups;
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    AccumScalar acc = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;

                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image =
                                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                (in_y < input_height);

                            if (!is_point_inside_image) {
                                continue;
                            }

                            for (int in_channel = 0; in_channel < filter_input_depth;
                                 ++in_channel) {
                                int32_t input_val =
                                    input_data[Offset(input_shape, batch, in_y, in_x,
                                                      in_channel + group * filter_input_depth)];
                                int32_t filter_val = filter_data[Offset(
                                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                                // Accumulate with 64 bits accumulator.
                                // int64_t += int8_t * int16_t so the highest value we can
                                // get from each accumulation is [-127, 127] * ([-32768,
                                // 32767] -
                                // [-32768, 32767]), which is [-8322945, 8322945].
                                // log2(8322945) = 22.99.
                                acc += filter_val * input_val;
                            }
                        }
                    }
                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }
                    int32_t scaled_acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
                    scaled_acc = std::max(scaled_acc, output_activation_min);
                    scaled_acc = std::min(scaled_acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int16_t>(scaled_acc);
                }
            }
        }
    }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
