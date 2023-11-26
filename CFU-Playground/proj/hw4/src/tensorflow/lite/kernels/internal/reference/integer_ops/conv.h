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

void print_3d_matrix(const int8_t* matrix, int channels, int height, int width) {
    printf("Input matrix max5: (channels, height, width) = (%d, %d, %d)\n", channels, height, width);
    channels = std::min(channels, 5);
    height = std::min(height, 5);
    width = std::min(width, 5);
    for (int c = 0; c < channels; ++c) {
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

void print_2d_col_matrix(const int8_t** matrix, int rows, int cols) {
    printf("Output matrix max5: (rows, cols) = (%d, %d)\n", rows, cols);
    rows = std::min(rows, 5);
    cols = std::min(cols, 5);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%4d ", matrix[r][c]);
        }
        printf("\n");
    }
}
namespace tflite {
namespace reference_integer_ops {

#define MAX_CHANNEL_SIZE 512
#define MAX_HWC_SIZE 512
#define MAX_WINDOW_COUNT 4096

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
    // const int filter_number = filter_shape.Dims(0);  // filter_number == output_depth
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    const int groups = input_depth / filter_input_depth;
    TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int filters_per_group = output_depth / groups;
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    int8_t weight_im2col[MAX_CHANNEL_SIZE][MAX_HWC_SIZE];
    int32_t input_im2col[MAX_HWC_SIZE][MAX_WINDOW_COUNT];
    int32_t result_im2col[MAX_CHANNEL_SIZE][MAX_WINDOW_COUNT];

    // perform matrix multiplication
    int HWC = filter_height * filter_width * filter_input_depth;
    int max_window_sliding_time = output_height * output_width;
    int filter_number = output_depth;

    // Check if exceeds the defined max size of the matrix
    printf("HWC: %d, max_window_sliding_time: %d, filter_number: %d\n", HWC, max_window_sliding_time, filter_number);
    printf("MAX_HWC_SIZE: %d, MAX_WINDOW_COUNT: %d, MAX_CHANNEL_SIZE: %d\n", MAX_HWC_SIZE, MAX_WINDOW_COUNT, MAX_CHANNEL_SIZE);
    if (HWC > MAX_HWC_SIZE || max_window_sliding_time > MAX_WINDOW_COUNT || filter_number > MAX_CHANNEL_SIZE) {
        printf("ERROR: matrix size exceeds the defined max size. \n");
        return;
    }

    for (int batch = 0; batch < batches; ++batch) {
        // im2col for input,  (HxWxC, window sliding time)
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                                int hwc_index = in_channel + filter_input_depth * (filter_x + filter_width * filter_y);
                                int window_index = out_y * output_width + out_x;
                                if (in_x < 0 || in_x >= input_width || in_y < 0 || in_y >= input_height) {
                                    input_im2col[hwc_index][window_index] = 0;
                                } else {
                                    int32_t input_val = input_data[Offset(input_shape, batch, in_y, in_x, in_channel + group * filter_input_depth)];
                                    input_im2col[hwc_index][window_index] = input_val + input_offset;
                                }
                            }
                        }
                    }
                }
            }
        }
        printf("im2col for input done. \n");

        // reshape filter to 2D (N, HxWxC)
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                        int32_t filter_val = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];

                        int number_index = out_channel;
                        int hwc_index = in_channel + filter_input_depth * (filter_x + filter_width * filter_y);
                        weight_im2col[number_index][hwc_index] = filter_val;
                        // printf(" %4d ", weight_im2col[number_index][hwc_index]);
                    }
                }
            }
            // printf("\n");
        }
        printf("reshape filter to 2D done. \n");

        // record MAC
        unsigned my_start = perf_get_mcycle();
        // perform matrix multiplication (N, HxWxC) x (HxWxC, window sliding time) = (N, window sliding time)
        for (int i = 0; i < filter_number; ++i) {
            for (int j = 0; j < max_window_sliding_time; ++j) {
                int32_t sum = 0;
                for (int k = 0; k < HWC; ++k) {
                    sum += weight_im2col[i][k] * input_im2col[k][j];
                }
                result_im2col[i][j] = sum;
            }
        }
        // printf("matrix multiplication done. \n");

        // convert the result matrix back to output format
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    int number_index = out_channel;
                    int window_index = out_y * output_width + out_x;
                    int32_t acc = result_im2col[number_index][window_index];
                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int8_t>(acc);
                }
            }
        }
        unsigned my_finish = perf_get_mcycle();
        my_cycles += (my_finish - my_start);

        // printf("convert the result matrix back to output format done. \n");
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
