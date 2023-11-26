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
// #include "cfu.h"
// #include "playground_util/print_params.h"
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

void im2col_cpu(float** src, const int& inHeight, const int& intWidth, const int& kHeight, const int& kWidth, float* srcIm2col) {
    const int outHeight = inHeight - kHeight + 1;
    const int outWidth = intWidth - kWidth + 1;
    int cnt = 0;
    for (int i = 0; i < kHeight; i++) {
        for (int j = 0; j < kWidth; j++) {
            // int id = i * kWidth + j;
            int ii = i;
            for (int x = 0; x < outHeight; x++) {
                int jj = j;
                for (int y = 0; y < outWidth; y++) {
                    srcIm2col[cnt] = src[ii][jj];
                    jj += 1;
                    cnt++;
                }
                ii += 1;
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
    my_print_conv_params(params, input_shape, filter_shape, output_shape);

    // Get parameters.
    const int32_t input_offset = params.input_offset;  // r = s(q - Z)
    // const int stride_width = params.stride_width;
    // const int stride_height = params.stride_height;
    // const int dilation_width_factor = params.dilation_width_factor;
    // const int dilation_height_factor = params.dilation_height_factor;
    // const int pad_width = params.padding_values.width;
    // const int pad_height = params.padding_values.height;
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
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    // Calculate the size of the column matrix for a single image.
    const int col_height = filter_height * filter_width * input_depth;
    const int col_width = output_height * output_width;
    // int8_t* col_data = new int8_t[col_height * col_width];
    // int8_t* col_data = (int8_t*)malloc(col_height * col_width * sizeof(int8_t));

    printf("col_height: %d, col_width: %d\n", col_height, col_width);
    // col_height: 9, col_width: 810

    int8_t input_im2col[int(1e6)];
    int8_t weight_im2col[int(1e6)];
    for (int batch = 0; batch < batches; ++batch) {
        // Perform im2col transformation for each image in the batch.
        // memset(input_data, 0, input_height * input_width * input_depth * sizeof(int8_t));
        // memset(weight_data, 0, filter_height * filter_width * input_depth * output_depth * sizeof(int8_t));
        // memset(result_data, 0, output_height * output_width * output_depth * sizeof(int32_t));

        // printf("batch: %d\n", batch);
        // im2col_cpu(filter_data, input_depth, input_height, input_width,
        //            filter_height, filter_width,
        //            pad_height, pad_width,
        //            stride_height, stride_width,
        //            dilation_height_factor, dilation_width_factor,
        //            weight_data);
        // printf("Im2Col done\n");

        for (int i = 0; i < input_depth; i++) {
            for (int j = 0; j < filter_height; j++) {
                for (int k = 0; k < filter_width; k++) {
                    weight_im2col[i * filter_height * filter_width + j * filter_width + k] =
                        filter_data[Offset(filter_shape, batch, j, k, i)];
                }
            }
        }

        int outHeight = input_height - filter_height + 1;
        int outWidth = input_width - filter_width + 1;
        im2col_cpu((float**)input_data, input_height, input_width, filter_height, filter_width, (float*)input_im2col);

        printf("Im2Col done\n");

        // Perform matrix multiplication by executing int8 FullyConnectedPerChannel()
        const int accum_depth = input_depth * filter_height * filter_width;
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[batch * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += filter_val * (input_val + input_offset);
            }
            if (bias_data) {
                acc += bias_data[out_c];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                                output_shift[out_c]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[out_c + output_depth * batch] = static_cast<int8_t>(acc);
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
    // const int dilation_width_factor = params.dilation_width_factor;
    // const int dilation_height_factor = params.dilation_height_factor;
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
                        // const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        const int in_y = in_y_origin + filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            // const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            const int in_x = in_x_origin + filter_x;

                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image =
                                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                (in_y < input_height);

                            if (!is_point_inside_image) {
                                continue;
                            }

                            // LR
                            unsigned my_start = perf_get_mcycle();
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
                            unsigned my_finish = perf_get_mcycle();
                            my_cycles += (my_finish - my_start);
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
