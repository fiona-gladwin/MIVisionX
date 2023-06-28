/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_AUGMENTATION_H
#define MIVISIONX_ROCAL_API_AUGMENTATION_H
#include "rocal_api_types.h"

RocalTensor  ROCAL_API_CALL
rocalSequenceRearrange(RocalContext p_context, RocalTensor input, std::vector<unsigned int>& new_order, bool is_output);

/// Accepts U8 and RGB24 inputs
/// \param context Rocal context
/// \param input Input Rocal tensor
/// \param is_output is the output tensor part of the graph output
/// \param alpha controls contrast of the image
/// \param beta controls brightness of the image
/// \param tensor_output_layout the layout of the output tensor
/// \param tensor_output_datatype the data type of the output tensor
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor ROCAL_API_CALL rocalCopyTensor(RocalContext context, RocalTensor input, bool is_output);

extern "C" RocalTensor ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context,
                                            RocalTensor p_input,
                                            unsigned dest_width, unsigned dest_height,
                                            std::vector<float> &mean,
                                            std::vector<float> &std_dev,
                                            bool is_output,
                                            RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                            std::vector<unsigned> max_size = {},
                                            unsigned resize_shorter = 0,
                                            unsigned resize_longer = 0,
                                            RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                            RocalIntParam mirror = NULL,
                                            RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                            RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context Rocal context
/// \param input Input Rocal tensor
/// \param crop_height crop width of the image
/// \param crop_width crop height of the image
/// \param start_x x-coordinate, start of the input image to be cropped
/// \param start_y y-coordinate, start of the input image to be cropped
/// \param mean mean value (specified for each channel) for image normalization
/// \param std_dev standard deviation value (specified for each channel) for image normalization
/// \param is_output is the output tensor part of the graph output
/// \param mirror controls horizontal flip of the image
/// \param tensor_output_layout the layout of the output tensor
/// \param tensor_output_type the data type of the output tensor
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalTensor input,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  std::vector<float> &mean,
                                                                  std::vector<float> &std_dev,
                                                                  bool is_output,
                                                                  RocalIntParam mirror = NULL,
                                                                  RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                                  RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor  ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input, bool is_output,
                                                RocalFloatParam crop_width = NULL,
                                                RocalFloatParam crop_height = NULL,
                                                RocalFloatParam crop_pox_x = NULL,
                                                RocalFloatParam crop_pos_y = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C"  RocalTensor  ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalTensor  input,
                                                      unsigned crop_width,
                                                      unsigned crop_height,
                                                      bool is_output,
                                                      float crop_pox_x,
                                                      float crop_pos_y,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor  ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalTensor input,
                                                        unsigned crop_width,
                                                        unsigned crop_height,
                                                        bool output,
                                                        RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                        RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                   unsigned dest_width, unsigned dest_height,
                                                   bool is_output,
                                                   RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                   std::vector<unsigned> max_size = {},
                                                   unsigned resize_shorter = 0,
                                                   unsigned resize_longer = 0,
                                                   RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                   RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                   RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context,
                                                      RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_datatype = ROCAL_UINT8);

extern "C" RocalTensor ROCAL_API_CALL rocalNormalize(RocalContext p_context,
                                                     RocalTensor p_input,
                                                     RocalTensorOutputType rocal_tensor_output_type,
                                                      bool is_output, bool batch = false,
                                                     std::vector<int> axes = {},
                                                     float mean = 0.0f, float std_dev = 0.0f,
                                                     float scale = 0.0f, float shift = 0.0f,
                                                     int ddof = 0, float epsilon = 0);

extern "C" RocalTensor ROCAL_API_CALL rocalPad(RocalContext p_context,
                                               RocalTensor p_input,
                                               RocalTensorOutputType rocal_tensor_output_type,
                                               bool is_output,
                                               float fill_value = 0.0f,
                                               std::vector<int>axes = {},
                                               std::vector<int>align = {});

extern "C" RocalTensor ROCAL_API_CALL rocalToDecibels(RocalContext p_context,
                                                      RocalTensor p_input,
                                                      RocalTensorLayout rocal_tensor_layout,
                                                      RocalTensorOutputType rocal_tensor_output_type,
                                                      bool is_output,
                                                      float cut_off_DB = -200.0,
                                                      float multiplier = 10.0,
                                                      float magnitude_reference = 0.0);

extern "C" RocalTensor ROCAL_API_CALL rocalPreEmphasisFilter(RocalContext p_context,
                                                            RocalTensor p_input,
                                                            RocalTensorOutputType rocal_tensor_output_type,
                                                            bool is_output,
                                                            RocalFloatParam p_preemph_coeff = NULL,
                                                            RocalAudioBorderType preemph_border_type = RocalAudioBorderType::CLAMP);

extern "C" RocalTensor ROCAL_API_CALL rocalSpectrogram(RocalContext p_context,
                                                       RocalTensor p_input,
                                                       RocalTensorOutputType rocal_tensor_output_type,
                                                       bool is_output,
                                                       std::vector<float>& window_fn,
                                                       bool center_windows = true,
                                                       bool reflect_padding = true,
                                                       RocalSpectrogramLayout spec_layout = RocalSpectrogramLayout::FT,
                                                       int power = 2, // Can be 1 or 2
                                                       int nfft_size = 2048,
                                                       int window_length = 512,
                                                       int window_step = 256);

extern "C" std::pair<RocalTensor,RocalTensor> ROCAL_API_CALL rocalNonSilentRegion(RocalContext p_context,
                                                           RocalTensor p_input,
                                                           bool is_output,
                                                           float cut_off_db = -0.60,
                                                           float reference_power = 0.0,
                                                           int reset_interval = 8192,
                                                           int window_length = 2048);

extern "C" RocalTensor ROCAL_API_CALL rocalMelFilterBank(RocalContext p_context,
                                                         RocalTensor p_input,
                                                         bool is_output,
                                                         float freq_high = 0.0,
                                                         float freq_low = 0.0,
                                                         RocalMelScaleFormula mel_formula = RocalMelScaleFormula::SLANEY,
                                                         int nfilter = 128,
                                                         bool normalize = true,
                                                         float sample_rate = 4410);

extern "C" RocalTensor ROCAL_API_CALL rocalSlice(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalTensor anchor,
                                                RocalTensor shape ,
                                                std::vector<float> fill_values = {},
                                                std::vector<unsigned> axes = {},
                                                bool normalized_anchor = false,
                                                bool normalized_shape = false,
                                                RocalOutOfBoundsPolicy policy = RocalOutOfBoundsPolicy::ERROR);

extern "C" RocalTensor ROCAL_API_CALL rocalTensorMulScalar(RocalContext p_context,
                                                           RocalTensor p_input,
                                                           bool is_output,
                                                           RocalTensorLayout rocal_tensor_layout,
                                                           RocalTensorOutputType rocal_tensor_output_type,
                                                           float freq_high = 0.0);

extern "C" RocalTensor ROCAL_API_CALL rocalNormalDistribution(RocalContext p_context,
                                                              RocalTensor p_input,
                                                              bool is_output,
                                                              float mean = 0.0,
                                                              float stddev = 0.0);

extern "C" RocalTensor ROCAL_API_CALL rocalTensorAddTensor(RocalContext p_context,
                                                           RocalTensor p_input1,
                                                           RocalTensor p_input2,
                                                           bool is_output,
                                                           RocalTensorLayout rocal_tensor_layout, // TODO : Swetha - not required for audio data - Check on this
                                                           RocalTensorOutputType rocal_tensor_output_type);
                                                        
extern "C" RocalTensor ROCAL_API_CALL rocalUniformDistribution(RocalContext p_context,
                                                              RocalTensor p_input,
                                                              bool is_output,
                                                              std::vector<float> &range);

extern "C" RocalTensor ROCAL_API_CALL rocalResample(RocalContext p_context,
                                                    RocalTensor p_input,
                                                    RocalTensor p_input_resample_rate,
                                                    RocalTensorOutputType rocal_tensor_output_type,
                                                    bool is_output,
                                                    float sample_hint);
#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H