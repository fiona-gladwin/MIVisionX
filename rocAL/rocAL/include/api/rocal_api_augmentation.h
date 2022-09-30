/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
rocalSequenceRearrange(
            RocalContext p_context, RocalTensor input, unsigned int* new_order, 
            unsigned int  new_sequence_length, unsigned int sequence_length, bool is_output );

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalCopyTensor(RocalContext context, RocalTensor input, bool is_output);

extern "C" RocalTensor ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, 
                                            RocalTensor p_input,
                                            RocalTensorLayout rocal_tensor_layout,
                                            RocalTensorOutputType rocal_tensor_output_type,
                                            unsigned resize_depth,
                                            unsigned resize_height,
                                            unsigned resize_width,
                                            int interpolation_type,
                                            std::vector<float> &mean,
                                            std::vector<float> &std_dev,
                                            bool is_output,
                                             RocalIntParam mirror = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalTensor input,
                                                                  RocalTensorLayout rocal_tensor_layout,
                                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                                  unsigned crop_depth,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  float start_z,
                                                                  std::vector<float> &mean,
                                                                  std::vector<float> &std_dev,
                                                                  bool is_output,
                                                                  RocalIntParam mirror = NULL);

extern "C" RocalTensor  ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input, bool is_output,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                RocalFloatParam crop_width = NULL,
                                                RocalFloatParam crop_height = NULL,
                                                RocalFloatParam crop_depth = NULL,
                                                RocalFloatParam crop_pox_x = NULL,
                                                RocalFloatParam crop_pos_y = NULL,
                                                RocalFloatParam crop_pos_z = NULL);

extern "C"  RocalTensor  ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalTensor  input,
                                                      RocalTensorLayout rocal_tensor_layout,
                                                      RocalTensorOutputType rocal_tensor_output_type,
                                                      unsigned crop_width,
                                                      unsigned crop_height,
                                                      unsigned crop_depth,
                                                      bool is_output,
                                                      float crop_pox_x,
                                                      float crop_pos_y,
                                                      float crop_pos_z);

extern "C" RocalTensor  ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalTensor input,
                                                        RocalTensorLayout rocal_tensor_layout,
                                                        RocalTensorOutputType rocal_tensor_output_type,
                                                        unsigned crop_width,
                                                        unsigned crop_height,
                                                        unsigned crop_depth,
                                                        bool output);

extern "C" RocalTensor ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                  RocalTensorLayout rocal_tensor_layout,
                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                   unsigned dest_width, unsigned dest_height,
                                                   bool is_output,
                                                   RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                   std::vector<unsigned> max_size = {},
                                                   unsigned resize_shorter = 0,
                                                   unsigned resize_longer = 0,
                                                   RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION);

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param size
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalResizeShorter(RocalContext context, RocalTensor input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                unsigned size,
                                                bool is_output );

extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context,
                                                      RocalTensor input,
                                                      RocalTensorLayout rocal_tensor_layout,
                                                      RocalTensorOutputType rocal_tensor_output_type,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL);


#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H