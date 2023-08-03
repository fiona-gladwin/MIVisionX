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

/*!
 * \brief Rearranges the order of the frames in the sequences with respect to new_order.
 * Frames can be repeated or dropped in the new_order.
 * \ingroup group_rocal_augmentations
 * \note: Accepts U8 and RGB24 input.
 * \param context context for the pipeline.
 * \param input Input Rocal Tensor
 * \param new_order represents the new order of the frames in the sequence in the range [0, sequence_length)
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSequenceRearrange(RocalContext p_context, RocalTensor p_input,
                                                              std::vector<unsigned int>& new_order,
                                                              bool is_output);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context context for the pipeline.
 * \param input Input Rocal Tensor
 * \param dest_width output width
 * \param dest_height ouput Height
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \param scaling_mode The resize scaling_mode to resize the image.
 * \param max_size Limits the size of the resized tensor.
 * \param resize_shorter The length of the shorter dimension of the image.
 * \param resize_longer The length of the larger dimension of the image.
 * \param interpolation_type The type of interpolation to be used for resize.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                  unsigned dest_width, unsigned dest_height,
                                                  bool is_output,
                                                  RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                  std::vector<unsigned> max_size = {},
                                                  unsigned resize_shorter = 0,
                                                  unsigned resize_longer = 0,
                                                  RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal Tensor
 * \param dest_width output width
 * \param dest_height output height
 * \param mean The channel mean values
 * \param std_dev The channel standard deviation values
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \param mirror Parameter to enable horizontal flip for output tensor.
 * \param scaling_mode The resize scaling_mode to resize the image.
 * \param max_size Limits the size of the resized tensor.
 * \param resize_shorter The length of the shorter dimension of the image.
 * \param resize_longer The length of the larger dimension of the image.
 * \param interpolation_type The type of interpolation to be used for resize.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, RocalTensor p_input, unsigned dest_width,
                                                                 unsigned dest_height, std::vector<float> &mean, std::vector<float> &std_dev,
                                                                 bool is_output,
                                                                 RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                                 std::vector<unsigned> max_size = {}, unsigned resize_shorter = 0,
                                                                 unsigned resize_longer = 0,
                                                                 RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                                 RocalIntParam mirror = NULL,
                                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal Tensor
 * \param dest_width output width
 * \param dest_height output height
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \param area Target area for the crop
 * \param x_center_drift Horizontal shift of the crop center from its original position in the input tensor
 * \param y_center_drift Vertical shift of the crop center from its original position in the input tensor
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropResize(RocalContext context, RocalTensor input,
                                                      unsigned dest_width, unsigned dest_height,
                                                      bool is_output,
                                                      RocalFloatParam area = NULL,
                                                      RocalFloatParam aspect_ratio = NULL,
                                                      RocalFloatParam x_center_drift = NULL,
                                                      RocalFloatParam y_center_drift = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param dest_width output width
 * \param dest_height output height
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \param area Target area for the crop
 * \param x_center_drift Horizontal shift of the crop center from its original position in the input image
 * \param y_center_drift Vertical shift of the crop center from its original position in the input image
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropResizeFixed(RocalContext context, RocalTensor input,
                                                           unsigned dest_width, unsigned dest_height,
                                                           bool is_output,
                                                           float area, float aspect_ratio,
                                                           float x_center_drift, float y_center_drift,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input. The output tensor dimension can be set to new values allowing the rotated image to fit,
 * otherwise; the tensor is cropped to fit the result.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal Tensor
 * \param is_output True: the output tensor is needed by user and will be copied to output buffers using the data
 * transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in
 * using it directly. This option allows certain optimizations to be achieved.
 * \param angle Rocal parameter defining the rotation angle value in degrees.
 * \param dest_width output width
 * \param dest_height output height
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return Returns a new tensor that keeps the result.
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRotate(RocalContext context, RocalTensor input, bool is_output,
                                                  RocalFloatParam angle = NULL,  unsigned dest_width = 0,
                                                  unsigned dest_height = 0,
                                                  RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input. The output tensor dimension can be set to new values allowing the rotated tensor to fit,
 * otherwise; the tensor is cropped to fit the result.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal Tensor
 * \param dest_width output width
 * \param dest_height output height
 * \param is_output Is the output tensor part of the graph output
 * \param angle The rotation angle value in degrees.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return Returns a new tensor that keeps the result.
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRotateFixed(RocalContext context, RocalTensor input, float angle,
                                                       bool is_output, unsigned dest_width = 0, unsigned dest_height = 0,
                                                       RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                       RocalTensorLayout output_layout = ROCAL_NONE,
                                                       RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param alpha controls contrast of the image
 * \param beta controls brightness of the image
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
                                                      RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param alpha controls contrast of the image
 * \param beta controls brightness of the image
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalTensor input,
                                                           float alpha, float beta,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param gamma gamma value for the image.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input,
                                                 bool is_output,
                                                 RocalFloatParam gamma = NULL,
                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param gamma gamma value for the image.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalTensor input,
                                                      float gamma,
                                                      bool is_output,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

 /*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param contrast_factor parameter representing the contrast factor for the contrast operation
 * \param contrast_center parameter representing the contrast center for the contrast operation
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalContrast(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam contrast_factor = NULL, RocalFloatParam contrast_center = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param contrast_factor  parameter representing the contrast factor for the contrast operation
 * \param contrast_center  parameter representing the contrast center for the contrast operation
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalContrastFixed(RocalContext context, RocalTensor input,
                                                         float contrast_factor, float contrast_center,
                                                         bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param horizonal_flag  determines whether the input tensor should be flipped horizontally
 * \param vertical_flag  determines whether the input tensor should be flipped vertically
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFlip(RocalContext context, RocalTensor input, bool is_output,
                                                RocalIntParam horizonal_flag = NULL, RocalIntParam vertical_flag = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param horizonal_flag  determines whether the input tensor should be flipped horizontally
 * \param vertical_flag  determines whether the input tensor should be flipped vertically
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFlipFixed(RocalContext context, RocalTensor input,
                                                     int horizonal_flag, int vertical_flag, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param kernel_size size ofthr kernel used for blurring
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlur(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalIntParam kernel_size = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param kernel_size size of the kernel used for blurring
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlurFixed(RocalContext context, RocalTensor input,
                                                     int kernel_size, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input1 Input1 Rocal tensor
 * \param input2 Input2 Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param ratio Rocal parameter defining the blending ratio, should be between 0.0 and 1.0
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlend(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                 bool is_output,
                                                 RocalFloatParam ratio = NULL,
                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input1 Input1 Rocal tensor
 * \param input2 Input2 Rocal tensor
 * \param ratio Float value defining the blending ratio, should be between 0.0 and 1.0.
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlendFixed(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                      float ratio, bool is_output,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param x0 float parameter representing the coefficient of affine tensor matrix
 * \param x1 float parameter representing the coefficient of affine tensor matrix
 * \param y0 float parameter representing the coefficient of affine tensor matrix
 * \param y1 float parameter representing the coefficient of affine tensor matrix
 * \param o0 float parameter representing the coefficient of affine tensor matrix
 * \param o1 float parameter representing the coefficient of affine tensor matrix
 * \param dest_height output height
 * \param dest_width output width
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalWarpAffine(RocalContext context, RocalTensor input, bool is_output,
                                                      unsigned dest_height = 0, unsigned dest_width = 0,
                                                      RocalFloatParam x0 = NULL, RocalFloatParam x1 = NULL,
                                                      RocalFloatParam y0= NULL, RocalFloatParam y1 = NULL,
                                                      RocalFloatParam o0 = NULL, RocalFloatParam o1 = NULL,
                                                      RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param x0 float parameter representing the coefficient of affine tensor matrix
 * \param x1 float parameter representing the coefficient of affine tensor matrix
 * \param y0 float parameter representing the coefficient of affine tensor matrix
 * \param y1 float parameter representing the coefficient of affine tensor matrix
 * \param o0 float parameter representing the coefficient of affine tensor matrix
 * \param o1 float parameter representing the coefficient of affine tensor matrix
 * \param dest_height output height
 * \param dest_width output width
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalWarpAffineFixed(RocalContext context, RocalTensor input, float x0, float x1,
                                                           float y0, float y1, float o0, float o1, bool is_output,
                                                           unsigned int dest_height = 0, unsigned int dest_width = 0,
                                                           RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFishEye(RocalContext context, RocalTensor input, bool is_output,
                                                   RocalTensorLayout output_layout = ROCAL_NONE,
                                                   RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param sdev standard deviation for the vignette effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVignette(RocalContext context, RocalTensor input,
                                                    bool is_output, RocalFloatParam sdev = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param sdev standard deviation for the vignette effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVignetteFixed(RocalContext context, RocalTensor input,
                                                         float sdev, bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param kernel_size kernel size used for the jitter effect
 * \param seed seed value for the random number generator
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJitter(RocalContext context, RocalTensor input,
                                                  bool is_output,
                                                  RocalIntParam kernel_size = NULL,
                                                  int seed = 0,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param kernel_size kernel size used for the jitter effect
 * \param seed seed value for the random number generator
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJitterFixed(RocalContext context, RocalTensor input,
                                                       int kernel_size, bool is_output, int seed = 0,
                                                       RocalTensorLayout output_layout = ROCAL_NONE,
                                                       RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param noise_prob probability of applying the Salt and Pepper noise.
 * \param salt_prob probability of applying salt noise
 * \param salt_val specifies the value of the salt noise
 * \param pepper_val specifies the value of the pepper noise
 * \param sdev standard deviation for SnPNoise
 * \param seed seed value for the random number generator
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoise(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam noise_prob = NULL, RocalFloatParam salt_prob = NULL,
                                                    RocalFloatParam salt_val = NULL, RocalFloatParam pepper_val = NULL,
                                                    int seed = 0,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param noise_prob probability of applying the Salt and Pepper noise.
 * \param salt_prob probability of applying salt noise
 * \param salt_val specifies the value of the salt noise
 * \param pepper_val specifies the value of the pepper noise
 * \param sdev standard deviation for SnPNoise
 * \param seed seed value for the random number generator
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoiseFixed(RocalContext context, RocalTensor input,
                                                         float noise_prob, float salt_prob,
                                                         float salt_val, float pepper_val,
                                                         bool is_output, int seed = 0,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param snow Float param representing the intensity of snow effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnow(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam snow = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param snow Float param representing the intensity of snow effect
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnowFixed(RocalContext context, RocalTensor input,
                                                     float snow, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param rain_value parameter represents the intensity of rain effect
 * \param rain_width parameter represents the width of the rain effect
 * \param rain_heigth parameter represents the width of the rain effect
 * \param rain_transparency parameter represents the transperancy of the rain effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRain(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam rain_value = NULL,
                                                RocalIntParam rain_width = NULL,
                                                RocalIntParam rain_height = NULL,
                                                RocalFloatParam rain_transparency = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param rain_value parameter represents the intensity of rain effect
 * \param rain_width parameter represents the width of the rain effect
 * \param rain_heigth parameter represents the width of the rain effect
 * \param rain_transparency parameter represents the transperancy of the rain effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRainFixed(RocalContext context, RocalTensor input,
                                                     float rain_value,
                                                     int rain_width,
                                                     int rain_height,
                                                     float rain_transparency,
                                                     bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param adjustment color temperature adjustment value
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTemp(RocalContext context, RocalTensor input,
                                                     bool is_output,
                                                     RocalIntParam adjustment = NULL,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param adjustment color temperature adjustment value
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \param is_output is the output tensor part of the graph output
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTempFixed(RocalContext context, RocalTensor input,
                                                          int adjustment, bool is_output,
                                                          RocalTensorLayout output_layout = ROCAL_NONE,
                                                          RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param fog_value parameter representing the intensity of fog effect
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFog(RocalContext context, RocalTensor input,
                                               bool is_output,
                                               RocalFloatParam fog_value = NULL,
                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param fog_value parameter representing the intensity of fog effect
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFogFixed(RocalContext context, RocalTensor input,
                                                    float fog_value, bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param strength parameter representing the strength of the lens correction.
 * \param zoom parameter representing the zoom factor of the lens correction.
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalLensCorrection(RocalContext context, RocalTensor input, bool is_output,
                                                          RocalFloatParam strength = NULL,
                                                          RocalFloatParam zoom = NULL,
                                                          RocalTensorLayout output_layout = ROCAL_NONE,
                                                          RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param strength parameter representing the strength of the lens correction.
 * \param zoom parameter representing the zoom factor of the lens correction.
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalLensCorrectionFixed(RocalContext context, RocalTensor input,
                                                               float strength, float zoom, bool is_output,
                                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalPixelate(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);


/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param exposure_factor exposure adjustment factor
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalExposure(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam exposure_factor = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param exposure_factor exposure adjustment factor
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalExposureFixed(RocalContext context, RocalTensor input,
                                                         float exposure_factor, bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param hue hue adjustment value in degrees
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalHue(RocalContext context, RocalTensor input,
                                               bool is_output,
                                               RocalFloatParam hue = NULL,
                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param hue hue adjustment value in degrees
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalHueFixed(RocalContext context, RocalTensor input,
                                                    float hue,
                                                    bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param saturation saturation adjustment value
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSaturation(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam saturation = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param saturation saturation adjustment value
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSaturationFixed(RocalContext context, RocalTensor input,
                                                           float saturation, bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCopy(RocalContext context, RocalTensor input, bool is_output);

/*! \brief Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalNop(RocalContext context, RocalTensor input, bool is_output);

/*! \brief Accepts RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param alpha parameter that controls the brightness of an image
 * \param beta parameter that helps in tuning the color balance of an image
 * \param hue parameter that adjusts the hue of an image
 * \param sat parameter that controls the intensity of colors
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param alpha parameter that controls the brightness of an image
 * \param beta parameter that helps in tuning the color balance of an image
 * \param hue parameter that adjusts the hue of an image
 * \param sat parameter that controls the intensity of colors
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \param is_output is the output tensor part of the graph output
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwistFixed(RocalContext context, RocalTensor input,
                                                           float alpha,
                                                           float beta,
                                                           float hue,
                                                           float sat,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param crop_height crop width of the tensor
 * \param crop_width crop height of the tensor
 * \param start_x x-coordinate, start of the input tensor to be cropped
 * \param start_y y-coordinate, start of the input tensor to be cropped
 * \param mean mean value (specified for each channel) for tensor normalization
 * \param std_dev standard deviation value (specified for each channel) for tensor normalization
 * \param is_output is the output tensor part of the graph output
 * \param mirror controls horizontal flip of the tensor
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalTensor input,
                                                               unsigned crop_height,
                                                               unsigned crop_width,
                                                               float start_x,
                                                               float start_y,
                                                               std::vector<float> &mean,
                                                               std::vector<float> &std_dev,
                                                               bool is_output,
                                                               RocalIntParam mirror = NULL,
                                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param crop_height crop width of the tensor
 * \param crop_width crop height of the tensor
 * \param crop_depth crop depth of the tensor
 * \param crop_pox_x x-coordinate, start of the input tensor to be cropped
 * \param crop_pox_y y-coordinate, start of the input tensor to be cropped
 * \param crop_pox_z z-coordinate, start of the input tensor to be cropped
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input, bool is_output,
                                                RocalFloatParam crop_width = NULL,
                                                RocalFloatParam crop_height = NULL,
                                                RocalFloatParam crop_depth = NULL,
                                                RocalFloatParam crop_pox_x = NULL,
                                                RocalFloatParam crop_pos_y = NULL,
                                                RocalFloatParam crop_pos_z = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param crop_height crop width of the tensor
 * \param crop_width crop height of the tensor
 * \param crop_depth crop depth of the tensor
 * \param crop_pox_x x-coordinate, start of the input tensor to be cropped
 * \param crop_pox_y y-coordinate, start of the input tensor to be cropped
 * \param crop_pox_z z-coordinate, start of the input tensor to be cropped
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalTensor  input,
                                                     unsigned crop_width,
                                                     unsigned crop_height,
                                                     unsigned crop_depth,
                                                     bool is_output,
                                                     float crop_pox_x,
                                                     float crop_pos_y,
                                                     float crop_pos_z,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param crop_height crop width of the tensor
 * \param crop_width crop height of the tensor
 * \param crop_depth crop depth of the tensor
 * \param is_output is the output tensor part of the graph output
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalTensor input,
                                                           unsigned crop_width,
                                                           unsigned crop_height,
                                                           unsigned crop_depth,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param dest_height output height
 * \param dest_width output width
 * \param crop_h crop width of the tensor
 * \param crop_w crop height of the tensor
 * \param is_output is the output tensor part of the graph output
 * \param mirror controls horizontal flip of the tensor
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeCropMirrorFixed(RocalContext context, RocalTensor input,
                                                                 unsigned dest_width, unsigned dest_height,
                                                                 bool is_output,
                                                                 unsigned crop_h,
                                                                 unsigned crop_w,
                                                                 RocalIntParam mirror,
                                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param dest_height output height
 * \param dest_width output width
 * \param crop_height crop width of the tensor
 * \param crop_width crop height of the tensor
 * \param is_output is the output tensor part of the graph output
 * \param mirror controls horizontal flip of the tensor
 * \param output_layout the layout of the output tensor
 * \param output_type the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeCropMirror(RocalContext context, RocalTensor input,
                                                            unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RocalFloatParam crop_height = NULL,
                                                            RocalFloatParam crop_width = NULL, RocalIntParam mirror = NULL,
                                                            RocalTensorLayout output_layout = ROCAL_NONE,
                                                            RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs and outputs cropped Images, valid bounding boxes and labels
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param num_of_attempts maximum number of attempts the function will make to find a valid crop
 * \param crop_area_factor specifies the proportion of the input image to be included in the cropped region
 * \param crop_aspect_ratio specifies the aspect ratio of the cropped region
 * \param crop_pos_x specifies a specific horizontal position for the crop
 * \param crop_pos_y specifies a specific vertical position for the crop
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRandomCrop(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam crop_area_factor = NULL,
                                                      RocalFloatParam crop_aspect_ratio = NULL,
                                                      RocalFloatParam crop_pos_x = NULL,
                                                      RocalFloatParam crop_pos_y = NULL,
                                                      int num_of_attempts = 20,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Accepts U8 and RGB24 inputs and outputs cropped tensor, valid bounding boxes and labels
 * \ingroup group_rocal_augmentations
 * \param context Rocal context
 * \param input Input Rocal tensor
 * \param is_output is the output tensor part of the graph output
 * \param num_of_attempts he maximum number of attempts the function will make to find a valid crop
 * \param crop_area_factor specifies the proportion of the input image to be included in the cropped region
 * \param crop_aspect_ratio specifies the aspect ratio of the cropped region
 * \param crop_pos_x specifies a specific horizontal position for the crop
 * \param crop_pos_y specifies a specific vertical position for the crop
 * \param output_layout the layout of the output tensor
 * \param output_datatype the data type of the output tensor
 * \return
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSSDRandomCrop(RocalContext context, RocalTensor input,
                                                         bool is_output,
                                                         RocalFloatParam threshold = NULL,
                                                         RocalFloatParam crop_area_factor  = NULL,
                                                         RocalFloatParam crop_aspect_ratio = NULL,
                                                         RocalFloatParam crop_pos_x = NULL,
                                                         RocalFloatParam crop_pos_y = NULL,
                                                         int num_of_attempts = 20,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H
