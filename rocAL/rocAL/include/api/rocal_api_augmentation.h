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
extern "C" RocalTensor ROCAL_API_CALL rocalColorTemperature(RocalContext p_context,
                                                            RocalTensor p_input,
                                                            RocalTensorLayout rocal_tensor_layout,
                                                            RocalTensorOutputType rocal_tensor_output_type,
                                                            bool is_output,
                                                            RocalIntParam p_adjust_value=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalRain(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_percentage=NULL,
        RocalIntParam p_width=NULL,
        RocalIntParam p_height=NULL,
        RocalFloatParam p_tranparency=NULL);



extern "C" RocalTensor ROCAL_API_CALL rocalLensCorrection(
                                                            RocalContext p_context,
                                                            RocalTensor p_input,
                                                            RocalTensorLayout rocal_tensor_layout,
                                                            RocalTensorOutputType rocal_tensor_output_type,
                                                            bool is_output,
                                                            RocalFloatParam p_strength=NULL,
                                                            RocalFloatParam p_zoom=NULL);
                                
extern "C" RocalTensor ROCAL_API_CALL rocalRotate(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                
                                                unsigned int dest_width=0,
                                                unsigned int dest_height=0,
                                                int outputformat=0,
                                                RocalFloatParam p_angle=NULL);
                

extern "C" RocalTensor ROCAL_API_CALL rocalResizeCrop(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                
                                                unsigned int dest_width=0,
                                                unsigned int dest_height=0,
                                                int outputformat=0,
                                                RocalFloatParam p_x1=NULL,
                                                RocalFloatParam p_x2=NULL,
                                                RocalFloatParam p_x3=NULL,
                                                RocalFloatParam p_x4=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalCropResize(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                
                                                unsigned int dest_width=0,
                                                unsigned int dest_height=0,
                                                int outputformat=0,
                                                RocalFloatParam p_x1=NULL,
                                                RocalFloatParam p_x2=NULL,
                                                RocalFloatParam p_x3=NULL,
                                                RocalFloatParam p_x4=NULL);

// extern "C" RocalTensor ROCAL_API_CALL rocalResizeCrop(RocalContext p_context,
//                                                 RocalTensor p_input,
//                                                 RocalTensorLayout rocal_tensor_layout,
//                                                 RocalTensorOutputType rocal_tensor_output_type,
//                                                 bool is_output,
                                                
//                                                 unsigned int dest_width=0,
//                                                 unsigned int dest_height=0,
//                                                 int outputformat=0,
//                                                 RocalFloatParam p_x1=NULL,
//                                                 RocalFloatParam p_x2=NULL,
//                                                 RocalFloatParam p_x3=NULL,
//                                                 RocalFloatParam p_x4=NULL,
//                                                 RocalFloatParam p_mirror=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalBlur(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalIntParam p_sdev=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalPixelate(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output);


extern "C" RocalTensor ROCAL_API_CALL rocalFisheye(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output);


extern "C" RocalTensor ROCAL_API_CALL rocalVignette(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_sdev=NULL);
                                        
extern "C" RocalTensor ROCAL_API_CALL rocalSnow(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_snow=NULL);
    
extern "C" RocalTensor ROCAL_API_CALL rocalFog(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_fog=NULL);


extern "C" RocalTensor ROCAL_API_CALL rocalHue(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_hue=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalJitter(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_sdev=NULL);
            
extern "C" RocalTensor ROCAL_API_CALL rocalSaturation(RocalContext p_context,
                                                RocalTensor p_input,
                                                RocalTensorLayout rocal_tensor_layout,
                                                RocalTensorOutputType rocal_tensor_output_type,
                                                bool is_output,
                                                RocalFloatParam p_sat=NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout, RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalNoise(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL, RocalFloatParam hue = NULL, RocalFloatParam sat = NULL, int seed=11110);

extern "C" RocalTensor ROCAL_API_CALL rocalBlend(RocalContext context, RocalTensor input,RocalTensor input1, RocalTensorLayout rocal_tensor_layout, RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input, RocalTensorLayout rocal_tensor_layout, RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL);



extern "C" RocalTensor ROCAL_API_CALL rocalContrast(RocalContext context, RocalTensor input, RocalTensorLayout rocal_tensor_layout,
                                                    RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                    RocalFloatParam c_fator = NULL, RocalFloatParam c_center = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalExposure(RocalContext context, RocalTensor input, RocalTensorLayout rocal_tensor_layout, RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL);
                                    
extern "C" RocalTensor ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                                  RocalTensorLayout rocal_tensor_layout,
                                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                                  unsigned resize_depth,
                                                                  unsigned resize_height,
                                                                  unsigned resize_width,
                                                                  int interpolation_type,
                                                                  bool is_output);

extern "C" RocalTensor ROCAL_API_CALL rocalWarpAffine(RocalContext p_context,
                                                    RocalTensor p_input,
                                                    RocalTensorLayout rocal_tensor_layout,
                                                    RocalTensorOutputType rocal_tensor_output_type,
                                                    bool is_output,
                                                    RocalFloatParam x0=0,
                                                    RocalFloatParam x1=0,
                                                    RocalFloatParam y0=0,
                                                    RocalFloatParam y1=0,
                                                    RocalFloatParam o0=0,
                                                    RocalFloatParam o1=0,
                                                    int interpolation_type=0);


extern "C" RocalTensor ROCAL_API_CALL rocalFlip(RocalContext context, RocalTensor input, RocalTensorLayout rocal_tensor_layout, RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam h_flag = NULL, RocalFloatParam v_flag = NULL);



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

extern "C" RocalTensor ROCAL_API_CALL rocalColorCast( RocalContext p_context,
                                                      RocalTensor p_input,
                                                      RocalTensorLayout rocal_tensor_layout,
                                                      RocalTensorOutputType rocal_tensor_output_type,
                                                      bool is_output,
                                                      RocalFloatParam R_value = NULL,
                                                      RocalFloatParam G_value = NULL,
                                                      RocalFloatParam B_value = NULL,
                                                      RocalFloatParam alpha_tensor = NULL);


extern "C" RocalTensor ROCAL_API_CALL rocalSpatter(RocalContext p_context,RocalTensor p_input,
                                                   RocalTensorLayout rocal_tensor_layout,
                                                   RocalTensorOutputType rocal_tensor_output_type,
                                                   bool is_output,
                                                   int R_value=0,
                                                   int G_value=0,
                                                   int B_value=0);

extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context,
                                                      RocalTensor input,
                                                      RocalTensorLayout rocal_tensor_layout,
                                                      RocalTensorOutputType rocal_tensor_output_type,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL);


extern "C" RocalTensor ROCAL_API_CALL rocalColorJitter(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL, RocalFloatParam hue = NULL, RocalFloatParam sat = NULL);
                    
extern "C" RocalTensor ROCAL_API_CALL rocalGridmask(RocalContext p_context,
                                                    RocalTensor p_input,
                                                    RocalTensorLayout rocal_tensor_layout,
                                                    RocalTensorOutputType rocal_tensor_output_type,
                                                    bool is_output,
                                                    int tileWidth=40,
                                                    float gridRatio=0.5,
                                                    float gridvalue=0.6,
                                                    unsigned int x=0,
                                                    unsigned int y=0);
extern "C" RocalTensor ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input,
                                                                  RocalTensorLayout rocal_tensor_layout,
                                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                                  unsigned crop_depth,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  float start_z,
                                                                  bool is_output);





/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
// extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
//                                                    RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalTensor input,
                                                        float alpha, float beta,
                                                        bool is_output);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
// extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input,
//                                               bool is_output,
//                                               RocalFloatParam alpha = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalTensor input, float alpha, bool is_output);

extern "C" RocalTensor ROCAL_API_CALL rocalCopyTensor(RocalContext context, RocalTensor input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalNopTensor(RocalContext context, RocalTensor input, bool is_output);


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


#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H