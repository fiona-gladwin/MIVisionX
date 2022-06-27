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


#include "node_gamma.h"
#include "node_exposure.h"
#include "node_resize.h"
#include "node_color_cast.h"
#include "node_brightness.h"
#include "node_crop_mirror_normalize.h"
#include "node_copy.h"
#include "node_nop.h"
#include "meta_node_crop_mirror_normalize.h"
#include "node_spatter.h"
#include "node_color_twist.h"
#include "node_crop.h"
#include "node_contrast.h"
#include "node_resize_mirror_normalize.h"
#include "node_flip.h"
#include "node_color_jitter.h"
#include "node_noise.h"



#include "commons.h"
#include "context.h"
#include "rocal_api.h"

RocalTensor ROCAL_API_CALL
rocalBrightness(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try
    {

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<BrightnessTensorNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

//Noise

RocalTensor ROCAL_API_CALL
rocalNoise(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta,
        RocalFloatParam p_hue,
        RocalFloatParam p_sat,
        int seed)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);

    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<NoiseTensorNode>({input}, {output})->init(alpha, beta, hue ,sat,seed);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


//contrast
RocalTensor ROCAL_API_CALL
rocalContrast(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam c_factor,
        RocalFloatParam c_center)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto contrast_factor = static_cast<FloatParam*>(c_factor);
    auto contrast_center = static_cast<FloatParam*>(c_center);
    try
    {

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<ContrastTensorNode>({input}, {output})->init(contrast_factor, contrast_center);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


//colorcast
RocalTensor
ROCAL_API_CALL rocalColorCast(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam R_value,
        RocalFloatParam G_value,
        RocalFloatParam B_value,
        RocalFloatParam alpha_tensor)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto red = static_cast<FloatParam*>(R_value);
    auto green = static_cast<FloatParam*>(G_value);
    auto blue = static_cast<FloatParam*>(B_value);
    auto alpha = static_cast<FloatParam*>(alpha_tensor);

    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<ColorCastTensorNode>({input}, {output})->init(red, green, blue ,alpha,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposure(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_alpha)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<ExposureTensorNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor
ROCAL_API_CALL rocalCrop(RocalContext p_context, 
                                            RocalTensor p_input,
                                            RocalTensorLayout rocal_tensor_layout,

                                            RocalTensorOutputType rocal_tensor_output_type,
                                            unsigned crop_depth,
                                            unsigned crop_height,
                                            unsigned crop_width,
                                            float start_x,
                                            float start_y,
                                            float start_z,
                                            bool is_output)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                std::cerr<<"\n Setting output type to FP16";
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                std::cerr<<"\n Setting output type to UINT8";
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        rocALTensorInfo output_info = input->info();
        // output_info.max_width(crop_width);
        // output_info.max_height(crop_height);
        // output_info.format(op_tensorFormat);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();
        context->master_graph->add_node<CropTensorNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}




RocalTensor ROCAL_API_CALL
rocalSpatter(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        int R_value,
        int G_value,
        int B_value)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<SpatterTensorNode>({input}, {output})->init(R_value,G_value,B_value,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalColorTwist(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta,
        RocalFloatParam p_hue,
        RocalFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);

    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<ColorTwistTensorNode>({input}, {output})->init(alpha, beta, hue ,sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorJitter(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta,
        RocalFloatParam p_hue,
        RocalFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);

    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<ColorJitterTensorNode>({input}, {output})->init(alpha, beta, hue ,sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


// RocalTensor ROCAL_API_CALL
// rocalGamma(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output,
//         RocalFloatParam p_alpha)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     auto alpha = static_cast<FloatParam*>(p_alpha);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<GammaNode>({input}, {output})->init(alpha);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalGammaFixed(
//         RocalContext p_context,
//         RocalTensor p_input,
//         float alpha,
//         bool is_output)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     try
//     {
//         if(!input || !context)
//             THROW("Null values passed as input")

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<GammaNode>({input}, {output})->init(alpha);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalBrightness(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output,
//         RocalFloatParam p_alpha,
//         RocalFloatParam p_beta)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     auto alpha = static_cast<FloatParam*>(p_alpha);
//     auto beta = static_cast<FloatParam*>(p_beta);
//     try
//     {

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<BrightnessNode>({input}, {output})->init(alpha, beta);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalBrightnessFixed(
//         RocalContext p_context,
//         RocalTensor p_input,
//         float alpha,
//         float beta,
//         bool is_output)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     try
//     {
//         if(!input || !context)
//             THROW("Null values passed as input")

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<BrightnessNode>({input}, {output})->init(alpha, beta);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

RocalTensor
ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, RocalTensorLayout rocal_tensor_layout,
                                    RocalTensorOutputType rocal_tensor_output_type, unsigned crop_depth, unsigned crop_height,
                                    unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
                                    std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    // float mean_acutal = 0, std_actual = 0; // Mean of vectors
    // for(unsigned i = 0; i < mean.size(); i++)
    // {
    //     mean_acutal += mean[i];
    //     std_actual  += std_dev[i];
    // }
    // mean_acutal /= mean.size();
    // std_actual /= std_dev.size();
    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                std::cerr<<"RocalTensorlayout::NHWC";
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                std::cerr<<"RocalTensorlayout::NCHW";

                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }

        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                std::cerr<<"\n Setting output type to FP16";
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                std::cerr<<"\n Setting output type to UINT8";
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        rocALTensorInfo output_info = input->info();
        // output_info.max_width(crop_width);
        // output_info.max_height(crop_height);
        // output_info.format(op_tensorFormat);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();
        context->master_graph->add_node<CropMirrorNormalizeTensorNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y, mean,
                                                                                        std_dev , mirror,layout );
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}


RocalTensor  ROCAL_API_CALL
rocalCopyTensor(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


// RocalTensor  ROCAL_API_CALL
// rocalNopTensor(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);
//         context->master_graph->add_tensor_node<NopNode>({input}, {output});
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }


RocalTensor ROCAL_API_CALL
rocalGamma(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    try
    {

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<GammaTensorNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

//resize
RocalTensor
ROCAL_API_CALL rocalResize(RocalContext p_context, 
                                            RocalTensor p_input,
                                            RocalTensorLayout rocal_tensor_layout,

                                            RocalTensorOutputType rocal_tensor_output_type,
                                            unsigned resize_depth,
                                            unsigned resize_height,
                                            unsigned resize_width,
                                            int interpolation_type,
                                            bool is_output)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || resize_width == 0 || resize_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }
        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                std::cerr<<"\n Setting output type to FP16";
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                std::cerr<<"\n Setting output type to UINT8";
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        rocALTensorInfo output_info = input->info();
        
        // Need to just set dims depending on NCHW or NHWC
        output_info.set_width(resize_width);
        output_info.set_height(resize_height);
        
        std::cerr<<"\n\n\nwidth     "<<output_info.get_width();
        std::cerr << " OUT W & H : " << output_info.max_width() << output_info.max_height() << "\n";
        std::cerr << " IN W & H : " << input->info().max_width() << input->info().max_height() << "\n";
        // output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();
        context->master_graph->add_node<ResizeTensorNode>({input}, {output})->init( interpolation_type, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}



//resizemirrornormalize

//resize
RocalTensor
ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, 
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
                                            RocalIntParam p_mirror)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);

    RocalTensorlayout op_tensorFormat;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || resize_width == 0 || resize_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        switch(rocal_tensor_layout)
        {
            case 0:
                op_tensorFormat = RocalTensorlayout::NHWC;
                layout=0;
                break;
            case 1:
                op_tensorFormat = RocalTensorlayout::NCHW;
                layout=1;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
        }
        switch(rocal_tensor_output_type)
        {
            case ROCAL_FP32:
                std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RocalTensorDataType::FP32;
                break;
            case ROCAL_FP16:
                std::cerr<<"\n Setting output type to FP16";
                op_tensorDataType = RocalTensorDataType::FP16;
                break;
            case ROCAL_UINT8:
                std::cerr<<"\n Setting output type to UINT8";
                op_tensorDataType = RocalTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
        }
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        rocALTensorInfo output_info = input->info();
        // Need to just set dims depending on NCHW or NHWC
        output_info.set_width(resize_width);
        output_info.set_height(resize_height);
        
        std::cerr<<"\n\n\nwidth     "<<output_info.get_width();
        std::cerr << " OUT W & H : " << output_info.max_width() << output_info.max_height() << "\n";
        std::cerr << " IN W & H : " << input->info().max_width() << input->info().max_height() << "\n";
        // output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();
        context->master_graph->add_node<ResizeMirrorNormalizeTensorNode>({input}, {output})->init( interpolation_type, mean,std_dev , mirror, layout);
        std::cerr<<"checking2222222222\n";
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}



RocalTensor ROCAL_API_CALL
rocalFlip(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalIntParam h_flag,
        RocalIntParam v_flag)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto horizontal_flag = static_cast<IntParam*>(h_flag);
    auto vertical_flag = static_cast<IntParam*>(v_flag);
    try
    {

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_node<FlipTensorNode>({input}, {output})->init(horizontal_flag, vertical_flag);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
