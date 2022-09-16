/*
MIT License

Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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


#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "rocal_api.h"

using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE 
#endif

#define DISPLAY
using namespace std::chrono;

template <typename T> 
void convert_float_to_uchar_buffer(T * input_float_buffer, unsigned char * output_uchar_buffer, size_t data_size)
{
    for(size_t i = 0; i < data_size; i++)
    {
        output_uchar_buffer[i] = (unsigned char)(*(input_float_buffer + i) * 255);
    }
}

void convert_nchw_to_nhwc(unsigned char * input_chw, unsigned char * output_hwc, int n, int h, int w, int c)
{
    int image_stride = h * w * c;
    int channel_stride = h * w;
    for(size_t idx = 0; idx < n; idx++)
    {
        unsigned char * input_image = input_chw + idx * image_stride;
        unsigned char * plane_R = input_image;
        unsigned char * plane_G = input_image + channel_stride;
        unsigned char * plane_B = input_image + channel_stride;

        unsigned char * output_image = output_hwc + idx * image_stride;
        for(size_t i = 0; i < h; i++)
        {
            for(size_t j = 0; j < w; j++)
            {
                *output_image++ = *plane_R;
                *output_image++ = *plane_G;
                *output_image++ = *plane_B;
                plane_R++;
                plane_G++;
                plane_B++;
            }
        }
    }
}


int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        printf( "Usage: rocal_external_source <image_dataset_folder/video_file> <processing_device=1/cpu=0>  decode_width decode_height batch_size gray_scale/rgb/rgbplanar display_on_off \n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    bool display = 0;// Display the images
    //int aug_depth = 1;// how deep is the augmentation tree
    int rgb = 2;// process color images
    int decode_width = 224;
    int decode_height = 224;
    int inputBatchSize = 64;
    int prefetch_queue_depth = 3;
    bool processing_device = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        inputBatchSize = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);


    std::cerr << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;
    RocalImageColor color_format = RocalImageColor::ROCAL_COLOR_RGB_PLANAR;
    if (rgb == 0) 
      color_format = RocalImageColor::ROCAL_COLOR_U8;
    else if (rgb == 1)
      color_format = RocalImageColor::ROCAL_COLOR_RGB24;
    else if (rgb == 2)
      color_format = RocalImageColor::ROCAL_COLOR_RGB_PLANAR;

    auto handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Could not create the Rocal contex\n";
        return -1;
    }


    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor input1;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;


    input1 = rocalJpegExternalFileSource(handle, folderPath1,  color_format, true, decode_width, decode_height, false, ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height, RocalDecoderType::ROCAL_DECODER_TJPEG);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "JPEG source could not initialize : "<<rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
        DIR *_src_dir;
    struct dirent *_entity;
    std::vector<std::string> file_names;
    if((_src_dir = opendir (folderPath1)) == nullptr)
    {
            std::cerr<<"\n ERROR: Failed opening the directory at "<<folderPath1;
            exit(0);
    }

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        std::string file_path = folderPath1;
        // file_path.append("/");
        file_path.append(_entity->d_name);
        std::cerr<<"\n _entity->d_name:: "<<file_path;
        file_names.push_back(file_path);
    }
    std::cerr<<"\n file names count :: "<<file_names.size();


#if 0
    const size_t num_values = 3;
    float values[num_values] = {0,10,135};
    double frequencies[num_values] = {1, 5, 5};

    RocalFloatParam rand_angle =   rocalCreateFloatRand( values , frequencies, num_values);
    // Creating successive blur nodes to simulate a deep branch of augmentations
    RocalTensor image2 = rocalCropResize(handle, image0, resize_w, resize_h, false, rand_crop_area);;
    for(int i = 0 ; i < aug_depth; i++)
    {
        image2 = rocalBlurFixed(handle, image2, 17.25, (i == (aug_depth -1)) ? true:false );
    }


    RocalTensor image4 = rocalColorTemp(handle, image0, false, color_temp_adj);

    RocalTensor image5 = rocalWarpAffine(handle, image4, false);

    RocalTensor image6 = rocalJitter(handle, image5, false);

    rocalVignette(handle, image6, true);



    RocalTensor image7 = rocalPixelate(handle, image0, false);

    RocalTensor image8 = rocalSnow(handle, image0, false);

    RocalTensor image9 = rocalBlend(handle, image7, image8, false);

    RocalTensor image10 = rocalLensCorrection(handle, image9, false);

    rocalExposure(handle, image10, true);
#else
  // uncomment the following to add augmentation if needed
    RocalTensor image0;
    int resize_w = 224, resize_h = 224;
    image0 = input1;
  // just do one augmentation to test
    image0 = rocalResize(handle, input1, resize_w, resize_h, true);
#endif

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    std::cerr << "<Remaining_images, augmentation_count> " << rocalGetRemainingImages(handle) << std::endl;
    // TODO:prefetch and feed data for the input pipeline
    for (int i=0; i< prefetch_queue_depth; i++) {
      
    }


    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    bool eos = false;
    int total_images = file_names.size();
    RocalTensorList output_tensor_list;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ?  ((tensorOutputType == RocalTensorOutputType::ROCAL_FP32) ? CV_32FC3 : CV_8UC3) : CV_8UC1);

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        std::vector<std::string> input_images;
        std::vector<int> label;
        for(int i = 0; i < inputBatchSize; i++)
        {
            input_images.push_back(file_names.back());
            file_names.pop_back();
            std::cerr<<"\n Input images :: "<<input_images[i];
        }
        if((file_names.size()) == 0)
        {
           eos = true;
        }
        if(index <= (total_images / inputBatchSize))
        {
            std::cerr<<"\n************************** Gonna process Batch *************************"<<index;
            rocalExternalSourceFeedInput(handle, input_images, label, NULL, {}, {}, 1, 1, RocalExtSourceMode (0), RocalTensorLayout (0), eos);
        }
        if(rocalRun(handle) != 0)
            break;
        uint pipeline_type = 1;
        switch(pipeline_type)
        {
            case 1: //classification pipeline
            {
                RocalTensorList labels = rocalGetImageLabels(handle);

                for(int i = 0; i < labels->size(); i++)
                {
                    int * labels_buffer = (int *)(labels->at(i)->buffer());
                    std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
                }
            }
            break;
            case 2: //detection pipeline
            {
                RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
                RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
                for(int i = 0; i < bbox_labels->size(); i++)
                {
                    int * labels_buffer = (int *)(bbox_labels->at(i)->buffer());
                    float *bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
                    std::cerr << "\n>>>>> BBOX LABELS : ";
                    for(int j = 0; j < bbox_labels->at(i)->info().dims().at(0); j++)
                        std::cerr << labels_buffer[j] << " ";
                    std::cerr << "\n>>>>> BBOXX : " <<bbox_coords->at(i)->info().dims().at(0) << " : \n";
                    for(int j = 0, j4 = 0; j < bbox_coords->at(i)->info().dims().at(0); j++, j4 = j * 4)
                        std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";

                }
            }
            break;
            // Cases for Mask & Keypoints is not added
            default:
            {
                std::cerr << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        output_tensor_list = rocalGetOutputTensors(handle);
        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        cv::Mat mat_input;
        cv::Mat mat_output;
        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            int h = output_tensor_list->at(idx)->info().max_shape().at(1) * output_tensor_list->at(idx)->info().dims().at(0);
            int w = output_tensor_list->at(idx)->info().max_shape().at(0);
            mat_input = cv::Mat(h, w, cv_color_format);
            mat_output = cv::Mat(h, w, cv_color_format);
            unsigned char *out_buffer;
            if(output_tensor_list->at(idx)->info().data_type() == RocalTensorDataType::FP32)
            {
                float * out_f_buffer;
                if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
                {
                    out_f_buffer = (float *)malloc(output_tensor_list->at(idx)->info().data_size());
                    output_tensor_list->at(idx)->copy_data(out_f_buffer);
                }
                else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                    out_f_buffer = (float *)output_tensor_list->at(idx)->buffer();

                out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size() / 4);
                convert_float_to_uchar_buffer(out_f_buffer, out_buffer, output_tensor_list->at(idx)->info().data_size() / 4);
                // if(out_f_buffer != nullptr) free(out_f_buffer);
            }
            if(output_tensor_list->at(idx)->info().data_type() == RocalTensorDataType::FP16)
            {
                half * out_f16_buffer;
                if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
                {
                    out_f16_buffer = (half *)malloc(output_tensor_list->at(idx)->info().data_size());
                    output_tensor_list->at(idx)->copy_data(out_f16_buffer);
                }
                else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                    out_f16_buffer = (half *)output_tensor_list->at(idx)->buffer();

                out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size() / 2);
                convert_float_to_uchar_buffer(out_f16_buffer, out_buffer, output_tensor_list->at(idx)->info().data_size() / 2);
                // if(out_f16_buffer != nullptr) free(out_f16_buffer);
            }
            else
            {
                if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
                {
                    out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size());
                    output_tensor_list->at(idx)->copy_data(out_buffer);
                }
                else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                    out_buffer = (unsigned char *)(output_tensor_list->at(idx)->buffer());
            }

            if(output_tensor_list->at(idx)->info().layout() == RocalTensorlayout::NCHW)
            {
                // cv::Mat mat_input_nchw = cv::Mat(cv_color_format, h, w);
                // mat_input_nchw = (unsigned char *)out_buffer;
                // cv::transposeND(mat_input_nchw, {0, 3, 1, 2}, mat_input); // Can be enabled only with OpenCV 4.6.0
                convert_nchw_to_nhwc(out_buffer, mat_input.data, output_tensor_list->at(idx)->info().dims().at(0), output_tensor_list->at(idx)->info().dims().at(2),
                                     output_tensor_list->at(idx)->info().dims().at(3), output_tensor_list->at(idx)->info().dims().at(1));            
            }
            else
                mat_input.data = (unsigned char *)out_buffer;

            mat_input.copyTo(mat_output(cv::Rect(0, 0, w, h)));
            std::string outName = "external_source_output";
            std::string out_filename = outName + ".png";   // in case the user specifies non png filename
            if (display)
                out_filename = outName + std::to_string(index) + std::to_string(idx) + ".png";   // in case the user specifies non png filename

            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
            {
                cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                cv::imwrite(out_filename, mat_color, compression_params);
            }
            else
            {
                cv::imwrite(out_filename, mat_output, compression_params);
            }
            // if(out_buffer != nullptr) free(out_buffer);
        }
        mat_input.release();
        mat_output.release();
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cerr << "Load     time " << rocal_timing.load_time << std::endl;
    std::cerr << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cerr << "Process  time " << rocal_timing.process_time << std::endl;
    std::cerr << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cerr << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    return 0;
}

