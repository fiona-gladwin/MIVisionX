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
#include <unistd.h>
#include <vector>
#include<string>

#include "rocal_api.h"

#include "opencv2/opencv.hpp"
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

#define DISPLAY 0
//#define RANDOMBBOXCROP

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

int test(int test_case, int reader_type, int pipeline_type, const char *path, const char *outName, int rgb, int gpu, int width, int height,int num_of_classes, int display_all);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT)
    {
        printf("Usage: rocal_unittests reader-type pipeline-type=1(classification)2(detection)3(keypoints) <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    int reader_type = atoi(argv[++argIdx]);
    int pipeline_type = atoi(argv[++argIdx]);
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;

    int rgb = 1; // process color images
    bool gpu = 1;
    int test_case = 0; // For Rotate
    int num_of_classes = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_of_classes = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display_all = atoi(argv[++argIdx]);

    test(test_case, reader_type, pipeline_type, path, outName, rgb, gpu, width, height, num_of_classes, display_all);

    return 0;
}

int test(int test_case, int reader_type, int pipeline_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all)
{
    size_t num_threads = 1;
    unsigned int inputBatchSize = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                             gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                             1);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

#if defined RANDOMBBOXCROP
    bool all_boxes_overlap = true;
    bool no_crop = false;
#endif

    RocalTensor input1;
    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;
    RocalMetaData metadata_output;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    switch (reader_type)
    {
        case 1: //image_partial decode
        {
            std::cout << ">>>>>>> Running PARTIAL DECODE" << std::endl;
            rocalCreateLabelReader(handle, path);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f/4, 4.0f/3};
            input1 = rocalFusedJpegCrop(handle, path, color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 2: //coco detection
        {
            std::cout << ">>>>>>> Running COCO READER" << std::endl;
            char const *json_path = "";
            if (strcmp(json_path, "") == 0)
            {
                std::cout << "\n json_path has to be set in rocal_unit test manually";
                exit(0);
            }
            metadata_output = rocalCreateCOCOReader(handle, json_path, true, false);
            // metadata_output = rocalCreateCOCOReader(handle, json_path, true, true);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, false, true, false);
            else
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, true, true, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
        case 3: //coco detection partial
        {
            std::cout << ">>>>>>> Running COCO READER PARTIAL" << std::endl;
            char const *json_path = "";
            if (strcmp(json_path, "") == 0)
            {
                std::cout << "\n json_path has to be set in rocal_unit test manually";
                exit(0);
            }
            rocalCreateCOCOReader(handle, json_path, true, false);
#if defined RANDOMBBOXCROP
            rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop);
#endif
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f/4, 4.0f/3};
            input1 = rocalJpegCOCOFileSourcePartial(handle, path, json_path, color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 4: //tf classification
        {
            std::cout << ">>>>>>> Running TF CLASSIFICATION READER" << std::endl;
            char key1[25] = "image/encoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            input1 = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
        case 5: //tf detection
        {
            std::cout << ">>>>>>> Running TF DETECTION READER" << std::endl;
            char key1[25] = "image/encoded";
            char key2[25] = "image/object/class/label";
            char key3[25] = "image/object/class/text";
            char key4[25] = "image/object/bbox/xmin";
            char key5[25] = "image/object/bbox/ymin";
            char key6[25] = "image/object/bbox/xmax";
            char key7[25] = "image/object/bbox/ymax";
            char key8[25] = "image/filename";
            rocalCreateTFReaderDetection(handle, path, true, key2, key3, key4, key5, key6, key7, key8);
            input1 = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
        case 6: //caffe classification
        {
            std::cout << ">>>>>>> Running CAFFE CLASSIFICATION READER" << std::endl;
            rocalCreateCaffeLMDBLabelReader(handle, path);
            input1 = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
        case 7: //caffe detection
        {
            std::cout << ">>>>>>> Running CAFFE DETECTION READER" << std::endl;
            rocalCreateCaffeLMDBReaderDetection(handle, path);
            input1 = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;

        case 8: //caffe2 classification
        {
            std::cout << ">>>>>>> Running CAFFE2 CLASSIFICATION READER" << std::endl;
            rocalCreateCaffe2LMDBLabelReader(handle, path, true);
            input1 = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
        case 9: //caffe2 detection
        {
            std::cout << ">>>>>>> Running CAFFE2 DETECTION READER" << std::endl;
            rocalCreateCaffe2LMDBReaderDetection(handle, path, true);
            input1 = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
#if 0

        case 10: //coco reader keypoints
        {
            std::cout << ">>>>>>> Running COCO KEYPOINTS READER" << std::endl;
            char const *json_path = "";
            if (strcmp(json_path, "") == 0)
            {
                std::cout << "\n json_path has to be set in rocal_unit test manually";
                exit(0);
            }
            float sigma = 3.0;
            rocalCreateCOCOReaderKeyPoints(handle, json_path, true, sigma, (unsigned)width, (unsigned)height);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, false, true, false);
            else
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, false, true, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
#endif

        default: //image pipeline
        {
            std::cout << ">>>>>>> Running IMAGE READER" << std::endl;
            metadata_output = rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
            else
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, true, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
    }

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = 100, resize_h = 300; // height and width

    RocalTensor image1;

    switch (test_case)
    {
    case 0:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Crop Mirror Normalize " << std::endl;
        image1 = rocalCropMirrorNormalize(handle, input1, resize_h, resize_w, 0, 0, mean, sdev, true, NULL, tensorLayout, tensorOutputType);
        break;
    }
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "Brightness" << std::endl;
        image1 = rocalBrightness(handle, input1, true, NULL, NULL, tensorLayout, tensorOutputType);
    }
    break;
    case 3:
    {
         std::cout << ">>>>>>> Running "
                  << "rocalResize" << std::endl;
        image1 = rocalResize(handle, input1, 0, 0, true, ROCAL_SCALING_MODE_NOT_SMALLER, {}, 256, 0, ROCAL_LINEAR_INTERPOLATION, tensorLayout, tensorOutputType);
    }
    break;
    case 26:
    {
         std::cout << ">>>>>>> Running "
                  << "rocalcrop" << std::endl;
        // image1 = rocalCrop(handle, input1, true, tensorLayout, tensorOutputType, resize_w, resize_h, 0, 0, 0);

    }
    break;
    case 50:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwist" << std::endl;
        image1 = rocalColorTwist(handle, input1, true, NULL, NULL, NULL, NULL, tensorLayout, tensorOutputType);
    }
    break;
    case 10:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Resize Mirror Normalize " << std::endl;
        image1 = rocalResizeMirrorNormalize(handle, input1, resize_w , resize_h, mean, sdev, true, ROCAL_SCALING_MODE_DEFAULT,
                                            {}, 0, 0, ROCAL_LINEAR_INTERPOLATION, NULL, tensorLayout, tensorOutputType);
        break;
    }
    default:
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    RocalTensorList output_tensor_list;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ?  ((tensorOutputType == RocalTensorOutputType::ROCAL_FP32) ? CV_32FC3 : CV_8UC3) : CV_8UC1);

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
        int label_id[inputBatchSize];
        int numOfClasses = 0;
        int image_name_length[inputBatchSize];
        switch(pipeline_type)
        {
            case 1: //classification pipeline
            {
                if (gpu == 1)
                    rocalGetImageLabels(handle, label_id, ROCAL_MEMCPY_TO_HOST);
                else
                    rocalGetImageLabels(handle, label_id);
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                char img_name[img_size];
                numOfClasses = num_of_classes;
                int label_one_hot_encoded[inputBatchSize * numOfClasses];
                rocalGetImageName(handle, img_name);
                if (num_of_classes != 0)
                {
                    rocalGetOneHotImageLabels(handle, label_one_hot_encoded, numOfClasses,0);
                }
                std::cerr << "\nPrinting image names of batch: " << img_name<<"\n";
                for (unsigned int i = 0; i < inputBatchSize; i++)
                {
                    std::cerr<<"\t Printing label_id : " << label_id[i] << std::endl;
                    if(num_of_classes != 0)
                    {
                        std::cout << "One Hot Encoded labels:"<<"\t";
                        for (int j = 0; j < numOfClasses; j++)
                        {
                            int idx_value = label_one_hot_encoded[(i*numOfClasses)+j];
                            if(idx_value == 0)
                                std::cout << idx_value;
                            else
                            {
                                std::cout << idx_value;
                            }
                        }
                    }
                    std::cout << "\n";
                }
            }
            break;
            case 2: //detection pipeline
            {
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                char img_name[img_size];
                rocalGetImageName(handle, img_name);
                std::cerr << "\nPrinting image names of batch: " << img_name;
                int bb_label_count[inputBatchSize];
                int size = rocalGetBoundingBoxCount(handle, bb_label_count);
                for (int i = 0; i < (int)inputBatchSize; i++)
                    std::cerr << "\n Number of box:  " << bb_label_count[i];
                int bb_labels[size];
                rocalGetBoundingBoxLabel(handle, bb_labels);
                float bb_coords[size * 4];
                rocalGetBoundingBoxCords(handle, bb_coords);
                int img_sizes_batch[inputBatchSize * 2];
                rocalGetImageSizes(handle, img_sizes_batch);
                for (int i = 0; i < (int)inputBatchSize; i++)
                {
                    std::cout<<"\nwidth:"<<img_sizes_batch[i*2];
                    std::cout<<"\nHeight:"<<img_sizes_batch[(i*2)+1];
                }
            }
            break;
            default:
            {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
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

            std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
            if (display_all)
                out_filename = std::string(outName) + std::to_string(index) + std::to_string(idx) + ".png";   // in case the user specifies non png filename

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
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    return 0;
}
