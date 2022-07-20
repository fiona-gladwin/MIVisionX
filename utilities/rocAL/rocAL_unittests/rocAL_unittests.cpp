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

#define DISPLAY 1
//#define RANDOMBBOXCROP

using namespace std::chrono;

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
    int test_case = 3; // For Rotate
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

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                             : RocalImageColor::ROCAL_COLOR_U8;

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

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    switch (reader_type)
    {
#if 0
        case 1: //image_partial decode
        {
            std::cout << ">>>>>>> Running PARTIAL DECODE" << std::endl;
            rocalCreateLabelReader(handle, path);
            input1 = rocalFusedJpegCrop(handle, path, color_format, num_threads, false, false);
        }
        break;
#endif
        case 2: //coco detection
        {
            std::cout << ">>>>>>> Running COCO READER" << std::endl;
            char const *json_path = "";
            if (strcmp(json_path, "") == 0)
            {
                std::cout << "\n json_path has to be set in rocal_unit test manually";
                exit(0);
            }
            rocalCreateCOCOReader(handle, json_path, true);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, false, true, false);
            else
                input1 = rocalJpegCOCOFileSource(handle, path, json_path, color_format, num_threads, false, true, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
#if 0
        case 3: //coco detection partial
        {
            std::cout << ">>>>>>> Running COCO READER PARTIAL" << std::endl;
            char const *json_path = "";
            if (strcmp(json_path, "") == 0)
            {
                std::cout << "\n json_path has to be set in rocal_unit test manually";
                exit(0);
            }
            rocalCreateCOCOReader(handle, json_path, true);
#if defined RANDOMBBOXCROP
            rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop);
#endif
            input1 = rocalJpegCOCOFileSourcePartial(handle, path, json_path, color_format, num_threads, false, true, false);
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
            rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
            else
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        }
        break;
    }

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = 300;
    int resize_h = 110;
    RocalTensor image1, image2;

    switch (test_case)
    {
    case 0:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Crop Mirror Normalize " << std::endl;
        image1 = rocalCropMirrorNormalize(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0, mean, sdev, true);
    }
    break;
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBrightness" << std::endl;
        image1 = rocalBrightness(handle, input1,tensorLayout, tensorOutputType, true);
    }
    break;
    case 2:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGamma" << std::endl;
        image1 = rocalGamma(handle, input1, tensorLayout, tensorOutputType, true);
    }
    break;
    case 3:
    {
         std::cout << ">>>>>>> Running "
                  << "rocalResize" << std::endl;
        image1 = rocalResize(handle, input1, tensorLayout, tensorOutputType, 3,300 , 300, 0,true);

    }
    break;
    case 4:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalExposure" << std::endl;
        image1 = rocalExposure(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 5:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorCast" << std::endl;
        image1 = rocalColorCast(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 17:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalspatter" << std::endl;
        image1 = rocalSpatter(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 24:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwist" << std::endl;
        image1 = rocalColorTwist(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 7:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlur" << std::endl;
        image1 = rocalBlur(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 8:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCrop" << std::endl;
        image1 = rocalCrop(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0,true);
    break;
    }
    case 9:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalContrast" << std::endl;
        image1 = rocalContrast(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 10:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Resize Mirror Normalize " << std::endl;
        image1 = rocalResizeMirrorNormalize(handle, input1, tensorLayout, tensorOutputType, 3,resize_w , resize_h, 0, mean, sdev,true);
        break;
    }
    case 6:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFlip" << std::endl;
        image1 = rocalFlip(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
    case 11:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalVignette" << std::endl;
        image1 = rocalVignette(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
    case 12:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalJitter" << std::endl;
        image1 = rocalJitter(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
    case 13:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalNoise" << std::endl;
        image1 = rocalNoise(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 14:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlend" << std::endl;
        image2 = rocalGamma(handle, input1, tensorLayout, tensorOutputType, true);
        image1 = rocalBlend(handle, input1,image2, tensorLayout, tensorOutputType, true);

    }
break;
case 15:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGridmask" << std::endl;
        image1 = rocalGridmask(handle, input1, tensorLayout, tensorOutputType, true);

    }
break;
case 16:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalwarp_affine" << std::endl;
        image1 = rocalWarpAffine(handle, input1, tensorLayout, tensorOutputType, true);

    }
break;
case 18:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorJitter" << std::endl;
        image1 = rocalColorJitter(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 19:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnow" << std::endl;
        image1 = rocalSnow(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;

case 20:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalfog" << std::endl;
        image1 = rocalFog(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 21:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSaturation" << std::endl;
        image1 = rocalSaturation(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 22:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalHue" << std::endl;
        image1 = rocalHue(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;

case 23:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFisheye" << std::endl;
        image1 = rocalFisheye(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
    case 25:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTemperature" << std::endl;
        image1 = rocalColorTemperature(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 26:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorPixelate" << std::endl;
        image1 = rocalPixelate(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 27:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalLensCorrection" << std::endl;
        image1 = rocalLensCorrection(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 28:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRain" << std::endl;
        image1 = rocalRain(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 29:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRotate" << std::endl;
        image1 = rocalRotate(handle, input1, tensorLayout, tensorOutputType, true,300, 300,0);
    }
break;
case 30:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalresizecrop" << std::endl;
        image1 = rocalCropResize(handle, input1, tensorLayout, tensorOutputType, true,300, 300,0);
    }
break;
// case 31:
//     {
//         std::cout << ">>>>>>> Running "
//                   << "rocalresizecropmirror" << std::endl;
//         image1 = rocalResizeCropMirror(handle, input1, tensorLayout, tensorOutputType, true,300, 300,0);
//     }
// break;
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
std::cerr<<"hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n\n\n";
    RocalTensorList output_tensor_list;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ?  ((tensorOutputType == RocalTensorOutputType::ROCAL_FP32) ? CV_8UC3 : CV_8UC3) : CV_8UC1);
    std::cerr<<"\n\ncv_color_format"<<cv_color_format<<"\n"<<color_format<<"\n"<<tensorOutputType;
    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        std::cerr<<"hello\n\n\n";
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
                // rocALTensorList* labels = rocalGetImageLabels(handle);

                // for(int i = 0; i < labels->size(); i++)
                // {
                //     int * labels_buffer = (int *)(labels->at(i)->buffer());
                //     std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
                // }
                // int img_size = rocalGetImageNameLen(handle, image_name_length);
                // char img_name[img_size];
                // numOfClasses = num_of_classes;
                // int label_one_hot_encoded[inputBatchSize * numOfClasses];
                // rocalGetImageName(handle, img_name);
                // if (num_of_classes != 0)
                // {
                //     rocalGetOneHotImageLabels(handle, label_one_hot_encoded, numOfClasses,0);
                // }
                // std::cerr << "\nPrinting image names of batch: " << img_name<<"\n";
                // for (unsigned int i = 0; i < inputBatchSize; i++)
                // {
                //     std::cerr<<"\t Printing label_id : " << label_id[i] << std::endl;
                //     if(num_of_classes != 0)
                //     {
                //         std::cout << "One Hot Encoded labels:"<<"\t";
                //         for (int j = 0; j < numOfClasses; j++)
                //         {
                //             int idx_value = label_one_hot_encoded[(i*numOfClasses)+j];
                //             if(idx_value == 0)
                //                 std::cout << idx_value;
                //             else
                //             {
                //                 std::cout << idx_value;
                //             }
                //         }
                //     }
                //     std::cout << "\n";
                // }
            }
            break;
            case 2: //detection pipeline
            {
                std::cerr<<"detection\n\n";
                rocALTensorList* bbox_labels = rocalGetBoundingBoxLabel(handle);
                rocALTensorList* bbox_coords = rocalGetBoundingBoxCords(handle);
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
#if 0
            case 3: // keypoints pipeline
            {
                int size = inputBatchSize;
                RocalJointsData *joints_data;
                rocalGetJointsDataPtr(handle, &joints_data);
                for (int i = 0; i < size; i++)
                {
                    std::cout << "ImageID: " << joints_data->image_id_batch[i] << std::endl;
                    std::cout << "AnnotationID: " << joints_data->annotation_id_batch[i] << std::endl;
                    std::cout << "ImagePath: " << joints_data->image_path_batch[i] << std::endl;
                    std::cout << "Center: " << joints_data->center_batch[i][0] << " " << joints_data->center_batch[i][1] << std::endl;
                    std::cout << "Scale: " << joints_data->scale_batch[i][0] << " " << joints_data->scale_batch[i][1] << std::endl;
                    std::cout << "Score: " << joints_data->score_batch[i] << std::endl;
                    std::cout << "Rotation: " << joints_data->rotation_batch[i] << std::endl;

                    for (int k = 0; k < 17; k++)
                    {
                    std::cout << "x : " << joints_data->joints_batch[i][k][0] << " , y : " << joints_data->joints_batch[i][k][1] << " , v : " << joints_data->joints_visibility_batch[i][k][0] << std::endl;
                    }
                }
            }
            break;
#endif
            default:
            {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }
        std::cerr<<"endddddd\n\n";
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
            int h = output_tensor_list->at(idx)->info().max_dims().at(1) * output_tensor_list->at(idx)->info().dims().at(0);
            int w = output_tensor_list->at(idx)->info().max_dims().at(0);
            mat_input = cv::Mat(h, w, cv_color_format);
            mat_output = cv::Mat(h, w, cv_color_format);
        //      switch (tensorOutputType)
        // {
        // case ROCAL_FP32:
        //     rocalCopyToTensorOutput(handle, mat_input.data, h * w * p);
        //     break;
        // case ROCAL_UINT8:
        //     rocalCopyToTensorOutput(handle, mat_input.data, h * w * p);
        //     break;
        // }
        if(0)
        {
            float * tempo= ( float *)(output_tensor_list->at(idx)->buffer());
            for(int j=0;j<(h*w*3);j++)
            {
            mat_input.data[j] =(unsigned char)(int)(tempo[j]*255.0f);
            }
        }
            mat_input.data = (unsigned char *)(output_tensor_list->at(idx)->buffer());
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
