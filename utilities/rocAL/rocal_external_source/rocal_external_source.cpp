/*
MIT License

Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <dirent.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rocal_api.h"

#if USE_OPENCV_4
#define CV_RGB2BGR COLOR_RGB2BGR
#endif

using namespace cv;
using namespace std::chrono;

int main(int argc, const char **argv) {
  const int MIN_ARG_COUNT = 2;
  if (argc < MIN_ARG_COUNT) {
    printf(
        "Usage: rocal_external_source <image_dataset_folder> "
        "<processing_device=1/cpu=0>  decode_width decode_height batch_size "
        "gray_scale/rgb/rgbplanar display_on_off external_source_mode<external_file_mode=0/raw_compressed_mode=1/raw_uncompresses_mode=2>\n");
    return -1;
  }
  int argIdx = 0;
  const char *folderPath = argv[++argIdx];
  bool display = 1;              // Display the images
  int rgb = 1;                   // process color images
  int decode_width = 224;        // Decoding width
  int decode_height = 224;       // Decoding height
  int inputBatchSize = 2;        // Batch size
  bool processing_device = 0;    // CPU Processing
  int mode = 0;                  // File mode

  if (argc >= argIdx + MIN_ARG_COUNT) processing_device = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) decode_width = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) decode_height = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) inputBatchSize = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) rgb = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) display = atoi(argv[++argIdx]);

  if (argc >= argIdx + MIN_ARG_COUNT) mode = atoi(argv[++argIdx]);

  std::cerr << "\n Mode:: " << mode << std::endl;
  std::cerr << ">>> Running on " << (processing_device ? "GPU" : "CPU")
            << std::endl;
  RocalImageColor color_format = RocalImageColor::ROCAL_COLOR_RGB24;
  if (rgb == 0)
    color_format = RocalImageColor::ROCAL_COLOR_U8;
  else if (rgb == 1)
    color_format = RocalImageColor::ROCAL_COLOR_RGB24;
  else if (rgb == 2)
    color_format = RocalImageColor::ROCAL_COLOR_RGB_PLANAR;

  int channels = 3;
  if (rgb == 0) channels = 1;

  auto handle =
      rocalCreate(inputBatchSize,
                  processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU
                                    : RocalProcessMode::ROCAL_PROCESS_CPU,
                  0, 1);

  if (rocalGetStatus(handle) != ROCAL_OK) {
    std::cerr << "Could not create the Rocal contex\n";
    return -1;
  }

  /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
  RocalImage input1;
  std::vector<uint32_t> srcsize_height, srcsize_width;
  uint32_t maxheight = 0, maxwidth = 0;
  DIR *_src_dir;
  struct dirent *_entity;
  std::vector<std::string> file_names;
  std::vector<unsigned char *> input_buffer;
  if ((_src_dir = opendir(folderPath)) == nullptr) {
    std::cerr << "\n ERROR: Failed opening the directory at " << folderPath;
    exit(0);
  }

  while ((_entity = readdir(_src_dir)) != nullptr) {
    if (_entity->d_type != DT_REG) continue;

    std::string file_path = folderPath;
    file_path.append(_entity->d_name);
    file_names.push_back(file_path);
  }
  if (mode != 0) {
    if (mode == 1) {
      // Mode 1 is Raw uncompressed
      // srcsize_height and srcsize_width resized based on total file count
      srcsize_height.resize(file_names.size());
      srcsize_width.resize(file_names.size());
      for (uint32_t i = 0; i < file_names.size(); i++) {
        FILE *_current_fPtr;
        _current_fPtr = fopen(file_names[i].c_str(), "rb");  // Open the file,
        if (!_current_fPtr)  // Check if it is ready for reading
          return 0;
        fseek(_current_fPtr, 0,
              SEEK_END);  // Take the file read pointer to the end
        size_t _current_file_size = ftell(
            _current_fPtr);  // Check how many bytes are there between and the
                             // current read pointer position (end of the file)
        unsigned char *input_data = static_cast<unsigned char *>(
            malloc(sizeof(unsigned char) * _current_file_size));
        if (_current_file_size == 0) {  // If file is empty continue
          fclose(_current_fPtr);
          _current_fPtr = nullptr;
          return 0;
        }

        fseek(_current_fPtr, 0,
              SEEK_SET);  // Take the file pointer back to the start
        size_t actual_read_size = fread(input_data, sizeof(unsigned char),
                                        _current_file_size, _current_fPtr);
        input_buffer.push_back(input_data);
        srcsize_height[i] = actual_read_size; // It stored the actual file size
      }
    }
    else if (mode == 2) {
      // Mode 2 is raw un-compressed mode
      // srcsize_height and srcsize_width resized based on total file count
      srcsize_height.resize(file_names.size());
      srcsize_width.resize(file_names.size());
      // Calculate max size and max height
      for (uint32_t i = 0; i < file_names.size(); i++) {
        Mat image;
        image = imread(file_names[i], 1);
        if (image.empty()) {
          std::cout << "Could not read the image: " << file_names[i]
                    << std::endl;
          return 1;
        }
        srcsize_height[i] = image.rows;
        srcsize_width[i] = image.cols;
        if (maxheight < srcsize_height[i]) maxheight = srcsize_height[i];
        if (maxwidth < srcsize_width[i]) maxwidth = srcsize_width[i];
      }
      // Allocate buffer for max size calculated
      unsigned long long imageDimMax =
          (unsigned long long)maxheight * (unsigned long long)maxwidth * 3;
      unsigned char *complete_image_buffer = static_cast<unsigned char *>(malloc(
          sizeof(unsigned char) * file_names.size() * imageDimMax));
      uint32_t elementsInRowMax = maxwidth * 3;

      for (uint32_t i = 0; i < file_names.size(); i++) {
        Mat image = imread(file_names[i], 1);
        if (image.empty()) {
          std::cout << "Could not read the image: " << file_names[i] << std::endl;
          return 1;
        }
        // Decode image
        cvtColor(image, image, cv::COLOR_BGR2RGB);
        unsigned char *ip_image = image.data;
        uint32_t elementsInRow = srcsize_width[i] * 3;
        // Copy the decoded data in allocated buffer
        for (uint32_t j = 0; j < srcsize_height[i]; j++) {
          unsigned char *temp_image = complete_image_buffer + (i * imageDimMax) + (j * elementsInRowMax);
          memcpy(temp_image, ip_image, elementsInRow * sizeof(unsigned char));
          ip_image += elementsInRow;
          input_buffer.push_back(temp_image);
        }
      }
    }
  }
  if (maxheight != 0 && maxwidth != 0)
    input1 = rocalJpegExternalFileSource(
        handle, folderPath, color_format, false, false, false,
        ROCAL_USE_USER_GIVEN_SIZE, maxwidth, maxheight,
        RocalDecoderType::ROCAL_DECODER_TJPEG, RocalExtSourceMode(mode));
  else
    input1 = rocalJpegExternalFileSource(
        handle, folderPath, color_format, false, false, false,
        ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height,
        RocalDecoderType::ROCAL_DECODER_TJPEG, RocalExtSourceMode(mode));
  if (rocalGetStatus(handle) != ROCAL_OK) {
    std::cerr << "JPEG source could not initialize : "
              << rocalGetErrorMessage(handle) << std::endl;
    return -1;
  }

  // uncomment the following to add augmentation if needed
  int resize_w = decode_width, resize_h = decode_height;
  // just do one augmentation to test
  rocalResize(handle, input1, resize_w, resize_h, true);

  if (rocalGetStatus(handle) != ROCAL_OK) {
    std::cerr << "Error while adding the augmentation nodes " << std::endl;
    auto err_msg = rocalGetErrorMessage(handle);
    std::cerr << err_msg << std::endl;
  }
  // Calling the API to verify and build the augmentation graph
  if (rocalVerify(handle) != ROCAL_OK) {
    std::cerr << "Could not verify the augmentation graph" << std::endl;
    return -1;
  }

  /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
  int n = rocalGetAugmentationBranchCount(handle);
  int h = n * rocalGetOutputHeight(handle);
  int w = rocalGetOutputWidth(handle);
  int p = (((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ||
            (color_format == RocalImageColor::ROCAL_COLOR_RGB_PLANAR))
               ? 3
               : 1);
  std::cerr << "output width " << w << " output height " << h
            << " color planes " << p << std::endl;
  const unsigned number_of_cols = 1;  // no augmented case
  auto cv_color_format = ((p == 3) ? CV_8UC3 : CV_8UC1);
  cv::Mat mat_output(h, w * number_of_cols, cv_color_format);
  cv::Mat mat_input(h, w, cv_color_format);
  cv::Mat mat_color;
  int col_counter = 0;
  bool eos = false;
  int total_images = file_names.size();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  int counter = 0;
  std::vector<std::string> names;
  std::vector<int> labels;
  names.resize(inputBatchSize);
  labels.resize(total_images);
  int iter_cnt = 0;
  int index = 0;
  // Assign some labels for all images
  for (int id = 0; id < total_images; id++) {
    labels[id] = 1;
  }
  while (!rocalIsEmpty(handle)) {
    std::vector<std::string> input_images;
    std::vector<unsigned char *> input_batch_buffer;
    std::vector<int> label_buffer;
    std::vector<unsigned> roi_width;
    std::vector<unsigned> roi_height;
    for (int i = 0; i < inputBatchSize; i++) {
      if (mode == 0) {
        input_images.push_back(file_names.back());
        file_names.pop_back();
        if ((file_names.size()) == 0) {
          eos = true;
        }
        label_buffer.push_back(labels.back());
        labels.pop_back();
      } else {
        if (mode == 1) {
          input_batch_buffer.push_back(input_buffer.back());
          input_buffer.pop_back();
          roi_height.push_back(srcsize_height.back());
          srcsize_height.pop_back();
          label_buffer.push_back(labels.back());
          labels.pop_back();
        } else {
          input_batch_buffer.push_back(input_buffer.back());
          input_buffer.pop_back();
          roi_width.push_back(srcsize_width.back());
          srcsize_width.pop_back();
          roi_height.push_back(srcsize_height.back());
          srcsize_height.pop_back();
          label_buffer.push_back(labels.back());
          labels.pop_back();
        }
        if ((file_names.size()) == 0 || input_buffer.size() == 0) {
          eos = true;
        }
      }
    }
    if (index <= (total_images / inputBatchSize)) {
      if (mode == 0)
        rocalExternalSourceFeedInput(handle, input_images, label_buffer, {}, {}, {},
                                     decode_width, decode_height, channels,
                                     RocalExtSourceMode(0),
                                     RocalTensorLayout(0), eos);
      else if (mode == 1)
        rocalExternalSourceFeedInput(handle, {}, label_buffer, input_batch_buffer, {},
                                     roi_height, decode_width, decode_height,
                                     channels, RocalExtSourceMode(mode),
                                     RocalTensorLayout(0), eos);
      else if (mode == 2)
        rocalExternalSourceFeedInput(handle, {}, label_buffer, input_batch_buffer,
                                     roi_width, roi_height, maxwidth, maxheight,
                                     channels, RocalExtSourceMode(mode),
                                     RocalTensorLayout(0), eos);
    }
    if (rocalRun(handle) != 0) break;

    if (display) rocalCopyToOutput(handle, mat_input.data, h * w * p);

    int label_ids[inputBatchSize];
    // Get labels and print labels
    rocalGetImageLabels(handle, label_ids);
    std::cout << "\nCurrent batch : " << iter_cnt << std::endl;
    for (auto label_id : label_ids) {
      std::cout << "Given Label :" << label_id << std::endl;
    }

    counter += inputBatchSize;
    iter_cnt++;

    if (!display) continue;
    // Dump the output image
    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    std::string out_filename =
        std::string("output") + std::to_string(index) + ".png";
    if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
      mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
      cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
      cv::imwrite(out_filename, mat_color, compression_params);
    } else if (color_format == RocalImageColor::ROCAL_COLOR_RGB_PLANAR) {
      // convert planar to packed for OPENCV
      for (int j = 0; j < n; j++) {
        int const kWidth = w;
        int const kHeight = rocalGetOutputHeight(handle);
        int single_h = kHeight / inputBatchSize;
        for (int n = 0; n < inputBatchSize; n++) {
          unsigned channel_size = kWidth * single_h * p;
          unsigned char *interleavedp = mat_output.data + channel_size * n;
          unsigned char *planarp = mat_input.data + channel_size * n;
          for (int i = 0; i < (kWidth * single_h); i++) {
            interleavedp[i * 3 + 0] = planarp[i + 0 * kWidth * single_h];
            interleavedp[i * 3 + 1] = planarp[i + 1 * kWidth * single_h];
            interleavedp[i * 3 + 2] = planarp[i + 2 * kWidth * single_h];
          }
        }
      }
      cv::imwrite(out_filename, mat_color, compression_params);
    } else {
      mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
      cv::imwrite(out_filename, mat_color, compression_params);
    }
    cv::waitKey(1);
    col_counter = (col_counter + 1) % number_of_cols;
    index++;
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto dur = duration_cast<microseconds>(t2 - t1).count();
  auto rocal_timing = rocalGetTimingInfo(handle);
  std::cerr << "Load     time " << rocal_timing.load_time << std::endl;
  std::cerr << "Decode   time " << rocal_timing.decode_time << std::endl;
  std::cerr << "Process  time " << rocal_timing.process_time << std::endl;
  std::cerr << "Transfer time " << rocal_timing.transfer_time << std::endl;
  std::cerr << ">>>>> " << counter
            << " images/frames Processed. Total Elapsed Time " << dur / 1000000
            << " sec " << dur % 1000000 << " us " << std::endl;
  rocalRelease(handle);
  mat_input.release();
  mat_output.release();
  return 0;
}