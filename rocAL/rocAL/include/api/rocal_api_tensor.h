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

#ifndef MIVISIONX_ROCAL_API_TENSOR_H
#define MIVISIONX_ROCAL_API_TENSOR_H
#include "rocal_api_types.h"

class rocalTensor {
public:
    virtual ~rocalTensor() = default;
    virtual void* buffer() = 0;
    // unsigned copy_to_external(); // Multiple API with different use cases
    virtual unsigned num_of_dims() = 0;
    virtual unsigned batch_size() = 0;
    virtual std::vector<size_t> dims() = 0;
    virtual RocalTensorLayout layout() = 0;
    virtual RocalTensorOutputType data_type() = 0;
    virtual RocalROICordsType roi_type() = 0;
    virtual RocalROICords *get_roi() = 0;
    virtual std::vector<size_t> shape() = 0;
};

class rocalTensorList {
public:
    virtual uint64_t size() = 0;
    virtual rocalTensor* at(size_t index) = 0;
    virtual void release() = 0;
    // at API
    // isDenseTensor
    // Add a copy API!
};

typedef rocalTensor * RocalTensor;
typedef rocalTensorList * RocalTensorList;
typedef std::vector<rocalTensorList *> RocalMetaData;

#endif //MIVISIONX_ROCAL_API_TENSOR_H
