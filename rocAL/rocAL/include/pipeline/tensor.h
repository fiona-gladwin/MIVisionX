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

#pragma once

#include <VX/vx.h>
#include <VX/vx_types.h>

#include <array>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>
#if ENABLE_HIP
#include "device_manager_hip.h"
#include "hip/hip_runtime.h"
#else
#include "device_manager.h"
#endif
#include "commons.h"

/*! \brief Converts Rocal Memory type to OpenVX memory type
 *
 * @param mem input Rocal type
 * @return the OpenVX type associated with input argument
 */
vx_enum vx_mem_type(RocalMemType mem);

/*! \brief Returns the size of the data type
 *
 * @param RocalTensorDataType input data type
 * @return the OpenVX data type size associated with input argument
 */
vx_uint64 tensor_data_size(RocalTensorDataType data_type);

/*! \brief Holds the information about a rocalTensor */
class rocalTensorInfo {
public:
    friend class rocalTensor;
    enum class Type {
        UNKNOWN = -1,
        REGULAR = 0,
        VIRTUAL = 1,
        HANDLE = 2
    };

    // Default constructor
    /*! initializes memory type to host and dimension as empty vector*/
    rocalTensorInfo();

    //! Initializer constructor with only fields common to all types (Image/ Video / Audio)
    rocalTensorInfo(std::vector<size_t> dims, RocalMemType mem_type,
                    RocalTensorDataType data_type);

    // Setting properties required for Image / Video
    void set_roi_type(RocalROIType roi_type) { _roi_type = roi_type; }
    void set_data_type(RocalTensorDataType data_type) {
        _data_type = data_type;
        _data_size = (_data_size / _data_type_size);
        _data_size *= data_type_size();
    }
    void set_max_dims() {
        if (_layout != RocalTensorlayout::NONE) {
            _max_dims.resize(2);  // Since 2 values will be stored in the vector
            _is_image = true;
            if (_layout == RocalTensorlayout::NHWC) {
                _max_dims[0] = _dims.at(2);
                _max_dims[1] = _dims.at(1);
            } else if (_layout == RocalTensorlayout::NCHW ||
                        _layout == RocalTensorlayout::NFHWC) {
                _max_dims[0] = _dims.at(3);
                _max_dims[1] = _dims.at(2);
            } else if (_layout == RocalTensorlayout::NFCHW) {
                _max_dims[0] = _dims.at(4);
                _max_dims[1] = _dims.at(3);
            }
            reallocate_tensor_roi_buffers();
        } else if (!_is_metadata) {  // For audio
            _max_dims.resize(2);       // Since 2 values will be stored in the vector
            _max_dims[0] = _dims.at(1);
            _max_dims[1] = _num_of_dims > 2 ? _dims.at(2) : 0;
        }
    }
    void set_tensor_layout(RocalTensorlayout layout) {
        _layout = layout;
        set_max_dims();
    }
    void set_dims(std::vector<size_t>& new_dims) {
        _data_size = _data_type_size;
        if (_num_of_dims == new_dims.size()) {
            for (unsigned i = 0; i < _num_of_dims; i++) {
                _dims.at(i) = new_dims[i];
                _data_size *= new_dims[i];
            }
            set_max_dims();
        } else {
            THROW("The size of number of dimensions does not match with the dimensions of existing tensor")
        }
    }
    void set_color_format(RocalColorFormat color_format) {
        _color_format = color_format;
    }
    unsigned num_of_dims() const { return _num_of_dims; }
    unsigned batch_size() const { return _batch_size; }
    uint64_t data_size() const { return _data_size; }
    std::vector<size_t> max_dims() const { return _max_dims; }
    std::vector<size_t> dims() const { return _dims; }
    RocalMemType mem_type() const { return _mem_type; }
    RocalROIType roi_type() const { return _roi_type; }
    RocalTensorDataType data_type() const { return _data_type; }
    RocalTensorlayout layout() const { return _layout; }
    std::shared_ptr<std::vector<RocalROI>> get_roi() const { return _roi; }
    RocalColorFormat color_format() const { return _color_format; }
    Type type() const { return _type; }
    uint64_t data_type_size() {
        _data_type_size = tensor_data_size(_data_type);
        return _data_type_size;
    }
    bool is_image() const { return _is_image; }
    void set_metadata() { _is_metadata = true; }
    bool is_metadata() const { return _is_metadata; }

private:
    Type _type = Type::UNKNOWN;  //!< tensor type, whether is virtual tensor, created from handle or is a regular tensor
    unsigned _num_of_dims;  //!< denotes the number of dimensions in the tensor
    std::vector<size_t> _dims;  //!< denotes the dimensions of the tensor
    unsigned _batch_size;       //!< the batch size
    RocalMemType _mem_type;     //!< memory type, currently either OpenCL or Host
    RocalROIType _roi_type = RocalROIType::XYWH;     //!< ROI type, currently either XYWH or LTRB
    RocalTensorDataType _data_type = RocalTensorDataType::FP32;  //!< tensor data type
    RocalTensorlayout _layout = RocalTensorlayout::NONE;     //!< layout of the tensor
    RocalColorFormat _color_format;  //!< color format of the image
    std::shared_ptr<std::vector<RocalROI>> _roi;
    uint64_t _data_type_size = tensor_data_size(_data_type);
    uint64_t _data_size = 0;
    std::vector<size_t> _max_dims;  //!< stores the the width and height dimensions in the tensor
    void reallocate_tensor_roi_buffers();
    bool _is_image = false;
    bool _is_metadata = false;
};

bool operator==(const rocalTensorInfo& rhs, const rocalTensorInfo& lhs);
/*! \brief Holds an OpenVX tensor and it's info
* Keeps the information about the tensor that can be queried using OVX API as
* well, but for simplicity and ease of use, they are kept in separate fields
*/
class rocalTensor {
public:
    int swap_handle(void* handle);
    const rocalTensorInfo& info() { return _info; }
    //! Default constructor
    rocalTensor() = delete;
    void* buffer() { return _mem_handle; }
    vx_tensor handle() { return _vx_handle; }
    vx_context context() { return _context; }
    void set_mem_handle(void* buffer) { _mem_handle = buffer; }
#if ENABLE_OPENCL
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#elif ENABLE_HIP
    unsigned copy_data(hipStream_t stream, void* host_memory, bool sync);
#endif
    unsigned copy_data(void* user_buffer);
    //! Default destructor
    /*! Releases the OpenVX Tensor object */
    ~rocalTensor();

    //! Constructor accepting the tensor information as input
    explicit rocalTensor(const rocalTensorInfo& tensor_info);
    int create(vx_context context);
    void update_tensor_roi(const std::vector<uint32_t>& width, const std::vector<uint32_t>& height);
    void reset_tensor_roi() { _info.reallocate_tensor_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since
    // tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);
    bool is_handle_set() { return (_vx_handle != 0); }
    void set_dims(std::vector<size_t>& dims) { _info.set_dims(dims); }

private:
    vx_tensor _vx_handle = nullptr;  //!< The OpenVX tensor
    void* _mem_handle = nullptr;  //!< Pointer to the tensor's internal buffer (opencl or host)
    rocalTensorInfo _info;  //!< The structure holding the info related to the stored OpenVX tensor
    vx_context _context = nullptr;
};

/*! \brief Contains a list of rocalTensors */
class rocalTensorList {
public:
    uint64_t size() { return _tensor_list.size(); }
    bool empty() { return _tensor_list.empty(); }
    rocalTensor* front() { return _tensor_list.front(); }
    void push_back(rocalTensor* tensor) {
        _tensor_list.emplace_back(tensor);
        _tensor_data_size.emplace_back(tensor->info().data_size());
    }
    std::vector<uint64_t> data_size() { return _tensor_data_size; }
    void release() {
        for (auto& tensor : _tensor_list) delete tensor;
    }
    rocalTensor* operator[](size_t index) { return _tensor_list[index]; }
    rocalTensor* at(size_t index) { return _tensor_list[index]; }
    void operator=(rocalTensorList& other) {
        for (unsigned idx = 0; idx < other.size(); idx++) {
            auto* new_tensor = new rocalTensor(other[idx]->info());
            if (new_tensor->create_from_handle(other[idx]->context()) != 0)
                THROW("Cannot create the tensor from handle")
            this->push_back(new_tensor);
        }
    }

private:
    std::vector<rocalTensor*> _tensor_list;
    std::vector<uint64_t> _tensor_data_size;
};