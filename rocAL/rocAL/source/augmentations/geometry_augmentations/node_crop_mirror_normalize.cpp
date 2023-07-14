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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_mirror_normalize.h"
#include "exception.h"


CropMirrorNormalizeNode::CropMirrorNormalizeNode(const std::vector<Tensor *> &inputs,
                                                 const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1]) {
        _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropMirrorNormalizeNode::create_node() {
    if(_node)
        return;

    if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")

    _crop_param->create_array(_graph);

    std::vector<float> multiplier_vec, offset_vec;
    int multiplier_offset_array_size = _batch_size * _inputs[0]->info().get_channels();
    if(!_std_dev[0])
        THROW("Standard deviation value cannot be 0");
    multiplier_vec.resize(multiplier_offset_array_size, -(_mean[0] / _std_dev[0]));
    offset_vec.resize(multiplier_offset_array_size, (1 / _std_dev[0]));
    
    if(_inputs[0]->info().get_channels() == 3) {
        if(!(_std_dev[0] && _std_dev[1] && _std_dev[2]))
            THROW("Standard deviation value cannot be 0");
        offset_vec[0] = 1 / _std_dev[0];
        offset_vec[1] = 1 / _std_dev[1];
        offset_vec[2] = 1 / _std_dev[2];
        multiplier_vec[0] = -(_mean[0] * offset_vec[0]);
        multiplier_vec[1] = -(_mean[1] * offset_vec[1]);
        multiplier_vec[2] = -(_mean[2] * offset_vec[2]);
        for (uint i = 1, j = 3; i < _batch_size; i++, j += 3) {
            multiplier_vec[j] = multiplier_vec[0];
            multiplier_vec[j + 1] = multiplier_vec[1];
            multiplier_vec[j + 2] = multiplier_vec[2];
            offset_vec[j] = offset_vec[0];
            offset_vec[j + 1] = offset_vec[1];
            offset_vec[j + 2] = offset_vec[2];
        }
    }
    _multiplier_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, multiplier_offset_array_size);
    _offset_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, multiplier_offset_array_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_multiplier_vx_array, multiplier_offset_array_size, multiplier_vec.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_offset_vx_array, multiplier_offset_array_size, offset_vec.data(), sizeof(vx_float32));
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the crop_mirror_normalize node (vxExtrppNode_CropMirrorNormalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))

    int input_layout = (int)_inputs[0]->info().layout();
    int output_layout = (int)_outputs[0]->info().layout();
    int roi_type = (int)_inputs[0]->info().roi_type();
    vx_scalar in_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar out_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    
    // Create vx_tensor for the crop coordinates
    vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> crop_tensor_dims = {_batch_size, 4};
    if(_inputs[0]->info().layout() == RocalTensorlayout::NFCHW || _inputs[0]->info().layout() == RocalTensorlayout::NFHWC)
        crop_tensor_dims = {_inputs[0]->info().dims()[0] * _inputs[0]->info().dims()[1], 4}; // For Sequences pre allocating the ROI to N * F to replicate in OpenVX extensions
    stride[0] = sizeof(vx_uint32);
    stride[1] = stride[0] * crop_tensor_dims[0];
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    allocate_host_or_pinned_mem(&_crop_coordinates, stride[1] * 4, _inputs[0]->info().mem_type());
    
    _crop_tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference) _graph->get()), num_of_dims, crop_tensor_dims.data(), VX_TYPE_UINT32, 0, 
                                                                  stride, (void *)_crop_coordinates, mem_type);
    if ((status = vxGetStatus((vx_reference)_crop_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(crop_tensor: failed " + TOSTR(status))
    
    _node = vxRppCropMirrorNormalize(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(),
                                             _multiplier_vx_array, _offset_vx_array, _mirror.default_array(), in_layout_vx, out_layout_vx, roi_type_vx);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop mirror normalize (vxExtrppNode_CropMirrorNormalize) failed: " + TOSTR(status))
}

void CropMirrorNormalizeNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    _mirror.update_array();
    
    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    RocalROI *src_roi = (RocalROI *)_crop_coordinates;
    for(unsigned i = 0; i < _batch_size; i++) {
        src_roi[i].x1 = x1[i];
        src_roi[i].y1 = y1[i];
        src_roi[i].x2 = crop_w_dims[i];
        src_roi[i].y2 = crop_h_dims[i];
    }
}

void CropMirrorNormalizeNode::init(int crop_h, int crop_w, float anchor_x, float anchor_y, std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror)
{
    // current implementation does a fixed crop with specified dims and anchor
    _crop_param->x1 = 0;
    _crop_param->y1 = 0;
    _crop_param->crop_h = crop_h;
    _crop_param->crop_w = crop_w;
    _crop_param->set_fixed_crop(anchor_x, anchor_y);
    _mean = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}

CropMirrorNormalizeNode::~CropMirrorNormalizeNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_crop_coordinates);
        if(err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        free(_crop_coordinates);
    }
    vxReleaseTensor(&_crop_tensor);
}
