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
#include "node_resize_crop_mirror.h"
#include "parameter_crop.h"
#include "exception.h"
ResizeCropMirrorNode::ResizeCropMirrorNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])

{
        _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void ResizeCropMirrorNode::create_node() {
    if(_node)
        return;
    
    if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")
    
    vx_status status = VX_SUCCESS;
    _crop_param->create_array(_graph);
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_Resize) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));
    
    // Create vx_tensor for the crop coordinates
    vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> crop_tensor_dims = {_batch_size, 4};
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
    
    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_INT32,&_interpolation_type);
   _node = vxExtrppNode_ResizeCropMirror(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(), _dst_roi_width, 
                               _dst_roi_height, _mirror.default_array(), interpolation_vx, _input_layout, _output_layout, _roi_type, _batch_size);

    // vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_Resize) node failed: "+ TOSTR(status))
}

void ResizeCropMirrorNode::update_node() {
    std::vector<uint32_t> src_h_dims, src_w_dims;
    // Using original width and height instead of the decoded width and height for computing resize dimensions
    src_w_dims = _inputs[0]->info().get_orig_roi_width_vec();
    src_h_dims = _inputs[0]->info().get_orig_roi_height_vec();
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_width = src_w_dims[i];
        _src_height = src_h_dims[i];
        _dst_width = _out_width;
        _dst_height = _out_height;
        adjust_out_roi_size();
        _dst_width = std::min(_dst_width, (unsigned)_outputs[0]->info().max_shape()[0]);
        _dst_height = std::min(_dst_height, (unsigned)_outputs[0]->info().max_shape()[1]);
        _dst_roi_width_vec.push_back(_dst_width);
        _dst_roi_height_vec.push_back(_dst_height);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();

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

void ResizeCropMirrorNode::init(unsigned dest_width, unsigned dest_height, unsigned crop_width, unsigned crop_height, IntParam *mirror, RocalResizeScalingMode scaling_mode,
                      const std::vector<unsigned>& max_size, RocalResizeInterpolationType interpolation_type) {
    _interpolation_type = (int)interpolation_type;
    _scaling_mode = scaling_mode;
    _out_width = dest_width;
    _out_height = dest_height;
    if(max_size.size() > 0) {
        _max_width = max_size[0];
        _max_height = max_size[1];
    }
    //cmn
    _crop_param->x1 = 0;
    _crop_param->y1 = 0;
    _crop_param->crop_h = crop_height;
    _crop_param->crop_w = crop_width;
    // _crop_param->set_fixed_crop(anchor_x, anchor_y);
    _mirror.set_param(core(mirror));

    std::cerr<<"crop_width and crop_height "<<dest_width<<"  "<<dest_height<<"  "<<crop_width<<"  "<<crop_height<<"\n";
}

void ResizeCropMirrorNode::adjust_out_roi_size() {
    bool has_max_size = (_max_width | _max_height) > 0;

    if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH) {
        if (!_dst_width) _dst_width = _src_width;
        if (!_dst_height) _dst_height = _src_height;

        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT) {
        if ((!_dst_width) & _dst_height) {  // Only height is passed
            _dst_width = std::lround(_src_width * (static_cast<float>(_dst_height) / _src_height));
        } else if ((!_dst_height) & _dst_width) {  // Only width is passed
            _dst_height = std::lround(_src_height * (static_cast<float>(_dst_width) / _src_width));
        }
        
        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else {
        float scale = 1.0f;
        float scale_w = static_cast<float>(_dst_width) / _src_width;
        float scale_h = static_cast<float>(_dst_height) / _src_height;
        if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER) {
            scale = std::max(scale_w, scale_h);
        } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER) {
            scale = (scale_w > 0 && scale_h > 0) ? std::min(scale_w, scale_h) : ((scale_w > 0) ? scale_w : scale_h);
        }
        
        if (has_max_size) {
            if (_max_width) scale = std::min(scale, static_cast<float>(_max_width) / _src_width);
            if (_max_height) scale = std::min(scale, static_cast<float>(_max_height) / _src_height);
        }

        if ((scale_h != scale) || (!_dst_height)) _dst_height = std::lround(_src_height * scale);
        if ((scale_w != scale) || (!_dst_width)) _dst_width = std::lround(_src_width * scale);
    }
}

ResizeCropMirrorNode::~ResizeCropMirrorNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipFree(_crop_coordinates);
        if(err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        free(_crop_coordinates);
    }
    vxReleaseTensor(&_crop_tensor);
}
