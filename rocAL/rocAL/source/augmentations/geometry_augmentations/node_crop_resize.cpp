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
#include "node_crop_resize.h"
#include "parameter_crop.h"
#include "exception.h"

CropResizeNode::CropResizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
    _crop_param = std::make_shared<RocalRandomCropParam>(_batch_size);
}

void CropResizeNode::create_node()
{
    if(_node)
        return;

    _crop_param->create_array(_graph);
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
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_crop_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(crop_tensor: failed " + TOSTR(status))
    // std::cerr<<
    _node = vxExtrppNode_ResizeCrop(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _crop_tensor, _outputs[0]->handle(),_dst_roi_width, _dst_roi_height,_crop_param->x1_arr, _crop_param->y1_arr, _crop_param->x2_arr, _crop_param->y2_arr,
                              _input_layout, _output_layout, _roi_type);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the resize_crop tensor (vxExtrppNode_ResizeCrop) failed: "+TOSTR(status))
}

void CropResizeNode::update_node()
{
    std::cerr<<"\ncheck in update_node  node_resize_crop.cpp";
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    
    // Obtain the crop coordinates and update the roi
    // auto x1 = _crop_param->get_x1_arr_val();
    // auto y1 = _crop_param->get_y1_arr_val();
    // RocalROI *src_roi = (RocalROI *)_crop_coordinates;
    // for(unsigned i = 0; i < _batch_size; i++) {
    //     src_roi[i].x1 = x1[i];
    //     src_roi[i].y1 = y1[i];
    //     src_roi[i].x2 = crop_w_dims[i];
    //     src_roi[i].y2 = crop_h_dims[i];
    // }

    // std::vector<uint32_t> src_h_dims, src_w_dims;
    // // Using original width and height instead of the decoded width and height for computing resize dimensions
    // src_w_dims = _inputs[0]->info().get_orig_roi_width_vec();
    // src_h_dims = _inputs[0]->info().get_orig_roi_height_vec();
    // for (unsigned i = 0; i < _batch_size; i++) {
    //     _src_width = src_w_dims[i];
    //     _src_height = src_h_dims[i];
    //     _dst_width = _out_width;
    //     _dst_height = _out_height;
    //     adjust_out_roi_size();
    //     _dst_width = std::min(_dst_width, (unsigned)_outputs[0]->info().max_shape()[0]);
    //     _dst_height = std::min(_dst_height, (unsigned)_outputs[0]->info().max_shape()[1]);
    //     _dst_roi_width_vec.push_back(_dst_width);
    //     _dst_roi_height_vec.push_back(_dst_height);
    // }
    // vx_status width_status, height_status;
    // width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // if(width_status != 0 || height_status != 0)
    //     WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    // _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    // _dst_roi_width_vec.clear();
    // _dst_roi_height_vec.clear();
}

// void CropResizeNode::init(unsigned int crop_h, unsigned int crop_w, float x_drift_, float y_drift_)
// {
//     _crop_param->crop_w = crop_w;
//     _crop_param->crop_h = crop_h;
//     _crop_param->x1     = x_drift_;
//     _crop_param->y1     = y_drift_;
//     FloatParam *x_drift  = ParameterFactory::instance()->create_single_value_float_param(x_drift_);
//     FloatParam *y_drift  = ParameterFactory::instance()->create_single_value_float_param(y_drift_);
//     _crop_param->set_x_drift_factor(core(x_drift));
//     _crop_param->set_y_drift_factor(core(y_drift));
// }

// This init is used only for centre crop
// void CropResizeNode::init(unsigned int crop_h, unsigned int crop_w)
// {
//     _crop_param->crop_w = crop_w;
//     _crop_param->crop_h = crop_h;
//     _crop_param->x1 = 0; 
//     _crop_param->y1 = 0;
//     _crop_param->set_fixed_crop(0.5, 0.5);    // for center_crop
// }

// void CropResizeNode::init(FloatParam *crop_h_factor, FloatParam  *crop_w_factor, FloatParam *x_drift, FloatParam *y_drift)
// {
//     _crop_param->set_x_drift_factor(core(x_drift));
//     _crop_param->set_y_drift_factor(core(y_drift));
//     _crop_param->set_crop_height_factor(core(crop_h_factor));
//     _crop_param->set_crop_width_factor(core(crop_w_factor));
//     _crop_param->set_random();
// }
void CropResizeNode::init(float area, float aspect_ratio, float x_center_drift, float y_center_drift)
{
    std::cerr<<"\ncheck in init  node_resize_crop.cpp";
    _crop_param->set_area_factor(ParameterFactory::instance()->create_single_value_param(area));
    _crop_param->set_aspect_ratio(ParameterFactory::instance()->create_single_value_param(aspect_ratio));
    _crop_param->set_x_drift_factor(ParameterFactory::instance()->create_single_value_param(x_center_drift));
    _crop_param->set_y_drift_factor(ParameterFactory::instance()->create_single_value_param(y_center_drift));
}
void CropResizeNode::init(FloatParam* area, FloatParam* aspect_ratio, FloatParam *x_center_drift, FloatParam *y_center_drift)
{
    _crop_param->set_area_factor(core(area));
    _crop_param->set_aspect_ratio(core(aspect_ratio));
    _crop_param->set_x_drift_factor(core(x_center_drift));
    _crop_param->set_y_drift_factor(core(y_center_drift));
    _crop_param->set_random();

}

// void CropResizeNode::adjust_out_roi_size() {
//     bool has_max_size = (_max_width | _max_height) > 0;

//     if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH) {
//         if (!_dst_width) _dst_width = _src_width;
//         if (!_dst_height) _dst_height = _src_height;

//         if (has_max_size) {
//             if (_max_width) _dst_width = std::min(_dst_width, _max_width);
//             if (_max_height) _dst_height = std::min(_dst_height, _max_height);
//         }
//     } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT) {
//         if ((!_dst_width) & _dst_height) {  // Only height is passed
//             _dst_width = std::lround(_src_width * (static_cast<float>(_dst_height) / _src_height));
//         } else if ((!_dst_height) & _dst_width) {  // Only width is passed
//             _dst_height = std::lround(_src_height * (static_cast<float>(_dst_width) / _src_width));
//         }
        
//         if (has_max_size) {
//             if (_max_width) _dst_width = std::min(_dst_width, _max_width);
//             if (_max_height) _dst_height = std::min(_dst_height, _max_height);
//         }
//     } else {
//         float scale = 1.0f;
//         float scale_w = static_cast<float>(_dst_width) / _src_width;
//         float scale_h = static_cast<float>(_dst_height) / _src_height;
//         if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER) {
//             scale = std::max(scale_w, scale_h);
//         } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER) {
//             scale = (scale_w > 0 && scale_h > 0) ? std::min(scale_w, scale_h) : ((scale_w > 0) ? scale_w : scale_h);
//         }
        
//         if (has_max_size) {
//             if (_max_width) scale = std::min(scale, static_cast<float>(_max_width) / _src_width);
//             if (_max_height) scale = std::min(scale, static_cast<float>(_max_height) / _src_height);
//         }

//         if ((scale_h != scale) || (!_dst_height)) _dst_height = std::lround(_src_height * scale);
//         if ((scale_w != scale) || (!_dst_width)) _dst_width = std::lround(_src_width * scale);
//     }
// }

CropResizeNode::~CropResizeNode() {
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
