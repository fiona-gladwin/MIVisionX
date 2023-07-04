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
#include "parameter_random_crop.h"
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
    _node = vxExtrppNode_ResizeCrop(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _crop_tensor, _outputs[0]->handle(),_dst_roi_width, _dst_roi_height,
                                   _input_layout, _output_layout, _roi_type);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the resize_crop tensor (vxExtrppNode_ResizeCrop) failed: "+TOSTR(status))
}

void CropResizeNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    
    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    auto x2 = _crop_param->get_croph_arr_val();
    auto y2 = _crop_param->get_cropw_arr_val();
    RocalROI *src_roi = (RocalROI *)_crop_coordinates;
    for(unsigned i = 0; i < _batch_size; i++) {
        src_roi[i].x1 = x1[i];
        src_roi[i].y1 = y1[i];
        src_roi[i].x2 = crop_w_dims[i];
        src_roi[i].y2 = crop_h_dims[i];
    }
}

void CropResizeNode::init(float area, float aspect_ratio, float x_center_drift, float y_center_drift)
{
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
