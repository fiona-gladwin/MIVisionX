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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_resize.h"
#include "exception.h"

CropResizeNode::CropResizeNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs),
        // _dest_width(_outputs[0]->info().get_width()),
        // _dest_height(_outputs[0]->info().get_height())
        _x1_arr(X1_RANGE[0], X1_RANGE[1]),
        _x2_arr(X1_RANGE[0], X1_RANGE[1]),
        _x3_arr(X1_RANGE[0], X1_RANGE[1]),
        _x4_arr(X1_RANGE[0], X1_RANGE[1])
{
    _crop_param = std::make_shared<RocalRandomCropParam>(_batch_size);
}

void CropResizeNode::create_node()
{
    if(_node)
        return;
    // if(_dest_width == 0 || _dest_height == 0)
    //     THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);
    _x1_arr.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _x2_arr.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _x3_arr.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _x4_arr.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().get_width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().get_height());
    std::cerr<<"\n\ndst_roi_width     "<<dst_roi_width[0]<<" " <<dst_roi_height[0]<<"\n\n";
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar toggleformat = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_outputtoggleformat);
    
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    std::cerr<<"layouttttttttttttttttt"<<_layout<<"\n\n\n\n";
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);


    // _node = vxExtrppNode_ResizeCrop(_graph->get(), _inputs[0]->handle(),  _src_tensor_roi, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _x1_arr.default_array(), _x2_arr.default_array(), _x3_arr.default_array(), _x4_arr.default_array(),toggleformat, layout, roi_type, _batch_size);
    _node = vxExtrppNode_ResizeCrop(_graph->get(), _inputs[0]->handle(),  _src_tensor_roi, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height,_crop_param->x1_arr, _crop_param->y1_arr, _crop_param->x2_arr, _crop_param->y2_arr,toggleformat, layout, roi_type, _batch_size);



    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropResizeNode::update_node()
{
    // _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    // _crop_param->update_array();
        _x1_arr.update_array();
        _x2_arr.update_array();
        _x3_arr.update_array();
        _x4_arr.update_array();
}

void CropResizeNode::init(int area, int aspect_ratio, int x_center_drift, int y_center_drift, int outputtoggleformat, int layout)
{
    // _crop_param->set_area_factor(ParameterFactory::instance()->create_single_value_param(area));
    // _crop_param->set_aspect_ratio(ParameterFactory::instance()->create_single_value_param(aspect_ratio));
    // _crop_param->set_x_drift_factor(ParameterFactory::instance()->create_single_value_param(x_center_drift));
    // _crop_param->set_y_drift_factor(ParameterFactory::instance()->create_single_value_param(y_center_drift));
    _x1_arr.set_param(area);
    _x2_arr.set_param(aspect_ratio);
    _x3_arr.set_param(x_center_drift);
    _x4_arr.set_param(y_center_drift);

    // _x1=area;
    // _x2=aspect_ratio;
    // _x3=x_center_drift;
    // _x4=y_center_drift;
    _outputtoggleformat=outputtoggleformat;
    _layout=layout;
}


void CropResizeNode::init(IntParam* area, IntParam* aspect_ratio, IntParam *x_center_drift, IntParam *y_center_drift, int outputtoggleformat, int layout)
{
    // _crop_param->set_area_factor(core(area));
    // _crop_param->set_aspect_ratio(core(aspect_ratio));
    // _crop_param->set_x_drift_factor(core(x_center_drift));
    // _crop_param->set_y_drift_factor(core(y_center_drift));
    _x1_arr.set_param(core(area));
    _x2_arr.set_param(core(aspect_ratio));
    _x3_arr.set_param(core(x_center_drift));
    _x4_arr.set_param(core(y_center_drift));
    _outputtoggleformat=outputtoggleformat;
    _layout=layout;

}
