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
#include "node_crop.h"
#include "parameter_crop.h"
#include "exception.h"

CropNode::CropNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs) {
    _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropNode::create_node() {
    if(_node)
        return;

    _crop_param->create_array(_graph);
    create_crop_tensor(_crop_tensor, &_crop_coordinates);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppCrop(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(),
                         input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the Crop node (vxExtRppCrop) failed: "+TOSTR(status))
}

void CropNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    RocalROI *crop_dims = static_cast<RocalROI *>(_crop_coordinates);
    for(unsigned i = 0; i < _batch_size; i++) {
        crop_dims[i].x1 = x1[i];
        crop_dims[i].y1 = y1[i];
        crop_dims[i].x2 = crop_w_dims[i];
        crop_dims[i].y2 = crop_h_dims[i];
    }
}

void CropNode::init(unsigned int crop_h, unsigned int crop_w, float x_drift, float y_drift) {
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1 = x_drift;
    _crop_param->y1 = y_drift;
    FloatParam *x_drift_param = ParameterFactory::instance()->create_single_value_float_param(x_drift);
    FloatParam *y_drift_param = ParameterFactory::instance()->create_single_value_float_param(y_drift);
    _crop_param->set_x_drift_factor(core(x_drift_param));
    _crop_param->set_y_drift_factor(core(y_drift_param));
}

// This init is used only for centre crop
void CropNode::init(unsigned int crop_h, unsigned int crop_w) {
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1 = 0; 
    _crop_param->y1 = 0;
    _crop_param->set_fixed_crop(0.5, 0.5);    // for center_crop
}

void CropNode::init(FloatParam *crop_h_factor, FloatParam  *crop_w_factor, FloatParam *x_drift, FloatParam *y_drift) {
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_crop_height_factor(core(crop_h_factor));
    _crop_param->set_crop_width_factor(core(crop_w_factor));
    _crop_param->set_random();
}

CropNode::~CropNode() {
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
