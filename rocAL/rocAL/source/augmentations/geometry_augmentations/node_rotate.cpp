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
#include "node_rotate.h"
#include "exception.h"


RotateNode::RotateNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs),
        _angle(ROTATE_ANGLE_RANGE[0], ROTATE_ANGLE_RANGE[1]) {}

void RotateNode::create_node() {
    if(_node)
        return;

    _angle.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_interpolation_type);

    // Create vx_scalar for layout and roi type to be passed to the node
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    _input_layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    _output_layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    _roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppRotate(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _angle.default_array(),
                           interpolation_vx, _input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rotate (vxExtRppRotate) node failed: " + TOSTR(status))
}

void RotateNode::init(float angle, RocalResizeInterpolationType interpolation_type) {
    _angle.set_param(angle);
    _interpolation_type = static_cast<int>(interpolation_type);
}

void RotateNode::init(FloatParam* angle, RocalResizeInterpolationType interpolation_type) {
    _angle.set_param(core(angle));
    _interpolation_type = static_cast<int>(interpolation_type);
}

void RotateNode::update_node() {
    _angle.update_array();
}
