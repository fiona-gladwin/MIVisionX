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
#include "node_lens_correction.h"
#include "exception.h"


LensCorrectionNode::LensCorrectionNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _strength(STRENGTH_RANGE[0], STRENGTH_RANGE[1]),
        _zoom(ZOOM_RANGE[0], ZOOM_RANGE[1])
{
}

void LensCorrectionNode::create_node() {
    if(_node)
        return;

    _strength.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _zoom.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_LensCorrection(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _strength.default_array(), _zoom.default_array(), _input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the lens correction  (vxExtrppNode_LensCorrection) node failed: "+ TOSTR(status))
}

void LensCorrectionNode::init(float alpha, float beta) {
    _strength.set_param(alpha);
    _zoom.set_param(beta);
}

void LensCorrectionNode::init(FloatParam* alpha, FloatParam* beta) {
    _strength.set_param(core(alpha));
    _zoom.set_param(core(beta));
}

void LensCorrectionNode::update_node() {
    _strength.update_array();
    _zoom.update_array();
}