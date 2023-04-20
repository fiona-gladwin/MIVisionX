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
#include "node_glitch.h"
#include "exception.h"


GlitchNode::GlitchNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _x_offset_r(GLITCH_RANGE[0], GLITCH_RANGE[1]),
        _y_offset_r(GLITCH_RANGE[0], GLITCH_RANGE[1]),
        _x_offset_g(GLITCH_RANGE[0], GLITCH_RANGE[1]),
        _y_offset_g(GLITCH_RANGE[0], GLITCH_RANGE[1]),
        _x_offset_b(GLITCH_RANGE[0], GLITCH_RANGE[1]),
        _y_offset_b(GLITCH_RANGE[0], GLITCH_RANGE[1])
{
}

void GlitchNode::create_node() {
    if(_node)
        return;

    _x_offset_r.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _y_offset_r.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _x_offset_g.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _y_offset_g.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _x_offset_b.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _y_offset_b.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_Glitch(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _x_offset_r.default_array(), _y_offset_r.default_array(), _x_offset_g.default_array(), _y_offset_g.default_array(),_x_offset_b.default_array(), _y_offset_b.default_array(),_input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the glitch (vxExtrppNode_Glitch) node failed: "+ TOSTR(status))
}

void GlitchNode::init(  int x_offset_r , int y_offset_r, int x_offset_g , int y_offset_g, int x_offset_b , int y_offset_b) {
    _x_offset_r.set_param(x_offset_r);
    _y_offset_r.set_param(y_offset_r);
    _x_offset_g.set_param(x_offset_g);
    _y_offset_g.set_param(y_offset_g);
    _x_offset_b.set_param(x_offset_b);
    _y_offset_b.set_param(y_offset_b);
}

void GlitchNode::init(IntParam* x_offset_r , IntParam* y_offset_r, IntParam* x_offset_g , IntParam* y_offset_g, IntParam* x_offset_b , IntParam* y_offset_b) {
    _x_offset_r.set_param(core(x_offset_r));
    _y_offset_r.set_param(core(y_offset_r));
    _x_offset_g.set_param(core(x_offset_g));
    _y_offset_g.set_param(core(y_offset_g));
    _x_offset_b.set_param(core(x_offset_b));
    _y_offset_b.set_param(core(y_offset_b));
}


void GlitchNode::update_node() {
    _x_offset_r.update_array();
    _y_offset_r.update_array();
    _x_offset_g.update_array();
    _y_offset_g.update_array();
    _x_offset_b.update_array();
    _y_offset_b.update_array();
}

