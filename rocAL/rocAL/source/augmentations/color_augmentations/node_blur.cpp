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
#include "node_blur.h"
#include "exception.h"


BlurNode::BlurNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _kernel_size(KERNEL_RANGE[0], KERNEL_RANGE[1])
{
}

void BlurNode::create_node() {
    std::cerr<<"blur in node_blur.cpp  create_node";

    if(_node)
        return;

    _kernel_size.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_Blur(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _kernel_size.default_array(), _input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness (vxExtrppNode_Blur) node failed: "+ TOSTR(status))
}

void BlurNode::init( int kernel_size) {
    std::cerr<<"blur in node_blur.cpp  init";
    _kernel_size.set_param(kernel_size);
}

void BlurNode::init( IntParam* kernel_size) {
    std::cerr<<"blur in node_blur.cpp  init";
    _kernel_size.set_param(core(kernel_size));
}


void BlurNode::update_node() {
    std::cerr<<"blur in node_blur.cpp  update_node";
    _kernel_size.update_array();
}

