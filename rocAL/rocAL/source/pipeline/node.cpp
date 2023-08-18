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

#include "node.h"
Node::~Node()
{
    if (_node) vxReleaseNode(&_node);
    _node = nullptr;
}

void
Node::create(std::shared_ptr<Graph> graph)
{
    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output images to the node")

    _graph = graph;
    create_node();
}

void
Node::update_parameters()
{
    update_node();
}

// Create vx_tensor for the crop coordinates
void Node::create_crop_tensor(vx_tensor &crop_tensor, void **crop_coordinates) {
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
    allocate_host_or_pinned_mem(crop_coordinates, stride[1] * 4, _inputs[0]->info().mem_type());

    crop_tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference) _graph->get()), num_of_dims, crop_tensor_dims.data(), VX_TYPE_UINT32, 0, 
                                                                  stride, (void *)(*crop_coordinates), mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)crop_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(crop_tensor: failed " + TOSTR(status))
}
