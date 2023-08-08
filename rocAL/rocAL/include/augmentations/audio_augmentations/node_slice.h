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

#pragma once
#include "node.h"
#include "graph.h"
#include "rocal_api_types.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class SliceNode : public Node {
public:
    SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SliceNode() = delete;
    void init(Tensor *anchor_param, Tensor *shape_param, std::vector<float> &fill_values_param, std::vector<unsigned> &axes,
              bool normalized_anchor, bool normalized_shape, RocalOutOfBoundsPolicy policy);
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _fill_values_array;
    float * _anchor_array , *_shape_array;
    Tensor *_anchor, *_shape;
    std::vector<float> _fill_values, _fill_values_vec, _anchor_vec, _shape_vec;
    bool _normalized_anchor = false, _normalized_shape = false;
    RocalOutOfBoundsPolicy _policy = RocalOutOfBoundsPolicy::ERROR;
    unsigned _dims_stride;
    int _axis_mask = 0;
};