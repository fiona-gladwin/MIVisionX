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
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class GlitchNode : public Node
{
public:
    GlitchNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    GlitchNode() = delete;

    void init( int x_offset_r , int y_offset_r, int x_offset_g , int y_offset_g, int x_offset_b , int y_offset_b);
    void init( IntParam* x_offset_r , IntParam* y_offset_r, IntParam* x_offset_g , IntParam* y_offset_g, IntParam* x_offset_b , IntParam* y_offset_b);

protected:
    void create_node() override ;
    void update_node() override;
private:

    ParameterVX<int> _x_offset_r;
    ParameterVX<int> _y_offset_r;
    ParameterVX<int> _x_offset_g;
    ParameterVX<int> _y_offset_g;
    ParameterVX<int> _x_offset_b;
    ParameterVX<int> _y_offset_b;
    constexpr static int  GLITCH_RANGE [2] = {0, 10};
 
};