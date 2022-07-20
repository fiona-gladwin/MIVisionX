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

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_vx.h"


class CropResizeNode : public Node
{
public:
    CropResizeNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs);
    CropResizeNode() = delete;
    void init(int  area, int aspect_ratio, int x_center_drift, int y_center_drift, int outputtoggleformat, int layout);
    void init(IntParam* area, IntParam* aspect_ratio, IntParam *x_center_drift, IntParam *y_center_drift, int outputtoggleformat, int layout);
    unsigned int get_dst_width() { return _outputs[0]->info().get_width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().get_height(); }
    std::shared_ptr<RocalRandomCropParam> get_crop_param() { return _crop_param; }
protected:
    void create_node() override;
    void update_node() override;
private:
    size_t _dest_width;
    size_t _dest_height;
    int _layout,_roi_type,_outputtoggleformat;
    ParameterVX<int> _x1_arr;
    ParameterVX<int> _x2_arr;
    ParameterVX<int> _x3_arr;
    ParameterVX<int> _x4_arr;
    
    std::shared_ptr<RocalRandomCropParam> _crop_param;
    vx_array _dst_roi_width ,_dst_roi_height;
    constexpr static int X1_RANGE [2] = {0, 100};

};



