/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include "rali_api_types.h"

class ResizeNode : public Node
{
public:
    ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeNode() = delete;
    unsigned int get_dst_width() { return _outputs[0]->info().width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().height_single(); }
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
    void init(unsigned dest_width, unsigned dest_height, RaliResizeScalingMode scaling_mode, unsigned max_size, RaliResizeInterpolationType interp_type,
              float crop_x, float crop_y, float crop_width, float crop_height, float is_relative_roi);
    void adjust_out_roi_size();
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _dst_roi_width, _dst_roi_height, _crop_x1, _crop_y1, _crop_x2, _crop_y2;
    unsigned _dest_width, _dest_height;
    float _crop_x, _crop_y, _crop_width, _crop_height;
    bool _is_relative_roi;
    bool _has_roi = false;
    unsigned _dim = 2; // Denotes 2D images
    vx_int32 _interp_type;
    RaliResizeScalingMode _scaling_mode;
    std::vector<uint32_t> _src_roi_size, _dst_roi_size, _max_roi_size, _dst_roi_width_vec, _dst_roi_height_vec;
    std::vector<uint32_t> _x1_arr_val, _y1_arr_val, _croph_arr_val, _cropw_arr_val, _x2_arr_val, _y2_arr_val;
    vx_array _x1_arr, _y1_arr, _croph_arr, _cropw_arr, _x2_arr, _y2_arr;
};
