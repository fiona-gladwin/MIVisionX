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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_resize.h"
#include "exception.h"


ResizeNode::ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void ResizeNode::create_node()
{
    if(_node)
        return;

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

   _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))

}

void ResizeNode::update_node()
{
    vx_status width_status, height_status;
    vx_size stride = sizeof(uint32_t);
    uint32_t *src_width = NULL;
    uint32_t *src_height = NULL;
    width_status = vxAccessArrayRange(_src_roi_width, 0, _batch_size, &stride, (void**)&src_width, VX_READ_ONLY);
    height_status = vxAccessArrayRange(_src_roi_height, 0, _batch_size, &stride, (void**)&src_height, VX_READ_ONLY);
    if(width_status != 0 || height_status != 0)
        THROW(" vxAccessArrayRange failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    width_status = vxCommitArrayRange(_src_roi_width, 0, _batch_size, src_width);
    height_status = vxCommitArrayRange(_src_roi_height, 0, _batch_size, src_height);
    if(width_status != 0 || height_status != 0)
        THROW(" vxCommitArrayRange failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    for (unsigned i = 0; i < _batch_size; i++)
    {
        _src_roi_size[0] = src_width[i];
        _src_roi_size[1] = src_height[i];
        _dst_roi_size[0] = _dest_width;
        _dst_roi_size[1] = _dest_height;
        adjust_out_roi_size();
        _dst_width.push_back(_dst_roi_size[0]);
        _dst_height.push_back(_dst_roi_size[1]);
        // std::cerr << "\nAFTER DST W : " << _dst_roi_size[0] << " DST H : " << _dst_roi_size[1];
    }
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_height failed " + TOSTR(width_status) + "  "+ TOSTR(height_status));
    _dst_width.clear();
    _dst_height.clear();
}

void ResizeNode::init(unsigned dest_width, unsigned dest_height, RaliResizeScalingMode scaling_mode, unsigned max_size)
{
    _scaling_mode = scaling_mode;
    _dest_width = dest_width;
    _dest_height = dest_height;
    _src_roi_size.resize(2);
    _dst_roi_size.resize(2);
    if(max_size > 0)
    {
        _max_roi_size.push_back(max_size);
        _max_roi_size.push_back(max_size);
    }
}

void ResizeNode::adjust_out_roi_size()
{
    unsigned dim = 2;
    std::vector<double> scale(dim, 1);
    std::vector<bool> has_dst_size_dim(dim, false);
    unsigned sizes_provided = 0;
    bool has_max_size = (_max_roi_size.size() > 0) ? true : false;
    for (unsigned i=0; i < dim; i++) {
        has_dst_size_dim[i] = (_src_roi_size[i] != 0) && (_dst_roi_size[i] != 0);
        sizes_provided += has_dst_size_dim[i];
        scale[i] = _src_roi_size[i] ? (_dst_roi_size[i] / static_cast<double>(_src_roi_size[i])) : 1;
    }
    if (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_STRETCH) {
        if (sizes_provided < dim) {
            for (unsigned i=0; i < dim; i++) {
                if (!has_dst_size_dim[i])
                    _dst_roi_size[i] = _src_roi_size[i];
            }
        }
        if (has_max_size) {
            for (unsigned i=0; i < dim; i++) {
                if (_max_roi_size[i] > 0 && _dst_roi_size[i] > _max_roi_size[i])
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    }
    else if (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_DEFAULT) {
        if (sizes_provided < dim) {
            double average_scale = 1;
            for (unsigned i=0; i < dim; i++) {
                if (has_dst_size_dim[i])
                    average_scale *= scale[i];
            }
            if (sizes_provided > 1)
                average_scale = std::pow(average_scale, 1.0 / sizes_provided);
            for(unsigned i=0; i < dim; i++) {
                if(!has_dst_size_dim[i])
                    _dst_roi_size[i] = std::round(_src_roi_size[i] * average_scale);   
            }    
        }
        if (has_max_size) {
            for (unsigned i=0; i < dim; i++) {
                if (_max_roi_size[i] > 0 && _dst_roi_size[i] > _max_roi_size[i])
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    }
    else {
        double final_scale = 0;
        bool first = true;
        for (unsigned i=0; i < dim; i++) {
            if (has_dst_size_dim[i]) {
                double s = scale[i];
                if (first ||
                    (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_NOT_SMALLER && s > final_scale) || 
                    (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_NOT_LARGER && s < final_scale))
                    final_scale = s;
                first = false;
            }
        }
        if(has_max_size) {
            for (unsigned i=0; i < dim; i++) {
                if(_max_roi_size[i] > 0) {
                    double s = static_cast<double>(_max_roi_size[i]) / _src_roi_size[i];
                    if (s < final_scale)
                        final_scale = s;
                }
            }
        }
        for (unsigned i=0; i < dim; i++) {
            if(!has_dst_size_dim[i] || (scale[i] != final_scale)) {
                _dst_roi_size[i] = std::round(_src_roi_size[i] * final_scale);
            }
        }
    }
}