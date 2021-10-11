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
    std::cerr << "================ RESIZE NODE =====================\n";
}

void ResizeNode::create_node()
{
    if(_node)
        return;
    std::cerr << "================ CREATE RESIZE NODE =====================\n";
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    
    // Initialize the crop array and vectors
    vx_status status;
    _x1_arr_val.resize(_batch_size);
    _cropw_arr_val.resize(_batch_size);
    _y1_arr_val.resize(_batch_size);
    _croph_arr_val.resize(_batch_size);
    _x2_arr_val.resize(_batch_size);
    _y2_arr_val.resize(_batch_size);

    _x1_arr =    vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _cropw_arr = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y1_arr =    vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _croph_arr = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _x2_arr =    vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y2_arr =    vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    status = vxAddArrayItems(_x1_arr, _batch_size, _x1_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))
    status = vxAddArrayItems(_y1_arr, _batch_size, _y1_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))
    status = vxAddArrayItems(_cropw_arr, _batch_size, _cropw_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))
    status = vxAddArrayItems(_croph_arr, _batch_size, _croph_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
            THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))
    status = vxAddArrayItems(_x2_arr, _batch_size, _x2_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))
    status = vxAddArrayItems(_y2_arr, _batch_size, _y2_arr_val.data(), sizeof(vx_uint32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(status))

   _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _interp_type, _batch_size);

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
        // Adjust crop coordinates
        _x1_arr_val[i] = static_cast<uint32_t>((_crop_x > 0) ? _is_relative_roi ?  (_crop_x * src_width[i]) : _crop_x : 0);
        _y1_arr_val[i] = static_cast<uint32_t>((_crop_y > 0) ? _is_relative_roi ?  (_crop_y * src_height[i]) : _crop_y : 0);
        _cropw_arr_val[i] = static_cast<uint32_t>((_crop_width > 0) ? _is_relative_roi ?  (_crop_width * src_width[i]) : _crop_width : src_width[i]);
        _croph_arr_val[i] = static_cast<uint32_t>((_crop_height > 0) ? _is_relative_roi ? (_crop_height * src_height[i]) : _crop_height : src_height[i]);
        _x2_arr_val[i] = _x1_arr_val[i] + _cropw_arr_val[i];
        _y2_arr_val[i] = _y1_arr_val[i] + _croph_arr_val[i];

        _src_roi_size[0] = _cropw_arr_val[i];
        _src_roi_size[1] = _croph_arr_val[i];
        _dst_roi_size[0] = _dest_width;
        _dst_roi_size[1] = _dest_height;
        adjust_out_roi_size();
        _dst_roi_width_vec.push_back(_dst_roi_size[0]);
        _dst_roi_height_vec.push_back(_dst_roi_size[1]);
        std::cerr << "\nAFTER DST W : " << _dst_roi_size[0] << " DST H : " << _dst_roi_size[1];
        std::cerr << " CROP X1 : " << _x1_arr_val[i] << " Y1 " << _y1_arr_val[i] << " X2 " << _x2_arr_val[i] << " Y2 " << _y2_arr_val[i] << "\n";
    }
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  "+ TOSTR(height_status));
    width_status = vxCopyArrayRange((vx_array)_x1_arr, 0, _batch_size, sizeof(vx_uint32), _x1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_y1_arr, 0, _batch_size, sizeof(vx_uint32), _y1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _x1_arr or _y1_arr failed " + TOSTR(width_status) + "  "+ TOSTR(height_status));
    width_status = vxCopyArrayRange((vx_array)_x2_arr, 0, _batch_size, sizeof(vx_uint32), _x2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_y2_arr, 0, _batch_size, sizeof(vx_uint32), _y2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _x2_arr or _y2_arr failed " + TOSTR(width_status) + "  "+ TOSTR(height_status));
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();
}

void ResizeNode::init(unsigned dest_width, unsigned dest_height, RaliResizeScalingMode scaling_mode, unsigned max_size, RaliResizeInterpolationType interp_type,
                      float crop_x, float crop_y, float crop_width, float crop_height, float is_relative_roi)
{
    std::cerr << "================ INITIALIZE RESIZE NODE =====================\n";
    _scaling_mode = scaling_mode;
    _dest_width = dest_width;
    _dest_height = dest_height;
    _src_roi_size.resize(2);
    _dst_roi_size.resize(2);
    _interp_type = (int)interp_type;
    if(max_size > 0)
    {
        _max_roi_size.push_back(max_size);
        _max_roi_size.push_back(max_size);
    }
    _crop_x = crop_x;
    _crop_y = crop_y;
    _crop_width = crop_width;
    _crop_height = crop_height;
    _is_relative_roi = is_relative_roi;
}

void ResizeNode::adjust_out_roi_size()
{
    std::vector<double> scale(_dim, 1);
    std::vector<bool> has_size(_dim, false);
    unsigned sizes_provided = 0;
    bool has_max_size = (_max_roi_size.size() > 0) ? true : false;
    for (unsigned i=0; i < _dim; i++) {
        has_size[i] = (_src_roi_size[i] != 0) && (_dst_roi_size[i] != 0);
        sizes_provided += has_size[i];
        scale[i] = _src_roi_size[i] ? (_dst_roi_size[i] / static_cast<double>(_src_roi_size[i])) : 1;
    }
    if (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_STRETCH) {
        if (sizes_provided < _dim) {
            for (unsigned i=0; i < _dim; i++) {
                if (!has_size[i])
                    _dst_roi_size[i] = _src_roi_size[i];
            }
        }
        if (has_max_size) {
            for (unsigned i=0; i < _dim; i++) {
                if ((_max_roi_size[i] > 0) && (_dst_roi_size[i] > _max_roi_size[i]))
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    }
    else if (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_DEFAULT) {
        if (sizes_provided < _dim) {
            double average_scale = 1;
            for (unsigned i=0; i < _dim; i++) {
                if (has_size[i])
                    average_scale *= scale[i];
            }
            if (sizes_provided > 1)
                average_scale = std::pow(average_scale, 1.0 / sizes_provided);
            for(unsigned i=0; i < _dim; i++) {
                if(!has_size[i])
                    _dst_roi_size[i] = std::round(_src_roi_size[i] * average_scale);   
            }    
        }
        if (has_max_size) {
            for (unsigned i=0; i < _dim; i++) {
                if ((_max_roi_size[i] > 0) && (_dst_roi_size[i] > _max_roi_size[i]))
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    }
    else {
        double final_scale = 0;
        bool first = true;
        for (unsigned i=0; i < _dim; i++) {
            if (has_size[i]) {
                double s = scale[i];
                if (first ||
                    (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_NOT_SMALLER && s > final_scale) || 
                    (_scaling_mode == RaliResizeScalingMode::RALI_SCALING_MODE_NOT_LARGER && s < final_scale))
                    final_scale = s;
                first = false;
            }
        }
        if(has_max_size) {
            for (unsigned i=0; i < _dim; i++) {
                if(_max_roi_size[i] > 0) {
                    double s = static_cast<double>(_max_roi_size[i]) / _src_roi_size[i];
                    if (s < final_scale)
                        final_scale = s;
                }
            }
        }
        for (unsigned i=0; i < _dim; i++) {
            if(!has_size[i] || (scale[i] != final_scale)) {
                _dst_roi_size[i] = std::round(_src_roi_size[i] * final_scale);
            }
        }
    }
}