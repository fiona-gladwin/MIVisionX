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
#include <graph.h>
#include <cmath>
#include "node_resize_mirror_normalize.h"
#include "exception.h"

ResizeMirrorNormalizeNode::ResizeMirrorNormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs), _mirror(_mirror_range[0], _mirror_range[1])
{
}

void ResizeMirrorNormalizeNode::create_node()
{
    if(_node)
        return;
    std::vector<uint32_t> roi_width(_batch_size, _inputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> roi_height(_batch_size, _inputs[0]->info().max_shape()[1]);
    _src_roi_width = vxCreateArray(vxGetContext((vx_reference) _graph->get()), VX_TYPE_UINT32, _batch_size);
    _src_roi_height = vxCreateArray(vxGetContext((vx_reference) _graph->get()), VX_TYPE_UINT32, _batch_size);
    
    vx_status width_status, height_status;
    width_status = vxAddArrayItems(_src_roi_width, _batch_size, roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_src_roi_height, _batch_size, roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizeTensor) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));
    
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);
    
    std::vector<float> mean_vec, std_dev_vec;
    int mean_std_array_size = _batch_size * _inputs[0]->info().get_channels();
    if(!_std_dev[0])
        THROW("Standard deviation value cannot be 0");
    mean_vec.resize(mean_std_array_size, _mean[0]);
    std_dev_vec.resize(mean_std_array_size, _std_dev[0]);

    if(_inputs[0]->info().get_channels() == 3) {
        if(!(_std_dev[0] && _std_dev[1] && _std_dev[2]))
            THROW("Standard deviation value cannot be 0");
        for (uint i = 0, j = 0; i < _batch_size; i++, j += 3 ) {
            mean_vec[j ] = _mean[0];
            mean_vec[j + 1] = _mean[1];
            mean_vec[j + 2] = _mean[2];

            std_dev_vec[j ] = _std_dev[0];
            std_dev_vec[j + 1] = _std_dev[1];
            std_dev_vec[j + 2] = _std_dev[2];
        }
    }

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
   
    vx_status status = VX_SUCCESS;
    _mean_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_std_array_size);
    _std_dev_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_std_array_size);
    status |= vxAddArrayItems(_mean_vx_array, mean_std_array_size, mean_vec.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_vx_array, mean_std_array_size, std_dev_vec.data(), sizeof(vx_float32));
    _mirror.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize_mirror_normalize node (vxExtrppNode_CropMirrorNormalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))    

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));

    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_INT32,&_interpolation_type);
   _node = vxExtrppNode_ResizeMirrorNormalize(_graph->get(), _inputs[0]->handle(),
                                             _src_tensor_roi, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height,
                                             interpolation_vx, _mean_vx_array, _std_dev_vx_array, _mirror.default_array(),
                                             _input_layout, _output_layout, _roi_type, _batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizeMirrorNormalize) node failed: "+ TOSTR(status))
}

void ResizeMirrorNormalizeNode::update_node()
{
    RocalROI* src_roi = _inputs[0]->info().get_roi();   // Check if it needs to be similar to resize
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_width = src_roi[i].x2;
        _src_height = src_roi[i].y2;
        _src_roi_width_vec.push_back(_src_width);
        _src_roi_height_vec.push_back(_src_height);
        // Min size and max size used for MLPerf MaskRCNN resize augmentation
        // TODO: Get the min_size and max_size as user arguments from python
        int min_size = 800;
        int max_size = 1333;
        int src_width = src_roi[i].x2;
        int src_height = src_roi[i].y2;
        int size = min_size;
        int output_width, output_height;

        float min_original_size = static_cast<float>(std::min(src_width, src_height));
        float max_original_size = static_cast<float>(std::max(src_width, src_height));
        if(max_original_size / min_original_size * size > max_size)
            size = static_cast<size_t>(round(max_size * min_original_size / max_original_size));

        if (((src_width <= src_height) && (src_width == size)) || ((src_height <= src_width) && (src_height == size)))
        {
            _dst_height = src_height;
            _dst_width = src_width;
            continue;
        }

        if(src_width < src_height) {
            _dst_width = size;
            _dst_height = static_cast<size_t>(size * src_height / src_width);	
        } else {
            _dst_height = size;
            _dst_width = static_cast<size_t>(size * src_width / src_height);
        }
        _dst_width = std::min(_dst_width, (unsigned)_outputs[0]->info().max_shape()[0]);
        _dst_height = std::min(_dst_height, (unsigned)_outputs[0]->info().max_shape()[1]);
        _dst_roi_width_vec.push_back(_dst_width);
        _dst_roi_height_vec.push_back(_dst_height);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_src_roi_width, 0, _batch_size, sizeof(vx_uint32), _src_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_src_roi_height, 0, _batch_size, sizeof(vx_uint32), _src_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _src_roi_width or _src_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    _src_roi_width_vec.clear();
    _src_roi_height_vec.clear();
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();
    _mirror.update_array();
}

void ResizeMirrorNormalizeNode::init(std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror)
{
    _interpolation_type = 1;
    _mean   = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}
