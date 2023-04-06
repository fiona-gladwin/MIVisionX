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
#include "node_snp_noise.h"
#include "exception.h"


NoiseNode::NoiseNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _noise_prob(NOISE_PROB_RANGE[0], NOISE_PROB_RANGE[1]),
        _salt_prob (SALT_PROB_RANGE[0], SALT_PROB_RANGE[1]),
        _noise_value(NOISE_RANGE[0], NOISE_RANGE[1]),
        _salt_value(SALT_RANGE[0], SALT_RANGE[1])
{
}

void NoiseNode::create_node() {
    if(_node)
        return;

    _noise_prob.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _salt_prob.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _noise_value.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _salt_value.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    vx_scalar seed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_seed);

    _node = vxExtrppNode_Noise(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(),_noise_prob.default_array(), _salt_prob.default_array(), _noise_value.default_array(), _salt_value.default_array(), seed, _input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Noise (vxExtrppNode_Noise) node failed: "+ TOSTR(status))
}

void NoiseNode::init( float noise_prob, float salt_prob, float noise_value , float salt_value,int seed) {
    _noise_prob.set_param(noise_prob);
    _salt_prob.set_param(salt_prob);
    _noise_value.set_param(noise_value);
    _salt_value.set_param(salt_value);
    _seed=seed;
}

void NoiseNode::init( FloatParam* noise_prob, FloatParam* salt_prob, FloatParam* noise_value, FloatParam* salt_value, int seed) {
    _noise_prob.set_param(core(noise_prob));
    _salt_prob.set_param(core(salt_prob));
    _noise_value.set_param(core(noise_value));
    _salt_value.set_param(core(salt_value));
    _seed=seed;
}

void NoiseNode::update_node() {
    _noise_prob.update_array();
    _salt_prob.update_array();
    _noise_value.update_array();
    _salt_value.update_array();
}

