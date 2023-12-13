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

SnPNoiseNode::SnPNoiseNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : Node(inputs, outputs),
                                                                                                      _noise_prob(NOISE_PROB_RANGE[0], NOISE_PROB_RANGE[1]),
                                                                                                      _salt_prob(SALT_PROB_RANGE[0], SALT_PROB_RANGE[1]),
                                                                                                      _salt_value(SALT_RANGE[0], SALT_RANGE[1]),
                                                                                                      _pepper_value(PEPPER_RANGE[0], PEPPER_RANGE[1]) {}

void SnPNoiseNode::create_node() {
    if (_node)
        return;

    _noise_prob.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _salt_prob.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _salt_value.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _pepper_value.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    vx_scalar seed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_seed);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppNoise(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _noise_prob.default_array(), _salt_prob.default_array(),
                          _salt_value.default_array(), _pepper_value.default_array(), seed, input_layout_vx, output_layout_vx,roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Noise (vxExtRppNoise) node failed: " + TOSTR(status))
}

void SnPNoiseNode::init(float noise_prob, float salt_prob, float salt_value, float pepper_value, int seed) {
    _noise_prob.set_param(noise_prob);
    _salt_prob.set_param(salt_prob);
    _salt_value.set_param(salt_value);
    _pepper_value.set_param(pepper_value);
    _seed = seed;
}

void SnPNoiseNode::init(FloatParam* noise_prob_param, FloatParam* salt_prob_param,
                        FloatParam* salt_value_param, FloatParam* pepper_value_param, int seed) {
    _noise_prob.set_param(core(noise_prob_param));
    _salt_prob.set_param(core(salt_prob_param));
    _salt_value.set_param(core(salt_value_param));
    _pepper_value.set_param(core(pepper_value_param));
    _seed = seed;
}

void SnPNoiseNode::update_node() {
    _noise_prob.update_array();
    _salt_prob.update_array();
    _salt_value.update_array();
    _pepper_value.update_array();
}