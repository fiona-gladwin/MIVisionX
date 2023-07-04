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

#include "internal_publishKernels.h"
#include "vx_ext_amd.h"
#define NUM_OF_DIMS 5

struct ToDecibelsLocalData
{
    RPPCommonHandle *handle;
    Rpp32u deviceType;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_float32 cutOffDB;
    vx_float32 multiplier;
    vx_float32 magnitudeReference;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dst_desc_ptr;
    void* roi_tensor_ptr_src;
    RpptROI* roi_ptr_src;
    void* roi_tensor_ptr_dst;
    RpptROI* roi_ptr_dst;
    RpptImagePatch *srcDims;
    size_t in_tensor_dims[NUM_OF_DIMS];
    size_t out_tensor_dims[NUM_OF_DIMS];
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

void update_destination_roi(const vx_reference *parameters, ToDecibelsLocalData *data)
{
    data->roi_ptr_dst = (RpptROI *)data->roi_tensor_ptr_dst;
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i=0; i < data->nbatchSize; i++)
    {
        data->roi_ptr_dst[i].xywhROI.xy.x = data->roi_ptr_src[i].xywhROI.xy.x;
        data->roi_ptr_dst[i].xywhROI.xy.y = data->roi_ptr_src[i].xywhROI.xy.y;
        // std::cerr<< "\n ToDecibels :" << data->roi_ptr_dst[i].xywhROI.xy.x;
        // std::cerr << "\n ToDecibels :" << data->roi_ptr_dst[i].xywhROI.xy.y;
    }

}

static vx_status VX_CALLBACK refreshToDecibels(vx_node node, const vx_reference *parameters, vx_uint32 num, ToDecibelsLocalData *data)
{
    vx_status status = VX_SUCCESS;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
        }
    }
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i = 0; i < data->nbatchSize; i++) {
        data->srcDims[i].height = data->roi_ptr_src[i].xywhROI.xy.y;
        data->srcDims[i].width = data->roi_ptr_src[i].xywhROI.xy.x;
    }
    update_destination_roi(parameters, data);
    return status;
}

static vx_status VX_CALLBACK validateToDecibels(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[NUM_OF_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 1);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processToDecibels(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ToDecibelsLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        refreshToDecibels(node, parameters, num, data);
        // rpp_status = rppt_ToDecibels_gpu((void *)data->hip_pSrc, data->src_desc_ptr, (void *)data->hip_pDst, data->src_desc_ptr,  data->alpha, data->beta, data->hip_roi_tensor_Ptr, data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        refreshToDecibels(node, parameters, num, data);
        rpp_status = rppt_to_decibels_host((float *)data->pSrc, data->src_desc_ptr, (float *)data->pDst, data->dst_desc_ptr, data->srcDims, data->cutOffDB, data->multiplier, data->magnitudeReference);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeToDecibels(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ToDecibelsLocalData *data = new ToDecibelsLocalData;
    // unsigned roiType;
    memset(data, 0, sizeof(*data));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[4], &data->cutOffDB));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[5], &data->multiplier));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->magnitudeReference));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->nbatchSize));

    // Querying for input tensor
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        data->src_desc_ptr->dataType = RpptDataType::F32;
    data->src_desc_ptr->offsetInBytes = 0;

    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1],VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        data->dst_desc_ptr->dataType = RpptDataType::F32;
    data->dst_desc_ptr->offsetInBytes = 0;

    // source_description_ptr
    data->src_desc_ptr->n = data->in_tensor_dims[0];
    data->src_desc_ptr->h = data->in_tensor_dims[2];
    data->src_desc_ptr->w = data->in_tensor_dims[1];
    data->src_desc_ptr->c = 1;
    data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
    data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
    data->src_desc_ptr->strides.cStride = 1;
    data->src_desc_ptr->numDims = 4;

    // source_description_ptr
    data->dst_desc_ptr->n = data->out_tensor_dims[0];
    data->dst_desc_ptr->w = data->out_tensor_dims[1];
    data->dst_desc_ptr->h = data->out_tensor_dims[2];
    data->dst_desc_ptr->c = 1;
    data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    data->dst_desc_ptr->strides.cStride = 1;
    data->dst_desc_ptr->numDims = 4;
    data->srcDims = (RpptImagePatch *)calloc(data->src_desc_ptr->n, sizeof(RpptImagePatch));

    refreshToDecibels(node, parameters, num, data);
    STATUS_ERROR_CHECK(createGraphHandle(node, &data->handle, data->src_desc_ptr->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeToDecibels(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ToDecibelsLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseGraphHandle(node, data->handle, data->deviceType));
    free(data->srcDims);

    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status ToDecibels_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.ToDecibels",
                                       VX_KERNEL_RPP_TODECIBELS,
                                       processToDecibels,
                                       9,
                                       validateToDecibels,
                                       initializeToDecibels,
                                       uninitializeToDecibels);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}