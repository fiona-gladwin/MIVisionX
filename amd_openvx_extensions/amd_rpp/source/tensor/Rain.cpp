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

struct RainLocalData {
    vxRppHandle * handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32u nbatchSize;
    vx_float32 *rainValue;
    vx_uint32 *rainWidth;
    vx_uint32 *rainHeight;
    vx_float32 *rainTransperancy;
    RpptDescPtr srcDescPtr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dstDescPtr;
    void *roiTensorPtr;
    RpptROI *roiPtr;
    RpptRoiType roiType;
    Rpp32s inputLayout;
    Rpp32s outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t ouputTensorDims[RPP_MAX_TENSOR_DIMS];
    vx_enum inputTensorType;
    vx_enum outputTensorType;
    RppiSize *srcDimensions; // TBR : Not present in tensor
    RppiSize maxSrcDimensions;  // TBR : Not present in tensor
};

static vx_status VX_CALLBACK refreshRain(vx_node node, const vx_reference *parameters, vx_uint32 num, RainLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->srcDescPtr->n, sizeof(vx_float32), data->rainValue, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->srcDescPtr->n, sizeof(vx_uint32), data->rainWidth, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->srcDescPtr->n, sizeof(vx_uint32), data->rainHeight, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->srcDescPtr->n, sizeof(vx_float32), data->rainTransperancy, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    for (int i = 0; i < data->inputTensorDims[0]; i++)
        {
            data->srcDimensions[i].width = data->srcDescPtr->w;
            data->srcDimensions[i].height = data->srcDescPtr->h;
        }
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->roiTensorPtr, sizeof(data->roiTensorPtr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->roiTensorPtr, sizeof(data->roiTensorPtr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    data->roiPtr = (RpptROI *)data->roiTensorPtr;
    if((data->inputLayout == 2 || data->inputLayout == 3)) { // For NFCHW and NFHWC formats
        unsigned num_of_frames = data->inputTensorDims[1]; // Num of frames 'F'
        for(int n = data->srcDescPtr->n - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for(int f = 0; f < num_of_frames; f++) {
                data->rainValue[index + f] = data->rainValue[n];
                data->rainWidth[index + f] = data->rainWidth[n];
                data->rainHeight[index + f] = data->rainHeight[n];
                data->rainTransperancy[index + f] = data->rainTransperancy[n];
                data->roiPtr[index + f].xywhROI.xy.x = data->roiPtr[n].xywhROI.xy.x;
                data->roiPtr[index + f].xywhROI.xy.y = data->roiPtr[n].xywhROI.xy.y;
                data->roiPtr[index + f].xywhROI.roiWidth = data->roiPtr[n].xywhROI.roiWidth;
                data->roiPtr[index + f].xywhROI.roiHeight = data->roiPtr[n].xywhROI.roiHeight;
            }
        }
    }

    return status;
}

static vx_status VX_CALLBACK validateRain(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #10 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    vx_tensor input;
    vx_parameter input_param;
    size_t in_num_tensor_dims;
    input_param = vxGetParameterByIndex(node, 0);
    STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(input, VX_TENSOR_NUMBER_OF_DIMS, &in_num_tensor_dims, sizeof(in_num_tensor_dims)));
    if(in_num_tensor_dims < 4)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Rain: tensor: #0 dimensions=%lu (must be greater than or equal to 4)\n", in_num_tensor_dims);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t out_num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 2);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &out_num_tensor_dims, sizeof(out_num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_tensor_dims, sizeof(out_num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&input);
    vxReleaseTensor(&output);
    vxReleaseParameter(&input_param);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processRain(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    RainLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        refreshRain(node, parameters, num, data);
        if(data->dstDescPtr->c==1 ) 
        rpp_status = rppi_rain_u8_pln1_batchPD_gpu((void *)data->pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->pDst, data->rainValue, data->rainWidth, data->rainHeight, data->rainTransperancy, data->nbatchSize, data->handle->rppHandle);
        else 
        rpp_status = rppi_rain_u8_pkd3_batchPD_gpu((void *)data->pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->pDst, data->rainValue, data->rainWidth, data->rainHeight, data->rainTransperancy, data->nbatchSize, data->handle->rppHandle);

        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        refreshRain(node, parameters, num, data);
        if(data->dstDescPtr->c==1 ) 
        rpp_status = rppi_rain_u8_pln1_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->rainValue, data->rainWidth, data->rainHeight, data->rainTransperancy, data->nbatchSize, data->handle->rppHandle);
        else 
        rpp_status = rppi_rain_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->rainValue, data->rainWidth, data->rainHeight, data->rainTransperancy, data->nbatchSize, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeRain(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RainLocalData *data = new RainLocalData;
    memset(data, 0, sizeof(*data));
    int roi_type;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->inputLayout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &data->outputLayout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[10], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = (roi_type == 0) ? RpptRoiType::XYWH : RpptRoiType::LTRB;

    // Querying for input tensor
    data->srcDescPtr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->srcDescPtr->numDims, sizeof(data->srcDescPtr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->srcDescPtr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->inputTensorType, sizeof(data->inputTensorType)));
    data->srcDescPtr->dataType = getRpptDataType(data->inputTensorType);
    data->srcDescPtr->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->srcDescPtr, data->inputLayout, data->inputTensorDims);

    // Querying for output tensor
    data->dstDescPtr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->dstDescPtr->numDims, sizeof(data->dstDescPtr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->ouputTensorDims, sizeof(vx_size) * data->dstDescPtr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->outputTensorType, sizeof(data->outputTensorType)));
    data->dstDescPtr->dataType = getRpptDataType(data->outputTensorType);
    data->dstDescPtr->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->dstDescPtr, data->outputLayout, data->ouputTensorDims);

    data->rainValue = (vx_float32 *)malloc(sizeof(vx_float32) * data->srcDescPtr->n);
    data->rainWidth = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->srcDescPtr->n);
    data->rainHeight = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->srcDescPtr->n);
    data->rainTransperancy = (vx_float32 *)malloc(sizeof(vx_float32) * data->srcDescPtr->n);
    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->srcDescPtr->n);
    
    data->nbatchSize = data->srcDescPtr->n;
    data->maxSrcDimensions.height = data->srcDescPtr->h;
    data->maxSrcDimensions.width = data->srcDescPtr->w;
    refreshRain(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->srcDescPtr->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeRain(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RainLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    free(data->rainHeight);
    free(data->rainWidth);
    free(data->rainTransperancy);
    free(data->rainValue);
    free(data->srcDimensions);

    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status Rain_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Rain",
                                       VX_KERNEL_RPP_RAIN,
                                       processRain,
                                       11,
                                       validateRain,
                                       initializeRain,
                                       uninitializeRain);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
