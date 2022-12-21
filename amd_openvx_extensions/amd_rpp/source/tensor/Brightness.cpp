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

#include "internal_publishKernels.h"

struct BrightnessLocalData {
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u deviceType;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_float32 *alpha;
    vx_float32 *beta;
    RpptDescPtr srcDescPtr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dstDescPtr;
    void *roiTensorPtr;
    RpptROI *roiPtr;
    RpptRoiType roiType;
    Rpp32s inputLayout;
    Rpp32s outputLayout;
    size_t inputTensorDims[NUM_OF_DIMS];
    size_t ouputTensorDims[NUM_OF_DIMS];
    vx_enum inputTensorType;
    vx_enum outputTensorType;
#if ENABLE_HIP
    void *pSrc_dev;
    void *pDst_dev;
#endif
};

static vx_status VX_CALLBACK refreshBrightness(vx_node node, const vx_reference *parameters, vx_uint32 num, BrightnessLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->nbatchSize, sizeof(vx_float32), data->alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_float32), data->beta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->roiTensorPtr, sizeof(data->roiTensorPtr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc_dev, sizeof(data->pSrc_dev)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst_dev, sizeof(data->pDst_dev)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->roiTensorPtr, sizeof(vx_uint32)));
        if (data->inputTensorType == vx_type_e::VX_TYPE_UINT8 && data->outputTensorType == vx_type_e::VX_TYPE_UINT8) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        } else if (data->inputTensorType == vx_type_e::VX_TYPE_INT8 && data->outputTensorType == vx_type_e::VX_TYPE_INT8) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        } else if (data->inputTensorType == vx_type_e::VX_TYPE_FLOAT32 && data->outputTensorType == vx_type_e::VX_TYPE_FLOAT32) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
#if defined(AMD_FP16_SUPPORT)
        else if (data->inputTensorType == vx_type_e::VX_TYPE_FLOAT16 && data->outputTensorType == vx_type_e::VX_TYPE_FLOAT16) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float16)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        }
#endif
    }
    data->roiPtr = (RpptROI *)data->roiTensorPtr;
    if((data->inputLayout == 2 || data->inputLayout == 3)) { // For NFCHW and NFHWC formats
        unsigned num_of_frames = data->inputTensorDims[1]; // Num of frames 'F'
        for(int n = data->nbatchSize - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for(int f = 0; f < num_of_frames; f++) {
                data->alpha[index + f] = data->alpha[n];
                data->beta[index + f] = data->beta[n];
                data->roiPtr[index + f].xywhROI.xy.x = data->roiPtr[n].xywhROI.xy.x;
                data->roiPtr[index + f].xywhROI.xy.y = data->roiPtr[n].xywhROI.xy.y;
                data->roiPtr[index + f].xywhROI.roiWidth = data->roiPtr[n].xywhROI.roiWidth;
                data->roiPtr[index + f].xywhROI.roiHeight = data->roiPtr[n].xywhROI.roiHeight;
            }
        }
    }

    return status;
}

static vx_status VX_CALLBACK validateBrightness(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    vx_tensor input;
    vx_parameter input_param;
    size_t in_num_tensor_dims;
    input_param = vxGetParameterByIndex(node, 0);
    STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(input, VX_TENSOR_NUMBER_OF_DIMS, &in_num_tensor_dims, sizeof(in_num_tensor_dims)));
    if(in_num_tensor_dims < 4)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Brightness: tensor: #0 dimensions=%lu (must be greater than or equal to 4)\n", in_num_tensor_dims);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t out_num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[NUM_OF_DIMS];
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

static vx_status VX_CALLBACK processBrightness(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    BrightnessLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        refreshBrightness(node, parameters, num, data);
        rpp_status = rppt_brightness_gpu((void *)data->pSrc_dev, data->srcDescPtr, (void *)data->pDst_dev, data->dstDescPtr,  data->alpha, data->beta, data->roiPtr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        refreshBrightness(node, parameters, num, data);
        rpp_status = rppt_brightness_host(data->pSrc, data->srcDescPtr, data->pDst, data->dstDescPtr, data->alpha, data->beta, data->roiPtr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeBrightness(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    BrightnessLocalData *data = new BrightnessLocalData;
    int roi_type;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    THROW("initialize : Brightness OpenCL backend is not supported")
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->inputLayout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->outputLayout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[8], &data->nbatchSize));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
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

    data->alpha = (vx_float32 *)malloc(sizeof(vx_float32) * data->srcDescPtr->n);
    data->beta = (vx_float32 *)malloc(sizeof(vx_float32) * data->srcDescPtr->n);
    refreshBrightness(node, parameters, num, data);

#if ENABLE_HIP
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.hipstream, data->nbatchSize);
#endif
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeBrightness(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    BrightnessLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_HIP
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->alpha);
    free(data->beta);
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

vx_status Brightness_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Brightness",
                                       VX_KERNEL_RPP_BRIGHTNESS,
                                       processBrightness,
                                       10,
                                       validateBrightness,
                                       initializeBrightness,
                                       uninitializeBrightness);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}