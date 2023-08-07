/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#define FARNEBACK_FRAME_WIDTH 960                       // Farneback algorithm frame width
#define FARNEBACK_FRAME_HEIGHT 540                      // Farneback algorithm frame height
#define FARNEBACK_OUTPUT_FRAME_SIZE 518400u             // 960 * 540
#define FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE 1036800u   // 960 * 540 * 2
#define FARNEBACK_OUTPUT_RGB_SIZE 1555200u              // 960 * 540 * 3
#define HUE_CONVERSION_FACTOR 0.0019607843f             // ((1 / 360.0) * (180 / 255.0))

struct OpticalFlowToColorLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    Rpp32f *pSrc;
    Rpp8u * pDst;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    RpptDescPtr pRgbScaleDesc;
    RpptDescPtr pHSVScaleDesc;
    RpptGenericDescPtr pMotionVectorCartesianDesc;
    RpptGenericDescPtr pMotionVectorPolarDesc;
    RpptGenericDescPtr pMotionVectorGenericDesc;
    Rpp32u imageMinMaxArrLength;
    Rpp32f *imageMinMaxArr;
    
    Rpp32f *pMotionVectorCartesian;
    Rpp32f *pMotionVectorPolar;
    RpptROI *pSrcRoi;
    RpptROI *pDstRoi;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

static vx_status VX_CALLBACK refreshOpticalFlowToColor(vx_node node, const vx_reference *parameters, vx_uint32 num, OpticalFlowToColorLocalData *data) {
    vx_status status = VX_SUCCESS;

    for (unsigned i = 0; i < data->pDstDesc->n; i++) {
        *data->pDstRoi = {0, 0, FARNEBACK_FRAME_WIDTH, FARNEBACK_FRAME_HEIGHT};
    }
    void *roi_tensor_ptr;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else {
        return VX_ERROR_NOT_IMPLEMENTED;
    }
    data->pSrcRoi = reinterpret_cast<RpptROI *>(roi_tensor_ptr);
    if (data->inputLayout == vxTensorLayout::VX_NFHWC || data->inputLayout == vxTensorLayout::VX_NFCHW) {
        unsigned num_of_frames = data->inputTensorDims[1]; // Num of frames 'F'
        for (int n = data->inputTensorDims[0] - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for (unsigned f = 0; f < num_of_frames; f++) {
                data->pSrcRoi[index + f].xywhROI = data->pSrcRoi[n].xywhROI;
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK validateOpticalFlowToColor(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #3 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #6 type=%d (must be size)\n", scalar_type);

    // Check for input tensor
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims != 5) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: OpticalFlowToColor: tensor: #0 dimensions=%lu (must be equal to 5)\n", num_tensor_dims);

    // Check for output tensor
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims != 5) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: OpticalFlowToColor: tensor: #2 dimensions=%lu (must be equal to 5)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processOpticalFlowToColor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_ERROR_NOT_IMPLEMENTED;
    OpticalFlowToColorLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshOpticalFlowToColor(node, parameters, num, data);
    
    const RpptSubpixelLayout subpixelLayout = RpptSubpixelLayout::BGRtype;
    const RpptRoiType roiTypeXYWH = RpptRoiType::XYWH;
    const RpptRoiType roiTypeLTRB = RpptRoiType::LTRB;
    const RpptAngleType angleType = RpptAngleType::DEGREES;
const RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;

    std::cerr << "Optical Flow to color process ...\t";
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        std::cerr << "OP SEQUENCE LENGTH : " << data->outputTensorDims[1] << "\n";
        for (unsigned sequence = 0; sequence < data->inputTensorDims[0]; sequence++) {
            Rpp32f *d_motionVectorCartesian = data->pSrc + (sequence * data->inputTensorDims[1] * data->pSrcDesc->strides.nStride);
            Rpp8u *rgbDst = data->pDst + (sequence * data->outputTensorDims[1] * data->pDstDesc->strides.nStride);

            for (unsigned frame = 0; frame < data->outputTensorDims[1]; frame++) {
                Rpp32f *d_motionVectorsPolarF32Comp1 = data->pMotionVectorPolar + FARNEBACK_OUTPUT_FRAME_SIZE;
                Rpp32f *d_motionVectorsPolarF32Comp2 = d_motionVectorsPolarF32Comp1 + FARNEBACK_OUTPUT_FRAME_SIZE;
                Rpp32f *d_motionVectorsPolarF32Comp3 = d_motionVectorsPolarF32Comp2 + FARNEBACK_OUTPUT_FRAME_SIZE;

                // ****************************************************************** post-processing ******************************************************************
                // convert from cartesian to polar coordinates
                rppt_cartesian_to_polar_gpu(d_motionVectorCartesian, data->pMotionVectorCartesianDesc, data->pMotionVectorPolar, data->pMotionVectorPolarDesc, angleType, data->pDstRoi, roiTypeXYWH, data->handle->rppHandle);

                // all ops in stream1 need to complete before rppt_multiply_scalar_gpu executes on stream1 and rppt_image_min_max executes on stream2
                // hipStreamSynchronize(stream1);
                hipDeviceSynchronize();
                std::cerr << "-C2P-";

                // normalize polar angle from 0 to 1 in hip stream1
                rppt_multiply_scalar_gpu(d_motionVectorsPolarF32Comp1, data->pMotionVectorGenericDesc, d_motionVectorsPolarF32Comp1, data->pMotionVectorGenericDesc, HUE_CONVERSION_FACTOR, data->pDstRoi, roiTypeXYWH, data->handle->rppHandle);
                hipDeviceSynchronize(); // could be a hipStreamSynchronize(stream2);
                
                rppt_image_min_max_gpu(data->pMotionVectorPolar, data->pMotionVectorGenericDesc, data->imageMinMaxArr, data->imageMinMaxArrLength, data->pDstRoi, roiTypeXYWH, data->handle->rppHandle); // could be handle2

                // all ops in stream2 need to complete before rppt_normalize_minmax_gpu executes on stream2
                hipDeviceSynchronize(); // could be a hipStreamSynchronize(stream2);

                // normalize polar magnitude from 0 to 1 in hip stream2
                rppt_normalize_minmax_gpu(data->pMotionVectorPolar, data->pMotionVectorGenericDesc, d_motionVectorsPolarF32Comp3, data->pMotionVectorGenericDesc, data->imageMinMaxArr, data->imageMinMaxArrLength, 0.0f, 1.0f, data->pDstRoi, roiTypeXYWH, data->handle->rppHandle); // could be handle2

                // all ops in all streams need to complete before rppt_hsv_to_rgbbgr_gpu executes on stream1
                // hipStreamSynchronize(stream2);
                // hipStreamSynchronize(stream1);
                hipDeviceSynchronize(); // could be a hipStreamSynchronize(stream2); followed by hipStreamSynchronize(stream1);
                std::cerr << "-Minmax-" << data->imageMinMaxArr[0] << " & " << data->imageMinMaxArr[1];
                // fused bitDepth + layout + colorType conversion of F32-PLN3 HSV to U8-PKD3 BGR in hip stream1
                rppt_hsv_to_rgbbgr_gpu(d_motionVectorsPolarF32Comp1, data->pHSVScaleDesc, rgbDst, data->pDstDesc, subpixelLayout, data->handle->rppHandle);

                // all ops in all streams need to complete at end of post-processing
                hipDeviceSynchronize();
                std::cerr << "-HSV2RGB-";
                
                
                d_motionVectorCartesian += data->pSrcDesc->strides.nStride;
                rgbDst += data->pDstDesc->strides.nStride;
            }
            std::cerr << "Next sequence\n";
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    std::cerr << "Optical flow to color process ends ...\n";
    return return_status;
}

static vx_status VX_CALLBACK initializeOpticalFlowToColor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    OpticalFlowToColorLocalData *data = new OpticalFlowToColorLocalData;
    memset(data, 0, sizeof(OpticalFlowToColorLocalData));

    vx_enum input_tensor_dtype, output_tensor_dtype;
    int roi_type, input_layout, output_layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = (roi_type == 0) ? RpptRoiType::XYWH : RpptRoiType::LTRB;
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pSrcDesc, data->inputLayout, data->inputTensorDims);

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pDstDesc, data->outputLayout, data->outputTensorDims);
    data->pSrcDesc->n = data->pDstDesc->n = 1;

    data->pRgbScaleDesc = new RpptDesc;
    data->pRgbScaleDesc->dataType = RpptDataType::U8;
    data->pRgbScaleDesc->offsetInBytes = 0;
    size_t dims1[] = {1, 3, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH};
    fillDescriptionPtrfromDims(data->pRgbScaleDesc, vxTensorLayout::VX_NHWC, dims1);
    
    data->pHSVScaleDesc = new RpptDesc;
    data->pHSVScaleDesc->dataType = RpptDataType::F32;
    data->pHSVScaleDesc->offsetInBytes = 0;
    size_t dims3[] = {1, 3, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH};
    fillDescriptionPtrfromDims(data->pHSVScaleDesc, vxTensorLayout::VX_NCHW, dims3);


    data->pMotionVectorCartesianDesc = new RpptGenericDesc;
    data->pMotionVectorCartesianDesc->dataType = RpptDataType::F32;
    data->pMotionVectorCartesianDesc->offsetInBytes = 0;
    size_t dims5[] = {1, 2, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH};
    fillGenericDescriptionPtrfromDims(data->pMotionVectorCartesianDesc, vxTensorLayout::VX_NCHW, dims5);
    
    data->pMotionVectorPolarDesc = new RpptGenericDesc;
    data->pMotionVectorPolarDesc->dataType = RpptDataType::F32;
    data->pMotionVectorPolarDesc->offsetInBytes = 0;
    size_t dims6[] = {1, 2, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH};
    fillGenericDescriptionPtrfromDims(data->pMotionVectorPolarDesc, vxTensorLayout::VX_NCHW, dims6);
    data->pMotionVectorPolarDesc->strides[0] *= 2;
    
    data->pMotionVectorGenericDesc = new RpptGenericDesc;
    data->pMotionVectorGenericDesc->dataType = RpptDataType::F32;
    data->pMotionVectorGenericDesc->offsetInBytes = 0;
    size_t dims7[] = {1, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH};
    fillGenericDescriptionPtrfromDims(data->pMotionVectorGenericDesc, vxTensorLayout::VX_NCHW, dims7);
    
#if ENABLE_HIP
    hipMalloc(&data->pMotionVectorCartesian, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f));
    hipMalloc(&data->pMotionVectorPolar, FARNEBACK_OUTPUT_FRAME_SIZE * 4 * sizeof(Rpp32f));

    // preinitialize saturation channel portion of the buffer for HSV and reuse on every iteration in post-processing
    Rpp32f saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE];
    std::fill(&saturationChannel[0], &saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE - 1], 1.0f);
    hipMemcpy((data->pMotionVectorPolar + 2 * FARNEBACK_OUTPUT_FRAME_SIZE), saturationChannel, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyHostToDevice);
    
    // allocate post-processing buffer for imageMinMax
    data->imageMinMaxArrLength = 2;
    hipHostMalloc(&data->imageMinMaxArr, data->imageMinMaxArrLength * sizeof(Rpp32f));
    hipHostMalloc(&data->pDstRoi, data->pDstDesc->n * sizeof(RpptROI));
#endif
    refreshOpticalFlowToColor(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeOpticalFlowToColor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    OpticalFlowToColorLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_HIP
    if (data->pMotionVectorCartesian) hipFree(data->pMotionVectorCartesian);
    if (data->pMotionVectorPolar) hipFree(data->pMotionVectorPolar);
    if (data->imageMinMaxArr) hipHostFree(data->imageMinMaxArr);
#endif
    delete(data->pSrcDesc);
    delete(data->pDstDesc);
    delete(data->pRgbScaleDesc);
    delete(data->pHSVScaleDesc);
    delete(data->pMotionVectorCartesianDesc);
    delete(data->pMotionVectorPolarDesc);
    delete(data->pMotionVectorGenericDesc);
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete(data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
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
        return VX_ERROR_NOT_SUPPORTED;

    return VX_SUCCESS;
}

vx_status OpticalFlowToColor_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.OpticalFlowToColor",
                                       VX_KERNEL_RPP_OPTICALFLOWTOCOLOR,
                                       processOpticalFlowToColor,
                                       7,
                                       validateOpticalFlowToColor,
                                       initializeOpticalFlowToColor,
                                       uninitializeOpticalFlowToColor);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
