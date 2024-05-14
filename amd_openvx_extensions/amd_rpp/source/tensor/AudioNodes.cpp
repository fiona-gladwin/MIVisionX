/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "audio_rpp.h"
#include "internal_publishKernels.h"

struct AudioLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32s *pSrcRoi;
    Rpp32s *pDstRoi;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
    vxRppAudioAugmentationName audio_augmentation;
    AudioAugmentationData *augmentationSpecificData;
};

void update_destination_roi(AudioLocalData *data, RpptROI *src_roi, RpptROI *dst_roi) {
    float scale_ratio;
    for (unsigned i = 0; i < data->pSrcDesc->n; i++) {
        scale_ratio = (data->augmentationSpecificData->resample.pInRateTensor[i] != 0) ? (data->augmentationSpecificData->resample.pOutRateTensor[i] / static_cast<float>(data->augmentationSpecificData->resample.pInRateTensor[i])) : 0;
        dst_roi[i].xywhROI.roiWidth = static_cast<int>(std::ceil(scale_ratio * src_roi[i].xywhROI.roiWidth));
        dst_roi[i].xywhROI.roiHeight = src_roi[i].xywhROI.roiHeight;
    }
}

void copy_src_dims_and_update_dst_roi(AudioLocalData *data, RpptROI *srcRoi, RpptROI *dstRoi) {
    for (unsigned i = 0; i < data->inputTensorDims[0]; i++) {
        data->augmentationSpecificData->melFilter.pSrcDims[i * 2] = srcRoi[i].xywhROI.roiWidth;
        data->augmentationSpecificData->melFilter.pSrcDims[i * 2 + 1] = srcRoi[i].xywhROI.roiHeight;
        dstRoi[i].xywhROI.roiWidth = data->augmentationSpecificData->melFilter.nfilter;
        dstRoi[i].xywhROI.roiHeight = srcRoi[i].xywhROI.roiHeight;
    }
}

void updateDstRoi(AudioLocalData *data, RpptROI *src_roi, RpptROI *dst_roi) {
    const Rpp32s num_frames = ((data->augmentationSpecificData->spectrogram.nfft / 2) + 1);
    for (unsigned i = 0; i < data->inputTensorDims[0]; i++) {
        data->augmentationSpecificData->spectrogram.pSrcLength[i] = static_cast<int>(src_roi[i].xywhROI.roiWidth);
        if (data->outputLayout == vxTensorLayout::VX_NTF) {
            dst_roi[i].xywhROI.roiWidth = ((data->augmentationSpecificData->spectrogram.pSrcLength[i] - data->augmentationSpecificData->spectrogram.windowOffset) / data->augmentationSpecificData->spectrogram.windowStep) + 1;
            dst_roi[i].xywhROI.roiHeight = num_frames;
        } else if (data->outputLayout == vxTensorLayout::VX_NFT) {
            dst_roi[i].xywhROI.roiWidth = num_frames;
            dst_roi[i].xywhROI.roiHeight = ((data->augmentationSpecificData->spectrogram.pSrcLength[i] - data->augmentationSpecificData->spectrogram.windowOffset) / data->augmentationSpecificData->spectrogram.windowStep) + 1;
        }
    }
}

// **************** Initialize function for resample ****************
void initializeResample(ResampleData data, float quality) {
    int lobes = std::round(0.007 * data.quality * data.quality - 0.09 * data.quality + 3);
    int lookupSize = lobes * 64 + 1;
    windowed_sinc(data.window, lookupSize, lobes);
}

static vx_status VX_CALLBACK refreshAudioNode(vx_node node, const vx_reference *parameters, vx_uint32 num, AudioLocalData *data) {
    vx_status status = VX_SUCCESS;
    int nDim = 2;  // Num dimensions for audio tensor
    void *roi_tensor_ptr_src, *roi_tensor_ptr_dst;
    RpptROI *src_roi, *dst_roi;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        if (parameters[2])
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->augmentationSpecificData->nsr.pDst2, sizeof(data->augmentationSpecificData->nsr.pDst2)));
        if (parameters[3])
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        if (parameters[4])
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
    }
    if (roi_tensor_ptr_src) src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_src);
    if (roi_tensor_ptr_dst) dst_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_dst);
    if (data->audio_augmentation == vxRppAudioAugmentationName::RESAMPLE) {
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[10], 0, data->pSrcDesc->n, sizeof(float), data->augmentationSpecificData->resample.pInRateTensor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[9], VX_TENSOR_BUFFER_HOST, &data->augmentationSpecificData->resample.pOutRateTensor, sizeof(data->augmentationSpecificData->resample.pOutRateTensor)));
        RpptROI *src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_src);
        update_destination_roi(data, src_roi, dst_roi);
        for (unsigned i = 0; i < data->pSrcDesc->n; i++) {
            data->pSrcRoi[i * nDim] = src_roi[i].xywhROI.roiWidth;
            data->pSrcRoi[i * nDim + 1] = src_roi[i].xywhROI.roiHeight;
        }
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::PRE_EMPHASIS_FILTER) {
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[10], 0, data->pSrcDesc->n, sizeof(float), data->augmentationSpecificData->preEmphasis.pPreemphCoeff, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        for (int n = 0; n < data->inputTensorDims[0]; n++)
            data->augmentationSpecificData->preEmphasis.pSampleSize[n] = src_roi[n].xywhROI.roiWidth * src_roi[n].xywhROI.roiHeight;
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::DOWNMIX) {
        for (int n = 0; n < data->inputTensorDims[0]; n++) {
            data->pSrcRoi[n * 2] = src_roi[n].xywhROI.roiWidth;
            data->pSrcRoi[n * 2 + 1] = src_roi[n].xywhROI.roiHeight;
        }
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::TO_DECIBELS) {
        for (unsigned i = 0; i < data->inputTensorDims[0]; i++) {
            data->augmentationSpecificData->toDecibels.pSrcDims[i].width = src_roi[i].xywhROI.roiHeight;
            data->augmentationSpecificData->toDecibels.pSrcDims[i].height = src_roi[i].xywhROI.roiWidth;
        }
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::MEL_FILTER_BANK) {
        copy_src_dims_and_update_dst_roi(data, src_roi, dst_roi);
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::SPECTROGRAM) {
        updateDstRoi(data, src_roi, dst_roi);
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::NON_SILENT_REGION_DETECTION) {
        unsigned *src_roi_ptr = static_cast<unsigned *>(roi_tensor_ptr_src);
        for (unsigned i = 0, j = 0; i < data->inputTensorDims[0]; i++, j += 4)
            data->augmentationSpecificData->nsr.pSrcLength[i] = src_roi_ptr[j + 2];
    }
    return status;
}

static vx_status VX_CALLBACK validateAudioNodes(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;

    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Validate for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate Resample: tensor #0 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    // Validate for output parameters
    vx_tensor output;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate Resample: tensor #1 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    if (parameters[2]) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
        if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate Resample: tensor #1 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    }
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processAudioNodes(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    AudioLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshAudioNode(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        if (data->audio_augmentation == vxRppAudioAugmentationName::RESAMPLE) {
            rpp_status = rppt_resample_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc,
                                            data->augmentationSpecificData->resample.pInRateTensor, data->augmentationSpecificData->resample.pOutRateTensor, data->pSrcRoi, data->augmentationSpecificData->resample.window, data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::PRE_EMPHASIS_FILTER) {
            rpp_status = rppt_pre_emphasis_filter_host((float *)data->pSrc, data->pSrcDesc, (float *)data->pDst, data->pDstDesc, (Rpp32s *)data->augmentationSpecificData->preEmphasis.pSampleSize, data->augmentationSpecificData->preEmphasis.pPreemphCoeff, RpptAudioBorderType(data->augmentationSpecificData->preEmphasis.borderType), data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::DOWNMIX) {
            rpp_status = rppt_down_mixing_host((float *)data->pSrc, data->pSrcDesc, (float *)data->pDst, data->pDstDesc, (Rpp32s *)data->pSrcRoi, false, data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::TO_DECIBELS) {
            rpp_status = rppt_to_decibels_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->augmentationSpecificData->toDecibels.pSrcDims, data->augmentationSpecificData->toDecibels.cutOffDB, data->augmentationSpecificData->toDecibels.multiplier, data->augmentationSpecificData->toDecibels.referenceMagnitude, data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::MEL_FILTER_BANK) {
            rpp_status = rppt_mel_filter_bank_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->augmentationSpecificData->melFilter.pSrcDims, data->augmentationSpecificData->melFilter.freqHigh, data->augmentationSpecificData->melFilter.freqLow,
                                                   data->augmentationSpecificData->melFilter.melFormula, data->augmentationSpecificData->melFilter.nfilter, data->augmentationSpecificData->melFilter.sampleRate, data->augmentationSpecificData->melFilter.normalize, data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::SPECTROGRAM) {
            rpp_status = rppt_spectrogram_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->augmentationSpecificData->spectrogram.pSrcLength, data->augmentationSpecificData->spectrogram.centerWindows, data->augmentationSpecificData->spectrogram.reflectPadding,
                                               data->augmentationSpecificData->spectrogram.pWindowFn, data->augmentationSpecificData->spectrogram.nfft, data->augmentationSpecificData->spectrogram.power, data->augmentationSpecificData->spectrogram.windowLength, data->augmentationSpecificData->spectrogram.windowStep, data->handle->rppHandle);
        } else if (data->audio_augmentation == vxRppAudioAugmentationName::NON_SILENT_REGION_DETECTION) {
            rpp_status = rppt_non_silent_region_detection_host(data->pSrc, data->pSrcDesc, data->augmentationSpecificData->nsr.pSrcLength, (Rpp32s*)data->pDst, data->augmentationSpecificData->nsr.pDst2, data->augmentationSpecificData->nsr.cutOffDB, data->augmentationSpecificData->nsr.windowLength, data->augmentationSpecificData->nsr.referencePower, data->augmentationSpecificData->nsr.resetInterval, data->handle->rppHandle);
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeAudioNodes(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    AudioLocalData *data = new AudioLocalData;
    memset(data, 0, sizeof(AudioLocalData));

    vx_enum input_tensor_dtype, output_tensor_dtype;
    vx_int32 input_layout, output_layout, aug_enum;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &input_layout));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[8], &output_layout));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &aug_enum));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[12], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);
    data->audio_augmentation = static_cast<vxRppAudioAugmentationName>(aug_enum);

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;
    fillAudioDescriptionPtrFromDims(data->pSrcDesc, data->inputTensorDims, data->inputLayout);

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillAudioDescriptionPtrFromDims(data->pDstDesc, data->outputTensorDims, data->outputLayout);

    if (parameters[3]) data->pSrcRoi = new Rpp32s[data->pSrcDesc->n * 2];
    if (parameters[4]) data->pDstRoi = new Rpp32s[data->pSrcDesc->n * 2];

    if (data->audio_augmentation == vxRppAudioAugmentationName::RESAMPLE) {
        float array_values[1];
        data->augmentationSpecificData->resample.pInRateTensor = new float[data->pSrcDesc->n];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, 1, sizeof(float), array_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->resample.quality = array_values[0];
        initializeResample(data->augmentationSpecificData->resample, data->augmentationSpecificData->resample.quality);
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::PRE_EMPHASIS_FILTER) {
        data->augmentationSpecificData->preEmphasis.pSampleSize = new unsigned[data->pSrcDesc->n];
        data->augmentationSpecificData->preEmphasis.pPreemphCoeff = new float[data->pSrcDesc->n];
        int array_values[1];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, 1, sizeof(int), array_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->preEmphasis.borderType = array_values[0];
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::TO_DECIBELS) {
        data->augmentationSpecificData->toDecibels.pSrcDims = new RpptImagePatch[data->pSrcDesc->n];
        float array_values[3];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, 3, sizeof(float), array_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->toDecibels.cutOffDB = array_values[0];
        data->augmentationSpecificData->toDecibels.multiplier = array_values[1];
        data->augmentationSpecificData->toDecibels.referenceMagnitude = array_values[2];
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::MEL_FILTER_BANK) {
        float int_values[3];
        int float_values[3];
        data->augmentationSpecificData->melFilter.pSrcDims = new Rpp32s[data->pSrcDesc->n * 2];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, 3, sizeof(int), int_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, 3, sizeof(float), float_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->melFilter.freqLow = float_values[0];
        data->augmentationSpecificData->melFilter.freqHigh = float_values[1];
        data->augmentationSpecificData->melFilter.sampleRate = float_values[2];
        data->augmentationSpecificData->melFilter.melFormula = static_cast<RpptMelScaleFormula>(int_values[0]);
        data->augmentationSpecificData->melFilter.nfilter = int_values[1];
        data->augmentationSpecificData->melFilter.normalize = static_cast<bool>(int_values[2]);
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::SPECTROGRAM) {
        int array_values[6];
        data->augmentationSpecificData->spectrogram.pSrcLength = new int[data->pSrcDesc->n];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, 6, sizeof(int), array_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->spectrogram.centerWindows = static_cast<bool>(array_values[0]);
        data->augmentationSpecificData->spectrogram.reflectPadding = static_cast<bool>(array_values[1]);
        data->augmentationSpecificData->spectrogram.power = array_values[2];
        data->augmentationSpecificData->spectrogram.nfft = array_values[3];
        data->augmentationSpecificData->spectrogram.windowLength = array_values[4];
        data->augmentationSpecificData->spectrogram.windowStep = array_values[5];
        data->augmentationSpecificData->spectrogram.windowOffset = (!data->augmentationSpecificData->spectrogram.centerWindows) ? data->augmentationSpecificData->spectrogram.windowLength : 0;
        data->augmentationSpecificData->spectrogram.pWindowFn = new float[data->augmentationSpecificData->spectrogram.windowLength];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->augmentationSpecificData->spectrogram.windowLength, sizeof(float), data->augmentationSpecificData->spectrogram.pWindowFn, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    } else if (data->audio_augmentation == vxRppAudioAugmentationName::NON_SILENT_REGION_DETECTION) {
        float int_values[2];
        int float_values[2];
        data->augmentationSpecificData->nsr.pSrcLength = new int[data->pSrcDesc->n];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, 2, sizeof(int), int_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, 2, sizeof(float), float_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->augmentationSpecificData->nsr.cutOffDB = float_values[0];
        data->augmentationSpecificData->nsr.referencePower = float_values[1];
        data->augmentationSpecificData->nsr.windowLength = int_values[0];
        data->augmentationSpecificData->nsr.resetInterval = int_values[1];
    }

    refreshAudioNode(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeAudioNodes(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    AudioLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete[] data->pSrcRoi;
    delete data->pSrcDesc;
    delete data->pDstDesc;
    delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32 &supported_target_affinity) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status AudioNodes_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.AudioNodes",
                                       VX_KERNEL_RPP_AUDIONODES,
                                       processAudioNodes,
                                       13,
                                       validateAudioNodes,
                                       initializeAudioNodes,
                                       uninitializeAudioNodes);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));   // Input
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));  // Output1
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));  // Output2
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));   // pSrcROI
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));   // pDstROI
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));    // Int scalars
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));    // Float scalars
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));   // input layout
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));   // output layout
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));   // Tensor-Sample Rate
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));    // Array-Sample Rate
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));  // Enum to identify the augmentation
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));  // Device Type
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
