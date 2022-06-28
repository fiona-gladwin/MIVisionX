/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#define NUM_OF_DIMS 4

struct ResizeMirrorNormalizeLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    Rpp32u nbatchSize;
    RpptDescPtr dst_desc_ptr;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *resize_h;
    vx_uint32 *resize_w;
    vx_uint32 *interpolation_type;
    vx_float32 *mean;
    vx_float32 *std_dev;
    vx_uint32 *mirror;
    vx_bool is_packed;                  // if true NHWC else NCHW
    size_t in_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    size_t out_tensor_dims[NUM_OF_DIMS];
    vx_uint32 channels;
    vx_uint32 batch_size;
    RpptROI *roi_tensor_Ptr;
    RpptRoiType roiType;
    vx_uint32 chnShift; // NHWC to NCHW
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    // RpptInterpolationType interpolation_type;
    RpptImagePatch *dstimgsize;

#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};
/*
 * Number of Dims is 4
 * If is_packed is true - NHWC
 * Dims[0] = N , Dims[1] = H, Dims[2] = W, Dims[3] = C
 * If is_packed is true - NCHW
 * Dims[0] = N , Dims[1] = C, Dims[2] = H, Dims[3] = W
 */
static vx_status VX_CALLBACK refreshResizeMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num, ResizeMirrorNormalizeLocalData *data)
{
    std::cerr<<"Refresh_ resizze_mirror_noramlize/n";

    vx_status status = VX_SUCCESS;
    std::cerr<<"\ndata->nbatchSize    "<<data->nbatchSize;

    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize * 4, sizeof(unsigned), data->roi_tensor_Ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_uint32), data->resize_w, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(vx_uint32), data->resize_h, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    std::cerr << "data->resize_w "<<*data->resize_w<<"  "<<*data->resize_h;
    // STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->nbatchSize, sizeof(vx_uint32), data->interpolation_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[10], &data->is_packed));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[7], 0, data->nbatchSize*3, sizeof(vx_float32), data->mean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[8], 0, data->nbatchSize*3, sizeof(vx_float32), data->std_dev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[9], 0, data->nbatchSize, sizeof(vx_uint32), data->mirror, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &data->chnShift));
    for (int i = 0; i < data->nbatchSize; i++)
    {
        // data->roi_tensor_Ptr[i].xywhROI.roiWidth =data->resize_w[i];
        // data->roi_tensor_Ptr[i].xywhROI.roiHeight =data->resize_h[i];
        data->dstimgsize[i].width = data->resize_w[i];
        data->dstimgsize[i].height = data->resize_h[i];
        std::cerr << "data->resize_w[i] " << data->resize_w[i] << "  " << data->resize_w[i]<<"\n";
    }
    std::cerr<<"data->dstimgsize[i].widthhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"<<data->dstimgsize[0].width<<"  "<<data->dstimgsize[0].height<<"\n";
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        std::cerr << "\n Comes to CPU";

        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //
        // STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float16)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        // }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8 && data->out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            std::cerr << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^";
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if(in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        // }
    }
    return status;
}

static vx_status VX_CALLBACK validateResizeMirrorNormalize(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    std::cerr<<"validation_resize_mirror_normalize\n";
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #11 type=%d (must be a boolean size)\n", scalar_type);

    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[12], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #2 type=%d (must be a boolean size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[13], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #13 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[14], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #14 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[15], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #15 type=%d (must be size)\n", scalar_type);
    std::cerr<<"validation_resize_mirror_normalize22222\n";

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[4];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 2);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processResizeMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    std::cerr<<"process_resize_mirror_normalize\n";
    vx_status vxstatus;
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ResizeMirrorNormalizeLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    Rpp32u N, C;
    N = data->nbatchSize;
    C = data->channels;

    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        vxstatus = refreshResizeMirrorNormalize(node, parameters, num, data);

        for (int i = 0; i < data->nbatchSize; i++)
        {
            std::cerr << "\n data->roi_tensor_Ptr values :: " << data->roi_tensor_Ptr[i].xywhROI.xy.x << " " << data->roi_tensor_Ptr[i].xywhROI.xy.y << " " << data->roi_tensor_Ptr[i].xywhROI.roiWidth << " " << data->roi_tensor_Ptr[i].xywhROI.roiHeight;
        }
        //     unsigned long long ioBufferSize = (unsigned long long)data->src_desc_ptr->h * (unsigned long long)data->src_desc_ptr->w * (unsigned long long)data->src_desc_ptr->c * (unsigned long long)data->src_desc_ptr->n;
        //     float *temp = ((float*)calloc( ioBufferSize,sizeof(float) ));

        //    if(1)
        //    {
        //             for (int i=0;i< ioBufferSize;i++)
        //             {
        //                 temp[i]=(float)*((unsigned char*)(data->pSrc) + i);
        //             }
        //    }

        int *temp = ((int *)calloc(100, sizeof(int)));
        std::cerr << "printing pSrc\n";
        
        // for (int i = 0; i < 100; i++)
        // {
        //     temp[i] = (int)*((unsigned char *)(data->pSrc) + i);
        //     std::cerr<<"\n "<<temp[i];
        //     // std::cerr<<" data->dstimgsize "<< data->dstimgsize[i].width<<" "<<data->dstimgsize[i].height<<"\n";
        //     // std::cerr<<"data->roi_tensor_Ptr[i].xywhROI.roiWidth" <<data->roi_tensor_Ptr[i].xywhROI.roiWidth<<"  "<<data->roi_tensor_Ptr[i].xywhROI.roiHeight<<"\n";

        //     // std::cerr<<temp[i]<<" ";
        // }
        std::cerr << " data->dstimgsize " << data->dstimgsize[0].width << " " << data->dstimgsize[0].height << "\n";

        std::cerr << "\n Gonna call RPP";
        // data->src_desc_ptr->dataType=RpptDataType::F32;
        // data->dst_desc_ptr->dataType=RpptDataType::F32;

        // rpp_status = rppt_resize_mirror_normalize_host(data->pSrc, data->src_desc_ptr,
        //                               data->pDst, data->dst_desc_ptr,
        //                               data->dstimgsize,
        //                               // data->interpolation_type,
        //                               RpptInterpolationType::BILINEAR,
        //                               data->mean,data->std_dev,
        //                               data->mirror,
        //                               data->roi_tensor_Ptr, data->roiType,
        //                               data->rppHandle);

        std::cerr << "\back to rpp";
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;

        //   float *temp = ((float*)calloc( 100,sizeof(float) ));

    //     for (int i = 0; i < 100; i++)
    //     {
    //         float temp1 = (int)*((unsigned char *)(data->pDst) + i);
    //         std::cout << temp1 << " ";
    //     }
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeResizeMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    std::cerr<<"intitalization_resize_mirror_normalize\n";
    ResizeMirrorNormalizeLocalData *data = new ResizeMirrorNormalizeLocalData;
    unsigned layout, roiType;
    memset(data, 0, sizeof(*data));
        std::cerr<<"intitalization_resize_mirror_normalize2222\n";

// std::cerr<<"\n INIT2";
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    // STATUS_extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrection(vx_graph graph, vx_tensor pSrc,vx_array srcROI, vx_tensor pDst, vx_array gamma,vx_scalar layout, vx_scalar roiType, vx_uint32 nbatchSize);

    ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    std::cerr<<"intitalization_resize_mirror_normalize33333\n";

    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[15], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        std::cerr<<"intitalization_resize_mirror_normalize 4444\n";

    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[14], &data->nbatchSize));
            std::cerr<<"intitalization_resize_mirror_normalize 5555\n";

    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[12], &layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        std::cerr<<"intitalization_resize_mirror_normalize 666\n";

    std::cerr << "layoutttttttttttt " << layout;
    // STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[13], &layout));
    // std::cerr<<"\n layout "<<layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[13], &roiType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (roiType == 1)
    {
        data->roiType = RpptRoiType::XYWH;
        std::cerr<<"roiType  XYWH\n";

    }
    else
    {
        data->roiType = RpptRoiType::LTRB;
        std::cerr<<"roiType  LTRB\n";

    }
    // Querying for input tensor
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8)
    {
        std::cerr << "datatype check UINT8";
        data->src_desc_ptr->dataType = RpptDataType::U8;
    }
    else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
    {
        std::cerr << "datatype check FLOAT32";
        data->src_desc_ptr->dataType = RpptDataType::F32;
    }
    // else if (data->src_desc_ptr->dataType == vx_type_e::VX_TYPE_FLOAT16)
    //     data->src_desc_ptr->dataType = RpptDataType::F16;
    else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8)
    {
        std::cerr << "datatype check INT8";
        data->src_desc_ptr->dataType = RpptDataType::I8;
    }

    data->src_desc_ptr->offsetInBytes = 0;
    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    if (data->out_tensor_type == vx_type_e::VX_TYPE_UINT8)
    {
        data->dst_desc_ptr->dataType = RpptDataType::U8;
        std::cerr << "dst datatype check UINT8";
    }
    else if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
    {
        data->dst_desc_ptr->dataType = RpptDataType::F32;
        std::cerr << "dst datatype check FLOAT32";
    }
    // else if (data->src_desc_ptr->dataType == vx_type_e::VX_TYPE_FLOAT16)
    //     data->src_desc_ptr->dataType = RpptDataType::F16;
    else if (data->out_tensor_type == vx_type_e::VX_TYPE_INT8)
    {
        std::cerr << "dst datatype check INT8";

        data->dst_desc_ptr->dataType = RpptDataType::I8;
    }
    data->dst_desc_ptr->offsetInBytes = 0;
    // declaring and pushing values to roi_tensor_Ptr
    data->roi_tensor_Ptr = (RpptROI *)calloc(data->nbatchSize, sizeof(RpptROI));
    data->dstimgsize = (RpptImagePatch *)calloc(data->nbatchSize, sizeof(RpptImagePatch));

    layout = 0;
    if (layout == 0) // NHWC
    {
        // source_description_ptr
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[1];
        data->src_desc_ptr->w = data->in_tensor_dims[2];
        data->src_desc_ptr->c = data->in_tensor_dims[3];
        std::cerr << "\n n h w c " << data->src_desc_ptr->n << " " << data->src_desc_ptr->h << " " << data->src_desc_ptr->w << " " << data->src_desc_ptr->c;
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
        data->src_desc_ptr->strides.cStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NHWC;

        // destination_description_ptr
        data->dst_desc_ptr->n = data->out_tensor_dims[0];
        data->dst_desc_ptr->h = data->out_tensor_dims[1];
        data->dst_desc_ptr->w = data->out_tensor_dims[2];
        data->dst_desc_ptr->c = data->out_tensor_dims[3];
        std::cerr << "\n dest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
        data->dst_desc_ptr->strides.cStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NHWC;
    }
    else // NCHW
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[2];
        data->src_desc_ptr->w = data->in_tensor_dims[3];
        data->src_desc_ptr->c = data->in_tensor_dims[1];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NCHW;

        data->dst_desc_ptr->n = data->out_tensor_dims[0];
        data->dst_desc_ptr->h = data->out_tensor_dims[2];
        data->dst_desc_ptr->w = data->out_tensor_dims[3];
        data->dst_desc_ptr->c = data->out_tensor_dims[1];
        std::cerr << "\ndest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.cStride = data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NHWC;
    }

    data->interpolation_type = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->resize_w = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->resize_h = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->mean = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize*3);
    data->std_dev = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize*3);
    data->mirror = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);


    refreshResizeMirrorNormalize(node, parameters, num, data);
#if ENABLE_OPENCL
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.cmdq, data->nbatchSize);
    std::cerr << "\n Finished rppCreateWithStreamAndBatchSize";
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeResizeMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ResizeMirrorNormalizeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->interpolation_type);
    free(data->resize_w);
    free(data->resize_h);
    free(data->mean);
    free(data->std_dev);
    free(data->mirror);
    free(data->roi_tensor_Ptr);
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

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif

    return VX_SUCCESS;
}

vx_status ResizeMirrorNormalize_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.ResizeMirrorNormalize",
                                       VX_KERNEL_RPP_RESIZEMIRRORNORMALIZE,
                                       processResizeMirrorNormalize,
                                       16,
                                       validateResizeMirrorNormalize,
                                       initializeResizeMirrorNormalize,
                                       uninitializeResizeMirrorNormalize);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel,11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 13, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 14, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 15, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
