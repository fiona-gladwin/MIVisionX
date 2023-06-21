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

#include <stdio.h>
#include <commons.h>
#include "hardware_video_decoder.h"

#ifdef ROCAL_VIDEO
HardWareVideoDecoder::HardWareVideoDecoder(){};

#if ENABLE_HIP

hipStream_t globalHipStream = nullptr;

bool allocateDevMemForYUVScaling(void **pUdevMem, void **pVdevMem, void **pScaledYUVdevMem, void **pScaledUdevMem, void **pScaledVdevMem, size_t &scaledYUVimageSize,
    uint32_t scaledLumaSize, uint32_t scaledYUVstride, uint32_t alignedScalingHeight, VADRMPRIMESurfaceDescriptor &vaDrmPrimeSurfaceDesc) {

    hipError_t hipStatus = hipSuccess;
    uint32_t chromaHeight = 0;
    uint32_t chromaStride = 0;
    size_t chromaImageSize = 0;
    size_t scaledChromaImageSize = 0;

    switch (vaDrmPrimeSurfaceDesc.fourcc) {
        case VA_FOURCC_NV12:
            scaledYUVimageSize = scaledYUVstride * (alignedScalingHeight + (alignedScalingHeight >> 1));
            chromaHeight = vaDrmPrimeSurfaceDesc.height / 2;
            chromaStride = vaDrmPrimeSurfaceDesc.layers[1].pitch[0] / 2;
            chromaImageSize = chromaStride * chromaHeight;
            scaledChromaImageSize = scaledYUVimageSize / 4;

            if (*pScaledYUVdevMem == nullptr) {
                hipStatus = hipMalloc(pScaledYUVdevMem, scaledYUVimageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for scaled YUV image!" << hipStatus << std::endl;
                    return false;
                }
            }
            if (*pUdevMem == nullptr) {
                hipStatus = hipMalloc(pUdevMem, chromaImageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for the U image!" << hipStatus << std::endl;
                    return false;
                }
            }
            if (*pVdevMem == nullptr) {
                hipStatus = hipMalloc(pVdevMem, chromaImageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for the V image!" << hipStatus << std::endl;
                    return false;
                }
            }
            if (*pScaledUdevMem == nullptr) {
                hipStatus = hipMalloc(pScaledUdevMem, scaledChromaImageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for scaled U image!" << hipStatus << std::endl;
                    return false;
                }
            }
            if (*pScaledVdevMem == nullptr) {
                hipStatus = hipMalloc(pScaledVdevMem, scaledChromaImageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for scaled V image!" << hipStatus << std::endl;
                    return false;
                }
            }
            break;
        case VA_FOURCC_Y800:
            if (*pScaledYUVdevMem == nullptr) {
                hipStatus = hipMalloc(pScaledYUVdevMem, scaledLumaSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for scaled YUV image!" << hipStatus << std::endl;
                    return false;
                }
            }
            break;
        case VA_FOURCC_444P:
            scaledYUVimageSize = scaledYUVstride * alignedScalingHeight * 3;
            if (*pScaledYUVdevMem == nullptr) {
                hipStatus = hipMalloc(pScaledYUVdevMem, scaledYUVimageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed to allocate the device memory for scaled YUV image!" << hipStatus << std::endl;
                    return false;
                }
            }
            break;
        default:
            std::cout << "Error! " << vaDrmPrimeSurfaceDesc.fourcc << " format is not supported!" << std::endl;
            return false;
        }

    return true;
}

bool scaleYUVimage(void *pYUVdevMem, void *pUdevMem, void *pVdevMem, void *pScaledYUVdevMem, void *pScaledUdevMem, void *pScaledVdevMem,
    size_t scaledYUVimageSize, uint32_t alignedScalingWidth, uint32_t alignedScalingHeight, uint32_t scaledYUVstride, uint32_t scaledLumaSize,
    VADRMPRIMESurfaceDescriptor &vaDrmPrimeSurfaceDesc, hipStream_t &hipStream) {

    hipError_t hipStatus = hipSuccess;
    size_t lumaSize = vaDrmPrimeSurfaceDesc.layers[0].pitch[0] * vaDrmPrimeSurfaceDesc.height;
    uint32_t chromaWidth = 0;
    uint32_t chromaHeight = 0;
    uint32_t chromaStride = 0;
    size_t chromaImageSize = 0;
    size_t scaledChromaImageSize = 0;
    uint32_t uOffset = 0;

    switch (vaDrmPrimeSurfaceDesc.fourcc) {
        case VA_FOURCC_NV12:
            chromaWidth = vaDrmPrimeSurfaceDesc.width / 2;
            chromaHeight = vaDrmPrimeSurfaceDesc.height / 2;
            chromaStride = vaDrmPrimeSurfaceDesc.layers[1].pitch[0] / 2;
            chromaImageSize = chromaStride * chromaHeight;
            scaledChromaImageSize = scaledYUVimageSize / 4;

            //extract the U and V components
            HipExec_ChannelExtract_U8U8_U16_hw(hipStream, chromaWidth, chromaHeight,
                (unsigned char *)pUdevMem, (unsigned char *)pVdevMem, chromaStride,
                (const unsigned char *)pYUVdevMem + lumaSize, vaDrmPrimeSurfaceDesc.layers[1].pitch[0]);

            //scale the Y, U, and V components of the NV12 image
            HipExec_ScaleImage_NV12_Nearest_hw(hipStream, alignedScalingWidth, alignedScalingHeight, (unsigned char *)pScaledYUVdevMem, scaledYUVstride,
                vaDrmPrimeSurfaceDesc.width, vaDrmPrimeSurfaceDesc.height, (const unsigned char *)pYUVdevMem, vaDrmPrimeSurfaceDesc.layers[0].pitch[0],
                (unsigned char *)pScaledUdevMem, (unsigned char *)pScaledVdevMem, (const unsigned char *)pUdevMem, (const unsigned char *)pVdevMem);

            // combine the scaled U and V components to the final scaled NV12 buffer
            HipExec_ChannelCombine_U16_U8U8_hw(hipStream, alignedScalingWidth / 2, alignedScalingHeight / 2,
                (unsigned char *)pScaledYUVdevMem + scaledLumaSize, scaledYUVstride,
                (unsigned char *)pScaledUdevMem, scaledYUVstride / 2,
                (unsigned char *)pScaledVdevMem, scaledYUVstride / 2);

            break;
        case VA_FOURCC_Y800:
            // if the surface format is YUV400, then there is only one Y component to scale
            HipExec_ScaleImage_U8_U8_Nearest_hw(hipStream, alignedScalingWidth, alignedScalingHeight, (unsigned char *)pScaledYUVdevMem, scaledYUVstride,
                vaDrmPrimeSurfaceDesc.width, vaDrmPrimeSurfaceDesc.height, (const unsigned char *)pYUVdevMem, vaDrmPrimeSurfaceDesc.layers[0].pitch[0]);

            break;
        case VA_FOURCC_444P:
            uOffset = alignedScalingWidth * alignedScalingHeight;
            HipExec_ScaleImage_YUV444_Nearest_hw(hipStream, alignedScalingWidth, alignedScalingHeight,
                (unsigned char *)pScaledYUVdevMem, scaledYUVstride, uOffset,
                vaDrmPrimeSurfaceDesc.width, vaDrmPrimeSurfaceDesc.height, (const unsigned char *)pYUVdevMem,
                vaDrmPrimeSurfaceDesc.layers[0].pitch[0], vaDrmPrimeSurfaceDesc.layers[1].offset[0]);

            break;
        default:
            std::cout << "Error! " << vaDrmPrimeSurfaceDesc.fourcc << " format is not supported!" << std::endl;
            return false;
        }

        hipStatus = hipStreamSynchronize(hipStream);
        if (hipStatus != hipSuccess) {
            std::cout << "ERROR: hipStreamSynchronize failed! (" << hipStatus << ")" << std::endl;
            return false;
        }

    return true;
}

bool allocateDevMemForRGBConversion(uint8_t **pRGBdevMem, uint32_t alignedScalingHeight,
    uint32_t scaledYUVstride, bool isScaling, VADRMPRIMESurfaceDescriptor &vaDrmPrimeSurfaceDesc) {

    hipError_t hipStatus = hipSuccess;
    size_t rgbImageStride = ALIGN16(isScaling ? scaledYUVstride * 3 : vaDrmPrimeSurfaceDesc.layers[0].pitch[0] * 3);

    switch (vaDrmPrimeSurfaceDesc.fourcc) {
        case VA_FOURCC_NV12:
        case VA_FOURCC_444P:
            //allocate HIP device memory for RGB frame
            if (*pRGBdevMem == nullptr) {
                size_t rgbImageSize = (isScaling ? alignedScalingHeight : ALIGN16(vaDrmPrimeSurfaceDesc.height)) * rgbImageStride;
                hipStatus = hipMalloc(pRGBdevMem, rgbImageSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMalloc failed!" << hipStatus << std::endl;
                    return false;
                }
            }
            break;
        default:
            std::cout << "Error! " << vaDrmPrimeSurfaceDesc.fourcc << " format is not supported!" << std::endl;
            return false;
    }

    return true;
}

bool colorConvertYUVtoRGB(void *pYUVdevMem, void *pScaledYUVdevMem, uint8_t *pRGBdevMem, uint32_t scaledLumaSize, uint32_t alignedScalingWidth,
    uint32_t alignedScalingHeight, uint32_t scaledYUVstride, bool isScaling, VADRMPRIMESurfaceDescriptor &vaDrmPrimeSurfaceDesc, hipStream_t &hipStream) {

    hipError_t hipStatus = hipSuccess;
    size_t lumaSize = isScaling ? scaledLumaSize : (vaDrmPrimeSurfaceDesc.layers[0].pitch[0] * vaDrmPrimeSurfaceDesc.height);
    size_t rgbImageStride = ALIGN16(isScaling ? scaledYUVstride * 3 : vaDrmPrimeSurfaceDesc.layers[0].pitch[0] * 3);

    switch (vaDrmPrimeSurfaceDesc.fourcc) {
        case VA_FOURCC_NV12:
            if (isScaling) {
                HipExec_ColorConvert_NV12_to_RGB_hw(hipStream, alignedScalingWidth, alignedScalingHeight, (unsigned char *)pRGBdevMem, rgbImageStride,
                (const unsigned char *)pScaledYUVdevMem, scaledYUVstride, (const unsigned char *)pScaledYUVdevMem + scaledLumaSize, scaledYUVstride);
            } else {
                HipExec_ColorConvert_NV12_to_RGB_hw(hipStream, vaDrmPrimeSurfaceDesc.width, vaDrmPrimeSurfaceDesc.height, (unsigned char *)pRGBdevMem, rgbImageStride,
                (const unsigned char *)pYUVdevMem, vaDrmPrimeSurfaceDesc.layers[0].pitch[0], (const unsigned char *)pYUVdevMem + lumaSize, vaDrmPrimeSurfaceDesc.layers[1].pitch[0]);
            }
            break;
        case VA_FOURCC_444P:
            if (isScaling) {
                HipExec_ColorConvert_YUV444_to_RGB_hw(hipStream, alignedScalingWidth, alignedScalingHeight, (unsigned char *)pRGBdevMem, rgbImageStride,
                    (const unsigned char *)pScaledYUVdevMem, scaledYUVstride, scaledLumaSize);
            } else {
                HipExec_ColorConvert_YUV444_to_RGB_hw(hipStream, vaDrmPrimeSurfaceDesc.width, vaDrmPrimeSurfaceDesc.height, (unsigned char *)pRGBdevMem, rgbImageStride,
                    (const unsigned char *)pYUVdevMem, vaDrmPrimeSurfaceDesc.layers[0].pitch[0], vaDrmPrimeSurfaceDesc.layers[1].offset[0]);
            }
            break;
        default:
            std::cout << "Error! " << vaDrmPrimeSurfaceDesc.fourcc << " format is not supported!" << std::endl;
            return false;
    }

    hipStatus = hipStreamSynchronize(hipStream);
    if (hipStatus != hipSuccess) {
        std::cout << "ERROR: hipStreamSynchronize failed! (" << hipStatus << ")" << std::endl;
        return false;
    }

    return true;

}
#endif
int HardWareVideoDecoder::seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number)
{
    auto seek_time = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), time_base);
    int ret = av_seek_frame(_fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        ERR("HardWareVideoDecoder::seek_frame Error in seeking frame. Unable to seek the given frame in a video");
        return ret;
    }
    return select_frame_pts;
}

//  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
//                                          const enum AVPixelFormat *pix_fmts)
//  {
//      const enum AVPixelFormat *p;
  
//      for (p = pix_fmts; *p != -1; p++) {
//          if (*p == hw_pix_fmt)
//              return *p;
//      }
  
//      fprintf(stderr, "Failed to get HW surface format.\n");
//      return AV_PIX_FMT_NONE;
//  }
 
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    (void)ctx, (void)pix_fmts;
    return AV_PIX_FMT_VAAPI;
}

int HardWareVideoDecoder::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx)
{
    int err = 0;
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type, NULL, NULL, 0)) < 0)    // DRM nodes thing I am not using here
        std::cerr << "[ERR] Context creation for hardware error";
    return err;
}

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status HardWareVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format)
{
    VideoDecoder::Status status = Status::OK;

    // Initialize the SwsContext
    SwsContext *swsctx = nullptr;
    bool isScaling = (out_width != _codec_width) || (out_height != _codec_height);
#if !ENABLE_HIP
    if ((out_width != _codec_width) || (out_height != _codec_height) || (out_pix_format != _dec_pix_fmt))
    {
        swsctx = sws_getCachedContext(nullptr, _codec_width, _codec_height, _dec_pix_fmt,
                                      out_width, out_height, out_pix_format, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsctx)
        {
            ERR("HardWareVideoDecoder::Decode Failed to get sws_getCachedContext");
            return Status::FAILED;
        }
    }
#endif
    // int select_frame_pts = seek_frame(_video_stream->avg_frame_rate, _video_stream->time_base, seek_frame_number);
    // if (select_frame_pts < 0)
    // {
    //     ERR("HardWareVideoDecoder::Decode Error in seeking frame. Unable to seek the given frame in a video");
    //     return Status::FAILED;
    // }
    unsigned frame_count = 0;
    bool end_of_stream = false;
    bool sequence_filled = false;
    uint8_t *dst_data[4] = {0};
    int dst_linesize[4] = {0};
    int image_size = out_height * out_stride * sizeof(unsigned char);
    AVPacket pkt;
    VASurfaceID va_surface = 0;
    VAStatus vastatus;

#if ENABLE_HIP    
    hipExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    hipExternalMemoryBufferDesc externalMemBufferDesc = {};
    hipError_t hipStatus;
    hipExternalMemory_t hipExtMem;
#endif

    // Variables required for scaling images
    void *pYUVdevMem = nullptr;
    void *pUdevMem = nullptr;
    void *pVdevMem = nullptr;
    void *pScaledYUVdevMem = nullptr;
    void *pScaledUdevMem = nullptr;
    void *pScaledVdevMem = nullptr;
    uint8_t *pRGBdevMem = nullptr;

    uint32_t scaledYUVstride = 0;
    uint32_t alignedScalingWidth = 0;
    uint32_t alignedScalingHeight = 0;
    size_t scaledYUVimageSize = 0;
    uint32_t scaledLumaSize = 0;
    if (isScaling) {
        alignedScalingWidth = ALIGN16(out_width);
        alignedScalingHeight = ALIGN16(out_height);
        scaledYUVstride = alignedScalingWidth;
        scaledLumaSize = scaledYUVstride * alignedScalingHeight;
    }
    std::string yuvformat= "";  // Not required
    do
    {
        int ret;
        // read packet from input file
        ret = av_read_frame(_fmt_ctx, &pkt);
        if (ret < 0 && ret != AVERROR_EOF)
        {
            ERR("HardWareVideoDecoder::Decode Failed to read the frame: ret=" + TOSTR(ret));
            status = Status::FAILED;
            break;
        }
        if (ret == 0 && pkt.stream_index != _video_stream_idx) continue;
        end_of_stream = (ret == AVERROR_EOF);
        if (end_of_stream)
        {
            // null packet for bumping process
            pkt.data = nullptr;
            pkt.size = 0;
        }

        // submit the packet to the decoder
        ret = avcodec_send_packet(_video_dec_ctx, &pkt);
        if (ret < 0)
        {
            ERR("HardWareVideoDecoder::Decode Error while sending packet to the decoder\n");
            status = Status::FAILED;
            break;
        }

        // get all the available frames from the decoder
        while (ret >= 0)
        {
            ret = avcodec_receive_frame(_video_dec_ctx, _dec_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if ((ret < 0)) continue;
            if (frame_count % stride == 0)
            {    
                va_surface = (uintptr_t)_dec_frame->data[3];
                vastatus = vaSyncSurface(_va_display, va_surface);
                if (vastatus != VA_STATUS_SUCCESS) {
                    std::cerr << "ERROR: vaSyncSurface failed! " << AVERROR(vastatus) << std::endl;
                    return Status::FAILED;
                }

                vastatus = vaExportSurfaceHandle(_va_display, va_surface,
                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY |
                    VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &_vaDrmPrimeSurfaceDesc);

                if (vastatus != VA_STATUS_SUCCESS) {
                    std::cerr << "ERROR: vaExportSurfaceHandle failed! " << AVERROR(vastatus) << std::endl;
                    return Status::FAILED;
                }
                switch (_vaDrmPrimeSurfaceDesc.fourcc) {
                    case VA_FOURCC_NV12:
                        yuvformat = "YUV420";
                        break;
                    case VA_FOURCC_Y800:
                        yuvformat = "YUV400";
                        break;
                    case VA_FOURCC_444P:
                        yuvformat = "YUV444";
                        break;
                    default:
                        std::cout << "Error! " << _vaDrmPrimeSurfaceDesc.fourcc << " format is not supported!" << AVERROR(vastatus) << std::endl;
                        return Status::FAILED;
                }
                
#if ENABLE_HIP
                // import the frame (DRM-PRIME FDs) into the HIP
                externalMemoryHandleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
                externalMemoryHandleDesc.handle.fd = _vaDrmPrimeSurfaceDesc.objects[0].fd;
                externalMemoryHandleDesc.size = _vaDrmPrimeSurfaceDesc.objects[0].size;

                hipStatus = hipImportExternalMemory(&hipExtMem, &externalMemoryHandleDesc);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipImportExternalMemory failed! (" << hipStatus << ")" << std::endl;
                    return Status::FAILED;
                }

                externalMemBufferDesc.offset = 0;
                externalMemBufferDesc.size = _vaDrmPrimeSurfaceDesc.objects[0].size;
                externalMemBufferDesc.flags = 0;
                hipStatus = hipExternalMemoryGetMappedBuffer(&pYUVdevMem, hipExtMem, &externalMemBufferDesc);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipExternalMemoryGetMappedBuffer failed! (" << hipStatus << ")" << std::endl;
                    return Status::FAILED;
                }

                if (isScaling) {
                    if (!allocateDevMemForYUVScaling(&pUdevMem, &pVdevMem, &pScaledYUVdevMem, &pScaledUdevMem, &pScaledVdevMem, scaledYUVimageSize, scaledLumaSize,
                            scaledYUVstride, alignedScalingHeight, _vaDrmPrimeSurfaceDesc)) {
                                std::cout << "ERROR: allocating device memories for YUV scaling failed!" << std::endl;
                                return Status::FAILED;
                    }
                    if (!scaleYUVimage(pYUVdevMem, pUdevMem, pVdevMem, pScaledYUVdevMem, pScaledUdevMem, pScaledVdevMem, scaledYUVimageSize, alignedScalingWidth,
                            alignedScalingHeight, scaledYUVstride, scaledLumaSize, _vaDrmPrimeSurfaceDesc, globalHipStream)) {
                                std::cout << "ERROR: allocating device memories for YUV scaling failed!" << std::endl;
                                return Status::FAILED;
                    }
                }

                // convert YUV to RGB format if the requested output frame format is RGB
                
                if (out_pix_format == AV_PIX_FMT_RGB24 && _vaDrmPrimeSurfaceDesc.fourcc != VA_FOURCC_Y800) {
                    if (!allocateDevMemForRGBConversion(&pRGBdevMem, alignedScalingHeight, scaledYUVstride, isScaling, _vaDrmPrimeSurfaceDesc)) {
                        std::cout << "ERROR: allocating device memories for RGB conversion failed!" << std::endl;
                        return Status::FAILED;
                    }
                    if (!colorConvertYUVtoRGB(pYUVdevMem, pScaledYUVdevMem, pRGBdevMem, scaledLumaSize, alignedScalingWidth, alignedScalingHeight, scaledYUVstride,
                        isScaling, _vaDrmPrimeSurfaceDesc, globalHipStream)) {
                            std::cout << "ERROR: YUV to RGB color conversion failed!" << std::endl;
                            return Status::FAILED;
                    }
                }
                size_t rgbFrameSize = isScaling ? alignedScalingHeight * ALIGN16(scaledYUVstride * 3) : _vaDrmPrimeSurfaceDesc.height * ALIGN16(_vaDrmPrimeSurfaceDesc.layers[0].pitch[0] * 3);
                // std::cerr << "rgbFrameSize : " << rgbFrameSize << " H : " << _vaDrmPrimeSurfaceDesc.height << " w : " << _vaDrmPrimeSurfaceDesc.width << "\n";
                hipStatus = hipMemcpyDtoH((void *)out_buffer, pRGBdevMem, rgbFrameSize);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipMemcpyDtoH failed! (" << hipStatus << ")" << std::endl;
                    return Status::FAILED;
                }
                
                hipStatus = hipDestroyExternalMemory(hipExtMem);
                if (hipStatus != hipSuccess) {
                    std::cout << "ERROR: hipDestroyExternalMemory failed! (" << hipStatus << ")" << std::endl;
                    return Status::FAILED;
                }

                for (int i = 0;  i < (int)_vaDrmPrimeSurfaceDesc.num_objects;  ++i) {
                    close(_vaDrmPrimeSurfaceDesc.objects[i].fd);
                }
#else
                //retrieve data from GPU to CPU
                if ((av_hwframe_transfer_data(_sw_frame, _dec_frame, 0)) < 0) {
                    ERR("HardWareVideoDecoder::Decode avcodec_receive_frame() failed");
                    return Status::FAILED;
                }

                dst_data[0] = out_buffer;
                dst_linesize[0] = out_stride;
                if (swsctx)
                    sws_scale(swsctx, _sw_frame->data, _sw_frame->linesize, 0, _sw_frame->height, dst_data, dst_linesize);
                else
                {
                    // copy from frame to out_buffer
                    memcpy(out_buffer, _sw_frame->data[0], _sw_frame->linesize[0] * out_height);
                }
#endif
                out_buffer = out_buffer + image_size;
            }
            ++frame_count;
            // av_frame_unref(_sw_frame);
            // av_frame_unref(_dec_frame);
            if (frame_count == sequence_length * stride)
            {
                sequence_filled = true;
                break;
            }
        }
        av_packet_unref(&pkt);
        if (sequence_filled)  break;
    } while (!end_of_stream);
    // avcodec_flush_buffers(_video_dec_ctx);
#if !ENABLE_HIP
    sws_freeContext(swsctx);
#endif
    return status;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status HardWareVideoDecoder::Initialize(const char *src_filename)
{
    VideoDecoder::Status status = Status::OK;
    int ret;
    AVDictionary *opts = NULL;

    // open input file, and initialize the context required for decoding
    _fmt_ctx = avformat_alloc_context();
    _src_filename = src_filename;

    // find if hardware decode is available
    AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    hw_type = av_hwdevice_find_type_by_name("vaapi");
    if (hw_type == AV_HWDEVICE_TYPE_NONE) {
        ERR("HardWareVideoDecoder::Initialize ERROR: vaapi is not supported for this device\n");
        return Status::FAILED;
    }
    else
        INFO("HardWareVideoDecoder::Initialize : Found vaapi device for the device\n");

    if (avformat_open_input(&_fmt_ctx, src_filename, NULL, NULL) < 0)
    {
        ERR("HardWareVideoDecoder::Initialize Couldn't Open video file " + STR(src_filename));
        return Status::FAILED;
    }
    if (avformat_find_stream_info(_fmt_ctx, NULL) < 0)
    {
        ERR("HardWareVideoDecoder::Initialize av_find_stream_info error");
        return Status::FAILED;
    }
    ret = av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &_decoder, 0);
    if (ret < 0)
    {
        ERR("HardWareVideoDecoder::Initialize Could not find %s stream in input file " +
                STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " " + STR(src_filename));
        return Status::FAILED;
    }
    // for hardware accelerated decoding, find config  -- Not required
    // for (int i = 0; ; i++) {
    //     const AVCodecHWConfig *config = avcodec_get_hw_config(_decoder, i);
    //     if (!config) {
    //         ERR("HardWareVideoDecoder::Initialize ERROR: decoder " + STR(_decoder->name) + " doesn't support device_type " + STR(av_hwdevice_get_type_name(hw_type)));
    //         return Status::FAILED;
    //     }
    //     if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
    //             config->device_type == hw_type) {
    //         break;
    //     }
    // }

    _video_stream_idx = ret;
    _video_stream = _fmt_ctx->streams[_video_stream_idx];

    if (!_video_stream)
    {
        ERR("HardWareVideoDecoder::Initialize Could not find video stream in the input, aborting");
        return Status::FAILED;
    }

    // find decoder for the stream -- // Not required
    _decoder = avcodec_find_decoder(_video_stream->codecpar->codec_id);
    if (!_decoder)
    {
        ERR("HardWareVideoDecoder::Initialize Failed to find " +
                STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }

    // Allocate a codec context for the decoder
    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx)
    {
        ERR("HardWareVideoDecoder::Initialize Failed to allocate the " +
                STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec context");
        return Status::NO_MEMORY;
    }

    // Copy codec parameters from input stream to output codec context
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video_stream->codecpar)) < 0)
    {
        ERR("HardWareVideoDecoder::Initialize Failed to copy " +
                STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec parameters to decoder context");
        return Status::FAILED;
    }

    // if (hw_decoder_init(_video_dec_ctx, hw_type, hw_device_ctx) < 0) {
    //     ERR("HardWareVideoDecoder::Initialize ERROR: Failed to create specified HW device");
    //     return Status::FAILED;
    // }
    
    int err = 0;
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, hw_type, NULL, NULL, 0)) < 0) {   // DRM nodes thing I am not using here
        std::cerr << "[ERR] Context creation for hardware error";
        return Status::FAILED;
    }
    
    // std::cerr << "Hardware decoder initialized\n";
    // _dec_pix_fmt = AV_PIX_FMT_NV12; // nv12 for vaapi // TODO - Need to check how?
    _hwctx = reinterpret_cast<AVHWDeviceContext *>(hw_device_ctx->data);
    _vactx = reinterpret_cast<AVVAAPIDeviceContext *>(_hwctx->hwctx);
    _va_display = _vactx->display;
    _video_dec_ctx->get_format = get_hw_format;
    _video_dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    
    // Init the decoders
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, &opts)) < 0)
    {
        ERR("HardWareVideoDecoder::Initialize Failed to open " +
                STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }
    _codec_width = _video_stream->codecpar->width;
    _codec_height = _video_stream->codecpar->height;
    
    _dec_frame = av_frame_alloc();
    if (!_dec_frame)
    {
        ERR("HardWareVideoDecoder::Decode Could not allocate _dec_frame");
        return Status::NO_MEMORY;
    }
#if !ENABLE_HIP
    _sw_frame = av_frame_alloc();
    if (!_sw_frame)
    {
        ERR("HardWareVideoDecoder::Decode Could not allocate _sw_frame");
        return Status::NO_MEMORY;
    }
#endif

#if ENABLE_HIP
    if(globalHipStream == nullptr) {
        // std::cerr << " Creating the HIP stream"; 
        hipError_t hipStatus = hipStreamCreate(&globalHipStream);
        if (hipStatus != hipSuccess) {
            std::cout << "ERROR: hipStreamCreate failed! (" << hipStatus << ")" << std::endl;
        }
    }
#endif
    return status;
}

void HardWareVideoDecoder::release()
{
    if (_video_dec_ctx)
        avcodec_free_context(&_video_dec_ctx);
    if (_fmt_ctx)
        avformat_close_input(&_fmt_ctx);
    if(_dec_frame) av_frame_free(&_dec_frame);
    // if(_sw_frame) av_frame_free(&_sw_frame);
}

HardWareVideoDecoder::~HardWareVideoDecoder()
{
    release();
}
#endif
