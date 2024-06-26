# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(amd_openvx_extensions)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
if(WIN32)
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()
find_package(OpenCV QUIET)
find_package(AMDRPP QUIET)
if(GPU_SUPPORT)
    if("${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            SET(BUILD_OPENCL True)
            find_package(miopengemm PATHS ${ROCM_PATH} QUIET)
        else()
            set(GPU_SUPPORT OFF)
            set(BUILD_OPENCL False)
        endif()
    elseif("${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
        set(DEFAULT_AMDGPU_TARGETS "gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102")
        set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
        find_package(HIP QUIET)
        if(HIP_FOUND)
            list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm)
        else()
            set(GPU_SUPPORT OFF)
        endif()
    endif()
    find_package(miopen     PATHS ${ROCM_PATH} QUIET)
    if (miopen_FOUND)
        #find the MIOPEN backend type
        set(MIOPEN_BACKEND_OPENCL_FOUND 0)
        set(MIOPEN_BACKEND_HIP_FOUND 0)

        if(EXISTS ${ROCM_PATH}/include/miopen/config.h)
            file(READ ${ROCM_PATH}/include/miopen/config.h MIOPEN_CONFIG_FILE)
            string(REGEX MATCH "MIOPEN_BACKEND_OPENCL ([0-9]*)" _ ${MIOPEN_CONFIG_FILE})
            set(MIOPEN_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
            string(REGEX MATCH "MIOPEN_BACKEND_HIP ([0-9]*)" _ ${MIOPEN_CONFIG_FILE})
            set(MIOPEN_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
        else()
            set(NEURAL_NET OFF)
            message("-- ${Red}WARNING: ${ROCM_PATH}/include/miopen/config.h file Not Found. please run the setup script to install the MIOpen package ${ColourReset}")
        endif()

    endif()
endif()

if(GPU_SUPPORT)
    find_package(FFmpeg QUIET)
    if(FFMPEG_FOUND)
        add_subdirectory(amd_media)
        message("-- ${Green}AMD OpenVX Media Extension -- amd_media module added${ColourReset}")
    else()
        message("-- ${Red}WARNING: FFMPEG Not Found -- amd_media module excluded${ColourReset}")
    endif(FFMPEG_FOUND)

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if(LOOM)
            add_subdirectory(amd_loomsl)
            message("-- ${Green}AMD OpenVX Loom Stich Library Extension -- amd_loomsl module added${ColourReset}")
        else()
            message("-- ${Cyan}LOOM Module turned OFF by user option -D LOOM=OFF ${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        #amd custom extenstion
        add_subdirectory(amd_custom)
        message("-- ${Green}AMD Custom Extension -- amd_custom module added${ColourReset}")
    else()
      message("-- ${Red}WARNING: GPU Support OpenCL/HIP Not Found -- amd_openvx_extensions modules for GPU excluded${ColourReset}")
    endif()
endif(GPU_SUPPORT)

if (NEURAL_NET)
    if(GPU_SUPPORT AND miopen_FOUND)
        if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND AND miopengemm_FOUND)
            if (MIOPEN_BACKEND_OPENCL_FOUND)
                add_subdirectory(amd_nn)
                message("-- ${Green}AMD OpenVX Neural Network Extension -- amd_nn module added with OpenCL backend${ColourReset}")
            else()
                set(NEURAL_NET OFF)
                message("-- ${Red}WARNING: MIOPEN with OpenCL backend Not Found -- amd_nn module excluded${ColourReset}")
            endif()
        elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            if (MIOPEN_BACKEND_HIP_FOUND)
                add_subdirectory(amd_nn)
                message("-- ${Green}AMD OpenVX Neural Network Extension -- amd_nn module added with HIP backend${ColourReset}")
            else()
                set(NEURAL_NET OFF)
                message("-- ${Red}WARNING: MIOPEN with HIP backend Not Found -- amd_nn module excluded${ColourReset}")
            endif()
        else()
            set(NEURAL_NET OFF)
            message("-- ${Red}WARNING: GPU support with OpenCL/MIOpenGEMM(for OpenCL)/HIP Not Found -- amd_nn module excluded${ColourReset}")
        endif()
    else()
        set(NEURAL_NET OFF)
        message("-- ${Red}WARNING: GPU_SUPPORT/MIOpen Not Found -- amd_nn module excluded${ColourReset}")
    endif()
else()
    message("-- ${Cyan}Neural Net Modules turned OFF by user option -D NEURAL_NET=OFF ${ColourReset}")
endif()

if(AMDRPP_FOUND AND GPU_SUPPORT)
    #find the RPP backend type
    set(RPP_BACKEND_OPENCL_FOUND 0)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h)
        file(READ ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message("-- ${Red}WARNING: ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h file Not Found. please run the setup script to install latest RPP package ${ColourReset}")
    endif()

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (RPP_BACKEND_OPENCL_FOUND)
            add_subdirectory(amd_rpp)
            message("-- ${Green}AMD ROCm Performance Primitives -- amd_rpp module added with OpenCL backend support${ColourReset}")
        else()
            message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend first!) -- amd_rpp module excluded.${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (RPP_BACKEND_HIP_FOUND)
            add_subdirectory(amd_rpp)
            message("-- ${Green}AMD ROCm Performance Primitives -- amd_rpp module added with the HIP backend support${ColourReset}")
        else()
            message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend first!) -- amd_rpp module excluded.${ColourReset}")
        endif()
    else()
        message("-- ${Red}WARNING: GPU support with OpenCL or HIP Not Found -- amd_rpp module excluded${ColourReset}")
    endif()
elseif(AMDRPP_FOUND AND ("${BACKEND}" STREQUAL "CPU" OR NOT(GPU_SUPPORT)))
    add_subdirectory(amd_rpp)
    message("-- ${Green}AMD ROCm Performance Primitives -- amd_rpp module added with CPU only backend support${ColourReset}")
else()
    message("-- ${Red}WARNING: AMD RPP Not Found -- OpenVX Extension amd_rpp module excluded${ColourReset}")
endif()

if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
        message("-- ${White}OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X${ColourReset}")
        add_subdirectory(amd_opencv)
        message("-- ${Green}AMD OpenVX OpenCV Extension -- amd_opencv module added${ColourReset}")
    else()
        message("-- ${Red}WARNING: OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING: OpenCV Not Found -- amd_opencv module excluded${ColourReset}")
endif()

if (MIGRAPHX)
    if(GPU_SUPPORT)
        if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            if (miopen_FOUND AND MIOPEN_BACKEND_HIP_FOUND)
                add_subdirectory(amd_migraphx)
                message("-- ${Green}AMD OpenVX MIGraphX Extension -- amd_migraphx module added with the HIP backend${ColourReset}")
            else()
                message("-- ${Red}WARNING: MIOPEN with the HIP backend Not Found -- amd_migraphx module excluded${ColourReset}")
            endif()
        elseif("${BACKEND}" STREQUAL "OPENCL")
            message("-- ${Red}WARNING: MIGraphX extension is not supported with the OCL backend -- amd_migraphx module excluded${ColourReset}")
        else()
            message("-- ${Red}WARNING: MIGraphX extension is only supported with the HIP backend, and the HIP backend Not Found -- amd_migraphx module excluded${ColourReset}")
        endif()
    else()
        message("-- ${Red}WARNING: GPU support with the HIP backend not found. -- amd_migraphx module excluded${ColourReset}")
    endif()
else()
    message("-- ${Cyan}MIGraphX Module turned OFF by user option -D MIGRAPHX=OFF ${ColourReset}")
endif()
