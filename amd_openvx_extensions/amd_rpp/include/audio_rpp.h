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

#ifndef _AUDIO_RPP_H_
#define _AUDIO_RPP_H_

#include "internal_publishKernels.h"

enum vxRppAudioAugmentationName {
    RESAMPLE = 0,
    PRE_EMPHASIS_FILTER = 1
};

struct ResampleData {
    float quality;
    Rpp32f *pInRateTensor;
    Rpp32f *pOutRateTensor;
    RpptResamplingWindow window;
};

struct PreEmphasisFilterData {
    vx_int32 borderType;
    Rpp32f *pPreemphCoeff;
    Rpp32u *pSampleSize;
};

struct ToDecibelsData {
    Rpp32f cutOffDB;
    Rpp32f multiplier;
    Rpp32f referenceMagnitude;
    RpptImagePatch *pSrcDims;
};

union AudioAugmentationData {
    ResampleData resample;
    PreEmphasisFilterData preEmphasis;
    ToDecibelsData toDecibels;
};

// ********************* Utility functions for Resample *********************
inline float sinc(float x) {
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * (1.0f / 6)) : std::sin(x) / x;
}

inline double hann(double x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

// initialization function used for filling the values in Resampling window (RpptResamplingWindow)
// using the coeffs and lobes value this function generates a LUT (look up table) which is further used in Resample audio augmentation
inline void windowed_sinc(RpptResamplingWindow &window, int coeffs, int lobes) {
    float scale = 2.0f * lobes / (coeffs - 1);
    float scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookupSize = window.lookup.size();
    int center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
        float x = (i - center) * scale;
        float y = (i - center) * scale_envelope;
        float w = sinc(x) * hann(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1 / scale;
    window.pCenter = _mm_set1_ps(window.center);
    window.pScale = _mm_set1_ps(window.scale);
}

#endif