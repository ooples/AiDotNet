// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for deformable convolution operations.
// Implements DCNv1 and DCNv2 (with modulation masks).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL kernels for deformable convolution operations.
    /// </summary>
    /// <remarks>
    /// Deformable convolution adds learnable 2D offsets to each sampling position,
    /// allowing the convolution to adaptively adjust its receptive field based on
    /// input content. DCNv2 adds modulation masks for additional control.
    /// Reference: Dai et al., "Deformable Convolutional Networks", ICCV 2017.
    /// </remarks>
    internal static class DeformableConvolutionKernels
    {
        /// <summary>
        /// Gets all deformable convolution kernel names.
        /// </summary>
        public static string[] GetKernelNames() =>
        [
            "deformable_conv2d",
            "deformable_conv2d_backward_input",
            "deformable_conv2d_backward_weights",
            "deformable_conv2d_backward_offset",
            "deformable_conv2d_backward_mask"
        ];

        /// <summary>
        /// Gets all deformable convolution kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// DEFORMABLE CONVOLUTION KERNELS
// ===========================================================================

// Bilinear interpolation helper for sampling at fractional positions
inline float bilinear_sample(
    __global const float* input,
    const int b,
    const int c,
    const float h,
    const float w,
    const int inHeight,
    const int inWidth,
    const int inChannels)
{
    // Get floor coordinates
    int h_low = (int)floor(h);
    int w_low = (int)floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    // Compute interpolation weights
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    // Sample four corners with bounds checking
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;

    if (h_low >= 0 && h_low < inHeight && w_low >= 0 && w_low < inWidth)
        v1 = input[((b * inChannels + c) * inHeight + h_low) * inWidth + w_low];

    if (h_low >= 0 && h_low < inHeight && w_high >= 0 && w_high < inWidth)
        v2 = input[((b * inChannels + c) * inHeight + h_low) * inWidth + w_high];

    if (h_high >= 0 && h_high < inHeight && w_low >= 0 && w_low < inWidth)
        v3 = input[((b * inChannels + c) * inHeight + h_high) * inWidth + w_low];

    if (h_high >= 0 && h_high < inHeight && w_high >= 0 && w_high < inWidth)
        v4 = input[((b * inChannels + c) * inHeight + h_high) * inWidth + w_high];

    // Bilinear interpolation
    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Forward pass for deformable convolution (DCNv2 with optional modulation)
// Input:   [batch, inChannels, inH, inW]
// Weights: [outChannels, inChannels/groups, kH, kW]
// Offsets: [batch, 2*kH*kW*deformGroups, outH, outW] (dx, dy for each kernel position)
// Mask:    [batch, kH*kW*deformGroups, outH, outW] (optional modulation)
// Output:  [batch, outChannels, outH, outW]
__kernel void deformable_conv2d(
    __global const float* input,
    __global const float* weights,
    __global const float* offsets,
    __global const float* mask,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups,
    const int deformGroups,
    const int hasMask)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int boc = get_global_id(2);

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    const int b = boc / outChannels;
    const int oc = boc % outChannels;

    // Determine group indices
    const int g = oc / (outChannels / groups);
    const int dg = oc / (outChannels / deformGroups);

    const int inChannelsPerGroup = inChannels / groups;
    const int kernelSize = kernelH * kernelW;

    float sum = 0.0f;

    // Base input position (center of kernel)
    int base_h = oh * strideH - padH;
    int base_w = ow * strideW - padW;

    for (int ic = 0; ic < inChannelsPerGroup; ic++) {
        const int actualIC = g * inChannelsPerGroup + ic;

        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                const int kernelIdx = kh * kernelW + kw;

                // Get offset for this kernel position
                const int offsetIdx = ((b * deformGroups + dg) * 2 * kernelSize + kernelIdx) * outHeight * outWidth
                                    + oh * outWidth + ow;
                const int offsetIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kernelIdx) * outHeight * outWidth
                                     + oh * outWidth + ow;

                float offset_h = offsets[offsetIdx];
                float offset_w = offsets[offsetIdxY];

                // Compute sampling position with offset and dilation
                float h = base_h + kh * dilationH + offset_h;
                float w = base_w + kw * dilationW + offset_w;

                // Sample using bilinear interpolation
                float val = bilinear_sample(input, b, actualIC, h, w, inHeight, inWidth, inChannels);

                // Apply modulation mask if using DCNv2
                if (hasMask != 0) {
                    const int maskIdx = ((b * deformGroups + dg) * kernelSize + kernelIdx) * outHeight * outWidth
                                      + oh * outWidth + ow;
                    val *= mask[maskIdx];
                }

                // Get weight
                const int weightIdx = ((oc * inChannelsPerGroup + ic) * kernelH + kh) * kernelW + kw;
                float weightVal = weights[weightIdx];

                sum += val * weightVal;
            }
        }
    }

    const int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = sum;
}

// Backward pass for input gradients with bilinear sampling
__kernel void deformable_conv2d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
    __global const float* offsets,
    __global const float* mask,
    __global float* gradInput,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups,
    const int deformGroups,
    const int hasMask)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * inChannels * inHeight * inWidth;

    if (idx >= totalSize) return;

    // Decompose index
    const int b = idx / (inChannels * inHeight * inWidth);
    const int rem1 = idx % (inChannels * inHeight * inWidth);
    const int ic = rem1 / (inHeight * inWidth);
    const int rem2 = rem1 % (inHeight * inWidth);
    const int ih = rem2 / inWidth;
    const int iw = rem2 % inWidth;

    const int g = ic / (inChannels / groups);
    const int inChannelsPerGroup = inChannels / groups;
    const int outChannelsPerGroup = outChannels / groups;
    const int kernelSize = kernelH * kernelW;

    float sumGrad = 0.0f;

    // Iterate over all output positions and kernel positions that might sample this input
    for (int oc = g * outChannelsPerGroup; oc < (g + 1) * outChannelsPerGroup; oc++) {
        const int dg = oc / (outChannels / deformGroups);
        const int icLocal = ic - g * inChannelsPerGroup;

        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                int base_h = oh * strideH - padH;
                int base_w = ow * strideW - padW;

                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        const int kernelIdx = kh * kernelW + kw;

                        // Get offset
                        const int offsetIdx = ((b * deformGroups + dg) * 2 * kernelSize + kernelIdx) * outHeight * outWidth
                                            + oh * outWidth + ow;
                        const int offsetIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kernelIdx) * outHeight * outWidth
                                             + oh * outWidth + ow;

                        float offset_h = offsets[offsetIdx];
                        float offset_w = offsets[offsetIdxY];

                        float h = base_h + kh * dilationH + offset_h;
                        float w = base_w + kw * dilationW + offset_w;

                        // Check if this input pixel is within the bilinear sampling region
                        int h_low = (int)floor(h);
                        int w_low = (int)floor(w);
                        int h_high = h_low + 1;
                        int w_high = w_low + 1;

                        // Check if (ih, iw) is one of the four corners
                        float lh = h - h_low;
                        float lw = w - w_low;
                        float hh = 1.0f - lh;
                        float hw = 1.0f - lw;

                        float weight_contrib = 0.0f;
                        if (ih == h_low && iw == w_low) weight_contrib = hh * hw;
                        else if (ih == h_low && iw == w_high) weight_contrib = hh * lw;
                        else if (ih == h_high && iw == w_low) weight_contrib = lh * hw;
                        else if (ih == h_high && iw == w_high) weight_contrib = lh * lw;
                        else continue;

                        float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                        const int weightIdx = ((oc * inChannelsPerGroup + icLocal) * kernelH + kh) * kernelW + kw;
                        float weightVal = weights[weightIdx];

                        float contrib = gradOutVal * weightVal * weight_contrib;

                        if (hasMask != 0) {
                            const int maskIdx = ((b * deformGroups + dg) * kernelSize + kernelIdx) * outHeight * outWidth
                                              + oh * outWidth + ow;
                            contrib *= mask[maskIdx];
                        }

                        sumGrad += contrib;
                    }
                }
            }
        }
    }

    gradInput[idx] = sumGrad;
}

// Backward pass for weight gradients
__kernel void deformable_conv2d_backward_weights(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* offsets,
    __global const float* mask,
    __global float* gradWeights,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups,
    const int deformGroups,
    const int hasMask)
{
    const int idx = get_global_id(0);
    const int inChannelsPerGroup = inChannels / groups;
    const int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;

    if (idx >= totalWeights) return;

    // Decompose weight index [oc, ic_local, kh, kw]
    int tmp = idx;
    const int kw = tmp % kernelW; tmp /= kernelW;
    const int kh = tmp % kernelH; tmp /= kernelH;
    const int icLocal = tmp % inChannelsPerGroup; tmp /= inChannelsPerGroup;
    const int oc = tmp;

    const int g = oc / (outChannels / groups);
    const int dg = oc / (outChannels / deformGroups);
    const int ic = g * inChannelsPerGroup + icLocal;
    const int kernelSize = kernelH * kernelW;
    const int kernelIdx = kh * kernelW + kw;

    float sumGrad = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                int base_h = oh * strideH - padH;
                int base_w = ow * strideW - padW;

                // Get offset
                const int offsetIdx = ((b * deformGroups + dg) * 2 * kernelSize + kernelIdx) * outHeight * outWidth
                                    + oh * outWidth + ow;
                const int offsetIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kernelIdx) * outHeight * outWidth
                                     + oh * outWidth + ow;

                float offset_h = offsets[offsetIdx];
                float offset_w = offsets[offsetIdxY];

                float h = base_h + kh * dilationH + offset_h;
                float w = base_w + kw * dilationW + offset_w;

                // Sample input with bilinear interpolation
                float inputVal = bilinear_sample(input, b, ic, h, w, inHeight, inWidth, inChannels);

                // Apply mask if using DCNv2
                if (hasMask != 0) {
                    const int maskIdx = ((b * deformGroups + dg) * kernelSize + kernelIdx) * outHeight * outWidth
                                      + oh * outWidth + ow;
                    inputVal *= mask[maskIdx];
                }

                float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                sumGrad += gradOutVal * inputVal;
            }
        }
    }

    gradWeights[idx] = sumGrad;
}

// Backward pass for offset gradients
__kernel void deformable_conv2d_backward_offset(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* weights,
    __global const float* offsets,
    __global const float* mask,
    __global float* gradOffsets,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups,
    const int deformGroups,
    const int hasMask)
{
    const int idx = get_global_id(0);
    const int kernelSize = kernelH * kernelW;
    const int totalOffsets = batch * deformGroups * 2 * kernelSize * outHeight * outWidth;

    if (idx >= totalOffsets) return;

    // Decompose offset index
    int tmp = idx;
    const int ow = tmp % outWidth; tmp /= outWidth;
    const int oh = tmp % outHeight; tmp /= outHeight;
    const int offsetComp = tmp % (2 * kernelSize); // 0 to 2*kernelSize-1
    tmp /= (2 * kernelSize);
    const int dg = tmp % deformGroups; tmp /= deformGroups;
    const int b = tmp;

    const int isYOffset = offsetComp >= kernelSize ? 1 : 0;
    const int kernelIdx = offsetComp % kernelSize;
    const int kh = kernelIdx / kernelW;
    const int kw = kernelIdx % kernelW;

    // Get current offset
    const int offsetIdxX = ((b * deformGroups + dg) * 2 * kernelSize + kernelIdx) * outHeight * outWidth
                         + oh * outWidth + ow;
    const int offsetIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kernelIdx) * outHeight * outWidth
                         + oh * outWidth + ow;

    float offset_h = offsets[offsetIdxX];
    float offset_w = offsets[offsetIdxY];

    int base_h = oh * strideH - padH;
    int base_w = ow * strideW - padW;

    float h = base_h + kh * dilationH + offset_h;
    float w = base_w + kw * dilationW + offset_w;

    // Compute bilinear gradient components
    int h_low = (int)floor(h);
    int w_low = (int)floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;

    float sumGrad = 0.0f;
    const int inChannelsPerGroup = inChannels / groups;
    const int outChannelsPerGroup = outChannels / groups;

    // Sum over output channels in this deform group
    for (int oc_offset = 0; oc_offset < outChannels / deformGroups; oc_offset++) {
        const int oc = dg * (outChannels / deformGroups) + oc_offset;
        const int g = oc / outChannelsPerGroup;

        float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

        if (hasMask != 0) {
            const int maskIdx = ((b * deformGroups + dg) * kernelSize + kernelIdx) * outHeight * outWidth
                              + oh * outWidth + ow;
            gradOutVal *= mask[maskIdx];
        }

        for (int ic = g * inChannelsPerGroup; ic < (g + 1) * inChannelsPerGroup; ic++) {
            const int icLocal = ic - g * inChannelsPerGroup;
            const int weightIdx = ((oc * inChannelsPerGroup + icLocal) * kernelH + kh) * kernelW + kw;
            float weightVal = weights[weightIdx];

            // Sample the four corners
            float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
            if (h_low >= 0 && h_low < inHeight && w_low >= 0 && w_low < inWidth)
                v1 = input[((b * inChannels + ic) * inHeight + h_low) * inWidth + w_low];
            if (h_low >= 0 && h_low < inHeight && w_high >= 0 && w_high < inWidth)
                v2 = input[((b * inChannels + ic) * inHeight + h_low) * inWidth + w_high];
            if (h_high >= 0 && h_high < inHeight && w_low >= 0 && w_low < inWidth)
                v3 = input[((b * inChannels + ic) * inHeight + h_high) * inWidth + w_low];
            if (h_high >= 0 && h_high < inHeight && w_high >= 0 && w_high < inWidth)
                v4 = input[((b * inChannels + ic) * inHeight + h_high) * inWidth + w_high];

            float grad_h, grad_w;
            if (isYOffset == 0) {
                // Gradient w.r.t. h offset
                grad_h = (1.0f - lw) * (v3 - v1) + lw * (v4 - v2);
                sumGrad += gradOutVal * weightVal * grad_h;
            } else {
                // Gradient w.r.t. w offset
                grad_w = (1.0f - lh) * (v2 - v1) + lh * (v4 - v3);
                sumGrad += gradOutVal * weightVal * grad_w;
            }
        }
    }

    gradOffsets[idx] = sumGrad;
}

// Backward pass for mask gradients (DCNv2)
__kernel void deformable_conv2d_backward_mask(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* weights,
    __global const float* offsets,
    __global float* gradMask,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups,
    const int deformGroups)
{
    const int idx = get_global_id(0);
    const int kernelSize = kernelH * kernelW;
    const int totalMask = batch * deformGroups * kernelSize * outHeight * outWidth;

    if (idx >= totalMask) return;

    // Decompose mask index
    int tmp = idx;
    const int ow = tmp % outWidth; tmp /= outWidth;
    const int oh = tmp % outHeight; tmp /= outHeight;
    const int kernelIdx = tmp % kernelSize; tmp /= kernelSize;
    const int dg = tmp % deformGroups; tmp /= deformGroups;
    const int b = tmp;

    const int kh = kernelIdx / kernelW;
    const int kw = kernelIdx % kernelW;

    // Get offset
    const int offsetIdxX = ((b * deformGroups + dg) * 2 * kernelSize + kernelIdx) * outHeight * outWidth
                         + oh * outWidth + ow;
    const int offsetIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kernelIdx) * outHeight * outWidth
                         + oh * outWidth + ow;

    float offset_h = offsets[offsetIdxX];
    float offset_w = offsets[offsetIdxY];

    int base_h = oh * strideH - padH;
    int base_w = ow * strideW - padW;

    float h = base_h + kh * dilationH + offset_h;
    float w = base_w + kw * dilationW + offset_w;

    float sumGrad = 0.0f;
    const int inChannelsPerGroup = inChannels / groups;
    const int outChannelsPerGroup = outChannels / groups;

    for (int oc_offset = 0; oc_offset < outChannels / deformGroups; oc_offset++) {
        const int oc = dg * (outChannels / deformGroups) + oc_offset;
        const int g = oc / outChannelsPerGroup;

        float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

        for (int ic = g * inChannelsPerGroup; ic < (g + 1) * inChannelsPerGroup; ic++) {
            const int icLocal = ic - g * inChannelsPerGroup;
            const int weightIdx = ((oc * inChannelsPerGroup + icLocal) * kernelH + kh) * kernelW + kw;
            float weightVal = weights[weightIdx];

            float inputVal = bilinear_sample(input, b, ic, h, w, inHeight, inWidth, inChannels);

            sumGrad += gradOutVal * weightVal * inputVal;
        }
    }

    gradMask[idx] = sumGrad;
}
";
        }
    }
}
