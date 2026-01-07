// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for deformable convolution operations (DCNv1/DCNv2).
// Deformable convolution learns offsets to sample positions, enabling adaptive receptive fields.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for deformable convolution operations.
    /// </summary>
    /// <remarks>
    /// Deformable convolution (DCN) extends standard convolution by learning offsets
    /// to each sampling position. This enables the network to adaptively adjust
    /// its receptive field based on the input content.
    ///
    /// DCNv2 additionally includes learnable modulation masks that weight each
    /// sampling position, providing finer control over feature aggregation.
    /// </remarks>
    internal static class CudaDeformableConvolutionKernels
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
#include <math.h>

// ===========================================================================
// DEFORMABLE CONVOLUTION KERNELS (DCNv2)
// ===========================================================================

// Bilinear sampling helper function
// Samples from input at fractional position (y, x) using bilinear interpolation
__device__ float bilinear_sample(
    const float* input,
    int batch, int channel, int height, int width,
    float y, float x)
{
    // Clamp to valid range
    if (y < -1.0f || y > (float)height || x < -1.0f || x > (float)width) {
        return 0.0f;
    }

    // Get integer and fractional parts
    int y0 = (int)floorf(y);
    int x0 = (int)floorf(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float ly = y - (float)y0;
    float lx = x - (float)x0;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Sample four corners with bounds checking
    float v00 = (y0 >= 0 && y0 < height && x0 >= 0 && x0 < width) ?
        input[((batch * channel) * height + y0) * width + x0] : 0.0f;
    float v01 = (y0 >= 0 && y0 < height && x1 >= 0 && x1 < width) ?
        input[((batch * channel) * height + y0) * width + x1] : 0.0f;
    float v10 = (y1 >= 0 && y1 < height && x0 >= 0 && x0 < width) ?
        input[((batch * channel) * height + y1) * width + x0] : 0.0f;
    float v11 = (y1 >= 0 && y1 < height && x1 >= 0 && x1 < width) ?
        input[((batch * channel) * height + y1) * width + x1] : 0.0f;

    // Bilinear interpolation
    return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
}

// Forward pass for deformable 2D convolution (DCNv2)
// Input:   [batch, inChannels, inH, inW]
// Weights: [outChannels, inChannels/groups, kH, kW]
// Offsets: [batch, deformGroups * 2 * kH * kW, outH, outW]
// Mask:    [batch, deformGroups * kH * kW, outH, outW] (optional for DCNv2)
// Output:  [batch, outChannels, outH, outW]
extern ""C"" __global__ void deformable_conv2d(
    const float* input,
    const float* weights,
    const float* offsets,
    const float* mask,
    float* output,
    int batch,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int deformGroups,
    int hasMask)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int boc = blockIdx.z;

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    int b = boc / outChannels;
    int oc = boc % outChannels;

    int inChannelsPerGroup = inChannels / groups;
    int outChannelsPerGroup = outChannels / groups;
    int g = oc / outChannelsPerGroup;
    int inChannelsPerDeformGroup = inChannels / deformGroups;

    float sum = 0.0f;

    for (int ic = g * inChannelsPerGroup; ic < (g + 1) * inChannelsPerGroup; ic++) {
        int deformGroup = ic / inChannelsPerDeformGroup;

        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                // Base position
                float base_y = oh * strideH - padH + kh * dilationH;
                float base_x = ow * strideW - padW + kw * dilationW;

                // Get offset index
                int offsetIdx = ((deformGroup * kernelH * kernelW + kh * kernelW + kw) * 2);
                int offsetBaseIdx = ((b * deformGroups * 2 * kernelH * kernelW + offsetIdx) * outHeight + oh) * outWidth + ow;

                // Apply learned offset
                float offset_y = offsets[offsetBaseIdx];
                float offset_x = offsets[offsetBaseIdx + outHeight * outWidth];

                float sample_y = base_y + offset_y;
                float sample_x = base_x + offset_x;

                // Bilinear sample
                float sampled = bilinear_sample(input, b, ic, inHeight, inWidth, sample_y, sample_x);

                // Apply mask if using DCNv2
                if (hasMask != 0) {
                    int maskIdx = ((b * deformGroups * kernelH * kernelW + deformGroup * kernelH * kernelW + kh * kernelW + kw) * outHeight + oh) * outWidth + ow;
                    sampled *= mask[maskIdx];
                }

                // Get weight
                int localIc = ic - g * inChannelsPerGroup;
                float w = weights[((oc * inChannelsPerGroup + localIc) * kernelH + kh) * kernelW + kw];

                sum += sampled * w;
            }
        }
    }

    int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = sum;
}

// Backward pass for input gradients
extern ""C"" __global__ void deformable_conv2d_backward_input(
    const float* gradOutput,
    const float* weights,
    const float* offsets,
    const float* mask,
    float* gradInput,
    int batch,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int deformGroups,
    int hasMask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInputSize = batch * inChannels * inHeight * inWidth;

    if (idx >= totalInputSize) return;

    int b = idx / (inChannels * inHeight * inWidth);
    int rem1 = idx % (inChannels * inHeight * inWidth);
    int ic = rem1 / (inHeight * inWidth);
    int rem2 = rem1 % (inHeight * inWidth);
    int ih = rem2 / inWidth;
    int iw = rem2 % inWidth;

    int inChannelsPerGroup = inChannels / groups;
    int outChannelsPerGroup = outChannels / groups;
    int g = ic / inChannelsPerGroup;
    int inChannelsPerDeformGroup = inChannels / deformGroups;
    int deformGroup = ic / inChannelsPerDeformGroup;

    float sumGrad = 0.0f;

    // For each output position that might sample from this input position
    for (int oh = 0; oh < outHeight; oh++) {
        for (int ow = 0; ow < outWidth; ow++) {
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    float base_y = oh * strideH - padH + kh * dilationH;
                    float base_x = ow * strideW - padW + kw * dilationW;

                    int offsetIdx = ((deformGroup * kernelH * kernelW + kh * kernelW + kw) * 2);
                    int offsetBaseIdx = ((b * deformGroups * 2 * kernelH * kernelW + offsetIdx) * outHeight + oh) * outWidth + ow;

                    float offset_y = offsets[offsetBaseIdx];
                    float offset_x = offsets[offsetBaseIdx + outHeight * outWidth];

                    float sample_y = base_y + offset_y;
                    float sample_x = base_x + offset_x;

                    // Check if this input position contributes to this sample
                    int y0 = (int)floorf(sample_y);
                    int x0 = (int)floorf(sample_x);

                    if (ih >= y0 && ih <= y0 + 1 && iw >= x0 && iw <= x0 + 1) {
                        float ly = sample_y - (float)y0;
                        float lx = sample_x - (float)x0;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;

                        float bilinearWeight = 0.0f;
                        if (ih == y0 && iw == x0) bilinearWeight = hy * hx;
                        else if (ih == y0 && iw == x0 + 1) bilinearWeight = hy * lx;
                        else if (ih == y0 + 1 && iw == x0) bilinearWeight = ly * hx;
                        else if (ih == y0 + 1 && iw == x0 + 1) bilinearWeight = ly * lx;

                        if (bilinearWeight > 0.0f) {
                            for (int oc = g * outChannelsPerGroup; oc < (g + 1) * outChannelsPerGroup; oc++) {
                                float gradOut = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                                int localIc = ic - g * inChannelsPerGroup;
                                float w = weights[((oc * inChannelsPerGroup + localIc) * kernelH + kh) * kernelW + kw];

                                float maskVal = 1.0f;
                                if (hasMask != 0) {
                                    int maskIdx = ((b * deformGroups * kernelH * kernelW + deformGroup * kernelH * kernelW + kh * kernelW + kw) * outHeight + oh) * outWidth + ow;
                                    maskVal = mask[maskIdx];
                                }

                                sumGrad += gradOut * w * bilinearWeight * maskVal;
                            }
                        }
                    }
                }
            }
        }
    }

    gradInput[idx] = sumGrad;
}

// Backward pass for weight gradients
extern ""C"" __global__ void deformable_conv2d_backward_weights(
    const float* input,
    const float* gradOutput,
    const float* offsets,
    const float* mask,
    float* gradWeights,
    int batch,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int deformGroups,
    int hasMask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inChannelsPerGroup = inChannels / groups;
    int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;

    if (idx >= totalWeights) return;

    int kw = idx % kernelW;
    int tmp = idx / kernelW;
    int kh = tmp % kernelH;
    tmp /= kernelH;
    int localIc = tmp % inChannelsPerGroup;
    int oc = tmp / inChannelsPerGroup;

    int outChannelsPerGroup = outChannels / groups;
    int g = oc / outChannelsPerGroup;
    int ic = g * inChannelsPerGroup + localIc;
    int inChannelsPerDeformGroup = inChannels / deformGroups;
    int deformGroup = ic / inChannelsPerDeformGroup;

    float sumGrad = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                float base_y = oh * strideH - padH + kh * dilationH;
                float base_x = ow * strideW - padW + kw * dilationW;

                int offsetIdx = ((deformGroup * kernelH * kernelW + kh * kernelW + kw) * 2);
                int offsetBaseIdx = ((b * deformGroups * 2 * kernelH * kernelW + offsetIdx) * outHeight + oh) * outWidth + ow;

                float offset_y = offsets[offsetBaseIdx];
                float offset_x = offsets[offsetBaseIdx + outHeight * outWidth];

                float sample_y = base_y + offset_y;
                float sample_x = base_x + offset_x;

                float sampled = bilinear_sample(input, b, ic, inHeight, inWidth, sample_y, sample_x);

                if (hasMask != 0) {
                    int maskIdx = ((b * deformGroups * kernelH * kernelW + deformGroup * kernelH * kernelW + kh * kernelW + kw) * outHeight + oh) * outWidth + ow;
                    sampled *= mask[maskIdx];
                }

                float gradOut = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                sumGrad += gradOut * sampled;
            }
        }
    }

    gradWeights[idx] = sumGrad;
}

// Backward pass for offset gradients
extern ""C"" __global__ void deformable_conv2d_backward_offset(
    const float* input,
    const float* weights,
    const float* gradOutput,
    const float* offsets,
    const float* mask,
    float* gradOffset,
    int batch,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int deformGroups,
    int hasMask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOffsets = batch * deformGroups * 2 * kernelH * kernelW * outHeight * outWidth;

    if (idx >= totalOffsets) return;

    // Decompose index
    int ow = idx % outWidth;
    int tmp = idx / outWidth;
    int oh = tmp % outHeight;
    tmp /= outHeight;
    int offsetComponent = tmp % 2; // 0 = y offset, 1 = x offset
    tmp /= 2;
    int kw = tmp % kernelW;
    tmp /= kernelW;
    int kh = tmp % kernelH;
    tmp /= kernelH;
    int deformGroup = tmp % deformGroups;
    int b = tmp / deformGroups;

    float base_y = oh * strideH - padH + kh * dilationH;
    float base_x = ow * strideW - padW + kw * dilationW;

    int offsetIdx = ((deformGroup * kernelH * kernelW + kh * kernelW + kw) * 2);
    int offsetBaseIdx = ((b * deformGroups * 2 * kernelH * kernelW + offsetIdx) * outHeight + oh) * outWidth + ow;

    float offset_y = offsets[offsetBaseIdx];
    float offset_x = offsets[offsetBaseIdx + outHeight * outWidth];

    float sample_y = base_y + offset_y;
    float sample_x = base_x + offset_x;

    // Compute gradient of bilinear sampling with respect to coordinates
    int y0 = (int)floorf(sample_y);
    int x0 = (int)floorf(sample_x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float ly = sample_y - (float)y0;
    float lx = sample_x - (float)x0;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    float sumGrad = 0.0f;

    int inChannelsPerDeformGroup = inChannels / deformGroups;
    int outChannelsPerGroup = outChannels / groups;
    int inChannelsPerGroup = inChannels / groups;

    for (int icLocal = 0; icLocal < inChannelsPerDeformGroup; icLocal++) {
        int ic = deformGroup * inChannelsPerDeformGroup + icLocal;
        int g = ic / inChannelsPerGroup;

        // Get input values
        float v00 = (y0 >= 0 && y0 < inHeight && x0 >= 0 && x0 < inWidth) ?
            input[((b * inChannels + ic) * inHeight + y0) * inWidth + x0] : 0.0f;
        float v01 = (y0 >= 0 && y0 < inHeight && x1 >= 0 && x1 < inWidth) ?
            input[((b * inChannels + ic) * inHeight + y0) * inWidth + x1] : 0.0f;
        float v10 = (y1 >= 0 && y1 < inHeight && x0 >= 0 && x0 < inWidth) ?
            input[((b * inChannels + ic) * inHeight + y1) * inWidth + x0] : 0.0f;
        float v11 = (y1 >= 0 && y1 < inHeight && x1 >= 0 && x1 < inWidth) ?
            input[((b * inChannels + ic) * inHeight + y1) * inWidth + x1] : 0.0f;

        // Gradient of bilinear sampling
        float grad_sample;
        if (offsetComponent == 0) {  // y offset
            grad_sample = -hx * v00 - lx * v01 + hx * v10 + lx * v11;
        } else {  // x offset
            grad_sample = -hy * v00 + hy * v01 - ly * v10 + ly * v11;
        }

        for (int ocLocal = 0; ocLocal < outChannelsPerGroup; ocLocal++) {
            int oc = g * outChannelsPerGroup + ocLocal;
            float gradOut = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

            int localIc = ic - g * inChannelsPerGroup;
            float w = weights[((oc * inChannelsPerGroup + localIc) * kernelH + kh) * kernelW + kw];

            float maskVal = 1.0f;
            if (hasMask != 0) {
                int maskIdx = ((b * deformGroups * kernelH * kernelW + deformGroup * kernelH * kernelW + kh * kernelW + kw) * outHeight + oh) * outWidth + ow;
                maskVal = mask[maskIdx];
            }

            sumGrad += gradOut * w * grad_sample * maskVal;
        }
    }

    gradOffset[idx] = sumGrad;
}

// Backward pass for mask gradients (DCNv2 only)
extern ""C"" __global__ void deformable_conv2d_backward_mask(
    const float* input,
    const float* weights,
    const float* gradOutput,
    const float* offsets,
    float* gradMask,
    int batch,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int deformGroups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalMask = batch * deformGroups * kernelH * kernelW * outHeight * outWidth;

    if (idx >= totalMask) return;

    // Decompose index
    int ow = idx % outWidth;
    int tmp = idx / outWidth;
    int oh = tmp % outHeight;
    tmp /= outHeight;
    int kw = tmp % kernelW;
    tmp /= kernelW;
    int kh = tmp % kernelH;
    tmp /= kernelH;
    int deformGroup = tmp % deformGroups;
    int b = tmp / deformGroups;

    float base_y = oh * strideH - padH + kh * dilationH;
    float base_x = ow * strideW - padW + kw * dilationW;

    int offsetIdx = ((deformGroup * kernelH * kernelW + kh * kernelW + kw) * 2);
    int offsetBaseIdx = ((b * deformGroups * 2 * kernelH * kernelW + offsetIdx) * outHeight + oh) * outWidth + ow;

    float offset_y = offsets[offsetBaseIdx];
    float offset_x = offsets[offsetBaseIdx + outHeight * outWidth];

    float sample_y = base_y + offset_y;
    float sample_x = base_x + offset_x;

    float sumGrad = 0.0f;

    int inChannelsPerDeformGroup = inChannels / deformGroups;
    int outChannelsPerGroup = outChannels / groups;
    int inChannelsPerGroup = inChannels / groups;

    for (int icLocal = 0; icLocal < inChannelsPerDeformGroup; icLocal++) {
        int ic = deformGroup * inChannelsPerDeformGroup + icLocal;
        int g = ic / inChannelsPerGroup;

        float sampled = bilinear_sample(input, b, ic, inHeight, inWidth, sample_y, sample_x);

        for (int ocLocal = 0; ocLocal < outChannelsPerGroup; ocLocal++) {
            int oc = g * outChannelsPerGroup + ocLocal;
            float gradOut = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

            int localIc = ic - g * inChannelsPerGroup;
            float w = weights[((oc * inChannelsPerGroup + localIc) * kernelH + kh) * kernelW + kw];

            sumGrad += gradOut * w * sampled;
        }
    }

    gradMask[idx] = sumGrad;
}
";
        }
    }
}
