// Copyright (c) AiDotNet. All rights reserved.
// CUDA pooling kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for pooling operations.
    /// </summary>
    internal static class CudaPoolingKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// POOLING KERNELS
// ===========================================================================

// Max Pooling 2D with optional indices for backward pass
extern ""C"" __global__ void maxpool2d(
    const float* input, float* output, int* indices,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW, int saveIndices)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;

            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                int inputIdx = ((b * channels + c) * inHeight + ih) * inWidth + iw;
                float val = input[inputIdx];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = ih * inWidth + iw;
                }
            }
        }
    }

    int outIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
    output[outIdx] = maxVal;
    if (saveIndices) {
        indices[outIdx] = maxIdx;
    }
}

// Max Pooling 2D backward pass
extern ""C"" __global__ void maxpool2d_backward(
    const float* gradOutput, const int* indices, float* gradInput,
    int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    int outIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
    float grad = gradOutput[outIdx];
    int maxIdx = indices[outIdx];

    int ih = maxIdx / inWidth;
    int iw = maxIdx % inWidth;
    int inputIdx = ((b * channels + c) * inHeight + ih) * inWidth + iw;

    atomicAdd(&gradInput[inputIdx], grad);
}

// Average Pooling 2D
extern ""C"" __global__ void avgpool2d(
    const float* input, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW, int countIncludePad)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;

            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                sum += input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                count++;
            } else if (countIncludePad) {
                count++;
            }
        }
    }

    int divisor = countIncludePad ? (kernelH * kernelW) : count;
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum / (float)max(divisor, 1);
}

// Average Pooling 2D backward pass
extern ""C"" __global__ void avgpool2d_backward(
    const float* gradOutput, float* gradInput,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW, int countIncludePad)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (iw >= inWidth || ih >= inHeight || b >= batch) return;

    float sum = 0.0f;

    for (int oh = 0; oh < outHeight; oh++) {
        for (int ow = 0; ow < outWidth; ow++) {
            int hStart = oh * strideH - padH;
            int wStart = ow * strideW - padW;
            int hEnd = hStart + kernelH;
            int wEnd = wStart + kernelW;

            if (ih >= hStart && ih < hEnd && iw >= wStart && iw < wEnd) {
                int poolSize;
                if (countIncludePad) {
                    poolSize = kernelH * kernelW;
                } else {
                    int hStartClamp = max(hStart, 0);
                    int hEndClamp = min(hEnd, inHeight);
                    int wStartClamp = max(wStart, 0);
                    int wEndClamp = min(wEnd, inWidth);
                    poolSize = (hEndClamp - hStartClamp) * (wEndClamp - wStartClamp);
                }

                sum += gradOutput[((b * channels + c) * outHeight + oh) * outWidth + ow] / (float)max(poolSize, 1);
            }
        }
    }

    gradInput[((b * channels + c) * inHeight + ih) * inWidth + iw] = sum;
}

// Global Average Pooling 2D
extern ""C"" __global__ void global_avgpool2d(
    const float* input, float* output, int batch, int channels, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx % channels;
    int b = idx / channels;

    if (b >= batch) return;

    float sum = 0.0f;
    int spatialSize = height * width;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            sum += input[((b * channels + c) * height + h) * width + w];
        }
    }

    output[b * channels + c] = sum / (float)spatialSize;
}

// Global Max Pooling 2D
extern ""C"" __global__ void global_maxpool2d(
    const float* input, float* output, int batch, int channels, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx % channels;
    int b = idx / channels;

    if (b >= batch) return;

    float maxVal = -INFINITY;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            float val = input[((b * channels + c) * height + h) * width + w];
            maxVal = fmaxf(maxVal, val);
        }
    }

    output[b * channels + c] = maxVal;
}

// Global Average Pooling 2D Backward
// Each thread handles one output element, broadcasting gradient back to all input positions
extern ""C"" __global__ void global_avgpool2d_backward(
    const float* gradOutput, float* gradInput, int batch, int channels, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatialSize = height * width;
    int totalElements = batch * channels * spatialSize;

    if (idx >= totalElements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (channels * height * width);

    // Gradient is divided equally among all spatial positions
    float scale = 1.0f / (float)spatialSize;
    gradInput[idx] = gradOutput[b * channels + c] * scale;
}

// Global Max Pooling 2D Backward with indices
// Each thread handles one output gradient, scattering it to the max input position
extern ""C"" __global__ void global_maxpool2d_backward(
    const float* gradOutput, const int* indices, float* gradInput,
    int batch, int channels, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batch * channels;

    if (idx >= totalOutputs) return;

    int c = idx % channels;
    int b = idx / channels;
    int spatialSize = height * width;

    // Get the gradient value and the index of the max element
    float grad = gradOutput[idx];
    int maxIdx = indices[idx];

    // Convert local spatial index to global input index
    int inputOffset = (b * channels + c) * spatialSize;
    atomicAdd(&gradInput[inputOffset + maxIdx], grad);
}

// Adaptive Average Pooling 2D
extern ""C"" __global__ void adaptive_avgpool2d(
    const float* input, float* output,
    int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    int hStart = (oh * inHeight) / outHeight;
    int hEnd = ((oh + 1) * inHeight) / outHeight;
    int wStart = (ow * inWidth) / outWidth;
    int wEnd = ((ow + 1) * inWidth) / outWidth;

    float sum = 0.0f;
    int count = 0;

    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            sum += input[((b * channels + c) * inHeight + ih) * inWidth + iw];
            count++;
        }
    }

    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum / (float)max(count, 1);
}

// ===========================================================================
// 3D POOLING KERNELS
// ===========================================================================

// Max Pooling 3D with optional indices for backward pass
// Input layout: NCDHW (batch, channels, depth, height, width)
extern ""C"" __global__ void maxpool3d(
    const float* input, float* output, int* indices,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int saveIndices)
{
    // Thread maps to (ow, oh) with grid z for (batch * channels * outDepth)
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;

    // Iterate over 3D kernel window
    for (int kd = 0; kd < kernelD; kd++) {
        int id = od * strideD + kd;
        if (id >= inDepth) continue;

        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh;
            if (ih >= inHeight) continue;

            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW + kw;
                if (iw >= inWidth) continue;

                // NCDHW layout: ((b * C + c) * D + d) * H * W + h * W + w
                int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                             + ih * inWidth + iw;
                float val = input[inputIdx];

                if (val > maxVal) {
                    maxVal = val;
                    // Store linear index within the spatial volume for this (b,c)
                    maxIdx = id * inHeight * inWidth + ih * inWidth + iw;
                }
            }
        }
    }

    // Output index: NCDHW layout
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    output[outIdx] = maxVal;

    if (saveIndices) {
        indices[outIdx] = maxIdx;
    }
}

// Max Pooling 3D backward pass
extern ""C"" __global__ void maxpool3d_backward(
    const float* gradOutput, const int* indices, float* gradInput,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int outDepth, int outHeight, int outWidth)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    float grad = gradOutput[outIdx];
    int maxIdx = indices[outIdx];

    // Decode maxIdx back to (id, ih, iw)
    int spatialHW = inHeight * inWidth;
    int id = maxIdx / spatialHW;
    int rem = maxIdx % spatialHW;
    int ih = rem / inWidth;
    int iw = rem % inWidth;

    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;
    atomicAdd(&gradInput[inputIdx], grad);
}

// Nearest Neighbor Upsample 3D
// Upsamples each spatial dimension by integer scale factors
extern ""C"" __global__ void nearest_upsample3d(
    const float* input, float* output,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int scaleD, int scaleH, int scaleW)
{
    int outDepth = inDepth * scaleD;
    int outHeight = inHeight * scaleH;
    int outWidth = inWidth * scaleW;

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    // Map output coord to input coord (nearest neighbor)
    int id = od / scaleD;
    int ih = oh / scaleH;
    int iw = ow / scaleW;

    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;

    output[outIdx] = input[inputIdx];
}

// Nearest Neighbor Upsample 3D backward pass
extern ""C"" __global__ void nearest_upsample3d_backward(
    const float* gradOutput, float* gradInput,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int scaleD, int scaleH, int scaleW)
{
    int outDepth = inDepth * scaleD;
    int outHeight = inHeight * scaleH;
    int outWidth = inWidth * scaleW;

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    // Map output coord to input coord
    int id = od / scaleD;
    int ih = oh / scaleH;
    int iw = ow / scaleW;

    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;

    atomicAdd(&gradInput[inputIdx], gradOutput[outIdx]);
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "maxpool2d",
                "maxpool2d_backward",
                "avgpool2d",
                "avgpool2d_backward",
                "global_avgpool2d",
                "global_maxpool2d",
                "global_avgpool2d_backward",
                "global_maxpool2d_backward",
                "adaptive_avgpool2d",
                "maxpool3d",
                "maxpool3d_backward",
                "nearest_upsample3d",
                "nearest_upsample3d_backward"
            };
        }
    }
}
