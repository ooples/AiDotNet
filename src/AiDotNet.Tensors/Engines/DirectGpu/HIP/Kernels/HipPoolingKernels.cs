// Copyright (c) AiDotNet. All rights reserved.
// HIP pooling kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipPoolingKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

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
    if (saveIndices) indices[outIdx] = maxIdx;
}

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

extern ""C"" __global__ void global_maxpool2d(
    const float* input, float* output, int* indices,
    int batch, int channels, int height, int width, int saveIndices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx % channels;
    int b = idx / channels;

    if (b >= batch) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;
    int spatialSize = height * width;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int spatialIdx = h * width + w;
            float val = input[((b * channels + c) * height + h) * width + w];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = spatialIdx;
            }
        }
    }

    int outIdx = b * channels + c;
    output[outIdx] = maxVal;
    if (saveIndices) indices[outIdx] = maxIdx;
}

// Global Average Pooling 2D Backward
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

    // Bounds check: ensure maxIdx is valid before writing to gradInput
    if (maxIdx < 0 || maxIdx >= spatialSize) return;

    // Convert local spatial index to global input index
    int inputOffset = (b * channels + c) * spatialSize;
    atomicAdd(&gradInput[inputOffset + maxIdx], grad);
}

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

extern ""C"" __global__ void maxpool3d(
    const float* input, float* output, int* indices,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int saveIndices)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;

    for (int kd = 0; kd < kernelD; kd++) {
        int id = od * strideD + kd;
        if (id >= inDepth) continue;

        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh;
            if (ih >= inHeight) continue;

            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW + kw;
                if (iw >= inWidth) continue;

                int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                             + ih * inWidth + iw;
                float val = input[inputIdx];

                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = id * inHeight * inWidth + ih * inWidth + iw;
                }
            }
        }
    }

    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    output[outIdx] = maxVal;
    if (saveIndices) indices[outIdx] = maxIdx;
}

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

    int spatialHW = inHeight * inWidth;
    int id = maxIdx / spatialHW;
    int rem = maxIdx % spatialHW;
    int ih = rem / inWidth;
    int iw = rem % inWidth;

    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;
    atomicAdd(&gradInput[inputIdx], grad);
}

// Average Pooling 3D
extern ""C"" __global__ void avgpool3d(
    const float* input, float* output,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int countIncludePad)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int od = linear_z % outDepth;
    int c = (linear_z / outDepth) % channels;
    int b = linear_z / (outDepth * channels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    float sum = 0.0f;
    int count = 0;

    for (int kd = 0; kd < kernelD; kd++) {
        int id = od * strideD + kd;
        if (id >= inDepth) continue;

        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh;
            if (ih >= inHeight) continue;

            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW + kw;
                if (iw >= inWidth) continue;

                int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                             + ih * inWidth + iw;
                sum += input[inputIdx];
                count++;
            }
        }
    }

    int divisor = countIncludePad ? (kernelD * kernelH * kernelW) : count;
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    output[outIdx] = sum / (float)max(divisor, 1);
}

// Average Pooling 3D backward pass
extern ""C"" __global__ void avgpool3d_backward(
    const float* gradOutput, float* gradInput,
    int batch, int channels,
    int inDepth, int inHeight, int inWidth,
    int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int countIncludePad)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z;

    int id = linear_z % inDepth;
    int c = (linear_z / inDepth) % channels;
    int b = linear_z / (inDepth * channels);

    if (iw >= inWidth || ih >= inHeight || id >= inDepth || b >= batch) return;

    float sum = 0.0f;

    for (int od = 0; od < outDepth; od++) {
        int dStart = od * strideD;
        int dEnd = dStart + kernelD;
        if (id < dStart || id >= dEnd) continue;

        for (int oh = 0; oh < outHeight; oh++) {
            int hStart = oh * strideH;
            int hEnd = hStart + kernelH;
            if (ih < hStart || ih >= hEnd) continue;

            for (int ow = 0; ow < outWidth; ow++) {
                int wStart = ow * strideW;
                int wEnd = wStart + kernelW;
                if (iw < wStart || iw >= wEnd) continue;

                int poolSize;
                if (countIncludePad) {
                    poolSize = kernelD * kernelH * kernelW;
                } else {
                    int dEndClamp = min(dEnd, inDepth);
                    int hEndClamp = min(hEnd, inHeight);
                    int wEndClamp = min(wEnd, inWidth);
                    poolSize = (dEndClamp - dStart) * (hEndClamp - hStart) * (wEndClamp - wStart);
                }

                int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
                           + oh * outWidth + ow;
                sum += gradOutput[outIdx] / (float)max(poolSize, 1);
            }
        }
    }

    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;
    gradInput[inputIdx] = sum;
}

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

    int id = od / scaleD;
    int ih = oh / scaleH;
    int iw = ow / scaleW;

    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;

    output[outIdx] = input[inputIdx];
}

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

    int id = od / scaleD;
    int ih = oh / scaleH;
    int iw = ow / scaleW;

    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth
               + oh * outWidth + ow;
    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth
                 + ih * inWidth + iw;

    atomicAdd(&gradInput[inputIdx], gradOutput[outIdx]);
}

// ===========================================================================
// 2D NEAREST NEIGHBOR UPSAMPLING
// ===========================================================================

extern ""C"" __global__ void nearest_neighbor_upsample(
    const float* input, float* output,
    int batchChannels, int height, int width,
    int scaleFactor, int totalOutputSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalOutputSize) return;

    int outHeight = height * scaleFactor;
    int outWidth = width * scaleFactor;
    int spatialOut = outHeight * outWidth;

    int bc = idx / spatialOut;
    int spatial = idx % spatialOut;
    int oh = spatial / outWidth;
    int ow = spatial % outWidth;

    int ih = oh / scaleFactor;
    int iw = ow / scaleFactor;
    int inputIdx = bc * height * width + ih * width + iw;

    output[idx] = input[inputIdx];
}

// Nearest Neighbor Upsample 2D backward pass
// Iterates over INPUT elements to avoid race conditions and atomic contention
// Each thread accumulates gradients from the scaleFactor x scaleFactor output region
extern ""C"" __global__ void nearest_neighbor_upsample_backward(
    const float* gradOutput, float* gradInput,
    int batchChannels, int height, int width,
    int scaleFactor, int totalInputSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalInputSize) return;

    int outHeight = height * scaleFactor;
    int outWidth = width * scaleFactor;
    int spatialIn = height * width;
    int spatialOut = outHeight * outWidth;

    // Decompose input index into batch-channel and spatial components
    int bc = idx / spatialIn;
    int spatial = idx % spatialIn;
    int ih = spatial / width;
    int iw = spatial % width;

    // Compute the top-left corner in output space
    int oh_start = ih * scaleFactor;
    int ow_start = iw * scaleFactor;

    // Accumulate gradients from the scaleFactor x scaleFactor output region
    float grad_sum = 0.0f;
    for (int dy = 0; dy < scaleFactor; dy++) {
        for (int dx = 0; dx < scaleFactor; dx++) {
            int oh = oh_start + dy;
            int ow = ow_start + dx;
            int outIdx = bc * spatialOut + oh * outWidth + ow;
            grad_sum += gradOutput[outIdx];
        }
    }

    gradInput[idx] = grad_sum;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "maxpool2d", "maxpool2d_backward", "avgpool2d", "avgpool2d_backward",
            "global_avgpool2d", "global_maxpool2d", "global_avgpool2d_backward", "global_maxpool2d_backward", "adaptive_avgpool2d",
            "maxpool3d", "maxpool3d_backward", "avgpool3d", "avgpool3d_backward",
            "nearest_upsample3d", "nearest_upsample3d_backward",
            "nearest_neighbor_upsample", "nearest_neighbor_upsample_backward"
        };
    }
}
