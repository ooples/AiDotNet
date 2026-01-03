// Copyright (c) AiDotNet. All rights reserved.
// HIP pooling kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipPoolingKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

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
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "maxpool2d", "maxpool2d_backward", "avgpool2d", "avgpool2d_backward",
            "global_avgpool2d", "global_maxpool2d", "adaptive_avgpool2d"
        };
    }
}
