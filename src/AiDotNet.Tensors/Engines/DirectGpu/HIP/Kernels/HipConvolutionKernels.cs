// Copyright (c) AiDotNet. All rights reserved.
// HIP convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipConvolutionKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

extern ""C"" __global__ void im2col(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPatches = batch * outH * outW;
    if (idx >= totalPatches) return;

    int b = idx / (outH * outW);
    int rem = idx % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;

    int patchSize = channels * kernelH * kernelW;
    float* outPtr = output + idx * patchSize;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                float val = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    val = input[((b * channels + c) * height + ih) * width + iw];
                }
                int outIdx = (c * kernelH + kh) * kernelW + kw;
                outPtr[outIdx] = val;
            }
        }
    }
}

extern ""C"" __global__ void col2im(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * height * width;
    if (idx >= totalSize) return;

    int b = idx / (channels * height * width);
    int rem1 = idx % (channels * height * width);
    int c = rem1 / (height * width);
    int rem2 = rem1 % (height * width);
    int ih = rem2 / width;
    int iw = rem2 % width;

    float sum = 0.0f;
    int patchSize = channels * kernelH * kernelW;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int oh_base = ih + padH - kh * dilationH;
            int ow_base = iw + padW - kw * dilationW;
            if (oh_base % strideH == 0 && ow_base % strideW == 0) {
                int oh = oh_base / strideH;
                int ow = ow_base / strideW;
                if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                    int patchIdx = (b * outH + oh) * outW + ow;
                    int colIdx = (c * kernelH + kh) * kernelW + kw;
                    sum += input[patchIdx * patchSize + colIdx];
                }
            }
        }
    }
    output[idx] = sum;
}

extern ""C"" __global__ void conv2d_direct(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % outChannels;
    int b = blockIdx.z / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float kernelVal = kernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * kernelVal;
                }
            }
        }
    }
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

extern ""C"" __global__ void conv2d_backward_input(
    const float* gradOutput, const float* kernel, float* gradInput,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z % inChannels;
    int b = blockIdx.z / inChannels;

    if (iw >= inWidth || ih >= inHeight || b >= batch) return;

    float sum = 0.0f;
    for (int oc = 0; oc < outChannels; oc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh_base = ih + padH - kh * dilationH;
                int ow_base = iw + padW - kw * dilationW;
                if (oh_base % strideH == 0 && ow_base % strideW == 0) {
                    int oh = oh_base / strideH;
                    int ow = ow_base / strideW;
                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth) {
                        float gradVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                        float kernelVal = kernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                        sum += gradVal * kernelVal;
                    }
                }
            }
        }
    }
    gradInput[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
}

extern ""C"" __global__ void conv2d_backward_kernel(
    const float* input, const float* gradOutput, float* gradKernel,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int kw = blockIdx.x * blockDim.x + threadIdx.x;
    int kh = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z % inChannels;
    int oc = blockIdx.z / inChannels;

    if (kw >= kernelW || kh >= kernelH || oc >= outChannels) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float gradVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                    sum += inVal * gradVal;
                }
            }
        }
    }
    gradKernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw] = sum;
}

extern ""C"" __global__ void depthwise_conv2d(
    const float* input, const float* kernel, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float inVal = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                float kernelVal = kernel[(c * kernelH + kh) * kernelW + kw];
                sum += inVal * kernelVal;
            }
        }
    }
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum;
}

extern ""C"" __global__ void conv_transpose2d(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % outChannels;
    int b = blockIdx.z / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih_base = oh + padH - kh;
                int iw_base = ow + padW - kw;
                if (ih_base % strideH == 0 && iw_base % strideW == 0) {
                    int ih = ih_base / strideH;
                    int iw = iw_base / strideW;
                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                        float kernelVal = kernel[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                        sum += inVal * kernelVal;
                    }
                }
            }
        }
    }
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

extern ""C"" __global__ void conv3d_direct(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inDepth, int inHeight, int inWidth,
    int outChannels, int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilationD, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int od = blockIdx.z % outDepth;
    int oc = (blockIdx.z / outDepth) % outChannels;
    int b = blockIdx.z / (outDepth * outChannels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

    float sum = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kd = 0; kd < kernelD; kd++) {
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    int id = od * strideD - padD + kd * dilationD;
                    int ih = oh * strideH - padH + kh * dilationH;
                    int iw = ow * strideW - padW + kw * dilationW;
                    if (id >= 0 && id < inDepth && ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        float inVal = input[(((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw];
                        float kernelVal = kernel[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                        sum += inVal * kernelVal;
                    }
                }
            }
        }
    }
    output[(((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "im2col", "col2im", "conv2d_direct", "conv2d_backward_input",
            "conv2d_backward_kernel", "depthwise_conv2d", "conv_transpose2d", "conv3d_direct"
        };
    }
}
