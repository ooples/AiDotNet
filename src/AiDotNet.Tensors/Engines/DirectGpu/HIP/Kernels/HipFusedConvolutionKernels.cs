// Copyright (c) AiDotNet. All rights reserved.
// HIP fused convolution kernels - Conv2D + BatchNorm/Bias + Activation in single pass.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    /// <summary>
    /// HIP fused convolution kernels for AMD GPU CNN performance optimization.
    /// </summary>
    internal static class HipFusedConvolutionKernels
    {
        public static string GetSource()
        {
            return @"
#include <hip/hip_runtime.h>
#include <math.h>

// ===========================================================================
// FUSED CONVOLUTION KERNELS: CONV2D + BIAS/BATCHNORM + ACTIVATION
// ===========================================================================

// Fused Conv2D + Bias + ReLU
extern ""C"" __global__ void conv2d_bias_relu(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    float result = sum + bias[oc];
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = fmaxf(0.0f, result);
}

// Fused Conv2D + Bias + GELU
extern ""C"" __global__ void conv2d_bias_gelu(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    float x = sum + bias[oc];
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = 0.5f * x * (1.0f + tanhf(inner));
}

// Fused Conv2D + Bias + Sigmoid
extern ""C"" __global__ void conv2d_bias_sigmoid(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    float x = sum + bias[oc];
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = 1.0f / (1.0f + expf(-x));
}

// Fused Conv2D + Bias (no activation)
extern ""C"" __global__ void conv2d_bias(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum + bias[oc];
}

// ===========================================================================
// FUSED CONV2D + BATCHNORM + ACTIVATION (INFERENCE MODE)
// ===========================================================================

// Fused Conv2D with folded BatchNorm + ReLU
extern ""C"" __global__ void conv2d_batchnorm_relu(
    const float* input, const float* foldedWeights, const float* foldedBias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = foldedWeights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    float result = sum + foldedBias[oc];
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = fmaxf(0.0f, result);
}

// Fused Conv2D with folded BatchNorm + GELU
extern ""C"" __global__ void conv2d_batchnorm_gelu(
    const float* input, const float* foldedWeights, const float* foldedBias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = foldedWeights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    float x = sum + foldedBias[oc];
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = 0.5f * x * (1.0f + tanhf(inner));
}

// Fused Conv2D with folded BatchNorm (no activation)
extern ""C"" __global__ void conv2d_batchnorm(
    const float* input, const float* foldedWeights, const float* foldedBias, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int oc = idx2 % outChannels;
    int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = foldedWeights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum + foldedBias[oc];
}

// ===========================================================================
// DEPTHWISE FUSED KERNELS
// ===========================================================================

// Fused Depthwise Conv2D + Bias + ReLU
extern ""C"" __global__ void depthwise_conv2d_bias_relu(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int c = idx2 % channels;
    int b = idx2 / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;

            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float inVal = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                float wVal = weights[(c * kernelH + kh) * kernelW + kw];
                sum += inVal * wVal;
            }
        }
    }

    float result = sum + bias[c];
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = fmaxf(0.0f, result);
}

// Fused Depthwise Conv2D + BatchNorm + ReLU
extern ""C"" __global__ void depthwise_conv2d_batchnorm_relu(
    const float* input, const float* foldedWeights, const float* foldedBias, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.z;
    int c = idx2 % channels;
    int b = idx2 / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;

            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float inVal = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                float wVal = foldedWeights[(c * kernelH + kh) * kernelW + kw];
                sum += inVal * wVal;
            }
        }
    }

    float result = sum + foldedBias[c];
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = fmaxf(0.0f, result);
}
";
        }

        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "conv2d_bias_relu",
                "conv2d_bias_gelu",
                "conv2d_bias_sigmoid",
                "conv2d_bias",
                "conv2d_batchnorm_relu",
                "conv2d_batchnorm_gelu",
                "conv2d_batchnorm",
                "depthwise_conv2d_bias_relu",
                "depthwise_conv2d_batchnorm_relu"
            };
        }
    }
}
