// Copyright (c) AiDotNet. All rights reserved.
// Convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for convolution operations.
    /// Implements im2col + GEMM approach for efficiency.
    /// </summary>
    internal static class ConvolutionKernels
    {
        /// <summary>
        /// Gets all convolution kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// CONVOLUTION KERNELS
// ===========================================================================

// Im2Col transformation for efficient convolution via GEMM
// Transforms input patches into columns for matrix multiplication
__kernel void im2col(
    __global const float* input,
    __global float* output,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int outH,
    const int outW)
{
    const int idx = get_global_id(0);
    const int totalPatches = batch * outH * outW;
    if (idx >= totalPatches) return;

    const int b = idx / (outH * outW);
    const int rem = idx % (outH * outW);
    const int oh = rem / outW;
    const int ow = rem % outW;

    const int patchSize = channels * kernelH * kernelW;
    __global float* outPtr = output + idx * patchSize;

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

// Col2Im transformation for convolution backward pass
// Accumulates gradients from columns back to input gradient
__kernel void col2im(
    __global const float* input,
    __global float* output,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int outH,
    const int outW)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * height * width;
    if (idx >= totalSize) return;

    const int b = idx / (channels * height * width);
    const int rem1 = idx % (channels * height * width);
    const int c = rem1 / (height * width);
    const int rem2 = rem1 % (height * width);
    const int ih = rem2 / width;
    const int iw = rem2 % width;

    float sum = 0.0f;
    const int patchSize = channels * kernelH * kernelW;

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

// Direct Conv2D kernel for small kernels (3x3, 5x5)
// Uses shared memory for input tile caching
__kernel void conv2d_direct(
    __global const float* input,
    __global const float* weights,
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
    const int dilationW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

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

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

// Conv2D backward pass for input gradients
__kernel void conv2d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
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
    const int dilationW)
{
    const int iw = get_global_id(0);
    const int ih = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int ic = idx2 % inChannels;
    const int b = idx2 / inChannels;

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
                        float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                        sum += gradVal * wVal;
                    }
                }
            }
        }
    }

    gradInput[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
}

// Conv2D backward pass for weight gradients
__kernel void conv2d_backward_weights(
    __global const float* input,
    __global const float* gradOutput,
    __global float* gradKernel,
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
    const int dilationW)
{
    const int kw = get_global_id(0);
    const int kh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int ic = idx2 % inChannels;
    const int oc = idx2 / inChannels;

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

// Depthwise Conv2D - each channel is convolved independently
__kernel void depthwise_conv2d(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int c = idx2 % channels;
    const int b = idx2 / channels;

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

    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum;
}

// Transposed Conv2D (deconvolution) with output padding support
__kernel void conv_transpose2d(
    __global const float* input,
    __global const float* weights,
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
    const int outputPadH,
    const int outputPadW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

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
                        // Note: weights layout is [inChannels, outChannels, kH, kW]
                        float wVal = weights[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                        sum += inVal * wVal;
                    }
                }
            }
        }
    }

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

// Transposed Conv2D backward pass for input gradients
__kernel void conv_transpose2d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
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
    const int outputPadH,
    const int outputPadW,
    const int totalInput)
{
    const int idx = get_global_id(0);
    if (idx >= totalInput) return;

    const int iw = idx % inWidth;
    const int ih = (idx / inWidth) % inHeight;
    const int ic = (idx / (inWidth * inHeight)) % inChannels;
    const int b = idx / (inWidth * inHeight * inChannels);

    // Effective output dimensions excluding output padding
    const int outHeight_eff = outHeight - outputPadH;
    const int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    for (int oc = 0; oc < outChannels; oc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;

                // Only consider positions within effective output (excluding output padding region)
                if (oh >= 0 && oh < outHeight_eff && ow >= 0 && ow < outWidth_eff) {
                    int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    sum += gradOutput[goIdx] * weights[kIdx];
                }
            }
        }
    }

    gradInput[idx] = sum;
}

// Transposed Conv2D backward pass for kernel gradients
__kernel void conv_transpose2d_backward_weights(
    __global const float* input,
    __global const float* gradOutput,
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
    const int outputPadH,
    const int outputPadW,
    const int totalKernel)
{
    const int idx = get_global_id(0);
    if (idx >= totalKernel) return;

    const int kw = idx % kernelW;
    const int kh = (idx / kernelW) % kernelH;
    const int oc = (idx / (kernelW * kernelH)) % outChannels;
    const int ic = idx / (kernelW * kernelH * outChannels);

    // Effective output dimensions excluding output padding
    const int outHeight_eff = outHeight - outputPadH;
    const int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int ih = 0; ih < inHeight; ih++) {
            for (int iw = 0; iw < inWidth; iw++) {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;

                // Only consider positions within effective output (excluding output padding region)
                if (oh >= 0 && oh < outHeight_eff && ow >= 0 && ow < outWidth_eff) {
                    int inIdx = ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                    int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                    sum += input[inIdx] * gradOutput[goIdx];
                }
            }
        }
    }

    gradWeights[idx] = sum;
}

// Conv3D for volumetric data
__kernel void conv3d_direct(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inDepth,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outDepth,
    const int outHeight,
    const int outWidth,
    const int kernelD,
    const int kernelH,
    const int kernelW,
    const int strideD,
    const int strideH,
    const int strideW,
    const int padD,
    const int padH,
    const int padW,
    const int dilationD,
    const int dilationH,
    const int dilationW)
{
    const int ow = get_global_id(0);
    const int idx1 = get_global_id(1);
    const int oh = idx1 % outHeight;
    const int od = idx1 / outHeight;
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

    if (ow >= outWidth || od >= outDepth || b >= batch) return;

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
                        float wVal = weights[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                        sum += inVal * wVal;
                    }
                }
            }
        }
    }

    output[(((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "im2col",
                "col2im",
                "conv2d_direct",
                "conv2d_backward_input",
                "conv2d_backward_weights",
                "depthwise_conv2d",
                "conv_transpose2d",
                "conv_transpose2d_backward_input",
                "conv_transpose2d_backward_weights",
                "conv3d_direct"
            };
        }
    }
}
