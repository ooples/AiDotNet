// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for locally connected convolution operations.
// Unlike standard convolution, locally connected uses unique weights per spatial position.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for locally connected convolution operations.
/// </summary>
/// <remarks>
/// Locally connected layers differ from standard convolution in that each spatial
/// position has its own unique set of weights. This increases parameter count but
/// allows the network to learn location-specific features.
/// </remarks>
internal static class HipLocallyConnectedKernels
{
    /// <summary>
    /// Gets all locally connected convolution kernel names.
    /// </summary>
    public static string[] GetKernelNames() =>
    [
        "locally_connected_conv2d",
        "locally_connected_conv2d_backward_input",
        "locally_connected_conv2d_backward_weights",
        "locally_connected_conv2d_backward_bias",
        "locally_connected_conv2d_fused"
    ];

    /// <summary>
    /// Gets all locally connected convolution kernel sources.
    /// </summary>
    public static string GetSource()
    {
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// Forward pass for locally connected 2D convolution
// Each output spatial position uses a unique set of weights
// Input:   [batch, inChannels, inH, inW]
// Weights: [outH, outW, outC, inC, kH, kW]
// Bias:    [outC] (optional)
// Output:  [batch, outC, outH, outW]
extern ""C"" __global__ void locally_connected_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
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
    int hasBias)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int boc = blockIdx.z;

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    int b = boc / outChannels;
    int oc = boc % outChannels;

    // Compute weighted sum using position-specific weights
    float sum = 0.0f;

    // Weight offset for this output position: [oh, ow, oc, :, :, :]
    // Layout: [outH, outW, outC, inC, kH, kW]
    int weightBaseIdx = ((oh * outWidth + ow) * outChannels + oc) * inChannels * kernelH * kernelW;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;

                // Bounds check (locally connected typically doesn't use padding)
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inputVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];

                    // Weight index: [oh, ow, oc, ic, kh, kw]
                    int weightIdx = weightBaseIdx + (ic * kernelH + kh) * kernelW + kw;
                    float weightVal = weights[weightIdx];

                    sum += inputVal * weightVal;
                }
            }
        }
    }

    // Add bias if provided
    if (hasBias != 0) {
        sum += bias[oc];
    }

    // Write output
    int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = sum;
}

// Backward pass for input gradients
// Computes gradient with respect to input tensor
extern ""C"" __global__ void locally_connected_conv2d_backward_input(
    const float* gradOutput,
    const float* weights,
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
    int strideW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInputSize = batch * inChannels * inHeight * inWidth;

    if (idx >= totalInputSize) return;

    // Decompose index to [b, ic, ih, iw]
    int b = idx / (inChannels * inHeight * inWidth);
    int rem1 = idx % (inChannels * inHeight * inWidth);
    int ic = rem1 / (inHeight * inWidth);
    int rem2 = rem1 % (inHeight * inWidth);
    int ih = rem2 / inWidth;
    int iw = rem2 % inWidth;

    float sumGrad = 0.0f;

    // Find all output positions this input contributes to
    for (int oh = 0; oh < outHeight; oh++) {
        for (int ow = 0; ow < outWidth; ow++) {
            // Check if input pixel (ih, iw) is in the kernel window for (oh, ow)
            int kh_rel = ih - oh * strideH;
            int kw_rel = iw - ow * strideW;

            if (kh_rel >= 0 && kh_rel < kernelH && kw_rel >= 0 && kw_rel < kernelW) {
                // This input pixel contributes to output at (oh, ow)
                for (int oc = 0; oc < outChannels; oc++) {
                    // Get gradient from output
                    float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

                    // Get weight for this connection
                    int weightIdx = ((((oh * outWidth + ow) * outChannels + oc) * inChannels + ic) * kernelH + kh_rel) * kernelW + kw_rel;
                    float weightVal = weights[weightIdx];

                    sumGrad += gradOutVal * weightVal;
                }
            }
        }
    }

    gradInput[idx] = sumGrad;
}

// Backward pass for weight gradients
// Computes gradient with respect to weights tensor
extern ""C"" __global__ void locally_connected_conv2d_backward_weights(
    const float* gradOutput,
    const float* input,
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
    int strideW)
{
    // Global thread indices for weight gradient
    // Weights shape: [outH, outW, outC, inC, kH, kW]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;

    if (idx >= totalWeights) return;

    // Decompose index to [oh, ow, oc, ic, kh, kw]
    int tmp = idx;
    int kw = tmp % kernelW; tmp /= kernelW;
    int kh = tmp % kernelH; tmp /= kernelH;
    int ic = tmp % inChannels; tmp /= inChannels;
    int oc = tmp % outChannels; tmp /= outChannels;
    int ow = tmp % outWidth; tmp /= outWidth;
    int oh = tmp;

    float sumGrad = 0.0f;

    // Sum over batch dimension
    for (int b = 0; b < batch; b++) {
        int ih = oh * strideH + kh;
        int iw = ow * strideW + kw;

        // Bounds check
        if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
            float gradOutVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
            float inputVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
            sumGrad += gradOutVal * inputVal;
        }
    }

    gradWeights[idx] = sumGrad;
}

// Backward pass for bias gradients
// Computes gradient with respect to bias tensor
// Output shape: [outC] (sum over batch and spatial dims)
extern ""C"" __global__ void locally_connected_conv2d_backward_bias(
    const float* gradOutput,
    float* gradBias,
    int batch,
    int outChannels,
    int outHeight,
    int outWidth)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= outChannels) return;

    float sumGrad = 0.0f;

    // Sum over batch and spatial dimensions
    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                sumGrad += gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
            }
        }
    }

    gradBias[oc] = sumGrad;
}

// Fused locally connected conv2d with activation
// Supports common activation functions fused into the kernel
extern ""C"" __global__ void locally_connected_conv2d_fused(
    const float* input,
    const float* weights,
    const float* bias,
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
    int hasBias,
    int activationType)  // 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int boc = blockIdx.z;

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    int b = boc / outChannels;
    int oc = boc % outChannels;

    float sum = 0.0f;

    int weightBaseIdx = ((oh * outWidth + ow) * outChannels + oc) * inChannels * kernelH * kernelW;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inputVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    int weightIdx = weightBaseIdx + (ic * kernelH + kh) * kernelW + kw;
                    float weightVal = weights[weightIdx];
                    sum += inputVal * weightVal;
                }
            }
        }
    }

    if (hasBias != 0) {
        sum += bias[oc];
    }

    // Apply activation function
    float result = sum;
    if (activationType == 1) {  // ReLU
        result = fmaxf(0.0f, sum);
    } else if (activationType == 2) {  // Sigmoid
        result = 1.0f / (1.0f + expf(-sum));
    } else if (activationType == 3) {  // Tanh
        result = tanhf(sum);
    }

    int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = result;
}
";
    }
}
