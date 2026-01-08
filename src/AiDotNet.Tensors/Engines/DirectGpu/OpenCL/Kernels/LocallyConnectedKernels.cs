// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for locally connected convolution operations.
// Unlike standard convolution, locally connected uses unique weights per spatial position.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL kernels for locally connected convolution operations.
    /// </summary>
    /// <remarks>
    /// Locally connected layers differ from standard convolution in that each spatial
    /// position has its own unique set of weights. This increases parameter count but
    /// allows the network to learn location-specific features.
    /// </remarks>
    internal static class LocallyConnectedKernels
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
// ===========================================================================
// LOCALLY CONNECTED CONVOLUTION KERNELS
// ===========================================================================

// Forward pass for locally connected 2D convolution
// Each output spatial position uses a unique set of weights
// Input:   [batch, inChannels, inH, inW]
// Weights: [outH, outW, outC, inC, kH, kW]
// Bias:    [outC] (optional)
// Output:  [batch, outC, outH, outW]
__kernel void locally_connected_conv2d(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
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
    const int hasBias)
{
    // Global thread indices
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int boc = get_global_id(2);

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    const int b = boc / outChannels;
    const int oc = boc % outChannels;

    // Compute weighted sum using position-specific weights
    float sum = 0.0f;

    // Weight offset for this output position: [oh, ow, oc, :, :, :]
    // Layout: [outH, outW, outC, inC, kH, kW]
    const int weightBaseIdx = ((oh * outWidth + ow) * outChannels + oc) * inChannels * kernelH * kernelW;

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
    const int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = sum;
}

// Backward pass for input gradients
// Computes gradient with respect to input tensor
__kernel void locally_connected_conv2d_backward_input(
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
    const int strideW)
{
    // Global thread indices for input gradient
    const int idx = get_global_id(0);
    const int totalInputSize = batch * inChannels * inHeight * inWidth;

    if (idx >= totalInputSize) return;

    // Decompose index to [b, ic, ih, iw]
    const int b = idx / (inChannels * inHeight * inWidth);
    const int rem1 = idx % (inChannels * inHeight * inWidth);
    const int ic = rem1 / (inHeight * inWidth);
    const int rem2 = rem1 % (inHeight * inWidth);
    const int ih = rem2 / inWidth;
    const int iw = rem2 % inWidth;

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
__kernel void locally_connected_conv2d_backward_weights(
    __global const float* gradOutput,
    __global const float* input,
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
    const int strideW)
{
    // Global thread indices for weight gradient
    // Weights shape: [outH, outW, outC, inC, kH, kW]
    const int idx = get_global_id(0);
    const int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;

    if (idx >= totalWeights) return;

    // Decompose index to [oh, ow, oc, ic, kh, kw]
    int tmp = idx;
    const int kw = tmp % kernelW; tmp /= kernelW;
    const int kh = tmp % kernelH; tmp /= kernelH;
    const int ic = tmp % inChannels; tmp /= inChannels;
    const int oc = tmp % outChannels; tmp /= outChannels;
    const int ow = tmp % outWidth; tmp /= outWidth;
    const int oh = tmp;

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
__kernel void locally_connected_conv2d_backward_bias(
    __global const float* gradOutput,
    __global float* gradBias,
    const int batch,
    const int outChannels,
    const int outHeight,
    const int outWidth)
{
    const int oc = get_global_id(0);

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
__kernel void locally_connected_conv2d_fused(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
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
    const int hasBias,
    const int activationType)  // 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int boc = get_global_id(2);

    if (ow >= outWidth || oh >= outHeight || boc >= batch * outChannels) return;

    const int b = boc / outChannels;
    const int oc = boc % outChannels;

    float sum = 0.0f;

    const int weightBaseIdx = ((oh * outWidth + ow) * outChannels + oc) * inChannels * kernelH * kernelW;

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
        result = fmax(0.0f, sum);
    } else if (activationType == 2) {  // Sigmoid
        result = 1.0f / (1.0f + exp(-sum));
    } else if (activationType == 3) {  // Tanh
        result = tanh(sum);
    }

    const int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
    output[outIdx] = result;
}
";
        }
    }
}
