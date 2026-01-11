// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for ConvLSTM (Convolutional LSTM) neural network operations.
// Implements spatiotemporal sequence learning with convolutional gates.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for ConvLSTM operations used in spatiotemporal neural networks.
/// Implements full forward and backward passes for ConvLSTM cells with 4 gates.
/// </summary>
internal static class HipConvLSTMKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define EPSILON 1e-15f

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

__device__ float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// CONVLSTM CELL FORWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void convlstm_cell_forward(
    const float* input,
    const float* prevH,
    const float* prevC,
    const float* weightsFi, const float* weightsIi, const float* weightsCi, const float* weightsOi,
    const float* weightsFh, const float* weightsIh, const float* weightsCh, const float* weightsOh,
    const float* biasF, const float* biasI, const float* biasC, const float* biasO,
    float* newH,
    float* newC,
    float* gateF, float* gateI, float* gateC, float* gateO,
    int batch, int inputChannels, int hiddenChannels,
    int height, int width, int kernelH, int kernelW, int pad)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenChannels * height * width;

    if (gid >= totalElements) return;

    int b = gid / (hiddenChannels * height * width);
    int rem1 = gid % (hiddenChannels * height * width);
    int hc = rem1 / (height * width);
    int rem2 = rem1 % (height * width);
    int h = rem2 / width;
    int w = rem2 % width;

    float sumF = biasF[hc];
    float sumI = biasI[hc];
    float sumC = biasC[hc];
    float sumO = biasO[hc];

    for (int ic = 0; ic < inputChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = h - pad + kh;
                int iw = w - pad + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float inVal = input[((b * inputChannels + ic) * height + ih) * width + iw];
                    int wIdx = ((hc * inputChannels + ic) * kernelH + kh) * kernelW + kw;

                    sumF += inVal * weightsFi[wIdx];
                    sumI += inVal * weightsIi[wIdx];
                    sumC += inVal * weightsCi[wIdx];
                    sumO += inVal * weightsOi[wIdx];
                }
            }
        }
    }

    for (int hc2 = 0; hc2 < hiddenChannels; hc2++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = h - pad + kh;
                int iw = w - pad + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float hVal = prevH[((b * hiddenChannels + hc2) * height + ih) * width + iw];
                    int wIdx = ((hc * hiddenChannels + hc2) * kernelH + kh) * kernelW + kw;

                    sumF += hVal * weightsFh[wIdx];
                    sumI += hVal * weightsIh[wIdx];
                    sumC += hVal * weightsCh[wIdx];
                    sumO += hVal * weightsOh[wIdx];
                }
            }
        }
    }

    float f = sigmoid(sumF);
    float i = sigmoid(sumI);
    float c = tanhf(sumC);
    float o = sigmoid(sumO);

    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i * c;
    float newHVal = o * tanhf(newCVal);

    newC[gid] = newCVal;
    newH[gid] = newHVal;
    gateF[gid] = f;
    gateI[gid] = i;
    gateC[gid] = c;
    gateO[gid] = o;
}

// ===========================================================================
// CONVLSTM CELL BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void convlstm_cell_backward_input(
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    const float* weightsFi, const float* weightsIi, const float* weightsCi, const float* weightsOi,
    float* gradInput,
    int batch, int inputChannels, int hiddenChannels,
    int height, int width, int kernelH, int kernelW, int pad)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * inputChannels * height * width;

    if (gid >= totalElements) return;

    int b = gid / (inputChannels * height * width);
    int rem1 = gid % (inputChannels * height * width);
    int ic = rem1 / (height * width);
    int rem2 = rem1 % (height * width);
    int h = rem2 / width;
    int w = rem2 % width;

    float gradSum = 0.0f;

    for (int hc = 0; hc < hiddenChannels; hc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh = h + pad - kh;
                int ow = w + pad - kw;

                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int outIdx = ((b * hiddenChannels + hc) * height + oh) * width + ow;

                    float f = gateF[outIdx];
                    float i = gateI[outIdx];
                    float c = gateC[outIdx];
                    float o = gateO[outIdx];
                    float prevCVal = prevC[outIdx];
                    float newCVal = newC[outIdx];

                    float dH = gradH[outIdx];
                    float tanhNewC = tanhf(newCVal);
                    float dO = dH * tanhNewC * sigmoid_derivative(o);
                    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
                    float dC = gradC[outIdx] + dC_from_H;
                    float dF = dC * prevCVal * sigmoid_derivative(f);
                    float dI = dC * c * sigmoid_derivative(i);
                    float dCGate = dC * i * tanh_derivative(c);

                    int wIdx = ((hc * inputChannels + ic) * kernelH + kh) * kernelW + kw;

                    gradSum += dF * weightsFi[wIdx];
                    gradSum += dI * weightsIi[wIdx];
                    gradSum += dCGate * weightsCi[wIdx];
                    gradSum += dO * weightsOi[wIdx];
                }
            }
        }
    }

    gradInput[gid] = gradSum;
}

extern ""C"" __global__ void convlstm_cell_backward_prevh(
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    const float* weightsFh, const float* weightsIh, const float* weightsCh, const float* weightsOh,
    float* gradPrevH,
    int batch, int hiddenChannels,
    int height, int width, int kernelH, int kernelW, int pad)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenChannels * height * width;

    if (gid >= totalElements) return;

    int b = gid / (hiddenChannels * height * width);
    int rem1 = gid % (hiddenChannels * height * width);
    int hc2 = rem1 / (height * width);
    int rem2 = rem1 % (height * width);
    int h = rem2 / width;
    int w = rem2 % width;

    float gradSum = 0.0f;

    for (int hc = 0; hc < hiddenChannels; hc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh = h + pad - kh;
                int ow = w + pad - kw;

                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int outIdx = ((b * hiddenChannels + hc) * height + oh) * width + ow;

                    float f = gateF[outIdx];
                    float i = gateI[outIdx];
                    float c = gateC[outIdx];
                    float o = gateO[outIdx];
                    float prevCVal = prevC[outIdx];
                    float newCVal = newC[outIdx];

                    float dH = gradH[outIdx];
                    float tanhNewC = tanhf(newCVal);
                    float dO = dH * tanhNewC * sigmoid_derivative(o);
                    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
                    float dC = gradC[outIdx] + dC_from_H;
                    float dF = dC * prevCVal * sigmoid_derivative(f);
                    float dI = dC * c * sigmoid_derivative(i);
                    float dCGate = dC * i * tanh_derivative(c);

                    int wIdx = ((hc * hiddenChannels + hc2) * kernelH + kh) * kernelW + kw;

                    gradSum += dF * weightsFh[wIdx];
                    gradSum += dI * weightsIh[wIdx];
                    gradSum += dCGate * weightsCh[wIdx];
                    gradSum += dO * weightsOh[wIdx];
                }
            }
        }
    }

    gradPrevH[gid] = gradSum;
}

extern ""C"" __global__ void convlstm_cell_backward_prevc(
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateO,
    const float* newC,
    float* gradPrevC,
    int batch, int hiddenChannels, int height, int width)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenChannels * height * width;

    if (gid >= totalElements) return;

    float f = gateF[gid];
    float o = gateO[gid];
    float newCVal = newC[gid];
    float tanhNewC = tanhf(newCVal);

    float dH = gradH[gid];
    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
    float dC = gradC[gid] + dC_from_H;

    gradPrevC[gid] = dC * f;
}

extern ""C"" __global__ void convlstm_cell_backward_weights_input(
    const float* input,
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    float* gradWeightsFi, float* gradWeightsIi, float* gradWeightsCi, float* gradWeightsOi,
    int batch, int inputChannels, int hiddenChannels,
    int height, int width, int kernelH, int kernelW, int pad)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = hiddenChannels * inputChannels * kernelH * kernelW;

    if (gid >= totalWeights) return;

    int hc = gid / (inputChannels * kernelH * kernelW);
    int rem1 = gid % (inputChannels * kernelH * kernelW);
    int ic = rem1 / (kernelH * kernelW);
    int rem2 = rem1 % (kernelH * kernelW);
    int kh = rem2 / kernelW;
    int kw = rem2 % kernelW;

    float gradF = 0.0f, gradI = 0.0f, gradC_ = 0.0f, gradO = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                int ih = oh - pad + kh;
                int iw = ow - pad + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int outIdx = ((b * hiddenChannels + hc) * height + oh) * width + ow;
                    int inIdx = ((b * inputChannels + ic) * height + ih) * width + iw;

                    float inputVal = input[inIdx];
                    float f = gateF[outIdx];
                    float i = gateI[outIdx];
                    float c = gateC[outIdx];
                    float o = gateO[outIdx];
                    float prevCVal = prevC[outIdx];
                    float newCVal = newC[outIdx];

                    float dH = gradH[outIdx];
                    float tanhNewC = tanhf(newCVal);
                    float dO_val = dH * tanhNewC * sigmoid_derivative(o);
                    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
                    float dC = gradC[outIdx] + dC_from_H;
                    float dF_val = dC * prevCVal * sigmoid_derivative(f);
                    float dI_val = dC * c * sigmoid_derivative(i);
                    float dC_val = dC * i * tanh_derivative(c);

                    gradF += dF_val * inputVal;
                    gradI += dI_val * inputVal;
                    gradC_ += dC_val * inputVal;
                    gradO += dO_val * inputVal;
                }
            }
        }
    }

    gradWeightsFi[gid] = gradF;
    gradWeightsIi[gid] = gradI;
    gradWeightsCi[gid] = gradC_;
    gradWeightsOi[gid] = gradO;
}

extern ""C"" __global__ void convlstm_cell_backward_weights_hidden(
    const float* prevH,
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    float* gradWeightsFh, float* gradWeightsIh, float* gradWeightsCh, float* gradWeightsOh,
    int batch, int hiddenChannels,
    int height, int width, int kernelH, int kernelW, int pad)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = hiddenChannels * hiddenChannels * kernelH * kernelW;

    if (gid >= totalWeights) return;

    int hc = gid / (hiddenChannels * kernelH * kernelW);
    int rem1 = gid % (hiddenChannels * kernelH * kernelW);
    int hc2 = rem1 / (kernelH * kernelW);
    int rem2 = rem1 % (kernelH * kernelW);
    int kh = rem2 / kernelW;
    int kw = rem2 % kernelW;

    float gradF = 0.0f, gradI = 0.0f, gradC_ = 0.0f, gradO = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                int ih = oh - pad + kh;
                int iw = ow - pad + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int outIdx = ((b * hiddenChannels + hc) * height + oh) * width + ow;
                    int hIdx = ((b * hiddenChannels + hc2) * height + ih) * width + iw;

                    float hVal = prevH[hIdx];
                    float f = gateF[outIdx];
                    float i = gateI[outIdx];
                    float c = gateC[outIdx];
                    float o = gateO[outIdx];
                    float prevCVal = prevC[outIdx];
                    float newCVal = newC[outIdx];

                    float dH = gradH[outIdx];
                    float tanhNewC = tanhf(newCVal);
                    float dO_val = dH * tanhNewC * sigmoid_derivative(o);
                    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
                    float dC = gradC[outIdx] + dC_from_H;
                    float dF_val = dC * prevCVal * sigmoid_derivative(f);
                    float dI_val = dC * c * sigmoid_derivative(i);
                    float dC_val = dC * i * tanh_derivative(c);

                    gradF += dF_val * hVal;
                    gradI += dI_val * hVal;
                    gradC_ += dC_val * hVal;
                    gradO += dO_val * hVal;
                }
            }
        }
    }

    gradWeightsFh[gid] = gradF;
    gradWeightsIh[gid] = gradI;
    gradWeightsCh[gid] = gradC_;
    gradWeightsOh[gid] = gradO;
}

extern ""C"" __global__ void convlstm_cell_backward_biases(
    const float* gradH,
    const float* gradC,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    float* gradBiasF, float* gradBiasI, float* gradBiasC, float* gradBiasO,
    int batch, int hiddenChannels, int height, int width)
{
    int hc = blockIdx.x * blockDim.x + threadIdx.x;

    if (hc >= hiddenChannels) return;

    float gradF = 0.0f, gradI = 0.0f, gradC_ = 0.0f, gradO = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((b * hiddenChannels + hc) * height + h) * width + w;

                float f = gateF[idx];
                float i = gateI[idx];
                float c = gateC[idx];
                float o = gateO[idx];
                float prevCVal = prevC[idx];
                float newCVal = newC[idx];

                float dH = gradH[idx];
                float tanhNewC = tanhf(newCVal);
                float dO_val = dH * tanhNewC * sigmoid_derivative(o);
                float dC_from_H = dH * o * tanh_derivative(tanhNewC);
                float dC = gradC[idx] + dC_from_H;
                float dF_val = dC * prevCVal * sigmoid_derivative(f);
                float dI_val = dC * c * sigmoid_derivative(i);
                float dC_val = dC * i * tanh_derivative(c);

                gradF += dF_val;
                gradI += dI_val;
                gradC_ += dC_val;
                gradO += dO_val;
            }
        }
    }

    gradBiasF[hc] = gradF;
    gradBiasI[hc] = gradI;
    gradBiasC[hc] = gradC_;
    gradBiasO[hc] = gradO;
}
";
    }

    /// <summary>
    /// Gets the list of kernel names provided by this source.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "convlstm_cell_forward",
            "convlstm_cell_backward_input",
            "convlstm_cell_backward_prevh",
            "convlstm_cell_backward_prevc",
            "convlstm_cell_backward_weights_input",
            "convlstm_cell_backward_weights_hidden",
            "convlstm_cell_backward_biases"
        };
    }
}
