// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for ConvLSTM (Convolutional LSTM) neural network operations.
// Implements spatiotemporal sequence learning with convolutional gates.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for ConvLSTM operations used in spatiotemporal neural networks.
/// Implements full forward and backward passes for ConvLSTM cells with 4 gates.
/// </summary>
internal static class ConvLSTMKernels
{
    public static string GetSource()
    {
        return @"
#define EPSILON 1e-15f

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

inline float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

inline float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// CONVLSTM CELL FORWARD KERNEL
// ===========================================================================

__kernel void convlstm_cell_forward(
    __global const float* input,
    __global const float* prevH,
    __global const float* prevC,
    __global const float* weightsFi, __global const float* weightsIi,
    __global const float* weightsCi, __global const float* weightsOi,
    __global const float* weightsFh, __global const float* weightsIh,
    __global const float* weightsCh, __global const float* weightsOh,
    __global const float* biasF, __global const float* biasI,
    __global const float* biasC, __global const float* biasO,
    __global float* newH,
    __global float* newC,
    __global float* gateF, __global float* gateI,
    __global float* gateC, __global float* gateO,
    const int batch, const int inputChannels, const int hiddenChannels,
    const int height, const int width, const int kernelH, const int kernelW, const int pad)
{
    int gid = get_global_id(0);
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

    float f = sigmoid_fn(sumF);
    float i = sigmoid_fn(sumI);
    float c = tanh(sumC);
    float o = sigmoid_fn(sumO);

    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i * c;
    float newHVal = o * tanh(newCVal);

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

__kernel void convlstm_cell_backward_input(
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateI,
    __global const float* gateC, __global const float* gateO,
    __global const float* prevC, __global const float* newC,
    __global const float* weightsFi, __global const float* weightsIi,
    __global const float* weightsCi, __global const float* weightsOi,
    __global float* gradInput,
    const int batch, const int inputChannels, const int hiddenChannels,
    const int height, const int width, const int kernelH, const int kernelW, const int pad)
{
    int gid = get_global_id(0);
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
                    float tanhNewC = tanh(newCVal);
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

__kernel void convlstm_cell_backward_prevh(
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateI,
    __global const float* gateC, __global const float* gateO,
    __global const float* prevC, __global const float* newC,
    __global const float* weightsFh, __global const float* weightsIh,
    __global const float* weightsCh, __global const float* weightsOh,
    __global float* gradPrevH,
    const int batch, const int hiddenChannels,
    const int height, const int width, const int kernelH, const int kernelW, const int pad)
{
    int gid = get_global_id(0);
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
                    float tanhNewC = tanh(newCVal);
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

__kernel void convlstm_cell_backward_prevc(
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateO,
    __global const float* newC,
    __global float* gradPrevC,
    const int batch, const int hiddenChannels, const int height, const int width)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenChannels * height * width;

    if (gid >= totalElements) return;

    float f = gateF[gid];
    float o = gateO[gid];
    float newCVal = newC[gid];
    float tanhNewC = tanh(newCVal);

    float dH = gradH[gid];
    float dC_from_H = dH * o * tanh_derivative(tanhNewC);
    float dC = gradC[gid] + dC_from_H;

    gradPrevC[gid] = dC * f;
}

__kernel void convlstm_cell_backward_weights_input(
    __global const float* input,
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateI,
    __global const float* gateC, __global const float* gateO,
    __global const float* prevC, __global const float* newC,
    __global float* gradWeightsFi, __global float* gradWeightsIi,
    __global float* gradWeightsCi, __global float* gradWeightsOi,
    const int batch, const int inputChannels, const int hiddenChannels,
    const int height, const int width, const int kernelH, const int kernelW, const int pad)
{
    int gid = get_global_id(0);
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
                    float tanhNewC = tanh(newCVal);
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

__kernel void convlstm_cell_backward_weights_hidden(
    __global const float* prevH,
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateI,
    __global const float* gateC, __global const float* gateO,
    __global const float* prevC, __global const float* newC,
    __global float* gradWeightsFh, __global float* gradWeightsIh,
    __global float* gradWeightsCh, __global float* gradWeightsOh,
    const int batch, const int hiddenChannels,
    const int height, const int width, const int kernelH, const int kernelW, const int pad)
{
    int gid = get_global_id(0);
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
                    float tanhNewC = tanh(newCVal);
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

__kernel void convlstm_cell_backward_biases(
    __global const float* gradH,
    __global const float* gradC,
    __global const float* gateF, __global const float* gateI,
    __global const float* gateC, __global const float* gateO,
    __global const float* prevC, __global const float* newC,
    __global float* gradBiasF, __global float* gradBiasI,
    __global float* gradBiasC, __global float* gradBiasO,
    const int batch, const int hiddenChannels, const int height, const int width)
{
    int hc = get_global_id(0);

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
                float tanhNewC = tanh(newCVal);
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
