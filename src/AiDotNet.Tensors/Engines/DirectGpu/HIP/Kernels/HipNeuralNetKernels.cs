// Copyright (c) AiDotNet. All rights reserved.
// HIP neural network kernels - activation gradients, loss functions, optimizers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipNeuralNetKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

// ===========================================================================
// ACTIVATION GRADIENT KERNELS
// ===========================================================================

extern ""C"" __global__ void relu_backward(
    const float* gradOutput, const float* input, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
}

extern ""C"" __global__ void sigmoid_backward(
    const float* gradOutput, const float* output, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float s = output[idx];
    gradInput[idx] = gradOutput[idx] * s * (1.0f - s);
}

extern ""C"" __global__ void tanh_backward(
    const float* gradOutput, const float* output, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float t = output[idx];
    gradInput[idx] = gradOutput[idx] * (1.0f - t * t);
}

extern ""C"" __global__ void gelu_backward(
    const float* gradOutput, const float* input, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x = input[idx];
    float x3 = x * x * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    float tanhVal = tanhf(inner);
    float sech2 = 1.0f - tanhVal * tanhVal;
    float dinnerDx = sqrt2OverPi * (1.0f + 3.0f * coeff * x * x);
    float grad = 0.5f * (1.0f + tanhVal) + 0.5f * x * sech2 * dinnerDx;
    gradInput[idx] = gradOutput[idx] * grad;
}

extern ""C"" __global__ void softmax_backward(
    const float* gradOutput, const float* output, float* gradInput, int batchSize, int features)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    int baseIdx = batch * features;
    float dot = 0.0f;
    for (int f = 0; f < features; f++) {
        dot += gradOutput[baseIdx + f] * output[baseIdx + f];
    }
    for (int f = 0; f < features; f++) {
        int idx = baseIdx + f;
        gradInput[idx] = output[idx] * (gradOutput[idx] - dot);
    }
}

extern ""C"" __global__ void leaky_relu(
    const float* input, float* output, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x >= 0.0f ? x : alpha * x;
}

extern ""C"" __global__ void leaky_relu_backward(
    const float* gradOutput, const float* input, float* gradInput, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = input[idx] >= 0.0f ? gradOutput[idx] : alpha * gradOutput[idx];
}

extern ""C"" __global__ void elu(
    const float* input, float* output, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

extern ""C"" __global__ void elu_backward(
    const float* gradOutput, const float* input, const float* output, float* gradInput, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    gradInput[idx] = x >= 0.0f ? gradOutput[idx] : gradOutput[idx] * (output[idx] + alpha);
}

extern ""C"" __global__ void silu(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float sig = 1.0f / (1.0f + expf(-x));
    output[idx] = x * sig;
}

extern ""C"" __global__ void swish_backward(
    const float* gradOutput, const float* input, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float sig = 1.0f / (1.0f + expf(-x));
    float swishVal = x * sig;
    gradInput[idx] = gradOutput[idx] * (swishVal + sig * (1.0f - swishVal));
}

extern ""C"" __global__ void mish(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float sp = logf(1.0f + expf(x));
    output[idx] = x * tanhf(sp);
}

extern ""C"" __global__ void softplus(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = logf(1.0f + expf(input[idx]));
}

extern ""C"" __global__ void hardswish(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    output[idx] = x * relu6 / 6.0f;
}

// ===========================================================================
// LOSS FUNCTION KERNELS
// ===========================================================================

extern ""C"" __global__ void cross_entropy_loss(
    const float* predictions, const float* targets, float* loss,
    int batchSize, int numClasses)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    int baseIdx = batch * numClasses;
    float sampleLoss = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        float pred = fmaxf(predictions[baseIdx + c], 1e-7f);
        sampleLoss -= targets[baseIdx + c] * logf(pred);
    }
    loss[batch] = sampleLoss;
}

extern ""C"" __global__ void cross_entropy_backward(
    const float* predictions, const float* targets, float* gradInput,
    int batchSize, int numClasses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * numClasses;
    if (idx >= total) return;

    float pred = fmaxf(predictions[idx], 1e-7f);
    gradInput[idx] = (-targets[idx] / pred) / (float)batchSize;
}

extern ""C"" __global__ void bce_loss(
    const float* predictions, const float* targets, float* loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float p = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
    float t = targets[idx];
    loss[idx] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
}

extern ""C"" __global__ void bce_backward(
    const float* predictions, const float* targets, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float p = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
    float t = targets[idx];
    gradInput[idx] = (p - t) / (p * (1.0f - p) * (float)size);
}

extern ""C"" __global__ void mse_loss(
    const float* predictions, const float* targets, float* loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    loss[idx] = diff * diff;
}

extern ""C"" __global__ void mse_backward(
    const float* predictions, const float* targets, float* gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    gradInput[idx] = 2.0f * (predictions[idx] - targets[idx]) / (float)size;
}

extern ""C"" __global__ void smooth_l1_loss(
    const float* predictions, const float* targets, float* loss,
    int size, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float diff = fabsf(predictions[idx] - targets[idx]);
    loss[idx] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
}

extern ""C"" __global__ void smooth_l1_backward(
    const float* predictions, const float* targets, float* gradInput,
    int size, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    float absDiff = fabsf(diff);
    float grad;
    if (absDiff < beta) {
        grad = diff / beta;
    } else {
        grad = diff > 0.0f ? 1.0f : -1.0f;
    }
    gradInput[idx] = grad / (float)size;
}

// ===========================================================================
// UTILITY KERNELS
// ===========================================================================

extern ""C"" __global__ void clamp(
    const float* input, float* output, float minVal, float maxVal, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fmaxf(fminf(input[idx], maxVal), minVal);
}

extern ""C"" __global__ void l2_norm_squared(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x * x;
}

extern ""C"" __global__ void scale(
    const float* input, float* output, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] * scalar;
}

extern ""C"" __global__ void copy_buffer(const float* src, float* dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = src[idx];
}

// ===========================================================================
// COMPARISON KERNELS
// ===========================================================================

extern ""C"" __global__ void greater_than(
    const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] > B[idx] ? 1.0f : 0.0f;
}

extern ""C"" __global__ void less_than(
    const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] < B[idx] ? 1.0f : 0.0f;
}

extern ""C"" __global__ void equals(
    const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = fabsf(A[idx] - B[idx]) < 1e-6f ? 1.0f : 0.0f;
}

extern ""C"" __global__ void where_cond(
    const float* condition, const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = condition[idx] != 0.0f ? A[idx] : B[idx];
}

// ===========================================================================
// STATISTICS KERNELS
// ===========================================================================

extern ""C"" __global__ void compute_mean_var(
    const float* input, float* mean, float* variance, int batchSize, int features)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= features) return;

    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        sum += input[b * features + f];
    }
    float m = sum / (float)batchSize;
    mean[f] = m;

    float varSum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        float diff = input[b * features + f] - m;
        varSum += diff * diff;
    }
    variance[f] = varSum / (float)batchSize;
}

extern ""C"" __global__ void argmax_axis(
    const float* input, float* indices, int outerSize, int axisSize)
{
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerSize) return;

    int baseIdx = outer * axisSize;
    float maxVal = input[baseIdx];
    int maxIdx = 0;
    for (int i = 1; i < axisSize; i++) {
        float val = input[baseIdx + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    indices[outer] = (float)maxIdx;
}

extern ""C"" __global__ void argmin_axis(
    const float* input, float* indices, int outerSize, int axisSize)
{
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerSize) return;

    int baseIdx = outer * axisSize;
    float minVal = input[baseIdx];
    int minIdx = 0;
    for (int i = 1; i < axisSize; i++) {
        float val = input[baseIdx + i];
        if (val < minVal) {
            minVal = val;
            minIdx = i;
        }
    }
    indices[outer] = (float)minIdx;
}

// ===========================================================================
// OPTIMIZER KERNELS
// ===========================================================================

extern ""C"" __global__ void sgd_step(
    float* param, const float* gradient, float* velocity,
    float learningRate, float momentum, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;
    param[idx] -= learningRate * v;
}

extern ""C"" __global__ void adam_step(
    float* param, const float* gradient, float* m, float* v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int t, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / (1.0f - powf(beta1, (float)t));
    float vHat = vVal / (1.0f - powf(beta2, (float)t));

    param[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

extern ""C"" __global__ void adamw_step(
    float* param, const float* gradient, float* m, float* v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int t, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / (1.0f - powf(beta1, (float)t));
    float vHat = vVal / (1.0f - powf(beta2, (float)t));

    // AdamW: decoupled weight decay
    param[idx] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[idx]);
}

// ===========================================================================
// DROPOUT AND EMBEDDING KERNELS
// ===========================================================================

extern ""C"" __global__ void dropout_forward(
    const float* input, float* output, const float* mask,
    float scale, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] * mask[idx] * scale;
}

extern ""C"" __global__ void dropout_backward(
    const float* gradOutput, const float* mask, float* gradInput,
    float scale, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = gradOutput[idx] * mask[idx] * scale;
}

extern ""C"" __global__ void embedding_forward(
    const float* indices, const float* embeddingTable, float* output,
    int numIndices, int embeddingDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numIndices) return;

    int idx = (int)indices[i];
    for (int d = 0; d < embeddingDim; d++) {
        output[i * embeddingDim + d] = embeddingTable[idx * embeddingDim + d];
    }
}

extern ""C"" __global__ void embedding_backward(
    const float* gradOutput, const float* indices, float* gradEmbedding,
    int numIndices, int embeddingDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numIndices) return;

    int idx = (int)indices[i];
    for (int d = 0; d < embeddingDim; d++) {
        atomicAdd(&gradEmbedding[idx * embeddingDim + d], gradOutput[i * embeddingDim + d]);
    }
}

// ===========================================================================
// TRANSPOSE KERNELS
// ===========================================================================

extern ""C"" __global__ void transpose_2d(
    const float* A, float* B, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    B[col * rows + row] = A[row * cols + col];
}

extern ""C"" __global__ void batched_transpose(
    const float* A, float* B, int batch, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (col >= cols || row >= rows || b >= batch) return;

    int inIdx = b * rows * cols + row * cols + col;
    int outIdx = b * cols * rows + col * rows + row;
    B[outIdx] = A[inIdx];
}

extern ""C"" __global__ void permute_general(
    const float* input, float* output,
    const int* inputStrides, const int* outputStrides, const int* permutation,
    int ndims, int totalSize)
{
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outIdx >= totalSize) return;

    int remaining = outIdx;
    int inputIdx = 0;
    int outCoords[8];

    for (int d = 0; d < ndims; d++)
    {
        outCoords[d] = remaining / outputStrides[d];
        remaining = remaining % outputStrides[d];
    }

    for (int d = 0; d < ndims; d++)
    {
        int inputDim = permutation[d];
        inputIdx += outCoords[d] * inputStrides[inputDim];
    }

    output[outIdx] = input[inputIdx];
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "relu_backward", "sigmoid_backward", "tanh_backward", "gelu_backward",
            "softmax_backward", "leaky_relu", "leaky_relu_backward",
            "elu", "elu_backward", "silu", "swish_backward", "mish", "softplus", "hardswish",
            "cross_entropy_loss", "cross_entropy_backward", "bce_loss", "bce_backward",
            "mse_loss", "mse_backward", "smooth_l1_loss", "smooth_l1_backward",
            "clamp", "l2_norm_squared", "scale", "copy_buffer",
            "greater_than", "less_than", "equals", "where_cond",
            "compute_mean_var", "argmax_axis", "argmin_axis",
            "sgd_step", "adam_step", "adamw_step",
            "dropout_forward", "dropout_backward", "embedding_forward", "embedding_backward",
            "transpose_2d", "batched_transpose", "permute_general"
        };
    }
}
