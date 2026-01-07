// Copyright (c) AiDotNet. All rights reserved.
// CUDA neural network kernels - activation gradients, loss functions, optimizers.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for neural network operations including activation gradients,
    /// loss functions, and optimizer step updates.
    /// </summary>
    internal static class CudaNeuralNetKernels
    {
        public static string GetSource()
        {
            return @"
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

extern ""C"" __global__ void not_equal_scalar(
    const float* A, float* C, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = fabsf(A[idx] - scalar) >= 1e-6f ? 1.0f : 0.0f;
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

// Mean reduction along axis: output[i] = mean(input[i, :])
extern ""C"" __global__ void mean_axis(
    const float* input, float* output, int outerSize, int reduceSize)
{
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerSize) return;

    int baseIdx = outer * reduceSize;
    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        sum += input[baseIdx + i];
    }
    output[outer] = sum / (float)reduceSize;
}

// Max reduction along axis: output[i] = max(input[i, :])
extern ""C"" __global__ void max_axis(
    const float* input, float* output, int outerSize, int reduceSize)
{
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerSize) return;

    int baseIdx = outer * reduceSize;
    float maxVal = input[baseIdx];
    for (int i = 1; i < reduceSize; i++) {
        float val = input[baseIdx + i];
        if (val > maxVal) maxVal = val;
    }
    output[outer] = maxVal;
}

// Variance reduction along axis: output[i] = var(input[i, :])
extern ""C"" __global__ void var_axis(
    const float* input, const float* mean, float* variance, int outerSize, int reduceSize)
{
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerSize) return;

    int baseIdx = outer * reduceSize;
    float m = mean[outer];
    float varSum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        float diff = input[baseIdx + i] - m;
        varSum += diff * diff;
    }
    variance[outer] = varSum / (float)reduceSize;
}

// ===========================================================================
// BROADCAST OPERATIONS
// ===========================================================================

// Broadcast multiply: C = A * B where B is broadcast along last axis
// A has shape (outerSize * innerSize), B has shape (innerSize), C has shape (outerSize * innerSize)
// output[i * innerSize + j] = input[i * innerSize + j] * broadcast[j]
extern ""C"" __global__ void broadcast_multiply_last_axis(
    const float* input, const float* broadcast, float* output,
    int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * innerSize;
    if (idx >= totalSize) return;

    int innerIdx = idx % innerSize;
    output[idx] = input[idx] * broadcast[innerIdx];
}

// Broadcast multiply: C = A * B where B is broadcast along first axis
// A has shape (outerSize * innerSize), B has shape (outerSize), C has shape (outerSize * innerSize)
// output[i * innerSize + j] = input[i * innerSize + j] * broadcast[i]
extern ""C"" __global__ void broadcast_multiply_first_axis(
    const float* input, const float* broadcast, float* output,
    int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * innerSize;
    if (idx >= totalSize) return;

    int outerIdx = idx / innerSize;
    output[idx] = input[idx] * broadcast[outerIdx];
}

// General broadcast multiply for tensors with compatible shapes
// Uses strides to handle arbitrary broadcasting patterns
// aStrides and bStrides specify how to map output index to input indices
extern ""C"" __global__ void broadcast_multiply_general(
    const float* A, const float* B, float* C,
    const int* aStrides, const int* bStrides, const int* cShape,
    int rank, int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    // Convert flat index to multi-dimensional index
    int aIdx = 0;
    int bIdx = 0;
    int remaining = idx;

    for (int d = rank - 1; d >= 0; d--) {
        int dimIdx = remaining % cShape[d];
        remaining /= cShape[d];
        aIdx += dimIdx * aStrides[d];
        bIdx += dimIdx * bStrides[d];
    }

    C[idx] = A[aIdx] * B[bIdx];
}

// ===========================================================================
// CAPSULE NETWORK OPERATIONS
// ===========================================================================

// Squash activation for capsule networks
// squash(v) = ||v||^2 / (1 + ||v||^2) * v / ||v||
// Applied per capsule where each capsule is a vector of length capsuleDim
// Input shape: (numCapsules, capsuleDim), output shape: same
extern ""C"" __global__ void squash(
    const float* input, float* output,
    int numCapsules, int capsuleDim, float epsilon)
{
    int capsuleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsuleIdx >= numCapsules) return;

    int baseIdx = capsuleIdx * capsuleDim;

    // Compute squared norm of this capsule
    float normSquared = 0.0f;
    for (int i = 0; i < capsuleDim; i++) {
        float val = input[baseIdx + i];
        normSquared += val * val;
    }

    // Compute scale: ||v||^2 / (1 + ||v||^2) / ||v||
    // = ||v|| / (1 + ||v||^2)
    float norm = sqrtf(normSquared + epsilon);
    float scale = normSquared / ((1.0f + normSquared) * norm);

    // Apply scaling to each element
    for (int i = 0; i < capsuleDim; i++) {
        output[baseIdx + i] = input[baseIdx + i] * scale;
    }
}

// Squash backward for gradient computation
extern ""C"" __global__ void squash_backward(
    const float* gradOutput, const float* input, float* gradInput,
    int numCapsules, int capsuleDim, float epsilon)
{
    int capsuleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsuleIdx >= numCapsules) return;

    int baseIdx = capsuleIdx * capsuleDim;

    // Compute squared norm
    float normSquared = 0.0f;
    for (int i = 0; i < capsuleDim; i++) {
        float val = input[baseIdx + i];
        normSquared += val * val;
    }

    // Simplified gradient: scale by 1/(1 + ||v||^2)
    float scale = 1.0f / (1.0f + normSquared);
    for (int i = 0; i < capsuleDim; i++) {
        gradInput[baseIdx + i] = gradOutput[baseIdx + i] * scale;
    }
}

// ===========================================================================
// TILE/REPEAT KERNELS
// ===========================================================================

// Tile tensor along batch dimension (axis 0)
// Input shape: [1, innerSize], Output shape: [repeats, innerSize]
extern ""C"" __global__ void tile_batch(
    const float* input, float* output,
    int repeats, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = repeats * innerSize;
    if (idx >= totalSize) return;

    int innerIdx = idx % innerSize;
    output[idx] = input[innerIdx];
}

// General tile along any axis
// For tiling [d0, d1, ..., dn] by factor R along axis A:
// Output: [d0, ..., d_{A-1}, d_A * R, d_{A+1}, ..., dn]
// outerSize = product of dimensions before axis
// axisSize = dimension at axis (original)
// innerSize = product of dimensions after axis
extern ""C"" __global__ void tile_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * axisSize * repeats * innerSize;
    if (idx >= totalSize) return;

    // Decompose output index
    int outputAxisSize = axisSize * repeats;
    int innerIdx = idx % innerSize;
    int temp = idx / innerSize;
    int outputAxisIdx = temp % outputAxisSize;
    int outerIdx = temp / outputAxisSize;

    // Map to input index (mod to handle repeat)
    int inputAxisIdx = outputAxisIdx % axisSize;
    int inputIdx = outerIdx * axisSize * innerSize + inputAxisIdx * innerSize + innerIdx;

    output[idx] = input[inputIdx];
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
        // Atomic add for thread safety
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

// General permute for arbitrary axis permutations
// Supports up to 8 dimensions
extern ""C"" __global__ void permute_general(
    const float* input, float* output,
    const int* inputStrides, const int* outputStrides, const int* permutation,
    int ndims, int totalSize)
{
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outIdx >= totalSize) return;

    // Compute multi-dimensional output indices
    int remaining = outIdx;
    int inputIdx = 0;

    // Unroll for common dimensions (up to 8)
    int outCoords[8];
    for (int d = 0; d < ndims; d++)
    {
        outCoords[d] = remaining / outputStrides[d];
        remaining = remaining % outputStrides[d];
    }

    // Apply inverse permutation to get input coordinates and compute input index
    for (int d = 0; d < ndims; d++)
    {
        int inputDim = permutation[d];
        inputIdx += outCoords[d] * inputStrides[inputDim];
    }

    output[outIdx] = input[inputIdx];
}

// ===========================================================================
// SPECIALIZED LAYER KERNELS (RBF, Spiking NN)
// ===========================================================================

extern ""C"" __global__ void rbf_forward(
    const float* input, const float* centers, const float* epsilons, float* output,
    int batchSize, int numCenters, int inputDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * numCenters;
    if (idx >= total) return;

    int centerIdx = idx % numCenters;
    int batchIdx = idx / numCenters;

    float distSq = 0.0f;
    for (int d = 0; d < inputDim; d++) {
        float diff = input[batchIdx * inputDim + d] - centers[centerIdx * inputDim + d];
        distSq += diff * diff;
    }

    output[idx] = expf(-epsilons[centerIdx] * distSq);
}

extern ""C"" __global__ void stdp_update(
    float* weights, const float* preTrace, const float* postTrace,
    const float* preSpike, const float* postSpike,
    float ltpRate, float ltdRate, float homeostasisRate,
    float minWeight, float maxWeight, int numPre, int numPost)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numPre * numPost;
    if (idx >= total) return;

    int postIdx = idx % numPost;
    int preIdx = idx / numPost;

    float w = weights[idx];
    float deltaW = 0.0f;

    // LTP: Pre-synaptic trace * Post-synaptic spike
    if (postSpike[postIdx] > 0.0f) {
        deltaW += ltpRate * preTrace[preIdx];
    }

    // LTD: Post-synaptic trace * Pre-synaptic spike
    if (preSpike[preIdx] > 0.0f) {
        deltaW -= ltdRate * postTrace[postIdx];
    }

    // Homeostasis
    deltaW -= homeostasisRate * w * (postSpike[postIdx] > 0.0f ? 1.0f : 0.0f);

    w += deltaW;
    w = fmaxf(minWeight, fminf(maxWeight, w));
    weights[idx] = w;
}

extern ""C"" __global__ void update_traces(
    float* traces, const float* spikes, const float* input,
    float decay, float threshold, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float spike = spikes[idx] > threshold ? 1.0f : 0.0f;
    // Simple trace update: trace = trace * decay + spike
    traces[idx] = traces[idx] * decay + spike;
}

extern ""C"" __global__ void sgd_momentum_update(
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

// ===========================================================================
// RANDOM NUMBER GENERATION (Simple LCG)
// ===========================================================================

// PCG32 implementation for better quality than LCG
__device__ uint pcg32_random(ulong* state, ulong* inc)
{
    ulong oldstate = *state;
    // Advance internal state
    *state = oldstate * 6364136223846793005ULL + (*inc | 1);
    // Calculate output function (XSH-RR), uses old state for max ILP
    uint xorshifted = (uint)(((oldstate >> 18) ^ oldstate) >> 27);
    uint rot = (uint)(oldstate >> 59);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

extern ""C"" __global__ void generate_random_uniform(
    float* output, int size, float min, float max, ulong seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Seeding: use global seed + index to ensure different stream per thread
    ulong state = seed + (ulong)idx;
    ulong inc = (ulong)idx; 
    
    // Warm up
    pcg32_random(&state, &inc);
    
    uint rnd = pcg32_random(&state, &inc);
    
    // Convert to [0, 1) float
    float r = (float)rnd / 4294967296.0f;
    
    output[idx] = min + r * (max - min);
}

extern ""C"" __global__ void generate_random_normal(
    float* output, int size, float mean, float stdDev, ulong seed)
{
    // Box-Muller transform generates 2 numbers
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx >= size) return;

    ulong state = seed + (ulong)idx;
    ulong inc = (ulong)idx; 
    pcg32_random(&state, &inc);

    uint u1_int = pcg32_random(&state, &inc);
    uint u2_int = pcg32_random(&state, &inc);

    float u1 = (float)u1_int / 4294967296.0f;
    float u2 = (float)u2_int / 4294967296.0f;
    
    // Avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;

    float mag = stdDev * sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * 3.14159265f * u2) + mean;
    float z1 = mag * sinf(2.0f * 3.14159265f * u2) + mean;

    output[pairIdx] = z0;
    if (pairIdx + 1 < size) {
        output[pairIdx + 1] = z1;
    }
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                // Activation gradients
                "relu_backward",
                "sigmoid_backward",
                "tanh_backward",
                "gelu_backward",
                "softmax_backward",
                "leaky_relu",
                "leaky_relu_backward",
                "elu",
                "elu_backward",
                "silu",
                "swish_backward",
                "mish",
                "softplus",
                "hardswish",
                // Loss functions
                "cross_entropy_loss",
                "cross_entropy_backward",
                "bce_loss",
                "bce_backward",
                "mse_loss",
                "mse_backward",
                "smooth_l1_loss",
                "smooth_l1_backward",
                // Utilities
                "clamp",
                "l2_norm_squared",
                "scale",
                "copy_buffer",
                // Comparisons
                "greater_than",
                "less_than",
                "equals",
                "where_cond",
                // Statistics
                "compute_mean_var",
                "argmax_axis",
                "argmin_axis",
                "mean_axis",
                "max_axis",
                "var_axis",
                // Broadcast operations
                "broadcast_multiply_last_axis",
                "broadcast_multiply_first_axis",
                "broadcast_multiply_general",
                // Capsule network operations
                "squash",
                "squash_backward",
                // Tile/repeat operations
                "tile_batch",
                "tile_axis",
                // Optimizers
                "sgd_step",
                "sgd_momentum_update",
                "adam_step",
                "adamw_step",
                // Dropout and embedding
                "dropout_forward",
                "dropout_backward",
                "embedding_forward",
                "embedding_backward",
                // Transpose
                "transpose_2d",
                "batched_transpose",
                "permute_general",
                // Specialized
                "rbf_forward",
                "stdp_update",
                "update_traces",
                "generate_random_uniform",
                "generate_random_normal"
            };
        }
    }
}
