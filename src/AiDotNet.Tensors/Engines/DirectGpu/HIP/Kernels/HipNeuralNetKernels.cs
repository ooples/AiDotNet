// Copyright (c) AiDotNet. All rights reserved.
// HIP neural network kernels - activation gradients, loss functions, optimizers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipNeuralNetKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

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
extern ""C"" __global__ void broadcast_multiply_general(
    const float* A, const float* B, float* C,
    const int* aStrides, const int* bStrides, const int* cShape,
    int rank, int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

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
extern ""C"" __global__ void squash(
    const float* input, float* output,
    int numCapsules, int capsuleDim, float epsilon)
{
    int capsuleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsuleIdx >= numCapsules) return;

    int baseIdx = capsuleIdx * capsuleDim;

    float normSquared = 0.0f;
    for (int i = 0; i < capsuleDim; i++) {
        float val = input[baseIdx + i];
        normSquared += val * val;
    }

    float norm = sqrtf(normSquared + epsilon);
    float scale = normSquared / ((1.0f + normSquared) * norm);

    for (int i = 0; i < capsuleDim; i++) {
        output[baseIdx + i] = input[baseIdx + i] * scale;
    }
}

extern ""C"" __global__ void squash_backward(
    const float* gradOutput, const float* input, float* gradInput,
    int numCapsules, int capsuleDim, float epsilon)
{
    int capsuleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsuleIdx >= numCapsules) return;

    int baseIdx = capsuleIdx * capsuleDim;

    float normSquared = 0.0f;
    for (int i = 0; i < capsuleDim; i++) {
        float val = input[baseIdx + i];
        normSquared += val * val;
    }

    float scale = 1.0f / (1.0f + normSquared);
    for (int i = 0; i < capsuleDim; i++) {
        gradInput[baseIdx + i] = gradOutput[baseIdx + i] * scale;
    }
}

// ===========================================================================
// TILE/REPEAT KERNELS
// ===========================================================================

// Tile tensor along batch dimension (axis 0)
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
extern ""C"" __global__ void tile_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * axisSize * repeats * innerSize;
    if (idx >= totalSize) return;

    int outputAxisSize = axisSize * repeats;
    int innerIdx = idx % innerSize;
    int temp = idx / innerSize;
    int outputAxisIdx = temp % outputAxisSize;
    int outerIdx = temp / outputAxisSize;

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

// ===========================================================================
// BATCHED GEMM KERNEL
// ===========================================================================

// Batched General Matrix Multiply: C[b] = alpha * A[b] * B[b] + beta * C[b]
// Uses tiled approach with shared memory for better memory access patterns
#define BATCHED_TILE_SIZE 16

extern ""C"" __global__ void batched_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batchCount,
    float alpha, float beta)
{
    // Each block handles one tile of output
    int batch = blockIdx.z;
    int row = blockIdx.y * BATCHED_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * BATCHED_TILE_SIZE + threadIdx.x;

    if (batch >= batchCount) return;

    // Calculate batch offsets
    int aStride = M * K;
    int bStride = K * N;
    int cStride = M * N;

    const float* Abatch = A + batch * aStride;
    const float* Bbatch = B + batch * bStride;
    float* Cbatch = C + batch * cStride;

    __shared__ float As[BATCHED_TILE_SIZE][BATCHED_TILE_SIZE];
    __shared__ float Bs[BATCHED_TILE_SIZE][BATCHED_TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles of K dimension
    int numTiles = (K + BATCHED_TILE_SIZE - 1) / BATCHED_TILE_SIZE;
    for (int t = 0; t < numTiles; t++)
    {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = t * BATCHED_TILE_SIZE + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = Abatch[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int bRow = t * BATCHED_TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = Bbatch[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BATCHED_TILE_SIZE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
    {
        int outIdx = row * N + col;
        Cbatch[outIdx] = alpha * sum + beta * Cbatch[outIdx];
    }
}

#undef BATCHED_TILE_SIZE
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
            "compute_mean_var", "argmax_axis", "argmin_axis", "mean_axis", "max_axis", "var_axis",
            "broadcast_multiply_last_axis", "broadcast_multiply_first_axis", "broadcast_multiply_general",
            "squash", "squash_backward",
            "tile_batch", "tile_axis",
            "sgd_step", "adam_step", "adamw_step",
            "dropout_forward", "dropout_backward", "embedding_forward", "embedding_backward",
            "transpose_2d", "batched_transpose", "permute_general", "batched_gemm"
        };
    }
}
