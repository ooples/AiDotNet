// Copyright (c) AiDotNet. All rights reserved.
// Neural network utility kernels for gradient computation, loss functions, and optimizers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for neural network operations including activation gradients,
    /// loss functions, optimizers, and utility operations.
    /// </summary>
    internal static class NeuralNetKernels
    {
        /// <summary>
        /// Gets all neural network kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// ACTIVATION GRADIENT KERNELS
// ===========================================================================

// ReLU backward: grad * (input > 0)
__kernel void relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
}

// Sigmoid backward: grad * output * (1 - output)
__kernel void sigmoid_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float o = output[idx];
    gradInput[idx] = gradOutput[idx] * o * (1.0f - o);
}

// Tanh backward: grad * (1 - output^2)
__kernel void tanh_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float o = output[idx];
    gradInput[idx] = gradOutput[idx] * (1.0f - o * o);
}

// GELU backward
__kernel void gelu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = input[idx];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanhInner = tanh(inner);
    float sech2 = 1.0f - tanhInner * tanhInner;
    float dInner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);

    gradInput[idx] = gradOutput[idx] * (0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * dInner);
}

// Softmax backward (assumes gradOutput already multiplied with loss derivative)
__kernel void softmax_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int batchSize,
    const int features)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    // Compute dot(gradOutput, output)
    float dotProd = 0.0f;
    for (int f = 0; f < features; f++) {
        int idx = b * features + f;
        dotProd += gradOutput[idx] * output[idx];
    }

    // gradInput = output * (gradOutput - dotProd)
    for (int f = 0; f < features; f++) {
        int idx = b * features + f;
        gradInput[idx] = output[idx] * (gradOutput[idx] - dotProd);
    }
}

// Leaky ReLU forward
__kernel void leaky_relu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * x;
}

// Leaky ReLU backward
__kernel void leaky_relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : alpha * gradOutput[idx];
}

// ELU forward: x > 0 ? x : alpha * (exp(x) - 1)
__kernel void elu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
}

// ELU backward
__kernel void elu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* output,
    __global float* gradInput,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    gradInput[idx] = x > 0.0f ? gradOutput[idx] : gradOutput[idx] * (output[idx] + alpha);
}

// Swish forward: x * sigmoid(x)
__kernel void swish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sig;
}

// Swish backward
__kernel void swish_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    float swishVal = x * sig;
    gradInput[idx] = gradOutput[idx] * (swishVal + sig * (1.0f - swishVal));
}

// SiLU (Sigmoid Linear Unit) - same as Swish with beta=1
__kernel void silu_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x / (1.0f + exp(-x));
}

// Mish: x * tanh(softplus(x))
__kernel void mish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sp = log(1.0f + exp(x));
    output[idx] = x * tanh(sp);
}

// Softplus: log(1 + exp(x))
__kernel void softplus_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    // Use stable version for large x
    output[idx] = x > 20.0f ? x : log(1.0f + exp(x));
}

// Hardswish: x * min(max(x + 3, 0), 6) / 6
__kernel void hardswish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x * fmin(fmax(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// SELU: scale * (x if x > 0, else alpha * (exp(x) - 1))
// Standard parameters: alpha ≈ 1.6733, scale ≈ 1.0507
__kernel void selu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const float scale,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = scale * (x > 0.0f ? x : alpha * (exp(x) - 1.0f));
}

// Hardsigmoid: clip((x + 3) / 6, 0, 1)
__kernel void hardsigmoid_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = fmin(fmax((x + 3.0f) / 6.0f, 0.0f), 1.0f);
}

// Hardtanh: clip(x, minVal, maxVal)
__kernel void hardtanh_forward(
    __global const float* input,
    __global float* output,
    const float minVal,
    const float maxVal,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmin(fmax(input[idx], minVal), maxVal);
}

// ===========================================================================
// LOSS FUNCTION KERNELS
// ===========================================================================

// Cross-entropy loss (with softmax input)
// Returns per-sample loss, needs reduction afterwards
__kernel void cross_entropy_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int batchSize,
    const int numClasses)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int c = 0; c < numClasses; c++) {
        maxVal = fmax(maxVal, predictions[b * numClasses + c]);
    }

    // Compute log-sum-exp
    float sumExp = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        sumExp += exp(predictions[b * numClasses + c] - maxVal);
    }
    float logSumExp = maxVal + log(sumExp);

    // Compute cross-entropy: -sum(target * log(softmax(pred)))
    float loss = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        int idx = b * numClasses + c;
        float logProb = predictions[idx] - logSumExp;
        loss -= targets[idx] * logProb;
    }
    losses[b] = loss;
}

// Cross-entropy backward (combined softmax + cross-entropy gradient)
__kernel void cross_entropy_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int batchSize,
    const int numClasses)
{
    const int idx = get_global_id(0);
    const int c = idx % numClasses;
    const int b = idx / numClasses;

    if (b >= batchSize) return;

    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int i = 0; i < numClasses; i++) {
        maxVal = fmax(maxVal, predictions[b * numClasses + i]);
    }

    // Compute softmax
    float sumExp = 0.0f;
    for (int i = 0; i < numClasses; i++) {
        sumExp += exp(predictions[b * numClasses + i] - maxVal);
    }
    float softmax = exp(predictions[idx] - maxVal) / sumExp;

    // Gradient is (softmax - target) / batchSize
    gradInput[idx] = (softmax - targets[idx]) / (float)batchSize;
}

// Binary cross-entropy loss
__kernel void bce_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float p = predictions[idx];
    float t = targets[idx];
    // Clamp to avoid log(0)
    p = fmax(fmin(p, 1.0f - 1e-7f), 1e-7f);
    losses[idx] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
}

// Binary cross-entropy backward
__kernel void bce_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float p = predictions[idx];
    float t = targets[idx];
    // Clamp to avoid division by zero
    p = fmax(fmin(p, 1.0f - 1e-7f), 1e-7f);
    gradInput[idx] = (p - t) / (p * (1.0f - p)) / (float)size;
}

// MSE loss
__kernel void mse_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    losses[idx] = diff * diff;
}

// MSE backward
__kernel void mse_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = 2.0f * (predictions[idx] - targets[idx]) / (float)size;
}

// Smooth L1 (Huber) loss
__kernel void smooth_l1_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size,
    const float beta)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = fabs(predictions[idx] - targets[idx]);
    losses[idx] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
}

// Smooth L1 backward
__kernel void smooth_l1_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size,
    const float beta)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    float absDiff = fabs(diff);
    float grad = absDiff < beta ? diff / beta : (diff > 0.0f ? 1.0f : -1.0f);
    gradInput[idx] = grad / (float)size;
}

// ===========================================================================
// OPTIMIZER KERNELS
// ===========================================================================

// SGD with momentum update
__kernel void sgd_momentum_update(
    __global float* param,
    __global const float* gradient,
    __global float* velocity,
    const float learningRate,
    const float momentum,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;
    param[idx] -= learningRate * v;
}

// Adam optimizer update
__kernel void adam_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];

    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment estimate
    float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - pow(beta1, (float)step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)step));

    // Update parameters
    float update = learningRate * m_hat / (sqrt(v_hat) + epsilon);
    if (weightDecay > 0.0f) {
        update += learningRate * weightDecay * param[idx];
    }
    param[idx] -= update;
}

// AdamW optimizer update (decoupled weight decay)
__kernel void adamw_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];

    // Decoupled weight decay (applied directly to params, not gradients)
    if (weightDecay > 0.0f) {
        param[idx] *= (1.0f - learningRate * weightDecay);
    }

    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment estimate
    float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - pow(beta1, (float)step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)step));

    // Update parameters
    param[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

// ===========================================================================
// UTILITY KERNELS
// ===========================================================================

// Clamp values between min and max
__kernel void clamp_values(
    __global const float* input,
    __global float* output,
    const float minVal,
    const float maxVal,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmin(fmax(input[idx], minVal), maxVal);
}

// Clip by value (symmetric around 0)
__kernel void clip_by_value(
    __global const float* input,
    __global float* output,
    const float clipValue,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmin(fmax(input[idx], -clipValue), clipValue);
}

// Transpose 2D matrix
__kernel void transpose2d(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= rows || col >= cols) return;

    output[col * rows + row] = input[row * cols + col];
}

// Batched transpose
__kernel void batched_transpose(
    __global const float* input,
    __global float* output,
    const int batch,
    const int rows,
    const int cols)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int b = get_global_id(2);

    if (row >= rows || col >= cols || b >= batch) return;

    int inIdx = (b * rows + row) * cols + col;
    int outIdx = (b * cols + col) * rows + row;
    output[outIdx] = input[inIdx];
}

// Fill buffer with constant value
__kernel void fill_buffer(
    __global float* buffer,
    const float value,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    buffer[idx] = value;
}

// Copy buffer
__kernel void copy_buffer(
    __global const float* src,
    __global float* dst,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    dst[idx] = src[idx];
}

// Comparison: greater than
__kernel void greater_than(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] > B[idx] ? 1.0f : 0.0f;
}

// Comparison: less than
__kernel void less_than(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] < B[idx] ? 1.0f : 0.0f;
}

// Comparison: equal
__kernel void equal_values(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] == B[idx] ? 1.0f : 0.0f;
}

// Where (conditional select)
__kernel void where_select(
    __global const float* condition,
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = condition[idx] > 0.0f ? A[idx] : B[idx];
}

// Mean along axis
__kernel void mean_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        sum += input[outer * reduceSize + i];
    }
    output[outer] = sum / (float)reduceSize;
}

// Variance along axis
__kernel void var_axis(
    __global const float* input,
    __global const float* mean,
    __global float* variance,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float m = mean[outer];
    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        float diff = input[outer * reduceSize + i] - m;
        sum += diff * diff;
    }
    variance[outer] = sum / (float)reduceSize;
}

// ArgMax along axis
__kernel void argmax_axis(
    __global const float* input,
    __global float* indices,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outer * reduceSize + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    indices[outer] = (float)maxIdx;
}

// ArgMin along axis
__kernel void argmin_axis(
    __global const float* input,
    __global float* indices,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float minVal = INFINITY;
    int minIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outer * reduceSize + i];
        if (val < minVal) {
            minVal = val;
            minIdx = i;
        }
    }
    indices[outer] = (float)minIdx;
}

// Broadcast multiply: C = A * B where B is broadcast along last axis
// A has shape (outerSize * innerSize), B has shape (innerSize), C has shape (outerSize * innerSize)
__kernel void broadcast_multiply_last_axis(
    __global const float* input,
    __global const float* broadcast,
    __global float* output,
    const int outerSize,
    const int innerSize)
{
    const int idx = get_global_id(0);
    const int totalSize = outerSize * innerSize;
    if (idx >= totalSize) return;

    const int innerIdx = idx % innerSize;
    output[idx] = input[idx] * broadcast[innerIdx];
}

// Broadcast multiply: C = A * B where B is broadcast along first axis
// A has shape (outerSize * innerSize), B has shape (outerSize), C has shape (outerSize * innerSize)
__kernel void broadcast_multiply_first_axis(
    __global const float* input,
    __global const float* broadcast,
    __global float* output,
    const int outerSize,
    const int innerSize)
{
    const int idx = get_global_id(0);
    const int totalSize = outerSize * innerSize;
    if (idx >= totalSize) return;

    const int outerIdx = idx / innerSize;
    output[idx] = input[idx] * broadcast[outerIdx];
}

// General broadcast multiply for tensors with compatible shapes
__kernel void broadcast_multiply_general(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* aStrides,
    __global const int* bStrides,
    __global const int* cShape,
    const int rank,
    const int totalSize)
{
    const int idx = get_global_id(0);
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

// Squash activation for capsule networks
// squash(v) = ||v||^2 / (1 + ||v||^2) * v / ||v||
__kernel void squash(
    __global const float* input,
    __global float* output,
    const int numCapsules,
    const int capsuleDim,
    const float epsilon)
{
    const int capsuleIdx = get_global_id(0);
    if (capsuleIdx >= numCapsules) return;

    const int baseIdx = capsuleIdx * capsuleDim;

    float normSquared = 0.0f;
    for (int i = 0; i < capsuleDim; i++) {
        float val = input[baseIdx + i];
        normSquared += val * val;
    }

    float norm = sqrt(normSquared + epsilon);
    float scale = normSquared / ((1.0f + normSquared) * norm);

    for (int i = 0; i < capsuleDim; i++) {
        output[baseIdx + i] = input[baseIdx + i] * scale;
    }
}

__kernel void squash_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int numCapsules,
    const int capsuleDim,
    const float epsilon)
{
    const int capsuleIdx = get_global_id(0);
    if (capsuleIdx >= numCapsules) return;

    const int baseIdx = capsuleIdx * capsuleDim;

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

// Tile tensor along batch dimension (axis 0)
__kernel void tile_batch(
    __global const float* input,
    __global float* output,
    const int repeats,
    const int innerSize)
{
    const int idx = get_global_id(0);
    const int totalSize = repeats * innerSize;
    if (idx >= totalSize) return;

    const int innerIdx = idx % innerSize;
    output[idx] = input[innerIdx];
}

// General tile along any axis
__kernel void tile_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int axisSize,
    const int innerSize,
    const int repeats)
{
    const int idx = get_global_id(0);
    const int totalSize = outerSize * axisSize * repeats * innerSize;
    if (idx >= totalSize) return;

    const int outputAxisSize = axisSize * repeats;
    const int innerIdx = idx % innerSize;
    int temp = idx / innerSize;
    const int outputAxisIdx = temp % outputAxisSize;
    const int outerIdx = temp / outputAxisSize;

    const int inputAxisIdx = outputAxisIdx % axisSize;
    const int inputIdx = outerIdx * axisSize * innerSize + inputAxisIdx * innerSize + innerIdx;

    output[idx] = input[inputIdx];
}

// Dropout forward
__kernel void dropout_forward(
    __global const float* input,
    __global float* output,
    __global float* mask,
    const int size,
    const float dropoutRate,
    const ulong seed,
    const int training)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    if (!training) {
        output[idx] = input[idx];
        mask[idx] = 1.0f;
        return;
    }

    // Simple LCG random number generator
    ulong state = seed + (ulong)idx * 6364136223846793005UL;
    state = state * 6364136223846793005UL + 1442695040888963407UL;
    float rand = (float)(state >> 33) / (float)(1UL << 31);

    float scale = 1.0f / (1.0f - dropoutRate);
    if (rand < dropoutRate) {
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    } else {
        output[idx] = input[idx] * scale;
        mask[idx] = scale;
    }
}

// Dropout backward
__kernel void dropout_backward(
    __global const float* gradOutput,
    __global const float* mask,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = gradOutput[idx] * mask[idx];
}

// Embedding lookup
__kernel void embedding_lookup(
    __global const float* indices,
    __global const float* embeddingTable,
    __global float* output,
    const int numIndices,
    const int embeddingDim)
{
    const int idx = get_global_id(0);
    if (idx >= numIndices) return;

    int index = (int)indices[idx];

    for (int d = 0; d < embeddingDim; d++) {
        output[idx * embeddingDim + d] = embeddingTable[index * embeddingDim + d];
    }
}

// Embedding backward (scatter add gradients)
__kernel void embedding_backward(
    __global const float* gradOutput,
    __global const float* indices,
    __global float* gradEmbedding,
    const int numIndices,
    const int embeddingDim)
{
    const int idx = get_global_id(0);
    if (idx >= numIndices) return;

    int index = (int)indices[idx];

    for (int d = 0; d < embeddingDim; d++) {
        // Note: This needs atomic add for thread safety in practice
        gradEmbedding[index * embeddingDim + d] += gradOutput[idx * embeddingDim + d];
    }
}

// Fused multiply-add: D = A * B + C
__kernel void fma_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global float* D,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    D[idx] = fma(A[idx], B[idx], C[idx]);
}

// Gather operation
__kernel void gather_kernel(
    __global const float* source,
    __global const float* indices,
    __global float* output,
    const int numIndices,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= numIndices) return;

    int index = (int)indices[idx];
    for (int f = 0; f < featureSize; f++) {
        output[idx * featureSize + f] = source[index * featureSize + f];
    }
}

// ScatterAdd operation
__kernel void scatter_add_kernel(
    __global const float* source,
    __global const float* indices,
    __global float* destination,
    const int sourceSize,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= sourceSize) return;

    int destIdx = (int)indices[idx];
    for (int f = 0; f < featureSize; f++) {
        // Note: Needs atomic add for thread safety
        destination[destIdx * featureSize + f] += source[idx * featureSize + f];
    }
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
                // Activation gradients
                "relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "softmax_backward",
                "leaky_relu_forward", "leaky_relu_backward",
                "elu_forward", "elu_backward",
                "swish_forward", "swish_backward",
                "silu_forward", "mish_forward", "softplus_forward", "hardswish_forward",
                "selu_forward", "hardsigmoid_forward", "hardtanh_forward",
                // Loss functions
                "cross_entropy_loss", "cross_entropy_backward",
                "bce_loss", "bce_backward",
                "mse_loss", "mse_backward",
                "smooth_l1_loss", "smooth_l1_backward",
                // Optimizers
                "sgd_momentum_update", "adam_update", "adamw_update",
                // Utilities
                "clamp_values", "clip_by_value",
                "transpose2d", "batched_transpose",
                "fill_buffer", "copy_buffer",
                "greater_than", "less_than", "equal_values", "where_select",
                "mean_axis", "var_axis", "argmax_axis", "argmin_axis",
                "broadcast_multiply_last_axis", "broadcast_multiply_first_axis", "broadcast_multiply_general",
                "squash", "squash_backward",
                "tile_batch", "tile_axis",
                "dropout_forward", "dropout_backward",
                "embedding_lookup", "embedding_backward",
                "fma_kernel", "gather_kernel", "scatter_add_kernel"
            };
        }
    }
}
