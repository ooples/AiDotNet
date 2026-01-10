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

    // Guard against t==0 which causes division by zero
    int safe_t = t < 1 ? 1 : t;
    float mHat = mVal / (1.0f - powf(beta1, (float)safe_t));
    float vHat = vVal / (1.0f - powf(beta2, (float)safe_t));

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

    // Guard against t==0 which causes division by zero
    int safe_t = t < 1 ? 1 : t;
    float mHat = mVal / (1.0f - powf(beta1, (float)safe_t));
    float vHat = vVal / (1.0f - powf(beta2, (float)safe_t));

    // AdamW: decoupled weight decay
    param[idx] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[idx]);
}

extern ""C"" __global__ void rmsprop_step(
    float* param, const float* gradient, float* squaredAvg,
    float learningRate, float rho, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update moving average of squared gradients
    float sqAvg = rho * squaredAvg[idx] + (1.0f - rho) * grad * grad;
    squaredAvg[idx] = sqAvg;

    // Update parameters
    param[idx] -= learningRate * grad / (sqrtf(sqAvg) + epsilon);
}

extern ""C"" __global__ void adagrad_step(
    float* param, const float* gradient, float* accumulatedGrad,
    float learningRate, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Accumulate squared gradients
    float accum = accumulatedGrad[idx] + grad * grad;
    accumulatedGrad[idx] = accum;

    // Update parameters
    param[idx] -= learningRate * grad / (sqrtf(accum) + epsilon);
}

extern ""C"" __global__ void nag_step(
    float* param, const float* gradient, float* velocity,
    float learningRate, float momentum, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Nesterov Accelerated Gradient (NAG):
    // Standard formulation matching PyTorch's Nesterov momentum:
    // v_t = momentum * v_{t-1} + grad
    // param = param - lr * (grad + momentum * v_t)
    float v = velocity[idx];
    float vNew = momentum * v + grad;
    velocity[idx] = vNew;

    // NAG update: apply gradient with lookahead via momentum
    param[idx] -= learningRate * (grad + momentum * vNew);
}

extern ""C"" __global__ void lars_step(
    float* param, const float* gradient, float* velocity,
    float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Note: LARS computes local learning rate based on layer-wise norms
    // This simplified version applies uniform scaling; full LARS requires
    // computing norms per layer before calling this kernel
    float grad = gradient[idx];
    float p = param[idx];

    // Apply weight decay
    if (weightDecay > 0.0f) {
        grad += weightDecay * p;
    }

    // Update velocity with momentum
    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;

    // Update parameters
    param[idx] = p - learningRate * v;
}

extern ""C"" __global__ void lamb_step(
    float* param, const float* gradient, float* m, float* v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, float trustRatio, int t, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    // Adam-like moment updates
    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    // Bias correction
    // Guard against t==0 which causes division by zero
    int safe_t = t < 1 ? 1 : t;
    float mHat = mVal / (1.0f - powf(beta1, (float)safe_t));
    float vHat = vVal / (1.0f - powf(beta2, (float)safe_t));

    // LAMB: Adam update direction with weight decay
    float adamUpdate = mHat / (sqrtf(vHat) + epsilon);
    float update = adamUpdate + weightDecay * p;

    // Apply LAMB trust ratio to scale the update
    param[idx] = p - learningRate * trustRatio * update;
}

// Vanilla SGD update (no momentum)
extern ""C"" __global__ void sgd_update(
    float* param, const float* gradient,
    float learningRate, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    param[idx] -= learningRate * grad;
}

// AdaDelta optimizer update
extern ""C"" __global__ void adadelta_update(
    float* param, const float* gradient, float* accumGrad, float* accumUpdate,
    float rho, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float ag = rho * accumGrad[idx] + (1.0f - rho) * grad * grad;
    accumGrad[idx] = ag;

    float rmsUpdate = sqrtf(accumUpdate[idx] + epsilon);
    float rmsGrad = sqrtf(ag + epsilon);
    float update = (rmsUpdate / rmsGrad) * grad;

    accumUpdate[idx] = rho * accumUpdate[idx] + (1.0f - rho) * update * update;

    param[idx] -= update;
}

// AMSGrad optimizer update
extern ""C"" __global__ void amsgrad_update(
    float* param, const float* gradient, float* m, float* v, float* vMax,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    float vMaxVal = fmaxf(vMax[idx], vVal);
    vMax[idx] = vMaxVal;

    float mHat = mVal / (1.0f - powf(beta1, (float)step));

    param[idx] -= learningRate * mHat / (sqrtf(vMaxVal) + epsilon);
}

// AdaMax optimizer update
extern ""C"" __global__ void adamax_update(
    float* param, const float* gradient, float* m, float* u,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float uVal = fmaxf(beta2 * u[idx], fabsf(grad));
    u[idx] = uVal;

    float biasCorrection = 1.0f - powf(beta1, (float)step);

    param[idx] -= (learningRate / biasCorrection) * mVal / (uVal + epsilon);
}

// Lion optimizer update
extern ""C"" __global__ void lion_update(
    float* param, const float* gradient, float* m,
    float learningRate, float beta1, float beta2, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float mVal = m[idx];

    float interp = beta1 * mVal + (1.0f - beta1) * grad;
    float update = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);

    m[idx] = beta2 * mVal + (1.0f - beta2) * grad;

    if (weightDecay > 0.0f) {
        update += weightDecay * param[idx];
    }

    param[idx] -= learningRate * update;
}

// Nadam optimizer update
extern ""C"" __global__ void nadam_update(
    float* param, const float* gradient, float* m, float* v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    float beta1Pow = powf(beta1, (float)step);
    float beta2Pow = powf(beta2, (float)step);
    float mHat = mVal / (1.0f - beta1Pow);
    float vHat = vVal / (1.0f - beta2Pow);

    float mNesterov = beta1 * mHat + (1.0f - beta1) * grad / (1.0f - beta1Pow);

    param[idx] -= learningRate * mNesterov / (sqrtf(vHat) + epsilon);
}

// FTRL optimizer update
extern ""C"" __global__ void ftrl_update(
    float* param, const float* gradient, float* z, float* n,
    float learningRate, float l1Reg, float l2Reg, float beta, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float nVal = n[idx];
    float zVal = z[idx];
    float pVal = param[idx];

    float nNew = nVal + grad * grad;
    n[idx] = nNew;

    float sigma = (sqrtf(nNew) - sqrtf(nVal)) / learningRate;

    zVal = zVal + grad - sigma * pVal;
    z[idx] = zVal;

    float zSign = (zVal > 0.0f) ? 1.0f : ((zVal < 0.0f) ? -1.0f : 0.0f);
    float zAbs = fabsf(zVal);

    if (zAbs <= l1Reg) {
        param[idx] = 0.0f;
    } else {
        float denom = (beta + sqrtf(nNew)) / learningRate + l2Reg;
        param[idx] = -zSign * (zAbs - l1Reg) / denom;
    }
}

// ===========================================================================
// ADDITIONAL OPTIMIZER KERNELS
// ===========================================================================

// Proximal Gradient Descent with L1 regularization
extern ""C"" __global__ void proximal_gradient_step(
    float* param, const float* gradient, 
    float learningRate, float l1Lambda, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float p = param[idx] - learningRate * gradient[idx];
    // Soft thresholding for L1 regularization
    float threshold = l1Lambda * learningRate;
    if (p > threshold) {
        param[idx] = p - threshold;
    } else if (p < -threshold) {
        param[idx] = p + threshold;
    } else {
        param[idx] = 0.0f;
    }
}

// Conjugate Gradient update
extern ""C"" __global__ void conjugate_gradient_step(
    float* param, float* direction, const float* gradient, 
    float* prevGradient, float beta, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    direction[idx] = -gradient[idx] + beta * direction[idx];
    param[idx] += alpha * direction[idx];
    prevGradient[idx] = gradient[idx];
}

// ---------------------------------------------------------------------------
// L-BFGS Two-Loop Recursion GPU Kernels
// ---------------------------------------------------------------------------
// These kernels provide GPU-accelerated primitives for L-BFGS optimization.
// The host orchestrates the sequential two-loop algorithm by calling these
// kernels in order. Each kernel is fully parallelized on GPU.
//
// Algorithm (host orchestration):
// 1. lbfgs_copy_vector: q = gradient
// 2. Backward loop (i = m-1 down to 0):
//    a. lbfgs_dot_product_reduce: compute s[i]·q, store in partial sums
//    b. Host: alpha[i] = rho[i] * dot_result
//    c. lbfgs_axpy: q = q - alpha[i] * y[i]
// 3. lbfgs_scale_vector: r = gamma * q (initial Hessian scaling)
// 4. Forward loop (i = 0 to m-1):
//    a. lbfgs_dot_product_reduce: compute y[i]·r
//    b. Host: beta = rho[i] * dot_result
//    c. lbfgs_axpy: r = r + (alpha[i] - beta) * s[i]
// 5. lbfgs_apply_direction: param = param - lr * r
// ---------------------------------------------------------------------------

// Copy vector: dst = src (used for q = gradient initialization)
extern ""C"" __global__ void lbfgs_copy_vector(
    const float* src, float* dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = src[idx];
}

// Parallel reduction for dot product
// Computes partial sums that must be summed on host or with another kernel
// blockPartials[blockIdx.x] = sum of a[i]*b[i] for this block's elements
extern ""C"" __global__ void lbfgs_dot_product_reduce(
    const float* a, const float* b, float* blockPartials, int size)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute product, or 0 if out of bounds
    sdata[tid] = (idx < size) ? (a[idx] * b[idx]) : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write this block's partial sum
    if (tid == 0) {
        blockPartials[blockIdx.x] = sdata[0];
    }
}

// Final reduction of block partial sums to single value
// Call with 1 block; handles numBlocks > blockDim.x via strided loop
extern ""C"" __global__ void lbfgs_reduce_partials(
    const float* blockPartials, float* result, int numBlocks)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // Each thread accumulates multiple partials if numBlocks > blockDim.x
    // This handles the case where we have more partial blocks than threads
    float sum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sum += blockPartials[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// AXPY operation: y = y + alpha * x
// Used for q = q - alpha[i] * y[i] (with negative alpha)
// and r = r + (alpha[i] - beta) * s[i]
extern ""C"" __global__ void lbfgs_axpy(
    float* y, const float* x, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    y[idx] = y[idx] + alpha * x[idx];
}

// Scale and copy: dst = gamma * src
// Used for initial Hessian scaling: r = gamma * q
// where gamma = (s·y)/(y·y) for scaled identity initial Hessian
extern ""C"" __global__ void lbfgs_scale_vector(
    const float* src, float* dst, float gamma, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = gamma * src[idx];
}

// Apply final L-BFGS direction to parameters
// param = param - learningRate * direction
extern ""C"" __global__ void lbfgs_apply_direction(
    float* param, const float* direction, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    param[idx] = param[idx] - learningRate * direction[idx];
}

// Compute rho[i] = 1 / (y[i]·s[i]) for history storage
// This is computed once when adding to history
extern ""C"" __global__ void lbfgs_compute_rho(
    const float* dotProduct, float* rho)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float dot = dotProduct[0];
        // Guard against division by zero
        rho[0] = (fabsf(dot) > 1e-10f) ? (1.0f / dot) : 0.0f;
    }
}

// Update history buffers s[newest] = x_new - x_old, y[newest] = g_new - g_old
extern ""C"" __global__ void lbfgs_update_history(
    float* s_newest, float* y_newest,
    const float* x_new, const float* x_old,
    const float* g_new, const float* g_old,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    s_newest[idx] = x_new[idx] - x_old[idx];
    y_newest[idx] = g_new[idx] - g_old[idx];
}

// BFGS update (simplified - full version requires matrix operations)
extern ""C"" __global__ void bfgs_step(
    float* param, const float* gradient, const float* invHessianDiag,
    float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Diagonal approximation: x = x - lr * H^-1 * g
    param[idx] -= learningRate * invHessianDiag[idx] * gradient[idx];
}

// Levenberg-Marquardt with damping
extern ""C"" __global__ void levenberg_marquardt_step(
    float* param, const float* gradient, const float* hessianDiag,
    float lambda, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Damped update: (H + λI)^-1 * g
    float dampedHess = hessianDiag[idx] + lambda;
    param[idx] -= learningRate * gradient[idx] / (dampedHess + 1e-8f);
}

// Trust Region update
extern ""C"" __global__ void trust_region_step(
    float* param, const float* gradient, const float* hessianDiag,
    float trustRadius, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Simplified trust region with diagonal Hessian
    float grad = gradient[idx];
    float hess = hessianDiag[idx];
    float step = -grad / (hess + 1e-8f);
    
    // Limit step by trust radius
    float stepNorm = fabsf(step);
    if (stepNorm > trustRadius) {
        step = (step / stepNorm) * trustRadius;
    }
    
    param[idx] += learningRate * step;
}

// ADMM (Alternating Direction Method of Multipliers)
extern ""C"" __global__ void admm_step(
    float* param, const float* gradient, float* dual, 
    const float* consensus, float rho, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // ADMM primal update: minimize f(x) + (rho/2)||x - z + u||^2
    float augmentedGrad = gradient[idx] + rho * (param[idx] - consensus[idx] + dual[idx]);
    param[idx] -= learningRate * augmentedGrad;
    
    // Dual update: u = u + (x - z)
    dual[idx] += param[idx] - consensus[idx];
}

// Newton's Method (simplified with diagonal Hessian)
extern ""C"" __global__ void newton_method_step(
    float* param, const float* gradient, const float* hessianDiag,
    float learningRate, float damping, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Newton step: x = x - lr * H^-1 * g
    float hess = hessianDiag[idx];
    param[idx] -= learningRate * gradient[idx] / (hess + damping);
}

// DFP (Davidon-Fletcher-Powell) - diagonal approximation
extern ""C"" __global__ void dfp_step(
    float* param, const float* gradient, const float* invHessianDiag,
    float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // DFP update with diagonal approximation
    param[idx] -= learningRate * invHessianDiag[idx] * gradient[idx];
}

// Coordinate Descent (per-coordinate update)
// This kernel should be launched with <<<1, 1>>> since it updates a single coordinate
extern ""C"" __global__ void coordinate_descent_step(
    float* param, const float* gradient,
    int coordinate, float learningRate, int size)
{
    // Only allow the first thread to execute to prevent data race
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (coordinate >= size) return;
    param[coordinate] -= learningRate * gradient[coordinate];
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
// LSTM KERNELS
// ===========================================================================

// LSTM cell forward pass
// Computes gates and cell/hidden states for a single timestep
// Gates: i=input, f=forget, g=cell candidate, o=output
// Formulas:
//   i = sigmoid(Wi*x + Ui*h_prev + bi)
//   f = sigmoid(Wf*x + Uf*h_prev + bf)
//   g = tanh(Wg*x + Ug*h_prev + bg)
//   o = sigmoid(Wo*x + Uo*h_prev + bo)
//   c = f * c_prev + i * g
//   h = o * tanh(c)
extern ""C"" __global__ void lstm_cell_forward(
    const float* gates,      // Pre-computed gates: [batch, 4*hidden] = Wi*x + Ui*h + b
    const float* cellPrev,   // Previous cell state: [batch, hidden]
    float* cellNext,         // Output cell state: [batch, hidden]
    float* hiddenNext,       // Output hidden state: [batch, hidden]
    float* gateActivations,  // Store activated gates for backward: [batch, 4*hidden]
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;
    int gateOffset = b * 4 * hiddenSize;

    // Load pre-activated gates
    float gi = gates[gateOffset + h];                    // input gate
    float gf = gates[gateOffset + hiddenSize + h];       // forget gate
    float gg = gates[gateOffset + 2 * hiddenSize + h];   // cell candidate
    float go = gates[gateOffset + 3 * hiddenSize + h];   // output gate

    // Apply activations
    float i = 1.0f / (1.0f + expf(-gi));  // sigmoid
    float f = 1.0f / (1.0f + expf(-gf));  // sigmoid
    float g = tanhf(gg);                   // tanh
    float o = 1.0f / (1.0f + expf(-go));  // sigmoid

    // Compute new cell state
    float cPrev = cellPrev[idx];
    float c = f * cPrev + i * g;

    // Compute new hidden state
    float tanhC = tanhf(c);
    float hNew = o * tanhC;

    // Store outputs
    cellNext[idx] = c;
    hiddenNext[idx] = hNew;

    // Store activated gates for backward pass
    gateActivations[gateOffset + h] = i;
    gateActivations[gateOffset + hiddenSize + h] = f;
    gateActivations[gateOffset + 2 * hiddenSize + h] = g;
    gateActivations[gateOffset + 3 * hiddenSize + h] = o;
}

// LSTM cell backward pass
// Computes gradients for gates, cell state, and hidden state
extern ""C"" __global__ void lstm_cell_backward(
    const float* gradHidden,      // Gradient from next layer/timestep: [batch, hidden]
    const float* gradCellNext,    // Gradient of cell from next timestep: [batch, hidden]
    const float* gateActivations, // Stored activated gates: [batch, 4*hidden]
    const float* cellPrev,        // Previous cell state: [batch, hidden]
    const float* cellNext,        // Current cell state: [batch, hidden]
    float* gradGates,             // Output: gradient of pre-activation gates: [batch, 4*hidden]
    float* gradCellPrev,          // Output: gradient of previous cell: [batch, hidden]
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;
    int gateOffset = b * 4 * hiddenSize;

    // Load activated gates
    float i = gateActivations[gateOffset + h];
    float f = gateActivations[gateOffset + hiddenSize + h];
    float g = gateActivations[gateOffset + 2 * hiddenSize + h];
    float o = gateActivations[gateOffset + 3 * hiddenSize + h];

    // Load cell states
    float cPrev = cellPrev[idx];
    float c = cellNext[idx];
    float tanhC = tanhf(c);

    // Gradient from hidden state output
    float dH = gradHidden[idx];
    
    // Gradient through output gate: h = o * tanh(c)
    float dO = dH * tanhC;
    float dTanhC = dH * o;
    
    // Gradient through tanh(c)
    float dC = dTanhC * (1.0f - tanhC * tanhC);
    
    // Add gradient from next timestep's cell
    dC += gradCellNext[idx];
    
    // Gradient through cell update: c = f * c_prev + i * g
    float dF = dC * cPrev;
    float dI = dC * g;
    float dG = dC * i;
    float dCPrev = dC * f;

    // Gradient through sigmoid/tanh activations to pre-activations
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // tanh'(x) = 1 - tanh(x)^2
    float gradGi = dI * i * (1.0f - i);
    float gradGf = dF * f * (1.0f - f);
    float gradGg = dG * (1.0f - g * g);
    float gradGo = dO * o * (1.0f - o);

    // Store gradients
    gradGates[gateOffset + h] = gradGi;
    gradGates[gateOffset + hiddenSize + h] = gradGf;
    gradGates[gateOffset + 2 * hiddenSize + h] = gradGg;
    gradGates[gateOffset + 3 * hiddenSize + h] = gradGo;
    gradCellPrev[idx] = dCPrev;
}

// Fused LSTM gate computation: computes Wi*x + Ui*h + b for all 4 gates
extern ""C"" __global__ void lstm_gates_precompute(
    const float* input,         // Input: [batch, input_size]
    const float* hiddenPrev,    // Previous hidden: [batch, hidden]
    const float* weightsIH,     // Input-to-hidden weights: [4*hidden, input_size]
    const float* weightsHH,     // Hidden-to-hidden weights: [4*hidden, hidden]
    const float* bias,          // Bias: [4*hidden]
    float* gates,               // Output gates: [batch, 4*hidden]
    int batchSize, int inputSize, int hiddenSize)
{
    // PERFORMANCE TODO: Replace naive dot-product loops with cuBLAS cublasSgemm
    // for the weightsIH × input and weightsHH × hiddenPrev operations.
    // Current implementation is ~10-100x slower than cuBLAS for large matrices.
    // For production, use cublasSgemmBatched or separate cublasSgemm calls.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * 4 * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / (4 * hiddenSize);
    int g = idx % (4 * hiddenSize);

    float sum = bias[g];

    // Input-to-hidden: weightsIH[g, :] dot input[b, :]
    for (int i = 0; i < inputSize; i++) {
        sum += weightsIH[g * inputSize + i] * input[b * inputSize + i];
    }

    // Hidden-to-hidden: weightsHH[g, :] dot hiddenPrev[b, :]
    for (int h = 0; h < hiddenSize; h++) {
        sum += weightsHH[g * hiddenSize + h] * hiddenPrev[b * hiddenSize + h];
    }

    gates[idx] = sum;
}

// ===========================================================================
// GRU KERNELS
// ===========================================================================

// GRU cell forward pass
// Gates: r=reset, z=update, n=new
// Formulas:
//   r = sigmoid(Wr*x + Ur*h_prev + br)
//   z = sigmoid(Wz*x + Uz*h_prev + bz)
//   n = tanh(Wn*x + r*(Un*h_prev) + bn)
//   h = (1-z)*n + z*h_prev
extern ""C"" __global__ void gru_cell_forward(
    const float* gatesRZ,        // Pre-computed r,z gates: [batch, 2*hidden]
    const float* gateN_input,    // Wn*x + bn part: [batch, hidden]
    const float* gateN_hidden,   // Un*h_prev part: [batch, hidden]
    const float* hiddenPrev,     // Previous hidden: [batch, hidden]
    float* hiddenNext,           // Output hidden: [batch, hidden]
    float* gateActivations,      // Store for backward: [batch, 3*hidden] (r, z, n)
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;

    // Load pre-activated gates
    float gr = gatesRZ[b * 2 * hiddenSize + h];
    float gz = gatesRZ[b * 2 * hiddenSize + hiddenSize + h];

    // Apply sigmoid
    float r = 1.0f / (1.0f + expf(-gr));
    float z = 1.0f / (1.0f + expf(-gz));

    // Compute candidate: n = tanh(Wn*x + r*(Un*h_prev) + bn)
    float nInput = gateN_input[idx];
    float nHidden = gateN_hidden[idx];
    float nPre = nInput + r * nHidden;
    float n = tanhf(nPre);

    // Compute new hidden: h = (1-z)*n + z*h_prev
    float hPrev = hiddenPrev[idx];
    float hNew = (1.0f - z) * n + z * hPrev;

    // Store output
    hiddenNext[idx] = hNew;

    // Store activations for backward
    int actOffset = b * 3 * hiddenSize;
    gateActivations[actOffset + h] = r;
    gateActivations[actOffset + hiddenSize + h] = z;
    gateActivations[actOffset + 2 * hiddenSize + h] = n;
}

// GRU cell backward pass
extern ""C"" __global__ void gru_cell_backward(
    const float* gradHidden,      // Gradient from next layer/timestep: [batch, hidden]
    const float* gateActivations, // Stored activations (r, z, n): [batch, 3*hidden]
    const float* hiddenPrev,      // Previous hidden: [batch, hidden]
    const float* gateN_hidden,    // Un*h_prev from forward: [batch, hidden]
    float* gradGatesRZ,           // Output: gradient of r,z pre-activations: [batch, 2*hidden]
    float* gradGateN,             // Output: gradient of n pre-activation: [batch, hidden]
    float* gradHiddenPrev,        // Output: gradient of previous hidden: [batch, hidden]
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;

    // Load activations
    int actOffset = b * 3 * hiddenSize;
    float r = gateActivations[actOffset + h];
    float z = gateActivations[actOffset + hiddenSize + h];
    float n = gateActivations[actOffset + 2 * hiddenSize + h];

    float hPrev = hiddenPrev[idx];
    float dH = gradHidden[idx];

    // Gradient through h = (1-z)*n + z*h_prev
    float dZ = dH * (hPrev - n);
    float dN = dH * (1.0f - z);
    float dHPrev = dH * z;

    // Gradient through n = tanh(...)
    float dNPre = dN * (1.0f - n * n);

    // Gradient through reset gate in candidate
    float nHidden = gateN_hidden[idx];
    float dR = dNPre * nHidden;
    dHPrev += dNPre * r;  // Gradient through Un*h_prev

    // Gradient through sigmoid activations
    float gradGr = dR * r * (1.0f - r);
    float gradGz = dZ * z * (1.0f - z);

    // Store gradients
    gradGatesRZ[b * 2 * hiddenSize + h] = gradGr;
    gradGatesRZ[b * 2 * hiddenSize + hiddenSize + h] = gradGz;
    gradGateN[idx] = dNPre;
    gradHiddenPrev[idx] = dHPrev;
}

// ===========================================================================
// SCATTER/GATHER OPERATIONS FOR GNNs
// ===========================================================================

// Scatter add: dst[indices[i]] += src[i] for each i
// Used for accumulating gradients in embedding backward and GNN message passing
extern ""C"" __global__ void scatter_add(
    const float* src,        // Source values: [numElements]
    const int* indices,      // Target indices: [numElements]
    float* dst,              // Destination (accumulated): [dstSize]
    int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    int dstIdx = indices[idx];
    atomicAdd(&dst[dstIdx], src[idx]);
}

// Batched scatter add for multi-dimensional tensors
extern ""C"" __global__ void scatter_add_batched(
    const float* src,        // Source: [numElements, featureSize]
    const int* indices,      // Target indices: [numElements]
    float* dst,              // Destination: [numNodes, featureSize]
    int numElements, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements * featureSize) return;

    int elemIdx = idx / featureSize;
    int featIdx = idx % featureSize;
    int dstIdx = indices[elemIdx];

    atomicAdd(&dst[dstIdx * featureSize + featIdx], src[idx]);
}

// Scatter max for graph pooling
// Uses atomic compare-and-swap on 32-bit floats for memory-safe max operation
extern ""C"" __global__ void scatter_max(
    const float* src,
    const int* indices,
    float* dst,
    int* argmax,  // Store indices of max values (can be NULL if not needed)
    int numElements, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements * featureSize) return;

    int elemIdx = idx / featureSize;
    int featIdx = idx % featureSize;
    int dstIdx = indices[elemIdx];
    int dstOffset = dstIdx * featureSize + featIdx;

    float val = src[idx];

    // Atomic max using compare-and-swap on float (reinterpreted as uint)
    float old = dst[dstOffset];
    while (val > old) {
        float assumed = old;
        old = __uint_as_float(atomicCAS((unsigned int*)&dst[dstOffset],
                        __float_as_uint(assumed),
                        __float_as_uint(val)));
        if (old == assumed) {
            // Successfully updated dst, now update argmax if provided
            // Note: argmax update is not atomic with dst update, but for graph pooling
            // the eventual consistency is acceptable since we track which element contributed
            if (argmax != NULL) {
                argmax[dstOffset] = elemIdx;
            }
            break;
        }
    }
}

// Scatter mean: accumulate then divide by count
extern ""C"" __global__ void scatter_mean_accumulate(
    const float* src,
    const int* indices,
    float* dst,
    int* counts,
    int numElements, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements * featureSize) return;

    int elemIdx = idx / featureSize;
    int featIdx = idx % featureSize;
    int dstIdx = indices[elemIdx];

    atomicAdd(&dst[dstIdx * featureSize + featIdx], src[idx]);
    
    // Only count once per element (when featIdx == 0)
    if (featIdx == 0) {
        atomicAdd(&counts[dstIdx], 1);
    }
}

extern ""C"" __global__ void scatter_mean_normalize(
    float* dst,
    const int* counts,
    int numNodes, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes * featureSize) return;

    int nodeIdx = idx / featureSize;
    int count = counts[nodeIdx];
    if (count > 0) {
        dst[idx] /= (float)count;
    }
}

// ===========================================================================
// ADDITIONAL NORMALIZATION BACKWARD KERNELS
// ===========================================================================

// Group normalization backward - Pass 1: Compute group-wise sums
// This kernel accumulates sum(dy * gamma) and sum(dy * gamma * xHat) for each group
// REQUIREMENT: C must be evenly divisible by G (C % G == 0)
extern ""C"" __global__ void groupnorm_backward_sums(
    const float* gradOutput,   // [N, C, H, W]
    const float* input,        // [N, C, H, W]
    const float* mean,         // [N, G]
    const float* invStd,       // [N, G]
    const float* gamma,        // [C]
    float* sumDy,              // [N, G] - sum of dy * gamma per group
    float* sumDyXhat,          // [N, G] - sum of dy * gamma * xHat per group
    float* gradGamma,          // [C] - accumulated
    float* gradBeta,           // [C] - accumulated
    int N, int C, int H, int W, int G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    // Validate divisibility assumption (only first thread checks to avoid divergence)
    if (idx == 0 && C % G != 0) return; // Invalid configuration - C must be divisible by G

    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int channelsPerGroup = C / G;
    int g = c / channelsPerGroup;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * G + g];
    float s = invStd[n * G + g];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    // Accumulate gamma and beta gradients
    atomicAdd(&gradGamma[c], dy * xHat);
    atomicAdd(&gradBeta[c], dy);

    // Accumulate group-wise sums for input gradient computation
    int groupIdx = n * G + g;
    atomicAdd(&sumDy[groupIdx], dyGam);
    atomicAdd(&sumDyXhat[groupIdx], dyGam * xHat);
}

// Group normalization backward - Pass 2: Compute final input gradients
// Uses the group-wise sums to compute: dX = invStd * (dy*gamma - (1/groupSize)*(sumDy + xHat*sumDyXhat))
// where groupSize = channelsPerGroup * H * W is the number of elements in each group
// REQUIREMENT: C must be evenly divisible by G (C % G == 0)
extern ""C"" __global__ void groupnorm_backward(
    const float* gradOutput,   // [N, C, H, W]
    const float* input,        // [N, C, H, W]
    const float* mean,         // [N, G]
    const float* invStd,       // [N, G]
    const float* gamma,        // [C]
    const float* sumDy,        // [N, G] - precomputed sum of dy * gamma per group
    const float* sumDyXhat,    // [N, G] - precomputed sum of dy * gamma * xHat per group
    float* gradInput,          // [N, C, H, W]
    int N, int C, int H, int W, int G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    // Validate divisibility assumption (only first thread checks to avoid divergence)
    if (idx == 0 && C % G != 0) return; // Invalid configuration - C must be divisible by G

    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int channelsPerGroup = C / G;
    int g = c / channelsPerGroup;
    int groupSize = channelsPerGroup * H * W;
    float invN = 1.0f / (float)groupSize;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * G + g];
    float s = invStd[n * G + g];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    int groupIdx = n * G + g;
    float sDy = sumDy[groupIdx];
    float sDyXhat = sumDyXhat[groupIdx];

    // Full group normalization backward formula:
    // dX = invStd * (dy*gamma - invN*(sumDy + xHat*sumDyXhat))
    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// Instance normalization backward - Pass 1: Compute instance-wise sums
// For instance norm, each (n, c) pair is its own instance (like group norm with G=C)
extern ""C"" __global__ void instancenorm_backward_sums(
    const float* gradOutput,   // [N, C, H, W]
    const float* input,        // [N, C, H, W]
    const float* mean,         // [N, C]
    const float* invStd,       // [N, C]
    const float* gamma,        // [C]
    float* sumDy,              // [N, C] - sum of dy * gamma per instance
    float* sumDyXhat,          // [N, C] - sum of dy * gamma * xHat per instance
    float* gradGamma,          // [C]
    float* gradBeta,           // [C]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * C + c];
    float s = invStd[n * C + c];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    atomicAdd(&gradGamma[c], dy * xHat);
    atomicAdd(&gradBeta[c], dy);

    int instanceIdx = n * C + c;
    atomicAdd(&sumDy[instanceIdx], dyGam);
    atomicAdd(&sumDyXhat[instanceIdx], dyGam * xHat);
}

// Instance normalization backward - Pass 2: Compute final input gradients
extern ""C"" __global__ void instancenorm_backward(
    const float* gradOutput,   // [N, C, H, W]
    const float* input,        // [N, C, H, W]
    const float* mean,         // [N, C]
    const float* invStd,       // [N, C]
    const float* gamma,        // [C]
    const float* sumDy,        // [N, C] - precomputed sum of dy * gamma per instance
    const float* sumDyXhat,    // [N, C] - precomputed sum of dy * gamma * xHat per instance
    float* gradInput,          // [N, C, H, W]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int instanceSize = H * W;
    float invN = 1.0f / (float)instanceSize;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * C + c];
    float s = invStd[n * C + c];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    int instanceIdx = n * C + c;
    float sDy = sumDy[instanceIdx];
    float sDyXhat = sumDyXhat[instanceIdx];

    // Full instance normalization backward formula:
    // dX = invStd * (dy*gamma - invN*(sumDy + xHat*sumDyXhat))
    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// ===========================================================================
// CONV3D BACKWARD KERNELS
// ===========================================================================

// Conv3D backward input gradient
extern ""C"" __global__ void conv3d_backward_input(
    const float* gradOutput,  // [N, outC, outD, outH, outW]
    const float* kernel,      // [outC, inC, kD, kH, kW]
    float* gradInput,         // [N, inC, D, H, W]
    int N, int inC, int D, int H, int W,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * inC * D * H * W;
    if (idx >= totalSize) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int ic = (idx / (W * H * D)) % inC;
    int n = idx / (W * H * D * inC);

    float sum = 0.0f;

    for (int oc = 0; oc < outC; oc++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int od = (d + padD - kd);
                    int oh = (h + padH - kh);
                    int ow = (w + padW - kw);

                    if (od % strideD == 0 && oh % strideH == 0 && ow % strideW == 0) {
                        od /= strideD;
                        oh /= strideH;
                        ow /= strideW;

                        if (od >= 0 && od < outD && oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                            int gradOutIdx = ((n * outC + oc) * outD + od) * outH * outW + oh * outW + ow;
                            int kernelIdx = ((oc * inC + ic) * kD + kd) * kH * kW + kh * kW + kw;
                            sum += gradOutput[gradOutIdx] * kernel[kernelIdx];
                        }
                    }
                }
            }
        }
    }

    gradInput[idx] = sum;
}

// Conv3D backward weight gradient
extern ""C"" __global__ void conv3d_backward_weights(
    const float* gradOutput,  // [N, outC, outD, outH, outW]
    const float* input,       // [N, inC, D, H, W]
    float* gradKernel,        // [outC, inC, kD, kH, kW]
    int N, int inC, int D, int H, int W,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalKernelSize = outC * inC * kD * kH * kW;
    if (idx >= totalKernelSize) return;

    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int ic = (idx / (kW * kH * kD)) % inC;
    int oc = idx / (kW * kH * kD * inC);

    float sum = 0.0f;

    for (int n = 0; n < N; n++) {
        for (int od = 0; od < outD; od++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int d = od * strideD + kd - padD;
                    int h = oh * strideH + kh - padH;
                    int w = ow * strideW + kw - padW;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int gradOutIdx = ((n * outC + oc) * outD + od) * outH * outW + oh * outW + ow;
                        int inputIdx = ((n * inC + ic) * D + d) * H * W + h * W + w;
                        sum += gradOutput[gradOutIdx] * input[inputIdx];
                    }
                }
            }
        }
    }

    gradKernel[idx] = sum;
}

// ===========================================================================
// GLOBAL POOLING BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void global_avgpool_backward(
    const float* gradOutput,  // [N, C]
    float* gradInput,         // [N, C, H, W]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float scale = 1.0f / (float)(H * W);
    gradInput[idx] = gradOutput[n * C + c] * scale;
}

extern ""C"" __global__ void global_maxpool_backward(
    const float* gradOutput,  // [N, C]
    const float* input,       // [N, C, H, W]
    const int* maxIndices,    // [N, C] - stored during forward
    float* gradInput,         // [N, C, H, W]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int spatialIdx = h * W + w;
    int maxIdx = maxIndices[n * C + c];

    gradInput[idx] = (spatialIdx == maxIdx) ? gradOutput[n * C + c] : 0.0f;
}

extern ""C"" __global__ void adaptive_avgpool_backward(
    const float* gradOutput,   // [N, C, outH, outW]
    float* gradInput,          // [N, C, H, W]
    int N, int C, int H, int W, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float sum = 0.0f;

    // For each output position that includes this input position
    for (int oh = 0; oh < outH; oh++) {
        int hStart = (oh * H) / outH;
        int hEnd = ((oh + 1) * H) / outH;
        if (h < hStart || h >= hEnd) continue;

        for (int ow = 0; ow < outW; ow++) {
            int wStart = (ow * W) / outW;
            int wEnd = ((ow + 1) * W) / outW;
            if (w < wStart || w >= wEnd) continue;

            int poolSize = (hEnd - hStart) * (wEnd - wStart);
            int gradOutIdx = ((n * C + c) * outH + oh) * outW + ow;
            sum += gradOutput[gradOutIdx] / (float)poolSize;
        }
    }

    gradInput[idx] = sum;
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
                // Optimizers
                "sgd_step",
                "adam_step",
                "adamw_step",
                "rmsprop_step",
                "adagrad_step",
                "nag_step",
                "lars_step",
                "lamb_step",
                "sgd_update",
                "adadelta_update",
                "amsgrad_update",
                "adamax_update",
                "lion_update",
                "nadam_update",
                "ftrl_update",
                // Additional optimizers
                "proximal_gradient_step",
                "conjugate_gradient_step",
                // L-BFGS two-loop recursion kernels
                "lbfgs_copy_vector",
                "lbfgs_dot_product_reduce",
                "lbfgs_reduce_partials",
                "lbfgs_axpy",
                "lbfgs_scale_vector",
                "lbfgs_apply_direction",
                "lbfgs_compute_rho",
                "lbfgs_update_history",
                "bfgs_step",
                "levenberg_marquardt_step",
                "trust_region_step",
                "admm_step",
                "newton_method_step",
                "dfp_step",
                "coordinate_descent_step",
                // Dropout and embedding
                "dropout_forward",
                "dropout_backward",
                "embedding_forward",
                "embedding_backward",
                // Transpose
                "transpose_2d",
                "batched_transpose",
                "permute_general",
                // LSTM kernels
                "lstm_cell_forward",
                "lstm_cell_backward",
                "lstm_gates_precompute",
                // GRU kernels
                "gru_cell_forward",
                "gru_cell_backward",
                // Scatter/Gather for GNNs
                "scatter_add",
                "scatter_add_batched",
                "scatter_max",
                "scatter_mean_accumulate",
                "scatter_mean_normalize",
                // Additional normalization backward
                "groupnorm_backward_sums",
                "groupnorm_backward",
                "instancenorm_backward_sums",
                "instancenorm_backward",
                // Conv3D backward
                "conv3d_backward_input",
                "conv3d_backward_weights",
                // Global pooling backward
                "global_avgpool_backward",
                "global_maxpool_backward",
                "adaptive_avgpool_backward"
            };
        }
    }
}
