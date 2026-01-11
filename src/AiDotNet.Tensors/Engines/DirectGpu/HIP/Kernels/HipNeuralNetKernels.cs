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

// ---------------------------------------------------------------------------
// Triplet Loss (for metric learning)
// Formula: loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
// ---------------------------------------------------------------------------
extern ""C"" __global__ void triplet_loss(
    const float* anchor, const float* positive, const float* negative,
    float* loss, int batchSize, int embeddingDim, float margin)
{
    int tripletIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tripletIdx >= batchSize) return;

    int offset = tripletIdx * embeddingDim;

    // Compute squared distances
    float posDist2 = 0.0f;
    float negDist2 = 0.0f;
    for (int j = 0; j < embeddingDim; j++) {
        float diffPos = anchor[offset + j] - positive[offset + j];
        float diffNeg = anchor[offset + j] - negative[offset + j];
        posDist2 += diffPos * diffPos;
        negDist2 += diffNeg * diffNeg;
    }

    // max(0, pos_dist² - neg_dist² + margin)
    loss[tripletIdx] = fmaxf(0.0f, posDist2 - negDist2 + margin);
}

extern ""C"" __global__ void triplet_loss_backward(
    const float* anchor, const float* positive, const float* negative,
    float* gradAnchor, float* gradPositive, float* gradNegative,
    int batchSize, int embeddingDim, float margin)
{
    int tripletIdx = blockIdx.x;
    int tid = threadIdx.x;
    if (tripletIdx >= batchSize) return;

    int offset = tripletIdx * embeddingDim;

    // First, compute squared distances (one thread computes, others wait)
    __shared__ float posDist2;
    __shared__ float negDist2;
    __shared__ int isActive;

    if (tid == 0) {
        posDist2 = 0.0f;
        negDist2 = 0.0f;
        for (int j = 0; j < embeddingDim; j++) {
            float diffPos = anchor[offset + j] - positive[offset + j];
            float diffNeg = anchor[offset + j] - negative[offset + j];
            posDist2 += diffPos * diffPos;
            negDist2 += diffNeg * diffNeg;
        }
        isActive = (posDist2 - negDist2 + margin) > 0.0f ? 1 : 0;
    }
    __syncthreads();

    float scale = 2.0f / (float)batchSize;

    // Stride loop to handle embeddingDim > blockDim.x (max 1024)
    for (int featureIdx = tid; featureIdx < embeddingDim; featureIdx += blockDim.x) {
        int globalIdx = offset + featureIdx;
        if (isActive) {
            float diffPos = anchor[globalIdx] - positive[globalIdx];
            float diffNeg = anchor[globalIdx] - negative[globalIdx];

            // d_loss/d_anchor = 2*(anchor - positive) - 2*(anchor - negative)
            gradAnchor[globalIdx] = scale * (diffPos - diffNeg);
            // d_loss/d_positive = -2*(anchor - positive)
            gradPositive[globalIdx] = -scale * diffPos;
            // d_loss/d_negative = 2*(anchor - negative)
            gradNegative[globalIdx] = scale * diffNeg;
        } else {
            gradAnchor[globalIdx] = 0.0f;
            gradPositive[globalIdx] = 0.0f;
            gradNegative[globalIdx] = 0.0f;
        }
    }
}

// ===========================================================================
// ADDITIONAL LOSS KERNELS
// ===========================================================================

extern ""C"" __global__ void huber_loss(const float* predicted, const float* actual, float* output, float delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    float abs_diff = fabsf(diff);
    output[idx] = (abs_diff <= delta) ? 0.5f * diff * diff : delta * (abs_diff - 0.5f * delta);
}

extern ""C"" __global__ void huber_gradient(const float* predicted, const float* actual, float* gradient, float delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    gradient[idx] = (fabsf(diff) <= delta) ? diff : delta * ((diff > 0.0f) ? 1.0f : -1.0f);
}

extern ""C"" __global__ void focal_loss(const float* predicted, const float* actual, float* output, float alpha, float gamma, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pred = fmaxf(fminf(predicted[idx], 1.0f - epsilon), epsilon);
    output[idx] = alpha * powf(1.0f - pred, gamma) * actual[idx] * (-logf(pred));
}

extern ""C"" __global__ void focal_gradient(const float* predicted, const float* actual, float* gradient, float alpha, float gamma, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pred = fmaxf(fminf(predicted[idx], 1.0f - epsilon), epsilon);
    float focal_weight = powf(1.0f - pred, gamma);
    float d_focal = gamma * powf(1.0f - pred, gamma - 1.0f);
    gradient[idx] = alpha * actual[idx] * (d_focal * logf(pred) - focal_weight / pred);
}

extern ""C"" __global__ void mae_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fabsf(predicted[idx] - actual[idx]);
}

extern ""C"" __global__ void mae_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    gradient[idx] = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
}

extern ""C"" __global__ void log_cosh_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Numerically stable: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
    // This avoids overflow that occurs with coshf(x) for |x| > ~20
    float x = predicted[idx] - actual[idx];
    float ax = fabsf(x);
    output[idx] = ax + log1pf(expf(-2.0f * ax)) - 0.69314718056f;
}

extern ""C"" __global__ void log_cosh_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Gradient of log(cosh(x)) is tanh(x), but guard for large inputs
    float x = predicted[idx] - actual[idx];
    gradient[idx] = (fabsf(x) > 20.0f) ? copysignf(1.0f, x) : tanhf(x);
}

extern ""C"" __global__ void quantile_loss(const float* predicted, const float* actual, float* output, float quantile, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = actual[idx] - predicted[idx];
    output[idx] = (diff > 0.0f) ? quantile * diff : -(1.0f - quantile) * diff;
}

extern ""C"" __global__ void quantile_gradient(const float* predicted, const float* actual, float* gradient, float quantile, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradient[idx] = (actual[idx] > predicted[idx]) ? -quantile : (1.0f - quantile);
}

extern ""C"" __global__ void hinge_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fmaxf(0.0f, 1.0f - actual[idx] * predicted[idx]);
}

extern ""C"" __global__ void hinge_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradient[idx] = (actual[idx] * predicted[idx] < 1.0f) ? -actual[idx] : 0.0f;
}

extern ""C"" __global__ void squared_hinge_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float margin = fmaxf(0.0f, 1.0f - actual[idx] * predicted[idx]);
    output[idx] = margin * margin;
}

extern ""C"" __global__ void squared_hinge_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradient[idx] = -2.0f * actual[idx] * fmaxf(0.0f, 1.0f - actual[idx] * predicted[idx]);
}

extern ""C"" __global__ void poisson_loss(const float* predicted, const float* actual, float* output, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pred = fmaxf(predicted[idx], epsilon);
    output[idx] = pred - actual[idx] * logf(pred);
}

extern ""C"" __global__ void poisson_gradient(const float* predicted, const float* actual, float* gradient, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pred = fmaxf(predicted[idx], epsilon);
    gradient[idx] = 1.0f - actual[idx] / pred;
}

extern ""C"" __global__ void exponential_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = expf(-actual[idx] * predicted[idx]);
}

extern ""C"" __global__ void exponential_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradient[idx] = -actual[idx] * expf(-actual[idx] * predicted[idx]);
}

extern ""C"" __global__ void modified_huber_loss(const float* predicted, const float* actual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float z = actual[idx] * predicted[idx];
    if (z >= 1.0f) output[idx] = 0.0f;
    else if (z >= -1.0f) { float t = 1.0f - z; output[idx] = t * t; }
    else output[idx] = -4.0f * z;
}

extern ""C"" __global__ void modified_huber_gradient(const float* predicted, const float* actual, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float z = actual[idx] * predicted[idx];
    if (z >= 1.0f) gradient[idx] = 0.0f;
    else if (z >= -1.0f) gradient[idx] = -2.0f * actual[idx] * (1.0f - z);
    else gradient[idx] = -4.0f * actual[idx];
}

extern ""C"" __global__ void categorical_cross_entropy_loss(const float* predicted, const float* actual, float* output, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = -actual[idx] * logf(fmaxf(predicted[idx], epsilon));
}

extern ""C"" __global__ void categorical_cross_entropy_gradient(const float* predicted, const float* actual, float* gradient, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradient[idx] = -actual[idx] / fmaxf(predicted[idx], epsilon);
}

extern ""C"" __global__ void charbonnier_loss(const float* predicted, const float* actual, float* output, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    output[idx] = sqrtf(diff * diff + epsilon * epsilon) - epsilon;
}

extern ""C"" __global__ void charbonnier_gradient(const float* predicted, const float* actual, float* gradient, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    gradient[idx] = diff / sqrtf(diff * diff + epsilon * epsilon);
}

extern ""C"" __global__ void elastic_net_loss(const float* predicted, const float* actual, float* output, float l1_weight, float l2_weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    float pred = predicted[idx];
    output[idx] = diff * diff + l1_weight * fabsf(pred) + l2_weight * pred * pred;
}

extern ""C"" __global__ void elastic_net_gradient(const float* predicted, const float* actual, float* gradient, float l1_weight, float l2_weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predicted[idx] - actual[idx];
    float pred = predicted[idx];
    float sign_pred = (pred > 0.0f) ? 1.0f : ((pred < 0.0f) ? -1.0f : 0.0f);
    gradient[idx] = 2.0f * diff + l1_weight * sign_pred + 2.0f * l2_weight * pred;
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

extern ""C"" __global__ void fill_buffer(float* dst, float value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = value;
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

// LARS (Layer-wise Adaptive Rate Scaling) optimizer
// Reference: Large Batch Training of Convolutional Networks (You et al., 2017)
// Requires precomputed layer-wise norms: weightNorm = ||w||_2, gradNorm = ||g||_2
// Local learning rate: lr_local = trustCoeff * weightNorm / (gradNorm + weightDecay * weightNorm + eps)
extern ""C"" __global__ void lars_step(
    float* param, const float* gradient, float* velocity,
    float learningRate, float momentum, float weightDecay, float trustCoeff,
    float weightNorm, float gradNorm, float epsilon, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    // Compute LARS local learning rate from precomputed layer norms
    float localLr = learningRate;
    if (weightNorm > 0.0f && gradNorm > 0.0f) {
        float denom = gradNorm + weightDecay * weightNorm + epsilon;
        localLr = learningRate * trustCoeff * weightNorm / denom;
    }

    // Apply weight decay to gradient
    if (weightDecay > 0.0f) {
        grad += weightDecay * p;
    }

    // Update velocity with momentum (SGD with momentum)
    float v = momentum * velocity[idx] + localLr * grad;
    velocity[idx] = v;

    // Update parameters
    param[idx] = p - v;
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

    // Guard against step==0 to prevent division-by-zero in bias correction
    int safe_step = (step < 1) ? 1 : step;

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

    float mHat = mVal / (1.0f - powf(beta1, (float)safe_step));

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

    // Guard against step==0 to prevent division-by-zero in bias correction
    int safe_step = (step < 1) ? 1 : step;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float uVal = fmaxf(beta2 * u[idx], fabsf(grad));
    u[idx] = uVal;

    float biasCorrection = 1.0f - powf(beta1, (float)safe_step);

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

    // Guard against step==0 to prevent division-by-zero in bias correction
    int safe_step = (step < 1) ? 1 : step;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    float beta1Pow = powf(beta1, (float)safe_step);
    float beta2Pow = powf(beta2, (float)safe_step);
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

// BFGS update using diagonal Hessian approximation
// Note: Full BFGS requires O(n²) memory for dense Hessian inverse. This diagonal
// approximation enables efficient GPU parallelization while preserving per-parameter
// curvature scaling. For full matrix BFGS, use L-BFGS with two-loop recursion.
extern ""C"" __global__ void bfgs_step(
    float* param, const float* gradient, const float* invHessianDiag,
    float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Diagonal quasi-Newton: x = x - lr * diag(H^-1) * g
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

// Trust Region update using diagonal Hessian approximation
// Uses per-parameter trust region constraint for GPU-parallel execution.
// Each parameter's step is independently bounded by trustRadius.
extern ""C"" __global__ void trust_region_step(
    float* param, const float* gradient, const float* hessianDiag,
    float trustRadius, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float hess = hessianDiag[idx];

    // Newton step with diagonal Hessian: step = -H^-1 * g
    float step = -grad / (hess + 1e-8f);

    // Apply per-parameter trust region constraint
    float stepMag = fabsf(step);
    if (stepMag > trustRadius) {
        step = (step / stepMag) * trustRadius;
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

// Newton's Method using diagonal Hessian approximation
// Uses per-parameter curvature (diagonal elements) for GPU-parallel second-order optimization.
// Damping parameter prevents instability when Hessian diagonal is near zero or negative.
extern ""C"" __global__ void newton_method_step(
    float* param, const float* gradient, const float* hessianDiag,
    float learningRate, float damping, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Damped Newton step: x = x - lr * g / (H_ii + damping)
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
// LSTM KERNELS
// ===========================================================================

extern ""C"" __global__ void lstm_cell_forward(
    const float* gates,
    const float* cellPrev,
    float* cellNext,
    float* hiddenNext,
    float* gateActivations,
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;
    int gateOffset = b * 4 * hiddenSize;

    float gi = gates[gateOffset + h];
    float gf = gates[gateOffset + hiddenSize + h];
    float gg = gates[gateOffset + 2 * hiddenSize + h];
    float go = gates[gateOffset + 3 * hiddenSize + h];

    float i = 1.0f / (1.0f + expf(-gi));
    float f = 1.0f / (1.0f + expf(-gf));
    float g = tanhf(gg);
    float o = 1.0f / (1.0f + expf(-go));

    float cPrev = cellPrev[idx];
    float c = f * cPrev + i * g;
    float tanhC = tanhf(c);
    float hNew = o * tanhC;

    cellNext[idx] = c;
    hiddenNext[idx] = hNew;

    gateActivations[gateOffset + h] = i;
    gateActivations[gateOffset + hiddenSize + h] = f;
    gateActivations[gateOffset + 2 * hiddenSize + h] = g;
    gateActivations[gateOffset + 3 * hiddenSize + h] = o;
}

extern ""C"" __global__ void lstm_cell_backward(
    const float* gradHidden,
    const float* gradCellNext,
    const float* gateActivations,
    const float* cellPrev,
    const float* cellNext,
    float* gradGates,
    float* gradCellPrev,
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;
    int gateOffset = b * 4 * hiddenSize;

    float i = gateActivations[gateOffset + h];
    float f = gateActivations[gateOffset + hiddenSize + h];
    float g = gateActivations[gateOffset + 2 * hiddenSize + h];
    float o = gateActivations[gateOffset + 3 * hiddenSize + h];

    float cPrev = cellPrev[idx];
    float c = cellNext[idx];
    float tanhC = tanhf(c);

    float dH = gradHidden[idx];
    float dO = dH * tanhC;
    float dTanhC = dH * o;
    float dC = dTanhC * (1.0f - tanhC * tanhC);
    dC += gradCellNext[idx];

    float dF = dC * cPrev;
    float dI = dC * g;
    float dG = dC * i;
    float dCPrev = dC * f;

    float gradGi = dI * i * (1.0f - i);
    float gradGf = dF * f * (1.0f - f);
    float gradGg = dG * (1.0f - g * g);
    float gradGo = dO * o * (1.0f - o);

    gradGates[gateOffset + h] = gradGi;
    gradGates[gateOffset + hiddenSize + h] = gradGf;
    gradGates[gateOffset + 2 * hiddenSize + h] = gradGg;
    gradGates[gateOffset + 3 * hiddenSize + h] = gradGo;
    gradCellPrev[idx] = dCPrev;
}

extern ""C"" __global__ void lstm_gates_precompute(
    const float* input,
    const float* hiddenPrev,
    const float* weightsIH,
    const float* weightsHH,
    const float* bias,
    float* gates,
    int batchSize, int inputSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * 4 * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / (4 * hiddenSize);
    int g = idx % (4 * hiddenSize);

    float sum = bias[g];

    for (int i = 0; i < inputSize; i++) {
        sum += weightsIH[g * inputSize + i] * input[b * inputSize + i];
    }

    for (int h = 0; h < hiddenSize; h++) {
        sum += weightsHH[g * hiddenSize + h] * hiddenPrev[b * hiddenSize + h];
    }

    gates[idx] = sum;
}

// ===========================================================================
// GRU KERNELS
// ===========================================================================

extern ""C"" __global__ void gru_cell_forward(
    const float* gatesRZ,
    const float* gateN_input,
    const float* gateN_hidden,
    const float* hiddenPrev,
    float* hiddenNext,
    float* gateActivations,
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;

    float gr = gatesRZ[b * 2 * hiddenSize + h];
    float gz = gatesRZ[b * 2 * hiddenSize + hiddenSize + h];

    float r = 1.0f / (1.0f + expf(-gr));
    float z = 1.0f / (1.0f + expf(-gz));

    float nInput = gateN_input[idx];
    float nHidden = gateN_hidden[idx];
    float nPre = nInput + r * nHidden;
    float n = tanhf(nPre);

    float hPrev = hiddenPrev[idx];
    float hNew = (1.0f - z) * n + z * hPrev;

    hiddenNext[idx] = hNew;

    int actOffset = b * 3 * hiddenSize;
    gateActivations[actOffset + h] = r;
    gateActivations[actOffset + hiddenSize + h] = z;
    gateActivations[actOffset + 2 * hiddenSize + h] = n;
}

extern ""C"" __global__ void gru_cell_backward(
    const float* gradHidden,
    const float* gateActivations,
    const float* hiddenPrev,
    const float* gateN_hidden,
    float* gradGatesRZ,
    float* gradGateN,
    float* gradHiddenPrev,
    int batchSize, int hiddenSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    int b = idx / hiddenSize;
    int h = idx % hiddenSize;

    int actOffset = b * 3 * hiddenSize;
    float r = gateActivations[actOffset + h];
    float z = gateActivations[actOffset + hiddenSize + h];
    float n = gateActivations[actOffset + 2 * hiddenSize + h];

    float hPrev = hiddenPrev[idx];
    float dH = gradHidden[idx];

    float dZ = dH * (hPrev - n);
    float dN = dH * (1.0f - z);
    float dHPrev = dH * z;

    float dNPre = dN * (1.0f - n * n);

    float nHidden = gateN_hidden[idx];
    float dR = dNPre * nHidden;
    dHPrev += dNPre * r;

    float gradGr = dR * r * (1.0f - r);
    float gradGz = dZ * z * (1.0f - z);

    gradGatesRZ[b * 2 * hiddenSize + h] = gradGr;
    gradGatesRZ[b * 2 * hiddenSize + hiddenSize + h] = gradGz;
    gradGateN[idx] = dNPre;
    gradHiddenPrev[idx] = dHPrev;
}

// ===========================================================================
// SCATTER/GATHER OPERATIONS FOR GNNs
// ===========================================================================

extern ""C"" __global__ void scatter_add(
    const float* src,
    const int* indices,
    float* dst,
    int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    int dstIdx = indices[idx];
    atomicAdd(&dst[dstIdx], src[idx]);
}

extern ""C"" __global__ void scatter_add_batched(
    const float* src,
    const int* indices,
    float* dst,
    int numElements, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements * featureSize) return;

    int elemIdx = idx / featureSize;
    int featIdx = idx % featureSize;
    int dstIdx = indices[elemIdx];

    atomicAdd(&dst[dstIdx * featureSize + featIdx], src[idx]);
}

extern ""C"" __global__ void scatter_max(
    const float* src,
    const int* indices,
    float* dst,
    int* argmax,
    int numElements, int featureSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements * featureSize) return;

    int elemIdx = idx / featureSize;
    int featIdx = idx % featureSize;
    int dstIdx = indices[elemIdx];
    int dstOffset = dstIdx * featureSize + featIdx;

    float val = src[idx];
    float old = dst[dstOffset];
    while (val > old) {
        float assumed = old;
        old = __uint_as_float(atomicCAS((unsigned int*)&dst[dstOffset], 
                        __float_as_uint(assumed),
                        __float_as_uint(val)));
        if (old == assumed) {
            if (argmax != 0) argmax[dstOffset] = elemIdx;
            break;
        }
    }
}

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
// CRITICAL: Host MUST validate that C % G == 0 before launching this kernel.
// Launching with C not divisible by G will produce incorrect results.
extern ""C"" __global__ void groupnorm_backward_sums(
    const float* gradOutput,
    const float* input,
    const float* mean,
    const float* invStd,
    const float* gamma,
    float* sumDy,
    float* sumDyXhat,
    float* gradGamma,
    float* gradBeta,
    int N, int C, int H, int W, int G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

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

    atomicAdd(&gradGamma[c], dy * xHat);
    atomicAdd(&gradBeta[c], dy);

    int groupIdx = n * G + g;
    atomicAdd(&sumDy[groupIdx], dyGam);
    atomicAdd(&sumDyXhat[groupIdx], dyGam * xHat);
}

// Group normalization backward - Pass 2: Compute final input gradients
extern ""C"" __global__ void groupnorm_backward(
    const float* gradOutput,
    const float* input,
    const float* mean,
    const float* invStd,
    const float* gamma,
    const float* sumDy,
    const float* sumDyXhat,
    float* gradInput,
    int N, int C, int H, int W, int G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

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

    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// Instance normalization backward - Pass 1: Compute instance-wise sums
extern ""C"" __global__ void instancenorm_backward_sums(
    const float* gradOutput,
    const float* input,
    const float* mean,
    const float* invStd,
    const float* gamma,
    float* sumDy,
    float* sumDyXhat,
    float* gradGamma,
    float* gradBeta,
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
    const float* gradOutput,
    const float* input,
    const float* mean,
    const float* invStd,
    const float* gamma,
    const float* sumDy,
    const float* sumDyXhat,
    float* gradInput,
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

    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// ===========================================================================
// CONV3D BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void conv3d_backward_input(
    const float* gradOutput,
    const float* kernel,
    float* gradInput,
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

extern ""C"" __global__ void conv3d_backward_weights(
    const float* gradOutput,
    const float* input,
    float* gradKernel,
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
    const float* gradOutput,
    float* gradInput,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float scale = 1.0f / (float)(H * W);
    gradInput[idx] = gradOutput[n * C + c] * scale;
}

extern ""C"" __global__ void global_maxpool_backward(
    const float* gradOutput,
    const float* input,
    const int* maxIndices,
    float* gradInput,
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
    const float* gradOutput,
    float* gradInput,
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
            "triplet_loss", "triplet_loss_backward",
            // Additional loss functions
            "huber_loss", "huber_gradient", "focal_loss", "focal_gradient",
            "mae_loss", "mae_gradient", "log_cosh_loss", "log_cosh_gradient",
            "quantile_loss", "quantile_gradient", "hinge_loss", "hinge_gradient",
            "squared_hinge_loss", "squared_hinge_gradient", "poisson_loss", "poisson_gradient",
            "exponential_loss", "exponential_gradient", "modified_huber_loss", "modified_huber_gradient",
            "categorical_cross_entropy_loss", "categorical_cross_entropy_gradient",
            "charbonnier_loss", "charbonnier_gradient", "elastic_net_loss", "elastic_net_gradient",
            // NOTE: contrastive_loss and contrastive_loss_backward not implemented in HIP backend
            "clamp", "l2_norm_squared", "scale", "copy_buffer", "fill_buffer",
            "greater_than", "less_than", "equals", "where_cond",
            "compute_mean_var", "argmax_axis", "argmin_axis",
            "sgd_step", "adam_step", "adamw_step", "rmsprop_step", "adagrad_step", "nag_step", "lars_step", "lamb_step",
            "sgd_update", "adadelta_update", "amsgrad_update", "adamax_update", "lion_update", "nadam_update", "ftrl_update",
            "proximal_gradient_step", "conjugate_gradient_step",
            // L-BFGS two-loop recursion kernels
            "lbfgs_copy_vector", "lbfgs_dot_product_reduce", "lbfgs_reduce_partials", "lbfgs_axpy",
            "lbfgs_scale_vector", "lbfgs_apply_direction", "lbfgs_compute_rho", "lbfgs_update_history",
            "bfgs_step", "levenberg_marquardt_step", "trust_region_step", "admm_step", "newton_method_step", "dfp_step", "coordinate_descent_step",
            "dropout_forward", "dropout_backward", "embedding_forward", "embedding_backward",
            "transpose_2d", "batched_transpose", "permute_general",
            // LSTM kernels
            "lstm_cell_forward", "lstm_cell_backward", "lstm_gates_precompute",
            // GRU kernels
            "gru_cell_forward", "gru_cell_backward",
            // Scatter/Gather for GNNs
            "scatter_add", "scatter_add_batched", "scatter_max",
            "scatter_mean_accumulate", "scatter_mean_normalize",
            // Additional normalization backward
            "groupnorm_backward_sums", "groupnorm_backward",
            "instancenorm_backward_sums", "instancenorm_backward",
            // Conv3D backward
            "conv3d_backward_input", "conv3d_backward_weights",
            // Global pooling backward
            "global_avgpool_backward", "global_maxpool_backward", "adaptive_avgpool_backward",
            // Batched GEMM
            "batched_gemm"
        };
    }
}
