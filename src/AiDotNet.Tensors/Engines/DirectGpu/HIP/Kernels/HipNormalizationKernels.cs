// Copyright (c) AiDotNet. All rights reserved.
// HIP normalization kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipNormalizationKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

extern ""C"" __global__ void batchnorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* runningMean, float* runningVar,
    float* saveMean, float* saveInvVar,
    int batch, int channels, int spatialSize,
    float epsilon, float momentum, int training)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;

    float mean = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            mean += input[(b * channels + c) * spatialSize + s];
        }
    }
    mean /= (float)batchSpatial;

    float var = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)batchSpatial;

    float invVar = 1.0f / sqrtf(var + epsilon);
    saveMean[c] = mean;
    saveInvVar[c] = invVar;

    if (training) {
        runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
        runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
    }

    float g = gamma[c];
    float b_val = beta[c];
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            int idx = (b * channels + c) * spatialSize + s;
            float normalized = (input[idx] - mean) * invVar;
            output[idx] = g * normalized + b_val;
        }
    }
}

extern ""C"" __global__ void batchnorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveMean, const float* saveInvVar,
    float* gradInput, float* gradGamma, float* gradBeta,
    int batch, int channels, int spatialSize, float epsilon)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;
    float mean = saveMean[c];
    float invVar = saveInvVar[c];
    float g = gamma[c];

    float dGamma = 0.0f;
    float dBeta = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            int idx = (b * channels + c) * spatialSize + s;
            float normalized = (input[idx] - mean) * invVar;
            dGamma += gradOutput[idx] * normalized;
            dBeta += gradOutput[idx];
        }
    }
    gradGamma[c] = dGamma;
    gradBeta[c] = dBeta;

    float sumDy = dBeta;
    float sumDyXmu = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            int idx = (b * channels + c) * spatialSize + s;
            sumDyXmu += gradOutput[idx] * (input[idx] - mean);
        }
    }

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            int idx = (b * channels + c) * spatialSize + s;
            float xmu = input[idx] - mean;
            float dxhat = gradOutput[idx] * g;
            gradInput[idx] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)batchSpatial);
        }
    }
}

extern ""C"" __global__ void layernorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batchSize, int normalizedSize, float epsilon)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float mean = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        mean += input[b * normalizedSize + i];
    }
    mean /= (float)normalizedSize;

    float var = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        float diff = input[b * normalizedSize + i] - mean;
        var += diff * diff;
    }
    var /= (float)normalizedSize;

    float invVar = 1.0f / sqrtf(var + epsilon);
    saveMean[b] = mean;
    saveInvVar[b] = invVar;

    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float normalized = (input[idx] - mean) * invVar;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

extern ""C"" __global__ void layernorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveMean, const float* saveInvVar,
    float* gradInput, float* gradGamma, float* gradBeta,
    int batchSize, int normalizedSize, float epsilon)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float mean = saveMean[b];
    float invVar = saveInvVar[b];

    float sumDy = 0.0f;
    float sumDyXmu = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float dy = gradOutput[idx] * gamma[i];
        sumDy += dy;
        sumDyXmu += dy * (input[idx] - mean);
    }

    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float xmu = input[idx] - mean;
        float dxhat = gradOutput[idx] * gamma[i];
        gradInput[idx] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)normalizedSize);
    }
}

extern ""C"" __global__ void layernorm_grad_params(
    const float* gradOutput, const float* input,
    const float* saveMean, const float* saveInvVar,
    float* gradGamma, float* gradBeta,
    int batchSize, int normalizedSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    float dBeta = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float mean = saveMean[b];
        float invVar = saveInvVar[b];
        float normalized = (input[idx] - mean) * invVar;
        dGamma += gradOutput[idx] * normalized;
        dBeta += gradOutput[idx];
    }
    gradGamma[i] = dGamma;
    gradBeta[i] = dBeta;
}

extern ""C"" __global__ void groupnorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batch, int numGroups, int channels, int spatialSize, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g = idx % numGroups;
    int b = idx / numGroups;

    if (b >= batch) return;

    int channelsPerGroup = channels / numGroups;
    int groupSize = channelsPerGroup * spatialSize;

    float mean = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            mean += input[(b * channels + c) * spatialSize + s];
        }
    }
    mean /= (float)groupSize;

    float var = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)groupSize;

    float invVar = 1.0f / sqrtf(var + epsilon);
    int saveIdx = b * numGroups + g;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            int inputIdx = (b * channels + c) * spatialSize + s;
            float normalized = (input[inputIdx] - mean) * invVar;
            output[inputIdx] = gamma[c] * normalized + beta[c];
        }
    }
}

extern ""C"" __global__ void instancenorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batch, int channels, int spatialSize, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx % channels;
    int b = idx / channels;

    if (b >= batch) return;

    float mean = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        mean += input[(b * channels + c) * spatialSize + s];
    }
    mean /= (float)spatialSize;

    float var = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        float diff = input[(b * channels + c) * spatialSize + s] - mean;
        var += diff * diff;
    }
    var /= (float)spatialSize;

    float invVar = 1.0f / sqrtf(var + epsilon);
    int saveIdx = b * channels + c;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    float g = gamma[c];
    float bt = beta[c];
    for (int s = 0; s < spatialSize; s++) {
        int inputIdx = (b * channels + c) * spatialSize + s;
        float normalized = (input[inputIdx] - mean) * invVar;
        output[inputIdx] = g * normalized + bt;
    }
}

extern ""C"" __global__ void rmsnorm_forward(
    const float* input, float* output,
    const float* gamma, float* saveRms,
    int batchSize, int normalizedSize, float epsilon)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float meanSq = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        float x = input[b * normalizedSize + i];
        meanSq += x * x;
    }
    meanSq /= (float)normalizedSize;

    float rms = sqrtf(meanSq + epsilon);
    float invRms = 1.0f / rms;
    saveRms[b] = rms;

    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        output[idx] = input[idx] * invRms * gamma[i];
    }
}

// RMS Normalization backward pass
extern ""C"" __global__ void rmsnorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveRms,
    float* gradInput, float* gradGamma,
    int batchSize, int normalizedSize, float epsilon)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float rms = saveRms[b];
    float invRms = 1.0f / rms;
    float invRms3 = invRms * invRms * invRms;

    // Compute sum of (gradOutput * gamma * input) for this sample
    float sumGradGammaX = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        sumGradGammaX += gradOutput[idx] * gamma[i] * input[idx];
    }

    // Compute gradInput for this sample
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float x = input[idx];
        float dy = gradOutput[idx];
        float g = gamma[i];

        // d/dx (x / rms * gamma) = gamma * (1/rms - x^2 / (n * rms^3))
        gradInput[idx] = g * dy * invRms - x * sumGradGammaX * invRms3 / (float)normalizedSize;
    }
}

// RMS Normalization gradient accumulation for gamma
extern ""C"" __global__ void rmsnorm_grad_gamma(
    const float* gradOutput, const float* input,
    const float* saveRms, float* gradGamma,
    int batchSize, int normalizedSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float rms = saveRms[b];
        float invRms = 1.0f / rms;
        dGamma += gradOutput[idx] * input[idx] * invRms;
    }
    gradGamma[i] = dGamma;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "batchnorm_forward", "batchnorm_backward",
            "layernorm_forward", "layernorm_backward", "layernorm_grad_params",
            "groupnorm_forward", "instancenorm_forward", "rmsnorm_forward",
            "rmsnorm_backward", "rmsnorm_grad_gamma"
        };
    }
}
