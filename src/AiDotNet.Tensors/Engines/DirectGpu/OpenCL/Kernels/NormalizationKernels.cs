// Copyright (c) AiDotNet. All rights reserved.
// Normalization kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for normalization operations.
    /// </summary>
    internal static class NormalizationKernels
    {
        /// <summary>
        /// Gets all normalization kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// NORMALIZATION KERNELS
// ===========================================================================

// Batch Normalization forward pass
// Input: [batch, channels, spatialSize]
// Computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
__kernel void batchnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* runningMean,
    __global float* runningVar,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon,
    const float momentum,
    const int training)
{
    const int c = get_global_id(0);
    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;

    // Compute mean
    float mean = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            mean += input[(b * channels + c) * spatialSize + s];
        }
    }
    mean /= (float)batchSpatial;

    // Compute variance
    float var = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatialSize; s++) {
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)batchSpatial;

    float invVar = 1.0f / sqrt(var + epsilon);

    // Save for backward pass
    saveMean[c] = mean;
    saveInvVar[c] = invVar;

    // Update running statistics if training
    if (training) {
        runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
        runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
    }

    // Normalize and apply affine transform
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

// Batch Normalization backward pass
__kernel void batchnorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradInput,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int c = get_global_id(0);
    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;
    float mean = saveMean[c];
    float invVar = saveInvVar[c];
    float g = gamma[c];

    // Compute gradGamma and gradBeta
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

    // Compute gradInput
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

// Layer Normalization forward pass
// Normalizes over the last dimension (features)
__kernel void layernorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batchSize,
    const int normalizedSize,
    const float epsilon)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        mean += input[b * normalizedSize + i];
    }
    mean /= (float)normalizedSize;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        float diff = input[b * normalizedSize + i] - mean;
        var += diff * diff;
    }
    var /= (float)normalizedSize;

    float invVar = 1.0f / sqrt(var + epsilon);

    saveMean[b] = mean;
    saveInvVar[b] = invVar;

    // Normalize and apply affine transform
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float normalized = (input[idx] - mean) * invVar;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

// Layer Normalization backward pass
__kernel void layernorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradInput,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batchSize,
    const int normalizedSize,
    const float epsilon)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    float mean = saveMean[b];
    float invVar = saveInvVar[b];

    // Compute intermediate sums
    float sumDy = 0.0f;
    float sumDyXmu = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float dy = gradOutput[idx] * gamma[i];
        sumDy += dy;
        sumDyXmu += dy * (input[idx] - mean);
    }

    // Compute gradInput
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        float xmu = input[idx] - mean;
        float dxhat = gradOutput[idx] * gamma[i];
        gradInput[idx] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)normalizedSize);
    }
}

// Layer Normalization gradient accumulation for gamma and beta
__kernel void layernorm_grad_params(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batchSize,
    const int normalizedSize)
{
    const int i = get_global_id(0);
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

// Group Normalization forward pass
__kernel void groupnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int numGroups,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int g = idx % numGroups;
    const int b = idx / numGroups;

    if (b >= batch) return;

    int channelsPerGroup = channels / numGroups;
    int groupSize = channelsPerGroup * spatialSize;

    // Compute mean for this group
    float mean = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            mean += input[(b * channels + c) * spatialSize + s];
        }
    }
    mean /= (float)groupSize;

    // Compute variance
    float var = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)groupSize;

    float invVar = 1.0f / sqrt(var + epsilon);

    int saveIdx = b * numGroups + g;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    // Normalize and apply affine transform
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            int inputIdx = (b * channels + c) * spatialSize + s;
            float normalized = (input[inputIdx] - mean) * invVar;
            output[inputIdx] = gamma[c] * normalized + beta[c];
        }
    }
}

// Instance Normalization forward pass
__kernel void instancenorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int c = idx % channels;
    const int b = idx / channels;

    if (b >= batch) return;

    // Compute mean for this instance
    float mean = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        mean += input[(b * channels + c) * spatialSize + s];
    }
    mean /= (float)spatialSize;

    // Compute variance
    float var = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        float diff = input[(b * channels + c) * spatialSize + s] - mean;
        var += diff * diff;
    }
    var /= (float)spatialSize;

    float invVar = 1.0f / sqrt(var + epsilon);

    int saveIdx = b * channels + c;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    // Normalize and apply affine transform
    float g = gamma[c];
    float bt = beta[c];
    for (int s = 0; s < spatialSize; s++) {
        int inputIdx = (b * channels + c) * spatialSize + s;
        float normalized = (input[inputIdx] - mean) * invVar;
        output[inputIdx] = g * normalized + bt;
    }
}

// RMS Normalization forward pass
// y = x / sqrt(mean(x^2) + eps) * gamma
__kernel void rmsnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global float* saveRms,
    const int batchSize,
    const int normalizedSize,
    const float epsilon)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    // Compute mean of squares
    float meanSq = 0.0f;
    for (int i = 0; i < normalizedSize; i++) {
        float x = input[b * normalizedSize + i];
        meanSq += x * x;
    }
    meanSq /= (float)normalizedSize;

    float rms = sqrt(meanSq + epsilon);
    float invRms = 1.0f / rms;
    saveRms[b] = rms;

    // Normalize and scale
    for (int i = 0; i < normalizedSize; i++) {
        int idx = b * normalizedSize + i;
        output[idx] = input[idx] * invRms * gamma[i];
    }
}

// RMS Normalization backward pass
// Computes gradients for input and gamma
__kernel void rmsnorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveRms,
    __global float* gradInput,
    __global float* gradGamma,
    const int batchSize,
    const int normalizedSize,
    const float epsilon)
{
    const int b = get_global_id(0);
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
        // Simplified: gamma * invRms - x * sumGradGammaX * invRms^3 / n
        gradInput[idx] = g * dy * invRms - x * sumGradGammaX * invRms3 / (float)normalizedSize;
    }
}

// RMS Normalization gradient accumulation for gamma
__kernel void rmsnorm_grad_gamma(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* saveRms,
    __global float* gradGamma,
    const int batchSize,
    const int normalizedSize)
{
    const int i = get_global_id(0);
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

// Scatter-Add backward pass (essentially a gather)
// gradSource[i] = gradDestination[indices[i]]
__kernel void scatter_add_backward(
    __global const float* gradDestination,
    __global const int* indices,
    __global float* gradSource,
    const int numIndices,
    const int featureSize)
{
    const int i = get_global_id(0);
    if (i >= numIndices) return;

    int srcIdx = indices[i];
    for (int f = 0; f < featureSize; f++) {
        gradSource[i * featureSize + f] = gradDestination[srcIdx * featureSize + f];
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
                "batchnorm_forward",
                "batchnorm_backward",
                "layernorm_forward",
                "layernorm_backward",
                "layernorm_grad_params",
                "groupnorm_forward",
                "instancenorm_forward",
                "rmsnorm_forward",
                "rmsnorm_backward",
                "rmsnorm_grad_gamma",
                "scatter_add_backward"
            };
        }
    }
}
