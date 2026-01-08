// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for capsule network operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaCapsuleKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// Capsule prediction transform for DigitCapsuleLayer
// Computes: pred[b,i,c,d] = sum_k(input[b,i,k] * weights[i,c,k,d])
// Input: [batchSize, inputCapsules, inputDim]
// Weights: [inputCapsules, outputCapsules, inputDim, outputDim]
// Output: [batchSize, inputCapsules, outputCapsules, outputDim]
extern ""C"" __global__ void capsule_predictions(
    const float* input,
    const float* weights,
    float* output,
    int batchSize,
    int inputCapsules,
    int inputDim,
    int outputCapsules,
    int outputDim)
{
    // Each thread computes one output element: output[b, i, c, d]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batchSize * inputCapsules * outputCapsules * outputDim;
    if (idx >= totalOutputs) return;

    // Decode linear index to (b, i, c, d)
    int d = idx % outputDim;
    int temp = idx / outputDim;
    int c = temp % outputCapsules;
    temp = temp / outputCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    // Compute sum_k(input[b,i,k] * weights[i,c,k,d])
    float sum = 0.0f;
    for (int k = 0; k < inputDim; k++)
    {
        // input[b,i,k] at offset: b*inputCapsules*inputDim + i*inputDim + k
        int inputIdx = b * inputCapsules * inputDim + i * inputDim + k;

        // weights[i,c,k,d] at offset: i*outputCapsules*inputDim*outputDim + c*inputDim*outputDim + k*outputDim + d
        int weightIdx = i * outputCapsules * inputDim * outputDim + c * inputDim * outputDim + k * outputDim + d;

        sum += input[inputIdx] * weights[weightIdx];
    }

    output[idx] = sum;
}

// Capsule transform for CapsuleLayer
// Computes: transformed[b,i,j,d] = sum_k(input[b,i,k] * weights[i,k,j,d])
// Input: [batchSize, inputCapsules, inputDim]
// Weights: [inputCapsules, inputDim, numCapsules, capsuleDim]
// Output: [batchSize, inputCapsules, numCapsules, capsuleDim]
extern ""C"" __global__ void capsule_transform(
    const float* input,
    const float* weights,
    float* output,
    int batchSize,
    int inputCapsules,
    int inputDim,
    int numCapsules,
    int capsuleDim)
{
    // Each thread computes one output element: output[b, i, j, d]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batchSize * inputCapsules * numCapsules * capsuleDim;
    if (idx >= totalOutputs) return;

    // Decode linear index to (b, i, j, d)
    int d = idx % capsuleDim;
    int temp = idx / capsuleDim;
    int j = temp % numCapsules;
    temp = temp / numCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    // Compute sum_k(input[b,i,k] * weights[i,k,j,d])
    float sum = 0.0f;
    for (int k = 0; k < inputDim; k++)
    {
        // input[b,i,k] at offset: b*inputCapsules*inputDim + i*inputDim + k
        int inputIdx = b * inputCapsules * inputDim + i * inputDim + k;

        // weights[i,k,j,d] at offset: i*inputDim*numCapsules*capsuleDim + k*numCapsules*capsuleDim + j*capsuleDim + d
        int weightIdx = i * inputDim * numCapsules * capsuleDim + k * numCapsules * capsuleDim + j * capsuleDim + d;

        sum += input[inputIdx] * weights[weightIdx];
    }

    output[idx] = sum;
}

// Dynamic routing weighted sum
// Computes: output[b,c,d] = sum_i(coupling[b,i,c] * predictions[b,i,c,d])
// Coupling: [batchSize, inputCapsules, outputCapsules]
// Predictions: [batchSize, inputCapsules, outputCapsules, capsuleDim]
// Output: [batchSize, outputCapsules, capsuleDim]
extern ""C"" __global__ void capsule_weighted_sum(
    const float* coupling,
    const float* predictions,
    float* output,
    int batchSize,
    int inputCapsules,
    int outputCapsules,
    int capsuleDim)
{
    // Each thread computes one output element: output[b, c, d]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batchSize * outputCapsules * capsuleDim;
    if (idx >= totalOutputs) return;

    // Decode linear index to (b, c, d)
    int d = idx % capsuleDim;
    int temp = idx / capsuleDim;
    int c = temp % outputCapsules;
    int b = temp / outputCapsules;

    // Compute sum_i(coupling[b,i,c] * predictions[b,i,c,d])
    float sum = 0.0f;
    for (int i = 0; i < inputCapsules; i++)
    {
        // coupling[b,i,c] at offset: b*inputCapsules*outputCapsules + i*outputCapsules + c
        int couplingIdx = b * inputCapsules * outputCapsules + i * outputCapsules + c;

        // predictions[b,i,c,d] at offset: b*inputCapsules*outputCapsules*capsuleDim + i*outputCapsules*capsuleDim + c*capsuleDim + d
        int predIdx = b * inputCapsules * outputCapsules * capsuleDim + i * outputCapsules * capsuleDim + c * capsuleDim + d;

        sum += coupling[couplingIdx] * predictions[predIdx];
    }

    output[idx] = sum;
}

// Dynamic routing agreement update
// Computes: agreement[b,i,c] = sum_d(predictions[b,i,c,d] * output[b,c,d])
// Predictions: [batchSize, inputCapsules, outputCapsules, capsuleDim]
// Output: [batchSize, outputCapsules, capsuleDim]
// Agreement: [batchSize, inputCapsules, outputCapsules]
extern ""C"" __global__ void capsule_agreement(
    const float* predictions,
    const float* output,
    float* agreement,
    int batchSize,
    int inputCapsules,
    int outputCapsules,
    int capsuleDim)
{
    // Each thread computes one agreement element: agreement[b, i, c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batchSize * inputCapsules * outputCapsules;
    if (idx >= totalOutputs) return;

    // Decode linear index to (b, i, c)
    int c = idx % outputCapsules;
    int temp = idx / outputCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    // Compute sum_d(predictions[b,i,c,d] * output[b,c,d])
    float sum = 0.0f;
    for (int d = 0; d < capsuleDim; d++)
    {
        // predictions[b,i,c,d] at offset: b*inputCapsules*outputCapsules*capsuleDim + i*outputCapsules*capsuleDim + c*capsuleDim + d
        int predIdx = b * inputCapsules * outputCapsules * capsuleDim + i * outputCapsules * capsuleDim + c * capsuleDim + d;

        // output[b,c,d] at offset: b*outputCapsules*capsuleDim + c*capsuleDim + d
        int outIdx = b * outputCapsules * capsuleDim + c * capsuleDim + d;

        sum += predictions[predIdx] * output[outIdx];
    }

    agreement[idx] = sum;
}

";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "capsule_predictions",
                "capsule_transform",
                "capsule_weighted_sum",
                "capsule_agreement"
            };
        }
    }
}
