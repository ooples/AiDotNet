// Copyright (c) AiDotNet. All rights reserved.
// CUDA activation kernels and simple elementwise ops.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaActivationKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

extern "C" __global__ void relu(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x > 0.0f ? x : 0.0f;
}

extern "C" __global__ void sigmoid(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void tanh_activation(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = tanhf(input[idx]);
}

extern "C" __global__ void gelu(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x = input[idx];
    float x3 = x * x * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" __global__ void swish(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

extern "C" __global__ void softmax(const float* input, float* output, int batchSize, int features)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    float maxVal = -INFINITY;
    int baseIdx = batch * features;
    for (int f = 0; f < features; f++)
    {
        float v = input[baseIdx + f];
        if (v > maxVal) maxVal = v;
    }

    float sumExp = 0.0f;
    for (int f = 0; f < features; f++)
    {
        float expVal = expf(input[baseIdx + f] - maxVal);
        output[baseIdx + f] = expVal;
        sumExp += expVal;
    }

    float invSum = 1.0f / sumExp;
    for (int f = 0; f < features; f++)
    {
        output[baseIdx + f] *= invSum;
    }
}

extern "C" __global__ void add_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] + B[idx];
}

extern "C" __global__ void multiply_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] * B[idx];
}

extern "C" __global__ void scale_vector(const float* A, float* B, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = A[idx] * scalar;
}

extern "C" __global__ void bias_add(float* data, const float* bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx >= size) return;
    int col = idx % cols;
    data[idx] += bias[col];
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "relu",
                "sigmoid",
                "tanh_activation",
                "gelu",
                "swish",
                "softmax",
                "add_vectors",
                "multiply_vectors",
                "scale_vector",
                "bias_add"
            };
        }
    }
}
