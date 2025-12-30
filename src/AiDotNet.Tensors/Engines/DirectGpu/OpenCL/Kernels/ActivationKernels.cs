// Copyright (c) AiDotNet. All rights reserved.
// Activation function kernels for neural network layers.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for common activation functions.
    /// </summary>
    internal static class ActivationKernels
    {
        /// <summary>
        /// Gets all activation kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// ACTIVATION KERNELS
// ===========================================================================

// ReLU: max(0, x)
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmax(0.0f, input[idx]);
}

// Leaky ReLU: x > 0 ? x : alpha * x
__kernel void leaky_relu(
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

// Sigmoid: 1 / (1 + exp(-x))
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
__kernel void tanh_activation(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = tanh(input[idx]);
}

// GELU (Gaussian Error Linear Unit) - approximation
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = input[idx];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}

// Swish: x * sigmoid(x)
__kernel void swish(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x / (1.0f + exp(-x));
}

// Softmax (per batch row)
// Two-pass: first find max, then compute exp and normalize
__kernel void softmax(
    __global const float* input,
    __global float* output,
    const int batchSize,
    const int features)
{
    const int batch = get_global_id(0);
    if (batch >= batchSize) return;

    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int f = 0; f < features; f++) {
        maxVal = fmax(maxVal, input[batch * features + f]);
    }

    // Compute exp and sum
    float sumExp = 0.0f;
    for (int f = 0; f < features; f++) {
        float expVal = exp(input[batch * features + f] - maxVal);
        output[batch * features + f] = expVal;
        sumExp += expVal;
    }

    // Normalize
    float invSum = 1.0f / sumExp;
    for (int f = 0; f < features; f++) {
        output[batch * features + f] *= invSum;
    }
}

// ===========================================================================
// ELEMENT-WISE OPERATIONS
// ===========================================================================

// Vector addition: C = A + B
__kernel void add_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] + B[idx];
}

// Vector multiplication: C = A * B (element-wise)
__kernel void multiply_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] * B[idx];
}

// Scalar multiplication: B = A * scalar
__kernel void scale_vector(
    __global const float* A,
    __global float* B,
    const float scalar,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = A[idx] * scalar;
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
                "relu", "leaky_relu", "sigmoid", "tanh_activation",
                "gelu", "swish", "softmax",
                "add_vectors", "multiply_vectors", "scale_vector"
            };
        }
    }
}
