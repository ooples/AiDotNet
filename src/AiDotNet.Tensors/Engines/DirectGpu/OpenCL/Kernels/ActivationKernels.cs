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

// Vector subtraction: C = A - B
__kernel void subtract_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] - B[idx];
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

// Vector division: C = A / B (element-wise)
__kernel void divide_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] / B[idx];
}

// Vector min: C = min(A, B)
__kernel void min_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = fmin(A[idx], B[idx]);
}

// Vector max: C = max(A, B)
__kernel void max_vectors(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = fmax(A[idx], B[idx]);
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

// Absolute value
__kernel void abs_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = fabs(A[idx]);
}

// Exponential
__kernel void exp_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = exp(A[idx]);
}

// Natural log
__kernel void log_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = log(A[idx]);
}

// Base-2 log
__kernel void log2_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = log2(A[idx]);
}

// Base-2 exp
__kernel void exp2_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = exp2(A[idx]);
}

// Base-10 exp
__kernel void exp10_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = pow(10.0f, A[idx]);
}

// exp(x) - 1
__kernel void expm1_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = exp(A[idx]) - 1.0f;
}

// log(1 + x)
__kernel void log1p_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = log(1.0f + A[idx]);
}

// Square root
__kernel void sqrt_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = sqrt(A[idx]);
}

// Sign
__kernel void sign_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = A[idx];
    B[idx] = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}

// Power with scalar exponent
__kernel void power_scalar(
    __global const float* A,
    __global float* B,
    const float exponent,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = pow(A[idx], exponent);
}

// ===========================================================================
// TRIGONOMETRIC OPERATIONS
// ===========================================================================

// Sine
__kernel void sin_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = sin(A[idx]);
}

// Cosine
__kernel void cos_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cos(A[idx]);
}

// Tangent
__kernel void tan_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = tan(A[idx]);
}

// Arc sine
__kernel void asin_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = asin(A[idx]);
}

// Arc cosine
__kernel void acos_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = acos(A[idx]);
}

// Arc tangent
__kernel void atan_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = atan(A[idx]);
}

// ===========================================================================
// HYPERBOLIC OPERATIONS
// ===========================================================================

// Hyperbolic sine
__kernel void sinh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = sinh(A[idx]);
}

// Hyperbolic cosine
__kernel void cosh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cosh(A[idx]);
}

// Inverse hyperbolic sine
__kernel void asinh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = asinh(A[idx]);
}

// Inverse hyperbolic cosine
__kernel void acosh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = acosh(A[idx]);
}

// Inverse hyperbolic tangent
__kernel void atanh_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = atanh(A[idx]);
}

// ===========================================================================
// ADDITIONAL UNARY OPERATIONS
// ===========================================================================

// Reciprocal: 1/x
__kernel void reciprocal_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = 1.0f / A[idx];
}

// Cube root
__kernel void cbrt_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = cbrt(A[idx]);
}

// Base-10 log
__kernel void log10_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = log10(A[idx]);
}

// Negate
__kernel void negate_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = -A[idx];
}

// Floor
__kernel void floor_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = floor(A[idx]);
}

// Ceiling
__kernel void ceil_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = ceil(A[idx]);
}

// Round
__kernel void round_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = round(A[idx]);
}

// Truncate
__kernel void trunc_vector(
    __global const float* A,
    __global float* B,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    B[idx] = trunc(A[idx]);
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
                // Activations
                "relu", "leaky_relu", "sigmoid", "tanh_activation",
                "gelu", "swish", "softmax",
                // Element-wise binary
                "add_vectors", "subtract_vectors", "multiply_vectors",
                "divide_vectors", "min_vectors", "max_vectors",
                // Scalar ops
                "scale_vector", "power_scalar",
                // Unary math
                "abs_vector", "exp_vector", "log_vector",
                "log2_vector", "exp2_vector", "exp10_vector",
                "expm1_vector", "log1p_vector", "sqrt_vector",
                "sign_vector",
                // Trigonometric
                "sin_vector", "cos_vector", "tan_vector",
                "asin_vector", "acos_vector", "atan_vector",
                // Hyperbolic
                "sinh_vector", "cosh_vector",
                "asinh_vector", "acosh_vector", "atanh_vector",
                // Additional unary
                "reciprocal_vector", "cbrt_vector", "log10_vector",
                "negate_vector", "floor_vector", "ceil_vector",
                "round_vector", "trunc_vector"
            };
        }
    }
}
