// Copyright (c) AiDotNet. All rights reserved.
// Activation type enum for type-safe fused kernel selection.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Specifies the activation function to use in fused GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Purpose:</b></para>
/// <para>
/// This enum provides type-safe selection of activation functions for fused GPU kernels.
/// Using an enum instead of strings ensures compile-time validation and better IDE support.
/// </para>
/// <para><b>Fused Operations:</b></para>
/// <para>
/// DirectGpu kernels can fuse GEMM + Bias + Activation into a single kernel launch,
/// eliminating memory round-trips between operations. This can provide 20-50% speedup
/// compared to separate operations.
/// </para>
/// </remarks>
public enum ActivationType
{
    /// <summary>
    /// No activation function applied (identity).
    /// Output = GEMM + Bias
    /// </summary>
    None = 0,

    /// <summary>
    /// Rectified Linear Unit: f(x) = max(0, x)
    /// Fast and effective, most common choice for hidden layers.
    /// </summary>
    ReLU = 1,

    /// <summary>
    /// Gaussian Error Linear Unit: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// Popular in transformer architectures (BERT, GPT).
    /// </summary>
    GELU = 2,

    /// <summary>
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    /// Maps output to range (0, 1), useful for binary classification.
    /// </summary>
    Sigmoid = 3,

    /// <summary>
    /// Hyperbolic Tangent: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// Maps output to range (-1, 1), useful for hidden layers in some architectures.
    /// </summary>
    Tanh = 4
}
