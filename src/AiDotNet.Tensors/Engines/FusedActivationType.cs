// Copyright (c) AiDotNet. All rights reserved.
// Fused activation type enum for IEngine abstraction layer.

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Specifies the activation function to use in fused operations.
/// </summary>
/// <remarks>
/// <para><b>Purpose:</b></para>
/// <para>
/// This enum provides type-safe selection of activation functions for fused operations
/// in IEngine. The engine implementation decides whether to use GPU fused kernels or
/// CPU sequential operations based on hardware availability and operation size.
/// </para>
/// <para><b>Performance Benefits:</b></para>
/// <para>
/// Fused operations combine multiple steps (e.g., MatMul + Bias + Activation) into a
/// single operation, eliminating intermediate memory allocations and transfers.
/// On GPU, this can provide 20-50% speedup over separate operations.
/// </para>
/// </remarks>
public enum FusedActivationType
{
    /// <summary>
    /// No activation function applied (identity).
    /// Output = Linear transformation only.
    /// </summary>
    None = 0,

    /// <summary>
    /// Rectified Linear Unit: f(x) = max(0, x)
    /// Fast and effective, most common choice for hidden layers.
    /// </summary>
    ReLU = 1,

    /// <summary>
    /// Gaussian Error Linear Unit: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Popular in transformer architectures (BERT, GPT).
    /// </summary>
    GELU = 2,

    /// <summary>
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    /// Maps output to range (0, 1), useful for binary classification and gates.
    /// </summary>
    Sigmoid = 3,

    /// <summary>
    /// Hyperbolic Tangent: f(x) = tanh(x)
    /// Maps output to range (-1, 1), useful for hidden layers in RNNs.
    /// </summary>
    Tanh = 4,

    /// <summary>
    /// Leaky ReLU: f(x) = x if x > 0, else alpha * x
    /// Prevents "dying ReLU" problem by allowing small gradients for negative inputs.
    /// </summary>
    LeakyReLU = 5,

    /// <summary>
    /// Swish/SiLU: f(x) = x * sigmoid(x)
    /// Self-gated activation, often outperforms ReLU in deep networks.
    /// </summary>
    Swish = 6,

    /// <summary>
    /// Softmax: exp(x_i) / sum(exp(x_j))
    /// Normalizes outputs to probability distribution, used for classification.
    /// </summary>
    Softmax = 7
}
