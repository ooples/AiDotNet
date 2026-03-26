namespace AiDotNet.Enums;

/// <summary>
/// Relative computational cost of an operation.
/// </summary>
public enum ComputeCost
{
    /// <summary>Simple comparison/max (ReLU, LeakyReLU, ThresholdedReLU). O(1) per element, no transcendentals.</summary>
    Low,
    /// <summary>One transcendental function (Sigmoid, Tanh, ELU). Single exp/log per element.</summary>
    Medium,
    /// <summary>Multiple transcendentals or reductions (GELU, Swish, Mish, Softmax). Multiple exp/tanh per element or requires full-vector scan.</summary>
    High
}
