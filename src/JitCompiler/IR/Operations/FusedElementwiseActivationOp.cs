namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused element-wise operation with activation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines element-wise math with activation.
///
/// Examples:
///   Add + ReLU
///   Multiply + Sigmoid
///   Subtract + Tanh
///
/// Very common in residual connections and skip connections.
/// Saves memory by not storing intermediate results.
/// </para>
/// </remarks>
public class FusedElementwiseActivationOp : IROp
{
    /// <summary>
    /// Gets or sets the element-wise operation type.
    /// </summary>
    public string ElementwiseOp { get; set; } = "Add";

    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (2 inputs for binary element-wise ops).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ElementwiseOp) || string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
