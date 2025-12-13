namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused matrix multiply + add + activation (full dense layer).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The ultimate fusion - entire dense layer in one op!
///
/// Combines:
///   MatMul + Add bias + Activation → One operation
///
/// Example:
///   output = activation(input @ weights + bias)
///
/// This is THE most common pattern in neural networks.
/// Can be 3-5x faster than three separate operations!
/// </para>
/// </remarks>
public class FusedDenseLayerOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (input, weights, bias).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
