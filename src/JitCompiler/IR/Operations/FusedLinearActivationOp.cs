namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused linear + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines linear layer with activation function.
///
/// Instead of:
///   t1 = Linear(input, weights, bias)
///   t2 = ReLU(t1)
///
/// We do:
///   t2 = LinearReLU(input, weights, bias)
///
/// Common in neural networks - almost every layer has an activation!
/// </para>
/// </remarks>
public class FusedLinearActivationOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs.
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
