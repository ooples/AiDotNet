namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused bias + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adds bias and applies activation together.
///
/// output = activation(input + bias)
///
/// Common after linear/conv layers without built-in bias.
/// </para>
/// </remarks>
public class FusedBiasActivationOp : IROp
{
    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Validates inputs (input, bias).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
