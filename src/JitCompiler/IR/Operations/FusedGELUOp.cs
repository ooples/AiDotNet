namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused GELU activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Gaussian Error Linear Unit.
///
/// GELU(x) = x * Phi(x), where Phi is the standard Gaussian CDF.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Very popular in transformers (BERT, GPT, etc.)
/// </para>
/// </remarks>
public class FusedGELUOp : IROp
{
    /// <summary>Whether to use the approximate version.</summary>
    public bool Approximate { get; set; } = true;

    /// <summary>Validates inputs (single input).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
