namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused add + layer normalization operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines residual addition with LayerNorm.
///
/// Common in Transformer blocks:
///   output = LayerNorm(x + residual)
///
/// Reduces memory traffic by avoiding intermediate storage.
/// </para>
/// </remarks>
public class FusedAddLayerNormOp : IROp
{
    /// <summary>Gets or sets the normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Validates inputs (a, b, gamma, beta).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 4) return false;
        return true;
    }
}
