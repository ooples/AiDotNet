namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused layer normalization + add operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines LayerNorm with residual addition.
///
/// Very common in Transformers:
///   output = LayerNorm(x) + residual
///
/// Fusing reduces memory reads/writes.
/// </para>
/// </remarks>
public class FusedLayerNormAddOp : IROp
{
    /// <summary>Gets or sets the normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Validates inputs (x, gamma, beta, residual).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 4) return false;
        return true;
    }
}
