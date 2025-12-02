namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused multi-head attention operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Multi-head attention for transformers.
///
/// Splits Q, K, V into multiple heads, applies attention, then concatenates.
/// This is the complete attention layer including all projections.
/// </para>
/// </remarks>
public class FusedMultiHeadAttentionOp : IROp
{
    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the head dimension.</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>Gets or sets whether to use causal masking.</summary>
    public bool CausalMask { get; set; } = false;

    /// <summary>Gets or sets dropout probability.</summary>
    public double Dropout { get; set; } = 0.0;

    /// <summary>Validates inputs (query, key, value).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 3) return false;
        return true;
    }
}
