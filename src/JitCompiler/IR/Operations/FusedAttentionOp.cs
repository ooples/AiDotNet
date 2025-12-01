namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused attention operation (Q*K^T + softmax + matmul V).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The core of Transformer models!
///
/// Attention:
///   scores = Q @ K^T / sqrt(d_k)
///   weights = softmax(scores)
///   output = weights @ V
///
/// This is the most expensive part of transformers.
/// Fusing allows optimizations like Flash Attention for massive speedups.
/// </para>
/// </remarks>
public class FusedAttentionOp : IROp
{
    /// <summary>Gets or sets the softmax axis.</summary>
    public int SoftmaxAxis { get; set; } = -1;

    /// <summary>Gets or sets the scaling factor (typically 1/sqrt(d_k)).</summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>Gets or sets whether to use causal masking.</summary>
    public bool CausalMask { get; set; } = false;

    /// <summary>Validates inputs (Q, K, V).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        return true;
    }
}
