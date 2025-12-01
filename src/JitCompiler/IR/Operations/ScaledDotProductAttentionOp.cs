namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents scaled dot-product attention in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
/// </para>
/// </remarks>
public class ScaledDotProductAttentionOp : IROp
{
    /// <summary>
    /// Optional scaling factor. If not specified, uses 1/sqrt(d_k).
    /// </summary>
    public double? Scale { get; set; }

    /// <summary>
    /// Whether to apply causal (autoregressive) masking.
    /// </summary>
    public bool IsCausal { get; set; }

    /// <summary>
    /// Dropout probability for attention weights.
    /// </summary>
    public double DropoutProbability { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value, optional mask
        if (InputIds.Length < 3 || InputIds.Length > 4) return false;
        return true;
    }

    public override string ToString()
    {
        var causalStr = IsCausal ? ", causal" : "";
        return $"t{OutputId} = ScaledDotProductAttention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}{causalStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
