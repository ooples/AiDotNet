namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents a simplified attention operation for GPU code generation.
/// </summary>
/// <remarks>
/// <para>
/// This is a simplified version of attention used for GPU kernel generation.
/// Computes Attention(Q, K, V) = softmax(QK^T * scale) * V
/// </para>
/// </remarks>
public class AttentionOp : IROp
{
    /// <summary>
    /// Scaling factor for the attention scores.
    /// Typically 1/sqrt(head_dim).
    /// </summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; } = 1;

    /// <summary>
    /// Head dimension (d_k).
    /// </summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>
    /// Sequence length.
    /// </summary>
    public int SeqLength { get; set; } = 512;

    /// <summary>
    /// Whether to apply causal (autoregressive) masking.
    /// </summary>
    public bool IsCausal { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value (optionally mask)
        if (InputIds.Length < 3 || InputIds.Length > 4) return false;
        return true;
    }

    public override string ToString()
    {
        var causalStr = IsCausal ? ", causal" : "";
        return $"t{OutputId} = Attention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}, scale={Scale}{causalStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
