namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents multi-head attention in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Multi-head attention allows the model to jointly attend to information
/// from different representation subspaces.
/// </para>
/// </remarks>
public class MultiHeadAttentionOp : IROp
{
    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    public int EmbedDim { get; set; }

    /// <summary>
    /// Key dimension per head.
    /// </summary>
    public int KeyDim { get; set; }

    /// <summary>
    /// Value dimension per head.
    /// </summary>
    public int ValueDim { get; set; }

    /// <summary>
    /// Dropout probability.
    /// </summary>
    public double DropoutProbability { get; set; }

    /// <summary>
    /// Whether this is self-attention (Q=K=V from same source).
    /// </summary>
    public bool IsSelfAttention { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value, W_q, W_k, W_v, W_o, optional mask
        if (InputIds.Length < 7) return false;
        if (NumHeads <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = MultiHeadAttention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}, heads={NumHeads}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
