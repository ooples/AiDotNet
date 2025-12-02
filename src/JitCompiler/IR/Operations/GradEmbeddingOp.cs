namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for EmbeddingOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = embedding[indices]
/// Backward: grad_embedding = scatter_add(grad_y, indices, embedding_shape)
/// Gradients are scattered back to embedding table positions.
/// </para>
/// </remarks>
public class GradEmbeddingOp : BackwardOp
{
    /// <summary>Shape of the embedding table.</summary>
    public int[] EmbeddingShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and indices
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradEmbedding[shape={string.Join(",", EmbeddingShape)}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
