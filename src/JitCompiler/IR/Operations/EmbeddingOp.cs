namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents embedding lookup operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Looks up embeddings for input indices from an embedding table.
/// </para>
/// </remarks>
public class EmbeddingOp : IROp
{
    /// <summary>
    /// Size of the vocabulary.
    /// </summary>
    public int NumEmbeddings { get; set; }

    /// <summary>
    /// Size of each embedding vector.
    /// </summary>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Optional padding index that will output zeros.
    /// </summary>
    public int? PaddingIdx { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: indices, embedding_weights
        if (InputIds.Length != 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Embedding(t{InputIds[0]}, t{InputIds[1]}, dim={EmbeddingDim}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
