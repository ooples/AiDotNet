using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Single-vector query embedder contract for dense retrievers that need
/// a per-query embedding (DPR-style, Karpukhin et al. 2020 §3.1). Concrete
/// implementations wrap a pretrained encoder (BERT-base / RoBERTa / E5 /
/// BGE / …) and produce a <see cref="Vector{T}"/> in the same embedding
/// space as the document store the retriever is reading from.
/// </summary>
/// <typeparam name="T">Numeric type for the embedding.</typeparam>
public interface IQueryEmbedder<T>
{
    /// <summary>
    /// Embeds <paramref name="query"/> into a dense vector matching the
    /// retriever's document-store dimensionality. The returned vector
    /// should be L2-normalised so cosine similarity at the store reduces
    /// to a plain dot product.
    /// </summary>
    Vector<T> EmbedQuery(string query);
}
