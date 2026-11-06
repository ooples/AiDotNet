using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// An <see cref="IQueryEmbedder{T}"/> that embeds queries with an <see cref="IEmbeddingModel{T}"/>, the same model
/// that embedded the documents, so the query lands in the store's embedding space.
/// </summary>
/// <typeparam name="T">Numeric type for the embedding.</typeparam>
/// <remarks>
/// <para>
/// Retrievers that score by a query embedding (such as <see cref="MultiVectorRetriever{T}"/>) take an
/// <see cref="IQueryEmbedder{T}"/>. When queries and documents share one encoder, this adapter is that embedder.
/// </para>
/// <para><b>For Beginners:</b> A retriever compares your question with stored documents as vectors. Both must be
/// turned into vectors by the same model, or the comparison is meaningless; this wraps that model for the query side.
/// </para>
/// </remarks>
public class EmbeddingModelQueryEmbedder<T> : IQueryEmbedder<T>
{
    private readonly IEmbeddingModel<T> _embeddingModel;

    /// <summary>Creates a query embedder backed by <paramref name="embeddingModel"/>.</summary>
    /// <param name="embeddingModel">The model that embedded the store's documents.</param>
    public EmbeddingModelQueryEmbedder(IEmbeddingModel<T> embeddingModel)
    {
        Guard.NotNull(embeddingModel);
        _embeddingModel = embeddingModel;
    }

    /// <inheritdoc />
    public Vector<T> EmbedQuery(string query) => _embeddingModel.Embed(query);
}
