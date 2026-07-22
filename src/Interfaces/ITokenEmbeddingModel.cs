using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for a model that produces per-token (contextual) embeddings for a text.
/// </summary>
/// <typeparam name="T">The numeric data type used for the embedding values.</typeparam>
/// <remarks>
/// <para>
/// Unlike a single-vector embedding model (<see cref="IEmbeddingModel{T}"/>) which collapses an
/// entire text into one dense vector, a token embedding model returns <i>one vector per token</i>.
/// This is the representation required by late-interaction retrievers and rerankers such as
/// ColBERT (Khattab &amp; Zaharia 2020), where the relevance of a (query, document) pair is scored
/// by matching individual query tokens against individual document tokens rather than comparing
/// two pooled vectors.
/// </para>
/// <para><b>For Beginners:</b> Think of this as describing a sentence word-by-word instead of
/// summarising the whole sentence in one number-list.
///
/// - Single-vector model: "The quick brown fox" -&gt; one vector.
/// - Token embedding model: "The quick brown fox" -&gt; four vectors (one for each word).
///
/// Keeping a vector per word lets a reranker ask, for every word in your question, "which word in
/// this document matches me best?" - a much finer-grained comparison than a single overall vector.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TokenEmbeddingModel")]
public interface ITokenEmbeddingModel<T>
{
    /// <summary>
    /// Gets the dimensionality of each per-token embedding vector.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Produces one embedding vector for each token in the supplied text.
    /// </summary>
    /// <param name="text">The text to embed at the token level.</param>
    /// <returns>
    /// An array of vectors, one per token, each of length <see cref="EmbeddingDimension"/>.
    /// Returns an empty array when the text contains no tokens.
    /// </returns>
    Vector<T>[] EmbedTokens(string text);
}
