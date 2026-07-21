using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// Adapts any real single-vector <see cref="IEmbeddingModel{T}"/> into a per-token
/// <see cref="ITokenEmbeddingModel{T}"/> by tokenizing the text and embedding each token through the
/// underlying model. This is what lets ColBERT-style late interaction run over the library's existing
/// real embedders (Word2Vec / GloVe / FastText / ONNX sentence-transformer / hosted providers) instead of
/// any placeholder — the token vectors are exactly the real model's output for each token.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ColBERT scores a query against a document by comparing them token-by-token,
/// so it needs a vector <i>per word</i>. Most embedding models return a single vector for a whole string.
/// This adapter bridges the two: it splits the text into words and asks the real model to embed each word,
/// giving you genuine per-token embeddings with no fake/placeholder data.</para>
/// <para>Contextual (transformer) token embeddings are strictly stronger than embedding words in isolation;
/// supply a contextual token model directly if you have one. This adapter makes any word/sentence embedder
/// usable, which is the correct real fallback — not a synthetic hash.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for embedding vectors.</typeparam>
public sealed class EmbeddingModelTokenAdapter<T> : ITokenEmbeddingModel<T>
{
    // Whitespace + common punctuation. Kept explicit (not a regex) so it behaves identically across TFMs.
    private static readonly char[] TokenSeparators =
    {
        ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']',
        '{', '}', '<', '>', '/', '\\', '|', '-', '_', '+', '=', '*', '&', '%', '$', '#', '@', '~', '`'
    };

    private readonly IEmbeddingModel<T> _inner;

    /// <summary>
    /// Creates a token-embedding adapter over a real single-vector embedding model.
    /// </summary>
    /// <param name="embeddingModel">The real embedding model used to embed each token. Required.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="embeddingModel"/> is null.</exception>
    public EmbeddingModelTokenAdapter(IEmbeddingModel<T> embeddingModel)
    {
        if (embeddingModel is null)
            throw new ArgumentNullException(nameof(embeddingModel));
        _inner = embeddingModel;
    }

    /// <inheritdoc/>
    public int EmbeddingDimension => _inner.EmbeddingDimension;

    /// <inheritdoc/>
    public Vector<T>[] EmbedTokens(string text)
    {
        if (text is null)
            throw new ArgumentNullException(nameof(text));
        if (text.Length == 0)
            return Array.Empty<Vector<T>>();

        var tokens = text.ToLowerInvariant().Split(TokenSeparators, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
            return Array.Empty<Vector<T>>();

        var result = new Vector<T>[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            result[i] = _inner.Embed(tokens[i]);
        return result;
    }
}
