using System;
using System.Collections.Generic;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// A deterministic, dependency-free <see cref="ITokenEmbeddingModel{T}"/> that maps each token to a
/// seeded, hash-derived unit vector. Intended as an offline default for late-interaction components
/// (e.g. <c>ColbertReranker</c>) so they are testable without downloading a neural model.
/// </summary>
/// <typeparam name="T">The numeric data type used for the embedding values (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Each token is lower-cased, hashed with a stable FNV-1a hash (deterministic across processes and
/// runtimes, unlike <see cref="string.GetHashCode()"/>), and that hash seeds a Gaussian random
/// vector which is then L2-normalised. Consequences:
/// </para>
/// <list type="bullet">
/// <item><description>The same token always maps to the same unit vector (cosine similarity 1.0 with itself).</description></item>
/// <item><description>Different tokens map to near-orthogonal vectors (cosine similarity near 0).</description></item>
/// </list>
/// <para>
/// This gives a lexical, exact-match relevance signal that is perfect for unit-testing the MaxSim
/// math without pretending to capture real semantics. Replace it with a genuine contextual token
/// embedder for production quality.
/// </para>
/// <para><b>For Beginners:</b> This is a stand-in "word to vectors" converter for testing. It is not
/// intelligent - it just turns each word into a fixed list of numbers in a repeatable way, so
/// identical words look identical and different words look different. Good enough to prove the
/// reranking logic works; swap in a real model for production.
/// </para>
/// </remarks>
public sealed class HashingTokenEmbeddingModel<T> : ITokenEmbeddingModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static readonly char[] TokenSeparators =
    {
        ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', '\''
    };

    private readonly int _embeddingDimension;

    /// <summary>
    /// Gets the dimensionality of each per-token embedding vector.
    /// </summary>
    public int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Initializes a new instance of the <see cref="HashingTokenEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="embeddingDimension">The dimensionality of each token embedding (default: 128).</param>
    public HashingTokenEmbeddingModel(int embeddingDimension = 128)
    {
        if (embeddingDimension <= 0)
            throw new ArgumentException("Embedding dimension must be greater than zero", nameof(embeddingDimension));

        _embeddingDimension = embeddingDimension;
    }

    /// <summary>
    /// Produces one deterministic unit-length embedding vector for each token in the supplied text.
    /// </summary>
    /// <param name="text">The text to embed at the token level.</param>
    /// <returns>An array of unit vectors, one per token; empty when the text has no tokens.</returns>
    public Vector<T>[] EmbedTokens(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));

        var tokens = text.ToLowerInvariant().Split(TokenSeparators, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
            return Array.Empty<Vector<T>>();

        var result = new Vector<T>[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            result[i] = EmbedToken(tokens[i]);
        }

        return result;
    }

    private Vector<T> EmbedToken(string token)
    {
        // Stable FNV-1a hash so the seed (and therefore the vector) is identical across
        // processes and runtimes. string.GetHashCode() is randomized per process in modern .NET.
        int seed = ComputeStableSeed(token);
        var random = RandomHelper.CreateSeededRandom(seed);

        var values = new T[_embeddingDimension];
        for (int i = 0; i < _embeddingDimension; i++)
        {
            values[i] = NumOps.FromDouble(random.NextGaussian());
        }

        // L2-normalise so dot products between token vectors equal cosine similarity.
        return VectorHelper.Normalize(new Vector<T>(values));
    }

    private static int ComputeStableSeed(string token)
    {
        unchecked
        {
            uint hash = 2166136261u;
            foreach (char c in token)
            {
                hash ^= c;
                hash *= 16777619u;
            }

            return (int)hash;
        }
    }
}
