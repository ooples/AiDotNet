using System;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration;

/// <summary>
/// Deterministic <see cref="IColBertEmbedder{T}"/> for unit tests.
/// Splits text on whitespace, lower-cases each token, hashes it with SHA-256,
/// maps the first 32 hash bytes to a signed-unit-vector slot, and L2-normalises.
/// Identical (case-folded) tokens produce identical unit vectors — cosine
/// similarity = 1.0, the perfect MaxSim score per
/// <c>Σ_q max_d cos(E_q, E_d)</c> — while non-identical tokens produce
/// near-orthogonal vectors (cosine ≈ 0).
/// </summary>
/// <remarks>
/// <para>
/// Required because <see cref="ColBERTRetriever{T}.RetrieveCore"/> throws
/// <see cref="NotSupportedException"/> when constructed without an embedder
/// (ColBERT has no defensible lexical-overlap fallback per the class doc;
/// the entire architecture is the token-level contextual representation).
/// Unit tests run without an ONNX checkpoint, so they wire this in instead.
/// </para>
/// </remarks>
internal sealed class TestColBertEmbedder<T> : IColBertEmbedder<T>
{
    private const int EmbedDim = 32;

    public Tensor<T> EmbedQuery(string query) => Embed(query);
    public Tensor<T> EmbedDocument(string document) => Embed(document);

    private static Tensor<T> Embed(string text)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var tokens = (text ?? string.Empty).Split(
            new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0) tokens = new[] { string.Empty };
        var result = new Tensor<T>(new[] { tokens.Length, EmbedDim });
        using var sha = SHA256.Create();
        for (int t = 0; t < tokens.Length; t++)
        {
            var lower = tokens[t].ToLowerInvariant();
            var bytes = Encoding.UTF8.GetBytes(lower);
            var hash = sha.ComputeHash(bytes);
            double sumSq = 0.0;
            for (int d = 0; d < EmbedDim; d++)
            {
                double v = ((sbyte)hash[d % hash.Length]) / 127.0;
                result[t, d] = numOps.FromDouble(v);
                sumSq += v * v;
            }
            double norm = Math.Sqrt(sumSq);
            if (norm > 0)
            {
                double invNorm = 1.0 / norm;
                for (int d = 0; d < EmbedDim; d++)
                    result[t, d] = numOps.FromDouble(
                        numOps.ToDouble(result[t, d]) * invNorm);
            }
        }
        return result;
    }
}
