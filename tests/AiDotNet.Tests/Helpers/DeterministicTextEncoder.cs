using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// A test stand-in for a CLIP text encoder: every text maps to a fixed embedding, either one the
/// test supplies or a reproducible pseudo-random unit-scale vector seeded from the text.
/// </summary>
/// <remarks>
/// The text-queried segmentation models take any <see cref="ITextEncoder{T}"/>; a real CLIP tower
/// would make every fixture minutes slower without exercising anything the models own. Reference
/// tests register exact embeddings so expected logits can be computed by hand.
/// </remarks>
public sealed class DeterministicTextEncoder<T> : ITextEncoder<T>
{
    private readonly Dictionary<string, double[]> _table = new Dictionary<string, double[]>(StringComparer.Ordinal);
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an encoder producing <paramref name="dimension"/>-wide embeddings.</summary>
    public DeterministicTextEncoder(int dimension = 16)
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        TextEmbeddingDimension = dimension;
    }

    /// <inheritdoc/>
    public int MaxSequenceLength => 77;

    /// <inheritdoc/>
    public int TextEmbeddingDimension { get; }

    /// <summary>Gets how many texts have been encoded, so tests can check the encoder was consulted.</summary>
    public int EncodeCount { get; private set; }

    /// <summary>Fixes the embedding returned for <paramref name="text"/>.</summary>
    public DeterministicTextEncoder<T> With(string text, params double[] embedding)
    {
        if (embedding.Length != TextEmbeddingDimension)
            throw new ArgumentException("Embedding width must equal TextEmbeddingDimension.", nameof(embedding));
        _table[text] = embedding;
        return this;
    }

    /// <inheritdoc/>
    public Tensor<T> EncodeText(string text)
    {
        EncodeCount++;
        if (!_table.TryGetValue(text, out var values))
        {
            values = new double[TextEmbeddingDimension];
            var rng = RandomHelper.CreateSeededRandom(StableSeed(text));
            for (int i = 0; i < values.Length; i++) values[i] = rng.NextDouble() * 2.0 - 1.0;
        }

        var result = new Tensor<T>(new[] { TextEmbeddingDimension });
        for (int i = 0; i < values.Length; i++) result[i] = _numOps.FromDouble(values[i]);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var result = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++) result[i] = EncodeText(texts[i]);
        return result;
    }

    private static int StableSeed(string text)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (char c in text) hash = (hash ^ c) * 16777619;
            return (int)(hash & 0x7FFFFFFF);
        }
    }
}
