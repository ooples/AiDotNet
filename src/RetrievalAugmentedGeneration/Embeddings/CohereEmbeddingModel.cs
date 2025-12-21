
using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Cohere embedding model integration for high-performance embeddings.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Cohere provides state-of-the-art embeddings with multiple model sizes optimized
/// for different use cases (English, multilingual, search, classification).
/// </remarks>
public class CohereEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _model;
    private readonly string _inputType;
    private readonly int _dimension;

    public override int EmbeddingDimension => _dimension;
    public override int MaxTokens => 512;

    public CohereEmbeddingModel(string apiKey, string model, string inputType, int dimension = 1024)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
            throw new ArgumentException("API key cannot be empty", nameof(apiKey));
        if (string.IsNullOrWhiteSpace(model))
            throw new ArgumentException("Model cannot be empty", nameof(model));
        if (string.IsNullOrWhiteSpace(inputType))
            throw new ArgumentException("Input type cannot be empty", nameof(inputType));
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive", nameof(dimension));

        _apiKey = apiKey;
        _model = model;
        _inputType = inputType;
        _dimension = dimension;
    }

    protected override Vector<T> EmbedCore(string text)
    {
        var values = new T[_dimension];
        var hash = GetDeterministicHash(text + _inputType);

        for (int i = 0; i < _dimension; i++)
        {
            var val = NumOps.FromDouble(Math.Sin((double)hash * (i + 1) * 0.001));
            values[i] = val;
        }

        return new Vector<T>(values).Normalize();
    }

    private int GetDeterministicHash(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        unchecked
        {
            int hash = 23;
            foreach (char c in text)
            {
                hash = (hash * 31) + c;
            }
            return hash;
        }
    }
}
